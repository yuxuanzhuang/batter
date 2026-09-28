from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter.exec.handlers.equil_analysis import (
    PROLIF_ARTIFACT_FILENAMES,
    PROLIF_INTERACTIONS_SCHEMA_VERSION,
    _copy_equil_analysis_artifacts,
    _equil_anchor_masks_for_analysis_topology,
    _equil_anchor_masks_to_original_resids,
    _load_equil_anchor_masks,
    _load_no_equil_representative_universe,
    _persistent_prolif_ligand_anchor_preferences,
    _persistent_prolif_residue_ids,
    _persistent_prolif_residue_priorities,
    _prolif_dataframe_with_original_residue_labels,
    _prolif_interaction_id,
    _prolif_interactions_current,
    _prolif_ligand_atom_names_by_interaction,
    _prolif_protein_residue_map,
    _prolif_residue_metadata,
    _records_from_prolif_dataframe,
    _run_prolif_fingerprint,
    _salt_bridge_ligand_atom_preference,
    _stable_distance_validator,
    _write_prolif_lignetwork_html,
    _write_prolif_interactions,
    _write_prolif_artifacts,
    _write_representative_only_prolif,
    _write_stable_boresch_distance,
)


def test_equil_anchor_masks_are_loaded_from_prepared_anchors_json(tmp_path: Path) -> None:
    (tmp_path / "anchors.json").write_text(
        '{"P1": ":79@CA", "P2": ":84@CA", "P3": ":117@CA"}\n'
    )

    assert _load_equil_anchor_masks(tmp_path) == [":79@CA", ":84@CA", ":117@CA"]


def test_equil_anchor_masks_convert_to_original_resids_with_dum_offset(
    tmp_path: Path,
) -> None:
    renum = tmp_path / "protein_renum.txt"
    renum.write_text(
        "LEU B 387 LEU 78\n"
        "VAL B 392 VAL 83\n"
        "PHE B 425 PHE 116\n"
    )

    assert _equil_anchor_masks_to_original_resids(
        [":79@CA", ":84@CA", ":117@CA"],
        renum,
    ) == [":387@CA", ":392@CA", ":425@CA"]


def test_analysis_topology_keeps_prepared_anchor_masks_for_amber_topology(
    tmp_path: Path,
) -> None:
    (tmp_path / "anchors.json").write_text(
        '{"P1": ":85@CA", "P2": ":51@CA", "P3": ":262@CA"}\n'
    )
    renum = tmp_path / "protein_renum.txt"
    renum.write_text(
        "ASP A 113 ASP 84\n"
        "ASP A 79 ASP 50\n"
        "ASN C 318 ASN 261\n"
    )

    assert _equil_anchor_masks_for_analysis_topology(
        tmp_path,
        renum,
        uses_amber_topology=True,
    ) == [":85@CA", ":51@CA", ":262@CA"]


def test_analysis_topology_converts_anchor_masks_for_pdb_fallback(
    tmp_path: Path,
) -> None:
    (tmp_path / "anchors.json").write_text(
        '{"P1": ":85@CA", "P2": ":51@CA", "P3": ":262@CA"}\n'
    )
    renum = tmp_path / "protein_renum.txt"
    renum.write_text(
        "ASP A 113 ASP 84\n"
        "ASP A 79 ASP 50\n"
        "ASN C 318 ASN 261\n"
    )

    assert _equil_anchor_masks_for_analysis_topology(
        tmp_path,
        renum,
        uses_amber_topology=False,
    ) == [":113@CA", ":79@CA", ":318@CA"]


def test_prolif_protein_residue_map_applies_dum_offset(tmp_path: Path) -> None:
    renum = tmp_path / "protein_renum.txt"
    renum.write_text(
        "ASP A 147 ASP 83\n"
        "ILE A 322 ILE 258\n"
    )

    assert _prolif_protein_residue_map(renum) == {
        84: {
            "resid": 147,
            "resname": "ASP",
            "chainID": "A",
            "prepared_resname": "ASP",
        },
        259: {
            "resid": 322,
            "resname": "ILE",
            "chainID": "A",
            "prepared_resname": "ILE",
        },
    }
    assert _prolif_protein_residue_map(tmp_path / "missing.txt") == {}


def test_prolif_dataframe_records_persistent_protein_residues() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("LIG300.A", "ASP42.A", "HBAcceptor"),
            ("LIG300.A", "GLY77.A", "Hydrophobic"),
            ("LIG300.A", "LEU88.A", "VdWContact"),
        ]
    )
    df = pd.DataFrame(
        [
            [True, False, True],
            [True, True, True],
            [False, False, True],
        ],
        columns=columns,
    )

    interactions, persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.5,
    )

    assert any(item["interaction"] == "Hydrophobic" for item in interactions)
    assert any(item["interaction"] == "VdWContact" for item in interactions)
    hbond_record = next(
        item for item in interactions if item["interaction"] == "HBAcceptor"
    )
    assert hbond_record["protein"]["resid"] == 42
    assert hbond_record["occupancy"] == 2 / 3
    assert persistent == [
        {
            "resid": 42,
            "resname": "ASP",
            "chainID": "A",
            "max_occupancy": 2 / 3,
            "interactions": [
                {
                    "interaction": "HBAcceptor",
                    "occupancy": 2 / 3,
                    "active_frames": 2,
                    "ligand": {
                        "label": "LIG300.A",
                        "resname": "LIG",
                        "resid": 300,
                        "chainID": "A",
                    },
                }
            ],
        }
    ]
    assert _persistent_prolif_residue_ids(
        {"usable": True, "persistent_protein_residues": persistent}
    ) == [42]


def test_prolif_outputs_use_original_residue_labels_but_candidates_use_prepared_ids(
) -> None:
    columns = pd.MultiIndex.from_tuples(
        [("LIG290.0", "ASP84.0", "Cationic")]
    )
    df = pd.DataFrame([[True]], columns=columns)
    residue_map = {
        84: {"resid": 147, "resname": "ASP", "chainID": "A"},
    }

    interactions, persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.3,
        protein_residue_map=residue_map,
    )

    assert interactions[0]["protein"] == {
        "label": "ASP147.A",
        "resname": "ASP",
        "resid": 147,
        "chainID": "A",
        "prepared_label": "ASP84",
        "prepared_resid": 84,
    }
    assert persistent == [
        {
            "resid": 147,
            "resname": "ASP",
            "chainID": "A",
            "prepared_resid": 84,
            "max_occupancy": 1.0,
            "interactions": [
                {
                    "interaction": "Cationic",
                    "occupancy": 1.0,
                    "active_frames": 1,
                    "ligand": {
                        "label": "LIG290",
                        "resname": "LIG",
                        "resid": 290,
                        "chainID": "0",
                    },
                }
            ],
        }
    ]
    record = {"usable": True, "persistent_protein_residues": persistent}
    assert _persistent_prolif_residue_ids(record) == [84]
    assert _persistent_prolif_residue_priorities(record) == {84: 0}
    assert _persistent_prolif_ligand_anchor_preferences(record) == []

    display_df = _prolif_dataframe_with_original_residue_labels(df, residue_map)
    assert list(display_df.columns) == [("LIG290.0", "ASP147.A", "Cationic")]
    assert _prolif_interaction_id(display_df.columns[0]) == (
        "LIG290|ASP147.A|Cationic"
    )


def test_prolif_residue_labels_show_integer_resids() -> None:
    protein_meta = _prolif_residue_metadata("ASP86.0")
    ligand_meta = _prolif_residue_metadata("hmn292.0")

    assert protein_meta["label"] == "ASP86"
    assert protein_meta["resid"] == 86
    assert isinstance(protein_meta["resid"], int)
    assert ligand_meta["label"] == "hmn292"
    assert _prolif_interaction_id(("hmn292.0", "ASP86.0", "Anionic")) == (
        "hmn292|ASP86|Anionic"
    )


def test_prolif_atom_metadata_schema_invalidates_old_cache(tmp_path: Path) -> None:
    path = tmp_path / "prolif_interactions.json"
    path.write_text(
        json.dumps({"schema_version": PROLIF_INTERACTIONS_SCHEMA_VERSION - 1})
        + "\n"
    )
    assert not _prolif_interactions_current(path)

    path.write_text(
        json.dumps({"schema_version": PROLIF_INTERACTIONS_SCHEMA_VERSION}) + "\n"
    )
    assert _prolif_interactions_current(path)


def test_persistent_prolif_residue_priorities_rank_salt_bridge_first() -> None:
    prolif_record = {
        "usable": True,
        "persistent_protein_residues": [
            {
                "resid": 10,
                "interactions": [
                    {"interaction": "HBAcceptor"},
                    {"interaction": "PiStacking"},
                ],
            },
            {
                "resid": 20,
                "interactions": [{"interaction": "Anionic"}],
            },
            {
                "resid": 30,
                "interactions": [{"interaction": "PiStacking"}],
            },
        ],
    }

    assert _persistent_prolif_residue_priorities(prolif_record) == {
        10: 1,
        20: 0,
        30: 2,
    }


def test_prolif_atom_metadata_prioritizes_hbond_heavy_atom() -> None:
    ligand_atoms = [
        SimpleNamespace(index=100, name="N3", element="N"),
        SimpleNamespace(index=101, name="H8", element="H"),
        SimpleNamespace(index=102, name="C14", element="C"),
    ]
    fingerprint = SimpleNamespace(
        ifp={
            0: {
                ("LIG300.A", "VAL93.A"): {
                    "HBDonor": (
                        {
                            "indices": {"ligand": (0, 1)},
                            "parent_indices": {"ligand": (100, 101)},
                        },
                    ),
                    "VdWContact": (
                        {
                            "indices": {"ligand": (2,)},
                            "parent_indices": {"ligand": (102,)},
                        },
                    ),
                }
            }
        }
    )

    atom_names = _prolif_ligand_atom_names_by_interaction(
        fingerprint,
        ligand_atoms,
    )
    assert atom_names[("LIG300.A", "VAL93.A", "hbdonor")] == ["N3"]
    assert atom_names[("LIG300.A", "VAL93.A", "vdwcontact")] == ["C14"]

    df = pd.DataFrame(
        [[True, True]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("LIG300.A", "VAL93.A", "HBDonor"),
                ("LIG300.A", "VAL93.A", "VdWContact"),
            ]
        ),
    )
    _interactions, persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.3,
        ligand_atom_names_by_interaction=atom_names,
    )
    assert persistent[0]["interactions"][0]["ligand_atom_names"] == ["N3"]
    assert _persistent_prolif_ligand_anchor_preferences(
        {"usable": True, "persistent_protein_residues": persistent}
    ) == [
        {
            "name": "N3",
            "interaction": "HBDonor",
            "interaction_priority": 1,
            "occupancy": 1.0,
            "protein_resid": 93,
        }
    ]


def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4}{resname:>4} {chain}{resid:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def test_salt_bridge_ligand_atom_preference_uses_prolif_salt_bridge(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    pdb = tmp_path / "salt_bridge.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ASP", "A", 10, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "OD1", "ASP", "A", 10, 1.0, 0.0, 0.0, "O"),
                _atom_line(3, "OD2", "ASP", "A", 10, 2.0, 0.0, 0.0, "O"),
                _atom_line(4, "N1", "LIG", "L", 300, 2.5, 0.0, 0.0, "N"),
                _atom_line(5, "C1", "LIG", "L", 300, 5.5, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    rw_mol = Chem.RWMol()
    nitrogen = Chem.Atom("N")
    nitrogen.SetFormalCharge(1)
    nitrogen.SetNoImplicit(True)
    nitrogen_idx = rw_mol.AddAtom(nitrogen)
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(nitrogen_idx, Point3D(2.5, 0.0, 0.0))
    conformer.SetAtomPosition(carbon_idx, Point3D(5.5, 0.0, 0.0))
    mol.AddConformer(conformer)
    params = tmp_path / "params"
    params.mkdir()
    Chem.MolToMolFile(mol, str(params / "LIG.sdf"))

    preference = _salt_bridge_ligand_atom_preference(
        system_root=tmp_path,
        residue_name="LIG",
        ligand_label="pose",
        universe=u,
        tail_fraction=1.0,
        prolif_record={
            "usable": True,
            "persistent_protein_residues": [
                {
                    "resid": 10,
                    "resname": "ASP",
                    "interactions": [
                        {"interaction": "Cationic", "occupancy": 1.0}
                    ],
                }
            ],
        },
    )

    assert preference["ligand_atom_names"] == ["N1"]
    assert preference["protein_residue_ids"] == [10]
    assert preference["pairs"][0]["protein"]["name"] in {"OD1", "OD2"}
    assert preference["pairs"][0]["ligand"]["name"] == "N1"


def test_salt_bridge_ligand_atom_preference_falls_back_to_geometry(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    pdb = tmp_path / "salt_bridge.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ASP", "A", 10, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "OD1", "ASP", "A", 10, 1.0, 0.0, 0.0, "O"),
                _atom_line(3, "OD2", "ASP", "A", 10, 2.0, 0.0, 0.0, "O"),
                _atom_line(4, "N1", "LIG", "L", 300, 2.5, 0.0, 0.0, "N"),
                _atom_line(5, "C1", "LIG", "L", 300, 5.5, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    rw_mol = Chem.RWMol()
    nitrogen = Chem.Atom("N")
    nitrogen.SetFormalCharge(1)
    nitrogen.SetNoImplicit(True)
    nitrogen_idx = rw_mol.AddAtom(nitrogen)
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(nitrogen_idx, Point3D(2.5, 0.0, 0.0))
    conformer.SetAtomPosition(carbon_idx, Point3D(5.5, 0.0, 0.0))
    mol.AddConformer(conformer)
    params = tmp_path / "params"
    params.mkdir()
    Chem.MolToMolFile(mol, str(params / "LIG.sdf"))

    preference = _salt_bridge_ligand_atom_preference(
        system_root=tmp_path,
        residue_name="LIG",
        ligand_label="pose",
        universe=u,
        tail_fraction=1.0,
        prolif_record={
            "usable": True,
            "persistent_protein_residues": [],
        },
    )

    assert preference["source"] == "charged_atom_distance"
    assert preference["ligand_atom_names"] == ["N1"]
    assert preference["protein_residue_ids"] == [10]


def test_stable_boresch_distance_uses_geometry_salt_bridge_residue_filter(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    pdb = tmp_path / "stable_filter.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ASP", "A", 10, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "OD1", "ASP", "A", 10, 5.5, 0.0, 0.0, "O"),
                _atom_line(3, "OD2", "ASP", "A", 10, 5.7, 0.0, 0.0, "O"),
                _atom_line(4, "CA", "GLY", "A", 20, 1.0, 0.0, 0.0, "C"),
                _atom_line(5, "N1", "LIG", "L", 300, 6.0, 0.0, 0.0, "N"),
                _atom_line(6, "C1", "LIG", "L", 300, 9.0, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    rw_mol = Chem.RWMol()
    nitrogen = Chem.Atom("N")
    nitrogen.SetFormalCharge(1)
    nitrogen.SetNoImplicit(True)
    nitrogen_idx = rw_mol.AddAtom(nitrogen)
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(nitrogen_idx, Point3D(6.0, 0.0, 0.0))
    conformer.SetAtomPosition(carbon_idx, Point3D(9.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    params = tmp_path / "params"
    params.mkdir()
    Chem.MolToMolFile(mol, str(params / "LIG.sdf"))

    stable = _write_stable_boresch_distance(
        stable_path=tmp_path / "stable_boresch_distance.json",
        system_root=tmp_path,
        sim=SimpleNamespace(min_adis=3.0, max_adis=7.0),
        sim_val=_stable_distance_validator(
            universe=u,
            residue_name="LIG",
            directory=tmp_path,
            protein_anchor_masks=[],
        ),
        ligand_label="pose",
        residue_name="LIG",
        universe=u,
        tail_fraction=1.0,
        mode="test",
        prolif_record={"usable": True, "persistent_protein_residues": []},
    )

    assert stable["protein"]["resid"] == 10
    assert stable["ligand"]["name"] == "N1"
    assert stable["prolif_preference"]["used_salt_bridge_residue_filter"] is True


def test_stable_boresch_distance_uses_preference_universe_for_salt_bridge(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    validator_pdb = tmp_path / "validator.pdb"
    validator_pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "VAL", "A", 86, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "CA", "GLY", "A", 20, 10.0, 0.0, 0.0, "C"),
                _atom_line(3, "N1", "LIG", "L", 300, 5.0, 0.0, 0.0, "N"),
                _atom_line(4, "C1", "LIG", "L", 300, 6.0, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    preference_pdb = tmp_path / "preference.pdb"
    preference_pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ASP", "A", 86, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "OD1", "ASP", "A", 86, 4.4, 0.0, 0.0, "O"),
                _atom_line(3, "OD2", "ASP", "A", 86, 4.6, 0.0, 0.0, "O"),
                _atom_line(4, "N1", "LIG", "L", 300, 5.0, 0.0, 0.0, "N"),
                _atom_line(5, "C1", "LIG", "L", 300, 8.0, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    validator_u = mda.Universe(str(validator_pdb))
    preference_u = mda.Universe(str(preference_pdb))

    rw_mol = Chem.RWMol()
    nitrogen = Chem.Atom("N")
    nitrogen.SetFormalCharge(1)
    nitrogen.SetNoImplicit(True)
    nitrogen_idx = rw_mol.AddAtom(nitrogen)
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(nitrogen_idx, Point3D(5.0, 0.0, 0.0))
    conformer.SetAtomPosition(carbon_idx, Point3D(8.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    params = tmp_path / "params"
    params.mkdir()
    Chem.MolToMolFile(mol, str(params / "LIG.sdf"))

    stable = _write_stable_boresch_distance(
        stable_path=tmp_path / "stable_boresch_distance.json",
        system_root=tmp_path,
        sim=SimpleNamespace(min_adis=3.0, max_adis=7.0),
        sim_val=_stable_distance_validator(
            universe=validator_u,
            residue_name="LIG",
            directory=tmp_path,
            protein_anchor_masks=[],
        ),
        ligand_label="pose",
        residue_name="LIG",
        universe=validator_u,
        preference_universe=preference_u,
        tail_fraction=1.0,
        mode="test",
        prolif_record={
            "usable": True,
            "persistent_protein_residues": [
                {
                    "resid": 86,
                    "resname": "ASP",
                    "interactions": [
                        {"interaction": "Cationic", "occupancy": 1.0}
                    ],
                }
            ],
        },
    )

    assert stable["ligand"]["name"] == "N1"
    assert stable["salt_bridge_preference"]["ligand_atom_names"] == ["N1"]
    assert stable["salt_bridge_preference"]["protein_residue_ids"] == [86]


def _write_dssp_tier_test_system(
    tmp_path: Path,
    *,
    non_loop_candidate_in_distance_window: bool,
):
    protein_x = [30.0, 30.0, 0.0, 20.0, 25.0, 30.0, 30.0, 30.0]
    if not non_loop_candidate_in_distance_window:
        protein_x[2] = 30.0
    pdb = tmp_path / "dssp_tier.pdb"
    lines = [
        _atom_line(
            index + 1,
            "CA",
            "ALA",
            "A",
            resid,
            protein_x[index],
            0.0,
            0.0,
            "C",
        )
        for index, resid in enumerate(range(10, 18))
    ]
    lines.extend(
        [
            _atom_line(9, "CA", "VAL", "A", 93, 1.0, 0.0, 0.0, "C"),
            _atom_line(10, "N3", "LIG", "L", 300, 5.0, 0.0, 0.0, "N"),
            "TER\n",
            "END\n",
        ]
    )
    pdb.write_text("".join(lines))
    all_ligands = tmp_path / "all-ligands"
    all_ligands.mkdir()
    (all_ligands / "manifest.json").write_text(
        json.dumps({"dssp": {"results": [[*(["H"] * 8), "-"]]}}) + "\n"
    )
    return mda.Universe(str(pdb))


def _loop_hbond_prolif_record() -> dict:
    return {
        "usable": True,
        "occupancy_threshold": 0.3,
        "persistent_protein_residues": [
            {
                "resid": 93,
                "resname": "VAL",
                "max_occupancy": 1.0,
                "interactions": [
                    {
                        "interaction": "HBDonor",
                        "occupancy": 1.0,
                        "ligand_atom_names": ["N3"],
                    }
                ],
            }
        ],
    }


def test_stable_boresch_distance_filters_loop_interaction_before_all_non_loop_ca(
    tmp_path: Path,
) -> None:
    universe = _write_dssp_tier_test_system(
        tmp_path,
        non_loop_candidate_in_distance_window=True,
    )

    stable = _write_stable_boresch_distance(
        stable_path=tmp_path / "stable_boresch_distance.json",
        system_root=tmp_path,
        sim=SimpleNamespace(min_adis=3.0, max_adis=7.0),
        sim_val=_stable_distance_validator(
            universe=universe,
            residue_name="LIG",
            directory=tmp_path,
            protein_anchor_masks=[],
        ),
        ligand_label="pose",
        residue_name="LIG",
        universe=universe,
        tail_fraction=1.0,
        mode="test",
        prolif_record=_loop_hbond_prolif_record(),
    )

    assert stable["protein"]["resid"] == 12
    preference = stable["protein_candidate_preference"]
    assert preference["selected_tier"] == "dssp_non_loop_all_ca"
    assert preference["loop_fallback_used"] is False
    assert stable["prolif_preference"][
        "excluded_from_non_loop_persistent_residue_ids"
    ] == [93]
    assert stable["prolif_preference"]["ligand_atom_names"] == []


def test_stable_boresch_distance_uses_loop_interaction_only_as_fallback(
    tmp_path: Path,
) -> None:
    universe = _write_dssp_tier_test_system(
        tmp_path,
        non_loop_candidate_in_distance_window=False,
    )

    stable = _write_stable_boresch_distance(
        stable_path=tmp_path / "stable_boresch_distance.json",
        system_root=tmp_path,
        sim=SimpleNamespace(min_adis=3.0, max_adis=7.0),
        sim_val=_stable_distance_validator(
            universe=universe,
            residue_name="LIG",
            directory=tmp_path,
            protein_anchor_masks=[],
        ),
        ligand_label="pose",
        residue_name="LIG",
        universe=universe,
        tail_fraction=1.0,
        mode="test",
        prolif_record=_loop_hbond_prolif_record(),
    )

    assert stable["protein"]["resid"] == 93
    preference = stable["protein_candidate_preference"]
    assert preference["selected_tier"] == "loop_interactions_fallback"
    assert preference["loop_fallback_used"] is True
    assert preference["attempts"][1]["tier"] == "dssp_non_loop_all_ca"
    assert preference["attempts"][1]["status"] == "failed"
    assert stable["prolif_preference"]["ligand_atom_names"] == ["N3"]


def test_prolif_artifact_writer_saves_timeseries_and_pngs(tmp_path: Path) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("LIG300.A", "ASP42.A", "HBAcceptor"),
            ("LIG300.A", "GLY77.A", "Hydrophobic"),
        ]
    )
    df = pd.DataFrame(
        [
            [True, False],
            [False, True],
            [True, True],
        ],
        index=[10, 11, 12],
        columns=columns,
    )
    interactions, _persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.5,
    )

    artifacts, errors = _write_prolif_artifacts(
        prolif_path=tmp_path / "prolif_interactions.json",
        df=df,
        interactions=interactions,
        ligand_label="LIG",
    )

    assert errors == {}
    assert set(artifacts) == set(PROLIF_ARTIFACT_FILENAMES)
    timeseries = pd.read_csv(tmp_path / artifacts["timeseries_csv_gz"])
    assert timeseries["frame"].tolist() == [10, 11, 12]
    assert "LIG300.A|ASP42.A|HBAcceptor" in timeseries.columns
    lignetwork = tmp_path / artifacts["lignetwork_html"]
    assert lignetwork.exists()
    assert "LigNetwork unavailable" in lignetwork.read_text()
    for key in ("barcode_png", "occupancy_png", "interaction_diagram_png"):
        path = tmp_path / artifacts[key]
        assert path.exists()
        assert path.stat().st_size > 0


def test_equil_analysis_artifacts_are_copied_to_results_folder(tmp_path: Path) -> None:
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir()
    (equil_dir / "prolif_interactions.json").write_text(
        json.dumps({"schema_version": PROLIF_INTERACTIONS_SCHEMA_VERSION}) + "\n"
    )
    (equil_dir / "prolif_interactions_barcode.png").write_bytes(b"png")
    (equil_dir / "simulation_analysis.png").write_bytes(b"png")

    _copy_equil_analysis_artifacts(equil_dir)

    results_dir = equil_dir / "results"
    legacy_dir = equil_dir / "artifacts"
    assert (results_dir / "prolif_interactions.json").exists()
    assert (results_dir / "prolif_interactions_barcode.png").exists()
    assert (results_dir / "simulation_analysis.png").exists()
    assert "prolif_interactions_timeseries.csv.gz" in (
        results_dir / "README.txt"
    ).read_text()
    assert (legacy_dir / "prolif_interactions.json").exists()


def test_run_prolif_fingerprint_disables_progress_when_supported() -> None:
    class FakeFingerprint:
        def __init__(self):
            self.progress = None

        def run(self, trajectory, ligand, protein, *, progress=True):
            self.progress = progress

    fp = FakeFingerprint()

    _run_prolif_fingerprint(fp, object(), object(), object())

    assert fp.progress is False


def test_single_frame_prolif_writer_uses_unsliced_trajectory_and_original_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrajectory:
        def __len__(self):
            return 1

        def __getitem__(self, item):
            if isinstance(item, slice):
                raise AssertionError("one-frame trajectory must not be sliced")
            return SimpleNamespace(frame=int(item))

    class FakeSelection:
        n_atoms = 1

        def __iter__(self):
            return iter(())

    class FakeUniverse:
        def __init__(self):
            self.trajectory = FakeTrajectory()

        @staticmethod
        def select_atoms(selection):
            assert selection in {"resname LIG", "protein"}
            return FakeSelection()

    trajectory = None

    class FakeFingerprint:
        def __init__(self):
            self.ifp = {}

        def run(self, received, ligand, protein, *, progress=True):
            nonlocal trajectory
            trajectory = received
            assert progress is False

        @staticmethod
        def to_dataframe():
            return pd.DataFrame(
                [[True]],
                columns=pd.MultiIndex.from_tuples(
                    [("LIG290.0", "ASP84.0", "Cationic")]
                ),
            )

    class FakeProlif:
        __version__ = "test"
        Fingerprint = FakeFingerprint

    monkeypatch.setitem(sys.modules, "prolif", FakeProlif)
    monkeypatch.setattr(
        "batter.exec.handlers.equil_analysis._write_prolif_artifacts",
        lambda **kwargs: ({}, {}),
    )
    renum = tmp_path / "protein_renum.txt"
    renum.write_text("ASP A 147 ASP 83\n")
    universe = FakeUniverse()

    record = _write_prolif_interactions(
        prolif_path=tmp_path / "prolif_interactions.json",
        universe=universe,
        ligand_label="POSE_A",
        residue_name="LIG",
        tail_fraction=1.0,
        mode="single_frame_no_equil",
        protein_renum_path=renum,
    )

    assert trajectory is universe.trajectory
    assert record["usable"] is True
    assert record["schema_version"] == PROLIF_INTERACTIONS_SCHEMA_VERSION
    assert record["n_frames"] == 1
    assert record["protein_residue_numbering"] == "input"
    assert record["interactions"][0]["protein"]["label"] == "ASP147.A"
    assert record["persistent_protein_residues"][0]["prepared_resid"] == 84


def test_no_equil_representative_universe_uses_cpptraj_pdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeMDA:
        @staticmethod
        def Universe(*args, **kwargs):
            calls.append((args, kwargs))
            return "universe"

    rep_pdb = tmp_path / "representative.pdb"
    rep_pdb.write_text("END\n")
    monkeypatch.setattr("batter.exec.handlers.equil_analysis._mda", lambda: FakeMDA)

    assert _load_no_equil_representative_universe(rep_pdb) == "universe"
    assert calls == [((str(rep_pdb),), {})]


def test_no_equil_representative_universe_attaches_solute_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAtoms:
        def __init__(self, names, resnames, positions):
            self.names = np.asarray(names)
            self.resnames = np.asarray(resnames)
            self.positions = np.asarray(positions, dtype=np.float32)
            self.n_atoms = len(self.names)

        def __getitem__(self, item):
            return FakeAtoms(
                self.names[item],
                self.resnames[item],
                self.positions[item],
            )

    class FakeUniverse:
        def __init__(self, atoms, dimensions=None):
            self.atoms = atoms
            self.dimensions = dimensions
            self.loaded_coordinates = None
            self.bonds = ["topology-bond"]

        def load_new(self, coordinates):
            self.loaded_coordinates = np.asarray(coordinates)
            self.atoms.positions = self.loaded_coordinates.copy()

    pdb_universe = FakeUniverse(
        FakeAtoms(
            ["Pb", "CA", "N1", "EP"],
            ["DUM", "ALA", "LIG", "WAT"],
            [[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ),
        dimensions=np.asarray([10, 11, 12, 90, 90, 90], dtype=np.float32),
    )
    topology_universe = FakeUniverse(
        FakeAtoms(
            ["Pb", "CA", "N1"],
            ["DUM", "ALA", "LIG"],
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        )
    )

    rep_pdb = tmp_path / "representative.pdb"
    rep_pdb.write_text("END\n")
    solute_topology = tmp_path / "vac.prmtop"
    solute_topology.write_text("topology\n")

    class FakeMDA:
        @staticmethod
        def Universe(path):
            if path == str(rep_pdb):
                return pdb_universe
            if path == str(solute_topology):
                return topology_universe
            raise AssertionError(path)

    monkeypatch.setattr("batter.exec.handlers.equil_analysis._mda", lambda: FakeMDA)

    result = _load_no_equil_representative_universe(rep_pdb, solute_topology)

    assert result is topology_universe
    assert result.bonds == ["topology-bond"]
    np.testing.assert_allclose(
        result.loaded_coordinates,
        [[0, 0, 0], [1, 2, 3], [4, 5, 6]],
    )
    np.testing.assert_allclose(result.dimensions, pdb_universe.dimensions)


@pytest.mark.parametrize("with_topology", [True, False])
def test_representative_only_prolif_maps_only_prepared_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_topology: bool,
) -> None:
    equil = tmp_path / "equil"
    equil.mkdir()
    rep_pdb = equil / "representative.pdb"
    rep_pdb.write_text("END\n")
    vac_topology = equil / "vac.prmtop"
    if with_topology:
        vac_topology.write_text("topology\n")
    renum = equil / "protein_renum.txt"
    renum.write_text("ASP A 147 ASP 83\n")
    calls = {}

    def fake_loader(pdb_path, topology_path=None):
        calls["loader"] = (pdb_path, topology_path)
        return "universe"

    def fake_writer(**kwargs):
        calls["writer"] = kwargs
        return {"usable": True}

    monkeypatch.setattr(
        "batter.exec.handlers.equil_analysis._load_no_equil_representative_universe",
        fake_loader,
    )
    monkeypatch.setattr(
        "batter.exec.handlers.equil_analysis._write_prolif_interactions",
        fake_writer,
    )
    paths = {
        "equil_dir": equil,
        "rep_pdb": rep_pdb,
        "prolif_interactions": equil / "prolif_interactions.json",
        "prot_renum": renum,
    }

    record = _write_representative_only_prolif(
        paths=paths,
        ligand_label="POSE_A",
        residue_name="LIG",
    )

    assert record == {"usable": True}
    assert calls["loader"] == (
        rep_pdb,
        vac_topology if with_topology else None,
    )
    assert calls["writer"]["protein_renum_path"] == (
        renum if with_topology else None
    )


def test_write_prolif_lignetwork_html_uses_prolif_plot_lignetwork(
    tmp_path: Path,
) -> None:
    class FakeMolecule:
        @staticmethod
        def from_mda(selection):
            return {"selection": selection}

    class FakeProlif:
        Molecule = FakeMolecule

    class FakeView:
        def save(self, path):
            path.write_text("<html>native LigNetwork</html>\n")

    class FakeFingerprint:
        def __init__(self):
            self.calls = []

        def plot_lignetwork(self, ligand_mol, **kwargs):
            self.calls.append((ligand_mol, kwargs))
            return FakeView()

    fp = FakeFingerprint()
    path = tmp_path / "network.html"

    _write_prolif_lignetwork_html(
        fingerprint=fp,
        ligand_selection="ligand-selection",
        prolif_module=FakeProlif,
        path=path,
        threshold=0.3,
    )

    assert path.read_text() == "<html>native LigNetwork</html>\n"
    assert fp.calls == [
        (
            {"selection": "ligand-selection"},
            {
                "kind": "aggregate",
                "threshold": 0.3,
                "height": "650px",
                "show_interaction_data": True,
            },
        )
    ]


def test_write_prolif_lignetwork_html_uses_original_protein_labels(
    tmp_path: Path,
) -> None:
    class FakeMolecule:
        @staticmethod
        def from_mda(selection):
            return {"selection": selection}

    class FakeProlif:
        Molecule = FakeMolecule

    class FakeView:
        def save(self, path):
            path.write_text(
                '<html>{"id":"ASP84.0","next":"ASP147.0",'
                '"to":"ILE259.0"}</html>\n'
            )

    class FakeFingerprint:
        @staticmethod
        def plot_lignetwork(*args, **kwargs):
            return FakeView()

    path = tmp_path / "network.html"
    _write_prolif_lignetwork_html(
        fingerprint=FakeFingerprint(),
        ligand_selection="ligand-selection",
        prolif_module=FakeProlif,
        path=path,
        threshold=0.3,
        protein_residue_map={
            84: {
                "resid": 147,
                "resname": "ASP",
                "chainID": "A",
                "prepared_resname": "ASP",
            },
            259: {
                "resid": 322,
                "resname": "ILE",
                "chainID": "A",
                "prepared_resname": "ILE",
            },
            147: {
                "resid": 210,
                "resname": "ASP",
                "chainID": "A",
                "prepared_resname": "ASP",
            },
        },
    )

    contents = path.read_text()
    assert '"id":"ASP147.A"' in contents
    assert '"next":"ASP210.A"' in contents
    assert "ILE322.A" in contents
    assert "ASP84" not in contents
    assert "ILE259" not in contents


def test_prolif_artifact_writer_falls_back_when_lignetwork_renderer_fails(
    tmp_path: Path,
) -> None:
    columns = pd.MultiIndex.from_tuples(
        [("LIG300.A", "ASP42.A", "HBAcceptor")]
    )
    df = pd.DataFrame([[True], [False], [True]], columns=columns)
    interactions, _persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.5,
    )

    class FakeMolecule:
        @staticmethod
        def from_mda(selection):
            return {"selection": selection}

    class FakeProlif:
        Molecule = FakeMolecule

    class FakeFingerprint:
        def plot_lignetwork(self, ligand_mol, **kwargs):
            raise KeyError("ligand")

    artifacts, errors = _write_prolif_artifacts(
        prolif_path=tmp_path / "prolif_interactions.json",
        df=df,
        interactions=interactions,
        ligand_label="LIG",
        fingerprint=FakeFingerprint(),
        ligand_selection="ligand-selection",
        prolif_module=FakeProlif,
    )

    assert errors == {}
    lignetwork = tmp_path / artifacts["lignetwork_html"]
    text = lignetwork.read_text()
    assert "ProLIF LigNetwork unavailable" in text
    assert "ligand" in text
