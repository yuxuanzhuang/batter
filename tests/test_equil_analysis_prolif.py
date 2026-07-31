from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter.exec.handlers.equil_analysis import (
    PROLIF_ARTIFACT_FILENAMES,
    _copy_equil_analysis_artifacts,
    _equil_anchor_masks_to_original_resids,
    _load_equil_anchor_masks,
    _persistent_prolif_residue_ids,
    _persistent_prolif_residue_priorities,
    _prolif_interaction_id,
    _prolif_residue_metadata,
    _records_from_prolif_dataframe,
    _run_prolif_fingerprint,
    _salt_bridge_ligand_atom_preference,
    _write_prolif_lignetwork_html,
    _write_prolif_artifacts,
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
    (equil_dir / "prolif_interactions.json").write_text('{"schema_version": 3}\n')
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
