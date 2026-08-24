from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import MDAnalysis as mda
import pandas as pd

import batter._internal.ops.build_complex as build_complex_mod
from batter.analysis.sim_validation import STABLE_BORESCH_DISTANCE_SCHEMA_VERSION
from batter._internal.ops.build_complex import (
    _apply_stable_boresch_distance_preference,
    _load_stable_boresch_distance,
    _user_anchor_triplet_was_provided,
    _user_p1_was_provided,
)


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


def test_stable_boresch_preference_updates_p1_and_l1_target(tmp_path: Path) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "CA", "ALA", "A", 20, 8.0, 0.0, 0.0, "C"),
                _atom_line(3, "CA", "ALA", "A", 30, 8.0, 8.0, 0.0, "C"),
                _atom_line(4, "C1", "LIG", "A", 100, 4.0, 0.0, 0.0, "C"),
                _atom_line(5, "C2", "LIG", "A", 100, 5.0, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    preference = _apply_stable_boresch_distance_preference(
        u=u,
        mol="LIG",
        stable_record={
            "protein": {"resid": 10, "name": "CA"},
            "ligand": {"name": "C1"},
            "distance": {"std": 0.2},
            "vector": {"mean": [4.0, 0.0, 0.0]},
        },
        P1=":20@CA",
        P2=":20@CA",
        P3=":30@CA",
        lig_name_str="C1 C2",
        l1_x=1.0,
        l1_y=1.0,
        l1_z=1.0,
        l1_range=6.0,
    )

    assert preference is not None
    assert preference["P1"] == ":10@CA"
    assert preference["p1_vmd"] == "10"
    assert preference["lig_name_str"].split()[0] == "C1"
    assert np.allclose(
        [preference["l1_x"], preference["l1_y"], preference["l1_z"]],
        [4.0, 0.0, 0.0],
    )
    assert np.isclose(preference["l1_range"], 6.0)


def test_stable_boresch_preference_renumbers_original_residue_ids(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "CYS", "A", 86, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "CA", "VAL", "A", 118, 20.0, 0.0, 0.0, "C"),
                _atom_line(3, "CA", "THR", "A", 87, 8.0, 0.0, 0.0, "C"),
                _atom_line(4, "CA", "ILE", "A", 132, 8.0, 8.0, 0.0, "C"),
                _atom_line(5, "C13", "LIG", "A", 100, 4.0, 0.0, 0.0, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))
    renum_data = pd.DataFrame(
        [
            {
                "old_resname": "CYS",
                "old_chain": "A",
                "old_resid": 118,
                "new_resname": "CYS",
                "new_resid": 85,
            }
        ]
    )

    preference = _apply_stable_boresch_distance_preference(
        u=u,
        mol="LIG",
        stable_record={
            "protein": {"resid": 118, "resname": "CYS", "name": "CA", "segid": "A"},
            "ligand": {"name": "C13"},
            "distance": {"std": 0.1},
            "vector": {"mean": [4.0, 0.0, 0.0]},
        },
        P1=":82@CA",
        P2=":87@CA",
        P3=":132@CA",
        lig_name_str="C13 C2",
        l1_x=1.0,
        l1_y=1.0,
        l1_z=1.0,
        l1_range=8.0,
        renum_data=renum_data,
    )

    assert preference is not None
    assert preference["P1"] == ":86@CA"
    assert preference["p1_vmd"] == "86"


def test_stable_boresch_preference_does_not_renumber_prepared_residue_twice(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ASP", "A", 85, 0.0, 0.0, 0.0, "C"),
                _atom_line(2, "CA", "GLY", "A", 58, 20.0, 0.0, 0.0, "C"),
                _atom_line(3, "CA", "VAL", "A", 51, 8.0, 0.0, 0.0, "C"),
                _atom_line(4, "CA", "THR", "A", 262, 8.0, 8.0, 0.0, "C"),
                _atom_line(5, "N1", "LIG", "A", 286, 4.0, 0.0, 0.0, "N"),
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))
    renum_data = pd.DataFrame(
        [
            {
                "old_resname": "ASP",
                "old_chain": "A",
                "old_resid": 113,
                "new_resname": "ASP",
                "new_resid": 84,
            },
            {
                "old_resname": "GLY",
                "old_chain": "A",
                "old_resid": 85,
                "new_resname": "GLY",
                "new_resid": 57,
            },
        ]
    )

    preference = _apply_stable_boresch_distance_preference(
        u=u,
        mol="LIG",
        stable_record={
            "protein": {"resid": 85, "resname": "ASP", "name": "CA"},
            "ligand": {"name": "N1"},
            "distance": {"std": 0.0},
            "vector": {"mean": [4.0, 0.0, 0.0]},
        },
        P1=":85@CA",
        P2=":51@CA",
        P3=":262@CA",
        lig_name_str="N1",
        l1_x=1.0,
        l1_y=1.0,
        l1_z=1.0,
        l1_range=8.0,
        renum_data=renum_data,
    )

    assert preference is not None
    assert preference["P1"] == ":85@CA"


def test_user_anchor_triplet_detects_fully_explicit_config() -> None:
    assert not _user_anchor_triplet_was_provided({"user_anchor_atoms": []})
    assert not _user_anchor_triplet_was_provided(
        {"user_anchor_atoms": ["resid 10 and name CA"]}
    )
    assert _user_anchor_triplet_was_provided(
        {"user_anchor_atoms": ["a", "b", "c"]}
    )


def test_user_p1_detects_single_explicit_anchor() -> None:
    assert not _user_p1_was_provided({"user_anchor_atoms": []})
    assert not _user_p1_was_provided({"user_anchor_atoms": ["", "  "]})
    assert _user_p1_was_provided(
        {"user_anchor_atoms": ["resid 10 and name CA"]}
    )


def test_stable_boresch_selection_keeps_user_p1(monkeypatch) -> None:
    candidates = [
        {"protein": {"mask": ":58@CA"}},
        {"protein": {"mask": ":85@CA"}},
    ]

    def _fake_apply(**kwargs):
        return {
            "P1": kwargs["stable_record"]["protein"]["mask"],
            "stable_ligand_name": "N1",
        }

    monkeypatch.setattr(
        build_complex_mod,
        "_apply_stable_boresch_distance_preference",
        _fake_apply,
    )
    monkeypatch.setattr(
        build_complex_mod,
        "_stable_preference_has_safe_boresch_frame",
        lambda **_kwargs: True,
    )

    preference = build_complex_mod._select_stable_boresch_distance_preference(
        u=object(),
        mol="LIG",
        stable_record={"ranked_pairs": candidates},
        P1=":85@CA",
        P2=":51@CA",
        P3=":262@CA",
        lig_name_str="N1 C2 C3",
        l1_x=1.0,
        l1_y=1.0,
        l1_z=1.0,
        l1_range=8.0,
        required_P1=":85@CA",
    )

    assert preference is not None
    assert preference["P1"] == ":85@CA"
    assert preference["stable_rank"] == 2


def test_stable_boresch_loader_ignores_stale_and_unusable_records(
    tmp_path: Path,
) -> None:
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir()
    stable_path = equil_dir / "stable_boresch_distance.json"

    stable_path.write_text(json.dumps({"schema_version": 1}) + "\n")
    assert _load_stable_boresch_distance(equil_dir) is None

    stable_path.write_text(
        json.dumps({"schema_version": 4, "usable": False, "reason": "no pair"})
        + "\n"
    )
    assert _load_stable_boresch_distance(equil_dir) is None

    current = {
        "schema_version": STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
        "usable": True,
        "protein": {"resid": 10, "name": "CA"},
        "ligand": {"name": "C1"},
    }
    stable_path.write_text(json.dumps(current) + "\n")
    assert _load_stable_boresch_distance(equil_dir) == current
