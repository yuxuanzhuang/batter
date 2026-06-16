from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import MDAnalysis as mda

# sim_validation imports networkx unconditionally but does not use it in this test.
sys.modules.setdefault("networkx", types.ModuleType("networkx"))

from batter.analysis.sim_validation import SimValidator


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


def _make_test_universe(tmp_path: Path) -> mda.Universe:
    pdb = tmp_path / "full.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 92, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "ALA", "A", 61, 100.0, 0.0, 0.0, "C"),
        _atom_line(3, "CA", "ALA", "A", 257, 0.0, 100.0, 0.0, "C"),
        _atom_line(4, "C1", "LIG", "A", 300, 3.0, 0.0, 0.0, "C"),
        _atom_line(5, "C2", "LIG", "A", 300, 50.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    return mda.Universe(str(pdb))


def _make_validator(
    u: mda.Universe, workdir: Path, protein_anchor_masks: list[str] | None = None
) -> SimValidator:
    validator = SimValidator.__new__(SimValidator)
    validator.universe = u
    validator.workdir = workdir
    validator.ligand = "LIG"
    validator.protein_anchor_masks = protein_anchor_masks or []
    validator.results = {}
    return validator


class _DummyTrajectory:
    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.n_frames


class _DummyUniverse:
    def __init__(self, n_frames: int) -> None:
        self.trajectory = _DummyTrajectory(n_frames)


def test_ligand_bs_uses_initial_binding_site_atoms(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    validator = _make_validator(u, tmp_path, [":61@CA", ":257@CA", ":61@CA"])
    validator._ligand_bs()

    assert np.allclose(validator.results["ligand_bs"], np.array([3.0]))


def test_ligand_bs_requires_site_or_anchor_atoms(tmp_path: Path) -> None:
    pdb = tmp_path / "far.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 92, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "C1", "LIG", "A", 300, 30.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    validator = _make_validator(u, tmp_path)

    with pytest.raises(ValueError, match="binding-site atoms"):
        validator._ligand_bs()


def test_representative_snapshot_uses_last_quarter_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _make_validator(_DummyUniverse(8), tmp_path)
    called = {}

    def fake_ligand_dihedral(start_frame: int = 0) -> None:
        called["start_frame"] = start_frame
        validator.results["ligand_dihedrals"] = np.array(
            [
                [90.0, -90.0],
                [90.0, -90.0],
            ]
        )
        validator.results["ligand_dihedral_frame_indices"] = np.array([6, 7])

    monkeypatch.setattr(validator, "_ligand_dihedral", fake_ligand_dihedral)

    rep_idx = validator.find_representative_snapshot(
        savefig=True, output_filename="tail_dihed_hist.png"
    )

    assert called["start_frame"] == 6
    assert rep_idx == 6
    assert validator.results["representative_frame_index"] == 6
    assert validator.results["representative_analysis_start_frame"] == 6
