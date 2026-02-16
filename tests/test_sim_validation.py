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


def _make_validator(u: mda.Universe, workdir: Path) -> SimValidator:
    validator = SimValidator.__new__(SimValidator)
    validator.universe = u
    validator.workdir = workdir
    validator.ligand = "LIG"
    validator.results = {}
    return validator


def test_ligand_bs_uses_min_distance_to_anchor_atoms(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    anchors_dir = tmp_path / "q_build_files"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    (anchors_dir / "protein_anchors.txt").write_text(":92@CA\n:61@CA\n:257@CA\n")

    validator = _make_validator(u, tmp_path)
    validator._ligand_bs()

    assert np.allclose(validator.results["ligand_bs"], np.array([3.0]))


def test_ligand_bs_requires_three_anchor_atoms(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    validator = _make_validator(u, tmp_path)

    with pytest.raises(ValueError, match="three protein anchor atoms"):
        validator._ligand_bs()
