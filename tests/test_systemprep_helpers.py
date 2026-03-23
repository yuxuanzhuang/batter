from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import pytest

from batter.systemprep.helpers import find_anchor_atoms


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


def _write_pdb(path: Path, lines: list[str]) -> None:
    path.write_text("".join([*lines, "TER\n", "END\n"]))


def _make_protein(tmp_path: Path) -> mda.Universe:
    pdb = tmp_path / "protein.pdb"
    _write_pdb(
        pdb,
        [
            _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "CB", "ALA", "A", 11, 0.0, 3.0, 0.0, "C"),
            _atom_line(3, "CG", "ALA", "A", 12, 3.0, 0.0, 0.0, "C"),
        ],
    )
    return mda.Universe(str(pdb))


def _make_ligand(tmp_path: Path, xyz: tuple[float, float, float]) -> mda.Universe:
    pdb = tmp_path / "ligand.pdb"
    x, y, z = xyz
    _write_pdb(
        pdb,
        [
            _atom_line(1, "C1", "LIG", "L", 1, x, y, z, "C"),
            _atom_line(2, "C2", "LIG", "L", 1, x + 0.5, y, z, "C"),
        ],
    )
    return mda.Universe(str(pdb))


def test_find_anchor_atoms_checks_unbound_threshold_pass(tmp_path: Path) -> None:
    u_prot = _make_protein(tmp_path)
    u_lig = _make_ligand(tmp_path, (1.0, 1.0, 0.0))

    result = find_anchor_atoms(
        u_prot=u_prot,
        u_lig=u_lig,
        lig_sdf=None,
        anchor_atoms=[
            "resid 10 and name CA",
            "resid 11 and name CB",
            "resid 12 and name CG",
        ],
        unbound_threshold=5.0,
    )

    assert result[3] == ":10@CA"
    assert result[4] == ":11@CB"
    assert result[5] == ":12@CG"
    assert result[6] > 0.0


def test_find_anchor_atoms_checks_unbound_threshold_fail(tmp_path: Path) -> None:
    u_prot = _make_protein(tmp_path)
    u_lig = _make_ligand(tmp_path, (20.0, 20.0, 20.0))

    with pytest.raises(ValueError, match="Ligand appears unbound"):
        find_anchor_atoms(
            u_prot=u_prot,
            u_lig=u_lig,
            lig_sdf=None,
            anchor_atoms=[
                "resid 10 and name CA",
                "resid 11 and name CB",
                "resid 12 and name CG",
            ],
            unbound_threshold=8.0,
        )
