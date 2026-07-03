from __future__ import annotations

from pathlib import Path

from batter._internal.ops import box


def _pdb_atom(
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
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def _overflow_pdb_atom(
    serial: int,
    name: str,
    resname: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3}  {resid:5d}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def test_cleanup_periodic_water_pdbs_removes_duplicate_boundary_water(
    tmp_path: Path,
) -> None:
    water_pdb = tmp_path / "solvate_pre_wat_00.pdb"
    report = tmp_path / "periodic_water_cleanup.json"
    water_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "O", "WAT", "W", 1, 1.000, 1.000, 9.800, "O"),
                _pdb_atom(2, "H1", "WAT", "W", 1, 1.100, 1.000, 9.800, "H"),
                _pdb_atom(3, "H2", "WAT", "W", 1, 1.000, 1.100, 9.800, "H"),
                "TER\n",
                _pdb_atom(4, "O", "WAT", "W", 2, 1.100, 1.000, -0.100, "O"),
                _pdb_atom(5, "H1", "WAT", "W", 2, 1.200, 1.000, -0.100, "H"),
                _pdb_atom(6, "H2", "WAT", "W", 2, 1.100, 1.100, -0.100, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )

    summary = box._cleanup_periodic_water_pdbs(
        [water_pdb],
        box=[10.0, 10.0, 10.0],
        report_path=report,
    )

    assert summary["removed_water_residues"] == 1
    assert summary["removed_water_water"] == 1
    assert summary["kept_water_residues_by_file"] == {"solvate_pre_wat_00.pdb": 1}
    assert report.exists()
    assert sum(1 for line in water_pdb.read_text().splitlines() if " WAT " in line) == 3


def test_cleanup_periodic_water_pdbs_removes_boundary_water_near_nonwater(
    tmp_path: Path,
) -> None:
    water_pdb = tmp_path / "solvate_pre_wat_00.pdb"
    nonwater_pdb = tmp_path / "solvate_pre_others.pdb"
    water_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "O", "WAT", "W", 1, 1.000, 1.000, -0.100, "O"),
                _pdb_atom(2, "H1", "WAT", "W", 1, 1.100, 1.000, -0.100, "H"),
                _pdb_atom(3, "H2", "WAT", "W", 1, 1.000, 1.100, -0.100, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )
    nonwater_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "Cl-", "Cl-", "X", 1, 1.000, 1.000, 9.850, "CL"),
                "TER\n",
                "END\n",
            ]
        )
    )

    summary = box._cleanup_periodic_water_pdbs(
        [water_pdb],
        nonwater_pdbs=[nonwater_pdb],
        box=[10.0, 10.0, 10.0],
    )

    assert summary["removed_water_residues"] == 1
    assert summary["removed_water_nonwater"] == 1
    assert summary["kept_water_residues_by_file"] == {"solvate_pre_wat_00.pdb": 0}
    assert not water_pdb.exists()
    assert nonwater_pdb.exists()


def test_cleanup_periodic_water_pdbs_removes_raw_water_near_nonwater(
    tmp_path: Path,
) -> None:
    water_pdb = tmp_path / "solvate_pre_wat_00.pdb"
    nonwater_pdb = tmp_path / "solvate_pre_others.pdb"
    water_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "O", "WAT", "W", 1, 1.000, 1.000, 1.000, "O"),
                _pdb_atom(2, "H1", "WAT", "W", 1, 1.100, 1.000, 1.000, "H"),
                _pdb_atom(3, "H2", "WAT", "W", 1, 1.000, 1.100, 1.000, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )
    nonwater_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "C1", "LIG", "X", 1, 1.300, 1.000, 1.000, "C"),
                "TER\n",
                "END\n",
            ]
        )
    )

    summary = box._cleanup_periodic_water_pdbs(
        [water_pdb],
        nonwater_pdbs=[nonwater_pdb],
        box=[10.0, 10.0, 10.0],
    )

    assert summary["removed_water_residues"] == 1
    assert summary["removed_water_nonwater"] == 1
    assert summary["kept_water_residues_by_file"] == {"solvate_pre_wat_00.pdb": 0}
    assert not water_pdb.exists()


def test_renumber_pdb_atom_line_blanks_overflow_insert_code() -> None:
    original = _overflow_pdb_atom(1, "O", "WAT", 10697, 1.0, 2.0, 3.0, "O")

    renumbered = box._renumber_pdb_atom_line(original, serial=40001, resid=10000)

    assert renumbered[21:27] == "W   1 "
    assert float(renumbered[30:38]) == 1.0
    assert float(renumbered[38:46]) == 2.0
    assert float(renumbered[46:54]) == 3.0
    assert "10697" not in renumbered


def test_renumber_pdb_atom_line_recovers_existing_malformed_overflow() -> None:
    original = (
        "ATOM  39997  O   WAT W100003     32.977   8.082  60.024"
        "  1.00  0.00      SYST\n"
    )

    renumbered = box._renumber_pdb_atom_line(original, serial=1, resid=1)

    assert renumbered[21:27] == "W   1 "
    assert float(renumbered[30:38]) == 32.977
    assert float(renumbered[38:46]) == 8.082
    assert float(renumbered[46:54]) == 60.024


def test_cleanup_normalizes_water_pdbs_without_removal(tmp_path: Path) -> None:
    water_pdb = tmp_path / "solvate_pre_wat_00.pdb"
    water_pdb.write_text(
        "".join(
            [
                _overflow_pdb_atom(1, "O", "WAT", 10697, 1.0, 2.0, 3.0, "O"),
                _overflow_pdb_atom(2, "H1", "WAT", 10697, 1.1, 2.0, 3.0, "H"),
                _overflow_pdb_atom(3, "H2", "WAT", 10697, 1.0, 2.1, 3.0, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )

    summary = box._cleanup_periodic_water_pdbs(
        [water_pdb],
        box=[10.0, 10.0, 10.0],
    )

    assert summary["removed_water_residues"] == 0
    assert summary["kept_water_residues_by_file"] == {"solvate_pre_wat_00.pdb": 1}
    atom_lines = [
        line for line in water_pdb.read_text().splitlines() if line.startswith("ATOM")
    ]
    assert all(line[21:27] == "W   1 " for line in atom_lines)
    normalized_xyz = [
        float(atom_lines[0][30:38]),
        float(atom_lines[0][38:46]),
        float(atom_lines[0][46:54]),
    ]
    assert normalized_xyz == [1.0, 2.0, 3.0]
