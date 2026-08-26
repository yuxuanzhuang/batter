import json
from pathlib import Path

from batter._internal.ops import restraints


def _pdb_atom(
    serial: int,
    name: str,
    resname: str,
    resid: int,
    x: float,
) -> str:
    record = "ATOM" if resname == "ALA" else "HETATM"
    return (
        f"{record:<6}{serial:5d} {name:<4} {resname:>3} A{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    )


def test_collect_calpha_and_lig_uses_non_loop_protein_group(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "all-ligands"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dssp": {
                    "results": [["-", "H", "H", "H", "H", "-", "E", "E"]]
                }
            }
        )
    )
    vac_pdb = tmp_path / "vac.pdb"
    vac_pdb.write_text(
        "".join(
            [
                _pdb_atom(resid, "CA", "ALA", resid, float(resid))
                for resid in range(1, 9)
            ]
            + [
                _pdb_atom(9, "C1", "LIG", 9, 9.0),
                _pdb_atom(10, "C2", "LIG", 9, 10.0),
                "END\n",
            ]
        )
    )

    protein_serials, ligand_serials = restraints._collect_calpha_and_lig(
        vac_pdb,
        "9",
        system_root=tmp_path,
    )

    assert protein_serials == ["2", "3", "4", "5"]
    assert ligand_serials == ["9", "10"]


def test_canonicalize_restraint_expr_uses_vac_pdb_atom_case() -> None:
    atm_num = ["0", ":220@C9", ":220@C5", ":220@C4", ":220@Cl1"]

    expr = restraints._canonicalize_restraint_expr(
        ":220@C9 :220@C5 :220@C4 :220@CL1",
        atm_num,
    )

    assert expr == ":220@C9 :220@C5 :220@C4 :220@Cl1"
    assert restraints._mask_index(atm_num, ":220@CL1") == 4


def test_write_assign_values_are_keyed_by_cpptraj_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run_with_log(command: str, working_dir: Path) -> None:
        assert "cpptraj" in command
        (working_dir / "assign.dat").write_text(
            "#Frame r0 r2 r3\n"
            "1 10.0 30.0 40.0\n"
        )

    monkeypatch.setattr(restraints, "run_with_log", fake_run_with_log)

    vals = restraints._write_assign_and_read_vals(
        tmp_path,
        [
            ":1@A :1@B",
            ":1@B :1@C",
            ":1@C :1@D",
            ":1@D :1@E",
        ],
        tmp_path / "dummy.prmtop",
        tmp_path / "dummy.inpcrd",
    )

    assert vals == [10.0, None, 30.0, 40.0]


def test_ligand_dihedral_force_helper_uses_active_force() -> None:
    assert (
        restraints._ligand_dihedral_force_constant(
            ":1@C11 :1@C12 :1@C13 :1@Cl2",
            10.0,
        )
        == 10.0
    )
    assert (
        restraints._ligand_dihedral_force_constant(
            ":1@C11 :1@C12 :1@C13 :1@Cl2",
            0.0,
        )
        == 0.0
    )


def test_missing_zero_force_ligand_dihedral_reference_uses_zero() -> None:
    val = restraints._ligand_dihedral_reference_value(
        [12.0, None],
        1,
        ":1@A :1@B :1@C :1@D",
        0.0,
        "z",
    )

    assert val == 0.0


def test_missing_nonzero_ligand_dihedral_reference_is_skipped() -> None:
    val = restraints._ligand_dihedral_reference_value(
        [12.0, None],
        1,
        ":1@A :1@B :1@C :1@D",
        10.0,
        "l",
    )

    assert val is None
