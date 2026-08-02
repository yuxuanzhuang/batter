from pathlib import Path

from batter._internal.ops import restraints


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
