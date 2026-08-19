from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np


def _load_internal_module(module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    package_roots = {
        "batter._internal": repo_root / "batter" / "_internal",
        "batter._internal.builders": repo_root / "batter" / "_internal" / "builders",
        "batter._internal.ops": repo_root / "batter" / "_internal" / "ops",
    }

    for pkg_name, pkg_path in package_roots.items():
        module = types.ModuleType(pkg_name)
        module.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = module

    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


restraints = _load_internal_module("batter._internal.ops.restraints")


class _FakeAtoms(list):
    @property
    def n_atoms(self) -> int:
        return len(self)


class _FakeAtom:
    def __init__(self, name: str, position: tuple[float, float, float]):
        self.name = name
        self.position = np.asarray(position, dtype=float)
        self.element = "C"
        self.type = "C"


class _FakeResidue:
    def __init__(self, atoms: list[_FakeAtom]):
        self.atoms = _FakeAtoms(atoms)


def _write_four_atom_pdb(path: Path, coords: list[tuple[float, float, float]]) -> None:
    lines = []
    for idx, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"HETATM{idx:5d}  C{idx:<2d} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


def test_load_common_core_indices_uses_alt_to_ref_mapping_direction(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"1": 10, "4": 12, "7": 13}))

    ref_indices, alt_indices = restraints._load_common_core_indices(mapping)

    assert ref_indices == [10, 12, 13]
    assert alt_indices == [1, 4, 7]


def test_load_common_core_indices_supports_scmask_format(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "scmk1_cc_solvent_indices": [9, "3", 5],
                "scmk2_cc_solvent_indices": ["8", 2],
            }
        )
    )

    ref_indices, alt_indices = restraints._load_common_core_indices(mapping)

    assert ref_indices == [3, 5, 9]
    assert alt_indices == [2, 8]


def test_common_core_boresch_preference_requires_more_than_three_atoms() -> None:
    assert restraints._common_core_boresch_preference_names(
        ["A1", "A2", "A3"],
        label="alternate",
    ) == []

    assert restraints._common_core_boresch_preference_names(
        ["A1", "A2", "A2", "", "A3", "A4"],
        label="alternate",
    ) == ["A1", "A2", "A3", "A4"]


def test_common_core_boresch_preference_threshold_controls_selection() -> None:
    receptor_atoms = [
        _FakeAtom("P1", (0.0, 0.0, 0.0)),
        _FakeAtom("P2", (0.0, -4.0, 0.0)),
        _FakeAtom("P3", (4.0, -4.0, 0.0)),
    ]
    residue = _FakeResidue(
        [
            _FakeAtom("R1", (1.0, 0.0, 0.0)),
            _FakeAtom("R2", (2.0, 0.0, 0.0)),
            _FakeAtom("R3", (1.0, 1.0, 0.0)),
            _FakeAtom("R4", (1.0, 0.0, 1.0)),
            _FakeAtom("R5", (3.0, 2.0, 1.0)),
        ]
    )

    below_threshold = restraints._common_core_boresch_preference_names(
        ["R1", "R2", "R3"],
        label="reference",
    )
    assert below_threshold == []

    above_threshold = restraints._common_core_boresch_preference_names(
        ["R1", "R2", "R3", "R4"],
        label="reference",
    )
    names = restraints._frame_safe_boresch_atom_names_from_residue(
        residue,
        receptor_atoms=receptor_atoms,
        label="reference",
        preferred_atom_names=above_threshold,
        preferred_first_names=above_threshold,
    )

    assert set(names) <= set(above_threshold)


def test_septop_ref_boresch_ignores_abfe_ligand_anchors() -> None:
    ref = _FakeResidue(
        [
            _FakeAtom("R1", (0.0, 0.0, 0.0)),
            _FakeAtom("R2", (1.0, 0.0, 0.0)),
            _FakeAtom("R3", (2.0, 0.0, 0.0)),
            _FakeAtom("R4", (1.0, 1.0, 0.0)),
            _FakeAtom("R5", (1.0, 0.0, 1.0)),
        ]
    )

    names = restraints._resolve_ref_boresch_atom_names(
        ref,
        anchor_names=["R1", "R2", "R3"],
    )

    coords = {atom.name: atom.position for atom in ref.atoms}
    assert len(names) == 3
    assert names != ["R1", "R2", "R3"]
    area2 = np.linalg.norm(
        np.cross(
            coords[names[1]] - coords[names[0]],
            coords[names[2]] - coords[names[0]],
        )
    )
    assert area2 > 0.25


def test_septop_alt_boresch_ignores_atom_mapping(tmp_path: Path) -> None:
    ref = _FakeResidue(
        [
            _FakeAtom("R1", (0.0, 0.0, 0.0)),
            _FakeAtom("R2", (1.0, 0.0, 0.0)),
            _FakeAtom("R3", (0.0, 1.0, 0.0)),
        ]
    )
    alt = _FakeResidue(
        [
            _FakeAtom("A0", (9.0, 9.0, 9.0)),
            _FakeAtom("A1", (0.0, 0.0, 0.0)),
            _FakeAtom("A2", (1.0, 0.0, 0.0)),
            _FakeAtom("A3", (0.0, 1.0, 0.0)),
        ]
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"1": 0, "2": 1, "3": 2}))

    names = restraints._resolve_alt_boresch_atom_names(
        ref_residue=ref,
        alt_residue=alt,
        ref_names=["R1", "R2", "R3"],
        mapping_path=mapping,
    )

    coords = {atom.name: atom.position for atom in alt.atoms}
    assert len(names) == 3
    assert names != ["A1", "A2", "A3"]
    area2 = np.linalg.norm(
        np.cross(
            coords[names[1]] - coords[names[0]],
            coords[names[2]] - coords[names[0]],
        )
    )
    assert area2 > 0.25


def test_septop_alt_boresch_empty_mapping_selects_independent_frame(
    tmp_path: Path,
) -> None:
    ref = _FakeResidue(
        [
            _FakeAtom("R1", (0.0, 0.0, 0.0)),
            _FakeAtom("R2", (1.0, 0.0, 0.0)),
            _FakeAtom("R3", (0.0, 1.0, 0.0)),
        ]
    )
    alt = _FakeResidue(
        [
            _FakeAtom("A1", (0.0, 0.0, 0.0)),
            _FakeAtom("A2", (1.0, 0.0, 0.0)),
            _FakeAtom("A3", (2.0, 0.0, 0.0)),
            _FakeAtom("A4", (1.0, 1.0, 0.0)),
            _FakeAtom("A5", (1.0, 0.0, 1.0)),
        ]
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}")

    names = restraints._resolve_alt_boresch_atom_names(
        ref_residue=ref,
        alt_residue=alt,
        ref_names=["R1", "R2", "R3"],
        mapping_path=mapping,
    )

    coords = {atom.name: atom.position for atom in alt.atoms}
    assert len(names) == 3
    assert names != ["A1", "A2", "A3"]
    area2 = np.linalg.norm(
        np.cross(
            coords[names[1]] - coords[names[0]],
            coords[names[2]] - coords[names[0]],
        )
    )
    assert area2 > 0.25


def test_septop_boresch_selection_rejects_endpoint_torsions() -> None:
    receptor_atoms = [
        _FakeAtom("P1", (49.458, 34.401, 64.572)),
        _FakeAtom("P2", (42.774, 32.383, 64.365)),
        _FakeAtom("P3", (35.111, 39.926, 64.300)),
    ]
    alt = _FakeResidue(
        [
            _FakeAtom("O1", (47.744, 45.981, 68.011)),
            _FakeAtom("C1", (46.423, 46.027, 67.494)),
            _FakeAtom("C3", (46.882, 44.481, 65.612)),
            _FakeAtom("C8", (43.478, 41.915, 66.788)),
            _FakeAtom("C10", (42.996, 39.390, 69.743)),
            _FakeAtom("C12", (44.414, 38.838, 70.098)),
        ]
    )
    atoms_by_name = {atom.name: atom for atom in alt.atoms}

    bad_values = restraints._boresch_frame_values(
        receptor_atoms,
        [atoms_by_name["C8"], atoms_by_name["O1"], atoms_by_name["C12"]],
    )
    assert bad_values is not None
    _, bad_torsion_margin = restraints._boresch_frame_margins(bad_values)
    assert bad_torsion_margin < restraints.BORESCH_MIN_TORSION_MARGIN_DEG

    names = restraints._frame_safe_boresch_atom_names_from_residue(
        alt,
        receptor_atoms=receptor_atoms,
        label="alternate",
    )
    values = restraints._boresch_frame_values(
        receptor_atoms,
        [atoms_by_name[name] for name in names],
    )
    assert values is not None
    angle_margin, torsion_margin = restraints._boresch_frame_margins(values)

    assert names != ["C8", "O1", "C12"]
    assert angle_margin >= restraints.BORESCH_MIN_ANGLE_MARGIN_DEG
    assert torsion_margin >= restraints.BORESCH_MIN_TORSION_MARGIN_DEG


def test_boresch_selection_relaxes_local_geometry_before_first_three() -> None:
    receptor_atoms = [
        _FakeAtom("P1", (0.0, -3.0, 1.0)),
        _FakeAtom("P2", (0.0, -4.0, 0.0)),
        _FakeAtom("P3", (4.0, -4.0, 0.0)),
    ]
    ligand = _FakeResidue(
        [
            _FakeAtom("A1", (0.0, 0.0, 0.0)),
            _FakeAtom("A2", (1.0, 0.0, 0.0)),
            _FakeAtom("A3", (2.0, 0.0, 0.0)),
            _FakeAtom("A4", (1.0, 0.05, 0.0)),
            _FakeAtom("A5", (1.0, 0.0, 0.05)),
        ]
    )

    names = restraints._frame_safe_boresch_atom_names_from_residue(
        ligand,
        receptor_atoms=receptor_atoms,
        label="alternate",
    )

    atoms_by_name = {atom.name: atom for atom in ligand.atoms}
    values = restraints._boresch_frame_values(
        receptor_atoms,
        [atoms_by_name[name] for name in names],
    )

    assert names != ["A1", "A2", "A3"]
    assert values is not None


def test_ligand_dihedral_reference_uses_original_input_metadata(tmp_path: Path) -> None:
    input_pdb = tmp_path / "input_pose.pdb"
    fallback_pdb = tmp_path / "l00" / "LIG.pdb"
    fallback_pdb.parent.mkdir()
    _write_four_atom_pdb(
        input_pdb,
        [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        ],
    )
    _write_four_atom_pdb(
        fallback_pdb,
        [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, -1.0),
        ],
    )

    store_dir = tmp_path / "artifacts" / "ligand_params" / "LIG"
    store_dir.mkdir(parents=True)
    (store_dir / "metadata.json").write_text(
        json.dumps({"input_path": input_pdb.as_posix()})
    )
    (tmp_path / "artifacts" / "ligand_params" / "index.json").write_text(
        json.dumps({"ligands": [{"ligand": "LigandA", "store_dir": store_dir.as_posix()}]})
    )

    ctx = types.SimpleNamespace(
        system_root=tmp_path,
        ligand="LigandA",
        residue_name="LIG",
    )

    values, source = restraints._reference_dihedral_values_from_input(
        ctx,
        fallback_pdb.parent,
        [(1, 2, 3, 4)],
    )

    expected = restraints._dihedral_degrees(
        restraints._load_reference_positions(input_pdb),
        (1, 2, 3, 4),
    )
    assert source == input_pdb
    assert abs(values[0] - expected) < 1e-6


def test_ligand_dihedral_reference_prefers_staged_input_over_cached_metadata(
    tmp_path: Path,
) -> None:
    staged_input = tmp_path / "staged_pose.pdb"
    cached_input = tmp_path / "cached_pose.pdb"
    _write_four_atom_pdb(
        staged_input,
        [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        ],
    )
    _write_four_atom_pdb(
        cached_input,
        [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, -1.0),
        ],
    )

    store_dir = tmp_path / "artifacts" / "ligand_params" / "SHARED_HASH"
    store_dir.mkdir(parents=True)
    (store_dir / "metadata.json").write_text(
        json.dumps({"input_path": cached_input.as_posix()})
    )
    (tmp_path / "artifacts" / "ligand_params" / "index.json").write_text(
        json.dumps(
            {
                "ligands": [
                    {
                        "ligand": "LigandA",
                        "store_dir": store_dir.as_posix(),
                        "input_path": staged_input.as_posix(),
                    }
                ]
            }
        )
    )

    ctx = types.SimpleNamespace(
        system_root=tmp_path,
        ligand="LigandA",
        residue_name="LIG",
    )

    values, source = restraints._reference_dihedral_values_from_input(
        ctx,
        tmp_path / "window",
        [(1, 2, 3, 4)],
    )

    expected = restraints._dihedral_degrees(
        restraints._load_reference_positions(staged_input),
        (1, 2, 3, 4),
    )
    assert source == staged_input
    assert abs(values[0] - expected) < 1e-6


def test_colvar_block_to_rst_translates_com_distance() -> None:
    block = """&colvar
 cv_type = 'COM_DISTANCE'
 cv_ni = 30, cv_i = 2,0,4777,4776,4775,4774,4786,4787,4788,4772,4771,4769,4770,4773,4778,4779,4780,4781,4782,4783,4784,4785,4794,4798,4799,4792,4793,4795,4796,4797,
 anchor_position =     0.0000,     0.0000,     0.0000,   999.0000
 anchor_strength =    10.0000,    10.0000,
/
"""

    got = restraints._colvar_block_to_rst(block)

    assert got is not None
    assert "iat=-1,-1," in got
    assert "r1=0.0, r2=0.0, r3=0.0, r4=999.0," in got
    assert "rk2=10.0, rk3=10.0," in got
    assert "igr1=2,0" in got
    assert "igr2=4777,4776,4775,4774,4786,4787,4788,4772,4771,4769,4770,4773," in got
    assert "4793,4795,4796,4797,0" in got
    assert got.rstrip().endswith("&end")


def test_colvar_block_to_rst_translates_distance() -> None:
    block = """&colvar
 cv_type = 'DISTANCE'
 cv_ni = 2, cv_i = 1323,4415
 anchor_position = 23.945, 24.245, 999, 999
 anchor_strength = 20.000000, 20.000000
/
"""

    got = restraints._colvar_block_to_rst(block)

    assert got is not None
    assert "iat=1323,4415," in got
    assert "r1=23.945, r2=24.245, r3=999.0, r4=999.0," in got
    assert "rk2=20.0, rk3=20.0," in got
    assert "igr2=" not in got
    assert got.rstrip().endswith("&end")


def test_append_colvar_rst_blocks_mirrors_tagged_extra_blocks(tmp_path: Path) -> None:
    cv_file = tmp_path / "cv.in"
    disang_file = tmp_path / "disang.rest"

    cv_file.write_text(
        "cv_file\n"
        "&colvar\n"
        " cv_type = 'DISTANCE'\n"
        " cv_ni = 2, cv_i = 11,22\n"
        " anchor_position = 1.0, 2.0, 999.0, 999.0\n"
        " anchor_strength = 5.0, 5.0\n"
        "/\n"
        "\n"
        "# EXTRA_CONFORMATIONAL_REST BEGIN\n"
        "&colvar\n"
        " cv_type = 'DISTANCE'\n"
        " cv_ni = 2, cv_i = 33,44\n"
        " anchor_position = 0.0, 0.0, 3.0, 3.3\n"
        " anchor_strength = 7.5, 7.5\n"
        "/\n"
        "# EXTRA_CONFORMATIONAL_REST END\n"
    )
    disang_file.write_text("# existing restraints\n")

    restraints._append_colvar_rst_blocks(cv_file, disang_file)

    mirrored = disang_file.read_text()
    assert "# Mirrored from cv.in" in mirrored
    assert mirrored.count("&rst") == 2
    assert "iat=11,22," in mirrored
    assert "iat=33,44," in mirrored


def test_equil_anchor_restraint_expressions_allow_single_ligand_anchor() -> None:
    exprs, ligand_count = restraints._equil_anchor_restraint_expressions(
        ":10@CA",
        ":20@CA",
        ":30@CA",
        ":385@DU1",
        None,
        None,
    )

    assert ligand_count == 3
    assert exprs == [
        ":10@CA :20@CA",
        ":20@CA :30@CA",
        ":30@CA :10@CA",
        ":10@CA :385@DU1",
        ":20@CA :10@CA :385@DU1",
        ":30@CA :20@CA :10@CA :385@DU1",
    ]
    assert all("None" not in expr for expr in exprs)


def test_build_restraints_z_allows_single_atom_ligand_anchor(tmp_path: Path) -> None:
    work_dir = tmp_path
    windows_dir = work_dir / "z00"
    windows_dir.mkdir()
    build_dir = work_dir / "z_build_files"
    build_dir.mkdir()

    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       0.000   2.000   0.000  1.00  0.00           C\n",
                "HETATM    4  NA  SOD A   4       1.000   1.000   0.000  1.00  0.00          NA\n",
                "END\n",
            ]
        )
    )
    (windows_dir / "ion.pdb").write_text(
        "HETATM    1  NA  SOD A   1       1.000   1.000   0.000  1.00  0.00          NA\nEND\n"
    )
    for path in [
        windows_dir / "vac_ligand.prmtop",
        windows_dir / "full.prmtop",
        windows_dir / "full.inpcrd",
    ]:
        path.write_text("stub\n")

    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":4@NA",
                "L2": None,
                "L3": None,
                "lig_res": "3",
            }
        )
    )

    ctx = types.SimpleNamespace(
        working_dir=work_dir,
        window_dir=windows_dir,
        equil_dir=work_dir / "equil",
        comp="z",
        ligand="ion",
        residue_name="SOD",
        sim=types.SimpleNamespace(
            hmr="no",
            dec_method="dd",
            rest=[0, 0, 5.0, 250.0, 0, 10.0, 20.0],
        ),
        extra={},
        win=0,
    )

    original_scan = restraints._scan_dihedrals_from_prmtop
    original_assign = restraints._write_assign_and_read_vals
    try:
        restraints._scan_dihedrals_from_prmtop = lambda *args, **kwargs: []
        restraints._write_assign_and_read_vals = lambda *args, **kwargs: [1.0] * 6
        restraints._build_restraints_v_o_z(None, ctx)
    finally:
        restraints._scan_dihedrals_from_prmtop = original_scan
        restraints._write_assign_and_read_vals = original_assign

    disang_text = (windows_dir / "disang.rest").read_text()
    assert "None" not in disang_text
    assert disang_text.count("#Lig_TR") == 3


def test_ion_guard_adds_one_lower_wall_per_bulk_ion_for_z(tmp_path: Path) -> None:
    windows_dir = tmp_path / "z00"
    windows_dir.mkdir()
    (windows_dir / "full.pdb").write_text(
        "".join(
            [
                "HETATM    1  NA  SOD A   1       0.000   0.000   0.000  1.00  0.00          NA\n",
                "HETATM    2  NA  SOD A   2      30.000   0.000   0.000  1.00  0.00          NA\n",
                "HETATM    3  NA  SOD A   3       7.000   0.000   0.000  1.00  0.00          NA\n",
                "HETATM    4  CL  CL  A   4      12.000   0.000   0.000  1.00  0.00          CL\n",
                "END\n",
            ]
        )
    )
    disang = windows_dir / "disang.rest"
    disang.write_text("# base restraints\n")

    ctx = types.SimpleNamespace(
        comp="z",
        window_dir=windows_dir,
        sim=types.SimpleNamespace(ion_guard="yes", cation="Na+", anion="Cl-"),
    )

    written = restraints._append_ion_guard_restraints(
        ctx,
        disang,
        ligand_resnames=["SOD"],
    )

    text = disang.read_text()
    assert written == 2
    assert text.count("#Ion_Guard") == 2
    assert "iat=3,1," in text
    assert "iat=4,1," in text
    assert "iat=3,2," not in text
    assert "iat=4,2," not in text
    assert "r2=   15.0000" in text
    assert "rk2= 10.0000000" in text
    assert "rk3=  0.0000000" in text


def test_ion_guard_can_be_disabled(tmp_path: Path) -> None:
    windows_dir = tmp_path / "z00"
    windows_dir.mkdir()
    (windows_dir / "full.pdb").write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  C1  LIG A   2      30.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  NA  NA  A   3       7.000   0.000   0.000  1.00  0.00          NA\n"
        "END\n"
    )
    disang = windows_dir / "disang.rest"
    disang.write_text("# base restraints\n")

    ctx = types.SimpleNamespace(
        comp="z",
        window_dir=windows_dir,
        sim=types.SimpleNamespace(ion_guard="no", cation="Na+", anion="Cl-"),
    )

    written = restraints._append_ion_guard_restraints(
        ctx,
        disang,
        ligand_resnames=["LIG"],
    )

    assert written == 0
    assert "#Ion_Guard" not in disang.read_text()


def test_build_restraints_l_allows_monoatomic_ion_without_dihedrals(
    tmp_path: Path,
) -> None:
    windows_dir = tmp_path / "l00"
    windows_dir.mkdir()
    build_dir = tmp_path / "l_build_files"
    build_dir.mkdir()

    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       0.000   2.000   0.000  1.00  0.00           C\n",
                "HETATM    4  NA  SOD A   4       1.000   1.000   0.000  1.00  0.00          NA\n",
                "END\n",
            ]
        )
    )
    (windows_dir / "vac_ligand.pdb").write_text(
        "HETATM    1  NA  SOD A   1       1.000   1.000   0.000  1.00  0.00          NA\nEND\n"
    )
    for path in [
        windows_dir / "vac_ligand.prmtop",
        windows_dir / "full.prmtop",
        windows_dir / "full.inpcrd",
    ]:
        path.write_text("stub\n")
    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":4@NA",
                "L2": None,
                "L3": None,
                "lig_res": "4",
            }
        )
    )

    ctx = types.SimpleNamespace(
        build_dir=build_dir,
        window_dir=windows_dir,
        comp="l",
        win=0,
        ligand="ion",
        residue_name="SOD",
        sim=types.SimpleNamespace(
            hmr="no",
            component_lambdas={"l": [0.0]},
            lig_dihcf_force=10.0,
            lig_distance_force=5.0,
            lig_angle_force=250.0,
        ),
    )

    original_scan = restraints._scan_dihedrals_from_prmtop
    try:
        restraints._scan_dihedrals_from_prmtop = lambda *args, **kwargs: []
        restraints._build_restraints_l(None, ctx)
    finally:
        restraints._scan_dihedrals_from_prmtop = original_scan

    disang_text = (windows_dir / "disang.rest").read_text()
    metadata = json.loads((windows_dir / "ligand_dihedral_restraints.json").read_text())
    assert "None" not in disang_text
    assert "#Lig_TR" not in disang_text
    assert "#Lig_D" not in disang_text
    assert metadata["boresch_restraints"] == []
    assert metadata["restraints"] == []
    assert metadata["reference_source"] is None


def test_append_x_septop_boresch_uses_reduced_small_endpoint(
    tmp_path: Path,
) -> None:
    windows_dir = tmp_path / "x00"
    windows_dir.mkdir()
    build_dir = tmp_path / "x_build_files"
    build_dir.mkdir()
    disang = windows_dir / "disang.rest"
    disang.write_text("")

    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       0.000   2.000   0.000  1.00  0.00           C\n",
                "HETATM    4  NA  REF A   4       1.000   1.000   0.000  1.00  0.00          NA\n",
                "HETATM    5  C1  ALT A   5       1.000   1.000   1.000  1.00  0.00           C\n",
                "HETATM    6  C2  ALT A   5       1.500   1.000   1.000  1.00  0.00           C\n",
                "HETATM    7  C3  ALT A   5       1.000   1.500   1.000  1.00  0.00           C\n",
                "END\n",
            ]
        )
    )
    for path in [windows_dir / "full.prmtop", windows_dir / "full.inpcrd"]:
        path.write_text("stub\n")
    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":4@NA",
                "L2": None,
                "L3": None,
                "lig_res": "4",
            }
        )
    )

    ctx = types.SimpleNamespace(
        build_dir=build_dir,
        window_dir=windows_dir,
        system_root=tmp_path,
        ligand="ref",
        residue_name="REF",
        extra={"residue_ref": "REF", "residue_alt": "ALT", "ligand_alt": "alt"},
        sim=types.SimpleNamespace(hmr="no", dec_method="dd", rest=[0, 0, 5, 250, 0, 10, 20]),
    )

    original_assign = restraints._write_assign_and_read_vals
    try:
        restraints._write_assign_and_read_vals = lambda *args, **kwargs: [
            1.0,
            2.0,
            2.0,
            1.5,
            90.0,
            180.0,
            2.5,
            100.0,
            170.0,
            95.0,
            160.0,
            150.0,
        ]
        exprs = restraints._append_x_septop_boresch_restraints(ctx, disang)
    finally:
        restraints._write_assign_and_read_vals = original_assign

    disang_text = disang.read_text()
    assert len(exprs) == 9
    assert disang_text.count("#Lig_TR_REF") == 3
    assert disang_text.count("#Lig_TR_ALT") == 6
    assert "Skipping SEPTOP Boresch restraints" not in disang_text


def test_append_x_septop_boresch_reselects_receptor_frame_to_keep_stable_l1(
    tmp_path: Path,
) -> None:
    windows_dir = tmp_path / "x00"
    windows_dir.mkdir()
    build_dir = tmp_path / "x_build_files"
    build_dir.mkdir()
    disang = windows_dir / "disang.rest"
    disang.write_text("")

    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  GLY A   2       0.000   8.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  SER A   3       0.000  16.000   0.000  1.00  0.00           C\n",
                "ATOM      4  CA  THR A   4       0.000   8.000   8.000  1.00  0.00           C\n",
                "HETATM    5  N1  REF L  10       4.000   0.000   0.000  1.00  0.00           N\n",
                "HETATM    6  C1  REF L  10       4.000   1.000   1.000  1.00  0.00           C\n",
                "HETATM    7  C2  REF L  10       4.000   2.000  -1.000  1.00  0.00           C\n",
                "HETATM    8  O1  REF L  10       5.000   0.000   2.000  1.00  0.00           O\n",
                "HETATM    9  N1  ALT M  11       4.200   0.000   0.300  1.00  0.00           N\n",
                "HETATM   10  C1  ALT M  11       4.200   1.000   1.300  1.00  0.00           C\n",
                "HETATM   11  C2  ALT M  11       4.200   2.000  -0.700  1.00  0.00           C\n",
                "HETATM   12  O1  ALT M  11       5.200   0.000   2.300  1.00  0.00           O\n",
                "END\n",
            ]
        )
    )
    for path in [windows_dir / "full.prmtop", windows_dir / "full.inpcrd"]:
        path.write_text("stub\n")
    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":10@C1",
                "L2": ":10@C2",
                "L3": ":10@O1",
                "lig_res": "10",
            }
        )
    )
    for ligand in ("ref", "alt"):
        stable_dir = tmp_path / "simulations" / ligand / "equil"
        stable_dir.mkdir(parents=True)
        (stable_dir / "stable_boresch_distance.json").write_text(
            json.dumps({"usable": True, "ligand": {"name": "N1"}})
        )

    ctx = types.SimpleNamespace(
        build_dir=build_dir,
        window_dir=windows_dir,
        system_root=tmp_path,
        ligand="ref",
        residue_name="REF",
        comp="x",
        extra={
            "ligand_ref": "ref",
            "ligand_alt": "alt",
            "residue_ref": "REF",
            "residue_alt": "ALT",
            "user_anchor_atoms": [],
        },
        sim=types.SimpleNamespace(hmr="no", dec_method="dd", rest=[0, 0, 5, 250, 0, 10, 20]),
    )

    original_assign = restraints._write_assign_and_read_vals
    try:
        restraints._write_assign_and_read_vals = (
            lambda _workdir, exprs, *_args, **_kwargs: [1.0] * len(exprs)
        )
        exprs = restraints._append_x_septop_boresch_restraints(ctx, disang)
    finally:
        restraints._write_assign_and_read_vals = original_assign

    diagnostic = json.loads((windows_dir / "boresch_anchor_guard.json").read_text())
    assert diagnostic["receptor"]["reselected"]
    assert diagnostic["receptor"]["final"]["P3"]["input_mask"] == ":4@CA"
    assert diagnostic["endpoints"]["ref"]["final_ligand_names"][0] == "N1"
    assert diagnostic["endpoints"]["alt"]["final_ligand_names"][0] == "N1"
    assert diagnostic["endpoints"]["ref"]["final"]["L1"]["amber_iat"] == 5
    assert any(":4@CA" in expr for expr in exprs)


def test_build_restraints_y_omits_ligand_com_block(tmp_path: Path) -> None:
    windows_dir = tmp_path / "y00"
    windows_dir.mkdir()
    (windows_dir / "vac.pdb").write_text("ATOM      1  C1  LIG A   2       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

    ctx = types.SimpleNamespace(
        window_dir=windows_dir,
        ligand="lig",
        residue_name="LIG",
    )

    restraints._build_restraints_y(None, ctx)

    assert (windows_dir / "cv.in").read_text() == "cv_file\n"
    assert "&colvar" not in (windows_dir / "cv.in").read_text()
    assert not (windows_dir / "disang.rest").read_text().strip()


def test_build_restraints_d_uses_local_frame_pose_restraints(tmp_path: Path) -> None:
    windows_dir = tmp_path / "d00"
    windows_dir.mkdir()
    build_dir = tmp_path / "d_build_files"
    build_dir.mkdir()
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir()
    (equil_dir / "extra_conf_restraints.json").write_text(
        json.dumps(
            {
                "blocks": [
                    "&colvar\n"
                    " cv_type = 'DISTANCE'\n"
                    " cv_ni = 2, cv_i = 1,2\n"
                    " anchor_position = 0.0, 0.0, 3.0, 3.3\n"
                    " anchor_strength = 10.0, 10.0\n"
                    "/\n"
                ]
            }
        )
    )
    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":10@C1",
                "L2": ":10@C2",
                "L3": ":10@O1",
                "lig_res": "10",
            }
        )
    )
    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       0.000   2.000   0.000  1.00  0.00           C\n",
                "ATOM      4  CA  ALA A   4       2.000   2.000   0.000  1.00  0.00           C\n",
                "ATOM      5  C1  LIG A  10       1.000   1.000   0.000  1.00  0.00           C\n",
                "ATOM      6  C2  LIG A  10       1.500   1.000   0.000  1.00  0.00           C\n",
                "ATOM      7  O1  LIG A  10       1.000   1.500   0.000  1.00  0.00           O\n",
                "ATOM      8  H1  LIG A  10       1.000   1.000   0.500  1.00  0.00           H\n",
                "ATOM      9  C1  LIG A  11      20.000  20.000   0.000  1.00  0.00           C\n",
                "END\n",
            ]
        )
    )

    ctx = types.SimpleNamespace(
        working_dir=tmp_path,
        window_dir=windows_dir,
        equil_dir=equil_dir,
        comp="d",
        residue_name="LIG",
        sim=types.SimpleNamespace(lig_distance_force=7.5, dec_method="dd"),
        extra={"extra_conformation_restraints": "unused.json"},
        win=0,
    )

    restraints._build_restraints_d(None, ctx)

    cv_text = (windows_dir / "cv.in").read_text()
    assert cv_text.count("&colvar") == 12
    assert "EXTRA_CONFORMATIONAL_REST" not in cv_text
    assert "cv_type = 'DISTANCE'" in cv_text
    assert "cv_type = 'COM_DISTANCE'" not in cv_text
    assert "cv_i = 1,5," in cv_text
    assert "cv_i = 1,6," in cv_text
    assert "cv_i = 1,7," in cv_text
    assert "cv_i = 5,6," in cv_text
    assert "cv_i = 5,7," in cv_text
    assert "cv_i = 6,7," in cv_text
    assert ",8," not in cv_text

    disang_text = (windows_dir / "disang.rest").read_text()
    assert "ABFE_diff local_frame bound-pose restraints" in disang_text
    assert "#Lig_TR" not in disang_text
    assert "EXTRA_CONFORMATIONAL_REST" not in disang_text
    assert disang_text.count("&rst") == cv_text.count("&colvar")
    assert "iat=1,5," in disang_text
    assert "iat=1,6," in disang_text
    assert "iat=1,7," in disang_text
    assert "iat=5,6," in disang_text
    assert "igr2=" not in disang_text
    assert "rk2=7.5, rk3=7.5" in disang_text

    metadata = json.loads((windows_dir / "abfe_diff_pose_restraints.json").read_text())
    assert metadata["mode"] == "local_frame"
    assert metadata["ligand_heavy_atom_serials"] == [5, 6, 7]
    assert metadata["ligand_pose_atom_serials"] == [5, 6, 7]
    assert metadata["anchor_atom_serials"] == [1, 2, 3]
    assert len(metadata["restraints"]) == 12
    assert sum(1 for item in metadata["restraints"] if item["kind"] == "external_pose") == 9
    assert sum(1 for item in metadata["restraints"] if item["kind"] == "ligand_internal") == 3


def test_build_restraints_d_can_use_dense_pose_restraints(tmp_path: Path) -> None:
    windows_dir = tmp_path / "d00"
    windows_dir.mkdir()
    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       0.000   2.000   0.000  1.00  0.00           C\n",
                "ATOM      4  CA  ALA A   4       2.000   2.000   0.000  1.00  0.00           C\n",
                "ATOM      5  C1  LIG A  10       1.000   1.000   0.000  1.00  0.00           C\n",
                "ATOM      6  C2  LIG A  10       1.500   1.000   0.000  1.00  0.00           C\n",
                "ATOM      7  O1  LIG A  10       1.000   1.500   0.000  1.00  0.00           O\n",
                "END\n",
            ]
        )
    )

    ctx = types.SimpleNamespace(
        window_dir=windows_dir,
        comp="d",
        residue_name="LIG",
        sim=types.SimpleNamespace(
            lig_distance_force=7.5,
            abfe_diff_pose_restraint_type="dense",
        ),
        extra={},
        win=0,
    )

    restraints._build_restraints_d(None, ctx)

    cv_text = (windows_dir / "cv.in").read_text()
    assert cv_text.count("&colvar") == 12
    assert "cv_i = 1,5," in cv_text
    assert "cv_i = 4,7," in cv_text
    assert "cv_i = 5,6," not in cv_text

    metadata = json.loads((windows_dir / "abfe_diff_pose_restraints.json").read_text())
    assert metadata["mode"] == "dense"
    assert sorted(metadata["anchor_atom_serials"]) == [1, 2, 3, 4]
    assert len(metadata["restraints"]) == 12
    assert all(item["kind"] == "external_pose" for item in metadata["restraints"])


def test_build_restraints_x_keeps_only_protein_com_block(tmp_path: Path) -> None:
    work_dir = tmp_path
    windows_dir = work_dir / "x00"
    windows_dir.mkdir()
    build_dir = work_dir / "x_build_files"
    build_dir.mkdir()

    vac_pdb = windows_dir / "vac.pdb"
    vac_pdb.write_text(
        "".join(
            f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    {float(idx):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
            for idx in range(1, 7)
        )
        + "END\n"
    )

    for path in [
        windows_dir / "REF.prmtop",
        windows_dir / "ALT.prmtop",
        windows_dir / "full.prmtop",
        windows_dir / "full.inpcrd",
    ]:
        path.write_text("stub\n")

    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":4@C1",
                "L2": ":4@C2",
                "L3": ":4@C3",
                "lig_res": "1",
            }
        )
    )

    ctx = types.SimpleNamespace(
        working_dir=work_dir,
        window_dir=windows_dir,
        equil_dir=work_dir / "x-1",
        ligand="lig",
        residue_name="REF",
        comp="x",
        extra={"residue_ref": "REF", "residue_alt": "ALT"},
        sim=types.SimpleNamespace(hmr="no", dec_method="dd", rest=[0, 0, 0, 0, 0, 10.0, 20.0]),
    )

    restraints._build_restraints_x(None, ctx)

    cv_text = (windows_dir / "cv.in").read_text()
    assert cv_text.count("&colvar") == 1
    assert "cv_ni = 8, cv_i = 1,0,1,2,3,4,5,6," in cv_text
    assert "cv_i = 2,0," not in cv_text

    disang_text = (windows_dir / "disang.rest").read_text()
    assert disang_text.count("&rst") == 1
    assert "igr1=1,0" in disang_text
    assert "igr1=2,0" not in disang_text


def test_ion_guard_uses_rbfe_scmask_site_atom(tmp_path: Path) -> None:
    work_dir = tmp_path
    windows_dir = work_dir / "x00"
    windows_dir.mkdir()
    equil_dir = work_dir / "x-1"
    equil_dir.mkdir()
    (equil_dir / "scmask.json").write_text(
        json.dumps(
            {
                "scmk1_cc_solvent_indices": [10],
                "scmk1_cc_site_indices": [11],
                "scmk2_cc_solvent_indices": [20],
                "scmk2_cc_site_indices": [21],
            }
        )
    )
    (windows_dir / "full.pdb").write_text(
        "".join(
            [
                f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    {float(idx):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
                for idx in range(1, 10)
            ]
            + [
                "HETATM   10  C1  REF A  10      10.000   0.000   0.000  1.00  0.00           C\n",
                "HETATM   11  C1  REF A  11      11.000   0.000   0.000  1.00  0.00           C\n",
                "HETATM   12  NA  NA  A  12      12.000   0.000   0.000  1.00  0.00          NA\n",
                "END\n",
            ]
        )
    )
    disang = windows_dir / "disang.rest"
    disang.write_text("")

    ctx = types.SimpleNamespace(
        comp="x",
        window_dir=windows_dir,
        residue_name="REF",
        sim=types.SimpleNamespace(ion_guard="yes", cation="Na+", anion="Cl-"),
    )

    written = restraints._append_ion_guard_restraints(
        ctx,
        disang,
        ligand_resnames=["REF", "ALT"],
    )

    text = disang.read_text()
    assert written == 1
    assert text.count("#Ion_Guard") == 1
    assert "iat=12,10," not in text
    assert "iat=12,11," in text


def test_build_restraints_v_omits_ligand_com_block(tmp_path: Path) -> None:
    work_dir = tmp_path
    windows_dir = work_dir / "v00"
    windows_dir.mkdir()
    build_dir = work_dir / "v_build_files"
    build_dir.mkdir()

    (windows_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       3.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      4  C1  LIG A   4       4.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      5  C2  LIG A   4       5.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      6  C3  LIG A   4       6.000   0.000   0.000  1.00  0.00           C\n",
            ]
        )
        + "END\n"
    )
    for path in [
        windows_dir / "lig.pdb",
        windows_dir / "vac_ligand.prmtop",
        windows_dir / "full.prmtop",
        windows_dir / "full.inpcrd",
    ]:
        path.write_text("stub\n")

    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":4@C1",
                "L2": ":4@C2",
                "L3": ":4@C3",
                "lig_res": "1",
            }
        )
    )

    ctx = types.SimpleNamespace(
        working_dir=work_dir,
        window_dir=windows_dir,
        comp="v",
        ligand="lig",
        residue_name="LIG",
        sim=types.SimpleNamespace(hmr="no", dec_method="dd", rest=[0, 0, 0, 0, 0, 10.0, 20.0]),
        extra={},
    )

    original_assign = restraints._write_assign_and_read_vals
    try:
        restraints._write_assign_and_read_vals = lambda *args, **kwargs: [1.0] * 9
        restraints._build_restraints_v_o_z(None, ctx)
    finally:
        restraints._write_assign_and_read_vals = original_assign

    cv_text = (windows_dir / "cv.in").read_text()
    assert cv_text.count("&colvar") == 1
    assert "cv_i = 2,0," not in cv_text

    disang_text = (windows_dir / "disang.rest").read_text()
    assert "igr1=1,0" in disang_text
    assert "igr1=2,0" not in disang_text


def test_ligand_dihedral_force_helper_uses_active_force() -> None:
    assert (
        restraints._ligand_dihedral_force_constant(
            ":1@C1 :1@C2 :1@C3 :1@C4"
        )
        == 10.0
    )
    assert (
        restraints._ligand_dihedral_force_constant(
            ":1@C1 :1@C12 :1@C3 :1@C4"
        )
        == 10.0
    )


def test_equil_disang_writes_auto_ligand_dihedrals_with_zero_force(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path
    build_dir = work_dir / "q_build_files"
    build_dir.mkdir()
    (build_dir / "equil-LIG.pdb").write_text("HEADER\n")
    (work_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":1@CA",
                "P2": ":2@CA",
                "P3": ":3@CA",
                "L1": ":4@C1",
                "L2": ":4@C2",
                "L3": ":4@C3",
                "lig_res": "4",
            }
        )
    )
    (work_dir / "vac.pdb").write_text(
        "".join(
            [
                "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  CA  ALA A   3       3.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      4  C1  LIG A   4       4.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      5  C2  LIG A   4       5.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      6  C3  LIG A   4       6.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      7  C4  LIG A   4       7.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      8  C12 LIG A   4       8.000   0.000   0.000  1.00  0.00           C\n",
            ]
        )
        + "END\n"
    )
    for name in ("lig.pdb", "LIG.prmtop", "full.prmtop", "full.inpcrd"):
        (work_dir / name).write_text("stub\n")

    ctx = types.SimpleNamespace(
        working_dir=work_dir,
        build_dir=build_dir,
        equil_dir=work_dir,
        win=-1,
        ligand="lig",
        residue_name="LIG",
        extra={},
        sim=types.SimpleNamespace(
            hmr="no",
            rest=[0, 0, 0, 0, 0, 10.0, 20.0],
            release_eq=[100],
        ),
    )

    original_scan = restraints._scan_dihedrals_from_prmtop
    original_assign = restraints._write_assign_and_read_vals
    try:
        restraints._scan_dihedrals_from_prmtop = lambda *args, **kwargs: [
            ":4@C1 :4@C2 :4@C3 :4@C4",
            ":4@C12 :4@C2 :4@C3 :4@C4",
        ]
        restraints._write_assign_and_read_vals = lambda *args, **kwargs: [1.0] * 11
        restraints.write_equil_restraints(ctx)
    finally:
        restraints._scan_dihedrals_from_prmtop = original_scan
        restraints._write_assign_and_read_vals = original_assign

    disang_text = (work_dir / "disang.rest").read_text()
    lig_d_lines = [line for line in disang_text.splitlines() if "#Lig_D" in line]
    assert len(lig_d_lines) == 2
    assert any(
        "iat=4,5,6,7," in line
        and "rk2=  0.0000000, rk3=  0.0000000, &end #Lig_D" in line
        for line in lig_d_lines
    )
    assert any(
        "iat=8,5,6,7," in line
        and "rk2=  0.0000000, rk3=  0.0000000, &end #Lig_D" in line
        for line in lig_d_lines
    )
