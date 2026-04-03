from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types


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


def test_colvar_block_to_rst_translates_com_distance() -> None:
    block = """&colvar
 cv_type = 'COM_DISTANCE'
 cv_ni = 30, cv_i = 2,0,4777,4776,4775,4774,4786,4787,4788,4772,4771,4769,4770,4773,4778,4779,4780,4781,4782,4783,4784,4785,4794,4798,4799,4792,4793,4795,4796,4797,
 anchor_position =     0.0000,     0.0000,     3.0000,   999.0000
 anchor_strength =    10.0000,    10.0000,
/
"""

    got = restraints._colvar_block_to_rst(block)

    assert got is not None
    assert "iat=-1,-1," in got
    assert "r1=0.0, r2=0.0, r3=3.0, r4=999.0," in got
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


def test_build_restraints_x_strides_common_core_ligand_com(tmp_path: Path) -> None:
    work_dir = tmp_path
    windows_dir = work_dir / "x00"
    windows_dir.mkdir()
    equil_dir = work_dir / "x-1"
    equil_dir.mkdir()
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
    (equil_dir / "scmask.json").write_text(
        json.dumps(
            {
                "scmk1_cc_solvent_indices": list(range(101, 126)),
                "scmk2_cc_solvent_indices": list(range(201, 226)),
            }
        )
    )

    ctx = types.SimpleNamespace(
        working_dir=work_dir,
        window_dir=windows_dir,
        equil_dir=equil_dir,
        ligand="lig",
        residue_name="REF",
        comp="x",
        extra={"residue_ref": "REF", "residue_alt": "ALT"},
        sim=types.SimpleNamespace(hmr="no", dec_method="dd", rest=[0, 0, 0, 0, 0, 10.0, 20.0]),
    )

    restraints._build_restraints_x(None, ctx)

    cv_text = (windows_dir / "cv.in").read_text()
    assert "cv_ni = 11, cv_i = 2,0,101,104,107,110,113,116,119,122,125," in cv_text
    assert "cv_ni = 11, cv_i = 2,0,201,204,207,210,213,216,219,222,225," in cv_text
