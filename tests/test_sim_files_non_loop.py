from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import types

import pytest


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


sim_files = _load_internal_module("batter._internal.ops.sim_files")


def _write_minimal_equil_templates(amber_dir: Path) -> None:
    amber_dir.mkdir(parents=True, exist_ok=True)
    (amber_dir / "mini.in").write_text("_lig_name_\n")
    (amber_dir / "eqnvt.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt0-water.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt-water.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt-water-eq.in").write_text(
        "restraintmask = '((@CA & _non_loop_) | :_lig_name_) & !@H='\n"
    )
    (amber_dir / "eqnpt-disappear.in").write_text(
        "_temperature_ _lig_name_\ninfe = _enable_infe_\nDISANG=disang_file.rest\n"
    )
    (amber_dir / "eqnpt-appear.in").write_text(
        "_temperature_ _lig_name_\ninfe = _enable_infe_\nDISANG=disang_file.rest\n"
    )
    (amber_dir / "mdin-equil").write_text(
        "&cntrl\n  temp0=_temperature_,\n  nstlim=_num-steps_,\n/\n"
    )


def _ctx(tmp_path: Path, *, with_manifest: bool, dssp_results: list[list[str]] | None):
    run_root = tmp_path / "run"
    work = run_root / "simulations" / "LIG" / "equil"
    build_dir = work / "q_build_files"
    amber_dir = work / "q_amber_files"
    work.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_equil_templates(amber_dir)

    # Used by write_sim_files to parse anchor atom ids.
    (work / "disang.rest").write_text("a b c d e f 1 2 3\n")

    # Fallback renumber map (new_resid column is 5th).
    (build_dir / "protein_renum.txt").write_text(
        "ALA A 10 ALA 1\n"
        "ALA A 11 ALA 2\n"
        "ALA A 12 ALA 3\n"
        "ALA A 13 ALA 4\n"
        "ALA A 14 ALA 5\n"
        "ALA A 15 ALA 6\n"
    )

    if with_manifest:
        all_ligs = run_root / "all-ligands"
        all_ligs.mkdir(parents=True, exist_ok=True)
        (all_ligs / "manifest.json").write_text(
            json.dumps({"dssp": {"results": dssp_results}}, indent=2)
        )

    sim = SimpleNamespace(
        temperature=300.0,
        membrane_simulation=False,
        eq_steps=2500,
    )
    return SimpleNamespace(
        ligand="LIG",
        residue_name="LIG",
        param_dir_dict={},
        working_dir=work,
        system_root=run_root,
        comp="q",
        win=-1,
        sim=sim,
        extra={},
        amber_dir=amber_dir,
        build_dir=build_dir,
    )


def test_non_loop_mask_from_dssp_assignments_filters_short_runs() -> None:
    assignments = ["-", "H", "H", "H", "H", "-", "E", "E", "-", "E", "E", "E", "E", "E", "-"]
    got = sim_files._non_loop_mask_from_dssp_assignments(assignments, min_len=4, shift=0)
    assert got == "1-4,9-13"


def test_write_sim_files_replaces_non_loop_from_dssp_manifest(tmp_path: Path) -> None:
    dssp = [["-", "H", "H", "H", "H", "-", "E", "E", "E", "E", "-", "-"]]
    ctx = _ctx(tmp_path, with_manifest=True, dssp_results=dssp)

    sim_files.write_sim_files(ctx, infe=False)

    eqnpt_eq = (ctx.working_dir / "eqnpt_eq.in").read_text()
    assert "_non_loop_" not in eqnpt_eq
    assert "::" not in eqnpt_eq
    assert ":3-6,8-11" in eqnpt_eq


def test_write_sim_files_non_loop_falls_back_to_renum_when_missing_dssp(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, with_manifest=False, dssp_results=None)

    sim_files.write_sim_files(ctx, infe=False)

    eqnpt_eq = (ctx.working_dir / "eqnpt_eq.in").read_text()
    assert "_non_loop_" not in eqnpt_eq
    assert "::" not in eqnpt_eq
    assert ":2-7" in eqnpt_eq


def test_write_sim_files_keeps_infe_disabled(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, with_manifest=False, dssp_results=None)

    sim_files.write_sim_files(ctx, infe=True)

    assert "infe = 0" in (ctx.working_dir / "eqnpt_disappear.in").read_text()
    assert "infe = 0" in (ctx.working_dir / "eqnpt_appear.in").read_text()


def test_write_cmass_dump_block_uses_dumpave_footer() -> None:
    handle = io.StringIO()

    sim_files._write_cmass_dump_block(handle, istep1=2500)

    assert handle.getvalue() == (
        " &wt type='DUMPFREQ', istep1=2500, /\n"
        " &wt type='END', /\n"
        "DISANG=disang.rest\n"
        "DUMPAVE=cmass.txt\n"
        "LISTIN=POUT\n"
        "LISTOUT=POUT\n"
    )


def test_modern_fe_templates_do_not_enable_infe() -> None:
    template_dir = Path(sim_files.__file__).resolve().parents[1] / "templates" / "amber_files_orig"

    for name in ("mini-uno", "mini-unorest", "mini-unorest-dd", "mini-unorest-lig", "mini-ex"):
        content = (template_dir / name).read_text()
        assert "  infe = 1," not in content
        assert "  infe = 0," in content


def test_modern_templates_use_dumpave_not_pmd() -> None:
    template_dir = Path(sim_files.__file__).resolve().parents[1] / "templates" / "amber_files_orig"
    template_names = (
        "eqnpt-appear.in",
        "eqnpt-disappear.in",
        "eqnpt-eq.in",
        "eqnpt-lig.in",
        "eqnpt-water-eq.in",
        "eqnpt-water.in",
        "eqnpt-uno-eq.in",
        "eqnpt-uno.in",
        "eqnpt.in",
        "eqnpt0-lig.in",
        "eqnpt0-uno.in",
        "eqnpt0-water.in",
        "eqnpt0.in",
        "eqnvt.in",
        "mdin-equil",
        "mini-ex",
        "mini-uno",
        "mini-unorest",
        "mini-unorest-dd",
        "mini-unorest-lig",
        "mini.in",
    )

    for name in template_names:
        content = (template_dir / name).read_text()
        assert "&pmd" not in content
        assert "output_file = 'cmass.txt'" not in content
        assert "cv_file = 'cv.in'" not in content
        assert "DUMPAVE=cmass.txt" in content
        assert "LISTIN=POUT" in content
        assert "LISTOUT=POUT" in content


def test_sim_files_source_has_no_infe_one_writes() -> None:
    content = Path(sim_files.__file__).read_text()
    assert 'mdin.write("  infe = 1,\\n")' not in content
    assert 'mdin.write(" &pmd \\n")' not in content
    assert "DUMPAVE=cmass.txt" in content


@pytest.mark.parametrize(
    ("resid_shift", "expected_mask"),
    [
        (1, "(:2-3) & @CA"),
        (2, "(:3-4) & @CA"),
    ],
)
def test_maybe_extra_mask_applies_residue_shift(
    tmp_path: Path, resid_shift: int, expected_mask: str
) -> None:
    work = tmp_path / "work"
    build_dir = tmp_path / "build"
    (work / "build_files").mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    (work / "full.pdb").write_text(
        "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A  11       1.000   0.000   0.000  1.00  0.00           C\n"
        "TER\nEND\n"
    )
    (work / "build_files" / "protein_renum.txt").write_text(
        "ALA A 10 ALA 1\n"
        "ALA A 11 ALA 2\n"
    )

    ctx = SimpleNamespace(
        extra={"extra_restraints": "resid 10:11", "extra_restraint_fc": 12.5},
        win=-1,
        equil_dir=work,
        build_dir=build_dir,
    )

    mask, force_const = sim_files._maybe_extra_mask(ctx, work, resid_shift=resid_shift)

    assert mask == expected_mask
    assert force_const == pytest.approx(12.5)
    saved = json.loads((work / "extra_restraints.json").read_text())
    assert saved["resid_shift"] == resid_shift


def test_maybe_extra_mask_reuses_equil_json_for_non_minus_one_windows(tmp_path: Path) -> None:
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir(parents=True, exist_ok=True)
    (equil_dir / "extra_restraints.json").write_text(
        json.dumps({"mask": "(:9) & @CA", "force_const": 9.0})
    )

    ctx = SimpleNamespace(
        extra={"extra_restraints": "resid 10"},
        win=0,
        equil_dir=equil_dir,
        build_dir=tmp_path / "build",
    )
    mask, force_const = sim_files._maybe_extra_mask(
        ctx, tmp_path / "unused", resid_shift=2
    )
    assert mask == "(:9) & @CA"
    assert force_const == pytest.approx(9.0)


def test_sim_files_y_uses_first_ligand_atom_position_restraint(tmp_path: Path) -> None:
    windows_dir = tmp_path / "y00"
    amber_dir = tmp_path / "amber"
    windows_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)

    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  C1  LIG A   2       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C2  LIG A   2       2.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )

    (amber_dir / "mini-unorest-lig").write_text(
        "&cntrl\n"
        "  nmropt = 1,\n"
        "  restraintmask = '(:_lig_name_ | @Na+,Cl-) & !@H=',\n"
        "/\n"
    )
    (amber_dir / "mini.in").write_text("_lig_name_\n")
    (amber_dir / "eqnpt-lig.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt0-lig.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "mdin-unorest-lig").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  dt = _step_,\n"
        "  nmropt = 1,\n"
        "  restraintmask = ':1',\n"
        "/\n"
    )

    ctx = SimpleNamespace(
        residue_name="LIG",
        window_dir=windows_dir,
        amber_dir=amber_dir,
        win=0,
        sim=SimpleNamespace(temperature=300.0, dic_n_steps={"y": 5000}, ntwx=250),
    )

    sim_files.sim_files_y(ctx, [0.0])

    mini_text = (windows_dir / "mini.in").read_text()
    eq_text = (windows_dir / "eq.in").read_text()
    template_text = (windows_dir / "mdin-template").read_text()

    assert "nmropt = 1" in mini_text
    assert ":LIG" in mini_text
    assert "@2" not in mini_text
    assert "nmropt = 1" in eq_text
    assert "restraintmask = '(@CA | :LIG | :1) & !@H='" in eq_text
    assert "@2" not in eq_text
    assert "restraintmask = '(:1 | @2) & !@H='" in template_text
    assert "nmropt = 0" in template_text


def test_sim_files_z_applies_first_atom_position_restraint_only_in_mdin_template(
    tmp_path: Path,
) -> None:
    windows_dir = tmp_path / "z00"
    amber_dir = tmp_path / "amber"
    windows_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)

    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  LIG A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C1  LIG A   2       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C2  LIG A   2       3.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (amber_dir / "mdin-unorest").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = _ntwx_,\n"
        "  ntwprt = _num-atoms_,\n"
        "  dt = _step_,\n"
        "  restraint_wt = 50.0,\n"
        "  restraintmask = ':1-2',\n"
        "/\n"
    )
    (amber_dir / "mini-unorest").write_text(
        "&cntrl\n"
        "  restraintmask = '(@CA,C,N,P31,Na+,Cl- | :_lig_name_ | :2) & !@H=',\n"
        "/\n"
    )
    (amber_dir / "mini.in").write_text("_lig_name_\n")
    (amber_dir / "eqnpt0-uno.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt-uno.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt-uno-eq.in").write_text("_temperature_ _lig_name_ _non_loop_\n")

    ctx = SimpleNamespace(
        working_dir=tmp_path / "work_unused",
        window_dir=windows_dir,
        amber_dir=amber_dir,
        system_root=tmp_path / "system_unused",
        build_dir=tmp_path / "build_unused",
        residue_name="LIG",
        comp="z",
        win=0,
        extra={"infe": 0},
        sim=SimpleNamespace(
            temperature=300.0,
            dic_n_steps={"z": 4000},
            ntwx=250,
            all_atoms="no",
            dec_method="sdr",
        ),
    )

    original_resolve = sim_files._resolve_non_loop_mask
    try:
        sim_files._resolve_non_loop_mask = lambda *args, **kwargs: ":1"
        sim_files.sim_files_z(ctx, [0.0])
    finally:
        sim_files._resolve_non_loop_mask = original_resolve

    eq_text = (windows_dir / "eq.in").read_text()
    template_text = (windows_dir / "mdin-template").read_text()
    mini_text = (windows_dir / "mini.in").read_text()

    assert "restraintmask = '((@CA & :1) | :LIG | :1-2 ) & !@H='" in eq_text
    assert "@3" not in eq_text

    assert "restraintmask = '(:1-2 | @3) & !@H='" in template_text

    assert ":LIG" in mini_text
    assert "@3" not in mini_text


def test_sim_files_x_uses_first_atoms_for_solvent_ligand_position_restraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim_files, "_resolve_non_loop_mask", lambda *args, **kwargs: ":1")

    work_dir = tmp_path / "work"
    windows_dir = work_dir / "x00"
    build_dir = work_dir / "x_build_files"
    amber_dir = tmp_path / "amber"
    equil_dir = work_dir / "x-1"
    windows_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)
    equil_dir.mkdir(parents=True)

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
                "scmk1_all_indices": [10, 11, 12],
                "scmk1_cc_solvent_indices": [10],
                "scmk1_cc_site_indices": [11],
                "scmk2_all_indices": [20, 21, 22],
                "scmk2_cc_solvent_indices": [20],
                "scmk2_cc_site_indices": [21],
            }
        )
    )
    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  C1  REF A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C1  REF A   2       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C2  REF A   2       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C1  ALT A   3       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C3  ALT A   4       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C4  ALT A   4       5.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (amber_dir / "mdin-ex").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = 100,\n"
        "  ntwprt = 10,\n"
        "  dt = _step_,\n"
        "  restraint_wt = 50.0,\n"
        "  restraintmask = ':1-2',\n"
        "/\n"
    )
    (amber_dir / "mini-ex").write_text(
        "&cntrl\n"
        "  restraintmask = '(@CA,C,N,P31,Na+,Cl- | :_lig1_name_ | :_lig2_name_ | :2) & !@H=',\n"
        "/\n"
    )

    ctx = SimpleNamespace(
        comp="x",
        residue_name="REF",
        extra={"residue_alt": "ALT"},
        working_dir=work_dir,
        window_dir=windows_dir,
        amber_dir=amber_dir,
        win=0,
        build_dir=tmp_path / "build_unused",
        system_root=tmp_path / "system_unused",
        sim=SimpleNamespace(
            temperature=300.0,
            dic_n_steps={"x": 4000},
            ntwx=250,
            all_atoms="no",
        ),
    )

    sim_files.sim_files_x(ctx, [0.0])

    eq_text = (windows_dir / "eq.in").read_text()
    template_text = (windows_dir / "mdin-template").read_text()
    mini_text = (windows_dir / "mini.in").read_text()
    mini_eq_text = (windows_dir / "mini_eq.in").read_text()

    assert "((@CA & :1) | (@10-11) | :1-2 ) & !@H=" in eq_text
    assert re.search(r"\|\s*@10\s*\|", eq_text) is None

    assert "(:1-2 | @10) & !@H=" in template_text
    assert re.search(r"(^|[^0-9])@20([^0-9]|$)", template_text) is None

    assert ":REF" in mini_text
    assert ":ALT" in mini_text
    assert re.search(r"\|\s*@10\s*\|", mini_text) is None

    assert ":REF" in mini_eq_text
    assert ":ALT" in mini_eq_text
    assert re.search(r"\|\s*@10\s*\|", mini_eq_text) is None
