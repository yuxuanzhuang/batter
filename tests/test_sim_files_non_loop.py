from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

sim_files = pytest.importorskip("batter._internal.ops.sim_files")


def _write_minimal_equil_templates(amber_dir: Path) -> None:
    amber_dir.mkdir(parents=True, exist_ok=True)
    (amber_dir / "mini.in").write_text("_lig_name_\n")
    (amber_dir / "eqnvt.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt0-water.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt-water.in").write_text("_temperature_ _lig_name_\n")
    (amber_dir / "eqnpt-water-eq.in").write_text(
        "restraintmask = '((@CA & :_non_loop_) | :_lig_name_) & !@H='\n"
    )
    (amber_dir / "eqnpt-disappear.in").write_text(
        "_temperature_ _lig_name_ _enable_infe_ disang_file\n"
    )
    (amber_dir / "eqnpt-appear.in").write_text(
        "_temperature_ _lig_name_ _enable_infe_ disang_file\n"
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
    # keep 2-5 (H run len=4) and 10-14 (E run len=5); drop 7-8 (len=2)
    assignments = ["-", "H", "H", "H", "H", "-", "E", "E", "-", "E", "E", "E", "E", "E", "-"]
    got = sim_files._non_loop_mask_from_dssp_assignments(assignments, min_len=4)
    assert got == "2-5,10-14"


def test_write_sim_files_replaces_non_loop_from_dssp_manifest(tmp_path: Path) -> None:
    dssp = [["-", "H", "H", "H", "H", "-", "E", "E", "E", "E", "-", "-"]]
    ctx = _ctx(tmp_path, with_manifest=True, dssp_results=dssp)

    sim_files.write_sim_files(ctx, infe=False)

    eqnpt_eq = (ctx.working_dir / "eqnpt_eq.in").read_text()
    assert "_non_loop_" not in eqnpt_eq
    assert ":2-5,7-10" in eqnpt_eq


def test_write_sim_files_non_loop_falls_back_to_renum_when_missing_dssp(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, with_manifest=False, dssp_results=None)

    sim_files.write_sim_files(ctx, infe=False)

    eqnpt_eq = (ctx.working_dir / "eqnpt_eq.in").read_text()
    assert "_non_loop_" not in eqnpt_eq
    assert ":1-6" in eqnpt_eq
