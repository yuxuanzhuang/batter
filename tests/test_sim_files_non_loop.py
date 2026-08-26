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
runfiles = importlib.import_module("batter._internal.ops.runfiles")


def test_default_fe_seed_schedule_uses_ten_states(tmp_path: Path) -> None:
    steps_per_state, n_states, dynlmb, total_steps = (
        sim_files.build_dyna_steps_run_per_lambda()
    )

    assert steps_per_state == 10_000
    assert n_states == 10
    assert dynlmb == pytest.approx(1.0 / 9.0)
    assert total_steps == 100_000

    window_dir = tmp_path / "z-1"
    ctx = SimpleNamespace(
        window_dir=window_dir,
        ligand="lig",
        comp="z",
        win=-1,
        sim=SimpleNamespace(hmr="yes", system_name="sys"),
    )
    runfiles.write_fe_run_file(ctx, [0.0, 0.5, 1.0])

    run_local = (window_dir / "run-local.bash").read_text()
    assert (
        "lambda_eq_list=(0.0000 0.1111 0.2222 0.3333 0.4444 "
        "0.5556 0.6667 0.7778 0.8889 1.0000)"
    ) in run_local


def test_fe_window_equilibration_defaults_to_fifty_ps() -> None:
    assert sim_files.fe_window_equil_steps(0.002) == 25_000
    assert sim_files.fe_window_equil_steps(0.001) == 50_000
    assert sim_files.DEFAULT_FE_HANDOFF_RESTRAINT_START == pytest.approx(1.0)
    assert sim_files.DEFAULT_FE_HANDOFF_RESTRAINT_END == pytest.approx(0.0)
    assert sim_files.DEFAULT_FE_HANDOFF_STAGES == 5


def _assert_fe_handoff(window_dir: Path, *, steps: int, dum_weight: float) -> None:
    paths = sorted(window_dir.glob("eq-handoff-[0-9][0-9].in")) + [
        window_dir / "eq.in"
    ]
    assert len(paths) == 5
    expected_weights = [1.0, 0.75, 0.5, 0.25, 0.0]
    observed_steps = 0
    for index, (path, weight) in enumerate(zip(paths, expected_weights)):
        text = path.read_text()
        assert f"FE target-window handoff stage {index + 1}/5" in text
        assert "  ntr = 1," in text
        assert "  nmropt = 1," in text
        assert "  ntwx = 0," in text
        assert "  ntwv = 0," in text
        assert "restraintmask" not in text
        assert "type='REST'" not in text
        assert "FE constant DUM positional restraint" in text
        assert f"\n{dum_weight:g}\nATOM 1 2\nEND\n" in text
        if weight > 0:
            assert "FE ligand anchor/common-core handoff positional restraint" in text
        else:
            assert "FE ligand anchor/common-core handoff positional restraint" not in text
        nstlim = re.search(r"\bnstlim\s*=\s*(\d+)", text)
        assert nstlim is not None
        observed_steps += int(nstlim.group(1))
    assert observed_steps == steps

    metadata = json.loads((window_dir / "eq-handoff.json").read_text())
    assert metadata["total_steps"] == steps
    assert metadata["dum_atom_indices"] == [1, 2]
    assert [stage["ligand_weight"] for stage in metadata["stages"]] == pytest.approx(
        expected_weights
    )


def test_eqnpt0_uno_template_uses_short_z_seed_equilibration() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "batter" / "_internal" / "templates" / "amber_files_orig" / "eqnpt0-uno.in"

    assert "  nstlim = 2000," in template.read_text()


def test_shipped_mdin_templates_wrap_coordinates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / "batter" / "_internal" / "templates" / "amber_files_orig"
    md_inputs = [
        path
        for path in template_dir.iterdir()
        if path.is_file() and (path.name.startswith("mdin") or path.name.startswith("eqn"))
    ]

    assert md_inputs
    for path in md_inputs:
        text = path.read_text()
        if "iwrap" in text:
            assert "iwrap = 0," not in text, path.name
            assert "iwrap = 1," in text, path.name


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


def test_solvent_ligand_restraint_mask_abfe_diff_uses_full_residue(tmp_path: Path) -> None:
    pdb = tmp_path / "vac.pdb"
    pdb.write_text(
        "".join(
            [
                "ATOM      1  C1  LIG A   5       0.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      2  C2  LIG A   5       1.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      3  C1  LIG A   6       5.000   0.000   0.000  1.00  0.00           C\n",
                "ATOM      4  C2  LIG A   6       6.000   0.000   0.000  1.00  0.00           C\n",
                "END\n",
            ]
        )
    )

    assert sim_files._solvent_ligand_restraint_mask(pdb, resid=6, comp="z") == "@3"
    assert sim_files._solvent_ligand_restraint_mask(pdb, resid=6, comp="d") == ":6"


def test_fe_ntwprt_atom_count_includes_ion_prefix_before_water(tmp_path: Path) -> None:
    (tmp_path / "vac.pdb").write_text(
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  LIG A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (tmp_path / "full.pdb").write_text(
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  LIG A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3 Na+  Na+ A   2       2.000   0.000   0.000  1.00  0.00          NA\n"
        "ATOM      4 Cl-  Cl- A   3       3.000   0.000   0.000  1.00  0.00          CL\n"
        "ATOM      5  O   WAT A   4       4.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      6  H1  WAT A   4       4.700   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    )

    assert sim_files._fe_ntwprt_atom_count(tmp_path, "no") == 4
    assert sim_files._fe_ntwprt_atom_count(tmp_path, "yes") == 6


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


def test_restraintmask_long_mask_converts_to_legacy_group(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prmtop = repo_root / "tests" / "data" / "ligand_params" / "b74b7e78c757" / "lig.prmtop"
    assert prmtop.exists()

    long_mask = "(" + " | ".join(["@1"] * 80) + ")"
    mdin = tmp_path / "mdin-test.in"
    mdin.write_text(
        "&cntrl\n"
        "  ntr = 1,\n"
        "  restraint_wt = 5.0,\n"
        f"  restraintmask = '{long_mask}',\n"
        "/\n"
    )

    sim_files._apply_restraintmask_length_limit(mdin, prmtop)

    text = mdin.read_text()
    assert "restraintmask =" not in text
    assert "Converted from restraintmask" in text
    assert "ATOM 1 1" in text


def test_restraintmask_short_mask_converts_to_legacy_group(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prmtop = repo_root / "tests" / "data" / "ligand_params" / "b74b7e78c757" / "lig.prmtop"
    assert prmtop.exists()

    mdin = tmp_path / "mdin-test.in"
    mdin.write_text(
        "&cntrl\n"
        "  ntr = 1,\n"
        "  restraint_wt = 50,\n"
        "  restraintmask = '@1',\n"
        "/\n"
    )

    sim_files._apply_restraintmask_length_limit(mdin, prmtop)

    text = mdin.read_text()
    assert "restraintmask =" not in text
    assert "Converted from restraintmask" in text
    assert "50" in text
    assert "ATOM 1 1" in text


def test_extra_restraints_are_included_in_legacy_group_conversion(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prmtop = repo_root / "tests" / "data" / "ligand_params" / "b74b7e78c757" / "lig.prmtop"
    assert prmtop.exists()

    mdin = tmp_path / "mdin-test.in"
    text = (
        "&cntrl\n"
        "  ntr = 1,\n"
        "  restraint_wt = 5.0,\n"
        "  restraintmask = '@1',\n"
        "/\n"
    )
    mdin.write_text(sim_files._patch_restraint_block(text, "@2", 50.0))

    sim_files._apply_restraintmask_length_limit(mdin, prmtop)

    converted = mdin.read_text()
    assert "restraintmask =" not in converted
    assert "Converted from restraintmask" in converted
    assert "50" in converted
    assert "ATOM 1 2" in converted


def test_restraintmask_without_prmtop_is_left_as_mask(tmp_path: Path) -> None:
    mdin = tmp_path / "mdin-test.in"
    original = (
        "&cntrl\n"
        "  ntr = 1,\n"
        "  restraint_wt = 5.0,\n"
        "  restraintmask = '(@CA,C,N,P31 | :apo | :1) & !@H=',\n"
        "/\n"
    )
    mdin.write_text(original)

    sim_files._apply_restraintmask_length_limit(mdin, prmtop_path=None)

    assert mdin.read_text() == original


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


def test_apply_fe_handoff_replaces_existing_restraint_schedule(
    tmp_path: Path,
) -> None:
    mdin = tmp_path / "eq.in"
    (tmp_path / "vac.pdb").write_text(
        "HETATM    1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "HETATM    2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "HETATM    3  N1  LIG A   3       2.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    4  C2  LIG A   3       3.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    5  C3  LIG A   3       4.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    mdin.write_text(
        "&cntrl\n"
        "  nmropt = 0,\n"
        "  ntr = 0,\n"
        "  restraint_wt = 50,\n"
        "  restraintmask = ':1-2',\n"
        "/\n"
        " &wt type='REST', istep1=0, istep2=100, value1=50, value2=5, /\n"
        " &wt type='DUMPFREQ', istep1=250, /\n"
        " &wt type='END', /\n"
        "DISANG=disang.rest\n"
    )

    sim_files._apply_fe_handoff_restraint(
        mdin,
        restraint_mask="(:3@N1 | :3@C2 | :3@C3) & !@H=",
        total_steps=25_000,
    )

    text = mdin.read_text()
    _assert_fe_handoff(tmp_path, steps=25_000, dum_weight=50.0)
    assert "restraintmask" not in text
    assert "value1=50" not in text
    assert "type='REST'" not in text


def test_fe_handoff_schedule_survives_legacy_group_conversion(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "vac.pdb"
    topology.write_text(
        "HETATM    1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "HETATM    2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "HETATM    3  C1  LIG A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    mdin = tmp_path / "eq.in"
    mdin.write_text("&cntrl\n/\n &wt type='END', /\n")

    sim_files._apply_fe_handoff_restraint(
        mdin,
        restraint_mask="@3",
        total_steps=25_000,
    )
    before = mdin.read_text()
    sim_files._apply_restraintmask_length_limit(mdin, topology)

    text = mdin.read_text()
    _assert_fe_handoff(tmp_path, steps=25_000, dum_weight=10.0)
    assert text == before
    assert "restraintmask =" not in text
    assert "FE constant DUM positional restraint\n10\nATOM 1 2\nEND\nEND\n" in text


def test_ligand_handoff_prefers_persisted_boresch_anchor_names(
    tmp_path: Path,
) -> None:
    (tmp_path / "disang.rest").write_text(
        "# Anchor atoms :1@CA :2@CA :3@CA :10@N1 :10@C2 :10@C3 comp=z\n"
    )

    mask = sim_files._ligand_handoff_restraint_mask(
        window_dir=tmp_path,
        vac_pdb=tmp_path / "missing.pdb",
        ligand_resids=(10, 20),
    )

    assert mask == (
        "(:10@N1 | :10@C2 | :10@C3 | :20@N1 | :20@C2 | :20@C3) & !@H="
    )


def test_septop_handoff_requires_four_mapped_atoms_before_using_core(
    tmp_path: Path,
) -> None:
    window_dir = tmp_path / "x00"
    seed_dir = tmp_path / "x-1"
    window_dir.mkdir()
    seed_dir.mkdir()
    guard = {
        "endpoints": {
            "ref": {
                "final": {
                    "L1": {"resolved": True, "name": "N1"},
                    "L2": {"resolved": True, "name": "C2"},
                    "L3": {"resolved": True, "name": "C3"},
                }
            },
            "alt": {
                "final": {
                    "L1": {"resolved": True, "name": "N4"},
                    "L2": {"resolved": True, "name": "C5"},
                    "L3": {"resolved": True, "name": "C6"},
                }
            },
        }
    }
    (seed_dir / "boresch_anchor_guard.json").write_text(json.dumps(guard))
    low_mapping = {
        "scmk1_cc_site_indices": [10, 11, 12],
        "scmk1_cc_solvent_indices": [20, 21, 22],
        "scmk2_cc_site_indices": [30, 31, 32],
        "scmk2_cc_solvent_indices": [40, 41, 42],
    }

    anchor_mask = sim_files._rbfe_handoff_restraint_mask(
        window_dir=window_dir,
        vac_pdb=tmp_path / "missing.pdb",
        scmask=low_mapping,
        ref_resid=5,
        septop=True,
    )
    core_mask = sim_files._rbfe_handoff_restraint_mask(
        window_dir=window_dir,
        vac_pdb=tmp_path / "missing.pdb",
        scmask={
            **low_mapping,
            "scmk1_cc_site_indices": [10, 11, 12, 13],
            "scmk1_cc_solvent_indices": [20, 21, 22, 23],
            "scmk2_cc_site_indices": [30, 31, 32, 33],
            "scmk2_cc_solvent_indices": [40, 41, 42, 43],
        },
        ref_resid=5,
        septop=True,
    )

    assert anchor_mask == (
        "(:5@N1 | :5@C2 | :5@C3 | :6@N1 | :6@C2 | :6@C3 | "
        ":7@N4 | :7@C5 | :7@C6 | :8@N4 | :8@C5 | :8@C6) & !@H="
    )
    assert core_mask == "(@10-13,20-23,30-33,40-43) & !@H="


def test_component_l_cmass_dumpfreq_is_capped() -> None:
    assert sim_files._component_l_cmass_dumpfreq(25000) == 1000
    assert sim_files._component_l_cmass_dumpfreq(500) == 500
    assert sim_files._component_l_cmass_dumpfreq(0) == 1


def test_write_l_mdin_uses_dense_cmass_dumpfreq_without_changing_ntwx(tmp_path: Path) -> None:
    src = tmp_path / "mdin-equil"
    dst = tmp_path / "mdin-template"
    src.write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = _ntwx_,\n"
        "  ntwr = _ntwr_,\n"
        "  nstlim = _num-steps_,\n"
        "  infe = _enable_infe_,\n"
        "/\n"
        " &wt type='DUMPFREQ', istep1=_ntwx_, /\n"
        " &wt type='END', /\n"
        "DISANG=disang_file.rest\n"
        "DUMPAVE=cmass.txt\n"
    )

    sim_files._write_l_mdin_from_equil_template(
        src=src,
        dst=dst,
        mol="LIG",
        replacements={
            "_ntwx_": "25000",
            "_ntwr_": "25000",
            "_enable_infe_": "0",
        },
        total_steps=250000,
        ntwx=25000,
        eq_seed=False,
        cmass_dumpfreq=25000,
    )

    text = dst.read_text()
    assert "ntwx = 25000" in text
    assert "ntwr = 25000" in text
    assert "nstlim = 250000" in text
    assert "type='DUMPFREQ', istep1=1000" in text


def test_write_l_mdin_can_chunk_production_nstlim(tmp_path: Path) -> None:
    src = tmp_path / "mdin-equil"
    dst = tmp_path / "mdin-template"
    src.write_text(
        "&cntrl\n"
        "  nstlim = _num-steps_,\n"
        "  infe = _enable_infe_,\n"
        "/\n"
    )

    sim_files._write_l_mdin_from_equil_template(
        src=src,
        dst=dst,
        mol="LIG",
        replacements={"_enable_infe_": "0"},
        total_steps=1_000_000,
        chunk_steps=250_000,
        ntwx=25_000,
        eq_seed=False,
    )

    text = dst.read_text()
    assert "! total_steps=1000000" in text
    assert "nstlim = 250000" in text


def test_write_l_mdin_can_enable_mcwat_fe(tmp_path: Path) -> None:
    src = tmp_path / "mdin-equil"
    dst = tmp_path / "mdin-template"
    src.write_text(
        "&cntrl\n"
        "  nstlim = _num-steps_,\n"
        "  mcwat = _enable_mcwat_,\n"
        "  nmd = 1000,\n"
        "  nmc = 1000,\n"
        "  mcwatmask = ':_lig_name_',\n"
        "  mcligshift = 15,\n"
        "  mcwatretry = 3000,\n"
        "  mcresstr = \"WAT\",\n"
        "  infe = _enable_infe_,\n"
        "/\n"
    )

    sim_files._write_l_mdin_from_equil_template(
        src=src,
        dst=dst,
        mol="LIG",
        replacements={"_enable_infe_": "0"},
        total_steps=1_000_000,
        chunk_steps=250_000,
        ntwx=25_000,
        eq_seed=False,
        mcwat_fe_mask=":291",
    )

    text = dst.read_text()
    assert "  mcwat = 1,\n" in text
    assert "  nmd = 1000,\n" in text
    assert "  nmc = 1000,\n" in text
    assert "  mcwatmask = \":291\",\n" in text
    assert "  mcligshift = 15,\n" in text
    assert "  mcwatretry = 3000,\n" in text
    assert "  mcresstr = \"WAT\",\n" in text


def test_modern_fe_templates_do_not_enable_infe() -> None:
    template_dir = Path(sim_files.__file__).resolve().parents[1] / "templates" / "amber_files_orig"

    for name in ("mini-uno", "mini-unorest", "mini-unorest-dd", "mini-unorest-lig", "mini-ex"):
        content = (template_dir / name).read_text()
        assert "  infe = 1," not in content
        assert "  infe = 0," in content


def test_mini_writers_force_shake_constraints(
    tmp_path: Path,
) -> None:
    src = tmp_path / "mini.in"
    fe_dst = tmp_path / "fe-mini.in"
    fe_eq_dst = tmp_path / "fe-mini-eq.in"
    eq_dst = tmp_path / "eq-mini.in"
    src.write_text(
        "&cntrl\n"
        "  ntf = 1,\n"
        "  ntc = 1,\n"
        "  restraintmask = ':_lig_name_',\n"
        "/\n"
    )

    sim_files._sub_write_fe_mini(src, fe_dst, {"_lig_name_": "LIG"})
    sim_files._sub_write_fe_mini(src, fe_eq_dst, {"_lig_name_": "LIG"})
    sim_files._sub_write_fe_mini(src, eq_dst, {"_lig_name_": "LIG"})

    assert "  ntf = 2," in fe_dst.read_text()
    assert "  ntc = 2," in fe_dst.read_text()
    assert "  ntf = 2," in fe_eq_dst.read_text()
    assert "  ntc = 2," in fe_eq_dst.read_text()
    assert "  ntf = 2," in eq_dst.read_text()
    assert "  ntc = 2," in eq_dst.read_text()


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


def test_maybe_extra_mask_uses_absolute_indices_for_all_selected_atoms(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    build_dir = tmp_path / "build"
    work.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    (work / "full.pdb").write_text(
        "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  ALA A  10       1.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  C1  LIG B 900       2.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    4  H1  LIG B 900       3.000   0.000   0.000  1.00  0.00           H\n"
        "TER\nEND\n"
    )

    ctx = SimpleNamespace(
        extra={
            "extra_restraints": "name CB or resname LIG",
            "extra_restraint_fc": 12.5,
        },
        win=-1,
        equil_dir=work,
        build_dir=build_dir,
    )

    mask, force_const = sim_files._maybe_extra_mask(ctx, work, resid_shift=2)

    assert mask == "@2-4"
    assert force_const == pytest.approx(12.5)
    saved = json.loads((work / "extra_restraints.json").read_text())
    assert saved["mask"] == "@2-4"
    assert saved["selection"] == "name CB or resname LIG"


def test_maybe_extra_mask_reuses_equil_json_for_non_minus_one_windows(tmp_path: Path) -> None:
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir(parents=True, exist_ok=True)
    (equil_dir / "extra_restraints.json").write_text(
        json.dumps({"mask": "@9", "force_const": 9.0})
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
    assert mask == "@9"
    assert force_const == pytest.approx(9.0)


def test_sim_files_y_uses_first_ligand_atom_position_restraint(tmp_path: Path) -> None:
    windows_dir = tmp_path / "y00"
    amber_dir = tmp_path / "amber"
    windows_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)

    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      3  C1  LIG A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C2  LIG A   3       3.000   0.000   0.000  1.00  0.00           C\n"
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
        "  nstlim = _num-steps_,\n"
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
    assert "nstlim = 10000" in eq_text
    assert "restraintmask" not in eq_text
    assert "@CA" not in eq_text
    _assert_fe_handoff(windows_dir, steps=50_000, dum_weight=10.0)
    assert "restraintmask = '(:1 | @3) & !@H='" in template_text
    assert "nmropt = 0" in template_text


def test_sim_files_z_keeps_bulk_ligand_first_atom_out_of_mdin_template(
    tmp_path: Path,
) -> None:
    windows_dir = tmp_path / "z00"
    amber_dir = tmp_path / "amber"
    windows_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)

    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      3  C1  LIG A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C2  LIG A   3       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C1  LIG A   4       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C2  LIG A   4       5.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (windows_dir / "full.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      3  C1  LIG A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C2  LIG A   3       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C1  LIG A   4       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C2  LIG A   4       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      7 Na+  Na+ A   5       6.000   0.000   0.000  1.00  0.00          NA\n"
        "ATOM      8 Cl-  Cl- A   6       7.000   0.000   0.000  1.00  0.00          CL\n"
        "ATOM      9  O   WAT A   7       8.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )
    (amber_dir / "mdin-unorest").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = _ntwx_,\n"
        "  ntwprt = _num-atoms_,\n"
        "  nstlim = _num-steps_,\n"
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
            mcwat_fe="yes",
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

    assert "restraintmask" not in eq_text
    assert "@CA" not in eq_text
    assert "nstlim = 5000" in eq_text
    assert "ntwx = 0" in eq_text
    _assert_fe_handoff(windows_dir, steps=25_000, dum_weight=10.0)

    assert "restraintmask = ':1-2'," in template_text
    assert "@5" not in template_text
    assert "ntwprt = 8" in template_text
    assert "  mcwat = 1,\n" in template_text
    assert "  nmd = 1000,\n" in template_text
    assert "  nmc = 1000,\n" in template_text
    assert "  mcwatmask = \":3\",\n" in template_text
    assert "  mcligshift = 15,\n" in template_text
    assert "  mcwatretry = 3000,\n" in template_text
    assert "  mcresstr = \"WAT\",\n" in template_text

    assert ":LIG" in mini_text
    assert "@5" not in mini_text


def test_sim_files_d_sdr_uses_three_copy_charge_balanced_masks(
    tmp_path: Path,
) -> None:
    windows_dir = tmp_path / "d-1"
    amber_dir = tmp_path / "amber"
    windows_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)

    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  C1  LIG A  10       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  LIG A  10       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C1  LIG A  30       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C2  LIG A  30       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C1  LIG A  20       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C2  LIG A  20       3.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (amber_dir / "mdin-diff-sdr").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = _ntwx_,\n"
        "  ntwprt = _num-atoms_,\n"
        "  nstlim = _num-steps_,\n"
        "  dt = _step_,\n"
        "  nmropt = 1,\n"
        "  restraint_wt = 50.0,\n"
        "  restraintmask = ':1-2',\n"
        "  icfe = 1,\n"
        "  clambda = lbd_val,\n"
        "  timask1 = ':mk1',\n"
        "  timask2 = ':mk2',\n"
        "  scmask1=':mk1',\n"
        "  scmask2=':mk2',\n"
        "  crgmask = ':mk3',\n"
        "  gti_bat_sc      = 1,\n"
        "/\n"
    )
    (amber_dir / "mini-diff-sdr").write_text(
        "&cntrl\n"
        "  restraintmask = ':_lig_name_',\n"
        "  timask1 = ':mk1',\n"
        "  timask2 = ':mk2',\n"
        "  scmask1=':mk1',\n"
        "  scmask2=':mk2',\n"
        "  crgmask = ':mk3',\n"
        "  gti_bat_sc      = 1,\n"
        "/\n"
    )
    (amber_dir / "eqnpt0-uno.in").write_text(
        "&cntrl\n"
        "  nmropt = 0,\n"
        "  temp0 = _temperature_,\n"
        "  ntp = 3,\n"
        "  csurften = 3,\n"
        "  restraintmask = ':_lig_name_',\n"
        "  mcwat = 1,\n"
        "/\n"
    )
    (amber_dir / "eqnpt-uno.in").write_text(
        "&cntrl\n"
        "  nmropt = 0,\n"
        "  temp0 = _temperature_,\n"
        "  ntp = 3,\n"
        "  csurften = 3,\n"
        "  restraintmask = ':_lig_name_',\n"
        "  mcwat = 1,\n"
        "/\n"
    )
    (amber_dir / "eqnpt-uno-eq.in").write_text(
        "&cntrl\n"
        "  nmropt = 0,\n"
        "  temp0 = _temperature_,\n"
        "  ntp = 3,\n"
        "  csurften = 3,\n"
        "  restraintmask = '((@CA & _non_loop_) | :_lig_name_) & !@H=',\n"
        "  mcwat = 1,\n"
        "/\n"
    )

    ctx = SimpleNamespace(
        working_dir=tmp_path / "work_unused",
        window_dir=windows_dir,
        amber_dir=amber_dir,
        system_root=tmp_path / "system_unused",
        build_dir=tmp_path / "build_unused",
        residue_name="LIG",
        comp="d",
        win=-1,
        extra={"infe": 0},
        sim=SimpleNamespace(
            temperature=300.0,
            dic_n_steps={"d": 4000},
            ntwx=250,
            all_atoms="no",
            dec_method="sdr",
        ),
    )

    original_resolve = sim_files._resolve_non_loop_mask
    try:
        sim_files._resolve_non_loop_mask = lambda *args, **kwargs: ":1"
        sim_files.sim_files_z(ctx, [0.0, 0.5, 1.0])
    finally:
        sim_files._resolve_non_loop_mask = original_resolve

    template_text = (windows_dir / "mdin-template").read_text()
    mini_text = (windows_dir / "mini.in").read_text()
    eq_text = (windows_dir / "eq.in").read_text()
    eqnpt0_text = (windows_dir / "eqnpt0.in").read_text()
    eqnpt_text = (windows_dir / "eqnpt.in").read_text()

    assert "timask1 = ':10,20'" in template_text
    assert "timask2 = ':30'" in template_text
    assert "scmask1=':10'" in template_text
    assert "scmask2=''" in template_text
    assert "crgmask = ':20'" in template_text
    assert "gti_bat_sc      = 1" in template_text
    assert "ti_vdw_mask" not in template_text
    assert "restraintmask = '(:30,20) & !@H='" in template_text
    assert "@CA" not in template_text
    assert "nmropt = 1" in template_text
    assert ":LIG" not in template_text
    assert "mcwat" not in template_text

    assert "timask1 = ':10,20'" in mini_text
    assert "timask2 = ':30'" in mini_text
    assert "scmask1=':10'" in mini_text
    assert "scmask2=''" in mini_text
    assert "crgmask = ':20'" in mini_text
    assert "gti_bat_sc      = 1" in mini_text
    assert "ti_vdw_mask" not in mini_text
    assert "restraintmask = '(@CA,C,N,P31 | :30,20) & !@H='" in mini_text
    assert ":LIG" not in mini_text

    assert "mcwat" not in eq_text
    assert "mcwatmask" not in eq_text
    assert "nmropt = 1" in eq_text
    assert "gti_bat_sc      = 1" in eq_text
    assert "restraintmask = '((@CA & :1) | :30,20) & !@H='" in eq_text
    assert ":LIG" not in eq_text
    assert "ntp = 1" in eqnpt0_text
    assert "csurften = 0" in eqnpt0_text
    assert "nmropt = 1" in eqnpt0_text
    assert "restraintmask = '(@CA,C,N,P31 | :30,20) & !@H='" in eqnpt0_text
    assert "ntp = 1" in eqnpt_text
    assert "csurften = 0" in eqnpt_text
    assert "nmropt = 1" in eqnpt_text
    assert "restraintmask = '(@CA,C,N,P31 | :30,20) & !@H='" in eqnpt_text
    assert "timask1 = ':10,20'" in eqnpt_text
    assert "timask2 = ':30'" in eqnpt_text
    assert "scmask1=':10'" in eqnpt_text
    assert "scmask2=''" in eqnpt_text
    assert "crgmask = ':20'" in eqnpt_text
    assert "gti_bat_sc      = 1" in eqnpt_text
    assert "nstlim = 100000" in eq_text
    assert "dynlmb = 0.1111111111111111" in eq_text
    assert "mbar_states = 03" in eq_text
    assert "FE target-window handoff" not in eq_text
    assert "type='REST'" not in eq_text


def test_abfe_diff_d_run_file_uses_ten_seed_lambda_states(tmp_path: Path) -> None:
    window_dir = tmp_path / "d-1"
    ctx = SimpleNamespace(
        window_dir=window_dir,
        ligand="lig",
        comp="d",
        win=-1,
        sim=SimpleNamespace(hmr="yes", system_name="sys", fe_type="uno_rest_diff"),
    )

    runfiles.write_fe_run_file(ctx, [0.0, 0.5, 1.0])

    run_local = (window_dir / "run-local.bash").read_text()
    assert (
        "lambda_eq_list=(0.0000 0.1111 0.2222 0.3333 0.4444 "
        "0.5556 0.6667 0.7778 0.8889 1.0000)"
    ) in run_local
    assert "lambda_set_list=(0.0000 0.5000 1.0000)" in run_local
    assert "RBFE minimization seed" in run_local
    assert "eq_init.rst7" in run_local
    assert "cd ../d-1" in run_local
    assert "Equilibration stage 0" not in run_local


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
                    "scmk1_all_indices": [3, 4, 5],
                    "scmk1_cc_solvent_indices": [3],
                    "scmk1_cc_site_indices": [4],
                    "scmk2_all_indices": [6, 7, 8],
                    "scmk2_cc_solvent_indices": [6],
                    "scmk2_cc_site_indices": [7],
            }
        )
    )
    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      3  C1  REF A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C1  REF A   4       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C2  REF A   4       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C1  ALT A   5       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      7  C3  ALT A   6       6.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      8  C4  ALT A   6       7.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (windows_dir / "full.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      3  C1  REF A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C1  REF A   4       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C2  REF A   4       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C1  ALT A   5       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      7  C3  ALT A   6       6.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      8  C4  ALT A   6       7.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      9 Na+  Na+ A   7       8.000   0.000   0.000  1.00  0.00          NA\n"
        "ATOM     10 Cl-  Cl- A   8       9.000   0.000   0.000  1.00  0.00          CL\n"
        "ATOM     11  O   WAT A   9      10.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )
    (amber_dir / "mdin-ex").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = 100,\n"
        "  ntwprt = _num-atoms_,\n"
        "  dt = _step_,\n"
        "  nmropt = 0,\n"
        "  restraint_wt = 50.0,\n"
        "  restraintmask = ':1-2',\n"
        "/\n"
    )
    (amber_dir / "mini-ex").write_text(
        "&cntrl\n"
        "  ntf = 1,\n"
        "  ntc = 1,\n"
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

    assert "restraintmask" not in eq_text
    assert "@CA" not in eq_text
    assert "nmropt = 1" in eq_text
    _assert_fe_handoff(windows_dir, steps=25_000, dum_weight=5.0)

    assert "(:1-2 | @3 | @6) & !@H=" in template_text
    assert "ntwprt = 10" in template_text

    assert ":REF" in mini_text
    assert ":ALT" in mini_text
    assert "  ntf = 1," in mini_text
    assert "  ntc = 2," in mini_text
    assert re.search(r"\|\s*@3\s*\|", mini_text) is None

    assert ":REF" in mini_eq_text
    assert ":ALT" in mini_eq_text
    assert "  ntf = 2," in mini_eq_text
    assert "  ntc = 2," in mini_eq_text
    assert re.search(r"\|\s*@3\s*\|", mini_eq_text) is None


def test_sim_files_x_septop_enables_lambda_dependent_boresch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim_files, "_resolve_non_loop_mask", lambda *args, **kwargs: ":1")

    work_dir = tmp_path / "work"
    windows_dir = work_dir / "x00"
    amber_dir = tmp_path / "amber"
    equil_dir = work_dir / "x-1"
    windows_dir.mkdir(parents=True)
    amber_dir.mkdir(parents=True)
    equil_dir.mkdir(parents=True)

    (equil_dir / "scmask.json").write_text(
        json.dumps(
            {
                "scmk1_all_indices": [10, 11, 12],
                "scmk1_cc_solvent_indices": [],
                "scmk1_cc_site_indices": [],
                "scmk2_all_indices": [20, 21, 22],
                "scmk2_cc_solvent_indices": [],
                "scmk2_cc_site_indices": [],
            }
        )
    )
    (windows_dir / "vac.pdb").write_text(
        "ATOM      1  Pb  DUM A   1       0.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      2  Pb  DUM A   2       1.000   0.000   0.000  1.00  0.00          PB\n"
        "ATOM      3  C1  REF A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  C1  REF A   4       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  C1  ALT A   5       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      6  C1  ALT A   6       5.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (amber_dir / "mdin-ex").write_text(
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        "  ntwx = 100,\n"
        "  ntwprt = 10,\n"
        "  dt = _step_,\n"
        "  nmropt = 1,\n"
        "  restraint_wt = 50.0,\n"
        "  restraintmask = ':1-2',\n"
        "  timask1 = 'timk1',\n"
        "  timask2 = 'timk2',\n"
        "  scmask1='scmk1',\n"
        "  scmask2='scmk2',\n"
        "  gti_vdw_exp     = 2\n"
        "/\n"
    )
    (amber_dir / "mini-ex").write_text(
        "&cntrl\n"
        "  ntf = 1,\n"
        "  ntc = 1,\n"
        "  nmropt = 1,\n"
        "  restraintmask = ':_lig1_name_ | :_lig2_name_',\n"
        "  gti_vdw_exp     = 2\n"
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
            fe_type="relative_septop",
            temperature=300.0,
            dic_n_steps={"x": 4000},
            ntwx=250,
            all_atoms="no",
        ),
    )

    sim_files.sim_files_x(ctx, [0.0, 1.0])

    eq_text = (windows_dir / "eq.in").read_text()
    template_text = (windows_dir / "mdin-template").read_text()

    assert "nmropt = 1" in eq_text
    assert "gti_bat_sc      = 1" in eq_text
    assert "gti_bat_sc      = 1" in template_text
    mini_text = (windows_dir / "mini.in").read_text()
    mini_eq_text = (windows_dir / "mini_eq.in").read_text()
    assert "gti_bat_sc      = 1" in mini_text
    assert "  ntf = 1," in mini_text
    assert "  ntc = 2," in mini_text
    assert "  ntf = 2," in mini_eq_text
    assert "  ntc = 2," in mini_eq_text
    assert "scmask1='@10-12'" in eq_text
    assert "scmask2='@20-22'" in eq_text
    assert "scmask1='@10-12'" in template_text
    assert "scmask2='@20-22'" in template_text
    assert "restraintmask" not in eq_text
    assert "@CA" not in eq_text
    _assert_fe_handoff(windows_dir, steps=25_000, dum_weight=5.0)
    assert "(:1-2 | @6 | @4) & !@H=" in template_text
    assert (windows_dir / "lambda.sch").read_text() == (
        "TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n"
    )
