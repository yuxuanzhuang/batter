from __future__ import annotations

import os
from pathlib import Path

import pytest

from batter import api as api_mod


def test_resolve_execution_run_defaults_to_latest(tmp_path: Path) -> None:
    executions = tmp_path / "executions"
    old = executions / "old-run"
    new = executions / "new-run"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    run_id, run_dir = api_mod._resolve_execution_run(tmp_path, None)
    assert run_id == "new-run"
    assert run_dir == new


def test_resolve_execution_run_raises_when_no_runs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No executions found"):
        api_mod._resolve_execution_run(tmp_path, None)


def test_api_read_cinnabar_outputs_reads_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "cinnabar"
    bundle_dir.mkdir()
    relative_csv = bundle_dir / "cinnabar_relative.csv"
    absolute_csv = bundle_dir / "cinnabar_absolute.csv"
    cycle_nodes_csv = bundle_dir / "cycle_closure_nodes.csv"
    cycle_edges_csv = bundle_dir / "cycle_closure_edges.csv"
    relative_csv.write_text(
        "labelA,labelB,DDG (kcal/mol),uncertainty (kcal/mol)\nA,B,1.0,0.2\n"
    )
    absolute_csv.write_text(
        "label,DG (kcal/mol),uncertainty (kcal/mol)\nA,-5.0,0.1\n"
    )
    cycle_nodes_csv.write_text(
        "label,dG_cc,dG_wcc1,path_dependent_error,path_independent_error\n"
        "A,-5.1,-5.2,0.3,0.4\n"
    )
    cycle_edges_csv.write_text(
        "labelA,labelB,ddG_cc,ddG_wcc1,pair_error\nA,B,1.1,1.2,0.5\n"
    )

    relative_df, absolute_df = api_mod.read_cinnabar_outputs(bundle_dir)

    assert relative_df.loc[0, "DDG_uncorrected (kcal/mol)"] == pytest.approx(1.0)
    assert relative_df.loc[0, "DDG_cycle_closure (kcal/mol)"] == pytest.approx(1.2)
    assert relative_df.loc[0, "uncertainty_cycle_closure (kcal/mol)"] == pytest.approx(
        0.5
    )
    assert absolute_df.loc[0, "DG_uncorrected (kcal/mol)"] == pytest.approx(-5.0)
    assert absolute_df.loc[0, "DG_cycle_closure (kcal/mol)"] == pytest.approx(-5.2)
