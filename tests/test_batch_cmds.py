from __future__ import annotations

from pathlib import Path

from batter.cli import batch_cmds


def _setup_abfe_component(exec_path: Path, ligand: str = "L1", comp: str = "z") -> Path:
    lig_dir = exec_path / "simulations" / ligand
    comp_dir = lig_dir / "fe" / comp
    (comp_dir / f"{comp}-1").mkdir(parents=True, exist_ok=True)
    (comp_dir / f"{comp}00").mkdir(parents=True, exist_ok=True)
    return comp_dir


def test_collect_batch_tasks_skips_pre_window_failed(tmp_path, monkeypatch) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    comp_dir = _setup_abfe_component(exec_path, ligand="L1", comp="z")
    (comp_dir / "z-1" / "FAILED").write_text("FAILED\n")

    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])
    monkeypatch.setattr(batch_cmds, "_write_batch_run_script", lambda *args, **kwargs: None)

    tasks = batch_cmds._collect_batch_tasks(exec_path)
    assert tasks == []


def test_collect_batch_tasks_keeps_unfinished_without_pre_window_failure(
    tmp_path, monkeypatch
) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    _setup_abfe_component(exec_path, ligand="L1", comp="z")

    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])
    monkeypatch.setattr(batch_cmds, "_write_batch_run_script", lambda *args, **kwargs: None)

    tasks = batch_cmds._collect_batch_tasks(exec_path)
    assert len(tasks) == 1
    assert tasks[0].component == "z"
    assert tasks[0].ligand == "L1"


def test_collect_remd_tasks_skips_pre_window_failed(tmp_path, monkeypatch) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    comp_dir = _setup_abfe_component(exec_path, ligand="L1", comp="z")
    (comp_dir / "run-local-remd.bash").write_text("#!/bin/bash\nN_WINDOWS=1\n")
    (comp_dir / "z-1" / "FAILED").write_text("FAILED\n")

    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])

    tasks = batch_cmds._collect_remd_tasks(exec_path)
    assert tasks == []
