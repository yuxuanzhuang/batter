from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

from batter.cli.run import cli
from batter.api import run_analysis_from_execution
from batter.runtime.fe_repo import FERecord, WindowResult
from batter.pipeline.step import ExecResult


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_run_invokes_run_from_yaml(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text("dummy: true\n")

    class DummySection:
        def __init__(self, **values):
            self.__dict__.update(values)

        def model_copy(self, update: dict | None = None):
            data = dict(self.__dict__)
            if update:
                data.update(update)
            return DummySection(**data)

    class DummyRunConfig:
        def __init__(self, system=None, run=None):
            self.system = system or DummySection(output_folder="out")
            self.run = run or DummySection(run_id="auto", dry_run=False)

        def model_copy(self, update: dict | None = None):
            data = {"system": self.system, "run": self.run}
            if update:
                data.update(update)
            return DummyRunConfig(system=data["system"], run=data["run"])

        def resolved_sim_config(self):
            return object()

    monkeypatch.setattr(
        "batter.cli.run.RunConfig.load",
        staticmethod(lambda path: DummyRunConfig()),
    )

    called = {}

    def fake_run_from_yaml(path, **kwargs):
        called["path"] = path
        called["kwargs"] = kwargs

    monkeypatch.setattr("batter.cli.run.run_from_yaml", fake_run_from_yaml)

    result = runner.invoke(
        cli,
        [
            "run",
            str(yaml_path),
            "--on-failure",
            "prune",
            "--run-id",
            "test",
            "--allow-run-id-mismatch",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert called["path"] == yaml_path
    assert called["kwargs"]["on_failure"] == "prune"
    assert called["kwargs"]["run_overrides"]["run_id"] == "test"
    assert called["kwargs"]["run_overrides"]["allow_run_id_mismatch"] is True
    assert called["kwargs"]["run_overrides"]["dry_run"] is True


def test_cli_fe_list_table(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    df = pd.DataFrame(
        [
            {
                "run_id": "run1",
                "system_name": "sys",
                "fe_type": "abfe",
                "temperature": 300.0,
                "method": "mbar",
                "total_dG": -5.1,
                "total_se": 0.2,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]
    )
    monkeypatch.setattr("batter.cli.run.list_fe_runs", lambda path: df)

    result = runner.invoke(cli, ["fe", "list", str(work_dir)])
    assert result.exit_code == 0
    assert "run1" in result.output
    assert "sys" in result.output


def test_cli_fe_show(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    record = FERecord(
        run_id="run1",
        ligand="LIG1",
        mol_name="LIG",
        system_name="sys",
        fe_type="abfe",
        temperature=300.0,
        total_dG=-5.0,
        total_se=0.2,
        components=["elec"],
        windows=[
            WindowResult(component="elec", lam=0.0, dG=-1.0),
        ],
    )
    monkeypatch.setattr(
        "batter.cli.run.load_fe_run",
        lambda path, run_id, ligand=None: record,
    )

    result = runner.invoke(
        cli, ["fe", "show", str(work_dir), "run1", "--ligand", "LIG1"]
    )
    assert result.exit_code == 0
    assert "run_id     : run1" in result.output
    assert "total_dG   : -5.000" in result.output


def test_cli_fe_analyze_invokes_api(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    called: dict[str, Any] = {}

    def fake_run(work_dir, run_id, *, ligand, components=None, n_workers, sim_range):
        called["work_dir"] = work_dir
        called["run_id"] = run_id
        called["ligand"] = ligand
        called["components"] = components
        called["n_workers"] = n_workers
        called["sim_range"] = sim_range

    monkeypatch.setattr("batter.cli.run.run_analysis_from_execution", fake_run)
    monkeypatch.setattr("batter.api.run_analysis_from_execution", fake_run)
    result = runner.invoke(
        cli,
        [
            "fe",
            "analyze",
            str(tmp_path),
            "run1",
            "--ligand",
            "LIG1",
            "--workers",
            "3",
            "--sim-range",
            "0,5",
        ],
    )
    assert result.exit_code == 0
    assert called["work_dir"] == tmp_path
    assert called["run_id"] == "run1"
    assert called["ligand"] == "LIG1"
    assert called["n_workers"] == 3
    assert called["sim_range"] == (0, 5)


def test_run_analysis_from_execution(monkeypatch, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    run_src = Path("tests/data/equil_finished/executions/rep1")
    run_dest = work_dir / "executions" / "rep1"
    run_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_src, run_dest, dirs_exist_ok=True)
    run_meta = run_dest / "artifacts" / "config" / "run_meta.json"
    run_meta.write_text(json.dumps({"protocol": "abfe", "system_name": "adrb2_A"}, indent=2))

    called: list[tuple[str, str | None, int | None]] = []

    def fake_analyze(step, system, params):
        called.append(
            (
                system.meta.get("ligand"),
                params.get("components"),
                params.get("analysis_n_workers"),
            )
        )
        return ExecResult()

    fake_module = types.ModuleType("batter.exec.handlers.fe_analysis")
    fake_module.analyze_handler = fake_analyze
    monkeypatch.setitem(sys.modules, "batter.exec.handlers.fe_analysis", fake_module)
    monkeypatch.setattr("batter.orchestrate.run.save_fe_records", lambda *args, **kwargs: [])

    run_analysis_from_execution(
        work_dir,
        "rep1",
        components=["z"],
        n_workers=2,
        sim_range=(0, -1),
    )

    assert len(called) == 2
    assert all(call[1] == ["z"] for call in called)
    assert all(call[2] == 2 for call in called)


def test_cli_clone_exec(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    work_dir = tmp_path / "work"
    src_exec = work_dir / "executions" / "src"
    src_exec.mkdir(parents=True)

    called = {}

    def fake_clone_execution(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("batter.cli.run.clone_execution", fake_clone_execution)

    result = runner.invoke(
        cli,
        [
            "clone-exec",
            str(work_dir),
            "src",
            "--only-equil",
            "--symlink",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert called["src_run_id"] == "src"
    assert called["only_equil"] is True
    assert called["symlink"] is True
    assert called["force"] is True
