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
from tests.data import EQUIL_FINISHED_DIR, FE_FINISHED_EXECUTION_DIR


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_run_invokes_run_from_yaml(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
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
        def __init__(self, run=None):
            default_run = {"output_folder": "out", "run_id": "auto", "dry_run": False}
            run_data = dict(default_run)
            if run:
                run_data.update(run)
            self.run = DummySection(**run_data)

        def model_copy(self, update: dict | None = None):
            data = {"run": self.run}
            if update:
                data.update(update)
            run_payload = data["run"]
            if isinstance(run_payload, DummySection):
                run_values = dict(run_payload.__dict__)
            else:
                run_values = dict(run_payload)
            return DummyRunConfig(run=run_values)

        def resolved_sim_config(self):
            return object()

    monkeypatch.setattr(
        "batter.cli.run_cmds.RunConfig.load",
        staticmethod(lambda path: DummyRunConfig()),
    )

    called = {}

    def fake_run_from_yaml(path, **kwargs):
        called["path"] = path
        called["kwargs"] = kwargs

    monkeypatch.setattr("batter.cli.run_cmds.run_from_yaml", fake_run_from_yaml)

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
    monkeypatch.setattr("batter.cli.fe_cmds.list_fe_runs", lambda path: df)

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
        "batter.cli.fe_cmds.load_fe_run",
        lambda path, run_id, ligand=None: record,
    )

    result = runner.invoke(
        cli, ["fe", "show", str(work_dir), "run1", "--ligand", "LIG1"]
    )
    assert result.exit_code == 0
    assert "run_id     : run1" in result.output
    assert "total_dG   : -5.000" in result.output


def test_cli_run_slurm_submit_uses_header(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text("dummy: true\n")
    header_root = tmp_path / "headers"

    class DummySection:
        def __init__(self, **values):
            self.__dict__.update(values)

        def model_copy(self, update: dict | None = None):
            data = dict(self.__dict__)
            if update:
                data.update(update)
            return DummySection(**data)

    class DummyRunConfig:
        def __init__(self, run=None):
            default_run = {
                "output_folder": tmp_path / "out",
                "run_id": "auto",
                "dry_run": False,
                "slurm_header_dir": header_root,
                "allow_run_id_mismatch": False,
            }
            run_data = dict(default_run)
            if run:
                run_data.update(run)
            self.run = DummySection(**run_data)
            self.create = DummySection(system_name="sys")
            self.protocol = "abfe"

        def model_copy(self, update: dict | None = None):
            data = {"run": self.run}
            if update:
                data.update(update)
            run_payload = data["run"]
            run_values = dict(run_payload.__dict__)
            return DummyRunConfig(run=run_values)

        def resolved_sim_config(self):
            return object()

    monkeypatch.setattr(
        "batter.cli.run_cmds.RunConfig.load",
        staticmethod(lambda path: DummyRunConfig()),
    )

    rendered = {}

    def fake_render(name, header_path, body_path, replacements, header_root=None):
        rendered["header_root"] = header_root
        return "#SBATCH --dummy\n"

    monkeypatch.setattr(
        "batter.cli.run_cmds.render_slurm_with_header_body", fake_render
    )

    class DummyProc:
        returncode = 0
        stdout = "Submitted batch job 123"
        stderr = ""

    calls = {}

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        calls["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["run", str(yaml_path), "--slurm-submit"])
    assert result.exit_code == 0
    # ensure header_root threaded through to renderer
    assert rendered["header_root"] == header_root
    # job script should be written under temp path
    scripts = list(tmp_path.glob("*_job_manager.sbatch"))
    assert len(scripts) == 1
    assert scripts[0].exists()
    script_text = scripts[0].read_text()
    assert "#SBATCH --job-name=fep_" in script_text
    assert "/simulations/manager" in script_text
    assert "__JOB_NAME__" not in script_text
    # sbatch invoked on the generated script
    assert scripts[0].name in calls["cmd"]


def test_cli_fe_analyze_invokes_api(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    called: dict[str, Any] = {}

    def fake_run(
        work_dir,
        run_id,
        *,
        ligand,
        components=None,
        n_workers,
        analysis_start_step,
        overwrite=True,
        raise_on_error=True,
    ):
        called["work_dir"] = work_dir
        called["run_id"] = run_id
        called["ligand"] = ligand
        called["components"] = components
        called["n_workers"] = n_workers
        called["analysis_start_step"] = analysis_start_step
        called["overwrite"] = overwrite
        called["raise_on_error"] = raise_on_error

    monkeypatch.setattr("batter.cli.fe_cmds.run_analysis_from_execution", fake_run)
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
            "--analysis-start-step",
            "2500",
        ],
    )
    assert result.exit_code == 0
    assert called["work_dir"] == tmp_path
    assert called["run_id"] == "run1"
    assert called["ligand"] == "LIG1"
    assert called["n_workers"] == 3
    assert called["analysis_start_step"] == 2500
    assert called["overwrite"] is True
    assert called["raise_on_error"] is True


def test_cli_fe_analyze_can_disable_raise(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    called: dict[str, Any] = {}

    def fake_run(
        work_dir,
        run_id,
        *,
        ligand,
        components=None,
        n_workers,
        analysis_start_step,
        overwrite=True,
        raise_on_error=True,
    ):
        called["raise_on_error"] = raise_on_error

    monkeypatch.setattr("batter.cli.fe_cmds.run_analysis_from_execution", fake_run)
    monkeypatch.setattr("batter.api.run_analysis_from_execution", fake_run)
    result = runner.invoke(
        cli,
        [
            "fe",
            "analyze",
            str(tmp_path),
            "run1",
            "--no-raise-on-error",
        ],
    )
    assert result.exit_code == 0
    assert called["raise_on_error"] is False


def test_cli_fe_analyze_uses_latest_when_run_id_omitted(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    called: dict[str, Any] = {}

    def fake_run(
        work_dir,
        run_id,
        *,
        ligand,
        components=None,
        n_workers,
        analysis_start_step,
        overwrite=True,
        raise_on_error=True,
    ):
        called["work_dir"] = work_dir
        called["run_id"] = run_id

    monkeypatch.setattr("batter.cli.fe_cmds.run_analysis_from_execution", fake_run)
    monkeypatch.setattr("batter.api.run_analysis_from_execution", fake_run)

    result = runner.invoke(cli, ["fe", "analyze", str(tmp_path)])

    assert result.exit_code == 0
    assert called["work_dir"] == tmp_path
    assert called["run_id"] is None
    assert "latest execution" in result.output


def _copy_finished_run(tmp_path: Path) -> Path:
    src = FE_FINISHED_EXECUTION_DIR
    dest = tmp_path / "work" / "executions" / "rep1"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dest,
        dirs_exist_ok=True,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    return tmp_path / "work"


def test_cli_fe_analyze_on_finished_run(
    tmp_path: Path, runner: CliRunner, monkeypatch
) -> None:
    work_dir = _copy_finished_run(tmp_path)

    # failed because one ligand is missing files
    called: list[bool] = []

    def fake_run(*args, raise_on_error=True, **kwargs):
        called.append(raise_on_error)
        if raise_on_error:
            raise RuntimeError("boom")

    monkeypatch.setattr("batter.api.run_analysis_from_execution", fake_run)
    monkeypatch.setattr("batter.cli.fe_cmds.run_analysis_from_execution", fake_run)

    result = runner.invoke(cli, ["fe", "analyze", str(work_dir), "rep1"])
    assert result.exit_code == 1

    result = runner.invoke(
        cli,
        ["fe", "analyze", str(work_dir), "rep1", "--no-raise-on-error"],
    )
    assert result.exit_code == 0
    assert called == [True, False]


def test_cli_clone_exec(tmp_path: Path, runner: CliRunner, monkeypatch) -> None:
    work_dir = _copy_finished_run(tmp_path)
    src_exec = work_dir / "executions" / "src"
    src_exec.mkdir(parents=True)

    called = {}

    def fake_clone_execution(**kwargs):
        called.update(kwargs)
        return work_dir / "executions" / "src-clone"

    monkeypatch.setattr("batter.cli.exec_cmds.clone_execution", fake_clone_execution)

    result = runner.invoke(
        cli,
        [
            "clone-exec",
            str(work_dir),
            "src",
            "--only-equil",
            "--mode",
            "symlink",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert called["src_run_id"] == "src"
    assert called["only_equil"] is True
    assert called["mode"] == "symlink"
    assert called["overwrite"] is True
