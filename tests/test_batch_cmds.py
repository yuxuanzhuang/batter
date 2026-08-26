from __future__ import annotations

from pathlib import Path

import pytest

from batter.cli import batch_cmds
from batter.cli.root import cli
from click.testing import CliRunner


def _setup_abfe_component(exec_path: Path, ligand: str = "L1", comp: str = "z") -> Path:
    lig_dir = exec_path / "simulations" / ligand
    comp_dir = lig_dir / "fe" / comp
    (comp_dir / f"{comp}-1").mkdir(parents=True, exist_ok=True)
    (comp_dir / f"{comp}00").mkdir(parents=True, exist_ok=True)
    return comp_dir


def _assert_header_mpi_flags_are_authoritative(text: str) -> None:
    assert (
        'if [[ "$use_srun" -eq 1 && -z "$mpi_flags" '
        '&& "$nodes" -gt 0 && "$win" -gt 0 ]]; then'
    ) in text
    assert (
        'mpi_flags="--nodes=${nodes} --ntasks=${win} --exclusive '
        '--gpus-per-task=1 --gpu-bind=closest"'
    ) in text
    assert 'mpi_flags="${mpi_flags} ${extra_flags}"' not in text


def test_parse_slurm_time_limit_minutes() -> None:
    assert batch_cmds._parse_slurm_time_limit_minutes("15") == 15
    assert batch_cmds._parse_slurm_time_limit_minutes("90:00") == 90
    assert batch_cmds._parse_slurm_time_limit_minutes("00:15:00") == 15
    assert (
        batch_cmds._parse_slurm_time_limit_minutes("2-01:30:00")
        == 2 * 24 * 60 + 90
    )


@pytest.mark.parametrize("remd", [False, True])
def test_batch_cli_rejects_signal_too_close_to_time_limit(
    tmp_path: Path, monkeypatch, remd: bool
) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    comp_dir = _setup_abfe_component(exec_path, ligand="L1", comp="z")
    if remd:
        (comp_dir / "run-local-remd.bash").write_text("#!/bin/bash\nN_WINDOWS=1\n")
    else:
        monkeypatch.setattr(
            batch_cmds,
            "_write_batch_run_script",
            lambda *args, **kwargs: comp_dir / "run-local-batch.bash",
        )
    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])

    out = tmp_path / ("remd.sbatch" if remd else "batch.sbatch")
    args = [
        "batch",
        "-e",
        str(exec_path),
        "--output",
        str(out),
        "--signal-mins",
        "14.5",
    ]
    if remd:
        args.insert(1, "--remd")

    runner = CliRunner()
    result = runner.invoke(cli, args)

    assert result.exit_code != 0
    assert "at least 1 minute shorter" in result.output
    assert not out.exists()


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


def test_remd_finished_time_uses_latest_numbered_restart(
    tmp_path: Path, monkeypatch
) -> None:
    comp_dir = tmp_path / "z"
    win0 = comp_dir / "z00"
    win0.mkdir(parents=True)
    (win0 / "md-01.rst7").write_text("one\n")
    (win0 / "md-03.rst7").write_text("three\n")
    (win0 / "md-current.rst7").write_text("legacy\n")

    monkeypatch.setattr(
        batch_cmds,
        "_remd_time_from_rst",
        lambda path: path.stem,
    )

    assert batch_cmds._remd_finished_time(comp_dir, "z") == "md-03"


def test_batch_cli_remd_renders_run_local_remd(
    tmp_path: Path, monkeypatch
) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    comp_dir = _setup_abfe_component(exec_path, ligand="L1", comp="z")
    (comp_dir / "run-local-remd.bash").write_text("#!/bin/bash\nN_WINDOWS=1\n")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    out = tmp_path / "remd.sbatch"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "batch",
            "--remd",
            "-e",
            str(exec_path),
            "--output",
            str(out),
            "--no-auto-resubmit",
        ],
    )

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "bash ./run-local-remd.bash" in text
    assert "bash ./run-local-batch.bash" not in text
    assert "#SBATCH --time=00:15:00" in text
    _assert_header_mpi_flags_are_authoritative(text)


def test_batch_cli_uses_header_mpi_flags_without_appending(
    tmp_path: Path, monkeypatch
) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    comp_dir = _setup_abfe_component(exec_path, ligand="L1", comp="z")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])
    monkeypatch.setattr(
        batch_cmds,
        "_write_batch_run_script",
        lambda *args, **kwargs: comp_dir / "run-local-batch.bash",
    )

    out = tmp_path / "batch.sbatch"
    result = CliRunner().invoke(
        cli,
        [
            "batch",
            "-e",
            str(exec_path),
            "--output",
            str(out),
            "--no-auto-resubmit",
        ],
    )

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "bash ./run-local-batch.bash" in text
    _assert_header_mpi_flags_are_authoritative(text)


def test_batch_cli_remd_explains_missing_rbfe_transformations(tmp_path: Path) -> None:
    exec_path = tmp_path / "executions" / "rep2"
    _setup_abfe_component(exec_path, ligand="L1", comp="z")
    config_dir = exec_path / "artifacts" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "rbfe_network.json").write_text('{"pairs": [["L1", "L2"]]}\n')

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "batch",
            "--remd",
            "-e",
            str(exec_path),
            "--no-auto-resubmit",
        ],
    )

    assert result.exit_code != 0
    assert "missing" in result.output
    assert "simulations/transformations" in result.output
    assert "fe/x" in result.output
