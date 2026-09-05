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


def test_batch_cli_help_reports_remd_as_default() -> None:
    result = CliRunner().invoke(cli, ["batch", "--help"])

    assert result.exit_code == 0, result.output
    assert "--remd / --no-remd" in result.output
    assert "[default: remd]" in result.output


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
    if not remd:
        args.insert(1, "--no-remd")

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


def test_collect_remd_tasks_refreshes_grouped_runtime_helpers(
    tmp_path, monkeypatch
) -> None:
    exec_path = tmp_path / "executions" / "rep1"
    comp_dir = _setup_abfe_component(exec_path, ligand="L1", comp="z")
    run_script = comp_dir / "run-local-remd.bash"
    check_script = comp_dir / "check_run.bash"
    run_script.write_text("#!/bin/bash\nN_WINDOWS=99\n# stale\n")
    check_script.write_text("# stale\n")

    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])

    tasks = batch_cmds._collect_remd_tasks(exec_path)

    assert len(tasks) == 1
    assert tasks[0].n_windows == 1
    assert "N_WINDOWS=1" in run_script.read_text()
    assert "archive_failed_md_segment" in run_script.read_text()
    assert check_script.read_text() == batch_cmds.BATCH_CHECK_TEMPLATE.read_text()


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


def test_batch_cli_defaults_to_remd_and_renders_run_local_remd(
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
            "-e",
            str(exec_path),
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "bash ./run-local-remd.bash" in text
    assert "bash ./run-local-batch.bash" not in text
    assert "#SBATCH --time=00:15:00" in text
    assert " batch --remd " in text
    _assert_header_mpi_flags_are_authoritative(text)


def test_batch_cli_no_remd_uses_standard_runner_and_preserves_mode_on_resubmit(
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
            "--no-remd",
            "-e",
            str(exec_path),
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "bash ./run-local-batch.bash" in text
    assert " batch --no-remd " in text
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
            "-e",
            str(exec_path),
            "--no-auto-resubmit",
        ],
    )

    assert result.exit_code != 0
    assert "missing" in result.output
    assert "simulations/transformations" in result.output
    assert "fe/x" in result.output



def test_collect_remd_marks_finished_using_only_window_zero(tmp_path, monkeypatch):
    execution = tmp_path / "executions/rep1"
    comp = _setup_abfe_component(execution, ligand="L1", comp="z")
    (comp / "run-local-remd.bash").write_text("N_WINDOWS=2\n")
    (comp / "z01").mkdir()
    w0 = comp / "z00"
    (w0 / "mdin-remd-template").write_text(
        "! total_steps=2500000\n! target_dt=0.004\n dt=0.002,\n"
    )
    (w0 / "production-start.ps").write_text("50\n")
    restart = w0 / "md-01.rst7"
    monkeypatch.setattr(batch_cmds, "components_under", lambda _: ["z"])
    monkeypatch.setattr(batch_cmds, "_remd_time_from_rst", lambda p: p.read_text())
    # 9899 ps production: still pending, despite including 50 ps equilibration.
    restart.write_text("9949")
    assert len(batch_cmds._collect_remd_tasks(execution)) == 1
    assert not (comp / "FINISHED").exists()
    # 9900 ps production: within 100 ps, even without any z01 restart.
    restart.write_text("9950")
    assert batch_cmds._collect_remd_tasks(execution) == []
    for directory in (comp, w0, comp / "z01"):
        assert (directory / "FINISHED").read_text() == "FINISHED\n"
    # Existing component completion also repairs a missing window marker.
    (comp / "z01/FINISHED").unlink()
    assert batch_cmds._collect_remd_tasks(execution) == []
    assert (comp / "z01/FINISHED").exists()
