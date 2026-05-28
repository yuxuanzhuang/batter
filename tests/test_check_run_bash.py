from __future__ import annotations

from pathlib import Path
import subprocess


def _run_check_min_energy(output_file: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = (
        f"source '{repo_root}/batter/_internal/templates/run_files_orig/check_run.bash' "
        f"&& check_min_energy '{output_file}' -1000"
    )
    return subprocess.run(
        ["bash", "-lc", cmd],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def test_check_min_energy_prefers_eamber():
    data_dir = Path(__file__).resolve().parent / "data" / "md_output"
    result = _run_check_min_energy(data_dir / "mini.out")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "EAMBER: -64851.9795 kcal/mol" in result.stdout


def test_archive_existing_log_file_moves_log(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    log_file = tmp_path / "run.log"
    log_file.write_text("old log\n")

    cmd = (
        f"source '{check_run}' "
        f"&& archive_existing_log_file '{log_file}'"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    archived_logs = list((tmp_path / "ARCHIVED_LOGS").glob("*_run.log"))
    assert len(archived_logs) == 1
    assert archived_logs[0].read_text() == "old log\n"
    assert not log_file.exists()


def test_reduce_dt_on_failure_updates_template_and_current(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"
    current = tmp_path / "mdin-current"

    tmpl.write_text("nstlim = 10,\ndt = 0.004,\n")
    current.write_text("nstlim = 8,\ndt = 0.004,\n")

    cmd = (
        f"source '{check_run}' "
        "&& reduce_dt_on_failure 'mdin-template' 0.001 'MD segment 1' 3"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dt=0.003000" in tmpl.read_text()
    assert "dt=0.003000" in current.read_text()
    assert "! target_dt=0.004" in tmpl.read_text()


def test_target_dt_and_remaining_steps_use_reduced_dt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"

    tmpl.write_text("! target_dt=0.004\n! total_steps=10\nnstlim = 10,\ndt = 0.003,\n")

    cmd = (
        f"source '{check_run}' "
        "&& dt=$(parse_dt_ps mdin-template) "
        "&& target_dt=$(parse_target_dt_ps mdin-template) "
        "&& total=$(parse_total_steps mdin-template) "
        "&& total_ps=$(awk -v s=\"$total\" -v dt=\"$target_dt\" 'BEGIN{printf \"%.6f\", s*dt}') "
        "&& remaining_steps_from_time \"$total_ps\" 0.020 \"$dt\""
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "7"


def test_scaled_nstlim_uses_target_dt_duration(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"

    tmpl.write_text("! target_dt=0.004\nnstlim = 1000000,\ndt = 0.002,\n")

    cmd = f"source '{check_run}' && scaled_nstlim_for_dt mdin-template"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "2000000"


def test_apply_retry_dt_reduction_is_idempotent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"

    tmpl.write_text("! total_steps=10\nnstlim = 10,\ndt = 0.004,\n")

    cmd = (
        f"source '{check_run}' "
        "&& apply_retry_dt_reduction mdin-template 4 0.001 test "
        "&& apply_retry_dt_reduction mdin-template 4 0.001 test"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = tmpl.read_text()
    assert "! target_dt=0.004" in text
    assert "dt=0.003000" in text


def test_retry_dt_schedule_uses_attempt_thresholds(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"

    tmpl.write_text("! target_dt=0.004\nnstlim = 10,\ndt = 0.004,\n")

    cmd = (
        f"source '{check_run}' "
        "&& retry_adjusted_dt_ps mdin-template 1 "
        "&& retry_adjusted_dt_ps mdin-template 3 "
        "&& retry_adjusted_dt_ps mdin-template 5 "
        "&& retry_adjusted_dt_ps mdin-template 6 "
        "&& retry_adjusted_dt_ps mdin-template 8 "
        "&& retry_adjusted_dt_ps mdin-template 9 "
        "&& retry_adjusted_dt_ps mdin-template 10"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "0.004000",
        "0.003000",
        "0.003000",
        "0.002000",
        "0.002000",
        "0.001000",
        "0.001000",
    ]


def test_write_mdin_current_uses_job_attempt_dt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"
    current = tmp_path / "mdin-current"
    rendered = tmp_path / "rendered-current"

    tmpl.write_text("! target_dt=0.004\nirest = 1,\nntx = 5,\nnstlim = 10,\ndt = 0.004,\n")
    current.write_text("irest = 1,\nntx = 5,\nnstlim = 6,\ndt = 0.004,\n")
    (tmp_path / "job_attempt.txt").write_text("5\n")

    cmd = (
        f"source '{check_run}' "
        "&& write_mdin_current 'mdin-template' 8 0 'mdin-current' > 'rendered-current'"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered_text = rendered.read_text()
    assert "nstlim = 8," in rendered_text
    assert "dt=0.003000" in rendered_text


def test_apply_retry_dt_reduction_corrects_template_and_regenerates_current(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"
    current = tmp_path / "mdin-current"

    tmpl.write_text("! target_dt=0.004\nirest = 1,\nntx = 5,\nnstlim = 10,\ndt = 0.001,\n")
    current.write_text("irest = 1,\nntx = 5,\nnstlim = 8,\ndt = 0.001,\n")
    (tmp_path / "job_attempt.txt").write_text("4\n")

    cmd = (
        f"source '{check_run}' "
        "&& apply_retry_dt_reduction 'mdin-template' '' 0.001 'startup'"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dt=0.003000" in tmpl.read_text()
    current_text = current.read_text()
    assert "nstlim = 8," in current_text
    assert "dt=0.003000" in current_text


def test_write_mdin_current_same_file_redirect_keeps_template_dt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"
    current = tmp_path / "mdin-current"

    tmpl.write_text("irest = 1,\nntx = 5,\nnstlim = 10,\ndt = 0.004,\n")
    current.write_text("old current content\n")

    cmd = (
        f"source '{check_run}' "
        "&& write_mdin_current 'mdin-template' 8 1 'mdin-current' > 'mdin-current'"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    current_text = current.read_text()
    assert "nstlim = 8," in current_text
    assert "dt = 0.004," in current_text


def test_completed_time_ps_from_ascii_restart(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    rst = tmp_path / "eq.rst7"
    rst.write_text(
        "Cpptraj Generated Restart\n"
        "64844  8.0000000E+01\n"
        "  1.0  2.0  3.0\n"
    )

    cmd = f"source '{check_run}' && completed_time_ps_from_rst '{rst}'"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "8.0000000E+01"


def test_cleanup_stale_md_artifacts_strict_archives_suspect_current_restart(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    (tmp_path / "mdin-template").write_text("nstlim = 10,\ndt = 0.004,\n")
    (tmp_path / "md-current.rst7").write_text("time=50.0000000000\n")
    (tmp_path / "md-01.out").write_text(
        "CONTROL DATA FOR THE RUN\n"
        " NSTEP =    10000   TIME(PS) =      50.000  TEMP(K) =   298.0\n"
    )

    cmd = f"source '{check_run}' && cleanup_stale_empty_md_artifacts strict"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "suspect restart md-current.rst7" in result.stdout
    assert not (tmp_path / "md-current.rst7").exists()
    assert not (tmp_path / "md-01.out").exists()
    assert list((tmp_path / "WRONG_FAIL").glob("*/md-current.rst7"))


def test_cleanup_stale_md_artifacts_relaxed_keeps_interrupted_current_restart(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    (tmp_path / "mdin-template").write_text("nstlim = 10,\ndt = 0.004,\n")
    (tmp_path / "md-current.rst7").write_text("time=50.0000000000\n")
    (tmp_path / "md-01.out").write_text(
        "CONTROL DATA FOR THE RUN\n"
        " NSTEP =    10000   TIME(PS) =      50.000  TEMP(K) =   298.0\n"
    )

    cmd = f"source '{check_run}' && cleanup_stale_empty_md_artifacts relaxed"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "md-current.rst7").exists()
    assert (tmp_path / "md-01.out").exists()
    assert not (tmp_path / "WRONG_FAIL").exists()
