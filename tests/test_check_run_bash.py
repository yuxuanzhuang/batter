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


def test_write_mdin_current_preserves_lower_existing_dt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    tmpl = tmp_path / "mdin-template"
    current = tmp_path / "mdin-current"
    rendered = tmp_path / "rendered-current"

    tmpl.write_text("irest = 1,\nntx = 5,\nnstlim = 10,\ndt = 0.004,\n")
    current.write_text("irest = 1,\nntx = 5,\nnstlim = 6,\ndt = 0.003,\n")

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
    assert "dt=0.003" in rendered_text
