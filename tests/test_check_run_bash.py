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
