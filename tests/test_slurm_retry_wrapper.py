from __future__ import annotations

import subprocess
from pathlib import Path


def _slurmm_am_body() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "batter"
        / "_internal"
        / "templates"
        / "run_files_orig"
        / "SLURMM-Am.body"
    )


def _run_slurmm_am_body(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    (tmp_path / "SLURMM-Am.body").write_text(_slurmm_am_body().read_text())
    (tmp_path / "run-local.bash").write_text("#!/bin/bash\nexit 1\n")

    return subprocess.run(
        ["bash", "SLURMM-Am.body"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def test_slurmm_am_body_increments_attempt_file_on_failure(tmp_path: Path) -> None:
    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "2"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"


def test_slurmm_am_body_does_not_increment_attempt_for_titan_xp_output(
    tmp_path: Path,
) -> None:
    archived = tmp_path / "WRONG_FAIL" / "20260101_job_attempt_1"
    archived.mkdir(parents=True)
    (archived / "eqnpt_eq.out").write_text("Running on GPU TITAN Xp\n")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "1"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
    assert "not incrementing failure count" in result.stdout


def test_slurmm_am_body_ignores_stale_titan_xp_archive_for_current_failure(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "WRONG_FAIL" / "20260101_010101_job_attempt_1"
    current = tmp_path / "WRONG_FAIL" / "20260102_010101_job_attempt_1"
    stale.mkdir(parents=True)
    current.mkdir(parents=True)
    (stale / "md-01.out").write_text("Running on GPU TITAN Xp\n")
    (current / "md-01.out").write_text("Running on GPU A100\n")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "2"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
    assert "TITAN Xp in WRONG_FAIL/20260101_010101_job_attempt_1" not in result.stdout
    assert "not incrementing failure count" not in result.stdout


def test_slurmm_am_body_ignores_stale_live_titan_xp_long_equil_output(
    tmp_path: Path,
) -> None:
    current = tmp_path / "WRONG_FAIL" / "20260728_222657_job_attempt_1"
    current.mkdir(parents=True)
    (tmp_path / "eqnpt_eq.out").write_text("CUDA Device Name: NVIDIA TITAN Xp\n")
    (current / "eqnpt_disappear.out").write_text(
        "MDOUT: eqnpt_disappear.out\nCUDA Device Name: NVIDIA A40\n"
    )

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "2"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
    assert "Detected TITAN Xp in eqnpt_eq.out" not in result.stdout
    assert "not incrementing failure count" not in result.stdout


def test_slurmm_am_body_empty_current_archive_marker_blocks_stale_titan_fallback(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "WRONG_FAIL" / "20260101_010101_job_attempt_1"
    stale.mkdir(parents=True)
    (stale / "eqnpt_eq.out").write_text("CUDA Device Name: NVIDIA TITAN Xp\n")
    (tmp_path / "ATTEMPT_FAILED_ARCHIVE").write_text("")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "2"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
    assert "not incrementing failure count" not in result.stdout


def test_slurmm_am_body_does_not_increment_attempt_for_titan_xp_md_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "md-01.out").write_text("Running on GPU TITAN Xp\n")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "1"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
    assert "not incrementing failure count" in result.stdout


def test_slurmm_am_body_increments_attempt_for_titan_xp_pre_md_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "mini.in.out").write_text("Running on GPU TITAN Xp\n")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "2"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
    assert "not incrementing failure count" not in result.stdout


def test_slurmm_am_body_retries_until_fifth_attempt(tmp_path: Path) -> None:
    (tmp_path / "job_attempt.txt").write_text("4\n")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "5"
    assert not (tmp_path / "FAILED").exists()


def test_slurmm_am_body_fails_on_fifth_attempt(tmp_path: Path) -> None:
    (tmp_path / "job_attempt.txt").write_text("5\n")

    result = _run_slurmm_am_body(tmp_path)

    assert result.returncode != 0
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "5"
    assert (tmp_path / "FAILED").read_text().strip() == "FAILED"
