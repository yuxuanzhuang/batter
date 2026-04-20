from __future__ import annotations

import subprocess
from pathlib import Path


def test_slurmm_am_body_increments_attempt_file_on_failure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    body = (
        repo_root
        / "batter"
        / "_internal"
        / "templates"
        / "run_files_orig"
        / "SLURMM-Am.body"
    )

    (tmp_path / "SLURMM-Am.body").write_text(body.read_text())
    (tmp_path / "run-local.bash").write_text("#!/bin/bash\nexit 1\n")

    result = subprocess.run(
        ["bash", "SLURMM-Am.body"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "job_attempt.txt").read_text().strip() == "2"
    assert (tmp_path / "ATTEMPT_FAILED").read_text().strip() == "FAILED"
