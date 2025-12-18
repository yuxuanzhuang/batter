from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def _run_parse_total_steps(check_run: Path, tmpl: Path) -> subprocess.CompletedProcess[str]:
    cmd = (
        f"source {shlex.quote(str(check_run))}; "
        f"parse_total_steps {shlex.quote(str(tmpl))}"
    )
    return subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, cwd=tmpl.parent
    )


def test_parse_total_steps_prefers_last_marker(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = (
        repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    )
    tmpl = tmp_path / "mdin-template"
    tmpl.write_text(
        "! total_steps=10\n"
        "nstlim = 5,\n"
        "# total_steps = 24\n"
    )

    result = _run_parse_total_steps(check_run, tmpl)

    assert result.returncode == 0
    assert result.stdout.strip() == "24"


def test_parse_total_steps_requires_marker(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_run = (
        repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"
    )
    tmpl = tmp_path / "mdin-template"
    tmpl.write_text("nstlim = 100,\n")

    result = _run_parse_total_steps(check_run, tmpl)

    assert result.returncode != 0
    assert "total_steps comment not found" in result.stderr
