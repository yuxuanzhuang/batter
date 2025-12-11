from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_stub_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def test_run_local_handles_template_segments(tmp_path: Path, monkeypatch) -> None:
    """run-local.bash should honor mdin-template total_steps via md-current rolling restarts."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())

    # minimal required inputs
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "mini.in.rst7").write_text("rst")
    # total_steps=20, nstlim=10 → two segments
    (work / "mdin-template").write_text(
        "! eq_steps=20\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 10,\n"
    )

    # stub pmemd/cpptraj that just writes requested outputs
    stub = work / "stub.sh"
    _write_stub_exe(
        stub,
        """#!/usr/bin/env bash
out=""
rst=""
nc=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) shift; out="$1";;
    -r) shift; rst="$1";;
    -x) shift; nc="$1";;
  esac
  shift
done
[[ -n "$out" ]] && echo "ok" > "$out"
[[ -n "$rst" ]] && echo "ok" > "$rst"
[[ -n "$nc" ]] && echo "ok" > "$nc"
exit 0
""",
    )
    cpptraj_stub = work / "cpptraj"
    _write_stub_exe(
        cpptraj_stub,
        """#!/usr/bin/env bash
# honor -x output.pdb
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-x" ]]; then shift; echo "pdb" > "$1"; exit 0; fi
  shift
done
""",
    )

    env = os.environ.copy()
    env["PMEMD_EXEC"] = str(stub)
    env["PATH"] = f"{work}:{env.get('PATH','')}"

    cmd = ["bash", "-lc", "source run-local.bash"]
    subprocess.run(cmd, cwd=work, check=True, env=env)

    # After two segments we should have rolling restarts and output
    assert (work / "md-current.rst7").exists()
    assert (work / "md-previous.rst7").exists()
    assert (work / "output.pdb").exists()
