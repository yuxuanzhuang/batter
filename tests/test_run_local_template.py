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
    (work / "eq.rst7").write_text("eqrst")
    # total_steps=20, nstlim=10 → two segments
    (work / "mdin-template").write_text(
        "! total_steps=20\n"
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
seg=0
if [[ "$out" =~ md-([0-9]+)\\.out ]]; then
  seg=$((10#${BASH_REMATCH[1]}))
elif [[ "$out" =~ md([0-9]+)\\.out ]]; then
  seg=$((10#${BASH_REMATCH[1]}))
fi
chunk_ps=""
if [[ -f mdin-current ]]; then
  nstlim=$(sed -nE 's/.*nstlim[[:space:]]*=[[:space:]]*([0-9]+).*/\\1/p' mdin-current | head -n 1)
  dt=$(sed -nE 's/.*dt[[:space:]]*=[[:space:]]*([-+0-9.eEdD]+).*/\\1/p' mdin-current | head -n 1)
  if [[ -n "$dt" ]]; then dt=${dt//d/e}; dt=${dt//D/e}; fi
  if [[ -z "$dt" ]]; then dt=0.001; fi
  if [[ -n "$nstlim" ]]; then
    chunk_ps=$(awk -v n="$nstlim" -v dt="$dt" 'BEGIN{printf "%.6f", n*dt}')
  fi
fi
if [[ -n "$out" ]]; then
  if [[ "$seg" -gt 0 && -n "$chunk_ps" ]]; then
    time=$(awk -v s="$seg" -v c="$chunk_ps" 'BEGIN{printf "%.6f", s*c}')
    echo "TIME(PS) = $time" > "$out"
  else
    echo "ok" > "$out"
  fi
fi
if [[ -n "$rst" ]]; then
  if [[ "$seg" -gt 0 && -n "$chunk_ps" ]]; then
    time=$(awk -v s="$seg" -v c="$chunk_ps" 'BEGIN{printf "%.10f", s*c}')
    echo "time=$time" > "$rst"
  else
    echo "time=0.0" > "$rst"
  fi
fi
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
    ncdump_stub = work / "ncdump"
    _write_stub_exe(
        ncdump_stub,
        """#!/usr/bin/env bash
file=""
for arg in "$@"; do
  if [[ "$arg" != -* ]]; then
    file="$arg"
  fi
done
time=$(sed -nE 's/^time=([0-9.+-eE]+).*/\\1/p' "$file" | tail -n 1)
if [[ -z "$time" ]]; then time=0; fi
cat <<EOF
        double time ;
                time:units = "picosecond" ;
 time = $time ;
EOF
exit 0
""",
    )

    env = os.environ.copy()
    env["PMEMD_EXEC"] = str(stub)
    env["CPPTRAJ_EXEC"] = str(cpptraj_stub)
    env["PATH"] = f"{work}:{env.get('PATH','')}"

    cmd = ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"]
    subprocess.run(cmd, cwd=work, check=True, env=env)
    assert (work / "md-current.rst7").exists()
    assert not (work / "output.pdb").exists()

    subprocess.run(cmd, cwd=work, check=True, env=env)

    # After two segments we should have rolling restarts and output
    assert (work / "md-current.rst7").exists()
    assert (work / "md-previous.rst7").exists()
    assert (work / "output.pdb").exists()
