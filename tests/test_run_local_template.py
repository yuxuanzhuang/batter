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
    (work / "run.log").write_text("old log\n")
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
# honor either -x output.pdb or stdin trajout output.pdb ...
target=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-x" ]]; then shift; target="$1"; break; fi
  shift
done
if [[ -z "$target" ]]; then
  target=$(awk '/^trajout[[:space:]]+/ { print $2; exit }' < /dev/stdin)
fi
if [[ -n "$target" ]]; then
  echo "pdb" > "$target"
fi
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
    archived_logs = list((work / "ARCHIVED_LOGS").glob("*_run.log"))
    assert len(archived_logs) == 1
    assert archived_logs[0].read_text() == "old log\n"
    assert (work / "run.log").exists()
    assert (work / "md-current.rst7").exists()
    assert not (work / "output.pdb").exists()

    subprocess.run(cmd, cwd=work, check=True, env=env)

    # After two segments we should have rolling restarts and output
    assert (work / "md-current.rst7").exists()
    assert (work / "md-previous.rst7").exists()
    assert (work / "output.pdb").exists()


def test_run_local_cleans_empty_md_artifacts_before_restart(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())

    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text("eqrst")
    (work / "run.log").write_text("old log\n")
    (work / "mdin-template").write_text(
        "! total_steps=10\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 10,\n"
        "dt = 0.001,\n"
    )

    for name in ["md-01.out", "md-01.nc", "cmass.txt", "md-current.rst7"]:
        (work / name).write_text("")

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
if [[ -n "$out" ]]; then
  echo "TIME(PS) = 0.010000" > "$out"
fi
if [[ -n "$rst" ]]; then
  echo "time=0.0100000000" > "$rst"
fi
[[ -n "$nc" ]] && echo "ok" > "$nc"
exit 0
""",
    )
    cpptraj_stub = work / "cpptraj"
    _write_stub_exe(
        cpptraj_stub,
        """#!/usr/bin/env bash
target=$(awk '/^trajout[[:space:]]+/ { print $2; exit }' < /dev/stdin)
if [[ -n "$target" ]]; then
  echo "pdb" > "$target"
fi
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

    result = subprocess.run(
        ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"],
        cwd=work,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[INFO] Removed stale empty file md-01.out" in result.stdout
    assert "[INFO] Removed stale empty file md-current.rst7" in result.stdout
    assert "Running segment 1 -> md-01.out" in result.stdout
    assert not (work / "ATTEMPT_FAILED").exists()
    assert (work / "md-01.out").read_text().strip() == "TIME(PS) = 0.010000"
    assert (work / "md-current.rst7").read_text().strip() == "time=0.0100000000"
    assert (work / "md-01.nc").read_text().strip() == "ok"
    assert not (work / "cmass.txt").exists()


def test_run_local_archives_incomplete_md_out_before_restart(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())

    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text("eqrst")
    (work / "run.log").write_text("old log\n")
    (work / "mdin-template").write_text(
        "! total_steps=10\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 10,\n"
        "dt = 0.001,\n"
    )
    (work / "md-01.out").write_text("job started but amber never wrote headers\n")
    (work / "md-01.nc").write_text("partial nc\n")

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
[[ -n "$out" ]] && echo "TIME(PS) = 0.010000" > "$out"
[[ -n "$rst" ]] && echo "time=0.0100000000" > "$rst"
[[ -n "$nc" ]] && echo "ok" > "$nc"
exit 0
""",
    )
    cpptraj_stub = work / "cpptraj"
    _write_stub_exe(
        cpptraj_stub,
        """#!/usr/bin/env bash
target=$(awk '/^trajout[[:space:]]+/ { print $2; exit }' < /dev/stdin)
[[ -n "$target" ]] && echo "pdb" > "$target"
exit 0
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
[[ -n "$time" ]] || time=0
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

    result = subprocess.run(
        ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"],
        cwd=work,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[INFO] Archived incomplete MD output md-01.out before restart." in result.stdout
    assert "Running segment 1 -> md-01.out" in result.stdout
    archived = list((work / "WRONG_FAIL").glob("*/md-01.out"))
    assert len(archived) == 1
    assert archived[0].read_text() == "job started but amber never wrote headers\n"
    assert (work / "md-01.out").read_text().strip() == "TIME(PS) = 0.010000"
    assert (work / "md-01.nc").read_text().strip() == "ok"


def test_run_local_remaining_steps_follow_reduced_dt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text("eqrst")
    (work / "md-current.rst7").write_text("time=0.0200000000\n")
    (work / "mdin-template").write_text(
        "! total_steps=10\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 10,\n"
        "dt = 0.004,\n"
    )

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
sed -nE 's/.*nstlim[[:space:]]*=[[:space:]]*([0-9]+).*/\\1/p' mdin-current | head -n 1 > run_steps.txt
[[ -n "$out" ]] && echo "TIME(PS) = 0.041000" > "$out"
[[ -n "$rst" ]] && echo "time=0.0410000000" > "$rst"
[[ -n "$nc" ]] && echo "ok" > "$nc"
exit 0
""",
    )
    cpptraj_stub = work / "cpptraj"
    _write_stub_exe(
        cpptraj_stub,
        """#!/usr/bin/env bash
target=$(awk '/^trajout[[:space:]]+/ { print $2; exit }' < /dev/stdin)
[[ -n "$target" ]] && echo "pdb" > "$target"
exit 0
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
[[ -n "$time" ]] || time=0
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
    env["PMEMD_DPFP_EXEC"] = str(stub)
    env["CPPTRAJ_EXEC"] = str(cpptraj_stub)
    env["RETRY_COUNT"] = "4"
    env["PATH"] = f"{work}:{env.get('PATH','')}"

    subprocess.run(
        ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"],
        cwd=work,
        check=True,
        env=env,
    )

    assert (work / "run_steps.txt").read_text().strip() == "7"
    mdin_text = (work / "mdin-template").read_text()
    assert "! target_dt=0.004" in mdin_text
    assert "dt=0.003000" in mdin_text


def test_run_local_scales_segment_steps_when_dt_reduced(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text(
        "Cpptraj Generated Restart\n"
        "64844  8.0000000E+01\n"
        "  1.0  2.0  3.0\n"
    )
    (work / "mdin-template").write_text(
        "! target_dt=0.004\n"
        "! total_steps=1000000\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 1000000,\n"
        "ntwr = 2500,\n"
        "dt = 0.002,\n"
    )

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
sed -nE 's/.*nstlim[[:space:]]*=[[:space:]]*([0-9]+).*/\\1/p' mdin-current | head -n 1 > run_steps.txt
[[ -n "$out" ]] && echo "TIME(PS) = 4080.000000" > "$out"
[[ -n "$rst" ]] && echo "time=4080.0000000000" > "$rst"
[[ -n "$nc" ]] && echo "ok" > "$nc"
exit 0
""",
    )
    cpptraj_stub = work / "cpptraj"
    _write_stub_exe(
        cpptraj_stub,
        """#!/usr/bin/env bash
target=$(awk '/^trajout[[:space:]]+/ { print $2; exit }' < /dev/stdin)
[[ -n "$target" ]] && echo "pdb" > "$target"
exit 0
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
[[ -n "$time" ]] || exit 1
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
    env["RETRY_COUNT"] = "6"
    env["PATH"] = f"{work}:{env.get('PATH','')}"

    result = subprocess.run(
        ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"],
        cwd=work,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "Running segment 1 -> md-01.out for 2000000 steps (4000.000000 ps)" in result.stdout
    assert (work / "run_steps.txt").read_text().strip() == "2000000"


def test_run_local_subtracts_initial_restart_time_for_production_progress(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text(
        "Cpptraj Generated Restart\n"
        "64844  8.0000000E+01\n"
        "  1.0  2.0  3.0\n"
    )
    (work / "md-current.rst7").write_text("time=2080.0000000000\n")
    (work / "md-01.out").write_text("CONTROL DATA FOR THE RUN\nTIME(PS) = 2080.000000\n")
    (work / "md-01.nc").write_text("traj\n")
    (work / "mdin-template").write_text(
        "! target_dt=0.004\n"
        "! total_steps=1000000\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 1000000,\n"
        "ntwr = 2500,\n"
        "dt = 0.002,\n"
    )

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
sed -nE 's/.*nstlim[[:space:]]*=[[:space:]]*([0-9]+).*/\\1/p' mdin-current | head -n 1 > run_steps.txt
[[ -n "$out" ]] && echo "TIME(PS) = 4080.000000" > "$out"
[[ -n "$rst" ]] && echo "time=4080.0000000000" > "$rst"
[[ -n "$nc" ]] && echo "ok" > "$nc"
exit 0
""",
    )
    cpptraj_stub = work / "cpptraj"
    _write_stub_exe(
        cpptraj_stub,
        """#!/usr/bin/env bash
target=$(awk '/^trajout[[:space:]]+/ { print $2; exit }' < /dev/stdin)
[[ -n "$target" ]] && echo "pdb" > "$target"
exit 0
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
[[ -n "$time" ]] || exit 1
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
    env["RETRY_COUNT"] = "6"
    env["PATH"] = f"{work}:{env.get('PATH','')}"

    result = subprocess.run(
        ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"],
        cwd=work,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "Current completed production time: 2000 ps / 4000.000000 ps" in result.stdout
    assert (work / "production-start.ps").read_text().strip() == "8.0000000E+01"
    assert (work / "run_steps.txt").read_text().strip() == "1000000"
