from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_stub_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _restart_text(time_ps: str) -> str:
    return (
        "Stub Amber restart\n"
        f"1  {time_ps}\n"
        "  0.0  0.0  0.0\n"
    )


def _restart_time(path: Path) -> str:
    return path.read_text().splitlines()[1].split()[1]


def test_production_md_uses_expected_reference_restart() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / "batter" / "_internal" / "templates" / "run_files_orig"

    expected_refs = {
        "run-local.bash": "${win_00}/eq.rst7",
        "run-local-rbfe.bash": "${win_00}/eq.rst7",
        "run-local-vacuum.bash": "$rst_in",
        "run-equil.bash": "$rst_in",
    }

    for name, expected_ref in expected_refs.items():
        text = (template_dir / name).read_text()
        assert (
            "-c $rst_in -o ${out_tag}.out -r $rst_out "
            f"-x ${{out_tag}}.nc -ref {expected_ref}"
        ) in text


def test_abfe_local_template_uses_merged_prmtop_for_cpptraj_restart_split() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"

    text = script.read_text()
    assert "$CPPTRAJ_EXEC -p $PRMTOP_MERGED -i /dev/stdin" in text
    assert "$CPPTRAJ_EXEC -p full.prmtop -i /dev/stdin" not in text


def test_abfe_equil_template_uses_merged_prmtop() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-equil.bash"

    text = script.read_text()
    assert 'PRMTOP="full_merged.prmtop"' in text
    assert 'PRMTOP="full.hmr.prmtop"' not in text


def test_run_local_handles_template_segments(tmp_path: Path, monkeypatch) -> None:
    """run-local.bash should honor mdin-template total_steps via explicit segment restarts."""
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
restart=""
ref=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) shift; restart="$1";;
    -o) shift; out="$1";;
    -r) shift; rst="$1";;
    -x) shift; nc="$1";;
    -ref) shift; ref="$1";;
  esac
  shift
done
if [[ "$out" == md-*.out ]]; then
  printf "%s %s %s\\n" "$out" "$restart" "$ref" >> md_ref_calls.txt
fi
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
    printf "Stub Amber restart\\n1  %s\\n  0.0  0.0  0.0\\n" "$time" > "$rst"
  else
    printf "Stub Amber restart\\n1  0.0\\n  0.0  0.0  0.0\\n" > "$rst"
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
    assert (work / "md-01.rst7").exists()
    assert not (work / "output.pdb").exists()

    subprocess.run(cmd, cwd=work, check=True, env=env)

    # After completion, segment restarts should be cleaned up.
    assert not (work / "md-01.rst7").exists()
    assert not (work / "md-02.rst7").exists()
    assert (work / "output.pdb").exists()
    assert (work / "md_ref_calls.txt").read_text().splitlines() == [
        "md-01.out eq.rst7 ../COMPONENT00/eq.rst7",
        "md-02.out md-01.rst7 ../COMPONENT00/eq.rst7",
    ]


def test_run_local_does_not_skip_exact_100ps_debug_window_before_md(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text(_restart_text("2.0000000E+01"))
    (work / "mdin-template").write_text(
        "! target_dt=0.004\n"
        "! total_steps=25000\n"
        "&cntrl\n"
        "  irest = 1,\n"
        "  ntx = 5,\n"
        "  ntwr = 2500,\n"
        "  ntwx = 25000,\n"
        "  nstlim = 25000,\n"
        "  dt = 0.001,\n"
        "/\n"
        " &wt type='DUMPFREQ', istep1=1000, /\n"
        "DUMPAVE=cmass.txt\n"
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
sed -nE 's/.*ntwx[[:space:]]*=[[:space:]]*([0-9]+).*/\\1/p' mdin-current | head -n 1 > ntwx.txt
sed -nE 's/.*istep1[[:space:]]*=[[:space:]]*([0-9]+).*/\\1/p' mdin-current | head -n 1 > dumpfreq.txt
[[ -n "$out" ]] && echo "TIME(PS) = 120.000000" > "$out"
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  120.0000000000\\n  0.0  0.0  0.0\\n" > "$rst"
[[ -n "$nc" ]] && echo "ok" > "$nc"
echo "cmass" > cmass-01.txt
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

    env = os.environ.copy()
    env["PMEMD_EXEC"] = str(stub)
    env["CPPTRAJ_EXEC"] = str(cpptraj_stub)
    env["PATH"] = f"{work}:{env.get('PATH','')}"

    subprocess.run(
        ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"],
        cwd=work,
        check=True,
        env=env,
    )

    assert (work / "run_steps.txt").read_text().strip() == "25000"
    assert (work / "ntwx.txt").read_text().strip() == "25000"
    assert (work / "dumpfreq.txt").read_text().strip() == "1000"
    assert not (work / "md-01.rst7").exists()
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

    for name in ["md-01.out", "md-01.nc", "cmass.txt", "cmass-01.txt", "md-01.rst7"]:
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
  printf "Stub Amber restart\\n1  0.0100000000\\n  0.0  0.0  0.0\\n" > "$rst"
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
    assert "[INFO] Removed stale empty file cmass-01.txt" in result.stdout
    assert "[INFO] Removed stale empty file md-01.rst7" in result.stdout
    assert "Running segment 1 -> md-01.out" in result.stdout
    assert not (work / "ATTEMPT_FAILED").exists()
    assert (work / "md-01.out").read_text().strip() == "TIME(PS) = 0.010000"
    assert not (work / "md-01.rst7").exists()
    assert (work / "md-01.nc").read_text().strip() == "ok"
    assert not (work / "cmass.txt").exists()
    assert not (work / "cmass-01.txt").exists()


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
    (work / "cmass-01.txt").write_text("partial cmass\n")

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
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  0.0100000000\\n  0.0  0.0  0.0\\n" > "$rst"
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
    archived_cmass = list((work / "WRONG_FAIL").glob("*/cmass-01.txt"))
    assert len(archived_cmass) == 1
    assert archived_cmass[0].read_text() == "partial cmass\n"
    assert (work / "md-01.out").read_text().strip() == "TIME(PS) = 0.010000"
    assert (work / "md-01.nc").read_text().strip() == "ok"


def test_run_local_resumes_from_explicit_segment_restart(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text(_restart_text("20.0000000000"))
    (work / "md-01.rst7").write_text(_restart_text("50.0000000000"))
    (work / "md-01.out").write_text(
        "CONTROL DATA FOR THE RUN\n"
        " NSTEP =    11700   TIME(PS) =      55.100  TEMP(K) =   298.0\n"
    )
    (work / "md-01.nc").write_text("partial traj\n")
    (work / "mdin-template").write_text(
        "! total_steps=200000\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 10,\n"
        "dt = 0.001,\n"
    )

    stub = work / "stub.sh"
    _write_stub_exe(
        stub,
        """#!/usr/bin/env bash
out=""
rst=""
nc=""
restart=""
ref=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) shift; restart="$1";;
    -o) shift; out="$1";;
    -r) shift; rst="$1";;
    -x) shift; nc="$1";;
    -ref) shift; ref="$1";;
  esac
  shift
done
printf "%s\\n" "$restart" > restart_in.txt
printf "%s\\n" "$ref" > reference_in.txt
[[ -n "$out" ]] && printf "CONTROL DATA FOR THE RUN\\n|  Final Performance Info:\\n|  Total wall time: 1 seconds\\nTIME(PS) = 50.010000\\n" > "$out"
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  50.0100000000\\n  0.0  0.0  0.0\\n" > "$rst"
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

    assert "Archived incomplete MD segment" not in result.stdout
    assert "Running segment 2 -> md-02.out" in result.stdout
    assert (work / "restart_in.txt").read_text().strip() == "md-01.rst7"
    assert (work / "reference_in.txt").read_text().strip() == "../COMPONENT00/eq.rst7"
    assert (work / "md-01.out").exists()
    assert (work / "md-01.nc").exists()
    assert _restart_time(work / "md-01.rst7") == "50.0000000000"
    assert _restart_time(work / "md-02.rst7") == "50.0100000000"
    assert (work / "md-02.out").exists()
    assert not (work / "WRONG_FAIL").exists()


def test_run_local_archives_invalid_segment_restart_and_uses_previous_segment(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    work = tmp_path
    (work / "run-local.bash").write_text(script.read_text())
    (work / "check_run.bash").write_text(check_run.read_text())
    (work / "full.hmr.prmtop").write_text("prmtop")
    (work / "full_merged.prmtop").write_text("prmtop")
    (work / "eq.rst7").write_text(_restart_text("20.0000000000"))
    (work / "md-01.rst7").write_text(_restart_text("50.0000000000"))
    (work / "md-02.rst7").write_text("not a restart\n")
    (work / "md-01.out").write_text(
        "CONTROL DATA FOR THE RUN\n"
        "|  Final Performance Info:\n"
        " NSTEP =    10000   TIME(PS) =      50.000  TEMP(K) =   298.0\n"
    )
    (work / "md-02.out").write_text(
        "CONTROL DATA FOR THE RUN\n"
        " NSTEP =    11700   TIME(PS) =      55.100  TEMP(K) =   298.0\n"
    )
    (work / "md-02.nc").write_text("partial traj\n")
    (work / "mdin-template").write_text(
        "! total_steps=200000\n"
        "irest = 1,\n"
        "ntx   = 5,\n"
        "nstlim = 10,\n"
        "dt = 0.001,\n"
    )

    stub = work / "stub.sh"
    _write_stub_exe(
        stub,
        """#!/usr/bin/env bash
out=""
rst=""
nc=""
restart=""
ref=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) shift; restart="$1";;
    -o) shift; out="$1";;
    -r) shift; rst="$1";;
    -x) shift; nc="$1";;
    -ref) shift; ref="$1";;
  esac
  shift
done
printf "%s\\n" "$restart" > restart_in.txt
printf "%s\\n" "$ref" > reference_in.txt
[[ -n "$out" ]] && printf "CONTROL DATA FOR THE RUN\\n|  Final Performance Info:\\n|  Total wall time: 1 seconds\\nTIME(PS) = 50.010000\\n" > "$out"
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  50.0100000000\\n  0.0  0.0  0.0\\n" > "$rst"
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

    assert "Archived invalid MD restart md-02.rst7" in result.stdout
    assert "Running segment 2 -> md-02.out" in result.stdout
    assert (work / "restart_in.txt").read_text().strip() == "md-01.rst7"
    assert (work / "reference_in.txt").read_text().strip() == "../COMPONENT00/eq.rst7"
    assert _restart_time(work / "md-02.rst7") == "50.0100000000"
    assert (work / "md-02.out").read_text().startswith("CONTROL DATA FOR THE RUN")
    assert list((work / "WRONG_FAIL").glob("*/md-02.rst7"))
    assert list((work / "WRONG_FAIL").glob("*/md-02.out"))
    assert list((work / "WRONG_FAIL").glob("*/md-02.nc"))


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
    (work / "md-01.rst7").write_text(_restart_text("0.0200000000"))
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
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  0.0410000000\\n  0.0  0.0  0.0\\n" > "$rst"
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

    assert (work / "run_steps.txt").read_text().strip() == "10"
    mdin_text = (work / "mdin-template").read_text()
    assert "! target_dt=0.004" in mdin_text
    assert "dt=0.002000" in mdin_text


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
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  4080.0000000000\\n  0.0  0.0  0.0\\n" > "$rst"
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

    assert "Running segment 1 -> md-01.out for 4000000 steps (4000.000000 ps)" in result.stdout
    assert (work / "run_steps.txt").read_text().strip() == "4000000"


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
    (work / "md-01.rst7").write_text(_restart_text("2080.0000000000"))
    (work / "md-01.out").write_text(
        "CONTROL DATA FOR THE RUN\n"
        "|  Final Performance Info:\n"
        "|  Total wall time: 1 seconds\n"
        "TIME(PS) = 2080.000000\n"
    )
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
[[ -n "$rst" ]] && printf "Stub Amber restart\\n1  4080.0000000000\\n  0.0  0.0  0.0\\n" > "$rst"
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
    assert (work / "run_steps.txt").read_text().strip() == "2000000"
