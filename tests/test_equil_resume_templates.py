from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_file(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _make_pmemd_stub(path: Path) -> None:
    _write_file(
        path,
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
if [[ -n "${CALL_LOG:-}" && -n "$out" ]]; then
  echo "$out" >> "$CALL_LOG"
fi
if [[ -n "${FAIL_OUT:-}" && "$out" == "$FAIL_OUT" ]]; then
  printf "segmentation fault\\n"
fi
if [[ -n "$out" ]]; then
  if [[ "$out" == mini* ]]; then
    printf " EAMBER = -2000.0000\\n" > "$out"
  else
    printf "ok\\n" > "$out"
  fi
fi
[[ -n "$rst" ]] && printf "rst\\n" > "$rst"
[[ -n "$nc" ]] && printf "nc\\n" > "$nc"
exit 0
""",
        executable=True,
    )


def _make_cpptraj_stub(path: Path) -> None:
    _write_file(
        path,
        """#!/usr/bin/env bash
targets=()
while IFS= read -r line; do
  if [[ "$line" == trajout* ]]; then
    set -- $line
    targets+=("$2")
  fi
done
for target in "${targets[@]}"; do
  printf "pdb\\n" > "$target"
done
""",
        executable=True,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _common_env(work: Path) -> dict[str, str]:
    env = os.environ.copy()
    pmemd_stub = work / "pmemd-stub.sh"
    cpptraj_stub = work / "cpptraj"
    _make_pmemd_stub(pmemd_stub)
    _make_cpptraj_stub(cpptraj_stub)

    env["PMEMD_EXEC"] = str(pmemd_stub)
    env["PMEMD_DPFP_EXEC"] = str(pmemd_stub)
    env["PMEMD_CPU_EXEC"] = str(pmemd_stub)
    env["PMEMD_CPU_MPI_EXEC"] = str(pmemd_stub)
    env["CPPTRAJ_EXEC"] = str(cpptraj_stub)
    env["CALL_LOG"] = str(work / "call.log")
    env["PATH"] = f"{work}:{env.get('PATH', '')}"
    env["SLURM_JOB_CPUS_PER_NODE"] = "1"
    return env


def _read_calls(work: Path) -> list[str]:
    call_log = work / "call.log"
    if not call_log.exists():
        return []
    return [line.strip() for line in call_log.read_text().splitlines() if line.strip()]


def _setup_run_local_only_eq(work: Path) -> tuple[dict[str, str], list[str]]:
    repo_root = _repo_root()
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-local.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    script_text = script.read_text().replace("NWINDOWS", "1").replace("COMPONENT", "z")
    _write_file(work / "run-local.bash", script_text)
    _write_file(work / "check_run.bash", check_run.read_text())

    for name in [
        "full.hmr.prmtop",
        "full_merged.prmtop",
        "full.inpcrd",
        "mini.in",
        "eqnpt0.in",
        "eqnpt.in",
        "eqnpt_eq.in",
    ]:
        _write_file(work / name, "x\n")

    env = _common_env(work)
    env["ONLY_EQ"] = "1"
    cmd = ["bash", "-lc", f"PATH={work}:$PATH; source run-local.bash"]
    return env, cmd


def _setup_run_equil_only_eq(work: Path) -> tuple[dict[str, str], list[str]]:
    repo_root = _repo_root()
    script = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "run-equil.bash"
    check_run = repo_root / "batter" / "_internal" / "templates" / "run_files_orig" / "check_run.bash"

    _write_file(work / "run-equil.bash", script.read_text())
    _write_file(work / "check_run.bash", check_run.read_text())

    for name in [
        "full.hmr.prmtop",
        "full_merged.prmtop",
        "full.inpcrd",
        "mini.in",
        "eqnvt.in",
        "eqnpt0.in",
        "eqnpt.in",
        "eqnpt_eq.in",
        "eqnpt_disappear.in",
        "eqnpt_appear.in",
    ]:
        _write_file(work / name, "x\n")

    _write_file(
        work / "mdin-template",
        "! total_steps=0\n"
        "nstlim = 10,\n"
        "dt = 0.001,\n",
    )
    _write_file(
        work / "check_penetration.py",
        "from pathlib import Path\n"
        "Path('RING_PENETRATION').unlink(missing_ok=True)\n",
    )

    env = _common_env(work)
    env["ONLY_EQ"] = "1"
    cmd = ["bash", "run-equil.bash"]
    return env, cmd


def test_run_local_only_eq_skips_existing_steps(tmp_path: Path) -> None:
    env, cmd = _setup_run_local_only_eq(tmp_path)
    for name in ["mini.rst7", "mini2.rst7", "eqnpt_pre.rst7", "eqnpt00.rst7"]:
        _write_file(tmp_path / name, "rst\n")

    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)

    calls = _read_calls(tmp_path)
    assert "mini.out" not in calls
    assert "mini2.out" not in calls
    assert "eqnpt_pre.out" not in calls
    assert "eqnpt00.out" not in calls
    assert "eqnpt01.out" in calls
    assert "eqnpt_eq.out" in calls
    assert (tmp_path / "EQ_FINISHED").exists()


def test_run_local_only_eq_reruns_existing_steps_after_failure_when_enabled(tmp_path: Path) -> None:
    env, cmd = _setup_run_local_only_eq(tmp_path)
    env["RERUN_EQ_STEPS_AFTER_FAILURE"] = "1"
    for name in ["mini.rst7", "mini2.rst7", "eqnpt_pre.rst7", "eqnpt00.rst7"]:
        _write_file(tmp_path / name, "rst\n")
    _write_file(tmp_path / "ATTEMPT_FAILED", "FAILED\n")

    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)

    calls = _read_calls(tmp_path)
    assert "mini.out" in calls
    assert "mini2.out" in calls
    assert "eqnpt_pre.out" in calls
    assert "eqnpt00.out" in calls


def test_run_equil_skips_existing_steps(tmp_path: Path) -> None:
    env, cmd = _setup_run_equil_only_eq(tmp_path)
    for name in ["mini.rst7", "mini2.rst7", "eqnvt.rst7", "eqnpt_pre.rst7", "eqnpt00.rst7"]:
        _write_file(tmp_path / name, "rst\n")

    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)

    calls = _read_calls(tmp_path)
    assert "mini.out" not in calls
    assert "mini2.out" not in calls
    assert "eqnvt.out" not in calls
    assert "eqnpt_pre.out" not in calls
    assert "eqnpt00.out" not in calls
    assert "eqnpt01.out" in calls
    assert "eqnpt_appear.out" in calls


def test_run_equil_reruns_existing_steps_after_failure_when_enabled(tmp_path: Path) -> None:
    env, cmd = _setup_run_equil_only_eq(tmp_path)
    env["RERUN_EQ_STEPS_AFTER_FAILURE"] = "1"
    for name in ["mini.rst7", "mini2.rst7", "eqnvt.rst7", "eqnpt_pre.rst7", "eqnpt00.rst7"]:
        _write_file(tmp_path / name, "rst\n")
    _write_file(tmp_path / "ATTEMPT_FAILED", "FAILED\n")

    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)

    calls = _read_calls(tmp_path)
    assert "mini.out" in calls
    assert "mini2.out" in calls
    assert "eqnpt_pre.out" in calls
    assert "eqnpt00.out" in calls


def test_run_equil_direct_step_failure_leaves_failed_marker(
    tmp_path: Path,
) -> None:
    env, cmd = _setup_run_equil_only_eq(tmp_path)
    env["FAIL_OUT"] = "eqnpt_pre.out"
    for name in ["mini.rst7", "mini2.rst7", "eqnvt.rst7"]:
        _write_file(tmp_path / name, "rst\n")

    failed_run = subprocess.run(cmd, cwd=tmp_path, env=env, check=False)
    assert failed_run.returncode != 0
    assert (tmp_path / "ATTEMPT_FAILED").exists()


def test_run_equil_reruns_nvt_after_direct_failure_when_enabled(
    tmp_path: Path,
) -> None:
    env, cmd = _setup_run_equil_only_eq(tmp_path)
    for name in ["mini.rst7", "mini2.rst7"]:
        _write_file(tmp_path / name, "rst\n")
    _write_file(
        tmp_path / "check_penetration.py",
        "from pathlib import Path\n"
        "Path('RING_PENETRATION').write_text('ring\\n')\n",
    )

    failed_run = subprocess.run(cmd, cwd=tmp_path, env=env, check=False)
    assert failed_run.returncode != 0
    assert (tmp_path / "ATTEMPT_FAILED").exists()
    assert (tmp_path / "eqnvt.rst7").exists()

    (tmp_path / "call.log").unlink(missing_ok=True)
    _write_file(
        tmp_path / "check_penetration.py",
        "from pathlib import Path\n"
        "Path('RING_PENETRATION').unlink(missing_ok=True)\n",
    )
    env["RERUN_EQ_STEPS_AFTER_FAILURE"] = "1"

    rerun = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = _read_calls(tmp_path)
    assert "Skipping NVT preparation" not in rerun.stdout
    assert "rerunning NVT preparation despite existing artifact eqnvt.rst7" in rerun.stdout
    assert "eqnpt_pre.out" in calls
