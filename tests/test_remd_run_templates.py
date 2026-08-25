from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _prepare_component(
    tmp_path: Path,
    *,
    script_name: str,
    template_name: str,
    total_steps: int,
    dt: float = 0.002,
    nstlim: int = 10,
) -> tuple[Path, Path, Path]:
    repo_root = _repo_root()
    script_tpl = (
        repo_root / "batter" / "_internal" / "templates" / "remd_run_files" / script_name
    )
    check_run = (
        repo_root
        / "batter"
        / "_internal"
        / "templates"
        / "run_files_orig"
        / "check_run.bash"
    )

    comp_dir = tmp_path / "z"
    win0 = comp_dir / "z00"
    win0.mkdir(parents=True, exist_ok=True)

    script_text = script_tpl.read_text().replace("COMPONENT", "z").replace("NWINDOWS", "1")
    (comp_dir / script_name).write_text(script_text)
    (comp_dir / "check_run.bash").write_text(check_run.read_text())
    (comp_dir / script_name).chmod(0o755)
    (comp_dir / "check_run.bash").chmod(0o755)

    if script_name == "run-local-remd.bash":
        (comp_dir / "remd").mkdir(parents=True, exist_ok=True)

    (comp_dir / "run.log").write_text("old log\n")
    tmpl = win0 / template_name
    tmpl.write_text(
        f"! total_steps={total_steps}\n"
        "&cntrl\n"
        "  ntx = 5,\n"
        "  irest = 1,\n"
        f"  nstlim = {nstlim},\n"
        f"  dt = {dt:.3f},\n"
        "  ntwr = 10,\n"
        "/\n"
        " &wt type='DUMPFREQ', istep1=10, /\n"
        " &wt type='END', /\n"
        "DUMPAVE=cmass.txt\n"
        "LISTIN=POUT\n"
        "LISTOUT=POUT\n"
    )
    (win0 / "eq.rst7").write_text("rst\n")
    return comp_dir, win0, tmpl


def _extract_dt(template_path: Path) -> float:
    match = re.search(
        r"^\s*dt\s*=\s*([-+0-9.eEdD]+)",
        template_path.read_text(),
        flags=re.MULTILINE,
    )
    assert match is not None, template_path.read_text()
    return float(match.group(1).replace("D", "e").replace("d", "e"))


def _restart_text(time_ps: float) -> str:
    return (
        "Stub Amber restart\n"
        f"1  {time_ps:.7E}\n"
        "  0.0  0.0  0.0\n"
    )


def _write_success_pmemd_stub(path: Path) -> None:
    _write_exe(
        path,
        "#!/usr/bin/env bash\n"
        "groupfile=\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ $1 == -groupfile ]]; then groupfile=$2; shift 2; else shift; fi\n"
        "done\n"
        "[[ -f $groupfile ]] || exit 2\n"
        "while IFS= read -r line; do\n"
        "  set -- $line\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    case $1 in\n"
        "      -o|-x|-l|-e|-r) mkdir -p \"$(dirname \"$2\")\"; echo ok > \"$2\"; shift 2 ;;\n"
        "      *) shift ;;\n"
        "    esac\n"
        "  done\n"
        "done < \"$groupfile\"\n"
        "exit 0\n",
    )


def _remove_production_is_complete(check_run: Path) -> None:
    text = check_run.read_text()
    start = text.index("\nproduction_is_complete() {")
    end = text.index("\nmdin_set_cntrl_value()", start)
    check_run.write_text(text[:start] + text[end:])


@pytest.mark.parametrize("script_name", ["run-local-remd.bash", "run-local-batch.bash"])
def test_remd_run_templates_use_merged_prmtop(script_name: str) -> None:
    template = (
        _repo_root()
        / "batter"
        / "_internal"
        / "templates"
        / "remd_run_files"
        / script_name
    )
    text = template.read_text()

    assert 'PRMTOP="full_merged.prmtop"' in text
    assert "full.hmr.prmtop" not in text


@pytest.mark.parametrize(
    ("script_name", "template_name"),
    [
        ("run-local-remd.bash", "mdin-remd-template"),
        ("run-local-batch.bash", "mdin-batch-template"),
    ],
)
def test_remd_run_templates_zero_step_finish(
    tmp_path: Path, script_name: str, template_name: str
) -> None:
    comp_dir, win0, _ = _prepare_component(
        tmp_path,
        script_name=script_name,
        template_name=template_name,
        total_steps=0,
    )

    result = subprocess.run(
        ["bash", f"./{script_name}"],
        cwd=comp_dir,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    archived_logs = list((comp_dir / "ARCHIVED_LOGS").glob("*_run.log"))
    assert len(archived_logs) == 1
    assert archived_logs[0].read_text() == "old log\n"
    assert (comp_dir / "FINISHED").exists()
    assert (win0 / "FINISHED").exists()


@pytest.mark.parametrize(
    ("script_name", "template_name"),
    [
        ("run-local-remd.bash", "mdin-remd-template"),
        ("run-local-batch.bash", "mdin-batch-template"),
    ],
)
def test_remd_run_templates_finish_with_old_check_run_without_completion_helper(
    tmp_path: Path, script_name: str, template_name: str
) -> None:
    comp_dir, win0, _ = _prepare_component(
        tmp_path,
        script_name=script_name,
        template_name=template_name,
        total_steps=0,
    )
    _remove_production_is_complete(comp_dir / "check_run.bash")

    result = subprocess.run(
        ["bash", f"./{script_name}"],
        cwd=comp_dir,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "production_is_complete: command not found" not in result.stderr
    assert (comp_dir / "FINISHED").exists()
    assert (win0 / "FINISHED").exists()


@pytest.mark.parametrize(
    ("script_name", "template_name"),
    [
        ("run-local-remd.bash", "mdin-remd-template"),
        ("run-local-batch.bash", "mdin-batch-template"),
    ],
)
def test_remd_run_templates_reduce_dt_after_retry_failure(
    tmp_path: Path, script_name: str, template_name: str
) -> None:
    comp_dir, win0, tmpl = _prepare_component(
        tmp_path,
        script_name=script_name,
        template_name=template_name,
        total_steps=20,
        dt=0.004,
        nstlim=10,
    )
    (win0 / "cmass-01.txt").write_text("stale\n")

    fail_stub = tmp_path / "pmemd-fail.sh"
    _write_exe(
        fail_stub,
        "#!/usr/bin/env bash\n"
        "exit 1\n",
    )

    env = os.environ.copy()
    env["PMEMD_MPI_EXEC"] = str(fail_stub)
    env["MPI_EXEC"] = "/bin/bash"
    env["MPI_FLAGS"] = " "
    env["RETRY_COUNT"] = "3"

    result = subprocess.run(
        ["bash", f"./{script_name}"],
        cwd=comp_dir,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert _extract_dt(tmpl) == pytest.approx(0.002)
    assert not (win0 / "cmass-01.txt").exists()


@pytest.mark.parametrize(
    ("script_name", "template_name", "current_name"),
    [
        ("run-local-remd.bash", "mdin-remd-template", "mdin-remd-current"),
        ("run-local-batch.bash", "mdin-batch-template", "mdin-current"),
    ],
)
def test_remd_run_templates_write_segmented_cmass_dumpave(
    tmp_path: Path,
    script_name: str,
    template_name: str,
    current_name: str,
) -> None:
    comp_dir, win0, _ = _prepare_component(
        tmp_path,
        script_name=script_name,
        template_name=template_name,
        total_steps=10,
        nstlim=10,
    )

    pmemd_stub = tmp_path / "pmemd-success.sh"
    _write_success_pmemd_stub(pmemd_stub)

    env = os.environ.copy()
    env["PMEMD_MPI_EXEC"] = str(pmemd_stub)
    env["MPI_EXEC"] = "/bin/bash"
    env["MPI_FLAGS"] = " "

    result = subprocess.run(
        ["bash", f"./{script_name}"],
        cwd=comp_dir,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = (win0 / current_name).read_text()
    assert "irest = 1," in text
    assert re.search(r"^\s*ntx\s*=\s*5,", text, flags=re.MULTILINE)
    assert "DUMPAVE=z00/cmass-01.txt" in text
    assert "DUMPAVE=cmass.txt" not in text


def test_run_local_remd_preserves_template_nstlim_for_short_tail(
    tmp_path: Path,
) -> None:
    comp_dir, win0, _ = _prepare_component(
        tmp_path,
        script_name="run-local-remd.bash",
        template_name="mdin-remd-template",
        total_steps=15,
        nstlim=10,
    )
    (win0 / "md-current.rst7").write_text(_restart_text(0.020))

    pmemd_stub = tmp_path / "pmemd-success.sh"
    _write_success_pmemd_stub(pmemd_stub)

    env = os.environ.copy()
    env["PMEMD_MPI_EXEC"] = str(pmemd_stub)
    env["MPI_EXEC"] = "/bin/bash"
    env["MPI_FLAGS"] = " "

    result = subprocess.run(
        ["bash", "./run-local-remd.bash"],
        cwd=comp_dir,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = (win0 / "mdin-remd-current").read_text()
    assert "nstlim = 10," in text
    assert "nstlim = 5," not in text
    assert "numexchg = 1," in text


def test_run_local_remd_caps_dumpfreq_to_exchange_block(tmp_path: Path) -> None:
    comp_dir, win0, tmpl = _prepare_component(
        tmp_path,
        script_name="run-local-remd.bash",
        template_name="mdin-remd-template",
        total_steps=200,
        nstlim=200,
    )
    tmpl.write_text(tmpl.read_text().replace("istep1=10", "istep1=25000"))

    pmemd_stub = tmp_path / "pmemd-success.sh"
    _write_success_pmemd_stub(pmemd_stub)

    env = os.environ.copy()
    env["PMEMD_MPI_EXEC"] = str(pmemd_stub)
    env["MPI_EXEC"] = "/bin/bash"
    env["MPI_FLAGS"] = " "

    result = subprocess.run(
        ["bash", "./run-local-remd.bash"],
        cwd=comp_dir,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = (win0 / "mdin-remd-current").read_text()
    assert "type='DUMPFREQ', istep1=200" in text
