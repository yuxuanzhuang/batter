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
    comp_dir, _win0, tmpl = _prepare_component(
        tmp_path,
        script_name=script_name,
        template_name=template_name,
        total_steps=20,
        dt=0.004,
        nstlim=10,
    )

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
    assert _extract_dt(tmpl) == pytest.approx(0.003)
