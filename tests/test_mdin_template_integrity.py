from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FILE_ROOT = REPO_ROOT / "batter" / "_internal" / "templates" / "run_files_orig"
REMD_RUN_FILE_ROOT = (
    REPO_ROOT / "batter" / "_internal" / "templates" / "remd_run_files"
)
CHECK_RUN = RUN_FILE_ROOT / "check_run.bash"


def _stage_runner(tmp_path: Path, script_name: str, template_text: str) -> Path:
    runner = tmp_path / script_name
    runner.write_text((RUN_FILE_ROOT / script_name).read_text())
    (tmp_path / "check_run.bash").write_text(CHECK_RUN.read_text())
    (tmp_path / "mdin-template").write_text(template_text)
    return runner


@pytest.mark.parametrize(
    "script_name",
    [
        "run-local.bash",
        "run-local-rbfe.bash",
        "run-local-vacuum.bash",
        "run-equil.bash",
    ],
)
def test_production_runner_rejects_empty_mdin_template(
    tmp_path: Path, script_name: str
) -> None:
    runner = _stage_runner(tmp_path, script_name, "")

    result = subprocess.run(
        ["bash", runner.name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing or empty mdin template" in result.stdout + result.stderr
    assert not (tmp_path / "FINISHED").exists()


def test_production_runner_propagates_missing_total_steps(tmp_path: Path) -> None:
    runner = _stage_runner(
        tmp_path,
        "run-local.bash",
        "irest = 1,\nntx = 5,\nnstlim = 10,\ndt = 0.004,\n",
    )

    result = subprocess.run(
        ["bash", runner.name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "total_steps comment not found" in result.stderr
    assert "Failed to parse total_steps" in result.stdout
    assert not (tmp_path / "FINISHED").exists()


@pytest.mark.parametrize(
    "template_text, expected_error",
    [
        ("! total_steps=10\nnstlim = 10,\n", "Required positive dt not found"),
        (
            "! total_steps=10\nnstlim = 10,\ndt = not-a-number,\n",
            "Required positive dt not found",
        ),
        (
            "! target_dt=not-a-number\n! total_steps=10\nnstlim = 10,\ndt = 0.004,\n",
            "Malformed target_dt marker",
        ),
    ],
)
def test_production_runner_rejects_invalid_timestep_metadata(
    tmp_path: Path, template_text: str, expected_error: str
) -> None:
    runner = _stage_runner(tmp_path, "run-local.bash", template_text)

    result = subprocess.run(
        ["bash", runner.name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert expected_error in result.stdout + result.stderr
    assert not (tmp_path / "FINISHED").exists()


def test_target_dt_update_preserves_template_when_copy_fails(tmp_path: Path) -> None:
    template = tmp_path / "mdin-template"
    original = "! total_steps=10\nnstlim = 10,\ndt = 0.004,\n"
    template.write_text(original)

    command = (
        f"source {shlex.quote(str(CHECK_RUN))}; "
        "cat() { return 73; }; "
        "ensure_target_dt_marker mdin-template 0.004"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "original preserved" in result.stderr
    assert template.read_text() == original
    assert not list(tmp_path.glob("mdin-template.tmp*"))


def test_failure_bookkeeping_survives_dt_reduction_failure(tmp_path: Path) -> None:
    previous_restart = tmp_path / "md-00.rst7"
    previous_restart.write_text("previous restart\n")

    command = (
        "set -e; "
        f"source {shlex.quote(str(CHECK_RUN))}; "
        "SIM_COMMAND_STATUS=17; "
        "check_sim_failure 'MD segment 1' run.log md-01.rst7 md-00.rst7 3"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Could not reduce dt" in result.stderr
    assert (tmp_path / "ATTEMPT_FAILED").read_text() == "FAILED\n"
    assert not previous_restart.exists()


def test_batch_runner_rejects_existing_empty_batch_template(tmp_path: Path) -> None:
    component = tmp_path / "z"
    window = component / "z00"
    window.mkdir(parents=True)

    script_text = (REMD_RUN_FILE_ROOT / "run-local-batch.bash").read_text()
    script_text = script_text.replace("COMPONENT", "z").replace("NWINDOWS", "1")
    runner = component / "run-local-batch.bash"
    runner.write_text(script_text)
    (component / "check_run.bash").write_text(CHECK_RUN.read_text())
    (window / "mdin-batch-template").write_text("")
    (window / "mdin-template").write_text(
        "! total_steps=10\n&cntrl\nnstlim = 10,\ndt = 0.004,\n/\n"
    )

    result = subprocess.run(
        ["bash", runner.name],
        cwd=component,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Existing batch template is empty or invalid" in result.stderr
    assert not (component / "FINISHED").exists()


def test_batch_runner_allows_fallback_when_batch_template_is_absent(
    tmp_path: Path,
) -> None:
    component = tmp_path / "z"
    window = component / "z00"
    window.mkdir(parents=True)

    script_text = (REMD_RUN_FILE_ROOT / "run-local-batch.bash").read_text()
    script_text = script_text.replace("COMPONENT", "z").replace("NWINDOWS", "1")
    runner = component / "run-local-batch.bash"
    runner.write_text(script_text)
    (component / "check_run.bash").write_text(CHECK_RUN.read_text())
    (window / "mdin-template").write_text(
        "! total_steps=0\n&cntrl\nnstlim = 10,\ndt = 0.004,\n/\n"
    )
    (window / "eq.rst7").write_text("restart\n")

    result = subprocess.run(
        ["bash", runner.name],
        cwd=component,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (component / "FINISHED").exists()
    assert (window / "FINISHED").exists()


def test_remd_runner_rejects_empty_primary_template(tmp_path: Path) -> None:
    component = tmp_path / "z"
    window = component / "z00"
    window.mkdir(parents=True)

    script_text = (REMD_RUN_FILE_ROOT / "run-local-remd.bash").read_text()
    script_text = script_text.replace("COMPONENT", "z").replace("NWINDOWS", "1")
    runner = component / "run-local-remd.bash"
    runner.write_text(script_text)
    (component / "check_run.bash").write_text(CHECK_RUN.read_text())
    (window / "mdin-remd-template").write_text("")

    result = subprocess.run(
        ["bash", runner.name],
        cwd=component,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing or empty mdin-remd-template" in result.stdout + result.stderr
    assert not (component / "FINISHED").exists()


def test_remd_current_rewrite_propagates_install_failure(tmp_path: Path) -> None:
    template = tmp_path / "mdin-remd-template"
    current = tmp_path / "mdin-remd-current"
    template.write_text("nstlim = 10,\ndt = 0.002,\n")
    current.write_text("nstlim = 10,\ndt = 0.004,\n")

    command = (
        f"source {shlex.quote(str(CHECK_RUN))}; "
        "mv() { return 73; }; "
        "sync_current_mdin_from_template mdin-remd-template mdin-remd-current"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "dt = 0.004" in current.read_text()
