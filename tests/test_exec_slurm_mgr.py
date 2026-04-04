from pathlib import Path
import subprocess

import pytest

from batter.exec.slurm_mgr import (
    SlurmJobManager,
    SlurmJobSpec,
    _atomic_append_jsonl_unique,
)


def test_atomic_append_jsonl_unique(tmp_path):
    path = tmp_path / "registry.jsonl"
    rec = {"workdir": "/tmp/job1"}

    _atomic_append_jsonl_unique(path, rec)
    _atomic_append_jsonl_unique(path, rec)  # duplicate should no-op

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_slurm_job_manager_status(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    script.write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(registry_file=tmp_path / "queue.jsonl")
    manager.add(spec)

    # registry file should have exactly one entry
    manager.add(spec)
    lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_registry_filters_by_stage(tmp_path):
    wd_eq = tmp_path / "eq_job"
    wd_fe = tmp_path / "fe_job"
    wd_eq.mkdir()
    wd_fe.mkdir()

    manager = SlurmJobManager(registry_file=tmp_path / "queue.jsonl")
    manager.set_stage("equil")

    manager.add(SlurmJobSpec(workdir=wd_eq, stage="equil"))
    manager.add(SlurmJobSpec(workdir=wd_fe, stage="fe"))

    jobs_equil = manager.jobs()
    assert {j.workdir for j in jobs_equil} == {wd_eq}

    manager.set_stage("fe")
    jobs_fe = manager.jobs()
    assert {j.workdir for j in jobs_fe} == {wd_fe}



def test_submission_failure_raises(monkeypatch, tmp_path):
    workdir = tmp_path / "fail"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    script.write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=0,
        submit_retry_limit=3,
        submit_retry_delay_s=0.0,
    )

    attempts = {"count": 0}

    def fail_submit_once(spec: SlurmJobSpec) -> str:
        attempts["count"] += 1
        raise RuntimeError("QOS limit")

    monkeypatch.setattr(manager, "_submit_once", fail_submit_once)

    with pytest.raises(RuntimeError, match="after 4 attempt"):
        manager._wait_loop([spec])

    assert attempts["count"] == 4


def test_submit_waits_for_slot_before_each_submission(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, max_active_jobs=1)

    wait_calls = {"count": 0}

    def fake_wait_for_slot(*args, **kwargs):
        wait_calls["count"] += 1

    monkeypatch.setattr(manager, "wait_for_slot", fake_wait_for_slot)
    monkeypatch.setattr(manager, "_submit_once", lambda spec: "42")

    assert manager._submit(spec) == "42"
    assert manager._submit(spec) == "42"
    assert wait_calls["count"] == 2


def test_submission_rate_limit_uses_longer_backoff(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        submit_retry_limit=2,
        submit_retry_delay_s=60.0,
    )

    sleeps: list[float] = []

    monkeypatch.setattr(manager, "wait_for_slot", lambda *args, **kwargs: None)
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))

    def fail_submit_once(spec: SlurmJobSpec) -> str:
        raise RuntimeError("sbatch: error: Reached jobs per hour limit")

    monkeypatch.setattr(manager, "_submit_once", fail_submit_once)

    with pytest.raises(RuntimeError, match="after 3 attempt"):
        manager._submit(spec)

    assert sleeps == [900.0, 1800.0]


def test_submit_rebuilds_script_with_header(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#SBATCH -J old\nBODY\n")
    header_root = tmp_path / "headers"
    header_root.mkdir()
    (header_root / "SLURMM-Am.header").write_text("#HEADER\n")

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        header_name="SLURMM-Am.header",
        header_root=header_root,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, header_root=header_root)

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        class Dummy:
            returncode = 0
            stdout = "Submitted batch job 99"
            stderr = ""

        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    jobid = manager._submit_once(spec)
    assert jobid == "99"
    script_txt = (workdir / "SLURMM-run").read_text()
    assert script_txt.startswith("#HEADER")
    assert "BODY" in script_txt
    assert "SBATCH -J old" not in script_txt


def test_submit_rebuild_does_not_duplicate_header_on_repeat(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n#SBATCH -J old\nBODY\n")
    header_root = tmp_path / "headers"
    header_root.mkdir()
    header_text = (
        "#!/bin/bash\n"
        "# SYSTEMNAME, STAGE, POSE are placeholders to be replaced when generating the script\n"
        "source /path/to/amber.sh\n"
    )
    (header_root / "SLURMM-Am.header").write_text(header_text)

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        header_name="SLURMM-Am.header",
        header_root=header_root,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, header_root=header_root)

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        class Dummy:
            returncode = 0
            stdout = "Submitted batch job 99"
            stderr = ""

        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    manager._submit_once(spec)
    manager._submit_once(spec)

    script_txt = (workdir / "SLURMM-run").read_text()
    assert script_txt.count("source /path/to/amber.sh") == 1
    assert script_txt.count("SYSTEMNAME, STAGE, POSE") == 1
    assert "BODY" in script_txt


def test_submit_uses_submit_dir(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    submit_dir = tmp_path / "batch"
    submit_dir.mkdir()
    script = submit_dir / "batch.sh"
    script.write_text("#!/bin/bash\necho hi\n")

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel=script.name,
        batch_script=script,
        submit_dir=submit_dir,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0)

    calls = {}

    class Dummy:
        returncode = 0
        stdout = "Submitted batch job 42"
        stderr = ""

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    jobid = manager._submit_once(spec)
    assert jobid == "42"
    assert calls["cwd"] == submit_dir
    assert script.name in calls["cmd"]
