from pathlib import Path

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

    # no sentinels yet -> pending
    done, status = manager._status(spec)
    assert done is False
    assert status is None

    # finished sentinel
    spec.finished_path().write_text("")
    done, status = manager._status(spec)
    assert done is True
    assert status == "FINISHED"

    # registry file should have exactly one entry
    manager.add(spec)
    lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_timeout_resubmits_without_failure(monkeypatch, tmp_path):
    workdir = tmp_path / "timeout"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    script.write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=0,
    )

    submissions = {"count": 0}

    def fake_submit(spec: SlurmJobSpec) -> str:
        submissions["count"] += 1
        spec.jobid_path().write_text(str(submissions["count"]))
        return str(submissions["count"])

    class StopLoop(Exception):
        pass

    states = iter(["TIMEOUT", "TIMEOUT", "STOP"])

    def fake_slurm_state(jobid: str | None):
        if not jobid:
            return None
        try:
            state = next(states)
        except StopIteration:
            return None
        if state == "STOP":
            raise StopLoop()
        return state

    monkeypatch.setattr("batter.exec.slurm_mgr._slurm_state", fake_slurm_state)
    monkeypatch.setattr(manager, "_submit", fake_submit)

    with pytest.raises(StopLoop):
        manager._wait_loop([spec])

    assert not spec.failed_path().exists()
    assert manager._retries.get(spec.workdir, 0) == 0
    assert submissions["count"] == 2


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


def test_completed_resubmit_does_not_count_retry(monkeypatch, tmp_path):
    workdir = tmp_path / "completed"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    script.write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=0,  # would fail immediately if counted
    )

    submissions = {"count": 0}

    def fake_submit(spec: SlurmJobSpec) -> str:
        submissions["count"] += 1
        spec.jobid_path().write_text(str(submissions["count"]))
        # simulate the resubmitted job writing FINISHED
        if submissions["count"] >= 2:
            spec.finished_path().write_text("")
        return str(submissions["count"])

    def fake_slurm_state(jobid: str | None):
        if not jobid:
            return None
        return "COMPLETED"

    monkeypatch.setattr("batter.exec.slurm_mgr._slurm_state", fake_slurm_state)
    monkeypatch.setattr(manager, "_submit", fake_submit)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    manager._wait_loop([spec])

    assert submissions["count"] == 2  # initial + resubmission
    assert manager._retries.get(spec.workdir, 0) == 0
    assert spec.failed_path().exists() is False
