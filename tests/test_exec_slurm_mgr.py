from pathlib import Path

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
