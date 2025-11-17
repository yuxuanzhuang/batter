from __future__ import annotations

from typing import List, Set

import pytest

from batter.exec.slurm_mgr import SlurmJobManager


def test_wait_for_slot_no_wait(monkeypatch):
    mgr = SlurmJobManager(poll_s=0.1, dry_run=True)
    sleep_called = []

    monkeypatch.setattr(
        "batter.exec.slurm_mgr._active_job_ids", lambda user=None: {"1", "2"}
    )
    mgr.jobs = lambda: ["a"]
    monkeypatch.setattr("time.sleep", lambda s: sleep_called.append(s))

    # cap enough to avoid waiting
    mgr.wait_for_slot(max_active=10)
    assert sleep_called == []


def test_wait_for_slot_enforces_cap(monkeypatch):
    mgr = SlurmJobManager(poll_s=0.1, dry_run=True)
    sleep_called: List[float] = []

    call_count = {"n": 0}
    def fake_active(user=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"1", "2"}
        mgr._submitted_job_ids.clear()
        return set()

    monkeypatch.setattr("batter.exec.slurm_mgr._active_job_ids", fake_active)
    mgr._submitted_job_ids.update({"j1", "j2", "j3"})
    mgr.jobs = lambda: []
    monkeypatch.setattr("time.sleep", lambda s: sleep_called.append(s))

    mgr.wait_for_slot(max_active=3)
    assert sleep_called == [0.1]
