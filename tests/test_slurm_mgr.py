from __future__ import annotations

from typing import List

import pytest

from batter.exec.slurm_mgr import SlurmJobManager


def test_wait_for_slot_no_wait(monkeypatch):
    mgr = SlurmJobManager(poll_s=0.1, dry_run=True, max_active_jobs=10)
    sleep_called = []

    monkeypatch.setattr(
        "batter.exec.slurm_mgr._num_active_job", lambda user=None, partition=None: 2
    )
    mgr.jobs = lambda: ["a"]
    monkeypatch.setattr("time.sleep", lambda s: sleep_called.append(s))

    # cap enough to avoid waiting
    mgr.wait_for_slot()
    assert sleep_called == []


def test_wait_for_slot_enforces_cap(monkeypatch):
    mgr = SlurmJobManager(poll_s=0.1, dry_run=True, max_active_jobs=3)
    sleep_called: List[float] = []

    call_count = {"n": 0}
    def fake_active(user=None, partition=None):
        call_count["n"] += 1
        return 3 if call_count["n"] == 1 else 0

    monkeypatch.setattr("batter.exec.slurm_mgr._num_active_job", fake_active)
    mgr.jobs = lambda: []
    monkeypatch.setattr("time.sleep", lambda s: sleep_called.append(s))

    mgr.wait_for_slot()
    assert sleep_called == [0.1]
