from __future__ import annotations

import os
from pathlib import Path

import pytest

from batter import api as api_mod


def test_resolve_execution_run_defaults_to_latest(tmp_path: Path) -> None:
    executions = tmp_path / "executions"
    old = executions / "old-run"
    new = executions / "new-run"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    run_id, run_dir = api_mod._resolve_execution_run(tmp_path, None)
    assert run_id == "new-run"
    assert run_dir == new


def test_resolve_execution_run_raises_when_no_runs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No executions found"):
        api_mod._resolve_execution_run(tmp_path, None)
