from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple


def select_run_id(sys_root: Path | str, protocol: str, system_name: str, requested: str | None) -> Tuple[str, Path]:
    """
    Choose the run identifier for the current execution and ensure its directory exists.

    Returns
    -------
    (run_id, run_dir)
    """
    runs_dir = Path(sys_root) / "executions"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if requested and requested != "auto":
        run_dir = runs_dir / requested
        run_dir.mkdir(parents=True, exist_ok=True)
        return requested, run_dir

    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.name, latest

    rid = generate_run_id(protocol, system_name)
    run_dir = runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return rid, run_dir


def generate_run_id(protocol: str, system_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{protocol}-{system_name}-{ts}"
