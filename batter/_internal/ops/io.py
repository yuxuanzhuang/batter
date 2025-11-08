"""Utility I/O helpers for builder workflows."""

from __future__ import annotations

from pathlib import Path
import shutil


def reset_dir(path: Path) -> None:
    """Remove and recreate a directory to ensure a clean workspace.

    Parameters
    ----------
    path : Path
        Directory to delete and re-create.
    """
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
