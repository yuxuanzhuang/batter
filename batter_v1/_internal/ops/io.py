from __future__ import annotations

from pathlib import Path
import shutil


def reset_dir(path: Path) -> None:
    """
    Delete `path` if it exists and recreate it empty.
    """
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)