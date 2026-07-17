from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def bundled_parmed_path() -> Path:
    """Return the bundled ParmEd source checkout path."""
    return Path(__file__).resolve().parents[2] / "extern" / "ParmEd"


def import_parmed() -> ModuleType:
    """Import ParmEd, falling back to the bundled source checkout."""
    try:
        return importlib.import_module("parmed")
    except ModuleNotFoundError as exc:
        if exc.name != "parmed":
            raise

    bundled = bundled_parmed_path()
    if not bundled.exists():
        raise

    bundled_path = str(bundled)
    if bundled_path not in sys.path:
        sys.path.insert(0, bundled_path)
    return importlib.import_module("parmed")
