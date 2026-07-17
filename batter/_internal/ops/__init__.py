"""Internal operation modules.

Keep this package initializer lightweight.  Several operation modules import
large scientific stacks, and eager imports here make unrelated CLI paths pay
that cost before they know which stage is being resumed.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "amber",
    "batch",
    "box",
    "build_complex",
    "helpers",
    "io",
    "remd",
    "restraints",
    "runfiles",
    "sim_files",
    "simprep",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
