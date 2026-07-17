"""Builder implementations.

Avoid importing concrete builders at package import time; some builders pull in
heavy scientific modules that are only needed for specific pipeline stages.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PrepareEquilBuilder"]


def __getattr__(name: str) -> Any:
    if name == "PrepareEquilBuilder":
        from .equil import PrepareEquilBuilder

        return PrepareEquilBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
