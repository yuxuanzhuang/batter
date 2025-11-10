from __future__ import annotations

import re
from typing import List

__all__ = ["atoi", "natural_keys"]

_NAT_SPLIT = re.compile(r"(\d+)")


def atoi(text: str) -> int | str:
    """Parse digit substrings as integers for natural sorting."""
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[int | str]:
    """
    Split a string into a sequence of ints/strings suitable for natural sorts.

    Examples
    --------
    >>> sorted(["win2", "win10", "win1"], key=natural_keys)
    ['win1', 'win2', 'win10']
    """
    return [atoi(c) for c in _NAT_SPLIT.split(text)]
