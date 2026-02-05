"""Shared CLI helpers."""

from __future__ import annotations

import re
import shlex
import sys


def _upsert_sbatch_option(text: str, flag: str, value: str) -> str:
    """
    Replace or insert a ``#SBATCH --<flag>=...`` line with ``value``.
    """
    pattern = re.compile(rf"^#SBATCH\s+--{re.escape(flag)}=.*$", re.MULTILINE)
    repl = f"#SBATCH --{flag}={value}"
    if pattern.search(text):
        return pattern.sub(repl, text, count=1)

    lines = text.splitlines()
    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    lines.insert(insert_idx, repl)
    return "\n".join(lines)


def _which_batter() -> str:
    """
    Resolve the executable used to invoke ``batter``.

    Returns
    -------
    str
        Shell-escaped token (``batter`` path or ``python -m batter.cli``).
    """
    import shutil

    exe = shutil.which("batter")
    if exe:
        return shlex.quote(exe)
    # last resort: run module (works inside editable installs)
    return shlex.quote(sys.executable) + " -m batter.cli"
