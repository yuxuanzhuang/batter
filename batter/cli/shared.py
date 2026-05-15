"""Shared CLI helpers."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path


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


def _batter_path_export_block() -> str:
    """
    Return shell code that exposes the submit-time BATTER bin directory.

    SLURM batch shells often do not inherit the interactive conda/module PATH.
    The generated manager script may call an absolute ``batter`` executable, but
    BATTER subprocesses still launch tools like ``tleap`` by name. Prepending
    the submit-time executable directory lets those sibling tools resolve.
    """
    import shutil

    exe = shutil.which("batter") or sys.executable
    env_bin = str(Path(exe).parent)
    return (
        "# BATTER environment captured at submit time\n"
        f"BATTER_ENV_BIN={shlex.quote(env_bin)}\n"
        'export PATH="$BATTER_ENV_BIN:$PATH"\n'
    )
