from __future__ import annotations

import os
import shlex
import signal
import subprocess as sp
from typing import Mapping, Sequence

from loguru import logger

from importlib import resources

__all__ = [
    "run_with_log",
    "antechamber",
    "tleap",
    "cpptraj",
    "parmchk2",
    "charmmlipid2amber",
    "usalign",
    "obabel",
    "vmd",
]

def _exec_from_env(key: str, default: str) -> str:
    """Resolve an executable name/path from the environment."""
    return os.environ.get(key, default)


# External executables used throughout BATTER. These derive from environment
# variables so overrides propagate into subprocesses.
antechamber = _exec_from_env("BATTER_ANTECHAMBER", "antechamber")
tleap = _exec_from_env("BATTER_TLEAP", "tleap")
cpptraj = _exec_from_env("BATTER_CPPTRAJ", "cpptraj")
parmchk2 = _exec_from_env("BATTER_PARMCHK2", "parmchk2")
charmmlipid2amber = _exec_from_env(
    "BATTER_CHARMM_LIPID2AMBER", "charmmlipid2amber.py"
)
usalign = _exec_from_env(
    "BATTER_USALIGN", str(resources.files("batter") / "utils" / "USalign")
)
obabel = _exec_from_env("BATTER_OBABEL", "obabel")
vmd = _exec_from_env("BATTER_VMD", "vmd")


def run_with_log(
    command: str | Sequence[str],
    level: str = "debug",
    working_dir: str | os.PathLike[str] | None = None,
    *,
    error_match: str | None = None,
    timeout: float | None = None,
    shell: bool = True,
    env: Mapping[str, str] | None = None,
) -> sp.CompletedProcess[str]:
    """
    Run a subprocess command and stream stdout/stderr through ``loguru``.

    Raises
    ------
    RuntimeError
        On non-zero return codes, signal-based exits, matched ``error_match``,
        or timeout.
    """
    if working_dir is None:
        working_dir = os.getcwd()

    log_methods = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical,
    }
    log = log_methods.get(level)
    if log is None:
        raise ValueError(f"Invalid log level: {level}")

    if isinstance(command, str) and not shell:
        cmd: str | Sequence[str] = shlex.split(command)
    else:
        cmd = command

    logger.debug(f"Running command: {command!r}")
    logger.debug(f"Working directory: {working_dir}")

    try:
        result = sp.run(
            cmd,
            shell=shell,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
            check=False,
            cwd=working_dir,
            timeout=timeout,
            env=(None if env is None else {**os.environ, **env}),
        )
    except sp.TimeoutExpired as e:  # pragma: no cover - exercised in integration
        logger.info(f"Command timed out after {timeout}s: {command!r}")
        if e.output:
            log("Command output before timeout:")
            for line in e.output.splitlines():
                log(line)
        if e.stderr:
            log("Command error output before timeout:")
            for line in e.stderr.splitlines():
                log(line)
        raise RuntimeError(f"Command timed out after {timeout}s: {command!r}") from e

    if result.stdout:
        log("Command output:")
        for line in result.stdout.splitlines():
            log(line)
    if result.stderr:
        log("Command errors:")
        for line in result.stderr.splitlines():
            log(line)

    if error_match and (error_match in result.stdout or error_match in result.stderr):
        raise RuntimeError(
            f"Command {command!r} reported an error matching {error_match!r}."
        )

    rc = result.returncode
    if rc == 0:
        return result

    if rc < 0:
        sig = -rc
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:  # pragma: no cover
            sig_name = f"SIG{sig}"
        raise RuntimeError(
            f"Command {command!r} died with signal {sig_name} ({sig})."
        )

    raise RuntimeError(f"Command {command!r} failed with return code {rc}.")
