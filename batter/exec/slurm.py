"""Execution backend that submits steps to Slurm via ``sbatch``."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem

from .base import ExecBackend, Resources

__all__ = ["SlurmBackend"]


_SBATCH_TPL = """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.%j.out
#SBATCH --error={log_dir}/{job_name}.%j.err
{partition}{account}{time}{cpus}{gpus}{mem}{extras}

set -euo pipefail

echo "[batter] node: $(hostname)"
echo "[batter] start: $(date -Is)"

# user environment (optional)
{env_block}

# step payload
{payload}

echo "[batter] done: $(date -Is)"
"""


def _fmt(flag: str, value: Optional[str | int]) -> str:
    """Render a ``#SBATCH`` line when ``value`` is provided."""
    return f"#SBATCH --{flag}={value}\n" if value not in (None, "", 0) else ""


class SlurmBackend(ExecBackend):
    """Slurm backend that materializes lightweight job scripts."""

    name: str = "slurm"

    def run(self, step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        """Submit ``step`` to Slurm.

        Parameters
        ----------
        step : Step
            Pipeline step metadata.
        system : SimSystem
            Simulation system whose ``root`` directory stores scripts and logs.
        params : dict
            Backend-specific options. Recognised keys include ``resources``,
            ``env`` (exported variables), and ``payload`` (shell snippet).

        Returns
        -------
        ExecResult
            Artifacts referencing the generated script and log paths together
            with the submitted job identifier (if available).
        """
        root = system.root
        sbatch_dir = root / "sbatch"
        log_dir = root / "logs"
        sbatch_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # resources
        rdict = params.get("resources", {}) or {}
        res = Resources(
            time=rdict.get("time"),
            cpus=rdict.get("cpus"),
            gpus=rdict.get("gpus"),
            mem=rdict.get("mem"),
            partition=rdict.get("partition"),
            account=rdict.get("account"),
            extra=rdict.get("extra", {}),
        )

        sbatch_text = _SBATCH_TPL.format(
            job_name=step.name,
            log_dir=log_dir.as_posix(),
            partition=_fmt("partition", res.partition),
            account=_fmt("account", res.account),
            time=_fmt("time", res.time),
            cpus=_fmt("cpus-per-task", res.cpus),
            gpus=_fmt("gpus", res.gpus),
            mem=_fmt("mem", res.mem),
            extras="".join(_fmt(k, v) for k, v in (res.extra or {}).items()),
            env_block=self._format_env(params.get("env", {})),
            payload=self._payload(step, system, params),
        )

        script_path = sbatch_dir / f"{step.name}.sh"
        script_path.write_text(sbatch_text)
        script_path.chmod(0o755)

        job_id = self._submit(script_path)
        logger.info("SLURM: submitted step {!r} as job {}", step.name, job_id or "<unknown>")

        return ExecResult(
            job_ids=[job_id] if job_id else [],
            artifacts={
                "script": script_path,
                "stdout": log_dir / f"{step.name}.{job_id}.out" if job_id else None,
                "stderr": log_dir / f"{step.name}.{job_id}.err" if job_id else None,
            },
        )

    # --------------------- helpers ---------------------

    @staticmethod
    def _format_env(env: Dict[str, str]) -> str:
        """Create export statements for user-provided environment variables."""
        if not env:
            return ":\n"
        lines = [f'export {k}={shlex.quote(str(v))}' for k, v in env.items()]
        return "\n".join(lines)

    @staticmethod
    def _payload(step: Step, system: SimSystem, params: Dict[str, Any]) -> str:
        """Return the shell snippet that drives the step."""
        payload = params.get("payload")
        if isinstance(payload, str) and payload.strip():
            return payload

        return f'echo "[batter] no payload for step {step.name}; system root: {system.root}"'

    @staticmethod
    def _submit(script_path: Path) -> Optional[str]:
        """Submit a script via ``utils.slurm_job`` or ``sbatch``.

        Parameters
        ----------
        script_path : pathlib.Path
            Script to submit.

        Returns
        -------
        str or None
            Parsed Slurm job identifier when submission succeeds.
        """
        try:
            from batter.utils.slurm_job import submit_job  # type: ignore
            job_id = submit_job(script_path)  # expected to return string/int id
            return str(job_id)
        except Exception:
            pass

        try:
            res = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
            # typical output: "Submitted batch job 123456"
            out = res.stdout.strip()
            for tok in out.split():
                if tok.isdigit():
                    return tok
            return None
        except Exception as e:
            logger.error("Failed to submit sbatch: {}", e)
            return None
