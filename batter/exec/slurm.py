from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from batter.pipeline.step import Step, ExecResult
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
    return f"#SBATCH --{flag}={value}\n" if value not in (None, "", 0) else ""


class SlurmBackend(ExecBackend):
    """
    Slurm backend that materializes simple sbatch scripts and submits them.

    Behavior
    --------
    - For each step, a script is written to ``<system.root>/sbatch/<step>.sh``.
    - Logs go under ``<system.root>/logs/``.
    - Submission uses, in order of preference:
        1) ``utils.slurm_job.submit_job(script_path)`` if available.
        2) ``sbatch <script>`` subprocess.
    - The payload is determined by ``params.get("payload")`` (string shell code).
      If unset, a minimal placeholder is used.

    Resources
    ---------
    - Read from ``params.get("resources", {})`` with keys matching :class:`Resources`:
      ``time``, ``cpus``, ``gpus``, ``mem``, ``partition``, ``account``, and
      optional ``extra`` mapping (converted to additional SBATCH lines).

    Environment
    -----------
    - Optional ``params.get("env", {})`` key (dict) exports ``KEY=VALUE`` before payload.

    Returns
    -------
    ExecResult
        With Slurm job id (if submission succeeded) and basic artifacts.
    """
    name: str = "slurm"

    def run(self, step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
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
        if not env:
            return ":\n"
        lines = [f'export {k}={shlex.quote(str(v))}' for k, v in env.items()]
        return "\n".join(lines)

    @staticmethod
    def _payload(step: Step, system: SimSystem, params: Dict[str, Any]) -> str:
        # if user provided a payload, use it
        payload = params.get("payload")
        if isinstance(payload, str) and payload.strip():
            return payload

        # default placeholder: print what would have been run
        return f'echo "[batter] no payload for step {step.name}; system root: {system.root}"'

    @staticmethod
    def _submit(script_path: Path) -> Optional[str]:
        """
        Submit a script via utils.slurm_job if present, else sbatch.

        Returns
        -------
        str or None
            Slurm job ID if it can be parsed; otherwise ``None``.
        """
        # 1) try package helper if available
        try:
            from batter.utils.slurm_job import submit_job  # type: ignore
            job_id = submit_job(script_path)  # expected to return string/int id
            return str(job_id)
        except Exception:
            pass

        # 2) fallback to sbatch
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