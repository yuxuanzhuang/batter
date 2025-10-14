# batter/exec/handlers/equil.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.exec.slurm_mgr import SlurmJobManager, SlurmJobSpec


def _phase_paths(root: Path) -> dict[str, Path]:
    p = root / "equil"
    return {
        "phase_dir": p,
        "script": p / "SLURMM-run",
        "finished": p / "FINISHED",
        "failed": p / "FAILED",
        "jobid": p / "JOBID",
        "stdout": p / "slurm.out",
        "stderr": p / "slurm.err",
        "rst7": root / "artifacts" / "equil" / "equil.rst7",
    }


def _read_partition(params: Dict[str, Any]) -> str:
    sim = params.get("sim", {}) or {}
    part = sim.get("partition") or sim.get("queue")
    return str(part) if part else "normal"


def equil_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Submit & monitor the equilibration SLURM job.

    Behavior
    --------
    - Always uses SLURM (no local execution).
    - If FINISHED exists, returns immediately.
    - If FAILED exists, clears it and resubmits.
    - If job not active (or was killed), resubmits.
    - Blocks until FINISHED/FAILED; raises on FAILED/unknown terminal state.
    """
    paths = _phase_paths(system.root)
    lig = (system.meta or {}).get("ligand", system.name)
    part = _read_partition(params)

    # Fast path: already done
    if paths["finished"].exists():
        logger.info(f"[equil:{lig}] FINISHED detected — skipping submit/monitor.")
        arts = {"rst7": paths["rst7"], "finished": paths["finished"]}
        for k in ("stdout", "stderr"):
            if paths[k].exists():
                arts[k] = paths[k]
        # best-effort job id
        try:
            arts["job_id"] = paths["jobid"].read_text().strip()
        except Exception:
            pass
        return ExecResult(job_ids=[], artifacts=arts)

    # If previously FAILED, clear the sentinel so the manager can retry cleanly
    if paths["failed"].exists():
        try:
            paths["failed"].unlink(missing_ok=True)
        except Exception:
            pass

    # Require the submit script to exist
    script = paths["script"]
    if not script.exists():
        raise FileNotFoundError(f"[equil:{lig}] SLURM submit script missing: {script}")

    # Build job spec (submit from equil/ directory; pass partition)
    job_name = f"{system.name.replace(':', '_')}_equil"
    spec = SlurmJobSpec(
        workdir=paths["phase_dir"],
        script_rel=script.name,                   # relative (submitted with cwd=workdir)
        finished_name=paths["finished"].name,
        failed_name=paths["failed"].name,
        name=job_name,
        extra_sbatch=["-p", part],
    )

    # Ensure running, then block until done (15-min poll; no overall timeout)
    mgr = SlurmJobManager(poll_s=60 * 15)
    mgr.ensure_running(spec)
    # best-effort read of job id for logging
    try:
        jid = paths["jobid"].read_text().strip()
    except Exception:
        jid = None
    logger.info(f"[equil:{lig}] SLURM job ensured running (jobid={jid or 'unknown'}, partition={part}).")

    mgr.wait_until_done([spec])

    # Decide outcome based on sentinels
    if paths["finished"].exists():
        logger.success(f"[equil:{lig}] FINISHED")
        arts = {"rst7": paths["rst7"], "finished": paths["finished"]}
        for k in ("stdout", "stderr"):
            if paths[k].exists():
                arts[k] = paths[k]
        if jid:
            arts["job_id"] = jid
        return ExecResult(job_ids=[jid] if jid else [], artifacts=arts)

    if paths["failed"].exists():
        msg = f"[equil:{lig}] FAILED — see {paths['stderr']} / {paths['stdout']}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Neither FINISHED nor FAILED (shouldn't happen if manager did its job)
    raise RuntimeError(f"[equil:{lig}] Unknown terminal state in {paths['phase_dir']}")