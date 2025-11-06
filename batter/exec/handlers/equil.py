from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.exec.slurm_mgr import SlurmJobManager, SlurmJobSpec
from batter.pipeline.payloads import StepPayload
from batter.orchestrate.state_registry import register_phase_state

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


def _read_partition(payload: StepPayload) -> str:
    if payload.sim is not None:
        sim_cfg = payload.sim
        part = getattr(sim_cfg, "partition", None) or getattr(sim_cfg, "queue", None)
        if part:
            return str(part)
        sim_dict = sim_cfg.model_dump()
        part = sim_dict.get("partition") or sim_dict.get("queue")
        if part:
            return str(part)
    sim_extra = payload.get("sim", {})
    if isinstance(sim_extra, dict):
        part = sim_extra.get("partition") or sim_extra.get("queue")
        if part:
            return str(part)
    part = payload.get("partition") or payload.get("queue")
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
    payload = StepPayload.model_validate(params)
    paths = _phase_paths(system.root)
    lig = system.meta.get("ligand", system.name)
    part = _read_partition(payload)

    finished_rel = paths["finished"].relative_to(system.root).as_posix()
    failed_rel = paths["failed"].relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "equil",
        required=[[finished_rel], [failed_rel]],
        success=[[finished_rel]],
        failure=[[failed_rel]],
    )

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
    job_name = f"fep_{os.path.abspath(system.root)}_eq"
    spec = SlurmJobSpec(
        workdir=paths["phase_dir"],
        script_rel=script.name,                   # relative (submitted with cwd=workdir)
        finished_name=paths["finished"].name,
        failed_name=paths["failed"].name,
        name=job_name,
        extra_sbatch=["-p", part],
    )

    mgr = payload.get("job_mgr")
    mgr.add(spec)
    return ExecResult(job_ids=[], artifacts={"workdir": paths["phase_dir"]})
