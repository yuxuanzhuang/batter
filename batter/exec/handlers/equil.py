"""Slurm-backed equilibration handler."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from batter.exec.slurm_mgr import SlurmJobManager, SlurmJobSpec
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.exec.handlers.batch import render_batch_slurm_script

def _phase_paths(root: Path) -> dict[str, Path]:
    """Return resolved paths for equilibration artifacts under ``root``."""
    phase_dir = root / "equil"
    return {
        "phase_dir": phase_dir,
        "script": phase_dir / "SLURMM-run",
        "finished": phase_dir / "FINISHED",
        "failed": phase_dir / "FAILED",
        "jobid": phase_dir / "JOBID",
        "stdout": phase_dir / "slurm.out",
        "stderr": phase_dir / "slurm.err",
        "rst7": root / "artifacts" / "equil" / "equil.rst7",
    }


def equil_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """Submit and register the equilibration job with the Slurm manager.

    Parameters
    ----------
    step : Step
        Pipeline step metadata (unused but provided for symmetry).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Raw handler payload; validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Result containing either existing artifacts (when already finished) or
        the work directory to be monitored by the manager.

    Raises
    ------
    FileNotFoundError
        If the expected submission script is missing.
    RuntimeError
        When ``payload['job_mgr']`` is not a :class:`SlurmJobManager`.
    """
    payload = StepPayload.model_validate(params)
    paths = _phase_paths(system.root)
    lig = system.meta.get("ligand", system.name)

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

    # Build job spec (submit from equil/ directory; pass partition via sbatch flags)
    job_name = f"fep_{os.path.abspath(system.root)}_eq"
    batch_script = None
    if payload.get("batch_mode"):
        batch_root = payload.get("batch_run_root")
        batch_script = render_batch_slurm_script(
            batch_root=batch_root or (system.root.parent.parent / "batch_run"),
            target_dir=paths["phase_dir"],
            run_script="run-local.bash",
            env=None,
            system_name=getattr(payload.get("sim"), "system_name", system.name),
            stage="equil",
            pose=system.meta.get("ligand", system.name),
            header_root=getattr(payload.get("sim"), "slurm_header_dir", None),
        )
    script_rel = batch_script.name if batch_script else script.name
    submit_dir = batch_script.parent if batch_script else None
    spec = SlurmJobSpec(
        workdir=paths["phase_dir"],
        script_rel=script_rel,
        finished_name=paths["finished"].name,
        failed_name=paths["failed"].name,
        name=job_name,
        batch_script=batch_script,
        submit_dir=submit_dir,
    )

    mgr = payload.get("job_mgr")
    if not isinstance(mgr, SlurmJobManager):
        raise RuntimeError("Equilibration handler requires payload['job_mgr'] to be a SlurmJobManager instance")
    mgr.add(spec)
    return ExecResult(job_ids=[], artifacts={"workdir": paths["phase_dir"]})
