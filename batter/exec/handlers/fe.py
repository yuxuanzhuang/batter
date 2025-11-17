"""Handlers that queue free-energy equilibration and production jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from batter.exec.slurm_mgr import SlurmJobManager, SlurmJobSpec
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils import components_under

# ---------------- utilities ----------------


def _read_partition(payload: StepPayload) -> str:
    """Resolve the desired Slurm partition from ``payload``."""
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


# ---------------- discovery helpers ----------------


def _equil_window_dir(root: Path, comp: str) -> Path:
    """Return the equilibration window directory for ``comp``."""
    return root / "fe" / comp / f"{comp}-1"


def _production_window_dirs(root: Path, comp: str) -> List[Path]:
    """Return production window directories for ``comp``."""
    base = root / "fe" / comp
    if not base.exists():
        return []
    out: List[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name == f"{comp}-1":
            continue
        if p.name.startswith(comp):
            tail = p.name[len(comp) :]
            if tail and tail.lstrip("-").isdigit():
                out.append(p)
    return out


def _spec_from_dir(
    workdir: Path,
    *,
    finished_name: str,
    part: str,
    job_name: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> SlurmJobSpec:
    """Build a :class:`SlurmJobSpec` for ``workdir``."""
    return SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        finished_name=finished_name,
        failed_name="FAILED",
        name=job_name,
        extra_sbatch=["-p", part],
        extra_env=extra_env or {},
    )


# ---------------- handlers ----------------


def fe_equil_handler(
    step: Step, system: SimSystem, params: Dict[str, Any]
) -> ExecResult:
    """Queue equilibration jobs for each component of a ligand.

    Parameters
    ----------
    step, system : ignored
        Included for parity with the handler signature.
    params : dict
        Handler payload containing the job manager and configuration values.

    Returns
    -------
    ExecResult
        Number of jobs enqueued (without waiting for completion).
    """
    payload = StepPayload.model_validate(params)
    lig = system.meta.get("ligand", system.name)
    part = _read_partition(payload)
    max_jobs = int(payload.get("max_active_jobs", 500))

    job_mgr = payload.get("job_mgr")
    if not isinstance(job_mgr, SlurmJobManager):
        raise ValueError(
            "[fe_equil] payload['job_mgr'] must be an instance of SlurmJobManager."
        )

    comps = components_under(system.root)
    if not comps:
        raise FileNotFoundError(
            f"[fe_equil:{lig}] No components found under {system.root/'fe'}"
        )

    # quota enforced inside the job manager before each add

    register_phase_state(
        system.root,
        "fe_equil",
        required=[
            ["fe/{comp}/{comp}-1/EQ_FINISHED"],
            ["fe/{comp}/{comp}-1/FAILED"],
        ],
        success=[["fe/{comp}/{comp}-1/EQ_FINISHED"]],
        failure=[["fe/{comp}/{comp}-1/FAILED"]],
    )

    count = 0
    for comp in comps:
        wd = _equil_window_dir(system.root, comp)
        if not wd.exists():
            logger.warning(
                f"[fe_equil:{lig}] missing equil window dir: {wd} — skipping"
            )
            continue

        # clear FAILED if present
        failed = wd / "FAILED"
        if failed.exists():
            try:
                failed.unlink()
            except Exception:
                pass

        env = {"ONLY_EQ": "1", "INPCRD": "full.inpcrd"}
        job_name = f"fep_{os.path.abspath(system.root)}_{comp}_fe_equil"
        spec = _spec_from_dir(
            wd,
            finished_name="EQ_FINISHED",
            part=part,
            job_name=job_name,
            extra_env=env,
        )
        job_mgr.add(spec, max_active=max_jobs)
        count += 1

    if count == 0:
        raise RuntimeError(f"[fe_equil:{lig}] No component equil windows to submit.")

    logger.debug(
        f"[fe_equil:{lig}] enqueued {count} component equil job(s) (partition={part})."
    )
    # Don’t claim success/terminal state; we’re not waiting here.
    return ExecResult(job_ids=[], artifacts={"count": count})


def fe_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """Queue production jobs for each component/window combination.

    Parameters
    ----------
    step, system : ignored
        Provided for handler API compatibility.
    params : dict
        Handler payload containing the job manager and configuration values.

    Returns
    -------
    ExecResult
        Number of jobs enqueued (without waiting for completion).
    """
    payload = StepPayload.model_validate(params)
    lig = system.meta.get("ligand", system.name)
    part = _read_partition(payload)
    max_jobs = int(payload.get("max_active_jobs", 500))

    job_mgr = payload.get("job_mgr")
    if not isinstance(job_mgr, SlurmJobManager):
        raise ValueError(
            "[fe] payload['job_mgr'] must be an instance of SlurmJobManager."
        )

    comps = components_under(system.root)
    if not comps:
        raise FileNotFoundError(
            f"[fe:{lig}] No components found under {system.root/'fe'}"
        )

    register_phase_state(
        system.root,
        "fe",
        required=[
            ["fe/{comp}/{comp}{win:02d}/FINISHED"],
            ["fe/{comp}/{comp}{win:02d}/FAILED"],
        ],
        success=[["fe/{comp}/{comp}{win:02d}/FINISHED"]],
        failure=[["fe/{comp}/{comp}{win:02d}/FAILED"]],
    )

    count = 0
    for comp in comps:
        for wd in _production_window_dirs(system.root, comp):
            # clear FAILED if present
            failed = wd / "FAILED"
            if failed.exists():
                try:
                    failed.unlink()
                except Exception:
                    pass

            env = {"INPCRD": f"../{comp}-1/eqnpt04.rst7"}
            job_name = f"fep_{os.path.abspath(system.root)}_{comp}_{wd.name}_fe"
            spec = _spec_from_dir(
                wd,
                finished_name="FINISHED",
                part=part,
                job_name=job_name,
                extra_env=env,
            )
            job_mgr.add(spec, max_active=max_jobs)
            count += 1

    if count == 0:
        raise RuntimeError(f"[fe:{lig}] No production windows to submit.")

    logger.debug(f"[fe:{lig}] enqueued {count} production job(s) (partition={part}).")
    # Don’t claim success/terminal state; we’re not waiting here.
    return ExecResult(job_ids=[], artifacts={"count": count})
