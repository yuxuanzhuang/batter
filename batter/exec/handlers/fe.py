# batter/exec/handlers/fe.py
from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.exec.slurm_mgr import SlurmJobSpec  # job manager is passed via params["job_mgr"]
from batter.utils import components_under
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload

# ---------------- utilities ----------------

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

def _active_job_count(user: Optional[str] = None) -> int:
    user = user or os.environ.get("USER")
    if not user:
        return 0
    try:
        out = subprocess.check_output(["squeue", "-h", "-u", user, "-o", "%i"], text=True)
        return sum(1 for ln in out.splitlines() if ln.strip())
    except Exception:
        return 0

def _ensure_job_quota(max_active: int, user: Optional[str] = None, poll_s: int = 60) -> None:
    """
    Enforce a one-time cap on active jobs at the start of a ligand.
    (Do not re-check per job; the global manager will handle the rest.)
    """
    if max_active <= 0:
        return
    while True:
        n = _active_job_count(user)
        if n < max_active:
            if n > 0:
                logger.debug(f"[SLURM] Active jobs={n} < cap={max_active} — proceeding with submissions.")
            break
        logger.warning(f"[SLURM] Active jobs={n} ≥ cap={max_active}; sleeping {poll_s}s before submitting…")
        time.sleep(poll_s)

# ---------------- discovery helpers ----------------

def _equil_window_dir(root: Path, comp: str) -> Path:
    # <ligand>/fe/<comp>/<comp>-1
    return root / "fe" / comp / f"{comp}-1"

def _production_window_dirs(root: Path, comp: str) -> List[Path]:
    """
    Return all production window dirs under <ligand>/fe/<comp> matching <comp>0, <comp>1, ...
    (Skip equil dir <comp>-1).
    """
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
            tail = p.name[len(comp):]
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

def fe_equil_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Enqueue FE-equilibration jobs for all components of a ligand (non-blocking).

    - Workdir per job: <ligand>/fe/<comp>/<comp>-1
    - Success sentinel: EQ_FINISHED
    - Env: ONLY_EQ=1, INPCRD=full.inpcrd
    - Applies one-time job cap check per ligand
    - Requires a global manager at params["job_mgr"]
    """
    payload = StepPayload.model_validate(params)
    lig = system.meta.get("ligand", system.name)
    part = _read_partition(payload)
    max_jobs = int(payload.get("max_active_jobs", 2000))

    job_mgr = payload.get("job_mgr")
    if job_mgr is None:
        raise ValueError("[fe_equil] params must include a global 'job_mgr' (SlurmJobManager).")

    comps = components_under(system.root)
    if not comps:
        raise FileNotFoundError(f"[fe_equil:{lig}] No components found under {system.root/'fe'}")

    _ensure_job_quota(max_jobs)

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
            logger.warning(f"[fe_equil:{lig}] missing equil window dir: {wd} — skipping")
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
        job_mgr.add(spec)
        count += 1

    if count == 0:
        raise RuntimeError(f"[fe_equil:{lig}] No component equil windows to submit.")

    logger.debug(f"[fe_equil:{lig}] enqueued {count} component equil job(s) (partition={part}).")
    # Don’t claim success/terminal state; we’re not waiting here.
    return ExecResult(job_ids=[], artifacts={"count": count})

def fe_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Enqueue FE production jobs for all components and windows of a ligand (non-blocking).

    - Workdir per job: <ligand>/fe/<comp>/<comp{idx}>
    - Success sentinel: FINISHED
    - Env: INPCRD=../{comp}-1/eqnpt04.rst7
    - Applies one-time job cap check per ligand
    - Requires a global manager at params["job_mgr"]
    """
    payload = StepPayload.model_validate(params)
    lig = system.meta.get("ligand", system.name)
    part = _read_partition(payload)
    max_jobs = int(payload.get("max_active_jobs", 2000))

    job_mgr = payload.get("job_mgr")
    if job_mgr is None:
        raise ValueError("[fe] params must include a global 'job_mgr' (SlurmJobManager).")

    comps = components_under(system.root)
    if not comps:
        raise FileNotFoundError(f"[fe:{lig}] No components found under {system.root/'fe'}")

    _ensure_job_quota(max_jobs)

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
            job_mgr.add(spec)
            count += 1

    if count == 0:
        raise RuntimeError(f"[fe:{lig}] No production windows to submit.")

    logger.debug(f"[fe:{lig}] enqueued {count} production job(s) (partition={part}).")
    # Don’t claim success/terminal state; we’re not waiting here.
    return ExecResult(job_ids=[], artifacts={"count": count})
