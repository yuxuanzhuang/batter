"""Handlers that queue free-energy equilibration and production jobs."""

from __future__ import annotations

import os
import subprocess
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from batter.exec.slurm_mgr import SlurmJobManager, SlurmJobSpec
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils import components_under
from batter.exec.handlers.batch import render_batch_slurm_script
from textwrap import dedent

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
    job_name: str,
    failed_name: str = "FAILED",
    script_rel: str = "SLURMM-run",
    extra_env: Optional[Dict[str, str]] = None,
    batch_script: Path | None = None,
    submit_dir: Path | None = None,
) -> SlurmJobSpec:
    """Build a :class:`SlurmJobSpec` for ``workdir``."""
    script_name = batch_script.name if batch_script else script_rel
    return SlurmJobSpec(
        workdir=workdir,
        script_rel=script_name,
        finished_name=finished_name,
        failed_name=failed_name,
        name=job_name,
        extra_env=extra_env or {},
        batch_script=batch_script,
        submit_dir=submit_dir,
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
            job_name=job_name,
            extra_env=env,
        )
        job_mgr.add(spec)
        count += 1

    if count == 0:
        raise RuntimeError(f"[fe_equil:{lig}] No component equil windows to submit.")

    logger.debug(f"[fe_equil:{lig}] enqueued {count} component equil job(s).")
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
    max_jobs = int(payload.get("max_active_jobs", 500))
    remd_enabled = False
    if payload.sim is not None:
        remd_enabled = str(getattr(payload.sim, "remd", "no")).lower() == "yes"
    batch_mode = bool(payload.get("batch_mode"))

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

    if remd_enabled:
        register_phase_state(
            system.root,
            "fe",
            required=[["fe/{comp}/FINISHED"], ["fe/{comp}/FAILED"]],
            success=[["fe/{comp}/FINISHED"]],
            failure=[["fe/{comp}/FAILED"]],
        )
    else:
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
    if batch_mode and not remd_enabled:
        run_root = system.root.parent.parent if system.root.name else system.root
        batch_root = payload.get("batch_run_root") or (run_root / "batch_run")
        lig = system.meta.get("ligand", system.name)
        batch_gpus = payload.get("batch_gpus")
        helper = _write_ligand_fe_batch_runner(
            system_root=system.root,
            batch_root=batch_root,
            ligand=lig,
            batch_gpus=batch_gpus,
            gpus_per_task=payload.get("batch_gpus_per_task") or 1,
        )
        safe_lig = lig.replace("/", "_")
        extra_sbatch: list[str] = []
        if batch_gpus:
            extra_sbatch += ["--gres", f"gpu:{batch_gpus}"]
        batch_script = render_batch_slurm_script(
            batch_root=batch_root,
            target_dir=batch_root,
            run_script=helper.name,
            env=None,
            system_name=getattr(payload.get("sim"), "system_name", system.name),
            stage="fe",
            pose=safe_lig,
            header_root=getattr(payload.get("sim"), "slurm_header_dir", None),
        )
        spec = SlurmJobSpec(
            workdir=batch_root,
            script_rel=batch_script.name,
            finished_name=f"fe_{safe_lig}.FINISHED",
            failed_name=f"fe_{safe_lig}.FAILED",
            name=f"fe_{safe_lig}",
            batch_script=batch_script,
            submit_dir=batch_root,
            extra_sbatch=extra_sbatch,
        )
        job_mgr.add(spec)
        return ExecResult(job_ids=[], artifacts={"batch_run": batch_root})

    for comp in comps:
        if remd_enabled:
            comp_dir = system.root / "fe" / comp
            failed = comp_dir / "FAILED"
            if failed.exists():
                try:
                    failed.unlink()
                except Exception:
                    pass

            job_name = f"fep_{os.path.abspath(system.root)}_{comp}_remd"
            spec = SlurmJobSpec(
                workdir=comp_dir,
                script_rel="SLURMM-BATCH-remd",
                finished_name="FINISHED",
                failed_name="FAILED",
                name=job_name,
            )
            job_mgr.add(spec)
            count += 1
            continue

        for wd in _production_window_dirs(system.root, comp):
            failed = wd / "FAILED"
            if failed.exists():
                try:
                    failed.unlink()
                except Exception:
                    pass

            env = {"INPCRD": f"../{comp}-1/eqnpt04.rst7"}
            job_name = f"fep_{os.path.abspath(system.root)}_{comp}_{wd.name}_fe"
            batch_script = None
            if batch_mode:
                batch_root = payload.get("batch_run_root") or (
                    system.root.parent.parent / "batch_run"
                )
                batch_script = render_batch_slurm_script(
                    batch_root=batch_root,
                    target_dir=wd,
                    run_script="run-local.bash",
                    env=env,
                    system_name=getattr(payload.get("sim"), "system_name", system.name),
                    stage="fe",
                    pose=f"{system.meta.get('ligand', system.name)}_{comp}_{wd.name}",
                    header_root=getattr(payload.get("sim"), "slurm_header_dir", None),
                )
            spec = _spec_from_dir(
                wd,
                finished_name="FINISHED",
                job_name=job_name,
                extra_env=env,
                batch_script=batch_script,
                submit_dir=batch_script.parent if batch_script else None,
            )
            job_mgr.add(spec)
            count += 1

    if count == 0:
        raise RuntimeError(f"[fe:{lig}] No production windows to submit.")

    logger.debug(f"[fe:{lig}] enqueued {count} production job(s).")
    # Don’t claim success/terminal state; we’re not waiting here.
    return ExecResult(job_ids=[], artifacts={"count": count})


def _write_ligand_fe_batch_runner(
    *,
    system_root: Path,
    batch_root: Path,
    ligand: str,
    batch_gpus: int | None = None,
    gpus_per_task: int = 1,
) -> Path:
    """Render a helper that launches all production windows for a single ligand in parallel."""
    batch_root.mkdir(parents=True, exist_ok=True)
    safe_lig = ligand.replace("/", "_")
    helper = batch_root / f"run_fe_{safe_lig}.sh"
    gpus_per_task = max(1, int(gpus_per_task))
    gpu_line = (
        f'TOTAL_GPUS="{batch_gpus}"'
        if batch_gpus
        else 'TOTAL_GPUS="${SLURM_GPUS_ON_NODE:-1}"'
    )
    text = (
        dedent(
            f"""
        #!/usr/bin/env bash
        set -euo pipefail
        {gpu_line}
        GPUS_PER_TASK={gpus_per_task}
        if [[ -z "$TOTAL_GPUS" ]]; then
            if [[ -n "${{SLURM_GPUS:-}}" ]]; then TOTAL_GPUS="${{SLURM_GPUS}}"; else TOTAL_GPUS="1"; fi
        fi
        slots=$((TOTAL_GPUS / GPUS_PER_TASK))
        if [[ $slots -lt 1 ]]; then slots=1; fi

        status=0
        declare -a pids=()
        running=0
        for w in "{(system_root / 'fe').as_posix()}"/*/*; do
            comp_dir=$(dirname "$w")
            comp=$(basename "$comp_dir")
            base=$(basename "$w")
            if [[ "$base" == "$comp-1" ]]; then
                continue
            fi
            if [[ -x "$w/run-local.bash" ]]; then
                echo "[batter-batch] fe running $w"
                (
                    cd "$w"
                    srun -N 1 -n 1 --gpus-per-task $GPUS_PER_TASK /bin/bash run-local.bash
                ) &
                pids+=($!)
                running=$((running + 1))
                if [[ $running -ge $slots ]]; then
                    if wait -n; then :; else status=$?; fi
                    running=$((running - 1))
                fi
            fi
        done

        for pid in "${{pids[@]:-}}"; do
            if wait "$pid"; then :; else status=$?; fi
        done

        if [[ $status -eq 0 ]]; then
            touch "{(batch_root / f'fe_{safe_lig}.FINISHED').as_posix()}"
        else
            touch "{(batch_root / f'fe_{safe_lig}.FAILED').as_posix()}"
        fi
        exit $status
        """
        ).strip()
        + "\n"
    )
    helper.write_text(text)
    try:
        helper.chmod(0o755)
    except Exception:
        pass
    return helper
