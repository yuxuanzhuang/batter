"""Slurm-backed equilibration handler."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from textwrap import dedent

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
    batch_mode = bool(payload.get("batch_mode"))
    mgr = payload.get("job_mgr")
    if not isinstance(mgr, SlurmJobManager):
        raise RuntimeError(
            "Equilibration handler requires payload['job_mgr'] to be a SlurmJobManager instance"
        )

    # Batch path: submit one SLURM script that loops over all ligand equil dirs
    if batch_mode:
        if getattr(mgr, "_equil_batch_added", False):
            return ExecResult(
                job_ids=[], artifacts={"batch_run": payload.get("batch_run_root")}
            )
        run_root = system.root.parent.parent if system.root.name else system.root
        batch_root = payload.get("batch_run_root") or (run_root / "batch_run")
        helper_script = _write_equil_batch_runner(
            run_root,
            batch_root,
            batch_gpus=payload.get("batch_gpus"),
            gpus_per_task=payload.get("batch_gpus_per_task") or 1,
        )
        batch_script = render_batch_slurm_script(
            batch_root=batch_root,
            target_dir=batch_root,
            run_script=helper_script.name,
            env=None,
            system_name=getattr(payload.get("sim"), "system_name", system.name),
            stage="equil",
            pose="all",
            header_root=getattr(payload.get("sim"), "slurm_header_dir", None),
        )
        spec = SlurmJobSpec(
            workdir=batch_root,
            script_rel=batch_script.name,
            finished_name="equil_all.FINISHED",
            failed_name="equil_all.FAILED",
            name=job_name,
            batch_script=batch_script,
            submit_dir=batch_root,
        )
        mgr.add(spec)
        setattr(mgr, "_equil_batch_added", True)
        return ExecResult(job_ids=[], artifacts={"batch_run": batch_root})

    # Default per-ligand submit
    spec = SlurmJobSpec(
        workdir=paths["phase_dir"],
        script_rel=script.name,
        finished_name=paths["finished"].name,
        failed_name=paths["failed"].name,
        name=job_name,
    )

    mgr.add(spec)
    return ExecResult(job_ids=[], artifacts={"workdir": paths["phase_dir"]})


def _write_equil_batch_runner(
    run_root: Path,
    batch_root: Path,
    *,
    batch_gpus: int | None = None,
    gpus_per_task: int = 1,
) -> Path:
    """Create a helper script that runs all ligand equil jobs in parallel."""
    batch_root.mkdir(parents=True, exist_ok=True)
    helper = batch_root / "run_all_equil.sh"
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
        for d in "{(run_root / 'simulations').as_posix()}"/*/equil; do
            if [[ -x "$d/run-local.bash" ]]; then
                echo "[batter-batch] running $d"
                (
                    cd "$d"
                    $MPI_EXEC -N 1 -n 1 --gpus-per-task $GPUS_PER_TASK /bin/bash run-local.bash
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
            touch "{(batch_root / 'equil_all.FINISHED').as_posix()}"
        else
            touch "{(batch_root / 'equil_all.FAILED').as_posix()}"
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
