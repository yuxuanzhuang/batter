# batter/exec/handlers/fe.py
from __future__ import annotations

import os
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.exec.slurm_mgr import SlurmJobManager, SlurmJobSpec as _BaseSlurmJobSpec

# ---------------- small shim to add env to SlurmJobSpec ----------------

@dataclass
class SlurmJobSpec(_BaseSlurmJobSpec):
    """
    Extend base SlurmJobSpec with per-job environment injection.

    These are passed to sbatch as: --export=ALL,KEY=VAL,KEY2=VAL2...
    """
    extra_env: Dict[str, str] = field(default_factory=dict)

# monkey-patch the manager's _submit to honor extra_env
# (If your SlurmJobManager already supports this, you can drop the patching.)
from batter.exec.slurm_mgr import SlurmJobManager as _BaseSlurmJobManager

class SlurmJobManager(_BaseSlurmJobManager):
    def _submit(self, spec: SlurmJobSpec) -> str:
        script_abs = spec.resolve_script_abs()
        if not script_abs.exists():
            listing = ", ".join(sorted(p.name for p in spec.workdir.iterdir())) if spec.workdir.exists() else "(missing workdir)"
            raise FileNotFoundError(
                f"SLURM script not found: {script_abs}\n"
                f"in workdir: {spec.workdir}\n"
                f"contents: {listing}"
            )

        try:
            script_abs.chmod(script_abs.stat().st_mode | 0o111)
        except Exception:
            pass

        # Build sbatch command (relative to workdir)
        cmd: List[str] = ["sbatch"]
        if spec.name:
            cmd += ["--job-name", spec.name]
        if spec.extra_sbatch:
            cmd += list(spec.extra_sbatch)

        # Environment export
        if getattr(spec, "extra_env", None):
            kv = [f"{k}={v}" for k, v in spec.extra_env.items()]
            cmd += ["--export", "ALL," + ",".join(kv)]

        cmd.append(spec.script_arg())

        logger.debug(f"[SLURM] sbatch: {' '.join(cmd)} (cwd={spec.workdir})")
        out = subprocess.check_output(cmd, cwd=spec.workdir, text=True).strip()

        import re
        m = re.search(r"Submitted batch job\s+(\d+)", out, re.I)
        if not m:
            raise RuntimeError(f"Could not parse sbatch output: {out}")
        jobid = m.group(1)
        (spec.workdir / "JOBID").write_text(f"{jobid}\n")
        logger.info(f"[SLURM] submitted {spec.workdir.name} → job {jobid}")
        return jobid

# ---------------- utilities ----------------

def _read_partition(params: Dict[str, Any]) -> str:
    sim = params.get("sim", {}) or {}
    part = sim.get("partition") or sim.get("queue")
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
    if max_active <= 0:
        return
    while True:
        n = _active_job_count(user)
        if n < max_active:
            if n > 0:
                logger.info(f"[SLURM] Active jobs={n} < cap={max_active} — proceeding with submissions.")
            break
        logger.warning(f"[SLURM] Active jobs={n} ≥ cap={max_active}; sleeping {poll_s}s before submitting…")
        time.sleep(poll_s)

# ---------------- discovery helpers ----------------

def _components_under(root: Path) -> List[str]:
    fe_root = root / "fe"
    if not fe_root.exists():
        return []
    return [p.name for p in sorted(fe_root.iterdir()) if p.is_dir()]

def _equil_window_dir(root: Path, comp: str) -> Path:
    # <ligand>/fe/<comp>/<comp>-1
    return root / "fe" / comp / f"{comp}-1"

def _production_window_dirs(root: Path, comp: str) -> List[Path]:
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
    workdir: Path, *, finished_name: str, part: str, job_name: str, extra_env: Optional[Dict[str, str]] = None
) -> SlurmJobSpec:
    extra = ["-p", part]
    return SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        finished_name=finished_name,
        failed_name="FAILED",
        name=job_name,
        extra_sbatch=extra,
        extra_env=extra_env or {},
    )

# ---------------- handlers ----------------

def fe_equil_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Submit & monitor FE-equilibration jobs for all components of a ligand.

    - Root: <ligand>/fe/<comp>/<comp>-1
    - Success sentinel: EQ_FINISHED
    - Adds env: ONLY_EQ=1, INPCRD=full.inpcrd
    - Checks job cap once per ligand
    """
    lig = (system.meta or {}).get("ligand", system.name)
    part = _read_partition(params)
    max_jobs = int(params.get("max_active_jobs", 1000))

    comps = _components_under(system.root)
    if not comps:
        raise FileNotFoundError(f"[fe_equil:{lig}] No components under {system.root/'fe'}")

    _ensure_job_quota(max_jobs)

    specs: List[SlurmJobSpec] = []
    for comp in comps:
        wd = _equil_window_dir(system.root, comp)
        if not wd.exists():
            logger.warning(f"[fe_equil:{lig}] missing window dir: {wd} — skipping")
            continue

        # clear FAILED if present
        failed = wd / "FAILED"
        if failed.exists():
            try:
                failed.unlink()
            except Exception:
                pass

        job_name = f"{system.root.name}_{comp}_{comp}-1"
        env = {"ONLY_EQ": "1", "INPCRD": "full.inpcrd"}
        spec = _spec_from_dir(
            wd,
            finished_name="EQ_FINISHED",
            part=part,
            job_name=job_name,
            extra_env=env,
        )
        specs.append(spec)

    if not specs:
        raise RuntimeError(f"[fe_equil:{lig}] No component equil windows to submit.")

    mgr = SlurmJobManager(poll_s=60 * 15)
    for s in specs:
        try:
            mgr.ensure_running(s)
        except Exception as e:
            logger.error(f"[fe_equil:{lig}] submit failed for {s.workdir}: {e}")

    mgr.wait_until_done(specs)

    arts: Dict[str, Dict[str, Any]] = {}
    for s in specs:
        key = str(s.workdir)
        arts.setdefault(key, {})
        jid = (s.workdir / "JOBID")
        if jid.exists():
            arts[key]["job_id"] = jid.read_text().strip()
        if s.finished_path().exists():
            arts[key]["finished"] = s.finished_path()
        for logn in ("slurm.out", "slurm.err"):
            p = s.workdir / logn
            if p.exists():
                arts[key][logn.rstrip(".out").rstrip(".err") or logn] = p

    logger.success(f"[fe_equil:{lig}] all components reached terminal state.")
    return ExecResult(job_ids=[], artifacts=arts)


def fe_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Submit & monitor FE production jobs for all components and windows.

    - Root: <ligand>/fe/<comp>/<comp{idx}>
    - Success sentinel: FINISHED
    - Adds env: INPCRD=../{comp}-1/eqnpt04.rst7
    - Checks job cap once per ligand
    """
    lig = (system.meta or {}).get("ligand", system.name)
    part = _read_partition(params)
    max_jobs = int(params.get("max_active_jobs", 1000))

    comps = _components_under(system.root)
    if not comps:
        raise FileNotFoundError(f"[fe:{lig}] No components under {system.root/'fe'}")

    _ensure_job_quota(max_jobs)

    specs: List[SlurmJobSpec] = []
    for comp in comps:
        for wd in _production_window_dirs(system.root, comp):
            # clear FAILED if present
            failed = wd / "FAILED"
            if failed.exists():
                try:
                    failed.unlink()
                except Exception:
                    pass

            job_name = f"{system.root.name}_{comp}_{wd.name}"
            env = {"INPCRD": f"../{comp}-1/eqnpt04.rst7"}

            spec = _spec_from_dir(
                wd,
                finished_name="FINISHED",
                part=part,
                job_name=job_name,
                extra_env=env,
            )
            specs.append(spec)

    if not specs:
        raise RuntimeError(f"[fe:{lig}] No production windows to submit.")

    mgr = SlurmJobManager(poll_s=60 * 15)
    for s in specs:
        try:
            mgr.ensure_running(s)
        except Exception as e:
            logger.error(f"[fe:{lig}] submit failed for {s.workdir}: {e}")

    mgr.wait_until_done(specs)

    arts: Dict[str, Dict[str, Any]] = {}
    for s in specs:
        key = str(s.workdir)
        arts.setdefault(key, {})
        jid = (s.workdir / "JOBID")
        if jid.exists():
            arts[key]["job_id"] = jid.read_text().strip()
        if s.finished_path().exists():
            arts[key]["finished"] = s.finished_path()
        for logn in ("slurm.out", "slurm.err"):
            p = s.workdir / logn
            if p.exists():
                arts[key][logn.rstrip(".out").rstrip(".err") or logn] = p

    logger.success(f"[fe:{lig}] all production jobs reached terminal state.")
    return ExecResult(job_ids=[], artifacts=arts)