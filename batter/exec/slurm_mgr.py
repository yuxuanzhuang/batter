"""Utilities for monitoring and resubmitting Slurm-managed jobs.

Notes
-----
This module relies on :mod:`fcntl` and is therefore intended for POSIX
systems (e.g. typical HPC clusters running Slurm).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger


# ---------- atomic registry append ----------
def _atomic_append_jsonl_unique(
    path: Path, rec: dict, unique_key: str = "workdir"
) -> None:
    """Append ``rec`` to a JSONL file if ``unique_key`` is not already present.

    Parameters
    ----------
    path : pathlib.Path
        Target JSONL file (created if missing).
    rec : dict
        Record to append.
    unique_key : str, optional
        Key whose value must be unique across existing rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key_val = rec.get(unique_key)
    if key_val is None:
        raise ValueError(f"Record missing unique key '{unique_key}': {rec}")

    # Open for read+append; create if missing.
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Rewind and scan existing lines
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = json.loads(line)
                except Exception:
                    continue
                if prev.get(unique_key) == key_val:
                    # Already registered for this folder → no-op
                    return

            # Not found → append
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---- SLURM state sets ----
SLURM_OK_STATES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "STAGE_OUT",
    "SUSPENDED",
}
SLURM_FINAL_BAD = {
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "OUT_OF_MEMORY",
}
JOBID_RE = re.compile(r"Submitted batch job\s+(\d+)", re.I)


# ---- Helpers ----
def _read_text(p: Path) -> Optional[str]:
    """Return stripped file contents or ``None`` if the file is unreadable."""
    try:
        return p.read_text().strip()
    except Exception:
        return None


def _write_text(p: Path, txt: str) -> None:
    """Write ``txt`` to ``p`` creating parent directories as required."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)


def _jobid_file(root: Path) -> Path:
    """Convenience helper returning ``root / 'JOBID'``."""
    return root / "JOBID"


def _state_from_squeue(jobid: str) -> Optional[str]:
    """Query ``squeue`` for ``jobid`` returning the job state."""
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", jobid, "-o", "%T"],
            text=True,
            stderr=subprocess.DEVNULL,  # hide "slurm_load_jobs error: Invalid job id specified"
        ).strip()
        if out:
            return out.split()[0]
    except Exception:
        pass
    return None


def _state_from_sacct(jobid: str) -> Optional[str]:
    """Query ``sacct`` for ``jobid`` returning the job state."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jobid, "-X", "-n", "-o", "State"],
            text=True,
            stderr=subprocess.DEVNULL,  # hide warnings/errors
        )
        for ln in out.splitlines():
            ln = ln.strip()
            if ln:
                return ln.split()[0]
    except Exception:
        pass
    return None


def _num_active_job(user: Optional[str] = None, partition: Optional[str] = None) -> int:
    """Return the number of active Slurm jobs for ``user`` (optionally within a partition).

    Parameters
    ----------
    user : str, optional
        Unix user name. If ``None``, defaults to ``$USER``.
    partition : str, optional
        Partition/queue name to filter with ``squeue -p``.

    Returns
    -------
    int
        Number of jobs currently reported by ``squeue`` for the user.
    """
    user = user or os.environ.get("USER")
    if not user:
        return 0
    cmd = ["squeue", "-h", "-u", user]
    if partition:
        cmd += ["-p", partition]
    cmd += ["-o", "%i"]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return 0

    n_ids = [line.strip() for line in out.splitlines() if line.strip()]
    logger.debug(
        f"[SQUEUE] active jobs for user '{user}'{f' partition={partition}' if partition else ''}: {n_ids}"
    )
    return len(n_ids)


def _slurm_state(jobid: Optional[str]) -> Optional[str]:
    """Return the best-effort Slurm state for ``jobid``."""
    if not jobid:
        return None
    return _state_from_squeue(jobid) or _state_from_sacct(jobid)


def _parse_gpu_env(value: str) -> Optional[int]:
    """Parse a GPU count from common SLURM/CUDA environment variables."""
    if not value:
        return None
    txt = value.strip()
    if not txt:
        return None
    if txt.isdigit():
        count = int(txt)
        return count if count > 0 else None
    tokens = [t for t in re.split(r"[,:]", txt) if t]
    digits = [t for t in tokens if t.isdigit()]
    if len(digits) > 1:
        return len(digits)
    if digits:
        try:
            count = int(digits[-1])
            return count if count > 0 else None
        except Exception:
            pass
    if tokens:
        return len(tokens)
    return None


def _infer_stage_from_workdir(path: Path | None) -> Optional[str]:
    """Heuristically infer a stage name from ``path`` for legacy queue entries."""
    if not path:
        return None
    parts = [p.lower() for p in path.parts]
    if "equil" in parts:
        return "equil"
    if "fe" in parts:
        # component equil windows are named <comp>-1; production windows differ
        if path.name.endswith("-1"):
            return "fe_equil"
        return "fe"
    return None


# ---- Spec ----
@dataclass
class SlurmJobSpec:
    """Descriptor for a Slurm job managed by :class:`SlurmJobManager`.

    Parameters
    ----------
    workdir : pathlib.Path
        Working directory containing submission scripts and sentinel files.
    script_rel : str, optional
        Preferred relative submission script path (default: ``"SLURMM-run"``).
    finished_name : str, optional
        Name of the sentinel file indicating success.
    failed_name : str, optional
        Name of the sentinel file indicating failure.
    name : str, optional
        Friendly display name used in logs.
    extra_sbatch : Sequence[str], optional
        Additional ``sbatch`` flags appended during submission.
    extra_env : dict, optional
        Additional environment variables exported before submission.
    batch_script : pathlib.Path, optional
        Optional wrapper script used when batch_mode is enabled.
    submit_dir : pathlib.Path, optional
        Working directory used when submitting (defaults to ``workdir``).
    """

    workdir: Path
    script_rel: str = "SLURMM-run"
    finished_name: str = "FINISHED"
    failed_name: str = "FAILED"
    name: Optional[str] = None
    stage: Optional[str] = None
    body_rel: Optional[str] = None
    header_name: Optional[str] = None
    header_template: Optional[Path] = None
    header_root: Optional[Path] = None
    extra_sbatch: Sequence[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    batch_script: Path | None = None
    submit_dir: Path | None = None

    # allow a few common variants (case, alt names)
    alt_script_names: Sequence[str] = (
        "SLURMM-run",
        "SLURMM-Run",
        "slurmm-run",
        "run.sh",
    )

    # --- absolute paths for checks/sentinels ---
    def finished_path(self) -> Path:
        """Sentinel path signalling successful completion."""
        return self.workdir / self.finished_name

    def failed_path(self) -> Path:
        """Sentinel path signalling failure."""
        return self.workdir / self.failed_name

    def jobid_path(self) -> Path:
        """Path containing the most recent Slurm job identifier."""
        return self.workdir / "JOBID"

    def resolve_script_abs(self) -> Path:
        """Return the absolute path to the submission script."""
        preferred = self.workdir / self.script_rel
        candidates = [preferred] + [
            self.workdir / n for n in self.alt_script_names if n != self.script_rel
        ]
        for p in candidates:
            if p.exists():
                return p
        return preferred

    def script_arg(self) -> str:
        """Return the workdir-relative script argument for ``sbatch``."""
        base = self.submit_dir or self.workdir
        candidate = base / self.script_rel
        abs_script = candidate if candidate.exists() else self.resolve_script_abs()
        try:
            return str(abs_script.relative_to(base))
        except ValueError:
            return abs_script.name


# ---- Manager ----
class SlurmJobManager:
    """Submit, monitor, and resubmit Slurm jobs for BATTER executions."""

    def __init__(
        self,
        poll_s: float = 20.0,
        max_retries: int = 3,
        resubmit_backoff_s: float = 30.0,
        registry_file: Optional[Path] = None,
        dry_run: bool = False,
        sbatch_flags: Optional[Sequence[str]] = None,
        submit_retry_limit: int = 3,
        submit_retry_delay_s: float = 60.0,
        max_active_jobs: Optional[int] = None,
        batch_mode: bool = False,
        batch_gpus: Optional[int] = None,
        gpus_per_task: int = 1,
        srun_extra: Optional[Sequence[str]] = None,
        stage: Optional[str] = None,
        header_root: Optional[Path] = None,
        partition: Optional[str] = None,
    ):
        """Initialise the manager.

        Parameters
        ----------
        poll_s : float, optional
            Polling interval in seconds.
        max_retries : int, optional
            Maximum number of automatic resubmissions per job.
        resubmit_backoff_s : float, optional
            Delay between a failed job and an attempted resubmission.
        registry_file : pathlib.Path, optional
            Optional JSONL file acting as a persistent queue shared across
            processes.
        dry_run : bool, optional
            When ``True`` do not submit jobs; only mark that a submission
            would occur.
        sbatch_flags : Sequence[str], optional
            Global ``sbatch`` flags appended to every submission.
        submit_retry_limit : int, optional
            Number of submission retries on failure per job.
        submit_retry_delay_s : float, optional
            Delay between submission retries.
        max_active_jobs : int, optional
            Maximum number of active jobs allowed for the user. When ``None``,
            no limit is enforced.
        batch_mode : bool, optional
            When ``True``, batch scripts (e.g., SLURMM-BATCH) may be supplied via
            ``batch_script``/``submit_dir`` on the job specs.
        batch_gpus : int, optional
            Reserved for future inline execution modes.
        gpus_per_task : int, optional
            Reserved for future inline execution modes.
        srun_extra : Sequence[str], optional
            Reserved for future inline execution modes.
        """
        self.poll_s = float(poll_s)
        self.max_retries = int(max_retries)
        self.resubmit_backoff_s = float(resubmit_backoff_s)
        self.dry_run = dry_run
        self.triggered = False
        self.submit_retry_limit = max(0, int(submit_retry_limit))
        self.submit_retry_delay_s = float(submit_retry_delay_s)
        self.max_active_jobs = (
            max_active_jobs if max_active_jobs is None else int(max_active_jobs)
        )
        if self.max_active_jobs is not None and self.max_active_jobs <= 0:
            raise ValueError("max_active_jobs must be positive or None")
        self.partition = partition

        self.sbatch_flags: List[str] = list(sbatch_flags or [])

        # Central registry: in-memory accumulation (per-process)
        self._inmem_specs: Dict[Path, SlurmJobSpec] = {}
        # Optional on-disk queue for cross-process accumulation
        self._registry_file = registry_file
        # retry accounting (by workdir)
        self._retries: Dict[Path, int] = {}
        self._submitted_job_ids: set[str] = set()
        self.n_active: int = 0
        self._stage: Optional[str] = stage
        self._header_root = header_root
        self.batch_mode = bool(batch_mode)
        self.batch_gpus = (
            None if batch_gpus is None or int(batch_gpus) <= 0 else int(batch_gpus)
        )
        self.gpus_per_task = max(1, int(gpus_per_task))
        self.srun_extra: List[str] = list(srun_extra or [])

    def set_stage(self, stage: Optional[str]) -> None:
        """Limit registry loading to ``stage`` and default new specs to this stage."""
        self._stage = stage

    def _stage_matches(self, stage: Optional[str], workdir: Path | None = None) -> bool:
        """Return ``True`` if ``stage`` is compatible with the manager's active stage."""
        if not self._stage:
            return True
        if stage:
            return stage == self._stage
        # Best-effort inference for legacy entries without stage metadata
        inferred = _infer_stage_from_workdir(workdir) if workdir else None
        return inferred == self._stage

    def _filter_stage(self, specs: Dict[Path, SlurmJobSpec]) -> Dict[Path, SlurmJobSpec]:
        """Filter ``specs`` to the active stage (if set)."""
        if not self._stage:
            return specs
        return {
            wd: spec for wd, spec in specs.items() if self._stage_matches(spec.stage, wd)
        }

    def _resolve_header_root(self, spec: SlurmJobSpec) -> Path:
        root = spec.header_root or self._header_root
        if not root:
            env_root = os.environ.get("BATTER_SLURM_HEADER_DIR")
            if env_root:
                return Path(env_root)
        return Path(root) if root else Path.home() / ".batter"

    def _rebuild_script_with_header(self, spec: SlurmJobSpec, script_abs: Path) -> None:
        """Rebuild the submission script by prepending a header to the stored body, if present."""
        body_path = spec.workdir / spec.body_rel if spec.body_rel else script_abs
        if not body_path.exists():
            candidate = script_abs.with_suffix(script_abs.suffix + ".body")
            if candidate.exists():
                body_path = candidate
            else:
                return

        try:
            body_text = body_path.read_text()
            # drop any baked-in SBATCH lines from the body
            body_lines = [
                ln for ln in body_text.splitlines() if not ln.lstrip().startswith("#SBATCH")
            ]
            body_text = "\n".join(body_lines)
        except Exception as exc:
            logger.warning(f"[SLURM] Failed to read body {body_path}: {exc}")
            return

        header_root = self._resolve_header_root(spec)
        header_text = ""
        if spec.header_name:
            user_header = header_root / spec.header_name
            if user_header.exists():
                try:
                    header_text = user_header.read_text()
                except Exception as exc:
                    logger.warning(f"[SLURM] Failed to read header {user_header}: {exc}")
            elif spec.header_template and spec.header_template.exists():
                try:
                    header_text = spec.header_template.read_text()
                except Exception:
                    header_text = ""

        combined = header_text
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += body_text

        try:
            script_abs.write_text(combined)
        except Exception as exc:
            logger.warning(f"[SLURM] Could not write rebuilt script {script_abs}: {exc}")

    # ---------- Registry API ----------
    def add(self, spec: SlurmJobSpec) -> None:
        """Queue ``spec`` for later submission.

        Parameters
        ----------
        spec : SlurmJobSpec
            Job specification to store. Persisted to ``registry_file`` when
            configured.
        """
        if self.dry_run:
            self.triggered = True
            return

        if spec.stage is None and self._stage is not None:
            spec.stage = self._stage

        self._inmem_specs[spec.workdir] = spec

        if self._registry_file is not None:
            rec = {
                "workdir": str(spec.workdir),
                "script_rel": spec.script_rel,
                "finished_name": spec.finished_name,
                "failed_name": spec.failed_name,
                "name": spec.name,
                "stage": spec.stage,
                "body_rel": spec.body_rel,
                "header_name": spec.header_name,
                "header_template": str(spec.header_template) if spec.header_template else None,
                "header_root": str(spec.header_root) if spec.header_root else None,
                "extra_sbatch": list(spec.extra_sbatch or []),
                "extra_env": dict(getattr(spec, "extra_env", {}) or {}),
                "batch_script": str(spec.batch_script) if spec.batch_script else None,
                "submit_dir": str(spec.submit_dir) if spec.submit_dir else None,
            }
            _atomic_append_jsonl_unique(self._registry_file, rec, unique_key="workdir")

    def wait_for_slot(
        self,
        poll_s: float | None = None,
        user: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> None:
        """Block until the number of active jobs drops below ``max_active_jobs``.

        Parameters
        ----------
        poll_s : float, optional
            Polling interval in seconds. Defaults to :attr:`poll_s`.
        user : str, optional
            User name to query in ``squeue``. Defaults to ``$USER``.
        """
        if self.max_active_jobs is None:
            return
        max_active = self.max_active_jobs
        interval = self.poll_s if poll_s is None else poll_s
        target_partition = partition or self.partition
        while True:
            n_active = _num_active_job(user=user, partition=target_partition)
            self.n_active = n_active
            if n_active < max_active:
                if n_active > 0:
                    logger.debug(
                        f"[SLURM_mgr] outstanding={n_active} < cap={max_active}, submitting"
                    )
                break
            logger.warning(
                f"[SLURM_mgr] outstanding={n_active} ≥ cap={max_active} — waiting {interval}s",
            )
            time.sleep(interval)

    def _load_registry_specs(self) -> Dict[Path, SlurmJobSpec]:
        """Load job specifications from the persistent registry."""
        out: Dict[Path, SlurmJobSpec] = {}
        if not self._registry_file or not self._registry_file.exists():
            return out
        with open(self._registry_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                wd = Path(rec["workdir"])
                stage = rec.get("stage")
                if not self._stage_matches(stage, wd):
                    continue
                out[wd] = SlurmJobSpec(
                    workdir=wd,
                    script_rel=rec.get("script_rel", "SLURMM-run"),
                    finished_name=rec.get("finished_name", "FINISHED"),
                    failed_name=rec.get("failed_name", "FAILED"),
                    name=rec.get("name"),
                    stage=stage,
                    body_rel=rec.get("body_rel"),
                    header_name=rec.get("header_name"),
                    header_template=Path(rec["header_template"]) if rec.get("header_template") else None,
                    header_root=Path(rec["header_root"]) if rec.get("header_root") else None,
                    extra_sbatch=rec.get("extra_sbatch") or [],
                    extra_env=rec.get("extra_env") or {},
                    batch_script=Path(rec["batch_script"]) if rec.get("batch_script") else None,
                    submit_dir=Path(rec["submit_dir"]) if rec.get("submit_dir") else None,
                )
        return out

    def jobs(self) -> List[SlurmJobSpec]:
        """Return the union of in-memory and on-disk queued specs (dedup by workdir)."""
        merged: Dict[Path, SlurmJobSpec] = self._load_registry_specs()
        merged.update(self._inmem_specs)
        return list(self._filter_stage(merged).values())

    def clear(self) -> None:
        """Clear in-memory specs, retry bookkeeping, and remove the on-disk queue if present."""
        self._inmem_specs.clear()
        self._retries.clear()
        if self._registry_file and self._registry_file.exists():
            try:
                self._registry_file.unlink()
            except Exception:
                pass

    # ---------- Core ops ----------
    def _submit(self, spec: SlurmJobSpec) -> str:
        """Submit ``spec`` via ``sbatch`` retrying on failure."""
        attempts = 0
        while True:
            try:
                return self._submit_once(spec)
            except Exception as exc:
                if self.submit_retry_limit == 0 or attempts >= self.submit_retry_limit:
                    raise RuntimeError(
                        f"SLURM submission failed for {spec.workdir} after {attempts + 1} attempt(s) "
                        f"due to: {exc}"
                    )
                attempts += 1
                delay = self.submit_retry_delay_s
                logger.warning(
                    f"[SLURM] submission attempt {attempts}/{self.submit_retry_limit} "
                    f"failed for {spec.workdir.name}: {exc}; retrying in {delay:.0f}s"
                )
                time.sleep(delay)

    def _submit_once(self, spec: SlurmJobSpec) -> str:
        """Submit ``spec`` via ``sbatch`` and persist the resulting job id (single attempt)."""
        # resolve script path (allow separate submission directory)
        if spec.submit_dir:
            candidate = Path(spec.submit_dir) / spec.script_rel
            script_abs = candidate if candidate.exists() else spec.resolve_script_abs()
        else:
            script_abs = spec.resolve_script_abs()

        # If a body is present, rebuild the script with the current header
        self._rebuild_script_with_header(spec, script_abs)

        if not script_abs.exists():
            listing = (
                ", ".join(sorted(p.name for p in spec.workdir.iterdir()))
                if spec.workdir.exists()
                else "(missing workdir)"
            )
            raise FileNotFoundError(
                f"SLURM script not found: {script_abs}\n"
                f"in workdir: {spec.workdir}\n"
                f"contents: {listing}"
            )

        try:
            script_abs.chmod(script_abs.stat().st_mode | 0o111)
        except Exception:
            pass

        # base + global flags + per-job flags
        cmd: List[str] = ["sbatch"]
        if self.sbatch_flags:
            cmd += self.sbatch_flags  # global flags first
        if spec.name:
            cmd += ["--job-name", spec.name]
        if spec.extra_sbatch:
            cmd += list(spec.extra_sbatch)  # job-specific flags after

        if spec.extra_env:
            kv = [f"{k}={v}" for k, v in spec.extra_env.items()]
            cmd += ["--export", "ALL," + ",".join(kv)]

        cmd.append(spec.script_arg())

        submit_cwd = spec.submit_dir or spec.workdir

        if self.dry_run:
            logger.info(f"[DRY-RUN] sbatch (cwd={submit_cwd}): {' '.join(cmd)}")
            # fabricate a dummy JOBID to keep downstream logic harmless
            _write_text(spec.jobid_path(), "0\n")
            return "0"

        logger.debug(f"[SLURM] sbatch: {' '.join(cmd)} (cwd={submit_cwd})")
        proc = subprocess.run(
            cmd,
            cwd=submit_cwd,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            raise RuntimeError(
                f"sbatch returned {proc.returncode}; stdout={stdout!r} stderr={stderr!r}"
            )
        out = proc.stdout.strip()
        m = JOBID_RE.search(out)
        if not m:
            raise RuntimeError(f"Could not parse sbatch output: {out}")
        jobid = m.group(1)
        _write_text(spec.jobid_path(), f"{jobid}\n")
        self._submitted_job_ids.add(jobid)
        self.n_active += 1
        logger.debug(f"[SLURM] submitted {spec.workdir.name} → job {jobid} #{self.n_active} active")
        return jobid

    def _status(self, spec: SlurmJobSpec) -> Tuple[bool, Optional[str]]:
        """Return ``(done, status)`` tuple for ``spec``."""
        if spec.finished_path().exists():
            return True, "FINISHED"
        if spec.failed_path().exists():
            return True, "FAILED"
        jobid = _read_text(spec.jobid_path())
        return False, _slurm_state(jobid)

    # ---------- Compatibility one-off API ----------
    def ensure_running(self, spec: SlurmJobSpec) -> None:
        """Ensure the given spec is submitted or already active/done (does not register)."""
        done, status = self._status(spec)
        if done:
            logger.debug(
                f"[SLURM] {spec.workdir.name}: already {status}; not submitting"
            )
            return
        if self.dry_run:
            self.triggered = True
            return
        state = _slurm_state(_read_text(spec.jobid_path()))
        if state in SLURM_OK_STATES:
            logger.debug(
                f"[SLURM] {spec.workdir.name}: active ({state}); not submitting"
            )
            return
        self._submit(spec)

    def wait_until_done(self, specs: Iterable[SlurmJobSpec]) -> None:
        """Submit if needed and watch the given set until done/fail (legacy interface)."""
        if self.dry_run:
            self.triggered = True
            return
        self._wait_loop(list(specs))

    # ---------- Global wait ----------
    def wait_all(self) -> None:
        """Submit/monitor all registered jobs together and block until completion."""
        specs_map = self._load_registry_specs()
        specs_map.update(self._inmem_specs)
        specs_map = self._filter_stage(specs_map)
        if not specs_map and not self.dry_run:
            logger.debug("[SLURM] wait_all: nothing to monitor.")
            return
        elif self.dry_run:
            self.triggered = True
            return
        specs = list(specs_map.values())
        self._wait_loop(specs)
        # clear registry for next phase
        self.clear()

    # ---------- Shared wait logic ----------
    def _wait_loop(self, specs: List[SlurmJobSpec]) -> None:
        """Internal polling loop shared by :meth:`wait_until_done` and :meth:`wait_all`."""
        # optional progress bar (tqdm)
        try:
            from tqdm import tqdm  # type: ignore

            use_tqdm = True
        except Exception:
            tqdm = None  # type: ignore
            use_tqdm = False

        # Initial submissions for provided specs only (do not re-submit already active jobs)
        self.wait_for_slot()
        for s in (
            tqdm(specs, desc="SLURM submissions", leave=True, dynamic_ncols=True)
            if use_tqdm
            else specs
        ):
            if (
                self.max_active_jobs is not None
                and self.n_active >= self.max_active_jobs
            ):
                logger.info(
                    f"[SLURM] reached max_active_jobs={self.max_active_jobs}; "
                    f"deferring further submissions"
                )
                self.wait_for_slot()
            try:
                self.ensure_running(s)
            except Exception as e:
                logger.error(f"[SLURM] submit failed for {s.workdir}: {e}")
                raise

        pending = {s.workdir: s for s in specs}
        retries = {s.workdir: self._retries.get(s.workdir, 0) for s in specs}

        total = len(specs)
        completed: set[Path] = set()
        last_log = 0.0
        pbar = (
            tqdm(total=total, desc="SLURM jobs", leave=True, dynamic_ncols=True)
            if use_tqdm
            else None
        )

        while pending:
            done_now: List[Path] = []

            # quick counts for progress display
            running_cnt = 0
            resub_cnt = 0
            failed_cnt = 0

            for wd, s in list(pending.items()):
                done, status = self._status(s)
                if done:
                    if status == "FAILED":
                        failed_cnt += 1
                    done_now.append(wd)
                    continue

                jobid = _read_text(s.jobid_path())
                state = _slurm_state(jobid)

                if state in SLURM_OK_STATES:
                    running_cnt += 1
                    continue

                # job missing or ended without sentinel → resubmit
                resub_reason = state or "MISSING"
                timeout_state = state == "TIMEOUT"
                completed_state = state == "COMPLETED"

                if state in SLURM_FINAL_BAD:
                    if timeout_state:
                        logger.debug(
                            f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} hit TIMEOUT; "
                            "resubmitting without counting as failure"
                        )
                    else:
                        logger.warning(
                            f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} reached "
                            f"state={state}; attempting resubmit"
                        )
                elif completed_state:
                    logger.debug(
                        f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} completed without FINISHED; "
                        "resubmitting without counting against retries"
                    )

                r = retries[wd]
                if (
                    not timeout_state
                    and not completed_state
                    and r >= self.max_retries
                ):
                    logger.error(
                        f"[SLURM] {wd.name}: exceeded max_retries={self.max_retries} "
                        f"(state={resub_reason}); marking FAILED"
                    )
                    s.failed_path().touch()
                    failed_cnt += 1
                    done_now.append(wd)
                    continue

                resub_cnt += 1
                if timeout_state:
                    logger.debug(
                        f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} state=TIMEOUT; "
                        "resubmitting (timeout retries are unlimited)"
                    )
                elif not completed_state:
                    logger.warning(
                        f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} "
                        f"state={resub_reason}; resubmitting ({r + 1}/{self.max_retries})"
                    )
                time.sleep(self.resubmit_backoff_s)
                try:
                    self._submit(s)
                    if not timeout_state and not completed_state:
                        retries[wd] = r + 1
                        self._retries[wd] = retries[wd]  # keep central book
                except Exception as e:
                    logger.error(f"[SLURM] {wd.name}: resubmit failed: {e}")
                    raise

            # remove finished from pending and update progress
            for wd in done_now:
                pending.pop(wd, None)
                if wd not in completed:
                    completed.add(wd)
                    if pbar:
                        pbar.update(1)

            # render progress info
            if pbar:
                pbar.set_postfix(
                    {
                        "running": running_cnt,
                        # "resub": resub_cnt,
                        "failed": failed_cnt,
                        # "pending": len(pending),
                    }
                )
            else:
                # fallback: log a compact status every ~30s
                now = time.time()
                if now - last_log > 30 or not pending:
                    logger.info(
                        f"[SLURM] progress {len(completed)}/{total} "
                        f"(running={running_cnt}, resub={resub_cnt}, "
                        f"failed={failed_cnt}, pending={len(pending)})"
                    )
                    last_log = now

            if pending:
                time.sleep(self.poll_s)

        if pbar:
            pbar.close()
        logger.info("[SLURM] All jobs complete.")
