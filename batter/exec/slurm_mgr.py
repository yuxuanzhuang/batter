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


def _num_active_job(user: Optional[str] = None) -> int:
    """Return the number of active Slurm jobs for ``user``.

    Parameters
    ----------
    user : str, optional
        Unix user name. If ``None``, defaults to ``$USER``.

    Returns
    -------
    int
        Number of jobs currently reported by ``squeue`` for the user.
    """
    user = user or os.environ.get("USER")
    if not user:
        return 0
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-u", user, "-o", "%i"],
            text=True,
        )
    except Exception:
        return 0

    n_ids = [line.strip() for line in out.splitlines() if line.strip()]
    logger.debug(f"[SQUEUE] active jobs for user '{user}': {n_ids}")
    return len(n_ids)


def _slurm_state(jobid: Optional[str]) -> Optional[str]:
    """Return the best-effort Slurm state for ``jobid``."""
    if not jobid:
        return None
    return _state_from_squeue(jobid) or _state_from_sacct(jobid)


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
    """

    workdir: Path
    script_rel: str = "SLURMM-run"
    finished_name: str = "FINISHED"
    failed_name: str = "FAILED"
    name: Optional[str] = None
    extra_sbatch: Sequence[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)

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
        abs_script = self.resolve_script_abs()
        try:
            return str(abs_script.relative_to(self.workdir))
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

        self.sbatch_flags: List[str] = list(sbatch_flags or [])

        # Central registry: in-memory accumulation (per-process)
        self._inmem_specs: Dict[Path, SlurmJobSpec] = {}
        # Optional on-disk queue for cross-process accumulation
        self._registry_file = registry_file
        # retry accounting (by workdir)
        self._retries: Dict[Path, int] = {}
        self._submitted_job_ids: set[str] = set()
        self.n_active: int = 0

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

        self._inmem_specs[spec.workdir] = spec

        if self._registry_file is not None:
            rec = {
                "workdir": str(spec.workdir),
                "script_rel": spec.script_rel,
                "finished_name": spec.finished_name,
                "failed_name": spec.failed_name,
                "name": spec.name,
                "extra_sbatch": list(spec.extra_sbatch or []),
                "extra_env": dict(getattr(spec, "extra_env", {}) or {}),
            }
            _atomic_append_jsonl_unique(self._registry_file, rec, unique_key="workdir")

    def wait_for_slot(
        self,
        poll_s: float | None = None,
        user: Optional[str] = None,
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
        while True:
            n_active = _num_active_job(user=user)
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
                out[wd] = SlurmJobSpec(
                    workdir=wd,
                    script_rel=rec.get("script_rel", "SLURMM-run"),
                    finished_name=rec.get("finished_name", "FINISHED"),
                    failed_name=rec.get("failed_name", "FAILED"),
                    name=rec.get("name"),
                    extra_sbatch=rec.get("extra_sbatch") or [],
                    extra_env=rec.get("extra_env") or {},
                )
        return out

    def jobs(self) -> List[SlurmJobSpec]:
        """Return the union of in-memory and on-disk queued specs (dedup by workdir)."""
        merged: Dict[Path, SlurmJobSpec] = self._load_registry_specs()
        merged.update(self._inmem_specs)
        return list(merged.values())

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
        script_abs = spec.resolve_script_abs()
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

        if self.dry_run:
            logger.info(f"[DRY-RUN] sbatch (cwd={spec.workdir}): {' '.join(cmd)}")
            # fabricate a dummy JOBID to keep downstream logic harmless
            _write_text(spec.jobid_path(), "0\n")
            return "0"

        logger.debug(f"[SLURM] sbatch: {' '.join(cmd)} (cwd={spec.workdir})")
        proc = subprocess.run(
            cmd,
            cwd=spec.workdir,
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
        logger.debug(f"[SLURM] submitted {spec.workdir.name} → job {jobid}")
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
        if not specs_map and not self.dry_run:
            logger.debug("[SLURM] wait_all: nothing to monitor.")
            return
        elif self.dry_run:
            self.triggered = True
            return
        self._wait_loop(list(specs_map.values()))
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

                r = retries[wd]
                if not timeout_state and r >= self.max_retries:
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
                else:
                    logger.warning(
                        f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} "
                        f"state={resub_reason}; resubmitting ({r + 1}/{self.max_retries})"
                    )
                time.sleep(self.resubmit_backoff_s)
                try:
                    self._submit(s)
                    if not timeout_state:
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
