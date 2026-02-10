from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast
from loguru import logger

__all__ = [
    "SlurmJobSpec",
    "SlurmJobManager",
]


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


# ---------- atomic registry append ----------
def _atomic_append_jsonl_unique(path: Path, rec: dict, unique_key: str = "workdir") -> None:
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

    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
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
                    return

            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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


def _states_from_squeue(jobids: Sequence[str]) -> Dict[str, str]:
    """Return ``{jobid: state}`` for jobids currently visible in squeue.

    Parameters
    ----------
    jobids : Sequence[str]
        Job IDs to query.

    Returns
    -------
    dict
        Mapping from jobid to state (e.g. RUNNING, PENDING).
    """
    ids = [j for j in jobids if j and j.isdigit()]
    if not ids:
        return {}

    cmd = ["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}

    states: Dict[str, str] = {}
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) >= 2:
            jid, st = parts[0], parts[1].split()[0]
            states[jid] = st
    return states


def _states_from_sacct(jobids: Sequence[str]) -> Dict[str, str]:
    """Return ``{jobid: state}`` using sacct for completed/missing jobs.

    Notes
    -----
    We request JobIDRaw so we can match child steps back to the root job.

    Parameters
    ----------
    jobids : Sequence[str]
        Job IDs to query.

    Returns
    -------
    dict
        Mapping from jobid to state.
    """
    ids = [j for j in jobids if j and j.isdigit()]
    if not ids:
        return {}

    cmd = ["sacct", "-j", ",".join(ids), "-X", "-n", "-o", "JobIDRaw,State"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}

    states: Dict[str, str] = {}
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 2:
            jid = parts[0]
            st = parts[1].split()[0]
            # keep root state if already set
            states.setdefault(jid, st)
    return states


def _num_active_job(user: Optional[str] = None, partition: Optional[str] = None) -> int:
    """Return the number of active Slurm jobs for ``user`` (optionally in a partition).

    Parameters
    ----------
    user : str, optional
        Unix username (defaults to ``$USER``).
    partition : str, optional
        Partition/queue name to filter with ``squeue -p``.

    Returns
    -------
    int
        Number of job IDs currently reported by ``squeue``.
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


def _infer_stage_from_workdir(path: Path | None) -> Optional[str]:
    """Heuristically infer a stage name from ``path`` for legacy entries.

    Parameters
    ----------
    path : pathlib.Path or None
        Work directory.

    Returns
    -------
    str or None
        Stage name such as ``equil``, ``fe_equil``, ``fe`` or ``None`` if unknown.
    """
    if not path:
        return None
    parts = [p.lower() for p in path.parts]
    if "equil" in parts:
        return "equil"
    if "fe" in parts:
        if path.name.endswith("-1"):
            return "fe_equil"
        return "fe"
    return None


@dataclass
class SlurmJobSpec:
    """Descriptor for a Slurm job managed by :class:`SlurmJobManager`.

    Parameters
    ----------
    workdir : pathlib.Path
        Working directory containing submission scripts and sentinel files.
    script_rel : str, optional
        Preferred relative submission script path.
    finished_name : str, optional
        Sentinel file name indicating success.
    failed_name : str, optional
        Sentinel file name indicating failure.
    name : str, optional
        Friendly display name.
    stage : str, optional
        Logical stage used for registry filtering.
    extra_sbatch : Sequence[str], optional
        Extra ``sbatch`` flags (job-specific).
    extra_env : dict, optional
        Extra environment variables to export (job-specific).
    submit_dir : pathlib.Path, optional
        Directory to submit from (defaults to ``workdir``).

    Notes
    -----
    The remaining fields are legacy compatibility fields used by older BATTER
    versions and/or existing registry entries. The manager may ignore them.
    """

    workdir: Path
    script_rel: str = "SLURMM-run"
    finished_name: str = "FINISHED"
    failed_name: str = "FAILED"
    name: Optional[str] = None
    stage: Optional[str] = None

    # legacy / compatibility fields
    body_rel: Optional[str] = None
    header_name: Optional[str] = None
    header_template: Optional[Path] = None
    header_root: Optional[Path] = None
    batch_script: Path | None = None

    extra_sbatch: Sequence[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    submit_dir: Path | None = None

    alt_script_names: Sequence[str] = (
        "SLURMM-run",
        "SLURMM-Run",
        "slurmm-run",
        "run.sh",
    )

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
        """Return the absolute path to the submission script.

        Returns
        -------
        pathlib.Path
            Existing script path if found, otherwise the preferred path.
        """
        preferred = self.workdir / self.script_rel
        candidates = [preferred] + [
            self.workdir / n for n in self.alt_script_names if n != self.script_rel
        ]
        for p in candidates:
            if p.exists():
                return p
        return preferred

    def script_arg(self) -> str:
        """Return the submission-script path argument for ``sbatch``.

        Returns
        -------
        str
            Script path relative to ``submit_dir`` when possible.
        """
        base = self.submit_dir or self.workdir
        candidate = base / self.script_rel
        abs_script = candidate if candidate.exists() else self.resolve_script_abs()
        try:
            return str(abs_script.relative_to(base))
        except ValueError:
            return abs_script.name


class SlurmJobManager:
    """Submit, monitor, and resubmit Slurm jobs for BATTER executions.

    Parameters
    ----------
    poll_s : float, optional
        Poll interval (seconds) between status checks.
    max_retries : int, optional
        Maximum automatic resubmissions per workdir (excluding TIMEOUT and COMPLETED-without-sentinel).
    resubmit_backoff_s : float, optional
        Sleep before resubmitting a job after detecting termination/missing state.
    registry_file : pathlib.Path, optional
        JSONL queue file for cross-process coordination.
    dry_run : bool, optional
        When True, do not submit; record that submission would occur.
    sbatch_flags : Sequence[str], optional
        Global sbatch flags appended to every submission.
    submit_retry_limit : int, optional
        Number of retries for the *submission command* itself.
    submit_retry_delay_s : float, optional
        Delay between submission retries.
    max_active_jobs : int, optional
        Cap on concurrent jobs for the user (checked via one `squeue -u` call).
    partition : str, optional
        Partition filter used by max_active_jobs checks.

    Other Parameters
    ----------------
    batch_mode, batch_gpus, gpus_per_task, srun_extra, stage, header_root
        Accepted for compatibility with older code paths. This manager does not
        implement batch execution; values are stored/ignored.
    **_ignored
        Extra kwargs are accepted and ignored for compatibility.
    """

    def __init__(
        self,
        poll_s: float = 60.0,
        max_retries: int = 3,
        resubmit_backoff_s: float = 30.0,
        registry_file: Optional[Path] = None,
        dry_run: bool = False,
        sbatch_flags: Optional[Sequence[str]] = None,
        submit_retry_limit: int = 3,
        submit_retry_delay_s: float = 60.0,
        max_active_jobs: Optional[int] = None,
        partition: Optional[str] = None,
        # --- compatibility kwargs ---
        batch_mode: bool = False,
        batch_gpus: Optional[int] = None,
        gpus_per_task: int = 1,
        srun_extra: Optional[Sequence[str]] = None,
        stage: Optional[str] = None,
        header_root: Optional[Path] = None,
        **_ignored: Any,
    ) -> None:
        self.poll_s = float(poll_s)
        self.max_retries = int(max_retries)
        self.resubmit_backoff_s = float(resubmit_backoff_s)
        self._registry_file = registry_file
        self.dry_run = bool(dry_run)
        self.triggered = False

        self.sbatch_flags: List[str] = list(sbatch_flags or [])
        self.submit_retry_limit = max(0, int(submit_retry_limit))
        self.submit_retry_delay_s = float(submit_retry_delay_s)

        self.max_active_jobs = None if max_active_jobs is None else int(max_active_jobs)
        if self.max_active_jobs is not None and self.max_active_jobs <= 0:
            raise ValueError("max_active_jobs must be positive or None")
        self.partition = partition

        # compatibility settings (stored; not implemented)
        self.batch_mode = bool(batch_mode)
        self.batch_gpus = None if batch_gpus is None else int(batch_gpus)
        self.gpus_per_task = max(1, int(gpus_per_task))
        self.srun_extra: List[str] = list(srun_extra or [])
        self._header_root = header_root

        # stage filtering
        self._stage: Optional[str] = stage

        # in-memory queue
        self._inmem_specs: Dict[Path, SlurmJobSpec] = {}
        self._retries: Dict[Path, int] = {}
        self._submitted_job_ids: set[str] = set()
        self.n_active: int = 0

    # ---------- stage API ----------
    def set_stage(self, stage: Optional[str]) -> None:
        """Set the active stage filter for registry loading/submission.

        Parameters
        ----------
        stage : str or None
            Stage key such as ``equil``, ``fe_equil``, ``fe``, etc.
            If None, stage filtering is disabled.
        """
        self._stage = stage

    def _stage_matches(self, stage: Optional[str], workdir: Path | None = None) -> bool:
        """Return True if a spec with ``stage`` should be managed under current stage."""
        if not self._stage:
            return True
        if stage:
            return stage == self._stage
        inferred = _infer_stage_from_workdir(workdir)
        return inferred == self._stage

    # ---------- registry API ----------
    def add(self, spec: SlurmJobSpec) -> None:
        """Queue ``spec`` for later submission and optionally persist to registry.

        Parameters
        ----------
        spec : SlurmJobSpec
            Job specification.
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
                # legacy fields preserved
                "body_rel": spec.body_rel,
                "header_name": spec.header_name,
                "header_template": str(spec.header_template) if spec.header_template else None,
                "header_root": str(spec.header_root) if spec.header_root else None,
                "extra_sbatch": list(spec.extra_sbatch or []),
                "extra_env": dict(spec.extra_env or {}),
                "batch_script": str(spec.batch_script) if spec.batch_script else None,
                "submit_dir": str(spec.submit_dir) if spec.submit_dir else None,
            }
            _atomic_append_jsonl_unique(self._registry_file, rec, unique_key="workdir")

    def _load_registry_specs(self) -> Dict[Path, SlurmJobSpec]:
        """Load job specs from the persistent JSONL registry (best-effort)."""
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

                wd = Path(rec.get("workdir", ""))
                if not wd:
                    continue

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
        """Return merged in-memory + registry specs (dedup by workdir)."""
        merged = self._load_registry_specs()
        merged.update(self._inmem_specs)
        if self._stage:
            merged = {
                wd: sp for wd, sp in merged.items() if self._stage_matches(sp.stage, wd)
            }
        return list(merged.values())

    def clear(self) -> None:
        """Clear in-memory queue/retry book and remove on-disk registry if present."""
        self._inmem_specs.clear()
        self._retries.clear()
        if self._registry_file and self._registry_file.exists():
            try:
                self._registry_file.unlink()
            except Exception:
                pass

    # ---------- throttling ----------
    def wait_for_slot(
        self,
        poll_s: float | None = None,
        user: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> None:
        """Block until active jobs drop below ``max_active_jobs``.

        Parameters
        ----------
        poll_s : float, optional
            Polling interval in seconds (defaults to :attr:`poll_s`).
        user : str, optional
            Unix username (defaults to ``$USER``).
        partition : str, optional
            Partition to filter on (defaults to manager partition).
        """
        if self.max_active_jobs is None:
            return

        max_active = self.max_active_jobs
        interval = self.poll_s if poll_s is None else float(poll_s)
        target_partition = partition or self.partition

        while True:
            n_active = _num_active_job(user=user, partition=target_partition)
            self.n_active = n_active
            if n_active < max_active:
                if n_active > 0:
                    logger.debug(
                        f"[SLURM_mgr] outstanding={n_active} < cap={max_active}, submitting"
                    )
                return
            logger.warning(
                f"[SLURM_mgr] outstanding={n_active} ≥ cap={max_active} — waiting {interval}s"
            )
            time.sleep(interval)

    # ---------- submission ----------
    def _submit(self, spec: SlurmJobSpec) -> str:
        """Submit a job via ``sbatch`` with retry-on-submission-failure."""
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
        """Single-attempt sbatch submit; persist JOBID."""
        # resolve script path (allow separate submission dir)
        if spec.submit_dir:
            candidate = Path(spec.submit_dir) / spec.script_rel
            script_abs = candidate if candidate.exists() else spec.resolve_script_abs()
        else:
            script_abs = spec.resolve_script_abs()

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

        cmd: List[str] = ["sbatch"]
        if self.sbatch_flags:
            cmd += self.sbatch_flags
        if spec.name:
            cmd += ["--job-name", spec.name]
        if spec.extra_sbatch:
            cmd += list(spec.extra_sbatch)

        if spec.extra_env:
            kv = [f"{k}={v}" for k, v in spec.extra_env.items()]
            cmd += ["--export", "ALL," + ",".join(kv)]

        cmd.append(spec.script_arg())
        submit_cwd = spec.submit_dir or spec.workdir

        if self.dry_run:
            logger.info(f"[DRY-RUN] sbatch (cwd={submit_cwd}): {' '.join(cmd)}")
            self.triggered = True
            _write_text(spec.jobid_path(), "0\n")
            return "0"

        logger.debug(f"[SLURM] sbatch: {' '.join(cmd)} (cwd={submit_cwd})")
        proc = subprocess.run(cmd, cwd=submit_cwd, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch returned {proc.returncode}; stdout={proc.stdout.strip()!r} "
                f"stderr={proc.stderr.strip()!r}"
            )

        out = proc.stdout.strip()
        m = JOBID_RE.search(out)
        if not m:
            raise RuntimeError(f"Could not parse sbatch output: {out}")
        jobid = m.group(1)

        _write_text(spec.jobid_path(), f"{jobid}\n")
        self._submitted_job_ids.add(jobid)
        self.n_active += 1
        logger.debug(f"[SLURM] submitted {spec.workdir.name} → job {jobid} (active≈{self.n_active})")
        return jobid

    # ---------- status ----------
    def _sentinel_done(self, spec: SlurmJobSpec) -> Tuple[bool, Optional[str]]:
        """Check FINISHED/FAILED sentinels only (no Slurm calls)."""
        if spec.finished_path().exists():
            return True, "FINISHED"
        if spec.failed_path().exists():
            return True, "FAILED"
        return False, None

    def ensure_running(self, spec: SlurmJobSpec) -> None:
        """Ensure the spec is submitted or already done/active.

        Parameters
        ----------
        spec : SlurmJobSpec
            Job spec.

        Notes
        -----
        This method does not register specs; it's a one-off submit-if-needed.
        """
        done, status = self._sentinel_done(spec)
        if done:
            logger.debug(f"[SLURM] {spec.workdir.name}: already {status}; not submitting")
            return
        if self.dry_run:
            self.triggered = True
            return
        # Submit unconditionally if no sentinel; wait_loop handles dedup & resub logic.
        self._submit(spec)

    # ---------- global wait ----------
    def wait_all(self) -> None:
        """Submit/monitor all registered jobs and block until completion."""
        specs_map = self._load_registry_specs()
        specs_map.update(self._inmem_specs)
        if self._stage:
            specs_map = {
                wd: sp for wd, sp in specs_map.items() if self._stage_matches(sp.stage, wd)
            }

        if not specs_map:
            if self.dry_run:
                self.triggered = True
            else:
                logger.debug("[SLURM] wait_all: nothing to monitor.")
            return

        if self.dry_run:
            self.triggered = True
            return

        self._wait_loop(list(specs_map.values()))
        self.clear()

    def wait_until_done(self, specs: Iterable[SlurmJobSpec]) -> None:
        """Legacy interface: monitor a given set until complete."""
        if self.dry_run:
            self.triggered = True
            return
        self._wait_loop(list(specs))

    def _wait_loop(self, specs: List[SlurmJobSpec]) -> None:
        """Internal polling loop with batched scheduler queries."""
        # optional progress bar
        try:
            from tqdm import tqdm  # type: ignore

            use_tqdm = True
        except Exception:
            tqdm = None  # type: ignore
            use_tqdm = False

        # initial submission pass
        self.wait_for_slot()
        submit_iter = (
            tqdm(specs, desc="SLURM submissions", leave=True, dynamic_ncols=True)
            if use_tqdm
            else specs
        )
        for s in submit_iter:
            if self.max_active_jobs is not None and self.n_active >= self.max_active_jobs:
                logger.info(
                    f"[SLURM] reached max_active_jobs={self.max_active_jobs}; deferring submissions"
                )
                self.wait_for_slot()
            try:
                # do not resubmit here; just ensure it has a JOBID if needed
                if not _read_text(s.jobid_path()):
                    self.ensure_running(s)
            except Exception as e:
                logger.error(f"[SLURM] submit failed for {s.workdir}: {e}")
                raise

        pending: Dict[Path, SlurmJobSpec] = {s.workdir: s for s in specs}
        retries: Dict[Path, int] = {s.workdir: self._retries.get(s.workdir, 0) for s in specs}

        total = len(specs)
        completed: set[Path] = set()
        last_log = 0.0
        pbar = (
            tqdm(total=total, desc="SLURM jobs", leave=True, dynamic_ncols=True)
            if use_tqdm
            else None
        )

        while pending:
            # build current jobid list
            wd_jobid: Dict[Path, str] = {}
            jobids: List[str] = []
            for wd, sp in pending.items():
                jid = _read_text(sp.jobid_path())
                if jid:
                    wd_jobid[wd] = jid
                    jobids.append(jid)

            # batched state lookup
            squeue_states = _states_from_squeue(jobids)
            missing = [jid for jid in jobids if jid not in squeue_states]
            sacct_states = _states_from_sacct(missing) if missing else {}

            def state_for(wd: Path) -> Optional[str]:
                jid = wd_jobid.get(wd)
                if not jid:
                    return None
                return squeue_states.get(jid) or sacct_states.get(jid)

            done_now: List[Path] = []
            running_cnt = 0
            resub_cnt = 0
            failed_cnt = 0

            for wd, sp in list(pending.items()):
                # sentinel checks first (no slurm calls)
                done, st = self._sentinel_done(sp)
                if done:
                    if st == "FAILED":
                        failed_cnt += 1
                    done_now.append(wd)
                    continue

                jid = wd_jobid.get(wd)
                state = state_for(wd)

                if state in SLURM_OK_STATES:
                    running_cnt += 1
                    continue

                # ended or missing => resubmit / fail-out
                timeout_state = state == "TIMEOUT"
                completed_state = state == "COMPLETED"
                resub_reason = state or "MISSING"

                # retry budget: TIMEOUT and COMPLETED-without-sentinel are unlimited (per prior behavior)
                r = retries.get(wd, 0)
                if (not timeout_state and not completed_state) and r >= self.max_retries:
                    logger.error(
                        f"[SLURM] {wd.name}: exceeded max_retries={self.max_retries} "
                        f"(state={resub_reason}); marking FAILED"
                    )
                    try:
                        sp.failed_path().touch()
                    except Exception:
                        pass
                    failed_cnt += 1
                    done_now.append(wd)
                    continue

                # resubmit
                resub_cnt += 1
                if timeout_state:
                    logger.debug(
                        f"[SLURM] {wd.name}: job{(' ' + jid) if jid else ''} state=TIMEOUT; resubmitting"
                    )
                elif completed_state:
                    logger.debug(
                        f"[SLURM] {wd.name}: job{(' ' + jid) if jid else ''} COMPLETED without FINISHED; resubmitting"
                    )
                else:
                    logger.warning(
                        f"[SLURM] {wd.name}: job{(' ' + jid) if jid else ''} state={resub_reason}; "
                        f"resubmitting ({r + 1}/{self.max_retries})"
                    )

                time.sleep(self.resubmit_backoff_s)
                self.wait_for_slot()
                try:
                    self._submit(sp)
                    if not timeout_state and not completed_state:
                        retries[wd] = r + 1
                        self._retries[wd] = retries[wd]
                except Exception as e:
                    logger.error(f"[SLURM] {wd.name}: resubmit failed: {e}")
                    raise

            # finalize done items
            for wd in done_now:
                pending.pop(wd, None)
                if wd not in completed:
                    completed.add(wd)
                    if pbar:
                        pbar.update(1)

            if pbar:
                pbar.set_postfix({"running": running_cnt, "failed": failed_cnt})
            else:
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

    def _resolve_header_root(self, spec: SlurmJobSpec) -> Path:
        """Resolve where header files live."""
        root = spec.header_root or self._header_root
        if not root:
            env_root = os.environ.get("BATTER_SLURM_HEADER_DIR")
            if env_root:
                return Path(env_root)
        return Path(root) if root else Path.home() / ".batter"


    def _rebuild_script_with_header(self, spec: SlurmJobSpec, script_abs: Path) -> None:
        """Rebuild the sbatch script by prepending header to stored body (if configured).

        This supports your old workflow:
        - old generator writes a body-only script (or separate *.body file)
        - manager prepends the appropriate header at submit time
        """
        # If no header requested, do nothing
        if not spec.header_name and not spec.header_template:
            return

        # Locate body source
        body_path = (spec.workdir / spec.body_rel) if spec.body_rel else script_abs
        if not body_path.exists():
            candidate = script_abs.with_suffix(script_abs.suffix + ".body")
            if candidate.exists():
                body_path = candidate
            else:
                # no body to rebuild from
                return

        # Read body, drop baked-in SBATCH lines so header owns SBATCH
        try:
            body_text = body_path.read_text()
        except Exception as exc:
            logger.warning(f"[SLURM] Failed to read body {body_path}: {exc}")
            return

        body_lines = [
            ln for ln in body_text.splitlines()
            if not ln.lstrip().startswith("#SBATCH")
        ]
        body_text = "\n".join(body_lines)

        # Read header
        header_text = ""
        header_root = self._resolve_header_root(spec)

        if spec.header_name:
            user_header = header_root / cast(str, spec.header_name)
            if user_header.exists():
                try:
                    header_text = user_header.read_text()
                except Exception as exc:
                    logger.warning(f"[SLURM] Failed to read header {user_header}: {exc}")
                    header_text = ""
            elif spec.header_template and spec.header_template.exists():
                try:
                    header_text = spec.header_template.read_text()
                except Exception:
                    header_text = ""

        # Combine and overwrite the submit script
        combined = header_text
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += body_text
        if not combined.endswith("\n"):
            combined += "\n"

        try:
            script_abs.write_text(combined)
        except Exception as exc:
            logger.warning(f"[SLURM] Could not write rebuilt script {script_abs}: {exc}")