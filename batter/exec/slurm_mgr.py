from __future__ import annotations
import os
import re
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from loguru import logger

# ---- SLURM state sets ----
SLURM_OK_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "STAGE_OUT", "SUSPENDED"}
SLURM_FINAL_BAD = {"CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}
JOBID_RE = re.compile(r"Submitted batch job\s+(\d+)", re.I)

# ---- Helpers ----
def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text().strip()
    except Exception:
        return None

def _write_text(p: Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)

def _jobid_file(root: Path) -> Path:
    return root / "JOBID"

def _state_from_squeue(jobid: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["squeue", "-h", "-j", jobid, "-o", "%T"], text=True).strip()
        if out:
            return out.split()[0]
    except Exception:
        pass
    return None

def _state_from_sacct(jobid: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jobid, "-X", "-n", "-o", "State"], text=True
        ).strip()
        if out:
            return out.split()[0]
    except Exception:
        pass
    return None

def _slurm_state(jobid: Optional[str]) -> Optional[str]:
    if not jobid:
        return None
    return _state_from_squeue(jobid) or _state_from_sacct(jobid)


@dataclass
class SlurmJobSpec:
    workdir: Path
    script_rel: str = "SLURMM-run"          # default relative to workdir
    finished_name: str = "FINISHED"
    failed_name: str = "FAILED"
    name: Optional[str] = None
    extra_sbatch: Sequence[str] = field(default_factory=list)

    # allow a few common variants (case, alt names)
    alt_script_names: Sequence[str] = ("SLURMM-run", "SLURMM-Run", "slurmm-run", "run.sh")

    # --- absolute paths for checks/sentinels ---
    def finished_path(self) -> Path: return self.workdir / self.finished_name
    def failed_path(self)   -> Path: return self.workdir / self.failed_name
    def jobid_path(self)    -> Path: return self.workdir / "JOBID"

    def resolve_script_abs(self) -> Path:
        """
        Return the first existing absolute script path under workdir.
        Falls back to the preferred name even if missing (for error msgs).
        """
        preferred = self.workdir / self.script_rel
        candidates = [preferred] + [self.workdir / n for n in self.alt_script_names if n != self.script_rel]
        for p in candidates:
            if p.exists():
                return p
        return preferred

    def script_arg(self) -> str:
        """
        Relative path passed to sbatch. Use just the filename (or workdir-relative path)
        so submission happens *in* workdir.
        """
        abs_script = self.resolve_script_abs()
        try:
            # best effort to keep it relative to workdir
            return str(abs_script.relative_to(self.workdir))
        except ValueError:
            # fallback: just the basename
            return abs_script.name


# ---- Manager ----
class SlurmJobManager:
    def __init__(self, poll_s: float = 20.0, max_retries: int = 3, resubmit_backoff_s: float = 30.0):
        self.poll_s = float(poll_s)
        self.max_retries = int(max_retries)
        self.resubmit_backoff_s = float(resubmit_backoff_s)

    def _submit(self, spec: SlurmJobSpec) -> str:
        # Resolve the absolute path for existence/chmod checks
        script_abs = spec.resolve_script_abs()
        if not script_abs.exists():
            listing = ", ".join(sorted(p.name for p in spec.workdir.iterdir())) if spec.workdir.exists() else "(missing workdir)"
            raise FileNotFoundError(
                f"SLURM script not found: {script_abs}\n"
                f"in workdir: {spec.workdir}\n"
                f"contents: {listing}"
            )

        # Best effort: ensure executable bit
        try:
            script_abs.chmod(script_abs.stat().st_mode | 0o111)
        except Exception:
            pass

        # Build sbatch command using a path RELATIVE to workdir
        cmd: List[str] = ["sbatch"]
        if spec.name:
            cmd += ["--job-name", spec.name]
        if spec.extra_sbatch:
            cmd += list(spec.extra_sbatch)
        cmd.append(spec.script_arg())

        logger.debug(f"[SLURM] sbatch: {' '.join(cmd)} (cwd={spec.workdir})")
        out = subprocess.check_output(cmd, cwd=spec.workdir, text=True).strip()

        m = JOBID_RE.search(out)
        if not m:
            raise RuntimeError(f"Could not parse sbatch output: {out}")
        jobid = m.group(1)
        _write_text(spec.jobid_path(), f"{jobid}\n")
        logger.info(f"[SLURM] submitted {spec.workdir.name} → job {jobid}")
        return jobid

    def _status(self, spec: SlurmJobSpec) -> Tuple[bool, Optional[str]]:
        """
        Returns (done, status):
          - (True, 'FINISHED'|'FAILED') if a sentinel exists
          - (False, <SLURM_STATE|None>) otherwise
        """
        if spec.finished_path().exists():
            return True, "FINISHED"
        if spec.failed_path().exists():
            return True, "FAILED"
        jobid = _read_text(spec.jobid_path())
        return False, _slurm_state(jobid)

    # --- public high-level API ---
    def ensure_running(self, spec: SlurmJobSpec) -> None:
        done, status = self._status(spec)
        if done:
            logger.info(f"[SLURM] {spec.workdir.name}: already {status}; not submitting")
            return
        state = _slurm_state(_read_text(spec.jobid_path()))
        if state in SLURM_OK_STATES:
            logger.info(f"[SLURM] {spec.workdir.name}: active ({state}); not submitting")
            return
        self._submit(spec)

    def wait_until_done(self, specs: Iterable[SlurmJobSpec]) -> None:
        """
        For a set of specs:
          - submit if needed
          - watch until FINISHED/FAILED
          - if job vanishes and no sentinel: resubmit (bounded)
        """
        specs = list(specs)
        retries = {s.workdir: 0 for s in specs}

        # initial submissions
        for s in specs:
            try:
                self.ensure_running(s)
            except Exception as e:
                logger.error(f"[SLURM] submit failed for {s.workdir}: {e}")

        pending = {s.workdir: s for s in specs}
        while pending:
            done_now: List[Path] = []
            for wd, s in list(pending.items()):
                done, status = self._status(s)
                if done:
                    logger.info(f"[SLURM] {wd.name}: {status}")
                    done_now.append(wd)
                    continue

                jobid = _read_text(s.jobid_path())
                state = _slurm_state(jobid)
                if state in SLURM_OK_STATES:
                    logger.debug(f"[SLURM] {wd.name}: {state} (waiting)")
                    continue

                # job missing or ended without sentinel → resubmit
                r = retries[wd]
                if r >= self.max_retries:
                    logger.error(f"[SLURM] {wd.name}: exceeded max_retries={self.max_retries} (state={state or 'MISSING'})")
                    done_now.append(wd)  # stop waiting; leave without fabricating a sentinel
                    continue

                logger.warning(f"[SLURM] {wd.name}: job{(' ' + jobid) if jobid else ''} state={state or 'MISSING'}; resubmitting ({r+1}/{self.max_retries})")
                time.sleep(self.resubmit_backoff_s)
                try:
                    self._submit(s)
                    retries[wd] = r + 1
                except Exception as e:
                    logger.error(f"[SLURM] {wd.name}: resubmit failed: {e}")

            for wd in done_now:
                pending.pop(wd, None)

            if pending:
                time.sleep(self.poll_s)