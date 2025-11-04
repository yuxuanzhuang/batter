from __future__ import annotations
import os
import stat
import json
import shutil
from pathlib import Path
from typing import Iterable, Literal, Optional, Callable
from loguru import logger


CopyMode = Literal["copy", "hardlink", "symlink"]


def _select_copy_fn(mode: CopyMode) -> Callable[[str, str], str]:
    if mode == "copy":
        return shutil.copy2
    if mode == "hardlink":
        return os.link  # may fail across filesystems; caller handles
    if mode == "symlink":
        # we implement symlink at the directory-walk level (files/dirs)
        # copy_function not used in that case.
        return shutil.copy2
    raise ValueError(f"Unknown copy mode: {mode}")


def _should_copy_path(p: Path, only_equil: bool) -> bool:
    """Filter which paths to include when only_equil=True."""
    if not only_equil:
        return True
    # Keep: inputs/, artifacts/ (but drop fe artifacts), simulations/*/{inputs, artifacts, equil}
    rel = "/".join(p.parts)  # cheap comparable
    # prune fe* except scaffold created later
    parts = p.parts
    if "fe" in parts:
        # Allow top-level 'fe' only when we explicitly make minimal tree later; here we skip source fe content.
        return False
    return True


def _strip_run_state(dst_run_dir: Path, reset_states: bool) -> None:
    """Remove SLURM/marker state so the cloned run can be re-submitted cleanly."""
    # Always remove a lingering slurm queue for safety when cloning
    slurm_q = dst_run_dir / ".slurm"
    if slurm_q.exists():
        shutil.rmtree(slurm_q, ignore_errors=True)

    if not reset_states:
        return

    # Remove JOBIDs and phase sentinels so orchestrator will (re)submit
    markers = {"FINISHED", "FAILED", "EQ_FINISHED", "UNBOUND"}
    for p in dst_run_dir.rglob("*"):
        name = p.name
        if name == "JOBID" and p.is_file():
            try: p.unlink()
            except Exception: pass
        if name in markers and p.is_file():
            try: p.unlink()
            except Exception: pass


def _ensure_fe_scaffold(dst_run_dir: Path, src_run_dir: Path) -> None:
    """
    For only_equil clones, create empty fe/ scaffolding per ligand/component
    using the src directory layout (no heavy files). This keeps later steps simple.
    """
    # Source layout: simulations/<LIG>/(inputs|artifacts|equil|fe/...)
    sim_root = src_run_dir / "simulations"
    if not sim_root.exists():
        return
    for lig_dir in sim_root.iterdir():
        if not lig_dir.is_dir():
            continue
        dst_lig = dst_run_dir / "simulations" / lig_dir.name
        # If the source had fe/, create an empty fe/ at the destination to keep paths consistent
        src_fe = lig_dir / "fe"
        if src_fe.exists():
            (dst_lig / "fe").mkdir(parents=True, exist_ok=True)


def clone_execution(
    work_dir: Path,
    src_run_id: str,
    dst_run_id: Optional[str] = None,
    *,
    mode: CopyMode = "hardlink",
    only_equil: bool = True,
    reset_states: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Clone <work_dir>/executions/<src_run_id> → <work_dir>/executions/<dst_run_id>.

    Parameters
    ----------
    work_dir
        BATTER work directory (the same passed to run_from_yaml; contains 'executions/').
    src_run_id
        Existing run_id to clone from.
    dst_run_id
        Target run_id. If None, suffix '-CLONE' is appended to src_run_id.
    mode
        'copy'      → real copy of files
        'hardlink'  → hardlink files (fast, but same filesystem required)
        'symlink'   → symlink files/dirs into the new run
    only_equil
        If True, copy inputs/artifacts/simulations/*/{inputs,artifacts,equil} but no FE data.
    reset_states
        If True, remove JOBID files, .slurm/ queue, FINISHED/FAILED/EQ_FINISHED/UNBOUND markers in the clone.
    overwrite
        If True, allow replacing an existing destination run directory.

    Returns
    -------
    Path
        The path to the new run directory.
    """
    runs_dir = Path(work_dir) / "executions"
    src = runs_dir / src_run_id
    if not src.exists():
        raise FileNotFoundError(f"Source run_id not found: {src}")

    dst_run_id = dst_run_id or f"{src_run_id}-CLONE"
    dst = runs_dir / dst_run_id

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination run already exists: {dst}")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    # Walk and replicate
    copy_fn = _select_copy_fn(mode)

    def _copy_file(s: Path, d: Path):
        d.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            # Use relative symlinks when possible
            rel = os.path.relpath(s, start=d.parent)
            try:
                os.symlink(rel, d)
            except FileExistsError:
                pass
        elif mode == "hardlink":
            try:
                os.link(s, d)
            except OSError:
                # Fallback to real copy if cross-filesystem
                shutil.copy2(s, d)
        else:
            shutil.copy2(s, d)

    for s in src.rglob("*"):
        rel = s.relative_to(src)
        d = dst / rel

        # filter content if only_equil
        if not _should_copy_path(rel, only_equil=only_equil):
            continue

        if s.is_dir():
            # For symlink mode, we symlink directories wholesale to reduce inode churn.
            if mode == "symlink":
                if d.exists():
                    continue
                d.parent.mkdir(parents=True, exist_ok=True)
                rel_dir = os.path.relpath(s, start=d.parent)
                try:
                    os.symlink(rel_dir, d, target_is_directory=True)
                except FileExistsError:
                    pass
                continue
            else:
                d.mkdir(parents=True, exist_ok=True)
                continue

        # files
        _copy_file(s, d)

    if only_equil:
        _ensure_fe_scaffold(dst, src)

    _strip_run_state(dst, reset_states=reset_states)

    # Start a fresh log for the clone
    (dst / "batter.run.log").write_text("")

    logger.info(f"Cloned run '{src_run_id}' → '{dst_run_id}' in mode={mode}, only_equil={only_equil}, reset_states={reset_states}")
    return dst