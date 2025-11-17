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
    dst_root: Path | None = None,
    mode: CopyMode = "hardlink",
    only_equil: bool = True,
    reset_states: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Clone an execution folder to a new run id.

    Parameters
    ----------
    work_dir : Path
        Work directory containing ``executions/``.
    src_run_id : str
        Existing run identifier to clone.
    dst_run_id : str, optional
        Destination run identifier. When ``None``, ``"-CLONE"`` is appended to
        ``src_run_id``.
    mode : {"copy", "hardlink", "symlink"}
        Copy strategy for files.
    only_equil : bool
        When ``True``, copy only inputs/artifacts/equilibration directories.
    reset_states : bool
        Remove SLURM metadata and phase sentinels from the cloned run.
    overwrite : bool
        Allow deleting an existing destination before cloning.

    Returns
    -------
    Path
        Path to the newly cloned execution directory.

    Raises
    ------
    FileNotFoundError
        If ``src_run_id`` does not exist.
    FileExistsError
        If the destination already exists and ``overwrite`` is ``False``.
    """
    src_runs = Path(work_dir) / "executions"
    dst_runs = Path(dst_root or work_dir) / "executions"
    src = src_runs / src_run_id
    if not src.exists():
        raise FileNotFoundError(f"Source run_id not found: {src}")

    dst_run_id = dst_run_id or f"{src_run_id}-CLONE"
    dst = dst_runs / dst_run_id

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

    logger.info(
        f"Cloned run '{src_run_id}' → '{dst_run_id}' "
        f"under root '{dst_runs}' in mode={mode}, only_equil={only_equil}, reset_states={reset_states}"
    )
    return dst
