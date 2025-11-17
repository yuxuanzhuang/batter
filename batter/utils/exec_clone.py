from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Literal, Optional

from loguru import logger

CopyMode = Literal["copy", "hardlink", "symlink"]


def _copy_file(src: Path, dst: Path, mode: CopyMode) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        try:
            os.symlink(rel, dst)
        except FileExistsError:
            pass
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    shutil.copy2(src, dst)


def _copy_dir_contents(
    src: Path,
    dst: Path,
    mode: CopyMode,
    *,
    ignore_ext: Iterable[str] | None = None,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ignore_ext = set(ignore_ext or [])
    for entry in src.iterdir():
        if entry.name.startswith(".") and entry.is_dir():
            continue
        if entry.is_dir():
            _copy_dir_contents(entry, dst / entry.name, mode, ignore_ext=ignore_ext)
            continue
        if entry.suffix in ignore_ext:
            continue
        _copy_file(entry, dst / entry.name, mode)


def _copy_if_exists(src: Path, dst: Path, mode: CopyMode) -> None:
    if not src.exists():
        return
    if src.is_dir():
        _copy_dir_contents(src, dst, mode)
        return
    _copy_file(src, dst, mode)


def _strip_run_state(dst_run_dir: Path, reset_states: bool) -> None:
    slurm_q = dst_run_dir / ".slurm"
    if slurm_q.exists():
        shutil.rmtree(slurm_q, ignore_errors=True)
    if not reset_states:
        return
    markers = {"FINISHED", "FAILED", "EQ_FINISHED", "UNBOUND"}
    for p in dst_run_dir.rglob("*"):
        if p.name == "JOBID" and p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
        if p.name in markers and p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def _copy_simulation_dir(src_run: Path, dst_run: Path, mode: CopyMode, only_equil: bool) -> None:
    sim_src = src_run / "simulations"
    if not sim_src.exists():
        return
    for lig_dir in sim_src.iterdir():
        if not lig_dir.is_dir():
            continue
        dst_lig = dst_run / "simulations" / lig_dir.name
        for artifact in ("inputs", "params", "equil"):
            _copy_if_exists(lig_dir / artifact, dst_lig / artifact, mode)
        if only_equil:
            (dst_lig / "fe").mkdir(parents=True, exist_ok=True)
            continue
        fe_src = lig_dir / "fe"
        if not fe_src.exists():
            continue
        _copy_dir_contents(fe_src, dst_lig / "fe", mode, ignore_ext={".rst7", ".nc", ".log", ".out"})


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
    src_root = Path(work_dir) / "executions"
    dst_root = dst_root or work_dir
    dst_execute_root = Path(dst_root) / "executions"
    src = src_root / src_run_id
    if not src.exists():
        raise FileNotFoundError(f"Source run_id not found: {src}")
    dst_run_id = dst_run_id or f"{src_run_id}-clone"
    dst = dst_execute_root / dst_run_id
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination run already exists: {dst}")
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("artifacts/config", "artifacts/ligand_params"):
        _copy_dir_contents(src / name, dst / name, mode)
    for fname in ("system.pkl", "mols.txt"):
        _copy_if_exists(src / fname, dst / fname, mode)

    _copy_simulation_dir(src, dst, mode, only_equil)

    _strip_run_state(dst, reset_states=reset_states)
    (dst / "batter.run.log").write_text("")

    logger.info(
        f"Cloned execution '{src_run_id}' → '{dst_run_id}' "
        f"under root '{dst_execute_root}' (mode={mode}, only_equil={only_equil})."
    )
    return dst
