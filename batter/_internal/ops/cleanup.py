"""Cleanup helpers for generated BATTER simulation work directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger


@dataclass
class CleanupStats:
    """Small summary for debug logging."""

    files: int = 0
    dirs: int = 0
    bytes: int = 0

    def add(self, other: "CleanupStats") -> None:
        self.files += other.files
        self.dirs += other.dirs
        self.bytes += other.bytes


_STATUS_KEEP = {
    "FINISHED",
    "FAILED",
    "UNBOUND",
    "JOBID",
    "EQ_FINISHED",
    "ATTEMPT_FAILED",
    "prepare_equil.ok",
    "prepare_equil.failed",
    "prepare_fe.ok",
    "prepare_fe_windows.ok",
    "pre_prepare_fe.ok",
    "pre_prepare_fe.failed",
    "job_attempt.txt",
    "production-start.ps",
    "run.log",
}

_RUN_INPUT_KEEP_NAMES = {
    "run-local.bash",
    "run-local-batch.bash",
    "run-local-remd.bash",
    "check_run.bash",
    "SLURMM-run",
    "SLURMM-run.body",
    "SLURMM-BATCH-remd",
    "lambda.sch",
    "mdin-template",
    "mdin-current",
    "mdin-batch-template",
    "mdin-remd-template",
    "mini.in",
    "mini_eq.in",
    "mini_noshake.in",
    "eq.in",
    "eqnvt.in",
    "eqnpt.in",
    "eqnpt0.in",
    "eqnpt0-water.in",
    "eqnpt_eq.in",
}

_SYSTEM_KEEP_NAMES = {
    "full.prmtop",
    "full.inpcrd",
    "full.hmr.prmtop",
    "full_merged.prmtop",
    "full.pdb",
    "equil-reference.pdb",
    "build_amber_renum.txt",
    "protein_renum.txt",
    "other_parts.prmtop",
    "other_parts.inpcrd",
    "other_parts.pdb",
    "eq.rst7",
    "mini_eq.rst7",
    "md-current.rst7",
    "md-previous.rst7",
    "representative.pdb",
    "representative.rst7",
    "representative_complex.pdb",
    "representative_pose.pdb",
    "initial_pose.pdb",
    "equilibration_analysis_results.npz",
    "simulation_analysis.png",
    "dihed_hist.png",
    "stable_boresch_distance.json",
    "prolif_interactions.json",
    "prolif_interactions_timeseries.csv.gz",
    "prolif_interactions_barcode.png",
    "prolif_interactions_occupancy.png",
    "prolif_lignetwork.html",
    "prolif_interaction_diagram.png",
    "anchors.txt",
    "anchors.json",
    "cv.in",
    "disang.rest",
    "restraints.in",
    "extra_conf_restraints.json",
    "ligand_dihedral_restraints.json",
    "ligand_dihedral_schedule.json",
    "sdr_info.txt",
    "vac.inpcrd",
    "vac.pdb",
    "vac.prmtop",
    "vac_ligand.pdb",
    "vac_ligand.prmtop",
}

_KEEP_DIR_NAMES = {
    "artifacts",
    "params",
    "inputs",
    "remd",
}

_PREP_DEBUG_GLOBS = (
    "STAGE-POSE-*.out",
    "STAGE-POSE-*.err",
    "assign.*",
    "tleap*.in",
    "tleap*.log",
    "leap.log",
    "logfile",
    "parmed-hmr.in",
    "parmed-hmr.log",
    "mdinfo",
    "build*.pdb",
    "output.pdb",
    "fe-*.pdb",
    "rec_amber*.pdb",
    "rec_file-clean.pdb",
    "rec_file.pdb",
    "solvate_*.pdb",
    "solvate_*.prmtop",
    "solvate_*.inpcrd",
    "solvate_pre_*.pdb",
    "other_parts.*",
    "vac_orig.pdb",
    "dum.*",
    "dummy.pdb",
    "apo_dummy.pdb",
    "build_amber_sslink",
    "rec_amber_renum.txt",
    "rec_amber_sslink",
    "ref_*.inpcrd",
    "ref_*.pdb",
    "ref_*.prmtop",
    "alter_ligand*.pdb",
    "alter_ligand*.prmtop",
    "full_pre.pdb",
    "lipid_hydrogen_repair_*.json",
    "periodic_water_cleanup.json",
)

_EQUIL_RUN_DEBUG_GLOBS = (
    "traj*.nc",
    "md-*.nc",
    "eqnpt*.nc",
    "eqnvt.nc",
    "mini*.nc",
    "md-*.out",
    "eqnpt*.out",
    "eqnvt.out",
    "mini*.out",
    "STAGE-POSE-*.out",
    "STAGE-POSE-*.err",
    "mdinfo",
)

_FE_RUN_DEBUG_GLOBS = (
    "traj*.nc",
    "md-*.nc",
    "md[0-9]*.nc",
    "mdin-*.nc",
    "mini*.nc",
    "eq*.nc",
    "md-*.out",
    "md[0-9]*.out",
    "mdin-*.out",
    "mini*.out",
    "eq*.out",
    "*.mdinfo",
    "*.log",
    "*.mden",
    "mdinfo",
    "logfile",
    "cmass.txt",
    "cmass-*.txt",
    "restraints_curr.in",
    "restraints.dat",
    "restraints.log",
)

_DEBUG_DIR_NAMES = {
    ".restraintmask_cache",
    "amber_files",
    "ARCHIVED_LOGS",
    "WRONG_FAIL",
}


def _path_size(path: Path) -> int:
    try:
        if path.is_symlink() or path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file() or child.is_symlink():
                    total += child.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def _delete_path(path: Path) -> CleanupStats:
    stats = CleanupStats(bytes=_path_size(path))
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            stats.dirs = 1
        else:
            path.unlink()
            stats.files = 1
    except FileNotFoundError:
        return CleanupStats()
    except OSError as exc:
        logger.warning("[cleanup] Could not remove {}: {}", path, exc)
        return CleanupStats()
    return stats


def _remove_globs(root: Path, patterns: Iterable[str], keep: Callable[[Path], bool]) -> CleanupStats:
    stats = CleanupStats()
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.exists() and not path.is_symlink():
                continue
            if keep(path):
                continue
            stats.add(_delete_path(path))
    return stats


def _remove_named_dirs(root: Path, names: Iterable[str]) -> CleanupStats:
    stats = CleanupStats()
    for name in names:
        path = root / name
        if path.exists() or path.is_symlink():
            stats.add(_delete_path(path))
    return stats


def _is_required_top_level(path: Path) -> bool:
    name = path.name
    if name in _STATUS_KEEP or name in _RUN_INPUT_KEEP_NAMES or name in _SYSTEM_KEEP_NAMES:
        return True
    if name.startswith("md-"):
        return True
    if name.startswith("equil-") and name.endswith(".pdb"):
        return True
    if name.startswith("cmass"):
        return True
    if name.startswith("anchors-") and name.endswith(".txt"):
        return True
    if name.startswith("prolif_"):
        return True
    if path.is_dir() and name in _KEEP_DIR_NAMES:
        return True
    if path.is_file() and name.endswith(".rest"):
        return True
    return False


def _is_required_post_fe(path: Path) -> bool:
    return _is_required_top_level(path)


def _eq_restart_priority(path: Path) -> tuple[int, float, str]:
    name = path.name
    priority = {
        "eqnpt_appear.rst7": 1000,
        "eq.rst7": 950,
        "eqnpt_disappear.rst7": 900,
        "eqnpt_eq.rst7": 850,
        "eqnpt_pre.rst7": 650,
        "eqnvt.rst7": 600,
    }.get(name)
    if priority is None:
        priority = 0
        if name.startswith("eqnpt") and name.endswith(".rst7"):
            stem = name.removesuffix(".rst7")
            try:
                priority = 700 + int(stem[-2:])
            except ValueError:
                priority = 500
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return priority, mtime, name


def _select_eq_restart_to_keep(
    directory: Path,
    keep_names: Iterable[str] | None,
) -> set[Path]:
    if keep_names is not None:
        for name in keep_names:
            path = directory / name
            if path.is_file():
                return {path}
        return set()

    candidates = [
        path
        for path in directory.glob("eq*.rst7")
        if path.is_file() or path.is_symlink()
    ]
    if not candidates:
        return set()
    return {max(candidates, key=_eq_restart_priority)}


def _remove_intermediate_eq_restarts(
    directory: Path,
    *,
    keep_names: Iterable[str] | None = None,
) -> CleanupStats:
    stats = CleanupStats()
    if not directory.is_dir():
        return stats
    keep = _select_eq_restart_to_keep(directory, keep_names)
    seen: set[Path] = set()
    for pattern in ("eq*.rst7", "eq*.rst7.[0-9]*"):
        for path in directory.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            if path in keep:
                continue
            if not path.exists() and not path.is_symlink():
                continue
            stats.add(_delete_path(path))
    return stats


def _prune_directory_contents(
    directory: Path,
    keep: Callable[[Path], bool],
) -> CleanupStats:
    stats = CleanupStats()
    if not directory.is_dir():
        return stats
    for child in list(directory.iterdir()):
        if keep(child):
            continue
        stats.add(_delete_path(child))
    try:
        if directory.exists() and not any(directory.iterdir()):
            stats.add(_delete_path(directory))
    except OSError:
        pass
    return stats


def _keep_equil_build_file(path: Path, ligand: str | None) -> bool:
    name = path.name
    if name in {"protein_renum.txt", "protein_sslink", "protein_anchors.txt"}:
        return True
    if name.startswith("anchors") and name.endswith(".txt"):
        return True
    if name.endswith(".txt") or name.endswith(".json"):
        return True
    if ligand and name == f"{ligand}.pdb":
        return True
    if ligand is None and name.endswith(".pdb"):
        return True
    return False


def cleanup_prepare_equil(equil_dir: Path, *, ligand: str | None = None) -> CleanupStats:
    """Remove preparation scratch from an equilibration directory."""

    equil_dir = Path(equil_dir)
    stats = CleanupStats()
    if not equil_dir.is_dir():
        return stats

    build_files = equil_dir / "q_build_files"
    if build_files.is_dir():
        protein_renum = build_files / "protein_renum.txt"
        if protein_renum.exists() and not (equil_dir / "protein_renum.txt").exists():
            shutil.copy2(protein_renum, equil_dir / "protein_renum.txt")
        if ligand:
            ligand_pdb = build_files / f"{ligand}.pdb"
            if ligand_pdb.exists() and not (equil_dir / ligand_pdb.name).exists():
                shutil.copy2(ligand_pdb, equil_dir / ligand_pdb.name)

    stats.add(_remove_named_dirs(equil_dir, ("q_amber_files", "q_build_files", "q_run_files")))
    stats.add(_remove_named_dirs(equil_dir, _DEBUG_DIR_NAMES))
    stats.add(_remove_globs(equil_dir, _PREP_DEBUG_GLOBS, _is_required_top_level))
    _log_cleanup("prepare_equil", equil_dir, stats)
    return stats


def cleanup_prepare_fe_component(
    comp_dir: Path,
    *,
    comp: str,
    drop_build_files: bool,
) -> CleanupStats:
    """Remove preparation scratch from one FE component directory."""

    comp_dir = Path(comp_dir)
    stats = CleanupStats()
    if not comp_dir.is_dir():
        return stats

    if drop_build_files:
        stats.add(_remove_named_dirs(comp_dir, (f"{comp}_amber_files",)))
        stats.add(_remove_named_dirs(comp_dir, (f"{comp}_build_files",)))
    stats.add(_remove_named_dirs(comp_dir, (f"{comp}_run_files",)))
    stats.add(_remove_named_dirs(comp_dir, _DEBUG_DIR_NAMES))
    stats.add(_remove_globs(comp_dir, _PREP_DEBUG_GLOBS, _is_required_top_level))

    for child in comp_dir.iterdir():
        if child.is_dir() and (child.name == f"{comp}-1" or child.name.startswith(comp)):
            stats.add(_remove_named_dirs(child, _DEBUG_DIR_NAMES))
            stats.add(_remove_globs(child, _PREP_DEBUG_GLOBS, _is_required_top_level))
    _log_cleanup(f"prepare_fe:{comp_dir.name}", comp_dir, stats)
    return stats


def cleanup_prepare_fe_root(
    fe_root: Path,
    *,
    components: Iterable[str],
    drop_build_files: bool,
) -> CleanupStats:
    """Remove preparation scratch from a full ``fe`` or ``pre_fe`` tree."""

    stats = CleanupStats()
    for comp in components:
        stats.add(
            cleanup_prepare_fe_component(
                Path(fe_root) / comp,
                comp=str(comp),
                drop_build_files=drop_build_files,
            )
        )
    return stats


def cleanup_equil_after_analysis(equil_dir: Path) -> CleanupStats:
    """Remove equilibration trajectories and stage logs after analysis artifacts exist."""

    equil_dir = Path(equil_dir)
    stats = CleanupStats()
    if not equil_dir.is_dir():
        return stats
    stats.add(_remove_named_dirs(equil_dir, _DEBUG_DIR_NAMES))
    stats.add(_remove_globs(equil_dir, _EQUIL_RUN_DEBUG_GLOBS, _is_required_top_level))
    stats.add(_remove_intermediate_eq_restarts(equil_dir))
    _log_cleanup("equil_analysis", equil_dir, stats)
    return stats


def _component_names(fe_root: Path, components: Iterable[str] | None) -> list[str]:
    if components is not None:
        return [str(comp) for comp in components]
    if not fe_root.is_dir():
        return []
    return sorted(
        child.name
        for child in fe_root.iterdir()
        if child.is_dir()
        and len(child.name) == 1
        and (child / f"{child.name}-1").is_dir()
    )


def cleanup_fe_after_analysis(
    fe_root: Path,
    *,
    components: Iterable[str] | None = None,
) -> CleanupStats:
    """Remove FE runtime trajectories/logs after ``Results.dat`` exists."""

    fe_root = Path(fe_root)
    stats = CleanupStats()
    if not (fe_root / "Results" / "Results.dat").is_file():
        return stats

    for comp in _component_names(fe_root, components):
        comp_dir = fe_root / comp
        if not comp_dir.is_dir():
            continue
        stats.add(
            _remove_named_dirs(
                comp_dir,
                (f"{comp}_amber_files", f"{comp}_build_files", f"{comp}_run_files"),
            )
        )
        stats.add(_remove_named_dirs(comp_dir, _DEBUG_DIR_NAMES))
        stats.add(_remove_globs(comp_dir, _PREP_DEBUG_GLOBS, _is_required_top_level))
        stats.add(_remove_globs(comp_dir, _FE_RUN_DEBUG_GLOBS, _is_required_post_fe))
        stats.add(_remove_intermediate_eq_restarts(comp_dir))

        for child in comp_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name != f"{comp}-1" and not child.name.startswith(comp):
                continue
            stats.add(_remove_named_dirs(child, _DEBUG_DIR_NAMES))
            stats.add(_remove_globs(child, _PREP_DEBUG_GLOBS, _is_required_top_level))
            stats.add(_remove_globs(child, _FE_RUN_DEBUG_GLOBS, _is_required_post_fe))
            keep_names = (
                ("eqnpt04.rst7", "eq.rst7", "eqnpt_eq.rst7")
                if child.name == f"{comp}-1"
                else ("eq.rst7",)
            )
            stats.add(_remove_intermediate_eq_restarts(child, keep_names=keep_names))

    _log_cleanup("fe_analysis", fe_root, stats)
    return stats


def cleanup_fe_equil_after_success(
    fe_root: Path,
    *,
    components: Iterable[str] | None = None,
) -> CleanupStats:
    """Remove FE-equilibration runtime outputs after ``EQ_FINISHED`` exists."""

    fe_root = Path(fe_root)
    stats = CleanupStats()
    for comp in _component_names(fe_root, components):
        equil_dir = fe_root / comp / f"{comp}-1"
        if not (equil_dir / "EQ_FINISHED").is_file():
            continue
        stats.add(_remove_named_dirs(equil_dir, _DEBUG_DIR_NAMES))
        stats.add(_remove_globs(equil_dir, _PREP_DEBUG_GLOBS, _is_required_top_level))
        stats.add(_remove_globs(equil_dir, _FE_RUN_DEBUG_GLOBS, _is_required_post_fe))
        stats.add(
            _remove_intermediate_eq_restarts(
                equil_dir,
                keep_names=("eqnpt04.rst7", "eq.rst7", "eqnpt_eq.rst7"),
            )
        )
    _log_cleanup("fe_equil", fe_root, stats)
    return stats


def _log_cleanup(label: str, root: Path, stats: CleanupStats) -> None:
    if not (stats.files or stats.dirs):
        return
    logger.debug(
        "[cleanup:{}] removed {} file(s), {} dir(s), {:.2f} MiB under {}",
        label,
        stats.files,
        stats.dirs,
        stats.bytes / (1024 * 1024),
        root,
    )
