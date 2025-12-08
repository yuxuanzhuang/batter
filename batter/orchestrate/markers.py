"""Phase completion helpers for orchestrated simulation runs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

from loguru import logger

from batter.systems.core import SimSystem
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step
from batter.utils import components_under
from batter.orchestrate.state_registry import get_phase_state, PhaseState


def partition_children_by_status(children: List[SimSystem], phase: str) -> Tuple[List[SimSystem], List[SimSystem]]:
    """Split systems into success and failure buckets for a given phase.

    Parameters
    ----------
    children
        Sequence of per-ligand :class:`SimSystem` descriptors.
    phase
        Phase identifier whose sentinel criteria should be evaluated.

    Returns
    -------
    tuple[list[SimSystem], list[SimSystem]]
        Two lists containing systems that satisfied the success spec and those that
        appear to have failed (either explicitly or due to missing sentinels).
    """
    ok, bad = [], []
    for child in children:
        spec = _phase_spec(child.root, phase)
        success_spec = spec.success or spec.required
        is_success = _spec_satisfied(child.root, success_spec, phase)
        if is_success:
            ok.append(child)
            continue
        is_failure = _spec_satisfied(child.root, spec.failure, phase)
        if is_failure or not is_success:
            bad.append(child)
    return ok, bad


def _remove_patterns(root: Path, spec: List[List[str]]) -> bool:
    removed = False
    for group in spec:
        for pattern in group:
            for p in _expand_pattern(root, pattern):
                if not p.exists():
                    continue
                try:
                    if p.is_dir():
                        for child in p.rglob("*"):
                            if child.is_file():
                                child.unlink(missing_ok=True)
                        p.rmdir()
                    else:
                        p.unlink()
                    removed = True
                except Exception:
                    logger.warning("Could not remove sentinel %s", p)
    return removed


def handle_phase_failures(children: List[SimSystem], phase_name: str, mode: str) -> List[SimSystem]:
    """Post-process phase results, pruning/retrying/raising on failure.

    Parameters
    ----------
    children
        Systems evaluated for the phase.
    phase_name
        Name of the phase being processed.
    mode
        Failure-handling strategy. ``"prune"`` removes failed systems, ``"retry"``
        clears success/failure sentinels so the systems rerun, and any other value
        raises an exception when failures are present.

    Returns
    -------
    list[SimSystem]
        Systems that should proceed to the next phase (possibly pruned).

    Raises
    ------
    RuntimeError
        If failures occur and ``mode`` is not ``"prune"``.
    """
    ok, bad = partition_children_by_status(children, phase_name)
    if bad:
        bad_names = ", ".join(c.meta.get("ligand", c.name) for c in bad)
        mode_lower = (mode or "").lower()
        if mode_lower == "prune":
            logger.warning(f"[{phase_name}] Pruning {len(bad)} ligand(s) that FAILED: {bad_names}")
            return ok
        if mode_lower == "retry":
            retried = []
            for c in bad:
                spec = _phase_spec(c.root, phase_name)
                removed_failure = _remove_patterns(c.root, spec.failure)
                removed_success = _remove_patterns(c.root, spec.success)
                if removed_failure or removed_success:
                    retried.append(c)
                else:
                    logger.warning(
                        "[%s] retry requested but no sentinels removed for %s",
                        phase_name,
                        c.meta.get("ligand", c.name),
                    )
            if retried:
                names = ", ".join(c.meta.get("ligand", c.name) for c in retried)
                logger.warning(f"[{phase_name}] Resetting failure state for {len(retried)} ligand(s): {names}")
            return ok + retried
        raise RuntimeError(f"[{phase_name}] {len(bad)} ligand(s) FAILED: {bad_names}")
    if not ok:
        raise RuntimeError(f"[{phase_name}] No ligands completed successfully.")
    return children


def filter_needing_phase(children: List[SimSystem], phase_name: str) -> List[SimSystem]:
    """Return only systems that still require work for the given phase.

    Parameters
    ----------
    children
        Candidate systems for the phase.
    phase_name
        Phase identifier whose completion criteria are checked.

    Returns
    -------
    list[SimSystem]
        Subset of systems lacking the necessary success sentinels.
    """
    need, done = [], []
    for child in children:
        if is_done(child, phase_name):
            done.append(child)
        else:
            need.append(child)
    if done:
        names = ", ".join(c.meta.get("ligand", c.name) for c in done)
        logger.debug(f"[skip] {phase_name}: {len(done)} ligand(s) already complete → {names}")
    return need


def run_phase_skipping_done(
    phase: Pipeline,
    children: List[SimSystem],
    phase_name: str,
    backend,
    *,
    max_workers: int | None = None,
    on_failure: str | None = None,
) -> bool:
    """Execute a phase for systems that still need it, skipping completed ones.

    Parameters
    ----------
    phase
        Pipeline configured for the phase.
    children
        Per-ligand systems targeted by the phase.
    phase_name
        Name of the phase (used for logging and state lookups).
    backend
        Execution backend implementing ``run_parallel``.
    max_workers
        Optional parallelism limit passed through to the backend.
    on_failure
        Failure-handling policy from the run config. When set to ``"prune"``
        or ``"retry"``, backend exceptions are logged and execution proceeds
        to sentinel-based pruning instead of raising immediately.

    Returns
    -------
    bool
        ``True`` if all systems were already complete (phase skipped),
        ``False`` otherwise.
    """
    todo = filter_needing_phase(children, phase_name)
    if not todo:
        logger.info(f"[skip] {phase_name}: all ligands already complete.")
        return True
    logger.info(
        f"{phase_name}: {len(todo)} ligand(s) not finished → running phase..."
        f"(of {len(children)} total)."
    )
    try:
        backend.run_parallel(phase, todo, description=phase_name, max_workers=max_workers)
    except Exception as exc:
        mode = (on_failure or "").lower()
        if mode in {"prune", "retry"}:
            logger.error(
                "[{}] backend reported failures (on_failure={}): {} — proceeding to sentinel-based pruning.",
                phase_name,
                mode,
                exc,
            )
            return False
        raise
    return False


def is_done(system: SimSystem, phase_name: str) -> bool:
    """Return ``True`` if a system satisfies the success criteria for a phase.

    Parameters
    ----------
    system
        Per-ligand :class:`SimSystem` descriptor.
    phase_name
        Name of the phase being checked.

    Returns
    -------
    bool
        ``True`` if the system meets the success specification, ``False`` otherwise.
    """
    spec = _phase_spec(system.root, phase_name)
    required_spec = spec.required or spec.success
    return _spec_satisfied(system.root, required_spec, phase_name)


def _phase_spec(root: Path, phase: str) -> PhaseState:
    """Fetch the phase specification for ``phase`` under ``root``."""

    return get_phase_state(root, phase)


def _spec_satisfied(root: Path, spec: List[List[str]], phase: str) -> bool:
    """Evaluate whether any clause in the DNF spec is satisfied on disk."""

    if not spec:
        return False
    progress = _load_progress(root, phase)
    updates: Dict[str, str] = {}
    comp_cache = components_under(root)
    win_cache: dict[str, List[int]] = {}
    for group in spec:
        paths: List[Path] = []
        for pattern in group:
            expanded = _expand_pattern(root, pattern, comp_cache, win_cache)
            if not expanded:
                paths = []
                break
            paths.extend(expanded)
        if paths and _all_exist_with_progress(root, paths, progress, updates):
            _write_progress(root, progress, updates, phase)
            return True
    if updates:
        _write_progress(root, progress, updates, phase)
    return False


def _expand_pattern(
    root: Path,
    pattern: str,
    comp_cache: List[str] | None = None,
    win_cache: dict[str, List[int]] | None = None,
) -> List[Path]:
    """Expand a single sentinel pattern, interpolating components and windows."""

    if "{comp" not in pattern and "{win" not in pattern:
        return [root / pattern]

    comps = comp_cache if comp_cache is not None else components_under(root)
    if not comps:
        return []

    expanded: List[Path] = []
    if "{win" in pattern:
        for comp in comps:
            cache = win_cache if win_cache is not None else {}
            if comp not in cache:
                cache[comp] = _production_windows_under(root, comp)
            wins = cache.get(comp, [])
            if not wins:
                return []
            for win in wins:
                expanded.append(root / pattern.format(comp=comp, win=win))
        return expanded

    for comp in comps:
        expanded.append(root / pattern.format(comp=comp))
    return expanded


def _production_windows_under(root: Path, comp: str) -> List[int]:
    """Return sorted production window indices for ``comp`` under ``root``."""

    base = root / "fe" / comp
    if not base.exists():
        return []
    out: List[int] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name == f"{comp}-1":
            continue
        if not name.startswith(comp):
            continue
        tail = name[len(comp):]
        try:
            idx = int(tail)
        except ValueError:
            continue
        if idx >= 0:
            out.append(idx)
    return sorted(out)


def _progress_path(root: Path, phase: str) -> Path:
    base = root / "artifacts"
    if phase.startswith("fe") or phase.startswith("prepare_fe"):
        base = root / "fe" / "artifacts"
    return base / "progress" / f"{phase}.csv"


def _load_progress(root: Path, phase: str) -> Dict[str, str]:
    path = _progress_path(root, phase)
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rel, state = line.split(",", 1)
            out[rel] = state
    except Exception:
        return {}
    return out


def _write_progress(
    root: Path, existing: Dict[str, str], updates: Dict[str, str], phase: str
) -> None:
    if not updates:
        return
    merged = dict(existing)
    merged.update(updates)
    path = _progress_path(root, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k},{v}\n" for k, v in sorted(merged.items())]
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text("".join(lines))
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _all_exist_with_progress(
    root: Path, paths: List[Path], progress: Dict[str, str], updates: Dict[str, str]
) -> bool:
    for p in paths:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.as_posix()
        state = progress.get(rel)
        if state:
            continue
        if p.exists():
            updates[rel] = "1"
            continue
        return False
    return True
