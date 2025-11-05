from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from loguru import logger

from batter.systems.core import SimSystem
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step
from batter.utils import components_under

REQUIRED_MARKERS = {
    "system_prep": [["artifacts/config/sim_overrides.json"]],
    "param_ligands": [["artifacts/ligand_params/index.json"]],
    "prepare_equil": [["equil/full.prmtop", "equil/artifacts/prepare_equil.ok"]],
    "equil": [["equil/FINISHED"], ["equil/FAILED"]],
    "equil_analysis": [["equil/representative.pdb"], ["equil/UNBOUND"]],
    "prepare_fe": [["fe/artifacts/prepare_fe.ok", "fe/artifacts/prepare_fe_windows.ok"]],
    "fe_equil": [["fe/{comp}/{comp}-1/EQ_FINISHED"]],
    "fe": [["fe/{comp}/{comp}{win:02d}/FINISHED"]],
    "analyze": [["fe/artifacts/analyze.ok"]],
}

SUCCESS_MARKERS = {
    "system_prep": [["artifacts/config/sim_overrides.json"]],
    "param_ligands": [["artifacts/ligand_params/index.json"]],
    "prepare_equil": [["equil/full.prmtop", "equil/artifacts/prepare_equil.ok"]],
    "equil": [["equil/FINISHED"]],
    "equil_analysis": [["equil/representative.pdb"], ["equil/UNBOUND"]],
    "prepare_fe": [["fe/artifacts/prepare_fe.ok", "fe/artifacts/prepare_fe_windows.ok"]],
    "fe_equil": [["fe/{comp}/{comp}-1/EQ_FINISHED"]],
    "fe": [["fe/{comp}/{comp}{win:02d}/FINISHED"]],
    "analyze": [["fe/artifacts/analyze.ok"]],
}

FAILURE_MARKERS = {
    "equil": [["equil/FAILED"]],
    "fe_equil": [["fe/{comp}/{comp}-1/FAILED"]],
    "fe": [["fe/{comp}/{comp}{win:02d}/FAILED"]],
}


def partition_children_by_status(children: List[SimSystem], phase: str) -> Tuple[List[SimSystem], List[SimSystem]]:
    ok, bad = [], []
    succ = SUCCESS_MARKERS.get(phase, [])
    fail = FAILURE_MARKERS.get(phase, [])
    for child in children:
        root = child.root
        is_success = _expand_and_check(root, succ)
        if is_success:
            ok.append(child)
            continue
        is_failure = _expand_and_check(root, fail) if fail else False
        if is_failure or not is_success:
            bad.append(child)
    return ok, bad


def handle_phase_failures(children: List[SimSystem], phase: str, mode: str) -> List[SimSystem]:
    ok, bad = partition_children_by_status(children, phase)
    if bad:
        bad_names = ", ".join(c.meta.get("ligand", c.name) for c in bad)
        if (mode or "").lower() == "prune":
            logger.warning(f"[{phase}] Pruning {len(bad)} ligand(s) that FAILED: {bad_names}")
            return ok
        raise RuntimeError(f"[{phase}] {len(bad)} ligand(s) FAILED: {bad_names}")
    if not ok:
        raise RuntimeError(f"[{phase}] No ligands completed successfully.")
    return children


def filter_needing_phase(children: List[SimSystem], phase_name: str) -> List[SimSystem]:
    if phase_name not in REQUIRED_MARKERS:
        return list(children)
    need = [c for c in children if not is_done(c, phase_name)]
    done = [c for c in children if c not in need]
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
) -> bool:
    todo = filter_needing_phase(children, phase_name)
    if not todo:
        logger.info(f"[skip] {phase_name}: all ligands already complete.")
        return True
    logger.info(
        f"{phase_name}: {len(todo)} ligand(s) not finished → running phase..."
        f"(of {len(children)} total)."
    )
    backend.run_parallel(phase, todo, description=phase_name, max_workers=max_workers)
    return False


def is_done(system: SimSystem, phase_name: str) -> bool:
    root = system.root
    spec = REQUIRED_MARKERS.get(phase_name, [])
    if not spec:
        return False

    if phase_name not in {"fe_equil", "fe"}:
        return _dnf_satisfied(root, spec)

    comps = components_under(root)
    if not comps:
        return False

    if phase_name == "fe_equil":
        for comp in comps:
            expanded_groups = []
            for group in spec:
                expanded_groups.append([p.format(comp=comp, win="") for p in group])
            if not _dnf_satisfied(root, expanded_groups):
                return False
        return True

    if phase_name == "fe":
        for comp in comps:
            wins = _production_windows_under(root, comp)
            if not wins:
                return False
            for win in wins:
                expanded_groups = []
                for group in spec:
                    expanded_groups.append([p.format(comp=comp, win=win) for p in group])
                if not _dnf_satisfied(root, expanded_groups):
                    return False
        return True

    return False


def _expand_and_check(root: Path, spec) -> bool:
    if not spec:
        return False
    if all(isinstance(g, list) and all("{" not in s for s in g) for g in spec):
        return _dnf_satisfied(root, spec)
    comps = components_under(root) or []
    for group in spec:
        required: List[Path] = []
        for patt in group:
            if "{comp}" in patt or "{win" in patt:
                if not comps:
                    required.append(root / patt.format(comp="", win=0))
                else:
                    for comp in comps:
                        wins = _production_windows_under(root, comp) if "{win" in patt else [None]
                        for w in wins:
                            required.append(root / patt.format(comp=comp, win=(0 if w is None else w)))
            else:
                required.append(root / patt)
        if all(p.exists() for p in required):
            return True
    return False


def _production_windows_under(root: Path, comp: str) -> List[int]:
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


def _dnf_satisfied(root: Path, marker_spec) -> bool:
    if not marker_spec:
        return False

    if all(isinstance(m, str) for m in marker_spec):
        return any((root / m).exists() for m in marker_spec)

    for group in marker_spec:
        if all((root / p).exists() for p in group):
            return True

    return False
