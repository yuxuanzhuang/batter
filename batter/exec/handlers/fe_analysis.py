# batter/exec/handlers/analyze.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem

from batter.analysis.analysis import analyze_lig_task
from batter.utils import components_under


def _production_window_indices(fe_root: Path, comp: str) -> List[int]:
    """
    Return sorted integer indices N for windows <ligand>/fe/<comp>/<compN> (N >= 0).
    (We intentionally skip the equil dir '<comp>-1'.)
    """
    base = fe_root / comp
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
        tail = name[len(comp):]  # e.g., "0", "1", ...
        try:
            idx = int(tail)
        except ValueError:
            continue
        if idx >= 0:
            out.append(idx)
    return sorted(out)


def _infer_component_windows_dict(fe_root: Path, components: List[str]) -> Dict[str, List[int]]:
    """
    Best-effort reconstruction of the windows list per component by directory scanning.
    If a component has no windows, returns an empty list for it.
    """
    d: Dict[str, List[int]] = {}
    for comp in components:
        d[comp] = _production_window_indices(fe_root, comp)
    return d


def analyze_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Run FE analysis for a ligand (single 'lig' rooted at <ligand>/fe).

    Expects the simulation phases to have produced the standard directory layout:
        <ligand>/fe/<comp>/<comp-1> (equil)
        <ligand>/fe/<comp>/<comp0>, <comp1>, ...

    Artifacts written by the analysis are returned (Results.dat, fe_timeseries.*).
    """
    lig = system.meta.get("ligand")
    mol = system.meta.get("residue_name")

    sim = params.get("sim", {}) or {}
    # Pull analysis settings, with safe defaults
    components: List[str] = list(sim.get("components") or components_under(system.root / "fe"))
    temperature: float = float(sim.get("temperature", 300.0))
    water_model: str = str(sim.get("water_model", "tip3p")).lower()
    rocklin_correction = bool(sim.get("rocklin_correction", False))
    n_workers: int = int(sim.get("n_workers", 4))

    # Optional: (start_idx, end_idx) subset of windows to analyze; else analyze all available
    sim_range = sim.get("sim_range", None)
    if sim_range is not None:
        try:
            sim_range = (int(sim_range[0]), int(sim_range[1]))
        except Exception:
            logger.warning(f"[analyze:{lig}] Ignoring invalid sim_range={sim_range!r}")
            sim_range = None

    # Optional: restraint tuple for Boresch analytical term (k’s etc.)
    # Default to zeros if not provided or not applicable.
    rest: Tuple[float, float, float, float, float] = tuple(sim.get("rest", (0, 0, 0, 0, 0)))  # type: ignore

    fe_root = system.root / "fe"
    if not fe_root.exists():
        raise FileNotFoundError(f"[analyze:{lig}] Missing FE folder: {fe_root}")

    # Try to reconstruct windows per component if the pipeline didn’t inject it
    component_windows_dict = params.get("component_windows_dict")
    if not component_windows_dict:
        component_windows_dict = _infer_component_windows_dict(fe_root, components)

    # Build the input dict expected by your analyze_* helpers
    input_dict = {
        "fe_folder": str(fe_root),
        "components": components,
        "rest": rest,
        "temperature": temperature,
        "water_model": water_model,
        "rocklin_correction": rocklin_correction,
        "component_windows_dict": component_windows_dict,
        "sim_range": sim_range,
        "raise_on_error": True,
        "n_workers": n_workers,
    }

    logger.debug(f"[analyze:{lig}] Starting FE analysis "
                f"(components={components}, T={temperature}K, rocklin={rocklin_correction}, mol={mol})")

    # We treat the ligand’s fe/ folder itself as “lig” root (lig = ".")
    try:
        analyze_lig_task(
            fe_folder=str(fe_root),
            lig='.',
            components=components,
            rest=rest,
            temperature=temperature,
            water_model=water_model,
            component_windows_dict=component_windows_dict,
            rocklin_correction=rocklin_correction,
            sim_range=sim_range,
            raise_on_error=True,
            mol=mol,
            n_workers=n_workers,
        )
    except Exception as e:
        logger.error(f"[analyze:{lig}] Analysis failed: {e}")
        raise

    # Collect artifacts
    results_dir = fe_root / "Results"
    arts: Dict[str, Path] = {}
    if results_dir.exists():
        res_file = results_dir / "Results.dat"
        if res_file.exists():
            arts["results_dat"] = res_file
        ts_json = results_dir / "fe_timeseries.json"
        if ts_json.exists():
            arts["fe_timeseries_json"] = ts_json
        ts_png = results_dir / "fe_timeseries.png"
        if ts_png.exists():
            arts["fe_timeseries_png"] = ts_png

    analyzed_finished = fe_root / "artifacts" /  "analyze.ok"
    open(analyzed_finished, "w").close()

    logger.debug(f"[analyze:{lig}] FE analysis done. Artifacts: {', '.join(p.name for p in arts.values()) or 'none'}")
    return ExecResult(job_ids=[], artifacts=arts)