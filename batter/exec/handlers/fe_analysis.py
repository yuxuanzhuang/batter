"""Run post-processing analysis on free-energy simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from loguru import logger

from batter.analysis.analysis import analyze_lig_task
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
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
    """Run FE analysis for a ligand rooted at ``<system.root>/fe``.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Mapping with the generated ``Results.dat`` and optional timeseries
        artefacts.
    """
    lig = system.meta.get("ligand")
    mol = system.meta.get("residue_name")

    payload = StepPayload.model_validate(params)
    sim_cfg = payload.sim

    fe_root = system.root / "fe"
    if not fe_root.exists():
        raise FileNotFoundError(f"[analyze:{lig}] Missing FE folder: {fe_root}")

    default_components = components_under(fe_root)
    components: List[str] = list(default_components)
    temperature: float = 310.0
    water_model: str = "tip3p"
    rocklin_correction: bool = False
    n_workers: int = 4
    rest: Tuple[str, ...] = tuple()
    sim_range_default: Optional[Tuple[int, int]] = None

    if sim_cfg is not None:
        if sim_cfg.components:
            components = list(sim_cfg.components)
        temperature = float(sim_cfg.temperature)
        water_model = str(sim_cfg.water_model).lower()
        rocklin_correction = bool(sim_cfg.rocklin_correction)
        rest = tuple(sim_cfg.rest)
        n_workers = int(getattr(sim_cfg, "analysis_n_workers", n_workers))
        sim_range_default = getattr(sim_cfg, "analysis_fe_range", None)

    components = list(payload.get("components", components))
    temperature = float(payload.get("temperature", temperature))
    water_model = str(payload.get("water_model", water_model)).lower()
    rocklin_correction = bool(payload.get("rocklin_correction", rocklin_correction))
    n_workers = int(payload.get("n_workers", n_workers))

    # Optional: (start_idx, end_idx) subset of windows to analyze; else analyze all available
    sim_range = payload.get("sim_range", sim_range_default)
    if sim_range is not None:
        try:
            sim_range = (int(sim_range[0]), int(sim_range[1]))
        except Exception:
            logger.warning(f"[analyze:{lig}] Ignoring invalid sim_range={sim_range!r}")
            sim_range = None

    # Try to reconstruct windows per component if the pipeline didn’t inject it
    component_windows_dict = payload.get("component_windows_dict")
    if not component_windows_dict:
        component_windows_dict = _infer_component_windows_dict(fe_root, components)


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

    analyzed_finished = fe_root / "artifacts" / "analyze.ok"
    open(analyzed_finished, "w").close()

    analyze_rel = analyzed_finished.relative_to(system.root).as_posix()
    results_rel = (results_dir / "Results.dat").relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "analyze",
        required=[[analyze_rel, results_rel]],
        success=[[analyze_rel, results_rel]],
    )

    logger.debug(f"[analyze:{lig}] FE analysis done. Artifacts: {', '.join(p.name for p in arts.values()) or 'none'}")
    return ExecResult(job_ids=[], artifacts=arts)
