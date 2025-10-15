# batter/exec/handlers/prepare_fe.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Type

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.config.simulation import SimulationConfig

from batter._internal.builders.fe_alchemical import AlchemicalFEBuilder

# -----------------------------
# helpers
# -----------------------------
def _system_root_for(child_root: Path) -> Path:
    """work/<sys>/simulations/<lig> → work/<sys>"""
    try:
        return child_root.parents[1]
    except Exception:
        return child_root

def _load_param_dir_dict(system_root: Path) -> Dict[str, str]:
    """
    Read artifacts/ligand_params/index.json → {residue_name: store_dir}
    """
    index_json = system_root / "artifacts" / "ligand_params" / "index.json"
    data = json.loads(index_json.read_text())
    out: Dict[str, str] = {}
    for entry in data.get("ligands", []):
        resn = entry.get("residue_name")
        store_dir = entry.get("store_dir")
        if resn and store_dir:
            out[resn] = store_dir
    if not out:
        raise RuntimeError(f"No ligand param entries found in {index_json}")
    return out

# -----------------------------
# prepare_fe (scaffolding / amber templates)
# -----------------------------
def prepare_fe_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Prepare per-ligand FE scaffolding (no window expansion here):
      - builds the initial FE directory for each requested component
      - writes artifacts/prepare_fe/prepare_fe.ok on success

    Notes
    -----
    - Windows are created in the separate 'prepare_fe_windows' step.
    - Builders internally call the shared `write_run_file()` to materialize run scripts.
    """
    # 1) Parse sim config + components
    sim = SimulationConfig.model_validate(params["sim"])
    components = list(getattr(sim, "components", []) or [])
    if not components:
        # fall back to a single 'z' if not specified (optional)
        components = ["z"]

    ligand = (system.meta or {}).get("ligand")
    residue_name = (system.meta or {}).get("residue_name")
    if not ligand or not residue_name:
        raise ValueError("System meta must include 'ligand' and 'residue_name'.")

    child_root = system.root
    system_root = _system_root_for(child_root)
    param_dir_dict = _load_param_dir_dict(system_root)

    # You can pass component-specific lambda schedules via params["component_windows"]
    comp_windows: dict = params.get("component_windows", {}) or {}
    infe: bool = bool(params.get("infe", False))

    artifacts: Dict[str, Any] = {}
    logger.info(f"[prepare_fe] start ligand={ligand} residue={residue_name} components={components}")

    # Build per component (scaffold / templates only; win=-1)
    for comp in components:
        workdir = child_root / "fe" / comp
        workdir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[prepare_fe] building component '{comp}' in {workdir}")
        builder = AlchemicalFEBuilder(
            ligand=ligand,
            residue_name=residue_name,
            param_dir_dict=param_dir_dict,
            sim_config=sim,
            component=comp,
            component_windows=comp_windows.get(comp, []),
            working_dir=workdir,
            system_root=system_root,
            win=-1,                              # scaffold/init pass
            extra={"infe": infe},
        )
        builder.build()  # will create <comp>-1, amber templates, run files, etc.

        artifacts[f"{comp}_workdir"] = str(workdir)

    # emit the common OK marker used by the orchestrator
    marker = system_root / "artifacts" / "prepare_fe" / "prepare_fe.ok"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")

    logger.success(f"[prepare_fe] finished ligand={ligand} → {marker}")
    return ExecResult(job_ids=[], artifacts={"prepare_fe_ok": marker, **artifacts})


# -----------------------------
# prepare_fe_windows (expand per-lambda windows)
# -----------------------------
def prepare_fe_windows_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Expand FE windows for each requested component:
      - copies <comp>-1 to <comp>-2, <comp>-3, ... (depending on lambda schedule)
      - keeps run scripts consistent in each window (builders call write_run_file)
      - writes artifacts/fe/windows.json summarizing windows

    Builders re-use the same interface; here we just iterate components and request
    per-window builds by calling with win >= 1.
    """
    sim = SimulationConfig.model_validate(params["sim"])
    components = list(getattr(sim, "components", []) or [])
    if not components:
        components = ["z"]

    ligand = (system.meta or {}).get("ligand")
    residue_name = (system.meta or {}).get("residue_name")
    if not ligand or not residue_name:
        raise ValueError("System meta must include 'ligand' and 'residue_name'.")

    child_root = system.root
    system_root = _system_root_for(child_root)
    param_dir_dict = _load_param_dir_dict(system_root)

    comp_windows: dict = params.get("component_windows", {}) or {}
    infe: bool = bool(params.get("infe", False))

    windows_summary: Dict[str, Any] = {}
    logger.info(f"[prepare_fe_windows] start ligand={ligand} residue={residue_name} components={components}")

    for comp in components:
        lambdas = list(comp_windows.get(comp, []))
        if not lambdas:
            logger.warning(f"[prepare_fe_windows] component '{comp}' has no lambda schedule — skipping window expansion.")
            windows_summary[comp] = {"n_windows": 0, "lambdas": []}
            continue

        workdir = child_root / "fe" / comp

        for win_idx, _ in enumerate(lambdas):
            logger.debug(f"[prepare_fe_windows] {comp} → creating window {win_idx} in {workdir}")
            builder = AlchemicalFEBuilder(
                ligand=ligand,
                residue_name=residue_name,
                param_dir_dict=param_dir_dict,
                sim_config=sim,
                component=comp,
                component_windows=lambdas,
                working_dir=workdir,
                system_root=system_root,
                win=win_idx,
                extra={"infe": infe},
            )
            builder.build()

        windows_summary[comp] = {"n_windows": len(lambdas), "lambdas": lambdas}

    # write a canonical windows.json under artifacts/fe/
    windows_json = system_root / "artifacts" / "fe" / "windows.json"
    windows_json.parent.mkdir(parents=True, exist_ok=True)
    windows_json.write_text(json.dumps(windows_summary, indent=2) + "\n")

    logger.success(f"[prepare_fe_windows] finished ligand={ligand} → {windows_json}")
    return ExecResult(job_ids=[], artifacts={"windows_json": windows_json})