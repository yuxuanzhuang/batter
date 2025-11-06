"""Prepare alchemical FE inputs for a ligand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Type

from loguru import logger

from batter._internal.builders.fe_alchemical import AlchemicalFEBuilder
from batter.config.simulation import SimulationConfig
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem

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
    """Construct the initial FE directory layout for a ligand.

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
        Metadata describing the generated directories.
    """
    # 1) Parse sim config + components
    payload = StepPayload.model_validate(params)
    if payload.sim is None:
        raise ValueError("[prepare_fe] Missing simulation configuration in payload.")
    sim = payload.sim
    components = list(getattr(sim, "components", []) or [])
    if not components:
        raise ValueError("No components specified in sim config.")

    ligand = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    if not ligand or not residue_name:
        raise ValueError("System meta must include 'ligand' and 'residue_name'.")

    child_root = system.root
    system_root = _system_root_for(child_root)
    param_dir_dict = _load_param_dir_dict(system_root)

    comp_windows: dict = sim.component_lambdas  # type: ignore[attr-defined]
    sys_params = payload.sys_params or SystemParams()
    extra_restraints: Optional[dict] = sys_params.get("extra_restraints", None)
    extra_restraints_fc: float = float(sys_params.get("extra_restraints_fc", 10.0))
    extra_conformation_restraints: Optional[Path] = sys_params.get("extra_conformation_restraints", None)

    infe = False
    if extra_restraints is not None:
        infe = False
        sim.barostat = '1'
    if extra_conformation_restraints is not None:
        infe = True
        # cannot do NFE with barostat 1 (Berendsen)
        sim.barostat = '2'
    
    artifacts: Dict[str, Any] = {}
    logger.debug(f"[prepare_fe] start ligand={ligand} residue={residue_name} components={components}")

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
            component_windows=comp_windows[comp],
            working_dir=workdir,
            system_root=system_root,
            infe=infe,
            win=-1,
            extra={
            "extra_restraints": extra_restraints,
            "extra_restraints_fc": extra_restraints_fc,
            "extra_conformation_restraints": extra_conformation_restraints,
            }
        )
        builder.build()  # will create <comp>-1, amber templates, run files, etc.

        artifacts[f"{comp}_workdir"] = str(workdir)

    # emit the common OK marker used by the orchestrator
    marker = child_root / "fe" / "artifacts" / "prepare_fe.ok"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")

    logger.debug(f"[prepare_fe] finished ligand={ligand} → {marker}")
    marker_rel = marker.relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "prepare_fe",
        required=[[marker_rel]],
        success=[[marker_rel]],
    )
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
    payload = StepPayload.model_validate(params)
    if payload.sim is None:
        raise ValueError("[prepare_fe_windows] Missing simulation configuration in payload.")
    sim = payload.sim
    components = list(getattr(sim, "components", []) or [])
    if not components:
        raise RuntimeError("No components specified in sim config for FE window preparation.")

    ligand = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    if not ligand or not residue_name:
        raise ValueError("System meta must include 'ligand' and 'residue_name'.")

    child_root = system.root
    system_root = _system_root_for(child_root)

    param_dir_dict = _load_param_dir_dict(system_root)

    comp_windows: dict = payload.get("component_lambdas") or sim.component_lambdas  # type: ignore[attr-defined]
    sys_params = payload.sys_params or SystemParams()
    extra_restraints: Optional[dict] = sys_params.get("extra_restraints", None)
    extra_restraints_fc: float = float(sys_params.get("extra_restraints_fc", 10.0))
    extra_conformation_restraints: Optional[Path] = sys_params.get("extra_conformation_restraints", None)

    infe = False
    if extra_restraints is not None:
        infe = False
        sim.barostat = '1'
    if extra_conformation_restraints is not None:
        infe = True
        # cannot do NFE with barostat 1 (Berendsen)
        sim.barostat = '2'
    
    windows_summary: Dict[str, Any] = {}
    logger.debug(f"[prepare_fe_windows] start ligand={ligand} residue={residue_name} components={components}")

    for comp in components:
        workdir = child_root / "fe" / comp
        lambdas = comp_windows[comp]

        for win_idx, _ in enumerate(lambdas):
            logger.debug(f"[prepare_fe_windows] {comp} → creating window {win_idx} in {workdir}")
            builder = AlchemicalFEBuilder(
                ligand=ligand,
                residue_name=residue_name,
                param_dir_dict=param_dir_dict,
                sim_config=sim,
                component=comp,
                infe=infe,
                component_windows=lambdas,
                working_dir=workdir,
                system_root=system_root,
                win=win_idx,
                extra={
                    "extra_restraints": extra_restraints,
                    "extra_restraints_fc": extra_restraints_fc,
                    "extra_conformation_restraints": extra_conformation_restraints,
                }
        )
            builder.build()

        windows_summary[comp] = {"n_windows": len(lambdas), "lambdas": lambdas}

    # write a canonical windows.json under artifacts/fe/
    windows_json = child_root / "fe" / "artifacts" / "windows.json"
    windows_json.parent.mkdir(parents=True, exist_ok=True)
    windows_json.write_text(json.dumps(windows_summary, indent=2) + "\n")

    prepare_finished = child_root / "fe" / "artifacts" / "prepare_fe_windows.ok"
    open(prepare_finished, "w").close()

    windows_rel = prepare_finished.relative_to(system.root).as_posix()
    prepare_rel = (child_root / "fe" / "artifacts" / "prepare_fe.ok").relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "prepare_fe",
        required=[[prepare_rel, windows_rel]],
        success=[[prepare_rel, windows_rel]],
    )

    logger.debug(f"[prepare_fe_windows] finished ligand={ligand} → {windows_json}")
    return ExecResult(job_ids=[], artifacts={"windows_json": windows_json})
