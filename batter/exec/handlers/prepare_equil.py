"""Prepare equilibration inputs for a ligand."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from batter._internal.builders.equil import PrepareEquilBuilder
from batter.config.simulation import SimulationConfig
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem


def prepare_equil_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """Build equilibration inputs for the current ligand.

    Parameters
    ----------
    step : Step
        Pipeline step metadata (unused).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Contains the output directory and any generated metadata.
    """
    # 1) Parse sim config
    payload = StepPayload.model_validate(params)
    if payload.sim is None:
        raise ValueError("[prepare_equil] Missing simulation configuration in payload.")
    sim = payload.sim
    partition = payload.get("partition") or payload.get("queue") or "normal"

    # 2) Resolve ligand name (e.g., folder name under ligands/)
    ligand = system.meta["ligand"]
    residue_name = system.meta["residue_name"]
    working_dir: Path = system.root / "equil"
    comp_windows: dict = payload.get("component_windows", {})
    sys_params = payload.sys_params or SystemParams()
    extra_restraints: Optional[str] = sys_params.get("extra_restraints", None)
    extra_restraint_fc = float(sys_params.get("extra_restraint_fc", 10.0))
    extra_conformation_restraints: Optional[Path] = sys_params.get(
        "extra_conformation_restraints", None
    )
    
    infe = bool(sim.infe)
    
    system_root = system.root

    # 3) Read parameter index (produced by param_ligands)
    param_dir_dict = system.meta['param_dir_dict']

    logger.debug(
        f"[prepare_equil] start for ligand={ligand} "
        f"| residue={residue_name} | workdir={working_dir}"
    )

    # 4) Build equilibration system
    builder = PrepareEquilBuilder(
        ligand=ligand,
        sim_config=sim,
        component_windows_dict=comp_windows,
        working_dir=working_dir,
        infe=infe,
        system_root=system_root.parents[1],
        residue_name=residue_name,
        param_dir_dict=param_dir_dict,
        extra={
            "extra_restraints": extra_restraints,
            "extra_restraint_fc": extra_restraint_fc,
            "extra_conformation_restraints": extra_conformation_restraints,
            "partition": partition,
        }
    )
    def _mark_prepare_equil_failed() -> None:
        equil_dir = system_root / "equil"
        os.makedirs(equil_dir, exist_ok=True)
        (equil_dir / "prepare_equil.failed").touch()

    try:
        ok = builder.build()
    except Exception:
        _mark_prepare_equil_failed()
        raise
    if not ok:
        _mark_prepare_equil_failed()
        raise RuntimeError(f"[prepare_equil] anchor detection failed for ligand={ligand}")

    os.makedirs(system_root / "equil", exist_ok=True)
    prepare_finished = system_root / "equil" / "prepare_equil.ok"
    open(prepare_finished, "w").close()

    prepare_rel = prepare_finished.relative_to(system.root).as_posix()
    full_prmtop = (system_root / "equil" / "full.prmtop").relative_to(system.root).as_posix()
    failed_rel = (system_root / "equil" / "prepare_equil.failed").relative_to(
        system.root
    ).as_posix()
    register_phase_state(
        system.root,
        "prepare_equil",
        required=[[full_prmtop, prepare_rel], [failed_rel]],
        success=[[full_prmtop, prepare_rel]],
        failure=[[failed_rel]],
    )

    logger.debug(f"[prepare_equil] finished for ligand={ligand}")

    return ExecResult(
        [],
        {
            "ligand": ligand,
            "prepared": True,
            "residue_name": residue_name,
            "workdir": str(working_dir),
        },
    )
