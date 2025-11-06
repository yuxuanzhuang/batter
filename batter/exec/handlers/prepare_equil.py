from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import os
from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.config.simulation import SimulationConfig
from batter._internal.builders.equil import PrepareEquilBuilder
from batter.orchestrate.state_registry import register_phase_state


def prepare_equil_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Pipeline handler for the `prepare_equil` step.

    - Reads artifacts/ligand_params/index.json to find per-ligand parameter store
    - Validates SimulationConfig from params["sim"]
    - Invokes the PrepareEquilBuilder
    """
    # 1) Parse sim config
    sim = SimulationConfig.model_validate(params["sim"])

    # 2) Resolve ligand name (e.g., folder name under ligands/)
    ligand = system.meta["ligand"]
    residue_name = system.meta["residue_name"]
    working_dir: Path = system.root / "equil"
    comp_windows: dict = params.get("component_windows", {})
    extra_restraints: Optional[dict] = params['sys_params'].get("extra_restraints", None)
    extra_restraints_fc: float = float(params['sys_params'].get("extra_restraints_fc", 10.0))
    extra_conformation_restraints: Optional[Path] = params['sys_params'].get("extra_conformation_restraints", None)
    
    infe = False
    if extra_restraints is not None:
        infe = False
        sim.barostat = '1'
    if extra_conformation_restraints is not None:
        infe = True
        # cannot do NFE with barostat 1 (Berendsen)
        sim.barostat = '2'
    
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
            "extra_restraints_fc": extra_restraints_fc,
            "extra_conformation_restraints": extra_conformation_restraints,
        }
    )
    ok = builder.build()
    if not ok:
        raise RuntimeError(f"[prepare_equil] anchor detection failed for ligand={ligand}")

    os.makedirs(system_root / "equil" / "artifacts", exist_ok=True)
    prepare_finished = system_root / "equil" / "artifacts" / "prepare_equil.ok"
    open(prepare_finished, "w").close()

    prepare_rel = prepare_finished.relative_to(system.root).as_posix()
    full_prmtop = (system_root / "equil" / "full.prmtop").relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "prepare_equil",
        required=[[full_prmtop, prepare_rel]],
        success=[[full_prmtop, prepare_rel]],
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
