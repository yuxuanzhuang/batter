from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.config.simulation import SimulationConfig
from batter._internal.builders.equil import PrepareEquilBuilder


def _find_ligand_entry(ligand: str, index_json: Path) -> Optional[Dict[str, Any]]:
    """Read artifacts/ligand_params/index.json and return matching ligand entry."""
    if not index_json.exists():
        raise FileNotFoundError(f"[prepare_equil] Missing ligand param index: {index_json}")
    index = json.loads(index_json.read_text())
    for entry in index.get("ligands", []):
        if entry["ligand"] == ligand:
            return entry
    return None


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
    ligand = system.meta.get("ligand", Path(system.root).name)
    working_dir: Path = system.root / "equil"
    comp_windows: dict = params.get("component_windows", {})
    infe: bool = bool(params.get("infe", False))

    # 3) Locate parent system root (…/work/<system>)
    try:
        system_root = system.root.parents[1]
    except Exception:
        system_root = system.root

    # 4) Read parameter index (produced by param_ligands)
    index_json = system_root / "artifacts" / "ligand_params" / "index.json"
    param_dir_dict = {}
    try:
        index_data = json.loads(index_json.read_text())
        for entry in index_data.get("ligands", []):
            store_dir = entry.get("store_dir")
            resn = entry.get("residue_name")
            param_dir_dict[resn] = store_dir
    except Exception as e:
        raise RuntimeError(f"Failed to parse ligand param index {index_json}: {e}")

    entry = _find_ligand_entry(ligand, index_json)
    if entry is None:
        raise FileNotFoundError(
            f"[prepare_equil] Ligand '{ligand}' not found in {index_json}. "
            "Ensure param_ligands ran successfully."
        )

    residue_name = entry.get("residue_name", ligand[:3])

    logger.info(
        f"[prepare_equil] start for ligand={ligand} "
        f"| residue={residue_name} | workdir={working_dir}"
    )

    # 5) Build equilibration system
    builder = PrepareEquilBuilder(
        ligand=ligand,
        sim_config=sim,
        component_windows_dict=comp_windows,
        working_dir=working_dir,
        infe=infe,
        system_root=system_root,
        residue_name=residue_name,
        param_dir_dict=param_dir_dict,
    )
    ok = builder.build()
    if not ok:
        raise RuntimeError(f"[prepare_equil] anchor detection failed for ligand={ligand}")

    logger.info(f"[prepare_equil] finished for ligand={ligand}")

    return ExecResult(
        [],
        {
            "ligand": ligand,
            "prepared": True,
            "residue_name": residue_name,
            "workdir": str(working_dir),
        },
    )