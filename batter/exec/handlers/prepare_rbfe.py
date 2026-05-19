"""Prepare planned RBFE network and atom-mapping artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from loguru import logger

from batter.config.run import RBFENetworkArgs
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem


def _rbfe_config_from_payload(
    payload: StepPayload,
    sys_params: SystemParams,
) -> RBFENetworkArgs:
    rbfe_raw = sys_params.get("rbfe", None) or payload.get("rbfe")
    if isinstance(rbfe_raw, RBFENetworkArgs):
        return rbfe_raw
    if rbfe_raw:
        return RBFENetworkArgs.model_validate(rbfe_raw)
    return RBFENetworkArgs()


def prepare_rbfe_handler(
    step: Step, system: SimSystem, params: Dict[str, Any]
) -> ExecResult:
    """Build the run-scoped RBFE plan before per-ligand equilibration."""
    payload = StepPayload.model_validate(params)
    sys_params = payload.sys_params or SystemParams()
    lig_map = {
        str(name): Path(path)
        for name, path in sys_params.ligand_paths.items()
    }
    if len(lig_map) < 2:
        raise RuntimeError("[prepare_rbfe] RBFE requires at least two staged ligands.")

    rbfe_cfg = _rbfe_config_from_payload(payload, sys_params)
    config_dir = system.root / "artifacts" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    from batter.orchestrate.run import _build_rbfe_network_plan

    planned = _build_rbfe_network_plan(
        list(lig_map.keys()),
        lig_map,
        rbfe_cfg,
        config_dir,
    )

    marker = config_dir / "prepare_rbfe.ok"
    marker.write_text("ok\n")
    network_rel = (config_dir / "rbfe_network.json").relative_to(system.root).as_posix()
    marker_rel = marker.relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "prepare_rbfe",
        required=[[network_rel, marker_rel]],
        success=[[network_rel, marker_rel]],
        failure=[],
    )

    logger.debug(
        f"[prepare_rbfe] planned {len(planned.get('pairs') or [])} "
        f"RBFE transformation(s) under {config_dir}"
    )
    return ExecResult(
        job_ids=[],
        artifacts={
            "rbfe_network": str(config_dir / "rbfe_network.json"),
            "rbfe_network_html": str(config_dir / "rbfe_network.html"),
            "prepare_rbfe_ok": str(marker),
        },
    )
