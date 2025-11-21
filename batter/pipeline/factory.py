from __future__ import annotations

from typing import List, Optional

from batter.config.simulation import SimulationConfig

from .payloads import StepPayload, SystemParams
from .pipeline import Pipeline
from .step import Step

__all__ = ["make_abfe_pipeline", "make_asfe_pipeline", "make_md_pipeline"]


def _step(
    name: str,
    requires: Optional[List[str]] = None,
    *,
    sim: Optional[SimulationConfig] = None,
    sys_params: Optional[SystemParams] = None,
    **extra,
) -> Step:
    """Small helper to keep steps consistent."""
    payload = StepPayload(sim=sim, sys_params=sys_params, **extra)
    return Step(name=name, requires=requires or [], payload=payload)


def make_abfe_pipeline(
    sim: SimulationConfig,
    sys_params: SystemParams | dict | None,
    only_fe_preparation: bool = False,
) -> Pipeline:
    """
    ABFE pipeline:

    system_prep → param_ligands → prepare_equil → equil → equil_analysis
    → prepare_fe → prepare_fe_windows → fe_equil → fe → analyze
    """
    steps: List[Step] = []
    params_model = (
        sys_params
        if isinstance(sys_params, SystemParams)
        else SystemParams.model_validate(sys_params or {})
    )

    # 0) system prep — runs once at system root
    steps.append(
        _step(
            name="system_prep",
            requires=[],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            name="param_ligands",
            requires=["system_prep"],
            sim=sim,
            sys_params=params_model,
        )
    )

    # Per-ligand steps
    steps.append(
        _step(
            "prepare_equil",
            requires=["param_ligands"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "equil",
            requires=["prepare_equil"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "equil_analysis",
            requires=["equil"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "prepare_fe",
            requires=["equil_analysis"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "prepare_fe_windows",
            requires=["prepare_fe"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "fe_equil",
            requires=["prepare_fe_windows"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "fe",
            requires=["fe_equil"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "analyze",
            requires=["fe"],
            sim=sim,
            sys_params=params_model,
        )
    )

    if only_fe_preparation:
        keep = {
            "system_prep",
            "param_ligands",
            "prepare_equil",
            "equil",
            "equil_analysis",
            "prepare_fe",
            "prepare_fe_windows",
            "fe_equil"
        }
        steps = [s for s in steps if s.name in keep]

    return Pipeline(steps)


def make_asfe_pipeline(
    sim: SimulationConfig,
    sys_params: SystemParams | dict | None,
    only_fe_preparation: bool = False,
) -> Pipeline:
    """
    ASFE pipeline:

    param_ligands → prepare_fe → prepare_fe_windows
    → fe_equil → fe → analyze
    """
    steps: List[Step] = []
    params_model = (
        sys_params
        if isinstance(sys_params, SystemParams)
        else SystemParams.model_validate(sys_params or {})
    )
    steps.append(
        _step(
            name="system_prep_asfe",
            requires=[],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            name="param_ligands",
            requires=["system_prep_asfe"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "prepare_fe",
            requires=["param_ligands"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "prepare_fe_windows",
            requires=["prepare_fe"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "fe_equil",
            requires=["prepare_fe_windows"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "fe",
            requires=["fe_equil"],
            sim=sim,
            sys_params=params_model,
        )
    )
    steps.append(
        _step(
            "analyze",
            requires=["fe"],
            sim=sim,
            sys_params=params_model,
        )
    )
    if only_fe_preparation:
        keep = {
            "system_prep_asfe",
            "param_ligands",
            "prepare_fe",
            "prepare_fe_windows",
            "fe_equil",
        }
        steps = [s for s in steps if s.name in keep]
    return Pipeline(steps)


def make_md_pipeline(
    sim: SimulationConfig,
    sys_params: SystemParams | dict | None,
    only_fe_preparation: bool = False,
) -> Pipeline:
    """
    MD-only pipeline focused on equilibration:

    system_prep → param_ligands → prepare_equil → equil → equil_analysis
    """
    params_model = (
        sys_params
        if isinstance(sys_params, SystemParams)
        else SystemParams.model_validate(sys_params or {})
    )
    steps: List[Step] = [
        _step(
            name="system_prep",
            requires=[],
            sim=sim,
            sys_params=params_model,
        ),
        _step(
            name="param_ligands",
            requires=["system_prep"],
            sim=sim,
            sys_params=params_model,
        ),
        _step(
            "prepare_equil",
            requires=["param_ligands"],
            sim=sim,
            sys_params=params_model,
        ),
        _step(
            "equil",
            requires=["prepare_equil"],
            sim=sim,
            sys_params=params_model,
        ),
        _step(
            "equil_analysis",
            requires=["equil"],
            sim=sim,
            sys_params=params_model,
        ),
    ]
    # only_fe_preparation has no effect here because the MD pipeline stops before FE.
    if only_fe_preparation:
        return Pipeline(steps)
    return Pipeline(steps)
