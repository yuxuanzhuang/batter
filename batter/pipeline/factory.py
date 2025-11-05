from __future__ import annotations

from typing import List
from batter.config.simulation import SimulationConfig
from .step import Step
from .pipeline import Pipeline

__all__ = ["make_abfe_pipeline", "make_asfe_pipeline"]


def _step(name: str, requires: List[str] | None = None, **params) -> Step:
    """Small helper to keep steps consistent."""
    return Step(name=name, requires=requires or [], params=params)


def make_abfe_pipeline(
    sim: SimulationConfig, sys_params: dict, only_fe_preparation: bool = False
) -> Pipeline:
    """
    ABFE pipeline (expanded):

    system_prep → param_ligands → prepare_equil → equil → equil_analysis
    → prepare_fe → prepare_fe_windows → fe_equil → fe → analyze
    """
    steps: List[Step] = []

    # 0) system prep — runs once at system root
    steps.append(
        _step(
            name="system_prep",
            requires=[],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            name="param_ligands",
            requires=["system_prep"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )

    # Per-ligand steps
    steps.append(
        _step(
            "prepare_equil",
            requires=["param_ligands"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "equil",
            requires=["prepare_equil"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "equil_analysis",
            requires=["equil"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "prepare_fe",
            requires=["equil_analysis"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "prepare_fe_windows",
            requires=["prepare_fe"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "fe_equil",
            requires=["prepare_fe_windows"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "fe",
            requires=["fe_equil"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "analyze",
            requires=["fe"],
            mode="abfe",
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )

    if only_fe_preparation:
        # Keep up to 'prepare_fe' (inclusive); we’ll still prune 'param_ligands' at child level in run.py
        keep = {"system_prep", "param_ligands", "prepare_equil", "equil", "prepare_fe", "prepare_fe_windows"}
        steps = [s for s in steps if s.name in keep]

    return Pipeline(steps)


def make_asfe_pipeline(
    sim: SimulationConfig, sys_params: dict, only_fe_preparation: bool = False
) -> Pipeline:
    """
    ASFE pipeline (unchanged here for completeness):

    param_ligands → prepare_fe → solvation → analyze
    """
    steps: List[Step] = []
    steps.append(
        _step(
            name="system_prep_asfe",
            requires=[],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            name="param_ligands",
            requires=["system_prep_asfe"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "prepare_fe",
            requires=["param_ligands"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    steps.append(
        _step(
            "prepare_fe_windows",
            requires=["prepare_fe"],
            sim=sim.model_dump(),
            sys_params=sys_params,
        )
    )
    if only_fe_preparation:
        keep = {"system_prep", "param_ligands", "prepare_fe", "prepare_fe_windows"}
        steps = [s for s in steps if s.name in keep]
    else:
        steps.append(
            _step(
                "fe_equil",
                requires=["prepare_fe_windows"],
                sim=sim.model_dump(),
                sys_params=sys_params,
            )
        )
        steps.append(
            _step(
                "fe",
                requires=["fe_equil"],
                sim=sim.model_dump(),
                sys_params=sys_params,
            )
        )
        steps.append(
            _step(
                "analyze",
                requires=["fe"],
                mode="asfe",
                sim=sim.model_dump(),
                sys_params=sys_params,
            )
        )
    return Pipeline(steps)
