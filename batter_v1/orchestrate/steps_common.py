from typing import Dict, Any, List
from batter.pipeline.step import Step
from batter.config.simulation import SimulationConfig

def step_prepare_fe(sim: SimulationConfig) -> Step:
    return Step(name="prepare_fe", params={"sim": sim.model_dump()})

def step_equil(sim: SimulationConfig) -> Step:
    return Step(name="equil", requires=["prepare_fe"], params={"sim": sim.model_dump()})

def step_windows(sim: SimulationConfig, components: List[str]) -> Step:
    return Step(
        name="windows",
        requires=["equil"],
        params={"sim": sim.model_dump(), "components": components},
    )

def step_solvation(sim: SimulationConfig) -> Step:
    # single-leg solvation (ASFE)
    return Step(name="solvation", requires=["prepare_fe"], params={"sim": sim.model_dump()})

def step_analysis(sim: SimulationConfig, mode: str) -> Step:
    return Step(name="analyze", requires=["windows" if mode!="asfe" else "solvation"],
                params={"sim": sim.model_dump(), "mode": mode})