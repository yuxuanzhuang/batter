from __future__ import annotations
from typing import List
from pydantic import BaseModel
from batter.config.simulation import SimulationConfig
from batter.pipeline.pipeline import Pipeline
from .protocols import FEProtocol, ProtocolContext
from .steps_common import step_prepare_fe, step_equil, step_windows, step_solvation, step_analysis


class ABFE(FEProtocol):
    name = "abfe"

    def validate(self, ctx: ProtocolContext) -> None:
        sim = ctx.sim
        if sim.fe_type not in {"uno_rest", "rest", "dd", "sdr", "dd-rest", "sdr-rest", "uno_dd"}:
            raise ValueError(f"ABFE expects absolute protocols; got fe_type={sim.fe_type}")
        if not sim.components:
            # your SimulationConfig exposes read-only tuple via property
            raise ValueError("ABFE requires non-empty components (derived from fe_type).")

    def plan(self, ctx: ProtocolContext) -> Pipeline:
        sim = ctx.sim
        steps = [step_prepare_fe(sim)]
        if not ctx.only_fe_preparation:
            steps += [step_equil(sim), step_windows(sim, list(sim.components)), step_analysis(sim, "abfe")]
        return Pipeline(steps=steps)

    def outputs(self) -> List[str]:
        return ["fe/index", "fe/<run_id>/record", "fe/<run_id>/windows"]


class ASFE(FEProtocol):
    name = "asfe"

    def validate(self, ctx: ProtocolContext) -> None:
        if ctx.sim.fe_type not in {"asfe"}:
            raise ValueError("ASFE expects fe_type='asfe'.")

    def plan(self, ctx: ProtocolContext) -> Pipeline:
        sim = ctx.sim
        steps = [step_prepare_fe(sim)]
        if not ctx.only_fe_preparation:
            steps += [step_solvation(sim), step_analysis(sim, "asfe")]
        return Pipeline(steps=steps)

    def outputs(self) -> List[str]:
        return ["fe/index", "fe/<run_id>/record", "fe/<run_id>/windows"]


# One-time registration
from .protocols import register_protocol
register_protocol(ABFE())
register_protocol(ASFE())