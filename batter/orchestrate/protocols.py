"""Protocol interfaces and registry used by the orchestrator."""

from __future__ import annotations
from typing import Protocol, Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from batter.config.simulation import SimulationConfig
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step

ProtocolName = Literal["abfe", "asfe", "md"]

class ProtocolContext(BaseModel):
    """
    What the protocol needs to plan a run.
    """
    sim: SimulationConfig
    only_fe_preparation: bool = False
    resume: bool = False     # optional: attach to partial runs
    dry_run: bool = False
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class FEProtocol(Protocol):
    """
    A protocol converts a validated SimulationConfig into a Pipeline.
    """

    name: ProtocolName

    def validate(self, ctx: ProtocolContext) -> None:
        """Raise ValueError if required config fields are missing/incompatible."""
        ...

    def plan(self, ctx: ProtocolContext) -> Pipeline:
        """Return a protocol-specific pipeline (DAG) of Steps."""
        ...

    def outputs(self) -> List[str]:
        """Logical artifact names produced (for indexing/saving FE results)."""
        ...


# --------- Registry ---------
_PROTOCOLS: Dict[ProtocolName, FEProtocol] = {}

def register_protocol(p: FEProtocol) -> None:
    _PROTOCOLS[p.name] = p

def get_protocol(name: ProtocolName) -> FEProtocol:
    try:
        return _PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"Unknown protocol: {name!r}. Available: {list(_PROTOCOLS)}")
