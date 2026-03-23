"""Protocol interfaces and registry used by the orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Protocol

from pydantic import BaseModel, Field

from batter.config.simulation import SimulationConfig
from batter.pipeline.pipeline import Pipeline

ProtocolName = Literal["abfe", "rbfe", "asfe", "md"]


class ProtocolContext(BaseModel):
    """Inputs required for protocol planning and validation."""

    sim: SimulationConfig
    only_fe_preparation: bool = False
    resume: bool = False
    dry_run: bool = False
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class FEProtocol(Protocol):
    """Protocol interface for turning a validated config into a pipeline."""

    name: ProtocolName

    def validate(self, ctx: ProtocolContext) -> None:
        """Raise ``ValueError`` when required config fields are incompatible."""
        ...

    def plan(self, ctx: ProtocolContext) -> Pipeline:
        """Return a protocol-specific pipeline."""
        ...

    def outputs(self) -> List[str]:
        """Return the logical artifact groups produced by the protocol."""
        ...


_PROTOCOLS: Dict[ProtocolName, FEProtocol] = {}


def register_protocol(p: FEProtocol) -> None:
    _PROTOCOLS[p.name] = p


def get_protocol(name: ProtocolName) -> FEProtocol:
    try:
        return _PROTOCOLS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown protocol: {name!r}. Available: {list(_PROTOCOLS)}"
        ) from exc
