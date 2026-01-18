from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RemdArgs(BaseModel):
    """REMD-specific controls nested under ``fe_sim.remd``."""

    model_config = ConfigDict(extra="forbid")

    nstlim: int = Field(
        100, ge=1, description="Total MD steps for each REMD segment (nstlim)."
    )
