from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from batter.config.utils import coerce_yes_no


class RemdArgs(BaseModel):
    """REMD-specific controls nested under ``fe_sim.remd``."""

    model_config = ConfigDict(extra="forbid")

    enable: Literal["yes", "no"] = Field(
        "no", description="Toggle REMD for the run (yes/no)."
    )
    nstlim: int = Field(
        100, ge=1, description="Total MD steps for each REMD segment (nstlim)."
    )
    numexchg: int = Field(
        3000, ge=1, description="Exchange attempt interval (steps between swaps)."
    )

    @field_validator("enable", mode="before")
    @classmethod
    def _coerce_enable(cls, v):
        return coerce_yes_no(v)
