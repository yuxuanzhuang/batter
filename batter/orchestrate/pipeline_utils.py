from __future__ import annotations

from typing import Optional

from batter.pipeline.pipeline import Pipeline
from batter.config.simulation import SimulationConfig
from batter.pipeline.factory import make_abfe_pipeline, make_asfe_pipeline


def select_pipeline(
    protocol: str,
    sim_cfg: SimulationConfig,
    only_fe_prep: bool,
    *,
    sys_params: Optional[dict] = None,
) -> Pipeline:
    """
    Return the protocol-specific pipeline for a run.

    Parameters
    ----------
    protocol
        Name of the requested protocol (e.g., "abfe", "asfe").
    sim_cfg
        Validated simulation configuration.
    only_fe_prep
        Whether to stop after FE preparation.
    sys_params
        Extra parameters consumed by system-level steps.
    """
    name = (protocol or "abfe").lower()
    if name == "abfe":
        return make_abfe_pipeline(
            sim_cfg,
            sys_params=sys_params or {},
            only_fe_preparation=only_fe_prep,
        )
    if name == "asfe":
        return make_asfe_pipeline(
            sim_cfg,
            sys_params=sys_params or {},
            only_fe_preparation=only_fe_prep,
        )
    if name == "rbfe":
        raise NotImplementedError("RBFE protocol is not yet implemented.")
    raise ValueError(f"Unsupported protocol: {protocol!r}")
