"""Selection helpers for choosing the correct pipeline implementation."""

from __future__ import annotations

from typing import Optional

from batter.pipeline.pipeline import Pipeline
from batter.config.simulation import SimulationConfig
from batter.pipeline.factory import make_abfe_pipeline, make_asfe_pipeline
from batter.pipeline.payloads import SystemParams


def select_pipeline(
    protocol: str,
    sim_cfg: SimulationConfig,
    only_fe_prep: bool,
    *,
    sys_params: Optional[SystemParams | dict] = None,
) -> Pipeline:
    """Return the protocol-specific pipeline for a run.

    Parameters
    ----------
    protocol : str
        Name of the requested protocol (e.g., ``"abfe"``).
    sim_cfg : SimulationConfig
        Validated simulation configuration produced by :class:`RunConfig`.
    only_fe_prep : bool
        When ``True``, stop after FE preparation steps.
    sys_params : SystemParams or dict, optional
        Extra parameters passed to system-level pipeline steps.

    Returns
    -------
    Pipeline
        Pipeline instance tailored to the requested protocol.

    Raises
    ------
    ValueError
        If the protocol name is not recognised.
    NotImplementedError
        Raised for protocols that are planned but not yet available (e.g., RBFE).
    """
    name = (protocol or "abfe").lower()
    params_model = (
        sys_params
        if isinstance(sys_params, SystemParams)
        else SystemParams.model_validate(sys_params or {})
    )

    if name == "abfe":
        return make_abfe_pipeline(
            sim_cfg,
            sys_params=params_model,
            only_fe_preparation=only_fe_prep,
        )
    if name == "asfe":
        return make_asfe_pipeline(
            sim_cfg,
            sys_params=params_model,
            only_fe_preparation=only_fe_prep,
        )
    if name == "rbfe":
        raise NotImplementedError("RBFE protocol is not yet implemented.")
    raise ValueError(f"Unsupported protocol: {protocol!r}")
