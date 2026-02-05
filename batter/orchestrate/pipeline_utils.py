"""Selection helpers for choosing the correct pipeline implementation."""

from __future__ import annotations

from typing import Optional

from batter.pipeline.pipeline import Pipeline
from batter.config.simulation import SimulationConfig
from batter.pipeline.factory import (
    make_abfe_pipeline,
    make_asfe_pipeline,
    make_md_pipeline,
    make_rbfe_pipeline,
)
from batter.pipeline.payloads import SystemParams


def select_pipeline(
    protocol: str,
    sim_cfg: SimulationConfig,
    only_fe_prep: bool,
    *,
    sys_params: Optional[SystemParams | dict] = None,
    partition: str | None = None,
) -> Pipeline:
    """Return the protocol-specific pipeline for a run.

    Parameters
    ----------
    protocol : str
        Name of the requested protocol (``"abfe"``, ``"rbfe"``, ``"asfe"``, or ``"md"``).
    sim_cfg : SimulationConfig
        Validated simulation configuration produced by :class:`RunConfig`.
    only_fe_prep : bool
        When ``True``, truncate the pipeline after FE preparation steps.
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
    """
    name = (protocol or "abfe").lower()
    params_model = (
        sys_params
        if isinstance(sys_params, SystemParams)
        else SystemParams.model_validate(sys_params or {})
    )
    extra = {"partition": partition} if partition else {}

    if name == "abfe":
        return make_abfe_pipeline(
            sim_cfg,
            sys_params=params_model,
            only_fe_preparation=only_fe_prep,
            extra=extra,
        )
    if name == "asfe":
        return make_asfe_pipeline(
            sim_cfg,
            sys_params=params_model,
            only_fe_preparation=only_fe_prep,
            extra=extra,
        )
    if name == "md":
        return make_md_pipeline(
            sim_cfg,
            sys_params=params_model,
            only_fe_preparation=only_fe_prep,
            extra=extra,
        )
    if name == "rbfe":
        return make_rbfe_pipeline(
            sim_cfg,
            sys_params=params_model,
            only_fe_preparation=only_fe_prep,
            extra=extra,
        )
    raise ValueError(f"Unsupported protocol: {protocol!r}")
