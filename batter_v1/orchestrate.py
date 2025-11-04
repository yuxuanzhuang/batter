from __future__ import annotations

from pathlib import Path
from loguru import logger

from batter.config.run import RunConfig
from batter.exec.local import LocalBackend
from batter.exec.slurm import SlurmBackend
from batter.systems.mabfe import MABFEBuilder
from batter.systems.core import SimSystem
from batter.pipeline.factory import (
    make_abfe_pipeline,
    make_asfe_pipeline,
)


def _select_pipeline(protocol: str, sim_cfg, only_fe_prep: bool):
    """
    Map protocol → concrete pipeline builder.
    """
    if protocol == "abfe":
        return make_abfe_pipeline(sim_cfg, only_fe_prep)
    if protocol == "asfe":
        return make_asfe_pipeline(sim_cfg, only_fe_prep)
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def run_from_yaml(path: Path | str) -> None:
    """
    Orchestrate a run from a top-level YAML using MABFEBuilder.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the run YAML (contains system/create/run + simulation).
    """
    path = Path(path)
    rc = RunConfig.load(path)

    # pick builder/backend
    if rc.system.type != "MABFE":
        raise ValueError(f"Unsupported system.type {rc.system.type!r}; only 'MABFE' is supported.")
    builder = MABFEBuilder()
    backend = SlurmBackend() if rc.backend == "slurm" else LocalBackend()

    # build system
    sys = SimSystem(name=rc.create.system_name, root=rc.system.output_folder)
    sys = builder.build(sys, rc.create)

    # pipeline
    sim_cfg = rc.resolved_sim_config()
    pipeline = _select_pipeline(rc.protocol, sim_cfg, rc.run.only_fe_preparation)

    logger.info("Planned steps: {}", [s.name for s in pipeline.ordered_steps()])
    if rc.run.dry_run:
        logger.info("Dry run enabled — no execution.")
        return

    # run
    pipeline.run(backend, sys)