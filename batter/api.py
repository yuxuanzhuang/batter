"""
Public API for BATTER.

This module exposes a small, stable interface for:
- Loading/saving simulation configs
- Orchestrating runs from a top-level YAML
- Managing portable artifacts and FE results

Examples
--------
Run from a top-level YAML::

    from batter.api import run_from_yaml
    run_from_yaml("examples/run_abfe.yaml")

Load, tweak, and re-save a simulation config::

    from batter.api import load_sim_config, save_sim_config
    cfg = load_sim_config("examples/sim_config.yaml")
    cfg.temperature = 298.15
    save_sim_config(cfg, "work/sim_config.resolved.yaml")

Inspect FE results in a work directory (portable across clusters)::

    from batter.api import list_fe_runs, load_fe_run
    idx = list_fe_runs("work/at1r_aai")
    rec = load_fe_run("work/at1r_aai", run_id=idx.iloc[-1]["run_id"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, TYPE_CHECKING

from ._version import __version__  # semantic version string

# --- Schemas / configs ---
from .config.simulation import SimulationConfig
from .config.run import RunConfig
from .config.io import read_yaml_config as load_sim_config, write_yaml_config as save_sim_config

# --- Orchestration entrypoint ---
from .orchestrate.run import run_from_yaml

# --- Portable runtime + FE results ---
from .runtime.portable import ArtifactStore
from .runtime.fe_repo import FEResultsRepository, FERecord, WindowResult

# --- System descriptor (read-only for users) ---
from .systems.core import SimSystem  # exposed for typing/provenance

if TYPE_CHECKING:
    import pandas as pd  # for type hints only

__all__ = [
    # version
    "__version__",
    # configs
    "SimulationConfig",
    "RunConfig",
    "load_sim_config",
    "save_sim_config",
    # orchestrator
    "run_from_yaml",
    # portable store + results
    "ArtifactStore",
    "FEResultsRepository",
    "FERecord",
    "WindowResult",
    # system descriptor
    "SimSystem",
    # convenience helpers
    "list_fe_runs",
    "load_fe_run",
]


# ---------------------- Convenience helpers ----------------------


def list_fe_runs(work_dir: Union[str, Path]) -> "pd.DataFrame":
    """
    List FE runs in a work directory.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Root directory for a BATTER run (portable across clusters).

    Returns
    -------
    pandas.DataFrame
        Index of runs, or an empty DataFrame if none exist.
        Columns typically include: ``run_id``, ``system_name``, ``fe_type``,
        ``temperature``, ``method``, ``total_dG``, ``total_se``,
        ``components``, ``created_at``.
    """
    store = ArtifactStore(Path(work_dir))
    repo = FEResultsRepository(store)
    return repo.index()


def load_fe_run(work_dir: Union[str, Path], run_id: str) -> FERecord:
    """
    Load a single FE record by run id.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Root directory for a BATTER run (portable across clusters).
    run_id : str
        Identifier of the FE run to load (see :func:`list_fe_runs`).

    Returns
    -------
    FERecord
        A structured record containing total dG, components, and per-window data.
    """
    store = ArtifactStore(Path(work_dir))
    repo = FEResultsRepository(store)
    return repo.load(run_id)