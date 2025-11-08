"""Public API for BATTER.

This module collects the stable entry points intended for external consumption.
They fall into four broad categories:

* **Configuration helpers** – load and dump ``RunConfig`` / ``SimulationConfig`` objects.
* **Execution** – orchestrate complete workflows from a YAML definition.
* **Portable results** – inspect and copy artifacts produced by a run.
* **Utilities** – clone the state of an execution for reproducibility.

Typical usage
-------------

Run a workflow from a top-level YAML::

    from batter.api import run_from_yaml
    run_from_yaml(\"examples/mabfe.yaml\")

Inspect FE records stored in a work directory::

    from batter.api import list_fe_runs, load_fe_run
    runs = list_fe_runs(\"work/adrb2\")
    latest = runs.iloc[-1][\"run_id\"]
    record = load_fe_run(\"work/adrb2\", latest)

For more examples, refer to ``docs/getting_started.rst`` and the tutorials.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from ._version import __version__  # semantic version string

# --- Schemas / configs ---
from .config.simulation import SimulationConfig
from .config.run import RunConfig
from .config import (
    load_run_config,
    dump_run_config,
    load_simulation_config as load_sim_config,
    dump_simulation_config as save_sim_config,
)

# --- Orchestration entrypoint ---
from .orchestrate.run import run_from_yaml

# --- Portable runtime + FE results ---
from .runtime.portable import ArtifactStore
from .runtime.fe_repo import FEResultsRepository, FERecord, WindowResult
from .utils.exec_clone import clone_execution

# --- System descriptor (read-only for users) ---
from .systems.core import SimSystem

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[assignment]  # for type hints only

__all__ = [
    # version
    "__version__",
    # configs
    "SimulationConfig",
    "RunConfig",
    "load_run_config",
    "dump_run_config",
    "load_sim_config",
    "save_sim_config",
    # orchestration
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
    # execution cloning
    "clone_execution",
]


def list_fe_runs(work_dir: Union[str, Path]) -> "pd.DataFrame":
    """
    Return an index of FE runs contained in a portable work directory.

    Parameters
    ----------
    work_dir : str or Path
        Path to the root directory of a BATTER execution (portable layout).

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per stored FE run. Columns typically include
        ``run_id``, ``system_name``, ``ligand``, ``mol_name``,
        ``fe_type``, ``temperature``,
        ``method``, ``total_dG``, ``total_se``, ``components``, and ``created_at``.
    """
    store = ArtifactStore(Path(work_dir))
    repo = FEResultsRepository(store)
    return repo.index()


def load_fe_run(work_dir: Union[str, Path], run_id: str) -> FERecord:
    """
    Load a single FE record by ``run_id`` from a portable work directory.

    Parameters
    ----------
    work_dir : str or Path
        Root directory of the BATTER execution.
    run_id : str
        Identifier of the FE run to load (as returned by :func:`list_fe_runs`).

    Returns
    -------
    FERecord
        Structured record containing total ΔG, standard error, components, and
        per-window results.
    """
    store = ArtifactStore(Path(work_dir))
    repo = FEResultsRepository(store)
    return repo.load(run_id)
