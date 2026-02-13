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
    # pass ``ligand`` when the run contains more than one ligand
    record = load_fe_run(\"work/adrb2\", latest, ligand=\"LIG1\")

Run FE analysis on an existing execution::

    from batter.api import run_analysis_from_execution
    run_analysis_from_execution(\"work/adrb2\", latest, ligand=\"LIG1\")

For more examples, refer to ``docs/getting_started.rst`` and the tutorials.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TYPE_CHECKING, Sequence, Union

from loguru import logger
from tqdm import tqdm

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
from .orchestrate.run import run_from_yaml, save_fe_records

# --- Portable runtime + FE results ---
from .runtime.portable import ArtifactStore
from .runtime.fe_repo import FEResultsRepository, FERecord, WindowResult
from .utils.exec_clone import clone_execution

# --- System descriptor (read-only for users) ---
from .systems.core import SimSystem, SystemMeta

from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import Step

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
    "run_analysis_from_execution",
    # execution cloning
    "clone_execution",
]


def _resolve_execution_run(work_root: Path, run_id: str | None) -> tuple[str, Path]:
    """Resolve an execution directory, defaulting to the most recent run."""
    requested = (run_id or "").strip() or None
    runs_root = work_root / "executions"

    if requested:
        run_dir = runs_root / requested
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Run '{requested}' does not exist under {work_root}."
            )
        return requested, run_dir

    if not runs_root.is_dir():
        raise FileNotFoundError(f"No executions found under {work_root}.")

    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No executions found under {work_root}.")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.info(
        f"No run_id provided; using latest execution '{latest.name}' under {work_root}."
    )
    return latest.name, latest


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
        DataFrame with one row per stored FE run. Columns include ``run_id``,
        ``ligand``, ``mol_name``, ``system_name``, ``temperature``, ``total_dG``,
        ``total_se``, ``canonical_smiles``, ``original_name``, ``original_path``,
        ``protocol``, ``analysis_start_step``, ``n_bootstraps``, ``status``,
        ``failure_reason``, and
        ``created_at``.
    """
    store = ArtifactStore(Path(work_dir))
    repo = FEResultsRepository(store)
    return repo.index()


def load_fe_run(
    work_dir: Union[str, Path], run_id: str, ligand: str | None = None
) -> FERecord:
    """
    Load a single FE record by ``run_id`` from a portable work directory.

    Parameters
    ----------
    work_dir : str or Path
        Root directory of the BATTER execution.
    run_id : str
        Identifier of the FE run to load (as returned by :func:`list_fe_runs`).
    ligand : str, optional
        Ligand identifier when multiple ligands were processed in the run. If omitted,
        the sole ligand is selected automatically or a ValueError is raised when
        multiple matches exist.

    Returns
    -------
    FERecord
        Structured record containing total ΔG, standard error, components, and
        per-window results.
    """
    store = ArtifactStore(Path(work_dir))
    repo = FEResultsRepository(store)
    if ligand:
        return repo.load(run_id, ligand)

    df = repo.index()
    matches = df[df["run_id"] == run_id]
    if matches.empty:
        raise KeyError(f"No FE records found for run_id '{run_id}'.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple ligands stored for run_id '{run_id}'. "
            "Call `load_fe_run` with the `ligand` argument or inspect `list_fe_runs`."
        )
    ligand_name = matches.iloc[0]["ligand"]
    return repo.load(run_id, ligand_name)


def run_analysis_from_execution(
    work_dir: Union[str, Path],
    run_id: str | None = None,
    *,
    ligand: str | None = None,
    components: Sequence[str] | None = None,
    n_workers: int | None = None,
    analysis_start_step: int | None = None,
    n_bootstraps: int | None = None,
    overwrite: bool = True,
    raise_on_error: bool = True,
) -> None:
    """
    Run FE analysis for a partially finished/finished execution.

    Parameters
    ----------
    work_dir : str or Path
        Root directory containing the portable execution store.
    run_id : str, optional
        Identifier of the execution (e.g., ``run-20240101``). When omitted,
        the most recently modified execution under ``<work_dir>/executions`` is used.
    ligand : str, optional
        Ligand identifier to target when only a subset should be analyzed.
    components : sequence of str, optional
        Components to include during analysis (overrides ``sim_cfg.components``).
    n_workers : int, optional
        Number of worker processes requested for the analysis handler.
    analysis_start_step : int, optional
        First production step to include in analysis (per window); overrides config.
    n_bootstraps : int, optional
        Number of MBAR bootstrap resamples; overrides config.
    overwrite: bool, optional
        When ``True`` (default), overwrite any existing analysis results for the run_id.
        When ``False``, skip ligands that already have analysis outputs.
    raise_on_error : bool, optional
        When ``True`` (default) propagate errors raised by the analysis handler.
        Set to ``False`` to log the failure and continue with other ligands.
    """
    work_root = Path(work_dir)
    run_id, run_dir = _resolve_execution_run(work_root, run_id)

    config_dir = run_dir / "artifacts" / "config"
    sim_cfg_path = config_dir / "sim.resolved.yaml"
    if not sim_cfg_path.exists():
        raise FileNotFoundError(
            f"Simulation configuration missing for run '{run_id}' at {sim_cfg_path}."
        )
    sim_cfg = load_sim_config(sim_cfg_path)

    run_meta_path = config_dir / "run_meta.json"
    run_meta: dict[str, Any] = {}
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text()) or {}
    protocol = run_meta.get("protocol", "abfe")
    system_name = run_meta.get("system_name") or sim_cfg.system_name

    index_path = run_dir / "artifacts" / "ligand_params" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Ligand index missing for run '{run_id}' at {index_path}."
        )
    ligands_payload = json.loads(index_path.read_text()) or {}
    entries = ligands_payload.get("ligands", [])
    if not entries:
        raise RuntimeError(f"No ligands recorded for run '{run_id}'.")

    param_dir_dict = {
        entry.get("residue_name"): entry.get("store_dir")
        for entry in entries
        if entry.get("residue_name") and entry.get("store_dir")
    }

    requested: Sequence[str] | None = [ligand] if ligand else None
    children: list[SimSystem] = []
    for entry in entries:
        lig_name = entry["ligand"]
        if requested and lig_name not in requested:
            continue
        child_root = run_dir / "simulations" / lig_name
        if not child_root.is_dir():
            raise FileNotFoundError(
                f"Simulation directory for ligand '{lig_name}' was not found at {child_root}."
            )
        meta = SystemMeta(
            ligand=lig_name,
            residue_name=entry.get("residue_name"),
            param_dir_dict=dict(param_dir_dict) if param_dir_dict else {},
        )
        children.append(
            SimSystem(
                name=f"{system_name}:{lig_name}:{run_id}",
                root=child_root,
                meta=meta,
            )
        )

    if requested and not children:
        raise KeyError(f"Ligand '{ligand}' not present in run '{run_id}'.")

    logger.info(f"Running analysis for {len(children)} ligands in run '{run_id}'.")
    logger.info(f"Number of workers: {n_workers}")
    payload_data: dict[str, Any] = {"sim": sim_cfg}
    if components:
        payload_data["components"] = list(components)
    if n_workers is not None:
        payload_data["analysis_n_workers"] = n_workers
        payload_data["n_workers"] = n_workers
    if analysis_start_step is not None:
        if analysis_start_step < 0:
            raise ValueError("analysis_start_step must be >= 0.")
        analysis_start_step_val = int(analysis_start_step)
        payload_data["analysis_start_step"] = analysis_start_step_val
        logger.info(f"Analysis start step set to: {analysis_start_step_val}")
    else:
        analysis_start_step_val = int(getattr(sim_cfg, "analysis_start_step", 0))
        payload_data["analysis_start_step"] = analysis_start_step_val
        logger.info(f"Analysis start step loaded: {analysis_start_step_val}")
    if n_bootstraps is not None:
        if n_bootstraps < 0:
            raise ValueError("n_bootstraps must be >= 0.")
        n_bootstraps_val = int(n_bootstraps)
        payload_data["n_bootstraps"] = n_bootstraps_val
        logger.info(f"MBAR bootstrap resamples set to: {n_bootstraps_val}")
    else:
        n_bootstraps_val = int(getattr(sim_cfg, "n_bootstraps", 0) or 0)
        payload_data["n_bootstraps"] = n_bootstraps_val
        logger.info(f"MBAR bootstrap resamples loaded: {n_bootstraps_val}")

    payload = StepPayload(**payload_data)
    params = payload.to_mapping()
    analyze_step = Step(name="analyze")
    from batter.exec.handlers.fe_analysis import analyze_handler

    def _analysis_outputs_present(fe_root: Path) -> bool:
        return (
            (fe_root / "Results" / "Results.dat").exists()
            and (fe_root / "analyze.ok").exists()
        )

    def _clear_analysis_outputs(fe_root: Path) -> None:
        shutil.rmtree(fe_root / "Results", ignore_errors=True)
        (fe_root / "analyze.ok").unlink(missing_ok=True)

    skipped = 0
    for child in tqdm(children, desc="Running analysis", unit="ligand"):
        fe_root = child.root / "fe"
        ligand_name = child.meta.get("ligand") or child.name
        if not overwrite and _analysis_outputs_present(fe_root):
            logger.info(
                f"Skipping analysis for ligand '{ligand_name}' (results already exist; overwrite=False)."
            )
            skipped += 1
            continue
        if overwrite:
            _clear_analysis_outputs(fe_root)
        try:
            analyze_handler(analyze_step, child, params)
        except Exception as exc:
            msg = f"Analysis failed for ligand '{ligand_name}' in run '{run_id}': {exc}"
            if raise_on_error:
                raise RuntimeError(msg) from exc
            logger.warning(msg)
            continue
    if skipped:
        logger.info(f"Skipped analysis for {skipped} ligand(s) with existing results.")

    store = ArtifactStore(work_root)
    repo = FEResultsRepository(store)
    failures = save_fe_records(
        run_dir=run_dir,
        run_id=run_id,
        children_all=children,
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol=protocol,
        analysis_start_step=analysis_start_step_val,
        n_bootstraps=n_bootstraps_val,
    )
    if failures:
        failed = ", ".join(
            [f"{name} ({status}: {reason})" for name, status, reason in failures]
        )
        logger.warning(f"Analysis recorded issues for run '{run_id}': {failed}")
    logger.info(f"Analysis complete for run '{run_dir}'.")
