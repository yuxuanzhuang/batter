from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from batter.config.simulation import SimulationConfig
from batter.orchestrate.run import save_fe_records
from batter.runtime.fe_repo import FEResultsRepository
from batter.runtime.portable import ArtifactStore
from batter.systems.core import SimSystem, SystemMeta


def _make_sim_cfg() -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "system_name": "sys",
            "fe_type": "rest",
            "dec_int": "mbar",
            "components": ["z"],
            "component_lambdas": {"z": [0.0, 1.0]},
            "lambdas": [0.0, 1.0],
            "temperature": 300.0,
            "analysis_fe_range": (0, -1),
        }
    )


@pytest.mark.parametrize("has_results", [False])
def test_save_fe_records_failure(tmp_path: Path, has_results: bool) -> None:
    run_dir = tmp_path / "run1"
    child_root = run_dir / "simulations" / "lig1"
    (child_root / "fe" / "Results").mkdir(parents=True, exist_ok=True)

    sim_cfg = _make_sim_cfg()
    child = SimSystem(
        name="sys:lig1:run1",
        root=child_root,
        meta=SystemMeta(ligand="lig1", residue_name="lig1"),
    )

    store = ArtifactStore(run_dir)
    repo = FEResultsRepository(store)

    failures = save_fe_records(
        run_dir=run_dir,
        run_id="run1",
        children_all=[child],
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol="abfe",
    )

    assert failures
    df = pd.read_csv(run_dir / "results" / "index.csv")
    row = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")].iloc[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "no_totals_found"
    failure_json = run_dir / "results" / "run1" / "lig1" / "failure.json"
    assert failure_json.exists()
