from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from batter.config.simulation import SimulationConfig
from batter.orchestrate.run import save_fe_records
from batter.runtime.fe_repo import FEResultsRepository
from batter.runtime.portable import ArtifactStore
from batter.systems.core import SimSystem, SystemMeta
from batter.orchestrate import run as run_mod


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
            "buffer_x": 15.0,
            "buffer_y": 15.0,
            "buffer_z": 15.0,
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


def test_compute_run_signature_excludes_run_section(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
run:
  output_folder: out
create:
  system_name: sys
fe_sim: {}
protocol: abfe
"""
    )
    sig, payload = run_mod._compute_run_signature(yaml_path, {"override": 1})
    assert isinstance(sig, str) and len(sig) == 64
    assert "run" not in payload["config"]
    assert set(payload["config"].keys()) <= {"create", "fe_sim", "fe"}
    assert payload["run_overrides"] == {}


def test_stored_payload_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "exec"
    path = run_mod._payload_path(run_dir)
    payload = {"config": {"a": 1}, "run_overrides": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    assert run_mod._stored_payload(run_dir) == payload


def test_resolve_signature_conflict_reports_diffs(tmp_path: Path, caplog) -> None:
    stored_payload = {"config": {"a": 1}}
    current_payload = {"config": {"a": 2}}
    keep = run_mod._resolve_signature_conflict(
        "aaa",
        "bbb",
        requested_run_id="auto",
        allow_run_id_mismatch=False,
        run_id="rid",
        run_dir=tmp_path,
        stored_payload=stored_payload,
        current_payload=current_payload,
    )
    assert keep is False
