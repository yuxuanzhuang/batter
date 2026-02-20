from __future__ import annotations

import json
from pathlib import Path

from batter.config.simulation import SimulationConfig
from batter.exec.handlers.fe_analysis import analyze_handler
from batter.pipeline.step import Step
from batter.systems.core import SimSystem, SystemMeta


def _sim_cfg() -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "system_name": "sys",
            "fe_type": "rest",
            "components": ["z", "x"],
            "component_lambdas": {"x": [0.0, 1.0]},
            "lambdas": [0.0, 1.0],
            "eq_steps": 1000,
            "n_bootstraps": 16,
            "buffer_x": 15.0,
            "buffer_y": 15.0,
            "buffer_z": 15.0,
        }
    )


def test_analyze_handler_rbfe_pair_forces_x_and_writes_summary(
    tmp_path: Path, monkeypatch
) -> None:
    pair_root = tmp_path / "simulations" / "transformations" / "A~B"
    (pair_root / "fe" / "x").mkdir(parents=True, exist_ok=True)

    called = {}

    def _fake_analyze_lig_task(**kwargs):
        called.update(kwargs)
        results_dir = Path(kwargs["lig_path"]) / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "Results.dat").write_text("x\t1.00\t0.10\nTotal\t1.00\t0.10\n")
        (results_dir / "fe_timeseries.json").write_text(
            json.dumps({"fe_value": [1.0], "fe_std": [0.1]})
        )
        (results_dir / "fe_timeseries.png").write_bytes(b"png")

    monkeypatch.setattr(
        "batter.exec.handlers.fe_analysis.analyze_lig_task", _fake_analyze_lig_task
    )

    system = SimSystem(
        name="sys:A~B:run",
        root=pair_root,
        meta=SystemMeta.from_mapping(
            {
                "ligand": "A~B",
                "residue_name": "AAA",
                "mode": "RBFE",
                "pair_id": "A~B",
                "ligand_ref": "A",
                "ligand_alt": "B",
            }
        ),
    )

    res = analyze_handler(
        Step(name="analyze"),
        system,
        {"sim": _sim_cfg()},
    )

    assert called["components"] == ["x"]
    assert called["n_bootstraps"] == 16
    summary = pair_root / "fe" / "Results" / "rbfe_pair_summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload["pair_id"] == "A~B"
    assert payload["ligand_ref"] == "A"
    assert payload["ligand_alt"] == "B"
    assert payload["total_dg_kcal_mol"] == 1.0
    assert payload["components"] == ["x"]
    assert "rbfe_pair_summary_json" in res.artifacts
