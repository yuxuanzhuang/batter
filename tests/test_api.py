from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest

from batter import api as api_mod
from batter.config.simulation import SimulationConfig


def test_resolve_execution_run_defaults_to_latest(tmp_path: Path) -> None:
    executions = tmp_path / "executions"
    old = executions / "old-run"
    new = executions / "new-run"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    run_id, run_dir = api_mod._resolve_execution_run(tmp_path, None)
    assert run_id == "new-run"
    assert run_dir == new


def test_resolve_execution_run_raises_when_no_runs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No executions found"):
        api_mod._resolve_execution_run(tmp_path, None)


def test_run_analysis_from_execution_uses_transformations_for_rsfe(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "executions" / "run1"
    config_dir = run_dir / "artifacts" / "config"
    lig_dir = run_dir / "artifacts" / "ligand_params"
    pair_root = run_dir / "simulations" / "transformations" / "LIGA~LIGB"
    config_dir.mkdir(parents=True, exist_ok=True)
    lig_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sim.resolved.yaml").write_text("placeholder\n")
    (config_dir / "run_meta.json").write_text(
        '{"protocol": "rsfe", "system_name": "sys"}'
    )
    (lig_dir / "index.json").write_text(
        '{"ligands": [{"ligand": "LIGA", "residue_name": "LIGA", "store_dir": "params/LIGA"}, '
        '{"ligand": "LIGB", "residue_name": "LIGB", "store_dir": "params/LIGB"}]}'
    )
    (pair_root / "fe").mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig.model_validate(
        {
            "system_name": "sys",
            "fe_type": "relative",
            "relative_scope": "solvation",
            "lambdas": [0.0, 1.0],
            "n_steps_dict": {"s_n_steps": 100, "h_n_steps": 100},
            "eq_steps": 1000,
            "buffer_x": 20.0,
            "buffer_y": 20.0,
            "buffer_z": 20.0,
        }
    )
    monkeypatch.setattr(api_mod, "load_sim_config", lambda _path: sim_cfg)
    monkeypatch.setattr(api_mod, "save_fe_records", lambda **_kwargs: [])

    seen: list[tuple[str | None, str | None, str | None]] = []

    def fake_analyze_handler(step, system, params):
        seen.append(
            (
                system.meta.get("pair_id"),
                system.meta.get("ligand_ref"),
                system.meta.get("ligand_alt"),
            )
        )

    fake_module = types.ModuleType("batter.exec.handlers.fe_analysis")
    fake_module.analyze_handler = fake_analyze_handler
    monkeypatch.setitem(sys.modules, "batter.exec.handlers.fe_analysis", fake_module)

    api_mod.run_analysis_from_execution(tmp_path, "run1")

    assert seen == [("LIGA~LIGB", "LIGA", "LIGB")]
