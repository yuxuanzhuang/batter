from __future__ import annotations

from pathlib import Path

from batter._internal.ops import amber as amber_ops
from batter.config.simulation import SimulationConfig


def _base_sim(**overrides) -> SimulationConfig:
    data = {
        "system_name": "sys",
        "fe_type": "rest",
        "lambdas": [0.0, 1.0],
        "num_equil_extends": 1,
        "eq_steps": 100,
        "buffer_x": 15.0,
        "buffer_y": 15.0,
        "buffer_z": 15.0,
    }
    data.update(overrides)
    return SimulationConfig(**data)


def test_write_amber_templates_populates_enable_mcwat(monkeypatch, tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "mdin-equil").write_text("_enable_mcwat_")

    monkeypatch.setattr(amber_ops, "amber_files_orig", template_dir)

    sim_yes = _base_sim()
    out_yes = tmp_path / "out_yes"
    amber_ops.write_amber_templates(out_yes, sim_yes, membrane=False, production=False)
    assert (out_yes / "mdin-equil").read_text() == "1"

    sim_no = _base_sim(enable_mcwat="no")
    out_no = tmp_path / "out_no"
    amber_ops.write_amber_templates(out_no, sim_no, membrane=False, production=False)
    assert (out_no / "mdin-equil").read_text() == "0"
