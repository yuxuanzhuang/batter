from pathlib import Path

import pytest

from batter.config.run import CreateArgs, FESimArgs
from batter.config.simulation import SimulationConfig
from batter.exec.handlers import prepare_fe as prepare_fe_mod
from batter.pipeline.step import Step
from batter.pipeline.payloads import StepPayload
from batter.systems.core import SimSystem


def test_prepare_fe_windows_always_writes_remd(monkeypatch, tmp_path: Path) -> None:
    # Minimal ligand param index
    run_root = tmp_path / "run"
    lig_root = run_root / "simulations" / "LIG"
    (run_root / "artifacts" / "ligand_params").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "ligand_params" / "index.json").write_text(
        '{"ligands": [{"residue_name": "LIG", "store_dir": "params/LIG"}]}'
    )

    # Dummy builder that creates window dirs and mdin placeholders
    created = []

    class DummyBuilder:
        def __init__(self, *, working_dir, component, win, **_kwargs):
            self.working_dir = working_dir
            self.component = component
            self.win = win

        def build(self):
            comp_dir = self.working_dir
            comp_dir.mkdir(parents=True, exist_ok=True)
            win_dir = comp_dir / f"{self.component}{self.win:02d}"
            win_dir.mkdir(parents=True, exist_ok=True)
            (win_dir / "mdin-00").write_text("&cntrl\n/")
            (win_dir / "mdin-01").write_text("&cntrl\n/")
            created.append(win_dir)

    monkeypatch.setattr(prepare_fe_mod, "AlchemicalFEBuilder", DummyBuilder)

    called = []

    def fake_prepare_remd(workdir, comp, sim, n_windows, partition=None):
        called.append((workdir, comp, n_windows))

    monkeypatch.setattr(prepare_fe_mod.remd_ops, "prepare_remd_component", fake_prepare_remd)

    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("dummy")
    create = CreateArgs(system_name="sys", ligand_paths={"LIG": lig_file})
    fe_args = FESimArgs(
        lambdas=[0.0],
        eq_steps=100,
        n_steps={"z": 1000},
    )
    sim_cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")

    payload = StepPayload(sim=sim_cfg)
    system = SimSystem(name="sys", root=lig_root, meta={"ligand": "LIG", "residue_name": "LIG"})

    prepare_fe_mod.prepare_fe_windows_handler(Step(name="prepare_fe_windows"), system, payload)

    assert called, "REMD prep should be invoked even when sim.remd is 'no'"
    workdir, comp, n_windows = called[0]
    assert workdir == lig_root / "fe" / "z"
    assert comp == "z"
    assert n_windows == 1
    # ensure dummy builder created window dirs
    assert (lig_root / "fe" / "z" / "z00").exists()
