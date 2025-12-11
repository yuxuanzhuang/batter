from pathlib import Path

from batter.exec.handlers.fe import fe_handler
from batter.exec.slurm_mgr import SlurmJobManager
from batter.pipeline.step import Step
from batter.config.simulation import SimulationConfig
from batter.systems.core import SimSystem


class DummyMgr(SlurmJobManager):
    def __init__(self):
        self.add_calls = []

    def add(self, spec):
        self.add_calls.append(spec)


def test_fe_batch_adds_gres_flag(tmp_path):
    root = tmp_path / "work"
    comp_dir = root / "fe" / "z"
    (comp_dir / "z-1").mkdir(parents=True)
    win = comp_dir / "z00"
    win.mkdir(parents=True)
    (win / "run-local.bash").write_text("#!/bin/bash\n")

    sim_cfg = SimulationConfig(
        system_name="sys",
        fe_type="uno_rest",
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        buffer_x=15.0,
        buffer_y=15.0,
        buffer_z=15.0,
    )

    mgr = DummyMgr()
    params = {
        "job_mgr": mgr,
        "sim": sim_cfg,
        "batch_mode": True,
        "batch_run_root": root / "batch_run",
        "batch_gpus": 2,
        "batch_gpus_per_task": 1,
    }
    system = SimSystem(name="sys:lig", root=root, meta={"ligand": "lig"})

    fe_handler(Step(name="fe"), system, params)

    assert len(mgr.add_calls) == 1
    spec = mgr.add_calls[0]
    assert spec.extra_sbatch == ["--gres", "gpu:2"]
    assert spec.workdir == params["batch_run_root"] / "lig"
