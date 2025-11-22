from pathlib import Path

import pytest

from batter.exec.handlers.fe import fe_handler
from batter.exec.slurm_mgr import SlurmJobManager
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import Step
from batter.systems.core import SimSystem
from batter.config.simulation import SimulationConfig


class DummyJobMgr(SlurmJobManager):
    def __init__(self):
        self.add_calls = []

    def add(self, spec):
        self.add_calls.append(spec)


@pytest.fixture
def remd_system(tmp_path):
    root = tmp_path / "work"
    comp_dir = root / "fe" / "z"
    comp_dir.mkdir(parents=True, exist_ok=True)
    # expected submit script location
    (comp_dir / "SLURMM-BATCH-remd").write_text("#!/bin/bash\necho remd\n")
    # sentinel files for state
    yield SimSystem(name="sys", root=root, meta={"ligand": "LIG"})


def test_fe_handler_remd_submits_per_component(remd_system):
    sim_cfg = SimulationConfig(
        system_name="sys",
        fe_type="uno_rest",
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=1000,
        buffer_x=15.0,
        buffer_y=15.0,
        buffer_z=15.0,
        remd="yes",
    )
    payload = StepPayload(sim=sim_cfg, job_mgr=DummyJobMgr())
    res = fe_handler(Step(name="fe"), remd_system, {"job_mgr": payload.job_mgr, "sim": sim_cfg})

    mgr = payload.job_mgr
    assert len(mgr.add_calls) == 1
    spec = mgr.add_calls[0]
    assert spec.workdir == remd_system.root / "fe" / "z"
    assert spec.script_rel == "SLURMM-BATCH-remd"
    assert spec.finished_name == "FINISHED"
    assert spec.failed_name == "FAILED"
