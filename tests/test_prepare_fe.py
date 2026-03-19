from pathlib import Path
import sys
import types

import pytest

from batter.config.run import CreateArgs, FESimArgs
from batter.config.simulation import SimulationConfig


fake_builder_mod = types.ModuleType("batter._internal.builders.fe_alchemical")


class _DummyBuilder:
    def __init__(self, *args, **kwargs):
        pass

    def build(self):
        raise AssertionError("RSFE guard should trigger before the FE builder runs.")


fake_builder_mod.AlchemicalFEBuilder = _DummyBuilder
sys.modules.setdefault("batter._internal.builders.fe_alchemical", fake_builder_mod)

fake_ops_pkg = types.ModuleType("batter._internal.ops")
fake_remd_mod = types.ModuleType("batter._internal.ops.remd")
fake_batch_mod = types.ModuleType("batter._internal.ops.batch")
fake_ops_pkg.remd = fake_remd_mod
fake_ops_pkg.batch = fake_batch_mod
sys.modules.setdefault("batter._internal.ops", fake_ops_pkg)
sys.modules.setdefault("batter._internal.ops.remd", fake_remd_mod)
sys.modules.setdefault("batter._internal.ops.batch", fake_batch_mod)

from batter.exec.handlers.prepare_fe import prepare_fe_handler
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import Step
from batter.systems.core import SimSystem


def test_prepare_fe_rejects_unimplemented_rsfe_x_component(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    pair_root = run_root / "simulations" / "transformations" / "LIGA~LIGB"
    params_dir = run_root / "artifacts" / "ligand_params"
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / "index.json").write_text(
        '{"ligands": ['
        '{"ligand": "LIGA", "residue_name": "LIGA", "store_dir": "params/LIGA"}, '
        '{"ligand": "LIGB", "residue_name": "LIGB", "store_dir": "params/LIGB"}'
        "]}",
    )

    lig_a = tmp_path / "liga.sdf"
    lig_b = tmp_path / "ligb.sdf"
    lig_a.write_text("dummy\n")
    lig_b.write_text("dummy\n")

    create = CreateArgs(system_name="sys", ligand_paths={"LIGA": lig_a, "LIGB": lig_b})
    fe_args = FESimArgs(lambdas=[0.0, 1.0], eq_steps=100, n_steps={"x": 1000})
    sim_cfg = SimulationConfig.from_sections(create, fe_args, protocol="rsfe")

    payload = StepPayload(sim=sim_cfg)
    system = SimSystem(
        name="sys:LIGA~LIGB",
        root=pair_root,
        meta={
            "ligand": "LIGA~LIGB",
            "residue_name": "LIGA",
            "mode": "RSFE",
            "pair_id": "LIGA~LIGB",
            "ligand_ref": "LIGA",
            "ligand_alt": "LIGB",
            "residue_ref": "LIGA",
            "residue_alt": "LIGB",
        },
    )

    with pytest.raises(NotImplementedError, match="ligand-only relative x-component"):
        prepare_fe_handler(Step(name="prepare_fe"), system, payload)
