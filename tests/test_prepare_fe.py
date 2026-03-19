from pathlib import Path
import sys
import types

from batter.config.run import CreateArgs, FESimArgs
from batter.config.simulation import SimulationConfig


fake_builder_mod = types.ModuleType("batter._internal.builders.fe_alchemical")
BUILDER_CALLS: list[dict] = []


class _DummyBuilder:
    def __init__(self, *args, **kwargs):
        BUILDER_CALLS.append(kwargs)
        self.working_dir = Path(kwargs["working_dir"])

    def build(self):
        self.working_dir.mkdir(parents=True, exist_ok=True)


fake_builder_mod.AlchemicalFEBuilder = _DummyBuilder
sys.modules.setdefault("batter._internal.builders.fe_alchemical", fake_builder_mod)

fake_ops_pkg = types.ModuleType("batter._internal.ops")
fake_remd_mod = types.ModuleType("batter._internal.ops.remd")
fake_batch_mod = types.ModuleType("batter._internal.ops.batch")
fake_remd_mod.prepare_remd_component = lambda *args, **kwargs: None
fake_batch_mod.prepare_batch_component = lambda *args, **kwargs: None
fake_ops_pkg.remd = fake_remd_mod
fake_ops_pkg.batch = fake_batch_mod
sys.modules.setdefault("batter._internal.ops", fake_ops_pkg)
sys.modules.setdefault("batter._internal.ops.remd", fake_remd_mod)
sys.modules.setdefault("batter._internal.ops.batch", fake_batch_mod)

from batter.exec.handlers.prepare_fe import prepare_fe_handler
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import Step
from batter.systems.core import SimSystem


def test_prepare_fe_routes_rsfe_components_and_scope(tmp_path: Path) -> None:
    BUILDER_CALLS.clear()
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
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        n_steps={"s": 1000, "h": 1000},
    )
    sim_cfg = SimulationConfig.from_sections(create, fe_args, protocol="rsfe")
    assert sim_cfg.components == ["s", "h"]

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

    result = prepare_fe_handler(Step(name="prepare_fe"), system, payload)

    assert result.artifacts["prepare_fe_ok"].exists()
    assert [call["component"] for call in BUILDER_CALLS] == ["s", "h"]
    assert all(call["extra"]["relative_scope"] == "solvation" for call in BUILDER_CALLS)
