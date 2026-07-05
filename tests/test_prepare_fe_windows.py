from pathlib import Path

import pytest

from batter.config.run import CreateArgs, FESimArgs
from batter.config.simulation import SimulationConfig
from batter.exec.handlers import prepare_fe as prepare_fe_mod
from batter.orchestrate.state_registry import get_phase_state
from batter.pipeline.step import Step
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.systems.core import SimSystem


def _write_required_window_files(window_dir: Path) -> None:
    window_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "run-local.bash",
        "check_run.bash",
        "mdin-template",
        "mdin-remd-template",
        "full.prmtop",
        "full.hmr.prmtop",
        "full.inpcrd",
        "full_merged.prmtop",
        "eq.rst7",
    ):
        (window_dir / name).write_text("x\n")


def _write_required_remd_component_files(workdir: Path) -> None:
    for name in (
        "run-local-remd.bash",
        "SLURMM-BATCH-remd",
        "check_run.bash",
        "lambda.sch",
    ):
        (workdir / name).write_text("x\n")
    (workdir / "remd").mkdir(parents=True, exist_ok=True)
    (workdir / "remd" / "mini.in.remd.groupfile").write_text("x\n")


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
            _write_required_window_files(win_dir)
            created.append(win_dir)

    monkeypatch.setattr(prepare_fe_mod, "AlchemicalFEBuilder", DummyBuilder)

    called = []

    def fake_prepare_remd(workdir, comp, sim, n_windows, partition=None):
        called.append((workdir, comp, n_windows))
        _write_required_remd_component_files(workdir)

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
    assert (lig_root / "fe" / "prepare_fe_windows.ok").exists()


def test_prepare_fe_windows_does_not_mark_complete_when_remd_files_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    lig_root = run_root / "simulations" / "LIG"
    (run_root / "artifacts" / "ligand_params").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "ligand_params" / "index.json").write_text(
        '{"ligands": [{"residue_name": "LIG", "store_dir": "params/LIG"}]}'
    )

    class DummyBuilder:
        def __init__(self, *, working_dir, component, win, **_kwargs):
            self.working_dir = working_dir
            self.component = component
            self.win = win

        def build(self):
            _write_required_window_files(
                self.working_dir / f"{self.component}{self.win:02d}"
            )

    monkeypatch.setattr(prepare_fe_mod, "AlchemicalFEBuilder", DummyBuilder)

    def fake_prepare_remd(workdir, comp, sim, n_windows, partition=None):
        return []

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

    stale_marker = lig_root / "fe" / "prepare_fe_windows.ok"
    stale_marker.parent.mkdir(parents=True, exist_ok=True)
    stale_marker.write_text("stale\n")

    with pytest.raises(RuntimeError, match="Incomplete prepare_fe_windows output") as exc:
        prepare_fe_mod.prepare_fe_windows_handler(
            Step(name="prepare_fe_windows"),
            system,
            payload,
        )

    msg = str(exc.value)
    assert "fe/z/run-local-remd.bash" in msg
    assert stale_marker.exists() is False


def test_prepare_fe_forwards_initial_anchor_atoms(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    lig_root = run_root / "simulations" / "LIG"
    (run_root / "artifacts" / "ligand_params").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "ligand_params" / "index.json").write_text(
        '{"ligands": [{"residue_name": "LIG", "store_dir": "params/LIG"}]}'
    )

    captured = []

    class DummyBuilder:
        def __init__(self, *, extra, **_kwargs):
            captured.append(extra)

        def build(self):
            return None

    monkeypatch.setattr(prepare_fe_mod, "AlchemicalFEBuilder", DummyBuilder)

    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("dummy")
    create = CreateArgs(
        system_name="sys",
        ligand_paths={"LIG": lig_file},
        anchor_atoms=[
            "resid 10 and name CA",
            "resid 20 and name CA",
            "resid 30 and name CA",
        ],
    )
    fe_args = FESimArgs(
        lambdas=[0.0],
        eq_steps=100,
        n_steps={"z": 1000},
    )
    sim_cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")

    payload = StepPayload(
        sim=sim_cfg,
        sys_params=SystemParams(anchor_atoms=tuple(create.anchor_atoms)),
    )
    system = SimSystem(
        name="sys",
        root=lig_root,
        meta={"ligand": "LIG", "residue_name": "LIG"},
    )

    prepare_fe_mod.prepare_fe_handler(Step(name="prepare_fe"), system, payload)

    assert captured
    assert captured[0]["user_anchor_atoms"] == create.anchor_atoms
    state = get_phase_state(lig_root, "prepare_fe")
    assert state.required == [["fe/prepare_fe.ok", "fe/prepare_fe_windows.ok"]]
    assert state.success == [["fe/prepare_fe.ok", "fe/prepare_fe_windows.ok"]]
