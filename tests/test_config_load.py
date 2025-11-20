from __future__ import annotations

import json
from pathlib import Path

import pytest
from loguru import logger
from pydantic import ValidationError

from batter.config import load_run_config, load_simulation_config
from batter.config.run import CreateArgs, FESimArgs, RunConfig, RunSection, MDSimArgs
from batter.config.simulation import SimulationConfig
from batter.config.utils import coerce_yes_no


def test_load_run_config_roundtrip(tmp_path: Path, monkeypatch) -> None:
    lig_file = tmp_path / "inputs" / "ligand.sdf"
    lig_file.parent.mkdir(parents=True, exist_ok=True)
    lig_file.write_text("dummy\n")
    monkeypatch.setenv("LIG_FILE", str(lig_file))

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        f"""
run:
  output_folder: "{tmp_path / 'work'}"
  run_id: auto
create:
  system_name: example
  ligand_paths:
    lig1: "${{LIG_FILE}}"
fe_sim: {{}}
"""
    )

    cfg = load_run_config(run_yaml)
    assert cfg.create.ligand_paths["LIG1"] == lig_file
    assert cfg.run.output_folder == tmp_path / "work"


def test_load_simulation_config(tmp_path: Path) -> None:
    sim_yaml = tmp_path / "sim.yaml"
    sim_yaml.write_text(
        """
system_name: sim-example
fe_type: uno_rest
lambdas: [0.0, 1.0]
num_equil_extends: 2
eq_steps: 1000
neutralize_only: "YES"
buffer_x: 20.0
buffer_y: 20.0
buffer_z: 20.0
"""
    )

    sim_cfg = load_simulation_config(sim_yaml)
    assert sim_cfg.system_name == "sim-example"
    assert sim_cfg.neutralize_only == "yes"


def test_run_config_relative_paths(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    protein = reference_dir / "protein.pdb"
    protein.write_text("HEADER\n")
    lig_json = reference_dir / "ligands.json"
    lig_json.write_text(json.dumps({"lig": str(protein)}))

    run_yaml = tmp_path / "run_rel.yaml"
    run_yaml.write_text(
        """
run:
  output_folder: work
  run_id: auto
create:
  system_name: example
  protein_input: reference/protein.pdb
  ligand_input: reference/ligands.json
fe_sim: {}
"""
    )

    cfg = load_run_config(run_yaml)
    assert cfg.create.protein_input == Path("reference/protein.pdb")
    assert cfg.create.ligand_input == Path("reference/ligands.json")


def base_sim_kwargs(**overrides):
    data = {
        "system_name": "sys",
        "fe_type": "rest",
        "lambdas": [0.0, 1.0],
        "num_equil_extends": 2,
        "eq_steps": 1000,
        "buffer_x": 15.0,
        "buffer_y": 15.0,
        "buffer_z": 15.0,
    }
    data.update(overrides)
    return data


def test_coerce_yes_no_invalid():
    with pytest.raises(ValueError):
        coerce_yes_no("maybe")


def test_run_section_requires_output_folder():
    with pytest.raises(ValidationError):
        RunSection(output_folder="")


def test_create_args_requires_ligand_spec():
    with pytest.raises(ValidationError):
        CreateArgs()


def test_fesim_args_invalid_yes_no():
    with pytest.raises(ValidationError):
        FESimArgs(remd="maybe")


def test_fesim_args_unsorted_lambdas():
    with pytest.raises(ValidationError):
        FESimArgs(lambdas=[0.5, 0.1])


def test_fesim_args_ingests_legacy_step_keys():
    args = FESimArgs.model_validate({"lambdas": [0, 1], "z_steps1": 60_000, "z_steps2": 70_000})
    assert args.steps1["z"] == 60_000
    assert args.steps2["z"] == 70_000


def test_fesim_args_ingests_legacy_component_lambdas():
    args = FESimArgs.model_validate({"lambdas": [0, 1], "z_lambdas": "0 0.5 1.0"})
    assert args.component_lambdas["z"] == [0.0, 0.5, 1.0]


def test_args_negative_force():
    with pytest.raises(ValidationError):
        FESimArgs(lig_distance_force=0.0)
    with pytest.raises(ValidationError):
        FESimArgs(lig_angle_force=0.0)
    with pytest.raises(ValidationError):
        FESimArgs(rec_com_force=0.0)
    with pytest.raises(ValidationError):
        FESimArgs(lig_com_force=0.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"p1": "bad-anchor"}, "Anchor must look"),
        ({"remd": "yes"}, "REMD not implemented"),
        ({"dec_int": "ti"}, "TI integration not implemented"),
        ({"fe_type": "custom", "lambdas": [0.0], "num_equil_extends": 1}, "dec_method"),
        (
            {
                "fe_type": "uno_rest",
                "lambdas": [0.0, 1.0],
                "n_steps_dict": {"z_steps1": 0, "z_steps2": 100},
            },
            "stage 1 steps must be > 0",
        ),
        ({"fe_type": "uno_rest", "lambdas": []}, "No lambdas defined"),
        (
            {"buffer_x": 4.0, "buffer_y": 15.0, "buffer_z": 15.0},
            "buffer_x must be >= 15.0",
        ),
        ({"neutralize_only": "maybe"}, "Invalid yes/no"),
    ],
)
def test_simulation_config_errors(overrides, message):
    kwargs = base_sim_kwargs(**overrides)
    with pytest.raises(Exception) as excinfo:
        SimulationConfig(**kwargs)
    assert message in str(excinfo.value)


def _minimal_create(tmp_path: Path, **updates) -> CreateArgs:
    lig = tmp_path / "lig.sdf"
    lig.write_text("dummy")
    data = {
        "system_name": "sys",
        "ligand_paths": {"LIG": lig},
    }
    data.update(updates)
    return CreateArgs(**data)


def test_sim_config_infe_flag_and_barostat(tmp_path: Path) -> None:
    conf_json = tmp_path / "conf.json"
    conf_json.write_text("[]")
    create = _minimal_create(tmp_path, extra_conformation_restraints=conf_json)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        num_equil_extends=1,
        eq_steps=100,
        steps1={"z": 50_000},
        steps2={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.infe is True
    assert cfg.barostat == 2
    assert cfg.release_eq == [0.0, 0.0]
    assert cfg.eq_steps1 == cfg.eq_steps2 == cfg.eq_steps == 100

    create2 = create.model_copy(
        update={"extra_conformation_restraints": None, "extra_restraints": "mask"}
    )
    cfg2 = SimulationConfig.from_sections(
        create2,
        FESimArgs(
            lambdas=[0, 1],
            num_equil_extends=1,
            eq_steps=100,
            steps1={"z": 50_000},
            steps2={"z": 300_000},
        ),
        protocol="abfe",
    )
    assert cfg2.infe is False
    assert cfg2.barostat == 1


def test_simulation_config_enable_mcwat_defaults_to_yes() -> None:
    cfg = SimulationConfig(**base_sim_kwargs())
    assert cfg.enable_mcwat == "yes"


def test_component_lambdas_override_and_default() -> None:
    base = base_sim_kwargs(
        component_windows={"c": [0.0, 0.25, 1.0]},
        lambdas=[0.0, 0.5, 1.0],
    )
    cfg = SimulationConfig(**base)
    assert cfg.component_lambdas["c"] == [0.0, 0.25, 1.0]
    # another active component should inherit the base lambdas
    assert cfg.component_lambdas["a"] == [0.0, 0.5, 1.0]


def test_sim_config_abfe_requires_z_steps(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=100,
        steps1={"z": 0},
        steps2={"z": 50},
    )
    with pytest.raises(ValueError, match="requires positive steps for component 'z'"):
        SimulationConfig.from_sections(create, fe_args, protocol="abfe")


def test_sim_config_asfe_requires_y_steps(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=100,
        steps1={"y": 0, "m": 10},
        steps2={"y": 0, "m": 20},
    )
    with pytest.raises(ValueError, match="requires positive steps for component 'y'"):
        SimulationConfig.from_sections(create, fe_args, protocol="asfe")


def test_component_lambdas_override_from_sections(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=100,
        component_lambdas={"z": [0.0, 0.2, 0.4, 1.0]},
        steps1={"z": 50_000},
        steps2={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.component_lambdas["z"] == [0.0, 0.2, 0.4, 1.0]


def _minimal_run_config(tmp_path: Path, protocol: str) -> RunConfig:
    create = _minimal_create(tmp_path)
    if protocol == "abfe":
        steps1 = {"z": 50_000}
        steps2 = {"z": 300_000}
    else:
        steps1 = {"y": 50_000, "m": 50_000}
        steps2 = {"y": 300_000, "m": 300_000}
    payload = {
        "protocol": protocol,
        "backend": "local",
        "run": {"output_folder": str(tmp_path / "out")},
        "create": create.model_dump(),
        "fe_sim": {
            "lambdas": [0.0, 1.0],
            "num_equil_extends": 1,
            "eq_steps": 1000,
            "steps1": steps1,
            "steps2": steps2,
        },
    }
    return RunConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [
        ("asfe", "asfe"),
        ("abfe", "uno_rest"),
    ],
)
def test_resolved_sim_config_sets_fe_type(protocol: str, expected: str, tmp_path: Path) -> None:
    cfg = _minimal_run_config(tmp_path, protocol)
    sim_cfg = cfg.resolved_sim_config()
    assert sim_cfg.fe_type == expected


def test_analysis_fe_range_default_small_num_fe_extends(tmp_path: Path, caplog) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=1000,
        num_fe_extends=2,
        steps1={"z": 50_000},
        steps2={"z": 300_000},
    )
    with caplog.at_level("WARNING"):
        cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.analysis_fe_range == (0, -1)


def test_analysis_fe_range_default_for_large_num_fe_extends(tmp_path: Path, caplog) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=1000,
        num_fe_extends=6,
        steps1={"z": 50_000},
        steps2={"z": 300_000},
    )
    with caplog.at_level("WARNING"):
        cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.analysis_fe_range == (2, -1)
    assert all("num_fe_extends" not in rec.message for rec in caplog.records)


def test_analysis_fe_range_respects_user_override(tmp_path: Path, caplog) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        num_equil_extends=1,
        eq_steps=1000,
        num_fe_extends=2,
        analysis_fe_range=(5, 7),
        steps1={"z": 50_000},
        steps2={"z": 300_000},
    )
    with caplog.at_level("WARNING"):
        cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.analysis_fe_range == (5, 7)
    assert all("num_fe_extends" not in rec.message for rec in caplog.records)


def test_enable_mcwat_propagates_from_fesim_args(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        num_equil_extends=0,
        eq_steps=100,
        enable_mcwat="no",
        steps1={"z": 50_000},
        steps2={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.enable_mcwat == "no"


def test_amber_setup_sh_defaults_and_override(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        num_equil_extends=0,
        eq_steps=10,
        steps1={"z": 10},
        steps2={"z": 20},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert (
        cfg.amber_setup_sh == "$GROUP_HOME/software/amber24/setup_amber.sh"
    )

    setup_sh = tmp_path / "amber.sh"
    setup_sh.write_text("#!/bin/bash\necho amber\n")

    run = RunConfig(
        version=1,
        protocol="abfe",
        backend="local",
        run=RunSection(
            output_folder=tmp_path / "out",
            amber_setup_sh=str(setup_sh),
        ),
        create=create,
        fe_sim=fe_args,
    )
    sim_cfg = run.resolved_sim_config()
    assert sim_cfg.amber_setup_sh == str(setup_sh)


def test_amber_setup_sh_warns_when_missing(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        num_equil_extends=0,
        eq_steps=10,
        steps1={"z": 10},
        steps2={"z": 20},
    )
    missing = tmp_path / "missing.sh"
    messages = []
    token = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        cfg = SimulationConfig.from_sections(
            create, fe_args, protocol="abfe", amber_setup_sh=str(missing)
        )
    finally:
        logger.remove(token)
    assert any("amber_setup_sh" in str(m) for m in messages)
    assert cfg.amber_setup_sh == str(missing)


def test_run_config_uses_md_sim_args(tmp_path: Path) -> None:
    lig = tmp_path / "lig.sdf"
    lig.write_text("dummy")
    run = RunConfig(
        version=1,
        protocol="md",
        backend="local",
        run=RunSection(output_folder=tmp_path / "work"),
        create=CreateArgs(system_name="sys", ligand_paths={"LIG": lig}),
        fe_sim={},  # intentionally empty; lambdas not required for MD
    )
    assert isinstance(run.fe_sim, MDSimArgs)
    assert run.fe_sim.dt == pytest.approx(0.004)
