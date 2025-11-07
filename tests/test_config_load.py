from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from batter.config import load_run_config, load_simulation_config
from batter.config.run import CreateArgs, FESimArgs, SystemSection
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
system:
  type: MABFE
  output_folder: "{tmp_path / 'work'}"
create:
  system_name: example
  ligand_paths:
    lig1: "${{LIG_FILE}}"
fe_sim: {{}}
run:
  run_id: auto
"""
    )

    cfg = load_run_config(run_yaml)
    assert cfg.create.ligand_paths["LIG1"] == lig_file
    assert cfg.system.output_folder == tmp_path / "work"


def test_load_simulation_config(tmp_path: Path) -> None:
    sim_yaml = tmp_path / "sim.yaml"
    sim_yaml.write_text(
        """
system_name: sim-example
fe_type: uno_rest
lambdas: [0.0, 1.0]
release_eq: [0.0, 1.0]
neutralize_only: "YES"
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
system:
  type: MABFE
  output_folder: work
create:
  system_name: example
  protein_input: reference/protein.pdb
  ligand_input: reference/ligands.json
fe_sim: {}
run:
  run_id: auto
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
        "release_eq": [0.0, 1.0],
    }
    data.update(overrides)
    return data


def test_coerce_yes_no_invalid():
    with pytest.raises(ValueError):
        coerce_yes_no("maybe")


def test_system_section_requires_output_folder():
    with pytest.raises(ValidationError):
        SystemSection(type="MABFE", output_folder="")


def test_create_args_requires_ligand_spec():
    with pytest.raises(ValidationError):
        CreateArgs()


def test_fesim_args_invalid_yes_no():
    with pytest.raises(ValidationError):
        FESimArgs(remd="maybe")


def test_fesim_args_unsorted_lambdas():
    with pytest.raises(ValidationError):
        FESimArgs(lambdas=[0.5, 0.1])


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
        ({"num_waters": 10}, "num_waters"),
        ({"fe_type": "custom", "lambdas": [0.0], "release_eq": [0.0]}, "dec_method"),
        (
            {
                "fe_type": "uno_rest",
                "lambdas": [0.0, 1.0],
                "n_steps_dict": {"z_steps1": 0, "z_steps2": 100},
            },
            "stage 1 steps must be > 0",
        ),
        (
            {"fe_type": "uno_rest", "lambdas": [], "release_eq": [0.0, 1.0]},
            "No lambdas defined",
        ),
        (
            {"buffer_x": 4.0, "buffer_y": 6.0, "buffer_z": 6.0},
            "buffer_x must be >= 5.0",
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
    cfg = SimulationConfig.from_sections(create, FESimArgs(lambdas=[0, 1], release_eq=[0, 1]))
    assert cfg.infe is True
    assert cfg.barostat == 2

    create2 = create.model_copy(update={"extra_conformation_restraints": None, "extra_restraints": "mask"})
    cfg2 = SimulationConfig.from_sections(create2, FESimArgs(lambdas=[0, 1], release_eq=[0, 1]))
    assert cfg2.infe is False
    assert cfg2.barostat == 1
