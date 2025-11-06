from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from batter.config import (
    load_run_config,
    dump_run_config,
    load_simulation_config,
    dump_simulation_config,
)
from batter.config.run import CreateArgs, FESimArgs, SystemSection
from batter.config.simulation import SimulationConfig
from batter.config.utils import coerce_yes_no


def data_dir() -> Path:
    return Path(__file__).parent / "data"


def base_sim_kwargs(**overrides):
    data = {
        "system_name": "sys",
        "fe_type": "rest",
        "lambdas": [0.0, 1.0],
        "release_eq": [0.0, 1.0],
    }
    data.update(overrides)
    return data


def test_load_and_dump_run_config(tmp_path: Path, monkeypatch) -> None:
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
  param_outdir: "./params"
fe_sim: {{}}
run:
  run_id: auto
"""
    )

    cfg = load_run_config(run_yaml)
    assert cfg.create.ligand_paths["LIG1"] == lig_file
    assert cfg.create.param_outdir == (tmp_path / "params")

    out_yaml = tmp_path / "roundtrip.yaml"
    dump_run_config(cfg, out_yaml)
    cfg_roundtrip = load_run_config(out_yaml)
    assert cfg_roundtrip.create.param_outdir == (tmp_path / "params")
    assert "LIG1" in cfg_roundtrip.create.ligand_paths


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
    assert cfg.create.protein_input == protein
    assert cfg.create.ligand_input == lig_json


def test_load_and_dump_simulation_config(tmp_path: Path) -> None:
    sim_yaml = tmp_path / "sim.yaml"
    sim_yaml.write_text(
        """
system_name: sim-example
fe_type: uno_rest
neutralize_only: "YES"
lambdas: [0.0, 1.0]
release_eq: [0.0, 1.0]
"""
    )

    sim_cfg = load_simulation_config(sim_yaml)
    assert sim_cfg.system_name == "sim-example"
    assert sim_cfg.neutralize_only == "yes"

    out_yaml = tmp_path / "sim_roundtrip.yaml"
    dump_simulation_config(sim_cfg, out_yaml)
    sim_cfg_roundtrip = load_simulation_config(out_yaml)
    assert sim_cfg_roundtrip.system_name == "sim-example"
    assert sim_cfg_roundtrip.neutralize_only == "yes"


@pytest.mark.parametrize("yaml_file", sorted(path for path in data_dir().glob("*.yaml")))
def test_example_run_configs_load(monkeypatch, yaml_file: Path) -> None:
    monkeypatch.chdir(yaml_file.parent)
    cfg = load_run_config(yaml_file.name)
    sim_cfg = cfg.resolved_sim_config()
    assert sim_cfg.system_name
    assert cfg.run.run_id


def test_coerce_yes_no_invalid():
    with pytest.raises(ValueError):
        coerce_yes_no("maybe")


def test_system_section_requires_output_folder():
    with pytest.raises(ValidationError):
        SystemSection(type="MABFE", output_folder="")


def test_create_args_requires_ligand_spec():
    with pytest.raises(ValidationError):
        CreateArgs()


def test_create_args_normalize_yes_no(tmp_path: Path):
    lig = tmp_path / "lig.sdf"
    with pytest.raises(ValidationError):
        CreateArgs(ligand_paths={"lig": lig}, neutralize_only="maybe")


def test_create_args_extra_restraints_conflict(tmp_path: Path):
    lig = tmp_path / "lig.sdf"
    conf = tmp_path / "conf.json"
    conf.write_text("[]")
    with pytest.raises(ValidationError):
        CreateArgs(
            ligand_paths={"lig": lig},
            extra_restraints="sel",
            extra_conformation_restraints=conf,
        )


def test_create_args_extra_conformation_missing(tmp_path: Path):
    lig = tmp_path / "lig.sdf"
    missing = tmp_path / "missing.json"
    with pytest.raises(ValidationError):
        CreateArgs(
            ligand_paths={"lig": lig},
            extra_conformation_restraints=missing,
        )


def test_create_args_extra_conformation_invalid(tmp_path: Path):
    lig = tmp_path / "lig.sdf"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bad": "value"}))
    with pytest.raises(ValidationError):
        CreateArgs(
            ligand_paths={"lig": lig},
            extra_conformation_restraints=bad,
        )


def test_fesim_args_invalid_yes_no():
    with pytest.raises(ValidationError):
        FESimArgs(remd="maybe")


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
