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
