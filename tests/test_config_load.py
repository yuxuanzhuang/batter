from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from loguru import logger
from pydantic import ValidationError

from batter.config import load_run_config, load_simulation_config
from batter.config.defaults import DEFAULT_N_BOOTSTRAPS, DEFAULT_NTPR
from batter.config.run import (
    CreateArgs,
    FESimArgs,
    MDSimArgs,
    RBFENetworkArgs,
    RunConfig,
    RunSection,
)
from batter.config.simulation import SimulationConfig
from batter.config.utils import apo_ligand_source_path, coerce_yes_no


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
    assert cfg.run.email_sender == "nobody@stanford.edu"


def test_run_config_infer_disulfide_bonds_override(tmp_path: Path) -> None:
    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("dummy\n")
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        f"""
run:
  output_folder: "{tmp_path / 'work'}"
protocol: abfe
create:
  system_name: example
  ligand_paths:
    lig1: "{lig_file}"
  infer_disulfide_bonds: false
fe_sim:
  z_lambdas: [0.0, 1.0]
  z_n_steps: 300000
"""
    )

    cfg = load_run_config(run_yaml)
    sim_cfg = cfg.resolved_sim_config()

    assert cfg.create.infer_disulfide_bonds is False
    assert sim_cfg.infer_disulfide_bonds is False


def test_run_config_ring_penetration_repair_options(tmp_path: Path) -> None:
    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("dummy\n")
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        f"""
run:
  output_folder: "{tmp_path / 'work'}"
protocol: abfe
create:
  system_name: example
  ligand_paths:
    lig1: "{lig_file}"
  fix_ring_penetration: false
  ring_penetration_fix_mode: ligand
fe_sim:
  z_lambdas: [0.0, 1.0]
  z_n_steps: 300000
"""
    )

    cfg = load_run_config(run_yaml)
    sim_cfg = cfg.resolved_sim_config()

    assert cfg.create.fix_ring_penetration is False
    assert cfg.create.ring_penetration_fix_mode == "ligand"
    assert sim_cfg.fix_ring_penetration is False
    assert sim_cfg.ring_penetration_fix_mode == "ligand"


def test_load_simulation_config(tmp_path: Path) -> None:
    sim_yaml = tmp_path / "sim.yaml"
    sim_yaml.write_text(
        """
system_name: sim-example
fe_type: uno_rest
lambdas: [0.0, 1.0]
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


def test_create_args_accepts_null_apo_ligand() -> None:
    args = CreateArgs(system_name="sys", ligand_paths={None: None})

    assert args.ligand_paths == {"APO": apo_ligand_source_path()}


def test_run_config_hoists_legacy_top_level_buffer_z(tmp_path: Path) -> None:
    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("dummy\n")
    run_yaml = tmp_path / "legacy_buffer.yaml"
    run_yaml.write_text(
        f"""
protocol: abfe
buffer_z: 12.5
z_n_steps: 1000
z_lambdas: [0.0, 1.0]
run:
  output_folder: "{tmp_path / 'work'}"
create:
  system_name: example
  ligand_paths:
    lig1: "{lig_file}"
fe_sim: {{}}
"""
    )

    cfg = load_run_config(run_yaml)
    sim_cfg = cfg.resolved_sim_config()

    assert cfg.fe_sim.buffer_z == 12.5
    assert sim_cfg.buffer_z == 12.5


def test_md_run_config_accepts_legacy_top_level_buffer_z() -> None:
    cfg = RunConfig.model_validate(
        {
            "protocol": "md",
            "buffer_z": 10.0,
            "run": {"output_folder": "work"},
            "create": {"system_name": "sys", "ligand_paths": {None: None}},
            "fe_sim": {},
        }
    )
    sim_cfg = cfg.resolved_sim_config()

    assert cfg.fe_sim.buffer_z == 10.0
    assert sim_cfg.buffer_z == 10.0


def test_run_config_accepts_rbfe_mapper_options(tmp_path: Path) -> None:
    lig1 = tmp_path / "lig1.sdf"
    lig2 = tmp_path / "lig2.sdf"
    atom_mapping = tmp_path / "atom_mapping.json"
    lig1.write_text("dummy\n")
    lig2.write_text("dummy\n")
    atom_mapping.write_text("{}\n")

    run_yaml = tmp_path / "rbfe_mapper_options.yaml"
    run_yaml.write_text(
        f"""
protocol: rbfe
run:
  output_folder: "{tmp_path / 'work'}"
  only_rbfe_network: true
create:
  system_name: rbfe-example
  ligand_paths:
    lig1: "{lig1}"
    lig2: "{lig2}"
fe_sim: {{}}
rbfe:
  mapping: konnektor
  skip_duplicate_ligands: true
  network_scorer: shape-difference
  atom_mapping_file: atom_mapping.json
  atom_mapper: lomap
  lomap:
    time: 7
    max3d: 2.0
    shift: false
  kartograf:
    atom_max_distance: 1.1
    allow_bond_breaks: true
    filter_element_changes: false
"""
    )

    cfg = load_run_config(run_yaml)
    assert cfg.rbfe is not None
    assert cfg.run.only_rbfe_network is True
    assert cfg.rbfe.atom_mapping_file == Path("atom_mapping.json")
    assert cfg.rbfe.resolve_paths(tmp_path).atom_mapping_file == atom_mapping.resolve()
    assert cfg.rbfe.skip_duplicate_ligands is True
    assert cfg.rbfe.atom_mapper == "lomap"
    assert cfg.rbfe.network_scorer == "shape_difference"
    assert cfg.rbfe.lomap.time == 7
    assert cfg.rbfe.lomap.max3d == 2.0
    assert cfg.rbfe.lomap.shift is False
    assert cfg.rbfe.kartograf.atom_max_distance == 1.1
    assert cfg.rbfe.kartograf.map_exact_ring_matches_only is True
    assert cfg.rbfe.kartograf.allow_partial_fused_rings is True
    assert cfg.rbfe.kartograf.allow_bond_breaks is True
    assert cfg.rbfe.kartograf.filter_element_changes is False
    assert cfg.rbfe.kartograf.filter_mismatched_attached_h_count is False


def test_run_config_rejects_only_rbfe_network_for_non_rbfe(tmp_path: Path) -> None:
    lig1 = tmp_path / "lig1.sdf"
    lig1.write_text("dummy\n")

    run_yaml = tmp_path / "abfe_only_rbfe_network.yaml"
    run_yaml.write_text(
        f"""
protocol: abfe
run:
  output_folder: "{tmp_path / 'work'}"
  only_rbfe_network: true
create:
  system_name: example
  ligand_paths:
    lig1: "{lig1}"
fe_sim: {{}}
"""
    )

    with pytest.raises(ValidationError, match="only_rbfe_network"):
        load_run_config(run_yaml)


def test_rbfe_kartograf_mapper_defaults() -> None:
    cfg = RBFENetworkArgs()

    assert cfg.kartograf.atom_max_distance == 0.95
    assert cfg.kartograf.map_exact_ring_matches_only is True
    assert cfg.kartograf.allow_partial_fused_rings is True
    assert cfg.kartograf.allow_bond_breaks is False
    assert cfg.network_scorer == "auto"
    assert cfg.add_atom_mapping_edges is False
    assert cfg.skip_duplicate_ligands is False
    assert cfg.minimal_mapping_atom == 3
    assert cfg.direction_policy == "larger_volume"


def test_rbfe_direction_policy_normalizes_hyphenated_value() -> None:
    cfg = RBFENetworkArgs(direction_policy="larger-volume")

    assert cfg.direction_policy == "larger_volume"


def test_rbfe_minimal_mapping_atom_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="minimal_mapping_atom"):
        RBFENetworkArgs(minimal_mapping_atom=0)


def test_run_config_rejects_rbfe_kartograf_hydrogen_mapping_options(
    tmp_path: Path,
) -> None:
    lig1 = tmp_path / "lig1.sdf"
    lig2 = tmp_path / "lig2.sdf"
    lig1.write_text("dummy\n")
    lig2.write_text("dummy\n")

    run_yaml = tmp_path / "rbfe_bad_hydrogen_options.yaml"
    run_yaml.write_text(
        f"""
protocol: rbfe
run:
  output_folder: "{tmp_path / 'work'}"
create:
  system_name: rbfe-example
  ligand_paths:
    lig1: "{lig1}"
    lig2: "{lig2}"
fe_sim: {{}}
rbfe:
  mapping: konnektor
  atom_mapper: kartograf
  kartograf:
    atom_map_hydrogens: true
"""
    )

    with pytest.raises(ValidationError, match="fixed for AMBER compatibility"):
        load_run_config(run_yaml)


def base_sim_kwargs(**overrides):
    data = {
        "system_name": "sys",
        "fe_type": "rest",
        "lambdas": [0.0, 1.0],
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


def test_create_args_rejects_reserved_ligand_name(tmp_path: Path) -> None:
    lig = tmp_path / "lig.sdf"
    lig.write_text("dummy\n")
    with pytest.raises(ValidationError, match="reserved"):
        CreateArgs(system_name="sys", ligand_paths={"transformations": lig})


def test_fesim_args_invalid_remd_type():
    with pytest.raises(ValidationError, match="fe_sim\\.remd"):
        FESimArgs(remd="maybe")


def test_fesim_args_unsorted_lambdas():
    with pytest.raises(ValidationError):
        FESimArgs(lambdas=[0.5, 0.1])


def test_fesim_args_rejects_stage1_steps():
    with pytest.raises(ValidationError):
        FESimArgs.model_validate(
            {"lambdas": [0, 1], "z_steps1": 60_000, "z_n_steps": 70_000}
        )


def test_fesim_args_ingests_legacy_component_lambdas():
    args = FESimArgs.model_validate({"lambdas": [0, 1], "z_lambdas": "0 0.5 1.0"})
    assert args.component_lambdas["z"] == [0.0, 0.5, 1.0]


def test_fesim_args_rejects_num_fe_extends() -> None:
    with pytest.raises(ValidationError, match="num_fe_extends is no longer supported"):
        FESimArgs.model_validate({"num_fe_extends": 2})


def test_fesim_args_rejects_analysis_range() -> None:
    with pytest.raises(ValidationError, match="analysis_range is no longer supported"):
        FESimArgs.model_validate({"analysis_range": [0, 100]})


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
        ({"dec_int": "ti"}, "TI integration not implemented"),
        ({"fe_type": "custom", "lambdas": [0.0]}, "dec_method"),
        (
            {
                "fe_type": "uno_rest",
                "lambdas": [0.0, 1.0],
                "n_steps_dict": {"z_n_steps": 0},
            },
            "steps must be > 0",
        ),
        ({"fe_type": "uno_rest", "lambdas": []}, "No lambdas defined"),
        (
            {"buffer_x": 4.0, "buffer_y": 15.0, "buffer_z": 10.0},
            "buffer_x must be >= 10.0",
        ),
        ({"neutralize_only": "maybe"}, "Invalid yes/no"),
    ],
)
def test_simulation_config_errors(overrides, message):
    kwargs = base_sim_kwargs(**overrides)
    with pytest.raises(Exception) as excinfo:
        SimulationConfig(**kwargs)
    assert message in str(excinfo.value)


def test_simulation_config_remd_enabled(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(
        create, fe_args, protocol="abfe", run_remd="yes"
    )
    assert cfg.remd == "yes"
    assert cfg.remd_nstlim == 100


def test_fesim_remd_block():
    args = FESimArgs.model_validate({"remd": {"nstlim": 200}})
    assert args.remd.nstlim == 200


def test_fesim_remd_numexchg_rejected():
    with pytest.raises(ValidationError):
        FESimArgs.model_validate({"remd": {"nstlim": 200, "numexchg": 1500}})


def test_fesim_remd_yes_rejected() -> None:
    with pytest.raises(ValidationError, match="run\\.remd"):
        FESimArgs.model_validate({"remd": "yes"})


def _minimal_create(tmp_path: Path, **updates) -> CreateArgs:
    lig = tmp_path / "lig.sdf"
    lig.write_text("dummy")
    data = {
        "system_name": "sys",
        "ligand_paths": {"LIG": lig},
    }
    data.update(updates)
    return CreateArgs(**data)


def test_sim_config_infer_disulfide_bonds_default(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(lambdas=[0.0, 1.0], n_steps={"z": 300_000})

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")

    assert cfg.infer_disulfide_bonds is True


def test_sim_config_ring_penetration_repair_defaults(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(lambdas=[0.0, 1.0], n_steps={"z": 300_000})

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")

    assert cfg.fix_ring_penetration is True
    assert cfg.ring_penetration_fix_mode == "auto"


def test_sim_config_infe_flag_and_barostat(tmp_path: Path) -> None:
    conf_json = tmp_path / "conf.json"
    conf_json.write_text("[]")
    create = _minimal_create(tmp_path, extra_conformation_restraints=conf_json)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        eq_steps=100,
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.infe is True
    assert cfg.barostat == 2
    assert cfg.release_eq == [0.0]
    assert cfg.eq_steps == 2500

    create2 = create.model_copy(
        update={"extra_conformation_restraints": None, "extra_restraints": "mask"}
    )
    cfg2 = SimulationConfig.from_sections(
        create2,
        FESimArgs(
            lambdas=[0, 1],
            eq_steps=100,
            n_steps={"z": 300_000},
        ),
        protocol="abfe",
    )
    assert cfg2.infe is False
    assert cfg2.barostat == 1


def test_run_config_load_resolves_relative_conformation_restraints(tmp_path: Path) -> None:
    conf_json = tmp_path / "rest.json"
    conf_json.write_text("[]")
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
protocol: abfe
backend: local
create:
  system_name: sys
  ligand_paths:
    LIG: lig.sdf
  extra_conformation_restraints: rest.json
run:
  output_folder: out
fe_sim:
  lambdas: [0.0, 1.0]
  z_n_steps: 300000
"""
    )

    cfg = RunConfig.load(yaml_path)

    assert cfg.create.extra_conformation_restraints == conf_json.resolve()


def test_simulation_config_enable_mcwat_defaults_to_yes() -> None:
    cfg = SimulationConfig(**base_sim_kwargs())
    assert cfg.enable_mcwat == "yes"


def test_simulation_config_mcwat_fe_defaults_to_no() -> None:
    cfg = SimulationConfig(**base_sim_kwargs())
    assert cfg.mcwat_fe == "no"


def test_simulation_config_ion_guard_defaults_to_yes() -> None:
    cfg = SimulationConfig(**base_sim_kwargs())
    assert cfg.ion_guard == "yes"


def test_default_ion_conc_is_50_mm(tmp_path: Path) -> None:
    cfg = SimulationConfig(**base_sim_kwargs())
    assert cfg.ion_conc == pytest.approx(0.05)
    assert cfg.ion_def[:2] == ["Na+", "Cl-"]
    assert cfg.ion_def[2] == pytest.approx(0.05)

    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(lambdas=[0.0, 1.0], n_steps={"z": 300_000})
    section_cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert section_cfg.ion_conc == pytest.approx(0.05)
    assert section_cfg.ion_def[2] == pytest.approx(0.05)


def test_ion_guard_from_sections_can_be_disabled(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        n_steps={"z": 300_000},
        ion_guard=False,
    )

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")

    assert cfg.ion_guard == "no"


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
        eq_steps=100,
        n_steps={"z": 0},
    )
    with pytest.raises(ValueError, match="requires positive steps for component 'z'"):
        SimulationConfig.from_sections(create, fe_args, protocol="abfe")


def test_sim_config_asfe_requires_y_steps(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        n_steps={"y": 0, "m": 20},
    )
    with pytest.raises(ValueError, match="requires positive steps for component 'y'"):
        SimulationConfig.from_sections(create, fe_args, protocol="asfe")


def test_component_lambdas_override_from_sections(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        component_lambdas={"z": [0.0, 0.2, 0.4, 1.0]},
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.component_lambdas["z"] == [0.0, 0.2, 0.4, 1.0]


def test_abfe_diff_uses_d_component_from_sections(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        component_lambdas={"d": [0.0, 0.25, 0.5, 1.0]},
        n_steps={"d": 300_000},
    )

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="ABFE_diff")

    assert cfg.fe_type == "uno_rest_diff"
    assert cfg.components == ["d"]
    assert cfg.dec_method == "sdr"
    assert cfg.component_lambdas["d"] == [0.0, 0.25, 0.5, 1.0]


def test_abfe_diff_can_add_ligand_conformational_component(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        component_lambdas={
            "d": [0.0, 0.5, 1.0],
            "l": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        n_steps={"d": 300_000, "l": 100_000},
        lig_dihcf_force=10.0,
    )

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="ABFE_diff")

    assert cfg.components == ["d", "l"]
    assert cfg.dec_method == "sdr"
    assert cfg.component_lambdas["l"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert cfg.dic_n_steps["l"] == 100_000
    assert cfg.lig_dihcf_force == 10.0


def test_ligand_rest_uses_l_component_from_sections(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        component_lambdas={"l": [0.0, 0.25, 0.5, 1.0]},
        n_steps={"l": 100_000},
        lig_dihcf_force=10.0,
    )

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="ligand-rest")

    assert cfg.fe_type == "ligand_rest"
    assert cfg.components == ["l"]
    assert cfg.dec_method == "sdr"
    assert cfg.component_lambdas["l"] == [0.0, 0.25, 0.5, 1.0]
    assert cfg.dic_n_steps["l"] == 100_000


def test_ligand_rest_requires_positive_dihedral_force(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=100,
        n_steps={"l": 100_000},
    )

    with pytest.raises(ValueError, match="ligand_rest requires positive lig_dihcf_force"):
        SimulationConfig.from_sections(create, fe_args, protocol="ligand_rest")


def test_run_config_abfe_diff_accepts_legacy_d_fields(tmp_path: Path) -> None:
    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("dummy\n")
    run_yaml = tmp_path / "abfe_diff.yaml"
    run_yaml.write_text(
        f"""
protocol: ABFE-diff
run:
  output_folder: "{tmp_path / 'work'}"
create:
  system_name: example
  ligand_paths:
    lig1: "{lig_file}"
fe_sim:
  d_lambdas: [0.0, 0.5, 1.0]
  d_n_steps: 300000
"""
    )

    cfg = load_run_config(run_yaml)
    sim_cfg = cfg.resolved_sim_config()

    assert cfg.protocol == "abfe_diff"
    assert sim_cfg.fe_type == "uno_rest_diff"
    assert sim_cfg.components == ["d"]
    assert sim_cfg.component_lambdas["d"] == [0.0, 0.5, 1.0]


def test_sim_config_from_sections_preserves_slurm_header_dir(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
    )
    header_dir = tmp_path / "slurm_headers"
    cfg = SimulationConfig.from_sections(
        create,
        fe_args,
        protocol="abfe",
        slurm_header_dir=header_dir,
    )
    assert cfg.slurm_header_dir == header_dir
    assert cfg.model_dump()["slurm_header_dir"] == header_dir


def _minimal_run_config(tmp_path: Path, protocol: str) -> RunConfig:
    create = _minimal_create(tmp_path)
    normalized_protocol = protocol.lower().replace("-", "_")
    if normalized_protocol == "abfe":
        n_steps = {"z": 300_000}
    elif normalized_protocol == "abfe_diff":
        n_steps = {"d": 300_000}
    elif normalized_protocol == "ligand_rest":
        n_steps = {"l": 100_000}
    elif normalized_protocol in {"rbfe", "rbfe_septop"}:
        n_steps = {"x": 300_000}
    else:
        n_steps = {"y": 300_000, "m": 300_000}
    extra_fe = {}
    if normalized_protocol == "ligand_rest":
        extra_fe["lig_dihcf_force"] = 10.0
    payload = {
        "protocol": protocol,
        "backend": "local",
        "run": {"output_folder": str(tmp_path / "out")},
        "create": create.model_dump(),
        "fe_sim": {
            "lambdas": [0.0, 1.0],
            "eq_steps": 1000,
            "n_steps": n_steps,
            **extra_fe,
        },
    }
    return RunConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [
        ("asfe", "asfe"),
        ("abfe", "uno_rest"),
        ("ABFE_diff", "uno_rest_diff"),
        ("ligand-rest", "ligand_rest"),
        ("rbfe-septop", "relative_septop"),
    ],
)
def test_resolved_sim_config_sets_fe_type(
    protocol: str, expected: str, tmp_path: Path
) -> None:
    cfg = _minimal_run_config(tmp_path, protocol)
    sim_cfg = cfg.resolved_sim_config()
    assert sim_cfg.fe_type == expected


def test_run_remd_toggle_overrides_fe_sim(tmp_path: Path) -> None:
    cfg = _minimal_run_config(tmp_path, "abfe")
    sim_cfg = cfg.resolved_sim_config()
    assert sim_cfg.remd == "no"

    cfg_yes = cfg.model_copy(
        update={"run": cfg.run.model_copy(update={"remd": "yes"})}
    )
    sim_cfg_yes = cfg_yes.resolved_sim_config()
    assert sim_cfg_yes.remd == "yes"


def test_resolved_sim_config_propagates_run_slurm_header_dir(tmp_path: Path) -> None:
    cfg = _minimal_run_config(tmp_path, "abfe")
    header_dir = tmp_path / "custom_headers"
    cfg = cfg.model_copy(
        update={"run": cfg.run.model_copy(update={"slurm_header_dir": header_dir})}
    )
    sim_cfg = cfg.resolved_sim_config()
    assert sim_cfg.slurm_header_dir == header_dir


def test_analysis_start_step_default(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.analysis_start_step == 0


def test_analysis_start_step_respects_user_override(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        analysis_start_step=5000,
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.analysis_start_step == 5000


def test_n_bootstraps_default(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert fe_args.n_bootstraps == DEFAULT_N_BOOTSTRAPS
    assert cfg.n_bootstraps == DEFAULT_N_BOOTSTRAPS


def test_n_bootstraps_respects_user_override(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
        n_bootstraps=64,
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.n_bootstraps == 64


def test_ntpr_default(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert fe_args.ntpr == DEFAULT_NTPR
    assert cfg.ntpr == DEFAULT_NTPR


def test_ntpr_respects_user_override(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"z": 300_000},
        ntpr=500,
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.ntpr == 500


def test_cinnabar_x_convergence_filter_default(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"x": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="rbfe")
    assert cfg.cinnabar_x_convergence_filter == (0.8, 1.0)
    assert cfg.cinnabar_x_convergence_fallback_filter == (0.5, 2.0)


def test_cinnabar_x_convergence_filter_can_be_disabled(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"x": 300_000},
        cinnabar_x_convergence_filter="off",
        cinnabar_x_convergence_fallback_filter="off",
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="rbfe")
    assert cfg.cinnabar_x_convergence_filter is None
    assert cfg.cinnabar_x_convergence_fallback_filter is None


def test_cinnabar_x_convergence_filter_respects_user_override(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0.0, 1.0],
        eq_steps=1000,
        n_steps={"x": 300_000},
        cinnabar_x_convergence_filter=[0.9, 0.5],
        cinnabar_x_convergence_fallback_filter=[0.6, 1.5],
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="rbfe")
    assert cfg.cinnabar_x_convergence_filter == (0.9, 0.5)
    assert cfg.cinnabar_x_convergence_fallback_filter == (0.6, 1.5)


def test_enable_mcwat_propagates_from_fesim_args(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        eq_steps=100,
        enable_mcwat="no",
        n_steps={"z": 300_000},
    )
    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")
    assert cfg.enable_mcwat == "no"


def test_mcwat_fe_propagates_from_fesim_args(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    fe_args = FESimArgs(
        lambdas=[0, 1],
        eq_steps=100,
        mcwat_fe="on",
        n_steps={"z": 300_000},
    )

    cfg = SimulationConfig.from_sections(create, fe_args, protocol="abfe")

    assert fe_args.mcwat_fe == "yes"
    assert cfg.mcwat_fe == "yes"
    assert FESimArgs(mcwat_fe="off").mcwat_fe == "no"


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


def test_resolved_sim_config_handles_md(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    payload = {
        "protocol": "md",
        "backend": "local",
        "run": {"output_folder": str(tmp_path / "out")},
        "create": create.model_dump(),
        "fe_sim": {
            "eq_steps": 1000,
            "dt": 0.002,
            "temperature": 300.0,
        },
    }
    cfg = RunConfig.model_validate(payload)
    sim_cfg = cfg.resolved_sim_config()
    assert sim_cfg.fe_type == "md"
    assert sim_cfg.eq_steps == 2500
    assert sim_cfg.temperature == 300.0


def test_md_rejects_fe_only_fields(tmp_path: Path) -> None:
    create = _minimal_create(tmp_path)
    payload = {
        "protocol": "md",
        "backend": "local",
        "run": {"output_folder": str(tmp_path / "out")},
        "create": create.model_dump(),
        "fe_sim": {"lambdas": [0.0, 1.0]},  # FE-only field should be rejected for MD
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)
