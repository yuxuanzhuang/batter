import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from batter import MABFESystem

import pytest
from batter.tests.data import (pipeline_ligands_files,
                    pipeline_reference_files,
                    pipeline_ligands_dict_json,
                    pipeline_sim_extra_rest_json,
                    pipeline_sim_dist_rest_json,
                    pipeline_abfe_input,
                    pipeline_equil_system,
)

@pytest.fixture
def system(
    request,
    tmp_path,
    pipeline_ligands_dict_json=pipeline_ligands_dict_json,
    pipeline_reference_files=pipeline_reference_files,
    pipeline_ligands_files=pipeline_ligands_files,
    pipeline_abfe_input=pipeline_abfe_input,
):
    """
    Build a MABFESystem based on a *given* simulation_input_json path.
    """
    # simulation_input_json path is provided through param (indirect)
    simulation_input_json = Path(request.param)

    # --- ligands mapping ---
    with open(pipeline_ligands_dict_json, "r") as f:
        ligand_data = json.load(f)
    ligand_data = {
        k: (pipeline_ligands_files / v).as_posix()
        for k, v in ligand_data.items()
    }

    # --- simulation input ---
    with open(simulation_input_json, "r") as f:
        simulation_inputs = json.load(f)

    system_name = simulation_inputs["protein"]
    protein_input = pipeline_reference_files / simulation_inputs["protein_input"]
    system_input = pipeline_reference_files / simulation_inputs["system_input"]
    system_rst7_input = pipeline_reference_files / simulation_inputs["system_rst7_input"]
    anchor_atoms = simulation_inputs["anchor_atoms"]
    extra_restraints = simulation_inputs.get("extra_restraints", None)

    # --- create & run (dry-run) ---
    sys_obj = MABFESystem(folder=tmp_path)

    sys_obj.create_system(
        system_name=system_name,
        protein_input=protein_input,
        system_topology=system_input,
        system_coordinate=system_rst7_input,
        ligand_paths=ligand_data,
        overwrite=True,
        retain_lig_prot=True,
        lipid_mol=["POPC"],
        anchor_atoms=anchor_atoms,
    )

    sys_obj.run_pipeline(
        input_file=pipeline_abfe_input,
        dry_run=True,
        only_equil=True,
        extra_restraints=extra_restraints,
    )

    return sys_obj


@pytest.mark.parametrize(
    "system",
    [pipeline_sim_extra_rest_json,
     pipeline_sim_dist_rest_json],
    indirect=True,
)
def test_existence_of_output_files(system):
    """
    # test existence of output files
    """
    expected_files = [
        'all-poses',
        'equil',
        'ff',

        # equil
        'equil/pose0',
        'equil/pose1',
        'equil/run_files',

        # pose files
        'equil/pose0/full.pdb',
        'equil/pose0/vac.pdb',
        'equil/pose0/full.hmr.prmtop',
        'equil/pose0/full.inpcrd',
        'equil/pose0/mdin-00'
    ]
    for file in expected_files:
        folder_path = f'{system.output_dir}/{file}'
        assert os.path.exists(folder_path), f"Expected file {file} does not exist."

@pytest.mark.skip(reason="disable temporarily")
def test_build_fe(tmp_path):

    with open(pipeline_simulation_input_json, "r") as f:
        simulation_inputs = json.load(f)

    extra_restraints = simulation_inputs.get("extra_restraints", None)

    # copy pipeline_equil_system folder to a temporary directory
    tmp_equil_system = tmp_path / "pipeline_equil_system"
    shutil.copytree(pipeline_equil_system, tmp_equil_system)

    # create MABFESystem instance
    system = MABFESystem(folder=tmp_equil_system)

    system.run_pipeline(
        input_file=pipeline_abfe_input,
        dry_run=True,  # Dry run for testing
        extra_restraints=extra_restraints,
    )