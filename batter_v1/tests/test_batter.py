"""
Tests for the ABFESystem.
"""

# Import package, test suite, and other packages as needed
import sys
from pytest import fixture
import tempfile
from pathlib import Path
import pytest

from batter import ABFESystem, MABFESystem

from .data import B2AR_CAU_INPUT

def test_batter_imported():
    """Sample test, will always pass so long as the import statement worked."""
    assert "batter" in sys.modules


@fixture(scope='module')
def abfe_system():
    protein_file = B2AR_CAU_INPUT / 'protein_input.pdb'
    system_file = B2AR_CAU_INPUT / 'system_input.pdb'
    ligand_sdf_file = B2AR_CAU_INPUT / 'e5e.sdf'
    system_inpcrd = B2AR_CAU_INPUT / 'system_input.inpcrd'


    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)  # Convert to pathlib.Path for compatibility
        test_system = ABFESystem(folder=temp_path)
        test_system.create_system(
            system_name='B2AR',
            protein_input=protein_file,
            system_topology=system_file,
            system_coordinate=system_inpcrd,
            ligand_paths=[ligand_sdf_file],
            anchor_atoms=[
                'resid 113 and name CA',
                'resid 82 and name CA',
                'resid 316 and name CA'
            ],
            lipid_mol=['POPC']
        )
        yield test_system, temp_path  # Pass both the System object and the temporary directory
        

@fixture(scope='module')
def mabfe_system(abfe_system):
    protein_file = B2AR_CAU_INPUT / 'protein_input.pdb'
    system_file = B2AR_CAU_INPUT / 'system_input.pdb'
    ligand1_sdf_file = B2AR_CAU_INPUT / 'cau.sdf'
    ligand2_sdf_file = B2AR_CAU_INPUT / 'e5e.sdf'
    system_inpcrd = B2AR_CAU_INPUT / 'system_input.inpcrd'
    ligand_dict = {
        'cau': ligand1_sdf_file,
        'e5e': ligand2_sdf_file,  # Using the same ligand for both
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)  # Convert to pathlib.Path for compatibility
        test_system = MABFESystem(folder=temp_path)
        test_system.create_system(
            system_name='B2AR',
            protein_input=protein_file,
            system_topology=system_file,
            system_coordinate=system_inpcrd,
            ligand_paths=ligand_dict,
            anchor_atoms=[
                'resid 113 and name CA',
                'resid 82 and name CA',
                'resid 316 and name CA'
            ],
            lipid_mol=['POPC']
        )
        yield test_system, temp_path  # Pass both the System object and the temporary directory

@pytest.mark.parametrize('system_fixture', ["abfe_system", "mabfe_system"])
def test_system_exist(system_fixture, request):
    """Test creating a System object."""
    test_system, temp_dir = request.getfixturevalue(system_fixture)

    all_poses_dir = temp_dir / 'all-poses'
    assert all_poses_dir.exists(), "All poses directory is missing"

    # Check files and directories in the root of temp_dir
    expected_files = ['B2AR_docked.pdb', 'reference.pdb', 'pose0.pdb']
    for file_name in expected_files:
        assert (all_poses_dir / file_name).exists(), f"Missing file: {file_name}"

    # Check ligand force field directory and files
    ligand_ff_dir = temp_dir / 'ff'
    assert ligand_ff_dir.exists(), "Ligand force field directory 'ff' is missing"

    if system_fixture == "abfe_system":
        expected_ligand_files = ['lig.frcmod', 'lig.lib', 'lig.mol2']
    elif system_fixture == "mabfe_system":
        expected_ligand_files = ['e5e.frcmod', 'e5e.lib', 'e5e.mol2',
                                 'cau.frcmod', 'cau.lib', 'cau.mol2']
    for file_name in expected_ligand_files:
        assert (ligand_ff_dir / file_name).exists(), f"Missing ligand file: {file_name}"


@pytest.mark.parametrize('system_fixture', ["abfe_system", "mabfe_system"])
def test_system_equil(system_fixture, request):
    """Test equilibration of a System object."""
    test_system, temp_dir = request.getfixturevalue(system_fixture)

    input_file = B2AR_CAU_INPUT / 'lipid.in'
    test_system.prepare(
        stage='equil',
        input_file=input_file
    )

    equil_dir = temp_dir / 'equil' / 'pose0'
    assert equil_dir.exists(), "Equilibration directory is missing"
    
    expected_files = ['full.hmr.prmtop', 'mdin-03', 'SLURMM-run', 'disang03.rest']
    for file_name in expected_files:
        assert (equil_dir / file_name).exists(), f"Missing file: {file_name}"