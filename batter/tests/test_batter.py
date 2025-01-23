"""
Tests for the ABFESystem.
"""

# Import package, test suite, and other packages as needed
import sys
from pytest import fixture
import tempfile
from pathlib import Path

from batter import ABFESystem

from .data import MOR_MP_INPUT

def test_batter_imported():
    """Sample test, will always pass so long as the import statement worked."""
    assert "batter" in sys.modules


@fixture
def abfe_system():
    protein_file = MOR_MP_INPUT / 'protein_input.pdb'
    system_file = MOR_MP_INPUT / 'system_input.pdb'
    ligand_file = MOR_MP_INPUT / 'ligand_input.pdb'
    system_inpcrd = MOR_MP_INPUT / 'system_input.inpcrd'


    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)  # Convert to pathlib.Path for compatibility
        test_system = ABFESystem(folder=temp_path)
        test_system.create_system(
            system_name='7T2G',
            protein_input=protein_file,
            system_topology=system_file,
            system_coordinate=system_inpcrd,
            ligand_paths=[ligand_file],
            lipid_mol=['POPC']
        )
        yield test_system, temp_path  # Pass both the System object and the temporary directory


def test_system_exist(abfe_system):
    """Test creating a System object."""
    test_system, temp_dir = abfe_system

    all_poses_dir = temp_dir / 'all-poses'
    assert all_poses_dir.exists(), "All poses directory is missing"

    # Check files and directories in the root of temp_dir
    expected_files = ['7T2G_docked.pdb', 'reference.pdb', 'pose0.pdb']
    for file_name in expected_files:
        assert (all_poses_dir / file_name).exists(), f"Missing file: {file_name}"

    # Check ligand force field directory and files
    ligand_ff_dir = temp_dir / 'ff'
    assert ligand_ff_dir.exists(), "Ligand force field directory 'ff' is missing"

    expected_ligand_files = ['sc3.frcmod', 'sc3.lib', 'sc3.mol2']
    for file_name in expected_ligand_files:
        assert (ligand_ff_dir / file_name).exists(), f"Missing ligand file: {file_name}"


def test_system_equil(abfe_system):
    """Test equilibration of a System object."""
    test_system, temp_dir = abfe_system

    input_file = MOR_MP_INPUT / 'lipid.in'
    test_system.prepare(
        stage='equil',
        input_file=input_file
    )

    equil_dir = temp_dir / 'equil' / 'pose0'
    assert equil_dir.exists(), "Equilibration directory is missing"
    
    expected_files = ['full.hmr.prmtop', 'mdin-03', 'SLURMM-run', 'disang03.rest']
    for file_name in expected_files:
        assert (equil_dir / file_name).exists(), f"Missing file: {file_name}"