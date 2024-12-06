"""
Unit and regression tests for the batter package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np
import pytest
from pytest import fixture
import tempfile
from pathlib import Path

import batter
from batter import System

from .data import MOR_MP_INPUT


def test_batter_imported():
    """Sample test, will always pass so long as the import statement worked."""
    assert "batter" in sys.modules


@fixture
def system():
    protein_file = MOR_MP_INPUT / 'protein_input.pdb'
    system_file = MOR_MP_INPUT / 'system_input.pdb'
    ligand_file = MOR_MP_INPUT / 'ligand_input.pdb'
    system_inpcrd = MOR_MP_INPUT / 'system_input.inpcrd'

    # Read the last line of the inpcrd file to get dimensions
    with open(system_inpcrd) as f:
        lines = f.readlines()
        box = np.array([float(x) for x in lines[-1].split()])

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)  # Convert to pathlib.Path for compatibility
        test_system = System(
            system_name='7T2G',
            protein_input=protein_file,
            system_input=system_file,
            system_dimensions=box,
            ligand_path=ligand_file,
            output_dir=temp_path,
            lipid_mol=['POPC']
        )
        yield test_system, temp_path  # Pass both the System object and the temporary directory


def test_system_exist(system):
    """Test creating a System object."""
    test_system, temp_dir = system

    all_poses_dir = temp_dir / 'all-poses'
    assert all_poses_dir.exists(), "All poses directory is missing"

    # Check files and directories in the root of temp_dir
    expected_files = ['7T2G_docked.pdb', 'reference.pdb', 'pose0.pdb']
    for file_name in expected_files:
        assert (all_poses_dir / file_name).exists(), f"Missing file: {file_name}"

    # Check ligand force field directory and files
    ligand_ff_dir = temp_dir / 'ff'
    assert ligand_ff_dir.exists(), "Ligand force field directory 'ff' is missing"

    expected_ligand_files = ['ligand.frcmod', 'ligand.lib', 'ligand.mol2']
    for file_name in expected_ligand_files:
        assert (ligand_ff_dir / file_name).exists(), f"Missing ligand file: {file_name}"


def test_system_equil(system):
    """Test equilibration of a System object."""
    test_system, temp_dir = system

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