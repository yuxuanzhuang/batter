from batter.ligand_process import LigandFactory
from .data import (
    g1i,
    KW6356_H,
)
import pytest
from loguru import logger
import json
import tempfile
import os

@pytest.fixture(params=[
    "gaff",
    "gaff2",
    "openff-2.2.1",
])
def ligand_ff(request):
    return request.param

def test_processligand_H(ligand_ff):
    output_dir = str(tempfile.mkdtemp())
    ligand_factory = LigandFactory()
    ligand_file=str(g1i)
    ligand_name = 'g1i'
    ligand = ligand_factory.create_ligand(
            ligand_file=ligand_file,
            index=0,
            output_dir=output_dir,
            ligand_name=ligand_name,
            retain_lig_prot=True,
            ligand_ff=ligand_ff,
    ) 
    ligand.prepare_ligand_parameters()
    
    # check prmtop is created
    assert os.path.exists(f'{output_dir}/{ligand_name}.prmtop')
    assert os.path.exists(f'{output_dir}/{ligand_name}.sdf')

@pytest.mark.skip(reason="Ligand processing without H tests are skipped for now")
def test_processligand_noH():
    output_dir = str(tempfile.mkdtemp())
    ligand_factory = LigandFactory()
    ligand_file=str(KW6356)
    ligand_name = 'KW6'
    ligand = ligand_factory.create_ligand(
            ligand_file=ligand_file,
            index=0,
            output_dir=output_dir,
            ligand_name=ligand_name,
            retain_lig_prot=False,
            ligand_ff='gaff2'
    ) 
    ligand.prepare_ligand_parameters()