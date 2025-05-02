from batter.ligand_process import LigandFactory
from .data import (
    KW6356,
    KW6356_H,
)
import pytest
from loguru import logger
import json
import tempfile

def test_processligand_H():
    output_dir = str(tempfile.mkdtemp())
    ligand_factory = LigandFactory()
    ligand_file=str(KW6356_H)
    ligand_name = 'KW6'
    ligand = ligand_factory.create_ligand(
            ligand_file=ligand_file,
            index=0,
            output_dir=output_dir,
            ligand_name=ligand_name,
            retain_lig_prot=True,
            ligand_ff='gaff2'
    ) 
    ligand.generate_unique_name([])
    ligand.prepare_ligand_parameters()

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
    ligand.generate_unique_name([])
    ligand.prepare_ligand_parameters()