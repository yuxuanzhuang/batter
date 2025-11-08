from batter_v1.ligand_process import (
    LigandFactory,
    _base26_triplet,
    _stable_hash_int,
    _convert_mol_name_to_unique
)

from .data import (
    g1i,
    KW6356_H,
)
import pytest
from loguru import logger
import json
import tempfile
import os
import re


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

def test_simple_three_letter_base():
    # "Acetone" → "ace"
    exist = set()
    out = _convert_mol_name_to_unique("Acetone", ind=0, smiles="CC(=O)C", exist_mol_names=exist)
    assert out == "ace"
    assert re.fullmatch(r"[a-z]{3}", out)


def test_short_name_is_padded_with_index_triplet():
    exist = set()
    out = _convert_mol_name_to_unique("A", ind=0, smiles="*", exist_mol_names=exist)
    assert len(out) == 3
    assert out.startswith("a")  # "a" then padded
    assert re.fullmatch(r"[a-z]{3}", out)


def test_name_with_no_letters_uses_index_triplet():
    exist = set()
    out = _convert_mol_name_to_unique("-+=", ind=5, smiles="*", exist_mol_names=exist)
    assert out == _base26_triplet(5)


def test_empty_name_uses_index_triplet():
    exist = set()
    out = _convert_mol_name_to_unique("", ind=42, smiles="*", exist_mol_names=exist)
    assert out == _base26_triplet(42)


def test_collision_triggers_smiles_hash():
    mol_name = "Acetone"  # → "ace"
    smiles = "CC(=O)C"
    base = "ace"
    seed = _stable_hash_int(smiles)
    cand0 = _base26_triplet(seed)
    exist = {base, cand0}
    out = _convert_mol_name_to_unique(mol_name, ind=0, smiles=smiles, exist_mol_names=exist)
    assert out not in exist
    assert re.fullmatch(r"[a-z]{3}", out)


def test_collision_without_smiles_uses_base_hash():
    mol_name = "naa"  # normalized base = "naa"
    base = "naa"
    exist = {base}
    seed = _stable_hash_int(base)
    # block a couple of hash-derived candidates
    exist |= {_base26_triplet(seed + k) for k in range(3)}
    out = _convert_mol_name_to_unique(mol_name, ind=7, smiles="", exist_mol_names=exist)
    assert out not in exist
    assert re.fullmatch(r"[a-z]{3}", out)


def test_determinism_same_inputs():
    exist = {"ace"}
    args = dict(mol_name="Acetone", ind=1, smiles="CC(=O)C")
    out1 = _convert_mol_name_to_unique(exist_mol_names=exist, **args)
    out2 = _convert_mol_name_to_unique(exist_mol_names=exist, **args)
    assert out1 == out2


def test_no_three_digits():
    args = dict(mol_name="129", ind=1, smiles="CC(=O)C")
    out = _convert_mol_name_to_unique(exist_mol_names=[], **args)
    assert out == 'l12'


def test_uniqueness_when_base_is_empty_and_conflict():
    exist = set()
    out1 = _convert_mol_name_to_unique("###", ind=0, smiles="S", exist_mol_names=exist)
    exist.add(out1)
    out2 = _convert_mol_name_to_unique("###", ind=0, smiles="Cl", exist_mol_names=exist)
    assert out2 != out1
    assert re.fullmatch(r"[a-z]{3}", out2)