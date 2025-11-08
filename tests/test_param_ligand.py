import sys
from pathlib import Path
from typing import Iterable

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batter.param.ligand import (  # noqa: E402
    FORBIDDEN_MOL_NAMES,
    LigandFactory,
    _base26_triplet,
    _convert_mol_name_to_unique,
    _stable_hash_int,
)


def _sample_ligand(name: str) -> Path:
    return ROOT / "tests" / "data" / "ligands" / name


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "aaa"),
        (1, "aab"),
        (25, "aaz"),
        (26, "aba"),
        (26**3 + 7, "aah"),
    ],
)
def test_base26_triplet_wraparound(value: int, expected: str) -> None:
    assert _base26_triplet(value) == expected


def test_stable_hash_int_is_deterministic() -> None:
    value = _stable_hash_int("batter-ligand")
    for _ in range(5):
        assert _stable_hash_int("batter-ligand") == value


@pytest.mark.parametrize(
    "mol_name,ind,smiles,expected",
    [
        ("Acetone", 0, "CC(=O)C", "ace"),
        ("A", 0, "*", "aaa"),
        ("-+=", 5, "*", _base26_triplet(5)),
        ("", 42, "*", _base26_triplet(42)),
    ],
)
def test_convert_mol_name_to_unique_basic_cases(
    mol_name: str, ind: int, smiles: str, expected: str
) -> None:
    out = _convert_mol_name_to_unique(
        mol_name=mol_name,
        ind=ind,
        smiles=smiles,
        exist_mol_names=set(),
    )
    assert out == expected


def test_convert_mol_name_to_unique_respects_forbidden_set() -> None:
    existing = set(FORBIDDEN_MOL_NAMES)
    name = _convert_mol_name_to_unique(
        mol_name="and",
        ind=3,
        smiles="C1=CC=CC=C1",
        exist_mol_names=existing,
    )
    assert name not in FORBIDDEN_MOL_NAMES
    assert len(name) == 3


def test_collision_triggers_smiles_hash() -> None:
    mol_name = "Acetone"
    smiles = "CC(=O)C"
    base = "ace"
    seed = _stable_hash_int(smiles)
    blocked: set[str] = {base, _base26_triplet(seed)}
    out = _convert_mol_name_to_unique(
        mol_name=mol_name,
        ind=0,
        smiles=smiles,
        exist_mol_names=blocked,
    )
    assert out not in blocked
    assert len(out) == 3


def test_collision_without_smiles_uses_hash_of_base() -> None:
    base = "naa"
    exist = {base}
    seed = _stable_hash_int(base)
    exist |= {_base26_triplet(seed + k) for k in range(3)}
    out = _convert_mol_name_to_unique(
        mol_name=base,
        ind=7,
        smiles="",
        exist_mol_names=exist,
    )
    assert out not in exist
    assert len(out) == 3


def test_determinism_same_inputs() -> None:
    exist = {"ace"}
    args = dict(mol_name="Acetone", ind=1, smiles="CC(=O)C")
    out1 = _convert_mol_name_to_unique(exist_mol_names=exist, **args)
    out2 = _convert_mol_name_to_unique(exist_mol_names=exist, **args)
    assert out1 == out2


def test_no_three_digits() -> None:
    out = _convert_mol_name_to_unique(
        mol_name="129",
        ind=1,
        smiles="CC(=O)C",
        exist_mol_names=set(),
    )
    assert out == "l12"


def test_unique_names_for_conflicting_inputs() -> None:
    exist: set[str] = set()
    out1 = _convert_mol_name_to_unique("###", ind=0, smiles="S", exist_mol_names=exist)
    exist.add(out1)
    out2 = _convert_mol_name_to_unique("###", ind=0, smiles="Cl", exist_mol_names=exist)
    assert out2 != out1
    assert len(out2) == 3


@pytest.mark.parametrize("filename", ["KW6356_h.sdf", "2_candidates.sdf"])
def test_ligand_factory_creates_sdf_processing(filename: str, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    factory = LigandFactory()
    lig = factory.create_ligand(
        ligand_file=_sample_ligand(filename),
        index=0,
        output_dir=output_dir,
        ligand_name="lig",
        retain_lig_prot=True,
        ligand_ff="gaff2",
    )
    assert Path(lig.ligand_sdf_path).exists()
    assert len(lig.name) == 3
    assert lig.name.lower() == lig.name


def test_ligand_factory_rejects_pdb(tmp_path: Path) -> None:
    factory = LigandFactory()
    with pytest.raises(ValueError):
        factory.create_ligand(
            ligand_file=tmp_path / "ligand.pdb",
            index=0,
            output_dir=tmp_path,
        )
