from batter._internal.ops.helpers import get_ligand_candidates
from tests.data import (
    TWO_CANDIDATE_SDF,
    THREE_CANDIDATE_SDF,
    data_path,
)


def test_get_ligand_candidates_with_two_candidates():
    candidates = get_ligand_candidates(TWO_CANDIDATE_SDF)

    assert len(candidates) == 5


def test_get_ligand_candidates_with_three_candidates():
    candidates = get_ligand_candidates(THREE_CANDIDATE_SDF)

    assert len(candidates) == 3
    assert candidates == [0, 2, 3]


def test_data_path_guard_rails():
    lig_dir = data_path("ligands")
    assert lig_dir.is_dir()
