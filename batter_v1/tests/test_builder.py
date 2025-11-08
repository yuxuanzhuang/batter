import pytest
from batter_v1.tests.data import two_candidates, three_candidates
from batter_v1.builder import get_ligand_candidates
import numpy as np


def test_get_ligand_candidates_with_two_candidates():
    candidates = get_ligand_candidates(two_candidates)
    
    assert len(candidates) == 5


def test_get_ligand_candidates_with_three_candidates():
    candidates = get_ligand_candidates(three_candidates)
    
    assert len(candidates) == 3
    assert candidates == [0, 2, 3]