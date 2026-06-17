from __future__ import annotations

from pathlib import Path

import pytest

from batter._internal.ops.ring_repair import (
    _ring_penetrations,
    repair_ring_penetrations,
    repair_ring_penetrations_with_ligand_torsions,
    repair_ring_penetrations_with_sidechain_rotamers,
)


_SEVEN_DFP_EXAMPLE = Path(
    "/home/users/yuzhuang/yuzhuang_scratch/az_fep/D2R/ABFE_runs_apo_rest/"
    "6cm4_rest_abfe/executions/rep1/simulations/7DFP_REP1/equil"
)
_SEVEN_DFP_PENETRATED_COORD = (
    _SEVEN_DFP_EXAMPLE / "full.inpcrd.pre_ring_repair_propagation_fix"
)
if not _SEVEN_DFP_PENETRATED_COORD.exists():
    _SEVEN_DFP_PENETRATED_COORD = _SEVEN_DFP_EXAMPLE / "full.inpcrd"


@pytest.mark.skipif(
    not (
        (_SEVEN_DFP_EXAMPLE / "full.prmtop").exists()
        and _SEVEN_DFP_PENETRATED_COORD.exists()
    ),
    reason="7DFP local ring-penetration example is not available",
)
def test_sidechain_ring_repair_resolves_7dfp_example() -> None:
    pmd = pytest.importorskip("parmed")

    structure = pmd.load_file(
        str(_SEVEN_DFP_EXAMPLE / "full.prmtop"),
        str(_SEVEN_DFP_PENETRATED_COORD),
    )
    initial_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    assert len(initial_pairs) == 2

    result = repair_ring_penetrations_with_sidechain_rotamers(
        structure, ligand_label="7DFP_REP1"
    )

    assert result.mode == "protein_sidechain"
    assert result.repaired is True
    assert result.initial_penetrations == 2
    assert result.final_penetrations == 0
    assert result.residue == {"index": 77, "resid": 78, "resname": "PHE"}
    assert result.rotations == [
        {"label": "chi1", "axis": "CA-CB", "angle_degrees": -90.0}
    ]

    final_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    assert final_pairs == []


@pytest.mark.skipif(
    not (
        (_SEVEN_DFP_EXAMPLE / "full.prmtop").exists()
        and _SEVEN_DFP_PENETRATED_COORD.exists()
    ),
    reason="7DFP local ring-penetration example is not available",
)
def test_ligand_torsion_ring_repair_resolves_7dfp_example() -> None:
    pmd = pytest.importorskip("parmed")

    structure = pmd.load_file(
        str(_SEVEN_DFP_EXAMPLE / "full.prmtop"),
        str(_SEVEN_DFP_PENETRATED_COORD),
    )
    result = repair_ring_penetrations_with_ligand_torsions(
        structure,
        ligand_resname="7df",
        ligand_label="7DFP_REP1",
    )

    assert result.mode == "ligand"
    assert result.repaired is True
    assert result.initial_penetrations == 2
    assert result.final_penetrations == 0
    assert result.residue == {"index": 270, "resid": 271, "resname": "7df"}
    assert result.rotations == [
        {"label": "ligand_torsion", "axis": "C16-C15", "angle_degrees": -45.0}
    ]

    final_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    assert final_pairs == []


def test_ring_repair_dispatch_requires_ligand_resname_for_ligand_mode() -> None:
    pmd = pytest.importorskip("parmed")
    structure = pmd.Structure()

    with pytest.raises(ValueError, match="ligand_resname"):
        repair_ring_penetrations(structure, fix_mode="ligand")
