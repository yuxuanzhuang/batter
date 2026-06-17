from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

import batter._internal.ops.ring_repair as ring_repair
from batter._internal.ops.ring_repair import (
    RingPenetrationRepairResult,
    _candidate_protein_residue_indices,
    _ring_penetrations,
    repair_ring_penetrations,
    repair_ring_penetrations_with_ligand_perturbations,
    repair_ring_penetrations_with_sidechain_rotamers,
)


_SEVEN_DFP_EXAMPLE = Path(
    "/home/users/yuzhuang/yuzhuang_scratch/az_fep/D2R/ABFE_runs_apo_rest/"
    "7jvr_rest_abfe/executions/rep1/simulations/7DFP_REP1/equil"
)
_SEVEN_DFP_PENETRATED_COORD = _SEVEN_DFP_EXAMPLE / "full.inpcrd"
_SIX_X19_EXAMPLE = Path(
    "/scratch/users/yuzhuang/az_fep/MD/6x1a_all_resolved/executions/rep1/"
    "simulations/6X19_MOLECULE3_REP1/equil"
)
_SIX_X19_PENETRATED_COORD = _SIX_X19_EXAMPLE / "full.inpcrd.pre_ring_repair"
if not _SIX_X19_PENETRATED_COORD.exists():
    _SIX_X19_PENETRATED_COORD = (
        _SIX_X19_EXAMPLE / "full.inpcrd.pre_iterative_ring_repair"
    )
if not _SIX_X19_PENETRATED_COORD.exists():
    _SIX_X19_PENETRATED_COORD = _SIX_X19_EXAMPLE / "full.inpcrd"


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
    assert result.selected_mode == "protein_sidechain"
    assert result.repaired is True
    assert result.initial_penetrations == 2
    assert result.final_penetrations == 0
    assert result.residue == {"index": 78, "resid": 79, "resname": "PHE"}
    assert result.rotations == [
        {"label": "chi1", "axis": "CA-CB", "angle_degrees": 45.0}
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
def test_ligand_perturbation_ring_repair_resolves_7dfp_example() -> None:
    pmd = pytest.importorskip("parmed")

    structure = pmd.load_file(
        str(_SEVEN_DFP_EXAMPLE / "full.prmtop"),
        str(_SEVEN_DFP_PENETRATED_COORD),
    )
    result = repair_ring_penetrations_with_ligand_perturbations(
        structure,
        ligand_resname="7df",
        ligand_label="7DFP_REP1",
    )

    assert result.mode == "ligand"
    assert result.selected_mode == "ligand"
    assert result.repaired is True
    assert result.initial_penetrations == 2
    assert result.final_penetrations == 0
    assert result.residue == {"index": 274, "resid": 275, "resname": "7df"}
    assert result.rotations == []
    assert result.perturbations
    assert all(
        item["label"] == "ligand_local_perturbation"
        for item in result.perturbations
    )

    final_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    assert final_pairs == []


@pytest.mark.skipif(
    not (
        (_SEVEN_DFP_EXAMPLE / "full.prmtop").exists()
        and _SEVEN_DFP_PENETRATED_COORD.exists()
    ),
    reason="7DFP local ring-penetration example is not available",
)
def test_auto_ring_repair_prefers_sidechain_for_7dfp_example() -> None:
    pmd = pytest.importorskip("parmed")

    structure = pmd.load_file(
        str(_SEVEN_DFP_EXAMPLE / "full.prmtop"),
        str(_SEVEN_DFP_PENETRATED_COORD),
    )
    result = repair_ring_penetrations(
        structure,
        fix_mode="auto",
        ligand_resname="7df",
        ligand_label="7DFP_REP1",
    )

    assert result.mode == "auto"
    assert result.selected_mode == "protein_sidechain"
    assert result.repaired is True
    assert result.final_penetrations == 0


@pytest.mark.skipif(
    not (
        (_SIX_X19_EXAMPLE / "full.prmtop").exists()
        and _SIX_X19_PENETRATED_COORD.exists()
    ),
    reason="6X19 local ring-penetration example is not available",
)
def test_auto_ring_repair_resolves_multisite_6x19_example() -> None:
    pmd = pytest.importorskip("parmed")

    structure = pmd.load_file(
        str(_SIX_X19_EXAMPLE / "full.prmtop"),
        str(_SIX_X19_PENETRATED_COORD),
    )
    initial_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    if not initial_pairs:
        pytest.skip("6X19 local example coordinates are already repaired")
    assert len(initial_pairs) == 3

    result = repair_ring_penetrations(
        structure,
        fix_mode="auto",
        ligand_resname="6x1",
        ligand_label="6X19_MOLECULE3_REP1",
    )

    assert result.mode == "auto"
    assert result.selected_mode == "auto"
    assert result.repaired is True
    assert result.initial_penetrations == 3
    assert result.final_penetrations == 0
    assert [step["mode"] for step in result.steps] == [
        "protein_sidechain",
        "ligand",
        "ligand",
    ]
    assert result.steps[0]["residue"] == {
        "index": 173,
        "resid": 174,
        "resname": "TRP",
    }
    assert result.steps[1]["residue"] == {
        "index": 395,
        "resid": 396,
        "resname": "6x1",
    }
    assert result.steps[1]["rotations"] == []
    assert result.steps[1]["perturbations"]
    assert result.steps[2]["rotations"] == []
    assert result.steps[2]["perturbations"]
    assert result.rotations == [
        {"label": "chi1", "axis": "CA-CB", "angle_degrees": 120.0}
    ]
    assert result.perturbations
    assert all(
        item["label"] == "ligand_local_perturbation"
        for item in result.perturbations
    )

    final_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    assert final_pairs == []


def test_auto_ring_repair_falls_back_to_ligand(monkeypatch) -> None:
    pmd = pytest.importorskip("parmed")
    structure = pmd.Structure()
    state = {"count": 3}

    def fake_ring_penetrations(*args, **kwargs):
        return [(1, 2)] * state["count"], [[3, 4, 5]] * state["count"], None

    def fake_sidechain(*args, **kwargs):
        return RingPenetrationRepairResult(
            mode="protein_sidechain",
            initial_penetrations=3,
            final_penetrations=3,
            repaired=False,
        )

    def fake_ligand(*args, **kwargs):
        state["count"] = 0
        return RingPenetrationRepairResult(
            mode="ligand",
            selected_mode="ligand",
            initial_penetrations=3,
            final_penetrations=0,
            repaired=True,
        )

    monkeypatch.setattr(ring_repair, "_ring_penetrations", fake_ring_penetrations)
    monkeypatch.setattr(
        ring_repair,
        "repair_ring_penetrations_with_sidechain_rotamers",
        fake_sidechain,
    )
    monkeypatch.setattr(
        ring_repair,
        "repair_ring_penetrations_with_ligand_perturbations",
        fake_ligand,
    )

    result = repair_ring_penetrations(
        structure,
        fix_mode="auto",
        ligand_resname="lig",
        ligand_label="example",
    )

    assert result.mode == "auto"
    assert result.selected_mode == "ligand"
    assert result.repaired is True


def test_nonaromatic_sidechain_penetrating_ligand_ring_is_protein_candidate() -> None:
    topology = nx.Graph()
    topology.add_node(
        1, resname="LYS", residue_idx=10, resid=42, name="CG", segid="A"
    )
    topology.add_node(
        2, resname="LYS", residue_idx=10, resid=42, name="CD", segid="A"
    )
    for node in range(3, 9):
        topology.add_node(
            node, resname="lig", residue_idx=99, resid=271, name=f"C{node}", segid=""
        )

    candidates = _candidate_protein_residue_indices(
        topology,
        pairs=[(1, 2)],
        rings=[[3, 4, 5, 6, 7, 8]],
    )

    assert candidates == [10]


def test_ring_repair_dispatch_requires_ligand_resname_for_ligand_mode() -> None:
    pmd = pytest.importorskip("parmed")
    structure = pmd.Structure()

    with pytest.raises(ValueError, match="ligand_resname"):
        repair_ring_penetrations(structure, fix_mode="ligand")
