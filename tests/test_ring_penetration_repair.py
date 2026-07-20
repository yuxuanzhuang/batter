from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import batter._internal.ops.ring_repair as ring_repair
from batter._internal.ops.ring_repair import (
    RingPenetrationRepairResult,
    _candidate_protein_residue_indices,
    _ring_penetrations,
    repair_ring_penetrations,
)


_RING_FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "ring_penetration"
_SIX_X19_EXAMPLE = _RING_FIXTURE_DIR / "6x19"


def _load_6x19_penetrated_structure():
    pmd = pytest.importorskip("parmed")

    return pmd.load_file(
        str(_SIX_X19_EXAMPLE / "full.prmtop"),
        str(_SIX_X19_EXAMPLE / "full.inpcrd"),
    )


def test_ring_penetration_detection_finds_multisite_6x19_fixture() -> None:
    structure = _load_6x19_penetrated_structure()
    initial_pairs, rings, topology = _ring_penetrations(
        structure, structure.coordinates
    )

    assert len(initial_pairs) == 3
    assert [
        [topology.nodes[idx]["resname"] for idx in pair]
        for pair in initial_pairs
    ] == [["6x1", "6x1"], ["MET", "MET"], ["PHE", "PHE"]]
    assert [
        sorted({topology.nodes[idx]["resname"] for idx in ring})
        for ring in rings
    ] == [["TRP"], ["6x1"], ["6x1"]]


def test_auto_ring_repair_resolves_multisite_6x19_fixture() -> None:
    structure = _load_6x19_penetrated_structure()
    initial_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
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
        "index": 0,
        "resid": 1,
        "resname": "TRP",
    }
    assert result.steps[1]["residue"] == {
        "index": 3,
        "resid": 4,
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


def test_ring_penetration_detection_includes_ligand_hydrogen_bonds() -> None:
    pmd = pytest.importorskip("parmed")
    structure = pmd.Structure()

    ring_atoms = []
    for name in ["ND1", "CE1", "NE2", "CD2", "CG"]:
        atom = pmd.Atom(name=name, type="C", atomic_number=6, mass=12.0)
        structure.add_atom(atom, "HID", 66)
        ring_atoms.append(atom)

    carbon = pmd.Atom(name="C19", type="C", atomic_number=6, mass=12.0)
    hydrogen = pmd.Atom(name="H23", type="H", atomic_number=1, mass=1.0)
    structure.add_atom(carbon, "4qk", 287)
    structure.add_atom(hydrogen, "4qk", 287)

    for first, second in zip(ring_atoms, [*ring_atoms[1:], ring_atoms[0]]):
        structure.bonds.append(pmd.Bond(first, second))
    structure.bonds.append(pmd.Bond(carbon, hydrogen))

    coordinates = []
    for index in range(5):
        angle = 2.0 * np.pi * index / 5.0
        coordinates.append([np.cos(angle), np.sin(angle), 0.0])
    coordinates.extend([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    structure.coordinates = np.asarray(coordinates, dtype=float)

    pairs, rings, topology = _ring_penetrations(structure, structure.coordinates)

    assert pairs == [(6, 7)]
    assert [topology.nodes[node]["name"] for node in pairs[0]] == ["C19", "H23"]
    assert {topology.nodes[node]["name"] for node in rings[0]} == {
        "ND1",
        "CE1",
        "NE2",
        "CD2",
        "CG",
    }


def test_ring_repair_dispatch_requires_ligand_resname_for_ligand_mode() -> None:
    pmd = pytest.importorskip("parmed")
    structure = pmd.Structure()

    with pytest.raises(ValueError, match="ligand_resname"):
        repair_ring_penetrations(structure, fix_mode="ligand")
