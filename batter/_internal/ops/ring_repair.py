from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import parmed as pmd
from loguru import logger

from batter.analysis.sim_validation import check_ring_penetration


_WATER_RESNAMES = {"HOH", "TIP3", "WAT"}
_AROMATIC_SIDECHAIN_BONDS = {
    "PHE": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "TYR": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HIS": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HID": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HIE": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HIP": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "TRP": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
}
_ROTATION_ANGLES = (15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180)


@dataclass
class RingPenetrationRepairResult:
    mode: str | None = None
    initial_penetrations: int = 0
    final_penetrations: int = 0
    repaired: bool = False
    residue: dict[str, Any] | None = None
    rotations: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None
    min_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "initial_penetrations": self.initial_penetrations,
            "final_penetrations": self.final_penetrations,
            "repaired": self.repaired,
            "residue": self.residue,
            "rotations": self.rotations,
            "score": self.score,
            "min_distance": self.min_distance,
        }


def _is_hydrogen(atom) -> bool:
    try:
        atomic_number = int(getattr(atom, "atomic_number", 0) or 0)
    except Exception:
        atomic_number = 0
    if atomic_number == 1:
        return True
    name = str(getattr(atom, "name", "")).strip().upper()
    return name.startswith("H") or (len(name) > 1 and name[0].isdigit() and name[1] == "H")


def _parmed_heavy_topology(structure: pmd.Structure) -> nx.Graph:
    graph = nx.Graph()
    for atom in structure.atoms:
        resname = str(atom.residue.name).strip()
        if resname in _WATER_RESNAMES or _is_hydrogen(atom):
            continue
        graph.add_node(
            atom.idx + 1,
            segid=str(getattr(atom.residue, "chain", "") or ""),
            resname=resname,
            name=str(atom.name).strip(),
            resid=int(atom.residue.number) + 1,
            residue_idx=int(atom.residue.idx),
        )

    for bond in structure.bonds:
        first = bond.atom1.idx + 1
        second = bond.atom2.idx + 1
        if graph.has_node(first) and graph.has_node(second):
            graph.add_edge(first, second)
    return graph


def _ring_penetrations(
    structure: pmd.Structure, coordinates: np.ndarray
) -> tuple[list[tuple[int, int]], list[list[int]], nx.Graph]:
    topology = _parmed_heavy_topology(structure)
    pairs, rings = _ring_penetrations_with_topology(topology, coordinates)
    return pairs, rings, topology


def _ring_penetrations_with_topology(
    topology: nx.Graph, coordinates: np.ndarray
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    coord = {
        node: np.asarray(coordinates[node - 1], dtype=float)
        for node in topology.nodes
    }
    pairs, rings = check_ring_penetration(topology, coord, verbose=0)
    return [tuple(int(x) for x in pair) for pair in pairs], rings


def _candidate_aromatic_residue_indices(
    topology: nx.Graph,
    pairs: list[tuple[int, int]],
    rings: list[list[int]],
) -> list[int]:
    residue_indices: list[int] = []

    def _add(node: int) -> None:
        attrs = topology.nodes[node]
        if attrs["resname"] not in _AROMATIC_SIDECHAIN_BONDS:
            return
        residue_idx = int(attrs["residue_idx"])
        if residue_idx not in residue_indices:
            residue_indices.append(residue_idx)

    for pair, ring in zip(pairs, rings):
        ring_residue_indices = {int(topology.nodes[node]["residue_idx"]) for node in ring}
        ring_resnames = {str(topology.nodes[node]["resname"]) for node in ring}
        if len(ring_residue_indices) == 1 and next(iter(ring_resnames)) in _AROMATIC_SIDECHAIN_BONDS:
            _add(ring[0])
        for node in pair:
            _add(node)

    return residue_indices


def _atom_index_by_name(residue: pmd.Residue, atom_name: str) -> int | None:
    target = atom_name.strip()
    for atom in residue.atoms:
        if str(atom.name).strip() == target:
            return int(atom.idx)
    return None


def _full_bond_graph(structure: pmd.Structure) -> nx.Graph:
    graph = nx.Graph()
    for atom in structure.atoms:
        graph.add_node(int(atom.idx))
    for bond in structure.bonds:
        graph.add_edge(int(bond.atom1.idx), int(bond.atom2.idx))
    return graph


def _moving_component(
    bond_graph: nx.Graph, proximal_idx: int, distal_idx: int
) -> list[int]:
    graph = bond_graph.copy()
    if graph.has_edge(proximal_idx, distal_idx):
        graph.remove_edge(proximal_idx, distal_idx)
    if distal_idx not in graph:
        return []
    return sorted(int(idx) for idx in nx.node_connected_component(graph, distal_idx))


def _rotatable_bonds(
    structure: pmd.Structure, residue_idx: int, bond_graph: nx.Graph
) -> list[dict[str, Any]]:
    residue = structure.residues[residue_idx]
    templates = _AROMATIC_SIDECHAIN_BONDS.get(str(residue.name).strip(), ())
    bonds: list[dict[str, Any]] = []
    for label, proximal_name, distal_name in templates:
        proximal_idx = _atom_index_by_name(residue, proximal_name)
        distal_idx = _atom_index_by_name(residue, distal_name)
        if proximal_idx is None or distal_idx is None:
            continue
        moving = _moving_component(bond_graph, proximal_idx, distal_idx)
        if not moving:
            continue
        bonds.append(
            {
                "label": label,
                "proximal_name": proximal_name,
                "distal_name": distal_name,
                "proximal_idx": proximal_idx,
                "distal_idx": distal_idx,
                "moving": moving,
            }
        )
    return bonds


def _candidate_ligand_residue_indices(
    topology: nx.Graph,
    pairs: list[tuple[int, int]],
    rings: list[list[int]],
    *,
    ligand_resname: str | None,
) -> list[int]:
    residue_indices: list[int] = []
    expected = str(ligand_resname or "").strip()

    def _add(node: int) -> None:
        attrs = topology.nodes[node]
        resname = str(attrs["resname"])
        if expected:
            if resname != expected:
                return
        elif resname in _AROMATIC_SIDECHAIN_BONDS:
            return
        residue_idx = int(attrs["residue_idx"])
        if residue_idx not in residue_indices:
            residue_indices.append(residue_idx)

    for pair, ring in zip(pairs, rings):
        for node in pair:
            _add(node)
        for node in ring:
            _add(node)
    return residue_indices


def _impacted_ligand_atom_indices(
    topology: nx.Graph,
    pairs: list[tuple[int, int]],
    rings: list[list[int]],
    *,
    residue_idx: int,
) -> set[int]:
    impacted: set[int] = set()
    for pair, ring in zip(pairs, rings):
        for node in [*pair, *ring]:
            if int(topology.nodes[node]["residue_idx"]) == int(residue_idx):
                impacted.add(int(node) - 1)
    return impacted


def _ligand_heavy_bond_graph(residue: pmd.Residue) -> nx.Graph:
    graph = nx.Graph()
    residue_atom_indices = {int(atom.idx) for atom in residue.atoms}
    for atom in residue.atoms:
        if _is_hydrogen(atom):
            continue
        graph.add_node(int(atom.idx))
    for atom in residue.atoms:
        first = int(atom.idx)
        for bond in atom.bonds:
            other = bond.atom1 if bond.atom2 is atom else bond.atom2
            second = int(other.idx)
            if second not in residue_atom_indices:
                continue
            if graph.has_node(first) and graph.has_node(second):
                graph.add_edge(first, second)
    return graph


def _ligand_rotatable_bonds(
    structure: pmd.Structure,
    residue_idx: int,
    impacted_atom_indices: set[int],
    bond_graph: nx.Graph,
) -> list[dict[str, Any]]:
    residue = structure.residues[residue_idx]
    heavy_graph = _ligand_heavy_bond_graph(residue)
    if heavy_graph.number_of_edges() == 0:
        return []

    bonds: list[dict[str, Any]] = []
    all_heavy = set(heavy_graph.nodes)
    for first_idx, second_idx in nx.bridges(heavy_graph):
        split_graph = heavy_graph.copy()
        split_graph.remove_edge(first_idx, second_idx)
        for component in nx.connected_components(split_graph):
            moving_heavy = set(int(idx) for idx in component)
            if not moving_heavy & impacted_atom_indices:
                continue
            stationary_heavy = all_heavy - moving_heavy
            if len(moving_heavy) < 2 or len(stationary_heavy) < 2:
                continue

            if first_idx in moving_heavy:
                distal_idx = int(first_idx)
                proximal_idx = int(second_idx)
            else:
                distal_idx = int(second_idx)
                proximal_idx = int(first_idx)

            moving = _moving_component(bond_graph, proximal_idx, distal_idx)
            residue_atom_indices = {int(atom.idx) for atom in residue.atoms}
            moving = [idx for idx in moving if idx in residue_atom_indices]
            if len(moving) < 2:
                continue

            proximal_name = str(structure.atoms[proximal_idx].name).strip()
            distal_name = str(structure.atoms[distal_idx].name).strip()
            bonds.append(
                {
                    "label": "ligand_torsion",
                    "proximal_name": proximal_name,
                    "distal_name": distal_name,
                    "proximal_idx": proximal_idx,
                    "distal_idx": distal_idx,
                    "moving": moving,
                }
            )
    return bonds


def _rotate_about_axis(
    coordinates: np.ndarray,
    moving_indices: list[int],
    proximal_idx: int,
    distal_idx: int,
    angle_degrees: float,
) -> np.ndarray:
    updated = coordinates.copy()
    axis = coordinates[distal_idx] - coordinates[proximal_idx]
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-8:
        return updated
    axis = axis / norm

    theta = np.deg2rad(float(angle_degrees))
    rel = coordinates[moving_indices] - coordinates[proximal_idx]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cross = np.cross(axis, rel)
    dot = rel @ axis
    updated[moving_indices] = (
        coordinates[proximal_idx]
        + rel * cos_theta
        + cross * sin_theta
        + np.outer(dot, axis) * (1.0 - cos_theta)
    )
    return updated


def _heavy_atom_indices(structure: pmd.Structure) -> list[int]:
    return [
        int(atom.idx)
        for atom in structure.atoms
        if str(atom.residue.name).strip() not in _WATER_RESNAMES and not _is_hydrogen(atom)
    ]


def _close_contact_score(
    structure: pmd.Structure,
    coordinates: np.ndarray,
    moved_indices: set[int],
    *,
    cutoff: float = 2.0,
) -> tuple[float, float]:
    heavy = set(_heavy_atom_indices(structure))
    moving = np.asarray(sorted(heavy & moved_indices), dtype=int)
    environment = np.asarray(sorted(heavy - moved_indices), dtype=int)
    if moving.size == 0 or environment.size == 0:
        return 0.0, float("inf")

    distances = np.linalg.norm(
        coordinates[moving, None, :] - coordinates[environment][None, :, :],
        axis=-1,
    )
    penalty = np.clip(float(cutoff) - distances, 0.0, None)
    return float(np.sum(penalty * penalty)), float(np.min(distances))


def _single_rotation_plan_candidates(
    rotatable: list[dict[str, Any]]
) -> list[list[tuple[dict[str, Any], float]]]:
    plans: list[list[tuple[dict[str, Any], float]]] = []
    for bond in rotatable:
        for angle in _ROTATION_ANGLES:
            plans.append([(bond, float(angle))])
    return plans


def _combined_rotation_plan_candidates(
    rotatable: list[dict[str, Any]]
) -> list[list[tuple[dict[str, Any], float]]]:
    plans: list[list[tuple[dict[str, Any], float]]] = []
    if len(rotatable) >= 2:
        first = rotatable[0]
        second = rotatable[1]
        for first_angle in _ROTATION_ANGLES:
            for second_angle in _ROTATION_ANGLES:
                plans.append([(first, float(first_angle)), (second, float(second_angle))])
    return plans


def repair_ring_penetrations_with_sidechain_rotamers(
    structure: pmd.Structure,
    *,
    ligand_label: str = "",
    max_candidates: int = 400,
) -> RingPenetrationRepairResult:
    """Resolve ring penetrations by rotating involved protein aromatic sidechains.

    The repair is intentionally local: it only considers aromatic protein
    sidechains that are part of a detected ring penetration and rotates chi2,
    chi1, or both. The structure is modified in place only when a candidate
    removes all detected ring penetrations.
    """
    coordinates = np.asarray(structure.coordinates, dtype=float).copy()
    pairs, rings, topology = _ring_penetrations(structure, coordinates)
    result = RingPenetrationRepairResult(
        mode="protein_sidechain", initial_penetrations=len(pairs)
    )
    if not pairs:
        return result

    candidate_residue_indices = _candidate_aromatic_residue_indices(topology, pairs, rings)
    if not candidate_residue_indices:
        result.final_penetrations = len(pairs)
        logger.warning(
            "[ring_repair:{}] Ring penetration detected, but no protein aromatic "
            "sidechain candidate was found for local rotamer repair.",
            ligand_label or "unknown",
        )
        return result

    bond_graph = _full_bond_graph(structure)
    best: tuple[float, float, int, list[dict[str, Any]], np.ndarray] | None = None
    tested = 0

    def _evaluate_plans(
        *,
        residue_idx: int,
        plans: list[list[tuple[dict[str, Any], float]]],
    ) -> None:
        nonlocal best, tested
        for plan in plans:
            tested += 1
            if tested > max_candidates:
                break
            candidate = coordinates.copy()
            moved_indices: set[int] = set()
            rotations: list[dict[str, Any]] = []
            for bond, angle in plan:
                candidate = _rotate_about_axis(
                    candidate,
                    bond["moving"],
                    bond["proximal_idx"],
                    bond["distal_idx"],
                    angle,
                )
                moved_indices.update(int(idx) for idx in bond["moving"])
                rotations.append(
                    {
                        "label": bond["label"],
                        "axis": f"{bond['proximal_name']}-{bond['distal_name']}",
                        "angle_degrees": angle,
                    }
                )

            candidate_pairs, _ = _ring_penetrations_with_topology(topology, candidate)
            if candidate_pairs:
                continue

            score, min_distance = _close_contact_score(
                structure, candidate, moved_indices
            )
            if (
                best is None
                or score < best[0] - 1.0e-8
                or (abs(score - best[0]) <= 1.0e-8 and min_distance > best[1])
            ):
                best = (score, min_distance, residue_idx, rotations, candidate)

    for residue_idx in candidate_residue_indices:
        rotatable = _rotatable_bonds(structure, residue_idx, bond_graph)
        if not rotatable:
            continue
        _evaluate_plans(
            residue_idx=residue_idx,
            plans=_single_rotation_plan_candidates(rotatable),
        )
        if best is not None or tested > max_candidates:
            break
        _evaluate_plans(
            residue_idx=residue_idx,
            plans=_combined_rotation_plan_candidates(rotatable),
        )
        if tested > max_candidates:
            break

    if best is None:
        result.final_penetrations = len(pairs)
        logger.warning(
            "[ring_repair:{}] Could not remove {} ring penetration(s) with local "
            "aromatic sidechain rotations.",
            ligand_label or "unknown",
            len(pairs),
        )
        return result

    score, min_distance, residue_idx, rotations, repaired_coordinates = best
    structure.coordinates = repaired_coordinates
    final_pairs, _ = _ring_penetrations_with_topology(topology, repaired_coordinates)
    residue = structure.residues[residue_idx]
    result.repaired = len(final_pairs) == 0
    result.final_penetrations = len(final_pairs)
    result.residue = {
        "index": int(residue_idx),
        "resid": int(residue.number) + 1,
        "resname": str(residue.name),
    }
    result.rotations = rotations
    result.score = float(score)
    result.min_distance = float(min_distance)

    if result.repaired:
        logger.info(
            "[ring_repair:{}] Removed {} ring penetration(s) by rotating {}{} "
            "{}; close-contact score={:.3f}, min_distance={:.2f} Å.",
            ligand_label or "unknown",
            result.initial_penetrations,
            residue.name,
            int(residue.number) + 1,
            ", ".join(
                f"{item['label']} {item['angle_degrees']:+.0f}°"
                for item in rotations
            ),
            score,
            min_distance,
        )
    return result


def repair_ring_penetrations_with_ligand_torsions(
    structure: pmd.Structure,
    *,
    ligand_resname: str,
    ligand_label: str = "",
    max_candidates: int = 400,
) -> RingPenetrationRepairResult:
    """Resolve ring penetrations by rotating local ligand torsions.

    Only heavy-atom ligand bonds that are graph bridges are considered. The
    fragment containing the penetrated ligand atom(s) is rotated, so the repair
    remains local to a ligand torsion rather than translating or rotating the
    whole ligand pose.
    """
    coordinates = np.asarray(structure.coordinates, dtype=float).copy()
    pairs, rings, topology = _ring_penetrations(structure, coordinates)
    result = RingPenetrationRepairResult(
        mode="ligand", initial_penetrations=len(pairs)
    )
    if not pairs:
        return result

    candidate_residue_indices = _candidate_ligand_residue_indices(
        topology,
        pairs,
        rings,
        ligand_resname=ligand_resname,
    )
    if not candidate_residue_indices:
        result.final_penetrations = len(pairs)
        logger.warning(
            "[ring_repair:{}] Ring penetration detected, but no ligand residue "
            "{} was found in the penetration geometry.",
            ligand_label or "unknown",
            ligand_resname,
        )
        return result

    bond_graph = _full_bond_graph(structure)
    best: tuple[float, float, int, list[dict[str, Any]], np.ndarray] | None = None
    tested = 0

    for residue_idx in candidate_residue_indices:
        impacted = _impacted_ligand_atom_indices(
            topology, pairs, rings, residue_idx=residue_idx
        )
        rotatable = _ligand_rotatable_bonds(
            structure, residue_idx, impacted, bond_graph
        )
        for plan in _single_rotation_plan_candidates(rotatable):
            tested += 1
            if tested > max_candidates:
                break
            candidate = coordinates.copy()
            moved_indices: set[int] = set()
            rotations: list[dict[str, Any]] = []
            for bond, angle in plan:
                candidate = _rotate_about_axis(
                    candidate,
                    bond["moving"],
                    bond["proximal_idx"],
                    bond["distal_idx"],
                    angle,
                )
                moved_indices.update(int(idx) for idx in bond["moving"])
                rotations.append(
                    {
                        "label": bond["label"],
                        "axis": f"{bond['proximal_name']}-{bond['distal_name']}",
                        "angle_degrees": angle,
                    }
                )

            candidate_pairs, _ = _ring_penetrations_with_topology(topology, candidate)
            if candidate_pairs:
                continue

            score, min_distance = _close_contact_score(
                structure, candidate, moved_indices
            )
            if (
                best is None
                or score < best[0] - 1.0e-8
                or (abs(score - best[0]) <= 1.0e-8 and min_distance > best[1])
            ):
                best = (score, min_distance, residue_idx, rotations, candidate)
        if tested > max_candidates:
            break

    if best is None:
        result.final_penetrations = len(pairs)
        logger.warning(
            "[ring_repair:{}] Could not remove {} ring penetration(s) with local "
            "ligand torsion rotations.",
            ligand_label or "unknown",
            len(pairs),
        )
        return result

    score, min_distance, residue_idx, rotations, repaired_coordinates = best
    structure.coordinates = repaired_coordinates
    final_pairs, _ = _ring_penetrations_with_topology(topology, repaired_coordinates)
    residue = structure.residues[residue_idx]
    result.repaired = len(final_pairs) == 0
    result.final_penetrations = len(final_pairs)
    result.residue = {
        "index": int(residue_idx),
        "resid": int(residue.number) + 1,
        "resname": str(residue.name),
    }
    result.rotations = rotations
    result.score = float(score)
    result.min_distance = float(min_distance)

    if result.repaired:
        logger.info(
            "[ring_repair:{}] Removed {} ring penetration(s) by rotating ligand "
            "{}{} {}; close-contact score={:.3f}, min_distance={:.2f} Å.",
            ligand_label or "unknown",
            result.initial_penetrations,
            residue.name,
            int(residue.number) + 1,
            ", ".join(
                f"{item['axis']} {item['angle_degrees']:+.0f}°"
                for item in rotations
            ),
            score,
            min_distance,
        )
    return result


def repair_ring_penetrations(
    structure: pmd.Structure,
    *,
    fix_mode: str = "protein_sidechain",
    ligand_resname: str | None = None,
    ligand_label: str = "",
    max_candidates: int = 400,
) -> RingPenetrationRepairResult:
    mode = str(fix_mode).strip().lower().replace("-", "_")
    if mode in {"protein", "sidechain", "protein_side_chain"}:
        mode = "protein_sidechain"
    if mode in {"lig", "ligand"}:
        mode = "ligand"

    if mode == "protein_sidechain":
        return repair_ring_penetrations_with_sidechain_rotamers(
            structure,
            ligand_label=ligand_label,
            max_candidates=max_candidates,
        )
    if mode == "ligand":
        if not ligand_resname:
            raise ValueError("ligand_resname is required for ligand ring repair mode.")
        return repair_ring_penetrations_with_ligand_torsions(
            structure,
            ligand_resname=ligand_resname,
            ligand_label=ligand_label,
            max_candidates=max_candidates,
        )
    raise ValueError(
        "fix_mode must be 'protein_sidechain' or 'ligand' for ring penetration repair."
    )
