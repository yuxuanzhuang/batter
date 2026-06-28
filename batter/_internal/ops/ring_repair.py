from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

from batter._internal.parmed_compat import import_parmed
from batter.analysis.sim_validation import check_ring_penetration, lsqp

pmd = import_parmed()


_WATER_RESNAMES = {"HOH", "TIP3", "WAT"}
_PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASH",
    "ASN",
    "ASP",
    "CYM",
    "CYS",
    "CYX",
    "GLH",
    "GLN",
    "GLU",
    "GLY",
    "HID",
    "HIE",
    "HIP",
    "HIS",
    "ILE",
    "LEU",
    "LYN",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
_PROTEIN_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
_AROMATIC_SIDECHAIN_BONDS = {
    "PHE": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "TYR": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HIS": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HID": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HIE": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "HIP": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
    "TRP": (("chi2", "CB", "CG"), ("chi1", "CA", "CB")),
}
_ROTATION_ANGLES = (
    15,
    -15,
    30,
    -30,
    45,
    -45,
    60,
    -60,
    90,
    -90,
    120,
    -120,
    150,
    -150,
    180,
)
_LIGAND_PERTURBATION_MAGNITUDES = (
    0.10,
    0.15,
    0.20,
    0.30,
    0.45,
    0.60,
    0.80,
    1.00,
    1.25,
    1.50,
)


@dataclass
class RingPenetrationRepairResult:
    mode: str | None = None
    selected_mode: str | None = None
    initial_penetrations: int = 0
    final_penetrations: int = 0
    repaired: bool = False
    residue: dict[str, Any] | None = None
    rotations: list[dict[str, Any]] = field(default_factory=list)
    perturbations: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None
    min_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_mode": self.selected_mode,
            "initial_penetrations": self.initial_penetrations,
            "final_penetrations": self.final_penetrations,
            "repaired": self.repaired,
            "residue": self.residue,
            "rotations": self.rotations,
            "perturbations": self.perturbations,
            "steps": self.steps,
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


def _exact_pair_still_penetrates_ring(
    coordinates: np.ndarray,
    pair: tuple[int, int],
    ring: list[int],
) -> bool:
    ring_atoms = np.asarray([coordinates[node - 1] for node in ring], dtype=float)
    if len(ring_atoms) < 3:
        return False

    try:
        axis, com, _ = lsqp(ring_atoms)
    except Exception:
        return False
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-8:
        return False
    axis = axis / norm

    projected = ring_atoms.copy()
    for index, atom in enumerate(projected):
        foot = np.dot(axis, atom - com) * axis + com
        projected[index] = com + (atom - foot)

    max_distance = float(np.max(np.linalg.norm(projected - com, axis=1)))
    first = np.asarray(coordinates[pair[0] - 1], dtype=float)
    second = np.asarray(coordinates[pair[1] - 1], dtype=float)
    first_side = float(np.dot(first - com, axis))
    second_side = float(np.dot(second - com, axis))
    if first_side * second_side > 0:
        return False

    denominator = float(np.dot(axis, second - first))
    if abs(denominator) <= 1.0e-10:
        return False
    scale = -float(np.dot(axis, first - com)) / denominator
    intersection = first + scale * (second - first)
    if float(np.linalg.norm(intersection - com)) > max_distance:
        return False

    winding = 0.0
    for index in range(len(projected)):
        p1 = projected[index] - intersection
        p2 = projected[(index + 1) % len(projected)] - intersection
        denominator = float(np.linalg.norm(p1) * np.linalg.norm(p2))
        if denominator <= 1.0e-10:
            return True
        cosine = float(np.dot(p1, p2) / denominator)
        winding += float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    winding_number = winding / (2.0 * np.pi)
    return 0.9 < winding_number < 1.1


def _active_original_penetrations(
    coordinates: np.ndarray,
    pairs: list[tuple[int, int]],
    rings: list[list[int]],
) -> int:
    return sum(
        1
        for pair, ring in zip(pairs, rings)
        if _exact_pair_still_penetrates_ring(coordinates, pair, ring)
    )


def _is_protein_resname(resname: str) -> bool:
    return str(resname).strip().upper() in _PROTEIN_RESNAMES


def _candidate_protein_residue_indices(
    topology: nx.Graph,
    pairs: list[tuple[int, int]],
    rings: list[list[int]],
) -> list[int]:
    residue_indices: list[int] = []

    def _add(node: int) -> None:
        attrs = topology.nodes[node]
        if not _is_protein_resname(str(attrs["resname"])):
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


def _impacted_residue_atom_indices(
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


def _sidechain_axis_label(proximal_name: str, distal_name: str) -> str:
    names = {str(proximal_name).strip(), str(distal_name).strip()}
    if names == {"CA", "CB"}:
        return "chi1"
    if "CB" in names:
        return "chi2"
    return "sidechain_torsion"


def _protein_residue_heavy_graph(residue: pmd.Residue) -> nx.Graph:
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


def _protein_sidechain_rotatable_bonds(
    structure: pmd.Structure,
    residue_idx: int,
    impacted_atom_indices: set[int],
    bond_graph: nx.Graph,
) -> list[dict[str, Any]]:
    residue = structure.residues[residue_idx]
    templates = _AROMATIC_SIDECHAIN_BONDS.get(str(residue.name).strip(), ())
    bonds: list[dict[str, Any]] = []
    seen_axes: set[tuple[int, int]] = set()

    def _append_bond(
        *,
        label: str,
        proximal_name: str,
        distal_name: str,
        proximal_idx: int,
        distal_idx: int,
    ) -> None:
        axis_key = tuple(sorted((int(proximal_idx), int(distal_idx))))
        if axis_key in seen_axes:
            return
        moving = _moving_component(bond_graph, proximal_idx, distal_idx)
        if not moving:
            return
        if impacted_atom_indices and not (set(moving) & impacted_atom_indices):
            return
        if any(
            str(structure.atoms[idx].name).strip() in _PROTEIN_BACKBONE_ATOMS
            and int(structure.atoms[idx].residue.idx) == int(residue_idx)
            for idx in moving
        ):
            return
        seen_axes.add(axis_key)
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

    for label, proximal_name, distal_name in templates:
        proximal_idx = _atom_index_by_name(residue, proximal_name)
        distal_idx = _atom_index_by_name(residue, distal_name)
        if proximal_idx is None or distal_idx is None:
            continue
        _append_bond(
            label=label,
            proximal_name=proximal_name,
            distal_name=distal_name,
            proximal_idx=proximal_idx,
            distal_idx=distal_idx,
        )

    heavy_graph = _protein_residue_heavy_graph(residue)
    residue_atom_indices = {int(atom.idx) for atom in residue.atoms}
    backbone_indices = {
        int(atom.idx)
        for atom in residue.atoms
        if str(atom.name).strip() in _PROTEIN_BACKBONE_ATOMS
    }
    for first_idx, second_idx in nx.bridges(heavy_graph):
        split_graph = heavy_graph.copy()
        split_graph.remove_edge(first_idx, second_idx)
        components = [set(int(idx) for idx in comp) for comp in nx.connected_components(split_graph)]
        if len(components) != 2:
            continue

        first_component, second_component = components
        if first_component & backbone_indices and not (second_component & backbone_indices):
            stationary_heavy = first_component
            moving_heavy = second_component
        elif second_component & backbone_indices and not (first_component & backbone_indices):
            stationary_heavy = second_component
            moving_heavy = first_component
        else:
            continue

        if first_idx in stationary_heavy and second_idx in moving_heavy:
            proximal_idx = int(first_idx)
            distal_idx = int(second_idx)
        elif second_idx in stationary_heavy and first_idx in moving_heavy:
            proximal_idx = int(second_idx)
            distal_idx = int(first_idx)
        else:
            continue

        if len(moving_heavy) < 2:
            continue
        if impacted_atom_indices and not (moving_heavy & impacted_atom_indices):
            continue

        moving = _moving_component(bond_graph, proximal_idx, distal_idx)
        moving = [idx for idx in moving if idx in residue_atom_indices]
        if len(moving) < 2:
            continue

        proximal_name = str(structure.atoms[proximal_idx].name).strip()
        distal_name = str(structure.atoms[distal_idx].name).strip()
        _append_bond(
            label=_sidechain_axis_label(proximal_name, distal_name),
            proximal_name=proximal_name,
            distal_name=distal_name,
            proximal_idx=proximal_idx,
            distal_idx=distal_idx,
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


def _ligand_displacement_groups_by_anchor(
    structure: pmd.Structure, residue_idx: int
) -> dict[int, list[int]]:
    residue = structure.residues[residue_idx]
    residue_atom_indices = {int(atom.idx) for atom in residue.atoms}
    groups: dict[int, list[int]] = {}
    for atom in residue.atoms:
        atom_idx = int(atom.idx)
        if _is_hydrogen(atom):
            continue
        moving = {atom_idx}
        for bond in atom.bonds:
            other = bond.atom1 if bond.atom2 is atom else bond.atom2
            other_idx = int(other.idx)
            if other_idx in residue_atom_indices and _is_hydrogen(other):
                moving.add(other_idx)
        groups[atom_idx] = sorted(moving)
    return groups


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


def _ring_plane_geometry(
    coordinates: np.ndarray, ring: list[int]
) -> tuple[np.ndarray, np.ndarray, float] | None:
    ring_atoms = np.asarray([coordinates[node - 1] for node in ring], dtype=float)
    if len(ring_atoms) < 3:
        return None
    try:
        axis, com, _ = lsqp(ring_atoms)
    except Exception:
        return None
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-8:
        return None
    axis = axis / norm

    projected = ring_atoms.copy()
    for index, atom in enumerate(projected):
        foot = np.dot(axis, atom - com) * axis + com
        projected[index] = com + (atom - foot)
    max_distance = float(np.max(np.linalg.norm(projected - com, axis=1)))
    return axis, np.asarray(com, dtype=float), max_distance


def _pair_ring_intersection(
    coordinates: np.ndarray, pair: tuple[int, int], ring: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    geometry = _ring_plane_geometry(coordinates, ring)
    if geometry is None:
        return None
    axis, com, _ = geometry
    first = np.asarray(coordinates[pair[0] - 1], dtype=float)
    second = np.asarray(coordinates[pair[1] - 1], dtype=float)
    denominator = float(np.dot(axis, second - first))
    if abs(denominator) <= 1.0e-10:
        return None
    scale = -float(np.dot(axis, first - com)) / denominator
    intersection = first + scale * (second - first)
    return axis, com, intersection


def _normalised(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        return None
    return np.asarray(vector, dtype=float) / norm


def _atom_label(atom: pmd.Atom) -> str:
    return (
        f"{str(atom.residue.name).strip()}{int(atom.residue.number) + 1}:"
        f"{str(atom.name).strip()}"
    )


def _add_ligand_perturbation_option(
    options: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    *,
    label: str,
    vectors_by_anchor: dict[int, np.ndarray],
    groups_by_anchor: dict[int, list[int]],
) -> None:
    moves: list[dict[str, Any]] = []
    signature_parts: list[tuple[int, float, float, float]] = []
    for anchor_idx, vector in sorted(vectors_by_anchor.items()):
        if anchor_idx not in groups_by_anchor:
            continue
        vector = np.asarray(vector, dtype=float)
        if float(np.linalg.norm(vector)) <= 1.0e-8:
            continue
        moves.append(
            {
                "anchor": int(anchor_idx),
                "indices": groups_by_anchor[anchor_idx],
                "vector": vector,
            }
        )
        signature_parts.append(
            (
                int(anchor_idx),
                round(float(vector[0]), 4),
                round(float(vector[1]), 4),
                round(float(vector[2]), 4),
            )
        )

    if not moves:
        return
    signature = (label, *signature_parts)
    if signature in seen:
        return
    seen.add(signature)
    options.append({"label": label, "moves": moves})


def _ligand_local_perturbation_options(
    structure: pmd.Structure,
    coordinates: np.ndarray,
    topology: nx.Graph,
    pairs: list[tuple[int, int]],
    rings: list[list[int]],
    *,
    residue_idx: int,
) -> list[dict[str, Any]]:
    groups_by_anchor = _ligand_displacement_groups_by_anchor(structure, residue_idx)
    if not groups_by_anchor:
        return []

    options: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def _is_target_ligand_node(node: int) -> bool:
        return int(topology.nodes[node]["residue_idx"]) == int(residue_idx)

    for pair, ring in zip(pairs, rings):
        geometry = _pair_ring_intersection(coordinates, pair, ring)
        if geometry is None:
            continue
        axis, _, intersection = geometry

        ligand_pair_atoms = [
            int(node) - 1 for node in pair if _is_target_ligand_node(node)
        ]
        if ligand_pair_atoms:
            pair_subsets = [[idx] for idx in ligand_pair_atoms]
            if len(ligand_pair_atoms) > 1:
                pair_subsets.append(ligand_pair_atoms)
            for subset in pair_subsets:
                for magnitude in _LIGAND_PERTURBATION_MAGNITUDES:
                    for sign in (1.0, -1.0):
                        vector = axis * sign * float(magnitude)
                        _add_ligand_perturbation_option(
                            options,
                            seen,
                            label="ligand_bond_atom_shift",
                            vectors_by_anchor={idx: vector for idx in subset},
                            groups_by_anchor=groups_by_anchor,
                        )

        ligand_ring_atoms = [
            int(node) - 1 for node in ring if _is_target_ligand_node(node)
        ]
        if not ligand_ring_atoms:
            continue

        ranked_ring_atoms = sorted(
            ligand_ring_atoms,
            key=lambda idx: float(np.linalg.norm(coordinates[idx] - intersection)),
        )
        subsets: list[list[int]] = []
        for size in (1, 2, 3):
            if len(ranked_ring_atoms) >= size:
                subsets.append(ranked_ring_atoms[:size])

        for subset in subsets:
            for magnitude in _LIGAND_PERTURBATION_MAGNITUDES:
                magnitude = float(magnitude)
                for sign in (1.0, -1.0):
                    _add_ligand_perturbation_option(
                        options,
                        seen,
                        label="ligand_ring_atom_lift",
                        vectors_by_anchor={
                            idx: axis * sign * magnitude for idx in subset
                        },
                        groups_by_anchor=groups_by_anchor,
                    )

                radial_vectors: dict[int, np.ndarray] = {}
                for idx in subset:
                    radial = coordinates[idx] - intersection
                    radial = radial - np.dot(radial, axis) * axis
                    radial_unit = _normalised(radial)
                    if radial_unit is not None:
                        radial_vectors[idx] = radial_unit * magnitude
                if radial_vectors:
                    _add_ligand_perturbation_option(
                        options,
                        seen,
                        label="ligand_ring_atom_radial_out",
                        vectors_by_anchor=radial_vectors,
                        groups_by_anchor=groups_by_anchor,
                    )
                    _add_ligand_perturbation_option(
                        options,
                        seen,
                        label="ligand_ring_atom_radial_in",
                        vectors_by_anchor={
                            idx: -vector for idx, vector in radial_vectors.items()
                        },
                        groups_by_anchor=groups_by_anchor,
                    )

                for sign in (1.0, -1.0):
                    mixed_vectors: dict[int, np.ndarray] = {}
                    for idx in subset:
                        radial = coordinates[idx] - intersection
                        radial = radial - np.dot(radial, axis) * axis
                        radial_unit = _normalised(radial)
                        if radial_unit is None:
                            mixed_vectors[idx] = axis * sign * magnitude
                        else:
                            mixed_unit = _normalised(axis * sign + radial_unit * 0.35)
                            if mixed_unit is None:
                                mixed_unit = axis * sign
                            mixed_vectors[idx] = mixed_unit * magnitude
                    _add_ligand_perturbation_option(
                        options,
                        seen,
                        label="ligand_ring_atom_mixed_shift",
                        vectors_by_anchor=mixed_vectors,
                        groups_by_anchor=groups_by_anchor,
                    )

        if len(ligand_ring_atoms) >= 3:
            for magnitude in _LIGAND_PERTURBATION_MAGNITUDES:
                for sign in (1.0, -1.0):
                    _add_ligand_perturbation_option(
                        options,
                        seen,
                        label="ligand_ring_pucker",
                        vectors_by_anchor={
                            idx: axis
                            * sign
                            * (1.0 if order % 2 == 0 else -1.0)
                            * float(magnitude)
                            for order, idx in enumerate(ligand_ring_atoms)
                        },
                        groups_by_anchor=groups_by_anchor,
                    )

    return options


def _ligand_perturbation_plan_candidates(
    options: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    plans: list[list[dict[str, Any]]] = [[option] for option in options]
    pair_window = min(24, len(options))
    for first_index in range(pair_window):
        first_anchors = {
            int(move["anchor"]) for move in options[first_index]["moves"]
        }
        for second_index in range(first_index + 1, pair_window):
            second_anchors = {
                int(move["anchor"]) for move in options[second_index]["moves"]
            }
            if first_anchors & second_anchors:
                continue
            plans.append([options[first_index], options[second_index]])
    return plans


def _apply_ligand_perturbation_plan(
    structure: pmd.Structure,
    coordinates: np.ndarray,
    plan: list[dict[str, Any]],
) -> tuple[np.ndarray, set[int], list[dict[str, Any]]]:
    vectors_by_atom: dict[int, np.ndarray] = {}
    vectors_by_anchor: dict[int, np.ndarray] = {}
    labels: list[str] = []
    for option in plan:
        label = str(option["label"])
        if label not in labels:
            labels.append(label)
        for move in option["moves"]:
            vector = np.asarray(move["vector"], dtype=float)
            anchor = int(move["anchor"])
            vectors_by_anchor[anchor] = vectors_by_anchor.get(
                anchor, np.zeros(3)
            ) + vector
            for idx in move["indices"]:
                idx = int(idx)
                vectors_by_atom[idx] = vectors_by_atom.get(idx, np.zeros(3)) + vector

    updated = coordinates.copy()
    for idx, vector in vectors_by_atom.items():
        updated[idx] = updated[idx] + vector

    if not vectors_by_atom:
        return updated, set(), []

    max_displacement = max(
        float(np.linalg.norm(vector)) for vector in vectors_by_atom.values()
    )
    perturbation = {
        "label": "ligand_local_perturbation",
        "components": labels,
        "anchors": [
            _atom_label(structure.atoms[idx]) for idx in sorted(vectors_by_anchor)
        ],
        "displacements": [
            {
                "atom": _atom_label(structure.atoms[idx]),
                "dx": float(vector[0]),
                "dy": float(vector[1]),
                "dz": float(vector[2]),
            }
            for idx, vector in sorted(vectors_by_anchor.items())
        ],
        "max_displacement": float(max_displacement),
    }
    return updated, set(vectors_by_atom), [perturbation]


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
    warn_on_failure: bool = True,
    accept_partial: bool = False,
) -> RingPenetrationRepairResult:
    """Resolve ring penetrations by rotating involved protein sidechains.

    The repair is intentionally local: it only considers protein sidechains
    that are part of a detected ring penetration and rotates sidechain bridge
    bonds while keeping the backbone side fixed. The structure is modified in
    place only when a candidate removes all detected ring penetrations. When
    ``accept_partial`` is enabled, the best local move that reduces the current
    full-system penetration count can be applied as one step in an iterative
    repair.
    """
    coordinates = np.asarray(structure.coordinates, dtype=float).copy()
    pairs, rings, topology = _ring_penetrations(structure, coordinates)
    result = RingPenetrationRepairResult(
        mode="protein_sidechain", initial_penetrations=len(pairs)
    )
    if not pairs:
        return result

    candidate_residue_indices = _candidate_protein_residue_indices(topology, pairs, rings)
    if not candidate_residue_indices:
        result.final_penetrations = len(pairs)
        if warn_on_failure:
            logger.warning(
                "[ring_repair:{}] Ring penetration detected, but no protein "
                "sidechain candidate was found for local rotamer repair.",
                ligand_label or "unknown",
            )
        return result

    bond_graph = _full_bond_graph(structure)
    best: tuple[int, float, float, int, list[dict[str, Any]], np.ndarray] | None = None
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

            remaining_original = _active_original_penetrations(
                candidate, pairs, rings
            )
            if remaining_original and (
                not accept_partial or remaining_original >= len(pairs)
            ):
                continue
            if remaining_original == 0 and not accept_partial:
                candidate_pairs, _ = _ring_penetrations_with_topology(
                    topology, candidate
                )
                if candidate_pairs:
                    continue

            score, min_distance = _close_contact_score(
                structure, candidate, moved_indices
            )
            if (
                best is None
                or remaining_original < best[0]
                or (
                    remaining_original == best[0]
                    and (
                        score < best[1] - 1.0e-8
                        or (
                            abs(score - best[1]) <= 1.0e-8
                            and min_distance > best[2]
                        )
                    )
                )
            ):
                best = (
                    remaining_original,
                    score,
                    min_distance,
                    residue_idx,
                    rotations,
                    candidate,
                )

    for residue_idx in candidate_residue_indices:
        impacted = _impacted_residue_atom_indices(
            topology, pairs, rings, residue_idx=residue_idx
        )
        rotatable = _protein_sidechain_rotatable_bonds(
            structure, residue_idx, impacted, bond_graph
        )
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
        if warn_on_failure:
            logger.warning(
                "[ring_repair:{}] Could not remove {} ring penetration(s) with "
                "local protein sidechain rotations.",
                ligand_label or "unknown",
                len(pairs),
            )
        return result

    _, score, min_distance, residue_idx, rotations, repaired_coordinates = best
    final_pairs, _ = _ring_penetrations_with_topology(topology, repaired_coordinates)
    if accept_partial and final_pairs and len(final_pairs) >= len(pairs):
        result.final_penetrations = len(pairs)
        return result

    structure.coordinates = repaired_coordinates
    residue = structure.residues[residue_idx]
    result.repaired = len(final_pairs) == 0
    result.final_penetrations = len(final_pairs)
    result.selected_mode = (
        "protein_sidechain" if result.repaired or accept_partial else None
    )
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


def repair_ring_penetrations_with_ligand_perturbations(
    structure: pmd.Structure,
    *,
    ligand_resname: str,
    ligand_label: str = "",
    max_candidates: int = 400,
    warn_on_failure: bool = True,
    accept_partial: bool = False,
) -> RingPenetrationRepairResult:
    """Resolve ring penetrations by locally perturbing involved ligand atoms.

    This mode never rotates the ligand as a rigid body or around a ligand
    torsion. It displaces only the ligand atoms directly involved in the
    penetration geometry, plus any hydrogens bonded to those moved heavy atoms.
    When ``accept_partial`` is enabled, one reducing local move can be applied
    as part of an iterative repair.
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
        if warn_on_failure:
            logger.warning(
                "[ring_repair:{}] Ring penetration detected, but no ligand residue "
                "{} was found in the penetration geometry.",
                ligand_label or "unknown",
                ligand_resname,
            )
        return result

    best: tuple[
        int, float, float, int, list[dict[str, Any]], np.ndarray
    ] | None = None
    tested = 0

    for residue_idx in candidate_residue_indices:
        options = _ligand_local_perturbation_options(
            structure,
            coordinates,
            topology,
            pairs,
            rings,
            residue_idx=residue_idx,
        )
        for plan in _ligand_perturbation_plan_candidates(options):
            tested += 1
            if tested > max_candidates:
                break
            candidate, moved_indices, perturbations = _apply_ligand_perturbation_plan(
                structure,
                coordinates,
                plan,
            )
            if not moved_indices:
                continue

            remaining_original = _active_original_penetrations(
                candidate, pairs, rings
            )
            if remaining_original and (
                not accept_partial or remaining_original >= len(pairs)
            ):
                continue

            candidate_pairs, _ = _ring_penetrations_with_topology(
                topology, candidate
            )
            candidate_count = len(candidate_pairs)
            if accept_partial:
                if candidate_count >= len(pairs):
                    continue
            elif candidate_count != 0:
                continue

            score, min_distance = _close_contact_score(
                structure, candidate, moved_indices
            )
            if (
                best is None
                or candidate_count < best[0]
                or (
                    candidate_count == best[0]
                    and (
                        score < best[1] - 1.0e-8
                        or (
                            abs(score - best[1]) <= 1.0e-8
                            and min_distance > best[2]
                        )
                    )
                )
            ):
                best = (
                    candidate_count,
                    score,
                    min_distance,
                    residue_idx,
                    perturbations,
                    candidate,
                )
        if tested > max_candidates:
            break

    if best is None:
        result.final_penetrations = len(pairs)
        if warn_on_failure:
            logger.warning(
                "[ring_repair:{}] Could not remove {} ring penetration(s) with "
                "local ligand coordinate perturbations.",
                ligand_label or "unknown",
                len(pairs),
            )
        return result

    _, score, min_distance, residue_idx, perturbations, repaired_coordinates = best
    final_pairs, _ = _ring_penetrations_with_topology(topology, repaired_coordinates)
    if accept_partial and final_pairs and len(final_pairs) >= len(pairs):
        result.final_penetrations = len(pairs)
        return result

    structure.coordinates = repaired_coordinates
    residue = structure.residues[residue_idx]
    result.repaired = len(final_pairs) == 0
    result.final_penetrations = len(final_pairs)
    result.selected_mode = "ligand" if result.repaired or accept_partial else None
    result.residue = {
        "index": int(residue_idx),
        "resid": int(residue.number) + 1,
        "resname": str(residue.name),
    }
    result.perturbations = perturbations
    result.score = float(score)
    result.min_distance = float(min_distance)

    if result.repaired:
        logger.info(
            "[ring_repair:{}] Removed {} ring penetration(s) by locally "
            "perturbing ligand {}{}; close-contact score={:.3f}, "
            "min_distance={:.2f} Å.",
            ligand_label or "unknown",
            result.initial_penetrations,
            residue.name,
            int(residue.number) + 1,
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
    warn_on_failure: bool = True,
    accept_partial: bool = False,
) -> RingPenetrationRepairResult:
    """Compatibility wrapper for the former ligand repair entry point."""
    return repair_ring_penetrations_with_ligand_perturbations(
        structure,
        ligand_resname=ligand_resname,
        ligand_label=ligand_label,
        max_candidates=max_candidates,
        warn_on_failure=warn_on_failure,
        accept_partial=accept_partial,
    )


def _repair_step_to_dict(result: RingPenetrationRepairResult) -> dict[str, Any]:
    return {
        "mode": result.selected_mode or result.mode,
        "residue": result.residue,
        "rotations": result.rotations,
        "perturbations": result.perturbations,
        "final_penetrations": result.final_penetrations,
        "score": result.score,
        "min_distance": result.min_distance,
    }


def _iterative_ring_penetration_repair(
    structure: pmd.Structure,
    *,
    mode: str,
    strategies: tuple[str, ...],
    ligand_resname: str | None,
    ligand_label: str,
    max_candidates: int,
    max_rounds: int = 8,
) -> RingPenetrationRepairResult:
    original_coordinates = np.asarray(structure.coordinates, dtype=float).copy()
    initial_pairs, _, _ = _ring_penetrations(structure, original_coordinates)
    result = RingPenetrationRepairResult(
        mode=mode,
        initial_penetrations=len(initial_pairs),
        final_penetrations=len(initial_pairs),
    )
    if not initial_pairs:
        return result

    steps: list[dict[str, Any]] = []
    rotation_steps: list[dict[str, Any]] = []
    perturbation_steps: list[dict[str, Any]] = []
    residues: list[dict[str, Any]] = []
    selected_modes: list[str] = []

    for _ in range(max_rounds):
        before_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
        before_count = len(before_pairs)
        if before_count == 0:
            break

        progress = False
        for strategy in strategies:
            if strategy == "protein_sidechain":
                step_result = repair_ring_penetrations_with_sidechain_rotamers(
                    structure,
                    ligand_label=ligand_label,
                    max_candidates=max_candidates,
                    warn_on_failure=False,
                    accept_partial=True,
                )
            elif strategy == "ligand":
                if not ligand_resname:
                    continue
                step_result = repair_ring_penetrations_with_ligand_perturbations(
                    structure,
                    ligand_resname=ligand_resname,
                    ligand_label=ligand_label,
                    max_candidates=max_candidates,
                    warn_on_failure=False,
                    accept_partial=True,
                )
            else:
                continue

            current_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
            current_count = len(current_pairs)
            if step_result.repaired or (
                (step_result.rotations or step_result.perturbations)
                and current_count < before_count
            ):
                progress = True
                selected_mode = step_result.selected_mode or strategy
                selected_modes.append(selected_mode)
                steps.append(_repair_step_to_dict(step_result))
                rotation_steps.extend(step_result.rotations)
                perturbation_steps.extend(step_result.perturbations)
                if step_result.residue is not None:
                    residues.append(step_result.residue)
                before_count = current_count
                if current_count == 0:
                    break

        if not progress:
            break

    final_pairs, _, _ = _ring_penetrations(structure, structure.coordinates)
    result.final_penetrations = len(final_pairs)
    result.repaired = len(final_pairs) == 0
    result.steps = steps
    result.rotations = rotation_steps
    result.perturbations = perturbation_steps
    result.residue = residues[-1] if residues else None
    if steps:
        result.score = steps[-1]["score"]
        result.min_distance = steps[-1]["min_distance"]
    if result.repaired:
        unique_modes = list(dict.fromkeys(selected_modes))
        if mode == "auto" and len(unique_modes) > 1:
            result.selected_mode = "auto"
        elif unique_modes:
            result.selected_mode = unique_modes[-1]
        logger.info(
            "[ring_repair:{}] Removed {} ring penetration(s) with {} local "
            "repair step(s): {}.",
            ligand_label or "unknown",
            result.initial_penetrations,
            len(steps),
            ", ".join(
                f"{step['mode']} {step['residue']['resname']}{step['residue']['resid']}"
                if step.get("residue")
                else str(step.get("mode"))
                for step in steps
            ),
        )
        return result

    structure.coordinates = original_coordinates
    logger.warning(
        "[ring_repair:{}] Could not remove {} ring penetration(s) with {} repair; "
        "original coordinates were kept.",
        ligand_label or "unknown",
        result.initial_penetrations,
        mode,
    )
    result.final_penetrations = result.initial_penetrations
    return result


def repair_ring_penetrations(
    structure: pmd.Structure,
    *,
    fix_mode: str = "auto",
    ligand_resname: str | None = None,
    ligand_label: str = "",
    max_candidates: int = 400,
) -> RingPenetrationRepairResult:
    mode = str(fix_mode).strip().lower().replace("-", "_")
    if mode in {"protein", "sidechain", "protein_side_chain"}:
        mode = "protein_sidechain"
    if mode in {"lig", "ligand"}:
        mode = "ligand"

    if mode == "auto":
        strategies = ("protein_sidechain",)
        if ligand_resname:
            strategies = ("protein_sidechain", "ligand")
        return _iterative_ring_penetration_repair(
            structure,
            mode="auto",
            strategies=strategies,
            ligand_resname=ligand_resname,
            ligand_label=ligand_label,
            max_candidates=max_candidates,
        )

    if mode == "protein_sidechain":
        return _iterative_ring_penetration_repair(
            structure,
            mode="protein_sidechain",
            strategies=("protein_sidechain",),
            ligand_resname=ligand_resname,
            ligand_label=ligand_label,
            max_candidates=max_candidates,
        )
    if mode == "ligand":
        if not ligand_resname:
            raise ValueError("ligand_resname is required for ligand ring repair mode.")
        return _iterative_ring_penetration_repair(
            structure,
            mode="ligand",
            strategies=("ligand",),
            ligand_resname=ligand_resname,
            ligand_label=ligand_label,
            max_candidates=max_candidates,
        )
    raise ValueError(
        "fix_mode must be 'auto', 'protein_sidechain', or 'ligand' for "
        "ring penetration repair."
    )
