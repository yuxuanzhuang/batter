from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
import numpy as np


DEFAULT_LIPID_RESNAMES = ("PA", "PC", "OL")


@dataclass
class LipidHydrogenRepairResult:
    scanned_centers: int
    suspicious_contacts: int
    repaired_centers: int
    repaired_hydrogens: int
    examples: list[dict[str, object]]
    inpcrd_updated: bool = False
    pdb_updated: bool = False

    @property
    def repaired(self) -> bool:
        return self.repaired_hydrogens > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned_centers": self.scanned_centers,
            "suspicious_contacts": self.suspicious_contacts,
            "repaired_centers": self.repaired_centers,
            "repaired_hydrogens": self.repaired_hydrogens,
            "examples": self.examples,
            "inpcrd_updated": self.inpcrd_updated,
            "pdb_updated": self.pdb_updated,
        }


def _is_hydrogen(atom) -> bool:
    return atom.name.startswith("H") or str(atom.type).startswith("h")


def _is_lipid_carbon(atom, lipid_resnames: set[str]) -> bool:
    atom_type = str(atom.type)
    return (
        atom.resname in lipid_resnames
        and not _is_hydrogen(atom)
        and (atom.name.startswith("C") or atom_type.startswith("c") or atom_type.startswith("C"))
    )


def _is_heavy(atom) -> bool:
    atom_type = str(atom.type)
    return not _is_hydrogen(atom) and atom.name != "EPW" and atom_type != "EP"


def _read_restart(path: Path) -> tuple[str, str, int, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not an Amber restart")
    title = lines[0]
    atom_line = lines[1]
    natom = int(atom_line.split()[0])
    values = np.fromstring(" ".join(lines[2:]), sep=" ", dtype=float)
    if values.size < natom * 3:
        raise ValueError(f"{path} does not contain {natom} atom coordinates")
    coords = values[: natom * 3].reshape((natom, 3)).copy()
    tail = values[natom * 3 :].copy()
    return title, atom_line, natom, coords, tail


def _write_restart(
    path: Path,
    title: str,
    atom_line: str,
    coords: np.ndarray,
    tail: np.ndarray,
) -> None:
    values = np.concatenate([coords.reshape(-1), tail])
    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{atom_line}\n")
        for i in range(0, values.size, 6):
            handle.write("".join(f"{value:12.7f}" for value in values[i : i + 6]) + "\n")


def _minimum_image(delta: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return delta
    return delta - box * np.round(delta / box)


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = _norm(vector)
    if length < 1.0e-8:
        return None
    return vector / length


def _perpendicular_axis(axis: np.ndarray) -> np.ndarray:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, trial))) > 0.8:
        trial = np.array([0.0, 1.0, 0.0])
    perp = trial - axis * float(np.dot(axis, trial))
    unit = _unit(perp)
    if unit is None:
        return np.array([0.0, 0.0, 1.0])
    return unit


def _hydrogen_directions(heavy_dirs: list[np.ndarray], hydrogen_count: int) -> list[np.ndarray]:
    if hydrogen_count <= 0:
        return []
    if not heavy_dirs:
        return []

    if hydrogen_count == 1:
        direction = _unit(-np.sum(heavy_dirs, axis=0))
        if direction is None:
            direction = -heavy_dirs[0]
        return [direction]

    if hydrogen_count == 2:
        heavy_sum = _unit(np.sum(heavy_dirs, axis=0))
        if heavy_sum is None:
            heavy_sum = heavy_dirs[0]
        base = -heavy_sum
        if len(heavy_dirs) >= 2:
            axis = _unit(np.cross(heavy_dirs[0], heavy_dirs[1]))
        else:
            axis = None
        if axis is None:
            axis = _perpendicular_axis(base)
        base_weight = 1.0 / math.sqrt(3.0)
        axis_weight = math.sqrt(2.0 / 3.0)
        direction_1 = _unit(base_weight * base + axis_weight * axis)
        direction_2 = _unit(base_weight * base - axis_weight * axis)
        if direction_1 is None or direction_2 is None:
            return []
        return [direction_1, direction_2]

    axis = heavy_dirs[0]
    e1 = _perpendicular_axis(axis)
    e2 = np.cross(axis, e1)
    cone = math.sqrt(8.0 / 9.0)
    directions = []
    for i in range(hydrogen_count):
        angle = 2.0 * math.pi * i / hydrogen_count
        radial = math.cos(angle) * e1 + math.sin(angle) * e2
        direction = _unit((-axis / 3.0) + cone * radial)
        if direction is None:
            return []
        directions.append(direction)
    return directions


def _replace_pdb_coordinates(line: str, coord: np.ndarray) -> str:
    return f"{line[:30]}{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{line[54:]}"


def _update_pdb_coordinates(pdb_path: Path, coords: np.ndarray) -> bool:
    if not pdb_path.exists():
        return False
    lines = pdb_path.read_text().splitlines(keepends=True)
    atom_line_indices = [
        i for i, line in enumerate(lines) if line.startswith(("ATOM  ", "HETATM"))
    ]
    if len(atom_line_indices) != len(coords):
        return False
    for atom_index, line_index in enumerate(atom_line_indices):
        newline = "\n" if lines[line_index].endswith("\n") else ""
        line = lines[line_index].rstrip("\n")
        lines[line_index] = _replace_pdb_coordinates(line, coords[atom_index]) + newline
    pdb_path.write_text("".join(lines))
    return True


def _backup_once(path: Path, suffix: str) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(path.name + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def repair_lipid_hydrogen_geometry(
    prmtop: str | Path,
    inpcrd: str | Path,
    pdb: str | Path | None = None,
    *,
    lipid_resnames: Iterable[str] = DEFAULT_LIPID_RESNAMES,
    suspicious_h_heavy_cutoff: float = 0.85,
    parent_bond_range: tuple[float, float] = (0.85, 1.25),
    backup_suffix: str = ".pre_lipid_h_repair",
    write_report: str | Path | None = None,
) -> LipidHydrogenRepairResult:
    """Repair tleap-placed lipid hydrogens that collide with dihedral neighbors.

    Split CHARMM-style lipid pieces such as PA/PC/OL can leave tleap adding a
    hydrogen on one side of a bonded lipid carbon directly toward the adjacent
    heavy atom that defines the local dihedral.  The C-H bond length is normal,
    but the H...heavy distance is impossible.  This scans lipid carbon centers
    for that pattern and rebuilds all hydrogens on each flagged center from the
    bonded heavy-neighbor geometry.
    """

    prmtop_path = Path(prmtop)
    inpcrd_path = Path(inpcrd)
    pdb_path = Path(pdb) if pdb is not None else None
    lipid_resname_set = {str(name) for name in lipid_resnames}

    title, atom_line, natom, coords, tail = _read_restart(inpcrd_path)
    universe = mda.Universe(str(prmtop_path), str(inpcrd_path), format="RESTRT")
    if len(universe.atoms) != natom:
        raise ValueError(
            f"Topology/restart atom count mismatch: {len(universe.atoms)} != {natom}"
        )

    box = tail[:3].astype(float) if tail.size >= 3 else None
    adjacency: list[list[int]] = [[] for _ in range(natom)]
    for i, j in universe.bonds.indices:
        ii = int(i)
        jj = int(j)
        adjacency[ii].append(jj)
        adjacency[jj].append(ii)

    flagged_centers: dict[int, list[dict[str, object]]] = {}
    scanned_centers = 0
    suspicious_contacts = 0

    for center_index, atom in enumerate(universe.atoms):
        if not _is_lipid_carbon(atom, lipid_resname_set):
            continue

        hydrogens = [
            idx
            for idx in adjacency[center_index]
            if universe.atoms[idx].resname in lipid_resname_set
            and _is_hydrogen(universe.atoms[idx])
        ]
        heavy_neighbors = [
            idx
            for idx in adjacency[center_index]
            if universe.atoms[idx].resname in lipid_resname_set
            and _is_heavy(universe.atoms[idx])
        ]
        if not hydrogens or not heavy_neighbors:
            continue
        scanned_centers += 1

        for hydrogen_index in hydrogens:
            parent_delta = _minimum_image(
                coords[hydrogen_index] - coords[center_index],
                box,
            )
            parent_distance = _norm(parent_delta)
            if not (parent_bond_range[0] <= parent_distance <= parent_bond_range[1]):
                continue
            for heavy_index in heavy_neighbors:
                h_heavy_distance = _norm(
                    _minimum_image(coords[hydrogen_index] - coords[heavy_index], box)
                )
                if h_heavy_distance < suspicious_h_heavy_cutoff:
                    suspicious_contacts += 1
                    h_atom = universe.atoms[hydrogen_index]
                    heavy_atom = universe.atoms[heavy_index]
                    flagged_centers.setdefault(center_index, []).append(
                        {
                            "hydrogen_index": hydrogen_index + 1,
                            "hydrogen": h_atom.name,
                            "hydrogen_resname": h_atom.resname,
                            "hydrogen_resid": int(h_atom.resid),
                            "parent_index": center_index + 1,
                            "parent": atom.name,
                            "parent_resname": atom.resname,
                            "parent_resid": int(atom.resid),
                            "neighbor_index": heavy_index + 1,
                            "neighbor": heavy_atom.name,
                            "neighbor_resname": heavy_atom.resname,
                            "neighbor_resid": int(heavy_atom.resid),
                            "parent_distance": round(parent_distance, 4),
                            "neighbor_distance": round(h_heavy_distance, 4),
                        }
                    )
                    break

    repaired_hydrogens: set[int] = set()
    for center_index in flagged_centers:
        hydrogens = [
            idx
            for idx in adjacency[center_index]
            if universe.atoms[idx].resname in lipid_resname_set
            and _is_hydrogen(universe.atoms[idx])
        ]
        heavy_neighbors = [
            idx
            for idx in adjacency[center_index]
            if universe.atoms[idx].resname in lipid_resname_set
            and _is_heavy(universe.atoms[idx])
        ]
        heavy_dirs = []
        for heavy_index in heavy_neighbors:
            direction = _unit(
                _minimum_image(coords[heavy_index] - coords[center_index], box)
            )
            if direction is not None:
                heavy_dirs.append(direction)
        directions = _hydrogen_directions(heavy_dirs, len(hydrogens))
        if len(directions) != len(hydrogens):
            continue
        for hydrogen_index, direction in zip(sorted(hydrogens), directions):
            current = _minimum_image(coords[hydrogen_index] - coords[center_index], box)
            bond_length = _norm(current)
            if not (parent_bond_range[0] <= bond_length <= parent_bond_range[1]):
                bond_length = 1.09
            coords[hydrogen_index] = coords[center_index] + bond_length * direction
            repaired_hydrogens.add(hydrogen_index)

    result = LipidHydrogenRepairResult(
        scanned_centers=scanned_centers,
        suspicious_contacts=suspicious_contacts,
        repaired_centers=len(flagged_centers),
        repaired_hydrogens=len(repaired_hydrogens),
        examples=[
            example
            for examples in list(flagged_centers.values())[:10]
            for example in examples[:1]
        ],
    )

    if result.repaired:
        _backup_once(inpcrd_path, backup_suffix)
        _write_restart(inpcrd_path, title, atom_line, coords, tail)
        result.inpcrd_updated = True
        if pdb_path is not None:
            _backup_once(pdb_path, backup_suffix)
            result.pdb_updated = _update_pdb_coordinates(pdb_path, coords)

    if write_report is not None and result.scanned_centers:
        Path(write_report).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    return result
