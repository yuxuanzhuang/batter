from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger

from batter.utils import run_with_log, tleap
from batter.utils.builder_utils import get_buffer_z
from batter.utils.lipid_hydrogen_repair import repair_lipid_hydrogen_geometry
from batter._internal.parmed_compat import import_parmed
from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_create_box
from batter._internal.ops.helpers import (
    Anchors,
    PROTEIN_COM_ATOM_SELECTION,
    load_anchors,
    run_parmed_hmr_if_enabled,
    merge_first_n_and_lipid_fragments_in_prmtop,
    revised_resids_for_lipid_fragments,
    save_anchors,
)
from batter._internal.ops.ring_repair import (
    repair_ring_penetrations,
)

pmd = import_parmed()


def _pdb_residue_records(path: Path) -> list[tuple[str, str, int]]:
    records: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "_"
        resid_text = line[22:26].strip()
        try:
            resid = int(resid_text)
        except ValueError:
            fields = line.split()
            if len(fields) < 5:
                continue
            resname = fields[3]
            if len(fields) >= 6 and re.fullmatch(r"-?\d+", fields[5]):
                chain = fields[4]
                resid_text = fields[5]
            else:
                chain = "_"
                resid_text = fields[4]
            try:
                resid = int(resid_text)
            except ValueError:
                continue
        key = (resname, chain, resid)
        if key in seen:
            continue
        seen.add(key)
        records.append(key)
    return records


def _write_identity_amber_renum(input_pdb: Path, renum_path: Path) -> None:
    lines = []
    for resname, chain, resid in _pdb_residue_records(input_pdb):
        lines.append(f"{resname} {chain} {resid:5d} {resname} {resid:5d}\n")
    renum_path.write_text("".join(lines))


def _run_pdb4amber_for_box_or_copy(
    input_pdb: Path, output_pdb: Path, *, working_dir: Path
) -> None:
    if shutil.which("pdb4amber"):
        run_with_log(
            f"pdb4amber -i {input_pdb.name} -o {output_pdb.name} -y",
            working_dir=working_dir,
        )
        return

    shutil.copy2(input_pdb, output_pdb)
    _write_identity_amber_renum(input_pdb, working_dir / "build_amber_renum.txt")
    (working_dir / "build_amber_sslink").write_text("")
    logger.warning(
        "pdb4amber was not found; copying {} to {} and writing identity "
        "build_amber_renum.txt without additional cleanup.",
        input_pdb,
        output_pdb,
    )


_PRE_RING_REPAIR_FILES = {
    "full_inpcrd": "full.inpcrd.pre_ring_repair",
    "full_pdb": "full.pdb.pre_ring_repair",
    "vac_inpcrd": "vac.inpcrd.pre_ring_repair",
    "vac_pdb": "vac.pdb.pre_ring_repair",
}
_MIN_SDR_SOLVATION_BUFFER_Z = 3.0
_WATER_RESNAMES = {
    "WAT",
    "HOH",
    "TIP3",
    "TIP3P",
    "TIP4P",
    "SPC",
    "SPCE",
    "OPC",
    "SOL",
}
_WATER_OXYGEN_NAMES = {"O", "OW", "OH2"}
_PERIODIC_WATER_CLASH_DISTANCE = 1.8
_FE_NONWATER_VAC_COMPONENTS = {"d", "l", "z"}


def _is_water_residue_name(name: str) -> bool:
    return str(name).strip().upper() in _WATER_RESNAMES


def _copy_structure_without_water(structure):
    nonwater = copy.copy(structure)
    water_selection = [
        _is_water_residue_name(atom.residue.name) for atom in nonwater.atoms
    ]
    if any(water_selection):
        nonwater.strip(water_selection)
    return nonwater


def _copy_structure_only_water(structure):
    water = copy.copy(structure)
    nonwater_selection = [
        not _is_water_residue_name(atom.residue.name) for atom in water.atoms
    ]
    if any(nonwater_selection):
        water.strip(nonwater_selection)
    return water


def _split_structure_nonwater_then_water(structure):
    nonwater = _copy_structure_without_water(structure)
    water = _copy_structure_only_water(structure)
    if len(water.atoms) == 0:
        return nonwater, water, nonwater
    return nonwater, water, nonwater + water


def _repair_lipid_hydrogens_in_amber_files(
    window_dir: Path,
    *,
    prefix: str,
    report_name: str,
) -> None:
    prmtop = window_dir / f"{prefix}.prmtop"
    inpcrd = window_dir / f"{prefix}.inpcrd"
    pdb = window_dir / f"{prefix}.pdb"
    if not prmtop.exists() or not inpcrd.exists():
        return
    result = repair_lipid_hydrogen_geometry(
        prmtop,
        inpcrd,
        pdb if pdb.exists() else None,
        write_report=window_dir / report_name,
    )
    if result.repaired:
        logger.debug(
            "Repaired {} lipid hydrogen(s) on {} carbon center(s) in {}.",
            result.repaired_hydrogens,
            result.repaired_centers,
            prmtop.name,
        )


def _repair_lipid_hydrogens_after_tleap_lipids(window_dir: Path) -> None:
    _repair_lipid_hydrogens_in_amber_files(
        window_dir,
        prefix="solvate_others",
        report_name="lipid_hydrogen_repair_solvate_others.json",
    )


def _repair_parmed_molecule_table_for_combine(structure: pmd.Structure) -> pmd.Structure:
    """Repair stale Amber molecule metadata before ParmEd Structure addition."""
    parm_data = getattr(structure, "parm_data", None)
    if parm_data is None or not hasattr(structure, "rediscover_molecules"):
        return structure

    def _table_is_valid() -> bool:
        solvent_pointers = parm_data.get("SOLVENT_POINTERS")
        atoms_per_molecule = parm_data.get("ATOMS_PER_MOLECULE")
        if (
            solvent_pointers is None
            or atoms_per_molecule is None
            or len(solvent_pointers) < 2
        ):
            return True
        expected_molecules = int(solvent_pointers[1])
        atom_count = len(getattr(structure, "atoms", []))
        return (
            len(atoms_per_molecule) == expected_molecules
            and sum(atoms_per_molecule) == atom_count
        )

    solvent_pointers = parm_data.get("SOLVENT_POINTERS")
    atoms_per_molecule = parm_data.get("ATOMS_PER_MOLECULE")
    if solvent_pointers is None or atoms_per_molecule is None or len(solvent_pointers) < 2:
        return structure

    if not _table_is_valid():
        structure.rediscover_molecules()
    if not _table_is_valid():
        from parmed.utils import tag_molecules

        molecule_atom_counts = [len(molecule) for molecule in tag_molecules(structure)]
        atom_count = len(getattr(structure, "atoms", []))
        if not molecule_atom_counts or sum(molecule_atom_counts) != atom_count:
            molecule_atom_counts = [atom_count] if atom_count else []
        parm_data["SOLVENT_POINTERS"] = [
            len(getattr(structure, "residues", [])),
            len(molecule_atom_counts),
            len(molecule_atom_counts) + 1,
        ]
        parm_data["ATOMS_PER_MOLECULE"] = molecule_atom_counts
    return structure


def _save_coordinate_snapshot(
    structure: pmd.Structure,
    coordinates: np.ndarray,
    *,
    inpcrd_path: Path,
    pdb_path: Path,
) -> None:
    current_coordinates = np.asarray(structure.coordinates, dtype=float).copy()
    try:
        structure.coordinates = np.asarray(coordinates, dtype=float).copy()
        structure.save(str(inpcrd_path), format="rst7", overwrite=True)
        structure.save(str(pdb_path), format="pdb", overwrite=True)
    finally:
        structure.coordinates = current_coordinates


def _save_pre_ring_repair_snapshots(
    window_dir: Path,
    *,
    vac: pmd.Structure,
    vac_coordinates: np.ndarray,
    combined: pmd.Structure,
    combined_coordinates: np.ndarray,
) -> dict[str, str]:
    _save_coordinate_snapshot(
        vac,
        vac_coordinates,
        inpcrd_path=window_dir / _PRE_RING_REPAIR_FILES["vac_inpcrd"],
        pdb_path=window_dir / _PRE_RING_REPAIR_FILES["vac_pdb"],
    )
    _save_coordinate_snapshot(
        combined,
        combined_coordinates,
        inpcrd_path=window_dir / _PRE_RING_REPAIR_FILES["full_inpcrd"],
        pdb_path=window_dir / _PRE_RING_REPAIR_FILES["full_pdb"],
    )
    return dict(_PRE_RING_REPAIR_FILES)


_HY36_DIGITS_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HY36_DIGITS_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


def _decode_pure_base36(value: str, digits: str) -> int:
    decoded = 0
    for char in value:
        decoded *= len(digits)
        decoded += digits.index(char)
    return decoded


def _hy36decode(width: int, value: str) -> int:
    """Decode a hybrid-36 PDB number.

    AmberTools can use hybrid-36 residue IDs once a PDB exceeds the decimal
    residue field. MDAnalysis decodes hybrid-36 atom serials, but not residue
    IDs, so BATTER normalizes those fields before handing the PDB to MDAnalysis.
    """
    if len(value) != width:
        raise ValueError(f"invalid hybrid-36 field width: {value!r}")

    first = value[0]
    if first in {"-", " "} or first.isdigit():
        return int(value)
    if first in _HY36_DIGITS_UPPER:
        return (
            _decode_pure_base36(value, _HY36_DIGITS_UPPER)
            - 10 * 36 ** (width - 1)
            + 10**width
        )
    if first in _HY36_DIGITS_LOWER:
        return (
            _decode_pure_base36(value, _HY36_DIGITS_LOWER)
            + 16 * 36 ** (width - 1)
            + 10**width
        )
    raise ValueError(f"invalid hybrid-36 field: {value!r}")


def _pdb_coordinate_fields_are_parseable(line: str) -> bool:
    if len(line) < 54:
        return False
    try:
        float(line[30:38])
        float(line[38:46])
        float(line[46:54])
    except ValueError:
        return False
    return True


def _normalize_decimal_resid_overflow_line(line: str) -> str | None:
    line_body = line.rstrip("\n")
    line_ending = line[len(line_body) :]
    match = re.match(
        r"(?P<resid>-?\d{5,})\s+"
        r"(?P<x>[-+]?\d+\.\d+)\s+"
        r"(?P<y>[-+]?\d+\.\d+)\s*"
        r"(?P<z>[-+]?\d+\.\d+)"
        r"(?P<rest>\s+.*)$",
        line_body[22:],
    )
    if match is None:
        return None

    try:
        resid = int(match.group("resid"))
        x = float(match.group("x"))
        y = float(match.group("y"))
        z = float(match.group("z"))
    except ValueError:
        return None

    normalized = (
        f"{line_body[:22]}{resid % 10000:04d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{match.group('rest')}"
        f"{line_ending}"
    )
    return normalized if _pdb_coordinate_fields_are_parseable(normalized) else None


def _normalize_hybrid36_resids_for_mdanalysis(pdb_path: Path) -> Path | None:
    """Return a temp PDB with hybrid-36 residue IDs converted for MDAnalysis.

    MDAnalysis' PDB parser treats non-decimal residue fields such as ``A6VB`` as
    missing and assigns resid 1, which merges consecutive waters into one
    residue. For MDAnalysis parsing only, convert such fields to their decimal
    residue number modulo the 4-column PDB field. MDAnalysis' existing wraparound
    logic then restores monotonically increasing residue IDs for normal Amber
    output order. Older Amber five-digit residue output is left unchanged when
    coordinate columns remain parseable; six-digit decimal residue overflow is
    folded back into the 4-column residue field so coordinates realign.
    """
    normalized_lines: list[str] = []
    changed = False

    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
                resid_field = line[22:26]
                try:
                    int(resid_field)
                except ValueError:
                    try:
                        resid = _hy36decode(4, resid_field)
                    except ValueError:
                        normalized_lines.append(line)
                    else:
                        changed = True
                        normalized_lines.append(
                            f"{line[:22]}{resid % 10000:04d}{line[26:]}"
                        )
                else:
                    if _pdb_coordinate_fields_are_parseable(line):
                        normalized_lines.append(line)
                        continue
                    normalized = _normalize_decimal_resid_overflow_line(line)
                    if normalized is None:
                        normalized_lines.append(line)
                        continue
                    changed = True
                    normalized_lines.append(normalized)
            else:
                normalized_lines.append(line)

    if not changed:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w")
    try:
        tmp.writelines(normalized_lines)
    finally:
        tmp.close()
    return Path(tmp.name)


@contextmanager
def _mdanalysis_pdb_path(pdb_path: Path) -> Iterator[Path]:
    normalized = _normalize_hybrid36_resids_for_mdanalysis(pdb_path)
    if normalized is None:
        yield pdb_path
        return

    try:
        yield normalized
    finally:
        normalized.unlink(missing_ok=True)


def _cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_res_blocks(selection, out_pdb: Path) -> None:
    lines = []
    if len(selection.residues) != 0:
        prev = selection.residues.resids[0]
        for res in selection.residues:
            if res.resid != prev:
                lines.append("TER\n")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            res.atoms.write(tmp.name)
            tmp.close()
            with open(tmp.name) as f:
                lines += [ln for ln in f if ln.startswith("ATOM")]
            prev = res.resid
    out_pdb.write_text("".join(lines))


def _pdb_line_resname(line: str) -> str:
    return line[17:20].strip() if len(line) >= 20 else ""


def _pdb_line_atom_name(line: str) -> str:
    return line[12:16].strip() if len(line) >= 16 else ""


def _renumber_pdb_atom_line(
    line: str,
    *,
    serial: int,
    resid: int,
    chain: str = "W",
) -> str:
    line = line.rstrip("\n")
    if len(line) < 54:
        line = line.ljust(54)
    xyz = _pdb_atom_xyz(line)
    resid_field = ((int(resid) - 1) % 9999) + 1
    if xyz is None:
        return (
            f"{line[:6]}{serial:5d}"
            f"{line[11:21]}{chain[:1]}{resid_field:4d} {line[27:]}\n"
        )
    return (
        f"{line[:6]}{serial:5d}"
        f"{line[11:21]}{chain[:1]}{resid_field:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}\n"
    )


def _iter_water_blocks_from_pdb(pdb_path: Path) -> Iterator[list[str]]:
    current: list[str] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if _pdb_line_resname(line).upper() not in _WATER_RESNAMES:
            continue
        atom_name = _pdb_line_atom_name(line).upper()
        if atom_name in _WATER_OXYGEN_NAMES and current:
            yield current
            current = []
        current.append(line)
    if current:
        yield current


def _pdb_atom_xyz(line: str) -> np.ndarray | None:
    try:
        return np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
    except Exception:
        normalized = _normalize_decimal_resid_overflow_line(line)
        if normalized is None:
            return None
        try:
            return np.asarray(
                [
                    float(normalized[30:38]),
                    float(normalized[38:46]),
                    float(normalized[46:54]),
                ],
                dtype=float,
            )
        except Exception:
            return None


def _translate_pdb_atom_line(line: str, delta: np.ndarray) -> str:
    xyz = _pdb_atom_xyz(line)
    if xyz is None:
        return line
    shifted = xyz + np.asarray(delta, dtype=float)
    if len(line) < 54:
        line = line.ljust(54)
    return (
        f"{line[:30]}{shifted[0]:8.3f}{shifted[1]:8.3f}{shifted[2]:8.3f}"
        f"{line[54:]}"
    )


def _translate_pdb_block(block: Sequence[str], delta: np.ndarray) -> list[str]:
    if not np.any(np.asarray(delta, dtype=float)):
        return list(block)
    return [_translate_pdb_atom_line(line, delta) for line in block]


def _pdb_atom_name_is_hydrogen(name: str) -> bool:
    stripped = name.strip().upper()
    return stripped.startswith("H") or (
        len(stripped) > 1 and stripped[0].isdigit() and stripped[1] == "H"
    )


def _extra_ligand_heavy_coords_from_build(
    build_pdb: Path,
    ligand_resname: str,
) -> np.ndarray:
    ligand_blocks: list[list[np.ndarray]] = []
    current_key: tuple[str, str, str] | None = None
    current_coords: list[np.ndarray] = []
    for line in build_pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if _pdb_line_resname(line) != ligand_resname:
            continue
        key = (line[21:22], line[22:26], line[26:27])
        if current_key is not None and key != current_key:
            ligand_blocks.append(current_coords)
            current_coords = []
        current_key = key
        if not _pdb_atom_name_is_hydrogen(_pdb_line_atom_name(line)):
            xyz = _pdb_atom_xyz(line)
            if xyz is not None:
                current_coords.append(xyz)
    if current_key is not None:
        ligand_blocks.append(current_coords)
    if len(ligand_blocks) <= 1:
        return np.empty((0, 3), dtype=float)
    coords = [coord for block in ligand_blocks[1:] for coord in block]
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _water_block_overlaps_coords(
    block: Sequence[str],
    coords: np.ndarray,
    *,
    box: np.ndarray,
    cutoff: float,
) -> bool:
    if coords.size == 0:
        return False
    cutoff2 = float(cutoff) * float(cutoff)
    for line in block:
        if _pdb_line_atom_name(line).upper() not in {"O", "OW", "OH2"}:
            continue
        water_xyz = _pdb_atom_xyz(line)
        if water_xyz is None:
            continue
        delta = coords - water_xyz
        if box.shape == (3,) and np.all(np.isfinite(box)) and np.all(box > 0.0):
            delta = delta - box * np.round(delta / box)
        if float(np.min(np.sum(delta * delta, axis=1))) < cutoff2:
            return True
    return False


def _box_array_is_valid(box: Sequence[float] | np.ndarray) -> bool:
    box_array = np.asarray(box, dtype=float)
    return (
        box_array.shape == (3,)
        and np.all(np.isfinite(box_array))
        and np.all(box_array > 0.0)
    )


def _periodic_cells(
    coords: np.ndarray,
    box: np.ndarray,
    *,
    bin_width: float,
) -> tuple[dict[tuple[int, int, int], list[int]], np.ndarray]:
    wrapped = coords - box * np.floor(coords / box)
    n_bins = np.maximum(np.floor(box / max(float(bin_width), 1.0e-6)).astype(int), 1)
    cell_width = box / n_bins
    bins = np.floor(wrapped / cell_width).astype(int)
    bins = np.minimum(bins, n_bins - 1)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for atom_i, key in enumerate(map(tuple, bins)):
        cells.setdefault(key, []).append(atom_i)
    return cells, n_bins


def _neighbor_periodic_cell_keys(
    key: tuple[int, int, int],
    n_bins: np.ndarray,
) -> Iterator[tuple[int, int, int]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (
                    (key[0] + dx) % int(n_bins[0]),
                    (key[1] + dy) % int(n_bins[1]),
                    (key[2] + dz) % int(n_bins[2]),
                )


def _minimum_image_distance(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    box: np.ndarray,
) -> tuple[float, float]:
    raw_delta = coord_b - coord_a
    raw_distance = float(np.sqrt(np.dot(raw_delta, raw_delta)))
    delta = raw_delta - box * np.round(raw_delta / box)
    return raw_distance, float(np.sqrt(np.dot(delta, delta)))


def _water_block_record(path: Path, block_index: int, block: Sequence[str]) -> dict[str, Any]:
    atom_coords: list[np.ndarray] = []
    atom_is_hydrogen: list[bool] = []
    oxygen_coord: np.ndarray | None = None
    for line in block:
        coord = _pdb_atom_xyz(line)
        if coord is None:
            continue
        atom_coords.append(coord)
        atom_is_hydrogen.append(_pdb_atom_is_hydrogen(line))
        if oxygen_coord is None and _pdb_line_atom_name(line).upper() in _WATER_OXYGEN_NAMES:
            oxygen_coord = coord
    return {
        "path": path,
        "block_index": block_index,
        "lines": list(block),
        "atom_coords": np.asarray(atom_coords, dtype=float),
        "atom_is_hydrogen": np.asarray(atom_is_hydrogen, dtype=bool),
        "oxygen_coord": oxygen_coord,
    }


def _read_nonwater_pdb_atoms(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    coords: list[np.ndarray] = []
    is_hydrogen: list[bool] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if _pdb_line_resname(line).upper() in _WATER_RESNAMES:
                continue
            coord = _pdb_atom_xyz(line)
            if coord is None:
                continue
            coords.append(coord)
            is_hydrogen.append(_pdb_atom_is_hydrogen(line))
    if not coords:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=bool)
    return np.asarray(coords, dtype=float), np.asarray(is_hydrogen, dtype=bool)


def _rewrite_cleaned_water_pdbs(
    water_pdbs: Sequence[Path],
    water_blocks: Sequence[dict[str, Any]],
    remove_blocks: set[tuple[Path, int]],
) -> dict[str, int]:
    blocks_by_path: dict[Path, list[dict[str, Any]]] = {Path(path): [] for path in water_pdbs}
    for block in water_blocks:
        blocks_by_path.setdefault(Path(block["path"]), []).append(block)

    kept_by_path: dict[str, int] = {}
    for path in water_pdbs:
        retained = [
            block
            for block in blocks_by_path.get(Path(path), [])
            if (Path(block["path"]), int(block["block_index"])) not in remove_blocks
        ]
        if not retained:
            Path(path).unlink(missing_ok=True)
            kept_by_path[Path(path).name] = 0
            continue

        serial = 1
        lines: list[str] = []
        for resid, block in enumerate(retained, start=1):
            for atom_line in block["lines"]:
                lines.append(
                    _renumber_pdb_atom_line(atom_line, serial=serial, resid=resid)
                )
                serial += 1
            lines.append("TER\n")
        lines.append("END\n")
        Path(path).write_text("".join(lines))
        kept_by_path[Path(path).name] = len(retained)
    return kept_by_path


def _cleanup_periodic_water_pdbs(
    water_pdbs: Sequence[Path],
    *,
    nonwater_pdbs: Sequence[Path] = (),
    box: Sequence[float] | np.ndarray,
    cutoff: float = _PERIODIC_WATER_CLASH_DISTANCE,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Remove water residues that clash before LEaP consumes water PDBs.

    The cleanup is intentionally water-only. It removes duplicate waters whose
    atoms overlap under the minimum-image convention, and removes waters whose
    atoms overlap non-water atoms. Protein, ligand, lipid, ion, and dummy atoms
    are left untouched.
    """
    water_paths = [Path(path) for path in water_pdbs if Path(path).exists()]
    summary: dict[str, Any] = {
        "water_files": [path.name for path in water_paths],
        "nonwater_files": [Path(path).name for path in nonwater_pdbs if Path(path).exists()],
        "removed_water_residues": 0,
        "removed_water_water": 0,
        "removed_water_nonwater": 0,
        "kept_water_residues_by_file": {},
    }
    if report_path is not None:
        report_path.unlink(missing_ok=True)
    if not water_paths:
        return summary

    box_array = np.asarray(box, dtype=float)
    if not _box_array_is_valid(box_array):
        return summary

    water_blocks: list[dict[str, Any]] = []
    for path in water_paths:
        for block_index, block in enumerate(_iter_water_blocks_from_pdb(path)):
            record = _water_block_record(path, block_index, block)
            if record["oxygen_coord"] is not None and record["atom_coords"].size:
                water_blocks.append(record)

    if not water_blocks:
        return summary

    remove_blocks: set[tuple[Path, int]] = set()
    oxygen_coords = np.asarray(
        [block["oxygen_coord"] for block in water_blocks],
        dtype=float,
    )
    cells, n_bins = _periodic_cells(
        oxygen_coords,
        box_array,
        bin_width=cutoff,
    )
    water_oxygen_cutoff2 = float(cutoff) * float(cutoff)
    visited_oxygen_pairs: set[tuple[int, int]] = set()
    for key, indices in cells.items():
        for neighbor_key in _neighbor_periodic_cell_keys(key, n_bins):
            for atom_i in indices:
                for atom_j in cells.get(neighbor_key, []):
                    if atom_j <= atom_i:
                        continue
                    pair = (atom_i, atom_j)
                    if pair in visited_oxygen_pairs:
                        continue
                    visited_oxygen_pairs.add(pair)
                    _, pbc_distance = _minimum_image_distance(
                        oxygen_coords[atom_i],
                        oxygen_coords[atom_j],
                        box_array,
                    )
                    if pbc_distance * pbc_distance >= water_oxygen_cutoff2:
                        continue
                    block = water_blocks[atom_j]
                    block_key = (Path(block["path"]), int(block["block_index"]))
                    if block_key not in remove_blocks:
                        remove_blocks.add(block_key)
                        summary["removed_water_water"] += 1

    water_atom_coords: list[np.ndarray] = []
    water_atom_hydrogen: list[bool] = []
    water_atom_blocks: list[int] = []
    for block_index, block in enumerate(water_blocks):
        for atom_coord, is_hydrogen in zip(
            block["atom_coords"],
            block["atom_is_hydrogen"],
        ):
            water_atom_coords.append(atom_coord)
            water_atom_hydrogen.append(bool(is_hydrogen))
            water_atom_blocks.append(block_index)

    if water_atom_coords:
        all_water_coords = np.asarray(water_atom_coords, dtype=float)
        all_water_hydrogen = np.asarray(water_atom_hydrogen, dtype=bool)
        atom_cells, atom_n_bins = _periodic_cells(
            all_water_coords,
            box_array,
            bin_width=cutoff,
        )
        heavy_hydrogen_cutoff = min(float(cutoff), 1.2)
        hydrogen_hydrogen_cutoff = min(float(cutoff), 1.0)
        heavy_heavy_cutoff = min(float(cutoff), 1.5)
        visited_atom_pairs: set[tuple[int, int]] = set()
        for key, atom_indices in atom_cells.items():
            for neighbor_key in _neighbor_periodic_cell_keys(key, atom_n_bins):
                for atom_i in atom_indices:
                    block_i = water_atom_blocks[atom_i]
                    block_i_key = (
                        Path(water_blocks[block_i]["path"]),
                        int(water_blocks[block_i]["block_index"]),
                    )
                    if block_i_key in remove_blocks:
                        continue
                    for atom_j in atom_cells.get(neighbor_key, []):
                        if atom_j <= atom_i:
                            continue
                        pair = (atom_i, atom_j)
                        if pair in visited_atom_pairs:
                            continue
                        visited_atom_pairs.add(pair)
                        block_j = water_atom_blocks[atom_j]
                        if block_i == block_j:
                            continue
                        block_j_key = (
                            Path(water_blocks[block_j]["path"]),
                            int(water_blocks[block_j]["block_index"]),
                        )
                        if block_j_key in remove_blocks:
                            continue
                        _, pbc_distance = _minimum_image_distance(
                            all_water_coords[atom_i],
                            all_water_coords[atom_j],
                            box_array,
                        )
                        if all_water_hydrogen[atom_i] and all_water_hydrogen[atom_j]:
                            pair_cutoff = hydrogen_hydrogen_cutoff
                        elif all_water_hydrogen[atom_i] or all_water_hydrogen[atom_j]:
                            pair_cutoff = heavy_hydrogen_cutoff
                        else:
                            pair_cutoff = heavy_heavy_cutoff
                        if pbc_distance < pair_cutoff:
                            remove_blocks.add(block_j_key)
                            summary["removed_water_water"] += 1

    other_coords, other_hydrogen = _read_nonwater_pdb_atoms(nonwater_pdbs)
    if other_coords.size:
        other_cells, other_n_bins = _periodic_cells(
            other_coords,
            box_array,
            bin_width=cutoff,
        )
        heavy_hydrogen_cutoff = min(float(cutoff), 1.2)
        hydrogen_hydrogen_cutoff = min(float(cutoff), 1.0)
        heavy_heavy_cutoff = min(float(cutoff), 1.5)

        for block_index, block in enumerate(water_blocks):
            block_key = (Path(block["path"]), int(block["block_index"]))
            if block_key in remove_blocks:
                continue
            atom_coords = block["atom_coords"]
            atom_hydrogen = block["atom_is_hydrogen"]
            water_cells, water_n_bins = _periodic_cells(
                atom_coords,
                box_array,
                bin_width=cutoff,
            )
            if not np.array_equal(water_n_bins, other_n_bins):
                raise RuntimeError("Internal periodic water cleanup bin mismatch.")

            removed = False
            for key, water_atom_indices in water_cells.items():
                for neighbor_key in _neighbor_periodic_cell_keys(key, other_n_bins):
                    other_indices = other_cells.get(neighbor_key)
                    if not other_indices:
                        continue
                    for water_atom_i in water_atom_indices:
                        for other_atom_i in other_indices:
                            _, pbc_distance = _minimum_image_distance(
                                atom_coords[water_atom_i],
                                other_coords[other_atom_i],
                                box_array,
                            )
                            if atom_hydrogen[water_atom_i] and other_hydrogen[other_atom_i]:
                                pair_cutoff = hydrogen_hydrogen_cutoff
                            elif atom_hydrogen[water_atom_i] or other_hydrogen[other_atom_i]:
                                pair_cutoff = heavy_hydrogen_cutoff
                            else:
                                pair_cutoff = heavy_heavy_cutoff
                            if pbc_distance < pair_cutoff:
                                remove_blocks.add(block_key)
                                summary["removed_water_nonwater"] += 1
                                removed = True
                                break
                        if removed:
                            break
                    if removed:
                        break
                if removed:
                    break

    if not remove_blocks:
        summary["kept_water_residues_by_file"] = _rewrite_cleaned_water_pdbs(
            water_paths,
            water_blocks,
            remove_blocks,
        )
        return summary

    summary["removed_water_residues"] = len(remove_blocks)
    summary["kept_water_residues_by_file"] = _rewrite_cleaned_water_pdbs(
        water_paths,
        water_blocks,
        remove_blocks,
    )
    if report_path is not None:
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.debug(
        "Removed {} water residue(s) with periodic clashes from {}.",
        summary["removed_water_residues"],
        ", ".join(path.name for path in water_paths),
    )
    return summary


def _pdb_min_z(pdb_path: Path) -> float:
    min_z = float("inf")
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        xyz = _pdb_atom_xyz(line)
        if xyz is None:
            continue
        min_z = min(min_z, float(xyz[2]))
    if not np.isfinite(min_z):
        raise ValueError(f"No readable atom z coordinates found in {pdb_path}")
    return min_z


def _write_membrane_water_chunks_from_build(
    window_dir: Path,
    *,
    ligand_resname: str,
    box: Sequence[float],
    z_max: float | None = None,
    reference_z_period: float | None = None,
    ligand_clash_cutoff: float = 2.2,
    max_waters_per_chunk: int = 8000,
) -> list[Path]:
    for pattern in (
        "solvate_pre_wat_*.pdb",
        "solvate_wat_*.pdb",
        "solvate_wat_*.prmtop",
        "solvate_wat_*.inpcrd",
        "tleap_solvate_wat_*.in",
        "tleap_solvate_wat_*.log",
    ):
        for old_path in window_dir.glob(pattern):
            old_path.unlink(missing_ok=True)

    chunks: list[Path] = []
    current_blocks: list[list[str]] = []
    build_pdb = window_dir / "build.pdb"
    extra_ligand_coords = _extra_ligand_heavy_coords_from_build(
        build_pdb, ligand_resname
    )
    box_array = np.asarray(box, dtype=float)
    z_min = _pdb_min_z(build_pdb) if z_max is not None else None
    z_period = (
        float(reference_z_period)
        if reference_z_period is not None
        and np.isfinite(float(reference_z_period))
        and float(reference_z_period) > 0.0
        else None
    )
    skipped_z = 0
    skipped_overlap = 0
    tiled_waters = 0

    def flush() -> None:
        if not current_blocks:
            return
        chunk_index = len(chunks)
        out_pdb = window_dir / f"solvate_pre_wat_{chunk_index:02d}.pdb"
        serial = 1
        lines: list[str] = []
        for resid, block in enumerate(current_blocks, start=1):
            for atom_line in block:
                lines.append(
                    _renumber_pdb_atom_line(atom_line, serial=serial, resid=resid)
                )
                serial += 1
            lines.append("TER\n")
        lines.append("END\n")
        out_pdb.write_text("".join(lines))
        chunks.append(out_pdb)
        current_blocks.clear()

    for block in _iter_water_blocks_from_pdb(build_pdb):
        oxygen_z = None
        for line in block:
            if _pdb_line_atom_name(line).upper() not in _WATER_OXYGEN_NAMES:
                continue
            water_xyz = _pdb_atom_xyz(line)
            if water_xyz is not None:
                oxygen_z = float(water_xyz[2])
                break
        if oxygen_z is None:
            continue

        shift_indices = [0]
        if z_min is not None and z_max is not None and z_period is not None:
            low_index = int(np.floor((float(z_min) - oxygen_z) / z_period))
            high_index = int(np.ceil((float(z_max) - oxygen_z) / z_period))
            shift_indices = list(range(low_index, high_index + 1))

        for shift_index in shift_indices:
            dz = (z_period or 0.0) * float(shift_index)
            shifted_oxygen_z = oxygen_z + dz
            if z_max is not None and shifted_oxygen_z > float(z_max) + 1.0e-6:
                if shift_index == 0:
                    skipped_z += 1
                continue
            if z_min is not None and shifted_oxygen_z < float(z_min) - 1.0e-6:
                continue

            shifted_block = _translate_pdb_block(
                block,
                np.asarray([0.0, 0.0, dz], dtype=float),
            )
            if _water_block_overlaps_coords(
                shifted_block,
                extra_ligand_coords,
                box=box_array,
                cutoff=ligand_clash_cutoff,
            ):
                skipped_overlap += 1
                continue
            if shift_index != 0:
                tiled_waters += 1
            current_blocks.append(shifted_block)
            if len(current_blocks) >= max_waters_per_chunk:
                flush()
    flush()
    if not chunks:
        raise ValueError(f"No membrane waters found in {build_pdb}")
    if tiled_waters:
        logger.debug(
            "Added {} periodically tiled membrane water(s) to cover the SDR z extent in {}.",
            tiled_waters,
            window_dir,
        )
    if skipped_z:
        logger.debug(
            "Removed {} membrane water(s) outside the SDR z extent in {}.",
            skipped_z,
            window_dir,
        )
    if skipped_overlap:
        logger.debug(
            "Removed {} membrane water(s) overlapping extra ligand copy/copies in {}.",
            skipped_overlap,
            window_dir,
        )
    return chunks


_REFERENCE_PROTON_RESTORE_EXCLUDED_RESNAMES = {
    "WAT",
    "HOH",
    "TIP3",
    "TIP3P",
    "TIP4P",
    "SPC",
    "SPCE",
    "OPC",
    "SOL",
    "NA",
    "NA+",
    "SOD",
    "K",
    "K+",
    "POT",
    "CL",
    "CL-",
    "CLA",
    "MG",
    "MG2",
    "CA",
    "CA2",
    "ZN",
    "DUM",
}


def _pdb_atom_coord(line: str) -> np.ndarray | None:
    if len(line) < 54:
        return None
    try:
        return np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
    except ValueError:
        return None


def _pdb_atom_is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element == "H"
    atom_name = line[12:16].strip().upper()
    return atom_name.startswith("H") or (
        len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == "H"
    )


def _replace_pdb_coord(line: str, coord: np.ndarray) -> str:
    return f"{line[:30]}{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{line[54:]}"


def _kabsch_transform(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = mobile_centered.T @ reference_centered
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def _read_pdb_residue_blocks(
    lines: Sequence[str],
) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    current_key: tuple[str, str, str, str] | None = None
    current: dict[str, Any] | None = None
    for line_index, line in enumerate(lines):
        if line.startswith("TER"):
            current_key = None
            current = None
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        coord = _pdb_atom_coord(line)
        if coord is None:
            continue
        key = (line[17:20].strip(), line[21:22], line[22:26], line[26:27])
        if current is None or key != current_key:
            current = {
                "resname": key[0],
                "atoms": {},
                "duplicate_atom_names": set(),
            }
            residues.append(current)
            current_key = key

        atom_name = line[12:16].strip()
        atoms = current["atoms"]
        if atom_name in atoms:
            current["duplicate_atom_names"].add(atom_name)
            continue
        atoms[atom_name] = {
            "line_index": line_index,
            "coord": coord,
            "is_hydrogen": _pdb_atom_is_hydrogen(line),
        }
    return residues


def _restore_reference_hydrogen_coordinates(
    target_pdb: Path,
    reference_pdb: Path,
    *,
    residue_names: Sequence[str] | None = None,
    exclude_residue_names: Sequence[str] | None = None,
    min_common_heavy_atoms: int = 3,
    heavy_rmsd_cutoff: float = 0.25,
) -> int:
    """Restore target hydrogen coordinates from an equilibrated reference.

    Residues are paired by occurrence within each residue name. For each pair,
    common heavy atoms are locally fit from reference to target, and matching
    target hydrogen coordinates are replaced by the transformed reference
    hydrogen coordinates. This preserves the target heavy-atom coordinates and
    avoids mapping newly solvated molecules onto the old reference.
    """
    if not target_pdb.exists() or not reference_pdb.exists():
        return 0

    include = {name.strip().upper() for name in residue_names or [] if name}
    exclude = {
        name.strip().upper()
        for name in (
            exclude_residue_names
            if exclude_residue_names is not None
            else _REFERENCE_PROTON_RESTORE_EXCLUDED_RESNAMES
        )
        if name
    }

    target_lines = target_pdb.read_text().splitlines(True)
    reference_lines = reference_pdb.read_text().splitlines(True)
    target_residues = _read_pdb_residue_blocks(target_lines)
    reference_by_name: dict[str, list[dict[str, Any]]] = {}
    for residue in _read_pdb_residue_blocks(reference_lines):
        reference_by_name.setdefault(str(residue["resname"]).upper(), []).append(residue)

    reference_indices: dict[str, int] = {}
    restored = 0
    skipped_rmsd = 0
    for target_residue in target_residues:
        resname = str(target_residue["resname"]).upper()
        if include and resname not in include:
            continue
        if resname in exclude:
            continue

        reference_residues = reference_by_name.get(resname)
        if not reference_residues:
            continue
        reference_index = reference_indices.get(resname, 0)
        reference_indices[resname] = reference_index + 1
        if reference_index >= len(reference_residues):
            continue
        reference_residue = reference_residues[reference_index]

        target_atoms = target_residue["atoms"]
        reference_atoms = reference_residue["atoms"]
        target_duplicates = target_residue["duplicate_atom_names"]
        reference_duplicates = reference_residue["duplicate_atom_names"]

        common_heavy_names = [
            name
            for name in sorted(set(target_atoms) & set(reference_atoms))
            if name not in target_duplicates
            and name not in reference_duplicates
            and not target_atoms[name]["is_hydrogen"]
            and not reference_atoms[name]["is_hydrogen"]
        ]
        if len(common_heavy_names) < min_common_heavy_atoms:
            continue

        reference_heavy = np.asarray(
            [reference_atoms[name]["coord"] for name in common_heavy_names],
            dtype=float,
        )
        target_heavy = np.asarray(
            [target_atoms[name]["coord"] for name in common_heavy_names],
            dtype=float,
        )
        rotation, translation = _kabsch_transform(reference_heavy, target_heavy)
        aligned_reference_heavy = reference_heavy @ rotation + translation
        rmsd = float(
            np.sqrt(np.mean(np.sum((aligned_reference_heavy - target_heavy) ** 2, axis=1)))
        )
        if not np.isfinite(rmsd) or rmsd > heavy_rmsd_cutoff:
            skipped_rmsd += 1
            continue

        for atom_name, target_atom in target_atoms.items():
            if atom_name in target_duplicates or atom_name in reference_duplicates:
                continue
            if not target_atom["is_hydrogen"]:
                continue
            reference_atom = reference_atoms.get(atom_name)
            if reference_atom is None or not reference_atom["is_hydrogen"]:
                continue
            new_coord = reference_atom["coord"] @ rotation + translation
            target_lines[target_atom["line_index"]] = _replace_pdb_coord(
                target_lines[target_atom["line_index"]],
                new_coord,
            )
            restored += 1

    if restored:
        target_pdb.write_text("".join(target_lines))
        logger.debug(
            "Restored {} existing hydrogen coordinate(s) in {} from {}.",
            restored,
            target_pdb.name,
            reference_pdb.name,
        )
    if skipped_rmsd:
        logger.debug(
            "Skipped {} residue(s) while restoring hydrogens in {} because heavy-atom RMSD exceeded {:.3f} Å.",
            skipped_rmsd,
            target_pdb.name,
            heavy_rmsd_cutoff,
        )
    return restored


def _restore_existing_protons_from_reference(window_dir: Path, target_pdb: Path) -> int:
    for reference_name in ("equil-reference.pdb", "rec_file.pdb"):
        reference_pdb = window_dir / reference_name
        if reference_pdb.exists():
            return _restore_reference_hydrogen_coordinates(target_pdb, reference_pdb)
    return 0


def _parmed_reference_translation(
    structure: pmd.Structure,
    reference_pdb: Path,
    *,
    selection: str = "protein and name CA",
) -> np.ndarray | None:
    if not reference_pdb.exists() or structure.coordinates is None:
        return None
    current_indices = [
        atom.idx
        for atom in structure.atoms
        if str(atom.name).strip() == "CA"
    ]
    if not current_indices:
        return None
    reference = mda.Universe(str(reference_pdb))
    reference_atoms = reference.select_atoms(selection)
    if reference_atoms.n_atoms != len(current_indices):
        logger.debug(
            "Could not compute reference-frame translation: {} current CA atoms, "
            "{} reference atoms from {!r}.",
            len(current_indices),
            reference_atoms.n_atoms,
            selection,
        )
        return None
    current = np.asarray(structure.coordinates, dtype=float)[current_indices]
    translation = np.median(reference_atoms.positions - current, axis=0)
    if not np.all(np.isfinite(translation)):
        return None
    return np.asarray(translation, dtype=float)


def _translate_parmed_structure(structure: pmd.Structure, translation: Sequence[float]) -> None:
    if structure.coordinates is None:
        return
    vector = np.asarray(translation, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return
    if float(np.linalg.norm(vector)) <= 1.0e-4:
        return
    structure.coordinates = np.asarray(structure.coordinates, dtype=float) + vector


_TERMINAL_AMIDE_CAP_ATOMS = {"N1": "N", "H1": "HN1", "H2": "HN2"}
_TERMINAL_METHYLAMIDE_RESNAMES = {"NMA", "NME"}
_N_TERMINAL_CAP_RESNAMES = {"ACE"}
_C_TERMINAL_CAP_RESNAMES = {"NMA", "NME", "NHE"}
_PROTEIN_TERMINAL_CAP_RESNAME_SET = (
    _N_TERMINAL_CAP_RESNAMES | _C_TERMINAL_CAP_RESNAMES
)
_PROTEIN_TERMINAL_CAP_RESNAMES = "ACE NMA NME NHE"
_PROTEIN_WITH_TERMINAL_CAPS = f"(protein or resname {_PROTEIN_TERMINAL_CAP_RESNAMES})"
_AMBER_PROTEIN_RESNAMES = {
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
_EMBEDDED_METHYLAMIDE_CARBON_ALIASES = ("CH3", "C1", "CM", "CR")
_SEPARATE_METHYLAMIDE_ATOM_ALIASES = {
    "N": "N",
    "H": "H",
    "HNT": "H",
    "HN": "H",
    "HN1": "H",
    "CH3": "C",
    "C": "C",
    "CA": "C",
    "CAT": "C",
    "C1": "C",
    "CM": "C",
    "CR": "C",
    "HH31": "H1",
    "HH32": "H2",
    "HH33": "H3",
    "H31": "H1",
    "H32": "H2",
    "H33": "H3",
    "H31H": "H1",
    "H32H": "H2",
    "H33H": "H3",
    "H1": "H1",
    "H2": "H2",
    "H3": "H3",
    "1HA": "H1",
    "2HA": "H2",
    "3HA": "H3",
    "HT1": "H1",
    "HT2": "H2",
    "HT3": "H3",
    "HR1": "H1",
    "HR2": "H2",
    "HR3": "H3",
    "HM1": "H1",
    "HM2": "H2",
    "HM3": "H3",
}
_ACE_CAP_ATOM_ALIASES = {
    "C": "C",
    "O": "O",
    "OY": "O",
    "CH3": "CH3",
    "CAY": "CH3",
    "CY": "CH3",
    "H1": "H1",
    "H2": "H2",
    "H3": "H3",
    "HY1": "H1",
    "HY2": "H2",
    "HY3": "H3",
    "1HY": "H1",
    "2HY": "H2",
    "3HY": "H3",
    "HH31": "H1",
    "HH32": "H2",
    "HH33": "H3",
}
_C_TERMINAL_CARBONYL_O_ALIASES = ("OT1", "OC1")
_C_TERMINAL_OXT_ALIASES = ("OT2", "OC2", "O2", "O1")


def _is_amber_protein_resname(resname: str) -> bool:
    return resname.strip().upper() in _AMBER_PROTEIN_RESNAMES


def _pdb_atom_name(line: str) -> str:
    return line[12:16].strip()


def _pdb_residue_key(line: str) -> tuple[str, int, str] | None:
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
        return None
    try:
        resid = int(line[22:26])
    except ValueError:
        return None
    return (line[21].strip(), resid, line[17:20].strip())


def _replace_pdb_atom_name(line: str, atom_name: str) -> str:
    line_body = line.rstrip("\n")
    line_ending = line[len(line_body) :]
    if len(line_body) < 16:
        line_body = line_body.ljust(16)
    atom_field = atom_name[:4] if len(atom_name) >= 4 else f" {atom_name:<3}"
    return f"{line_body[:12]}{atom_field}{line_body[16:]}{line_ending}"


def _replace_pdb_residue(line: str, *, resname: str, resid: int) -> str:
    line_body = line.rstrip("\n")
    line_ending = line[len(line_body) :]
    if len(line_body) < 26:
        line_body = line_body.ljust(26)
    return f"{line_body[:17]}{resname:>3}{line_body[20:22]}{resid:4d}{line_body[26:]}{line_ending}"


def _residue_keys_in_order(block: list[str]) -> list[tuple[str, int, str]]:
    keys: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in block:
        key = _pdb_residue_key(line)
        if key is not None and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _atom_names_for_residue(
    block: list[str], residue_key: tuple[str, int, str]
) -> set[str]:
    return {
        _pdb_atom_name(line)
        for line in block
        if _pdb_residue_key(line) == residue_key
    }


def _embedded_methylamide_cap_atoms(atom_names: set[str]) -> dict[str, str] | None:
    methyl_carbons = [
        name for name in _EMBEDDED_METHYLAMIDE_CARBON_ALIASES if name in atom_names
    ]
    if not methyl_carbons:
        return None

    aliases = {"N1": "N", methyl_carbons[0]: "C"}
    if "HN" in atom_names:
        aliases["HN"] = "H"
        methyl_hydrogens = ("H1", "H2", "H3")
    elif "HN1" in atom_names:
        aliases["HN1"] = "H"
        methyl_hydrogens = ("H1", "H2", "H3")
    elif "H" in atom_names:
        aliases["H"] = "H"
        methyl_hydrogens = ("H1", "H2", "H3")
    else:
        aliases["H1"] = "H"
        methyl_hydrogens = ("H2", "H3", "H4")

    for source, target in zip(methyl_hydrogens, ("H1", "H2", "H3")):
        if source in atom_names:
            aliases[source] = target
    for source, target in {
        "HH31": "H1",
        "HH32": "H2",
        "HH33": "H3",
        "H31": "H1",
        "H32": "H2",
        "H33": "H3",
        "H31H": "H1",
        "H32H": "H2",
        "H33H": "H3",
        "HR1": "H1",
        "HR2": "H2",
        "HR3": "H3",
        "HM1": "H1",
        "HM2": "H2",
        "HM3": "H3",
    }.items():
        if source in atom_names:
            aliases[source] = target

    return aliases


def _rewrite_separate_terminal_methylamide_cap(
    block: list[str], residue_keys: list[tuple[str, int, str]]
) -> tuple[list[str], bool] | None:
    if len(residue_keys) < 2:
        return None

    terminal_key = residue_keys[-1]
    if terminal_key[2] not in _TERMINAL_METHYLAMIDE_RESNAMES:
        return None

    previous_key = residue_keys[-2]
    if not _is_amber_protein_resname(previous_key[2]):
        return None

    atom_names = _atom_names_for_residue(block, terminal_key)
    if (
        "N" not in atom_names
        or "O" in atom_names
        or not any(
            name in atom_names
            for name in ("CH3", "C", "CA", "CAT", "C1", "CM", "CR")
        )
        or not atom_names.issubset(_SEPARATE_METHYLAMIDE_ATOM_ALIASES)
    ):
        return None

    rewritten: list[str] = []
    emitted_cap_atoms: set[str] = set()
    changed = terminal_key[2] != "NME"
    for line in block:
        key = _pdb_residue_key(line)
        atom_name = _pdb_atom_name(line)
        if key == previous_key and atom_name == "OXT":
            changed = True
            continue
        if key == terminal_key:
            cap_atom = _SEPARATE_METHYLAMIDE_ATOM_ALIASES[atom_name]
            if cap_atom in emitted_cap_atoms:
                changed = True
                continue
            emitted_cap_atoms.add(cap_atom)
            changed = changed or cap_atom != atom_name
            cap_line = _replace_pdb_atom_name(line, cap_atom)
            cap_line = _replace_pdb_residue(
                cap_line,
                resname="NME",
                resid=terminal_key[1],
            )
            rewritten.append(cap_line)
            continue
        rewritten.append(line)

    return rewritten, changed


def _rewrite_ace_caps_for_leap(pdb_path: Path) -> int:
    """
    Rewrite common ACE atom aliases into Amber ACE template atom names.

    Dabble-style ACE caps can use ``CAY/HY1/HY2/HY3/OY`` while Amber LEaP's
    ACE template expects ``CH3/H1/H2/H3/O``. If the aliases are left in place,
    LEaP creates untyped CAY/OY atoms and later fails during ``saveamberparm``.
    """
    lines = pdb_path.read_text().splitlines(True)
    rewritten: list[str] = []
    emitted_by_residue: dict[tuple[str, int, str], set[str]] = {}
    changed_residues: set[tuple[str, int, str]] = set()

    for line in lines:
        key = _pdb_residue_key(line)
        if key is None or key[2] != "ACE":
            rewritten.append(line)
            continue

        atom_name = _pdb_atom_name(line)
        target_name = _ACE_CAP_ATOM_ALIASES.get(atom_name)
        if target_name is None:
            rewritten.append(line)
            continue

        emitted = emitted_by_residue.setdefault(key, set())
        if target_name in emitted:
            changed_residues.add(key)
            continue

        emitted.add(target_name)
        if target_name != atom_name:
            line = _replace_pdb_atom_name(line, target_name)
            changed_residues.add(key)
        rewritten.append(line)

    if changed_residues:
        pdb_path.write_text("".join(rewritten))
    return len(changed_residues)


def _rewrite_cterminal_oxygen_aliases_for_leap(pdb_path: Path) -> int:
    """
    Normalize C-terminal protein oxygen aliases before LEaP.

    Some prepared protein PDBs carry terminal oxygens as ``OT1/OT2``,
    ``OC1/OC2``, or an extra ``O1`` atom. Amber's C-terminal residue templates
    expect the backbone carbonyl ``O`` and terminal ``OXT``; leaving aliases on
    the terminal amino-acid residue creates untyped atoms such as ``CASP@O1``.
    """
    lines = pdb_path.read_text().splitlines(True)
    rewritten: list[str] = []
    block: list[str] = []
    changed_count = 0

    def flush_block() -> None:
        nonlocal changed_count
        if not block:
            return

        residue_keys = _residue_keys_in_order(block)
        if not residue_keys:
            rewritten.extend(block)
            block.clear()
            return

        terminal_key = residue_keys[-1]
        if not _is_amber_protein_resname(terminal_key[2]):
            rewritten.extend(block)
            block.clear()
            return

        atom_names = _atom_names_for_residue(block, terminal_key)
        rename: dict[str, str] = {}
        drop: set[str] = set()
        has_o = "O" in atom_names
        has_oxt = "OXT" in atom_names

        if not has_o:
            for alias in _C_TERMINAL_CARBONYL_O_ALIASES:
                if alias in atom_names:
                    rename[alias] = "O"
                    has_o = True
                    break
            if (
                not has_o
                and "O1" in atom_names
                and any(alias in atom_names for alias in ("O2", "OT2", "OC2"))
            ):
                rename["O1"] = "O"
                has_o = True

        if has_oxt:
            for alias in _C_TERMINAL_OXT_ALIASES:
                if alias in atom_names:
                    drop.add(alias)
        else:
            for alias in _C_TERMINAL_OXT_ALIASES:
                if alias in atom_names and rename.get(alias) != "O":
                    rename[alias] = "OXT"
                    has_oxt = True
                    break
            if has_oxt:
                for alias in _C_TERMINAL_OXT_ALIASES:
                    if alias in atom_names and alias not in rename:
                        drop.add(alias)

        if not rename and not drop:
            rewritten.extend(block)
            block.clear()
            return

        changed = False
        for line in block:
            key = _pdb_residue_key(line)
            atom_name = _pdb_atom_name(line)
            if key == terminal_key and atom_name in drop:
                changed = True
                continue
            if key == terminal_key and atom_name in rename:
                line = _replace_pdb_atom_name(line, rename[atom_name])
                changed = True
            rewritten.append(line)
        if changed:
            changed_count += 1
        block.clear()

    for line in lines:
        if line.startswith("TER"):
            flush_block()
            rewritten.append(line)
        else:
            block.append(line)
    flush_block()

    if changed_count:
        pdb_path.write_text("".join(rewritten))
    return changed_count


def _rewrite_terminal_amide_caps_for_leap(
    pdb_path: Path,
    *,
    exclude_residue_names: Sequence[str] | None = None,
) -> int:
    """
    Rewrite terminal amide caps into Amber residue/atom names.

    Peptide inputs can encode a C-terminal amide on the final amino-acid residue
    itself. LEaP then treats cap atoms like ``N1`` as unknown atoms on ``CXXX``.
    Moving those atoms into following ``NHE`` or ``NME`` residues lets the
    standard aminoct library type the cap and bond it to the preceding residue.
    """
    lines = pdb_path.read_text().splitlines(True)
    excluded = {
        str(name).strip()
        for name in (exclude_residue_names or ())
        if str(name).strip()
    }
    rewritten: list[str] = []
    block: list[str] = []
    cap_count = 0
    used_resids = {
        (key[0], key[1])
        for key in (_pdb_residue_key(line) for line in lines)
        if key is not None
    }
    all_resids = {resid for _chain, resid in used_resids}
    next_cap_resid = max(all_resids, default=0) + 1

    def take_cap_resid(chain_id: str, after_resid: int) -> int:
        nonlocal next_cap_resid
        while (chain_id, next_cap_resid) in used_resids and next_cap_resid <= 9999:
            next_cap_resid += 1
        if next_cap_resid > 9999:
            candidate = min(max(int(after_resid) + 1, 1), 9999)
            for offset in range(9999):
                resid = ((candidate - 1 + offset) % 9999) + 1
                if (chain_id, resid) not in used_resids:
                    used_resids.add((chain_id, resid))
                    return resid
            raise ValueError(
                f"Unable to assign a unique PDB residue ID for terminal amide cap in {pdb_path}"
            )
        resid = next_cap_resid
        used_resids.add((chain_id, resid))
        next_cap_resid += 1
        return resid

    def flush_block() -> None:
        nonlocal cap_count
        if not block:
            return

        residue_keys = _residue_keys_in_order(block)
        if not residue_keys:
            rewritten.extend(block)
            block.clear()
            return

        terminal_key = residue_keys[-1]
        if terminal_key[2] in excluded:
            rewritten.extend(block)
            block.clear()
            return

        separate_methylamide = _rewrite_separate_terminal_methylamide_cap(
            block, residue_keys
        )
        if separate_methylamide is not None:
            methylamide_lines, changed = separate_methylamide
            rewritten.extend(methylamide_lines)
            if changed:
                cap_count += 1
            block.clear()
            return

        terminal_atom_names = _atom_names_for_residue(block, terminal_key)
        if (
            "N1" not in terminal_atom_names
            or not _is_amber_protein_resname(terminal_key[2])
        ):
            rewritten.extend(block)
            block.clear()
            return

        methylamide_atoms = _embedded_methylamide_cap_atoms(terminal_atom_names)
        if methylamide_atoms is not None:
            cap_atom_map = methylamide_atoms
            cap_resname = "NME"
        else:
            cap_atom_map = _TERMINAL_AMIDE_CAP_ATOMS
            cap_resname = "NHE"

        cap_lines: list[str] = []
        body_lines: list[str] = []
        has_amide_n = False
        cap_resid = take_cap_resid(terminal_key[0], terminal_key[1])

        for line in block:
            key = _pdb_residue_key(line)
            atom_name = _pdb_atom_name(line)
            if key == terminal_key and atom_name in cap_atom_map:
                has_amide_n = has_amide_n or atom_name == "N1"
                cap_atom = cap_atom_map[atom_name]
                cap_line = _replace_pdb_atom_name(line, cap_atom)
                cap_line = _replace_pdb_residue(
                    cap_line,
                    resname=cap_resname,
                    resid=cap_resid,
                )
                cap_lines.append(cap_line)
                continue
            if key == terminal_key and atom_name == "OXT":
                continue
            body_lines.append(line)

        if has_amide_n:
            rewritten.extend(body_lines)
            rewritten.extend(cap_lines)
            cap_count += 1
        else:
            rewritten.extend(block)
        block.clear()

    for line in lines:
        if line.startswith("TER"):
            flush_block()
            rewritten.append(line)
        else:
            block.append(line)
    flush_block()

    if cap_count:
        pdb_path.write_text("".join(rewritten))
    return cap_count


def _chain_id_from_renum(
    renum_df: pd.DataFrame, *, resid: int, resname: str
) -> str:
    """Return the original chain ID for a residue in an Amber-renumbered PDB."""
    candidates = renum_df.query(
        "new_resid == @resid and new_resname == @resname"
    )
    if candidates.empty:
        candidates = renum_df.query(
            "old_resid == @resid and old_resname == @resname"
        )
    if candidates.empty:
        raise ValueError(
            f"Unable to map Amber residue {resname} {resid} back to an input chain"
        )
    return candidates.old_chain.values[0]


def _renum_resname(row: pd.Series) -> str:
    new_resname = str(row.get("new_resname", "")).strip()
    return new_resname or str(row.get("old_resname", "")).strip()


def _renum_row_is_protein_like(row: pd.Series) -> bool:
    resname = _renum_resname(row)
    return (
        _is_amber_protein_resname(resname)
        or resname in _PROTEIN_TERMINAL_CAP_RESNAME_SET
    )


def _resnames_match_for_renum(residue_resname: str, row: pd.Series) -> bool:
    residue_resname = residue_resname.strip()
    row_resnames = {
        str(row.get("old_resname", "")).strip(),
        str(row.get("new_resname", "")).strip(),
    }
    if residue_resname in row_resnames:
        return True
    return (
        residue_resname in _C_TERMINAL_CAP_RESNAMES
        and any(name in _C_TERMINAL_CAP_RESNAMES for name in row_resnames)
    )


def _collapse_terminal_cap_resid_values(
    renum_df: pd.DataFrame, resids: list[int] | np.ndarray
) -> list[int]:
    collapsed = [int(resid) for resid in resids]
    rows = renum_df.reset_index(drop=True)
    if len(rows) != len(collapsed):
        return collapsed

    for pos, row in rows.iterrows():
        resname = _renum_resname(row)
        if resname in _N_TERMINAL_CAP_RESNAMES:
            search_range = range(pos + 1, len(rows))
        elif resname in _C_TERMINAL_CAP_RESNAMES:
            search_range = range(pos - 1, -1, -1)
        else:
            continue

        chain = str(row["old_chain"]).strip()
        for neighbor_pos in search_range:
            neighbor = rows.iloc[neighbor_pos]
            if str(neighbor["old_chain"]).strip() != chain:
                continue
            if _renum_resname(neighbor) in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
                continue
            collapsed[pos] = collapsed[neighbor_pos]
            break
    return collapsed


def _residue_chain_id(residue) -> str:
    try:
        chain_ids = residue.atoms.chainIDs
    except Exception:
        chain_ids = []
    if len(chain_ids):
        return str(chain_ids[0]).strip()
    return str(getattr(residue, "segid", "")).strip()


def _collapse_terminal_cap_resids_in_place(residues) -> None:
    if len(residues) == 0:
        return

    resids = np.array(residues.resids, dtype=int)
    chain_ids = [_residue_chain_id(residue) for residue in residues]
    resnames = [str(residue.resname).strip() for residue in residues]

    for pos, resname in enumerate(resnames):
        if resname in _N_TERMINAL_CAP_RESNAMES:
            search_range = range(pos + 1, len(residues))
        elif resname in _C_TERMINAL_CAP_RESNAMES:
            search_range = range(pos - 1, -1, -1)
        else:
            continue

        for neighbor_pos in search_range:
            if chain_ids[neighbor_pos] != chain_ids[pos]:
                continue
            if resnames[neighbor_pos] in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
                continue
            resids[pos] = resids[neighbor_pos]
            break

    residues.resids = resids


def _renum_old_resids_for_residues(residues, renum_df: pd.DataFrame) -> list[int]:
    rows = renum_df.reset_index(drop=True)
    row_pos = 0
    old_resids: list[int] = []

    for residue in residues:
        resname = str(residue.resname).strip()
        if resname in ["HIS", "HIE", "HIP", "HID"]:
            resname = "HIS"

        while row_pos < len(rows) and not _renum_row_is_protein_like(
            rows.iloc[row_pos]
        ):
            row_pos += 1

        if row_pos < len(rows) and _resnames_match_for_renum(
            resname, rows.iloc[row_pos]
        ):
            old_resids.append(int(rows.iloc[row_pos]["old_resid"]))
            row_pos += 1
            continue

        if resname in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
            old_resids.append(int(residue.resid))
            continue

        while row_pos < len(rows) and _renum_resname(
            rows.iloc[row_pos]
        ) in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
            row_pos += 1
        while row_pos < len(rows) and not _renum_row_is_protein_like(
            rows.iloc[row_pos]
        ):
            row_pos += 1
        if row_pos < len(rows):
            old_resids.append(int(rows.iloc[row_pos]["old_resid"]))
            row_pos += 1
        else:
            old_resids.append(int(residue.resid))

    return old_resids


def _renum_chain_ids_for_residues(residues, renum_df: pd.DataFrame) -> list[str]:
    rows = renum_df.reset_index(drop=True)
    row_pos = 0
    chain_ids: list[str] = []
    last_chain = "A"

    for residue in residues:
        resname = str(residue.resname).strip()
        if resname in ["HIS", "HIE", "HIP", "HID"]:
            resname = "HIS"

        while row_pos < len(rows) and not _renum_row_is_protein_like(
            rows.iloc[row_pos]
        ):
            row_pos += 1

        if row_pos < len(rows) and _resnames_match_for_renum(
            resname, rows.iloc[row_pos]
        ):
            last_chain = str(rows.iloc[row_pos]["old_chain"]).strip() or last_chain
            chain_ids.append(last_chain)
            row_pos += 1
            continue

        if resname in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
            chain_ids.append(last_chain)
            continue

        while row_pos < len(rows) and _renum_resname(
            rows.iloc[row_pos]
        ) in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
            last_chain = str(rows.iloc[row_pos]["old_chain"]).strip() or last_chain
            row_pos += 1
        while row_pos < len(rows) and not _renum_row_is_protein_like(
            rows.iloc[row_pos]
        ):
            row_pos += 1
        if row_pos < len(rows):
            last_chain = str(rows.iloc[row_pos]["old_chain"]).strip() or last_chain
            chain_ids.append(last_chain)
            row_pos += 1
        else:
            chain_ids.append(last_chain)

    return chain_ids


def _restore_protein_resids_from_renum(atom_group, renum_df: pd.DataFrame) -> None:
    residues = atom_group.select_atoms(_PROTEIN_WITH_TERMINAL_CAPS).residues
    if len(residues) == 0:
        return
    residues.resids = _renum_old_resids_for_residues(residues, renum_df)
    _collapse_terminal_cap_resids_in_place(residues)


def _ligand_charge_from_metadata(meta_path: Path) -> int | None:
    """Return the integer ligand charge recorded during parametrization."""
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        charge_val = data.get("ligand_charge")
        if charge_val is None:
            return None
        return int(round(float(charge_val)))
    except Exception as exc:
        logger.debug(f"Failed to read ligand charge from {meta_path}: {exc}")
        return None


def _read_disulfide_pairs(sslink_path: Path) -> list[tuple[int, int]]:
    """Read pdb4amber's 1-based residue-index disulfide pairs."""
    if not sslink_path.exists():
        return []

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for line_no, line in enumerate(sslink_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 2:
            logger.warning(
                f"Skipping malformed disulfide record {sslink_path}:{line_no}: {line!r}"
            )
            continue
        try:
            first, second = int(fields[0]), int(fields[1])
        except ValueError:
            logger.warning(
                f"Skipping malformed disulfide record {sslink_path}:{line_no}: {line!r}"
            )
            continue
        if first <= 0 or second <= 0 or first == second:
            logger.warning(
                f"Skipping invalid disulfide record {sslink_path}:{line_no}: {line!r}"
            )
            continue

        pair = tuple(sorted((first, second)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _map_disulfide_pairs_to_resids(
    pairs: list[tuple[int, int]], revised_resids: list[int] | np.ndarray
) -> list[tuple[int, int]]:
    """Map pdb4amber residue indices to the residue IDs written to LEaP PDBs."""
    revised = [int(resid) for resid in revised_resids]
    mapped: list[tuple[int, int]] = []
    for first, second in pairs:
        if first > len(revised) or second > len(revised):
            logger.warning(
                f"Skipping disulfide pair {first} {second}: only {len(revised)} residues were mapped"
            )
            continue
        mapped.append((revised[first - 1], revised[second - 1]))
    return mapped


def _merge_disulfide_pairs(
    pairs: list[tuple[int, int]], extra_pairs: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for first, second in [*pairs, *extra_pairs]:
        pair = tuple(sorted((int(first), int(second))))
        if pair in seen:
            continue
        seen.add(pair)
        merged.append(pair)
    return merged


def _infer_cyx_disulfide_pairs_from_atoms(
    atoms: mda.AtomGroup, *, max_sg_distance: float = 2.8
) -> list[tuple[int, int]]:
    """Infer close CYX SG-SG pairs that pdb4amber may omit from sslink."""
    records: list[tuple[int, np.ndarray]] = []
    for residue in atoms.select_atoms("protein and resname CYX").residues:
        sg_atoms = residue.atoms.select_atoms("name SG")
        if sg_atoms.n_atoms != 1:
            continue
        records.append(
            (int(residue.resid), np.asarray(sg_atoms[0].position, dtype=float))
        )

    candidates: list[tuple[float, tuple[int, int]]] = []
    for idx, (first_resid, first_pos) in enumerate(records):
        for second_resid, second_pos in records[idx + 1 :]:
            distance = float(np.linalg.norm(first_pos - second_pos))
            if distance <= float(max_sg_distance):
                candidates.append((distance, tuple(sorted((first_resid, second_resid)))))

    inferred: list[tuple[int, int]] = []
    used_resids: set[int] = set()
    for _distance, pair in sorted(candidates, key=lambda item: item[0]):
        first, second = pair
        if first in used_resids or second in used_resids:
            continue
        used_resids.update(pair)
        inferred.append(pair)
    return inferred


def _mark_disulfide_residue_names(residues, disulfide_resids: set[int]) -> None:
    """Ensure disulfide cysteines are written as CYX before LEaP loads them."""
    if not disulfide_resids:
        return

    for residue in residues:
        if (
            int(residue.resid) in disulfide_resids
            and residue.resname in {"CYS", "CYX"}
        ):
            residue.resname = "CYX"


def _is_disulfide_thiol_hydrogen_line(line: str, disulfide_resids: set[int]) -> bool:
    """Return True for cysteine SG hydrogen records that should not survive as CYX."""
    if not disulfide_resids or not line.startswith(("ATOM  ", "HETATM")):
        return False
    atom_name = line[12:16].strip()
    if atom_name not in {"HG", "HG1"}:
        return False
    resname = line[17:20].strip()
    if resname != "CYX":
        return False
    try:
        resid = int(line[22:26])
    except ValueError:
        return False
    return resid in disulfide_resids


def _write_leap_disulfide_bonds(
    handle, unit_name: str, disulfide_pairs: list[tuple[int, int]]
) -> None:
    """Write explicit LEaP SG-SG bonds for pdb4amber-detected disulfides."""
    if not disulfide_pairs:
        return

    for first, second in disulfide_pairs:
        handle.write(f"bond {unit_name}.{first}.SG {unit_name}.{second}.SG\n")
    handle.write("\n")


def _map_disulfide_pairs_to_leap_indices(
    disulfide_pairs: list[tuple[int, int]], pdb_path: Path
) -> list[tuple[int, int]]:
    """Map PDB residue IDs to the contiguous residue indices used by LEaP."""
    if not disulfide_pairs:
        return []

    residue_order: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in pdb_path.read_text().splitlines():
        key = _pdb_residue_key(line)
        if key is None or key in seen:
            continue
        seen.add(key)
        residue_order.append(key)

    if not residue_order:
        return disulfide_pairs

    leap_index = residue_order[0][1]
    resid_to_leap_index: dict[int, int] = {}
    ambiguous_resids: set[int] = set()
    for _chain, resid, _resname in residue_order:
        if resid in resid_to_leap_index:
            ambiguous_resids.add(resid)
        else:
            resid_to_leap_index[resid] = leap_index
        leap_index += 1

    mapped: list[tuple[int, int]] = []
    for first, second in disulfide_pairs:
        if first in ambiguous_resids or second in ambiguous_resids:
            logger.warning(
                f"Skipping disulfide pair {first} {second}: duplicate residue IDs in {pdb_path}"
            )
            continue
        try:
            mapped.append((resid_to_leap_index[first], resid_to_leap_index[second]))
        except KeyError:
            logger.warning(
                f"Skipping disulfide pair {first} {second}: residue ID not present in {pdb_path}"
            )
    return mapped


def _replace_anchor_mask_resid(mask: str | None, resid: int) -> str | None:
    if not mask:
        return mask
    return re.sub(r":-?\d+(?=@)", f":{resid}", mask, count=1)


def _find_ligand_resid_in_pdb(pdb_path: Path, ligand_resname: str) -> int | None:
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip() != ligand_resname:
            continue
        key = _pdb_residue_key(line)
        if key is not None:
            return key[1]
    return None


def _sync_ligand_anchor_residue_with_pdb(
    working_dir: Path, pdb_path: Path, ligand_resname: str
) -> None:
    anchors_path = working_dir / "anchors.json"
    if not anchors_path.exists():
        return

    actual_lig_res = _find_ligand_resid_in_pdb(pdb_path, ligand_resname)
    if actual_lig_res is None:
        logger.warning(
            f"Could not find ligand residue {ligand_resname!r} in {pdb_path}; leaving anchors unchanged"
        )
        return

    anchors = load_anchors(working_dir)
    if str(actual_lig_res) == str(anchors.lig_res):
        return

    save_anchors(
        working_dir,
        Anchors(
            P1=anchors.P1,
            P2=anchors.P2,
            P3=anchors.P3,
            L1=_replace_anchor_mask_resid(anchors.L1, actual_lig_res),
            L2=_replace_anchor_mask_resid(anchors.L2, actual_lig_res),
            L3=_replace_anchor_mask_resid(anchors.L3, actual_lig_res),
            lig_res=str(actual_lig_res),
        ),
    )
    logger.debug(
        "Updated ligand anchor residue from {} to {} after LEaP residue numbering.",
        anchors.lig_res,
        actual_lig_res,
    )


def _write_abfe_diff_charge_ligand_from_ref_vac(
    window_dir: Path,
    ligand_resname: str,
) -> Path:
    """Write a one-ligand PDB copied from the equilibrated pre_fe bulk ligand."""
    ref_vac_pdb = window_dir / "ref_vac.pdb"
    u_ref_vac = mda.Universe(ref_vac_pdb.as_posix())
    ligands = u_ref_vac.select_atoms(f"resname {ligand_resname}").residues
    if len(ligands) < 2:
        raise ValueError(
            f"ABFE_diff d construction requires two pre_fe ligand residues in "
            f"{ref_vac_pdb}; found {len(ligands)} for resname {ligand_resname!r}."
        )
    out_pdb = window_dir / "charge_ligand_aligned_solvent.pdb"
    ligands[1].atoms.write(out_pdb.as_posix())
    return out_pdb


def _rename_parmed_residues(
    structure: pmd.Structure,
    residue_indices: list[int] | tuple[int, ...],
    residue_name: str,
) -> None:
    """Rename residues in both ParmEd objects and AmberParm metadata."""
    labels = getattr(structure, "parm_data", {}).get("RESIDUE_LABEL")
    for index in residue_indices:
        if index < 0 or index >= len(structure.residues):
            continue
        residue = structure.residues[index]
        residue.name = residue_name
        for atom in residue.atoms:
            atom.residue.name = residue_name
        if labels is not None and index < len(labels):
            labels[index] = residue_name


def _make_residues_nonsteric(
    structure: pmd.Structure,
    residue_indices: list[int] | tuple[int, ...],
) -> None:
    """Give selected residues a private zero-LJ atom type while preserving charges."""
    selected = [0] * len(structure.atoms)
    for index in residue_indices:
        if index < 0 or index >= len(structure.residues):
            continue
        for atom in structure.residues[index].atoms:
            selected[atom.idx] = 1

    if not any(selected):
        return

    from parmed.tools.addljtype import AddLJType

    if getattr(structure, "chamber", False):
        AddLJType(structure, selected, 0.0, 0.0, 0.0, 0.0)
    else:
        AddLJType(structure, selected, 0.0, 0.0, None, None)
    structure.load_atom_info()


def _create_box_d_abfe_diff_from_pre_fe(ctx: BuildContext) -> None:
    """
    Build ABFE_diff d by reusing pre_fe topology pieces and appending one
    charge-balancing ligand copy, analogous to the x-component ParmEd combine.
    """
    sim = ctx.sim
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    amber_dir = ctx.amber_dir
    mol = ctx.residue_name

    required = [
        window_dir / "ref_vac.prmtop",
        window_dir / "ref_vac.pdb",
        window_dir / "other_parts.prmtop",
        window_dir / "other_parts.pdb",
        window_dir / f"{mol}.prmtop",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "ABFE_diff d x-style topology construction is missing required "
            "pre_fe piece(s): " + ", ".join(missing)
        )

    charge_ligand_pdb = _write_abfe_diff_charge_ligand_from_ref_vac(window_dir, mol)

    vac_p = pmd.load_file(
        str(window_dir / "ref_vac.prmtop"),
        str(window_dir / "ref_vac.pdb"),
    )
    other_part_p = pmd.load_file(
        str(window_dir / "other_parts.prmtop"),
        str(window_dir / "other_parts.pdb"),
    )
    charge_ligand_p = pmd.load_file(
        str(window_dir / f"{mol}.prmtop"),
        str(charge_ligand_pdb),
    )
    _rename_parmed_residues(charge_ligand_p, [0], mol)
    _repair_parmed_molecule_table_for_combine(charge_ligand_p)

    charge_ligand_index = len(vac_p.residues)
    combined = vac_p + charge_ligand_p + other_part_p
    vac, _, combined = _split_structure_nonwater_then_water(combined)
    _rename_parmed_residues(combined, [charge_ligand_index], mol)
    _rename_parmed_residues(vac, [charge_ligand_index], mol)
    _make_residues_nonsteric(combined, [charge_ligand_index])
    _make_residues_nonsteric(vac, [charge_ligand_index])

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)
    combined.save(str(window_dir / "full_pre.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)
    _sync_ligand_anchor_residue_with_pdb(build_dir, window_dir / "vac.pdb", mol)
    if work != build_dir:
        _sync_ligand_anchor_residue_with_pdb(work, window_dir / "vac.pdb", mol)

    u_full = mda.Universe(str(window_dir / "full.pdb"))
    u_vac = mda.Universe(str(window_dir / "vac.pdb"))

    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    if renum_txt.exists():
        renum_df2 = pd.read_csv(
            renum_txt,
            sep=r"\s+",
            header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
        )
        renum_df2["old_resname"] = renum_df2["old_resname"].replace(
            ["HIS", "HIE", "HIP", "HID"], "HIS"
        )
        renum_df2["new_resname"] = renum_df2["new_resname"].replace(
            ["HIS", "HIE", "HIP", "HID"], "HIS"
        )
        _restore_protein_resids_from_renum(u_full, renum_df2)
        _restore_protein_resids_from_renum(u_vac, renum_df2)

        chain_list = renum_df2.old_chain.values
        chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
        for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
            res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))
    u_vac.atoms.write(str(window_dir / "vac_orig.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    hmr_enabled = str(getattr(sim, "hmr", "no")).lower() == "yes"
    full_prmtop = (
        str(window_dir / "full.hmr.prmtop")
        if hmr_enabled
        else str(window_dir / "full.prmtop")
    )
    merge_first_n_and_lipid_fragments_in_prmtop(
        full_prmtop,
        6,
        getattr(sim, "lipid_mol", []),
        str(window_dir / "full_merged.prmtop"),
    )


@register_create_box("d")
@register_create_box("l")
@register_create_box("z")
def create_box(ctx: BuildContext) -> None:
    """
    Create the solvated box for the given component and window.
    """
    work = ctx.working_dir
    comp = ctx.comp
    param_dir = work.parent.parent / "params" if comp != "q" else work.parent / "params"
    sim = ctx.sim
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    amber_dir = ctx.amber_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    membrane_builder = sim.membrane_simulation
    use_membrane_reference_box = membrane_builder and comp != "q"
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol

    ligand = ctx.ligand
    mol = ctx.residue_name

    for attr in ("buffer_x", "buffer_y", "buffer_z"):
        if not hasattr(sim, attr):
            raise AttributeError(
                f"SimulationConfig missing '{attr}'. Please specify this buffer in the YAML."
            )
    buffer_x = float(sim.buffer_x)
    buffer_y = float(sim.buffer_y)
    buffer_z = float(sim.buffer_z)
    if (not membrane_builder) and (buffer_x < 5 or buffer_y < 5 or buffer_z < 5):
        raise ValueError("For water systems, buffer_x/y/z must be ≥ 5 Å.")

    if membrane_builder:
        buffer_x = 0.0
        buffer_y = 0.0
    else:
        # for non-equilibration non-membrane systems,
        # reduce the buffer by existing solvation shell
        if comp != 'q':
            solv_shell = sim.solv_shell
            buffer_x = max(0.0, buffer_x - solv_shell)
            buffer_y = max(0.0, buffer_y - solv_shell)
            buffer_z = max(0.0, buffer_z - solv_shell)


    sdr_abs_z: float | None = None
    if comp == "l":
        buffer_z_left = buffer_z
    elif comp != "q":
        sdr_dist, abs_z, buffer_z_left = map(float, open(window_dir / "sdr_info.txt").read().split())
        sdr_abs_z = abs_z
        if buffer_z_left < _MIN_SDR_SOLVATION_BUFFER_Z:
            logger.debug(
                "[create_box:{}] SDR solvation z buffer {:.3f} Å is below {:.1f} Å; using {:.1f} Å.",
                comp,
                buffer_z_left,
                _MIN_SDR_SOLVATION_BUFFER_Z,
                _MIN_SDR_SOLVATION_BUFFER_Z,
            )
            buffer_z_left = _MIN_SDR_SOLVATION_BUFFER_Z
    else:
        buffer_z_left = buffer_z

    reference_dimensions = None
    membrane_dimensions = None
    membrane_water_z_max = None
    if use_membrane_reference_box:
        reference_pdb = window_dir / "equil-reference.pdb"
        if not reference_pdb.exists():
            raise FileNotFoundError(
                f"Membrane FE box creation requires {reference_pdb} to preserve "
                "the equilibrated coordinate frame."
            )
        reference_universe = mda.Universe(str(reference_pdb))
        if reference_universe.dimensions is None:
            raise ValueError(f"{reference_pdb} does not contain box dimensions.")
        reference_dimensions = np.asarray(reference_universe.dimensions[:3], dtype=float)
        if reference_dimensions.shape != (3,) or not np.all(np.isfinite(reference_dimensions)):
            raise ValueError(f"{reference_pdb} contains invalid box dimensions.")
        membrane_dimensions = reference_dimensions.copy()
        if comp in {"e", "v", "o", "z", "d"}:
            if sdr_abs_z is None:
                raise ValueError(
                    f"Component {comp} requires SDR z information for membrane box creation."
                )
            membrane_dimensions[2] = float(sdr_abs_z)
            solvated_build_pdb = window_dir / "build.pdb"
            if solvated_build_pdb.exists():
                membrane_water_z_max = _pdb_min_z(solvated_build_pdb) + float(sdr_abs_z)

    if not hasattr(sim, "water_model"):
        raise AttributeError("SimulationConfig missing 'water_model'.")
    water_model = str(sim.water_model).upper()

    if not hasattr(sim, "ion_def"):
        raise AttributeError("SimulationConfig missing 'ion_def'.")
    ion_def = sim.ion_def

    if not hasattr(sim, "neut"):
        raise AttributeError("SimulationConfig missing 'neut'.")
    neut = str(sim.neut)

    if not hasattr(sim, "dec_method"):
        raise AttributeError("SimulationConfig missing 'dec_method'.")
    dec_method = str(sim.dec_method)

    if (
        comp == "d"
        and dec_method == "sdr"
        and getattr(sim, "fe_type", None) == "uno_rest_diff"
    ):
        _create_box_d_abfe_diff_from_pre_fe(ctx)
        return

    # ---- copy FF artifacts (resolve ff/ relative to window_dir: ../../param) ----
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "json"):
        src = param_dir / f"{ctx.residue_name}.{ext}"
        shutil.copy2(src, window_dir / src.name)

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{ctx.residue_name}.{ext}"
        shutil.copy2(src, window_dir / f"vac_ligand.{ext}")

    shutil.copy2(build_dir / f"{ligand}.pdb", window_dir / f"{ligand}.pdb")

    # other_mol
    if other_mol:
        raise NotImplementedError("Other molecules not supported now.")

    # tleap template
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    _cp(src_tleap, window_dir / "tleap.in")

    # water box keyword
    if water_model == "TIP3PF":
        # still uses leaprc.water.fb3
        water_box = "FB3BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    else:
        water_box = f"{water_model}BOX"

    build_ace_cap_count = _rewrite_ace_caps_for_leap(window_dir / "build.pdb")
    if build_ace_cap_count:
        logger.debug(
            "Rewrote {} ACE terminal cap(s) into Amber atom names before pre-solvation LEaP.",
            build_ace_cap_count,
        )
    build_cterm_o_count = _rewrite_cterminal_oxygen_aliases_for_leap(
        window_dir / "build.pdb"
    )
    if build_cterm_o_count:
        logger.debug(
            "Rewrote {} C-terminal oxygen alias residue(s) before pre-solvation LEaP.",
            build_cterm_o_count,
        )

    build_cap_count = _rewrite_terminal_amide_caps_for_leap(
        window_dir / "build.pdb",
        exclude_residue_names=[mol],
    )
    if build_cap_count:
        logger.debug(
            "Rewrote {} terminal protein amide cap(s) as Amber NHE/NME residues before pre-solvation LEaP.",
            build_cap_count,
        )

    # --- tleap solvate pre ---
    tleap_solv_pre = window_dir / "tleap_solvate_pre.in"
    _cp(window_dir / "tleap.in", tleap_solv_pre)
    with tleap_solv_pre.open("a") as f:
        f.write("# Load the necessary parameters\n")
        for om in other_mol:
            f.write(f"loadamberparams {om.lower()}.frcmod\n")
            f.write(f"{om} = loadmol2 {om.lower()}.mol2\n")
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        if water_model != "TIP3PF":
            f.write(f"source leaprc.water.{water_model.lower()}\n\n")
        else:
            f.write("source leaprc.water.fb3\n\n")
        build_input = "build-dry.pdb" if use_membrane_reference_box else "build.pdb"
        f.write(f"model = loadpdb {build_input}\n\n")
        if use_membrane_reference_box:
            assert membrane_dimensions is not None
            f.write(
                "set model box "
                f"{{{membrane_dimensions[0]:.6f} "
                f"{membrane_dimensions[1]:.6f} "
                f"{membrane_dimensions[2]:.6f}}}\n\n"
            )
        else:
            f.write(
                f"solvatebox model {water_box} {{ {buffer_x} {buffer_y} {buffer_z_left} }} 1\n\n"
            )
        f.write("desc model\n")
        f.write("savepdb model full_pre.pdb\n")
        f.write("quit\n")
    run_with_log(
        f"{tleap} -s -f {tleap_solv_pre.name} > tleap_solvate_pre.log",
        working_dir=window_dir,
    )
    _restore_existing_protons_from_reference(window_dir, window_dir / "full_pre.pdb")

    def _remove_stale_generated_files(patterns: tuple[str, ...]) -> None:
        for pattern in patterns:
            for path in window_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    if use_membrane_reference_box:
        _remove_stale_generated_files(
            (
                "solvate_pre_outside_wat.pdb",
                "solvate_outside_wat.pdb",
                "solvate_outside_wat.prmtop",
                "solvate_outside_wat.inpcrd",
                "tleap_solvate_outside_wat.in",
                "tleap_outside_wat.log",
                "solvate_pre_around_water.pdb",
                "solvate_around_wat.pdb",
                "solvate_around_wat.prmtop",
                "solvate_around_wat.inpcrd",
                "tleap_solvate_around_wat.in",
                "tleap_around_wat.log",
            )
        )
    else:
        _remove_stale_generated_files(
            (
                "solvate_pre_wat_*.pdb",
                "solvate_wat_*.pdb",
                "solvate_wat_*.prmtop",
                "solvate_wat_*.inpcrd",
                "tleap_solvate_wat_*.in",
                "tleap_solvate_wat_*.log",
            )
        )

    water_chunk_paths = (
        _write_membrane_water_chunks_from_build(
            window_dir,
            ligand_resname=mol,
            box=membrane_dimensions,
            z_max=membrane_water_z_max,
            reference_z_period=(
                float(reference_dimensions[2])
                if reference_dimensions is not None
                else None
            ),
        )
        if use_membrane_reference_box
        else []
    )

    # Count waters in build.pdb
    num_waters = sum(
        1 for ln in (window_dir / "build.pdb").read_text().splitlines() if "WAT" in ln
    )

    # pdb4amber is only used here for residue-renumbering and disulfide metadata.
    # For membrane FE systems, do not send the full solvent box through pdb4amber.
    pdb4amber_input = "build-dry.pdb" if use_membrane_reference_box else "build.pdb"
    _run_pdb4amber_for_box_or_copy(
        window_dir / pdb4amber_input,
        window_dir / "build_amber.pdb",
        working_dir=window_dir,
    )
    renum_df = pd.read_csv(
        window_dir / "build_amber_renum.txt",
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df["old_resname"] = renum_df["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    renum_df["new_resname"] = renum_df["new_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    revised_resids = revised_resids_for_lipid_fragments(
        (
            (row["old_resname"], row["old_chain"], row["old_resid"])
            for _, row in renum_df.iterrows()
        ),
        lipid_mol,
    )
    disulfide_pairs = _map_disulfide_pairs_to_resids(
        _read_disulfide_pairs(window_dir / "build_amber_sslink"), revised_resids
    )
    disulfide_resids = {resid for pair in disulfide_pairs for resid in pair}

    # MDAnalysis universes
    with _mdanalysis_pdb_path(window_dir / "full_pre.pdb") as full_pre_pdb:
        u = mda.Universe(str(full_pre_pdb))
        final_system = u.atoms
        system_dimensions = np.asarray(u.dimensions[:3], dtype=float).copy()

        if use_membrane_reference_box:
            assert membrane_dimensions is not None
            u.dimensions[:3] = membrane_dimensions
            system_dimensions = membrane_dimensions.copy()
            final_system = u.atoms
        elif membrane_builder:
            reference_pdb = window_dir / "equil-reference.pdb"
            if not reference_pdb.exists():
                raise FileNotFoundError(
                    f"Membrane equil box creation requires {reference_pdb} to trim "
                    "the LEaP-solvated box back to the reference x/y frame."
                )
            reference_universe = mda.Universe(str(reference_pdb))
            if reference_universe.dimensions is None:
                raise ValueError(f"{reference_pdb} does not contain box dimensions.")
            reference_dimensions_q = np.asarray(
                reference_universe.dimensions[:3],
                dtype=float,
            )
            if (
                reference_dimensions_q.shape != (3,)
                or not np.all(np.isfinite(reference_dimensions_q))
            ):
                raise ValueError(f"{reference_pdb} contains invalid box dimensions.")
            u.dimensions[0] = reference_dimensions_q[0]
            u.dimensions[1] = reference_dimensions_q[1]
            u.dimensions[2] = float(u.dimensions[2]) - 3.0
            u.atoms.positions[:, 2] -= 3.0
            system_dimensions = np.asarray(u.dimensions[:3], dtype=float).copy()
            final_system = u.atoms

            if lipid_mol:
                membrane_region = u.select_atoms(f'resname {" ".join(lipid_mol)}')
                membrane_phosphates = membrane_region.select_atoms("type P")
                if len(membrane_phosphates):
                    memb_z_max = membrane_phosphates.positions[:, 2].max() - 10.0
                    memb_z_min = membrane_phosphates.positions[:, 2].min() + 10.0
                    water_in_mem = u.select_atoms(
                        "byres (resname WAT and "
                        f"prop z > {memb_z_min} and prop z < {memb_z_max})"
                    )
                    final_system = final_system - water_in_mem

        if not use_membrane_reference_box:
            water_around_prot = u.select_atoms("resname WAT").residues[:num_waters].atoms
            final_system = final_system | water_around_prot

        if comp in ["e", "v", "o", "z", "d"] and not membrane_builder:
            min_pos = final_system.positions[:, 2].min()
            system_dimensions[2] = abs_z

            outside_wat_z = final_system.select_atoms(
                "byres (resname WAT and "
                f"(prop z > {abs_z + min_pos}))"
            )
            final_system = final_system - outside_wat_z

        if membrane_builder and not use_membrane_reference_box:
            half_x = float(system_dimensions[0]) / 2.0
            half_y = float(system_dimensions[1]) / 2.0
            outside_wat_xy = final_system.select_atoms(
                "byres (resname WAT and "
                f"((prop x > {half_x}) or (prop x < {-half_x}) or "
                f"(prop y > {half_y}) or (prop y < {-half_y})))"
            )
            final_system = final_system - outside_wat_xy

        # renumber residues
        revised_resids = np.array(revised_resids)
        total_residues = final_system.residues.n_residues
        final_resids = np.zeros(total_residues, dtype=int)
        final_resids[: len(revised_resids)] = revised_resids
        next_resnum = revised_resids[-1] + 1
        final_resids[len(revised_resids) :] = np.arange(
            next_resnum, total_residues - len(revised_resids) + next_resnum
        )
        final_system.residues.resids = final_resids
        if bool(getattr(sim, "infer_disulfide_bonds", True)):
            inferred_disulfide_pairs = _infer_cyx_disulfide_pairs_from_atoms(final_system)
            existing_disulfide_pairs = {tuple(sorted(pair)) for pair in disulfide_pairs}
            new_disulfide_pairs = [
                pair
                for pair in inferred_disulfide_pairs
                if tuple(sorted(pair)) not in existing_disulfide_pairs
            ]
            if new_disulfide_pairs:
                logger.info(
                    "Inferred additional CYX disulfide pair(s) from SG distances: {}. "
                    "Set create.infer_disulfide_bonds: false to disable this inference.",
                    ", ".join(
                        f"{first}-{second}" for first, second in new_disulfide_pairs
                    ),
                )
                disulfide_pairs = _merge_disulfide_pairs(
                    disulfide_pairs, new_disulfide_pairs
                )
                disulfide_resids = {resid for pair in disulfide_pairs for resid in pair}
        _mark_disulfide_residue_names(final_system.residues, disulfide_resids)

        # partitions
        final_system_dum = final_system.select_atoms("resname DUM")
        final_system_dum[0].position = final_system.select_atoms(PROTEIN_COM_ATOM_SELECTION).center_of_mass()
        ligand_residues = final_system.select_atoms(f"resname {mol}").residues
        if comp in {"z", "d"} and len(final_system_dum) > 1 and len(ligand_residues) > 1:
            final_system_dum[1].position = ligand_residues[1].atoms.center_of_mass()
        final_system_prot = final_system.select_atoms(_PROTEIN_WITH_TERMINAL_CAPS)
        final_system_others = final_system - final_system_prot - final_system_dum
        final_system_ligs = final_system.select_atoms(f"resname {mol}")
        final_system_other_mol = (
            final_system_others.select_atoms("not resname WAT") - final_system_ligs
        )
        if use_membrane_reference_box:
            final_system_water_notaround = None
            final_system_water_around = None
        else:
            final_system_water = final_system_others.select_atoms("resname WAT")
            final_system_water_notaround = final_system.select_atoms(
                f"byres (resname WAT and not (around 6 {_PROTEIN_WITH_TERMINAL_CAPS}))"
            )
            final_system_water_around = final_system_water - final_system_water_notaround

        # write parts
        _write_res_blocks(final_system_dum, window_dir / "solvate_pre_dum.pdb")

        # set chainIDs using renum_df and write protein by chains
        for residue, chain_id in zip(
            final_system_prot.residues,
            _renum_chain_ids_for_residues(final_system_prot.residues, renum_df),
        ):
            residue.atoms.chainIDs = chain_id
        _collapse_terminal_cap_resids_in_place(final_system_prot.residues)
        prot_lines = []
        for chain_name in np.unique(final_system_prot.atoms.chainIDs):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            final_system.select_atoms(f"chainID {chain_name}").write(tmp.name)
            tmp.close()
            with open(tmp.name) as f:
                prot_lines += [
                    ln
                    for ln in f
                    if ln.startswith("ATOM")
                    and not _is_disulfide_thiol_hydrogen_line(ln, disulfide_resids)
                ]
            prot_lines.append("TER\n")
        solvate_pre_prot = window_dir / "solvate_pre_prot.pdb"
        solvate_pre_prot.write_text("".join(prot_lines))
        ace_cap_count = _rewrite_ace_caps_for_leap(solvate_pre_prot)
        if ace_cap_count:
            logger.debug(
                "Rewrote {} ACE terminal cap(s) into Amber atom names before LEaP.",
                ace_cap_count,
            )
        cterm_o_count = _rewrite_cterminal_oxygen_aliases_for_leap(solvate_pre_prot)
        if cterm_o_count:
            logger.debug(
                "Rewrote {} C-terminal oxygen alias residue(s) before LEaP.",
                cterm_o_count,
            )
        cap_count = _rewrite_terminal_amide_caps_for_leap(solvate_pre_prot)
        if cap_count:
            logger.debug(
                "Rewrote {} terminal protein amide cap(s) as Amber NHE/NME residues before LEaP.",
                cap_count,
            )
        leap_disulfide_pairs = _map_disulfide_pairs_to_leap_indices(
            disulfide_pairs, solvate_pre_prot
        )

        _write_res_blocks(final_system_ligs, window_dir / "solvate_pre_ligands.pdb")

        other_lines_exist = len(final_system_other_mol.residues) != 0
        if other_lines_exist:
            _write_res_blocks(final_system_other_mol, window_dir / "solvate_pre_others.pdb")

        outside_wat_exist = (
            final_system_water_notaround is not None
            and len(final_system_water_notaround.residues) != 0
        )
        if outside_wat_exist:
            _write_res_blocks(
                final_system_water_notaround, window_dir / "solvate_pre_outside_wat.pdb"
            )

        around_wat_exist = (
            final_system_water_around is not None
            and len(final_system_water_around.residues) != 0
        )
        if around_wat_exist:
            _write_res_blocks(
                final_system_water_around, window_dir / "solvate_pre_around_water.pdb"
            )

    nonwater_pre_pdbs = [
        window_dir / "solvate_pre_dum.pdb",
        window_dir / "solvate_pre_prot.pdb",
        window_dir / "solvate_pre_ligands.pdb",
    ]
    if other_lines_exist:
        nonwater_pre_pdbs.append(window_dir / "solvate_pre_others.pdb")
    water_pre_pdbs = (
        list(water_chunk_paths)
        if use_membrane_reference_box
        else [
            path
            for path in (
                window_dir / "solvate_pre_outside_wat.pdb",
                window_dir / "solvate_pre_around_water.pdb",
            )
            if path.exists()
        ]
    )
    water_cleanup = _cleanup_periodic_water_pdbs(
        water_pre_pdbs,
        nonwater_pdbs=nonwater_pre_pdbs,
        box=system_dimensions,
        report_path=window_dir / "periodic_water_cleanup.json",
    )
    if use_membrane_reference_box:
        water_chunk_paths = [path for path in water_chunk_paths if path.exists()]
        if not water_chunk_paths:
            raise ValueError(
                f"All membrane water chunks were removed during periodic cleanup in {window_dir}."
            )

    # --- tleap parts (all with working_dir=window_dir) ---

    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_dum.in")
    with (window_dir / "tleap_solvate_dum.in").open("a") as f:
        f.write("dum = loadpdb solvate_pre_dum.pdb\n\n")
        f.write(
            f"set dum box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb dum solvate_dum.pdb\n")
        f.write("saveamberparm dum solvate_dum.prmtop solvate_dum.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_dum.in > tleap_dum.log", working_dir=window_dir
    )

    # prot
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_prot.in")
    with (window_dir / "tleap_solvate_prot.in").open("a") as f:
        f.write("prot = loadpdb solvate_pre_prot.pdb\n\n")
        _write_leap_disulfide_bonds(f, "prot", leap_disulfide_pairs)
        f.write(
            f"set prot box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb prot solvate_prot.pdb\n")
        f.write("saveamberparm prot solvate_prot.prmtop solvate_prot.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_prot.in > tleap_prot.log", working_dir=window_dir
    )

    # ligands
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_ligands.in")
    with (window_dir / "tleap_solvate_ligands.in").open("a") as f:
        f.write("# Load the necessary parameters\n")
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        if comp == "x":
            f.write(f"loadamberparams {mol}.frcmod\n")
            f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write("ligands = loadpdb solvate_pre_ligands.pdb\n\n")
        f.write(
            f"set ligands box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb ligands solvate_ligands.pdb\n")
        f.write(
            "saveamberparm ligands solvate_ligands.prmtop solvate_ligands.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_ligands.in > tleap_ligands.log",
        working_dir=window_dir,
    )

    # others
    if other_lines_exist:
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_others.in")
        with (window_dir / "tleap_solvate_others.in").open("a") as f:
            for om in other_mol:
                f.write(f"loadamberparams {om.lower()}.frcmod\n")
                f.write(f"{om} = loadmol2 {om.lower()}.mol2\n")
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("others = loadpdb solvate_pre_others.pdb\n\n")
            f.write(
                f"set others box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb others solvate_others.pdb\n")
            f.write(
                "saveamberparm others solvate_others.prmtop solvate_others.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_others.in > tleap_others.log",
            working_dir=window_dir,
        )
        _repair_lipid_hydrogens_after_tleap_lipids(window_dir)

    # charge accounting
    def _sum_unit_charge_from_log(logfile: Path) -> Tuple[int, int]:
        neu_cat = neu_ani = 0
        if not logfile.exists():
            return 0, 0
        for line in logfile.read_text().splitlines():
            if "The unperturbed charge of the unit" in line:
                q = float(line.split()[6].strip("'\",.:;#()]["))
                if q < 0:
                    neu_cat += round(float(re.sub(r"[+-]", "", str(q))))
                elif q > 0:
                    neu_ani += round(float(re.sub(r"[+-]", "", str(q))))
        return neu_cat, neu_ani

    neu_cat, neu_ani = _sum_unit_charge_from_log(window_dir / "tleap_prot.log")
    if (window_dir / "tleap_others.log").exists():
        nc2, na2 = _sum_unit_charge_from_log(window_dir / "tleap_others.log")
        neu_cat += nc2
        neu_ani += na2
    lig_charge = _ligand_charge_from_metadata(param_dir / f"{ctx.residue_name}.json")
    lig_cat = max(0, -lig_charge)
    lig_ani = max(0, lig_charge)

    charge_neut = neu_cat - neu_ani + lig_cat - lig_ani
    neu_cat = max(0, charge_neut)
    neu_ani = max(0, -charge_neut)

    box_volume = system_dimensions[0] * system_dimensions[1] * system_dimensions[2]
    num_ions = round(ion_def[2] * 6.02e23 * box_volume * 1e-27)
    # put a minimum of 5 ions
    num_ions = max(5, num_ions)
    if membrane_builder:
        num_ions //= 2
    num_cat = num_ions
    num_ani = num_ions - neu_cat + neu_ani
    if num_ani < 0:
        num_cat = neu_cat
        num_ions = neu_cat
        num_ani = 0

    water_part_prefixes: list[str] = []
    if use_membrane_reference_box:
        for chunk_index, chunk_path in enumerate(water_chunk_paths):
            unit_name = f"wat{chunk_index:02d}"
            prefix = f"solvate_wat_{chunk_index:02d}"
            water_part_prefixes.append(prefix)
            tleap_chunk = window_dir / f"tleap_{prefix}.in"
            _cp(window_dir / "tleap.in", tleap_chunk)
            with tleap_chunk.open("a") as f:
                if water_model != "TIP3PF":
                    f.write(f"source leaprc.water.{water_model.lower()}\n\n")
                else:
                    f.write("source leaprc.water.fb3\n\n")
                f.write(f"{unit_name} = loadpdb {chunk_path.name}\n\n")
                f.write(
                    f"set {unit_name} box "
                    f"{{{system_dimensions[0]:.6f} "
                    f"{system_dimensions[1]:.6f} "
                    f"{system_dimensions[2]:.6f}}}\n"
                )
                f.write(f"savepdb {unit_name} {prefix}.pdb\n")
                f.write(
                    f"saveamberparm {unit_name} {prefix}.prmtop {prefix}.inpcrd\nquit\n"
                )
            run_with_log(
                f"{tleap} -s -f {tleap_chunk.name} > tleap_{prefix}.log",
                working_dir=window_dir,
            )

    # outside water — ionization
    if (not use_membrane_reference_box) and (window_dir / "solvate_pre_outside_wat.pdb").exists():
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_outside_wat.in")
        with (window_dir / "tleap_solvate_outside_wat.in").open("a") as f:
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("outside_wat = loadpdb solvate_pre_outside_wat.pdb\n\n")
            if neut == "no":
                f.write(f"addionsrand outside_wat {ion_def[0]} {num_cat}\n")
                f.write(f"addionsrand outside_wat {ion_def[1]} {num_ani}\n")
            elif neut == "yes":
                if neu_cat:
                    f.write(f"addionsrand outside_wat {ion_def[0]} {neu_cat}\n")
                if neu_ani:
                    f.write(f"addionsrand outside_wat {ion_def[1]} {neu_ani}\n")
            f.write(
                f"set outside_wat box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb outside_wat solvate_outside_wat.pdb\n")
            f.write(
                "saveamberparm outside_wat solvate_outside_wat.prmtop solvate_outside_wat.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_outside_wat.in > tleap_outside_wat.log",
            working_dir=window_dir,
        )

    # around water
    if (not use_membrane_reference_box) and (window_dir / "solvate_pre_around_water.pdb").exists():
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_around_wat.in")
        with (window_dir / "tleap_solvate_around_wat.in").open("a") as f:
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("around_wat = loadpdb solvate_pre_around_water.pdb\n\n")
            f.write(
                f"set around_wat box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb around_wat solvate_around_wat.pdb\n")
            f.write(
                "saveamberparm around_wat solvate_around_wat.prmtop solvate_around_wat.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_around_wat.in > tleap_around_wat.log",
            working_dir=window_dir,
        )

    # combine with ParmEd
    dum_p = pmd.load_file(
        str(window_dir / "solvate_dum.prmtop"), str(window_dir / "solvate_dum.inpcrd")
    )
    prot_p = pmd.load_file(
        str(window_dir / "solvate_prot.prmtop"), str(window_dir / "solvate_prot.inpcrd")
    )
    ligand_p_1 = pmd.load_file(str(window_dir / f"{mol}.prmtop"))
    ligand_p_1.residues[0].name = mol
    ligand_p_1.save(str(window_dir / f"{mol}.prmtop"), overwrite=True)
    ligand_p_1 = pmd.load_file(str(window_dir / f"{mol}.prmtop"))

    lig_inp = pmd.load_file(str(window_dir / "solvate_ligands.inpcrd")).coordinates
    if dec_method == "dd" or comp in {"q", "l"}:
        ligands_p = ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp in ["z", "o", "s", "v"] and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp == "d" and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp in ["e"] and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1 + ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    else:
        raise ValueError(
            f"Unsupported comp={comp} with dec={dec_method} for custom ligand params."
        )

    combined = dum_p + prot_p + ligands_p
    vac = dum_p + prot_p + ligands_p
    other_parts = []

    if (window_dir / "solvate_others.prmtop").exists():
        others_p = pmd.load_file(
            str(window_dir / "solvate_others.prmtop"),
            str(window_dir / "solvate_others.inpcrd"),
        )
        combined += others_p
        other_parts.append(others_p)
    if use_membrane_reference_box:
        for prefix in water_part_prefixes:
            water_pmd = pmd.load_file(
                str(window_dir / f"{prefix}.prmtop"),
                str(window_dir / f"{prefix}.inpcrd"),
            )
            combined += water_pmd
            other_parts.append(water_pmd)
    elif (window_dir / "solvate_outside_wat.prmtop").exists():
        out_wat_pmd =  pmd.load_file(
            str(window_dir / "solvate_outside_wat.prmtop"),
            str(window_dir / "solvate_outside_wat.inpcrd"),
        )
        combined += out_wat_pmd
        other_parts.append(out_wat_pmd)
    if (not use_membrane_reference_box) and (window_dir / "solvate_around_wat.prmtop").exists():
        around_wat_pmd = pmd.load_file(
            str(window_dir / "solvate_around_wat.prmtop"),
            str(window_dir / "solvate_around_wat.inpcrd"),
        )
        combined += around_wat_pmd
        other_parts.append(around_wat_pmd)

    if not other_parts:
        raise ValueError("No non-vacuum solvent/other parts were generated.")
    other_parts_pmd = other_parts[0]
    for part in other_parts[1:]:
        other_parts_pmd = other_parts_pmd + part

    if comp == "q" and bool(getattr(sim, "fix_ring_penetration", True)):
        pre_repair_vac_coordinates = np.asarray(vac.coordinates, dtype=float).copy()
        pre_repair_combined_coordinates = np.asarray(
            combined.coordinates, dtype=float
        ).copy()
        repair_result = repair_ring_penetrations(
            vac,
            fix_mode=getattr(sim, "ring_penetration_fix_mode", "auto"),
            ligand_resname=mol,
            ligand_label=ligand,
        )
        if repair_result.initial_penetrations:
            pre_repair_files = _save_pre_ring_repair_snapshots(
                window_dir,
                vac=vac,
                vac_coordinates=pre_repair_vac_coordinates,
                combined=combined,
                combined_coordinates=pre_repair_combined_coordinates,
            )
            repair_metadata = repair_result.to_dict()
            repair_metadata["pre_repair_files"] = pre_repair_files
            (window_dir / "ring_penetration_repair.json").write_text(
                json.dumps(repair_metadata, indent=2, sort_keys=True)
            )
        if repair_result.repaired:
            combined_coordinates = np.asarray(combined.coordinates, dtype=float).copy()
            combined_coordinates[: len(vac.atoms)] = np.asarray(
                vac.coordinates, dtype=float
            )
            combined.coordinates = combined_coordinates

    if use_membrane_reference_box:
        reference_translation = _parmed_reference_translation(
            combined,
            window_dir / "equil-reference.pdb",
        )
        if reference_translation is not None:
            _translate_parmed_structure(combined, reference_translation)
            _translate_parmed_structure(vac, reference_translation)
            _translate_parmed_structure(other_parts_pmd, reference_translation)
            logger.debug(
                "Translated final {} system back to equil-reference frame by "
                "[{:.3f}, {:.3f}, {:.3f}] Å.",
                comp,
                float(reference_translation[0]),
                float(reference_translation[1]),
                float(reference_translation[2]),
            )

    if comp in _FE_NONWATER_VAC_COMPONENTS:
        vac, other_parts_pmd, combined = _split_structure_nonwater_then_water(combined)

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)
    _sync_ligand_anchor_residue_with_pdb(build_dir, window_dir / "vac.pdb", mol)
    if work != build_dir:
        _sync_ligand_anchor_residue_with_pdb(work, window_dir / "vac.pdb", mol)

    other_parts_pmd.save(str(window_dir / "other_parts.prmtop"), overwrite=True)
    other_parts_pmd.save(str(window_dir / "other_parts.inpcrd"), overwrite=True)
    other_parts_pmd.save(str(window_dir / "other_parts.pdb"), overwrite=True)

    u_full = mda.Universe(str(window_dir / "full.pdb"))
    u_vac = mda.Universe(str(window_dir / "vac.pdb"))

    # renumber protein residues back to original ids
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    renum_df2 = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df2["old_resname"] = renum_df2["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    renum_df2["new_resname"] = renum_df2["new_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    _restore_protein_resids_from_renum(u_full, renum_df2)
    _restore_protein_resids_from_renum(u_vac, renum_df2)

    # rebuild segments by chain
    seg_txt = window_dir / "build_amber_renum.txt"
    seg_df = pd.read_csv(
        seg_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    chain_list = renum_df2.old_chain.values
    chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
    for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
        res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))
    u_vac.atoms.write(str(window_dir / "vac_orig.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    # merge DUM + DUM + PROT plus all ligand copies before applying AMBER masks.
    merge_molecule_count = 7 if comp == "d" and dec_method == "sdr" else 6
    merge_first_n_and_lipid_fragments_in_prmtop(
        full_prmtop,
        merge_molecule_count,
        lipid_mol,
        str(window_dir / "full_merged.prmtop"),
    )
    return


@register_create_box("x")
def create_box_x(ctx: BuildContext) -> None:
    """
    Create the box for RBFE (x-component) ligand-pair systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir

    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    extra = ctx.extra or {}
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    res_ref = extra.get("residue_ref") or ctx.residue_name
    res_alt = extra.get("residue_alt")

    if not res_alt:
        raise ValueError(
            "RBFE component 'x' requires residue_alt in BuildContext.extra."
        )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        param_dir = work.parent.parent / "params"
        src = param_dir / f"{res_ref}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_x] Optional/absent: {src}")
        param_dir = work.parent.parent.parent / lig_alt / "params"
        src = param_dir / f"{res_alt}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_x] Optional/absent: {src}")

    membrane_builder = sim.membrane_simulation
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol
    
    # tleap template
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    _cp(src_tleap, window_dir / "tleap.in")

    # water box keyword
    water_model = str(sim.water_model).upper()

    if water_model == "TIP3PF":
        # still uses leaprc.water.fb3
        water_box = "FB3BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    else:
        water_box = f"{water_model}BOX"

    if water_model != "TIP3PF":
        water_line = f"source leaprc.water.{water_model.lower()}\n\n"
    else:
        water_line = "source leaprc.water.fb3\n\n"


    # combine with ParmEd
    vac_p = pmd.load_file(
        str(window_dir / "ref_vac.prmtop"), str(window_dir / "ref_vac.pdb")
    )
    other_part_p = pmd.load_file(
        str(window_dir / "other_parts.prmtop"),
        str(window_dir / "other_parts.pdb"),
    )
    ligand_alt = pmd.load_file(str(window_dir / f"{res_alt}.prmtop"))
    ligand_alt.residues[0].name = res_alt
    ligand_alt.save(str(window_dir / f"{res_alt}.prmtop"), overwrite=True)
    alter_ligands_p_site = pmd.load_file(
        str(window_dir / f"{res_alt}.prmtop"),
        str(window_dir / "alter_ligand_aligned_site.pdb"),
    )
    _repair_parmed_molecule_table_for_combine(alter_ligands_p_site)
    alter_ligands_p_solvent = pmd.load_file(
        str(window_dir / f"{res_alt}.prmtop"),
        str(window_dir / "alter_ligand_aligned_solvent.pdb"),
    )
    _repair_parmed_molecule_table_for_combine(alter_ligands_p_solvent)
    combined = vac_p + alter_ligands_p_site + alter_ligands_p_solvent + other_part_p

    # build the ion prmtop if exists
    if os.path.exists(window_dir / "ions.pdb"):
        tleap_ion_txt = (window_dir / "tleap.in").read_text().splitlines()
        tleap_ion_txt += [
            "# ion topology",
            water_line,
            f"ions = loadpdb ions.pdb",
            "saveamberparm ions ions.prmtop ions.inpcrd",
            "quit",
        ]
        _write(window_dir / "tleap_ions.in", "\n".join(tleap_ion_txt) + "\n")
        run_with_log(
            f"{tleap} -s -f tleap_ions.in > tleap_ions.log", working_dir=window_dir
        )
        ion_p = pmd.load_file(
            str(window_dir / "ions.prmtop"),
            str(window_dir / "ions.inpcrd"),
        )
        combined += ion_p

    vac, _, combined = _split_structure_nonwater_then_water(combined)

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    u_full = mda.Universe(str(window_dir / "full.pdb"))

    # renumber protein residues back to original ids
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    renum_df2 = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df2["old_resname"] = renum_df2["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    renum_df2["new_resname"] = renum_df2["new_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    _restore_protein_resids_from_renum(u_full, renum_df2)

    # rebuild segments by chain
    seg_txt = window_dir / "build_amber_renum.txt"
    seg_df = pd.read_csv(
        seg_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    chain_list = renum_df2.old_chain.values
    chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
    for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
        res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    merge_first_n_and_lipid_fragments_in_prmtop(
        full_prmtop,
        5,
        lipid_mol,
        str(window_dir / "full_merged.prmtop"),
    )

    # get mapping file

    mapping = json.load(open(window_dir / "mapping.json"))
    ref_site = u_full.select_atoms(f"resname {res_ref}").residues[0]
    ref_solvent = u_full.select_atoms(f"resname {res_ref}").residues[1]
    alt_site = u_full.select_atoms(f"resname {res_alt}").residues[0]
    alt_solvent = u_full.select_atoms(f"resname {res_alt}").residues[1]

    # select cc parts
    alt_index_list = [int(i) for i in mapping.keys()]
    ref_index_list = [int(i) for i in mapping.values()]
    cc_indices_site_t0 = ref_site.atoms[ref_index_list].indices + 1
    cc_indices_solvent_t0 = alt_solvent.atoms[alt_index_list].indices + 1
    cc_indices_solvent_t1 = ref_solvent.atoms[ref_index_list].indices + 1
    cc_indices_site_t1 = alt_site.atoms[alt_index_list].indices + 1
    all_indices_t0 = (
        np.concatenate((ref_site.atoms.indices, alt_solvent.atoms.indices)) + 1
    )
    all_indices_t1 = (
        np.concatenate((ref_solvent.atoms.indices, alt_site.atoms.indices)) + 1
    )

    dict_sc_mask = {
        "scmk1_all_indices": all_indices_t0.astype(int).tolist(),
        "scmk1_cc_site_indices": cc_indices_site_t0.astype(int).tolist(),
        "scmk1_cc_solvent_indices": cc_indices_solvent_t0.astype(int).tolist(),
        "scmk2_all_indices": all_indices_t1.astype(int).tolist(),
        "scmk2_cc_site_indices": cc_indices_site_t1.astype(int).tolist(),
        "scmk2_cc_solvent_indices": cc_indices_solvent_t1.astype(int).tolist(),
    }

    with open(window_dir / "scmask.json", "w") as f:
        json.dump(dict_sc_mask, f)

    return


@register_create_box("y")
def create_box_y(ctx: BuildContext) -> None:
    """
    Create the box for ligand-only (solvation FE) systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir
    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    mol = ctx.residue_name
    buffer_x = float(sim.buffer_x)
    buffer_y = float(sim.buffer_y)
    buffer_z = float(sim.buffer_z)
    if buffer_x < 10 or buffer_y < 10 or buffer_z < 10:
        raise ValueError(
            f"For water systems, buffer_x/y/z must be ≥ 10 Å; got {buffer_x}/{buffer_y}/{buffer_z}."
        )
    if not hasattr(sim, "water_model"):
        raise AttributeError("SimulationConfig missing 'water_model'.")
    water_model = str(sim.water_model).upper()

    if not hasattr(sim, "ion_def"):
        raise AttributeError("SimulationConfig missing 'ion_def'.")
    ion_def = sim.ion_def
    if len(ion_def) < 3:
        raise ValueError("`ion_def` must contain [cation, anion, concentration].")

    if not hasattr(sim, "neut"):
        raise AttributeError("SimulationConfig missing 'neut'.")
    neut = str(sim.neut).lower()

    comp = ctx.comp
    param_dir = (
        (work.parent.parent / "params") if comp != "q" else (work.parent / "params")
    )

    build_pdb = window_dir / "build.pdb"
    if not build_pdb.exists():
        fallback = build_dir / "build.pdb"
        if fallback.exists():
            _cp(fallback, build_pdb)
        else:
            raise FileNotFoundError(
                f"[create_box_y] build.pdb missing in {window_dir} (fallback: {fallback})."
            )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_y] Optional/absent: {src}")

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / f"vac_ligand.{ext}")

    # --- copy a base tleap template into window_dir ---
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    if not src_tleap.exists():
        raise FileNotFoundError(
            "No tleap template found (tleap.in[.amber16]) in amber_dir."
        )
    _cp(src_tleap, window_dir / "tleap.in")

    # --- build the vacuum unit from ligand PDB (vac.*) ---
    tleap_lig_txt = (window_dir / "tleap.in").read_text().splitlines()
    tleap_lig_txt += [
        "# ligand-only vacuum topology",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f'set {{{mol}.1}} name "{mol}"\n',
        f"lig = loadpdb {mol}.pdb",
        "desc lig",
        "savepdb lig vac.pdb",
        "saveamberparm lig vac.prmtop vac.inpcrd",
        "quit",
    ]
    _write(window_dir / "tleap_ligands.in", "\n".join(tleap_lig_txt) + "\n")
    run_with_log(
        f"{tleap} -s -f tleap_ligands.in > tleap_ligands.log", working_dir=window_dir
    )

    # --- determine water box keyword ---
    if water_model == "TIP3PF":
        water_box = "FB3BOX"  # leaprc.water.fb3
        water_leaprc = "leaprc.water.fb3"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
        water_leaprc = "leaprc.water.spce"
    else:
        water_box = f"{water_model}BOX"
        water_leaprc = f"leaprc.water.{water_model.lower()}"

    # --- read ligand net charge from tleap log (unperturbed unit charge line) ---
    def _unit_charge_from_log(logfile: Path) -> int:
        if not logfile.exists():
            return 0
        q = 0.0
        for ln in logfile.read_text().splitlines():
            if "The unperturbed charge of the unit" in ln:
                try:
                    q = float(ln.split()[6].strip("'\",.:;#()[]"))
                except Exception:
                    pass
        return int(round(q))

    lig_charge = _ligand_charge_from_metadata(param_dir / f"{ctx.residue_name}.json")
    # put a minimum of 5 ions
    box_volume_A3 = 2 * buffer_x * 2 * buffer_y * 2 * buffer_z
    num_ions = max(
        5,
        round(ion_def[2] * 6.02e23 * box_volume_A3 * 1e-27),
    )

    add_neu_cat = max(0, -lig_charge)
    add_neu_ani = max(0, lig_charge)

    tleap_solv_lines = (window_dir / "tleap.in").read_text().splitlines()
    tleap_solv_lines += [
        "# ligand-only solvation",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f"source {water_leaprc}",
        f'set {{{mol}.1}} name "{mol}"',
        f"model = loadpdb {build_pdb.name}",
        "",
        f"solvatebox model {water_box} {{ {buffer_x:.3f} {buffer_y:.3f} {buffer_z:.3f} }} 1",
        "",
        "# ions",
    ]
    if neut == "no":
        if num_ions > 0 or add_neu_cat > 0 or add_neu_ani > 0:
            tleap_solv_lines += [
                f"addionsrand model {ion_def[0]} {num_ions + add_neu_cat}",
                f"addionsrand model {ion_def[1]} {num_ions + add_neu_ani}",
            ]
    else:
        if add_neu_cat:
            tleap_solv_lines.append(f"addionsrand model {ion_def[0]} {add_neu_cat}")
        if add_neu_ani:
            tleap_solv_lines.append(f"addionsrand model {ion_def[1]} {add_neu_ani}")

    tleap_solv_lines += [
        "desc model",
        "savepdb model full_pre.pdb",
        "quit",
        "",
    ]
    _write(window_dir / "tleap_solvate.in", "\n".join(tleap_solv_lines))
    run_with_log(
        f"{tleap} -s -f tleap_solvate.in > tleap_solvate.log", working_dir=window_dir
    )
    _restore_existing_protons_from_reference(window_dir, window_dir / "full_pre.pdb")

    # --- process full_pre.pdb into final full.{prmtop,inpcrd,pdb} ---
    #
    u = mda.Universe(str(window_dir / "full_pre.pdb"))
    final_system = u.atoms
    system_dimensions = u.dimensions[:3]
    final_system_dum = final_system.select_atoms("resname DUM")
    final_system_lig = final_system.select_atoms(f"resname {mol}")
    final_system_others = final_system - final_system_dum - final_system_lig

    _write_res_blocks(final_system_dum, window_dir / "solvate_pre_dum.pdb")
    _write_res_blocks(final_system_lig, window_dir / "solvate_pre_lig.pdb")
    _write_res_blocks(final_system_others, window_dir / "solvate_pre_others.pdb")

    # tleap parts
    # dum
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_dum.in")
    with (window_dir / "tleap_solvate_dum.in").open("a") as f:
        f.write("dum = loadpdb solvate_pre_dum.pdb\n\n")
        f.write(
            f"set dum box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb dum solvate_dum.pdb\n")
        f.write("saveamberparm dum solvate_dum.prmtop solvate_dum.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_dum.in > tleap_dum.log", working_dir=window_dir
    )

    # ligand
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_lig.in")
    with (window_dir / "tleap_solvate_lig.in").open("a") as f:
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        f.write("lig = loadpdb solvate_pre_lig.pdb\n\n")
        f.write(
            f"set lig box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb lig solvate_ligands.pdb\n")
        f.write(
            "saveamberparm lig solvate_ligands.prmtop solvate_ligands.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_lig.in > tleap_lig.log", working_dir=window_dir
    )

    # others
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_others.in")
    with (window_dir / "tleap_solvate_others.in").open("a") as f:
        if water_model != "TIP3PF":
            f.write(f"source leaprc.water.{water_model.lower()}\n\n")
        else:
            f.write("source leaprc.water.fb3\n\n")
        f.write("others = loadpdb solvate_pre_others.pdb\n\n")
        f.write(
            f"set others box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb others solvate_others.pdb\n")
        f.write(
            "saveamberparm others solvate_others.prmtop solvate_others.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_others.in > tleap_others.log",
        working_dir=window_dir,
    )
    _repair_lipid_hydrogens_after_tleap_lipids(window_dir)

    # combine with ParmEd
    dum_p = pmd.load_file(
        str(window_dir / "solvate_dum.prmtop"), str(window_dir / "solvate_dum.inpcrd")
    )
    ligand_p = pmd.load_file(str(window_dir / f"{mol}.prmtop"))
    ligand_p.residues[0].name = mol
    lig_inp = pmd.load_file(str(window_dir / "solvate_ligands.inpcrd")).coordinates
    ligand_p.coordinates = lig_inp
    ligand_p.save(str(window_dir / f"{mol}.prmtop"), overwrite=True)

    others = pmd.load_file(
        str(window_dir / "solvate_others.prmtop"),
        str(window_dir / "solvate_others.inpcrd"),
    )
    combined = dum_p + ligand_p + others
    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac = dum_p + ligand_p
    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    return


@register_create_box("m")
def create_box_m(ctx: BuildContext) -> None:
    """
    Create the box for ligand-only (vacuum) systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir
    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    mol = ctx.residue_name
    
    comp = ctx.comp
    param_dir = (
        (work.parent.parent / "params") if comp != "q" else (work.parent / "params")
    )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_m] Optional/absent: {src}")

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / f"vac_ligand.{ext}")

    # --- copy a base tleap template into window_dir ---
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    if not src_tleap.exists():
        raise FileNotFoundError(
            "No tleap template found (tleap.in[.amber16]) in amber_dir."
        )
    _cp(src_tleap, window_dir / "tleap.in")

    # --- build the vacuum unit from ligand PDB (vac.*) ---
    tleap_lig_txt = (window_dir / "tleap.in").read_text().splitlines()
    tleap_lig_txt += [
        "# ligand-only vacuum topology",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f'set {{{mol}.1}} name "{mol}"\n',
        f"lig = loadpdb {mol}.pdb",
        # set box to 40
        "set lig box {40.000000 40.000000 40.000000}",
        "desc lig",
        "savepdb lig vac.pdb",
        "saveamberparm lig vac.prmtop vac.inpcrd",
        "quit",
    ]
    _write(window_dir / "tleap_ligands.in", "\n".join(tleap_lig_txt) + "\n")
    run_with_log(
        f"{tleap} -s -f tleap_ligands.in > tleap_ligands.log", working_dir=window_dir
    )

    # copy ligand_p to vac.prmtop
    ligand_p_file = window_dir / f"{mol}.prmtop"
    _cp(ligand_p_file, window_dir / "vac.prmtop")

    # copy vac to full
    _cp(window_dir / "vac.pdb", window_dir / "full.pdb")
    _cp(window_dir / "vac.prmtop", window_dir / "full.prmtop")
    _cp(window_dir / "vac.inpcrd", window_dir / "full.inpcrd")
    
    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    return
