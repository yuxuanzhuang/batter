"""Helper utilities for system preparation internals.

This module centralizes frequently reused routines that operate on MDAnalysis
universes, RDKit molecules, or simple file artifacts produced during system
building.  Most helpers revolve around anchor detection, solvent handling,
and mask formatting for downstream AMBER tooling.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import csv
import json
import os
import shutil
import shlex
import sys

from loguru import logger

from batter.data import charmmlipid2amber as charmmlipid2amber_csv
from batter._internal.parmed_compat import bundled_parmed_path
from batter.systemprep import (
    get_buffer_z,
    get_ligand_candidates,
    get_sdr_dist,
    select_ions_away_from_complex,
)
from batter.utils import run_with_log

__all__ = [
    "Anchors",
    "PROTEIN_COM_ATOM_SELECTION",
    "get_buffer_z",
    "get_sdr_dist",
    "get_ligand_candidates",
    "load_anchors",
    "num_to_mask",
    "copy_if_exists",
    "field_slice",
    "is_atom_line",
    "rewrite_prmtop_reference",
    "run_parmed_hmr_if_enabled",
    "save_anchors",
    "select_ions_away_from_complex",
    "amber_lipid_fragment_patterns",
    "merge_first_n_and_lipid_fragments_in_prmtop",
    "merge_first_n_molecules_in_prmtop",
    "revised_resids_for_lipid_fragments",
]


PROTEIN_COM_ATOM_SELECTION = "protein and name CA"


@dataclass(frozen=True)
class Anchors:
    """Atom masks that define protein anchors and optional ligand anchors."""

    P1: str
    P2: str
    P3: str
    L1: str | None
    L2: str | None
    L3: str | None
    lig_res: str

def _anchors_path(working_dir: Path) -> Path:
    """Return the canonical on-disk location for anchor metadata."""
    return working_dir / "anchors.json"

def save_anchors(working_dir: Path, anchors: Anchors) -> None:
    """Persist anchor metadata to ``anchors.json`` under ``working_dir``."""
    p = _anchors_path(working_dir)
    p.write_text(json.dumps(asdict(anchors), indent=2))
    logger.debug(f"[simprep] wrote anchors → {p}")

def load_anchors(working_dir: Path) -> Anchors:
    """Load and deserialize previously stored anchor masks."""
    p = _anchors_path(working_dir)
    data = json.loads(p.read_text())
    return Anchors(**data)


def num_to_mask(pdb_file: str | Path) -> list[str]:
    """Map PDB atom indices to Amber-style mask strings.

    The first entry is a dummy ``"0"`` to align with 1-based indexing so that
    ``atm_num[i]`` corresponds to atom ``i`` in the source file.

    Parameters
    ----------
    pdb_file : str or Path
        Path to the PDB file to read.

    Returns
    -------
    list[str]
        Mask strings aligned with atom indices (1-based).
    """
    pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    atm_num: list[str] = ["0"]  # align with Amber 1-based numbering
    with pdb_file.open() as f:
        for line in f:
            rec = line[0:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue
            atom_name = line[12:16].strip()
            resid = line[22:26].strip()
            atm_num.append(f":{resid}@{atom_name}")
    return atm_num


def is_atom_line(line: str) -> bool:
    """Return True when a PDB line is an ATOM/HETATM record."""
    tag = line[0:6].strip()
    return tag in {"ATOM", "HETATM"}


def field_slice(line: str, start: int, end: int) -> str:
    """Extract a fixed-width PDB-style field (0-based, end-exclusive)."""
    return line[start:end].strip()


def copy_if_exists(src: Path, dst: Path, *, on_missing: str = "warn") -> bool:
    """Copy ``src`` to ``dst`` when present.

    Parameters
    ----------
    on_missing
        ``"warn"`` to log a warning and continue, or ``"raise"`` to raise.
    """
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    if on_missing == "raise":
        raise FileNotFoundError(f"Missing required file: {src}")
    logger.warning(f"Expected file not found: {src} (continuing)")
    return False


def rewrite_prmtop_reference(text: str, *, hmr: bool) -> str:
    """Normalize generated run-script PRMTOP references to the merged topology."""
    _ = hmr
    return (
        text.replace("full.hmr.prmtop", "full_merged.prmtop")
        .replace("full.prmtop", "full_merged.prmtop")
    )


def run_parmed_hmr_if_enabled(sim_hmr: str | bool, amber_dir: Path, window_dir: Path) -> None:
    """Run parmed HMR conversion if enabled by the simulation config."""
    hmr = str(sim_hmr).lower() == "yes" if not isinstance(sim_hmr, bool) else sim_hmr
    if not hmr:
        logger.debug("[box] HMR disabled; skipping parmed-hmr.")
        return
    parmed_hmr = amber_dir / "parmed-hmr.in"
    if not parmed_hmr.exists():
        logger.warning("[box] parmed-hmr.in not found in amber_dir; skipping HMR.")
        return
    shutil.copy2(parmed_hmr, window_dir / "parmed-hmr.in")
    parmed_exe = shutil.which("parmed")
    env = None
    if parmed_exe:
        command = f"{shlex.quote(parmed_exe)} -O -n -i parmed-hmr.in > parmed-hmr.log"
    else:
        bundled = bundled_parmed_path()
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env = {
            "PYTHONPATH": (
                str(bundled)
                if not existing_pythonpath
                else f"{bundled}{os.pathsep}{existing_pythonpath}"
            )
        }
        logger.warning(
            "[box] parmed executable not found; using bundled ParmEd Python entrypoint for HMR."
        )
        command = (
            f"{shlex.quote(sys.executable)} -c "
            f"{shlex.quote('from parmed.scripts import clapp; clapp()')} "
            "-O -n -i parmed-hmr.in > parmed-hmr.log"
        )
    run_with_log(
        command,
        working_dir=window_dir,
        env=env,
    )


def format_ranges(numbers: Iterable[int]) -> str:
    """Compact integer sequences into comma-delimited ranges.

    Parameters
    ----------
    numbers : Iterable[int]
        Integer values (typically atom numbers) to compress.

    Returns
    -------
    str
        Comma-separated range specification (e.g., ``"1-3,5-6"``).
    """
    from itertools import groupby
    numbers = sorted(set(numbers))
    ranges = []

    for _, group in groupby(enumerate(numbers), key=lambda x: x[1] - x[0]):
        group = list(group)
        start = group[0][1]
        end = group[-1][1]
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
    
    return ",".join(ranges)


def _find_prmtop_flag_index(lines: list[str], flag_name: str) -> int:
    target = f"%FLAG {flag_name}"
    for i, line in enumerate(lines):
        if line.strip() == target:
            return i
    raise ValueError(f"Could not find section {target}")


def _prmtop_section_data_range(lines: list[str], flag_idx: int) -> tuple[int, int]:
    if flag_idx + 1 >= len(lines) or not lines[flag_idx + 1].startswith("%FORMAT"):
        raise ValueError(f"Missing %FORMAT line after {lines[flag_idx]}")
    start = flag_idx + 2
    end = start
    while end < len(lines) and not lines[end].startswith("%FLAG"):
        end += 1
    return start, end


def _parse_fixed_width_ints(section_lines: list[str], width: int = 8) -> list[int]:
    values = []
    for line in section_lines:
        for i in range(0, len(line), width):
            chunk = line[i : i + width]
            if chunk.strip():
                values.append(int(chunk))
    return values


def _format_fixed_width_ints(values: list[int], per_line: int, width: int = 8) -> list[str]:
    out = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        out.append("".join(f"{v:{width}d}" for v in chunk))
    return out


def _parse_fixed_width_strings(section_lines: list[str], width: int = 4) -> list[str]:
    values = []
    for line in section_lines:
        for i in range(0, len(line), width):
            chunk = line[i : i + width]
            if chunk:
                values.append(chunk.strip())
    return values


def _prmtop_int_section(lines: list[str], flag_name: str) -> list[int]:
    flag_idx = _find_prmtop_flag_index(lines, flag_name)
    start, end = _prmtop_section_data_range(lines, flag_idx)
    return _parse_fixed_width_ints(lines[start:end], width=8)


def _prmtop_string_section(lines: list[str], flag_name: str) -> list[str]:
    flag_idx = _find_prmtop_flag_index(lines, flag_name)
    start, end = _prmtop_section_data_range(lines, flag_idx)
    return _parse_fixed_width_strings(lines[start:end], width=4)


@lru_cache(maxsize=1)
def _known_amber_lipid_fragment_patterns() -> tuple[tuple[str, tuple[str, ...]], ...]:
    patterns: list[tuple[str, tuple[str, ...]]] = []
    rows_by_source: dict[str, list[tuple[int, str]]] = {}
    try:
        with open(charmmlipid2amber_csv, newline="") as handle:
            next(handle, None)
            reader = csv.DictReader(handle)
            for row in reader:
                source = str(row.get("residue", "")).strip().upper()
                replace = str(row.get("replace", "")).split()
                if not source or not replace:
                    continue
                try:
                    order = int(row.get("order", 0))
                except (TypeError, ValueError):
                    order = 0
                rows_by_source.setdefault(source, []).append((order, replace[-1].upper()))
    except OSError:
        rows_by_source = {}

    for source, rows in rows_by_source.items():
        sequence: list[str] = []
        for _order, resname in sorted(rows):
            if resname and (not sequence or sequence[-1] != resname):
                sequence.append(resname)
        if len(sequence) > 1:
            patterns.append((source, tuple(sequence)))

    fallback = (
        ("POPC", ("PA", "PC", "OL")),
        ("DPPC", ("PA", "PC", "PA")),
        ("DOPC", ("OL", "PC", "OL")),
    )
    for item in fallback:
        if item not in patterns:
            patterns.append(item)
    return tuple(patterns)


def amber_lipid_fragment_patterns(lipid_mol: Iterable[str] | None) -> tuple[tuple[str, ...], ...]:
    """Return Amber lipid fragment sequences relevant to configured lipid names."""
    requested = {str(name).strip().upper() for name in (lipid_mol or []) if str(name).strip()}
    if not requested:
        return ()

    selected: list[tuple[str, ...]] = []
    for source, pattern in _known_amber_lipid_fragment_patterns():
        pattern_set = set(pattern)
        if source in requested or pattern_set == requested:
            selected.append(pattern)

    unique: list[tuple[str, ...]] = []
    for pattern in sorted(selected, key=lambda item: (-len(item), item)):
        if pattern not in unique:
            unique.append(pattern)
    return tuple(unique)


def revised_resids_for_lipid_fragments(
    residue_records: Iterable[tuple[str, str, int]],
    lipid_mol: Iterable[str] | None,
) -> list[int]:
    """Assign one residue id to consecutive Amber fragments from one lipid."""
    records = [
        (str(resname).strip().upper(), str(chain).strip(), int(resid))
        for resname, chain, resid in residue_records
    ]
    patterns = amber_lipid_fragment_patterns(lipid_mol)
    requested = {str(name).strip().upper() for name in (lipid_mol or []) if str(name).strip()}
    fragment_names = set(requested)
    for pattern in patterns:
        fragment_names.update(pattern)

    revised: list[int] = []
    counter = 1
    i = 0
    while i < len(records):
        matched: tuple[str, ...] | None = None
        for pattern in patterns:
            stop = i + len(pattern)
            if stop > len(records):
                continue
            names = tuple(record[0] for record in records[i:stop])
            chains = {record[1] for record in records[i:stop]}
            if names == pattern and len(chains) == 1:
                matched = pattern
                break
        if matched is not None:
            revised.extend([counter] * len(matched))
            counter += 1
            i += len(matched)
            continue

        resname, chain, resid = records[i]
        if (
            revised
            and resname in fragment_names
            and records[i - 1][0] in fragment_names
            and chain == records[i - 1][1]
            and resid == records[i - 1][2]
        ):
            revised.append(revised[-1])
        else:
            revised.append(counter)
            counter += 1
        i += 1
    return revised


def _molecule_residue_label_groups(
    atoms_per_molecule: list[int],
    residue_labels: list[str],
    residue_pointers: list[int],
) -> list[tuple[str, ...]]:
    total_atoms = sum(atoms_per_molecule)
    residue_starts = [int(pointer) - 1 for pointer in residue_pointers]
    residue_ends = residue_starts[1:] + [total_atoms]
    groups: list[tuple[str, ...]] = []
    residue_index = 0
    atom_start = 0
    for atom_count in atoms_per_molecule:
        atom_end = atom_start + atom_count
        while residue_index + 1 < len(residue_ends) and residue_ends[residue_index] <= atom_start:
            residue_index += 1
        first_residue = residue_index
        last_residue = first_residue
        while last_residue + 1 < len(residue_starts) and residue_starts[last_residue + 1] < atom_end:
            last_residue += 1
        groups.append(tuple(residue_labels[first_residue : last_residue + 1]))
        atom_start = atom_end
    return groups


def _lipid_fragment_molecule_spans(
    lines: list[str],
    atoms_per_molecule: list[int],
    lipid_mol: Iterable[str] | None,
) -> list[tuple[int, int]]:
    patterns = amber_lipid_fragment_patterns(lipid_mol)
    if not patterns:
        return []
    residue_labels = [label.upper() for label in _prmtop_string_section(lines, "RESIDUE_LABEL")]
    residue_pointers = _prmtop_int_section(lines, "RESIDUE_POINTER")
    molecule_groups = _molecule_residue_label_groups(
        atoms_per_molecule,
        residue_labels,
        residue_pointers,
    )

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(molecule_groups):
        matched: tuple[str, ...] | None = None
        for pattern in patterns:
            stop = i + len(pattern)
            if stop > len(molecule_groups):
                continue
            groups = molecule_groups[i:stop]
            if all(len(group) == 1 for group in groups) and tuple(
                group[0] for group in groups
            ) == pattern:
                matched = pattern
                break
        if matched is None:
            i += 1
            continue
        spans.append((i, i + len(matched)))
        i += len(matched)
    return spans


def _merge_molecule_spans_in_prmtop(
    prmtop_path: str | Path,
    spans: Iterable[tuple[int, int]],
    output_path: str | Path,
) -> str:
    prmtop_path = Path(prmtop_path)
    output_path = Path(output_path)
    lines = prmtop_path.read_text().splitlines()

    apm_flag_idx = _find_prmtop_flag_index(lines, "ATOMS_PER_MOLECULE")
    apm_start, apm_end = _prmtop_section_data_range(lines, apm_flag_idx)
    apm_values = _parse_fixed_width_ints(lines[apm_start:apm_end], width=8)

    normalized_spans = []
    for start, stop in sorted(spans):
        if start < 0 or stop > len(apm_values) or start >= stop:
            raise ValueError(
                f"Invalid molecule span ({start}, {stop}) for {len(apm_values)} molecules"
            )
        if normalized_spans and start < normalized_spans[-1][1]:
            raise ValueError("Molecule merge spans must not overlap")
        normalized_spans.append((start, stop))

    new_apm_values: list[int] = []
    removed_molecules = 0
    cursor = 0
    for start, stop in normalized_spans:
        new_apm_values.extend(apm_values[cursor:start])
        if stop - start == 1:
            new_apm_values.append(apm_values[start])
        else:
            new_apm_values.append(sum(apm_values[start:stop]))
            removed_molecules += stop - start - 1
        cursor = stop
    new_apm_values.extend(apm_values[cursor:])

    lines[apm_start:apm_end] = _format_fixed_width_ints(new_apm_values, per_line=10, width=8)

    sp_flag_idx = _find_prmtop_flag_index(lines, "SOLVENT_POINTERS")
    sp_start, sp_end = _prmtop_section_data_range(lines, sp_flag_idx)
    sp_values = _parse_fixed_width_ints(lines[sp_start:sp_end], width=8)
    if len(sp_values) < 3:
        raise ValueError("SOLVENT_POINTERS section must contain at least 3 integers")
    sp_values[1] -= removed_molecules
    sp_values[2] -= removed_molecules
    lines[sp_start:sp_end] = _format_fixed_width_ints(sp_values, per_line=3, width=8)

    output_path.write_text("\n".join(lines) + "\n")
    return output_path.as_posix()


def merge_first_n_molecules_in_prmtop(prmtop_path: str, n: int, output_path: str | None = None) -> str:
    """
    Modify an AMBER prmtop file by:
      1) Merging the first n molecules in %FLAG ATOMS_PER_MOLECULE
         by replacing the first n entries with their sum.
      2) Modifying %FLAG SOLVENT_POINTERS by reducing the 2nd and 3rd
         integers by (n - 1).

    Parameters
    ----------
    prmtop_path : str
        Path to input prmtop file.
    n : int
        Number of first molecules to merge.
    output_path : str | None
        Path to write the modified prmtop. If None, writes to
        "<original_stem>_merged.prmtop".

    Returns
    -------
    str
        Path to the written output prmtop.

    Notes
    -----
    Assumptions:
    - %FLAG ATOMS_PER_MOLECULE uses integer format like %FORMAT(10I8)
    - %FLAG SOLVENT_POINTERS uses %FORMAT(3I8)
    - The function preserves the original section order and rewrites
      only these two sections using the same fixed-width formatting.
    """
    prmtop_path = Path(prmtop_path)
    lines = prmtop_path.read_text().splitlines()
    apm_values = _prmtop_int_section(lines, "ATOMS_PER_MOLECULE")

    if n < 1:
        raise ValueError("n must be >= 1")
    if n > len(apm_values):
        raise ValueError(
            f"n={n} is larger than the number of molecules in ATOMS_PER_MOLECULE "
            f"({len(apm_values)})"
        )

    return _merge_molecule_spans_in_prmtop(prmtop_path, [(0, n)], output_path)


def merge_first_n_and_lipid_fragments_in_prmtop(
    prmtop_path: str,
    n: int,
    lipid_mol: Iterable[str] | None,
    output_path: str | None = None,
) -> str:
    prmtop_path_obj = Path(prmtop_path)
    if output_path is None:
        output_path = prmtop_path_obj.with_name(f"{prmtop_path_obj.stem}_merged.prmtop").as_posix()

    lines = prmtop_path_obj.read_text().splitlines()
    apm_values = _prmtop_int_section(lines, "ATOMS_PER_MOLECULE")
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > len(apm_values):
        raise ValueError(
            f"n={n} is larger than the number of molecules in ATOMS_PER_MOLECULE "
            f"({len(apm_values)})"
        )

    spans = [(0, n)]
    lipid_spans = [
        span for span in _lipid_fragment_molecule_spans(lines, apm_values, lipid_mol)
        if span[1] <= 0 or span[0] >= n
    ]
    spans.extend(lipid_spans)
    return _merge_molecule_spans_in_prmtop(prmtop_path_obj, spans, output_path)
