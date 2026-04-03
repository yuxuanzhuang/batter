"""Helper utilities for system preparation internals.

This module centralizes frequently reused routines that operate on MDAnalysis
universes, RDKit molecules, or simple file artifacts produced during system
building.  Most helpers revolve around anchor detection, solvent handling,
and mask formatting for downstream AMBER tooling.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List
import json
import shutil

from loguru import logger

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
]


PROTEIN_COM_ATOM_SELECTION = "protein and name CA"


@dataclass(frozen=True)
class Anchors:
    """Atom masks that define the three protein and ligand anchor atoms."""

    P1: str
    P2: str
    P3: str
    L1: str
    L2: str
    L3: str
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
    """Rewrite run-script PRMTOP references for HMR on/off modes."""
    if hmr:
        return text.replace("full.prmtop", "full.hmr.prmtop")
    return text.replace("full.hmr.prmtop", "full.prmtop")


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
    run_with_log(
        "parmed -O -n -i parmed-hmr.in > parmed-hmr.log",
        working_dir=window_dir,
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

    def find_flag_index(flag_name: str) -> int:
        target = f"%FLAG {flag_name}"
        for i, line in enumerate(lines):
            if line.strip() == target:
                return i
        raise ValueError(f"Could not find section {target}")

    def get_section_data_range(flag_idx: int) -> tuple[int, int]:
        """
        Return [start, end) line indices of data lines for a FLAG section,
        excluding %FLAG and %FORMAT lines.
        """
        if flag_idx + 1 >= len(lines) or not lines[flag_idx + 1].startswith("%FORMAT"):
            raise ValueError(f"Missing %FORMAT line after {lines[flag_idx]}")
        start = flag_idx + 2
        end = start
        while end < len(lines) and not lines[end].startswith("%FLAG"):
            end += 1
        return start, end

    def parse_fixed_width_ints(section_lines: List[str], width: int = 8) -> List[int]:
        values = []
        for line in section_lines:
            # Amber prmtop integer sections are fixed-width, not whitespace-delimited.
            for i in range(0, len(line), width):
                chunk = line[i:i + width]
                if chunk.strip():
                    values.append(int(chunk))
        return values

    def format_fixed_width_ints(values: List[int], per_line: int, width: int = 8) -> List[str]:
        out = []
        for i in range(0, len(values), per_line):
            chunk = values[i:i + per_line]
            out.append("".join(f"{v:{width}d}" for v in chunk))
        return out

    # --- Modify ATOMS_PER_MOLECULE ---
    apm_flag_idx = find_flag_index("ATOMS_PER_MOLECULE")
    apm_start, apm_end = get_section_data_range(apm_flag_idx)
    apm_values = parse_fixed_width_ints(lines[apm_start:apm_end], width=8)

    if n < 1:
        raise ValueError("n must be >= 1")
    if n > len(apm_values):
        raise ValueError(
            f"n={n} is larger than the number of molecules in ATOMS_PER_MOLECULE "
            f"({len(apm_values)})"
        )

    merged_atoms = sum(apm_values[:n])
    new_apm_values = [merged_atoms] + apm_values[n:]
    new_apm_lines = format_fixed_width_ints(new_apm_values, per_line=10, width=8)

    # Replace section data lines
    lines[apm_start:apm_end] = new_apm_lines

    # Because line counts may have changed, refind SOLVENT_POINTERS after replacement
    sp_flag_idx = find_flag_index("SOLVENT_POINTERS")
    sp_start, sp_end = get_section_data_range(sp_flag_idx)
    sp_values = parse_fixed_width_ints(lines[sp_start:sp_end], width=8)

    if len(sp_values) < 3:
        raise ValueError("SOLVENT_POINTERS section must contain at least 3 integers")

    # Reduce 2nd and 3rd integers by (n - 1)
    decrement = n - 1
    sp_values[1] -= decrement
    sp_values[2] -= decrement

    new_sp_lines = format_fixed_width_ints(sp_values, per_line=3, width=8)
    lines[sp_start:sp_end] = new_sp_lines

    Path(output_path).write_text("\n".join(lines) + "\n")
    return output_path
