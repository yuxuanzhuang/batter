"""Helper utilities for system preparation internals.

This module centralizes frequently reused routines that operate on MDAnalysis
universes, RDKit molecules, or simple file artifacts produced during system
building.  Most helpers revolve around anchor detection, solvent handling,
and mask formatting for downstream AMBER tooling.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
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
