"""Helper utilities for system preparation internals.

This module centralizes frequently reused routines that operate on MDAnalysis
universes, RDKit molecules, or simple file artifacts produced during system
building.  Most helpers revolve around anchor detection, solvent handling,
and mask formatting for downstream AMBER tooling.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import json

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from loguru import logger

try:
    from rdkit import Chem
except Exception as e:  # pragma: no cover
    Chem = None  # type: ignore
    logger.warning("RDKit not available; get_ligand_candidates will fail if called. ({})", e)

__all__ = [
    "Anchors",
    "get_buffer_z",
    "get_sdr_dist",
    "get_ligand_candidates",
    "load_anchors",
    "num_to_mask",
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


def get_buffer_z(protein_file: str | Path, targeted_buf: float = 20.0) -> float:
    """Return the extra z-buffer needed to meet a target solvent thickness.

    Parameters
    ----------
    protein_file : str or Path
        Path to the receptor-only structure (PDB, GRO, etc.).
    targeted_buf : float, default 20.0
        Desired minimal water thickness (Å) above and below the protein.

    Returns
    -------
    float
        Additional buffer (Å) to add in both z directions.
    """
    u = mda.Universe(str(protein_file))
    protein = u.select_atoms("protein")
    prot_z_min = protein.positions[:, 2].min()
    prot_z_max = protein.positions[:, 2].max()

    sys_z_min = u.atoms.positions[:, 2].min()
    sys_z_max = u.atoms.positions[:, 2].max()

    buffer_top = sys_z_max - prot_z_max
    buffer_bottom = prot_z_min - sys_z_min
    current_buffer = min(buffer_top, buffer_bottom)

    required_extra = max(0.0, targeted_buf - current_buffer)
    return float(required_extra)


def get_sdr_dist(
    protein_file: str | Path,
    lig_resname: str,
    buffer_z: float,
    extra_buffer: float = 5.0,
) -> float:
    """Compute the ligand z-translation that centers it within the solvent.

    Parameters
    ----------
    protein_file : str or Path
        Path to the receptor structure (with ligand coordinates present).
    lig_resname : str
        Residue name used to select the ligand atoms.
    buffer_z : float
        Pre-computed buffer (Å) to maintain between protein surface and solvent.
    extra_buffer : float, default 5.0
        Additional spacing (Å) to keep the ligand slightly above the protein.

    Returns
    -------
    float
        Translation distance (Å) to add to ligand z coordinates.
    """
    u = mda.Universe(str(protein_file))
    ligand = u.select_atoms(f"resname {lig_resname}")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand {lig_resname} not found in {protein_file}")

    protein_ns = u.select_atoms("protein and not resname WAT Na+ Cl-")
    prot_z_max = protein_ns.positions[:, 2].max()
    prot_z_min = protein_ns.positions[:, 2].min()

    # target is a bit above the top protein surface
    targeted_lig_z = prot_z_max + buffer_z + float(extra_buffer)
    lig_z = float(ligand.positions[:, 2].mean())
    sdr_dist = targeted_lig_z - lig_z
    return float(sdr_dist)


def get_ligand_candidates(ligand_sdf: str | Path, removeHs: bool = True) -> List[int]:
    """Return ligand atom indices suitable for anchor selection.

    Criteria:

    * heavy atoms bound to at least two other heavy atoms (to avoid terminal atoms)
    * sp-hybridized carbons are skipped
    * if fewer than three candidates remain, all heavy atoms are used

    Parameters
    ----------
    ligand_sdf : str or Path
        Path to the ligand SDF file.
    removeHs : bool, default True
        Whether RDKit should remove hydrogens while reading the SDF.

    Returns
    -------
    list[int]
        Zero-based atom indices within the RDKit molecule.

    Raises
    ------
    RuntimeError
        If RDKit is unavailable.
    ValueError
        If the SDF file cannot be read.
    """
    if Chem is None:
        raise RuntimeError("RDKit is required for get_ligand_candidates but is not available.")

    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=removeHs)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise ValueError(f"Could not read ligand SDF: {ligand_sdf}")
    mol = mols[0]

    anchor_candidates: List[int] = []
    non_h: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue  # skip H
        # avoid sp carbons
        if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
            continue
        heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1)
        if heavy_neighbors >= 2:
            anchor_candidates.append(atom.GetIdx())
        non_h.append(atom.GetIdx())

    if len(anchor_candidates) < 3:
        logger.warning(
            "Fewer than three candidate ligand anchors found; using all non-H atoms instead."
        )
        anchor_candidates = non_h
    return anchor_candidates


def select_ions_away_from_complex(
    u: mda.Universe,
    total_charge: int,
    lig_resname: str,
) -> Optional[List[int]]:
    """Select ion indices that neutralize the system while staying distant.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing the solvated system.
    total_charge : int
        Total system charge to neutralize; positive selects Na+, negative Cl-.
    lig_resname : str
        Residue name used for ligand selection when defining the complex.

    Returns
    -------
    list[int] or None
        Zero-based atom indices of chosen ions, or ``None`` when no ions are required.

    Raises
    ------
    ValueError
        If insufficient ions are available or none satisfy the distance criteria.
    """
    if total_charge == 0:
        return None

    ion_type = "Na+" if total_charge > 0 else "Cl-"
    n_needed = abs(int(total_charge))

    complex_sel = u.select_atoms(f"protein or resname {lig_resname} or name P31")
    ions = u.select_atoms(f"resname {ion_type}")
    if len(ions) < n_needed:
        raise ValueError(f"Not enough {ion_type} ions to neutralize: need {n_needed}, have {len(ions)}.")

    chosen: List[int] = []

    def _pick(min_dist: float, remaining: int) -> int:
        for ion in ions:
            if ion.index in chosen:
                continue
            dmin = float(distance_array(ion.position, complex_sel.positions, box=u.dimensions).min())
            if dmin > min_dist:
                chosen.append(ion.index)
                remaining -= 1
                if remaining == 0:
                    break
        return remaining

    remaining = _pick(15.0, n_needed)
    if remaining > 0:
        logger.warning(
            f"Not enough {ion_type} ions found ≥15 Å from complex; relaxing to 10 Å."
        )
        remaining = _pick(10.0, remaining)

    if remaining > 0:
        raise ValueError(
            f"Insufficient {ion_type} ions ≥10 Å from complex. "
            f"Found {len(chosen)} / required {n_needed}."
        )
    return chosen


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
