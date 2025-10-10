# batter/_internal/ops/helpers.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from loguru import logger

try:
    from rdkit import Chem
except Exception as e:  # pragma: no cover
    Chem = None  # type: ignore
    logger.warning("RDKit not available; get_ligand_candidates will fail if called. ({})", e)

__all__ = [
    "get_buffer_z",
    "get_sdr_dist",
    "get_ligand_candidates",
    "select_ions_away_from_complex",
]


def get_buffer_z(protein_file: str | Path, targeted_buf: float = 20.0) -> float:
    """
    Extra z-buffer (Å) required to reach ``targeted_buf`` water thickness on BOTH
    sides of the protein along z.
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
    """
    Compute a vertical (z) shift that places the ligand mid-solvent above the protein.
    Returns the distance (Å) to add to the ligand z coordinates.
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
    """
    Candidate atoms for Boresch restraints:
    non-H atoms bonded to >= 2 heavy atoms; if <3 found, return all non-H atoms.

    Returns 0-based atom indices in the RDKit molecule.
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


def select_ions_away_from_complex(u: mda.Universe, total_charge: int, lig_resname: str) -> Optional[List[int]]:
    """
    Pick ion indices (Na+ or Cl-) at least ~15 Å from the complex (protein + ligand + P31).
    Falls back to 10 Å if needed; raises if still insufficient.
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
    """
    Build a list mapping atom numbers to Amber-style masks (':resid@atomname').

    The first entry is a dummy `0` to align 1-based atom numbering with indices.
    So `atm_num[i]` corresponds to atom i in the PDB file.

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