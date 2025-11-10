from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array
from loguru import logger

try:
    from rdkit import Chem
except Exception as e:  # pragma: no cover - RDKit optional at runtime
    Chem = None  # type: ignore
    logger.warning("RDKit not available; ligand helpers will fail if invoked. (%s)", e)

__all__ = [
    "find_anchor_atoms",
    "get_ligand_candidates",
    "select_ions_away_from_complex",
    "get_buffer_z",
    "get_sdr_dist",
]


def find_anchor_atoms(
    u_prot: mda.Universe,
    u_lig: mda.Universe,
    lig_sdf: Optional[str | Path],
    anchor_atoms: Sequence[str],
    ligand_anchor_atom: Optional[str] = None,
) -> Tuple[float, float, float, str, str, str, float]:
    """
    Identify Boresch-style anchor atoms and pocket geometry.

    Returns
    -------
    (l1_x, l1_y, l1_z, p1_str, p2_str, p3_str, r_dist)
        Translation vector from P1 to ligand COM (Å), formatted protein anchor
        strings (``:RESID@NAME``), and the ligand distance magnitude + 1 Å.
    """
    if len(anchor_atoms) != 3:
        raise ValueError("anchor_atoms must contain exactly 3 selection strings.")

    u_merge = mda.Merge(u_prot.atoms, u_lig.atoms)

    P1_atom = u_merge.select_atoms(anchor_atoms[0])
    P2_atom = u_merge.select_atoms(anchor_atoms[1])
    P3_atom = u_merge.select_atoms(anchor_atoms[2])

    if P1_atom.n_atoms == 0 or P2_atom.n_atoms == 0 or P3_atom.n_atoms == 0:
        raise ValueError(
            "Anchor atom not found with the provided selection string.\n"
            f"p1: {anchor_atoms[0]}, p2: {anchor_atoms[1]}, p3: {anchor_atoms[2]}\n"
            f"P1_atom.n_atoms={P1_atom.n_atoms}, "
            f"P2_atom.n_atoms={P2_atom.n_atoms}, "
            f"P3_atom.n_atoms={P3_atom.n_atoms}"
        )
    if P1_atom.n_atoms != 1 or P2_atom.n_atoms != 1 or P3_atom.n_atoms != 1:
        raise ValueError("More than one atom selected in the anchor atoms.")

    if ligand_anchor_atom:
        lig_sel = u_merge.select_atoms(ligand_anchor_atom)
        if lig_sel.n_atoms == 0:
            logger.warning(
                "Provided ligand anchor atom '%s' not found; using all ligand atoms instead.",
                ligand_anchor_atom,
            )
            lig_sel = u_lig.atoms
    else:
        lig_sel = u_lig.atoms
        if lig_sdf:
            try:
                candidates = get_ligand_candidates(lig_sdf)
                if candidates:
                    lig_sel = u_lig.atoms[candidates]
            except Exception as e:  # pragma: no cover - RDKit optional
                logger.warning(
                    "Could not derive ligand candidates from SDF '%s': %s. Using all ligand atoms.",
                    lig_sdf,
                    e,
                )

    r_vect = lig_sel.center_of_mass() - P1_atom.positions  # shape (1,3)
    l1_x, l1_y, l1_z = float(r_vect[0][0]), float(r_vect[0][1]), float(r_vect[0][2])
    logger.debug("l1_x=%.2f; l1_y=%.2f; l1_z=%.2f", l1_x, l1_y, l1_z)

    p1_formatted = f":{P1_atom.resids[0]}@{P1_atom.names[0]}"
    p2_formatted = f":{P2_atom.resids[0]}@{P2_atom.names[0]}"
    p3_formatted = f":{P3_atom.resids[0]}@{P3_atom.names[0]}"
    logger.debug(
        "Receptor anchors: P1=%s, P2=%s, P3=%s",
        p1_formatted,
        p2_formatted,
        p3_formatted,
    )

    l1_range = float(np.linalg.norm(r_vect)) + 1.0
    return l1_x, l1_y, l1_z, p1_formatted, p2_formatted, p3_formatted, l1_range


def get_ligand_candidates(ligand_sdf: str | Path, removeHs: bool = True) -> List[int]:
    """
    Return ligand atom indices suitable for anchor selection.

    Criteria: heavy atoms bound to at least two other heavy atoms while skipping
    hydrogens and sp-hybridised carbons. Falls back to all non-hydrogen atoms
    when fewer than three candidates survive.
    """
    if Chem is None:
        raise RuntimeError("RDKit is required for get_ligand_candidates.")

    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=removeHs)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise ValueError(f"Could not read ligand SDF: {ligand_sdf}")
    mol = mols[0]

    anchor_candidates: List[int] = []
    non_h: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue  # skip hydrogens
        non_h.append(atom.GetIdx())
        if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
            continue
        heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1)
        if heavy_neighbors >= 2:
            anchor_candidates.append(atom.GetIdx())

    if len(anchor_candidates) < 3:
        logger.warning(
            "Fewer than three candidate ligand anchors found; using all non-H atoms instead."
        )
        anchor_candidates = non_h
    return anchor_candidates


def select_ions_away_from_complex(
    universe: mda.Universe,
    total_charge: int,
    lig_resname: str,
) -> Optional[List[int]]:
    """
    Choose ion indices far from the complex to neutralize the system.

    Returns zero-based atom indices (or ``None`` if no ions are required).
    """
    if total_charge == 0:
        return None

    ion_type = "Na+" if total_charge > 0 else "Cl-"
    n_needed = abs(int(total_charge))

    complex_sel = universe.select_atoms(f"protein or resname {lig_resname} or name P31")
    ions = universe.select_atoms(f"resname {ion_type}")
    if len(ions) < n_needed:
        raise ValueError(
            f"Not enough {ion_type} ions to neutralize: need {n_needed}, have {len(ions)}."
        )

    chosen: List[int] = []

    def _pick(min_dist: float, remaining: int) -> int:
        for ion in ions:
            if ion.index in chosen:
                continue
            dmin = float(
                distance_array(ion.position, complex_sel.positions, box=universe.dimensions).min()
            )
            if dmin > min_dist:
                chosen.append(ion.index)
                remaining -= 1
                if remaining == 0:
                    break
        return remaining

    remaining = _pick(15.0, n_needed)
    if remaining > 0:
        logger.warning(
            "Not enough %s ions ≥15 Å from complex; relaxing to 10 Å.", ion_type
        )
        remaining = _pick(10.0, remaining)

    if remaining > 0:
        raise ValueError(
            f"Insufficient {ion_type} ions ≥10 Å from complex. "
            f"Found {len(chosen)} / required {n_needed}."
        )
    return chosen


def get_buffer_z(protein_file: str | Path, targeted_buf: float = 20.0) -> float:
    """
    Extra buffer (Å) needed along Z to maintain ``targeted_buf`` water thickness.
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
    Compute a Z-shift (Å) for the ligand to place it roughly mid-solvent.
    """
    u = mda.Universe(str(protein_file))
    ligand = u.select_atoms(f"resname {lig_resname}")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand {lig_resname} not found in {protein_file}")

    prot_sel = "protein and not (resname WAT SPC TIP3 TIP3P TIP4P OPC NA+ Na+ K+ Cl-)"
    protein = (
        u.select_atoms(prot_sel) if u.select_atoms(prot_sel).n_atoms else u.select_atoms("protein")
    )

    prot_z_max = protein.positions[:, 2].max()
    prot_z_min = protein.positions[:, 2].min()

    system_z_max = prot_z_max + buffer_z
    system_z_min = prot_z_min - buffer_z

    lig_z = float(ligand.positions[:, 2].mean())
    targeted_lig_z = system_z_max + float(extra_buffer)
    sdr_dist = targeted_lig_z - lig_z
    return float(sdr_dist)
