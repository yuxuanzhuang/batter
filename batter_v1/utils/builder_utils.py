# batter/utils/anchors.py
from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from loguru import logger

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
    lig_sdf: Optional[str],
    anchor_atoms: Sequence[str],
    ligand_anchor_atom: Optional[str] = None,
) -> Tuple[float, float, float, str, str, str, float]:
    """
    Identify Boresch-style anchor atoms and pocket geometry.

    Parameters
    ----------
    u_prot
        Protein universe.
    u_lig
        Ligand universe.
    lig_sdf
        Path to the ligand SDF (used only to derive candidate atoms if
        ``ligand_anchor_atom`` is not provided). May be None.
    anchor_atoms
        Three MDAnalysis selection strings for receptor anchors (P1, P2, P3).
        Each selection must resolve to exactly one atom.
    ligand_anchor_atom
        Optional MDAnalysis selection for the ligand anchor atom. If omitted,
        candidates are derived from `lig_sdf` via RDKit; on failure, all ligand
        atoms are used.

    Returns
    -------
    (l1_x, l1_y, l1_z, p1_str, p2_str, p3_str, r_dist)
        Translation vector from P1 to ligand COM (x,y,z), formatted protein
        anchor specs (":RESID@NAME"), and the distance magnitude + 1.0 (Å).
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
            f"P1_atom.n_atoms={P1_atom.n_atoms}, P2_atom.n_atoms={P2_atom.n_atoms}, P3_atom.n_atoms={P3_atom.n_atoms}"
        )
    if P1_atom.n_atoms != 1 or P2_atom.n_atoms != 1 or P3_atom.n_atoms != 1:
        raise ValueError("More than one atom selected in the anchor atoms.")

    # Pick ligand anchor set
    if ligand_anchor_atom:
        lig_sel = u_merge.select_atoms(ligand_anchor_atom)
        if lig_sel.n_atoms == 0:
            logger.warning(
                "Provided ligand anchor atom '{}' not found; using all ligand atoms instead.",
                ligand_anchor_atom,
            )
            lig_sel = u_lig.atoms
    else:
        # Try RDKit candidate picking if SDF supplied; otherwise use all atoms
        lig_sel = u_lig.atoms
        if lig_sdf:
            try:
                candidates = get_ligand_candidates(lig_sdf)
                if candidates:
                    lig_sel = u_lig.atoms[candidates]
            except Exception as e:
                logger.warning(
                    "Could not derive ligand candidates from SDF '{}': {}. Using all ligand atoms.",
                    lig_sdf,
                    e,
                )

    # Vector from P1 to ligand COM
    r_vect = lig_sel.center_of_mass() - P1_atom.positions  # shape (1,3)
    l1_x, l1_y, l1_z = float(r_vect[0][0]), float(r_vect[0][1]), float(r_vect[0][2])
    logger.debug("l1_x={:.2f}; l1_y={:.2f}; l1_z={:.2f}", l1_x, l1_y, l1_z)

    p1_formatted = f":{P1_atom.resids[0]}@{P1_atom.names[0]}"
    p2_formatted = f":{P2_atom.resids[0]}@{P2_atom.names[0]}"
    p3_formatted = f":{P3_atom.resids[0]}@{P3_atom.names[0]}"
    logger.debug(
        "Receptor anchors: P1={}, P2={}, P3={}",
        p1_formatted, p2_formatted, p3_formatted
    )

    l1_range = float(np.linalg.norm(r_vect)) + 1.0
    return l1_x, l1_y, l1_z, p1_formatted, p2_formatted, p3_formatted, l1_range


def get_ligand_candidates(ligand_sdf: str, removeHs: bool = True) -> List[int]:
    """
    Candidate ligand atoms for Boresch restraints from an SDF.

    Heuristic (RXRX-style):
      - Non-hydrogen atoms
      - Not sp-hybridized carbons
      - Connected to at least two heavy neighbors

    If < 3 candidates are found, all non-hydrogen atoms are returned.

    Parameters
    ----------
    ligand_sdf
        Path to ligand SDF.
    removeHs
        Whether to remove hydrogens when reading (RDKit).

    Returns
    -------
    list[int]
        0-based atom indices in the ligand.
    """
    try:
        from rdkit import Chem
    except Exception as e:
        raise RuntimeError(
            "RDKit is required to derive ligand anchor candidates from SDF."
        ) from e

    supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=removeHs)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise ValueError(f"Failed to read any molecule from {ligand_sdf}")
    mol = mols[0]

    anchor_candidates: List[int] = []
    non_h_indices: List[int] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 1:  # skip H
            continue
        non_h_indices.append(idx)
        # avoid sp-carbon
        if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
            continue

        heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1)
        if heavy_neighbors >= 2:
            anchor_candidates.append(idx)

    if len(anchor_candidates) < 3:
        logger.warning(
            "Found < 3 ligand anchor candidates; using all non-H ligand atoms instead."
        )
        anchor_candidates = non_h_indices

    return anchor_candidates


def select_ions_away_from_complex(
    universe: mda.Universe,
    total_charge: int,
    mol: str,
) -> Optional[List[int]]:
    """
    Choose ion indices far from the complex to neutralize the system.

    Parameters
    ----------
    universe
        MDAnalysis Universe containing ions & complex.
    total_charge
        Net charge to neutralize (positive => add Cl-, negative => add Na+).
    mol
        Ligand residue name used in the complex selection.

    Returns
    -------
    list[int] | None
        Selected ion atom indices (0-based) or None if no ions needed.
    """
    if total_charge > 0:
        ion_type = "Cl-"
    elif total_charge < 0:
        ion_type = "Na+"
    else:
        return None

    n_ions = abs(total_charge)
    complex_sys = universe.select_atoms(f"protein or resname {mol} or name P31")
    ions = universe.select_atoms(f"resname {ion_type}")
    if len(ions) < n_ions:
        raise ValueError(f"Not enough {ion_type} ions in the system to neutralize the charge.")

    sel_indices: List[int] = []

    def _try_with_cutoff(cut: float, needed: int) -> int:
        remaining = needed
        for ion in ions:
            if ion.index in sel_indices:
                continue
            dmin = distance_array(ion.position, complex_sys.positions, box=universe.dimensions).min()
            if dmin > cut:
                sel_indices.append(ion.index)
                remaining -= 1
                if remaining == 0:
                    break
        return remaining

    # First pass: 15 Å cutoff
    n_left = _try_with_cutoff(15.0, n_ions)
    if n_left > 0:
        logger.warning(
            "Not enough {} ions beyond 15 Å; trying 10 Å instead.", ion_type
        )
        n_left = _try_with_cutoff(10.0, n_left)
        if n_left > 0:
            raise ValueError(
                f"Only found {len(sel_indices)} suitable {ion_type} ions beyond 10 Å from the complex."
            )

    return sel_indices


def get_buffer_z(protein_file: str, targeted_buf: float = 20.0) -> float:
    """
    Extra buffer_z needed (Å) to reach a target water layer thickness on both
    sides of the protein along Z.

    Parameters
    ----------
    protein_file
        Path to a PDB (or trajectory-supported) file containing protein + solvent.
    targeted_buf
        Desired minimum solvent thickness (Å) above and below the protein.

    Returns
    -------
    float
        Extra Å to add to buffer_z.
    """
    u = mda.Universe(protein_file)
    protein = u.select_atoms("protein")

    prot_z_min = protein.positions[:, 2].min()
    prot_z_max = protein.positions[:, 2].max()

    sys_z_min = u.atoms.positions[:, 2].min()
    sys_z_max = u.atoms.positions[:, 2].max()

    buffer_top = sys_z_max - prot_z_max
    buffer_bottom = prot_z_min - sys_z_min
    current_buffer = float(min(buffer_top, buffer_bottom))

    return float(max(0.0, targeted_buf - current_buffer))


def get_sdr_dist(
    protein_file: str,
    lig_resname: str,
    buffer_z: float,
    extra_buffer: float = 5.0,
) -> float:
    """
    Compute a Z-shift (Å) for the ligand to place it roughly mid-solvent.

    Parameters
    ----------
    protein_file
        Path to system PDB with protein + solvent.
    lig_resname
        Ligand residue name.
    buffer_z
        Target buffer thickness (Å) (per side).
    extra_buffer
        Additional offset beyond buffer_z (Å).

    Returns
    -------
    float
        Suggested positive shift along +Z for the ligand.
    """
    u = mda.Universe(protein_file)
    ligand = u.select_atoms(f"resname {lig_resname}")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand {lig_resname} not found in {protein_file}")

    # Exclude water and ions; case-insensitive variants are handled via multiple names
    prot_sel = "protein and not (resname WAT SPC TIP3 TIP3P TIP4P OPC NA+ Na+ K+ Cl-)"
    protein = u.select_atoms(prot_sel) if u.select_atoms(prot_sel).n_atoms else u.select_atoms("protein")

    prot_z_max = protein.positions[:, 2].max()
    prot_z_min = protein.positions[:, 2].min()

    # theoretical box edges if we expand by buffer_z
    system_z_max = prot_z_max + buffer_z
    system_z_min = prot_z_min - buffer_z

    lig_z = float(ligand.positions[:, 2].mean())
    targeted_lig_z = system_z_max + extra_buffer  # place above protein by buffer + margin
    sdr_dist = float(targeted_lig_z - lig_z)
    return sdr_dist