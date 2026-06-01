from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array
from loguru import logger

try:
    from rdkit import Chem
except Exception as e:  # pragma: no cover - RDKit optional at runtime
    Chem = None  # type: ignore
    logger.warning(f"RDKit not available; ligand helpers will fail if invoked. ({e})")

__all__ = [
    "find_anchor_atoms",
    "get_ligand_candidates",
    "select_apo_receptor_anchor_atoms",
    "select_receptor_anchor_atoms",
    "select_ions_away_from_complex",
    "get_buffer_z",
    "get_sdr_dist",
]

_BACKBONE_ANCHOR_NAMES = ("CA", "C", "N")
_STABLE_DSSP_CODES = {"H", "E"}
_LIGAND_POLAR_ATOMIC_NUMBERS = {7, 8, 9, 15, 16, 17, 35, 53}
_SALT_BRIDGE_CUTOFF = 4.0
_HBOND_CUTOFF = 3.5
_POLAR_CONTACT_CUTOFF = 4.0
_PROTEIN_POSITIVE_ATOMS = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("LYS", "NZ"),
    ("HIP", "ND1"),
    ("HIP", "NE2"),
}
_PROTEIN_NEGATIVE_ATOMS = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
}
_PROTEIN_HBOND_DONOR_NAMES = {
    "N",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "OG",
    "OG1",
    "OH",
    "SG",
}
_PROTEIN_HBOND_ACCEPTOR_NAMES = {
    "O",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
}


@dataclass(frozen=True)
class _LigandInteractionSite:
    position: np.ndarray
    charge: int = 0
    donor: bool = False
    acceptor: bool = False
    polar: bool = False


def select_receptor_anchor_atoms(
    u_prot: mda.Universe,
    u_lig: mda.Universe,
    lig_sdf: Optional[str | Path] = None,
    *,
    protein_dssp: Any = None,
    host_min_distance: float = 5.0,
    host_max_distance: float = 15.0,
    min_anchor_distance: float = 8.0,
    target_angle: float = 90.0,
    max_candidates: int = 120,
    max_p1_candidates: int = 30,
) -> List[str]:
    """
    Automatically choose receptor P1/P2/P3 anchor selections.

    The workflow mirrors the restraint-search heuristics used in OpenFE in a
    static BATTER setup context: prefer stable backbone atoms, search near the
    first ligand, enforce the tutorial's P1-P2/P2-P3 distance and angle rules,
    and use the residue closest to ligand interaction atoms as the P1 anchor.

    Reference: OpenFE's Boresch restraint host-anchor workflow in
    ``openfe/protocols/restraint_utils/geometry/boresch/geometry.py`` and
    ``host.py``. This is a BATTER-specific implementation of the same selection
    criteria, not vendored OpenFE code.
    """
    if host_min_distance < 0 or host_max_distance <= host_min_distance:
        raise ValueError("host_min_distance must be >= 0 and below host_max_distance.")
    if min_anchor_distance <= 0:
        raise ValueError("min_anchor_distance must be positive.")

    base_candidates = _protein_backbone_anchor_candidates(u_prot)
    if base_candidates.n_atoms < 3:
        raise ValueError(
            "Could not auto-select receptor anchors: fewer than three protein "
            "backbone candidate atoms (CA/C/N) were found."
        )

    ligand_reference = _ligand_reference_atoms(u_lig, lig_sdf)
    ligand_interactions = _ligand_interaction_sites(u_lig, lig_sdf, ligand_reference)

    candidate_tiers = [
        (
            "DSSP stable secondary structure",
            _dssp_filtered_candidates(base_candidates, protein_dssp),
        ),
        ("trimmed protein chains", _chain_trimmed_candidates(base_candidates)),
        ("all protein backbone candidates", base_candidates),
    ]

    search_windows = [
        (host_min_distance, host_max_distance, False),
        (3.0, 20.0, True),
    ]
    for min_dist, max_dist, relaxed in search_windows:
        for tier_name, tier_candidates in candidate_tiers:
            if tier_candidates.n_atoms < 3:
                continue
            nearby = _distance_filtered_candidates(
                tier_candidates,
                ligand_reference,
                min_distance=min_dist,
                max_distance=max_dist,
            )
            if nearby.n_atoms < 3:
                continue
            anchors = _score_anchor_triplets(
                nearby,
                ligand_reference,
                ligand_interactions,
                min_anchor_distance=min_anchor_distance,
                target_angle=target_angle,
                max_candidates=max_candidates,
                max_p1_candidates=max_p1_candidates,
            )
            if anchors is None:
                continue
            if relaxed:
                logger.warning(
                    "Auto receptor-anchor selection used a relaxed host-ligand "
                    "distance window {:.1f}-{:.1f} A after the default {:.1f}-{:.1f} A "
                    "window produced no valid triplet.",
                    min_dist,
                    max_dist,
                    host_min_distance,
                    host_max_distance,
                )
            selections = [_atom_selection(atom, u_prot) for atom in anchors]
            logger.info(
                "Auto-selected receptor anchors from {}: P1={}, P2={}, P3={}",
                tier_name,
                selections[0],
                selections[1],
                selections[2],
            )
            return selections

    raise ValueError(
        "Could not auto-select receptor anchors satisfying BATTER tutorial "
        f"criteria: backbone atoms near the first ligand, P1-P2 and P2-P3 >= "
        f"{min_anchor_distance:.1f} A, and angle near {target_angle:.0f} degrees. "
        "Provide create.anchor_atoms manually or inspect the first ligand pose."
    )


def select_apo_receptor_anchor_atoms(
    u_prot: mda.Universe,
    *,
    protein_dssp: Any = None,
    min_anchor_distance: float = 8.0,
    target_angle: float = 90.0,
    max_candidates: int = 120,
    max_p1_candidates: int = 30,
) -> List[str]:
    """
    Automatically choose receptor P1/P2/P3 anchors without using a ligand pose.

    Apo MD runs only need stable, non-degenerate receptor anchors to keep the
    existing BATTER equilibration machinery wired. Unlike
    :func:`select_receptor_anchor_atoms`, this selector deliberately ignores the
    ligand position and scores protein backbone triplets around the protein core,
    preferring the usual BATTER geometry without making it a hard requirement.
    """
    if min_anchor_distance <= 0:
        raise ValueError("min_anchor_distance must be positive.")

    base_candidates = _protein_backbone_anchor_candidates(u_prot)
    if base_candidates.n_atoms < 3:
        raise ValueError(
            "Could not auto-select apo receptor anchors: fewer than three protein "
            "backbone candidate atoms (CA/C/N) were found."
        )

    candidate_tiers = [
        (
            "DSSP stable secondary structure",
            _dssp_filtered_candidates(base_candidates, protein_dssp),
        ),
        ("trimmed protein chains", _chain_trimmed_candidates(base_candidates)),
        ("all protein backbone candidates", base_candidates),
    ]
    for tier_name, tier_candidates in candidate_tiers:
        if tier_candidates.n_atoms < 3:
            continue
        anchors = _score_apo_anchor_triplets(
            tier_candidates,
            min_anchor_distance=min_anchor_distance,
            target_angle=target_angle,
            max_candidates=max_candidates,
            max_p1_candidates=max_p1_candidates,
        )
        if anchors is None:
            continue
        selections = [_atom_selection(atom, u_prot) for atom in anchors]
        logger.info(
            "Auto-selected apo receptor anchors from {}: P1={}, P2={}, P3={}",
            tier_name,
            selections[0],
            selections[1],
            selections[2],
        )
        return selections

    raise ValueError(
        "Could not auto-select apo receptor anchors: fewer than three distinct, "
        "non-degenerate protein backbone residues were usable. Provide "
        "create.anchor_atoms manually."
    )


def _protein_backbone_anchor_candidates(u_prot: mda.Universe) -> mda.AtomGroup:
    name_clause = " ".join(_BACKBONE_ANCHOR_NAMES)
    candidates = u_prot.select_atoms(
        f"protein and not resname NMA ACE and name {name_clause}"
    )
    if candidates.n_atoms >= 3:
        return _receptor_like_anchor_candidates(candidates)
    return _receptor_like_anchor_candidates(
        u_prot.select_atoms(f"protein and name {name_clause}")
    )


def _receptor_like_anchor_candidates(
    candidates: mda.AtomGroup,
    *,
    min_group_residues: int = 40,
    min_largest_fraction: float = 0.5,
) -> mda.AtomGroup:
    """
    Exclude short protein chains from automatic receptor-anchor selection.

    Some inputs keep a peptide binder or other short protein chain in
    ``protein_input`` before the receptor. Treating every protein chain as an
    anchor source can place apo dummy ligands next to the peptide instead of the
    receptor, inflating the staging box. Keep receptor-sized chains/fragments,
    but leave small single-chain test systems untouched.
    """
    if candidates.n_atoms < 3:
        return candidates

    grouped: dict[tuple[str, str], list[Any]] = {}
    for residue in candidates.residues:
        atoms = residue.atoms
        segid = str(atoms.segids[0]).strip() if hasattr(atoms, "segids") else ""
        try:
            chain_id = str(atoms.chainIDs[0]).strip()
        except Exception:
            chain_id = ""
        grouped.setdefault((segid, chain_id), []).append(residue)

    if len(grouped) <= 1:
        return candidates

    max_len = max(len(residues) for residues in grouped.values())
    threshold = max(
        int(min_group_residues), int(np.ceil(max_len * min_largest_fraction))
    )
    kept_groups = {
        key: residues for key, residues in grouped.items() if len(residues) >= threshold
    }
    if not kept_groups or len(kept_groups) == len(grouped):
        return candidates

    allowed_resindices = {
        int(residue.resindex)
        for residues in kept_groups.values()
        for residue in residues
    }
    allowed_atoms = candidates.universe.residues[sorted(allowed_resindices)].atoms
    filtered = candidates.intersection(allowed_atoms)
    if filtered.n_atoms < 3:
        return candidates

    skipped = [
        f"segid={segid or '?'} chain={chain_id or '?'} residues={len(residues)}"
        for (segid, chain_id), residues in sorted(grouped.items())
        if (segid, chain_id) not in kept_groups
    ]
    if skipped:
        logger.info(
            "Ignoring short protein chain(s) for receptor-anchor auto-selection: {}",
            "; ".join(skipped),
        )
    return filtered


def _one_backbone_atom_per_residue(candidates: mda.AtomGroup) -> list[Any]:
    """Pick one backbone anchor candidate per residue, preferring CA atoms."""
    atoms: list[Any] = []
    for residue in candidates.residues:
        residue_candidates = candidates.intersection(residue.atoms)
        ca_atoms = residue_candidates.select_atoms("name CA")
        if ca_atoms.n_atoms:
            atoms.append(ca_atoms[0])
        elif residue_candidates.n_atoms:
            atoms.append(residue_candidates[0])
    return atoms


def _score_apo_anchor_triplets(
    candidates: mda.AtomGroup,
    *,
    min_anchor_distance: float,
    target_angle: float,
    max_candidates: int,
    max_p1_candidates: int,
) -> list[Any] | None:
    atoms = _one_backbone_atom_per_residue(candidates)
    if len(atoms) < 3:
        return None

    protein_center = candidates.center_of_geometry()
    atom_records = [
        {
            "atom": atom,
            "center_distance": float(np.linalg.norm(atom.position - protein_center)),
        }
        for atom in atoms
    ]
    atom_records.sort(key=lambda item: (item["center_distance"], item["atom"].index))
    limited_records = atom_records[: max(3, int(max_candidates))]
    p1_records = atom_records[: max(1, int(max_p1_candidates))]

    best_score: float | None = None
    best_atoms: list[Any] | None = None
    for p1_record in p1_records:
        p1 = p1_record["atom"]
        for p2_record in limited_records:
            p2 = p2_record["atom"]
            if p2.residue.ix == p1.residue.ix:
                continue
            d12 = float(np.linalg.norm(p1.position - p2.position))
            for p3_record in limited_records:
                p3 = p3_record["atom"]
                if p3.residue.ix in {p1.residue.ix, p2.residue.ix}:
                    continue
                d23 = float(np.linalg.norm(p2.position - p3.position))
                angle = _angle_degrees(p1.position, p2.position, p3.position)
                if angle is None:
                    continue
                distance_shortfall = max(0.0, min_anchor_distance - d12) + max(
                    0.0,
                    min_anchor_distance - d23,
                )
                score = (
                    abs(angle - target_angle) / target_angle
                    + 0.10 * distance_shortfall
                    + 0.01
                    * (
                        p1_record["center_distance"]
                        + p2_record["center_distance"]
                        + p3_record["center_distance"]
                    )
                    + 0.01
                    * (
                        abs(d12 - min_anchor_distance)
                        + abs(d23 - min_anchor_distance)
                    )
                    + (0.0 if p1.name == "CA" else 0.05)
                    + (0.0 if p2.name == "CA" else 0.05)
                    + (0.0 if p3.name == "CA" else 0.05)
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_atoms = [p1, p2, p3]

    return best_atoms


def _valid_ligand_indices(u_lig: mda.Universe, indices: Sequence[int]) -> list[int]:
    n_atoms = u_lig.atoms.n_atoms
    return [int(idx) for idx in indices if 0 <= int(idx) < n_atoms]


def _ligand_reference_atoms(
    u_lig: mda.Universe,
    lig_sdf: Optional[str | Path],
) -> mda.AtomGroup:
    if lig_sdf:
        try:
            indices = _valid_ligand_indices(u_lig, get_ligand_candidates(lig_sdf))
            if indices:
                return u_lig.atoms[indices]
        except Exception as exc:  # pragma: no cover - RDKit optional
            logger.warning(
                "Could not derive ligand anchor candidates from '{}': {}. "
                "Using non-hydrogen ligand atoms for receptor-anchor auto-selection.",
                lig_sdf,
                exc,
            )

    non_h = u_lig.select_atoms("not name H*")
    return non_h if non_h.n_atoms else u_lig.atoms


def _ligand_interaction_sites(
    u_lig: mda.Universe,
    lig_sdf: Optional[str | Path],
    fallback: mda.AtomGroup,
) -> list[_LigandInteractionSite]:
    if Chem is None or not lig_sdf:
        return [_site_from_ligand_atom(atom) for atom in fallback]

    try:
        supplier = Chem.SDMolSupplier(str(lig_sdf), removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
        if not mols:
            return [_site_from_ligand_atom(atom) for atom in fallback]
        mol = mols[0]
        sites = []
        for rd_atom in mol.GetAtoms():
            idx = rd_atom.GetIdx()
            if rd_atom.GetAtomicNum() == 1 or not _valid_ligand_indices(u_lig, [idx]):
                continue
            charge = int(rd_atom.GetFormalCharge())
            donor = _rdkit_atom_is_hbond_donor(rd_atom)
            acceptor = _rdkit_atom_is_hbond_acceptor(rd_atom)
            polar = (
                charge != 0
                or donor
                or acceptor
                or rd_atom.GetAtomicNum() in _LIGAND_POLAR_ATOMIC_NUMBERS
            )
            if polar:
                sites.append(
                    _LigandInteractionSite(
                        position=np.asarray(u_lig.atoms[idx].position, dtype=float),
                        charge=charge,
                        donor=donor,
                        acceptor=acceptor,
                        polar=polar,
                    )
                )
        if sites:
            return sites
    except Exception as exc:  # pragma: no cover - RDKit optional
        logger.debug(
            "Could not derive ligand interaction atoms from '{}': {}", lig_sdf, exc
        )
    return [_site_from_ligand_atom(atom) for atom in fallback]


def _rdkit_atom_is_hbond_donor(atom: Any) -> bool:
    if atom.GetFormalCharge() > 0 and atom.GetAtomicNum() in {7, 8, 16}:
        return True
    if atom.GetAtomicNum() not in {7, 8, 16}:
        return False
    return atom.GetTotalNumHs(includeNeighbors=True) > 0


def _rdkit_atom_is_hbond_acceptor(atom: Any) -> bool:
    if atom.GetFormalCharge() > 0:
        return False
    if atom.GetAtomicNum() == 8:
        return True
    if atom.GetAtomicNum() == 7:
        return atom.GetTotalDegree() < 4
    if atom.GetAtomicNum() == 16:
        return True
    return False


def _site_from_ligand_atom(atom: Any) -> _LigandInteractionSite:
    element = _atom_element(atom)
    polar = element in {"N", "O", "P", "S", "F", "CL", "BR", "I"}
    return _LigandInteractionSite(
        position=np.asarray(atom.position, dtype=float),
        charge=0,
        donor=element in {"N", "O", "S"},
        acceptor=element in {"N", "O", "S"},
        polar=polar,
    )


def _dssp_filtered_candidates(
    candidates: mda.AtomGroup,
    protein_dssp: Any,
    *,
    min_structure_size: int = 6,
    trim_structure_ends: int = 2,
) -> mda.AtomGroup:
    empty = candidates.atoms[[]]
    if protein_dssp is None:
        return empty
    dssp = np.asarray(protein_dssp)
    if dssp.size == 0:
        return empty
    if dssp.ndim > 1:
        dssp = dssp[-1]

    protein_residues = candidates.universe.select_atoms(
        "protein and not resname NMA ACE"
    ).residues
    n_residues = min(len(protein_residues), len(dssp))
    if n_residues == 0:
        return empty

    allowed_resindices: set[int] = set()
    indexed_codes = [(idx, str(code)) for idx, code in enumerate(dssp[:n_residues])]
    for code, group_iter in groupby(indexed_codes, key=lambda item: item[1]):
        group = [idx for idx, _ in group_iter]
        if code not in _STABLE_DSSP_CODES or len(group) < min_structure_size:
            continue
        trimmed = group[trim_structure_ends:-trim_structure_ends]
        allowed_resindices.update(
            int(protein_residues[idx].resindex) for idx in trimmed
        )

    if not allowed_resindices:
        return empty
    allowed_atoms = candidates.universe.residues[sorted(allowed_resindices)].atoms
    return candidates.intersection(allowed_atoms)


def _chain_trimmed_candidates(
    candidates: mda.AtomGroup,
    *,
    min_chain_length: int = 30,
    trim_chain_start: int = 10,
    trim_chain_end: int = 10,
) -> mda.AtomGroup:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for residue in candidates.residues:
        atoms = residue.atoms
        segid = str(atoms.segids[0]).strip() if hasattr(atoms, "segids") else ""
        try:
            chain_id = str(atoms.chainIDs[0]).strip()
        except Exception:
            chain_id = ""
        grouped.setdefault((segid, chain_id), []).append(residue)

    allowed_resindices: set[int] = set()
    for residues in grouped.values():
        residues = sorted(
            residues, key=lambda residue: (int(residue.resid), residue.ix)
        )
        if len(residues) < min_chain_length:
            continue
        trimmed = residues[trim_chain_start:-trim_chain_end]
        allowed_resindices.update(int(residue.resindex) for residue in trimmed)

    if not allowed_resindices:
        return candidates.atoms[[]]
    allowed_atoms = candidates.universe.residues[sorted(allowed_resindices)].atoms
    return candidates.intersection(allowed_atoms)


def _distance_filtered_candidates(
    candidates: mda.AtomGroup,
    ligand_reference: mda.AtomGroup,
    *,
    min_distance: float,
    max_distance: float,
) -> mda.AtomGroup:
    ligand_center = ligand_reference.center_of_geometry()
    distances = distance_array(
        np.asarray([ligand_center], dtype=float),
        candidates.positions,
        box=candidates.universe.dimensions,
    )[0]
    mask = (distances >= float(min_distance)) & (distances <= float(max_distance))
    return candidates.atoms[mask]


def _angle_degrees(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float | None:
    v1 = p1 - p2
    v2 = p3 - p2
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return None
    cos_angle = float(np.dot(v1, v2) / denom)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _score_anchor_triplets(
    candidates: mda.AtomGroup,
    ligand_reference: mda.AtomGroup,
    ligand_interactions: Sequence[_LigandInteractionSite],
    *,
    min_anchor_distance: float,
    target_angle: float,
    max_candidates: int,
    max_p1_candidates: int,
) -> list[Any] | None:
    ligand_center = ligand_reference.center_of_geometry()
    ligand_distances = distance_array(
        np.asarray([ligand_center], dtype=float),
        candidates.positions,
        box=candidates.universe.dimensions,
    )[0]

    candidate_records = []
    for idx, atom in enumerate(candidates):
        interaction_rank, interaction_distance = _best_residue_interaction(
            atom.residue,
            ligand_interactions,
        )
        name_penalty = 0.0 if atom.name == "CA" else 0.05
        candidate_records.append(
            {
                "atom": atom,
                "ligand_distance": float(ligand_distances[idx]),
                "interaction_rank": interaction_rank,
                "interaction_distance": interaction_distance,
                "p1_score": 10.0 * interaction_rank
                + interaction_distance
                + 0.05 * abs(float(ligand_distances[idx]) - 8.0)
                + name_penalty,
            }
        )

    candidate_records.sort(key=lambda item: (item["p1_score"], item["atom"].index))
    limited_records = candidate_records[:max(3, int(max_candidates))]
    p1_records = candidate_records[:max(1, int(max_p1_candidates))]

    best_score: float | None = None
    best_atoms: list[Any] | None = None
    for p1_record in p1_records:
        p1 = p1_record["atom"]
        for p2_record in limited_records:
            p2 = p2_record["atom"]
            if p2.index == p1.index or p2.residue.ix == p1.residue.ix:
                continue
            d12 = float(np.linalg.norm(p1.position - p2.position))
            if d12 < min_anchor_distance:
                continue
            for p3_record in limited_records:
                p3 = p3_record["atom"]
                if (
                    p3.index in {p1.index, p2.index}
                    or p3.residue.ix in {p1.residue.ix, p2.residue.ix}
                ):
                    continue
                d23 = float(np.linalg.norm(p2.position - p3.position))
                if d23 < min_anchor_distance:
                    continue
                angle = _angle_degrees(p1.position, p2.position, p3.position)
                if angle is None:
                    continue
                score = (
                    abs(angle - target_angle) / target_angle
                    + 5.0 * p1_record["interaction_rank"]
                    + 0.05 * p1_record["interaction_distance"]
                    + 0.02 * abs(p1_record["ligand_distance"] - 8.0)
                    + 0.01
                    * (
                        abs(d12 - min_anchor_distance)
                        + abs(d23 - min_anchor_distance)
                    )
                    + (0.0 if p1.name == "CA" else 0.05)
                    + (0.0 if p2.name == "CA" else 0.05)
                    + (0.0 if p3.name == "CA" else 0.05)
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_atoms = [p1, p2, p3]

    return best_atoms


def _best_residue_interaction(
    residue: Any,
    ligand_sites: Sequence[_LigandInteractionSite],
) -> tuple[int, float]:
    if not ligand_sites:
        return 4, float("inf")

    best_rank = 4
    best_distance = float("inf")
    for protein_atom in residue.atoms:
        protein_element = _atom_element(protein_atom)
        if protein_element == "H":
            continue
        protein_charge = _protein_atom_charge(protein_atom)
        protein_donor = _protein_atom_is_hbond_donor(protein_atom)
        protein_acceptor = _protein_atom_is_hbond_acceptor(protein_atom)
        protein_polar = protein_charge != 0 or protein_donor or protein_acceptor

        for ligand_site in ligand_sites:
            distance = float(np.linalg.norm(protein_atom.position - ligand_site.position))
            rank = 4
            if (
                protein_charge != 0
                and ligand_site.charge != 0
                and protein_charge * ligand_site.charge < 0
                and distance <= _SALT_BRIDGE_CUTOFF
            ):
                rank = 0
            elif (
                (
                    protein_donor
                    and ligand_site.acceptor
                    or protein_acceptor
                    and ligand_site.donor
                )
                and distance <= _HBOND_CUTOFF
            ):
                rank = 1
            elif protein_polar and ligand_site.polar and distance <= _POLAR_CONTACT_CUTOFF:
                rank = 2
            elif distance < best_distance:
                rank = 3

            if (rank, distance) < (best_rank, best_distance):
                best_rank = rank
                best_distance = distance

    return best_rank, best_distance


def _protein_atom_charge(atom: Any) -> int:
    key = (str(atom.resname).upper(), str(atom.name).upper())
    if key in _PROTEIN_POSITIVE_ATOMS:
        return 1
    if key in _PROTEIN_NEGATIVE_ATOMS:
        return -1
    return 0


def _protein_atom_is_hbond_donor(atom: Any) -> bool:
    name = str(atom.name).upper()
    return name in _PROTEIN_HBOND_DONOR_NAMES or (
        _atom_element(atom) == "N" and _protein_atom_charge(atom) >= 0
    )


def _protein_atom_is_hbond_acceptor(atom: Any) -> bool:
    name = str(atom.name).upper()
    if _protein_atom_charge(atom) > 0:
        return False
    return name in _PROTEIN_HBOND_ACCEPTOR_NAMES or _atom_element(atom) == "O"


def _atom_element(atom: Any) -> str:
    try:
        element = str(atom.element).strip().upper()
    except Exception:
        element = ""
    if element:
        return element
    name = str(atom.name).strip().upper()
    if len(name) >= 2 and name[:2] in {"CL", "BR"}:
        return name[:2]
    return name[:1]


def _atom_selection(atom: Any, universe: mda.Universe) -> str:
    base = f"protein and resid {int(atom.resid)} and name {atom.name}"
    selectors = [base]
    segid = str(getattr(atom, "segid", "")).strip()
    try:
        chain_id = str(atom.chainID).strip()
    except Exception:
        chain_id = ""
    if chain_id:
        selectors.append(f"chainID {chain_id} and {base}")
    if segid:
        selectors.append(f"segid {segid} and {base}")
    selectors.append(f"index {int(atom.index)}")

    for selector in selectors:
        try:
            if universe.select_atoms(selector).n_atoms == 1:
                return selector
        except Exception:
            continue
    return f"index {int(atom.index)}"


def _unit_vector_or_none(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        return None
    return np.asarray(vector, dtype=float) / norm


def _synthetic_apo_l1_vector(
    p1_position: np.ndarray,
    p2_position: np.ndarray,
    protein_center: np.ndarray,
    distance: float,
) -> np.ndarray:
    p1_position = np.asarray(p1_position, dtype=float)
    protein_center = np.asarray(protein_center, dtype=float)

    direction = _unit_vector_or_none(p1_position - protein_center)
    if direction is not None:
        return direction * float(distance)

    p1_to_p2 = np.asarray(p2_position, dtype=float) - p1_position
    direction = _unit_vector_or_none(p1_to_p2)
    if direction is None:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=float)

    base = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(direction, base))) > 0.9:
        base = np.asarray([0.0, 1.0, 0.0], dtype=float)
    perpendicular = _unit_vector_or_none(np.cross(direction, base))
    if perpendicular is None:
        perpendicular = base
    return perpendicular * float(distance)


def find_anchor_atoms(
    u_prot: mda.Universe,
    u_lig: mda.Universe,
    lig_sdf: Optional[str | Path],
    anchor_atoms: Sequence[str],
    ligand_anchor_atom: Optional[str] = None,
    unbound_threshold: Optional[float] = None,
    protein_dssp: Any = None,
    apo_ligand: bool = False,
    apo_ligand_distance: Optional[float] = None,
) -> Tuple[float, float, float, str, str, str, float]:
    """
    Identify Boresch-style anchor atoms and pocket geometry.

    Returns
    -------
    (l1_x, l1_y, l1_z, p1_str, p2_str, p3_str, r_dist)
        Translation vector from P1 to ligand COM (Å), formatted protein anchor
        strings (``:RESID@NAME``), and the ligand distance magnitude + 1 Å.
        For apo dummy ligands, the vector is synthesized near P1 instead of
        using the dummy atom's input coordinates.

    Raises
    ------
    ValueError
        If ``unbound_threshold`` is set and the minimum distance between any
        ligand atom and the three anchor atoms is greater than or equal to the
        threshold.
    """
    if len(anchor_atoms) == 0:
        anchor_atoms = select_receptor_anchor_atoms(
            u_prot,
            u_lig,
            lig_sdf,
            protein_dssp=protein_dssp,
        )
    elif len(anchor_atoms) != 3:
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
        raise ValueError(
            "More than one atom selected in the anchor atoms. with the provided selection strings."
            f"\np1: {anchor_atoms[0]}, p2: {anchor_atoms[1]}, p3: {anchor_atoms[2]}\n"
            f"Selected atoms:\n"
            f"P1_atom={P1_atom}, \n"
            f"P2_atom={P2_atom}, \n"
            f"P3_atom={P3_atom}\n"
            f"P1_atom.n_atoms={P1_atom.n_atoms}, "
            f"P2_atom.n_atoms={P2_atom.n_atoms}, "
            f"P3_atom.n_atoms={P3_atom.n_atoms}"
        )

    if unbound_threshold is not None:
        if unbound_threshold < 0:
            raise ValueError("unbound_threshold must be >= 0.")
        anchor_positions = np.vstack(
            (P1_atom.positions, P2_atom.positions, P3_atom.positions)
        )
        min_anchor_dist = float(
            distance_array(
                u_lig.atoms.positions,
                anchor_positions,
                box=u_merge.dimensions,
            ).min()
        )
        if min_anchor_dist >= float(unbound_threshold):
            raise ValueError(
                "Ligand appears unbound during system prep: "
                f"minimum ligand-anchor distance ({min_anchor_dist:.3f} Å) "
                f">= unbound threshold ({float(unbound_threshold):.3f} Å)."
            )

    if ligand_anchor_atom:
        lig_sel = u_merge.select_atoms(ligand_anchor_atom)
        if lig_sel.n_atoms == 0:
            logger.warning(
                "Provided ligand anchor atom '{}' not found; using all ligand atoms instead.",
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
                    "Could not derive ligand candidates from SDF '{}': {}. Using all ligand atoms.",
                    lig_sdf,
                    e,
                )

    if apo_ligand:
        distance = float(
            apo_ligand_distance if apo_ligand_distance is not None else 5.0
        )
        protein_atoms = u_prot.select_atoms("protein")
        protein_center = protein_atoms.center_of_geometry()
        r_vect = _synthetic_apo_l1_vector(
            P1_atom.positions[0],
            P2_atom.positions[0],
            protein_center,
            distance,
        ).reshape(1, 3)
        logger.debug(
            "Using synthetic apo L1 vector with length {:.2f} Å instead of raw dummy coordinates.",
            distance,
        )
    else:
        r_vect = lig_sel.center_of_mass() - P1_atom.positions  # shape (1,3)
    l1_x, l1_y, l1_z = float(r_vect[0][0]), float(r_vect[0][1]), float(r_vect[0][2])
    logger.debug("l1_x={:.2f}; l1_y={:.2f}; l1_z={:.2f}", l1_x, l1_y, l1_z)

    p1_formatted = f":{P1_atom.resids[0]}@{P1_atom.names[0]}"
    p2_formatted = f":{P2_atom.resids[0]}@{P2_atom.names[0]}"
    p3_formatted = f":{P3_atom.resids[0]}@{P3_atom.names[0]}"
    logger.debug(
        "Receptor anchors: P1={}, P2={}, P3={}",
        p1_formatted,
        p2_formatted,
        p3_formatted,
    )

    l1_range = float(np.linalg.norm(r_vect)) + 1.0
    return l1_x, l1_y, l1_z, p1_formatted, p2_formatted, p3_formatted, l1_range


def get_ligand_candidates(ligand_sdf: str | Path, removeHs: bool = False) -> List[int]:
    """
    Return ligand atom indices suitable for anchor selection.

    Criteria: heavy atoms bound to at least two other heavy atoms while skipping
    hydrogens and sp-hybridised carbons. Falls back to all non-hydrogen atoms
    when fewer than three candidates survive.
    """
    if Chem is None:
        raise RuntimeError("RDKit is required for get_ligand_candidates.")

    # Preserve the file's original atom indexing. Some later paths use these
    # indices against full-H molecules, so physically removing Hs here can
    # create out-of-range RDKit accesses.
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
                distance_array(
                    ion.position, complex_sel.positions, box=universe.dimensions
                ).min()
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
            "Not enough {} ions ≥15 Å from complex; relaxing to 10 Å.", ion_type
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
    Extra buffer (Å) needed along Z to maintain ``targeted_buf`` water thickness from
    the solute to the protein.
    e.g. from the solute in the solvent to the protein.
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
) -> (float, float, float):
    """
    Compute a Z-shift (Å) for the ligand to place it roughly mid-solvent.

    return
    -------
    sdr_dist: Z-shift (Å) to apply to the ligand to achieve the targeted buffer distance from the protein.
    z_abs: Absolute z box size (Å) needed to fit the system with the targeted buffer.
    z_left: Remaining buffer distance on the top side of the ligand after placing the ligand.
    """
    u = mda.Universe(str(protein_file))
    ligand = u.select_atoms(f"resname {lig_resname}")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand {lig_resname} not found in {protein_file}")

    prot_sel = "protein and not (resname WAT SPC TIP3 TIP3P TIP4P OPC NA+ Na+ K+ Cl-)"
    protein = (
        u.select_atoms(prot_sel)
        if u.select_atoms(prot_sel).n_atoms
        else u.select_atoms("protein")
    )

    prot_z_max = protein.positions[:, 2].max()
    prot_z_min = protein.positions[:, 2].min()

    lig_cog = ligand.positions.mean(axis=0)
    lig_radius = np.max(np.linalg.norm(ligand.positions - lig_cog, axis=1))
    # z box size should be
    # |--(buffer_z)--ligand (rotatable)--(buffer_z)--protein-----|
    # this assume ligand will not stretch
    # add extra 2.0 Å to account for ligand flexibility.
    abs_z = prot_z_max - prot_z_min + 2 * buffer_z + 2 * lig_radius + 2.0
    # now determine the placement of the ligand in z to achieve the above buffer condition
    box_below_protein = prot_z_min
    buffer_z_left = buffer_z - box_below_protein + lig_radius
    buffer_z_left = max(0.0, buffer_z_left)
    z_shift = buffer_z + prot_z_max - lig_cog[2] + lig_radius + 1.0  # add extra 1.0 Å to avoid clashes
    return z_shift, float(abs_z), float(buffer_z_left)
