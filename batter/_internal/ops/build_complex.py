from __future__ import annotations

import os
import re
import glob
import json
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger

from batter.data import charmmlipid2amber as charmmlipid2amber_csv
from batter.utils import (
    run_with_log,
    tleap,
    cpptraj,
    charmmlipid2amber,
    vmd,
)

from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import (
    register_build_complex,
    register_sim_files,
)
from batter._internal.ops.helpers import (
    get_buffer_z,
    get_sdr_dist,
    get_ligand_candidates,
    num_to_mask,
    select_ions_away_from_complex,
    Anchors,
    revised_resids_for_lipid_fragments,
    save_anchors,
)
from batter._internal.templates import BUILD_FILES_DIR as build_files_orig  # type: ignore


_INITIAL_SALT_BRIDGE_DISTANCE_CUTOFF = 4.0
_PROTEIN_POSITIVE_SALT_ATOMS = frozenset(
    {
        ("ARG", "NH1"),
        ("ARG", "NH2"),
        ("ARG", "NE"),
        ("LYS", "NZ"),
        ("HIP", "ND1"),
        ("HIP", "NE2"),
    }
)
_PROTEIN_NEGATIVE_SALT_ATOMS = frozenset(
    {
        ("ASP", "OD1"),
        ("ASP", "OD2"),
        ("GLU", "OE1"),
        ("GLU", "OE2"),
    }
)
_BORESCH_GUARD_HELPERS = None


def _boresch_guard_helpers():
    global _BORESCH_GUARD_HELPERS
    if _BORESCH_GUARD_HELPERS is None:
        from batter._internal.ops.restraints import (
            BORESCH_MIN_ANGLE_MARGIN_DEG,
            BORESCH_MIN_TORSION_MARGIN_DEG,
            _boresch_frame_margins,
            _boresch_frame_values,
            _frame_safe_boresch_atom_names_from_residue,
        )

        _BORESCH_GUARD_HELPERS = (
            BORESCH_MIN_ANGLE_MARGIN_DEG,
            BORESCH_MIN_TORSION_MARGIN_DEG,
            _boresch_frame_margins,
            _boresch_frame_values,
            _frame_safe_boresch_atom_names_from_residue,
        )
    return _BORESCH_GUARD_HELPERS


def _atom_is_hydrogen(atom) -> bool:
    """Check whether an MDAnalysis atom is hydrogen.

    Parameters
    ----------
    atom
        MDAnalysis atom-like object. The object may expose ``element`` and
        ``name`` attributes depending on the source topology.

    Returns
    -------
    bool
        ``True`` when the atom is identified as hydrogen from its element or
        atom name, otherwise ``False``.
    """
    try:
        element = str(atom.element).strip().upper()
        if element:
            return element == "H"
    except Exception:
        pass

    name = str(getattr(atom, "name", "")).strip().upper()
    return name.startswith("H") or (len(name) > 1 and name[0].isdigit() and name[1] == "H")


_COVALENT_RADII_ANGSTROM = {
    "B": 0.84,
    "BR": 1.20,
    "C": 0.76,
    "CL": 1.02,
    "F": 0.57,
    "I": 1.39,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "SE": 1.20,
    "SI": 1.11,
}


def _atom_element_symbol(atom) -> str:
    for attr in ("element", "type", "name"):
        try:
            value = str(getattr(atom, attr, "")).strip()
        except Exception:
            continue
        if not value:
            continue
        letters = re.sub(r"[^A-Za-z]", "", value).upper()
        if not letters:
            continue
        if len(letters) >= 2 and letters[:2] in _COVALENT_RADII_ANGSTROM:
            return letters[:2]
        return letters[0]
    return "C"


def _ligand_heavy_adjacency(atoms: Sequence[object]) -> dict[int, set[int]]:
    """Estimate heavy-atom connectivity from topology bonds, then distances."""
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(atoms))}
    index_to_pos: dict[int, int] = {}
    for idx, atom in enumerate(atoms):
        try:
            index_to_pos[int(atom.index)] = idx
        except Exception:
            pass

    if index_to_pos:
        for idx, atom in enumerate(atoms):
            try:
                bonded_atoms = list(atom.bonded_atoms)
            except Exception:
                bonded_atoms = []
            for bonded in bonded_atoms:
                try:
                    bonded_idx = int(bonded.index)
                except Exception:
                    continue
                bonded_pos = index_to_pos.get(bonded_idx)
                if bonded_pos is not None and bonded_pos != idx:
                    adjacency[idx].add(bonded_pos)
                    adjacency[bonded_pos].add(idx)
        if any(adjacency.values()):
            return adjacency

    positions: list[np.ndarray] = []
    elements: list[str] = []
    for atom in atoms:
        try:
            positions.append(np.asarray(atom.position, dtype=float))
        except Exception:
            positions.append(np.full(3, np.nan))
        elements.append(_atom_element_symbol(atom))

    for i in range(len(atoms)):
        if positions[i].shape != (3,) or not np.all(np.isfinite(positions[i])):
            continue
        for j in range(i + 1, len(atoms)):
            if positions[j].shape != (3,) or not np.all(np.isfinite(positions[j])):
                continue
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance < 0.40:
                continue
            radius_i = _COVALENT_RADII_ANGSTROM.get(elements[i], 0.76)
            radius_j = _COVALENT_RADII_ANGSTROM.get(elements[j], 0.76)
            if distance <= radius_i + radius_j + 0.45:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def _ligand_heavy_neighbor_counts(atoms: Sequence[object]) -> dict[int, int]:
    """Estimate heavy-atom degrees from topology bonds, then covalent distances."""
    return {
        idx: len(neighbors)
        for idx, neighbors in _ligand_heavy_adjacency(atoms).items()
    }


def _ligand_ring_membership(atoms: Sequence[object]) -> dict[int, bool]:
    """Return whether each heavy atom participates in an inferred ring."""
    adjacency = _ligand_heavy_adjacency(atoms)
    membership = {idx: False for idx in range(len(atoms))}

    for idx, neighbors in adjacency.items():
        neighbor_list = list(neighbors)
        if len(neighbor_list) < 2:
            continue
        blocked = idx
        for start_pos, start in enumerate(neighbor_list[:-1]):
            targets = set(neighbor_list[start_pos + 1 :])
            stack = [start]
            visited = {blocked, start}
            while stack and targets:
                current = stack.pop()
                if current in targets:
                    membership[idx] = True
                    targets.clear()
                    break
                for neighbor in adjacency.get(current, set()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            if membership[idx]:
                break
    return membership


def _ligand_anchor_priority_class(heavy_neighbors: int, in_ring: bool) -> int:
    """Lower class is preferred for ligand L2/L3 anchor selection."""
    degree = int(heavy_neighbors)
    if bool(in_ring) and degree >= 2:
        return 0
    if degree > 2:
        return 1
    if degree >= 2:
        return 2
    return 3


def _ligand_anchor_pair_priority_rank(
    first_name: str,
    second_name: str,
    *,
    heavy_neighbors_by_name: dict[str, int],
    ring_by_name: dict[str, bool],
) -> int:
    """Rank L2/L3 pairs by ring/internal character before geometry tie-breaks."""
    degree1 = int(heavy_neighbors_by_name.get(first_name, 0))
    degree2 = int(heavy_neighbors_by_name.get(second_name, 0))
    low_degree_count = int(degree1 < 2) + int(degree2 < 2)
    class1 = _ligand_anchor_priority_class(degree1, ring_by_name.get(first_name, False))
    class2 = _ligand_anchor_priority_class(degree2, ring_by_name.get(second_name, False))
    degree_sum = min(degree1 + degree2, 99)
    return (
        low_degree_count * 10_000_000
        + (class1 + class2) * 1_000_000
        + class1 * 100_000
        + class2 * 10_000
        - degree_sum * 100
        - min(degree1, 9) * 10
        - min(degree2, 9)
    )


def _executable_path(command: str) -> str | None:
    path = shutil.which(command)
    if path:
        return path

    env_path = Path(sys.executable).resolve().parent / command
    if env_path.exists() and os.access(env_path, os.X_OK):
        return str(env_path)
    return None


def _executable_available(command: str) -> bool:
    return _executable_path(command) is not None


def _run_pdb4amber(input_pdb: Path, output_pdb: Path, *, working_dir: Path) -> None:
    executable = _executable_path("pdb4amber")
    if executable is None:
        raise FileNotFoundError(
            "pdb4amber is required but was not found in PATH. "
            "Activate the batter_dev/AmberTools environment before building complexes."
        )
    run_with_log(
        f"{shlex.quote(executable)} -i {shlex.quote(input_pdb.name)} "
        f"-o {shlex.quote(output_pdb.name)} -y",
        working_dir=working_dir,
    )


def _empty_atomgroup(u: mda.Universe):
    return u.atoms[[]]


def _write_atomgroup_pdb(ag, path: Path) -> None:
    if ag.n_atoms == 0:
        path.write_text("END\n")
        return
    ag.write(str(path))


def _unique_atomgroup(u: mda.Universe, *groups):
    atom_indices: list[int] = []
    for group in groups:
        if group is not None and group.n_atoms:
            atom_indices.extend(int(idx) for idx in group.ix)
    if not atom_indices:
        return _empty_atomgroup(u)
    return u.atoms[np.asarray(sorted(set(atom_indices)), dtype=int)]


def _center_of_atoms(ag, *, mass_weighted: bool = True) -> np.ndarray:
    if ag.n_atoms == 0:
        raise ValueError("Cannot compute center of an empty atom selection.")
    positions = np.asarray(ag.positions, dtype=float)
    if not mass_weighted:
        return positions.mean(axis=0)
    try:
        masses = np.asarray(ag.masses, dtype=float)
        if np.all(np.isfinite(masses)) and float(masses.sum()) > 0.0:
            return np.average(positions, axis=0, weights=masses)
    except Exception:
        pass
    return positions.mean(axis=0)


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _kabsch_transform(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_c = mobile - mobile_center
    reference_c = reference - reference_center
    covariance = mobile_c.T @ reference_c
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def _resname_group(u: mda.Universe, names: Sequence[str]):
    names_set = {str(name) for name in names if str(name)}
    if not names_set:
        return _empty_atomgroup(u)
    mask = np.asarray([str(resname) in names_set for resname in u.atoms.resnames])
    return u.atoms[mask]


def _pdb_residue_names(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            name = line[17:21].strip()
            if name:
                names.add(name)
    return names


def _charmm_lipid_source_residue_names() -> set[str]:
    try:
        df = pd.read_csv(charmmlipid2amber_csv, header=1)
    except Exception as exc:
        logger.warning(
            "Could not read CHARMM-to-AMBER lipid map {}; "
            "assuming lipid conversion is needed: {}",
            charmmlipid2amber_csv,
            exc,
        )
        return set()
    return {
        str(residue).strip()
        for residue in df.get("residue", pd.Series(dtype=str)).dropna()
        if str(residue).strip()
    }


def _lipids_need_charmm_to_amber_conversion(lipids_pdb: Path) -> bool:
    lipid_resnames = _pdb_residue_names(lipids_pdb)
    charmm_resnames = _charmm_lipid_source_residue_names()
    if not charmm_resnames:
        return True
    return bool(lipid_resnames & charmm_resnames)


def _python_split_rec_file(
    *,
    workdir: Path,
    mol: str,
    solv_shell: float,
    other_mol: Sequence[str],
    lipid_mol: Sequence[str],
    keep_all_waters: bool = False,
) -> None:
    """Python fallback for split-ini.tcl when VMD is unavailable."""
    rec_file = workdir / "rec_file.pdb"
    u = mda.Universe(str(rec_file))
    core_sel = f"(protein or resname ACE NMA NME NHE or resname {mol})"
    dum = u.select_atoms("resname DUM")
    prot = u.select_atoms(f"(protein or resname ACE NMA NME NHE) and not resname {mol}")
    lipid = _resname_group(u, lipid_mol)
    lig = u.select_atoms(f"resname {mol}")

    if other_mol:
        other_names = " ".join(str(name) for name in other_mol)
        othrs = u.select_atoms(
            f"resname {other_names} and not water and same residue as "
            f"(around {float(solv_shell):.6g} {core_sel})"
        )
    else:
        othrs = _empty_atomgroup(u)

    near_terms = [core_sel]
    if other_mol:
        near_terms.append(f"resname {' '.join(str(name) for name in other_mol)}")
    if lipid_mol:
        near_terms.append(f"resname {' '.join(str(name) for name in lipid_mol)}")
    near_sel = "(" + " or ".join(near_terms) + ")"
    if keep_all_waters:
        wat = u.select_atoms("water")
    else:
        wat = u.select_atoms(
            f"water and same residue as (around {float(solv_shell):.6g} {near_sel})"
        )
    if wat.n_atoms:
        wat.residues.resnames = "WAT"

    if keep_all_waters:
        ion = u.select_atoms("resname Na+ Cl- K+")
    else:
        ion = u.select_atoms(
            "resname Na+ Cl- K+ and same residue as "
            "(around 5 (protein or resname ACE NMA NME NHE))"
        )

    _write_atomgroup_pdb(dum, workdir / "dummy.pdb")
    _write_atomgroup_pdb(prot, workdir / "protein.pdb")
    _write_atomgroup_pdb(ion, workdir / "others.pdb")
    if othrs.n_atoms:
        combined = _unique_atomgroup(u, ion, othrs)
        _write_atomgroup_pdb(combined, workdir / "others.pdb")
    _write_atomgroup_pdb(lipid, workdir / "lipids.pdb")
    _write_atomgroup_pdb(wat, workdir / "crystalwat.pdb")
    ligand_pdb = workdir / f"{mol}.pdb"
    if lig.n_atoms:
        _write_atomgroup_pdb(lig, ligand_pdb)
    elif not ligand_pdb.exists() or ligand_pdb.stat().st_size == 0:
        _write_atomgroup_pdb(lig, ligand_pdb)
    else:
        logger.debug(
            "Python split found no atoms with residue name {}; keeping existing {}.",
            mol,
            ligand_pdb,
        )


def _python_measure_fit(
    *,
    workdir: Path,
    reference_pdb: str = "aligned-nc.pdb",
    mobile_pdb: str = "complex.pdb",
    output_pdb: str = "aligned.pdb",
) -> None:
    """Python fallback for measure-fit.tcl when VMD is unavailable."""
    ref = mda.Universe(str(workdir / reference_pdb))
    mob = mda.Universe(str(workdir / mobile_pdb))
    ref_sel = ref.select_atoms("protein and backbone")
    mob_sel = mob.select_atoms("protein and backbone")
    if ref_sel.n_atoms == 0 or mob_sel.n_atoms == 0:
        raise RuntimeError("Cannot align complex: empty protein backbone selection.")
    if ref_sel.n_atoms != mob_sel.n_atoms:
        n = min(ref_sel.n_atoms, mob_sel.n_atoms)
        logger.warning(
            "Backbone atom counts differ during Python fit (reference={}, mobile={}); "
            "using first {} atoms.",
            ref_sel.n_atoms,
            mob_sel.n_atoms,
            n,
        )
        ref_pos = ref_sel.positions[:n]
        mob_pos = mob_sel.positions[:n]
    else:
        ref_pos = ref_sel.positions
        mob_pos = mob_sel.positions
    rotation, translation = _kabsch_transform(np.asarray(mob_pos), np.asarray(ref_pos))
    mob.atoms.positions = np.asarray(mob.atoms.positions) @ rotation + translation
    mob.atoms.write(str(workdir / output_pdb))


def _python_nochain_for_alignment(workdir: Path) -> None:
    """Python equivalent of nochain.tcl for USalign input preparation."""
    for input_name, output_name in (
        ("reference_amber.pdb", "reference_amber-nc.pdb"),
        ("complex.pdb", "complex-nc.pdb"),
    ):
        u = mda.Universe(str(workdir / input_name))
        protein = u.select_atoms("protein")
        if protein.n_atoms == 0:
            raise RuntimeError(
                f"Cannot prepare {output_name}: no protein atoms in {input_name}."
            )
        try:
            protein.chainIDs = "X"
        except Exception:
            pass
        _write_atomgroup_pdb(protein, workdir / output_name)


def _translate_pdb_to_reference_frame(
    *,
    target_pdb: Path,
    reference_pdb: Path,
    selection: str = "protein and name CA",
) -> np.ndarray | None:
    """Translate a generated PDB back to the reference coordinate frame."""
    if not target_pdb.exists() or not reference_pdb.exists():
        return None

    target = mda.Universe(str(target_pdb))
    reference = mda.Universe(str(reference_pdb))
    target_sel = target.select_atoms(selection)
    reference_sel = reference.select_atoms(selection)
    if target_sel.n_atoms == 0 or reference_sel.n_atoms == 0:
        logger.debug(
            "Could not restore {} to reference frame: empty selection {!r}.",
            target_pdb.name,
            selection,
        )
        return None
    if target_sel.n_atoms != reference_sel.n_atoms:
        logger.debug(
            "Could not restore {} to reference frame: selection {!r} has {} target "
            "atoms and {} reference atoms.",
            target_pdb.name,
            selection,
            target_sel.n_atoms,
            reference_sel.n_atoms,
        )
        return None

    translation = np.median(reference_sel.positions - target_sel.positions, axis=0)
    if not np.all(np.isfinite(translation)):
        return None
    if float(np.linalg.norm(translation)) <= 1.0e-4:
        return translation

    target.atoms.positions = target.atoms.positions + translation
    target.atoms.write(str(target_pdb))
    logger.debug(
        "Translated {} back to reference frame by [{:.3f}, {:.3f}, {:.3f}] Å.",
        target_pdb.name,
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )
    return translation


def _translate_pdb_by_vector(target_pdb: Path, translation: Sequence[float]) -> None:
    if not target_pdb.exists():
        return
    vector = np.asarray(translation, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return
    if float(np.linalg.norm(vector)) <= 1.0e-4:
        return
    target = mda.Universe(str(target_pdb))
    target.atoms.positions = target.atoms.positions + vector
    target.atoms.write(str(target_pdb))


def _write_python_prep_script_marker(workdir: Path) -> None:
    (workdir / "prep.tcl").write_text(
        "# BATTER used Python ligand-anchor preparation; prep-ini.tcl was not run.\n"
    )


def _ligand_atom_by_name(u: mda.Universe, mol: str, atom_name: str):
    return u.select_atoms(f"resname {mol} and name {atom_name}")


def _receptor_atom_by_resid_name(u: mda.Universe, mol: str, resid: str, atom_name: str):
    return u.select_atoms(f"(not resname {mol}) and resid {resid} and name {atom_name}")


def _atom_from_anchor_mask(
    u: mda.Universe,
    mask: str,
    *,
    mol: str,
    ligand: bool,
):
    match = re.match(r"^:(-?\d+)@(.+)$", str(mask).strip())
    if match is None:
        return None
    resid = match.group(1)
    atom_name = match.group(2).strip()
    if ligand:
        selections = [
            f"resname {mol} and resid {resid} and name {atom_name}",
            f"resid {resid} and name {atom_name}",
        ]
    else:
        selections = [
            f"(not resname {mol}) and resid {resid} and name {atom_name}",
            f"protein and resid {resid} and name {atom_name}",
            f"resid {resid} and name {atom_name}",
        ]
    for selection in selections:
        atoms = u.select_atoms(selection)
        if atoms.n_atoms == 1:
            return atoms[0]
    return None


def _ligand_residue_for_boresch_guard(
    u: mda.Universe,
    *,
    mol: str,
    lig_resid: str,
):
    lig_resid = str(lig_resid or "").strip()
    if lig_resid:
        atoms = u.select_atoms(f"resname {mol} and resid {lig_resid}")
    else:
        atoms = u.select_atoms(f"resname {mol}")
    if atoms.n_atoms == 0:
        atoms = u.select_atoms(f"resname {mol}")
    if atoms.n_atoms == 0:
        return None
    return atoms.residues[0]


def _anchor_mask_from_atom(atom) -> str:
    return f":{int(atom.resid)}@{str(atom.name).strip()}"


def _dedupe_atoms_by_index(atoms: Sequence[object]) -> list[object]:
    selected: list[object] = []
    seen: set[int] = set()
    for atom in atoms:
        if atom is None:
            continue
        idx = int(atom.index)
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(atom)
    return selected


def _receptor_ca_anchor_candidates(u: mda.Universe, mol: str):
    for selection in (
        f"protein and name CA and not resname {mol}",
        f"(not resname {mol}) and name CA",
        "protein and name CA",
        "name CA",
    ):
        try:
            atoms = u.select_atoms(selection)
        except Exception:
            continue
        if atoms.n_atoms >= 3:
            return atoms
    return u.atoms[:0]


def _mask_index_from_atm_num(atm_num: Sequence[str], mask: str) -> int:
    try:
        return list(atm_num).index(mask)
    except ValueError:
        target = str(mask).lower()
        matches = [
            idx for idx, candidate in enumerate(atm_num)
            if str(candidate).lower() == target
        ]
        if len(matches) == 1:
            return matches[0]
        raise


def _atom_record_for_diagnostic(
    u: mda.Universe,
    atm_num: Sequence[str],
    mask: str | None,
    *,
    mol: str,
    ligand: bool,
) -> dict | None:
    if not mask:
        return None
    atom = _atom_from_anchor_mask(u, mask, mol=mol, ligand=ligand)
    if atom is None:
        return {"input_mask": mask, "resolved": False}
    amber_iat = int(atom.index) + 1
    canonical_mask = atm_num[amber_iat] if amber_iat < len(atm_num) else _anchor_mask_from_atom(atom)
    try:
        input_amber_iat = _mask_index_from_atm_num(atm_num, str(mask))
    except ValueError:
        input_amber_iat = None
    return {
        "input_mask": mask,
        "resolved": True,
        "canonical_pdb_mask": canonical_mask,
        "pdb_mask": _anchor_mask_from_atom(atom),
        "input_amber_iat": input_amber_iat,
        "amber_iat": amber_iat,
        "pdb_serial": int(getattr(atom, "id", amber_iat)),
        "atom_index0": int(atom.index),
        "resid": int(atom.resid),
        "resname": str(atom.resname).strip(),
        "name": str(atom.name).strip(),
        "segid": str(getattr(atom, "segid", "")).strip(),
        "chainID": str(getattr(atom, "chainID", "")).strip(),
    }


def _boresch_values_for_masks(
    u: mda.Universe,
    receptor_masks: Sequence[str],
    ligand_masks: Sequence[str],
    *,
    mol: str,
) -> dict | None:
    (
        _min_angle_margin,
        _min_torsion_margin,
        _boresch_frame_margins,
        _boresch_frame_values,
        _,
    ) = _boresch_guard_helpers()
    receptor_atoms = [
        _atom_from_anchor_mask(u, mask, mol=mol, ligand=False)
        for mask in receptor_masks[:3]
    ]
    ligand_atoms = [
        _atom_from_anchor_mask(u, mask, mol=mol, ligand=True)
        for mask in ligand_masks[:3]
    ]
    if any(atom is None for atom in receptor_atoms + ligand_atoms):
        return None
    values = _boresch_frame_values(receptor_atoms, ligand_atoms)
    if values is None:
        return None
    angle_margin, torsion_margin = _boresch_frame_margins(values)
    return {
        "values": [float(value) for value in values],
        "angle_margin_deg": float(angle_margin),
        "torsion_margin_deg": float(torsion_margin),
    }


def _write_abfe_anchor_guard_diagnostic(
    *,
    path: Path,
    fe_pdb: Path,
    mol: str,
    ligand_label: str,
    lig_resid: str,
    old_receptor_masks: Sequence[str],
    new_receptor_masks: Sequence[str],
    old_ligand_names: Sequence[str],
    new_ligand_names: Sequence[str],
    preferred_first_names: Sequence[str],
    allow_receptor_reselection: bool,
    user_anchor_triplet: bool,
) -> None:
    try:
        u = mda.Universe(str(fe_pdb))
        atm_num = num_to_mask(fe_pdb)
        old_ligand_masks = [f":{lig_resid}@{name}" for name in old_ligand_names[:3]]
        new_ligand_masks = [f":{lig_resid}@{name}" for name in new_ligand_names[:3]]
        data = {
            "schema_version": 1,
            "stage": "build_complex_z",
            "source_pdb": str(fe_pdb),
            "ligand_label": ligand_label,
            "residue_name": mol,
            "ligand_resid": str(lig_resid),
            "preferred_first_names": _dedupe_names(preferred_first_names),
            "allow_receptor_reselection": bool(allow_receptor_reselection),
            "user_anchor_triplet": bool(user_anchor_triplet),
            "receptor": {
                "reselected": list(old_receptor_masks[:3]) != list(new_receptor_masks[:3]),
                "old": {
                    key: _atom_record_for_diagnostic(
                        u,
                        atm_num,
                        mask,
                        mol=mol,
                        ligand=False,
                    )
                    for key, mask in zip(("P1", "P2", "P3"), old_receptor_masks[:3])
                },
                "final": {
                    key: _atom_record_for_diagnostic(
                        u,
                        atm_num,
                        mask,
                        mol=mol,
                        ligand=False,
                    )
                    for key, mask in zip(("P1", "P2", "P3"), new_receptor_masks[:3])
                },
            },
            "ligand": {
                "reselected": list(old_ligand_names[:3]) != list(new_ligand_names[:3]),
                "old_names": list(old_ligand_names[:3]),
                "final_names": list(new_ligand_names[:3]),
                "old": {
                    key: _atom_record_for_diagnostic(
                        u,
                        atm_num,
                        mask,
                        mol=mol,
                        ligand=True,
                    )
                    for key, mask in zip(("L1", "L2", "L3"), old_ligand_masks)
                },
                "final": {
                    key: _atom_record_for_diagnostic(
                        u,
                        atm_num,
                        mask,
                        mol=mol,
                        ligand=True,
                    )
                    for key, mask in zip(("L1", "L2", "L3"), new_ligand_masks)
                },
            },
            "boresch": {
                "old": _boresch_values_for_masks(
                    u,
                    old_receptor_masks,
                    old_ligand_masks,
                    mol=mol,
                ),
                "final": _boresch_values_for_masks(
                    u,
                    new_receptor_masks,
                    new_ligand_masks,
                    mol=mol,
                ),
            },
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not write Boresch anchor guard diagnostic {}: {}",
            path,
            exc,
        )


def _dihedral_degrees_from_positions(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> float | None:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p4 = np.asarray(p4, dtype=float)

    b0 = -(p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    norm = float(np.linalg.norm(b1))
    if norm < 1.0e-12:
        return None
    b1 /= norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if float(np.linalg.norm(v)) < 1.0e-12:
        return None
    if float(np.linalg.norm(w)) < 1.0e-12:
        return None
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def _boresch_values_from_positions(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
) -> tuple[float, ...] | None:
    values = (
        _angle_degrees(p2, p1, l1),
        _dihedral_degrees_from_positions(p3, p2, p1, l1),
        _angle_degrees(p1, l1, l2),
        _dihedral_degrees_from_positions(p2, p1, l1, l2),
        _dihedral_degrees_from_positions(p1, l1, l2, l3),
    )
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return tuple(float(value) for value in values)


def _preferred_l1_ligand_triplet_candidates(
    residue,
    preferred_first_names: Sequence[str],
) -> list[dict]:
    preferred_names = _dedupe_names(preferred_first_names)
    if not preferred_names:
        return []

    atoms = [atom for atom in residue.atoms if not _atom_is_hydrogen(atom)]
    if len(atoms) < 3:
        return []

    coords = {
        int(atom.index): np.array(atom.position, dtype=float, copy=True)
        for atom in atoms
    }
    all_positions = np.asarray([coords[int(atom.index)] for atom in atoms], dtype=float)
    centroid = all_positions.mean(axis=0)
    span = max(float(np.max(np.linalg.norm(all_positions - centroid, axis=1))), 1.0)
    heavy_neighbor_counts = _ligand_heavy_neighbor_counts(atoms)
    ring_membership = _ligand_ring_membership(atoms)
    atom_positions: dict[int, int] = {}
    for idx, atom in enumerate(atoms):
        try:
            atom_positions[int(atom.index)] = idx
        except Exception:
            atom_positions[idx] = idx
    heavy_neighbors_by_name: dict[str, int] = {}
    ring_by_name: dict[str, bool] = {}
    for idx, atom in enumerate(atoms):
        name = str(atom.name).strip()
        if not name:
            continue
        heavy_neighbors_by_name[name] = max(
            heavy_neighbors_by_name.get(name, 0),
            int(heavy_neighbor_counts.get(idx, 0)),
        )
        ring_by_name[name] = bool(ring_by_name.get(name, False)) or bool(
            ring_membership.get(idx, False)
        )

    candidates: list[dict] = []
    for preferred_rank, preferred_name in enumerate(preferred_names):
        l1_atoms = [
            atom for atom in atoms if str(atom.name).strip() == preferred_name
        ]
        for l1 in l1_atoms:
            l1_pos = coords[int(l1.index)]
            l1_centrality = float(np.linalg.norm(l1_pos - centroid) / span)
            for l2 in atoms:
                if int(l2.index) == int(l1.index):
                    continue
                l2_pos = coords[int(l2.index)]
                d12 = float(np.linalg.norm(l2_pos - l1_pos))
                if d12 < 0.5:
                    continue
                for l3 in atoms:
                    if int(l3.index) in {int(l1.index), int(l2.index)}:
                        continue
                    l3_pos = coords[int(l3.index)]
                    d13 = float(np.linalg.norm(l3_pos - l1_pos))
                    d23 = float(np.linalg.norm(l3_pos - l2_pos))
                    if min(d13, d23) < 0.5:
                        continue
                    area2 = float(
                        np.linalg.norm(np.cross(l2_pos - l1_pos, l3_pos - l1_pos))
                    )
                    sine = area2 / max(d12 * d13, 1.0e-12)
                    if sine < 0.25:
                        continue

                    spread = (min(d12, d13) + 0.5 * d23) / span
                    local_score = 4.0 * sine + 0.25 * spread - l1_centrality
                    try:
                        l2_index = atom_positions.get(int(l2.index), -1)
                    except Exception:
                        l2_index = -1
                    try:
                        l3_index = atom_positions.get(int(l3.index), -1)
                    except Exception:
                        l3_index = -1
                    low_degree_l2_l3_count = int(
                        heavy_neighbor_counts.get(l2_index, 0) < 2
                    ) + int(heavy_neighbor_counts.get(l3_index, 0) < 2)
                    l2_name = str(l2.name).strip()
                    l3_name = str(l3.name).strip()
                    candidates.append(
                        {
                            "preferred_rank": preferred_rank,
                            "names": [
                                str(l1.name).strip(),
                                l2_name,
                                l3_name,
                            ],
                            "positions": (l1_pos, l2_pos, l3_pos),
                            "score": local_score,
                            "low_degree_l2_l3_count": low_degree_l2_l3_count,
                            "terminal_l2_l3_count": low_degree_l2_l3_count,
                            "l2_l3_priority_rank": _ligand_anchor_pair_priority_rank(
                                l2_name,
                                l3_name,
                                heavy_neighbors_by_name=heavy_neighbors_by_name,
                                ring_by_name=ring_by_name,
                            ),
                        }
                    )

    candidates.sort(
        key=lambda item: (
            int(item["preferred_rank"]),
            int(item.get("l2_l3_priority_rank", 0)),
            -float(item["score"]),
            item["names"],
        )
    )
    return candidates


def _best_preferred_l1_triplet_for_receptor_frame(
    *,
    residue,
    receptor_atoms: Sequence[object],
    preferred_first_names: Sequence[str],
    triplet_candidates: Sequence[dict] | None = None,
) -> dict | None:
    (
        BORESCH_MIN_ANGLE_MARGIN_DEG,
        BORESCH_MIN_TORSION_MARGIN_DEG,
        _boresch_frame_margins,
        _,
        _,
    ) = _boresch_guard_helpers()

    candidates = list(triplet_candidates or [])
    if not candidates:
        candidates = _preferred_l1_ligand_triplet_candidates(
            residue,
            preferred_first_names,
        )
    if not candidates:
        return None

    if len(receptor_atoms) != 3:
        return None
    p1, p2, p3 = [
        np.asarray(atom.position, dtype=float) for atom in receptor_atoms
    ]

    best: dict | None = None
    for candidate in candidates:
        l1, l2, l3 = candidate["positions"]
        values = _boresch_values_from_positions(p1, p2, p3, l1, l2, l3)
        if values is None:
            continue
        angle_margin, torsion_margin = _boresch_frame_margins(values)
        if (
            angle_margin < BORESCH_MIN_ANGLE_MARGIN_DEG
            or torsion_margin < BORESCH_MIN_TORSION_MARGIN_DEG
        ):
            continue

        endpoint_score = 0.03 * angle_margin + 0.05 * torsion_margin
        score = float(candidate["score"]) + endpoint_score
        names = list(candidate["names"])
        scored_candidate = {
            "preferred_rank": int(candidate["preferred_rank"]),
            "names": names,
            "values": values,
            "margins": (angle_margin, torsion_margin),
            "score": score,
            "low_degree_l2_l3_count": int(
                candidate.get(
                    "low_degree_l2_l3_count",
                    candidate.get("terminal_l2_l3_count", 0),
                )
            ),
            "terminal_l2_l3_count": int(candidate.get("terminal_l2_l3_count", 0)),
            "l2_l3_priority_rank": int(candidate.get("l2_l3_priority_rank", 0)),
        }
        if best is None or (
            int(scored_candidate["preferred_rank"]),
            int(scored_candidate["l2_l3_priority_rank"]),
            -float(scored_candidate["score"]),
            names,
        ) < (
            int(best["preferred_rank"]),
            int(best.get("l2_l3_priority_rank", 0)),
            -float(best["score"]),
            best["names"],
        ):
            best = scored_candidate

    return best


def _select_receptor_p2_p3_for_preferred_l1(
    *,
    u: mda.Universe,
    mol: str,
    ligand_label: str,
    residue,
    p1_atom,
    current_p2_atom,
    current_p3_atom,
    preferred_first_names: Sequence[str],
    triplet_candidates: Sequence[dict] | None = None,
    min_anchor_distance: float = 8.0,
    max_candidates: int = 64,
) -> dict | None:
    ca_atoms = _receptor_ca_anchor_candidates(u, mol)
    if ca_atoms.n_atoms < 3:
        return None

    triplet_candidates = list(triplet_candidates or [])
    if not triplet_candidates:
        triplet_candidates = _preferred_l1_ligand_triplet_candidates(
            residue,
            preferred_first_names,
        )
    if not triplet_candidates:
        return None

    p1_pos = np.asarray(p1_atom.position, dtype=float)
    scored_atoms: list[tuple[float, float, int, object]] = []
    for atom in ca_atoms:
        if int(atom.index) == int(p1_atom.index):
            continue
        if int(atom.residue.ix) == int(p1_atom.residue.ix):
            continue
        distance_to_p1 = float(np.linalg.norm(np.asarray(atom.position, dtype=float) - p1_pos))
        scored_atoms.append(
            (
                abs(distance_to_p1 - 10.0),
                distance_to_p1,
                int(atom.index),
                atom,
            )
        )
    scored_atoms.sort(key=lambda item: item[:3])
    candidate_atoms = _dedupe_atoms_by_index(
        [
            current_p2_atom,
            current_p3_atom,
            *[item[3] for item in scored_atoms[: max(3, int(max_candidates))]],
        ]
    )

    def _candidate_pairs(*, one_change_only: bool):
        seen: set[tuple[int, int]] = set()
        if one_change_only:
            pairs = [
                (current_p2_atom, p3_atom)
                for p3_atom in candidate_atoms
                if int(p3_atom.index) != int(current_p3_atom.index)
            ]
            pairs.extend(
                (p2_atom, current_p3_atom)
                for p2_atom in candidate_atoms
                if int(p2_atom.index) != int(current_p2_atom.index)
            )
        else:
            pairs = [
                (p2_atom, p3_atom)
                for p2_atom in candidate_atoms
                for p3_atom in candidate_atoms
                if (
                    int(p2_atom.index) != int(current_p2_atom.index)
                    or int(p3_atom.index) != int(current_p3_atom.index)
                )
            ]
        for p2_atom, p3_atom in pairs:
            key = (int(p2_atom.index), int(p3_atom.index))
            if key in seen:
                continue
            seen.add(key)
            yield p2_atom, p3_atom

    def _score_receptor_pair(p2_atom, p3_atom) -> tuple[tuple, dict] | None:
        if int(p2_atom.index) == int(p1_atom.index):
            return None
        if int(p2_atom.residue.ix) == int(p1_atom.residue.ix):
            return None
        d12 = float(np.linalg.norm(np.asarray(p2_atom.position, dtype=float) - p1_pos))
        if d12 < float(min_anchor_distance):
            return None
        if int(p3_atom.index) in {int(p1_atom.index), int(p2_atom.index)}:
            return None
        if int(p3_atom.residue.ix) in {
            int(p1_atom.residue.ix),
            int(p2_atom.residue.ix),
        }:
            return None
        d23 = float(
            np.linalg.norm(
                np.asarray(p3_atom.position, dtype=float)
                - np.asarray(p2_atom.position, dtype=float)
            )
        )
        if d23 < float(min_anchor_distance):
            return None
        receptor_angle = _angle_degrees(
            p1_atom.position,
            p2_atom.position,
            p3_atom.position,
        )
        if not np.isfinite(receptor_angle):
            return None
        receptor_angle_margin = min(float(receptor_angle), 180.0 - float(receptor_angle))
        if receptor_angle_margin < 15.0:
            return None

        receptor_atoms = (p1_atom, p2_atom, p3_atom)
        triplet = _best_preferred_l1_triplet_for_receptor_frame(
            residue=residue,
            receptor_atoms=receptor_atoms,
            preferred_first_names=preferred_first_names,
            triplet_candidates=triplet_candidates,
        )
        if triplet is None:
            return None

        angle_margin, torsion_margin = triplet["margins"]
        change_count = int(int(p2_atom.index) != int(current_p2_atom.index)) + int(
            int(p3_atom.index) != int(current_p3_atom.index)
        )
        key = (
            change_count,
            int(int(p2_atom.index) != int(current_p2_atom.index)),
            int(triplet["preferred_rank"]),
            -float(torsion_margin),
            -float(angle_margin),
            abs(float(receptor_angle) - 90.0),
            abs(d12 - 10.0) + abs(d23 - 10.0),
            int(p2_atom.index),
            int(p3_atom.index),
        )
        result = {
            "P2": _anchor_mask_from_atom(p2_atom),
            "P3": _anchor_mask_from_atom(p3_atom),
            "p2_atom": p2_atom,
            "p3_atom": p3_atom,
            "names": triplet["names"],
            "values": triplet["values"],
            "margins": triplet["margins"],
            "receptor_angle": receptor_angle,
            "d12": d12,
            "d23": d23,
        }
        return key, result

    best_key: tuple | None = None
    best: dict | None = None
    for one_change_only in (True, False):
        for p2_atom, p3_atom in _candidate_pairs(one_change_only=one_change_only):
            scored = _score_receptor_pair(p2_atom, p3_atom)
            if scored is None:
                continue
            key, result = scored
            if best_key is None or key < best_key:
                best_key = key
                best = result
        if best is not None:
            break

    if best is not None:
        logger.debug(
            "[build_complex_z] Found alternate receptor P2/P3 for {} that keeps "
            "preferred ligand L1 {}: P2={}, P3={}, ligand anchors={}, "
            "Boresch margins=({:.1f}, {:.1f}) deg.",
            ligand_label,
            best["names"][0],
            best["P2"],
            best["P3"],
            best["names"],
            best["margins"][0],
            best["margins"][1],
        )
    return best


def _guard_abfe_boresch_anchor_frame(
    *,
    fe_pdb: Path,
    mol: str,
    ligand_label: str,
    P1: str,
    P2: str,
    P3: str,
    lig_resid: str,
    selected_names: Sequence[str],
    preferred_first_names: Sequence[str] = (),
    allow_receptor_reselection: bool = False,
) -> tuple[str, str, str, list[str]]:
    """Avoid endpoint Boresch frames, optionally changing P2/P3 to keep L1."""
    try:
        (
            _min_angle_margin,
            _min_torsion_margin,
            _boresch_frame_margins,
            _boresch_frame_values,
            _frame_safe_boresch_atom_names_from_residue,
        ) = _boresch_guard_helpers()
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not import Boresch frame guard for {}; "
            "keeping ligand anchors {}: {}",
            ligand_label,
            list(selected_names[:3]),
            exc,
        )
        return P1, P2, P3, list(selected_names[:3])

    try:
        u = mda.Universe(str(fe_pdb))
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not load {} for Boresch frame guard; "
            "keeping ligand anchors {}: {}",
            fe_pdb,
            list(selected_names[:3]),
            exc,
        )
        return P1, P2, P3, list(selected_names[:3])

    receptor_atoms = []
    for mask in (P1, P2, P3):
        atom = _atom_from_anchor_mask(u, mask, mol=mol, ligand=False)
        if atom is None:
            logger.warning(
                "[build_complex_z] Could not resolve receptor anchor {} in {}; "
                "keeping ligand anchors {}.",
                mask,
                fe_pdb.name,
                list(selected_names[:3]),
            )
            return P1, P2, P3, list(selected_names[:3])
        receptor_atoms.append(atom)

    residue = _ligand_residue_for_boresch_guard(u, mol=mol, lig_resid=lig_resid)
    if residue is None:
        logger.warning(
            "[build_complex_z] Could not resolve ligand residue {}:{} in {}; "
            "keeping ligand anchors {}.",
            mol,
            lig_resid,
            fe_pdb.name,
            list(selected_names[:3]),
        )
        return P1, P2, P3, list(selected_names[:3])

    preferred_names = _dedupe_names(preferred_first_names)
    if preferred_names:
        preferred_exc: Exception | None = None
        triplet_candidates: list[dict] = []
        try:
            triplet_candidates = _preferred_l1_ligand_triplet_candidates(
                residue,
                preferred_names,
            )
            preferred_triplet = _best_preferred_l1_triplet_for_receptor_frame(
                residue=residue,
                receptor_atoms=receptor_atoms,
                preferred_first_names=preferred_names,
                triplet_candidates=triplet_candidates,
            )
        except Exception as exc:
            preferred_triplet = None
            preferred_exc = exc
        if preferred_triplet is not None:
            return P1, P2, P3, preferred_triplet["names"]
        if allow_receptor_reselection:
            alternate = _select_receptor_p2_p3_for_preferred_l1(
                u=u,
                mol=mol,
                ligand_label=ligand_label,
                residue=residue,
                p1_atom=receptor_atoms[0],
                current_p2_atom=receptor_atoms[1],
                current_p3_atom=receptor_atoms[2],
                preferred_first_names=preferred_names,
                triplet_candidates=triplet_candidates,
            )
            if alternate is not None:
                logger.debug(
                    "[build_complex_z] Replacing receptor P2/P3 for {} to "
                    "keep preferred ligand L1 {}: ({}, {}) -> ({}, {}).",
                    ligand_label,
                    alternate["names"][0],
                    P2,
                    P3,
                    alternate["P2"],
                    alternate["P3"],
                )
                return P1, alternate["P2"], alternate["P3"], alternate["names"]
        logger.debug(
            "[build_complex_z] Preferred ligand L1 {} for {} did not pass "
            "the current/alternate receptor-frame guard: {}",
            preferred_names,
            ligand_label,
            preferred_exc or "no safe preferred-L1 Boresch triplet",
        )

    try:
        guarded_names = _frame_safe_boresch_atom_names_from_residue(
            residue,
            receptor_atoms=receptor_atoms,
            label=f"ABFE {ligand_label}",
            preferred_first_names=preferred_first_names,
        )[:3]
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not select guarded ABFE Boresch anchors for {}; "
            "keeping ligand anchors {}: {}",
            ligand_label,
            list(selected_names[:3]),
            exc,
        )
        return P1, P2, P3, list(selected_names[:3])

    residue_atoms_by_name = {
        str(atom.name).strip(): atom for atom in residue.atoms
    }
    ligand_atoms = [
        residue_atoms_by_name.get(str(name).strip()) for name in guarded_names
    ]
    if any(atom is None for atom in ligand_atoms):
        logger.warning(
            "[build_complex_z] Guarded ABFE Boresch anchors {} were not all found in {}; "
            "keeping ligand anchors {}.",
            guarded_names,
            fe_pdb.name,
            list(selected_names[:3]),
        )
        return P1, P2, P3, list(selected_names[:3])

    values = _boresch_frame_values(receptor_atoms, ligand_atoms)
    margins = _boresch_frame_margins(values or ())
    if list(selected_names[:3]) != guarded_names:
        logger.debug(
            "[build_complex_z] Replaced ABFE Boresch ligand anchors for {}: {} -> {} "
            "(angle margin {:.1f} deg, torsion margin {:.1f} deg).",
            ligand_label,
            list(selected_names[:3]),
            guarded_names,
            margins[0],
            margins[1],
        )
    return P1, P2, P3, guarded_names


def _guard_abfe_boresch_ligand_anchor_names(
    *,
    fe_pdb: Path,
    mol: str,
    ligand_label: str,
    P1: str,
    P2: str,
    P3: str,
    lig_resid: str,
    selected_names: Sequence[str],
    preferred_first_names: Sequence[str] = (),
) -> list[str]:
    """Compatibility wrapper returning only guarded ligand anchor names."""
    _p1, _p2, _p3, names = _guard_abfe_boresch_anchor_frame(
        fe_pdb=fe_pdb,
        mol=mol,
        ligand_label=ligand_label,
        P1=P1,
        P2=P2,
        P3=P3,
        lig_resid=lig_resid,
        selected_names=selected_names,
        preferred_first_names=preferred_first_names,
        allow_receptor_reselection=False,
    )
    return names


def _pick_ligand_anchor_names(
    *,
    u: mda.Universe,
    mol: str,
    ligand_names: Sequence[str],
    preferred_l1_names: Sequence[str] = (),
    p1_resid: str,
    p1_atom: str,
    p2_resid: str,
    p2_atom: str,
    l1_x: float,
    l1_y: float,
    l1_z: float,
    l1_range: float,
    min_adis: float,
    max_adis: float,
) -> list[str]:
    p1 = _receptor_atom_by_resid_name(u, mol, p1_resid, p1_atom)
    p2 = _receptor_atom_by_resid_name(u, mol, p2_resid, p2_atom)
    if p1.n_atoms == 0 or p2.n_atoms == 0:
        raise RuntimeError("anchor not found")
    p1_center = _center_of_atoms(p1)
    p2_center = _center_of_atoms(p2)
    target = p1_center + np.asarray([l1_x, l1_y, l1_z], dtype=float)

    ligand_heavy_atoms = [
        atom for atom in u.select_atoms(f"resname {mol}") if not _atom_is_hydrogen(atom)
    ]
    heavy_neighbor_counts = _ligand_heavy_neighbor_counts(ligand_heavy_atoms)
    ring_membership = _ligand_ring_membership(ligand_heavy_atoms)
    heavy_neighbors_by_name: dict[str, int] = {}
    ring_by_name: dict[str, bool] = {}
    for idx, atom in enumerate(ligand_heavy_atoms):
        name = str(atom.name).strip()
        if not name:
            continue
        heavy_neighbors_by_name[name] = max(
            heavy_neighbors_by_name.get(name, 0),
            int(heavy_neighbor_counts.get(idx, 0)),
        )
        ring_by_name[name] = bool(ring_by_name.get(name, False)) or bool(
            ring_membership.get(idx, False)
        )
    ligand_name_rank = {
        name: rank for rank, name in enumerate(_dedupe_names([str(x) for x in ligand_names]))
    }

    candidates: dict[str, np.ndarray] = {}
    for name in ligand_names:
        atoms = _ligand_atom_by_name(u, mol, str(name))
        if atoms.n_atoms == 0:
            continue
        if all(_atom_is_hydrogen(atom) for atom in atoms):
            continue
        center = _center_of_atoms(atoms)
        dist = float(np.linalg.norm(center - target))
        if dist >= float(l1_range):
            continue
        angle = _angle_degrees(p2_center, p1_center, center)
        if np.isfinite(angle):
            candidates[str(name)] = center

    preferred_l1 = _dedupe_names(
        str(name).strip()
        for name in preferred_l1_names
        if str(name).strip() in candidates
    )
    preferred_l1_rank = {name: rank for rank, name in enumerate(preferred_l1)}

    def _l1_score(name: str, center: np.ndarray) -> tuple[int, int, float, int, int] | None:
        angle = _angle_degrees(p2_center, p1_center, center)
        if not np.isfinite(angle):
            return None
        angle_diff = abs(angle - 90.0)
        if angle_diff <= 15.0:
            tolerance_rank = 0
        elif angle_diff <= 70.0:
            tolerance_rank = 1
        else:
            return None
        preferred_penalty = 0
        preferred_rank = ligand_name_rank.get(name, len(ligand_name_rank))
        if preferred_l1:
            preferred_penalty = 0 if name in preferred_l1_rank else 1
            preferred_rank = preferred_l1_rank.get(name, len(preferred_l1_rank))
        return (
            preferred_penalty,
            tolerance_rank,
            float(np.linalg.norm(center - target)),
            preferred_rank,
            ligand_name_rank.get(name, len(ligand_name_rank)),
        )

    best_triplet: tuple[str, str, str] | None = None
    best_score: tuple[float, ...] | None = None
    for aa1, aa1_center in candidates.items():
        aa1_score = _l1_score(aa1, aa1_center)
        if aa1_score is None:
            continue
        for aa2, aa2_center in candidates.items():
            if aa2 == aa1:
                continue
            d12 = float(np.linalg.norm(aa2_center - aa1_center))
            if not (float(min_adis) < d12 < float(max_adis)):
                continue
            angle2 = _angle_degrees(p1_center, aa1_center, aa2_center)
            if not np.isfinite(angle2):
                continue
            angle2_diff = abs(angle2 - 90.0)
            aa2_terminal = int(heavy_neighbors_by_name.get(aa2, 0) < 2)
            for aa3, aa3_center in candidates.items():
                if aa3 in {aa1, aa2}:
                    continue
                d23 = float(np.linalg.norm(aa3_center - aa2_center))
                if not (float(min_adis) < d23 < float(max_adis)):
                    continue
                angle3 = _angle_degrees(aa1_center, aa2_center, aa3_center)
                if not np.isfinite(angle3):
                    continue
                angle3_diff = abs(angle3 - 90.0)
                aa3_terminal = int(heavy_neighbors_by_name.get(aa3, 0) < 2)
                l2_l3_priority_rank = _ligand_anchor_pair_priority_rank(
                    aa2,
                    aa3,
                    heavy_neighbors_by_name=heavy_neighbors_by_name,
                    ring_by_name=ring_by_name,
                )
                score = (
                    float(aa1_score[0]),
                    float(aa2_terminal + aa3_terminal),
                    float(aa2_terminal),
                    float(aa3_terminal),
                    float(l2_l3_priority_rank),
                    float(aa1_score[1]),
                    float(aa1_score[2]),
                    float(angle2_diff + angle3_diff),
                    float(max(angle2_diff, angle3_diff)),
                    float(aa1_score[3]),
                    float(aa1_score[4]),
                    float(ligand_name_rank.get(aa2, len(ligand_name_rank))),
                    float(ligand_name_rank.get(aa3, len(ligand_name_rank))),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_triplet = (aa1, aa2, aa3)
    if best_triplet is None:
        raise RuntimeError("anchor not found")
    return list(best_triplet)


def _dedupe_names(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean = str(name).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _order_ligand_names_with_priority(
    names: Sequence[str],
    priority_names: Sequence[str],
) -> list[str]:
    base = _dedupe_names(names)
    priority = _dedupe_names(priority_names)
    base_set = set(base)
    priority_set = set(priority)
    return [name for name in priority if name in base_set] + [
        name for name in base if name not in priority_set
    ]


def _python_prep_complex(
    *,
    workdir: Path,
    mol: str,
    p1_atom: str,
    p1_vmd: str,
    p2_atom: str,
    p2_vmd: str,
    first_resid: str,
    last_resid: str,
    stage: str,
    l1_x: float,
    l1_y: float,
    l1_z: float,
    l1_range: float,
    min_adis: float,
    max_adis: float,
    sdr_dist: float,
    ligand_names: Sequence[str],
    other_mol: Sequence[str],
    lipid_mol: Sequence[str],
    preferred_l1_names: Sequence[str] = (),
) -> None:
    """Prepare staged complex files and ligand anchor names without VMD."""
    _write_python_prep_script_marker(workdir)
    u = mda.Universe(str(workdir / "aligned_amber.pdb"))
    receptor_backbone = u.select_atoms(
        f"(not resname {mol}) and resid {first_resid} to {last_resid} and name CA C N O"
    )
    if receptor_backbone.n_atoms == 0:
        raise RuntimeError("anchor not found")

    core = u.select_atoms(
        f"resid {first_resid} to {last_resid} and not water and not resname {mol} and not name H*"
    )
    lig = u.select_atoms(f"resname {mol}")
    water = u.select_atoms("resname WAT")
    others = _resname_group(u, other_mol)
    lipids = _resname_group(u, lipid_mol)
    ions = u.select_atoms("resname Na+ Cl- K+")
    all_atoms = _unique_atomgroup(u, core, lig, others, water, lipids, ions)
    shift = -_center_of_atoms(receptor_backbone)
    all_atoms.positions = all_atoms.positions + shift
    filini = workdir / f"{stage}-{mol}-ini.pdb"
    filpdb = workdir / f"{stage}-{mol}.pdb"
    _write_atomgroup_pdb(all_atoms, filini)
    shutil.copy2(filini, filpdb)

    prep_u = mda.Universe(str(filpdb))
    lig = prep_u.select_atoms(f"resname {mol}")
    if lig.n_atoms == 0:
        raise RuntimeError("anchor not found")
    lig.chainIDs = "S"
    lig.residues.resids = 1
    _write_atomgroup_pdb(lig, workdir / f"{mol}.pdb")
    lig_noh = lig.select_atoms("not name H*")
    _write_atomgroup_pdb(lig_noh, workdir / f"{mol}-noh.pdb")
    prep_u.atoms.write(str(filpdb))

    lig_heavy_names = _dedupe_names(
        str(atom.name) for atom in lig if not _atom_is_hydrogen(atom)
    )
    lig_heavy_name_set = set(lig_heavy_names)
    available_ligand_names = [
        name
        for name in _dedupe_names([str(name) for name in ligand_names])
        if name in lig_heavy_name_set
        and _ligand_atom_by_name(prep_u, mol, name).n_atoms > 0
    ]
    if not available_ligand_names:
        available_ligand_names = lig_heavy_names
    if lig_noh.n_atoms < 3 or len(available_ligand_names) < 3:
        anchors = available_ligand_names[: max(1, min(3, len(available_ligand_names)))]
    else:
        anchors = _pick_ligand_anchor_names(
            u=prep_u,
            mol=mol,
            ligand_names=available_ligand_names,
            preferred_l1_names=preferred_l1_names,
            p1_resid=p1_vmd,
            p1_atom=p1_atom,
            p2_resid=p2_vmd,
            p2_atom=p2_atom,
            l1_x=float(l1_x),
            l1_y=float(l1_y),
            l1_z=float(l1_z),
            l1_range=float(l1_range),
            min_adis=float(min_adis),
            max_adis=float(max_adis),
        )
    (workdir / "anchors.txt").write_text(" ".join(anchors) + "\n")

    dum = mda.Universe(str(workdir / "dum.pdb"))
    dum_atoms = dum.atoms
    dummy_center = _center_of_atoms(dum_atoms)
    receptor_center = _center_of_atoms(
        prep_u.select_atoms(
            f"(not resname {mol}) and resid {first_resid} to {last_resid} and name CA C N O"
        )
    )
    dum_atoms.positions = dum_atoms.positions + (receptor_center - dummy_center)
    _write_atomgroup_pdb(dum_atoms, workdir / "dum1.pdb")

    if float(sdr_dist) != 0.0:
        dum2 = mda.Universe(str(workdir / "dum.pdb"))
        dum2_atoms = dum2.atoms
        shifted_ligand_center = _center_of_atoms(lig_noh) + np.asarray(
            [0.0, 0.0, float(sdr_dist)],
            dtype=float,
        )
        dum2_atoms.positions = dum2_atoms.positions + (
            shifted_ligand_center - _center_of_atoms(dum2_atoms)
        )
        dum2_atoms.residues.resids = 2
        _write_atomgroup_pdb(dum2_atoms, workdir / "dum2.pdb")


def _sdf_heavy_atom_ordinals(sdf_file: str | Path) -> tuple[int | None, dict[int, int]]:
    """Build a heavy-atom ordinal map for an SDF molecule.

    Parameters
    ----------
    sdf_file
        Path to the ligand SDF file. Hydrogens are preserved when reading so
        returned indices match the original SDF atom ordering.

    Returns
    -------
    tuple[int | None, dict[int, int]]
        Total SDF atom count and a mapping from original SDF atom index to its
        zero-based heavy-atom ordinal. Returns ``(None, {})`` when RDKit is not
        available or the file cannot be read.
    """
    try:
        from rdkit import Chem
    except Exception:
        return None, {}

    try:
        supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
    except Exception:
        return None, {}
    if not mols:
        return None, {}

    mol = mols[0]
    heavy_ordinals: dict[int, int] = {}
    heavy_ordinal = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        heavy_ordinals[int(atom.GetIdx())] = heavy_ordinal
        heavy_ordinal += 1
    return int(mol.GetNumAtoms()), heavy_ordinals


def _map_sdf_atom_indices_to_ligand_names(
    sdf_file: str | Path,
    ligand_atoms,
    atom_indices: Sequence[int],
) -> tuple[list[str], list[int], int | None]:
    lig_names = [str(name) for name in ligand_atoms.names]
    sdf_atom_count, heavy_ordinals = _sdf_heavy_atom_ordinals(sdf_file)
    heavy_names = [
        str(atom.name)
        for atom in ligand_atoms
        if not _atom_is_hydrogen(atom)
    ]

    names: list[str] = []
    dropped: list[int] = []
    if heavy_ordinals and heavy_names:
        for idx in atom_indices:
            heavy_ordinal = heavy_ordinals.get(idx)
            if heavy_ordinal is not None and 0 <= heavy_ordinal < len(heavy_names):
                names.append(heavy_names[heavy_ordinal])
            else:
                dropped.append(idx)
    elif sdf_atom_count == len(lig_names):
        for idx in atom_indices:
            if 0 <= idx < len(lig_names):
                names.append(lig_names[idx])
            else:
                dropped.append(idx)
    else:
        for idx in atom_indices:
            if 0 <= idx < len(lig_names):
                names.append(lig_names[idx])
            else:
                dropped.append(idx)

    return names, dropped, sdf_atom_count


def _sdf_formally_charged_atom_indices(sdf_file: str | Path) -> list[int]:
    try:
        from rdkit import Chem
    except Exception:
        return []

    try:
        supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
    except Exception:
        return []
    if not mols:
        return []

    charged: list[int] = []
    for atom in mols[0].GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if int(atom.GetFormalCharge()) != 0:
            charged.append(int(atom.GetIdx()))
    return charged


def _sdf_formal_charge_by_ligand_atom_name(
    sdf_file: str | Path,
    ligand_atoms,
) -> dict[str, int]:
    charged_indices = _sdf_formally_charged_atom_indices(sdf_file)
    if not charged_indices:
        return {}

    try:
        from rdkit import Chem
        supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
    except Exception:
        return {}
    if not mols:
        return {}

    charges: dict[str, int] = {}
    for idx in charged_indices:
        names, _dropped, _sdf_atom_count = _map_sdf_atom_indices_to_ligand_names(
            sdf_file,
            ligand_atoms,
            [idx],
        )
        if not names:
            continue
        clean = str(names[0]).strip()
        if not clean:
            continue
        charge = int(mols[0].GetAtomWithIdx(int(idx)).GetFormalCharge())
        if charge:
            charges[clean] = charge
    return charges


def _protein_initial_salt_bridge_atom_charge(atom) -> int:
    key = (str(atom.resname).upper(), str(atom.name).upper())
    if key in _PROTEIN_POSITIVE_SALT_ATOMS:
        return 1
    if key in _PROTEIN_NEGATIVE_SALT_ATOMS:
        return -1
    return 0


def _initial_pose_salt_bridge_ligand_atom_names(
    *,
    sdf_file: str | Path,
    ligand_atoms,
    protein_atoms,
    distance_cutoff: float = _INITIAL_SALT_BRIDGE_DISTANCE_CUTOFF,
) -> list[str]:
    """Return ligand atom names in an initial-pose salt bridge, ordered by distance."""
    try:
        ligand_charges = _sdf_formal_charge_by_ligand_atom_name(sdf_file, ligand_atoms)
    except Exception:
        return []
    if not ligand_charges:
        return []

    ligand_atoms_by_name = {
        str(atom.name).strip(): atom
        for atom in ligand_atoms
        if str(atom.name).strip() in ligand_charges
    }
    if not ligand_atoms_by_name:
        return []

    contacts: list[tuple[float, str]] = []
    for protein_atom in protein_atoms:
        protein_charge = _protein_initial_salt_bridge_atom_charge(protein_atom)
        if protein_charge == 0:
            continue
        for ligand_name, ligand_atom in ligand_atoms_by_name.items():
            ligand_charge = int(ligand_charges.get(ligand_name, 0))
            if ligand_charge == 0 or protein_charge * ligand_charge >= 0:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(protein_atom.position, dtype=float)
                    - np.asarray(ligand_atom.position, dtype=float)
                )
            )
            if distance <= float(distance_cutoff):
                contacts.append((distance, ligand_name))

    names: list[str] = []
    seen: set[str] = set()
    for _distance, ligand_name in sorted(contacts, key=lambda item: (item[0], item[1])):
        if ligand_name in seen:
            continue
        seen.add(ligand_name)
        names.append(ligand_name)
    return names


def _candidate_ligand_atom_name_string(
    sdf_file: str | Path,
    ligand_atoms,
    *,
    ligand_label: str,
    stage: str,
) -> str:
    """Map RDKit candidate atom indices to final ligand atom names.

    Parameters
    ----------
    sdf_file
        Path to the ligand SDF used to generate candidate atom indices.
    ligand_atoms
        MDAnalysis atom group for the ligand as it appears in the final prepared
        PDB.
    ligand_label
        User-facing ligand identifier used for diagnostics.
    stage
        Build stage label used for diagnostics.

    Returns
    -------
    str
        Space-separated ligand atom names for Python anchor selection.

    Notes
    -----
    ``get_ligand_candidates`` returns RDKit atom indices from the SDF. Those are
    valid only when the final ligand atom list still has the same full-H atom
    order. If hydrogens were removed during prep, map candidates through heavy
    atom ordinals instead of indexing into the shorter PDB atom-name array.
    """
    lig_names = [str(name) for name in ligand_atoms.names]
    if not lig_names:
        raise ValueError(
            f"No atoms with ligand residue were found while preparing {ligand_label} ({stage})."
        )

    candidate_indices = [int(idx) for idx in get_ligand_candidates(str(sdf_file))]
    names, dropped, sdf_atom_count = _map_sdf_atom_indices_to_ligand_names(
        sdf_file,
        ligand_atoms,
        candidate_indices,
    )
    heavy_names = [
        str(atom.name)
        for atom in ligand_atoms
        if not _atom_is_hydrogen(atom)
    ]
    heavy_name_set = set(heavy_names)
    if heavy_name_set:
        hydrogen_names = [name for name in names if name not in heavy_name_set]
        if hydrogen_names:
            logger.warning(
                "[build_complex] Ignored {} hydrogen ligand candidate atom name(s) "
                "for {} ({}): {}.",
                len(hydrogen_names),
                ligand_label,
                stage,
                ", ".join(hydrogen_names[:10]),
            )
            names = [name for name in names if name in heavy_name_set]

    if dropped:
        logger.warning(
            "[build_complex] Ignored {} stale ligand candidate atom index/indices "
            "for {} ({}): {}. SDF atom count={}, final ligand atom count={}.",
            len(dropped),
            ligand_label,
            stage,
            ", ".join(str(idx) for idx in dropped[:10]),
            sdf_atom_count if sdf_atom_count is not None else "unknown",
            len(lig_names),
        )
    if not names:
        logger.warning(
            "[build_complex] No mapped ligand candidate atom names for {} ({}); "
            "using all final heavy ligand atoms for anchor search.",
            ligand_label,
            stage,
        )
        names = heavy_names if heavy_names else lig_names

    return " ".join(names)


def _is_apo_ligand_build(param_json: Path, ligand: str, mol: str) -> bool:
    try:
        metadata = json.loads(param_json.read_text())
    except Exception:
        metadata = {}
    if bool(metadata.get("apo")):
        return True
    return ligand.upper() == "APO" and mol.upper() == "APO"


def _write_ligand_pdb_with_parameter_names(
    ligand_pdb: Path,
    parameter_mol2: Path,
    output_pdb: Path,
    *,
    residue_name: str,
    ligand_label: str,
    apo_ligand: bool = False,
) -> None:
    """Write a ligand PDB whose atom names/count match its parameter mol2."""
    ante_mol = mda.Universe(str(parameter_mol2))
    lig_u = mda.Universe(str(ligand_pdb))

    if lig_u.atoms.n_atoms == ante_mol.atoms.n_atoms:
        output_atoms = lig_u.atoms
    elif apo_ligand and ante_mol.atoms.n_atoms == 1 and lig_u.atoms.n_atoms >= 1:
        logger.info(
            "[build_complex] Collapsing apo dummy ligand {} from {} source atoms "
            "to the single parameterized dummy atom.",
            ligand_label,
            lig_u.atoms.n_atoms,
        )
        output_atoms = lig_u.atoms[:1]
    else:
        raise ValueError(
            f"Ligand atom count mismatch for {ligand_label}: "
            f"{ligand_pdb} has {lig_u.atoms.n_atoms} atom(s), but "
            f"{parameter_mol2} has {ante_mol.atoms.n_atoms} atom(s)."
        )

    output_atoms.names = ante_mol.atoms.names
    output_atoms.residues.resnames = residue_name
    output_atoms.write(str(output_pdb))


def _copy_if_distinct(src: Path, dst: Path) -> None:
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass
    shutil.copy2(src, dst)


_STABLE_BORESCH_DISTANCE_JSON = "stable_boresch_distance.json"
_STABLE_BORESCH_DISTANCE_SCHEMA_VERSION = 8


def _user_anchor_triplet_was_provided(extra: dict | None) -> bool:
    if not extra:
        return False
    anchors = extra.get("user_anchor_atoms") or ()
    return sum(1 for anchor in anchors if str(anchor).strip()) >= 3


def _load_stable_boresch_distance(equil_dir: Path) -> dict | None:
    path = equil_dir / _STABLE_BORESCH_DISTANCE_JSON
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Ignoring unreadable stable Boresch distance {}: {}",
            path,
            exc,
        )
        return None
    if not isinstance(data, dict):
        logger.warning(
            "[build_complex_z] Ignoring malformed stable Boresch distance {}", path
        )
        return None
    try:
        schema_version = int(data.get("schema_version", 0))
    except Exception:
        schema_version = 0
    if schema_version < _STABLE_BORESCH_DISTANCE_SCHEMA_VERSION:
        logger.warning(
            "[build_complex_z] Ignoring stale stable Boresch distance {} "
            "(schema_version={}, expected >= {}).",
            path,
            schema_version,
            _STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
        )
        return None
    if data.get("usable") is False:
        logger.debug(
            "[build_complex_z] Stable Boresch distance {} is marked unusable: {}",
            path,
            data.get("reason", "no reason recorded"),
        )
        return None
    return data


def _stable_boresch_distance_candidates(stable_record: dict) -> list[dict]:
    ranked = stable_record.get("ranked_pairs")
    if isinstance(ranked, list):
        candidates = [item for item in ranked if isinstance(item, dict)]
        if candidates:
            return candidates
    return [stable_record]


def _stable_salt_bridge_ligand_atom_names(stable_record: dict | None) -> list[str]:
    if not isinstance(stable_record, dict):
        return []
    preference = stable_record.get("salt_bridge_preference")
    if not isinstance(preference, dict):
        return []
    return _dedupe_names(preference.get("ligand_atom_names") or [])


def _renumber_stable_protein_residue(
    *,
    renum_data: pd.DataFrame | None,
    stable_resid: int,
    stable_resname: str,
    stable_chain: str,
) -> int:
    if renum_data is None:
        return int(stable_resid)

    matches = renum_data.query("old_resid == @stable_resid")
    if matches.empty:
        return int(stable_resid)

    chain = str(stable_chain).strip()
    if chain:
        chain_matches = matches.query("old_chain == @chain")
        if not chain_matches.empty:
            matches = chain_matches

    resname = str(stable_resname).strip()
    if resname:
        resname_matches = matches.query(
            "old_resname == @resname or new_resname == @resname"
        )
        if not resname_matches.empty:
            matches = resname_matches

    # +1 matches the receptor numbering written into equil-<mol>.pdb because
    # the dummy atom occupies residue 1 in the VMD prep system.
    return int(matches["new_resid"].values[0]) + 1


def _apply_stable_boresch_distance_preference(
    *,
    u: mda.Universe,
    mol: str,
    stable_record: dict,
    P1: str,
    P2: str,
    P3: str,
    lig_name_str: str,
    l1_x: float,
    l1_y: float,
    l1_z: float,
    l1_range: float,
    renum_data: pd.DataFrame | None = None,
) -> dict | None:
    protein = stable_record.get("protein") or {}
    ligand = stable_record.get("ligand") or {}
    if not isinstance(protein, dict) or not isinstance(ligand, dict):
        return None

    try:
        stable_original_resid = int(protein["resid"])
        stable_resname = str(protein.get("resname", "")).strip()
        stable_protein_name = str(protein["name"]).strip()
        stable_ligand_name = str(ligand["name"]).strip()
    except Exception:
        logger.warning(
            "[build_complex_z] Stable Boresch distance JSON lacks atom metadata."
        )
        return None
    if not stable_protein_name or not stable_ligand_name:
        return None

    stable_chain = str(protein.get("segid") or protein.get("chainID") or "").strip()
    stable_resid = _renumber_stable_protein_residue(
        renum_data=renum_data,
        stable_resid=stable_original_resid,
        stable_resname=stable_resname,
        stable_chain=stable_chain,
    )

    candidate_names = [name for name in lig_name_str.split() if name]
    if stable_ligand_name not in candidate_names:
        logger.warning(
            "[build_complex_z] Stable ligand atom {} is not in the current Boresch "
            "candidate set {}; keeping default ligand-anchor search.",
            stable_ligand_name,
            " ".join(candidate_names),
        )
        return None

    selection = (
        f"(not resname {mol}) and resid {stable_resid} and name {stable_protein_name}"
    )
    stable_protein = u.select_atoms(selection)
    if stable_protein.n_atoms != 1:
        stable_protein = u.select_atoms(
            f"protein and resid {stable_resid} and name {stable_protein_name}"
        )
    if stable_protein.n_atoms != 1:
        logger.warning(
            "[build_complex_z] Stable protein atom selection {} matched {} atom(s); "
            "keeping default receptor P1.",
            selection,
            stable_protein.n_atoms,
        )
        return None

    atom = stable_protein[0]
    stable_P1 = f":{int(atom.resid)}@{atom.name}"
    if stable_P1 in {P2, P3}:
        logger.warning(
            "[build_complex_z] Stable protein atom {} duplicates P2/P3; "
            "keeping default receptor anchors.",
            stable_P1,
        )
        return None

    vector = stable_record.get("vector") or {}
    try:
        vector_mean = [float(x) for x in vector.get("mean", [])]
        if len(vector_mean) != 3:
            raise ValueError
        l1_x_new, l1_y_new, l1_z_new = vector_mean
    except Exception:
        l1_x_new, l1_y_new, l1_z_new = float(l1_x), float(l1_y), float(l1_z)

    try:
        distance_std = float((stable_record.get("distance") or {}).get("std", 0.0))
    except Exception:
        distance_std = 0.0
    l1_range_new = max(float(l1_range), 2.0 + 3.0 * max(distance_std, 0.0))

    preferred_names = [stable_ligand_name] + [
        name for name in candidate_names if name != stable_ligand_name
    ]
    return {
        "P1": stable_P1,
        "stable_original_P1": f":{stable_original_resid}@{stable_protein_name}",
        "p1_resid": str(int(atom.resid)),
        "p1_atom": str(atom.name),
        "p1_vmd": str(int(atom.resid)),
        "lig_name_str": " ".join(preferred_names),
        "l1_x": l1_x_new,
        "l1_y": l1_y_new,
        "l1_z": l1_z_new,
        "l1_range": l1_range_new,
        "stable_ligand_name": stable_ligand_name,
    }


def _stable_preference_has_safe_boresch_frame(
    *,
    u: mda.Universe,
    mol: str,
    preference: dict,
    P2: str,
    P3: str,
) -> bool:
    receptor_atoms = []
    for mask in (preference["P1"], P2, P3):
        atom = _atom_from_anchor_mask(u, mask, mol=mol, ligand=False)
        if atom is None:
            logger.debug(
                "[build_complex_z] Stable-pair receptor anchor {} could not be "
                "resolved for full-frame check.",
                mask,
            )
            return False
        receptor_atoms.append(atom)

    residue = _ligand_residue_for_boresch_guard(u, mol=mol, lig_resid="")
    if residue is None:
        logger.debug(
            "[build_complex_z] Ligand residue {} could not be resolved for "
            "stable-pair full-frame check.",
            mol,
        )
        return False

    stable_ligand_name = str(preference["stable_ligand_name"]).strip()
    triplet_candidates: list[dict] = []
    try:
        triplet_candidates = _preferred_l1_ligand_triplet_candidates(
            residue,
            [stable_ligand_name],
        )
        preferred_triplet = _best_preferred_l1_triplet_for_receptor_frame(
            residue=residue,
            receptor_atoms=receptor_atoms,
            preferred_first_names=[stable_ligand_name],
            triplet_candidates=triplet_candidates,
        )
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not run Boresch frame guard while checking "
            "stable pair {}: {}",
            stable_ligand_name,
            exc,
        )
        return True
    if preferred_triplet is None:
        alternate = _select_receptor_p2_p3_for_preferred_l1(
            u=u,
            mol=mol,
            ligand_label="ABFE stable-pair precheck",
            residue=residue,
            p1_atom=receptor_atoms[0],
            current_p2_atom=receptor_atoms[1],
            current_p3_atom=receptor_atoms[2],
            preferred_first_names=[stable_ligand_name],
            triplet_candidates=triplet_candidates,
        )
        if alternate is not None:
            logger.debug(
                "[build_complex_z] Stable pair P1={} L1={} requires alternate "
                "P2/P3 for a full-frame Boresch guard; accepting for later "
                "P2/P3 reselection.",
                preference.get("P1"),
                stable_ligand_name,
            )
            return True
        logger.debug(
            "[build_complex_z] Stable pair P1={} L1={} did not pass the "
            "current/alternate full-frame Boresch guard.",
            preference.get("P1"),
            stable_ligand_name,
        )
        return False
    return True


def _select_stable_boresch_distance_preference(
    *,
    u: mda.Universe,
    mol: str,
    stable_record: dict,
    P1: str,
    P2: str,
    P3: str,
    lig_name_str: str,
    l1_x: float,
    l1_y: float,
    l1_z: float,
    l1_range: float,
    renum_data: pd.DataFrame | None = None,
) -> dict | None:
    candidates = _stable_boresch_distance_candidates(stable_record)
    for rank, candidate in enumerate(candidates, start=1):
        preference = _apply_stable_boresch_distance_preference(
            u=u,
            mol=mol,
            stable_record=candidate,
            P1=P1,
            P2=P2,
            P3=P3,
            lig_name_str=lig_name_str,
            l1_x=l1_x,
            l1_y=l1_y,
            l1_z=l1_z,
            l1_range=l1_range,
            renum_data=renum_data,
        )
        if preference is None:
            continue
        if not _stable_preference_has_safe_boresch_frame(
            u=u,
            mol=mol,
            preference=preference,
            P2=P2,
            P3=P3,
        ):
            continue
        preference["stable_rank"] = rank
        preference["stable_candidate_count"] = len(candidates)
        return preference

    if candidates:
        logger.debug(
            "[build_complex_z] None of {} stable protein-ligand pair candidate(s) "
            "satisfied the full-frame Boresch guard; using default anchors.",
            len(candidates),
        )
    return None


def _unit_vector(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-8:
        return None
    return np.asarray(vec, dtype=float) / norm


def _perpendicular_unit_vector(vec: np.ndarray) -> np.ndarray:
    base = np.asarray([1.0, 0.0, 0.0], dtype=float)
    unit = _unit_vector(vec)
    if unit is not None and abs(float(np.dot(unit, base))) > 0.9:
        base = np.asarray([0.0, 1.0, 0.0], dtype=float)
    if unit is None:
        return base
    perp = np.cross(unit, base)
    perp_unit = _unit_vector(perp)
    return perp_unit if perp_unit is not None else base


def _apo_dummy_spacing(min_adis: float, max_adis: float) -> float:
    if max_adis > min_adis:
        return float((min_adis + max_adis) / 2.0)
    return float(max(4.0, min_adis + 1.0))


def _position_apo_dummy_atoms(
    pdb_file: Path,
    *,
    mol: str,
    p1_resid: int,
    p2_resid: int,
    h1_atom: str,
    h2_atom: str,
    l1_vector: np.ndarray,
    min_adis: float,
    max_adis: float,
) -> list[str]:
    """Place the apo dummy atom near the L1 reference and return its anchor name."""
    u = mda.Universe(str(pdb_file))
    lig_atoms = u.select_atoms(f"resname {mol}")
    if lig_atoms.n_atoms < 1:
        raise ValueError(
            f"Apo dummy ligand {mol} must contain at least one atom for an anchor placeholder."
        )

    p1 = u.select_atoms(
        f"(not resname {mol}) and (resid {p1_resid} and name {h1_atom})"
    )
    p2 = u.select_atoms(
        f"(not resname {mol}) and (resid {p2_resid} and name {h2_atom})"
    )
    if p1.n_atoms != 1 or p2.n_atoms != 1:
        raise ValueError(
            "Could not place apo dummy ligand: receptor P1/P2 anchors were not found "
            f"in {pdb_file}."
        )

    p1_pos = np.asarray(p1.positions[0], dtype=float)
    p2_pos = np.asarray(p2.positions[0], dtype=float)
    p1_to_p2 = p2_pos - p1_pos
    target_vec = np.asarray(l1_vector, dtype=float)
    target_dir = _unit_vector(target_vec)
    if target_dir is None:
        target_dir = _perpendicular_unit_vector(p1_to_p2)
        target_vec = target_dir * _apo_dummy_spacing(min_adis, max_adis)

    lig_atoms[0].position = p1_pos + target_vec
    u.atoms.write(str(pdb_file))
    return [str(lig_atoms[0].name)]


def _write_apo_anchor_outputs(
    build_dir: Path,
    *,
    ligand: str,
    mol: str,
    anchor_names: Sequence[str],
) -> None:
    missing = [
        path
        for path in (build_dir / f"equil-{mol}.pdb", build_dir / f"{mol}-noh.pdb")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Apo dummy prep did not produce required build output(s): "
            + ", ".join(str(path) for path in missing)
        )

    if len(anchor_names) < 1:
        raise ValueError("Apo dummy anchor output requires at least one atom name.")
    anchor_file = build_dir / "anchors.txt"
    anchor_file.write_text(" ".join(anchor_names) + "\n")

    tagged = build_dir / f"anchors-{ligand}.txt"
    if tagged.exists():
        tagged.unlink()
    anchor_file.rename(tagged)

    dum1 = build_dir / "dum1.pdb"
    if not dum1.exists():
        shutil.copy2(build_dir / "dum.pdb", dum1)


def build_complex(ctx: BuildContext, *, infe: bool = False) -> bool:
    """
    Creates the aligned + cleaned PDBs (protein/others/lipids), finds
    receptor/ligand anchors, generates `equil-<lig>.pdb` and
    `anchors-<ligand>.txt`. Returns False if anchors can’t be found.
    """
    sim = ctx.sim
    ligand = ctx.ligand
    mol = ctx.residue_name
    param_json = ctx.working_dir.parent / "params" / f"{mol}.json"
    apo_ligand = _is_apo_ligand_build(param_json, ligand, mol)

    # Pull many config knobs (renamed to locals for readability)
    H1 = sim.p1
    H2 = sim.p2
    H3 = sim.p3
    l1_x = sim.l1_x
    l1_y = sim.l1_y
    l1_z = sim.l1_z
    l1_range = sim.l1_range
    max_adis = sim.max_adis
    min_adis = sim.min_adis

    other_mol = sim.other_mol
    if not hasattr(sim, "lipid_mol"):
        raise AttributeError(
            "SimulationConfig is missing 'lipid_mol'. "
            "Please include it in the run configuration (use an empty list if not needed)."
        )
    lipid_mol = sim.lipid_mol
    if mol in other_mol or mol in lipid_mol:
        raise ValueError(
            f"The ligand {mol} cannot be in the other_mol/lipid_mol list: {other_mol} and {lipid_mol}"
        )

    logger.debug(
        f"[Equil] Building complex for ligand {ligand} with other_mol={other_mol} lipid_mol={lipid_mol}"
    )
    solv_shell = sim.solv_shell
    system_name = sim.system_name

    # Stage directories
    work = ctx.working_dir
    build_dir = ctx.build_dir
    run_dir = ctx.run_dir
    amber_dir = ctx.amber_dir
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(amber_dir, exist_ok=True)
    vmd_available = _executable_available(vmd)
    if not vmd_available:
        logger.warning(
            "VMD executable {!r} was not found; using Python fallbacks for "
            "build-complex split/fit/prep steps.",
            vmd,
        )

    # Copy baseline build templates
    shutil.copytree(build_files_orig, build_dir, dirs_exist_ok=True)

    # Bring the input ligand + reference files
    all_ligand_folder = ctx.system_root / "all-ligands"
    shutil.copy2(all_ligand_folder / "reference.pdb", build_dir / "reference.pdb")
    shutil.copy2(all_ligand_folder / f"{system_name}.pdb", build_dir / "rec_file.pdb")
    shutil.copy2(all_ligand_folder / f"{ligand}.pdb", build_dir / f"{ligand}.pdb")
    shutil.copy2(all_ligand_folder / f"{ligand}.pdb", work / f"{ligand}.pdb")

    # Ensure ligand atom names match antechamber mol2 (ligand.ff prepared earlier)
    shutil.copy2(work.parent / "params" / f"{mol}.mol2", build_dir / f"{mol}.mol2")
    shutil.copy2(work.parent / "params" / f"{mol}.sdf", build_dir / f"{mol}.sdf")

    _write_ligand_pdb_with_parameter_names(
        build_dir / f"{ligand}.pdb",
        build_dir / f"{mol}.mol2",
        build_dir / f"{mol}.pdb",
        residue_name=mol,
        ligand_label=ligand,
        apo_ligand=apo_ligand,
    )
    if apo_ligand:
        _copy_if_distinct(build_dir / f"{mol}.pdb", build_dir / f"{ligand}.pdb")
        _copy_if_distinct(build_dir / f"{mol}.pdb", work / f"{ligand}.pdb")

    # Prepare VMD split script
    split_ini = Path(build_dir / "split-ini.tcl")
    split_tcl = Path(build_dir / "split.tcl")
    with open(split_ini, "rt") as fin, open(split_tcl, "wt") as fout:
        other_mol_vmd = " ".join(other_mol) if other_mol else "XXX"
        lipid_mol_vmd = " ".join(lipid_mol) if lipid_mol else "XXX"
        for line in fin:
            if "lig" not in line:
                fout.write(
                    line.replace("SHLL", f"{solv_shell:4.2f}")
                    .replace("OTHRS", str(other_mol_vmd))
                    .replace("LIPIDS", str(lipid_mol_vmd))
                    .replace("MMM", f"'{mol}'")
                )

    if vmd_available:
        run_with_log(
            f"{vmd} -dispdev text -e {str(split_tcl)}",
            error_match="syntax error",
            shell=False,
            working_dir=build_dir,
        )
    else:
        _python_split_rec_file(
            workdir=build_dir,
            mol=mol,
            solv_shell=float(solv_shell),
            other_mol=other_mol,
            lipid_mol=lipid_mol,
        )
    # Protein PDB cleanup with pdb4amber
    shutil.copy2(build_dir / "protein.pdb", build_dir / "protein_vmd.pdb")
    _run_pdb4amber(
        build_dir / "protein_vmd.pdb",
        build_dir / "protein.pdb",
        working_dir=build_dir,
    )

    renum_txt = build_dir / "protein_renum.txt"
    renum_data = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )

    # Original receptor numbering (to detect missing residues)
    u_original = mda.Universe(str(build_dir / "rec_file.pdb"))
    first_res = int(u_original.residues[0].resid)
    recep_resid_num = len(mda.Universe(str(build_dir / "protein.pdb")).residues)

    # Adjust protein anchors to new numbering
    def _extract_resid(atom_spec: str) -> tuple[int, str]:
        # e.g. ":113@CA" or "113@CA"
        if atom_spec.startswith(":"):
            atom_spec = atom_spec[1:]
        r_s, a = atom_spec.split("@")
        return int(r_s), a

    h1_resid, h1_atom = _extract_resid(H1)
    h2_resid, h2_atom = _extract_resid(H2)
    h3_resid, h3_atom = _extract_resid(H3)

    def _entry(old_res: int):
        protein_chain = "A"
        e = renum_data.query("old_resid == @old_res and old_chain == @protein_chain")
        return e if not e.empty else renum_data.query("old_resid == @old_res")

    h1_entry = _entry(h1_resid)
    h2_entry = _entry(h2_resid)
    h3_entry = _entry(h3_resid)
    if h1_entry.empty or h2_entry.empty or h3_entry.empty:
        renum_data.to_csv(build_dir / "protein_renum_err.txt", sep="\t", index=False)
        raise ValueError(
            f"Could not find one or more receptor anchors in protein sequence; "
            f"renumber map written to {build_dir/'protein_renum_err.txt'}"
        )

    # +1 due to dummy atom
    p1_resid = int(h1_entry["new_resid"].values[0]) + 1
    p2_resid = int(h2_entry["new_resid"].values[0]) + 1
    p3_resid = int(h3_entry["new_resid"].values[0]) + 1
    p1_vmd = f"{p1_resid}"
    p2_vmd = f"{p2_resid}"

    P1 = f":{p1_resid}@{h1_atom}"
    P2 = f":{p2_resid}@{h2_atom}"
    P3 = f":{p3_resid}@{h3_atom}"

    (build_dir / "protein_anchors.txt").write_text(f"{P1}\n{P2}\n{P3}\n")
    logger.debug(f"[Equil] Receptor anchors: P1={P1}, P2={P2}, P3={P3}")

    # Truncate 4-letter residue names for AMBER (co-binders, lipids)
    if any(x[:3] != x for x in other_mol):
        logger.warning("Co-binder residue names truncated to 3 letters for AMBER.")
    other_mol = [x[:3] for x in other_mol]
    if lipid_mol:
        lipid_mol = [x[:3] for x in lipid_mol]

    # Convert CHARMM lipids to lipid21 if membrane
    if sim.membrane_simulation:
        lipids_pdb = build_dir / "lipids.pdb"
        lipids_amber_pdb = build_dir / "lipids_amber.pdb"
        if not lipids_pdb.exists():
            raise FileNotFoundError(
                f"Expected membrane lipid PDB was not created: {lipids_pdb}"
            )
        if _lipids_need_charmm_to_amber_conversion(lipids_pdb):
            run_with_log(
                f"{charmmlipid2amber} -c {charmmlipid2amber_csv} "
                f"-i {lipids_pdb} -o {lipids_amber_pdb}"
            )
            lipid_action = "Converted CHARMM lipids to AMBER"
        else:
            shutil.copy2(lipids_pdb, lipids_amber_pdb)
            lipid_action = "Using already-AMBER lipid residues without conversion"
        u_lip = mda.Universe(str(lipids_amber_pdb))
        lipid_resnames = list(set(u_lip.residues.resnames))
        logger.debug(f"[Equil] {lipid_action}: {lipid_resnames}")
        lipid_mol = lipid_resnames  # updated list

    # Merge raw complex (protein + ligand + others + (lipids) + crystal waters)
    parts: list[Path] = [
        build_dir / "protein.pdb",
        build_dir / f"{mol}.pdb",
        build_dir / "others.pdb",
    ]
    if sim.membrane_simulation:
        parts.append(build_dir / "lipids_amber.pdb")
    parts.append(build_dir / "crystalwat.pdb")
    merged = build_dir / "complex-merge.pdb"
    with open(merged, "w") as fout:
        for p in parts:
            if p.exists():
                with open(p) as fin:
                    for line in fin:
                        fout.write(line)

    # Strip CRYST1/CONECT/END
    complex_pdb = build_dir / "complex.pdb"
    with open(merged) as f_in, open(complex_pdb, "w") as f_out:
        for line in f_in:
            if "CRYST1" in line or "CONECT" in line or line.startswith("END"):
                continue
            f_out.write(line)

    # Avoid chain swapping when aligning
    _run_pdb4amber(
        build_dir / "reference.pdb",
        build_dir / "reference_amber.pdb",
        working_dir=build_dir,
    )
    if vmd_available:
        run_with_log(
            f"{vmd} -dispdev text -e nochain.tcl", shell=False, working_dir=build_dir
        )
    else:
        _python_nochain_for_alignment(build_dir)
    run_with_log(
        "./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc",
        working_dir=build_dir,
    )
    if vmd_available:
        run_with_log(
            f"{vmd} -dispdev text -e measure-fit.tcl", shell=False, working_dir=build_dir
        )
    else:
        _python_measure_fit(workdir=build_dir)

    # Clean aligned and put in AMBER format
    with (
        open(build_dir / "aligned.pdb", "r") as oldfile,
        open(build_dir / "aligned-clean.pdb", "w") as newfile,
    ):
        for line in oldfile:
            if len(line.split()) > 4:
                newfile.write(line)
    _run_pdb4amber(
        build_dir / "aligned-clean.pdb",
        build_dir / "aligned_amber.pdb",
        working_dir=build_dir,
    )

    # For membrane: restore box info and re-merge lipid partial residues into single resids
    if sim.membrane_simulation:
        u_aln = mda.Universe(str(build_dir / "aligned_amber.pdb"))
        u_aln.dimensions = u_original.dimensions
        renum_txt2 = build_dir / "aligned_amber_renum.txt"
        ren2 = pd.read_csv(
            renum_txt2,
            sep=r"\s+",
            header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
        )
        key_chain = ren2["old_chain"].astype(str)
        key_resid = ren2["old_resid"].astype(int)

        boundary = key_chain.ne(key_chain.shift(1)) | key_resid.ne(key_resid.shift(1))
        revised = boundary.cumsum().to_numpy(dtype=int)

        # Safety check (the renum table should have one row per residue in the universe)
        if revised.size != u_aln.residues.n_residues:
            raise ValueError(
                f"Residue count mismatch: renum rows={revised.size} vs universe={u_aln.residues.n_residues}"
            )

        u_aln.residues.resids = revised
        u_aln.atoms.write(str(build_dir / "aligned_amber.pdb"))

    sdf_file = build_dir / f"{mol}.sdf"
    pdb_file = build_dir / "aligned_amber.pdb"
    apo_anchor_names: list[str] | None = None
    if apo_ligand:
        apo_anchor_names = _position_apo_dummy_atoms(
            pdb_file,
            mol=mol,
            p1_resid=p1_resid,
            p2_resid=p2_resid,
            h1_atom=h1_atom,
            h2_atom=h2_atom,
            l1_vector=np.asarray([l1_x, l1_y, l1_z], dtype=float),
            min_adis=float(min_adis),
            max_adis=float(max_adis),
        )
        logger.debug(
            "[build_complex] Placed apo dummy ligand '{}' at the anchor reference "
            "and will use {} as ligand anchor(s).",
            ligand,
            " ".join(apo_anchor_names),
        )

    u = mda.Universe(str(pdb_file))
    lig_atoms = u.select_atoms(f"resname {mol}")
    lig_names = lig_atoms.names
    lig_heavy_names = _dedupe_names(
        str(atom.name) for atom in lig_atoms if not _atom_is_hydrogen(atom)
    )
    lig_heavy_count = sum(1 for atom in lig_atoms if not _atom_is_hydrogen(atom))
    salt_bridge_lig_names: list[str] = []
    if apo_anchor_names is None:
        lig_name_str = _candidate_ligand_atom_name_string(
            sdf_file,
            lig_atoms,
            ligand_label=ligand,
            stage="equil",
        )
        salt_bridge_lig_names = _initial_pose_salt_bridge_ligand_atom_names(
            sdf_file=sdf_file,
            ligand_atoms=lig_atoms,
            protein_atoms=u.select_atoms(f"protein and not resname {mol}"),
        )
        if salt_bridge_lig_names:
            lig_name_str = " ".join(
                _order_ligand_names_with_priority(
                    lig_name_str.split(),
                    salt_bridge_lig_names,
                )
            )
            logger.debug(
                "[build_complex] Prioritizing initial-pose salt-bridge ligand "
                "atom(s) for {}: {}",
                ligand,
                " ".join(salt_bridge_lig_names),
            )
    else:
        lig_name_str = " ".join(apo_anchor_names)
    anchor_file = build_dir / "anchors.txt"

    def _anchor_names_from_file(path: Path) -> list[str]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            return path.read_text().splitlines()[0].split()
        except Exception:
            return []

    def _partial_ligand_anchors_are_expected(path: Path) -> bool:
        names = _anchor_names_from_file(path)
        if not names or len(names) >= 3:
            return False
        if lig_heavy_count >= 3:
            return False
        logger.debug(
            "Ligand {} has only {} non-hydrogen atom(s); using reduced ligand "
            "anchor set {} and omitting unavailable Boresch angle/dihedral terms.",
            ligand,
            lig_heavy_count,
            " ".join(names),
        )
        return True

    def _run_python_prep(ligand_name_str: str) -> None:
        _python_prep_complex(
            workdir=build_dir,
            mol=mol,
            p1_atom=h1_atom,
            p1_vmd=p1_vmd,
            p2_atom=h2_atom,
            p2_vmd=p2_vmd,
            first_resid="1",
            last_resid=str(recep_resid_num),
            stage="equil",
            l1_x=float(l1_x),
            l1_y=float(l1_y),
            l1_z=float(l1_z),
            l1_range=float(l1_range),
            min_adis=float(min_adis),
            max_adis=float(max_adis),
            sdr_dist=0.0,
            ligand_names=str(ligand_name_str).split(),
            other_mol=other_mol,
            lipid_mol=lipid_mol,
            preferred_l1_names=salt_bridge_lig_names,
        )

    if apo_ligand:
        _run_python_prep(lig_name_str)
        _write_apo_anchor_outputs(
            build_dir,
            ligand=ligand,
            mol=mol,
            anchor_names=apo_anchor_names or list(lig_names),
        )
        return True

    try:
        _run_python_prep(lig_name_str)
    except RuntimeError:
        if _partial_ligand_anchors_are_expected(anchor_file):
            pass
        else:
            # fallback: all heavy ligand atoms
            lig_name_str2 = " ".join(
                _order_ligand_names_with_priority(
                    lig_heavy_names if lig_heavy_names else [str(x) for x in lig_names],
                    salt_bridge_lig_names,
                )
            )
            _run_python_prep(lig_name_str2)

    # Verify anchors.txt
    if anchor_file.stat().st_size == 0:
        logger.warning(
            f"Could not find ligand L1 for {ligand}. Most likely not in binding site."
        )
        return False

    # Ensure we got 3 ligand anchors
    with open(anchor_file) as f:
        line = f.readline().strip()
    if len(line.split()) < 3:
        if _partial_ligand_anchors_are_expected(anchor_file):
            os.rename(anchor_file, build_dir / f"anchors-{ligand}.txt")
            return True
        os.rename(anchor_file, build_dir / f"anchors-{ligand}.txt")
        logger.warning(
            f"Could not find ligand L2/L3 anchors for {ligand}. Try reducing min_adis."
        )
        return False

    os.rename(anchor_file, build_dir / f"anchors-{ligand}.txt")
    return True


@register_build_complex("d")
@register_build_complex("l")
@register_build_complex("z")
def build_complex_z(ctx) -> bool:
    """
    Z-component _build_complex:
    Copy/transform files from the per-ligand equil output, then detect/emit anchors.
    Returns True on success, False to indicate pruning.
    """
    # --- config / context ---
    ligand = ctx.ligand
    mol = ctx.residue_name
    sim = ctx.sim

    solv_shell = sim.solv_shell
    l1_x, l1_y, l1_z = sim.l1_x, sim.l1_y, sim.l1_z
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol
    l1_range = sim.l1_range
    max_adis = sim.max_adis
    min_adis = sim.min_adis
    buffer_z = sim.buffer_z

    hmr = sim.hmr
    membrane_builder = sim.membrane_simulation

    workdir = ctx.build_dir
    workdir.mkdir(parents=True, exist_ok=True)
    vmd_available = _executable_available(vmd)
    if not vmd_available:
        logger.warning(
            "VMD executable {!r} was not found; using Python fallbacks for build-complex "
            "split/fit/prep steps.",
            vmd,
        )
    child_root = ctx.working_dir  # .../simulations/<LIG>/fe/...
    sys_root = ctx.system_root  # .../work/<system>
    equil_dir = (
        sys_root / "simulations" / ligand / "equil"
    )  # .../work/<system>/simulations/<LIG>/equil
    ff_dir = (
        sys_root / "simulations" / ligand / "params"
    )  # .../work/<system>/simulations/<LIG>/params

    shutil.copytree(build_files_orig, workdir, dirs_exist_ok=True)

    # --- helpers to keep paths explicit ---
    def _p(name: str) -> Path:
        return workdir / name

    def _copy(src: Path, dst_name: str):
        if src.exists():
            shutil.copy2(src, _p(dst_name))
        else:
            raise FileNotFoundError(f"Missing required file: {src}")

    def _copy_first_existing(candidates: Sequence[Path], dst_name: str):
        for src in candidates:
            if src.exists():
                shutil.copy2(src, _p(dst_name))
                return src
        raise FileNotFoundError(
            "Missing required file; tried: "
            + ", ".join(str(src) for src in candidates)
        )

    # 1) copy artifacts from equil
    _copy_first_existing(
        (
            equil_dir / "q_build_files" / f"{ligand}.pdb",
            equil_dir / f"{ligand}.pdb",
            equil_dir / f"{mol}.pdb",
        ),
        f"{ligand}.pdb",
    )
    _copy(equil_dir / "representative.rst7", "representative.rst7")
    _copy(equil_dir / "representative.pdb", "aligned-nc.pdb")
    _copy(equil_dir / "build_amber_renum.txt", "build_amber_renum.txt")
    _copy_first_existing(
        (
            equil_dir / "q_build_files" / "protein_renum.txt",
            equil_dir / "protein_renum.txt",
        ),
        "protein_renum.txt",
    )

    for p in equil_dir.glob("full*.prmtop"):
        shutil.copy2(p, _p(p.name))
    for p in equil_dir.glob("vac*"):
        shutil.copy2(p, _p(p.name))

    # 2) Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf"):
        src = ff_dir / f"{mol}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_z] Missing ligand FF file: {src}")
        shutil.copy2(src, workdir / src.name)

    # 3) materialize the representative structure for split/rewrite steps.
    if _executable_available(cpptraj):
        run_with_log(
            f"{cpptraj} -p full.prmtop -y representative.rst7 -x rec_file.pdb",
            working_dir=workdir,
        )
    else:
        logger.warning(
            "cpptraj executable {!r} was not found; using representative.pdb "
            "as rec_file.pdb for build-complex FE setup.",
            cpptraj,
        )
        shutil.copy2(_p("aligned-nc.pdb"), _p("rec_file.pdb"))

    # 4) reapply chain IDs from renum map; optional lipid resid compaction
    renum = pd.read_csv(
        _p("build_amber_renum.txt"),
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    u = mda.Universe(str(_p("rec_file.pdb")))
    for residue in u.select_atoms("protein").residues:
        resid_str = residue.resid
        chain = renum.query("old_resid == @resid_str").old_chain.values
        if chain.size:
            residue.atoms.chainIDs = chain[0]

    if membrane_builder:
        # also skip ANC, which is a anchored dummy atom for rmsf restraint
        non_water_ag = u.select_atoms("not resname WAT Na+ Cl- K+ ANC")
        # fix lipid resids
        revised_resids = revised_resids_for_lipid_fragments(
            (
                (row["old_resname"], row["old_chain"], row["old_resid"])
                for _, row in renum.iterrows()
                if row["old_resname"] not in ["WAT", "Na+", "Cl-", "K+"]
            ),
            lipid_mol,
        )

        revised_resids = np.array(revised_resids)
        total_residues = non_water_ag.residues.n_residues
        final_resids = np.zeros(total_residues, dtype=int)
        final_resids[: len(revised_resids)] = revised_resids
        next_resnum = revised_resids[-1] + 1
        final_resids[len(revised_resids) :] = np.arange(
            next_resnum, total_residues - len(revised_resids) + next_resnum
        )
        non_water_ag.residues.resids = final_resids

    u.atoms.write(str(_p("rec_file.pdb")))
    shutil.copy2(_p("rec_file.pdb"), _p("equil-reference.pdb"))

    # 5) VMD split -> split.tcl generated under workdir
    other_mol_vmd = " ".join(other_mol) if other_mol else "XXX"
    lipid_mol_vmd = " ".join(lipid_mol) if lipid_mol else "XXX"
    with open(_p("split-ini.tcl"), "rt") as fin, open(_p("split.tcl"), "wt") as fout:
        for line in fin:
            if membrane_builder and line.startswith("set wat "):
                fout.write('set wat [atomselect 0 "water"]\n')
                continue
            if membrane_builder and line.startswith("set ion "):
                fout.write('set ion [atomselect 0 "resname \'Na+\' \'Cl-\' \'K+\'"]\n')
                continue
            fout.write(
                line.replace("SHLL", f"{solv_shell:4.2f}")
                .replace("OTHRS", str(other_mol_vmd))
                .replace("LIPIDS", str(lipid_mol_vmd))
                .replace("mmm", mol)
                .replace("MMM", mol)
            )
    if vmd_available:
        run_with_log(f"{vmd} -dispdev text -e split.tcl", shell=False, working_dir=workdir)
    else:
        _python_split_rec_file(
            workdir=workdir,
            mol=mol,
            solv_shell=float(solv_shell),
            other_mol=other_mol,
            lipid_mol=lipid_mol,
            keep_all_waters=bool(membrane_builder),
        )

    # 6) merge -> complex.pdb (strip headers/CRYST1/CONECT/END)
    pieces = [
        "dummy.pdb",
        "protein.pdb",
        f"{mol}.pdb",
        "lipids.pdb",
        "others.pdb",
        "crystalwat.pdb",
    ]
    if not all(_p(f).exists() for f in pieces):
        missing = [f for f in pieces if not _p(f).exists()]
        raise FileNotFoundError(f"Missing split output files: {', '.join(missing)}")
    (_p("complex-merge.pdb")).write_text("".join((_p(f).read_text()) for f in pieces))
    with open(_p("complex-merge.pdb")) as fin, open(_p("complex.pdb"), "wt") as fout:
        for ln in fin:
            if ("CRYST1" in ln) or ("CONECT" in ln) or ln.startswith("END"):
                continue
            fout.write(ln)

    # 7) read anchors/meta from equil header
    equil_info = equil_dir / f"equil-{mol}.pdb"
    if not equil_info.exists():
        raise FileNotFoundError(f"Missing {equil_info}")
    with equil_info.open() as f:
        data = f.readline().split()
        P1, P2, P3 = data[2].strip(), data[3].strip(), data[4].strip()
        first_res, recep_last = data[8].strip(), data[9].strip()
    p1_resid = P1.split("@")[0][1:]
    p1_atom = P1.split("@")[1]
    rec_res = int(recep_last) + 1
    p1_vmd = p1_resid

    p2_resid = P2.split("@")[0][1:]
    p2_atom = P2.split("@")[1]
    p2_vmd = p2_resid
    renum_data = pd.read_csv(
        _p("protein_renum.txt"),
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )

    # 8) SDR distance
    if buffer_z <= 20:
        buffer_z = 20
        logger.debug(
            f"buffer_z too small ({sim.buffer_z}); setting to 20 Å for SDR calculation."
        )

    sdr_dist, abs_z, buffer_z_left = get_sdr_dist(
        str(_p("complex.pdb")), lig_resname=mol, buffer_z=buffer_z
    )
    # save for future stages
    with open(_p("sdr_info.txt"), "wt") as f:
        f.write(f"{sdr_dist}\n{abs_z}\n{buffer_z_left}\n")
    logger.debug(f"[build_complex_z] SDR distance: {sdr_dist:.2f} Å, abs_z: {abs_z:.2f} Å, buffer_z_left: {buffer_z_left:.2f} Å")

    # 9) align & pdb4amber
    if vmd_available:
        run_with_log(
            f"{vmd} -dispdev text -e measure-fit.tcl", shell=False, working_dir=workdir
        )
    else:
        _python_measure_fit(workdir=workdir)
    with open(_p("aligned.pdb")) as fin, open(_p("aligned-clean.pdb"), "wt") as fout:
        for ln in fin:
            if len(ln.split()) > 3:
                fout.write(ln)
    if membrane_builder:
        shutil.copy2(_p("aligned-clean.pdb"), _p("aligned_amber.pdb"))
    else:
        _run_pdb4amber(
            _p("aligned-clean.pdb"),
            _p("aligned_amber.pdb"),
            working_dir=workdir,
        )
    u = mda.Universe(str(_p("aligned_amber.pdb")))

    # optional lipid resid fix post-amber
    if membrane_builder:
        non_water_ag = u.select_atoms("not resname WAT Na+ Cl- K+")
        non_water_ag.residues.resids = final_resids

        u.atoms.write(_p("aligned_amber.pdb"))
        u = mda.Universe(str(_p("aligned_amber.pdb")))

    # 10) ligand candidates for Boresch
    sdf_file = _p(f"{mol}.sdf")
    lig_atoms = u.select_atoms(f"resname {mol}")
    lig_names = lig_atoms.names
    lig_heavy_names = _dedupe_names(
        str(atom.name) for atom in lig_atoms if not _atom_is_hydrogen(atom)
    )
    lig_heavy_count = sum(1 for atom in lig_atoms if not _atom_is_hydrogen(atom))
    lig_name_str = _candidate_ligand_atom_name_string(
        sdf_file,
        lig_atoms,
        ligand_label=ligand,
        stage="fe-z",
    )
    salt_bridge_lig_names: list[str] = []
    default_anchor_state = {
        "P1": P1,
        "p1_resid": p1_resid,
        "p1_atom": p1_atom,
        "p1_vmd": p1_vmd,
        "lig_name_str": lig_name_str,
        "l1_x": l1_x,
        "l1_y": l1_y,
        "l1_z": l1_z,
        "l1_range": l1_range,
    }
    stable_preference_applied = False
    stable_preference = None
    stable_record = None

    extra = dict(ctx.extra or {})
    stable_record = _load_stable_boresch_distance(equil_dir)
    if stable_record is not None:
        salt_bridge_lig_names = _stable_salt_bridge_ligand_atom_names(stable_record)
        if salt_bridge_lig_names:
            lig_name_str = " ".join(
                _order_ligand_names_with_priority(
                    lig_name_str.split(),
                    salt_bridge_lig_names,
                )
            )
            logger.debug(
                "[build_complex_z] Prioritizing salt-bridge ligand atom(s) "
                "from equil analysis for {}: {}",
                ligand,
                " ".join(salt_bridge_lig_names),
            )
    if _user_anchor_triplet_was_provided(extra):
        logger.debug(
            "[build_complex_z] Explicit create.anchor_atoms triplet was provided; "
            "stable equilibration distance will not modify receptor anchors."
        )
    else:
        if stable_record is not None:
            stable_preference = _select_stable_boresch_distance_preference(
                u=u,
                mol=mol,
                stable_record=stable_record,
                P1=P1,
                P2=P2,
                P3=P3,
                lig_name_str=lig_name_str,
                l1_x=float(l1_x),
                l1_y=float(l1_y),
                l1_z=float(l1_z),
                l1_range=float(l1_range),
                renum_data=renum_data,
            )
            if stable_preference is not None:
                P1 = stable_preference["P1"]
                p1_resid = stable_preference["p1_resid"]
                p1_atom = stable_preference["p1_atom"]
                p1_vmd = stable_preference["p1_vmd"]
                lig_name_str = stable_preference["lig_name_str"]
                lig_name_str = " ".join(
                    _order_ligand_names_with_priority(
                        lig_name_str.split(),
                        salt_bridge_lig_names,
                    )
                )
                l1_x = stable_preference["l1_x"]
                l1_y = stable_preference["l1_y"]
                l1_z = stable_preference["l1_z"]
                l1_range = stable_preference["l1_range"]
                stable_preference_applied = True
                logger.debug(
                    "[build_complex_z] Using stable equilibration distance to prefer "
                    "P1={} (from {}) and ligand L1 candidate {} for {} "
                    "(rank {}/{}).",
                    P1,
                    stable_preference["stable_original_P1"],
                    stable_preference["stable_ligand_name"],
                    ligand,
                    stable_preference.get("stable_rank", 1),
                    stable_preference.get("stable_candidate_count", 1),
                )

    # 11) Python ligand-anchor preparation
    def _restore_anchor_state(state: dict) -> None:
        nonlocal P1, p1_resid, p1_atom, p1_vmd, lig_name_str, l1_x, l1_y, l1_z, l1_range
        P1 = state["P1"]
        p1_resid = state["p1_resid"]
        p1_atom = state["p1_atom"]
        p1_vmd = state["p1_vmd"]
        lig_name_str = state["lig_name_str"]
        l1_x = state["l1_x"]
        l1_y = state["l1_y"]
        l1_z = state["l1_z"]
        l1_range = state["l1_range"]

    def _anchor_names_from_file(path: Path) -> list[str]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            return path.read_text().splitlines()[0].split()
        except Exception:
            return []

    def _partial_ligand_anchors_are_expected(path: Path) -> bool:
        names = _anchor_names_from_file(path)
        if not names or len(names) >= 3:
            return False
        if lig_heavy_count >= 3:
            return False
        logger.debug(
            "Ligand {} has only {} non-hydrogen atom(s); using reduced FE ligand "
            "anchor set {} and omitting unavailable Boresch angle/dihedral terms.",
            ligand,
            lig_heavy_count,
            " ".join(names),
        )
        return True

    def _run_prep(ligand_name_str: str) -> None:
        _python_prep_complex(
            workdir=workdir,
            mol=mol,
            p1_atom=p1_atom,
            p1_vmd=p1_vmd,
            p2_atom=p2_atom,
            p2_vmd=p2_vmd,
            first_resid="2",
            last_resid=str(rec_res),
            stage="fe",
            l1_x=float(l1_x),
            l1_y=float(l1_y),
            l1_z=float(l1_z),
            l1_range=float(l1_range),
            min_adis=float(min_adis),
            max_adis=float(max_adis),
            sdr_dist=float(sdr_dist),
            ligand_names=str(ligand_name_str).split(),
            other_mol=other_mol,
            lipid_mol=lipid_mol,
            preferred_l1_names=salt_bridge_lig_names,
        )

    try:
        _run_prep(lig_name_str)
    except RuntimeError:
        if _partial_ligand_anchors_are_expected(_p("anchors.txt")):
            pass
        else:
            logger.debug(
                "[build_complex_z] Candidate ligand anchors failed for {}; "
                "retrying with all ligand atoms.",
                ligand,
            )
            all_lig_name_str = " ".join(
                _order_ligand_names_with_priority(
                    lig_heavy_names if lig_heavy_names else [str(x) for x in lig_names],
                    salt_bridge_lig_names,
                )
            )
            try:
                _run_prep(all_lig_name_str)
                lig_name_str = all_lig_name_str
            except RuntimeError:
                if _partial_ligand_anchors_are_expected(_p("anchors.txt")):
                    lig_name_str = all_lig_name_str
                else:
                    if not stable_preference_applied:
                        raise
                    logger.debug(
                        "[build_complex_z] Stable-distance preferred geometry failed for {}; "
                        "retrying original receptor anchor geometry.",
                        ligand,
                    )
                    _restore_anchor_state(default_anchor_state)
                    try:
                        _run_prep(lig_name_str)
                    except RuntimeError:
                        if _partial_ligand_anchors_are_expected(_p("anchors.txt")):
                            pass
                        else:
                            all_lig_name_str = " ".join(
                                _order_ligand_names_with_priority(
                                    lig_heavy_names if lig_heavy_names else [str(x) for x in lig_names],
                                    salt_bridge_lig_names,
                                )
                            )
                            _run_prep(all_lig_name_str)
                            lig_name_str = all_lig_name_str

    prep_translation = _translate_pdb_to_reference_frame(
        target_pdb=_p(f"fe-{mol}.pdb"),
        reference_pdb=_p("aligned_amber.pdb"),
    )
    if prep_translation is not None:
        _translate_pdb_by_vector(_p(f"{mol}.pdb"), prep_translation)
        _translate_pdb_by_vector(_p(f"{mol}-noh.pdb"), prep_translation)

    # 12) anchors.txt -> validate, rename with ligand tag, write header into fe-<mol>.pdb
    anchors_txt = _p("anchors.txt")
    anchor_names = _anchor_names_from_file(anchors_txt)
    if not anchor_names:
        logger.warning("anchors.txt missing or empty")
        return False
    good = len(anchor_names) >= 3 or _partial_ligand_anchors_are_expected(anchors_txt)
    tagged = _p(f"anchors-{ligand}.txt")
    anchors_txt.rename(tagged)
    if not good:
        logger.warning("anchors.txt too short; pruning")
        return False

    lig_resid = str(int(recep_last) + 2)
    fe_pdb = _p(f"fe-{mol}.pdb")
    if not fe_pdb.exists():
        raise FileNotFoundError(f"Missing {fe_pdb}")
    a = anchor_names
    if len(a) >= 3:
        old_receptor_masks = (P1, P2, P3)
        old_ligand_names = list(a[:3])
        preferred_first_names = _dedupe_names(
            [
                *salt_bridge_lig_names,
                *(
                    [stable_preference["stable_ligand_name"]]
                    if stable_preference_applied and stable_preference is not None
                    else []
                ),
            ]
        )
        user_anchor_triplet = _user_anchor_triplet_was_provided(extra)
        P1, P2, P3, a = _guard_abfe_boresch_anchor_frame(
            fe_pdb=fe_pdb,
            mol=mol,
            ligand_label=ligand,
            P1=P1,
            P2=P2,
            P3=P3,
            lig_resid=lig_resid,
            selected_names=a,
            preferred_first_names=preferred_first_names,
            allow_receptor_reselection=not user_anchor_triplet,
        )
        _write_abfe_anchor_guard_diagnostic(
            path=_p("boresch_anchor_guard.json"),
            fe_pdb=fe_pdb,
            mol=mol,
            ligand_label=ligand,
            lig_resid=lig_resid,
            old_receptor_masks=old_receptor_masks,
            new_receptor_masks=(P1, P2, P3),
            old_ligand_names=old_ligand_names,
            new_ligand_names=a,
            preferred_first_names=preferred_first_names,
            allow_receptor_reselection=not user_anchor_triplet,
            user_anchor_triplet=user_anchor_triplet,
        )
    tagged.write_text(" ".join(a[:3]) + "\n")
    L1 = f":{lig_resid}@{a[0]}"
    L2 = f":{lig_resid}@{a[1]}" if len(a) > 1 else None
    L3 = f":{lig_resid}@{a[2]}" if len(a) > 2 else None
    L2_label = L2 or "NA"
    L3_label = L3 or "NA"

    lines = fe_pdb.read_text().splitlines(True)
    with fe_pdb.open("wt") as fout:
        fout.write(
            f"{'REMARK A':<8s}  {P1:6s}  {P2:6s}  {P3:6s}  {L1:6s}  {L2_label:6s}  {L3_label:6s}  {first_res:6s}  {recep_last:4s}\n"
        )
        fout.writelines(lines[1:])

    save_anchors(
        workdir, Anchors(P1=P1, P2=P2, P3=P3, L1=L1, L2=L2, L3=L3, lig_res=lig_resid)
    )

    return True


@register_build_complex("x")
def build_complex_x(ctx) -> bool:
    """
    RBFE (x-component) build_complex.

    Builds the reference-ligand complex using the equilibrated reference ligand,
    and stages auxiliary files for the alternate ligand (for downstream RBFE steps).
    """
    extra = ctx.extra or {}
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    res_ref = extra.get("residue_ref") or ctx.residue_name
    res_alt = extra.get("residue_alt")

    if not lig_ref or not lig_alt or not res_ref or not res_alt:
        raise ValueError(
            "RBFE component 'x' requires pair metadata "
            "(ligand_ref/ligand_alt/residue_ref/residue_alt)."
        )

    # Reuse the z-build logic with a reference-ligand context to build the complex.
    ref_ctx = BuildContext(
        ligand=str(lig_ref),
        residue_name=str(res_ref),
        param_dir_dict=ctx.param_dir_dict,
        sim=ctx.sim,
        working_dir=ctx.working_dir,
        system_root=ctx.system_root,
        comp=ctx.comp,
        win=ctx.win,
        anchors=ctx.anchors,
        lipid_mol=ctx.lipid_mol,
        other_mol=ctx.other_mol,
        extra=dict(extra),
    )

    ok = build_complex_z(ref_ctx)
    if not ok:
        return False

    # Stage alternate-ligand inputs alongside build files for downstream RBFE steps.
    build_dir = ctx.build_dir
    sys_root = ctx.system_root
    all_ligs = sys_root / "all-ligands"
    alt_pdb = all_ligs / f"{lig_alt}.pdb"
    if alt_pdb.exists():
        shutil.copy2(alt_pdb, build_dir / alt_pdb.name)
    else:
        logger.warning("[build_complex_x] Missing alt ligand PDB: {}", alt_pdb)

    alt_params = sys_root / "simulations" / str(lig_alt) / "params"
    for ext in (".mol2", ".sdf"):
        src = alt_params / f"{res_alt}{ext}"
        if src.exists():
            shutil.copy2(src, build_dir / src.name)
        else:
            logger.warning("[build_complex_x] Missing alt ligand param file: {}", src)

    return True


@register_build_complex("y")
@register_build_complex("m")
def build_complex_lig(ctx) -> bool:
    """
    Component 'y' (ligand-only) build_complex:
    - No receptor complexing; just stage the ligand structural files.
    - Sets builder.mol and builder.corrected_sdr_dist for downstream code.
    """
    # Where to put staged files
    build_dir: Path = ctx.build_dir
    work: Path = ctx.working_dir

    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve locations
    ligand = ctx.ligand
    mol = ctx.residue_name
    sys_root = ctx.system_root
    all_ligand_folder = sys_root / "all-ligands"
    ff_dir = sys_root / "simulations" / ligand / "params"
    param_json = ff_dir / f"{mol}.json"
    apo_ligand = _is_apo_ligand_build(param_json, ligand, mol)

    shutil.copytree(build_files_orig, build_dir, dirs_exist_ok=True)

    # Inputs
    ligand_pdb = all_ligand_folder / f"{ligand}.pdb"
    if not ligand_pdb.exists():
        raise FileNotFoundError(f"[build_complex_y] Missing ligand pdb: {ligand_pdb}")

    # Copy <pose>.pdb into build_dir
    shutil.copy2(ligand_pdb, build_dir / f"{ligand}.pdb")
    shutil.copy2(all_ligand_folder / f"{ligand}.pdb", work / f"{ligand}.pdb")

    # Ensure ligand atom names match antechamber mol2 (ligand.ff prepared earlier)
    shutil.copy2(ff_dir / f"{mol}.mol2", build_dir / f"{mol}.mol2")
    shutil.copy2(ff_dir / f"{mol}.sdf", build_dir / f"{mol}.sdf")

    _write_ligand_pdb_with_parameter_names(
        build_dir / f"{ligand}.pdb",
        build_dir / f"{mol}.mol2",
        build_dir / f"{mol}.pdb",
        residue_name=mol,
        ligand_label=ligand,
        apo_ligand=apo_ligand,
    )
    if apo_ligand:
        _copy_if_distinct(build_dir / f"{mol}.pdb", build_dir / f"{ligand}.pdb")
        _copy_if_distinct(build_dir / f"{mol}.pdb", work / f"{ligand}.pdb")

    mol = ctx.residue_name

    # Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf"):
        src = ff_dir / f"{mol}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_y] Missing ligand FF file: {src}")
        shutil.copy2(src, build_dir / src.name)

    return True
