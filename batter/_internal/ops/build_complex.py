from __future__ import annotations

import os
import re
import glob
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger

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
    select_ions_away_from_complex,
    Anchors,
    save_anchors,
)
from batter._internal.templates import BUILD_FILES_DIR as build_files_orig  # type: ignore


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


def _executable_available(command: str) -> bool:
    return shutil.which(command) is not None


def _run_pdb4amber_or_copy(input_pdb: Path, output_pdb: Path, *, working_dir: Path) -> None:
    if _executable_available("pdb4amber"):
        run_with_log(
            f"pdb4amber -i {input_pdb.name} -o {output_pdb.name} -y",
            working_dir=working_dir,
        )
        return
    logger.warning(
        "pdb4amber was not found; copying {} to {} without additional cleanup.",
        input_pdb,
        output_pdb,
    )
    shutil.copy2(input_pdb, output_pdb)


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
    _write_atomgroup_pdb(lig, workdir / f"{mol}.pdb")


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
        "# VMD unavailable; BATTER used Python fallback for prep-ini.tcl.\n"
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
    """Avoid endpoint angle/torsion Boresch ligand-anchor triplets for ABFE."""
    try:
        from batter._internal.ops.restraints import (
            _boresch_frame_margins,
            _boresch_frame_values,
            _frame_safe_boresch_atom_names_from_residue,
        )
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not import Boresch frame guard for {}; "
            "keeping ligand anchors {}: {}",
            ligand_label,
            list(selected_names[:3]),
            exc,
        )
        return list(selected_names[:3])

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
        return list(selected_names[:3])

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
            return list(selected_names[:3])
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
        return list(selected_names[:3])

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
        return list(selected_names[:3])

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
        return list(selected_names[:3])

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
    return guarded_names


def _pick_ligand_anchor_names(
    *,
    u: mda.Universe,
    mol: str,
    ligand_names: Sequence[str],
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

    candidates: dict[str, np.ndarray] = {}
    for name in ligand_names:
        atoms = _ligand_atom_by_name(u, mol, str(name))
        if atoms.n_atoms == 0:
            continue
        center = _center_of_atoms(atoms)
        dist = float(np.linalg.norm(center - target))
        if dist >= float(l1_range):
            continue
        angle = _angle_degrees(p2_center, p1_center, center)
        if np.isfinite(angle):
            candidates[str(name)] = center

    aa1: str | None = None
    for tolerance in (15.0, 70.0):
        best_diff = float("inf")
        for name, center in candidates.items():
            angle = _angle_degrees(p2_center, p1_center, center)
            if not np.isfinite(angle) or abs(angle - 90.0) > tolerance:
                continue
            diff = float(np.linalg.norm(center - target))
            if diff < best_diff:
                best_diff = diff
                aa1 = name
        if aa1 is not None:
            break
    if aa1 is None:
        raise RuntimeError("anchor not found")

    aa1_center = candidates[aa1]
    aa2: str | None = None
    best_angle_diff = float("inf")
    for name, center in candidates.items():
        if name == aa1:
            continue
        distance = float(np.linalg.norm(center - aa1_center))
        if not (float(min_adis) < distance < float(max_adis)):
            continue
        angle = _angle_degrees(p1_center, aa1_center, center)
        if not np.isfinite(angle):
            continue
        angle_diff = abs(angle - 90.0)
        if angle_diff < best_angle_diff:
            best_angle_diff = angle_diff
            aa2 = name
    if aa2 is None:
        raise RuntimeError("anchor not found")

    aa2_center = candidates[aa2]
    aa3: str | None = None
    best_angle_diff = float("inf")
    for name, center in candidates.items():
        if name in {aa1, aa2}:
            continue
        distance = float(np.linalg.norm(center - aa2_center))
        if not (float(min_adis) < distance < float(max_adis)):
            continue
        angle = _angle_degrees(aa1_center, aa2_center, center)
        if not np.isfinite(angle):
            continue
        angle_diff = abs(angle - 90.0)
        if angle_diff < best_angle_diff:
            best_angle_diff = angle_diff
            aa3 = name
    if aa3 is None:
        raise RuntimeError("anchor not found")
    return [aa1, aa2, aa3]


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
) -> None:
    """Python fallback for prep-ini.tcl when VMD is unavailable."""
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

    anchors = _pick_ligand_anchor_names(
        u=prep_u,
        mol=mol,
        ligand_names=ligand_names,
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
        Space-separated ligand atom names for VMD anchor selection.

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
    sdf_atom_count, heavy_ordinals = _sdf_heavy_atom_ordinals(sdf_file)

    names: list[str] = []
    dropped: list[int] = []
    if sdf_atom_count == len(lig_names):
        for idx in candidate_indices:
            if 0 <= idx < len(lig_names):
                names.append(lig_names[idx])
            else:
                dropped.append(idx)
    elif heavy_ordinals:
        heavy_names = [
            str(atom.name)
            for atom in ligand_atoms
            if not _atom_is_hydrogen(atom)
        ]
        for idx in candidate_indices:
            heavy_ordinal = heavy_ordinals.get(idx)
            if heavy_ordinal is not None and 0 <= heavy_ordinal < len(heavy_names):
                names.append(heavy_names[heavy_ordinal])
            else:
                dropped.append(idx)
    else:
        for idx in candidate_indices:
            if 0 <= idx < len(lig_names):
                names.append(lig_names[idx])
            else:
                dropped.append(idx)

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
            "using all final ligand atoms for anchor search.",
            ligand_label,
            stage,
        )
        names = lig_names

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
_STABLE_BORESCH_DISTANCE_SCHEMA_VERSION = 4


def _user_anchor_atoms_were_provided(extra: dict | None) -> bool:
    if not extra:
        return False
    anchors = extra.get("user_anchor_atoms") or ()
    return any(str(anchor).strip() for anchor in anchors)


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
    try:
        from batter._internal.ops.restraints import (
            _frame_safe_boresch_atom_names_from_residue,
        )
    except Exception as exc:
        logger.warning(
            "[build_complex_z] Could not import Boresch frame guard while checking "
            "stable pair {}: {}",
            preference.get("stable_ligand_name"),
            exc,
        )
        return True

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

    try:
        _frame_safe_boresch_atom_names_from_residue(
            residue,
            receptor_atoms=receptor_atoms,
            label="ABFE stable-pair precheck",
            preferred_first_names=[preference["stable_ligand_name"]],
            require_preferred_first=True,
        )
    except Exception as exc:
        logger.debug(
            "[build_complex_z] Stable pair P1={} L1={} did not pass the "
            "full-frame Boresch guard: {}",
            preference.get("P1"),
            preference.get("stable_ligand_name"),
            exc,
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

    run_with_log(
        f"{vmd} -dispdev text -e {str(split_tcl)}",
        error_match="syntax error",
        shell=False,
        working_dir=build_dir,
    )
    # Protein PDB cleanup with pdb4amber
    shutil.copy2(build_dir / "protein.pdb", build_dir / "protein_vmd.pdb")
    run_with_log(
        "pdb4amber -i protein_vmd.pdb -o protein.pdb -y", working_dir=build_dir
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
        run_with_log(
            f"{charmmlipid2amber} -i {build_dir/'lipids.pdb'} -o {build_dir/'lipids_amber.pdb'}"
        )
        u_lip = mda.Universe(str(build_dir / "lipids_amber.pdb"))
        lipid_resnames = list(set(u_lip.residues.resnames))
        logger.debug(f"[Equil] Converted CHARMM lipids to AMBER: {lipid_resnames}")
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
    run_with_log(
        "pdb4amber -i reference.pdb -o reference_amber.pdb -y", working_dir=build_dir
    )
    run_with_log(
        f"{vmd} -dispdev text -e nochain.tcl", shell=False, working_dir=build_dir
    )
    run_with_log(
        "./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc",
        working_dir=build_dir,
    )
    run_with_log(
        f"{vmd} -dispdev text -e measure-fit.tcl", shell=False, working_dir=build_dir
    )

    # Clean aligned and put in AMBER format
    with (
        open(build_dir / "aligned.pdb", "r") as oldfile,
        open(build_dir / "aligned-clean.pdb", "w") as newfile,
    ):
        for line in oldfile:
            if len(line.split()) > 4:
                newfile.write(line)
    run_with_log(
        "pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y", working_dir=build_dir
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
    if apo_anchor_names is None:
        lig_name_str = _candidate_ligand_atom_name_string(
            sdf_file,
            lig_atoms,
            ligand_label=ligand,
            stage="equil",
        )
    else:
        lig_name_str = " ".join(apo_anchor_names)

    # Build VMD prep.tcl from template, try with candidate names first
    prep_ini = build_dir / "prep-ini.tcl"
    prep_tcl = build_dir / "prep.tcl"

    def _write_prep(ligand_name_str: str) -> None:
        with open(prep_ini, "rt") as fin, open(prep_tcl, "wt") as fout:
            other_mol_vmd = " ".join(other_mol)
            lipid_mol_vmd = " ".join(lipid_mol)
            for line in fin:
                fout.write(
                    line.replace("MMM", mol)
                    .replace("mmm", mol)
                    .replace("NN", h1_atom)
                    .replace("N2A", h2_atom)
                    .replace("P1A", f"{p1_vmd}")
                    .replace("P2A", f"{p2_vmd}")
                    .replace("FIRST", "1")
                    .replace("LAST", f"{recep_resid_num}")
                    .replace("STAGE", "equil")
                    .replace("XDIS", f"{l1_x:4.2f}")
                    .replace("YDIS", f"{l1_y:4.2f}")
                    .replace("ZDIS", f"{l1_z:4.2f}")
                    .replace("RANG", f"{l1_range:4.2f}")
                    .replace("DMAX", f"{max_adis:4.2f}")
                    .replace("DMIN", f"{min_adis:4.2f}")
                    .replace("SDRD", f"{0.0:4.2f}")
                    .replace("OTHRS", str(other_mol_vmd))
                    .replace("LIPIDS", str(lipid_mol_vmd))
                    .replace("LIGANDNAME", ligand_name_str)
                )

    _write_prep(lig_name_str)
    if apo_ligand:
        try:
            run_with_log(
                f"{vmd} -dispdev text -e prep.tcl",
                shell=False,
                working_dir=build_dir,
            )
        except RuntimeError:
            if not (build_dir / f"equil-{mol}.pdb").exists():
                raise
            logger.warning(
                "[build_complex] VMD exited while searching apo dummy anchors for {}; "
                "continuing with fixed dummy anchors.",
                ligand,
            )
        _write_apo_anchor_outputs(
            build_dir,
            ligand=ligand,
            mol=mol,
            anchor_names=apo_anchor_names or list(lig_names),
        )
        return True

    try:
        run_with_log(
            f"{vmd} -dispdev text -e prep.tcl",
            error_match="anchor not found",
            shell=False,
            working_dir=build_dir,
        )
    except RuntimeError:
        # fallback: all ligand atoms
        lig_name_str2 = " ".join([str(x) for x in lig_names])
        _write_prep(lig_name_str2)
        run_with_log(
            f"{vmd} -dispdev text -e prep.tcl",
            error_match="anchor not found",
            shell=False,
            working_dir=build_dir,
        )

    # Verify anchors.txt
    anchor_file = build_dir / "anchors.txt"
    if anchor_file.stat().st_size == 0:
        logger.warning(
            f"Could not find ligand L1 for {ligand}. Most likely not in binding site."
        )
        return False

    # Ensure we got 3 ligand anchors
    with open(anchor_file) as f:
        line = f.readline().strip()
    if len(line.split()) < 3:
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

    # 1) copy artifacts from equil
    _copy(equil_dir / "q_build_files" / f"{ligand}.pdb", f"{ligand}.pdb")
    _copy(equil_dir / "representative.rst7", "representative.rst7")
    _copy(equil_dir / "representative.pdb", "aligned-nc.pdb")
    _copy(equil_dir / "build_amber_renum.txt", "build_amber_renum.txt")
    _copy(equil_dir / "q_build_files" / "protein_renum.txt", "protein_renum.txt")

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

    # 3) extract receptor-only PDB from representative.rst7
    run_with_log(
        f"{cpptraj} -p full.prmtop -y representative.rst7 -x rec_file.pdb",
        working_dir=workdir,
    )

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
        revised_resids = []
        resid_counter = 1
        prev_resid = 0
        for i, row in renum.iterrows():
            # skip water and ions as they will not be present later
            if row["old_resname"] in ["WAT", "Na+", "Cl-", "K+"]:
                continue
            if row["old_resid"] != prev_resid or row["old_resname"] not in lipid_mol:
                revised_resids.append(resid_counter)
                resid_counter += 1
            else:
                revised_resids.append(resid_counter - 1)
            prev_resid = row["old_resid"]

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
        _run_pdb4amber_or_copy(
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
    lig_name_str = _candidate_ligand_atom_name_string(
        sdf_file,
        lig_atoms,
        ligand_label=ligand,
        stage="fe-z",
    )
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

    extra = dict(ctx.extra or {})
    if _user_anchor_atoms_were_provided(extra):
        logger.debug(
            "[build_complex_z] Explicit create.anchor_atoms were provided; "
            "stable equilibration distance will not modify receptor anchors."
        )
    else:
        stable_record = _load_stable_boresch_distance(equil_dir)
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

    # 11) prep.tcl
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

    def _write_prep(ligand_name_str: str) -> None:
        with open(_p("prep-ini.tcl"), "rt") as fin, open(_p("prep.tcl"), "wt") as fout:
            for line in fin:
                fout.write(
                    line.replace("MMM", mol)
                    .replace("mmm", mol)
                    .replace("NN", p1_atom)
                    .replace("P1A", p1_vmd)
                    .replace("N2A", p2_atom)
                    .replace("P2A", p2_vmd)
                    .replace("FIRST", "2")
                    .replace("LAST", str(rec_res))
                    .replace("STAGE", "fe")
                    .replace("XDIS", f"{l1_x:4.2f}")
                    .replace("YDIS", f"{l1_y:4.2f}")
                    .replace("ZDIS", f"{l1_z:4.2f}")
                    .replace("RANG", f"{l1_range:4.2f}")
                    .replace("DMAX", f"{max_adis:4.2f}")
                    .replace("DMIN", f"{min_adis:4.2f}")
                    .replace("SDRD", f"{sdr_dist:4.2f}")
                    .replace("LIGSITE", "0")  # no FB for ligand now
                    .replace("OTHRS", " ".join(other_mol) if other_mol else "XXX")
                    .replace("LIPIDS", " ".join(lipid_mol) if lipid_mol else "XXX")
                    .replace("LIGANDNAME", ligand_name_str)
                )

    def _run_prep(ligand_name_str: str) -> None:
        _write_prep(ligand_name_str)
        if vmd_available:
            run_with_log(
                f"{vmd} -dispdev text -e prep.tcl",
                error_match="anchor not found",
                shell=False,
                working_dir=workdir,
            )
        else:
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
            )

    try:
        _run_prep(lig_name_str)
    except RuntimeError:
        logger.debug(
            "[build_complex_z] Candidate ligand anchors failed for {}; "
            "retrying with all ligand atoms.",
            ligand,
        )
        all_lig_name_str = " ".join(str(x) for x in lig_names)
        try:
            _run_prep(all_lig_name_str)
            lig_name_str = all_lig_name_str
        except RuntimeError:
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
                all_lig_name_str = " ".join(str(x) for x in lig_names)
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
    if (not anchors_txt.exists()) or (anchors_txt.stat().st_size == 0):
        logger.warning("anchors.txt missing or empty")
        return False
    good = True
    with anchors_txt.open() as f:
        for ln in f:
            if len(ln.split()) < 3:
                good = False
                break
    tagged = _p(f"anchors-{ligand}.txt")
    anchors_txt.rename(tagged)
    if not good:
        logger.warning("anchors.txt too short; pruning")
        return False

    lig_resid = str(int(recep_last) + 2)
    fe_pdb = _p(f"fe-{mol}.pdb")
    if not fe_pdb.exists():
        raise FileNotFoundError(f"Missing {fe_pdb}")
    with tagged.open() as f:
        a = f.readline().split()
    a = _guard_abfe_boresch_ligand_anchor_names(
        fe_pdb=fe_pdb,
        mol=mol,
        ligand_label=ligand,
        P1=P1,
        P2=P2,
        P3=P3,
        lig_resid=lig_resid,
        selected_names=a,
        preferred_first_names=(
            [stable_preference["stable_ligand_name"]]
            if stable_preference_applied and stable_preference is not None
            else []
        ),
    )
    tagged.write_text(" ".join(a[:3]) + "\n")
    L1 = f":{lig_resid}@{a[0]}"
    L2 = f":{lig_resid}@{a[1]}"
    L3 = f":{lig_resid}@{a[2]}"

    lines = fe_pdb.read_text().splitlines(True)
    with fe_pdb.open("wt") as fout:
        fout.write(
            f"{'REMARK A':<8s}  {P1:6s}  {P2:6s}  {P3:6s}  {L1:6s}  {L2:6s}  {L3:6s}  {first_res:6s}  {recep_last:4s}\n"
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
