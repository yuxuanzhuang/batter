from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from math import ceil
import json
import re
import MDAnalysis as mda
import numpy as np
from loguru import logger

from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_restraints
from batter._internal.ops.helpers import (
    PROTEIN_COM_ATOM_SELECTION,
    num_to_mask,
    load_anchors,
    is_atom_line as _is_atom_line,
    field_slice as _field,
)
from batter.utils import run_with_log, cpptraj

ION_NAMES = {"Na+", "K+", "Cl-", "NA", "CL", "K"}  # NA/CL appear in some pdbs too
ION_GUARD_DISTANCE = 15.0
ION_GUARD_FORCE = 10.0
ION_GUARD_TAG = "Ion_Guard"
BULK_LIGAND_RESTRAINT_HALF_WIDTH = 3.0
BULK_LIGAND_RESTRAINT_FORCE = 10.0
BULK_LIGAND_RESTRAINT_TAG = "Bulk_Lig"
COM_RESTRAINT_ANCHORS = (0.0, 0.0, 0.0, 999.0)
ABFE_DIFF_POSE_RADIUS = 8.0
ABFE_DIFF_POSE_MAX_ANCHORS = 8
ABFE_DIFF_POSE_MIN_ANCHORS = 4
ABFE_DIFF_POSE_WIDTH = 0.5
ABFE_DIFF_POSE_LOCAL_ANCHORS = 3
ABFE_DIFF_POSE_LIGAND_ATOMS = 6
_ANCHOR_MASK_RE = re.compile(r"^:(-?\d+)@(.+)$")
BORESCH_MIN_ANGLE_MARGIN_DEG = 30.0
BORESCH_MIN_TORSION_MARGIN_DEG = 15.0
SEPTOP_COMMON_CORE_BORESCH_MIN_MAPPED_ATOMS = 4
LIGAND_DIHEDRAL_DEFAULT_FORCE = 10.0
BoreschCandidate = tuple[float, tuple[int, int, int], tuple[float, ...], float, float]

def _stride_atom_serials(
    atoms: Sequence[str | int],
    max_n: int,
) -> list[str]:
    """Return atom serials strided down to at most ``max_n`` entries."""
    if max_n <= 0:
        raise ValueError("max_n must be a positive integer")

    tokens = [str(atom).strip() for atom in atoms if str(atom).strip()]
    if len(tokens) <= max_n:
        return tokens

    step = ceil(len(tokens) / max_n)
    return tokens[::step]


def _collect_calpha_and_lig(
    vac_pdb: Path,
    lig_res: str,
    offset: int = 0,
    stride_to_max_number: int = 50,
) -> Tuple[List[str], List[str]]:
    """Return (protein_calpha_serials, ligand_heavy_atom_serials).

    If either list is longer than `stride_to_max_number`, it is strided so the
    returned list length is <= `stride_to_max_number`. It is to keep better performance in simulations
    """
    u = mda.Universe(str(vac_pdb))

    protein_calpha_serials = (
        (u.select_atoms(PROTEIN_COM_ATOM_SELECTION).indices + 1).astype(str).tolist()
    )
    ligand_heavy_atom_serials = (
        (
            u.select_atoms(f"not type H and resid {int(lig_res) + offset}").indices + 1
        )
        .astype(str)
        .tolist()
    )

    protein_calpha_serials = _stride_atom_serials(
        protein_calpha_serials, stride_to_max_number
    )
    ligand_heavy_atom_serials = _stride_atom_serials(
        ligand_heavy_atom_serials, stride_to_max_number
    )

    return protein_calpha_serials, ligand_heavy_atom_serials


def _select_bound_ligand_heavy_atoms(
    universe: mda.Universe,
    residue_name: str,
) -> mda.AtomGroup:
    """Return heavy atoms from the first ligand residue in the bound pose."""
    lig_atoms = universe.select_atoms(f"resname {residue_name}")
    if lig_atoms.n_atoms == 0:
        raise ValueError(
            f"[restraints:d] No residues with resname {residue_name!r} found in vac.pdb"
        )

    ligand_resids = sorted({int(resid) for resid in lig_atoms.resids})
    bound_resid = ligand_resids[0]
    bound_atoms = universe.select_atoms(f"resid {bound_resid}")
    heavy = bound_atoms[
        [
            not _is_hydrogen_atom(atom)
            for atom in bound_atoms
        ]
    ]
    if heavy.n_atoms == 0:
        raise ValueError(
            f"[restraints:d] Found zero heavy atoms for bound ligand residue {bound_resid}"
        )
    return heavy


def _is_hydrogen_atom(atom) -> bool:
    """Return True for hydrogen atoms using element, type, or PDB atom name."""
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element == "H":
        return True
    atom_type = str(getattr(atom, "type", "") or "").strip().upper()
    if atom_type == "H":
        return True
    name = str(getattr(atom, "name", "") or "").strip().upper()
    return bool(re.match(r"^\d*H", name))


def _select_binding_site_calpha_atoms(
    universe: mda.Universe,
    ligand_heavy: mda.AtomGroup,
    *,
    radius: float = ABFE_DIFF_POSE_RADIUS,
    min_atoms: int = ABFE_DIFF_POSE_MIN_ANCHORS,
    max_atoms: int = ABFE_DIFF_POSE_MAX_ANCHORS,
) -> list:
    """Choose nearby receptor C-alpha atoms for relative ligand-pose restraints."""
    ca_atoms = universe.select_atoms("protein and name CA")
    if ca_atoms.n_atoms == 0:
        ca_atoms = universe.select_atoms("name CA")
    if ca_atoms.n_atoms == 0:
        raise ValueError(
            "[restraints:d] Cannot build relative pose restraints without receptor CA atoms"
        )

    deltas = ca_atoms.positions[:, None, :] - ligand_heavy.positions[None, :, :]
    nearest = np.linalg.norm(deltas, axis=2).min(axis=1)
    order = np.argsort(nearest)
    within = [int(idx) for idx in order if nearest[int(idx)] <= radius]
    selected = within[:max_atoms]
    if len(selected) < min_atoms:
        selected = [int(idx) for idx in order[: min(max_atoms, ca_atoms.n_atoms)]]
    return [ca_atoms[idx] for idx in selected]


def _flat_bottom_distance_anchors(target: float, width: float) -> tuple[float, float, float, float]:
    """Return AMBER r1-r4 anchors for a symmetric flat-bottom distance well."""
    return (0.0, max(0.0, target - width), target + width, 999.0)


def _adjust_receptor_anchor_mask(mask: str, dec_method: str | None) -> str:
    """Return the receptor anchor mask matching SDR-renumbered FE PDBs."""
    if str(dec_method or "").lower() != "sdr":
        return mask
    match = _ANCHOR_MASK_RE.match(mask.strip())
    if not match:
        return mask
    return f":{int(match.group(1)) + 1}@{match.group(2)}"


def _mask_index(atm_num: Sequence[str], mask: str) -> int:
    """Return the 1-based PDB atom index for an Amber mask.

    Amber topology atom names for halogens are commonly upper-case (CL1), while
    PDB atom names written by Amber tools can be element-cased (Cl1). Prefer an
    exact match, then fall back to a unique case-insensitive match.
    """
    try:
        return atm_num.index(mask)
    except ValueError:
        target = mask.lower()
        matches = [idx for idx, candidate in enumerate(atm_num) if candidate.lower() == target]
        if len(matches) == 1:
            return matches[0]
        raise


def _canonical_mask(atm_num: Sequence[str], mask: str) -> str:
    """Return the mask spelling used by ``atm_num`` when it can be resolved."""
    try:
        return atm_num[_mask_index(atm_num, mask)]
    except ValueError:
        return mask


def _canonicalize_restraint_expr(expr: str, atm_num: Sequence[str]) -> str:
    """Normalize atom-mask spelling in a restraint expression to ``vac.pdb``."""
    return " ".join(_canonical_mask(atm_num, field) for field in expr.split())


def _resolve_anchor_atom_from_mask(
    universe: mda.Universe,
    atm_num: Sequence[str],
    mask: str,
) -> object | None:
    """Resolve an Amber-style atom mask to an MDAnalysis atom from ``vac.pdb``."""
    try:
        serial = _mask_index(atm_num, mask)
    except ValueError:
        return None
    if serial <= 0 or serial > universe.atoms.n_atoms:
        return None
    return universe.atoms[serial - 1]


def _load_abfe_diff_saved_anchors(ctx: BuildContext):
    work_dir = getattr(ctx, "working_dir", None)
    comp = getattr(ctx, "comp", "d")
    if work_dir is None:
        return None
    try:
        return load_anchors(Path(work_dir) / f"{comp}_build_files")
    except Exception as exc:
        logger.debug(f"[restraints:{comp}] could not load saved anchors: {exc}")
        return None


def _select_abfe_diff_receptor_anchors(
    ctx: BuildContext,
    universe: mda.Universe,
    vac_pdb: Path,
    ligand_heavy: mda.AtomGroup,
    *,
    count: int,
    radius: float,
) -> list:
    """Choose the receptor atoms defining the ABFE_diff ligand-pose frame."""
    comp = getattr(ctx, "comp", "d")
    count = max(ABFE_DIFF_POSE_LOCAL_ANCHORS, int(count))
    anchors = _load_abfe_diff_saved_anchors(ctx)
    selected: list = []

    if anchors is not None:
        atm_num = num_to_mask(vac_pdb.as_posix())
        dec_method = getattr(getattr(ctx, "sim", None), "dec_method", None)
        seen_indices: set[int] = set()
        for raw_mask in (anchors.P1, anchors.P2, anchors.P3):
            mask = _adjust_receptor_anchor_mask(str(raw_mask), dec_method)
            atom = _resolve_anchor_atom_from_mask(universe, atm_num, mask)
            if atom is None:
                logger.debug(
                    f"[restraints:{comp}] saved receptor anchor {mask!r} not found in {vac_pdb.name}"
                )
                continue
            if int(atom.index) in seen_indices:
                continue
            selected.append(atom)
            seen_indices.add(int(atom.index))
            if len(selected) >= count:
                return selected

    fallback = _select_binding_site_calpha_atoms(
        universe,
        ligand_heavy,
        radius=radius,
        min_atoms=ABFE_DIFF_POSE_LOCAL_ANCHORS,
        max_atoms=count,
    )
    seen = {int(atom.index) for atom in selected}
    selected.extend(atom for atom in fallback if int(atom.index) not in seen)
    if len(selected) < ABFE_DIFF_POSE_LOCAL_ANCHORS:
        raise ValueError(
            f"[restraints:{comp}] ABFE_diff local-frame restraints require at least "
            f"{ABFE_DIFF_POSE_LOCAL_ANCHORS} receptor anchors; found {len(selected)}"
        )
    return selected[:count]


def _preferred_ligand_anchor_atoms(
    ctx: BuildContext,
    universe: mda.Universe,
    vac_pdb: Path,
    ligand_heavy: mda.AtomGroup,
) -> list:
    """Return saved ligand anchor atoms when they are available in ``vac.pdb``."""
    anchors = _load_abfe_diff_saved_anchors(ctx)
    if anchors is None:
        return []
    atm_num = num_to_mask(vac_pdb.as_posix())
    ligand_indices = {int(atom.index) for atom in ligand_heavy}
    selected: list = []
    seen_indices: set[int] = set()
    for raw_mask in (anchors.L1, anchors.L2, anchors.L3):
        if not raw_mask:
            continue
        atom = _resolve_anchor_atom_from_mask(universe, atm_num, str(raw_mask))
        if atom is None or int(atom.index) not in ligand_indices:
            continue
        if int(atom.index) in seen_indices:
            continue
        selected.append(atom)
        seen_indices.add(int(atom.index))
    return selected


def _select_ligand_pose_atoms(
    ligand_heavy: mda.AtomGroup,
    *,
    count: int,
    preferred_atoms: Sequence[object] = (),
) -> list:
    """Choose a compact, spatially spread ligand scaffold for pose restraints."""
    count = max(3, int(count))
    heavy_atoms = list(ligand_heavy)
    if len(heavy_atoms) <= count:
        return heavy_atoms

    heavy_by_index = {int(atom.index): atom for atom in heavy_atoms}
    selected_indices: list[int] = []
    for atom in preferred_atoms:
        idx = int(atom.index)
        if idx in heavy_by_index and idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= count:
            return [heavy_by_index[idx] for idx in selected_indices]

    positions = np.asarray([atom.position for atom in heavy_atoms], dtype=float)
    atom_indices = [int(atom.index) for atom in heavy_atoms]
    centroid = positions.mean(axis=0)

    while len(selected_indices) < count:
        candidate_scores: list[tuple[float, int]] = []
        selected_set = set(selected_indices)
        for i, atom_index in enumerate(atom_indices):
            if atom_index in selected_set:
                continue
            if selected_indices:
                selected_positions = np.asarray(
                    [heavy_by_index[idx].position for idx in selected_indices],
                    dtype=float,
                )
                score = float(
                    np.linalg.norm(positions[i] - selected_positions, axis=1).min()
                )
            else:
                score = float(np.linalg.norm(positions[i] - centroid))
            candidate_scores.append((score, atom_index))
        if not candidate_scores:
            break
        _, chosen_index = max(candidate_scores, key=lambda item: (item[0], -item[1]))
        selected_indices.append(chosen_index)

    return [heavy_by_index[idx] for idx in selected_indices]


def _write_distance_restraint_block(
    handle,
    atom1: object,
    atom2: object,
    *,
    width: float,
    force_const: float,
) -> float:
    """Write one flat-bottom atom-atom distance restraint and return its target."""
    target = float(np.linalg.norm(atom1.position - atom2.position))
    _write_group_colvar_block(
        handle,
        anchor_atom=str(int(atom1.index) + 1),
        group_atoms=[str(int(atom2.index) + 1)],
        anchors=_flat_bottom_distance_anchors(target, width),
        strengths=(force_const, force_const),
    )
    return target


def _load_common_core_indices(mapping_path: Path) -> tuple[list[int], list[int]]:
    """Load 0-based ``(ref_indices, alt_indices)`` from RBFE mapping JSON.

    Prepared RBFE ``mapping.json`` files store ``componentB_to_componentA``,
    i.e. alternate-ligand atom index -> reference-ligand atom index.
    """
    if not mapping_path.exists():
        return [], []
    try:
        data = json.loads(mapping_path.read_text())
    except Exception as exc:
        logger.warning(f"[restraints:x] Failed to parse {mapping_path}: {exc}")
        return [], []

    if not isinstance(data, dict):
        logger.warning(f"[restraints:x] Unexpected mapping format in {mapping_path}: {type(data)}")
        return [], []

    if "scmk1_cc_solvent_indices" in data or "scmk2_cc_solvent_indices" in data:
        ref_indices = sorted(int(i) for i in data.get("scmk1_cc_solvent_indices", []))
        alt_indices = sorted(int(i) for i in data.get("scmk2_cc_solvent_indices", []))
        return ref_indices, alt_indices

    ref_indices: list[int] = []
    alt_indices: list[int] = []
    for raw_alt, raw_ref in data.items():
        try:
            ref_indices.append(int(raw_ref))
            alt_indices.append(int(raw_alt))
        except (TypeError, ValueError):
            continue
    ref_indices = sorted(set(ref_indices))
    alt_indices = sorted(set(alt_indices))
    return ref_indices, alt_indices


def _mapped_heavy_atom_names_from_residue(
    residue,
    mapped_indices: Iterable[int],
) -> list[str]:
    names: list[str] = []
    atoms = list(residue.atoms)
    for raw_idx in mapped_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(atoms):
            continue
        atom = atoms[idx]
        name = str(atom.name).strip()
        if not name or name in names or _is_hydrogen_atom(atom):
            continue
        names.append(name)
    return names


def _common_core_boresch_preference_names(
    names: Iterable[str],
    *,
    label: str,
) -> list[str]:
    """Return mapped common-core names only when there are enough to prefer."""
    unique: list[str] = []
    for name in names:
        clean = str(name).strip()
        if clean and clean not in unique:
            unique.append(clean)

    if len(unique) < SEPTOP_COMMON_CORE_BORESCH_MIN_MAPPED_ATOMS:
        if unique:
            logger.debug(
                "[restraints:x] ignoring {} mapped common-region Boresch "
                "preference: {} mapped heavy atom(s) < {} required: {}",
                label,
                len(unique),
                SEPTOP_COMMON_CORE_BORESCH_MIN_MAPPED_ATOMS,
                unique,
            )
        return []
    return unique


def _collect_common_core_heavy_ligand(
    vac_pdb: Path,
    lig_res: str,
    offset: int,
    mapped_indices: Iterable[int],
    stride_to_max_number: int = 10,
) -> List[str]:
    """Return 1-based atom serials for mapped heavy atoms in one ligand residue.

    If the resulting list is longer than `stride_to_max_number`, it is strided
    so the returned list length is <= `stride_to_max_number`.
    """
    u = mda.Universe(str(vac_pdb))
    lig_atoms = u.select_atoms(f"resid {int(lig_res) + offset}")
    if lig_atoms.n_atoms == 0:
        return []

    valid = sorted({int(i) for i in mapped_indices if 0 <= int(i) < lig_atoms.n_atoms})
    if not valid:
        return []

    cc_atoms = lig_atoms[valid].select_atoms("not name H*")
    if cc_atoms.n_atoms == 0:
        return []

    cc_serials = list((cc_atoms.indices + 1).astype(str))
    return _stride_atom_serials(cc_serials, stride_to_max_number)

def _scan_dihedrals_from_prmtop(prmtop_path: Path, ligand_atm_num: List[str]) -> List[str]:
    """Build ligand dihedral masks (non-H) from vac_ligand.prmtop."""
    mlines: List[str] = []
    spool = 0
    with prmtop_path.open() as fin:
        for line in fin:
            if "FLAG DIHEDRALS_WITHOUT_HYDROGEN" in line:
                spool = 1
                continue
            if "FLAG EXCLUDED_ATOMS_LIST" in line:
                spool = 0
            if spool and len(line.split()) > 3:
                mlines.append(line.rstrip())

    msk: List[str] = []
    # primary term
    for ln in mlines:
        data = ln.split()
        if int(data[3]) > 0:
            idx = [abs(int(x) // 3) + 1 for x in data[:4]]
            msk.append(
                f"{ligand_atm_num[idx[0]]} {ligand_atm_num[idx[1]]} {ligand_atm_num[idx[2]]} {ligand_atm_num[idx[3]]}"
            )
    # secondary term (if present)
    for ln in mlines:
        data = ln.split()
        if len(data) > 7 and int(data[8]) > 0:
            idx = [abs(int(x) // 3) + 1 for x in data[5:9]]
            msk.append(
                f"{ligand_atm_num[idx[0]]} {ligand_atm_num[idx[1]]} {ligand_atm_num[idx[2]]} {ligand_atm_num[idx[3]]}"
            )

    # de-duplicate on the central pair
    seen_pairs = set()
    uniq = []
    for m in msk:
        a, b, c, d = m.split()
        pair = tuple(sorted((b, c)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        uniq.append(m)
    return uniq

def _filter_sp_carbons(msk: List[str], mol2_path: Path) -> List[str]:
    """Drop dihedrals that include cg/c1 carbons (mol2 atom types)."""
    sp_atoms = set()
    if mol2_path.exists():
        with mol2_path.open() as fin:
            # naive/mol2-lite parse for atom types
            in_atoms = False
            for line in fin:
                if line.strip().startswith("@<TRIPOS>ATOM"):
                    in_atoms = True
                    continue
                if line.strip().startswith("@<TRIPOS>"):
                    in_atoms = False
                if in_atoms:
                    parts = line.split()
                    if len(parts) >= 6 and parts[5] in ("cg", "c1"):
                        # store atom name (parts[1] is atom id; we want name after '@' in masks,
                        # but masks carry serial@name – we filter by that name)
                        # We'll just record names we see and test substring after '@'
                        sp_atoms.add(parts[1])
    out = []
    for m in msk:
        _, b, c, _ = m.split()
        try:
            bname = b.split("@", 1)[1]
            cname = c.split("@", 1)[1]
        except Exception:
            out.append(m)
            continue
        if (bname in sp_atoms) or (cname in sp_atoms):
            continue
        out.append(m)
    return out


def _ligand_dihedral_force_constant(
    expr: str,
    active_force: float = LIGAND_DIHEDRAL_DEFAULT_FORCE,
) -> float:
    """Return the force constant for a prmtop-derived ligand dihedral."""
    return float(active_force)


def _restraint_reference_value(vals: Sequence[Optional[float]], index: int) -> Optional[float]:
    if index < 0 or index >= len(vals):
        return None
    value = vals[index]
    if value is None:
        return None
    return float(value)


def _ligand_dihedral_reference_value(
    vals: Sequence[Optional[float]],
    index: int,
    expr: str,
    force_const: float,
    comp: str,
) -> Optional[float]:
    value = _restraint_reference_value(vals, index)
    if value is not None:
        return value
    if float(force_const) == 0.0:
        logger.warning(
            f"[restraints:{comp}] missing reference for zero-force ligand dihedral {expr}; using 0.0"
        )
        return 0.0
    logger.warning(
        f"[restraints:{comp}] skipping ligand dihedral without reference value: {expr}"
    )
    return None


def _write_assign_and_read_vals(
    work: Path,
    rst_exprs: List[str],
    prmtop: Path,
    traj: Path,
) -> List[Optional[float]]:
    """Emit assign.in and parse assign.dat into reference values `vals`, same order as `rst_exprs`."""
    ain = work / "assign.in"
    with ain.open("w") as f:
        f.write(f"parm {prmtop.as_posix()}\n")
        f.write(f"trajin {traj.as_posix()}\n")
        for i, expr in enumerate(rst_exprs):
            parts = expr.split()
            if len(parts) == 2:
                f.write(f"distance r{i} {expr} noimage out assign.dat\n")
            elif len(parts) == 3:
                f.write(f"angle    r{i} {expr} out assign.dat\n")
            elif len(parts) == 4:
                f.write(f"dihedral r{i} {expr} out assign.dat\n")
    run_with_log(f"{cpptraj} -i {ain.name} > assign.log", working_dir=work)

    assign_dat = (work / "assign.dat").read_text().splitlines()
    if len(assign_dat) < 2:
        raise RuntimeError("assign.dat did not contain reference values")
    labels = assign_dat[0].split()
    raw_vals = assign_dat[1].split()
    if labels and labels[0].lower().lstrip("#") == "frame":
        labels = labels[1:]
        raw_vals = raw_vals[1:]

    vals: List[Optional[float]] = [None] * len(rst_exprs)
    for label, raw_val in zip(labels, raw_vals):
        if not re.fullmatch(r"r\d+", label):
            continue
        idx = int(label[1:])
        if 0 <= idx < len(vals):
            vals[idx] = float(raw_val)

    missing = [f"r{i}" for i, val in enumerate(vals) if val is None]
    if missing:
        logger.warning(
            f"[restraints] cpptraj did not report reference values for {', '.join(missing)}"
        )
    return vals


def _equil_anchor_restraint_expressions(
    P1: str,
    P2: str,
    P3: str,
    L1: Optional[str],
    L2: Optional[str],
    L3: Optional[str],
) -> tuple[List[str], int]:
    """Return equilibration anchor restraint expressions and ligand expression count."""
    rst: List[str] = [f"{P1} {P2}", f"{P2} {P3}", f"{P3} {P1}"]
    ligand_rst: List[str] = []
    if L1:
        ligand_rst.extend(
            [
                f"{P1} {L1}",
                f"{P2} {P1} {L1}",
                f"{P3} {P2} {P1} {L1}",
            ]
        )
        if L2:
            ligand_rst.extend(
                [
                    f"{P1} {L1} {L2}",
                    f"{P2} {P1} {L1} {L2}",
                ]
            )
            if L3:
                ligand_rst.append(f"{P1} {L1} {L2} {L3}")
    return rst + ligand_rst, len(ligand_rst)


def _ligand_anchor_count(L1: Optional[str], L2: Optional[str], L3: Optional[str]) -> int:
    return sum(1 for mask in (L1, L2, L3) if mask)


def _resid_from_anchor_mask(mask: str | None) -> str | None:
    if not mask:
        return None
    match = _ANCHOR_MASK_RE.match(str(mask).strip())
    return match.group(1) if match else None


def _validate_ligand_anchor_set(
    *,
    comp: str,
    L1: Optional[str],
    L2: Optional[str],
    L3: Optional[str],
    ligand_heavy_count: int,
) -> None:
    """Allow reduced ligand anchors only when the ligand is too small for Boresch."""
    anchor_count = _ligand_anchor_count(L1, L2, L3)
    if anchor_count == 3:
        return
    if 0 < anchor_count < 3 and 0 < int(ligand_heavy_count) < 3:
        labels = [str(mask) for mask in (L1, L2, L3) if mask]
        logger.debug(
            "[restraints:{}] ligand has only {} heavy atom(s); using reduced "
            "ligand anchor set {} and omitting unavailable Boresch terms.",
            comp,
            ligand_heavy_count,
            labels,
        )
        return
    raise ValueError(
        f"[restraints:{comp}] Boresch restraints require ligand anchors L1/L2/L3; "
        f"got L1={L1!r}, L2={L2!r}, L3={L3!r} for ligand with "
        f"{ligand_heavy_count} heavy atom(s)."
    )


def _heavy_atom_count_from_pdb(
    pdb_path: Path,
    *,
    resname: str | None = None,
    resid: str | int | None = None,
) -> int:
    """Count heavy atoms in a PDB, optionally restricted to one residue."""
    u = mda.Universe(str(pdb_path))
    selection = "all"
    clauses: list[str] = []
    if resname:
        clauses.append(f"resname {resname}")
    if resid not in (None, ""):
        clauses.append(f"resid {int(resid)}")
    if clauses:
        selection = " and ".join(clauses)
    atoms = u.select_atoms(selection)
    return sum(1 for atom in atoms if not _is_hydrogen_atom(atom))


def _gen_cv_blocks_from_distance_restraints(work_dir: Path,
                                            restraints: Iterable[Iterable]) -> list[str]:
    """
    Build &colvar DISTANCE blocks from JSON rows:
      [direction, res1, res2, cutoff, force_constant]
    Uses CA atoms from work_dir/full.pdb.
    """
    pdb = work_dir / "full.pdb"
    if not pdb.exists():
        raise FileNotFoundError(f"[extra_conf] Missing full.pdb under {work_dir}")

    u = mda.Universe(pdb.as_posix())
    blocks: list[str] = []
    for row in restraints:
        try:
            direction, res1, res2, cutoff, force_const = row
            direction = str(direction).strip()
            res1 = int(res1); res2 = int(res2)
            cutoff = float(cutoff); force_const = float(force_const)
        except Exception as e:
            raise ValueError(f"[extra_conf] Bad row {row!r}: {e}")

        try:
            a1 = u.select_atoms(f"resid {res1} and name CA")[0].index + 1
            a2 = u.select_atoms(f"resid {res2} and name CA")[0].index + 1
        except Exception:
            raise ValueError(f"[extra_conf] Could not find CA for resid {res1} or {res2} in {pdb.name}")

        # walls: add a small 0.3 Å buffer to avoid overlap
        if direction == ">=":
            lo = max(cutoff - 0.3, 0.0)
            hi = cutoff
            anchors = f"{lo:.3f}, {hi:.3f}, 999, 999"
        elif direction == "<=":
            hi = cutoff + 0.3
            anchors = f"0, 0, {cutoff:.3f}, {hi:.3f}"
        else:
            raise ValueError(f"[extra_conf] Invalid direction {direction!r}; expected '>=' or '<='.")

        blk  = "&colvar\n"
        blk += " cv_type = 'DISTANCE'\n"
        blk += f" cv_ni = 2, cv_i = {a1},{a2}\n"
        blk += f" anchor_position = {anchors}\n"
        blk += f" anchor_strength = {force_const:.6f}, {force_const:.6f}\n"
        blk += "/\n"
        blocks.append(blk)

    return blocks

def _append_or_replace_tagged_block(file_path: Path, tag: str, blocks: list[str]) -> None:
    """
    Idempotently insert or replace a tagged block in file_path.
    Tag markers:
        # {tag} BEGIN
        # {tag} END
    """
    begin = f"# {tag} BEGIN"
    end   = f"# {tag} END"
    new_block = begin + "\n" + "".join(blocks) + end + "\n"

    if not file_path.exists():
        raise FileNotFoundError(f"[extra_conf] {file_path} does not exist")

    text = file_path.read_text()

    # replace if already present, else append
    pattern = re.compile(rf"^#\s*{re.escape(tag)}\s+BEGIN.*?#\s*{re.escape(tag)}\s+END\s*$",
                         flags=re.DOTALL | re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(new_block, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + new_block

    file_path.write_text(text)

def _format_rst_number(value: float | int | str) -> str:
    """Format AMBER restraint scalars with at least one decimal place."""
    rendered = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return rendered if "." in rendered else f"{rendered}.0"


def _parse_colvar_csv(raw: str) -> list[str]:
    """Split a comma-delimited cv.in value while tolerating trailing commas."""
    return [part.strip().strip("'\"") for part in raw.split(",") if part.strip()]


def _extract_colvar_value(block: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([^\n/]+)", block)
    return match.group(1).strip() if match else None


def _iter_colvar_blocks(text: str) -> Iterable[str]:
    """Yield raw &colvar blocks from a cv.in file."""
    for match in re.finditer(r"&colvar\b(.*?)(?:^\s*/\s*$)", text, flags=re.DOTALL | re.MULTILINE):
        yield match.group(1)


def _format_igr_line(label: str, atoms: Sequence[str]) -> str:
    """Wrap igr atom lists over multiple lines and terminate with a trailing zero."""
    tokens = [str(atom).strip() for atom in atoms if str(atom).strip()]
    tokens.append("0")

    lines: list[str] = []
    for idx in range(0, len(tokens), 12):
        chunk = tokens[idx : idx + 12]
        prefix = f" {label}=" if idx == 0 else "      "
        suffix = "," if idx + 12 < len(tokens) else ""
        lines.append(f"{prefix}{','.join(chunk)}{suffix}\n")
    return "".join(lines)


def _render_distance_rst_block(
    atom1: str,
    atom2: str,
    anchors: Sequence[float],
    strengths: Sequence[float],
) -> str:
    return (
        "&rst\n"
        f" iat={atom1},{atom2},\n"
        " r1={r1}, r2={r2}, r3={r3}, r4={r4},\n"
        " rk2={rk2}, rk3={rk3},\n"
        "&end\n"
    ).format(
        r1=_format_rst_number(anchors[0]),
        r2=_format_rst_number(anchors[1]),
        r3=_format_rst_number(anchors[2]),
        r4=_format_rst_number(anchors[3]),
        rk2=_format_rst_number(strengths[0]),
        rk3=_format_rst_number(strengths[1]),
    )


def _render_com_distance_rst_block(
    anchor_atom: str,
    group_atoms: Sequence[str],
    anchors: Sequence[float],
    strengths: Sequence[float],
) -> str:
    return (
        "&rst\n"
        " iat=-1,-1,\n"
        " r1={r1}, r2={r2}, r3={r3}, r4={r4},\n"
        " rk2={rk2}, rk3={rk3},\n"
        "{igr1}"
        "{igr2}"
        "&end\n"
    ).format(
        r1=_format_rst_number(anchors[0]),
        r2=_format_rst_number(anchors[1]),
        r3=_format_rst_number(anchors[2]),
        r4=_format_rst_number(anchors[3]),
        rk2=_format_rst_number(strengths[0]),
        rk3=_format_rst_number(strengths[1]),
        igr1=_format_igr_line("igr1", [anchor_atom]),
        igr2=_format_igr_line("igr2", group_atoms),
    )


def _write_group_colvar_block(
    handle,
    *,
    anchor_atom: str,
    group_atoms: Sequence[str],
    anchors: Sequence[float],
    strengths: Sequence[float],
) -> None:
    """Write a DISTANCE/COM_DISTANCE &colvar block for one anchor atom."""
    handle.write("&colvar\n")
    if len(group_atoms) == 1:
        handle.write(" cv_type = 'DISTANCE'\n")
        handle.write(f" cv_ni = 2, cv_i = {anchor_atom},{group_atoms[0]},\n")
    else:
        handle.write(" cv_type = 'COM_DISTANCE'\n")
        handle.write(f" cv_ni = {len(group_atoms)+2}, cv_i = {anchor_atom},0,")
        for atom in group_atoms:
            handle.write(f"{atom},")
        handle.write("\n")
    handle.write(
        " anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % tuple(anchors)
    )
    handle.write(
        " anchor_strength = %10.4f, %10.4f,\n" % (strengths[0], strengths[1])
    )
    handle.write("/\n")


def _colvar_block_to_rst(block: str) -> str | None:
    """Translate a single AMBER &colvar block into an equivalent &rst block."""
    cv_type = _extract_colvar_value(block, "cv_type")
    cv_i = _extract_colvar_value(block, "cv_i")
    anchor_position = _extract_colvar_value(block, "anchor_position")
    anchor_strength = _extract_colvar_value(block, "anchor_strength")

    if not all((cv_type, cv_i, anchor_position, anchor_strength)):
        raise ValueError(f"Malformed &colvar block; missing required fields:\n{block}")

    atoms = _parse_colvar_csv(cv_i)
    anchors = [float(value) for value in _parse_colvar_csv(anchor_position)]
    strengths = [float(value) for value in _parse_colvar_csv(anchor_strength)]
    cv_type = cv_type.strip("'\"")

    if len(anchors) < 4 or len(strengths) < 2:
        raise ValueError(f"Malformed &colvar block; bad anchors/strengths:\n{block}")

    if cv_type == "DISTANCE":
        if len(atoms) != 2:
            raise ValueError(f"DISTANCE cv_i must contain exactly two atoms:\n{block}")
        return _render_distance_rst_block(atoms[0], atoms[1], anchors, strengths)

    if cv_type == "COM_DISTANCE":
        if len(atoms) < 3 or atoms[1] != "0":
            raise ValueError(f"COM_DISTANCE cv_i must be <atom>,0,<group...>:\n{block}")
        return _render_com_distance_rst_block(atoms[0], atoms[2:], anchors, strengths)

    logger.warning(f"[restraints] Unsupported cv_type={cv_type!r}; skipping disang mirror.")
    return None


def _append_colvar_rst_blocks(cv_file: Path, disang_file: Path) -> None:
    """Append &rst entries derived from every &colvar block in cv_file."""
    rst_blocks = []
    for block in _iter_colvar_blocks(cv_file.read_text()):
        rst_block = _colvar_block_to_rst(block)
        if rst_block:
            rst_blocks.append(rst_block)

    if not rst_blocks:
        return

    existing = disang_file.read_text() if disang_file.exists() else ""
    with disang_file.open("a") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        if existing.strip():
            handle.write("\n")
        handle.write("# Mirrored from cv.in\n")
        for rst_block in rst_blocks:
            handle.write(rst_block)


def _ion_guard_is_enabled(ctx: BuildContext) -> bool:
    value = getattr(ctx.sim, "ion_guard", "yes")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {
        "no",
        "false",
        "0",
        "off",
        "disable",
        "disabled",
    }


def _ion_token_aliases(value: object) -> set[str]:
    token = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    if not token:
        return set()
    aliases = {token}
    aliases.add(re.sub(r"\d+$", "", token))
    return {alias for alias in aliases if alias}


def _first_int_from_keys(data: dict, keys: Sequence[str]) -> int | None:
    for key in keys:
        values = data.get(key) or []
        if not values:
            continue
        try:
            value = int(values[0])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _ion_guard_indices_from_scmask(ctx: BuildContext) -> tuple[list[int], list[int]]:
    """Return (bound-site references, ligand-copy exclusions) from RBFE scmask.json."""
    if str(getattr(ctx, "comp", "")).lower() != "x":
        return [], []
    scmask_path = ctx.window_dir.parent / "x-1" / "scmask.json"
    if not scmask_path.exists():
        return [], []

    try:
        data = json.loads(scmask_path.read_text())
    except Exception as exc:
        logger.warning(f"[restraints:x] ion_guard could not read {scmask_path}: {exc}")
        return [], []

    site_idx = _first_int_from_keys(
        data,
        ("scmk1_cc_site_indices", "scmk2_cc_site_indices"),
    )
    solvent_idx = _first_int_from_keys(
        data,
        ("scmk1_cc_solvent_indices", "scmk2_cc_solvent_indices"),
    )

    reference_indices = [site_idx] if site_idx is not None else []
    exclude_indices: list[int] = []
    for idx in (site_idx, solvent_idx):
        if idx is not None and idx not in exclude_indices:
            exclude_indices.append(idx)
    return reference_indices, exclude_indices


def _first_ligand_heavy_atom_indices(
    universe: mda.Universe,
    ligand_resnames: Sequence[str],
    *,
    limit: int = 1,
) -> list[int]:
    wanted = {
        str(resname).strip()
        for resname in ligand_resnames
        if str(resname).strip()
    }
    if not wanted:
        return []

    refs: list[int] = []
    for residue in universe.residues:
        if str(residue.resname).strip() not in wanted:
            continue
        heavy_atoms = [atom for atom in residue.atoms if not _is_hydrogen_atom(atom)]
        if not heavy_atoms:
            continue
        idx = int(heavy_atoms[0].ix) + 1
        if idx not in refs:
            refs.append(idx)
        if len(refs) >= limit:
            break
    return refs


def _first_ligand_atom_indices(
    universe: mda.Universe,
    ligand_resname: str,
    *,
    limit: int = 2,
) -> list[int]:
    wanted = str(ligand_resname or "").strip()
    if not wanted:
        return []

    refs: list[int] = []
    for residue in universe.residues:
        if str(residue.resname).strip() != wanted or residue.atoms.n_atoms == 0:
            continue
        idx = int(residue.atoms[0].ix) + 1
        if idx not in refs:
            refs.append(idx)
        if len(refs) >= limit:
            break
    return refs


def _append_BULK_LIGAND_restraint(ctx: BuildContext, disang: Path) -> int:
    """Append the z-only flat-bottom restraint between site and bulk ligand atoms."""
    comp = str(getattr(ctx, "comp", "")).lower()
    if comp != "z":
        return 0

    vac_pdb = ctx.window_dir / "vac.pdb"
    if not vac_pdb.exists():
        logger.debug(
            f"[restraints:{comp}] bulk ligand z restraint skipped; missing {vac_pdb}"
        )
        return 0

    try:
        universe = mda.Universe(vac_pdb.as_posix())
    except Exception as exc:
        logger.warning(
            f"[restraints:{comp}] bulk ligand z restraint could not parse "
            f"{vac_pdb}: {exc}"
        )
        return 0

    atom_indices = _first_ligand_atom_indices(universe, ctx.residue_name, limit=2)
    if len(atom_indices) < 2:
        logger.warning(
            f"[restraints:{comp}] bulk ligand z restraint requires two "
            f"{ctx.residue_name!r} residues in {vac_pdb}"
        )
        return 0

    site_idx, bulk_idx = atom_indices[:2]
    existing = disang.read_text() if disang.exists() else ""
    with disang.open("a") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        if existing.strip():
            handle.write("\n")
        handle.write("# Bulk ligand z flat-bottom restraint\n")
        handle.write("&rst\n")
        handle.write("  iat=-1,-1,\n")
        handle.write("  fxyz=0,0,1,\n")
        handle.write(
            "  r1=-999.0, "
            f"r2={-BULK_LIGAND_RESTRAINT_HALF_WIDTH:.1f}, "
            f"r3={BULK_LIGAND_RESTRAINT_HALF_WIDTH:.1f}, "
            "r4=999.0,\n"
        )
        handle.write(
            f"  rk2={BULK_LIGAND_RESTRAINT_FORCE:.1f}, "
            f"rk3={BULK_LIGAND_RESTRAINT_FORCE:.1f},\n"
        )
        handle.write(f"  igr1=2,0,\n")
        handle.write(f"  igr2={bulk_idx},0,\n")
        handle.write(f"&end #{BULK_LIGAND_RESTRAINT_TAG}\n")

    logger.debug(
        f"[restraints:{comp}] bulk ligand z restraint wrote site atom "
        f"2 to bulk atom {bulk_idx}"
    )
    return 1


def _residue_ix_for_atom_indices(
    universe: mda.Universe,
    atom_indices: Sequence[int],
) -> set[int]:
    residue_ix: set[int] = set()
    n_atoms = int(universe.atoms.n_atoms)
    for idx in atom_indices:
        if idx < 1 or idx > n_atoms:
            continue
        residue_ix.add(int(universe.atoms[idx - 1].residue.ix))
    return residue_ix


def _ion_guard_ion_indices(
    universe: mda.Universe,
    ctx: BuildContext,
    *,
    exclude_atom_indices: set[int],
    exclude_residue_ix: set[int],
) -> list[int]:
    ion_aliases: set[str] = set()
    ion_aliases.update(_ion_token_aliases(getattr(ctx.sim, "cation", "Na+")))
    ion_aliases.update(_ion_token_aliases(getattr(ctx.sim, "anion", "Cl-")))
    if not ion_aliases:
        return []

    ion_indices: list[int] = []
    for atom in universe.atoms:
        idx = int(atom.ix) + 1
        if idx in exclude_atom_indices:
            continue
        if int(atom.residue.ix) in exclude_residue_ix:
            continue

        res_aliases = _ion_token_aliases(getattr(atom, "resname", ""))
        element_aliases = _ion_token_aliases(getattr(atom, "element", ""))
        name_aliases = _ion_token_aliases(getattr(atom, "name", ""))
        residue_is_single_atom = len(atom.residue.atoms) == 1

        if (
            res_aliases & ion_aliases
            or element_aliases & ion_aliases
            or (residue_is_single_atom and name_aliases & ion_aliases)
        ):
            ion_indices.append(idx)
    return ion_indices


def _append_ion_guard_restraints(
    ctx: BuildContext,
    disang: Path,
    *,
    ligand_resnames: Sequence[str],
) -> int:
    """Append FE-only ion lower-wall restraints to ``disang.rest``."""
    comp = str(getattr(ctx, "comp", "")).lower()
    if comp not in {"z", "x"} or not _ion_guard_is_enabled(ctx):
        return 0

    full_pdb = ctx.window_dir / "full.pdb"
    if not full_pdb.exists():
        logger.debug(f"[restraints:{comp}] ion_guard skipped; missing {full_pdb}")
        return 0

    try:
        universe = mda.Universe(full_pdb.as_posix())
    except Exception as exc:
        logger.warning(
            f"[restraints:{comp}] ion_guard could not parse {full_pdb}: {exc}"
        )
        return 0

    reference_indices, ligand_exclusion_indices = _ion_guard_indices_from_scmask(ctx)
    fallback_ligand_indices = _first_ligand_heavy_atom_indices(
        universe,
        ligand_resnames,
        limit=2,
    )
    if not reference_indices:
        for idx in fallback_ligand_indices[:1]:
            if idx not in reference_indices:
                reference_indices.append(idx)
            if reference_indices:
                break
    reference_indices = reference_indices[:1]
    if not ligand_exclusion_indices:
        ligand_exclusion_indices = fallback_ligand_indices
    if not reference_indices:
        logger.warning(
            f"[restraints:{comp}] ion_guard enabled but no ligand reference atoms "
            f"were found in {full_pdb}"
        )
        return 0

    exclude_atoms = set(reference_indices) | set(ligand_exclusion_indices)
    exclude_residues = _residue_ix_for_atom_indices(universe, sorted(exclude_atoms))
    ion_indices = _ion_guard_ion_indices(
        universe,
        ctx,
        exclude_atom_indices=exclude_atoms,
        exclude_residue_ix=exclude_residues,
    )
    if not ion_indices:
        logger.debug(
            f"[restraints:{comp}] ion_guard found no configured bulk ions in {full_pdb}"
        )
        return 0

    existing = disang.read_text() if disang.exists() else ""
    written = 0
    with disang.open("a") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        if existing.strip():
            handle.write("\n")
        handle.write(
            "# Ion guard lower-wall restraints: each configured ion to ligand "
            "binding-site reference atom\n"
        )
        for ion_idx in ion_indices:
            for ref_idx in reference_indices:
                if ion_idx == ref_idx:
                    continue
                iat = f"{ion_idx},{ref_idx},"
                handle.write(f"&rst iat={iat:<23s} ")
                handle.write(
                    "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #%s\n"
                    % (
                        0.0,
                        ION_GUARD_DISTANCE,
                        999.0,
                        999.0,
                        ION_GUARD_FORCE,
                        0.0,
                        ION_GUARD_TAG,
                    )
                )
                written += 1

    logger.debug(
        f"[restraints:{comp}] ion_guard wrote {written} restraints for "
        f"{len(ion_indices)} ions and {len(reference_indices)} ligand reference atom"
    )
    return written


def _maybe_append_extra_conf_blocks(ctx: BuildContext, work_dir: Path, cv_file: Path, *, comp: Optional[str]=None) -> None:
    """
    If ctx.extra['extra_conformation_restraints'] is set, parse JSON and append
    the generated &colvar blocks to cv_file (idempotently).
    For FE stage, pass comp (e.g., 'z' or 'o') to honor component gating.
    """
    spec_path = ctx.extra.get("extra_conformation_restraints")
    if not spec_path:
        return
    if ctx.win != -1:
        # load from equil dir
        block_json = ctx.equil_dir / "extra_conf_restraints.json"
        if not block_json.exists():
            raise FileNotFoundError(f"[extra_conf] Expected extra_conf_restraints.json in equil dir: {block_json}")
        _append_or_replace_tagged_block(cv_file, tag="EXTRA_CONFORMATIONAL_REST",
                                       blocks=json.load(block_json.open())['blocks'])
        return
    p = Path(spec_path)
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"[extra_conf] Could not parse {p}: {e}")

    if not isinstance(data, (list, tuple)) or not all(isinstance(r, (list, tuple)) for r in data):
        raise ValueError(f"[extra_conf] JSON must be a list of rows [dir, res1, res2, cutoff, k]. Got: {type(data)}")

    blocks = _gen_cv_blocks_from_distance_restraints(work_dir, data)
    # save blocks
    json.dump({'blocks': blocks}, (work_dir / "extra_conf_restraints.json").open("w"), indent=2)
    _append_or_replace_tagged_block(cv_file, tag="EXTRA_CONFORMATIONAL_REST", blocks=blocks)
    return


# ───────────────────────────── write_equil_restraints (integrated) ─────────────────────────────

def write_equil_restraints(ctx: BuildContext) -> None:
    """
    Generate, in ctx.working_dir:
      - assign.in / assign.dat (reference via cpptraj)
      - disangXX.rest (staged release weights from ctx.sim.release_eq)
      - disang.rest  (copy of last stage)
      - cv.in        (COM restraint)  + (OPTIONAL) EXTRA_CONFORMATIONAL_REST blocks
    Uses anchors saved previously to anchors.json.
    """
    work = ctx.working_dir
    build_dir = ctx.build_dir
    lig = ctx.ligand
    mol = ctx.residue_name

    vac_pdb         = work / "vac.pdb"
    vac_lig_pdb     = work / f"{lig}.pdb"
    vac_lig_prmtop  = work / f"{mol}.prmtop"
    hmr = str(ctx.sim.hmr).lower() == "yes"
    full_prmtop = work / ("full.hmr.prmtop" if hmr else "full.prmtop")
    full_inpcrd     = work / "full.inpcrd"
    lig_mol2        = work / f"{mol}.mol2"
    anchors_pdb     = build_dir / f"equil-{mol}.pdb"

    if not anchors_pdb.exists():
        raise FileNotFoundError(f"Anchor header not found: {anchors_pdb}")
    for p in (vac_pdb, vac_lig_pdb, vac_lig_prmtop, full_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing for restraints: {p}")

    anchors = load_anchors(work)
    P1, P2, P3, L1, L2, L3, lig_res = (
        anchors.P1, anchors.P2, anchors.P3,
        anchors.L1, anchors.L2, anchors.L3,
        anchors.lig_res,
    )

    hvy_h, _ = _collect_calpha_and_lig(vac_pdb, lig_res)

    atm_num         = num_to_mask(vac_pdb.as_posix())
    ligand_atm_num  = num_to_mask(vac_lig_pdb.as_posix())

    # base restraint expressions
    rst, ligand_anchor_rst_count = _equil_anchor_restraint_expressions(
        P1, P2, P3, L1, L2, L3
    )

    msk = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    msk = [m.replace(":1", f":{lig_res}") for m in msk]
    if lig_mol2.exists():
        msk = _filter_sp_carbons(msk, lig_mol2)

    full_rst = [_canonicalize_restraint_expr(expr, atm_num) for expr in (rst + msk)]

    vals = _write_assign_and_read_vals(work, full_rst, full_prmtop, full_inpcrd)

    rest = ctx.sim.rest              # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    release_eq = ctx.sim.release_eq  # e.g., [0, 20, 50, 80, 100]

    # cv.in (protein COM only; ligand solvent restraint is now ntr-based)
    cv_in = work / "cv.in"
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n")
        _write_group_colvar_block(
            cvf,
            anchor_atom="1",
            group_atoms=hvy_h,
            anchors=COM_RESTRAINT_ANCHORS,
            strengths=(5.0, 5.0),
        )

    # ---- integrate extra conformation restraints (equil) ----
    _maybe_append_extra_conf_blocks(ctx, work_dir=work, cv_file=cv_in)

    # single restraint file (no staged ramping)
    #rdsf = rest[1]
    #ldf = rest[2]
    #laf = rest[3]
    #ldhf = rest[4]

    # set all to 0 for equil
    rdsf = 0
    ldf  = 0
    laf  = 0
    ldhf = 0

    outp = work / "disang.rest"
    with outp.open("w") as df:
        l1_label = L1 or "NA"
        l2_label = L2 or "NA"
        l3_label = L3 or "NA"
        df.write(f"# Anchor atoms {P1} {P2} {P3} {l1_label} {l2_label} {l3_label}  stage=equil  weight=100\n")
        for i, expr in enumerate(full_rst):
            fields = expr.split()
            n = len(fields)

            # first 3 are protein distances
            if i < 3 and n == 2:
                iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                         % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                continue

            # TR block
            if 3 <= i < 3 + ligand_anchor_rst_count:
                if n == 2:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf))
                elif n == 3:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf))
                elif n == 4:
                    iat = (
                        f"{_mask_index(atm_num, fields[0])},"
                        f"{_mask_index(atm_num, fields[1])},"
                        f"{_mask_index(atm_num, fields[2])},"
                        f"{_mask_index(atm_num, fields[3])},"
                    )
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, laf, laf))
                continue

            # ligand dihedrals from the ligand prmtop.
            if n == 4:
                try:
                    force_const = 0.0
                    val = _ligand_dihedral_reference_value(
                        vals,
                        i,
                        expr,
                        force_const,
                        "equil",
                    )
                    if val is None:
                        continue
                    iat = (
                        f"{_mask_index(atm_num, fields[0])},"
                        f"{_mask_index(atm_num, fields[1])},"
                        f"{_mask_index(atm_num, fields[2])},"
                        f"{_mask_index(atm_num, fields[3])},"
                    )
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                            % (val - 180.0, val, val, val + 180.0, force_const, force_const))
                except Exception:
                    logger.warning(f"[equil] skipping bad ligand dihedral restraint: {expr}")

    _append_colvar_rst_blocks(cv_in, outp)
    logger.debug(f"[equil] restraints written in {work}")


# ───────────────────────────── FE component restraint writers (integrated) ─────────────────────────────

def _write_component_restraints(ctx: BuildContext, *, skip_lig_tr: bool = False, lig_only: bool = False) -> None:
    """
    Core FE writer: produces cv.in and disang.rest in ctx.window_dir.
    Also appends EXTRA_CONFORMATIONAL_REST blocks to cv.in if ctx.extra specifies a JSON file.
    """
    work = ctx.working_dir
    windows_dir = ctx.window_dir
    lig = ctx.ligand
    mol  = ctx.residue_name
    comp = ctx.comp

    vac_pdb         = windows_dir / "vac.pdb"
    vac_lig_pdb     = windows_dir / f"{lig}.pdb"
    vac_lig_prmtop  = windows_dir / "vac_ligand.prmtop"
    hmr = str(ctx.sim.hmr).lower() == "yes"
    full_prmtop = windows_dir / ("full.hmr.prmtop" if hmr else "full.prmtop")
    full_inpcrd     = windows_dir / "full.inpcrd"
    lig_mol2        = windows_dir / f"{mol}.mol2"

    for p in (vac_pdb, vac_lig_pdb, vac_lig_prmtop, full_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"[restraints:{comp}] missing required file: {p}")

    anchors = load_anchors(work / f"{ctx.comp}_build_files")
    P1, P2, P3 = anchors.P1, anchors.P2, anchors.P3
    p1_res = P1.split('@')[0][1:]
    p2_res = P2.split('@')[0][1:]
    p3_res = P3.split('@')[0][1:]
    p1_atom = P1.split('@')[1]
    p2_atom = P2.split('@')[1]
    p3_atom = P3.split('@')[1]
    # add 1 to Px resid if  dec_method == 'sdr'
    if ctx.sim.dec_method == 'sdr':
        P1 = f":{int(p1_res)+1}@{p1_atom}"
        P2 = f":{int(p2_res)+1}@{p2_atom}"
        P3 = f":{int(p3_res)+1}@{p3_atom}"
        
    L1, L2, L3 = anchors.L1, anchors.L2, anchors.L3
    lig_res    = anchors.lig_res

    if comp in ("v", "o", "z", ):
        offset = 1
    elif comp in ("e", "x"):
        offset = 3
    else:
        offset = 0
    hvy_h, hvy_lig = _collect_calpha_and_lig(vac_pdb, lig_res, offset)
    ligand_heavy_count = _heavy_atom_count_from_pdb(
        vac_pdb,
        resname=mol,
        resid=_resid_from_anchor_mask(L1),
    )
    if ligand_heavy_count == 0:
        ligand_heavy_count = len(hvy_lig)
    atm_num         = num_to_mask(vac_pdb.as_posix())
    ligand_atm_num  = num_to_mask(vac_lig_pdb.as_posix())

    rst: List[str]
    ligand_anchor_rst_count = 0
    if (not lig_only) and (not skip_lig_tr):
        _validate_ligand_anchor_set(
            comp=comp,
            L1=L1,
            L2=L2,
            L3=L3,
            ligand_heavy_count=ligand_heavy_count,
        )
        rst, ligand_anchor_rst_count = _equil_anchor_restraint_expressions(
            P1, P2, P3, L1, L2, L3
        )
    else:
        rst = [f"{P1} {P2}", f"{P2} {P3}", f"{P3} {P1}"]

    # ligand dihedrals
    lig_msks = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    lig_msks = [m.replace(":1", f":{lig_res}") for m in lig_msks]
    if lig_mol2.exists():
        lig_msks = _filter_sp_carbons(lig_msks, lig_mol2)

    rst_full = [_canonicalize_restraint_expr(expr, atm_num) for expr in (rst + lig_msks)]
    vals = _write_assign_and_read_vals(windows_dir, rst_full, full_prmtop, full_inpcrd)

    # weights (single stage in FE)
    rest = ctx.sim.rest  # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    rdhf, rdsf, ldf, laf, ldhf, rcom, lcom = rest

    # cv.in
    cv_in = windows_dir / "cv.in"
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n")
        _write_group_colvar_block(
            cvf,
            anchor_atom="1",
            group_atoms=hvy_h,
            anchors=COM_RESTRAINT_ANCHORS,
            strengths=(rcom, rcom),
        )
        if comp not in {"v", "o", "z"}:
            _write_group_colvar_block(
                cvf,
                anchor_atom="2",
                group_atoms=hvy_lig,
                anchors=COM_RESTRAINT_ANCHORS,
                strengths=(lcom, lcom),
            )

    # ---- integrate extra conformation restraints (FE) only for z/o ----
    if ctx.comp in {"z", "o"}:
        _maybe_append_extra_conf_blocks(ctx, work_dir=windows_dir, cv_file=cv_in, comp=ctx.comp)

    # disang.rest
    disang = windows_dir / "disang.rest"
    with disang.open("w") as df:
        l1_label = L1 or "NA"
        l2_label = L2 or "NA"
        l3_label = L3 or "NA"
        df.write(f"# Anchor atoms {P1} {P2} {P3} {l1_label} {l2_label} {l3_label}  comp={comp}\n")
        for i, expr in enumerate(rst_full):
            fields = expr.split()
            n = len(fields)
            # protein triangle
            if i < 3 and n == 2:
                iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                         % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                continue
            # TR (if included)
            if (
                (not lig_only)
                and (not skip_lig_tr)
                and (i >= 3)
                and (i < 3 + ligand_anchor_rst_count)
            ):
                if n == 2:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf))
                    continue
                if n == 3:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf))
                    continue
                if n == 4:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},{_mask_index(atm_num, fields[3])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, laf, laf))
                    continue
            # ligand dihedrals from the ligand prmtop.
            if n == 4:
                try:
                    force_const = 0.0
                    val = _ligand_dihedral_reference_value(
                        vals,
                        i,
                        expr,
                        force_const,
                        comp,
                    )
                    if val is None:
                        continue
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},{_mask_index(atm_num, fields[3])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                            % (val - 180.0, val, val, val + 180.0, force_const, force_const))
                except Exception:
                    logger.warning(f"[restraints:{comp}] skipping bad ligand dihedral restraint: {expr}")

    _append_BULK_LIGAND_restraint(ctx, disang)
    _append_ion_guard_restraints(ctx, disang, ligand_resnames=[mol])
    _append_colvar_rst_blocks(cv_in, disang)
    # analysis driver
    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write(f"# comp={comp}\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")
        start = 3 if (not lig_only) else 0
        for i, expr in enumerate((rst_full[start:]), start=start):
            arr = expr.split()
            tag = "distance" if len(arr) == 2 else ("angle" if len(arr) == 3 else "dihedral")
            fh.write(f"{tag} r{i} {expr} out restraints.dat\n")

    logger.debug(f"[restraints:{comp}] wrote cv.in (with extras if set), disang.rest, restraints.in in {windows_dir}")


def _lambda_weight_for_window(ctx: BuildContext) -> float:
    lambdas = list(getattr(ctx.sim, "component_lambdas", {}).get(ctx.comp, []) or [])
    if not lambdas:
        lambdas = list(getattr(ctx.sim, "lambdas", []) or [])
    if not lambdas:
        return 0.0
    if ctx.win < 0:
        return float(lambdas[0])
    if ctx.win >= len(lambdas):
        raise IndexError(
            f"[restraints:{ctx.comp}] window {ctx.win} outside lambda schedule of length {len(lambdas)}"
        )
    return float(lambdas[ctx.win])


def _ligand_atom_masks_from_vac_pdb(vac_pdb: Path, mol: str, lig_res: str) -> list[str]:
    universe = mda.Universe(vac_pdb.as_posix())
    ligand_atoms = universe.select_atoms(f"resname {mol} and resid {int(lig_res)}")
    if ligand_atoms.n_atoms == 0:
        ligand_atoms = universe.select_atoms(f"resname {mol}").residues[0].atoms
    masks = ["0"]
    for atom in ligand_atoms:
        masks.append(f":{int(atom.resid)}@{atom.name}")
    return masks


def _ligand_reference_candidates(ctx: BuildContext, windows_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    def _add(path_like) -> None:
        if not path_like:
            return
        path = Path(str(path_like)).expanduser()
        if path.exists() and path not in candidates:
            candidates.append(path)

    index_path = ctx.system_root / "artifacts" / "ligand_params" / "index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text())
            for entry in index_data.get("ligands", []):
                if str(entry.get("ligand")) != str(ctx.ligand):
                    continue
                input_path = entry.get("input_path")
                if input_path and not str(input_path).startswith("BATTER_APO_DUMMY"):
                    _add(input_path)
                meta_path = Path(str(entry.get("store_dir", ""))) / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    input_path = meta.get("input_path")
                    if input_path and not str(input_path).startswith("BATTER_APO_DUMMY"):
                        _add(input_path)
        except Exception as exc:
            logger.warning(f"[restraints:l] could not read ligand input metadata: {exc}")

    params_dir = ctx.system_root / "simulations" / ctx.ligand / "params"
    for meta_name in ("metadata.json", f"{ctx.residue_name}.metadata.json"):
        meta_path = params_dir / meta_name
        if meta_path.exists():
            try:
                input_path = json.loads(meta_path.read_text()).get("input_path")
                if input_path and not str(input_path).startswith("BATTER_APO_DUMMY"):
                    _add(input_path)
            except Exception:
                pass

    for base in (params_dir, windows_dir):
        for ext in ("sdf", "pdb", "mol2"):
            _add(base / f"{ctx.residue_name}.{ext}")
            _add(base / f"{ctx.ligand}.{ext}")

    return candidates


def _load_reference_positions(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    try:
        from rdkit import Chem

        mol = None
        if suffix in {".sdf", ".sd"}:
            supplier = Chem.SDMolSupplier(path.as_posix(), removeHs=False, sanitize=False)
            mol = supplier[0] if len(supplier) else None
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(path.as_posix(), removeHs=False, sanitize=False)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(
                path.as_posix(),
                removeHs=False,
                sanitize=False,
                cleanupSubstructures=False,
            )
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            coords = np.array(
                [
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())
                ],
                dtype=float,
            )
            return coords
    except Exception as exc:
        logger.debug(f"[restraints:l] RDKit could not read {path}: {exc}")

    universe = mda.Universe(path.as_posix())
    return np.asarray(universe.atoms.positions, dtype=float)


def _dihedral_degrees(coords: np.ndarray, indices_1based: Sequence[int]) -> float:
    i, j, k, l = [int(x) - 1 for x in indices_1based]
    p0, p1, p2, p3 = coords[[i, j, k, l]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm == 0.0:
        raise ValueError("zero-length central bond in dihedral reference")
    b1 /= norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _reference_dihedral_values_from_input(
    ctx: BuildContext,
    windows_dir: Path,
    relative_dihedrals: Sequence[Sequence[int]],
) -> tuple[list[float], Path]:
    required_atoms = max(max(dihedral) for dihedral in relative_dihedrals)
    failures: list[str] = []
    for candidate in _ligand_reference_candidates(ctx, windows_dir):
        try:
            coords = _load_reference_positions(candidate)
            if coords.shape[0] < required_atoms:
                failures.append(
                    f"{candidate} has {coords.shape[0]} atoms, needs {required_atoms}"
                )
                continue
            vals = [_dihedral_degrees(coords, dihedral) for dihedral in relative_dihedrals]
            return vals, candidate
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
    raise FileNotFoundError(
        "[restraints:l] could not compute ligand dihedral targets from input/parameter conformer. "
        + "; ".join(failures)
    )


def _ligand_boresch_expressions(P1: str, P2: str, P3: str, L1: str, L2: str, L3: str) -> list[str]:
    if not all([P1, P2, P3, L1, L2, L3]):
        raise ValueError("[restraints:l] Boresch restraints require P1/P2/P3 and L1/L2/L3 anchors")
    return [
        f"{P1} {L1}",
        f"{P2} {P1} {L1}",
        f"{P3} {P2} {P1} {L1}",
        f"{P1} {L1} {L2}",
        f"{P2} {P1} {L1} {L2}",
        f"{P1} {L1} {L2} {L3}",
    ]


def _write_ligand_dihedral_restraints(ctx: BuildContext) -> None:
    """
    Write lambda-scaled Boresch and ligand conformational restraints for component ``l``.

    ``l-1`` carries full-strength targets so ``eq.in`` can ramp the global
    NMR restraint scale with ``&wt type='REST'``. Production windows carry the
    fixed force constant scaled by their lambda value.
    """
    windows_dir = ctx.window_dir
    lig = ctx.ligand
    mol = ctx.residue_name
    comp = ctx.comp

    vac_pdb = windows_dir / "vac.pdb"
    vac_lig_prmtop = windows_dir / "vac_ligand.prmtop"
    hmr = str(ctx.sim.hmr).lower() == "yes"
    full_prmtop = windows_dir / ("full.hmr.prmtop" if hmr else "full.prmtop")
    full_inpcrd = windows_dir / "full.inpcrd"
    lig_mol2 = windows_dir / f"{mol}.mol2"

    for p in (vac_pdb, vac_lig_prmtop, full_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"[restraints:{comp}] missing required file: {p}")

    anchors = load_anchors(ctx.build_dir)
    P1, P2, P3 = anchors.P1, anchors.P2, anchors.P3
    L1, L2, L3 = anchors.L1, anchors.L2, anchors.L3
    lig_res = anchors.lig_res
    atm_num = num_to_mask(vac_pdb.as_posix())
    vac_lig_pdb = windows_dir / "vac_ligand.pdb"
    if not vac_lig_pdb.exists():
        vac_lig_pdb = windows_dir / f"{lig}.pdb"
    if not vac_lig_pdb.exists():
        vac_lig_pdb = windows_dir / f"{mol}.pdb"
    if vac_lig_pdb.exists():
        ligand_atm_num = num_to_mask(vac_lig_pdb.as_posix())
        ligand_heavy_count = _heavy_atom_count_from_pdb(vac_lig_pdb)
    else:
        ligand_atm_num = _ligand_atom_masks_from_vac_pdb(vac_pdb, mol, lig_res)
        ligand_heavy_count = _heavy_atom_count_from_pdb(
            vac_pdb,
            resname=mol,
            resid=_resid_from_anchor_mask(L1) or lig_res,
        )

    if _ligand_anchor_count(L1, L2, L3) == 3:
        boresch_exprs = _ligand_boresch_expressions(P1, P2, P3, L1, L2, L3)
        boresch_exprs = [
            _canonicalize_restraint_expr(expr, atm_num) for expr in boresch_exprs
        ]
        boresch_vals = _write_assign_and_read_vals(
            windows_dir, boresch_exprs, full_prmtop, full_inpcrd
        )
    else:
        _validate_ligand_anchor_set(
            comp=comp,
            L1=L1,
            L2=L2,
            L3=L3,
            ligand_heavy_count=ligand_heavy_count,
        )
        boresch_exprs = []
        boresch_vals = []

    raw_lig_msks = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    if lig_mol2.exists():
        raw_lig_msks = _filter_sp_carbons(raw_lig_msks, lig_mol2)
    relative_dihedrals: list[tuple[int, int, int, int]] = []
    for expr in raw_lig_msks:
        fields = expr.split()
        if len(fields) != 4:
            continue
        try:
            relative_dihedrals.append(tuple(_mask_index(ligand_atm_num, field) for field in fields))
        except ValueError:
            logger.warning(f"[restraints:{comp}] skipping ligand dihedral without source atom map: {expr}")
    lig_msks = [m.replace(":1", f":{lig_res}") for m in raw_lig_msks]
    lig_msks = [_canonicalize_restraint_expr(expr, atm_num) for expr in lig_msks]
    if not lig_msks:
        if ligand_heavy_count < 4:
            logger.warning(
                "[restraints:{}] ligand has only {} heavy atom(s); no ligand "
                "dihedral restraints can be generated.",
                comp,
                ligand_heavy_count,
            )
            vals: list[float] = []
            reference_source: Path | None = None
        else:
            raise ValueError(f"[restraints:{comp}] no ligand heavy-atom dihedrals found for {lig}")
    elif len(relative_dihedrals) != len(lig_msks):
        raise ValueError(
            f"[restraints:{comp}] could not map all ligand dihedrals to input conformer atom order"
        )
    else:
        vals, reference_source = _reference_dihedral_values_from_input(
            ctx,
            windows_dir,
            relative_dihedrals,
        )

    base_force = float(getattr(ctx.sim, "lig_dihcf_force", 0.0) or 0.0)
    window_weight = _lambda_weight_for_window(ctx)
    force_scale = 1.0 if ctx.win < 0 else window_weight
    force_const = base_force * force_scale
    base_distance_force = float(getattr(ctx.sim, "lig_distance_force", 0.0) or 0.0)
    base_angle_force = float(getattr(ctx.sim, "lig_angle_force", 0.0) or 0.0)
    distance_force = base_distance_force * force_scale
    angle_force = base_angle_force * force_scale
    if base_force <= 0.0:
        logger.warning(
            "[restraints:l] lig_dihcf_force is <= 0; component l will not restrain ligand conformations."
        )

    cv_in = windows_dir / "cv.in"
    cv_in.write_text("cv_file\n")

    restraint_records: list[dict[str, object]] = []
    boresch_records: list[dict[str, object]] = []
    used_msks: list[str] = []
    disang = windows_dir / "disang.rest"
    with disang.open("w") as df:
        reference_label = reference_source.as_posix() if reference_source else "none"
        l1_label = L1 or "NA"
        l2_label = L2 or "NA"
        l3_label = L3 or "NA"
        df.write(
            f"# Ligand Boresch and conformational restraints comp={comp} "
            f"base_distance_force={base_distance_force:.8g} "
            f"base_angle_force={base_angle_force:.8g} "
            f"base_dihedral_force={base_force:.8g} lambda={window_weight:.8g} "
            f"force_scale={force_scale:.8g} reference={reference_label}\n"
        )
        df.write(f"# Anchor atoms {P1} {P2} {P3} {l1_label} {l2_label} {l3_label}\n")
        for idx, (expr, val) in enumerate(zip(boresch_exprs, boresch_vals)):
            fields = expr.split()
            n = len(fields)
            try:
                iat = ",".join(str(_mask_index(atm_num, field)) for field in fields) + ","
            except ValueError as exc:
                raise ValueError(f"[restraints:{comp}] could not map Boresch restraint: {expr}") from exc
            df.write(f"&rst iat={iat:<23s} ")
            if n == 2:
                rk = distance_force
                r1, r2, r3, r4 = 0.0, float(val), float(val), 999.0
                kind = "distance"
            elif n == 3:
                rk = angle_force
                r1, r2, r3, r4 = 0.0, float(val), float(val), 180.0
                kind = "angle"
            elif n == 4:
                rk = angle_force
                r1, r2, r3, r4 = float(val) - 180.0, float(val), float(val), float(val) + 180.0
                kind = "dihedral"
            else:
                raise ValueError(f"[restraints:{comp}] invalid Boresch restraint: {expr}")
            df.write(
                "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, "
                "rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                % (r1, r2, r3, r4, rk, rk)
            )
            boresch_records.append(
                {
                    "index": idx,
                    "kind": kind,
                    "mask": expr,
                    "reference": float(val),
                    "base_distance_force_constant": base_distance_force,
                    "base_angle_force_constant": base_angle_force,
                    "lambda": window_weight,
                    "force_scale": force_scale,
                    "force_constant": rk,
                }
            )
        for idx, (expr, val) in enumerate(zip(lig_msks, vals)):
            fields = expr.split()
            if len(fields) != 4:
                continue
            try:
                iat = (
                    f"{_mask_index(atm_num, fields[0])},"
                    f"{_mask_index(atm_num, fields[1])},"
                    f"{_mask_index(atm_num, fields[2])},"
                    f"{_mask_index(atm_num, fields[3])},"
                )
            except ValueError:
                logger.warning(f"[restraints:{comp}] skipping unmapped ligand dihedral: {expr}")
                continue
            dih_force_const = _ligand_dihedral_force_constant(expr, force_const)
            df.write(f"&rst iat={iat:<23s} ")
            df.write(
                "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, "
                "rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                % (
                    float(val) - 180.0,
                    float(val),
                    float(val),
                    float(val) + 180.0,
                    dih_force_const,
                    dih_force_const,
                )
            )
            used_msks.append(expr)
            restraint_records.append(
                {
                    "index": idx,
                    "mask": expr,
                    "reference_degrees": float(val),
                    "base_force_constant": base_force,
                    "lambda": window_weight,
                    "force_scale": force_scale,
                    "force_constant": dih_force_const,
                }
            )

    if lig_msks and not used_msks:
        raise ValueError(f"[restraints:{comp}] no ligand dihedrals could be mapped into vac.pdb")

    (windows_dir / "ligand_dihedral_restraints.json").write_text(
        json.dumps(
            {
                "component": comp,
                "window": ctx.win,
                "boresch_restraints": boresch_records,
                "base_distance_force_constant": base_distance_force,
                "base_angle_force_constant": base_angle_force,
                "base_force_constant": base_force,
                "lambda": window_weight,
                "force_scale": force_scale,
                "restraints": restraint_records,
                "reference_source": reference_source.as_posix() if reference_source else None,
            },
            indent=2,
        )
        + "\n"
    )

    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write(f"# comp={comp} ligand Boresch and conformational restraints\n")
        fh.write("noexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")
        for idx, expr in enumerate(boresch_exprs):
            arr = expr.split()
            tag = "distance" if len(arr) == 2 else ("angle" if len(arr) == 3 else "dihedral")
            fh.write(f"{tag} tr{idx} {expr} out restraints.dat\n")
        for idx, expr in enumerate(used_msks):
            fh.write(f"dihedral r{idx} {expr} out restraints.dat\n")

    logger.debug(
        f"[restraints:{comp}] wrote {len(boresch_records)} Boresch and {len(restraint_records)} ligand dihedral restraints in {windows_dir}"
    )


# ───────────────────────────── registrations ─────────────────────────────

@register_restraints("d")
def _build_restraints_d(builder, ctx: BuildContext) -> None:
    """
    ABFE_diff bound-state restraints.

    The default ``local_frame`` mode pins a small ligand scaffold to a three-
    anchor receptor frame and optionally restrains the ligand scaffold internally.
    This keeps the bound dummy near its initial pose without the receptor-wide
    spring cage produced by the original dense C-alpha/heavy-atom network.
    """
    windows_dir = ctx.window_dir
    comp = ctx.comp
    vac_pdb = windows_dir / "vac.pdb"
    if not vac_pdb.exists():
        raise FileNotFoundError(f"[restraints:{comp}] missing required file: {vac_pdb}")

    universe = mda.Universe(vac_pdb.as_posix())
    ligand_heavy = _select_bound_ligand_heavy_atoms(universe, ctx.residue_name)

    force_const = float(getattr(ctx.sim, "lig_distance_force", 5.0) or 5.0)
    width = float(
        getattr(ctx.sim, "abfe_diff_pose_width", ABFE_DIFF_POSE_WIDTH)
        or ABFE_DIFF_POSE_WIDTH
    )
    mode = str(
        getattr(ctx.sim, "abfe_diff_pose_restraint_type", "local_frame")
        or "local_frame"
    ).lower().replace("-", "_")
    anchor_radius = float(
        getattr(ctx.sim, "abfe_diff_pose_anchor_radius", ABFE_DIFF_POSE_RADIUS)
        or ABFE_DIFF_POSE_RADIUS
    )

    if mode == "dense":
        anchor_atoms = _select_binding_site_calpha_atoms(
            universe,
            ligand_heavy,
            radius=anchor_radius,
            min_atoms=ABFE_DIFF_POSE_MIN_ANCHORS,
            max_atoms=ABFE_DIFF_POSE_MAX_ANCHORS,
        )
        ligand_pose_atoms = list(ligand_heavy)
        include_internal = False
    elif mode == "local_frame":
        anchor_atoms = _select_abfe_diff_receptor_anchors(
            ctx,
            universe,
            vac_pdb,
            ligand_heavy,
            count=int(
                getattr(
                    ctx.sim,
                    "abfe_diff_pose_anchor_count",
                    ABFE_DIFF_POSE_LOCAL_ANCHORS,
                )
                or ABFE_DIFF_POSE_LOCAL_ANCHORS
            ),
            radius=anchor_radius,
        )
        ligand_pose_atoms = _select_ligand_pose_atoms(
            ligand_heavy,
            count=int(
                getattr(
                    ctx.sim,
                    "abfe_diff_pose_ligand_atom_count",
                    ABFE_DIFF_POSE_LIGAND_ATOMS,
                )
                or ABFE_DIFF_POSE_LIGAND_ATOMS
            ),
            preferred_atoms=_preferred_ligand_anchor_atoms(
                ctx, universe, vac_pdb, ligand_heavy
            ),
        )
        include_internal = (
            str(getattr(ctx.sim, "abfe_diff_pose_internal_restraints", "yes")).lower()
            == "yes"
        )
    else:
        raise ValueError(
            f"[restraints:{comp}] unsupported ABFE_diff pose restraint mode: {mode!r}"
        )

    cv_in = windows_dir / "cv.in"
    metadata: list[dict[str, float | int | str]] = []
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n")
        for anchor in anchor_atoms:
            for lig_atom in ligand_pose_atoms:
                target = _write_distance_restraint_block(
                    cvf,
                    anchor,
                    lig_atom,
                    width=width,
                    force_const=force_const,
                )
                metadata.append(
                    {
                        "kind": "external_pose",
                        "anchor_atom_serial": int(anchor.index + 1),
                        "anchor_resid": int(anchor.resid),
                        "anchor_name": str(anchor.name),
                        "ligand_atom_serial": int(lig_atom.index + 1),
                        "ligand_atom_name": str(lig_atom.name),
                        "target_distance": target,
                        "flat_bottom_width": width,
                        "force_constant": force_const,
                    }
                )
        if include_internal:
            for i, atom1 in enumerate(ligand_pose_atoms):
                for atom2 in ligand_pose_atoms[i + 1 :]:
                    target = _write_distance_restraint_block(
                        cvf,
                        atom1,
                        atom2,
                        width=width,
                        force_const=force_const,
                    )
                    metadata.append(
                        {
                            "kind": "ligand_internal",
                            "ligand_atom_serial": int(atom1.index + 1),
                            "ligand_atom_name": str(atom1.name),
                            "ligand_atom2_serial": int(atom2.index + 1),
                            "ligand_atom2_name": str(atom2.name),
                            "target_distance": target,
                            "flat_bottom_width": width,
                            "force_constant": force_const,
                        }
                    )

    disang = windows_dir / "disang.rest"
    disang.write_text(
        f"# ABFE_diff {mode} bound-pose restraints; no Boresch ligand TR terms\n"
    )
    _append_colvar_rst_blocks(cv_in, disang)

    (windows_dir / "abfe_diff_pose_restraints.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "ligand_heavy_atom_serials": [
                    int(atom.index + 1) for atom in ligand_heavy
                ],
                "ligand_pose_atom_serials": [
                    int(atom.index + 1) for atom in ligand_pose_atoms
                ],
                "anchor_atom_serials": [
                    int(atom.index + 1) for atom in anchor_atoms
                ],
                "restraints": metadata,
            },
            indent=2,
        )
    )

    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write(
            f"# ABFE_diff {mode} pose restraints; no Boresch ligand TR metrics\n"
            "noexitonerror\n"
            "parm vac.prmtop\n"
        )
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")

    logger.debug(
        f"[restraints:{comp}] wrote ABFE_diff {mode} pose cv.in, disang.rest, and restraints.in in {windows_dir}"
    )


@register_restraints("v", "o", "z")
def _build_restraints_v_o_z(builder, ctx: BuildContext) -> None:
    _write_component_restraints(ctx, skip_lig_tr=False, lig_only=False)


@register_restraints("l")
def _build_restraints_l(builder, ctx: BuildContext) -> None:
    _write_ligand_dihedral_restraints(ctx)


@register_restraints("y")
def _build_restraints_y(builder, ctx: BuildContext) -> None:
    """
    Ligand-only (solvation FE) restraints:
      - cv.in: placeholder file; ligand solvent restraint now comes from ntr
      - disang.rest: empty (no mirrored ligand COM block)
      - restraints.in: minimal analysis driver (optional)
    """
    windows_dir = ctx.window_dir

    vac_pdb = windows_dir / "vac.pdb"
    if not vac_pdb.exists():
        raise FileNotFoundError(f"[restraints:y] Missing ligand-only vac.pdb: {vac_pdb}")

    # ---- cv.in (placeholder only; solvent ligand restraint is ntr-based) ----
    cv_in = windows_dir / "cv.in"
    cv_in.write_text("cv_file\n")

    disang = windows_dir / "disang.rest"
    disang.write_text("\n")

    # (Optional) very small analysis driver to keep downstream scripts happy
    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write("# ligand-only; no &rst metrics\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")

    logger.debug(f"[restraints:y] wrote placeholder cv.in, empty disang.rest, restraints.in in {windows_dir}")

@register_restraints("m")
def _build_restraints_m(builder, ctx: BuildContext) -> None:
    """
    Ligand-only (vacuum FE) restraints:
      - disang.rest: empty (no AMBER &rst blocks)
      - restraints.in: minimal analysis driver (optional)
    """
    windows_dir = ctx.window_dir
    lig = ctx.ligand
    mol = ctx.residue_name

    vac_pdb = windows_dir / "vac.pdb"
    if not vac_pdb.exists():
        raise FileNotFoundError(f"[restraints:y] Missing ligand-only vac.pdb: {vac_pdb}")

    # read ligand-only coords and collect heavy atom serials (1-based) for AMBER
    u_lig = mda.Universe(vac_pdb.as_posix())
    # prefer selecting by resname if present, otherwise just take all non-H
    try:
        lig_atoms = u_lig.select_atoms(f"resname {mol} and not name H*")
        if lig_atoms.n_atoms == 0:
            lig_atoms = u_lig.select_atoms("not name H*")
    except Exception:
        lig_atoms = u_lig.select_atoms("not name H*")

    if lig_atoms.n_atoms == 0:
        raise RuntimeError("[restraints:y] Found zero ligand heavy atoms in vac.pdb")

    hvy_serials = [str(a.ix + 1) for a in lig_atoms]  # 1-based serials for AMBER masks

    # strengths from sim.rest: [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    rest = ctx.sim.rest
    try:
        lcom = float(rest[6])
    except Exception:
        raise ValueError(f"[restraints:y] Invalid sim.rest; expected length ≥ 7, got: {rest}")

    # ---- disang.rest: empty (legacy behavior) ----
    (windows_dir / "disang.rest").write_text("\n")

    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write("# ligand-only; no &rst metrics\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")

    logger.debug(f"[restraints:y] wrote cv.in (ligand COM only), empty disang.rest, restraints.in in {windows_dir}")


def _atom_name_from_anchor_mask(mask: str | None) -> str:
    if not mask:
        raise ValueError("[restraints:x] Missing ligand anchor atom mask.")
    match = _ANCHOR_MASK_RE.match(str(mask).strip())
    if match:
        return match.group(2)
    if "@" in str(mask):
        return str(mask).rsplit("@", 1)[1].strip()
    return str(mask).strip()


def _optional_atom_name_from_anchor_mask(mask: str | None) -> str:
    return _atom_name_from_anchor_mask(mask) if mask else ""


def _first_residue_with_resname(universe: mda.Universe, resname: str, *, label: str):
    atoms = universe.select_atoms(f"resname {resname}")
    if atoms.n_atoms == 0:
        raise ValueError(
            f"[restraints:x] No {label} residue with resname {resname!r} in vac.pdb"
        )
    return atoms.residues[0]


def _heavy_atom_names_from_residue(residue) -> list[str]:
    names: list[str] = []
    for atom in residue.atoms:
        name = str(atom.name).strip()
        if name and not _is_hydrogen_atom(atom) and name not in names:
            names.append(name)
    return names


def _heavy_atoms_from_residue(residue) -> list:
    atoms: list = []
    seen_names: set[str] = set()
    for atom in residue.atoms:
        name = str(atom.name).strip()
        if not name or name in seen_names or _is_hydrogen_atom(atom):
            continue
        try:
            position = np.asarray(atom.position, dtype=float)
        except Exception:
            continue
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            continue
        atoms.append(atom)
        seen_names.add(name)
    return atoms


def _independent_boresch_atom_names_from_residue(residue, *, label: str) -> list[str]:
    """Choose a deterministic, non-collinear ligand anchor triplet without atom mapping."""
    atoms = _heavy_atoms_from_residue(residue)
    if len(atoms) < 3:
        selected = [str(atom.name).strip() for atom in atoms]
        raise ValueError(
            f"[restraints:x] Need at least 3 {label} heavy atoms for "
            f"Boresch restraints; got {selected}"
        )

    coords = np.asarray([np.asarray(atom.position, dtype=float) for atom in atoms])
    centroid = np.mean(coords, axis=0)
    span = float(np.max(np.linalg.norm(coords - centroid, axis=1)))
    span = max(span, 1.0)

    best_score: float | None = None
    best_indices: tuple[int, int, int] | None = None
    min_anchor_distance = 0.5
    min_sine = 0.25

    for i in range(len(atoms)):
        l1_centrality = np.linalg.norm(coords[i] - centroid) / span
        for j in range(len(atoms)):
            if j == i:
                continue
            d12 = float(np.linalg.norm(coords[j] - coords[i]))
            if d12 < min_anchor_distance:
                continue
            for k in range(len(atoms)):
                if k == i or k == j:
                    continue
                d13 = float(np.linalg.norm(coords[k] - coords[i]))
                d23 = float(np.linalg.norm(coords[k] - coords[j]))
                if min(d13, d23) < min_anchor_distance:
                    continue
                area2 = float(
                    np.linalg.norm(
                        np.cross(coords[j] - coords[i], coords[k] - coords[i])
                    )
                )
                sine = area2 / max(d12 * d13, 1.0e-12)
                if sine < min_sine:
                    continue
                spread = (min(d12, d13) + 0.5 * d23) / span
                score = 4.0 * sine + 0.25 * spread - l1_centrality
                if best_score is None or score > best_score:
                    best_score = score
                    best_indices = (i, j, k)

    if best_indices is None:
        logger.warning(
            "[restraints:x] Could not find a non-collinear {} Boresch anchor "
            "triplet; falling back to the first three heavy atoms.",
            label,
        )
        best_indices = (0, 1, 2)

    return [str(atoms[idx].name).strip() for idx in best_indices]


def _vector_angle_degrees(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float | None:
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1.0e-12:
        return None
    cos_angle = float(np.dot(v1, v2) / denom)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _vector_dihedral_degrees(
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
    if float(np.linalg.norm(v)) < 1.0e-12 or float(np.linalg.norm(w)) < 1.0e-12:
        return None
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def _angle_endpoint_margin_degrees(angle: float) -> float:
    return min(float(angle), 180.0 - float(angle))


def _torsion_endpoint_margin_degrees(torsion: float) -> float:
    folded = abs(((float(torsion) + 180.0) % 360.0) - 180.0)
    return min(folded, 180.0 - folded)


def _boresch_frame_values(
    receptor_atoms: Sequence,
    ligand_atoms: Sequence,
) -> tuple[float, ...] | None:
    if len(receptor_atoms) != 3 or len(ligand_atoms) != 3:
        return None
    p1, p2, p3 = [np.asarray(atom.position, dtype=float) for atom in receptor_atoms]
    l1, l2, l3 = [np.asarray(atom.position, dtype=float) for atom in ligand_atoms]
    values = (
        _vector_angle_degrees(p2, p1, l1),
        _vector_dihedral_degrees(p3, p2, p1, l1),
        _vector_angle_degrees(p1, l1, l2),
        _vector_dihedral_degrees(p2, p1, l1, l2),
        _vector_dihedral_degrees(p1, l1, l2, l3),
    )
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return tuple(float(value) for value in values)


def _boresch_frame_margins(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 5:
        return 0.0, 0.0
    angle_margin = min(
        _angle_endpoint_margin_degrees(values[0]),
        _angle_endpoint_margin_degrees(values[2]),
    )
    torsion_margin = min(
        _torsion_endpoint_margin_degrees(value)
        for value in (values[1], values[3], values[4])
    )
    return float(angle_margin), float(torsion_margin)


def _boresch_frame_values_from_positions(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
) -> tuple[float, ...] | None:
    values = (
        _vector_angle_degrees(p2, p1, l1),
        _vector_dihedral_degrees(p3, p2, p1, l1),
        _vector_angle_degrees(p1, l1, l2),
        _vector_dihedral_degrees(p2, p1, l1, l2),
        _vector_dihedral_degrees(p1, l1, l2, l3),
    )
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return tuple(float(value) for value in values)


def _anchor_mask_from_atom(atom) -> str:
    return f":{int(atom.resid)}@{str(atom.name).strip()}"


def _atom_record_for_anchor_guard(universe: mda.Universe, atm_num: Sequence[str], mask: str | None) -> dict | None:
    if not mask:
        return None
    try:
        amber_iat = _mask_index(atm_num, str(mask))
    except ValueError:
        return {"input_mask": mask, "resolved": False}
    if amber_iat <= 0 or amber_iat > universe.atoms.n_atoms:
        return {"input_mask": mask, "resolved": False, "amber_iat": int(amber_iat)}
    atom = universe.atoms[amber_iat - 1]
    return {
        "input_mask": mask,
        "resolved": True,
        "canonical_pdb_mask": atm_num[amber_iat],
        "pdb_mask": _anchor_mask_from_atom(atom),
        "amber_iat": int(amber_iat),
        "pdb_serial": int(getattr(atom, "id", amber_iat)),
        "atom_index0": int(atom.index),
        "resid": int(atom.resid),
        "resname": str(atom.resname).strip(),
        "name": str(atom.name).strip(),
        "segid": str(getattr(atom, "segid", "")).strip(),
        "chainID": str(getattr(atom, "chainID", "")).strip(),
    }


def _boresch_record_for_masks(
    universe: mda.Universe,
    atm_num: Sequence[str],
    receptor_masks: Sequence[str],
    ligand_masks: Sequence[str],
) -> dict | None:
    receptor_atoms = [
        _resolve_anchor_atom_from_mask(universe, atm_num, mask)
        for mask in receptor_masks[:3]
    ]
    ligand_atoms = [
        _resolve_anchor_atom_from_mask(universe, atm_num, mask)
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


def _preferred_l1_ligand_triplet_candidates(
    residue,
    preferred_first_names: Sequence[str],
) -> list[dict]:
    preferred_names = [
        str(name).strip()
        for name in preferred_first_names
        if str(name).strip()
    ]
    if not preferred_names:
        return []

    atoms = _heavy_atoms_from_residue(residue)
    if len(atoms) < 3:
        return []

    coords = {
        int(i): np.array(atom.position, dtype=float, copy=True)
        for i, atom in enumerate(atoms)
    }
    all_positions = np.asarray([coords[i] for i in range(len(atoms))], dtype=float)
    centroid = all_positions.mean(axis=0)
    span = max(float(np.max(np.linalg.norm(all_positions - centroid, axis=1))), 1.0)

    candidates: list[dict] = []
    for preferred_rank, preferred_name in enumerate(preferred_names):
        l1_indices = [
            i for i, atom in enumerate(atoms)
            if str(atom.name).strip() == preferred_name
        ]
        for i in l1_indices:
            l1_pos = coords[i]
            l1_centrality = float(np.linalg.norm(l1_pos - centroid) / span)
            for j in range(len(atoms)):
                if j == i:
                    continue
                l2_pos = coords[j]
                d12 = float(np.linalg.norm(l2_pos - l1_pos))
                if d12 < 0.5:
                    continue
                for k in range(len(atoms)):
                    if k == i or k == j:
                        continue
                    l3_pos = coords[k]
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
                    candidates.append(
                        {
                            "preferred_rank": preferred_rank,
                            "names": [
                                str(atoms[i].name).strip(),
                                str(atoms[j].name).strip(),
                                str(atoms[k].name).strip(),
                            ],
                            "positions": (l1_pos, l2_pos, l3_pos),
                            "score": local_score,
                        }
                    )

    candidates.sort(
        key=lambda item: (
            int(item["preferred_rank"]),
            -float(item["score"]),
            item["names"],
        )
    )
    return candidates


def _best_preferred_l1_triplet_for_receptor_frame(
    receptor_atoms: Sequence,
    triplet_candidates: Sequence[dict],
) -> dict | None:
    if len(receptor_atoms) != 3:
        return None
    p1, p2, p3 = [np.asarray(atom.position, dtype=float) for atom in receptor_atoms]

    best: dict | None = None
    for candidate in triplet_candidates:
        l1, l2, l3 = candidate["positions"]
        values = _boresch_frame_values_from_positions(p1, p2, p3, l1, l2, l3)
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
        scored = {
            "preferred_rank": int(candidate["preferred_rank"]),
            "names": list(candidate["names"]),
            "values": values,
            "margins": (angle_margin, torsion_margin),
            "score": score,
        }
        if best is None or (
            int(scored["preferred_rank"]),
            -float(scored["score"]),
            scored["names"],
        ) < (
            int(best["preferred_rank"]),
            -float(best["score"]),
            best["names"],
        ):
            best = scored
    return best


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


def _receptor_ca_candidates(universe: mda.Universe, ligand_resnames: Sequence[str]):
    excluded = " ".join(str(name).strip() for name in ligand_resnames if str(name).strip())
    selections = []
    if excluded:
        selections.extend(
            [
                f"protein and name CA and not resname {excluded}",
                f"(not resname {excluded}) and name CA",
            ]
        )
    selections.extend(["protein and name CA", "name CA"])
    for selection in selections:
        try:
            atoms = universe.select_atoms(selection)
        except Exception:
            continue
        if atoms.n_atoms >= 3:
            return atoms
    return universe.atoms[:0]


def _select_receptor_p2_p3_for_preferred_l1_sets(
    *,
    universe: mda.Universe,
    ligand_resnames: Sequence[str],
    p1_atom,
    current_p2_atom,
    current_p3_atom,
    endpoint_specs: Sequence[dict],
    min_anchor_distance: float = 8.0,
    max_candidates: int = 64,
) -> dict | None:
    ca_atoms = _receptor_ca_candidates(universe, ligand_resnames)
    if ca_atoms.n_atoms < 3:
        return None

    prepared_specs: list[dict] = []
    for spec in endpoint_specs:
        triplet_candidates = _preferred_l1_ligand_triplet_candidates(
            spec["residue"],
            spec.get("preferred_first_names") or [],
        )
        if triplet_candidates:
            prepared_specs.append({**spec, "triplet_candidates": triplet_candidates})
    if not prepared_specs:
        return None

    p1_pos = np.asarray(p1_atom.position, dtype=float)
    scored_atoms: list[tuple[float, float, int, object]] = []
    for atom in ca_atoms:
        if int(atom.index) == int(p1_atom.index):
            continue
        if int(atom.residue.ix) == int(p1_atom.residue.ix):
            continue
        distance_to_p1 = float(
            np.linalg.norm(np.asarray(atom.position, dtype=float) - p1_pos)
        )
        scored_atoms.append((abs(distance_to_p1 - 10.0), distance_to_p1, int(atom.index), atom))
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

    def _score_pair(p2_atom, p3_atom) -> tuple[tuple, dict] | None:
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
        receptor_angle = _vector_angle_degrees(
            p1_atom.position,
            p2_atom.position,
            p3_atom.position,
        )
        if receptor_angle is None or not np.isfinite(receptor_angle):
            return None
        receptor_angle_margin = min(float(receptor_angle), 180.0 - float(receptor_angle))
        if receptor_angle_margin < 15.0:
            return None

        receptor_atoms = (p1_atom, p2_atom, p3_atom)
        endpoint_results: dict[str, dict] = {}
        endpoint_score = 0.0
        min_angle_margin = float("inf")
        min_torsion_margin = float("inf")
        max_preferred_rank = 0
        for spec in prepared_specs:
            result = _best_preferred_l1_triplet_for_receptor_frame(
                receptor_atoms,
                spec["triplet_candidates"],
            )
            if result is None:
                return None
            endpoint_results[str(spec["key"])] = result
            endpoint_score += float(result["score"])
            angle_margin, torsion_margin = result["margins"]
            min_angle_margin = min(min_angle_margin, float(angle_margin))
            min_torsion_margin = min(min_torsion_margin, float(torsion_margin))
            max_preferred_rank = max(max_preferred_rank, int(result["preferred_rank"]))

        change_count = int(int(p2_atom.index) != int(current_p2_atom.index)) + int(
            int(p3_atom.index) != int(current_p3_atom.index)
        )
        key = (
            change_count,
            max_preferred_rank,
            -min_torsion_margin,
            -min_angle_margin,
            -endpoint_score,
            abs(float(receptor_angle) - 90.0),
            abs(d12 - 10.0) + abs(d23 - 10.0),
            int(p2_atom.index),
            int(p3_atom.index),
        )
        return key, {
            "P2": _anchor_mask_from_atom(p2_atom),
            "P3": _anchor_mask_from_atom(p3_atom),
            "p2_atom": p2_atom,
            "p3_atom": p3_atom,
            "endpoint_results": endpoint_results,
            "receptor_angle": float(receptor_angle),
            "d12": d12,
            "d23": d23,
        }

    best_key: tuple | None = None
    best: dict | None = None
    for one_change_only in (True, False):
        for p2_atom, p3_atom in _candidate_pairs(one_change_only=one_change_only):
            scored = _score_pair(p2_atom, p3_atom)
            if scored is None:
                continue
            key, result = scored
            if best_key is None or key < best_key:
                best_key = key
                best = result
        if best is not None:
            break
    return best


def _write_septop_anchor_guard_diagnostic(
    *,
    path: Path,
    universe: mda.Universe,
    atm_num: Sequence[str],
    ctx: BuildContext,
    old_receptor_masks: Sequence[str],
    final_receptor_masks: Sequence[str],
    endpoint_data: dict[str, dict],
    allow_receptor_reselection: bool,
    user_anchor_triplet: bool,
) -> None:
    try:
        data = {
            "schema_version": 1,
            "stage": "restraints_x_septop",
            "component": ctx.comp,
            "window_dir": str(ctx.window_dir),
            "build_dir": str(ctx.build_dir),
            "allow_receptor_reselection": bool(allow_receptor_reselection),
            "user_anchor_triplet": bool(user_anchor_triplet),
            "receptor": {
                "reselected": list(old_receptor_masks[:3]) != list(final_receptor_masks[:3]),
                "old": {
                    key: _atom_record_for_anchor_guard(universe, atm_num, mask)
                    for key, mask in zip(("P1", "P2", "P3"), old_receptor_masks[:3])
                },
                "final": {
                    key: _atom_record_for_anchor_guard(universe, atm_num, mask)
                    for key, mask in zip(("P1", "P2", "P3"), final_receptor_masks[:3])
                },
            },
            "endpoints": {},
        }
        for key, endpoint in endpoint_data.items():
            final_masks = endpoint.get("final_ligand_masks") or []
            initial_names = endpoint.get("initial_ligand_names") or []
            final_names = endpoint.get("final_ligand_names") or []
            data["endpoints"][key] = {
                "ligand": endpoint.get("ligand"),
                "residue_name": endpoint.get("residue_name"),
                "resid": endpoint.get("resid"),
                "preferred_first_names": endpoint.get("preferred_first_names") or [],
                "initial_ligand_names": initial_names,
                "final_ligand_names": final_names,
                "ligand_reselected": list(initial_names[:3]) != list(final_names[:3]),
                "final": {
                    label: _atom_record_for_anchor_guard(universe, atm_num, mask)
                    for label, mask in zip(("L1", "L2", "L3"), final_masks)
                },
                "boresch": _boresch_record_for_masks(
                    universe,
                    atm_num,
                    final_receptor_masks,
                    final_masks,
                ),
            }
        path.write_text(json.dumps(data, indent=2) + "\n")
    except Exception as exc:
        logger.warning(
            "[restraints:x] Could not write SEPTOP Boresch anchor diagnostic {}: {}",
            path,
            exc,
        )


def _user_anchor_triplet_was_provided(extra: dict | None) -> bool:
    if not extra:
        return False
    anchors = extra.get("user_anchor_atoms") or ()
    return sum(1 for anchor in anchors if str(anchor).strip()) >= 3


def _frame_safe_boresch_atom_names_from_residue(
    residue,
    *,
    receptor_atoms: Sequence,
    label: str,
    preferred_atom_names: Sequence[str] = (),
    preferred_first_names: Sequence[str] = (),
    require_preferred_first: bool = False,
    min_angle_margin: float = BORESCH_MIN_ANGLE_MARGIN_DEG,
    min_torsion_margin: float = BORESCH_MIN_TORSION_MARGIN_DEG,
) -> list[str]:
    """Choose ligand anchors after checking the full receptor-ligand Boresch frame."""
    atoms = _heavy_atoms_from_residue(residue)
    if len(atoms) < 3:
        selected = [str(atom.name).strip() for atom in atoms]
        raise ValueError(
            f"[restraints:x] Need at least 3 {label} heavy atoms for "
            f"Boresch restraints; got {selected}"
        )

    coords = np.asarray([np.asarray(atom.position, dtype=float) for atom in atoms])
    centroid = np.mean(coords, axis=0)
    span = float(np.max(np.linalg.norm(coords - centroid, axis=1)))
    span = max(span, 1.0)

    preferred_atom_set = {
        str(name).strip()
        for name in preferred_atom_names
        if str(name).strip()
    }
    preferred_names = [
        str(name).strip()
        for name in preferred_first_names
        if str(name).strip()
    ]
    preferred_set = set(preferred_names)

    def _scan_candidates(
        *,
        min_anchor_distance: float,
        min_sine: float,
        angle_margin_cutoff: float,
        torsion_margin_cutoff: float,
    ) -> tuple[
        BoreschCandidate | None,
        BoreschCandidate | None,
        BoreschCandidate | None,
        dict[str, BoreschCandidate],
    ]:
        best_valid: BoreschCandidate | None = None
        best_fallback: BoreschCandidate | None = None
        preferred_triplet_valid: BoreschCandidate | None = None
        preferred_valid: dict[str, BoreschCandidate] = {}

        for i in range(len(atoms)):
            first_name = str(atoms[i].name).strip()
            l1_centrality = np.linalg.norm(coords[i] - centroid) / span
            for j in range(len(atoms)):
                if j == i:
                    continue
                d12 = float(np.linalg.norm(coords[j] - coords[i]))
                if d12 < min_anchor_distance:
                    continue
                for k in range(len(atoms)):
                    if k == i or k == j:
                        continue
                    d13 = float(np.linalg.norm(coords[k] - coords[i]))
                    d23 = float(np.linalg.norm(coords[k] - coords[j]))
                    if min(d13, d23) < min_anchor_distance:
                        continue
                    area2 = float(
                        np.linalg.norm(
                            np.cross(coords[j] - coords[i], coords[k] - coords[i])
                        )
                    )
                    sine = area2 / max(d12 * d13, 1.0e-12)
                    if sine < min_sine:
                        continue

                    values = _boresch_frame_values(
                        receptor_atoms,
                        (atoms[i], atoms[j], atoms[k]),
                    )
                    if values is None:
                        continue
                    angle_margin, torsion_margin = _boresch_frame_margins(values)
                    spread = (min(d12, d13) + 0.5 * d23) / span
                    local_score = 4.0 * sine + 0.25 * spread - l1_centrality
                    endpoint_score = 0.03 * angle_margin + 0.05 * torsion_margin
                    score = local_score + endpoint_score
                    candidate = (
                        score,
                        (i, j, k),
                        values,
                        angle_margin,
                        torsion_margin,
                    )
                    if best_fallback is None or score > best_fallback[0]:
                        best_fallback = candidate
                    if (
                        angle_margin < angle_margin_cutoff
                        or torsion_margin < torsion_margin_cutoff
                    ):
                        continue
                    if best_valid is None or score > best_valid[0]:
                        best_valid = candidate
                    if preferred_atom_set:
                        candidate_names = {
                            str(atoms[idx].name).strip()
                            for idx in (i, j, k)
                        }
                        if candidate_names <= preferred_atom_set and (
                            preferred_triplet_valid is None
                            or score > preferred_triplet_valid[0]
                        ):
                            preferred_triplet_valid = candidate
                    if first_name in preferred_set and (
                        first_name not in preferred_valid
                        or score > preferred_valid[first_name][0]
                    ):
                        preferred_valid[first_name] = candidate

        return best_valid, best_fallback, preferred_triplet_valid, preferred_valid

    def _names_from_candidate(candidate: BoreschCandidate) -> list[str]:
        return [str(atoms[idx].name).strip() for idx in candidate[1]]

    best_valid, best_fallback, preferred_triplet_valid, preferred_valid = (
        _scan_candidates(
            min_anchor_distance=0.5,
            min_sine=0.25,
            angle_margin_cutoff=min_angle_margin,
            torsion_margin_cutoff=min_torsion_margin,
        )
    )

    if preferred_triplet_valid is not None:
        _, _, values, angle_margin, torsion_margin = preferred_triplet_valid
        names = _names_from_candidate(preferred_triplet_valid)
        logger.debug(
            "[restraints:x] Selected {} Boresch anchors {} from mapped common "
            "region with values {} (angle margin {:.1f}, torsion margin {:.1f}).",
            label,
            names,
            [round(value, 3) for value in values],
            angle_margin,
            torsion_margin,
        )
        return names

    for first_name in preferred_names:
        selected = preferred_valid.get(first_name)
        if selected is None:
            continue
        _, _, values, angle_margin, torsion_margin = selected
        names = _names_from_candidate(selected)
        logger.debug(
            "[restraints:x] Selected {} Boresch anchors {} with preferred L1 {} "
            "and values {} (angle margin {:.1f}, torsion margin {:.1f}).",
            label,
            names,
            first_name,
            [round(value, 3) for value in values],
            angle_margin,
            torsion_margin,
        )
        return names

    if require_preferred_first and preferred_names:
        raise ValueError(
            f"No {label} Boresch triplet with preferred L1 in {preferred_names} "
            f"satisfied angle margin >= {min_angle_margin:.1f} deg and torsion "
            f"margin >= {min_torsion_margin:.1f} deg."
        )

    selected = best_valid or best_fallback
    selected_source = (
        "strict_valid"
        if best_valid is not None
        else ("strict_fallback" if best_fallback is not None else None)
    )
    if best_valid is None:
        for min_anchor_distance, min_sine, angle_cutoff, torsion_cutoff in (
            (0.25, 0.10, min(15.0, min_angle_margin), min(5.0, min_torsion_margin)),
            (0.05, 0.01, 0.0, 0.0),
        ):
            (
                relaxed_valid,
                relaxed_fallback,
                _relaxed_preferred_triplet,
                _relaxed_preferred,
            ) = _scan_candidates(
                min_anchor_distance=min_anchor_distance,
                min_sine=min_sine,
                angle_margin_cutoff=angle_cutoff,
                torsion_margin_cutoff=torsion_cutoff,
            )
            relaxed_selected = relaxed_valid or relaxed_fallback
            if relaxed_selected is None:
                continue
            selected = relaxed_selected
            selected_source = "relaxed"
            logger.debug(
                "[restraints:x] Falling back to relaxed {} Boresch anchor "
                "search (min distance {:.2f} A, min sine {:.2f}, angle margin "
                ">= {:.1f} deg, torsion margin >= {:.1f} deg).",
                label,
                min_anchor_distance,
                min_sine,
                angle_cutoff,
                torsion_cutoff,
            )
            break

    if selected is None:
        logger.warning(
            "[restraints:x] Could not find a non-collinear {} Boresch anchor "
            "triplet with receptor-frame checks; falling back to ligand-only "
            "geometry.",
            label,
        )
        return _independent_boresch_atom_names_from_residue(residue, label=label)

    _, _, values, angle_margin, torsion_margin = selected
    names = _names_from_candidate(selected)
    if selected_source == "strict_fallback":
        logger.warning(
            "[restraints:x] Could not find a {} Boresch triplet satisfying "
            "angle margin >= {:.1f} deg and torsion margin >= {:.1f} deg; using "
            "{} with values {} (angle margin {:.1f}, torsion margin {:.1f}).",
            label,
            min_angle_margin,
            min_torsion_margin,
            names,
            [round(value, 3) for value in values],
            angle_margin,
            torsion_margin,
        )
    elif selected_source == "relaxed":
        logger.debug(
            "[restraints:x] Selected {} Boresch anchors {} with relaxed search "
            "and values {} (angle margin {:.1f}, torsion margin {:.1f}).",
            label,
            names,
            [round(value, 3) for value in values],
            angle_margin,
            torsion_margin,
        )
    else:
        logger.debug(
            "[restraints:x] Selected {} Boresch anchors {} with values {} "
            "(angle margin {:.1f}, torsion margin {:.1f}).",
            label,
            names,
            [round(value, 3) for value in values],
            angle_margin,
            torsion_margin,
        )
    return names


def _resolve_ref_boresch_atom_names(
    ref_residue,
    anchor_names: Sequence[str],
    receptor_atoms: Sequence | None = None,
    preferred_atom_names: Sequence[str] = (),
    preferred_first_names: Sequence[str] = (),
) -> list[str]:
    if any(anchor_names):
        logger.debug(
            "[restraints:x] selecting SEPTOP REF Boresch anchors with mapped "
            "common-region preference"
        )
    if receptor_atoms is not None:
        return _frame_safe_boresch_atom_names_from_residue(
            ref_residue,
            receptor_atoms=receptor_atoms,
            label="reference",
            preferred_atom_names=preferred_atom_names,
            preferred_first_names=preferred_first_names,
        )
    return _independent_boresch_atom_names_from_residue(
        ref_residue,
        label="reference",
    )


def _resolve_alt_boresch_atom_names(
    *,
    ref_residue,
    alt_residue,
    ref_names: Sequence[str],
    mapping_path: Path,
    receptor_atoms: Sequence | None = None,
    preferred_atom_names: Sequence[str] = (),
    preferred_first_names: Sequence[str] = (),
) -> list[str]:
    if mapping_path.exists():
        logger.debug(
            "[restraints:x] selecting SEPTOP ALT Boresch anchors with mapped "
            "common-region preference"
        )
    if receptor_atoms is not None:
        return _frame_safe_boresch_atom_names_from_residue(
            alt_residue,
            receptor_atoms=receptor_atoms,
            label="alternate",
            preferred_atom_names=preferred_atom_names,
            preferred_first_names=preferred_first_names,
        )
    return _independent_boresch_atom_names_from_residue(
        alt_residue,
        label="alternate",
    )


def _stable_ranked_ligand_atom_names(system_root: Path, ligand: str) -> list[str]:
    path = system_root / "simulations" / str(ligand) / "equil" / "stable_boresch_distance.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("[restraints:x] Could not read stable Boresch pairs {}: {}", path, exc)
        return []
    if not isinstance(data, dict) or data.get("usable") is False:
        return []
    raw_pairs = data.get("ranked_pairs")
    pairs = raw_pairs if isinstance(raw_pairs, list) else [data]
    names: list[str] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        ligand_record = pair.get("ligand") or {}
        if not isinstance(ligand_record, dict):
            continue
        name = str(ligand_record.get("name", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def _boresch_tr_expressions(
    P1: str,
    P2: str,
    P3: str,
    L1: str,
    L2: str,
    L3: str,
) -> list[str]:
    return [
        f"{P1} {L1}",
        f"{P2} {P1} {L1}",
        f"{P3} {P2} {P1} {L1}",
        f"{P1} {L1} {L2}",
        f"{P2} {P1} {L1} {L2}",
        f"{P1} {L1} {L2} {L3}",
    ]


def _reduced_or_boresch_tr_expressions(
    P1: str,
    P2: str,
    P3: str,
    lig_masks: Sequence[str],
) -> list[str]:
    if len(lig_masks) >= 3:
        return _boresch_tr_expressions(P1, P2, P3, *list(lig_masks[:3]))
    if len(lig_masks) == 2:
        L1, L2 = lig_masks
        return [
            f"{P1} {L1}",
            f"{P2} {P1} {L1}",
            f"{P3} {P2} {P1} {L1}",
            f"{P1} {L1} {L2}",
            f"{P2} {P1} {L1} {L2}",
        ]
    if len(lig_masks) == 1:
        L1 = lig_masks[0]
        return [
            f"{P1} {L1}",
            f"{P2} {P1} {L1}",
            f"{P3} {P2} {P1} {L1}",
        ]
    raise ValueError("[restraints:x] no ligand anchor masks available.")


def _append_x_septop_boresch_restraints(ctx: BuildContext, disang: Path) -> list[str]:
    """Append lambda-dependent Boresch restraints for both site ligands."""
    windows_dir = ctx.window_dir
    extra = ctx.extra or {}
    lig_ref = extra.get("ligand_ref") or ctx.ligand
    lig_alt = extra.get("ligand_alt")
    mol_ref = extra.get("residue_ref") or ctx.residue_name
    mol_alt = extra.get("residue_alt")
    if not mol_alt:
        raise ValueError("[restraints:x] SEPTOP RBFE requires residue_alt metadata.")

    vac_pdb = windows_dir / "vac.pdb"
    hmr = str(ctx.sim.hmr).lower() == "yes"
    full_prmtop = windows_dir / ("full.hmr.prmtop" if hmr else "full.prmtop")
    full_inpcrd = windows_dir / "full.inpcrd"
    for path in (vac_pdb, full_prmtop, full_inpcrd):
        if not path.exists():
            raise FileNotFoundError(f"[restraints:x] missing required file: {path}")

    anchors = load_anchors(ctx.build_dir)
    P1 = _adjust_receptor_anchor_mask(anchors.P1, ctx.sim.dec_method)
    P2 = _adjust_receptor_anchor_mask(anchors.P2, ctx.sim.dec_method)
    P3 = _adjust_receptor_anchor_mask(anchors.P3, ctx.sim.dec_method)
    old_receptor_masks = (P1, P2, P3)

    universe = mda.Universe(vac_pdb.as_posix())
    atm_num = num_to_mask(vac_pdb.as_posix())
    receptor_atoms = []
    for mask in (P1, P2, P3):
        atom = _resolve_anchor_atom_from_mask(universe, atm_num, mask)
        if atom is None:
            raise ValueError(f"[restraints:x] could not resolve SEPTOP receptor anchor {mask!r}")
        receptor_atoms.append(atom)
    ref_residue = _first_residue_with_resname(universe, str(mol_ref), label="reference ligand")
    alt_residue = _first_residue_with_resname(universe, str(mol_alt), label="alternate ligand")
    ref_heavy_names = _heavy_atom_names_from_residue(ref_residue)
    alt_heavy_names = _heavy_atom_names_from_residue(alt_residue)

    anchor_names = [
        _optional_atom_name_from_anchor_mask(anchors.L1),
        _optional_atom_name_from_anchor_mask(anchors.L2),
        _optional_atom_name_from_anchor_mask(anchors.L3),
    ]

    def _unique_names(names: Iterable[str]) -> list[str]:
        out: list[str] = []
        for name in names:
            clean = str(name).strip()
            if clean and clean not in out:
                out.append(clean)
        return out

    mapping_path = windows_dir / "mapping.json"
    ref_common_indices, alt_common_indices = _load_common_core_indices(mapping_path)
    ref_common_names = _mapped_heavy_atom_names_from_residue(
        ref_residue,
        ref_common_indices,
    )
    alt_common_names = _mapped_heavy_atom_names_from_residue(
        alt_residue,
        alt_common_indices,
    )
    ref_common_preference_names = _common_core_boresch_preference_names(
        ref_common_names,
        label="reference",
    )
    alt_common_preference_names = _common_core_boresch_preference_names(
        alt_common_names,
        label="alternate",
    )
    if ref_common_names or alt_common_names:
        logger.debug(
            "[restraints:x] mapped common-region ligand atoms for Boresch "
            "preference: ref={} alt={} usable_ref={} usable_alt={}",
            ref_common_names,
            alt_common_names,
            ref_common_preference_names,
            alt_common_preference_names,
        )

    ref_preferred = _unique_names(
        ref_common_preference_names
        + _stable_ranked_ligand_atom_names(ctx.system_root, str(lig_ref))
    )
    if anchor_names[0] and anchor_names[0] not in ref_preferred:
        ref_preferred.append(anchor_names[0])
    alt_preferred = (
        _stable_ranked_ligand_atom_names(ctx.system_root, str(lig_alt))
        if lig_alt
        else []
    )
    alt_preferred = _unique_names(alt_common_preference_names + alt_preferred)
    ref_names_from_reselect: list[str] | None = None
    alt_names_from_reselect: list[str] | None = None
    user_anchor_triplet = _user_anchor_triplet_was_provided(extra)
    allow_receptor_reselection = not user_anchor_triplet
    if allow_receptor_reselection:
        endpoint_specs: list[dict] = []
        current_frame_failed = False
        for key, residue, preferred in (
            ("ref", ref_residue, ref_preferred),
            ("alt", alt_residue, alt_preferred),
        ):
            if not preferred:
                continue
            candidates = _preferred_l1_ligand_triplet_candidates(residue, preferred)
            if not candidates:
                continue
            endpoint_specs.append(
                {
                    "key": key,
                    "residue": residue,
                    "preferred_first_names": preferred,
                }
            )
            current_result = _best_preferred_l1_triplet_for_receptor_frame(
                receptor_atoms,
                candidates,
            )
            if current_result is None:
                current_frame_failed = True
        if current_frame_failed and endpoint_specs:
            alternate = _select_receptor_p2_p3_for_preferred_l1_sets(
                universe=universe,
                ligand_resnames=[str(mol_ref), str(mol_alt)],
                p1_atom=receptor_atoms[0],
                current_p2_atom=receptor_atoms[1],
                current_p3_atom=receptor_atoms[2],
                endpoint_specs=endpoint_specs,
            )
            if alternate is not None:
                P2 = alternate["P2"]
                P3 = alternate["P3"]
                receptor_atoms = [
                    receptor_atoms[0],
                    alternate["p2_atom"],
                    alternate["p3_atom"],
                ]
                endpoint_results = alternate["endpoint_results"]
                if "ref" in endpoint_results:
                    ref_names_from_reselect = endpoint_results["ref"]["names"]
                if "alt" in endpoint_results:
                    alt_names_from_reselect = endpoint_results["alt"]["names"]
                logger.debug(
                    "[restraints:x] Replacing SEPTOP receptor P2/P3 to keep "
                    "preferred ligand L1 atoms: ({}, {}) -> ({}, {}).",
                    old_receptor_masks[1],
                    old_receptor_masks[2],
                    P2,
                    P3,
                )
    elif ref_preferred or alt_preferred:
        logger.debug(
            "[restraints:x] Explicit create.anchor_atoms triplet was provided; "
            "SEPTOP receptor P2/P3 will not be reselected."
        )

    if len(ref_heavy_names) < 3:
        ref_names = ref_heavy_names
        logger.warning(
            "[restraints:x] reference ligand {}:{} has only {} heavy atom(s); "
            "writing reduced external restraints {}.",
            mol_ref,
            int(ref_residue.resid),
            len(ref_heavy_names),
            ref_names,
        )
    else:
        ref_names = ref_names_from_reselect or _resolve_ref_boresch_atom_names(
            ref_residue,
            anchor_names,
            receptor_atoms=receptor_atoms,
            preferred_atom_names=ref_common_preference_names,
            preferred_first_names=ref_preferred,
        )
    if len(alt_heavy_names) < 3:
        alt_names = alt_heavy_names
        logger.warning(
            "[restraints:x] alternate ligand {}:{} has only {} heavy atom(s); "
            "writing reduced external restraints {}.",
            mol_alt,
            int(alt_residue.resid),
            len(alt_heavy_names),
            alt_names,
        )
    else:
        alt_names = alt_names_from_reselect or _resolve_alt_boresch_atom_names(
            ref_residue=ref_residue,
            alt_residue=alt_residue,
            ref_names=ref_names,
            mapping_path=mapping_path,
            receptor_atoms=receptor_atoms,
            preferred_atom_names=alt_common_preference_names,
            preferred_first_names=alt_preferred,
        )

    ref_lig_masks = [f":{int(ref_residue.resid)}@{name}" for name in ref_names]
    alt_lig_masks = [f":{int(alt_residue.resid)}@{name}" for name in alt_names]
    _write_septop_anchor_guard_diagnostic(
        path=windows_dir / "boresch_anchor_guard.json",
        universe=universe,
        atm_num=atm_num,
        ctx=ctx,
        old_receptor_masks=old_receptor_masks,
        final_receptor_masks=(P1, P2, P3),
        endpoint_data={
            "ref": {
                "ligand": lig_ref,
                "residue_name": mol_ref,
                "resid": int(ref_residue.resid),
                "preferred_first_names": ref_preferred,
                "initial_ligand_names": [name for name in anchor_names if name],
                "final_ligand_names": ref_names,
                "final_ligand_masks": ref_lig_masks,
            },
            "alt": {
                "ligand": lig_alt,
                "residue_name": mol_alt,
                "resid": int(alt_residue.resid),
                "preferred_first_names": alt_preferred,
                "initial_ligand_names": [],
                "final_ligand_names": alt_names,
                "final_ligand_masks": alt_lig_masks,
            },
        },
        allow_receptor_reselection=allow_receptor_reselection,
        user_anchor_triplet=user_anchor_triplet,
    )
    receptor_exprs = [f"{P1} {P2}", f"{P2} {P3}", f"{P3} {P1}"]
    ref_exprs = _reduced_or_boresch_tr_expressions(P1, P2, P3, ref_lig_masks)
    alt_exprs = _reduced_or_boresch_tr_expressions(P1, P2, P3, alt_lig_masks)
    rst_full = [
        _canonicalize_restraint_expr(expr, atm_num)
        for expr in (receptor_exprs + ref_exprs + alt_exprs)
    ]

    vals = _write_assign_and_read_vals(windows_dir, rst_full, full_prmtop, full_inpcrd)
    _rdhf, rdsf, ldf, laf, _ldhf, _rcom, _lcom = ctx.sim.rest

    existing = disang.read_text() if disang.exists() else ""
    with disang.open("a") as df:
        if existing and not existing.endswith("\n"):
            df.write("\n")
        df.write(
            "# SEPTOP lambda-dependent Boresch restraints "
            f"ref={mol_ref}:{int(ref_residue.resid)} alt={mol_alt}:{int(alt_residue.resid)}\n"
        )
        for i, expr in enumerate(rst_full):
            fields = expr.split()
            n = len(fields)
            if i < 3:
                tag = "Rec_C"
            elif i < 3 + len(ref_exprs):
                tag = "Lig_TR_REF"
            else:
                tag = "Lig_TR_ALT"
            if i < 3 and n == 2:
                iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write(
                    "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #%s\n"
                    % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf, tag)
                )
                continue
            if n == 2:
                iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write(
                    "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #%s\n"
                    % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf, tag)
                )
            elif n == 3:
                iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write(
                    "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #%s\n"
                    % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf, tag)
                )
            elif n == 4:
                iat = (
                    f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                    f"{_mask_index(atm_num, fields[2])},{_mask_index(atm_num, fields[3])},"
                )
                df.write(f"&rst iat={iat:<23s} ")
                df.write(
                    "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #%s\n"
                    % (
                        float(vals[i]) - 180.0,
                        float(vals[i]),
                        float(vals[i]),
                        float(vals[i]) + 180.0,
                        laf,
                        laf,
                        tag,
                    )
                )

    logger.debug(
        f"[restraints:x] SEPTOP Boresch anchors ref={ref_lig_masks} alt={alt_lig_masks} written to {disang}"
    )
    return rst_full[3:]

@register_restraints("x")
def _build_restraints_x(builder, ctx: BuildContext) -> None:
    """
    For two ligands
    """
    work = ctx.working_dir
    windows_dir = ctx.window_dir
    lig = ctx.ligand
    septop = str(getattr(ctx.sim, "fe_type", "")).lower() == "relative_septop"
    extra = ctx.extra or {}
    mol_ref = extra.get("residue_ref") or ctx.residue_name
    mol_alt = extra.get("residue_alt")
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    comp = ctx.comp

    vac_pdb         = windows_dir / "vac.pdb"
    vac_ref_prmtop  = windows_dir / f"{mol_ref}.prmtop"
    vac_alt_prmtop  = windows_dir / f"{mol_alt}.prmtop"
    hmr = str(ctx.sim.hmr).lower() == "yes"
    full_prmtop = windows_dir / ("full.hmr.prmtop" if hmr else "full.prmtop")
    full_inpcrd     = windows_dir / "full.inpcrd"
    lig_mol2        = windows_dir / f"{mol_ref}.mol2"

    for p in (vac_pdb, vac_ref_prmtop, vac_alt_prmtop, full_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"[restraints:{comp}] missing required file: {p}")

    anchors = load_anchors(work / f"{ctx.comp}_build_files")
    P1, P2, P3 = anchors.P1, anchors.P2, anchors.P3
    p1_res = P1.split('@')[0][1:]
    p2_res = P2.split('@')[0][1:]
    p3_res = P3.split('@')[0][1:]
    p1_atom = P1.split('@')[1]
    p2_atom = P2.split('@')[1]
    p3_atom = P3.split('@')[1]
    # add 1 to Px resid if  dec_method == 'sdr'
    if ctx.sim.dec_method == 'sdr':
        P1 = f":{int(p1_res)+1}@{p1_atom}"
        P2 = f":{int(p2_res)+1}@{p2_atom}"
        P3 = f":{int(p3_res)+1}@{p3_atom}"
        
    L1, L2, L3 = anchors.L1, anchors.L2, anchors.L3
    lig_res    = anchors.lig_res
    rest = ctx.sim.rest  # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    rdhf, rdsf, ldf, laf, ldhf, rcom, lcom = rest

    hvy_h, _ = _collect_calpha_and_lig(vac_pdb, lig_res, 1)

    # cv.in
    cv_in = windows_dir / "cv.in"
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n")
        _write_group_colvar_block(
            cvf,
            anchor_atom="1",
            group_atoms=hvy_h,
            anchors=COM_RESTRAINT_ANCHORS,
            strengths=(rcom, rcom),
        )

    # ---- integrate extra conformation restraints (FE) only for z/o ----
    _maybe_append_extra_conf_blocks(ctx, work_dir=windows_dir, cv_file=cv_in, comp=ctx.comp)
    
    disang = windows_dir / "disang.rest"
    disang.write_text("")
    septop_exprs: list[str] = []
    if septop:
        septop_exprs = _append_x_septop_boresch_restraints(ctx, disang)
    _append_ion_guard_restraints(
        ctx,
        disang,
        ligand_resnames=[name for name in (mol_ref, mol_alt) if name],
    )
    _append_colvar_rst_blocks(cv_in, disang)

    # analysis driver
    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write(f"# comp={comp}\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")
        for i, expr in enumerate(septop_exprs):
            arr = expr.split()
            tag = "distance" if len(arr) == 2 else ("angle" if len(arr) == 3 else "dihedral")
            fh.write(f"{tag} r{i} {expr} out restraints.dat\n")

    logger.debug(f"[restraints:{comp}] wrote cv.in (with extras if set), disang.rest, restraints.in in {windows_dir}")


def _build_restraints_x_boresch(builder, ctx: BuildContext) -> None:
    """
    For two ligands
    """
    work = ctx.working_dir
    windows_dir = ctx.window_dir
    lig = ctx.ligand
    extra = ctx.extra or {}
    mol_ref = extra.get("residue_ref") or ctx.residue_name
    mol_alt = extra.get("residue_alt")
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    comp = ctx.comp

    vac_pdb         = windows_dir / "vac.pdb"
    vac_ref_pdb     = windows_dir / f"{mol_ref}.pdb"
    vac_ref_prmtop  = windows_dir / f"{mol_ref}.prmtop"
    vac_alt_pdb     = windows_dir / f"{mol_alt}.pdb"
    vac_alt_prmtop  = windows_dir / f"{mol_alt}.prmtop"
    hmr = str(ctx.sim.hmr).lower() == "yes"
    full_prmtop = windows_dir / ("full.hmr.prmtop" if hmr else "full.prmtop")
    full_inpcrd     = windows_dir / "full.inpcrd"
    lig_mol2        = windows_dir / f"{mol_ref}.mol2"

    for p in (
        vac_pdb,
        vac_ref_pdb,
        vac_ref_prmtop,
        vac_alt_pdb,
        vac_alt_prmtop,
        full_prmtop,
        full_inpcrd,
    ):
        if not p.exists():
            raise FileNotFoundError(f"[restraints:{comp}] missing required file: {p}")

    anchors = load_anchors(work / f"{ctx.comp}_build_files")
    P1, P2, P3 = anchors.P1, anchors.P2, anchors.P3
    p1_res = P1.split('@')[0][1:]
    p2_res = P2.split('@')[0][1:]
    p3_res = P3.split('@')[0][1:]
    p1_atom = P1.split('@')[1]
    p2_atom = P2.split('@')[1]
    p3_atom = P3.split('@')[1]
    # add 1 to Px resid if  dec_method == 'sdr'
    if ctx.sim.dec_method == 'sdr':
        P1 = f":{int(p1_res)+1}@{p1_atom}"
        P2 = f":{int(p2_res)+1}@{p2_atom}"
        P3 = f":{int(p3_res)+1}@{p3_atom}"
        
    L1, L2, L3 = anchors.L1, anchors.L2, anchors.L3
    lig_res    = anchors.lig_res

    offset = 3
    hvy_h, hvy_lig = _collect_calpha_and_lig(vac_pdb, lig_res, offset)
    atm_num         = num_to_mask(vac_pdb.as_posix())
    ligand_atm_num  = num_to_mask(vac_ref_pdb.as_posix())

    # protein triad
    rst: List[str] = [f"{P1} {P2}", f"{P2} {P3}", f"{P3} {P1}"]
    # TR chain (unless skipping or ligand-only)
    if (not lig_only) and (not skip_lig_tr):
        rst += [
            f"{P1} {L1}",
            f"{P2} {P1} {L1}",
            f"{P3} {P2} {P1} {L1}",
            f"{P1} {L1} {L2}",
            f"{P2} {P1} {L1} {L2}",
            f"{P1} {L1} {L2} {L3}",
        ]

    # ligand dihedrals
    lig_msks = _scan_dihedrals_from_prmtop(vac_ref_prmtop, ligand_atm_num)
    lig_msks = [m.replace(":1", f":{lig_res}") for m in lig_msks]
    if lig_mol2.exists():
        lig_msks = _filter_sp_carbons(lig_msks, lig_mol2)

    rst_full = [_canonicalize_restraint_expr(expr, atm_num) for expr in (rst + lig_msks)]
    vals = _write_assign_and_read_vals(windows_dir, rst_full, full_prmtop, full_inpcrd)

    # weights (single stage in FE)
    rest = ctx.sim.rest  # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    rdhf, rdsf, ldf, laf, ldhf, rcom, lcom = rest

    # cv.in
    cv_in = windows_dir / "cv.in"
    with cv_in.open("w") as cvf:
        # protein COM restraint
        cvf.write("cv_file\n&colvar\n")
        if len(hvy_h) == 1:
            # if only one atom, use DISTANCE instead of COM_DISTANCE
            cvf.write(" cv_type = 'DISTANCE'\n")
            cvf.write(f" cv_ni = 2, cv_i = 1,{hvy_h[0]},\n")
        else:
            cvf.write(" cv_type = 'COM_DISTANCE'\n")
            cvf.write(f" cv_ni = {len(hvy_h)+2}, cv_i = 1,0,")
            for a in hvy_h:
                cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % COM_RESTRAINT_ANCHORS)
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (rcom, rcom))
        cvf.write("/\n")

        # ligand COM restraint
        cvf.write("&colvar\n")
        if len(hvy_lig) == 1:
            # if only one atom, use DISTANCE instead of COM_DISTANCE
            cvf.write(" cv_type = 'DISTANCE'\n")
            cvf.write(f" cv_ni = 2, cv_i = 1,{hvy_lig[0]},\n")
        else:
            cvf.write(" cv_type = 'COM_DISTANCE'\n")
            cvf.write(f" cv_ni = {len(hvy_lig)+2}, cv_i = 2,0,")
            for a in hvy_lig:
                cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % COM_RESTRAINT_ANCHORS)
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (lcom, lcom))
        cvf.write("/\n")

    # ---- integrate extra conformation restraints (FE) only for z/o ----
    if ctx.comp in {"z", "o"}:
        _maybe_append_extra_conf_blocks(ctx, work_dir=windows_dir, cv_file=cv_in, comp=ctx.comp)

    # disang.rest
    disang = windows_dir / "disang.rest"
    with disang.open("w") as df:
        df.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3}  comp={comp}\n")
        for i, expr in enumerate(rst_full):
            fields = expr.split()
            n = len(fields)
            # protein triangle
            if i < 3 and n == 2:
                iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                         % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                continue
            # TR (if included)
            if (not lig_only) and (i >= 3) and (i < 9):
                if n == 2:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf))
                    continue
                if n == 3:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf))
                    continue
                if n == 4:
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},{_mask_index(atm_num, fields[3])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, laf, laf))
                    continue
            # ligand dihedrals from the ligand prmtop.
            if n == 4:
                try:
                    force_const = 0.0
                    val = _ligand_dihedral_reference_value(
                        vals,
                        i,
                        expr,
                        force_const,
                        comp,
                    )
                    if val is None:
                        continue
                    iat = f"{_mask_index(atm_num, fields[0])},{_mask_index(atm_num, fields[1])},{_mask_index(atm_num, fields[2])},{_mask_index(atm_num, fields[3])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                            % (val - 180.0, val, val, val + 180.0, force_const, force_const))
                except Exception:
                    logger.warning(f"[restraints:{comp}] skipping bad ligand dihedral restraint: {expr}")

    _append_colvar_rst_blocks(cv_in, disang)
    # analysis driver
    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write(f"# comp={comp}\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")
        start = 3 if (not lig_only) else 0
        for i, expr in enumerate((rst_full[start:]), start=start):
            arr = expr.split()
            tag = "distance" if len(arr) == 2 else ("angle" if len(arr) == 3 else "dihedral")
            fh.write(f"{tag} r{i} {expr} out restraints.dat\n")

    logger.debug(f"[restraints:{comp}] wrote cv.in (with extras if set), disang.rest, restraints.in in {windows_dir}")
