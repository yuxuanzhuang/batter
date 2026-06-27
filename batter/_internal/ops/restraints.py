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
COM_RESTRAINT_ANCHORS = (0.0, 0.0, 0.0, 999.0)
ABFE_DIFF_POSE_RADIUS = 8.0
ABFE_DIFF_POSE_MAX_ANCHORS = 8
ABFE_DIFF_POSE_MIN_ANCHORS = 4
ABFE_DIFF_POSE_WIDTH = 0.5
ABFE_DIFF_POSE_LOCAL_ANCHORS = 3
ABFE_DIFF_POSE_LIGAND_ATOMS = 6
_ANCHOR_MASK_RE = re.compile(r"^:(-?\d+)@(.+)$")

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


def _resolve_anchor_atom_from_mask(
    universe: mda.Universe,
    atm_num: Sequence[str],
    mask: str,
) -> object | None:
    """Resolve an Amber-style atom mask to an MDAnalysis atom from ``vac.pdb``."""
    try:
        serial = atm_num.index(mask)
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
    """Load 0-based (ref_indices, alt_indices) from RBFE mapping JSON."""
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

    # scmk1_cc_indices as ref_indices
    ref_indices = sorted(data.get("scmk1_cc_solvent_indices", []))
    # scmk2_cc_indices as alt_indices
    alt_indices = sorted(data.get("scmk2_cc_solvent_indices", []))
    return ref_indices, alt_indices


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

def _write_assign_and_read_vals(work: Path, rst_exprs: List[str], prmtop: Path, traj: Path) -> List[float]:
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
    vals = assign_dat[1].split()
    # legacy rotation: shift first to end, drop last
    vals.append(vals.pop(0))
    vals = vals[:-1]
    return [float(v) for v in vals]


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

    full_rst = rst + msk

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
                iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                         % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                continue

            # TR block
            if 3 <= i < 3 + ligand_anchor_rst_count:
                if n == 2:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf))
                elif n == 3:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf))
                elif n == 4:
                    iat = (
                        f"{atm_num.index(fields[0])},"
                        f"{atm_num.index(fields[1])},"
                        f"{atm_num.index(fields[2])},"
                        f"{atm_num.index(fields[3])},"
                    )
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, laf, laf))
                continue

            # disable ligand dihedrals
            if False:
                if n == 4:
                    try:
                        iat = (
                            f"{atm_num.index(fields[0])},"
                            f"{atm_num.index(fields[1])},"
                            f"{atm_num.index(fields[2])},"
                            f"{atm_num.index(fields[3])},"
                        )
                        df.write(f"&rst iat={iat:<23s} ")
                        df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                                % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, ldhf, ldhf))
                    except:
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
    atm_num         = num_to_mask(vac_pdb.as_posix())
    ligand_atm_num  = num_to_mask(vac_lig_pdb.as_posix())

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
    lig_msks = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    lig_msks = [m.replace(":1", f":{lig_res}") for m in lig_msks]
    if lig_mol2.exists():
        lig_msks = _filter_sp_carbons(lig_msks, lig_mol2)

    rst_full = rst + lig_msks
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
        df.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3}  comp={comp}\n")
        for i, expr in enumerate(rst_full):
            fields = expr.split()
            n = len(fields)
            # protein triangle
            if i < 3 and n == 2:
                iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                         % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                continue
            # TR (if included)
            if (not lig_only) and (i >= 3) and (i < 9):
                if n == 2:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf))
                    continue
                if n == 3:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf))
                    continue
                if n == 4:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},{atm_num.index(fields[3])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, laf, laf))
                    continue
            # ligand dihedrals
            if False:
                if n == 4:
                    try:
                        iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},{atm_num.index(fields[3])},"
                        df.write(f"&rst iat={iat:<23s} ")
                        df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                                % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, ldhf, ldhf))
                    except:
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


def _write_ligand_dihedral_restraints(ctx: BuildContext) -> None:
    """
    Write only ligand conformational dihedral restraints for component ``l``.

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
    lig_res = anchors.lig_res

    atm_num = num_to_mask(vac_pdb.as_posix())
    vac_lig_pdb = windows_dir / "vac_ligand.pdb"
    if not vac_lig_pdb.exists():
        vac_lig_pdb = windows_dir / f"{lig}.pdb"
    if not vac_lig_pdb.exists():
        vac_lig_pdb = windows_dir / f"{mol}.pdb"
    if vac_lig_pdb.exists():
        ligand_atm_num = num_to_mask(vac_lig_pdb.as_posix())
    else:
        ligand_atm_num = _ligand_atom_masks_from_vac_pdb(vac_pdb, mol, lig_res)
    raw_lig_msks = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    if lig_mol2.exists():
        raw_lig_msks = _filter_sp_carbons(raw_lig_msks, lig_mol2)
    relative_dihedrals: list[tuple[int, int, int, int]] = []
    for expr in raw_lig_msks:
        fields = expr.split()
        if len(fields) != 4:
            continue
        try:
            relative_dihedrals.append(tuple(ligand_atm_num.index(field) for field in fields))
        except ValueError:
            logger.warning(f"[restraints:{comp}] skipping ligand dihedral without source atom map: {expr}")
    lig_msks = [m.replace(":1", f":{lig_res}") for m in raw_lig_msks]
    if not lig_msks:
        raise ValueError(f"[restraints:{comp}] no ligand heavy-atom dihedrals found for {lig}")
    if len(relative_dihedrals) != len(lig_msks):
        raise ValueError(
            f"[restraints:{comp}] could not map all ligand dihedrals to input conformer atom order"
        )

    vals, reference_source = _reference_dihedral_values_from_input(
        ctx,
        windows_dir,
        relative_dihedrals,
    )

    base_force = float(getattr(ctx.sim, "lig_dihcf_force", 0.0) or 0.0)
    window_weight = _lambda_weight_for_window(ctx)
    force_scale = 1.0 if ctx.win < 0 else window_weight
    force_const = base_force * force_scale
    if base_force <= 0.0:
        logger.warning(
            "[restraints:l] lig_dihcf_force is <= 0; component l will not restrain ligand conformations."
        )

    cv_in = windows_dir / "cv.in"
    cv_in.write_text("cv_file\n")

    restraint_records: list[dict[str, object]] = []
    used_msks: list[str] = []
    disang = windows_dir / "disang.rest"
    with disang.open("w") as df:
        df.write(
            f"# Ligand conformational dihedral restraints comp={comp} "
            f"base_force={base_force:.8g} lambda={window_weight:.8g} "
            f"force_scale={force_scale:.8g} reference={reference_source}\n"
        )
        for idx, (expr, val) in enumerate(zip(lig_msks, vals)):
            fields = expr.split()
            if len(fields) != 4:
                continue
            try:
                iat = (
                    f"{atm_num.index(fields[0])},"
                    f"{atm_num.index(fields[1])},"
                    f"{atm_num.index(fields[2])},"
                    f"{atm_num.index(fields[3])},"
                )
            except ValueError:
                logger.warning(f"[restraints:{comp}] skipping unmapped ligand dihedral: {expr}")
                continue
            df.write(f"&rst iat={iat:<23s} ")
            df.write(
                "r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, "
                "rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                % (
                    float(val) - 180.0,
                    float(val),
                    float(val),
                    float(val) + 180.0,
                    force_const,
                    force_const,
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
                    "force_constant": force_const,
                }
            )

    if not used_msks:
        raise ValueError(f"[restraints:{comp}] no ligand dihedrals could be mapped into vac.pdb")

    (windows_dir / "ligand_dihedral_restraints.json").write_text(
        json.dumps(
            {
                "component": comp,
                "window": ctx.win,
                "base_force_constant": base_force,
                "lambda": window_weight,
                "force_scale": force_scale,
                "restraints": restraint_records,
                "reference_source": reference_source.as_posix(),
            },
            indent=2,
        )
        + "\n"
    )

    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write(f"# comp={comp} ligand conformational dihedral restraints\n")
        fh.write("noexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")
        for idx, expr in enumerate(used_msks):
            fh.write(f"dihedral r{idx} {expr} out restraints.dat\n")

    logger.debug(
        f"[restraints:{comp}] wrote {len(restraint_records)} ligand dihedral restraints in {windows_dir}"
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

@register_restraints("x")
def _build_restraints_x(builder, ctx: BuildContext) -> None:
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
    _append_colvar_rst_blocks(cv_in, disang)

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

    rst_full = rst + lig_msks
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
                iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                         % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                continue
            # TR (if included)
            if (not lig_only) and (i >= 3) and (i < 9):
                if n == 2:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, ldf, ldf))
                    continue
                if n == 3:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (0.0, float(vals[i]), float(vals[i]), 180.0, laf, laf))
                    continue
                if n == 4:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},{atm_num.index(fields[3])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_TR\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, laf, laf))
                    continue
            # ligand dihedrals
            if False:
                if n == 4:
                    try:
                        iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},{atm_num.index(fields[3])},"
                        df.write(f"&rst iat={iat:<23s} ")
                        df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                                % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, ldhf, ldhf))
                    except:
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
