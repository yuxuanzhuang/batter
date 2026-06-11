from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from math import ceil
import json
import re
import MDAnalysis as mda
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


# ───────────────────────────── registrations ─────────────────────────────

@register_restraints("v", "o", "z")
def _build_restraints_v_o_z(builder, ctx: BuildContext) -> None:
    _write_component_restraints(ctx, skip_lig_tr=False, lig_only=False)


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
