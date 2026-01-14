from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Iterable

import json
import re
import MDAnalysis as mda
from loguru import logger

from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_restraints
from batter._internal.ops.helpers import num_to_mask, load_anchors
from batter.utils import run_with_log, cpptraj

ION_NAMES = {"Na+", "K+", "Cl-", "NA", "CL", "K"}  # NA/CL appear in some pdbs too


# ────────────────────────── small helpers (working-dir aware) ──────────────────────────

def _is_atom_line(line: str) -> bool:
    tag = line[0:6].strip()
    return tag == "ATOM" or tag == "HETATM"

def _field(line: str, start: int, end: int) -> str:
    return line[start:end].strip()

def _collect_backbone_heavy_and_lig(vac_pdb: Path, lig_res: str, offset: int = 0) -> List[List[str]]:
    """Return ([protein_backbone_heavy_atom_serials], [ligand_heavy_atom_serials])."""
    hvy = []
    hvy_lig = []
    with vac_pdb.open() as f:
        for line in f:
            if not _is_atom_line(line):
                continue
            resi = int(_field(line, 22, 26) or "0")
            if 2 <= resi < int(lig_res):
                name = _field(line, 12, 16)
                if name in ("CA", "N", "C", "O"):
                    hvy.append(_field(line, 6, 11))  # atom serial as string
            elif resi == int(lig_res) + offset:
                name = _field(line, 12, 16)
                if name and name[0] != "H":
                    hvy_lig.append(_field(line, 6, 11))
    return hvy, hvy_lig

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


# ───────────────────── extra conformation restraints (helpers) ─────────────────────

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

        # walls: add a small 0.3 Å buffer like your original code
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
    full_hmr_prmtop = work / "full.hmr.prmtop"
    full_inpcrd     = work / "full.inpcrd"
    lig_mol2        = work / f"{mol}.mol2"
    anchors_pdb     = build_dir / f"equil-{mol}.pdb"

    if not anchors_pdb.exists():
        raise FileNotFoundError(f"Anchor header not found: {anchors_pdb}")
    for p in (vac_pdb, vac_lig_pdb, vac_lig_prmtop, full_hmr_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing for restraints: {p}")

    anchors = load_anchors(work)
    P1, P2, P3, L1, L2, L3, lig_res = (
        anchors.P1, anchors.P2, anchors.P3,
        anchors.L1, anchors.L2, anchors.L3,
        anchors.lig_res,
    )

    hvy_h, _ = _collect_backbone_heavy_and_lig(vac_pdb, lig_res)

    atm_num         = num_to_mask(vac_pdb.as_posix())
    ligand_atm_num  = num_to_mask(vac_lig_pdb.as_posix())

    # base restraint expressions
    rst: List[str] = []
    rst += [f"{P1} {P2}", f"{P2} {P3}", f"{P3} {P1}"]  # protein triangle
    rst += [
        f"{P1} {L1}",
        f"{P2} {P1} {L1}",
        f"{P3} {P2} {P1} {L1}",
        f"{P1} {L1} {L2}",
        f"{P2} {P1} {L1} {L2}",
        f"{P1} {L1} {L2} {L3}",
    ]

    msk = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    msk = [m.replace(":1", f":{lig_res}") for m in msk]
    if lig_mol2.exists():
        msk = _filter_sp_carbons(msk, lig_mol2)

    full_rst = rst + msk

    vals = _write_assign_and_read_vals(work, full_rst, full_hmr_prmtop, full_inpcrd)

    rest = ctx.sim.rest              # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    release_eq = ctx.sim.release_eq  # e.g., [0, 20, 50, 80, 100]

    # cv.in (COM)
    cv_in = work / "cv.in"
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        cvf.write(f" cv_ni = {len(hvy_h)+2}, cv_i = 1,0,")
        for a in hvy_h:
            cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 1.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (rest[5], rest[5]))
        cvf.write("/\n")

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
        df.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3}  stage=equil  weight=100\n")
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
            if 3 <= i < 9:
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
    full_hmr_prmtop = windows_dir / "full.hmr.prmtop"
    full_inpcrd     = windows_dir / "full.inpcrd"
    lig_mol2        = windows_dir / f"{mol}.mol2"

    for p in (vac_pdb, vac_lig_pdb, vac_lig_prmtop, full_hmr_prmtop, full_inpcrd):
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
    hvy_h, hvy_lig = _collect_backbone_heavy_and_lig(vac_pdb, lig_res, offset)
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
    vals = _write_assign_and_read_vals(windows_dir, rst_full, full_hmr_prmtop, full_inpcrd)

    # weights (single stage in FE)
    rest = ctx.sim.rest  # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    rdhf, rdsf, ldf, laf, ldhf, rcom, lcom = rest

    # cv.in
    cv_in = windows_dir / "cv.in"
    with cv_in.open("w") as cvf:
        # protein COM restraint
        cvf.write("cv_file\n&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        cvf.write(f" cv_ni = {len(hvy_h)+2}, cv_i = 1,0,")
        for a in hvy_h: cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 1.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (rcom, rcom))
        cvf.write("/\n")

        # ligand COM restraint
        cvf.write("&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        cvf.write(f" cv_ni = {len(hvy_lig)+2}, cv_i = 2,0,")
        for a in hvy_lig: cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 1.0, 999.0))
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
      - cv.in: one COM_DISTANCE block using ligand heavy atoms
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

    # ---- cv.in (single ligand COM restraint) ----
    cv_in = windows_dir / "cv.in"
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n")
        cvf.write("&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        # cv_ni = (#heavy + 2), cv_i starts with "1,0," then heavy serials
        cvf.write(f" cv_ni = {len(hvy_serials) + 2}, cv_i = 1,0,")
        cvf.write(",".join(hvy_serials))
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 1.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (lcom, lcom))
        cvf.write("/\n")

    # ---- disang.rest: empty (legacy behavior) ----
    (windows_dir / "disang.rest").write_text("\n")

    # (Optional) very small analysis driver to keep downstream scripts happy
    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write("# ligand-only; no &rst metrics\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")

    logger.debug(f"[restraints:y] wrote cv.in (ligand COM only), empty disang.rest, restraints.in in {windows_dir}")

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

    # (Optional) very small analysis driver to keep downstream scripts happy
    rest_in = windows_dir / "restraints.in"
    with rest_in.open("w") as fh:
        fh.write("# ligand-only; no &rst metrics\nnoexitonerror\nparm vac.prmtop\n")
        for k in range(2, 11):
            fh.write(f"trajin md{k:02d}.nc\n")

    logger.debug(f"[restraints:y] wrote cv.in (ligand COM only), empty disang.rest, restraints.in in {windows_dir}")


@register_restraints("x")
def _build_restraints_x(builder, ctx: BuildContext) -> None:
    _write_component_restraints(ctx, skip_lig_tr=False, lig_only=False)
    # (append x-specific reference-ligand restraints here if needed)
