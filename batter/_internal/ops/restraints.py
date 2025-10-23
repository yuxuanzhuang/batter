from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import MDAnalysis as mda
from loguru import logger

from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_restraints
from batter._internal.ops.helpers import num_to_mask, load_anchors, Anchors, save_anchors
from batter.utils import run_with_log, cpptraj


ION_NAMES = {"Na+", "K+", "Cl-", "NA", "CL", "K"}  # NA/CL appear in some pdbs too


# ────────────────────────── small helpers (working-dir aware) ──────────────────────────

def _is_atom_line(line: str) -> bool:
    tag = line[0:6].strip()
    return tag == "ATOM" or tag == "HETATM"

def _field(line: str, start: int, end: int) -> str:
    return line[start:end].strip()

def _read_nonblank_lines(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]

def _collect_backbone_heavy_and_lig(vac_pdb: Path, lig_res: str, offset: int = 0) -> List[str]:
    """Return mask-style atom numbers (as strings) for backbone heavy atoms before ligand and for ligand (in
    the bulk heavy atoms."""
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
                    hvy.append(_field(line, 6, 11))  # atom serial as string (cpptraj masks use this later)
            elif resi == int(lig_res) + offset:
                name = _field(line, 12, 16)
                if name[0] != "H":  # non-H ligand heavy atoms
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
            for line in fin:
                bits = line.split()
                if len(bits) > 6 and bits[5] in ("cg", "c1"):
                    sp_atoms.add(bits[1])  # mol2 atom name
    out = []
    for m in msk:
        _, b, c, _ = m.split()
        if b.split("@")[1] in sp_atoms or c.split("@")[1] in sp_atoms:
            continue
        out.append(m)
    return out

def _write_assign_and_read_vals(work: Path, rst_exprs: List[str], prmtop: Path, traj: Path) -> List[float]:
    """
    Emit assign.in and parse assign.dat into reference values `vals`, same order as `rst_exprs`.
    """
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


# ───────────────────────────── write_equil_restraints (drop-in) ─────────────────────────────

def write_equil_restraints(ctx: BuildContext) -> None:
    """
    Generate, in ctx.working_dir:
      - assign.in / assign.dat (reference via cpptraj)
      - disangXX.rest (staged release weights from ctx.sim.release_eq)
      - disang.rest  (copy of last stage)
      - cv.in        (COM restraint)
    Uses anchors saved previously to anchors.json.
    """
    work = ctx.working_dir
    build_dir = ctx.build_dir
    lig = ctx.ligand
    mol = ctx.residue_name
    comp = ctx.comp
    stage = getattr(ctx, "stage", "prepare_equil")

    vac_pdb         = work / "vac.pdb"
    vac_lig_pdb     = work / f"{lig}.pdb"
    vac_lig_prmtop  = work / "vac_ligand.prmtop"
    full_hmr_prmtop = work / "full.hmr.prmtop"
    full_inpcrd     = work / "full.inpcrd"
    lig_mol2        = work / f"{mol}.mol2"
    anchors_pdb     = build_dir / f"equil-{mol}.pdb"

    if not anchors_pdb.exists():
        raise FileNotFoundError(f"Anchor header not found: {anchors_pdb}")
    for p in (vac_pdb, vac_lig_pdb, vac_lig_prmtop, full_hmr_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing for restraints: {p}")

    # anchors.json already written during build; load it
    anchors = load_anchors(work)
    P1, P2, P3, L1, L2, L3, lig_res = (
        anchors.P1, anchors.P2, anchors.P3,
        anchors.L1, anchors.L2, anchors.L3,
        anchors.lig_res,
    )

    # 1) protein backbone heavy atoms for COM pool
    hvy_h, _ = _collect_backbone_heavy_and_lig(vac_pdb, lig_res)

    # 2) atom-number ↔ mask mapping (cpptraj style)
    atm_num         = num_to_mask(vac_pdb.as_posix())
    ligand_atm_num  = num_to_mask(vac_lig_pdb.as_posix())

    # 3) base restraint expressions (protein triangle + TR chain)
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

    # 4) ligand dihedrals (non-H), replace :1 with :<lig_res>, filter sp carbons
    msk = _scan_dihedrals_from_prmtop(vac_lig_prmtop, ligand_atm_num)
    msk = [m.replace(":1", f":{lig_res}") for m in msk]
    if lig_mol2.exists():
        msk = _filter_sp_carbons(msk, lig_mol2)

    full_rst = rst + msk

    # 5) reference values (cpptraj)
    vals = _write_assign_and_read_vals(work, full_rst, full_hmr_prmtop, full_inpcrd)

    # 6) staged disangXX.rest and cv.in
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
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 0.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (rest[5], rest[5]))
        cvf.write("/\n")

    for idx, weight in enumerate(release_eq):
        rdsf = rest[1]
        ldf  = weight * rest[2] / 100.0
        laf  = weight * rest[3] / 100.0
        ldhf = weight * rest[4] / 100.0

        outp = work / f"disang{idx:02d}.rest"
        with outp.open("w") as df:
            df.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3}  stage={stage}  weight={weight}\n")
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

                # ligand dihedrals
                if n == 4:
                    iat = (
                        f"{atm_num.index(fields[0])},"
                        f"{atm_num.index(fields[1])},"
                        f"{atm_num.index(fields[2])},"
                        f"{atm_num.index(fields[3])},"
                    )
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, ldhf, ldhf))

    # copy the last stage as disang.rest
    (work / "disang.rest").write_text((work / f"disang{len(release_eq)-1:02d}.rest").read_text())
    logger.debug(f"[equil] restraints written in {work}")


# ───────────────────────────── FE component restraint writers ─────────────────────────────

def _write_component_restraints(ctx: BuildContext, *, skip_lig_tr: bool = False, lig_only: bool = False) -> None:
    """
    Core FE writer: mirrors write_equil_restraints structure, but produces a single disang.rest.
    Everything is rooted at ctx.working_dir.
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

    # weights (like equil but single stage at 100%)
    rest = ctx.sim.rest  # [rdhf, rdsf, ldf, laf, ldhf, rcom, lcom]
    rdhf, rdsf, ldf_base, laf_base, ldhf_base, rcom, lcom = rest
    ldf, laf, ldhf = ldf_base, laf_base, ldhf_base

    # cv.in
    cv_in = windows_dir / "cv.in"
    with cv_in.open("w") as cvf:
        # protein backbone heavy atoms for COM restraint
        cvf.write("cv_file\n&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        cvf.write(f" cv_ni = {len(hvy_h)+2}, cv_i = 1,0,")
        for a in hvy_h: cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 0.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (rcom, rcom))
        cvf.write("/\n")

        # ligand COM restraint
        cvf.write("&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        cvf.write(f" cv_ni = {len(hvy_lig)+2}, cv_i = 2,0,")
        for a in hvy_lig: cvf.write(a + ",")
        cvf.write("\n")
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 0.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (lcom, lcom))
        cvf.write("/\n")


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
            if n == 4:
                iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},{atm_num.index(fields[2])},{atm_num.index(fields[3])},"
                df.write(f"&rst iat={iat:<23s} ")
                df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Lig_D\n"
                         % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0, ldhf, ldhf))

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

    logger.debug(f"[restraints:{comp}] wrote cv.in, disang.rest, restraints.in in {windows_dir}")


# ───────────────────────────── registrations ─────────────────────────────

@register_restraints("v", "o", "z")
def _build_restraints_v_o_z(builder, ctx: BuildContext) -> None:
    # full protein + TR + ligand dihedrals
    _write_component_restraints(ctx, skip_lig_tr=False, lig_only=False)

@register_restraints("y")
def _build_restraints_y(builder, ctx: BuildContext) -> None:
    # ligand-only (no protein TR terms)
    _write_component_restraints(ctx, skip_lig_tr=True, lig_only=True)

@register_restraints("x")
def _build_restraints_x(builder, ctx: BuildContext) -> None:
    # same base as v/o/z; if you need extra reference-ligand blocks, append after this call.
    _write_component_restraints(ctx, skip_lig_tr=False, lig_only=False)
    # TODO: append x-specific reference-ligand restraints here (also rooted at ctx.working_dir).