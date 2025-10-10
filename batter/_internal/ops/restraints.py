from __future__ import annotations

from pathlib import Path
from typing import List

import MDAnalysis as mda
from loguru import logger

# reuse your existing helpers
from batter.utils import run_with_log
from batter._internal.ops.helpers import num_to_mask
from batter._internal.ops.simprep import load_anchors
from batter._internal.builders.interfaces import BuildContext

from batter.utils import cpptraj

def _read_anchor_header(equil_pdb: Path):
    """
    Parse the 1st line we write in equil-<lig>.pdb:
    REMARK A P1 P2 P3 L1 L2 L3 first_res recep_last
    Returns (P1,P2,P3,L1,L2,L3,lig_res)
    """
    with equil_pdb.open("r") as f:
        header = f.readline().split()
    if not header or header[0] != "REMARK":
        raise ValueError(f"Anchor header missing in {equil_pdb}")
    P1, P2, P3, L1, L2, L3 = header[2:8]
    lig_res = L1.split("@")[0][1:]  # strip leading ':'
    return P1, P2, P3, L1, L2, L3, lig_res


def _collect_backbone_heavy_before_lig(vac_pdb: Path, lig_res: str) -> List[str]:
    """Return mask-style atom numbers (as strings) for backbone heavy atoms before ligand."""
    hvy = []
    with vac_pdb.open() as f:
        for line in f:
            rec = line[0:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            resi = int(line[22:26].strip())
            if 2 <= resi < int(lig_res):
                name = line[12:16].strip()
                if name in ("CA", "N", "C", "O"):
                    hvy.append(line[6:11].strip())  # atom serial as string (cpptraj masks use this later)
    return hvy


def write_equil_restraints(ctx: BuildContext) -> None:
    """
    Generate:
      - assign.in / assign.dat (reference values via cpptraj)
      - disang*.rest (distance/angle/dihedral restraints, staged by release weights)
      - cv.in (COM restraint)
    All files are written into ctx.window_dir.
    """
    work = ctx.working_dir
    build_dir = ctx.working_dir / f"{ctx.comp}_build_files"
    lig = ctx.ligand
    comp = ctx.comp
    stage = ctx.stage if hasattr(ctx, "stage") else "prepare_equil"

    vac_pdb = work / "vac.pdb"
    vac_lig_pdb = work / "vac_ligand.pdb"
    vac_lig_prmtop = work / "vac_ligand.prmtop"
    full_hmr_prmtop = work / "full.hmr.prmtop"
    full_inpcrd = work / "full.inpcrd"
    lig_mol2 = work / f"{ctx.residue_name}.mol2"  # your box step should have written these in window_dir
    anchors_pdb = build_dir / f"equil-{lig.lower()}.pdb"

    if not anchors_pdb.exists():
        raise FileNotFoundError(f"Anchor header not found: {anchors_pdb}")
    for p in (vac_pdb, vac_lig_pdb, vac_lig_prmtop, full_hmr_prmtop, full_inpcrd):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing for restraints: {p}")

    anchors = load_anchors(ctx.working_dir)
    P1, P2, P3, L1, L2, L3, lig_res = (
        anchors.P1, anchors.P2, anchors.P3,
        anchors.L1, anchors.L2, anchors.L3,
        anchors.lig_res,
    )

    # 1) protein backbone heavy atoms for COM restraint pool
    hvy_h = _collect_backbone_heavy_before_lig(vac_pdb, lig_res)

    # 2) atom-number ↔ mask mapping (cpptraj style) for full and ligand
    pdb_file = vac_pdb.as_posix()
    ligand_pdb_file = vac_lig_pdb.as_posix()
    atm_num = num_to_mask(pdb_file)          # list[str]; index(atom mask) → serial
    ligand_atm_num = num_to_mask(ligand_pdb_file)

    # 3) define base restraint masks (protein shape + ligand TR/anchor)
    rst: List[str] = []
    # protein triangle
    rst.append(f"{P1} {P2}")
    rst.append(f"{P2} {P3}")
    rst.append(f"{P3} {P1}")
    # ligand TR/anchor chain
    rst.append(f"{P1} {L1}")
    rst.append(f"{P2} {P1} {L1}")
    rst.append(f"{P3} {P2} {P1} {L1}")
    rst.append(f"{P1} {L1} {L2}")
    rst.append(f"{P2} {P1} {L1} {L2}")
    rst.append(f"{P1} {L1} {L2} {L3}")

    # 4) add ligand dihedrals (non-H) from prmtop
    mlines: List[str] = []
    spool = 0
    with vac_lig_prmtop.open() as fin:
        for line in fin:
            if "FLAG DIHEDRALS_WITHOUT_HYDROGEN" in line:
                spool = 1
                continue
            if "FLAG EXCLUDED_ATOMS_LIST" in line:
                spool = 0
            if spool and len(line.split()) > 3:
                mlines.append(line.rstrip())

    msk: List[str] = []
    for row in mlines:
        dat = row.split()
        # first term
        if int(dat[3]) > 0:
            idx = [abs(int(x) // 3) + 1 for x in dat[:4]]
            msk.append(
                f"{ligand_atm_num[idx[0]]} {ligand_atm_num[idx[1]]} {ligand_atm_num[idx[2]]} {ligand_atm_num[idx[3]]}"
            )
        # second term (if present)
        if len(dat) > 7 and int(dat[8]) > 0:
            idx = [abs(int(x) // 3) + 1 for x in dat[5:9]]
            msk.append(
                f"{ligand_atm_num[idx[0]]} {ligand_atm_num[idx[1]]} {ligand_atm_num[idx[2]]} {ligand_atm_num[idx[3]]}"
            )

    # de-duplicate collinear centers (keep unique by the inner pair)
    seen_pairs = set()
    uniq = []
    for m in msk:
        a, b, c, d = m.split()
        pair = tuple(sorted((b, c)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        uniq.append(m)
    msk = uniq

    # replace ligand residue index to actual lig_res in masks (cpptraj :1 → :<lig_res>)
    msk = [m.replace(":1", f":{lig_res}") for m in msk]

    # 5) drop sp carbons from mol2 to avoid unstable dihedrals
    sp_atoms = set()
    if lig_mol2.exists():
        with lig_mol2.open() as fin:
            for line in fin:
                bits = line.split()
                if len(bits) > 6 and bits[5] in ("cg", "c1"):
                    sp_atoms.add(bits[1])  # mol2 atom name
    filt = []
    for m in msk:
        _, b, c, _ = m.split()
        bname = b.split("@")[1]
        cname = c.split("@")[1]
        if bname in sp_atoms or cname in sp_atoms:
            continue
        filt.append(m)
    rst.extend(filt)

    # 6) assign reference values via cpptraj
    assign_in = work / "assign.in"
    with assign_in.open("w") as f:
        f.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3}\n")
        f.write(f"parm {full_hmr_prmtop}\n")
        f.write(f"trajin {full_inpcrd}\n")
        for i, expr in enumerate(rst):
            parts = expr.split()
            if len(parts) == 2:
                f.write(f"distance r{i} {expr} noimage out assign.dat\n")
            elif len(parts) == 3:
                f.write(f"angle    r{i} {expr} out assign.dat\n")
            elif len(parts) == 4:
                f.write(f"dihedral r{i} {expr} out assign.dat\n")
    run_with_log(f"{cpptraj} -i {assign_in.name} > assign.log", working_dir=work)

    # parse assign.dat (first data row → values)
    assign_dat = (work / "assign.dat").read_text().splitlines()
    if len(assign_dat) < 2:
        raise RuntimeError("assign.dat did not contain reference values")
    vals = assign_dat[1].split()
    # legacy rotation: shift first to end, drop last
    vals.append(vals.pop(0))
    vals = vals[:-1]

    # 7) write staged disang files + cv.in
    rest = ctx.sim.rest      # e.g., [dummy, rdsf, ldf_base, laf_base, ldhf_base, rcom]
    release_eq = ctx.sim.release_eq

    # COM restraint pool: backbone heavies + two anchors (matches legacy)
    cv_in = work / "cv.in"
    with cv_in.open("w") as cvf:
        cvf.write("cv_file\n&colvar\n")
        cvf.write(" cv_type = 'COM_DISTANCE'\n")
        cvf.write(f" cv_ni = {len(hvy_h)+2}, cv_i = 1,0,")
        for a in hvy_h:
            cvf.write(a + ",")
        cvf.write("\n")
        # anchor_position and anchor_strength get set per-run by mdin in your flow; keep legacy defaults
        cvf.write(" anchor_position = %10.4f, %10.4f, %10.4f, %10.4f\n" % (0.0, 0.0, 0.0, 999.0))
        cvf.write(" anchor_strength = %10.4f, %10.4f,\n" % (rest[5], rest[5]))
        cvf.write("/\n")

    # produce disangXX.rest for each release weight
    for idx, weight in enumerate(release_eq):
        rdsf = rest[1]
        ldf = weight * rest[2] / 100.0
        laf = weight * rest[3] / 100.0
        ldhf = weight * rest[4] / 100.0

        outp = work / f"disang{idx:02d}.rest"
        with outp.open("w") as df:
            df.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3}  stage={stage}  weight={weight}\n")
            for i, expr in enumerate(rst):
                fields = expr.split()
                n = len(fields)

                # protein triangle (i < 3): distance only
                if i < 3 and n == 2:
                    iat = f"{atm_num.index(fields[0])},{atm_num.index(fields[1])},"
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end #Rec_C\n"
                             % (0.0, float(vals[i]), float(vals[i]), 999.0, rdsf, rdsf))
                    continue

                # skip ligand TR restraints for component 'a' (matches legacy behavior)
                if i >= 3 and i < 9 and comp == "a":
                    continue

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
                    # TR or dihedral: both use ±180 windows
                    df.write(f"&rst iat={iat:<23s} ")
                    df.write("r1=%10.4f, r2=%10.4f, r3=%10.4f, r4=%10.4f, rk2=%11.7f, rk3=%11.7f, &end %s\n"
                             % (float(vals[i]) - 180.0, float(vals[i]), float(vals[i]), float(vals[i]) + 180.0,
                                (laf if i < 9 else ldhf), (laf if i < 9 else ldhf),
                                "#Lig_TR" if i < 9 else "#Lig_D"))
    # copy lask one as disang.rest
    outp = work / "disang.rest"
    outp.write_text((work / f"disang{idx:02d}.rest").read_text())
    logger.debug(f'Finished writing restraints in {work}')