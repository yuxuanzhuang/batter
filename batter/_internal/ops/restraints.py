from __future__ import annotations

from pathlib import Path
from typing import List

import MDAnalysis as mda
from loguru import logger

# reuse your existing helpers
from batter.utils import run_with_log
from batter._internal.ops.helpers import num_to_mask
from batter._internal.builders.interfaces import BuildContext
from batter._internal.ops.helpers import Anchors, save_anchors, load_anchors
from batter._internal.builders.fe_registry import register_restraints

from batter.utils import cpptraj

def _read_anchor_line(pdb_path: Path, working_dir: Path) -> Anchors:
    with open(pdb_path, "r") as f:
        data = f.readline().split()
    P1, P2, P3 = data[2].strip(), data[3].strip(), data[4].strip()
    L1, L2, L3 = data[5].strip(), data[6].strip(), data[7].strip()
    lig_res = L1.split('@')[0][1:]

    anchors = Anchors(P1=P1, P2=P2, P3=P3, L1=L1, L2=L2, L3=L3, lig_res=lig_res)
    save_anchors(working_dir, anchors)  # persist JSON snapshot for reproducibility
    return anchors

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


def _precompute_common(builder) -> dict:
    comp = builder.comp
    stage = builder.stage
    mol   = builder.mol
    sim   = builder.sim_config

    pdb_file          = Path("vac.pdb")
    ligand_pdb_file   = Path("vac_ligand.pdb")
    reflig_pdb_file   = Path("vac_reference.pdb")
    stage_pdb         = Path(f"{stage}-{mol.lower()}.pdb")
    if not stage_pdb.exists():
        raise FileNotFoundError(f"Missing {stage_pdb} (needed to read anchors).")

    anchors = _read_anchor_line(stage_pdb, builder.build_dir)

    # weight / lambdas
    lambdas = builder.component_windows.get(comp, [])
    weight  = lambdas[builder.win if builder.win != -1 else 0] if lambdas else 1.0

    # index maps
    atm_num         = scripts_ops.num_to_mask(str(pdb_file))
    ligand_atm_num  = scripts_ops.num_to_mask(str(ligand_pdb_file))
    vac_atoms       = mda.Universe(str(ligand_pdb_file)).atoms.n_atoms

    # ref atoms for exchange
    ref_atoms = None
    if comp == 'x' and reflig_pdb_file.exists():
        ref_atoms = mda.Universe(str(reflig_pdb_file)).atoms.n_atoms

    # convenience bag
    return dict(
        comp=comp, stage=stage, mol=mol, sim=sim,
        pdb_file=pdb_file, ligand_pdb_file=ligand_pdb_file, reflig_pdb_file=reflig_pdb_file,
        anchors=anchors,
        lambdas=lambdas, weight=weight,
        atm_num=atm_num, ligand_atm_num=ligand_atm_num, vac_atoms=vac_atoms, ref_atoms=ref_atoms
    )

# ------------- public entrypoint -------------

def build_restraints(builder) -> None:
    ctx = _precompute_common(builder)
    comp = ctx["comp"]
    # Respect your original “skip on per-window lambdas” gate if needed
    # (Keep this guard in your builder before calling build_restraints).
    fn = _RESTRAINT_BUILDERS.get(comp, _build_restraints_default)
    return fn(builder, ctx)

# ---------------- utilities copied from your monolith ----------------

def _scan_dihedrals_from_prmtop(prmtop_path: Path, ligand_atm_num: List[str]) -> List[str]:
    """Build msk list (ligand dihedral masks) from vac_ligand.prmtop, like in your original code."""
    mlines: List[str] = []
    msk: List[str] = []
    spool = 0
    with open(prmtop_path) as fin:
        lines = [ln.rstrip() for ln in fin if ln.strip()]
    for line in lines:
        if 'FLAG DIHEDRALS_WITHOUT_HYDROGEN' in line:
            spool = 1
        elif 'FLAG EXCLUDED_ATOMS_LIST' in line:
            spool = 0
        if spool and (len(line.split()) > 3):
            mlines.append(line)

    # primary set
    for ln in mlines:
        data = ln.split()
        if int(data[3]) > 0:
            anum = [abs(int(x)//3)+1 for x in data]
            msk.append('%s %s %s %s' % (
                ligand_atm_num[anum[0]], ligand_atm_num[anum[1]], ligand_atm_num[anum[2]], ligand_atm_num[anum[3]]
            ))
    # secondary set (len > 7)
    for ln in mlines:
        data = ln.split()
        if len(data) > 7 and int(data[8]) > 0:
            anum = [abs(int(x)//3)+1 for x in data]
            msk.append('%s %s %s %s' % (
                ligand_atm_num[anum[5]], ligand_atm_num[anum[6]], ligand_atm_num[anum[7]], ligand_atm_num[anum[8]]
            ))

    # prune duplicates on same central bond (like your mat/excl dance)
    excl = msk[:]
    mat: List[int] = []
    for i in range(len(excl)):
        a1, b1, c1, d1 = excl[i].split()
        for j in range(i):
            a2, b2, c2, d2 = excl[j].split()
            if (b1 == b2 and c1 == c2) or (b1 == c2 and c1 == b2):
                if j not in mat:
                    mat.append(j)
    for j in mat:
        msk[j] = ''
    msk = [m for m in msk if m]
    return msk

def _filter_sp_carbons(msk: List[str], mol2_path: Path) -> List[str]:
    """Drop dihedrals that include cg/c1 carbons, as in your code."""
    sp_carb = []
    with open(mol2_path) as fin:
        lines = [ln.rstrip() for ln in fin if ln.strip()]
    for ln in lines:
        data = ln.split()
        if len(data) > 6 and (data[5] == 'cg' or data[5] == 'c1'):
            sp_carb.append(data[1])

    out = []
    for m in msk:
        a, b, c, d = m.split()
        b_name = b.split('@')[1]
        c_name = c.split('@')[1]
        if (b_name in sp_carb) or (c_name in sp_carb):
            continue
        out.append(m)
    return out

def _write_assign_and_read_vals(anchor_rst: List[str], atm_num: List[str], cpptraj_in: str, prmtop: str, traj: str) -> List[float]:
    """
    Emit assign.in (or assign2.in) and parse assign.dat into reference values `vals`,
    same order as anchor_rst.
    """
    with open(cpptraj_in, "w") as fh:
        fh.write("parm %s\n" % prmtop)
        fh.write("trajin %s\n" % traj)
        # r0.. for each rst entry
        for i, expr in enumerate(anchor_rst):
            arr = expr.split()
            if len(arr) == 2:
                fh.write(f"distance r{i} {expr} noimage out assign.dat\n")
            elif len(arr) == 3:
                fh.write(f"angle r{i} {expr} out assign.dat\n")
            elif len(arr) == 4:
                fh.write(f"dihedral r{i} {expr} out assign.dat\n")
    helpers_ops.run_with_log(f"{helpers_ops.cpptraj} -i {cpptraj_in} > {cpptraj_in.replace('.in','.log')}")
    with open("assign.dat") as fin:
        lines = [ln.rstrip() for ln in fin if ln.strip()]
    vals = lines[1].split()
    # reorder like original (pop first -> append; drop last)
    vals.append(vals.pop(0))
    vals = vals[:-1]
    return [float(v) for v in vals]

# ---------------- default (v/o/z share) ----------------

def _build_restraints_default(builder, ctx: dict) -> None:
    """
    Implements your common branch (used by v/o/z/y default path when appropriate).
    - builds rst list (protein triad, TR, dihedrals)
    - creates assign.in to get reference values
    - writes disang.rest and restraints.in
    """
    comp   = ctx["comp"]
    stage  = ctx["stage"]
    mol    = ctx["mol"]
    sim    = ctx["sim"]
    anchors: AnchorInfo = ctx["anchors"]
    atm_num        = ctx["atm_num"]
    ligand_atm_num = ctx["ligand_atm_num"]
    vac_atoms      = ctx["vac_atoms"]
    weight         = ctx["weight"]

    # Rest weights (your original "rest" vector mapping to rdhf/rdsf/ldf/laf/ldhf/rcom/lcom)
    rest = sim.rest
    if comp in {'v','e','w','f','x','o','z'}:
        rdhf, rdsf, ldf, laf, ldhf, rcom, lcom = rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6]
    else:
        # sensible fallback (unused for 'y', which overrides below)
        rdhf, rdsf, ldf, laf, ldhf, rcom, lcom = rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6]

    # Build anchor expressions
    P1, P2, P3 = anchors.P1, anchors.P2, anchors.P3
    L1, L2, L3 = anchors.L1, anchors.L2, anchors.L3

    rst: List[str] = []
    # protein distances
    rst.append(f"{P1} {P2}")
    rst.append(f"{P2} {P3}")
    rst.append(f"{P3} {P1}")
    # ligand TR anchors
    rst.append(f"{P1} {L1}")
    rst.append(f"{P2} {P1} {L1}")
    rst.append(f"{P3} {P2} {P1} {L1}")
    rst.append(f"{P1} {L1} {L2}")
    rst.append(f"{P2} {P1} {L1} {L2}")
    rst.append(f"{P1} {L1} {L2} {L3}")

    # ligand dihedrals from prmtop
    msk = _scan_dihedrals_from_prmtop(Path("vac_ligand.prmtop"), ligand_atm_num)
    # remove sp carbons
    msk = _filter_sp_carbons(msk, Path(f"{mol.lower()}.mol2"))

    # combine full list (protein+TR+lig dihed)
    full_rst = rst + msk

    # get reference values via cpptraj
    # (your original writes 'assign.in', parm full.hmr.prmtop, traj full.inpcrd)
    prmtop_ref = "full.hmr.prmtop"
    traj_ref   = "full.inpcrd"
    vals = _write_assign_and_read_vals_for(builder, full_rst, prmtop_ref, traj_ref)

    # Write disang.rest (full system)
    with open("disang.rest", "w") as fh:
        fh.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3} stage={stage} weight={weight}\n")
        for i, expr in enumerate(full_rst):
            arr = expr.split()
            if len(arr) == 2:
                nums = f"{atm_num.index(arr[0])},{atm_num.index(arr[1])},"
                fh.write(f"&rst iat={nums} r1=0.0, r2={vals[i]:10.4f}, r3={vals[i]:10.4f}, r4=999.0, rk2={rdsf:11.7f}, rk3={rdsf:11.7f}, &end #Rec_C\n")
            elif len(arr) == 3:
                nums = f"{atm_num.index(arr[0])},{atm_num.index(arr[1])},{atm_num.index(arr[2])},"
                fh.write(f"&rst iat={nums} r1=0.0, r2={vals[i]:10.4f}, r3={vals[i]:10.4f}, r4=180.0, rk2={laf:11.7f}, rk3={laf:11.7f}, &end #Lig_TR\n")
            elif len(arr) == 4:
                nums = f"{atm_num.index(arr[0])},{atm_num.index(arr[1])},{atm_num.index(arr[2])},{atm_num.index(arr[3])},"
                fh.write(f"&rst iat={nums} r1={vals[i]-180:10.4f}, r2={vals[i]:10.4f}, r3={vals[i]:10.4f}, r4={vals[i]+180:10.4f}, rk2={ldhf:11.7f}, rk3={ldhf:11.7f}, &end #Lig_D\n")
        fh.write("\n")

    # Analysis driver (restraints.in) — mirrors your monolith
    with open("restraints.in", "w") as fh:
        fh.write(f"# Anchor atoms {P1} {P2} {P3} {L1} {L2} {L3} stage={stage}\n")
        fh.write("noexitonerror\n")
        fh.write("parm vac.prmtop\n")
        for i in range(2, 11):
            fh.write(f"trajin md{i:02d}.nc\n")
        # dump TR and dihedrals — skip first 3 (protein-only) if you want exactly your original split
        for i, expr in enumerate(full_rst[3:], start=3):
            arr = expr.split()
            tag = "distance" if len(arr)==2 else ("angle" if len(arr)==3 else "dihedral")
            fh.write(f"{tag} a{i} {expr} out restraints.dat\n")

def _write_assign_and_read_vals_for(builder, rst_list: List[str], prmtop_ref: str, traj_ref: str) -> List[float]:
    # Use the helper that writes cpptraj input + parses assign.dat, but keep filename parity
    with open("assign.in", "w") as fh:
        fh.write(f"parm {prmtop_ref}\n")
        fh.write(f"trajin {traj_ref}\n")
        for i, expr in enumerate(rst_list):
            arr = expr.split()
            if len(arr) == 2:
                fh.write(f"distance r{i} {expr} noimage out assign.dat\n")
            elif len(arr) == 3:
                fh.write(f"angle r{i} {expr} out assign.dat\n")
            elif len(arr) == 4:
                fh.write(f"dihedral r{i} {expr} out assign.dat\n")
    helpers_ops.run_with_log(f"{helpers_ops.cpptraj} -i assign.in > assign.log")
    with open("assign.dat") as fin:
        lines = [ln.rstrip() for ln in fin if ln.strip()]
    vals = lines[1].split()
    vals.append(vals.pop(0))
    vals = vals[:-1]
    return [float(v) for v in vals]

# ---------------- v/o/z share this default ----------------

@register_restraints('v', 'o', 'z')
def _build_restraints_v_o_z(builder, ctx: dict) -> None:
    return _build_restraints_default(builder, ctx)

# ---------------- y (ligand-only) ----------------

@register_restraints('y')
def _build_restraints_y(builder, ctx: dict) -> None:
    """
    For component 'y' (ligand-only decouple/attach) you said the complex isn’t needed
    and you typically only apply ligand conformational restraints.
    This writes a minimal disang.rest for ligand dihedrals only.
    """
    stage  = ctx["stage"]
    mol    = ctx["mol"]
    ligand_atm_num = ctx["ligand_atm_num"]
    weight = ctx["weight"]
    rest   = ctx["sim"].rest
    ldhf   = rest[4]  # ligand dihedral force from your rest vector

    # Build dihedral masks from vac_ligand.prmtop
    msk = _scan_dihedrals_from_prmtop(Path("vac_ligand.prmtop"), ligand_atm_num)
    msk = _filter_sp_carbons(msk, Path(f"{mol.lower()}.mol2"))

    # Derive “vals” for dihedrals only (using full.hmr.prmtop / full.inpcrd as in your flow)
    vals = _write_assign_and_read_vals_for(builder, msk, "full.hmr.prmtop", "full.inpcrd")

    with open("disang.rest", "w") as fh:
        fh.write(f"# Ligand-only stage={stage} weight={weight}\n")
        for i, expr in enumerate(msk):
            a,b,c,d = expr.split()
            nums = f"{ligand_atm_num.index(a)},{ligand_atm_num.index(b)},{ligand_atm_num.index(c)},{ligand_atm_num.index(d)},"
            fh.write(f"&rst iat={nums} r1={vals[i]-180:10.4f}, r2={vals[i]:10.4f}, r3={vals[i]:10.4f}, r4={vals[i]+180:10.4f}, rk2={ldhf:11.7f}, rk3={ldhf:11.7f}, &end #Lig_D\n")
        fh.write("\n")

    with open("restraints.in", "w") as fh:
        fh.write(f"# Ligand-only stage={stage}\nnoexitonerror\nparm vac.prmtop\n")
        for i in range(2, 11):
            fh.write(f"trajin md{i:02d}.nc\n")
        for i, expr in enumerate(msk):
            fh.write(f"dihedral d{i} {expr} out restraints.dat\n")

# ---------------- x (exchange) = default + reference ligand section ----------------

@register_restraints('x')
def _build_restraints_x(builder, ctx: dict) -> None:
    # write the common part first
    _build_restraints_default(builder, ctx)

    # append the reference-ligand section (your "comp == 'x'" extra)
    anchors = ctx["anchors"]
    atm_num = ctx["atm_num"]
    rest    = ctx["sim"].rest
    ldf, laf, ldhf = rest[2], rest[3], rest[4]
    ref_atoms = ctx["ref_atoms"]
    if ref_atoms is None:
        logger.warning("[restraints:x] no vac_reference.pdb present; skipping x-specific reference block.")
        return

    # Re-use vals from a second assign (your assign2.in/assign2.dat workflow)
    # We reconstruct the same anchor order (TR + dihedrals) for the reference ligand
    # and then append to disang.rest
    # For brevity, we’ll just mirror the dihedral part using `ref_atoms` offset.

    # Build masks again from vac_reference.prmtop
    ligand_atm_num_ref = scripts_ops.num_to_mask(str(ctx["reflig_pdb_file"]))
    msk_ref = _scan_dihedrals_from_prmtop(Path("vac_reference.prmtop"), ligand_atm_num_ref)
    msk_ref = _filter_sp_carbons(msk_ref, Path(f"{builder.molr.lower()}.mol2"))

    vals_ref = _write_assign_and_read_vals_for(builder, msk_ref, "full-ref.hmr.prmtop", "rec_file.pdb")

    with open("disang.rest", "a") as fh:
        for i, expr in enumerate(msk_ref):
            a,b,c,d = expr.split()
            # apply +ref_atoms offset like your monolith
            nums = f"{atm_num.index(a)+ref_atoms},{atm_num.index(b)+ref_atoms},{atm_num.index(c)+ref_atoms},{atm_num.index(d)+ref_atoms},"
            fh.write(f"&rst iat={nums} r1={vals_ref[i]-180:10.4f}, r2={vals_ref[i]:10.4f}, r3={vals_ref[i]:10.4f}, r4={vals_ref[i]+180:10.4f}, rk2={ldhf:11.7f}, rk3={ldhf:11.7f}, &end #Lig_D_ref\n")
        fh.write("\n")