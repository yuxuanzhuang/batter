from __future__ import annotations

from pathlib import Path
import shutil
import glob
import os
import json

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger

from batter._internal.builders.fe_registry import register_build_complex, register_sim_files
from batter._internal.ops.helpers import get_sdr_dist, get_ligand_candidates, Anchors, save_anchors
from batter.utils import (
    run_with_log,
    cpptraj,
    vmd
)

from batter._internal.templates import BUILD_FILES_DIR as build_files_orig  # type: ignore


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


@register_build_complex("z")
def build_complex_z(b) -> bool:
    """
    Z-component _build_complex:
    Copy/transform files from the per-ligand equil output, then detect/emit anchors.
    Returns True on success, False to indicate pruning.
    """
    # --- config / context ---
    ligand = b.ctx.ligand
    sim = b.ctx.sim

    solv_shell   = sim.solv_shell
    l1_x, l1_y, l1_z = sim.l1_x, sim.l1_y, sim.l1_z
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol
    l1_range    = sim.l1_range
    max_adis    = sim.max_adis
    min_adis    = sim.min_adis
    buffer_z    = sim.buffer_z
    hmr         = sim.hmr

    workdir   = b.build_dir; workdir.mkdir(parents=True, exist_ok=True)
    child_root = b.ctx.working_dir                         # .../simulations/<LIG>/fe/...
    sys_root   = b.ctx.system_root                         # .../work/<system>
    equil_dir  = sys_root / "simulations" / ligand / "equil"  # .../work/<system>/simulations/<LIG>/equil
    ff_dir = sys_root/ "simulations" / ligand / "params"  # .../work/<system>/simulations/<LIG>/params

    shutil.copytree(build_files_orig, workdir, dirs_exist_ok=True)

    # --- helpers to keep paths explicit ---
    def _p(name: str) -> Path: return workdir / name
    def _copy(src: Path, dst_name: str):
        if src.exists():
            shutil.copy2(src, _p(dst_name))
        else:
            raise FileNotFoundError(f"Missing required file: {src}")

    # 1) copy artifacts from equil
    _copy(equil_dir / "q_build_files" / f"{ligand}.pdb",          f"{ligand}.pdb")
    _copy(equil_dir / "representative.rst7",                    "representative.rst7")
    _copy(equil_dir / "representative.pdb",                     "aligned-nc.pdb")
    _copy(equil_dir / "build_amber_renum.txt",                  "build_amber_renum.txt")
    _copy(equil_dir / "q_build_files" / "protein_renum.txt",      "protein_renum.txt")

    for p in equil_dir.glob("full*.prmtop"): shutil.copy2(p, _p(p.name))
    for p in equil_dir.glob("vac*"):          shutil.copy2(p, _p(p.name))

    # 2) deduce ligand resname from copied ligand PDB
    mol = b.ctx.residue_name
    
    # Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf", ".pdb"):
        src = ff_dir / f"{mol}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_y] Missing ligand FF file: {src}")
        shutil.copy2(src, workdir / src.name)

    prmtop_f = "full.prmtop" if str(hmr).lower() == "no" else "full.hmr.prmtop"

    # 3) extract receptor-only PDB from representative.rst7
    run_with_log(f"{cpptraj} -p {prmtop_f} -y representative.rst7 -x rec_file.pdb",
                 working_dir=workdir)

    # 4) reapply chain IDs from renum map; optional lipid resid compaction
    renum = pd.read_csv(_p("build_amber_renum.txt"), sep=r"\s+", header=None,
                        names=['old_resname','old_chain','old_resid','new_resname','new_resid'])
    u = mda.Universe(str(_p("rec_file.pdb")))
    for residue in u.select_atoms('protein').residues:
        resid_str = residue.resid
        chain = renum.query('old_resid == @resid_str').old_chain.values
        if chain.size: residue.atoms.chainIDs = chain[0]

    if b.membrane_builder:
        # also skip ANC, which is a anchored dummy atom for rmsf restraint
        non_water_ag = u.select_atoms('not resname WAT Na+ Cl- K+ ANC')
        # fix lipid resids
        revised_resids = []
        resid_counter = 1
        prev_resid = 0
        for i, row in renum.iterrows():
            # skip water and ions as they will not be present later
            if row['old_resname'] in ['WAT', 'Na+', 'Cl-', 'K+']:
                continue
            if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                revised_resids.append(resid_counter)
                resid_counter += 1
            else:
                revised_resids.append(resid_counter - 1)
            prev_resid = row['old_resid']
        
        revised_resids = np.array(revised_resids)
        total_residues = non_water_ag.residues.n_residues
        final_resids = np.zeros(total_residues, dtype=int)
        final_resids[:len(revised_resids)] = revised_resids
        next_resnum = revised_resids[-1] + 1
        final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
        non_water_ag.residues.resids = final_resids

    u.atoms.write(str(_p("rec_file.pdb")))
    shutil.copy2(_p("rec_file.pdb"), _p("equil-reference.pdb"))

    # 5) VMD split -> split.tcl generated under workdir
    other_mol_vmd = " ".join(other_mol) if other_mol else "XXX"
    lipid_mol_vmd = " ".join(lipid_mol) if lipid_mol else "XXX"
    with open(_p("split-ini.tcl"), "rt") as fin, open(_p("split.tcl"), "wt") as fout:
        for line in fin:
            fout.write(
                line.replace("SHLL", f"{solv_shell:4.2f}")
                    .replace("OTHRS", str(other_mol_vmd))
                    .replace("LIPIDS", str(lipid_mol_vmd))
                    .replace("mmm", mol)
                    .replace("MMM", mol)
            )
    run_with_log(f'{vmd} -dispdev text -e split.tcl', shell=False,
                    working_dir=workdir)

    # 6) merge -> complex.pdb (strip headers/CRYST1/CONECT/END)
    pieces = ["dummy.pdb","protein.pdb",f"{mol}.pdb","lipids.pdb","others.pdb","crystalwat.pdb"]
    if not all(_p(f).exists() for f in pieces):
        missing = [f for f in pieces if not _p(f).exists()]
        raise FileNotFoundError(f"Missing split output files: {', '.join(missing)}")
    (_p("complex-merge.pdb")).write_text(
        "".join((_p(f).read_text()) for f in pieces)
    )
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
    p1_resid = P1.split("@")[0][1:]; p1_atom = P1.split("@")[1]
    rec_res = int(recep_last) + 1;   p1_vmd  = p1_resid

    # 8) SDR distance
    if not buffer_z: buffer_z = 25
    sdr_dist = get_sdr_dist(str(_p("complex.pdb")), lig_resname=mol, buffer_z=buffer_z, extra_buffer=5)

    # 9) align & pdb4amber
    run_with_log(f"{vmd} -dispdev text -e measure-fit.tcl", shell=False, working_dir=workdir)
    with open(_p("aligned.pdb")) as fin, open(_p("aligned-clean.pdb"), "wt") as fout:
        for ln in fin:
            if len(ln.split()) > 3: fout.write(ln)
    run_with_log(f"pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y", working_dir=workdir)

    # optional lipid resid fix post-amber
    if b.membrane_builder:
        u = mda.Universe(_p("aligned_amber.pdb"))
        non_water_ag = u.select_atoms('not resname WAT Na+ Cl- K+')
        non_water_ag.residues.resids = final_resids

        u.atoms.write(_p("aligned_amber.pdb"))

    # 10) ligand candidates for Boresch
    sdf_file = _p(f"{mol}.sdf")
    candidates_indices = get_ligand_candidates(str(sdf_file))
    u3 = mda.Universe(str(_p("aligned_amber.pdb")))
    lig_names = u3.select_atoms(f"resname {mol}").names
    lig_name_str = " ".join(str(x) for x in lig_names[candidates_indices])

    # 11) prep.tcl
    with open(_p("prep-ini.tcl"), "rt") as fin, open(_p("prep.tcl"), "wt") as fout:
        for line in fin:
            fout.write(
                line.replace("MMM", mol)
                    .replace("mmm", mol)
                    .replace("NN", p1_atom)
                    .replace("P1A", p1_vmd)
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
                    .replace("LIGSITE", "0") # no FB for ligand now
                    .replace("OTHRS", " ".join(other_mol) if other_mol else "XXX")
                    .replace("LIPIDS", " ".join(lipid_mol) if lipid_mol else "XXX")
                    .replace("LIGANDNAME", lig_name_str)
            )
    try:
        run_with_log(f"{vmd} -dispdev text -e prep.tcl", error_match="anchor not found", shell=False,
                     working_dir=workdir)
    except RuntimeError:
        logger.info("Default candidates failed; retry with ALL ligand atoms.")
        lig_name_str = " ".join(str(x) for x in lig_names)
        with open(_p("prep-ini.tcl"), "rt") as fin, open(_p("prep.tcl"), "wt") as fout:
            for line in fin:
                fout.write(
                    line.replace("MMM", mol)
                        .replace("mmm", mol)
                        .replace("NN", p1_atom)
                        .replace("P1A", p1_vmd)
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
                        .replace("LIGSITE", "0") # no FB for ligand now
                        .replace("OTHRS", " ".join(other_mol) if other_mol else "XXX")
                        .replace("LIPIDS", " ".join(lipid_mol) if lipid_mol else "XXX")
                        .replace("LIGANDNAME", lig_name_str)
                )
        run_with_log(f"{vmd} -dispdev text -e prep.tcl", error_match="anchor not found", shell=False,
                     working_dir=workdir)

    # 12) anchors.txt -> validate, rename with ligand tag, write header into fe-<mol>.pdb
    anchors_txt = _p("anchors.txt")
    if (not anchors_txt.exists()) or (anchors_txt.stat().st_size == 0):
        logger.warning("anchors.txt missing or empty")
        return False
    good = True
    with anchors_txt.open() as f:
        for ln in f:
            if len(ln.split()) < 3:
                good = False; break
    tagged = _p(f"anchors-{ligand}.txt")
    anchors_txt.rename(tagged)
    if not good:
        logger.warning("anchors.txt too short; pruning")
        return False

    lig_resid = str(int(recep_last) + 2)
    with tagged.open() as f:
        a = f.readline().split()
        L1 = f":{lig_resid}@{a[0]}"; L2 = f":{lig_resid}@{a[1]}"; L3 = f":{lig_resid}@{a[2]}"

    fe_pdb = _p(f"fe-{mol}.pdb")
    if not fe_pdb.exists():
        raise FileNotFoundError(f"Missing {fe_pdb}")
    lines = fe_pdb.read_text().splitlines(True)
    with fe_pdb.open("wt") as fout:
        fout.write(f"{'REMARK A':<8s}  {P1:6s}  {P2:6s}  {P3:6s}  {L1:6s}  {L2:6s}  {L3:6s}  {first_res:6s}  {recep_last:4s}\n")
        fout.writelines(lines[1:])

    save_anchors(workdir, Anchors(P1=P1, P2=P2, P3=P3, L1=L1, L2=L2, L3=L3, lig_res=lig_resid))

    return True

@register_build_complex("y")
def build_complex_y(ctx) -> bool:
    """
    Component 'y' (ligand-only) build_complex:
    - No receptor complexing; just stage the ligand structural files.
    - Sets builder.mol and builder.corrected_sdr_dist for downstream code.
    """
    # Where to put staged files
    build_dir: Path = ctx.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve locations
    ligand = ctx.ligand
    system_root: Path = ctx.system_root
    all_ligands_dir = system_root / "all-ligands"
    ff_dir = ctx.working_dir.parent / "params"

    # Inputs
    ligand_pdb = all_ligands_dir / f"{ligand}.pdb"
    if not ligand_pdb.exists():
        raise FileNotFoundError(f"[build_complex_y] Missing ligand pdb: {ligand_pdb}")

    # Copy <pose>.pdb into build_dir
    shutil.copy2(ligand_pdb, build_dir / f"{ligand}.pdb")

    mol = ctx.residue_name

    # Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf", ".pdb"):
        src = ff_dir / f"{mol}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_y] Missing ligand FF file: {src}")
        shutil.copy2(src, build_dir / src.name)

    return True


@register_sim_files("z")
def sim_files_z(ctx, lambdas) -> None:
    """
    Create per-window MD input files for component 'z' (UNO-REST style),
    supporting decoupling methods 'sdr' and 'dd'.
    """
    work: Path = ctx.working_dir
    sim = ctx.sim
    comp = ctx.comp
    mol = ctx.residue_name
    win = ctx.win
    windows_dir = ctx.window_dir

    dec_method = getattr(sim, "dec_method", None)
    if dec_method not in {"sdr", "dd"}:
        raise ValueError(f"Decoupling method '{dec_method}' not recognized. Use 'sdr' or 'dd'.")

    temperature = sim.temperature
    num_sim = int(sim.num_fe_range)
    steps1 = sim.dic_steps1[comp]
    steps2 = sim.dic_steps2[comp]
    ntwx = sim.ntwx

    weight = lambdas[win if win != -1 else 0]

    vac_pdb = windows_dir / "vac.pdb"
    if not vac_pdb.exists():
        raise FileNotFoundError(f"Missing required file: {vac_pdb}")

    vac_atoms = mda.Universe(vac_pdb.as_posix()).atoms.n_atoms

    # find *last* residue index of this ligand in vac.pdb
    last_lig: Optional[str] = None
    with vac_pdb.open("rt") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[17:20].strip().lower() == mol.lower():
                last_lig = line[22:26].strip()
    if last_lig is None:
        raise ValueError(f"No ligand residue matching '{mol}' found in {vac_pdb.name}")

    amber_dir = ctx.amber_dir

    if dec_method == "sdr":
        mk2 = int(last_lig)
        mk1 = mk2 - 1
        template_mdin = amber_dir / "mdin-unorest"
        template_mini = amber_dir / "mini-unorest"

        for i in range(0, num_sim + 1):
            n_steps_run = str(steps1) if i == 0 else str(steps2)
            out_path = windows_dir / f"mdin-{i:02d}"
            with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
                for line in fin:
                    if i == 0:
                        if "ntx = 5" in line:
                            line = "ntx = 1,\n"
                        elif "irest" in line:
                            line = "irest = 0,\n"
                        elif "dt = " in line:
                            line = "dt = 0.001,\n"
                        elif "restraintmask" in line:
                            rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                            if rm == "":
                                line = f"restraintmask = '(@CA | :{mol}) & !@H='\n"
                            else:
                                line = f"restraintmask = '(@CA | :{mol} | {rm}) & !@H='\n"

                    line = (
                        line.replace("_temperature_", str(temperature))
                            .replace("_num-atoms_", str(vac_atoms))
                            .replace("_num-steps_", n_steps_run)
                            .replace("lbd_val", f"{float(weight):6.5f}")
                            .replace("mk1", str(mk1))
                            .replace("mk2", str(mk2))
                    )
                    fout.write(line)

            with out_path.open("a") as mdin:
                mdin.write(f"  mbar_states = {len(lambdas):02d}\n")
                mdin.write("  mbar_lambda =")
                for lam in lambdas:
                    mdin.write(f" {lam:6.5f},")
                mdin.write("\n")
                mdin.write("  infe = 1,\n")
                mdin.write(" /\n")
                mdin.write(" &pmd \n")
                mdin.write("  output_file = 'cmass.txt'\n")
                mdin.write(f"  output_freq = {int(ntwx):02d}\n")
                mdin.write("  cv_file = 'cv.in'\n")
                mdin.write(" /\n")
                mdin.write(" &wt type = 'END' , /\n")
                mdin.write("DISANG=disang.rest\n")
                mdin.write("LISTOUT=POUT\n")

        with template_mini.open("rt") as fin, (windows_dir / "mini.in").open("wt") as fout:
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                        .replace("lbd_val", f"{float(weight):6.5f}")
                        .replace("mk1", str(mk1))
                        .replace("mk2", str(mk2))
                        .replace("_lig_name_", mol)
                )
                fout.write(line)

    else:  # dd
        infe_flag = 1 if getattr(ctx, "infe", False) else 0
        mk1 = int(last_lig)
        template_mdin = amber_dir / "mdin-unorest-dd"
        template_mini = amber_dir / "mini-unorest-dd"

        for i in range(0, num_sim + 1):
            n_steps_run = str(steps1) if i == 0 else str(steps2)
            out_path = windows_dir / f"mdin-{i:02d}"
            with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
                for line in fin:
                    if i == 0:
                        if "ntx = 5" in line:
                            line = "ntx = 1,\n"
                        elif "irest" in line:
                            line = "irest = 0,\n"
                        elif "dt = " in line:
                            line = "dt = 0.001,\n"
                        elif "restraintmask" in line:
                            rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                            if rm == "":
                                line = f"restraintmask = '(@CA | :{mol}) & !@H='\n"
                            else:
                                line = f"restraintmask = '(@CA | :{mol} | {rm}) & !@H='\n"

                    line = (
                        line.replace("_temperature_", str(temperature))
                            .replace("_num-atoms_", str(vac_atoms))
                            .replace("_num-steps_", n_steps_run)
                            .replace("lbd_val", f"{float(weight):6.5f}")
                            .replace("mk1", str(mk1))
                    )
                    fout.write(line)

            with out_path.open("a") as mdin:
                mdin.write(f"  mbar_states = {len(lambdas):02d}\n")
                mdin.write("  mbar_lambda =")
                for lam in lambdas:
                    mdin.write(f" {lam:6.5f},")
                mdin.write("\n")
                mdin.write(f"  infe = {infe_flag},\n")
                mdin.write(" /\n")
                mdin.write(" &pmd \n")
                mdin.write("  output_file = 'cmass.txt'\n")
                mdin.write(f"  output_freq = {int(ntwx):02d}\n")
                mdin.write("  cv_file = 'cv.in'\n")
                mdin.write(" /\n")
                mdin.write(" &wt type = 'END' , /\n")
                mdin.write("DISANG=disang.rest\n")
                mdin.write("LISTOUT=POUT\n")

        with template_mini.open("rt") as fin, (windows_dir / "mini.in").open("wt") as fout:
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                        .replace("lbd_val", f"{float(weight):6.5f}")
                        .replace("mk1", str(mk1))
                        .replace("_lig_name_", mol)
                )
                fout.write(line)

    # Always emit mini_eq.in, eqnpt0.in, eqnpt.in from UNO templates:
    amber_dir = ctx.amber_dir

    with (amber_dir / "mini.in").open("rt") as fin, (windows_dir / "mini_eq.in").open("wt") as fout:
        for line in fin:
            fout.write(line.replace("_lig_name_", mol))

    with (amber_dir / "eqnpt0-uno.in").open("rt") as fin, (windows_dir / "eqnpt0.in").open("wt") as fout:
        for line in fin:
            if "mcwat" in line:
                fout.write("  mcwat = 0,\n")
            else:
                fout.write(line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol))

    with (amber_dir / "eqnpt-uno.in").open("rt") as fin, (windows_dir / "eqnpt.in").open("wt") as fout:
        for line in fin:
            if "mcwat" in line:
                fout.write("  mcwat = 0,\n")
            else:
                fout.write(line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol))

    (windows_dir / "lambda.sch").write_text("TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n")

    logger.debug(f"[sim_files_z] wrote mdin/mini/eq inputs in {windows_dir} for comp='z', win={win}, weight={weight:0.5f}")


@register_sim_files("y")
def sim_files_y(ctx, lambdas) -> None:
    """
    Generate MD input files for ligand-only component 'y'.
    """
    work: Path = ctx.working_dir
    sim = ctx.sim
    comp = ctx.comp
    mol = ctx.residue_name
    win = ctx.win
    windows_dir = ctx.window_dir


    temperature = sim.temperature
    num_sim = int(sim.num_fe_range)
    steps1 = sim.dic_steps1[comp]
    steps2 = sim.dic_steps2[comp]
    ntwx = sim.ntwx

    weight = lambdas[win if win != -1 else 0]
    mk1 = 2  # ligand-only marker convention

    amber_dir = ctx.amber_dir

    # mini.in from ligand template
    with (amber_dir / "mini-unorest-lig").open("rt") as fin, (windows_dir / "mini.in").open("wt") as fout:
        for line in fin:
            line = (
                line.replace("_temperature_", str(temperature))
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                    .replace("_lig_name_", mol)
            )
            fout.write(line)

    # mini_eq.in from generic mini template
    with (amber_dir / "mini.in").open("rt") as fin, (windows_dir / "mini_eq.in").open("wt") as fout:
        for line in fin:
            fout.write(line.replace("_lig_name_", mol))

    # eqnpt.in / eqnpt0.in from ligand templates
    with (amber_dir / "eqnpt-lig.in").open("rt") as fin, (windows_dir / "eqnpt.in").open("wt") as fout:
        for line in fin:
            fout.write(
                line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol)
            )
    with (amber_dir / "eqnpt0-lig.in").open("rt") as fin, (windows_dir / "eqnpt0.in").open("wt") as fout:
        for line in fin:
            fout.write(
                line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol)
            )

    # per-window production inputs
    template = amber_dir / "mdin-unorest-lig"
    for i in range(0, num_sim + 1):
        out_path = windows_dir / f"mdin-{i:02d}"
        n_steps_run = str(steps1) if i == 0 else str(steps2)

        with template.open("rt") as fin, out_path.open("wt") as fout:
            for line in fin:
                if i == 0:
                    if "ntx = 5" in line:
                        line = "ntx = 1,\n"
                    elif "irest" in line:
                        line = "irest = 0,\n"
                    elif "dt = " in line:
                        line = "dt = 0.001,\n"
                line = (
                    line.replace("_temperature_", str(temperature))
                        .replace("_num-steps_", n_steps_run)
                        .replace("lbd_val", f"{float(weight):6.5f}")
                        .replace("mk1", str(mk1))
                        .replace("disang_file", "disang")
                        .replace("_lig_name_", mol)
                )
                fout.write(line)

        with out_path.open("a") as mdin:
            mdin.write(f"  mbar_states = {len(lambdas)}\n")
            mdin.write("  mbar_lambda =")
            for lbd in lambdas:
                mdin.write(f" {lbd:6.5f},")
            mdin.write("\n")
            mdin.write("  infe = 1,\n")
            mdin.write(" /\n")
            mdin.write(" &pmd \n")
            mdin.write("  output_file = 'cmass.txt'\n")
            mdin.write(f"  output_freq = {int(ntwx):02d}\n")
            mdin.write("  cv_file = 'cv.in'\n")
            mdin.write(" /\n")
            mdin.write(" &wt type = 'END' , /\n")
            mdin.write("DISANG=disang.rest\n")
            mdin.write("LISTOUT=POUT\n")

    logger.debug(f"[sim_files_y] wrote mdin/mini/eq inputs in {windows_dir} for comp='y', win={win}, weight={weight:0.5f}")