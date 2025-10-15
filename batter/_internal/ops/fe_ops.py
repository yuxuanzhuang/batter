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
from batter.utils import (
    run_with_log,
    cpptraj,
    vmd
)


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
    lipid_mol = b.ctx.lipid_mol
    other_mol = b.ctx.other_mol
    sim = b.ctx.sim

    solv_shell   = sim.solv_shell
    l1_x, l1_y, l1_z = sim.l1_x, sim.l1_y, sim.l1_z
    l1_range    = sim.l1_range
    max_adis    = sim.max_adis
    min_adis    = sim.min_adis
    buffer_z    = sim.buffer_z
    hmr         = sim.hmr

    workdir   = b.build_dir; workdir.mkdir(parents=True, exist_ok=True)
    child_root = b.ctx.working_dir                         # .../simulations/<LIG>/fe/...
    sys_root   = b.ctx.system_root                         # .../work/<system>
    equil_dir  = sys_root / "equilibration" / ligand  # .../work/<system>/equilibration/<LIG>

    # --- helpers to keep paths explicit ---
    def _p(name: str) -> Path: return workdir / name
    def _copy(src: Path, dst_name: str):
        if src.exists():
            shutil.copy2(src, _p(dst_name))
        else:
            logger.warning(f"[build_complex_z] missing: {src}")

    # 1) copy artifacts from equil
    _copy(equil_dir / "build_files" / f"{ligand}.pdb",          f"{ligand}.pdb")
    _copy(equil_dir / "representative.rst7",                    "representative.rst7")
    _copy(equil_dir / "representative.pdb",                     "aligned-nc.pdb")
    _copy(equil_dir / "build_amber_renum.txt",                  "build_amber_renum.txt")
    _copy(equil_dir / "build_files" / "protein_renum.txt",      "protein_renum.txt")

    for p in equil_dir.glob("full*.prmtop"): shutil.copy2(p, _p(p.name))
    for p in equil_dir.glob("vac*"):          shutil.copy2(p, _p(p.name))

    # 2) deduce ligand resname from copied ligand PDB
    mol = mda.Universe(str(_p(f"{ligand}.pdb"))).residues[0].resname
    b.mol = mol

    prmtop_f = "full.prmtop" if str(hmr).lower() == "no" else "full.hmr.prmtop"

    # 3) extract receptor-only PDB from representative.rst7
    run_with_log(f"{cpptraj} -p {prmtop_f} -y {_p('representative.rst7')} -x {_p('rec_file.pdb')}")

    # 4) reapply chain IDs from renum map; optional lipid resid compaction
    renum = pd.read_csv(_p("build_amber_renum.txt"), sep=r"\s+", header=None,
                        names=['old_resname','old_chain','old_resid','new_resname','new_resid'])
    u = mda.Universe(str(_p("rec_file.pdb")))
    for residue in u.select_atoms('protein').residues:
        resid_str = residue.resid
        chain = renum.query('old_resid == @resid_str').old_chain.values
        if chain.size: residue.atoms.chainIDs = chain[0]

    if b.membrane_builder:
        revised, resid_counter, prev_resid = [], 1, None
        for _, row in renum.iterrows():
            if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                revised.append(resid_counter); resid_counter += 1
            else:
                revised.append(resid_counter - 1)
            prev_resid = row['old_resid']
        revised = np.asarray(revised, dtype=int)
        total = u.atoms.residues.n_residues
        final_resids = np.zeros(total, dtype=int)
        n = min(total, revised.size)
        final_resids[:n] = revised[:n]
        if n < total:
            final_resids[n:] = np.arange(final_resids[n-1] + 1, final_resids[n-1] + 1 + (total - n))
        u.atoms.residues.resids = final_resids

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
                    .replace("mmm", mol.lower())
                    .replace("MMM", f"'{mol}'")
            )
    run_with_log(f"{vmd} -dispdev text -e {_p('split.tcl')}", shell=False)

    # 6) merge -> complex.pdb (strip headers/CRYST1/CONECT/END)
    pieces = ["dummy.pdb","protein.pdb",f"{mol.lower()}.pdb","lipids.pdb","others.pdb","crystalwat.pdb"]
    (_p("complex-merge.pdb")).write_text(
        "".join(((_p(f).read_text()) if _p(f).exists() else "") for f in pieces)
    )
    with open(_p("complex-merge.pdb")) as fin, open(_p("complex.pdb"), "wt") as fout:
        for ln in fin:
            if ("CRYST1" in ln) or ("CONECT" in ln) or ln.startswith("END"):
                continue
            fout.write(ln)

    # 7) read anchors/meta from equil header
    equil_info = equil_dir / f"equil-{mol.lower()}.pdb"
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
    sdr_dist = get_sdr_dist(str(_p("complex.pdb")), lig_resname=mol.lower(), buffer_z=buffer_z, extra_buffer=5)
    logger.debug(f"SDR distance: {sdr_dist:.2f}")
    b.corrected_sdr_dist = sdr_dist

    # 9) align & pdb4amber
    run_with_log(f"{vmd} -dispdev text -e {_p('measure-fit.tcl')}", shell=False)
    with open(_p("aligned.pdb")) as fin, open(_p("aligned-clean.pdb"), "wt") as fout:
        for ln in fin:
            if len(ln.split()) > 3: fout.write(ln)
    run_with_log(f"pdb4amber -i {_p('aligned-clean.pdb')} -o {_p('aligned_amber.pdb')} -y")

    # optional lipid resid fix post-amber
    if b.membrane_builder and _p("aligned_amber_renum.txt").exists():
        u2 = mda.Universe(str(_p("aligned_amber.pdb")))
        ren = pd.read_csv(_p("aligned_amber_renum.txt"), sep=r"\s+", header=None,
                          names=["old_resname","old_resid","new_resname","new_resid"])
        revised, resid_counter, prev_resid = [], 1, None
        for _, row in ren.iterrows():
            if row["old_resid"] != prev_resid or row["old_resname"] not in lipid_mol:
                revised.append(resid_counter); resid_counter += 1
            else:
                revised.append(resid_counter - 1)
            prev_resid = row["old_resid"]
        revised = np.asarray(revised, dtype=int)
        total = u2.atoms.residues.n_residues
        final_resids = np.arange(1, total + 1, dtype=int)
        n = min(total, revised.size)
        final_resids[:n] = revised[:n]
        u2.atoms.residues.resids = final_resids
        u2.atoms.write(str(_p("aligned_amber.pdb")))

    # 10) ligand candidates for Boresch
    sdf_file = _p(f"{mol.lower()}.sdf")
    candidates_indices = get_ligand_candidates(str(sdf_file))
    u3 = mda.Universe(str(_p("aligned_amber.pdb")))
    lig_names = u3.select_atoms(f"resname {mol.lower()}").names
    lig_name_str = " ".join(str(x) for x in lig_names[candidates_indices])

    # 11) prep.tcl
    with open(_p("prep-ini.tcl"), "rt") as fin, open(_p("prep.tcl"), "wt") as fout:
        for line in fin:
            fout.write(
                line.replace("MMM", f"'{mol}'")
                    .replace("mmm", mol.lower())
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
                    .replace("LIGSITE", "1")
                    .replace("OTHRS", " ".join(other_mol) if other_mol else "XXX")
                    .replace("LIPIDS", " ".join(lipid_mol) if lipid_mol else "XXX")
                    .replace("LIGANDNAME", lig_name_str)
            )
    try:
        run_with_log(f"{vmd} -dispdev text -e {_p('prep.tcl')}", error_match="anchor not found", shell=False)
    except RuntimeError:
        logger.info("Default candidates failed; retry with ALL ligand atoms.")
        lig_name_str = " ".join(str(x) for x in lig_names)
        with open(_p("prep-ini.tcl"), "rt") as fin, open(_p("prep.tcl"), "wt") as fout:
            for line in fin:
                fout.write(
                    line.replace("MMM", f"'{mol}'")
                        .replace("mmm", mol.lower())
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
                        .replace("LIGSITE", "1")
                        .replace("OTHRS", " ".join(other_mol) if other_mol else "XXX")
                        .replace("LIPIDS", " ".join(lipid_mol) if lipid_mol else "XXX")
                        .replace("LIGANDNAME", lig_name_str)
                )
        run_with_log(f"{vmd} -dispdev text -e {_p('prep.tcl')}", error_match="anchor not found", shell=False)

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

    fe_pdb = _p(f"fe-{mol.lower()}.pdb")
    if not fe_pdb.exists():
        raise FileNotFoundError(f"Missing {fe_pdb}")
    lines = fe_pdb.read_text().splitlines(True)
    with fe_pdb.open("wt") as fout:
        fout.write(f"{'REMARK A':-8s}  {P1:6s}  {P2:6s}  {P3:6s}  {L1:6s}  {L2:6s}  {L3:6s}  {first_res:6s}  {recep_last:4s}\n")
        fout.writelines(lines[1:])

    # (optional) persist anchors JSON for downstream
    try:
        from batter._internal.ops.helper import Anchors, save_anchors
        save_anchors(workdir, Anchors(P1=P1, P2=P2, P3=P3, L1=L1, L2=L2, L3=L3, lig_res=lig_resid))
    except Exception as e:
        logger.debug(f"[build_complex_z] could not write anchors.json: {e}")

    return True

@register_build_complex("y")
def build_complex_y(builder) -> bool:
    """
    Component 'y' (ligand-only) build_complex:
    - No receptor complexing; just stage the ligand structural files.
    - Sets builder.mol and builder.corrected_sdr_dist for downstream code.
    """
    # Where to put staged files
    build_dir: Path = builder.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve locations
    ligand = builder.ctx.ligand
    system_root: Path = builder.ctx.system_root
    all_ligands_dir = system_root / "all-ligands"
    ff_dir = builder.ctx.working_dir.parent / "ff"   # e.g., <child_root>/fe/ff

    # Inputs
    ligand_pdb = all_ligands_dir / f"{ligand}.pdb"
    if not ligand_pdb.exists():
        raise FileNotFoundError(f"[build_complex_y] Missing ligand pdb: {ligand_pdb}")

    # Copy <pose>.pdb into build_dir
    shutil.copy2(ligand_pdb, build_dir / f"{ligand}.pdb")

    mol = builder.ctx.residue_name

    # Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf", ".pdb"):
        src = ff_dir / f"{mol.lower()}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_y] Missing ligand FF file: {src}")
        shutil.copy2(src, build_dir / src.name)

    # No SDR placement needed for ligand-only
    builder.corrected_sdr_dist = 0

    return True


@register_sim_files("z")
def sim_files_z(builder) -> None:
    """
    Create per-window MD input files for component 'z' (UNO-REST style),
    supporting decoupling methods 'sdr' and 'dd'.

    Expects the following templates to exist in ../<amber_files_folder>/:
      - mdin-unorest              (for SDR)
      - mdin-unorest-dd           (for DD)
      - mini-unorest              (for SDR)
      - mini-unorest-dd           (for DD)
      - mini.in (for mini_eq.in)  (generic)
      - eqnpt0-uno.in
      - eqnpt-uno.in
    """
    import os
    import MDAnalysis as mda

    sim = builder.sim_config
    comp = builder.comp
    mol = builder.mol
    win = builder.win

    dec_method = getattr(sim, "dec_method", None)
    if dec_method not in {"sdr", "dd"}:
        raise ValueError(f"Decoupling method '{dec_method}' not recognized. Use 'sdr' or 'dd'.")

    temperature = sim.temperature
    num_sim = int(sim.num_fe_range)
    steps1 = sim.dic_steps1[comp]
    steps2 = sim.dic_steps2[comp]
    ntwx = sim.ntwx

    lambdas = builder.component_windows_dict[comp]
    weight = lambdas[win if win != -1 else 0]

    # vac system info
    vac_atoms = mda.Universe("./vac.pdb").atoms.n_atoms

    # find *last* residue index of this ligand in vac.pdb
    last_lig = None
    with open("./vac.pdb", "rt") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[17:20].strip().lower() == mol.lower():
                last_lig = line[22:26].strip()
    if last_lig is None:
        raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

    amber_dir_rel = Path("..") / builder.amber_files_folder

    if dec_method == "sdr":
        # simultaneous decouple → two markers bracketing ligand
        mk2 = int(last_lig)
        mk1 = mk2 - 1
        template_mdin = amber_dir_rel / "mdin-unorest"
        template_mini = amber_dir_rel / "mini-unorest"

        for i in range(0, num_sim + 1):
            n_steps_run = str(steps1) if i == 0 else str(steps2)
            out_path = Path(f"./mdin-{i:02d}")
            with open(template_mdin, "rt") as fin, open(out_path, "wt") as fout:
                for line in fin:
                    # first window tweaks
                    if i == 0:
                        if "ntx = 5" in line:
                            line = "ntx = 1,\n"
                        elif "irest" in line:
                            line = "irest = 0,\n"
                        elif "dt = " in line:
                            line = "dt = 0.001,\n"
                        elif "restraintmask" in line:
                            # merge CA + ligand with any pre-existing mask
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

            # append MBAR + PME meta & NFE CV
            with open(out_path, "a") as mdin:
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

        # minimization template for SDR
        with open(template_mini, "rt") as fin, open("./mini.in", "wt") as fout:
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                        .replace("lbd_val", f"{float(weight):6.5f}")
                        .replace("mk1", str(mk1))
                        .replace("mk2", str(mk2))
                        .replace("_lig_name_", mol)
                )
                fout.write(line)

    else:  # dec_method == "dd"
        infe_flag = 1 if getattr(builder, "infe", False) else 0
        mk1 = int(last_lig)
        template_mdin = amber_dir_rel / "mdin-unorest-dd"
        template_mini = amber_dir_rel / "mini-unorest-dd"

        for i in range(0, num_sim + 1):
            n_steps_run = str(steps1) if i == 0 else str(steps2)
            out_path = Path(f"./mdin-{i:02d}")
            with open(template_mdin, "rt") as fin, open(out_path, "wt") as fout:
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

            with open(out_path, "a") as mdin:
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

        with open(template_mini, "rt") as fin, open("./mini.in", "wt") as fout:
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                        .replace("lbd_val", f"{float(weight):6.5f}")
                        .replace("mk1", str(mk1))
                        .replace("_lig_name_", mol)
                )
                fout.write(line)

    # Always emit mini_eq.in, eqnpt0.in, eqnpt.in from UNO templates:
    with open(amber_dir_rel / "mini.in", "rt") as fin, open("./mini_eq.in", "wt") as fout:
        for line in fin:
            fout.write(line.replace("_lig_name_", mol))

    with open(amber_dir_rel / "eqnpt0-uno.in", "rt") as fin, open("./eqnpt0.in", "wt") as fout:
        for line in fin:
            if "mcwat" in line:
                fout.write("  mcwat = 0,\n")
            else:
                fout.write(line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol))

    with open(amber_dir_rel / "eqnpt-uno.in", "rt") as fin, open("./eqnpt.in", "wt") as fout:
        for line in fin:
            if "mcwat" in line:
                fout.write("  mcwat = 0,\n")
            else:
                fout.write(line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol))

    # Lambda schedule file
    with open("./lambda.sch", "wt") as fout:
        fout.write("TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n")

    logger.debug(f"[sim_files_z] wrote mdin/mini/eq inputs for comp='z', win={win}, weight={weight:0.5f}")


@register_sim_files("y")
def sim_files_y(builder) -> None:
    """
    Generate MD input files for ligand-only component 'y'.

    Requires the following templates in ../<amber_files_folder>/:
      - mini-unorest-lig
      - mini.in                (for mini_eq.in)
      - eqnpt-lig.in
      - eqnpt0-lig.in
      - mdin-unorest-lig
    """
    sim = builder.sim_config
    comp = builder.comp           # 'y'
    mol = builder.mol
    win = builder.win

    temperature = sim.temperature
    num_sim = int(sim.num_fe_range)
    steps1 = sim.dic_steps1[comp]
    steps2 = sim.dic_steps2[comp]
    ntwx = sim.ntwx

    lambdas = builder.component_windows_dict[comp]
    weight = lambdas[win if win != -1 else 0]

    # mk1 is fixed to residue 2 for ligand-only setup (matches your original)
    mk1 = 2

    amber_dir_rel = Path("..") / builder.amber_files_folder

    # mini.in from ligand template
    with open(amber_dir_rel / "mini-unorest-lig", "rt") as fin, open("./mini.in", "wt") as fout:
        for line in fin:
            line = (
                line.replace("_temperature_", str(temperature))
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                    .replace("_lig_name_", mol)
            )
            fout.write(line)

    # mini_eq.in from generic mini template
    with open(amber_dir_rel / "mini.in", "rt") as fin, open("./mini_eq.in", "wt") as fout:
        for line in fin:
            fout.write(line.replace("_lig_name_", mol))

    # eqnpt.in / eqnpt0.in from ligand templates
    with open(amber_dir_rel / "eqnpt-lig.in", "rt") as fin, open("./eqnpt.in", "wt") as fout:
        for line in fin:
            fout.write(
                line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol)
            )
    with open(amber_dir_rel / "eqnpt0-lig.in", "rt") as fin, open("./eqnpt0.in", "wt") as fout:
        for line in fin:
            fout.write(
                line.replace("_temperature_", str(temperature)).replace("_lig_name_", mol)
            )

    # per-window production inputs
    for i in range(0, num_sim + 1):
        template = amber_dir_rel / "mdin-unorest-lig"
        out_path = Path(f"./mdin-{i:02d}")
        n_steps_run = str(steps1) if i == 0 else str(steps2)

        with open(template, "rt") as fin, open(out_path, "wt") as fout:
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

        # append MBAR, PME/NFE and restraints section
        with open(out_path, "a") as mdin:
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

    logger.debug(f"[sim_files_y] wrote mdin/mini/eq inputs for comp='y', win={win}, weight={weight:0.5f}")