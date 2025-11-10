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
from batter._internal.builders.fe_registry import register_build_complex, register_sim_files
from batter._internal.ops.helpers import (
    get_buffer_z,
    get_sdr_dist,
    get_ligand_candidates,
    select_ions_away_from_complex,
    Anchors,
    save_anchors
)
from batter._internal.templates import BUILD_FILES_DIR as build_files_orig  # type: ignore


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_complex(ctx: BuildContext, *, infe: bool = False) -> bool:
    """
    Creates the aligned + cleaned PDBs (protein/others/lipids), finds
    receptor/ligand anchors, generates `equil-<lig>.pdb` and
    `anchors-<ligand>.txt`. Returns False if anchors can’t be found.
    """
    sim = ctx.sim
    ligand = ctx.ligand
    mol = ctx.residue_name

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
        raise ValueError(f"The ligand {mol} cannot be in the other_mol/lipid_mol list: {other_mol} and {lipid_mol}")

    logger.debug(f"[Equil] Building complex for ligand {ligand} with other_mol={other_mol} lipid_mol={lipid_mol}")
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

    ante_mol = mda.Universe(str(build_dir / f"{mol}.mol2"))
    lig_u = mda.Universe(str(build_dir / f"{ligand}.pdb"))
    lig_u.atoms.names = ante_mol.atoms.names
    lig_u.atoms.residues.resnames = mol
    lig_u.atoms.write(str(build_dir / f"{mol}.pdb"))

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

    run_with_log(f"{vmd} -dispdev text -e {str(split_tcl)}", error_match="syntax error", shell=False, working_dir=build_dir)
    # Protein PDB cleanup with pdb4amber
    shutil.copy2(build_dir / "protein.pdb", build_dir / "protein_vmd.pdb")
    run_with_log("pdb4amber -i protein_vmd.pdb -o protein.pdb -y", working_dir=build_dir)

    renum_txt = build_dir / "protein_renum.txt"
    renum_data = pd.read_csv(
        renum_txt, sep=r"\s+", header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"]
    )

    # Original receptor numbering (to detect missing residues)
    u_original = mda.Universe(str(build_dir / "rec_file.pdb"))
    first_res = int(u_original.residues[0].resid)
    recep_resid_num = len(mda.Universe(str(build_dir / "protein.pdb")).residues)

    # Adjust protein anchors to new numbering
    def _extract_resid(atom_spec: str) -> tuple[int, str]:
        # e.g. ":113@CA" or "113@CA" formats used in your config
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
        run_with_log(f"{charmmlipid2amber} -i {build_dir/'lipids.pdb'} -o {build_dir/'lipids_amber.pdb'}")
        u_lip = mda.Universe(str(build_dir / "lipids_amber.pdb"))
        lipid_resnames = list(set(u_lip.residues.resnames))
        logger.debug(f"[Equil] Converted CHARMM lipids to AMBER: {lipid_resnames}")
        lipid_mol = lipid_resnames  # updated list

    # Merge raw complex (protein + ligand + others + (lipids) + crystal waters)
    parts: list[Path] = [build_dir / "protein.pdb", build_dir / f"{mol}.pdb", build_dir / "others.pdb"]
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
    run_with_log("pdb4amber -i reference.pdb -o reference_amber.pdb -y", working_dir=build_dir)
    run_with_log(f"{vmd} -dispdev text -e nochain.tcl", shell=False, working_dir=build_dir)
    run_with_log("./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc", working_dir=build_dir)
    run_with_log(f"{vmd} -dispdev text -e measure-fit.tcl", shell=False, working_dir=build_dir)

    # Clean aligned and put in AMBER format
    with open(build_dir / "aligned.pdb", "r") as oldfile, open(build_dir / "aligned-clean.pdb", "w") as newfile:
        for line in oldfile:
            if len(line.split()) > 4:
                newfile.write(line)
    run_with_log("pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y", working_dir=build_dir)

    # For membrane: restore box info and re-merge lipid partial residues into single resids
    if sim.membrane_simulation:
        u_aln = mda.Universe(str(build_dir / "aligned_amber.pdb"))
        u_aln.dimensions = u_original.dimensions
        renum_txt2 = build_dir / "aligned_amber_renum.txt"
        ren2 = pd.read_csv(
            renum_txt2, sep=r"\s+", header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"]
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
    candidates_indices = get_ligand_candidates(str(sdf_file))
    pdb_file = build_dir / "aligned_amber.pdb"
    u = mda.Universe(str(pdb_file))
    lig_names = u.select_atoms(f"resname {mol}").names
    lig_name_str = " ".join([str(x) for x in lig_names[candidates_indices]])

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
                        .replace("P1A", f"{p1_vmd}")
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
    try:
        run_with_log(f"{vmd} -dispdev text -e prep.tcl", error_match="anchor not found", shell=False, working_dir=build_dir)
    except RuntimeError:
        # fallback: all ligand atoms
        lig_name_str2 = " ".join([str(x) for x in lig_names])
        _write_prep(lig_name_str2)
        run_with_log(f"{vmd} -dispdev text -e prep.tcl", error_match="anchor not found", shell=False, working_dir=build_dir)

    # Verify anchors.txt
    anchor_file = build_dir / "anchors.txt"
    if anchor_file.stat().st_size == 0:
        logger.warning(f"Could not find ligand L1 for {ligand}. Most likely not in binding site.")
        return False

    # Ensure we got 3 ligand anchors
    with open(anchor_file) as f:
        line = f.readline().strip()
    if len(line.split()) < 3:
        os.rename(anchor_file, build_dir / f"anchors-{ligand}.txt")
        logger.warning(f"Could not find ligand L2/L3 anchors for {ligand}. Try reducing min_adis.")
        return False

    os.rename(anchor_file, build_dir / f"anchors-{ligand}.txt")
    return True


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

    solv_shell   = sim.solv_shell
    l1_x, l1_y, l1_z = sim.l1_x, sim.l1_y, sim.l1_z
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol
    l1_range    = sim.l1_range
    max_adis    = sim.max_adis
    min_adis    = sim.min_adis
    buffer_z    = sim.buffer_z
    hmr         = sim.hmr
    membrane_builder = sim.membrane_simulation

    workdir   = ctx.build_dir
    workdir.mkdir(parents=True, exist_ok=True)
    child_root = ctx.working_dir                         # .../simulations/<LIG>/fe/...
    sys_root   = ctx.system_root                         # .../work/<system>
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

    # 2) Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf"):
        src = ff_dir / f"{mol}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_z] Missing ligand FF file: {src}")
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

    if membrane_builder:
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
    if not buffer_z or buffer_z < 15:
        buffer_z = 25
    sdr_dist = get_sdr_dist(str(_p("complex.pdb")), lig_resname=mol, buffer_z=buffer_z, extra_buffer=5)

    # 9) align & pdb4amber
    run_with_log(f"{vmd} -dispdev text -e measure-fit.tcl", shell=False, working_dir=workdir)
    with open(_p("aligned.pdb")) as fin, open(_p("aligned-clean.pdb"), "wt") as fout:
        for ln in fin:
            if len(ln.split()) > 3: fout.write(ln)
    run_with_log(f"pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y", working_dir=workdir)

    # optional lipid resid fix post-amber
    if membrane_builder:
        u = mda.Universe(_p("aligned_amber.pdb"))
        non_water_ag = u.select_atoms('not resname WAT Na+ Cl- K+')
        non_water_ag.residues.resids = final_resids

        u.atoms.write(_p("aligned_amber.pdb"))

    # 10) ligand candidates for Boresch
    sdf_file = _p(f"{mol}.sdf")
    candidates_indices = get_ligand_candidates(str(sdf_file))
    lig_names = u.select_atoms(f"resname {mol}").names
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
    work: Path = ctx.working_dir
    
    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve locations
    ligand = ctx.ligand
    mol = ctx.residue_name
    sys_root   = ctx.system_root                         # .../work/<system>
    all_ligand_folder = sys_root / "all-ligands"
    ff_dir = sys_root / "simulations" / ligand / "params"  # .../work/<system>/simulations/<LIG>/params

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


    ante_mol = mda.Universe(str(build_dir / f"{mol}.mol2"))
    lig_u = mda.Universe(str(build_dir / f"{ligand}.pdb"))
    lig_u.atoms.names = ante_mol.atoms.names
    lig_u.atoms.residues.resnames = mol
    lig_u.atoms.write(str(build_dir / f"{mol}.pdb"))


    mol = ctx.residue_name

    # Copy ligand FF files from fe/ff → build_dir
    for ext in (".mol2", ".sdf"):
        src = ff_dir / f"{mol}{ext}"
        if not src.exists():
            raise FileNotFoundError(f"[build_complex_y] Missing ligand FF file: {src}")
        shutil.copy2(src, build_dir / src.name)

    return True
