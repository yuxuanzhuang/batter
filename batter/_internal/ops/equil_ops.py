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

# Context class (provided by your new BaseBuilder layer)
# It should expose: ligand (str), sim (SimulationConfig), working_dir (Path),
# component (str), win (int), etc.
from batter._internal.builders.base import BuildContext

from batter._internal.ops.helpers import (
    get_buffer_z,
    get_sdr_dist,
    get_ligand_candidates,
    select_ions_away_from_complex,
)

from batter._internal.templates import BUILD_FILES_DIR as build_files_orig  # type: ignore
from batter._internal.templates import AMBER_FILES_DIR as amber_files_orig  # type: ignore


# ---------------------------------------------------------------------------
# build_complex
# ---------------------------------------------------------------------------

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
    lipid_mol = getattr(sim, "lipid_mol", [])
    if mol in other_mol or mol in lipid_mol:
        raise ValueError(f"The ligand {mol} cannot be in the other_mol/lipid_mol list: {other_mol} and {lipid_mol}")

    logger.debug(f"[Equil] Building complex for ligand {ligand} with other_mol={other_mol} lipid_mol={lipid_mol}")
    solv_shell = sim.solv_shell
    system_name = sim.system_name

    # Stage directories
    work = ctx.working_dir
    build_dir = work / "q_build_files"
    run_dir = work / "run_files"
    amber_dir = work / "amber_files"
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

    # Run VMD split; note error_match follows your legacy pattern
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
    if sim._membrane_simulation and lipid_mol:
        run_with_log(f"{charmmlipid2amber} -i {build_dir/'lipids.pdb'} -o {build_dir/'lipids_amber.pdb'}")
        u_lip = mda.Universe(str(build_dir / "lipids_amber.pdb"))
        lipid_resnames = list(set(u_lip.residues.resnames))
        logger.debug(f"[Equil] Converted CHARMM lipids to AMBER: {lipid_resnames}")
        lipid_mol = lipid_resnames  # updated list

    # Merge raw complex (protein + ligand + others + (lipids) + crystal waters)
    parts: list[Path] = [build_dir / "protein.pdb", build_dir / f"{mol.lower()}.pdb", build_dir / "others.pdb"]
    if sim._membrane_simulation and lipid_mol:
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
    if sim._membrane_simulation and lipid_mol:
        u_aln = mda.Universe(str(build_dir / "aligned_amber.pdb"))
        u_aln.dimensions = u_original.dimensions
        renum_txt2 = build_dir / "aligned_amber_renum.txt"
        ren2 = pd.read_csv(
            renum_txt2, sep=r"\s+", header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"]
        )
        revised = []
        resid_counter = 1
        prev_resid = 0
        for _, row in ren2.iterrows():
            if row["old_resid"] != prev_resid or row["old_resname"] not in lipid_mol:
                revised.append(resid_counter)
                resid_counter += 1
            else:
                revised.append(resid_counter - 1)
            prev_resid = row["old_resid"]
        u_aln.atoms.residues.resids = revised
        u_aln.atoms.write(str(build_dir / "aligned_amber.pdb"))

    sdf_file = build_dir / f"{mol.lower()}.sdf"
    candidates_indices = get_ligand_candidates(str(sdf_file))
    pdb_file = build_dir / "aligned_amber.pdb"
    u = mda.Universe(str(pdb_file))
    lig_names = u.select_atoms(f"resname {mol.lower()}").names
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
                    line.replace("MMM", f"'{mol}'")
                        .replace("mmm", mol.lower())
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


# ---------------------------------------------------------------------------
# write_sim_files
# ---------------------------------------------------------------------------

def write_sim_files(ctx: BuildContext, *, infe: bool) -> None:
    """
    Port of EquilibrationBuilder._sim_files()

    Writes minimization/NVT/NPT inputs and mdin-XX files based on
    release schedule; fills in temperature, restraint file names, etc.
    """
    sim = ctx.sim
    work = ctx.working_dir

    amber_dir = work / "amber_files"
    os.makedirs(amber_dir, exist_ok=True)

    temperature = sim.temperature
    mol = mda.Universe(str(work / "vac.pdb")).residues.resnames[0] if (work / "vac.pdb").exists() else "lig"
    num_sim = len(sim.release_eq)

    # Parse anchors from disang.rest (first line, as in legacy)
    with open(work / "disang.rest", "r") as f:
        parts = f.readline().split()
        # positions: ... L1 L2 L3 ...
        L1 = parts[6].strip()
        L2 = parts[7].strip()
        L3 = parts[8].strip()

    # Copy AMBER templates from your template dir (amber_files_orig) and substitute
    shutil.copytree(amber_files_orig, amber_dir, dirs_exist_ok=True)

    # Generate template-based files
    def _sub_write(src: Path, dst: Path, repl: dict[str, str]) -> None:
        text = Path(src).read_text()
        for k, v in repl.items():
            text = text.replace(k, v)
        dst.write_text(text)

    # mini.in
    _sub_write(
        Path(amber_files_orig) / "mini.in",
        work / "mini.in",
        {"_lig_name_": mol},
    )

    # eqnvt.in
    _sub_write(
        Path(amber_files_orig) / "eqnvt.in",
        work / "eqnvt.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt0.in (membrane vs water variant)
    eqnpt0_src = Path(amber_files_orig) / ( "eqnpt0.in" if sim._membrane_simulation else "eqnpt0-water.in" )
    _sub_write(
        eqnpt0_src,
        work / "eqnpt0.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt.in
    eqnpt_src = Path(amber_files_orig) / ( "eqnpt.in" if sim._membrane_simulation else "eqnpt-water.in" )
    _sub_write(
        eqnpt_src,
        work / "eqnpt.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # mdin-XX files for gradual release
    steps1 = sim.eq_steps1
    steps2 = sim.eq_steps2
    infe_flag = "1" if infe else "0"

    for i, weight in enumerate(sim.release_eq):
        mdin_src = Path(amber_files_orig) / "mdin-equil"
        dst = work / f"mdin-{i:02d}"
        text = mdin_src.read_text()
        # First stage (i==0) needs irest/ntx reset
        if i == 0:
            text = re.sub(r"^\s*irest\s*=.*$", "  irest = 0,", text, flags=re.MULTILINE)
            text = re.sub(r"^\s*ntx\s*=.*$", "  ntx = 1,", text, flags=re.MULTILINE)

        num_steps = steps2 if weight == 0 else steps1
        text = (text
                .replace("_temperature_", f"{temperature}")
                .replace("_enable_infe_", infe_flag)
                .replace("_lig_name_", mol)
                .replace("_num-steps_", f"{num_steps}")
                .replace("disang_file", f"disang{i:02d}")
                )
        dst.write_text(text)

    logger.debug(f"[Equil] Simulation input files written under {work}")