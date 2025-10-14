# fe_ops.py
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
import glob
import shutil
import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger

# Reuse your existing helpers/constants (assumed available in your tree)
from batter.utils import run_with_log
from batter._internal import scripts
from batter.utils import vmd as VMD_BIN   # if you prefer, pass vmd path explicitly
# from batter._internal.const import cpptraj  # prefer passing cpptraj explicitly

# Your geometry helpers (must exist as before)
from batter._internal.ops.geometry import get_sdr_dist, get_ligand_candidates


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def build_complex_fe(
    *,
    build_dir: Path,
    pose: str,
    equil_dir: Path,          # path to the ligand’s equil folder (e.g. <...>/equil/<pose>)
    cpptraj_bin: str,         # e.g. "cpptraj" or resolved path
    vmd_bin: Optional[str] = None,  # if None, uses batter.utils.vmd
    membrane_builder: bool,
    lipid_mol: List[str],
    other_mol: List[str],
    hmr: str,                 # "yes" or "no"
    # simulation config fields used by legacy code:
    solv_shell: float,
    l1_x: float,
    l1_y: float,
    l1_z: float,
    l1_range: float,
    max_adis: float,
    min_adis: float,
    buffer_z: float,
    # paths to template trees the legacy code copies from:
    build_files_orig: Path,   # directory containing original build files (legacy: build_files_orig)
) -> Tuple[bool, Optional[str], str, float]:
    """
    Run the legacy FE 'build_complex' stage as a standalone op.

    Returns
    -------
    (ok, code, mol, sdr_dist)
      ok   : True if anchors found and files written; False if pruned/failed.
      code : Optional short code for anchor failures ('anch1' or 'anch2'); None on success.
      mol  : Detected ligand residue name (3-letter/legacy style).
      sdr_dist : computed SDR distance (float).
    """
    vmd_exec = vmd_bin or VMD_BIN

    build_dir = Path(build_dir).resolve()
    equil_dir = Path(equil_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    with _pushd(build_dir):
        # 1) seed the working folder with stage templates
        shutil.copytree(build_files_orig, ".", dirs_exist_ok=True)

        # 2) pull equilibrium artifacts for this pose
        #    expects files: {pose}.pdb, representative.rst7/pdb, renum files, full*.prmtop, vac*
        def cp(src: Path, dst: Path | None = None):
            dst = dst or Path(".")
            run_with_log(f"cp {src} {dst}")

        cp(equil_dir / f"{pose}.pdb", Path("./"))
        cp(equil_dir / "representative.rst7", Path("./"))
        cp(equil_dir / "representative.pdb", Path("./aligned-nc.pdb"))
        cp(equil_dir / "build_amber_renum.txt", Path("./"))
        cp(equil_dir / "protein_renum.txt", Path("./"))

        if not Path("protein_renum.txt").exists():
            raise FileNotFoundError(f"protein_renum.txt not found in {build_dir}")

        for f in glob.glob(str(equil_dir / "full*.prmtop")):
            cp(Path(f), Path("./"))
        for f in glob.glob(str(equil_dir / "vac*")):
            cp(Path(f), Path("./"))

        # Detect ligand resname (first residue in {pose}.pdb)
        mol = mda.Universe(f"{pose}.pdb").residues[0].resname
        run_with_log(f"cp {equil_dir}/{mol.lower()}.sdf ./")
        run_with_log(f"cp {equil_dir}/{mol.lower()}.mol2 ./")
        run_with_log(f"cp {equil_dir}/{mol.lower()}.pdb ./")

        # 3) write receptor-only PDB from representative state
        prmtop_f = "full.hmr.prmtop" if hmr == "yes" else "full.prmtop"
        run_with_log(f"{cpptraj_bin} -p {prmtop_f} -y representative.rst7 -x rec_file.pdb")

        # Fix chain IDs using build_amber_renum.txt
        renum = pd.read_csv(
            "build_amber_renum.txt",
            sep=r"\s+",
            header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
        )
        u = mda.Universe("rec_file.pdb")
        for residue in u.select_atoms("protein").residues:
            resid_str = residue.resid
            residue.atoms.chainIDs = renum.query("old_resid == @resid_str").old_chain.values[0]

        if membrane_builder:
            # Re-pack residue IDs for lipids to ensure continuity
            non_water_ag = u.select_atoms("not resname WAT Na+ Cl- K+ ANC")
            revised_resids = []
            resid_counter = 1
            prev_resid = 0
            for _, row in renum.iterrows():
                if row["old_resname"] in ["WAT", "Na+", "Cl-", "K+"]:
                    continue
                if row["old_resid"] != prev_resid or row["old_resname"] not in lipid_mol:
                    revised_resids.append(resid_counter)
                    resid_counter += 1
                else:
                    revised_resids.append(resid_counter - 1)
                prev_resid = row["old_resid"]

            revised_resids = np.array(revised_resids)
            total_res = non_water_ag.residues.n_residues
            final_resids = np.zeros(total_res, dtype=int)
            final_resids[: len(revised_resids)] = revised_resids
            next_resnum = revised_resids[-1] + 1
            final_resids[len(revised_resids) :] = np.arange(
                next_resnum, total_res - len(revised_resids) + next_resnum
            )
            non_water_ag.residues.resids = final_resids

        u.atoms.write("rec_file.pdb")
        run_with_log("cp rec_file.pdb equil-reference.pdb")

        # 4) split raw complex (VMD TCL)
        with open("split-ini.tcl", "rt") as fin, open("split.tcl", "wt") as fout:
            other_mol_vmd = " ".join(other_mol) if other_mol else "XXX"
            lipid_mol_vmd = " ".join(lipid_mol) if lipid_mol else "XXX"
            for line in fin:
                fout.write(
                    line.replace("SHLL", f"{solv_shell:4.2f}")
                    .replace("OTHRS", str(other_mol_vmd))
                    .replace("LIPIDS", str(lipid_mol_vmd))
                    .replace("mmm", mol.lower())
                    .replace("MMM", f"'{mol}'")
                )
        run_with_log(f"{vmd_exec} -dispdev text -e split.tcl", shell=False)

        # 5) merge + clean complex
        with open("complex-merge.pdb", "w") as out:
            for fname in ["dummy.pdb", "protein.pdb", f"{mol.lower()}.pdb", "lipids.pdb", "others.pdb", "crystalwat.pdb"]:
                if Path(fname).exists():
                    out.write(Path(fname).read_text())
        with open("complex-merge.pdb") as fin, open("complex.pdb", "w") as fout:
            for line in fin:
                if "CRYST1" not in line and "CONECT" not in line and "END" not in line:
                    fout.write(line)

        # 6) read anchors & protein size from equil-{mol}.pdb
        with open(equil_dir / f"equil-{mol.lower()}.pdb", "r") as f:
            data = f.readline().split()
            P1, P2, P3 = data[2].strip(), data[3].strip(), data[4].strip()
            first_res, recep_last = data[8].strip(), data[9].strip()
        p1_resid = P1.split("@")[0][1:]
        p1_atom = P1.split("@")[1]
        rec_res = int(recep_last) + 1
        p1_vmd = p1_resid

        # 7) align to reference (use input as reference for membranes)
        run_with_log(f"{vmd_exec} -dispdev text -e measure-fit.tcl", shell=False)

        # 8) AMBERize + (optionally) reapply lipid resids
        with open("aligned.pdb", "r") as fin, open("aligned-clean.pdb", "w") as fout:
            for line in fin:
                if len(line.split()) > 3:
                    fout.write(line)
        run_with_log("pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y")
        if membrane_builder:
            u = mda.Universe("aligned_amber.pdb")
            non_water_ag = u.select_atoms("not resname WAT Na+ Cl- K+")
            non_water_ag.residues.resids = final_resids  # from earlier
            u.atoms.write("aligned_amber.pdb")

        # 9) compute SDR distance (default buffer_z=25 if 0)
        if buffer_z == 0:
            buffer_z = 25.0
        sdr_dist = get_sdr_dist("complex.pdb", lig_resname=mol.lower(), buffer_z=buffer_z, extra_buffer=5)
        logger.debug(f"SDR distance: {sdr_dist:.02f}")

        # 10) VMD prep to pick ligand anchors (candidate atoms)
        candidates_indices = get_ligand_candidates(f"{mol.lower()}.sdf")
        pdb_file = "aligned_amber.pdb"
        u2 = mda.Universe(pdb_file)
        lig_names = u2.select_atoms(f"resname {mol.lower()}").names
        lig_name_str = " ".join(str(i) for i in lig_names[candidates_indices])

        def _render_prep_tcl(ligand_name_str: str) -> None:
            lipid_mol_vmd = " ".join(lipid_mol) if lipid_mol else "XXX"
            other_mol_vmd = " ".join(other_mol) if other_mol else "XXX"
            with open("prep-ini.tcl", "rt") as fin, open("prep.tcl", "wt") as fout:
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
                        .replace("OTHRS", str(other_mol_vmd))
                        .replace("LIPIDS", str(lipid_mol_vmd))
                        .replace("LIGANDNAME", ligand_name_str)
                    )

        try:
            _render_prep_tcl(lig_name_str)
            run_with_log(f"{vmd_exec} -dispdev text -e prep.tcl", error_match="anchor not found", shell=False)
        except RuntimeError:
            logger.info(
                "Failed to find anchors with selected candidates; retrying with all ligand atoms."
            )
            lig_name_str = " ".join(str(i) for i in lig_names)
            _render_prep_tcl(lig_name_str)
            run_with_log(f"{vmd_exec} -dispdev text -e prep.tcl", error_match="anchor not found", shell=False)

        # 11) validate anchors file
        anchor_file = Path("anchors.txt")
        if anchor_file.stat().st_size == 0:
            return (False, "anch1", mol, sdr_dist)
        with anchor_file.open() as f:
            for line in f:
                if len(line.split()) < 3:
                    anchor_file.rename(f"anchors-{pose}.txt")
                    return (False, "anch2", mol, sdr_dist)
        anchor_file.rename(f"anchors-{pose}.txt")

        # 12) read ligand anchors and write header into fe-<mol>.pdb
        with open(equil_dir / f"equil-{mol.lower()}.pdb", "r") as f:
            data = f.readline().split()
            P1, P2, P3 = data[2].strip(), data[3].strip(), data[4].strip()
            first_res, recep_last = data[8].strip(), data[9].strip()

        lig_resid = str(int(recep_last) + 2)
        with open(f"anchors-{pose}.txt", "r") as f:
            for line in f:
                splitdata = line.split()
                L1 = ":" + lig_resid + "@" + splitdata[0]
                L2 = ":" + lig_resid + "@" + splitdata[1]
                L3 = ":" + lig_resid + "@" + splitdata[2]

        with open(f"fe-{mol.lower()}.pdb", "r") as fin:
            data_lines = fin.read().splitlines(True)
        with open(f"fe-{mol.lower()}.pdb", "w") as fout:
            fout.write(
                "%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n"
                % ("REMARK A", P1, P2, P3, L1, L2, L3, first_res, recep_last)
            )
            fout.writelines(data_lines[1:])

        logger.info("fe build_complex finished for pose={} (mol={})", pose, mol)
        return (True, None, mol, float(sdr_dist))