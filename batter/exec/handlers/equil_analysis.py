from __future__ import annotations
import os
import subprocess
from pathlib import Path
import shutil
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import MDAnalysis as mda
from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem

from batter.analysis.sim_validation import SimValidator
from batter.utils import run_with_log, cpptraj


def _paths(root: Path) -> dict[str, Path]:
    eq = root / "equil"
    return {
        "equil_dir": eq,
        "finished": eq / "FINISHED",
        "failed":   eq / "FAILED",
        "unbound":  eq / "UNBOUND",
        "rep_pdb":  eq / "representative.pdb",
        "rep_rst":  eq / "representative.rst7",
        "build_files": eq / "q_build_files",
        "prot_renum":  eq / "q_build_files" / "protein_renum.txt",
        "full_pdb":    eq / "full.pdb",
    }


def _cpptraj_export_rep(rep_idx: int, prmtop: str,
                        start_eq: int, end_eq: int,
                        workdir: Path) -> None:
    """
    Write the representative frame to PDB and RST7 using cpptraj, with logged execution.

    Parameters
    ----------
    rep_idx : int
        0-based index of the representative frame (cpptraj is 1-based; we add 1).
    prmtop : str
        Path to the topology file (.prmtop).
    start_eq : int
        Index of the first equilibration segment to read (skipping earlier segments where
        release_eq != 0). This is used to determine which md-XX.nc files to read.
    end_eq : int
        The last equilibration segments named md-XX.nc.
    workdir : Path
        Directory containing the md-XX.nc trajectories; rep.in will be written here.
    """
    # Build cpptraj input
    lines: List[str] = [f"parm {prmtop}"]
    for i in range(start_eq, end_eq):  # skip the first equil traj (assumes md-00.nc exists and is skipped)
        lines.append(f"trajin md-{i:02d}.nc")
    # cpptraj is 1-indexed for frames
    one_based_frame = rep_idx + 1
    lines.append(f"trajout representative.pdb pdb onlyframes {one_based_frame}")
    lines.append(f"trajout representative.rst7 restart onlyframes {one_based_frame}")

    script = "\n".join(lines) + "\n"
    (workdir / "rep.in").write_text(script)

    run_with_log(f"{cpptraj} -i rep.in", working_dir=workdir)


def equil_analysis_handler(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Post-equilibration binding sanity check.
    Produces:
      - representative.pdb / representative.rst7 if bound
      - UNBOUND sentinel if unbound
    Prunes nothing by itself; the orchestrator can skip UNBOUND ligands downstream.
    """
    p = _paths(system.root)
    lig = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    logger.debug(f"Running equil_analysis_handler for ligand {lig} (residue {residue_name})")

    sim = params.get("sim")
    threshold = float(params.get("unbound_threshold", 8.0))
    release_eq = sim.get("release_eq")
    n_eq = len(release_eq)
    hmr = str(sim.get("hmr", "yes"))  # "yes"/"no"

    # hard requirements
    if not p["finished"].exists():
        if p["failed"].exists():
            raise FileNotFoundError(f"[equil_check:{lig}] equil FAILED; cannot proceed")
        raise FileNotFoundError(f"[equil_check:{lig}] equil not FINISHED")

    if p["unbound"].exists():
        logger.warning(f"[equil_check:{lig}] previously marked UNBOUND — keeping as is")
        return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

    # if representative already exists, we're done (idempotent)
    if p["rep_pdb"].exists() and p["rep_rst"].exists():
        logger.debug(f"[equil_check:{lig}] representative.* already present; skipping analysis")
        return ExecResult(job_ids=[], artifacts={"representative_pdb": p["rep_pdb"], "representative_rst7": p["rep_rst"]})

    if not p["full_pdb"].exists():
        raise FileNotFoundError(f"[equil_check:{lig}] missing {p['full_pdb']}")

    # Build trajectory list (skip the eq runs where release_eq is not 0
    start = int(np.where(np.array(release_eq, float) == 0.0)[0][0]) if np.any(np.array(release_eq) == 0.0) else None

    trajs = [p["equil_dir"] / f"md-{i:02d}.nc" for i in range(start, n_eq)]
    trajs = [t for t in trajs if t.exists()]

    # Run validation
    prmtop = "full.hmr.prmtop" if hmr == "yes" else "full.prmtop"

    try:
        u = mda.Universe(str(p["full_pdb"]), [str(t) for t in trajs])
        sim_val = SimValidator(u, ligand=residue_name, directory=p["equil_dir"])
        sim_val.plot_analysis(savefig=True)

        # bound vs unbound
        ligand_bs_last = float(np.asarray(sim_val.results["ligand_bs"][-1]).item())
        if ligand_bs_last > threshold:
            logger.warning(f"[equil_check:{lig}] UNBOUND (ligand_bs={ligand_bs_last:.2f} Å) > {threshold:.2f} Å")
            p["unbound"].write_text(f"UNBOUND with ligand_bs = {ligand_bs_last:.3f}\n")
            return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})
        rep_idx = int(sim_val.find_representative_snapshot())
        # pick representative frame and export using cpptraj
        _cpptraj_export_rep(rep_idx, prmtop, start, n_eq, p["equil_dir"])

    # if traj doesn't exist
    # use the last frame as representative
    except Exception as e:
        logger.warning(f"[equil_check:{lig}] error during simulation validation: {e}")
        # copy last frame as representative
        shutil.copyfile(p["equil_dir"] / f"md{n_eq-1:02d}.rst7", p["rep_rst"])
        # convert to pdb
        run_with_log(f'{cpptraj} -p {prmtop} -y representative.rst7 -x representative.pdb', working_dir=p["equil_dir"])

    # remap protein residue IDs back to original (protein_renum.txt)
    renum_txt = p["prot_renum"]
    if not renum_txt.exists():
        raise FileNotFoundError(f"[equil_check:{lig}] missing {renum_txt}; cannot renumber residues")
    else:
        renum = pd.read_csv(
            renum_txt, sep=r"\s+", header=None,
            names=["old_resname","old_chain","old_resid","new_resname","new_resid"]
        )
        uu = mda.Universe(str(p["rep_pdb"]))
        uu.select_atoms("protein").residues.resids = renum["old_resid"].values
        uu.atoms.write(str(p["rep_pdb"]))

    logger.debug(f"[equil_check:{lig}] representative frame written")
    assert p["rep_pdb"].exists() and p["rep_rst"].exists()
    return ExecResult(job_ids=[], artifacts={"representative_pdb": p["rep_pdb"], "representative_rst7": p["rep_rst"]})