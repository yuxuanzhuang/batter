"""Analyse equilibration trajectories to determine binding status."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List
import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
from loguru import logger

from batter.analysis.sim_validation import SimValidator
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils import cpptraj, run_with_log


def _paths(root: Path) -> dict[str, Path]:
    """Return commonly accessed equilibration paths under ``root``."""
    eq = root / "equil"
    return {
        "equil_dir": eq,
        "finished": eq / "FINISHED",
        "failed": eq / "FAILED",
        "unbound": eq / "UNBOUND",
        "rep_pdb": eq / "representative.pdb",
        "rep_rst": eq / "representative.rst7",
        "build_files": eq / "q_build_files",
        "prot_renum": eq / "q_build_files" / "protein_renum.txt",
        "full_pdb": eq / "full.pdb",
    }


def _sort_md_paths(paths: List[Path]) -> List[Path]:
    """Sort md-* files by their integer index (md-01, md01, etc.)."""

    def _idx(p: Path) -> int:
        stem = p.stem  # md-01 or md01
        for token in stem.split("-"):
            if token.isdigit():
                return int(token)
        try:
            return int("".join(filter(str.isdigit, stem)))
        except Exception:
            return -1

    return sorted(paths, key=_idx)


def _cpptraj_export_rep(
    rep_idx: int, prmtop: str, trajs: List[Path], workdir: Path
) -> None:
    """Export a representative frame to PDB/RST7 using cpptraj."""
    if not trajs:
        raise FileNotFoundError(
            "No md-*.nc trajectories found for equilibration analysis."
        )

    lines: List[str] = [f"parm {prmtop}"]
    for t in trajs:
        rel = t.name  # use local names; workdir is traj location
        lines.append(f"trajin {rel}")
    # cpptraj is 1-indexed for frames
    one_based_frame = rep_idx + 1
    lines.append(f"trajout representative.pdb pdb onlyframes {one_based_frame}")
    lines.append(f"trajout representative.rst7 restart onlyframes {one_based_frame}")

    script = "\n".join(lines) + "\n"
    (workdir / "rep.in").write_text(script)

    run_with_log(f"{cpptraj} -i rep.in", working_dir=workdir)


def equil_analysis_handler(
    step: Step, system: SimSystem, params: Dict[str, Any]
) -> ExecResult:
    """Inspect equilibration trajectories and generate representative files.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system providing context and filesystem roots.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Artifacts describing the binding state (representative structures or
        ``UNBOUND`` sentinel).

    Raises
    ------
    FileNotFoundError
        When required inputs are missing.
    ValueError
        When the payload lacks a simulation configuration.
    """
    p = _paths(system.root)
    lig = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    logger.debug(
        f"Running equil_analysis_handler for ligand {lig} (residue {residue_name})"
    )

    rep_rel = p["rep_pdb"].relative_to(system.root).as_posix()
    unbound_rel = p["unbound"].relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "equil_analysis",
        required=[[rep_rel], [unbound_rel]],
        success=[[rep_rel], [unbound_rel]],
    )

    payload = StepPayload.model_validate(params)
    sim = payload.sim
    if sim is None:
        raise ValueError(
            "[equil_analysis] Missing simulation configuration in payload."
        )
    threshold = float(
        payload.get("unbound_threshold", getattr(sim, "unbound_threshold", 8.0))
    )
    hmr = str(sim.hmr)

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
        logger.debug(
            f"[equil_check:{lig}] representative.* already present; skipping analysis"
        )
        return ExecResult(
            job_ids=[],
            artifacts={
                "representative_pdb": p["rep_pdb"],
                "representative_rst7": p["rep_rst"],
            },
        )

    if not p["full_pdb"].exists():
        raise FileNotFoundError(f"[equil_check:{lig}] missing {p['full_pdb']}")

    # Run validation
    prmtop = "full.hmr.prmtop" if hmr == "yes" else "full.prmtop"

    try:
        # Build trajectory list from completed equil segments
        trajs = _sort_md_paths(list(p["equil_dir"].glob("md-*.nc")))
        trajs = [t for t in trajs if t.exists()]
        # make sure each t is larger than 1 KB
        trajs = [t for t in trajs if t.stat().st_size > 1024]
        if not trajs:
            raise FileNotFoundError(
                f"[equil_check:{lig}] no md-*.nc trajectories found for analysis"
            )
        u = mda.Universe(str(p["full_pdb"]), [str(t) for t in trajs])
        sim_val = SimValidator(u, ligand=residue_name, directory=p["equil_dir"])
        sim_val.plot_analysis(savefig=True)

        # bound vs unbound
        ligand_bs_last = float(np.asarray(sim_val.results["ligand_bs"][-1]).item())
        if ligand_bs_last > threshold:
            logger.warning(
                f"[equil_check:{lig}] UNBOUND (ligand_bs={ligand_bs_last:.2f} Å) > {threshold:.2f} Å"
            )
            p["unbound"].write_text(f"UNBOUND with ligand_bs = {ligand_bs_last:.3f}\n")
            return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})
        rep_idx = int(sim_val.find_representative_snapshot())
        # pick representative frame and export using cpptraj
        _cpptraj_export_rep(rep_idx, prmtop, trajs, p["equil_dir"])

    # if traj doesn't exist
    # use the last frame as representative
    except Exception as e:
        logger.warning(f"[equil_check:{lig}] error during simulation validation: {e}")
        # copy last frame as representative
        last_rst = p["equil_dir"] / "md-current.rst7"
        if os.path.exists(last_rst):
            shutil.copyfile(last_rst, p["rep_rst"])
        else:
            raise FileNotFoundError(
                f"[equil_check:{lig}] no md-current.rst7 found for fallback representative"
            )
        # convert to pdb
        run_with_log(
            f"{cpptraj} -p {prmtop} -y representative.rst7 -x representative.pdb",
            working_dir=p["equil_dir"],
        )

    # remap protein residue IDs back to original (protein_renum.txt)
    renum_txt = p["prot_renum"]
    if not renum_txt.exists():
        raise FileNotFoundError(
            f"[equil_check:{lig}] missing {renum_txt}; cannot renumber residues"
        )
    else:
        renum = pd.read_csv(
            renum_txt,
            sep=r"\s+",
            header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
        )
        uu = mda.Universe(str(p["rep_pdb"]))
        uu.select_atoms("protein").residues.resids = renum["old_resid"].values
        uu.atoms.write(str(p["rep_pdb"]))

    logger.debug(f"[equil_check:{lig}] representative frame written")
    assert p["rep_pdb"].exists() and p["rep_rst"].exists()
    return ExecResult(
        job_ids=[],
        artifacts={
            "representative_pdb": p["rep_pdb"],
            "representative_rst7": p["rep_rst"],
        },
    )
