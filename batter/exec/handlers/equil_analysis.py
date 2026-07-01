"""Analyse equilibration trajectories to determine binding status."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import MDAnalysis as mda
import numpy as np
import pandas as pd
from loguru import logger
from MDAnalysis.analysis import align

from batter.analysis.sim_validation import (
    STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
    SimValidator,
)
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
        "stable_boresch_distance": eq / "stable_boresch_distance.json",
        "build_files": eq / "q_build_files",
        "prot_renum": eq / "q_build_files" / "protein_renum.txt",
        "full_pdb": eq / "full.pdb",
    }


def _stable_boresch_distance_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    try:
        schema_version = int(data.get("schema_version", 0))
    except Exception:
        return False
    return schema_version >= STABLE_BORESCH_DISTANCE_SCHEMA_VERSION


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


def _ligand_candidate_atom_names(
    *,
    system_root: Path,
    residue_name: str | None,
    ligand_label: str | None,
    universe: mda.Universe,
) -> list[str] | None:
    if not residue_name:
        return None
    sdf_file = system_root / "params" / f"{residue_name}.sdf"
    if not sdf_file.exists():
        return None
    lig_atoms = universe.select_atoms(f"resname {residue_name}")
    if lig_atoms.n_atoms == 0:
        return None
    try:
        from batter._internal.ops.build_complex import (
            _candidate_ligand_atom_name_string,
        )

        names = _candidate_ligand_atom_name_string(
            sdf_file,
            lig_atoms,
            ligand_label=ligand_label or residue_name,
            stage="equil-analysis",
        )
    except Exception as exc:
        logger.warning(
            "[equil_check:{}] Could not derive ligand anchor candidate names from {}: {}. "
            "Using all ligand heavy atoms for stable-distance search.",
            ligand_label,
            sdf_file,
            exc,
        )
        return None
    return [name for name in names.split() if name]


def _stable_distance_validator(
    *,
    universe: mda.Universe,
    residue_name: str | None,
    directory: Path,
    protein_anchor_masks: list[str],
) -> SimValidator:
    validator = SimValidator.__new__(SimValidator)
    validator.universe = universe
    validator.workdir = directory.resolve()
    validator.ligand = residue_name
    validator.protein_anchor_masks = protein_anchor_masks
    validator.results = {}
    return validator


def _write_stable_boresch_distance(
    *,
    stable_path: Path,
    system_root: Path,
    sim: Any,
    sim_val: SimValidator,
    ligand_label: str | None,
    residue_name: str | None,
    universe: mda.Universe,
    tail_fraction: float,
    mode: str,
) -> dict[str, Any]:
    ligand_candidate_names = _ligand_candidate_atom_names(
        system_root=system_root,
        residue_name=residue_name,
        ligand_label=ligand_label,
        universe=universe,
    )
    stable_record = sim_val.find_stable_boresch_distance(
        tail_fraction=tail_fraction,
        min_distance=float(getattr(sim, "min_adis", None) or 3.0),
        max_distance=float(getattr(sim, "max_adis", None) or 7.0),
        ligand_atom_names=ligand_candidate_names,
    )
    stable_record["mode"] = mode
    stable_record["usable"] = True
    stable_path.write_text(json.dumps(stable_record, indent=2) + "\n")
    logger.info(
        "[equil_check:{}] stable Boresch pair: {} to {} "
        "(mean={:.2f} Å, std={:.2f} Å, frames={} from frame {}, "
        "ranked_pairs={}, mode={}).",
        ligand_label,
        stable_record["protein"]["mask"],
        stable_record["ligand"]["mask"],
        stable_record["distance"]["mean"],
        stable_record["distance"]["std"],
        stable_record["n_frames"],
        stable_record["analysis_start_frame"],
        len(stable_record.get("ranked_pairs") or []),
        mode,
    )
    return stable_record


def _write_unusable_stable_boresch_distance(
    *,
    stable_path: Path,
    mode: str,
    reason: Exception,
) -> None:
    stable_record = {
        "schema_version": STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
        "source": "equil_analysis",
        "mode": mode,
        "usable": False,
        "reason": str(reason),
    }
    stable_path.write_text(json.dumps(stable_record, indent=2) + "\n")


_EQUIL_ANALYSIS_ARTIFACT_FILES = (
    "representative.pdb",
    "representative.rst7",
    "representative_complex.pdb",
    "representative_pose.pdb",
    "initial_pose.pdb",
    "equilibration_analysis_results.npz",
    "stable_boresch_distance.json",
    "simulation_analysis.png",
    "dihed_hist.png",
)


def _copy_equil_analysis_artifacts(equil_dir: Path) -> None:
    artifacts_dir = equil_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in _EQUIL_ANALYSIS_ARTIFACT_FILES:
        src = equil_dir / name
        if src.exists():
            shutil.copy2(src, artifacts_dir / name)


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
    sys_params = payload.sys_params
    user_anchor_atoms = list(
        (sys_params.get("anchor_atoms", []) if sys_params is not None else []) or []
    )
    threshold = float(
        payload.get("unbound_threshold", getattr(sim, "unbound_threshold", 8.0))
    )
    hmr = str(sim.hmr)
    prmtop = "full.hmr.prmtop" if hmr == "yes" else "full.prmtop"

    # hard requirements
    if not p["finished"].exists():
        if p["failed"].exists():
            raise FileNotFoundError(f"[equil_check:{lig}] equil FAILED; cannot proceed")
        raise FileNotFoundError(f"[equil_check:{lig}] equil not FINISHED")

    if p["unbound"].exists():
        logger.warning(f"[equil_check:{lig}] previously marked UNBOUND — keeping as is")
        return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

    # if representative already exists, we're done (idempotent). For auto-anchor
    # runs, still allow a later invocation to backfill the stable-distance JSON.
    stable_distance_needed = not user_anchor_atoms
    if (
        stable_distance_needed
        and p["stable_boresch_distance"].exists()
        and not _stable_boresch_distance_current(p["stable_boresch_distance"])
    ):
        logger.debug(
            "[equil_check:{}] stable Boresch distance JSON is stale; "
            "removing it so the current selector can regenerate it.",
            lig,
        )
        try:
            p["stable_boresch_distance"].unlink()
        except OSError as exc:
            logger.warning(
                "[equil_check:{}] Could not remove stale stable Boresch distance "
                "JSON {}: {}",
                lig,
                p["stable_boresch_distance"],
                exc,
            )
    if (
        p["rep_pdb"].exists()
        and p["rep_rst"].exists()
        and (
            not stable_distance_needed
            or _stable_boresch_distance_current(p["stable_boresch_distance"])
        )
    ):
        logger.debug(
            f"[equil_check:{lig}] representative.* already present; skipping analysis"
        )
        artifacts = {
            "representative_pdb": p["rep_pdb"],
            "representative_rst7": p["rep_rst"],
        }
        if p["stable_boresch_distance"].exists():
            artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
        return ExecResult(job_ids=[], artifacts=artifacts)

    if not p["full_pdb"].exists():
        if p["rep_pdb"].exists() and p["rep_rst"].exists():
            logger.warning(
                f"[equil_check:{lig}] missing {p['full_pdb']}; cannot backfill "
                "stable Boresch distance, keeping existing representative.*"
            )
            artifacts = {
                "representative_pdb": p["rep_pdb"],
                "representative_rst7": p["rep_rst"],
            }
            if p["stable_boresch_distance"].exists():
                artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
            return ExecResult(job_ids=[], artifacts=artifacts)
        raise FileNotFoundError(f"[equil_check:{lig}] missing {p['full_pdb']}")

    eq_steps = int(getattr(sim, "eq_steps", 0) or 0)
    if eq_steps == 0:
        eqnpt_appear = p["equil_dir"] / "eqnpt_appear.rst7"
        if not eqnpt_appear.exists():
            raise FileNotFoundError(
                f"[equil_check:{lig}] eq_steps=0 but missing {eqnpt_appear}"
            )
        shutil.copyfile(eqnpt_appear, p["rep_rst"])
        run_with_log(
            f"{cpptraj} -p {prmtop} -y representative.rst7 -x representative.pdb",
            working_dir=p["equil_dir"],
        )
        logger.debug(
            f"[equil_check:{lig}] eq_steps=0; copied {eqnpt_appear.name} as representative"
        )
        if user_anchor_atoms:
            logger.debug(
                "[equil_check:{}] explicit create.anchor_atoms were provided; "
                "skipping stable Boresch distance auto-anchor override.",
                lig,
            )
        else:
            try:
                u_static = mda.Universe(str(p["rep_pdb"]))
                anchor_masks = [
                    str(getattr(sim, "p1", "") or "").strip(),
                    str(getattr(sim, "p2", "") or "").strip(),
                    str(getattr(sim, "p3", "") or "").strip(),
                ]
                stable_val = _stable_distance_validator(
                    universe=u_static,
                    residue_name=residue_name,
                    directory=p["equil_dir"],
                    protein_anchor_masks=anchor_masks,
                )
                _write_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    system_root=system.root,
                    sim=sim,
                    sim_val=stable_val,
                    ligand_label=lig,
                    residue_name=residue_name,
                    universe=u_static,
                    tail_fraction=1.0,
                    mode="single_frame_no_equil",
                )
            except Exception as exc:
                _write_unusable_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    mode="single_frame_no_equil",
                    reason=exc,
                )
                logger.warning(
                    "[equil_check:{}] Could not identify a single-frame "
                    "protein-ligand distance for automatic Boresch anchor "
                    "refinement: {}",
                    lig,
                    exc,
                )
        _copy_equil_analysis_artifacts(p["equil_dir"])
        # Skip trajectory-based validation/analysis when no equilibration steps ran.
        artifacts = {
            "representative_pdb": p["rep_pdb"],
            "representative_rst7": p["rep_rst"],
        }
        if p["stable_boresch_distance"].exists():
            artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
        return ExecResult(job_ids=[], artifacts=artifacts)

    # Run validation

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
        anchor_masks = [
            str(getattr(sim, "p1", "") or "").strip(),
            str(getattr(sim, "p2", "") or "").strip(),
            str(getattr(sim, "p3", "") or "").strip(),
        ]
        sim_val = SimValidator(
            u,
            ligand=residue_name,
            directory=p["equil_dir"],
            protein_anchor_masks=anchor_masks,
        )
        sim_val.plot_analysis(savefig=True)

        # bound vs unbound
        ligand_bs_last = float(np.asarray(sim_val.results["ligand_bs"][-1]).item())
        if ligand_bs_last > threshold:
            logger.warning(
                f"[equil_check:{lig}] UNBOUND (ligand_bs={ligand_bs_last:.2f} Å) > {threshold:.2f} Å"
            )
            p["unbound"].write_text(f"UNBOUND with ligand_bs = {ligand_bs_last:.3f}\n")
            return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

        if user_anchor_atoms:
            logger.debug(
                "[equil_check:{}] explicit create.anchor_atoms were provided; "
                "skipping stable Boresch distance auto-anchor override.",
                lig,
            )
        else:
            try:
                _write_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    system_root=system.root,
                    sim=sim,
                    sim_val=sim_val,
                    ligand_label=lig,
                    residue_name=residue_name,
                    universe=u,
                    tail_fraction=0.25,
                    mode="trajectory_tail",
                )
            except Exception as exc:
                _write_unusable_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    mode="trajectory_tail",
                    reason=exc,
                )
                logger.warning(
                    "[equil_check:{}] Could not identify a stable protein-ligand "
                    "distance for automatic Boresch anchor refinement: {}",
                    lig,
                    exc,
                )
        rep_idx = int(sim_val.find_representative_snapshot())
        # pick representative frame and export using cpptraj
        _cpptraj_export_rep(rep_idx, prmtop, trajs, p["equil_dir"])
        sim_val.dump_results()

    # if traj doesn't exist
    # use the last frame as representative
    except Exception as e:
        logger.debug(f"[equil_check:{lig}] error during simulation validation: {e}")
        if p["rep_pdb"].exists() and p["rep_rst"].exists():
            logger.warning(
                f"[equil_check:{lig}] keeping existing representative.* after "
                "validation/backfill failure"
            )
        else:
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

    # align representative to initial complex and extract poses
    protein_align = (getattr(sim, "protein_align", None) or "name CA").strip()
    if protein_align and p["rep_pdb"].exists() and p["full_pdb"].exists():
        try:
            aligned_rep_output = p["equil_dir"] / "representative_complex.pdb"
            u_rep = mda.Universe(str(p["rep_pdb"]))
            u_ref = mda.Universe(str(p["full_pdb"]))
            _ = align.alignto(
                mobile=u_rep.atoms,
                reference=u_ref.atoms,
                select=f"({protein_align}) and name CA and not resname NMA ACE",
            )
            u_rep.atoms.write(aligned_rep_output)
            if residue_name:
                u_ref.select_atoms(f"resname {residue_name}").write(
                    p["equil_dir"] / "initial_pose.pdb"
                )
                u_rep.select_atoms(f"resname {residue_name}").write(
                    p["equil_dir"] / "representative_pose.pdb"
                )
        except Exception as exc:
            logger.warning(
                f"[equil_check:{lig}] Failed to align representative complex: {exc}"
            )

    # copy key outputs into equil/artifacts for downstream use
    _copy_equil_analysis_artifacts(p["equil_dir"])

    logger.debug(f"[equil_check:{lig}] representative frame written")
    assert p["rep_pdb"].exists() and p["rep_rst"].exists()
    artifacts = {
        "representative_pdb": p["rep_pdb"],
        "representative_rst7": p["rep_rst"],
    }
    if p["stable_boresch_distance"].exists():
        artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
    return ExecResult(job_ids=[], artifacts=artifacts)
