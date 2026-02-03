"""Utilities for reading and saving FE analysis results stored on disk."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

from loguru import logger

from batter.config.simulation import SimulationConfig
from batter.runtime.fe_repo import FEResultsRepository, FERecord
from batter.systems.core import SimSystem


def parse_results_dat(
    path: Path,
) -> Tuple[float | None, float | None, Dict[str, float]]:
    """Parse ``fe/Results/Results.dat`` and return totals plus per-component values.

    The file is expected to contain rows with ``LABEL FE SE``. ``total`` rows set the
    aggregate dG/SE, while other labels populate ``{label}_fe``/``{label}_se`` entries.
    """
    per_comp: Dict[str, float] = {}
    total_dG, total_se = None, None
    for raw in path.read_text().splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [t for t in ln.replace("\t", " ").split() if t]
        if len(parts) < 3:
            continue
        label, fe_str, se_str = parts[0], parts[1], parts[2]
        try:
            fe = float(fe_str)
            se = float(se_str)
        except ValueError:
            continue
        if label.lower() == "total":
            total_dG, total_se = fe, se
        else:
            per_comp[f"{label}_fe"] = fe
            per_comp[f"{label}_se"] = se
    return total_dG, total_se, per_comp


def fallback_totals_from_json(results_dir: Path) -> Tuple[float | None, float | None]:
    """Look for totals in JSON outputs produced by analysis.

    Attempts ``z_results.json`` first (for REST components), then searches for
    ``*_results.json`` siblings that expose ``fe``/``fe_error`` keys.
    """
    zjson = results_dir / "z_results.json"
    if zjson.exists():
        try:
            data = json.loads(zjson.read_text())
            fe = data.get("fe")
            se = data.get("fe_error")
            if fe is not None and se is not None:
                return float(fe), float(se)
        except Exception:
            pass

    for js in sorted(results_dir.glob("*_results.json")):
        try:
            data = json.loads(js.read_text())
            fe = data.get("fe")
            se = data.get("fe_error")
            if fe is not None and se is not None:
                return float(fe), float(se)
        except Exception:
            continue
    return None, None


def extract_ligand_metadata(
    child: SimSystem, original_map: Dict[str, str] | None = None
) -> tuple[str | None, str | None, str | None]:
    """Gather ligand metadata (canonical SMILES, original name/path) for FE exports.

    Metadata is pulled from the parameterization artifact (``metadata.json``) when
    available and combined with any original ligand-name mapping stored on disk.
    """
    canonical_smiles: str | None = None
    original_name: str | None = child.meta.get("ligand")
    original_path: str | None = None

    param_dirs = child.meta.get("param_dir_dict", {}) or {}
    residue_name = child.meta.get("residue_name")
    param_dir = param_dirs.get(residue_name) if residue_name else None
    if param_dir:
        meta_path = Path(param_dir) / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as exc:  # pragma: no cover - best-effort metadata
                logger.debug(f"Failed to read ligand metadata {meta_path}: {exc}")
                meta = {}
            canonical_smiles = meta.get("canonical_smiles")
            original_path = meta.get("input_path")
            original_name = meta.get("title") or original_name
            aliases = meta.get("aliases") or []
            if aliases and not meta.get("title"):
                original_name = aliases[0]
            else:
                original_name = meta.get("prepared_base") or original_name
    if original_map and child.meta.get("ligand"):
        original_name = original_map.get(child.meta.get("ligand"), original_name)
    return canonical_smiles, original_name, original_path


def save_fe_records(
    *,
    run_dir: Path,
    run_id: str,
    children_all: list[SimSystem],
    sim_cfg_updated: SimulationConfig,
    repo: FEResultsRepository,
    protocol: str,
    original_map: Dict[str, str] | None = None,
    analysis_start_step: int | None = None,
) -> list[tuple[str, str, str]]:
    """Persist FE totals for all ligands to the FE repository, recording failures.

    For each ligand, totals are pulled from ``Results.dat`` (or JSON fallbacks) and
    written to the portable FE repository alongside the raw ``Results/`` artifacts.
    Missing totals or save errors are captured as failures and returned as
    ``(ligand, status, reason)`` tuples.
    """
    failures: list[tuple[str, str, str]] = []
    analysis_start_step_val = analysis_start_step
    if analysis_start_step_val is None:
        analysis_start_step_val = getattr(sim_cfg_updated, "analysis_start_step", None)
    if analysis_start_step_val is not None:
        analysis_start_step_val = int(analysis_start_step_val)
    protein_align = getattr(sim_cfg_updated, "protein_align", None)
    for child in children_all:
        lig_name = child.meta["ligand"]
        mol_name = child.meta["residue_name"]
        results_dir = child.root / "fe" / "Results"
        total_dG, total_se = None, None
        is_rbfe = str(child.meta.get("mode", "")).upper() == "RBFE"

        dat = results_dir / "Results.dat"
        if dat.exists():
            try:
                tdg, tse, _ = parse_results_dat(dat)
                total_dG, total_se = tdg, tse
            except Exception as exc:
                logger.warning(f"[{lig_name}] Failed to parse Results.dat: {exc}")

        if total_dG is None or total_se is None:
            tdg, tse = fallback_totals_from_json(results_dir)
            total_dG = tdg if total_dG is None else total_dG
            total_se = tse if total_se is None else total_se

        if total_dG is None or total_se is None:
            reason = "no_totals_found"
            failures.append((lig_name, "failed", reason))
            canonical_smiles, original_name, original_path = extract_ligand_metadata(
                child, original_map
            )
            repo.record_failure(
                run_id=run_id,
                ligand=lig_name,
                system_name=sim_cfg_updated.system_name,
                temperature=sim_cfg_updated.temperature,
                status="failed",
                reason=reason,
                canonical_smiles=canonical_smiles,
                original_name=original_name,
                original_path=original_path,
                protocol=protocol,
                analysis_start_step=analysis_start_step_val,
            )
            if is_rbfe:
                ref = child.meta.get("ligand_ref")
                alt = child.meta.get("ligand_alt")
                res_ref = child.meta.get("residue_ref") or mol_name
                res_alt = child.meta.get("residue_alt")
                if ref:
                    _copy_equil_artifacts(
                        repo,
                        run_id,
                        lig_name,
                        res_ref,
                        run_dir / "simulations" / ref / "equil",
                        protein_align,
                        dest_subdir="Equil_ref",
                    )
                if alt:
                    _copy_equil_artifacts(
                        repo,
                        run_id,
                        lig_name,
                        res_alt or mol_name,
                        run_dir / "simulations" / alt / "equil",
                        protein_align,
                        dest_subdir="Equil_alt",
                    )
            else:
                _copy_equil_artifacts(
                    repo, run_id, lig_name, mol_name, child.root / "equil", protein_align
                )
            logger.warning(f"[{lig_name}] No totals found under {results_dir}")
            continue

        canonical_smiles, original_name, original_path = extract_ligand_metadata(
            child, original_map
        )

        try:
            rec = FERecord(
                run_id=run_id,
                ligand=lig_name,
                mol_name=mol_name,
                system_name=sim_cfg_updated.system_name,
                fe_type=sim_cfg_updated.fe_type,
                temperature=sim_cfg_updated.temperature,
                method=sim_cfg_updated.dec_int,
                total_dG=total_dG,
                total_se=total_se,
                components=list(sim_cfg_updated.components),
                windows=[],  # optional: can be populated later
                canonical_smiles=canonical_smiles,
                original_name=original_name,
                original_path=original_path,
                protocol=protocol,
                analysis_start_step=analysis_start_step_val,
            )
            repo.save(rec, copy_from=results_dir)
            if is_rbfe:
                ref = child.meta.get("ligand_ref")
                alt = child.meta.get("ligand_alt")
                res_ref = child.meta.get("residue_ref") or mol_name
                res_alt = child.meta.get("residue_alt")
                if ref:
                    _copy_equil_artifacts(
                        repo,
                        run_id,
                        lig_name,
                        res_ref,
                        run_dir / "simulations" / ref / "equil",
                        protein_align,
                        dest_subdir="Equil_ref",
                    )
                if alt:
                    _copy_equil_artifacts(
                        repo,
                        run_id,
                        lig_name,
                        res_alt or mol_name,
                        run_dir / "simulations" / alt / "equil",
                        protein_align,
                        dest_subdir="Equil_alt",
                    )
            else:
                _copy_equil_artifacts(
                    repo, run_id, lig_name, mol_name, child.root / "equil", protein_align
                )
            logger.info(
                f"Saved FE record for ligand {lig_name}"
                f"(ΔG={total_dG:.2f} ± {total_se:.2f} kcal/mol; run_id={run_id})"
            )
        except Exception as exc:
            reason = f"save_failed: {exc}"
            logger.warning(f"Could not save FE record for {lig_name}: {exc}")
            failures.append((lig_name, "failed", reason))
            repo.record_failure(
                run_id=run_id,
                ligand=lig_name,
                system_name=sim_cfg_updated.system_name,
                temperature=sim_cfg_updated.temperature,
                status="failed",
                reason=reason,
                canonical_smiles=canonical_smiles,
                original_name=original_name,
                original_path=original_path,
                protocol=protocol,
                analysis_start_step=analysis_start_step_val,
            )
            if is_rbfe:
                ref = child.meta.get("ligand_ref")
                alt = child.meta.get("ligand_alt")
                res_ref = child.meta.get("residue_ref") or mol_name
                res_alt = child.meta.get("residue_alt")
                if ref:
                    _copy_equil_artifacts(
                        repo,
                        run_id,
                        lig_name,
                        res_ref,
                        run_dir / "simulations" / ref / "equil",
                        protein_align,
                        dest_subdir="Equil_ref",
                    )
                if alt:
                    _copy_equil_artifacts(
                        repo,
                        run_id,
                        lig_name,
                        res_alt or mol_name,
                        run_dir / "simulations" / alt / "equil",
                        protein_align,
                        dest_subdir="Equil_alt",
                    )
            else:
                _copy_equil_artifacts(
                    repo, run_id, lig_name, mol_name, child.root / "equil", protein_align
                )

    return failures


def _copy_equil_artifacts(
    repo: FEResultsRepository,
    run_id: str,
    ligand: str,
    mol_name: str,
    equil_dir: Path,
    protein_align: str | None,
    *,
    dest_subdir: str = "Equil",
) -> None:
    if not equil_dir.exists():
        logger.warning(
            f"Equilibration directory {equil_dir} does not exist; skipping copy."
        )
        return
    dest_dir = repo.ligand_dir(run_id, ligand) / dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidates_map = {
        "equilibration_analysis_results.npz": "equilibration_analysis_results.npz",
        "simulation_analysis.png": "simulation_analysis.png",
        "dihed_hist.png": "dihed_hist.png",
        "representative.pdb": "representative.pdb",
        "representative_complex.pdb": "representative_complex.pdb",
        "representative_pose.pdb": "representative_pose.pdb",
        "initial_pose.pdb": "initial_pose.pdb",
        f"{mol_name}.sdf": f"{mol_name}.sdf",
        f"{mol_name}.prmtop": f"{mol_name}.prmtop",
        f"{mol_name}.pdb": f"{mol_name}.pdb",
        f"full.pdb": f"initial_complex.pdb",
    }

    for name in candidates_map:
        src = equil_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / candidates_map[name])
        else:
            logger.warning(f"Equilibration artifact {src} not found; skipping copy.")

    # add README file
    readme_path = dest_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(
            "This directory contains equilibration artifacts for the ligand.\n"
            "Files may include analysis results, representative structures, and input files.\n"
            "These were copied from the equilibration phase for reference.\n\n\n"
            f"File list:\n"
            f"- equilibration_analysis_results.npz: NumPy archive with equilibration analysis data.\n"
            f"- simulation_analysis.png: Plot of equilibration simulation metrics over time.\n"
            f"- dihed_hist.png: Histogram of dihedral angle distributions during equilibration.\n"
            f"- representative.pdb: Representative snapshot from equilibration that is used for further FEP.\n\n"
            f"- {mol_name}.sdf: Ligand structure file in SDF format.\n"
            f"- {mol_name}.prmtop: AMBER parameter/topology file for the ligand.\n"
            f"- {mol_name}.pdb: PDB structure file for the ligand.\n\n"
            f"- initial_complex.pdb: Initial complex structure used for equilibration.\n"
            f"- representative_complex.pdb: Aligned representative structure to initial complex.\n"
            f"- initial_pose.pdb: Initial ligand pose extracted from the initial complex.\n"
            f"- representative_pose.pdb: Representative ligand pose extracted from the representative complex.\n"
        )
