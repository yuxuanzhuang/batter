from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.orchestrate.state_registry import register_phase_state

# -----------------------
# Small helpers
# -----------------------
def _ensure_pdb(lig_path: Path, out_dir: Path) -> Path:
    """
    Ensure a PDB exists for a ligand file; if not PDB, convert via RDKit.
    Returns the path to the PDB file we wrote or found.
    """
    lig_path = Path(lig_path)
    if lig_path.suffix.lower() == ".pdb":
        return lig_path

    try:
        from rdkit import Chem
    except Exception as e:
        raise RuntimeError(
            f"Ligand {lig_path} is not PDB; RDKit is required to convert SDF/MOL2 → PDB."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f"{lig_path.stem}.pdb"

    if lig_path.suffix.lower() == ".sdf":
        suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            raise ValueError(f"RDKit could not read any molecule from {lig_path}")
        Chem.MolToPDBFile(mols[0], str(out_pdb))
    elif lig_path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(lig_path), removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit could not read {lig_path}")
        Chem.MolToPDBFile(mol, str(out_pdb))
    else:
        raise ValueError(f"Unsupported ligand format: {lig_path.suffix} for {lig_path}")

    return out_pdb


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# -----------------------
# Runner (MASFE)
# -----------------------
class _MASFESystemPrepRunner:
    """
    Minimal system_prep for MASFE (solvation FE):
      - No protein/topology/coordinates
      - Stage ligands into all-ligands/ as <NAME>.pdb (convert if needed)
      - Write a manifest for downstream handlers
    """

    def __init__(self, system: SimSystem) -> None:
        self.system = system
        self.output_dir = system.root
        self.ligand_stage_dir = self.output_dir / "all-ligands"
        self.ligand_stage_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, system_name: str, ligand_paths: Dict[str, str]) -> Dict[str, Any]:
        logger.info(f"[MASFE system_prep] system={system_name}, ligands={len(ligand_paths)}")
        staged_map: Dict[str, str] = {}

        # Convert all ligands to PDB (if needed) and stage to all-ligands/
        for name, src in sorted(ligand_paths.items()):
            src_p = Path(src)
            if not src_p.exists():
                raise FileNotFoundError(f"Ligand file not found: {src_p}")
            pdb = _ensure_pdb(src_p, self.ligand_stage_dir)
            # stage as <NAME>.pdb (uppercase key for consistency)
            dst = self.ligand_stage_dir / f"{name.upper()}.pdb"
            if pdb.resolve() != dst.resolve():
                _copy(pdb, dst)
            staged_map[name.upper()] = str(dst)

        # Minimal manifest
        manifest = {
            "system_name": system_name,
            "mode": "MASFE",
            "ligands": staged_map,
        }
        (self.ligand_stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


# -----------------------
# Handler entry point
# -----------------------
def system_prep_masfe(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    MASFE system_prep:
      - No receptor alignment or membrane logic
      - Stage ligands → PDB under all-ligands/
      - Write artifacts/config/sim_overrides.json with a small 'is_solvation' flag
    """
    logger.info(f"[system_prep_masfe] Preparing solvation FE system in {system.root}")

    # Expect the same sys_params envelope your orchestrator already passes
    payload = StepPayload.model_validate(params)
    sys_params = payload.sys_params or SystemParams({})
    lig_map = sys_params["ligand_paths"]  # should already be {NAME: abs_path}

    runner = _MASFESystemPrepRunner(system)
    manifest = runner.run(system_name=sys_params["system_name"], ligand_paths=lig_map)

    # Minimal overrides that tell downstream we are in MASFE/solvation mode.
    overrides = {
        "is_solvation": True,
        # Surface common FE knobs if you want handlers to see them early:
        "water_model": sys_params.get("water_model", "TIP3P"),
        "ion_conc": sys_params.get("ion_conc", 0.0),
        "cation": sys_params.get("cation", "Na+"),
        "anion": sys_params.get("anion", "Cl-"),
        # No anchors, no membrane here
    }
    (system.root / "artifacts" / "config").mkdir(parents=True, exist_ok=True)
    overrides_path = system.root / "artifacts" / "config" / "sim_overrides.json"
    overrides_path.write_text(json.dumps(overrides, indent=2))

    marker_rel = overrides_path.relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "system_prep_asfe",
        required=[[marker_rel]],
        success=[[marker_rel]],
    )

    outputs = [system.root / "all-ligands" / "manifest.json"]
    info = {"system_prep_ok": True, **manifest, "sim_updates": overrides}
    logger.info(f"[system_prep_masfe] Done (ligands: {len(manifest['ligands'])}).")
    return ExecResult(outputs, info)
