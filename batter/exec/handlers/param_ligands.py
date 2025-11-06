from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.orchestrate.state_registry import register_phase_state


# Reuse helpers from content-addressed ligand store
from batter.param.ligand import (
    _rdkit_load,
    _canonical_payload,
    _hash_id,
    batch_ligand_process,
    _convert_mol_name_to_unique,
)


LIGAND_FILES = ["mol2", "prmtop", "sdf", "json", "frcmod", "inpcrd", "lib"]

def copy_ligand_params(src_dir: Path, child_dir: Path, residue_name: str):
    """
    Copy ligand parameter files (lig.*) from src_dir into child_dir/params/,
    renaming them from 'lig.ext' to '{residue_name}.ext'.
    """
    child_params = child_dir / "params"
    child_params.mkdir(parents=True, exist_ok=True)

    for ext in LIGAND_FILES:
        src = src_dir / f"lig.{ext}"
        if not src.exists():
            continue  # skip missing files gracefully
        dst = child_params / f"{residue_name}.{ext}"
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
            logger.debug(f"Copied {src.name} → {dst}")
        except Exception as e:
            logger.warning(f"Failed to copy {src} to {dst}: {e}")


def _resolve_outdir(template: str | Path, system: SimSystem) -> Path:
    """
    Resolve {WORK} placeholder in output dir, then absolutize & expanduser.
    """
    if not isinstance(template, (str, Path)):
        raise TypeError("param_ligands.outdir must be a string")
    resolved = str(template).replace("{WORK}", system.root.as_posix())
    return Path(resolved).expanduser().resolve()


def param_ligands(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Parent-only parametrization pass:
      - Runs batch_ligand_process once into a content-addressed store
      - Links per-ligand params into <root>/ligands/<LIG>/params/*
      - Emits an index for downstream steps
    """
    payload = StepPayload.model_validate(params)
    sys_params = payload.sys_params or SystemParams()
    lig_root = system.root / "simulations"
    if not lig_root.exists():
        raise FileNotFoundError(f"[param_ligands] No 'ligands/' at {system.root}. Did staging run?")


    outdir = _resolve_outdir(sys_params["param_outdir"], system)
    charge = sys_params.get("charge", "am1bcc")
    ligand_ff = sys_params.get("ligand_ff", "gaff2")
    retain = bool(sys_params.get("retain_lig_prot", True))

    lig_map = sys_params["ligand_paths"]

    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[param_ligands] {len(lig_map)} ligands")
    logger.info(
        f"[param_ligands] parameterizing"
        f"(charge={charge}, ff={ligand_ff}, retain H={retain})"
    )

    # Run batch parametrization into content-addressed subfolders
    # Returns (hash_ids_in_order, residue_names_in_order)
    hashes, unique = batch_ligand_process(
        ligand_paths=lig_map,
        output_path=outdir,
        retain_lig_prot=retain,
        ligand_ff=ligand_ff,
        charge_method=charge,
        overwrite=False,
        run_with_slurm=False,
    )
    if not hashes:
        raise RuntimeError("[param_ligands] No ligands processed (empty hash list).")
    
    # generate unique list of resnames
    unique_resnames = []
    for i, (name, p) in enumerate(lig_map.items()):
        init_mol_name = name.lower()
        unique_resname = _convert_mol_name_to_unique(
                mol_name=init_mol_name,
                ind=i,
                smiles=unique[p][1],
                exist_mol_names=set(unique_resnames))
        unique_resnames.append(unique_resname)

    # Link artifacts per staged ligand and collect index rows
    artifacts_index_dir = system.root / "artifacts" / "ligand_params"
    artifacts_index_dir.mkdir(parents=True, exist_ok=True)

    index_entries: List[Dict[str, Any]] = []
    linked: List[Tuple[str, str]] = []  # (name, hash)

    for i, (name, d) in enumerate(lig_map.items()):
        hid = hashes[i]
        src_dir = outdir / hid
        meta_path = src_dir / "metadata.json"
        if not src_dir.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"[param_ligands] Missing params for staged ligand {name}: expected {src_dir}"
            )

        meta = json.loads(meta_path.read_text())
        residue_name = unique_resnames[i]
        title = meta.get("title", name)

        copy_ligand_params(src_dir, lig_root / Path(name), residue_name)

        linked.append((name, hid, residue_name))
        index_entries.append(
            {
                "ligand": name,
                "hash": hid,
                "store_dir": str(src_dir),
                "linked_dir": str(lig_root / Path(name) / "params"),
                "residue_name": residue_name,
                "title": title,
            }
        )

    # Save a machine-readable index for downstream steps
    index_payload = {
        "store": str(outdir),
        "ligands": index_entries,
        "config": {
            "ligand_ff": ligand_ff,
            "charge": charge,
            "retain_lig_prot": retain,
        },
    }
    index_path = artifacts_index_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2))
    marker_rel = index_path.relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "param_ligands",
        required=[[marker_rel]],
        success=[[marker_rel]],
    )

    # also save a simple TSV manifest (name\t hash)
    manifest = artifacts_index_dir / "ligand_manifest.tsv"
    with manifest.open("w") as mf:
        for name, lh, rn in linked:
            mf.write(f"{name}\t{lh}\t{rn}\n")

    logger.debug(f"[param_ligands] Linked params for staged ligands: {linked}")
    logger.debug(f"[param_ligands] Wrote index → {index_path}")

    # Return rich metadata so downstream steps can consume without re-reading disk, if desired
    return ExecResult(
        [],
        {
            "param_store": str(outdir),
            "index_json": str(artifacts_index_dir / "index.json"),
            "manifest_tsv": str(manifest),
            "hashes": hashes,
        },
    )
