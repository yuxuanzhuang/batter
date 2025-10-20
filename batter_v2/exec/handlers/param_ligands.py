from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem

# Reuse helpers from content-addressed ligand store
from batter.param.ligand import (
    _rdkit_load,
    _canonical_payload,
    _hash_id,
    batch_ligand_process,
)


def _resolve_outdir(template: str, system: SimSystem) -> Path:
    """
    Resolve {WORK} placeholder in output dir, then absolutize & expanduser.
    """
    if not isinstance(template, str):
        raise TypeError("param_ligands.outdir must be a string")
    resolved = template.replace("{WORK}", system.root.as_posix())
    return Path(resolved).expanduser().resolve()


def param_ligands(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """
    Parent-only parametrization pass:
      - Validates staged ligands under <root>/ligands/*/inputs/ligand.sdf
      - Runs batch_ligand_process once into a content-addressed store
      - Links per-ligand params into <root>/ligands/<LIG>/params/*
      - Emits an index for downstream steps
    """
    lig_root = system.root / "simulations"
    if not lig_root.exists():
        raise FileNotFoundError(f"[param_ligands] No 'ligands/' at {system.root}. Did staging run?")

    # each child dir must contain inputs/ligand.sdf
    staged = [d for d in sorted(lig_root.glob("*")) if (d / "inputs" / "ligand.sdf").exists()]
    if not staged:
        raise FileNotFoundError(f"[param_ligands] No staged ligands found under {lig_root}.")

    outdir = _resolve_outdir(params["outdir"], system)
    charge = params.get("charge", "am1bcc")
    ligand_ff = params.get("ligand_ff", "gaff2")
    retain = bool(params.get("retain_lig_prot", True))

    # alias (child folder name) -> path to sdf
    lig_map: Dict[str, str] = {d.name: (d / "inputs" / "ligand.sdf").as_posix() for d in staged}

    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[param_ligands] {len(lig_map)} ligands")
    logger.info(
        f"[param_ligands] parameterizing → {outdir} "
        f"(charge={charge}, ff={ligand_ff}, retain={retain})"
    )

    # Run batch parametrization into content-addressed subfolders
    # Returns (hash_ids_in_order, residue_names_in_order)
    hashes, residue_names = batch_ligand_process(
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

    # Build mapping alias->hash by recomputing hid from each staged input
    # (robust to ordering and guarantees alias alignment)
    alias_to_hash: Dict[str, str] = {}
    for d in staged:
        sdf = d / "inputs" / "ligand.sdf"
        mol = _rdkit_load(sdf, retain_h=retain)
        smi = _canonical_payload(mol)
        hid = _hash_id(smi, ligand_ff=ligand_ff, retain_h=retain)
        alias_to_hash[d.name] = hid

    # Link artifacts per staged ligand and collect index rows
    artifacts_index_dir = system.root / "artifacts" / "ligand_params"
    artifacts_index_dir.mkdir(parents=True, exist_ok=True)

    index_entries: List[Dict[str, Any]] = []
    linked: List[Tuple[str, str]] = []  # (alias, hash)

    for d in staged:
        alias = d.name
        hid = alias_to_hash[alias]
        src_dir = outdir / hid
        meta_path = src_dir / "metadata.json"
        if not src_dir.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"[param_ligands] Missing params for staged ligand {alias}: expected {src_dir}"
            )

        # load residue name and title from metadata (written by batch_ligand_process)
        try:
            meta = json.loads(meta_path.read_text())
            residue_name = meta.get("residue_name", alias[:3].lower())
            title = meta.get("title", alias)
        except Exception:
            residue_name, title = alias[:3].lower(), alias

        # link (or copy fallback) into child params/
        child_params = d / "params"
        child_params.mkdir(parents=True, exist_ok=True)

        for src in sorted(src_dir.glob("*")):
            dst = child_params / src.name
            if dst.exists():
                continue
            try:
                # Make symlink relative for portability across machines
                rel = os.path.relpath(src, start=child_params)
                dst.symlink_to(rel)
            except Exception:
                shutil.copy2(src, dst)

        linked.append((alias, hid))
        index_entries.append(
            {
                "ligand": alias,
                "hash": hid,
                "store_dir": str(src_dir),
                "linked_dir": str(child_params),
                "residue_name": residue_name,  # 3-char (lower)
                "title": title,                 # SDF internal _Name
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
    (artifacts_index_dir / "index.json").write_text(json.dumps(index_payload, indent=2))

    # also save a simple TSV manifest (alias\t hash)
    manifest = artifacts_index_dir / "ligand_manifest.tsv"
    with manifest.open("w") as mf:
        for alias, lh in linked:
            mf.write(f"{alias}\t{lh}\n")

    logger.info(f"[param_ligands] Linked params for staged ligands: {linked}")
    logger.info(f"[param_ligands] Wrote index → {artifacts_index_dir / 'index.json'}")

    # Return rich metadata so downstream steps can consume without re-reading disk, if desired
    return ExecResult(
        [],
        {
            "param_store": str(outdir),
            "index_json": str(artifacts_index_dir / "index.json"),
            "manifest_tsv": str(manifest),
            "alias_to_hash": alias_to_hash,
            "hashes": hashes,
            "residue_names": residue_names,
        },
    )