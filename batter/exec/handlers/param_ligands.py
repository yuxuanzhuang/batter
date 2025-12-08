"""Parameterise ligands and populate per-ligand artifacts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

from batter.orchestrate.state_registry import register_phase_state
from batter.param.ligand import (
    _convert_mol_name_to_unique,
    _hash_id,
    _rdkit_load,
    _canonical_payload,
    batch_ligand_process,
)
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem

LIGAND_FILES = ["mol2", "prmtop", "sdf", "json", "frcmod", "inpcrd", "lib"]

def copy_ligand_params(src_dir: Path, child_dir: Path, residue_name: str) -> None:
    """Copy ``lig.*`` artifacts into ``child_dir/params`` using ``residue_name``."""
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
    """Resolve ``{WORK}`` placeholders against ``system.root``."""
    if not isinstance(template, (str, Path)):
        raise TypeError("param_ligands.outdir must be a string")
    resolved = str(template).replace("{WORK}", system.root.as_posix())
    return Path(resolved).expanduser().resolve()


def _require(sys_params: SystemParams, key: str) -> Any:
    try:
        return sys_params[key]
    except KeyError as exc:
        raise KeyError(f"[param_ligands] Missing required sys_params[{key!r}]") from exc


def param_ligands(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """Run the ligand parametrisation pipeline and index results.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Mapping containing the parameter store path, JSON index, manifest, and raw hashes.
    """
    payload = StepPayload.model_validate(params)
    sys_params = payload.sys_params or SystemParams()
    lig_root = system.root / "simulations"
    if not lig_root.exists():
        raise FileNotFoundError(f"[param_ligands] No 'ligands/' at {system.root}. Did staging run?")


    outdir = _resolve_outdir(sys_params["param_outdir"], system)
    charge = _require(sys_params, "charge")
    ligand_ff = _require(sys_params, "ligand_ff")
    retain = bool(_require(sys_params, "retain_lig_prot"))

    lig_map = sys_params["ligand_paths"]

    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[param_ligands] {len(lig_map)} ligands")
    logger.info(
        f"[param_ligands] parameterizing"
        f"(charge={charge}, ff={ligand_ff}, retain H={retain})"
    )

    artifacts_index_dir = system.root / "artifacts" / "ligand_params"
    artifacts_index_dir.mkdir(parents=True, exist_ok=True)
    index_path = artifacts_index_dir / "index.json"

    try:
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
    except Exception as exc:
        # allow reuse of an existing index when present
        if index_path.exists():
            logger.error(
                "[param_ligands] encountered error but index exists; reusing cached ligands. Error: %s",
                exc,
            )
            existing_index = json.loads(index_path.read_text())
            index_entries = existing_index.get("ligands", [])
            # write a manifest to keep downstream in sync
            manifest = artifacts_index_dir / "ligand_manifest.tsv"
            with manifest.open("w") as mf:
                for entry in index_entries:
                    mf.write(
                        f"{entry.get('ligand')}\t{entry.get('hash')}\t{entry.get('residue_name')}\n"
                    )
            marker_rel = index_path.relative_to(system.root).as_posix()
            register_phase_state(
                system.root,
                "param_ligands",
                required=[[marker_rel]],
                success=[[marker_rel]],
            )
            return ExecResult(
                [],
                {
                    "param_store": existing_index.get("store", str(outdir)),
                    "index_json": str(index_path),
                    "manifest_tsv": str(manifest),
                    "hashes": [e.get("hash") for e in index_entries],
                },
            )

        # Attempt to salvage cached ligands: use existing param store entries only
        salvaged_hashes: List[str] = []
        unique = {}
        for name, path in lig_map.items():
            try:
                mol = _rdkit_load(path, retain_h=retain)
                smi = _canonical_payload(mol)
                hid = _hash_id(smi, ligand_ff=ligand_ff, retain_h=retain)
                cache_dir = outdir / hid
                if (cache_dir / "lig.prmtop").exists():
                    unique[path] = (hid, smi)
                    salvaged_hashes.append(hid)
            except Exception:
                continue

        if salvaged_hashes:
            logger.error(
                "[param_ligands] encountered error; salvaged %d cached ligands and will skip failures.",
                len(salvaged_hashes),
            )
            hashes = salvaged_hashes
        else:
            logger.error(
                "[param_ligands] encountered error and no cached ligands could be salvaged: %s",
                exc,
            )
            raise

    # generate unique list of resnames only for ligands we have data for
    unique_resnames: Dict[str, str] = {}
    seen_resnames: set[str] = set()
    for i, (name, p) in enumerate(lig_map.items()):
        smiles_val = unique.get(p, (None, None))[1]
        if smiles_val is None:
            continue
        init_mol_name = name.lower()
        unique_resname = _convert_mol_name_to_unique(
            mol_name=init_mol_name,
            ind=i,
            smiles=smiles_val,
            exist_mol_names=seen_resnames,
        )
        seen_resnames.add(unique_resname)
        unique_resnames[name] = unique_resname

    # Link artifacts per staged ligand and collect index rows
    index_entries: List[Dict[str, Any]] = []
    linked: List[Tuple[str, str]] = []  # (name, hash)

    for name, d in lig_map.items():
        if name not in unique_resnames:
            logger.warning("[param_ligands] Skipping ligand %s due to parametrization failure.", name)
            continue
        hid = unique.get(d, (None, None))[0]
        if hid is None:
            logger.warning("[param_ligands] Missing hash for ligand %s; skipping.", name)
            continue
        src_dir = outdir / hid
        meta_path = src_dir / "metadata.json"
        if not src_dir.exists() or not meta_path.exists():
            logger.warning(
                "[param_ligands] Missing params for staged ligand %s at %s; skipping.",
                name,
                src_dir,
            )
            continue

        meta = json.loads(meta_path.read_text())
        residue_name = unique_resnames.get(name)
        if residue_name is None:
            logger.warning("[param_ligands] Missing residue name for %s; skipping.", name)
            continue
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
