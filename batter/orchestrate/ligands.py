"""Helpers for resolving ligand inputs and staged ligands for a run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from batter.config.utils import sanitize_ligand_name


def resolve_ligand_map(
    run_cfg, yaml_dir: Path
) -> Tuple[Dict[str, Path], Dict[str, str]]:
    """Resolve ligands from RunConfig sources (paths list or JSON mapping).

    Entries from ``create.ligand_paths`` are merged with (and can be overridden by)
    ``create.ligand_input``; relative paths are resolved against the YAML location
    (or JSON file parent). Ligand identifiers are sanitized for filesystem safety and
    the original names are returned alongside the resolved paths.
    """
    lig_map: Dict[str, Path] = {}
    original_names: Dict[str, str] = {}

    paths = getattr(run_cfg.create, "ligand_paths", None) or dict()
    for name, value in paths.items():
        lig_path = Path(value)
        lig_path = lig_path if lig_path.is_absolute() else (yaml_dir / lig_path)
        sanitized = sanitize_ligand_name(str(name))
        lig_map[sanitized] = lig_path.resolve()
        original_names[sanitized] = str(name)

    lig_json = getattr(run_cfg.create, "ligand_input", None)
    if lig_json:
        jpath = Path(lig_json)
        jpath = jpath if jpath.is_absolute() else (yaml_dir / jpath)
        data = json.loads(jpath.read_text())

        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = ((Path(p).stem, p) for p in data)
        else:
            raise TypeError(f"{jpath} must be a dict or list, got {type(data).__name__}")

        for name, value in items:
            lig_path = Path(value)
            lig_path = lig_path if lig_path.is_absolute() else (jpath.parent / lig_path)
            sanitized = sanitize_ligand_name(str(name))
            lig_map[sanitized] = lig_path.resolve()
            original_names[sanitized] = str(name)

    if not lig_map:
        raise ValueError(
            "No ligands provided. Specify `create.ligand_paths` or `create.ligand_input` in your YAML."
        )

    missing = [str(p) for p in lig_map.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Ligand file(s) not found: {missing}")

    return lig_map, original_names


def discover_staged_ligands(run_dir: Path) -> Dict[str, Path]:
    """Inspect an execution directory to reconstruct ``{ligand: path}``.

    This is used to resume or continue runs without the original ligand inputs by
    scanning staged per-ligand simulation folders (or legacy ``inputs/`` layouts)
    under ``run_dir``.
    """
    lig_map: Dict[str, Path] = {}

    sim_dir = run_dir / "simulations"
    if sim_dir.exists():
        for sub in sim_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.upper()
            inp = sub / "inputs"
            if not inp.exists():
                continue
            for ext in (".sdf", ".mol2", ".pdb"):
                cand = inp / f"ligand{ext}"
                if cand.exists():
                    lig_map[name] = cand
                    break

    if not lig_map:
        inp_dir = run_dir / "inputs"
        if inp_dir.exists():
            for p in inp_dir.iterdir():
                if p.suffix.lower() in {".sdf", ".mol2", ".pdb"}:
                    lig_map[p.stem.upper()] = p

    return lig_map
