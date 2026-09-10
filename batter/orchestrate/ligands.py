"""Helpers for resolving ligand inputs and staged ligands for a run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from batter.config.utils import (
    apo_ligand_source_path,
    coerce_apo_ligand_name,
    is_apo_ligand_value,
    sanitize_user_ligand_name,
)


def resolve_ligand_map(
    run_cfg, yaml_dir: Path
) -> Tuple[Dict[str, Path], Dict[str, str]]:
    """Resolve ligands from RunConfig sources (paths list or JSON mapping).

    Entries from ``create.ligand_paths`` are merged with ``create.ligand_input``;
    the latter may override the exact same original key. Relative paths are resolved
    against the YAML location (or JSON file parent). Ligand identifiers are sanitized
    for filesystem safety and ambiguous sanitized names are rejected.
    """
    lig_map: Dict[str, Path] = {}
    original_names: Dict[str, str] = {}
    origins: Dict[str, str] = {}

    def store_ligand(
        sanitized: str,
        original: str,
        path: Path,
        *,
        source: str,
    ) -> None:
        if sanitized in lig_map:
            exact_json_override = (
                source == "ligand_input"
                and origins[sanitized] == "ligand_paths"
                and original_names[sanitized] == original
            )
            if not exact_json_override:
                raise ValueError(
                    "Ligand names "
                    f"{original_names[sanitized]!r} and {original!r} both normalize "
                    f"to {sanitized!r}. Choose distinct ligand identifiers."
                )
        lig_map[sanitized] = path.resolve()
        original_names[sanitized] = original
        origins[sanitized] = source

    paths = getattr(run_cfg.create, "ligand_paths", None) or dict()
    for name, value in paths.items():
        if is_apo_ligand_value(value):
            lig_path = apo_ligand_source_path()
            sanitized = coerce_apo_ligand_name(name)
        else:
            lig_path = Path(value)
            lig_path = lig_path if lig_path.is_absolute() else (yaml_dir / lig_path)
            sanitized = sanitize_user_ligand_name(str(name))
        store_ligand(
            sanitized,
            str(name),
            lig_path,
            source="ligand_paths",
        )

    lig_json = getattr(run_cfg.create, "ligand_input", None)
    if lig_json:
        jpath = Path(lig_json)
        jpath = jpath if jpath.is_absolute() else (yaml_dir / jpath)
        data = json.loads(jpath.read_text())

        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = (
                (
                    coerce_apo_ligand_name(p)
                    if is_apo_ligand_value(p)
                    else Path(p).stem,
                    p,
                )
                for p in data
            )
        else:
            raise TypeError(f"{jpath} must be a dict or list, got {type(data).__name__}")

        for name, value in items:
            if is_apo_ligand_value(value):
                lig_path = apo_ligand_source_path()
                sanitized = coerce_apo_ligand_name(name)
            else:
                lig_path = Path(value)
                lig_path = (
                    lig_path if lig_path.is_absolute() else (jpath.parent / lig_path)
                )
                sanitized = sanitize_user_ligand_name(str(name))
            store_ligand(
                sanitized,
                str(name),
                lig_path,
                source="ligand_input",
            )

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
    scanning staged per-ligand simulation folders, namespaced shared inputs, or
    legacy flat ``inputs/`` layouts under ``run_dir``.
    """
    lig_map: Dict[str, Path] = {}

    def record_ligand(raw_name: str, path: Path) -> None:
        name = sanitize_user_ligand_name(raw_name)
        previous = lig_map.get(name)
        if previous is not None and previous != path:
            raise ValueError(
                f"Multiple staged ligands normalize to {name!r}: "
                f"{previous} and {path}."
            )
        lig_map[name] = path

    sim_dir = run_dir / "simulations"
    if sim_dir.exists():
        for sub in sorted(sim_dir.iterdir()):
            if not sub.is_dir():
                continue
            inp = sub / "inputs"
            if not inp.exists():
                continue
            for ext in (".sdf", ".mol2", ".pdb"):
                cand = inp / f"ligand{ext}"
                if cand.exists():
                    record_ligand(sub.name, cand)
                    break

    if not lig_map:
        inp_dir = run_dir / "inputs"
        namespaced_dir = inp_dir / "ligands"
        if namespaced_dir.exists():
            for p in sorted(namespaced_dir.iterdir()):
                if p.suffix.lower() in {".sdf", ".mol2", ".pdb"}:
                    record_ligand(p.stem, p)

    if not lig_map:
        inp_dir = run_dir / "inputs"
        if inp_dir.exists():
            for p in sorted(inp_dir.iterdir()):
                # These are the canonical shared receptor/system names used by
                # legacy MABFE staging, not ligands.
                if p.name in {"protein.pdb", "system.pdb"}:
                    continue
                if p.suffix.lower() in {".sdf", ".mol2", ".pdb"}:
                    record_ligand(p.stem, p)

    return lig_map
