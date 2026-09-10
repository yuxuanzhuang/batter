"""Paths and compatibility helpers for system-preparation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SYSTEM_PREP_DIRNAME = "all-ligands"
SYSTEM_ARTIFACTS_DIRNAME = "system"
LIGAND_ARTIFACTS_DIRNAME = "ligands"
SYSTEM_PREP_MANIFEST_FILENAME = "manifest.json"
DOCKED_SYSTEM_FILENAME = "docked.pdb"


def system_prep_stage_dir(system_root: Path | str) -> Path:
    """Return the shared system-preparation artifact directory."""
    return Path(system_root) / SYSTEM_PREP_DIRNAME


def system_prep_manifest_path(system_root: Path | str) -> Path:
    """Return the system-preparation manifest path."""
    return system_prep_stage_dir(system_root) / SYSTEM_PREP_MANIFEST_FILENAME


def docked_system_pdb_path(system_root: Path | str) -> Path:
    """Return the collision-safe path for the prepared receptor/system PDB."""
    return (
        system_prep_stage_dir(system_root)
        / SYSTEM_ARTIFACTS_DIRNAME
        / DOCKED_SYSTEM_FILENAME
    )


def ligand_pdb_path(system_root: Path | str, ligand: str) -> Path:
    """Return the collision-safe path for a prepared ligand PDB."""
    return (
        system_prep_stage_dir(system_root)
        / LIGAND_ARTIFACTS_DIRNAME
        / f"{ligand}.pdb"
    )


def manifest_artifact_path(system_root: Path | str, artifact: Path | str) -> str:
    """Serialize an artifact relative to ``all-ligands`` when possible."""
    stage_dir = system_prep_stage_dir(system_root).resolve()
    path = Path(artifact)
    resolved = path.resolve()
    try:
        return resolved.relative_to(stage_dir).as_posix()
    except ValueError:
        return str(resolved)


def _load_manifest(system_root: Path | str) -> dict[str, Any]:
    path = system_prep_manifest_path(system_root)
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_value_path(system_root: Path | str, value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return system_prep_stage_dir(system_root) / path


def _first_existing_or_default(candidates: list[Path | None]) -> Path:
    usable = [path for path in candidates if path is not None]
    for path in usable:
        if path.is_file():
            return path
    if not usable:
        raise ValueError("No system-preparation artifact candidates were provided.")
    return usable[0]


def _manifest_first_candidates(
    system_root: Path | str,
    configured: Path | None,
    local_candidates: list[Path | None],
) -> list[Path | None]:
    """Prefer manifest paths inside this run, but not stale absolute locations."""
    if configured is None:
        return local_candidates
    try:
        configured.resolve().relative_to(system_prep_stage_dir(system_root).resolve())
    except ValueError:
        return [*local_candidates, configured]
    return [configured, *local_candidates]


def resolve_docked_system_pdb(
    system_root: Path | str,
    system_name: str | None = None,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve a prepared system PDB from new or legacy layouts."""
    payload = dict(manifest) if manifest is not None else _load_manifest(system_root)
    configured = _manifest_value_path(system_root, payload.get("docked"))
    legacy = (
        system_prep_stage_dir(system_root) / f"{system_name}.pdb"
        if system_name
        else None
    )
    return _first_existing_or_default(
        _manifest_first_candidates(
            system_root,
            configured,
            [docked_system_pdb_path(system_root), legacy],
        )
    )


def _manifest_ligand_value(manifest: Mapping[str, Any], ligand: str) -> Any:
    ligand_map = manifest.get("ligands")
    if not isinstance(ligand_map, Mapping):
        return None
    if ligand in ligand_map:
        return ligand_map[ligand]
    upper = ligand.upper()
    if upper in ligand_map:
        return ligand_map[upper]
    matches = [value for key, value in ligand_map.items() if str(key).upper() == upper]
    return matches[0] if len(matches) == 1 else None


def resolve_ligand_pdb(
    system_root: Path | str,
    ligand: str,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve a prepared ligand PDB from new or legacy layouts."""
    payload = dict(manifest) if manifest is not None else _load_manifest(system_root)
    configured = _manifest_value_path(
        system_root, _manifest_ligand_value(payload, ligand)
    )
    stage_dir = system_prep_stage_dir(system_root)
    upper = ligand.upper()
    return _first_existing_or_default(
        _manifest_first_candidates(
            system_root,
            configured,
            [
                ligand_pdb_path(system_root, ligand),
                ligand_pdb_path(system_root, upper),
                stage_dir / f"{ligand}.pdb",
                stage_dir / f"{upper}.pdb",
            ],
        )
    )
