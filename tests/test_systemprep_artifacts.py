from __future__ import annotations

import json
from pathlib import Path

from batter.systemprep.artifacts import (
    docked_system_pdb_path,
    ligand_pdb_path,
    manifest_artifact_path,
    resolve_docked_system_pdb,
    resolve_ligand_pdb,
    system_prep_manifest_path,
)


def test_system_and_same_named_ligand_have_distinct_paths(tmp_path: Path) -> None:
    system_pdb = docked_system_pdb_path(tmp_path)
    ligand_pdb = ligand_pdb_path(tmp_path, "7LD4")

    assert system_pdb != ligand_pdb
    assert system_pdb == tmp_path / "all-ligands" / "system" / "docked.pdb"
    assert ligand_pdb == tmp_path / "all-ligands" / "ligands" / "7LD4.pdb"


def test_manifest_resolvers_support_namespaced_relative_paths(tmp_path: Path) -> None:
    system_pdb = docked_system_pdb_path(tmp_path)
    ligand_pdb = ligand_pdb_path(tmp_path, "7LD4")
    system_pdb.parent.mkdir(parents=True)
    ligand_pdb.parent.mkdir(parents=True)
    system_pdb.write_text("system\n")
    ligand_pdb.write_text("ligand\n")

    manifest = {
        "docked": manifest_artifact_path(tmp_path, system_pdb),
        "ligands": {"7LD4": manifest_artifact_path(tmp_path, ligand_pdb)},
    }
    system_prep_manifest_path(tmp_path).write_text(json.dumps(manifest))

    assert resolve_docked_system_pdb(tmp_path, "7LD4") == system_pdb
    assert resolve_ligand_pdb(tmp_path, "7LD4") == ligand_pdb
    assert resolve_docked_system_pdb(tmp_path, "7LD4").read_text() == "system\n"
    assert resolve_ligand_pdb(tmp_path, "7LD4").read_text() == "ligand\n"


def test_manifest_resolvers_support_legacy_flat_layout(tmp_path: Path) -> None:
    stage_dir = tmp_path / "all-ligands"
    stage_dir.mkdir()
    legacy_system = stage_dir / "SYS.pdb"
    legacy_ligand = stage_dir / "LIG1.pdb"
    legacy_system.write_text("system\n")
    legacy_ligand.write_text("ligand\n")

    assert resolve_docked_system_pdb(tmp_path, "SYS") == legacy_system
    assert resolve_ligand_pdb(tmp_path, "LIG1") == legacy_ligand


def test_manifest_resolvers_prefer_relocated_local_legacy_files(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_stage = old_root / "all-ligands"
    new_stage = new_root / "all-ligands"
    old_stage.mkdir(parents=True)
    new_stage.mkdir(parents=True)
    old_system = old_stage / "SYS.pdb"
    old_ligand = old_stage / "LIG1.pdb"
    new_system = new_stage / "SYS.pdb"
    new_ligand = new_stage / "LIG1.pdb"
    for path in (old_system, old_ligand, new_system, new_ligand):
        path.write_text(path.parent.parent.name)
    manifest = {
        "docked": str(old_system),
        "ligands": {"LIG1": str(old_ligand)},
    }
    system_prep_manifest_path(new_root).write_text(json.dumps(manifest))

    assert resolve_docked_system_pdb(new_root, "SYS") == new_system
    assert resolve_ligand_pdb(new_root, "LIG1") == new_ligand


def test_reserved_artifact_names_are_safe_as_ligand_names(tmp_path: Path) -> None:
    paths = {
        ligand_pdb_path(tmp_path, name)
        for name in ("REFERENCE", "SYSTEM_INPUT", "SYSTEM_ALIGNED", "DOCKED")
    }

    assert len(paths) == 4
    assert all(path.parent.name == "ligands" for path in paths)
    assert docked_system_pdb_path(tmp_path) not in paths
