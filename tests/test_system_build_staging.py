from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from batter.systems.core import SimSystem
from batter.systems.mabfe import MABFEBuilder
from batter.systems.masfe import MASFEBuilder


def test_mabfe_builder_namespaces_ligands_and_resumes_without_shared_pdbs(
    tmp_path: Path,
) -> None:
    protein = tmp_path / "7LD4.pdb"
    system = tmp_path / "7LD4_dabbled.pdb"
    ligand = tmp_path / "adenosine.sdf"
    protein.write_text("protein\n")
    system.write_text("system\n")
    ligand.write_text("ligand\n")
    root = tmp_path / "run"
    args = SimpleNamespace(
        system_name="7LD4",
        protein_input=protein,
        system_input=system,
        system_coordinate=None,
        ligand_paths={"7LD4": ligand},
        lipid_mol=[],
        anchor_atoms=[],
        ligand_ff="gaff2",
        overwrite=False,
    )
    builder = MABFEBuilder()

    prepared = builder.build(SimSystem(name="7LD4", root=root), args)
    resumed = builder.build(SimSystem(name="7LD4", root=root), args)

    assert prepared.protein == root / "inputs" / "protein.pdb"
    assert prepared.ligands == (root / "inputs" / "ligands" / "7LD4.sdf",)
    assert resumed.ligands == prepared.ligands
    assert root / "inputs" / "system.pdb" not in resumed.ligands
    assert root / "inputs" / "protein.pdb" not in resumed.ligands


def test_masfe_builder_namespaces_ligands_on_initial_and_resumed_builds(
    tmp_path: Path,
) -> None:
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    root = tmp_path / "run"
    args = SimpleNamespace(
        system_name="SOLV",
        ligand_paths={"LIG1": ligand},
        ligand_ff="gaff2",
        overwrite=False,
    )
    builder = MASFEBuilder()

    prepared = builder.build(SimSystem(name="SOLV", root=root), args)
    resumed = builder.build(SimSystem(name="SOLV", root=root), args)

    expected = (root / "inputs" / "ligands" / "LIG1.sdf",)
    assert prepared.ligands == expected
    assert resumed.ligands == expected


@pytest.mark.parametrize("builder_type", [MABFEBuilder, MASFEBuilder])
def test_bulk_ligand_staging_rejects_duplicate_normalized_stems(
    builder_type, tmp_path: Path
) -> None:
    first = tmp_path / "first" / "lig.sdf"
    second = tmp_path / "second" / "LIG.mol2"
    parent = SimSystem(name="SYS", root=tmp_path / "run")

    with pytest.raises(ValueError, match="normalize to subsystem name 'LIG'"):
        builder_type().build_all_ligands(parent, [first, second])
