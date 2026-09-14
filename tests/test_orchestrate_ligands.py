from __future__ import annotations

import json
from pathlib import Path

import pytest

from batter.config.run import RunConfig
from batter.config.utils import apo_ligand_source_path, is_apo_ligand_path
from batter.orchestrate.ligands import discover_staged_ligands, resolve_ligand_map


def test_resolve_ligand_map_rejects_reserved_name_from_json(tmp_path: Path) -> None:
    lig = tmp_path / "lig.sdf"
    lig.write_text("dummy\n")
    lig_json = tmp_path / "ligands.json"
    lig_json.write_text(json.dumps({"transformations": str(lig)}))

    cfg = RunConfig.model_validate(
        {
            "run": {"output_folder": str(tmp_path / "out")},
            "create": {"system_name": "sys", "ligand_input": str(lig_json)},
            "fe_sim": {},
        }
    )

    with pytest.raises(ValueError, match="reserved"):
        resolve_ligand_map(cfg, tmp_path)


def test_resolve_ligand_map_accepts_null_apo_entry_from_json(tmp_path: Path) -> None:
    lig_json = tmp_path / "ligands.json"
    lig_json.write_text(json.dumps({"None": None}))

    cfg = RunConfig.model_validate(
        {
            "run": {"output_folder": str(tmp_path / "out")},
            "create": {"system_name": "sys", "ligand_input": str(lig_json)},
            "fe_sim": {},
        }
    )

    lig_map, original_names = resolve_ligand_map(cfg, tmp_path)

    assert lig_map == {"APO": apo_ligand_source_path().resolve()}
    assert original_names == {"APO": "None"}


def test_resolve_ligand_map_preserves_custom_null_apo_labels(tmp_path: Path) -> None:
    lig_json = tmp_path / "ligands.json"
    lig_json.write_text(json.dumps({"apo_rep1": None, "apo_rep2": None}))

    cfg = RunConfig.model_validate(
        {
            "run": {"output_folder": str(tmp_path / "out")},
            "create": {"system_name": "sys", "ligand_input": str(lig_json)},
            "fe_sim": {},
        }
    )

    lig_map, original_names = resolve_ligand_map(cfg, tmp_path)

    assert lig_map == {
        "APO_REP1": apo_ligand_source_path().resolve(),
        "APO_REP2": apo_ligand_source_path().resolve(),
    }
    assert original_names == {"APO_REP1": "apo_rep1", "APO_REP2": "apo_rep2"}


def test_resolve_ligand_map_rejects_names_that_sanitize_to_same_key(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.sdf"
    second = tmp_path / "second.sdf"
    first.write_text("first\n")
    second.write_text("second\n")
    lig_json = tmp_path / "ligands.json"
    lig_json.write_text(
        json.dumps({"lig-1": str(first), "LIG_1": str(second)})
    )
    cfg = RunConfig.model_validate(
        {
            "run": {"output_folder": str(tmp_path / "out")},
            "create": {"system_name": "sys", "ligand_input": str(lig_json)},
            "fe_sim": {},
        }
    )

    with pytest.raises(ValueError, match="both normalize to 'LIG_1'"):
        resolve_ligand_map(cfg, tmp_path)


def test_resolve_ligand_map_allows_exact_json_key_to_override_paths(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.sdf"
    replacement = tmp_path / "replacement.sdf"
    original.write_text("original\n")
    replacement.write_text("replacement\n")
    lig_json = tmp_path / "ligands.json"
    lig_json.write_text(json.dumps({"LIG1": str(replacement)}))
    cfg = RunConfig.model_validate(
        {
            "run": {"output_folder": str(tmp_path / "out")},
            "create": {
                "system_name": "sys",
                "ligand_paths": {"LIG1": str(original)},
                "ligand_input": str(lig_json),
            },
            "fe_sim": {},
        }
    )

    lig_map, original_names = resolve_ligand_map(cfg, tmp_path)

    assert lig_map == {"LIG1": replacement.resolve()}
    assert original_names == {"LIG1": "LIG1"}


def test_discover_staged_ligands_skips_rbfe_transformations_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "exec"
    # RBFE transformations root must not be interpreted as a ligand directory.
    (run_dir / "simulations" / "transformations").mkdir(parents=True)
    lig_file = run_dir / "simulations" / "LIG1" / "inputs" / "ligand.sdf"
    lig_file.parent.mkdir(parents=True)
    lig_file.write_text("dummy\n")

    lig_map = discover_staged_ligands(run_dir)
    assert set(lig_map.keys()) == {"LIG1"}


def test_discover_staged_ligands_rejects_ambiguous_normalized_names(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "exec"
    first = run_dir / "simulations" / "lig-1" / "inputs" / "ligand.sdf"
    second = run_dir / "simulations" / "LIG_1" / "inputs" / "ligand.sdf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first\n")
    second.write_text("second\n")

    with pytest.raises(ValueError, match="Multiple staged ligands normalize to 'LIG_1'"):
        discover_staged_ligands(run_dir)


def test_discovered_staged_apo_ligand_is_recognized(tmp_path: Path) -> None:
    run_dir = tmp_path / "exec"
    lig_file = run_dir / "simulations" / "APO" / "inputs" / "ligand.pdb"
    lig_file.parent.mkdir(parents=True)
    lig_file.write_text(apo_ligand_source_path().read_text())

    lig_map = discover_staged_ligands(run_dir)

    assert set(lig_map.keys()) == {"APO"}
    assert is_apo_ligand_path(lig_map["APO"])


def test_discover_staged_ligands_prefers_namespaced_shared_inputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "exec"
    protein = run_dir / "inputs" / "protein.pdb"
    system = run_dir / "inputs" / "system.pdb"
    ligand = run_dir / "inputs" / "ligands" / "7LD4.pdb"
    ligand.parent.mkdir(parents=True)
    protein.write_text("protein\n")
    system.write_text("system\n")
    ligand.write_text("ligand\n")

    lig_map = discover_staged_ligands(run_dir)

    assert lig_map == {"7LD4": ligand}


def test_discover_staged_ligands_ignores_legacy_shared_pdbs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "exec"
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "protein.pdb").write_text("protein\n")
    (inputs / "system.pdb").write_text("system\n")
    ligand = inputs / "LIG1.sdf"
    ligand.write_text("ligand\n")

    lig_map = discover_staged_ligands(run_dir)

    assert lig_map == {"LIG1": ligand}
