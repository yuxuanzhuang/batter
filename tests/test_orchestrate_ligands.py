from __future__ import annotations

import json
from pathlib import Path

import pytest

from batter.config.run import RunConfig
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


def test_discover_staged_ligands_skips_rbfe_transformations_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "exec"
    # RBFE transformations root must not be interpreted as a ligand directory.
    (run_dir / "simulations" / "transformations").mkdir(parents=True)
    lig_file = run_dir / "simulations" / "LIG1" / "inputs" / "ligand.sdf"
    lig_file.parent.mkdir(parents=True)
    lig_file.write_text("dummy\n")

    lig_map = discover_staged_ligands(run_dir)
    assert set(lig_map.keys()) == {"LIG1"}

