from __future__ import annotations

import json
import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")

# Stub minimal gufe module to satisfy imports when dependency is absent
sys.modules.setdefault("gufe", ModuleType("gufe"))
sys.modules["gufe"].SmallMoleculeComponent = object  # type: ignore[attr-defined]

import batter.exec.handlers.param_ligands as handler


def test_param_ligands_uses_string_paths_for_unique_map(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "simulations").mkdir(parents=True)

    system = SimpleNamespace(root=root)

    lig_file = tmp_path / "lig.sdf"
    lig_file.write_text("fake")

    outdir = tmp_path / "store"
    hash_id = "HASH123"
    store_dir = outdir / hash_id
    store_dir.mkdir(parents=True)
    (store_dir / "metadata.json").write_text(json.dumps({"title": "Lig"}))
    (store_dir / "lig.prmtop").write_text("ok")  # make cache detectable

    sys_params = {
        "param_outdir": outdir,
        "charge": "bcc",
        "ligand_ff": "gaff2",
        "retain_lig_prot": True,
        "ligand_paths": {"Lig1": lig_file},
    }

    monkeypatch.setattr(
        handler.StepPayload, "model_validate", staticmethod(lambda params: SimpleNamespace(sys_params=params["sys_params"], model_extra={}))
    )
    monkeypatch.setattr(
        handler, "_convert_mol_name_to_unique", lambda mol_name, ind, smiles, exist_mol_names: mol_name
    )
    monkeypatch.setattr(handler, "register_phase_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(handler, "copy_ligand_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        handler,
        "batch_ligand_process",
        lambda ligand_paths, output_path, retain_lig_prot, ligand_ff, charge_method, overwrite, run_with_slurm, on_failure: ([hash_id], {str(lig_file): (hash_id, "SMI")}),
    )

    result = handler.param_ligands(None, system, {"sys_params": sys_params})

    index_json = json.loads((root / "artifacts" / "ligand_params" / "index.json").read_text())
    assert index_json["ligands"][0]["ligand"] == "Lig1"
    assert index_json["ligands"][0]["hash"] == hash_id
    assert result.artifacts["hashes"] == [hash_id]
