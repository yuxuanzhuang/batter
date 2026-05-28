from __future__ import annotations

import json
from pathlib import Path

import pytest

from batter.config.utils import apo_ligand_source_path
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import Step
from batter.systems.core import SimSystem


def test_bundled_apo_dummy_source_has_one_atom() -> None:
    source = apo_ligand_source_path().read_text()

    assert source.count("HETATM") == 1
    assert "DU1" in source
    assert "DU2" not in source
    assert "DU3" not in source


def test_param_ligands_links_apo_dummy_params(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("MDAnalysis", exc_type=ImportError)
    handler_mod = pytest.importorskip(
        "batter.exec.handlers.param_ligands",
        exc_type=ImportError,
    )

    run_root = tmp_path / "run"
    (run_root / "simulations" / "APO" / "inputs").mkdir(parents=True)

    def fake_run_with_log(cmd, *, working_dir=None, **kwargs):
        work = Path(working_dir)
        (work / "lig.prmtop").write_text("%VERSION\n")
        (work / "lig.inpcrd").write_text("dummy\n")
        (work / "lig.lib").write_text("dummy\n")

    monkeypatch.setattr(handler_mod, "run_with_log", fake_run_with_log)

    payload = StepPayload(
        sys_params={
            "param_outdir": str(tmp_path / "params"),
            "charge": "am1bcc",
            "ligand_ff": "openff-2.3.0",
            "retain_lig_prot": True,
            "ligand_paths": {"APO": str(apo_ligand_source_path())},
        }
    )

    result = handler_mod.param_ligands(
        Step(name="param_ligands"),
        SimSystem(name="sys", root=run_root),
        payload.model_dump(),
    )

    index_path = Path(result.artifacts["index_json"])
    index = json.loads(index_path.read_text())
    assert index["ligands"][0]["ligand"] == "APO"
    assert index["ligands"][0]["residue_name"] == "apo"
    assert index["ligands"][0]["title"] == "apo dummy ligand"

    child_params = run_root / "simulations" / "APO" / "params"
    for ext in ("mol2", "sdf", "json", "frcmod", "prmtop", "inpcrd", "lib", "pdb"):
        assert (child_params / f"apo.{ext}").exists()

    mol2 = (child_params / "apo.mol2").read_text()
    sdf = (child_params / "apo.sdf").read_text()
    pdb = (child_params / "apo.pdb").read_text()
    assert "    1     0     1     0     1" in mol2
    assert "  1  0  0  0  0  0" in sdf
    assert "DU1" in mol2
    assert "DU1" in pdb
    assert sdf.count(" Pb ") == 1
    assert "DU2" not in mol2
    assert "DU2" not in pdb
    assert "DU3" not in mol2
    assert "DU3" not in pdb
