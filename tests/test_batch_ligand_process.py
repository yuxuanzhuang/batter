from __future__ import annotations

from pathlib import Path

import pytest

import batter.param.ligand as ligand_mod


class _FakeLigand:
    def __init__(self, ligand_file: str, output_dir: str, fail: bool = False):
        self.ligand_file = ligand_file
        self.output_dir = output_dir
        self._fail = fail
        self.name = Path(ligand_file).stem

    def prepare_ligand_parameters(self) -> None:
        if self._fail:
            raise RuntimeError("boom")
        # minimal marker to mimic cached params
        Path(self.output_dir, "lig.prmtop").write_text("ok")


class _FakeFactory:
    def __init__(self, fail_paths: set[Path]):
        self._fail_paths = {Path(p).resolve() for p in fail_paths}

    def create_ligand(
        self,
        ligand_file,
        index: int,
        output_dir,
        ligand_name=None,
        charge: str = "am1bcc",
        retain_lig_prot: bool = True,
        ligand_ff: str = "gaff2",
        unique_mol_names=None,
    ):
        fail = Path(ligand_file).resolve() in self._fail_paths
        return _FakeLigand(ligand_file, output_dir, fail=fail)


def _patch_hashing(monkeypatch, fail_paths: set[Path]):
    # deterministic payload/hash without RDKit
    monkeypatch.setattr(ligand_mod, "_rdkit_load", lambda p, retain_h: Path(p))
    monkeypatch.setattr(
        ligand_mod, "_canonical_payload", lambda mol: f"SMI-{Path(mol).name}"
    )
    # simple hash function
    monkeypatch.setattr(
        ligand_mod,
        "_hash_id",
        lambda payload, ligand_ff, retain_h: f"HASH-{payload}",
    )
    monkeypatch.setattr(
        ligand_mod,
        "LigandFactory",
        lambda: _FakeFactory(fail_paths),
    )


def test_batch_ligand_process_prunes_failed(monkeypatch, tmp_path: Path) -> None:
    lig1 = tmp_path / "a.sdf"
    lig2 = tmp_path / "b.sdf"
    lig1.write_text("fake")
    lig2.write_text("fake")

    fail_set = {lig2}
    _patch_hashing(monkeypatch, fail_set)

    hashes, unique = ligand_mod.batch_ligand_process(
        {"L1": str(lig1), "L2": str(lig2)},
        output_path=tmp_path / "out",
        on_failure="prune",
    )

    assert hashes == ["HASH-SMI-a.sdf"]
    assert set(unique.keys()) == {str(lig1)}


def test_batch_ligand_process_raises_without_prune(monkeypatch, tmp_path: Path) -> None:
    lig = tmp_path / "a.sdf"
    lig.write_text("fake")
    fail_set = {lig}
    _patch_hashing(monkeypatch, fail_set)

    with pytest.raises(RuntimeError, match="boom"):
        ligand_mod.batch_ligand_process(
            {"L1": str(lig)},
            output_path=tmp_path / "out",
            on_failure="raise",
        )
