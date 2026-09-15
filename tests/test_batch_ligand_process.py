from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import batter.param.ligand as ligand_mod


class _FakeLigand:
    def __init__(self, ligand_file: str, output_dir: str, fail: bool = False):
        self.ligand_file = ligand_file
        self.output_dir = output_dir
        self._fail = fail
        self.name = "lig"

    def prepare_ligand_parameters(self) -> None:
        if self._fail:
            raise RuntimeError("boom")
        for suffix in (
            "sdf",
            "mol2",
            "frcmod",
            "lib",
            "prmtop",
            "inpcrd",
            "pdb",
            "json",
        ):
            Path(self.output_dir, f"lig.{suffix}").write_text("ok")


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
    monkeypatch.setattr(
        ligand_mod,
        "_parameterization_payload",
        lambda mol: f"SMI-{Path(mol).name}",
    )
    monkeypatch.setattr(
        ligand_mod,
        "_atom_order_payload",
        lambda mol: f"ORDER-{Path(mol).name}",
    )
    # simple hash function
    monkeypatch.setattr(
        ligand_mod,
        "_hash_id",
        lambda payload, ligand_ff, retain_h, charge_method=None: f"HASH-{payload}",
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


def test_batch_ligand_process_prune_logs_failure_details(
    monkeypatch, tmp_path: Path
) -> None:
    lig1 = tmp_path / "a.sdf"
    lig2 = tmp_path / "b.sdf"
    lig1.write_text("fake")
    lig2.write_text("fake")

    _patch_hashing(monkeypatch, {lig2})
    messages: list[str] = []
    sink_id = ligand_mod.logger.add(
        lambda message: messages.append(str(message)), format="{message}", level="ERROR"
    )
    try:
        ligand_mod.batch_ligand_process(
            {"L1": str(lig1), "L2": str(lig2)},
            output_path=tmp_path / "out",
            on_failure="prune",
        )
    finally:
        ligand_mod.logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert (
        "[param_ligands] failed to prepare L2 (hash=HASH-SMI-b.sdf): boom"
        in rendered
    )
    assert "on_failure=prune" in rendered
    assert "%s" not in rendered


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


def test_batch_ligand_process_reuses_cache(monkeypatch, tmp_path: Path) -> None:
    lig = tmp_path / "a.sdf"
    lig.write_text("fake")
    _patch_hashing(monkeypatch, fail_paths=set())

    out = tmp_path / "out"
    cache_dir = out / "HASH-SMI-a.sdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # create all required files so the ligand is considered cached
    for suffix in (
        "sdf",
        "mol2",
        "frcmod",
        "lib",
        "prmtop",
        "inpcrd",
        "pdb",
        "json",
    ):
        (cache_dir / f"lig.{suffix}").write_text("ok")

    hashes, unique = ligand_mod.batch_ligand_process(
        {"L1": str(lig)},
        output_path=out,
        on_failure="raise",
    )

    assert hashes == ["HASH-SMI-a.sdf"]
    assert set(unique.keys()) == {str(lig)}


def test_batch_ligand_process_does_not_reuse_incomplete_cache(
    monkeypatch, tmp_path: Path
) -> None:
    lig = tmp_path / "a.sdf"
    lig.write_text("fake")
    _patch_hashing(monkeypatch, {lig})

    out = tmp_path / "out"
    cache_dir = out / "HASH-SMI-a.sdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "lig.prmtop").write_text("partial")

    hashes, unique = ligand_mod.batch_ligand_process(
        {"L1": str(lig)},
        output_path=out,
        on_failure="prune",
    )

    assert hashes == []
    assert unique == {}


def test_ligand_parameter_lock_serializes_same_hash_across_processes(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    output_root = tmp_path / "params"
    first_acquired = context.Event()
    release_first = context.Event()
    second_attempting = context.Event()
    second_acquired = context.Event()

    def hold_first_lock() -> None:
        with ligand_mod._ligand_parameter_lock(output_root, "abc123"):
            first_acquired.set()
            assert release_first.wait(timeout=5)

    def acquire_second_lock() -> None:
        assert first_acquired.wait(timeout=5)
        second_attempting.set()
        with ligand_mod._ligand_parameter_lock(output_root, "abc123"):
            second_acquired.set()

    first = context.Process(target=hold_first_lock)
    second = context.Process(target=acquire_second_lock)
    first.start()
    second.start()

    assert second_attempting.wait(timeout=5)
    time.sleep(0.1)
    assert not second_acquired.is_set()

    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert second_acquired.is_set()


def test_parameterization_payload_distinguishes_atom_order_not_coordinates() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("C[NH2+]CCO"))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, Point3D(float(index), 0.0, 0.0))
    mol.AddConformer(conformer)

    coordinate_variant = Chem.Mol(mol)
    coordinate_variant.GetConformer().SetAtomPosition(0, Point3D(99.0, 1.0, 2.0))
    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    assert ligand_mod._canonical_payload(mol) == ligand_mod._canonical_payload(reordered)
    assert ligand_mod._parameterization_payload(mol) == ligand_mod._parameterization_payload(
        coordinate_variant
    )
    assert ligand_mod._parameterization_payload(mol) != ligand_mod._parameterization_payload(
        reordered
    )


def test_parameter_cache_hash_includes_charge_method() -> None:
    payload = ligand_mod._parameterization_payload(
        Chem.AddHs(Chem.MolFromSmiles("CCO"))
    )

    am1bcc = ligand_mod._hash_id(payload, "openff-2.3.0", True, "am1bcc")
    gasteiger = ligand_mod._hash_id(payload, "openff-2.3.0", True, "gasteiger")

    assert am1bcc != gasteiger


def test_batch_ligand_process_separates_reordered_isomorphic_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC[NH2+]CO"))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, Point3D(float(index), 0.0, 0.0))
    mol.AddConformer(conformer)
    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    first = tmp_path / "first.sdf"
    second = tmp_path / "second.sdf"
    for path, molecule in ((first, mol), (second, reordered)):
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()

    monkeypatch.setattr(
        ligand_mod,
        "LigandFactory",
        lambda: _FakeFactory(fail_paths=set()),
    )
    hashes, unique = ligand_mod.batch_ligand_process(
        {"FIRST": first, "SECOND": second},
        output_path=tmp_path / "params",
        ligand_ff="gaff2",
        charge_method="bcc",
    )

    assert unique[str(first)][1] == unique[str(second)][1]
    assert len(hashes) == 2
    assert len(set(hashes)) == 2
