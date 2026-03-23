from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from batter.exec.handlers import system_prep as system_prep_mod
from batter.exec.handlers.system_prep import _SystemPrepRunner
from batter.systems.core import SimSystem


def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4}{resname:>4} {chain}{resid:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def _write_pdb(path: Path, lines: list[str]) -> None:
    path.write_text("".join([*lines, "TER\n", "END\n"]))


def _make_protein_pdb(path: Path) -> None:
    _write_pdb(
        path,
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
            _atom_line(3, "C", "ALA", "A", 1, 1.5, 1.0, 0.0, "C"),
            _atom_line(4, "O", "ALA", "A", 1, 1.5, 2.0, 0.0, "O"),
            _atom_line(5, "N", "ALA", "A", 2, 2.5, 0.5, 0.0, "N"),
            _atom_line(6, "CA", "ALA", "A", 2, 3.5, 1.0, 0.0, "C"),
            _atom_line(7, "C", "ALA", "A", 2, 4.5, 0.0, 0.0, "C"),
            _atom_line(8, "O", "ALA", "A", 2, 5.5, 0.5, 0.0, "O"),
        ],
    )


def _make_ligand_pdb(path: Path) -> None:
    _write_pdb(
        path,
        [
            _atom_line(1, "C1", "LIG", "L", 1, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "C2", "LIG", "L", 1, 1.0, 0.0, 0.0, "C"),
        ],
    )


def test_run_input_protein_dssp_persists_results(monkeypatch, tmp_path: Path) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)
    protein = tmp_path / "protein.pdb"
    _make_protein_pdb(protein)
    runner._protein_input = str(protein)
    runner.ligands_folder.mkdir(parents=True, exist_ok=True)

    expected = np.array([["H", "E", "-"]], dtype="<U1")

    class DummyDSSP:
        def __init__(self, _u):
            self.results = {}

        def run(self):
            self.results["dssp"] = expected
            return self

    monkeypatch.setattr(system_prep_mod, "DSSP", DummyDSSP)

    result = runner._run_input_protein_dssp()

    assert result["shape"] == [1, 3]
    assert result["results"] == [["H", "E", "-"]]

    dssp_npy = Path(result["npy"])
    dssp_json = Path(result["json"])
    assert dssp_npy.exists()
    assert dssp_json.exists()

    np.testing.assert_array_equal(np.load(dssp_npy, allow_pickle=False), expected)
    assert json.loads(dssp_json.read_text()) == [["H", "E", "-"]]


def test_run_includes_dssp_in_manifest(monkeypatch, tmp_path: Path) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)

    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.pdb"
    _make_protein_pdb(protein)
    _make_ligand_pdb(ligand)

    fake_dssp = {
        "npy": str(system.root / "all-ligands" / "protein_input_dssp.npy"),
        "json": str(system.root / "all-ligands" / "protein_input_dssp.json"),
        "shape": [1, 2],
        "results": [["H", "E"]],
    }

    monkeypatch.setattr(
        _SystemPrepRunner, "_run_input_protein_dssp", lambda self: fake_dssp
    )
    monkeypatch.setattr(_SystemPrepRunner, "_get_alignment", lambda self: None)

    def _fake_process_system(self) -> None:
        self.ligands_folder.mkdir(parents=True, exist_ok=True)
        _make_protein_pdb(self.ligands_folder / "reference.pdb")
        _make_protein_pdb(self.ligands_folder / f"{self.system_name}.pdb")

    def _fake_prepare_all_ligands(self) -> None:
        out = self.ligands_folder / "LIG1.pdb"
        _make_ligand_pdb(out)
        self.ligand_dict = {"LIG1": str(out)}

    monkeypatch.setattr(_SystemPrepRunner, "_process_system", _fake_process_system)
    monkeypatch.setattr(
        _SystemPrepRunner, "_prepare_all_ligands", _fake_prepare_all_ligands
    )
    monkeypatch.setattr(
        system_prep_mod,
        "find_anchor_atoms",
        lambda *args, **kwargs: (1.0, 2.0, 3.0, ":1@CA", ":2@CA", ":3@CA", 4.0),
    )

    manifest = runner.run(
        system_name="SYS",
        protein_input=str(protein),
        ligand_paths={"LIG1": str(ligand)},
        anchor_atoms=["resid 1 and name CA", "resid 2 and name CA", "resid 1 and name C"],
    )

    assert manifest["dssp"] == fake_dssp
    manifest_path = system.root / "all-ligands" / "manifest.json"
    assert manifest_path.exists()
    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["dssp"] == fake_dssp
