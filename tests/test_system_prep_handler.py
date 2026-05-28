from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter.config.utils import apo_ligand_source_path
from batter.exec.handlers import system_prep as system_prep_mod
from batter.exec.handlers.system_prep import (
    _SystemPrepRunner,
    _find_min_xy_box_rotation,
    _select_anchor_reference_ligand,
)
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


def _make_fragmented_protein_pdb(
    path: Path,
    *,
    second_resid: int,
    second_ca_x: float,
) -> None:
    _write_pdb(
        path,
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
            _atom_line(3, "C", "ALA", "A", 1, 1.5, 1.0, 0.0, "C"),
            _atom_line(4, "O", "ALA", "A", 1, 1.5, 2.0, 0.0, "O"),
            _atom_line(5, "N", "ALA", "A", second_resid, second_ca_x - 1.0, 0.0, 0.0, "N"),
            _atom_line(6, "CA", "ALA", "A", second_resid, second_ca_x, 0.0, 0.0, "C"),
            _atom_line(7, "C", "ALA", "A", second_resid, second_ca_x + 1.0, 0.0, 0.0, "C"),
            _atom_line(8, "O", "ALA", "A", second_resid, second_ca_x + 2.0, 0.0, 0.0, "O"),
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


def _make_ligand_pdb_at(path: Path, x: float) -> None:
    _write_pdb(
        path,
        [
            _atom_line(1, "C1", "LIG", "L", 1, x, 0.0, 0.0, "C"),
            _atom_line(2, "C2", "LIG", "L", 1, x + 1.0, 0.0, 0.0, "C"),
        ],
    )


def _make_diagonal_protein_pdb(path: Path) -> None:
    lines: list[str] = []
    serial = 1
    for resid, center in enumerate((0.0, 4.0, 8.0), start=1):
        x = center
        y = center
        z = 0.0
        lines.extend(
            [
                _atom_line(serial, "N", "ALA", "A", resid, x - 1.0, y - 0.5, z, "N"),
                _atom_line(serial + 1, "CA", "ALA", "A", resid, x, y, z, "C"),
                _atom_line(serial + 2, "C", "ALA", "A", resid, x + 1.0, y + 0.5, z, "C"),
                _atom_line(serial + 3, "O", "ALA", "A", resid, x + 1.6, y + 1.2, z, "O"),
            ]
        )
        serial += 4
    _write_pdb(path, lines)


def _make_offset_ligand_pdb(path: Path) -> None:
    _write_pdb(
        path,
        [
            _atom_line(1, "C1", "LIG", "L", 1, 2.5, 1.0, 0.0, "C"),
            _atom_line(2, "C2", "LIG", "L", 1, 3.5, 1.2, 0.0, "C"),
        ],
    )


def _atom_line_with_segid(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
    *,
    segid: str = "",
) -> str:
    line = _atom_line(serial, name, resname, chain, resid, x, y, z, element).rstrip("\n")
    if len(line) < 76:
        line = line.ljust(76)
    return f"{line[:72]}{segid:<4}{line[76:]}\n"


def _make_mixed_segid_protein_pdb(path: Path) -> None:
    _write_pdb(
        path,
        [
            _atom_line_with_segid(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N", segid="P69"),
            _atom_line_with_segid(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C", segid="P69"),
            _atom_line_with_segid(3, "C", "ALA", "A", 1, 1.5, 1.0, 0.0, "C", segid="P69"),
            _atom_line_with_segid(4, "O", "ALA", "A", 1, 1.5, 2.0, 0.0, "O", segid="P69"),
            _atom_line_with_segid(5, "H", "ALA", "A", 1, -0.6, 0.2, 0.0, "H", segid=""),
            _atom_line_with_segid(6, "HA", "ALA", "A", 1, 1.0, -0.8, 0.0, "H", segid=""),
            _atom_line_with_segid(7, "N", "ALA", "A", 2, 2.5, 0.5, 0.0, "N", segid="P69"),
            _atom_line_with_segid(8, "CA", "ALA", "A", 2, 3.5, 1.0, 0.0, "C", segid="P69"),
            _atom_line_with_segid(9, "C", "ALA", "A", 2, 4.5, 0.0, 0.0, "C", segid="P69"),
            _atom_line_with_segid(10, "O", "ALA", "A", 2, 5.5, 0.5, 0.0, "O", segid="P69"),
            _atom_line_with_segid(11, "H", "ALA", "A", 2, 2.0, 0.8, 0.0, "H", segid=""),
            _atom_line_with_segid(12, "HA", "ALA", "A", 2, 3.5, 1.8, 0.0, "H", segid=""),
        ],
    )


def _xy_area(atomgroup) -> float:
    spans = np.ptp(atomgroup.positions, axis=0)
    return float(spans[0] * spans[1])


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


def test_find_min_xy_box_rotation_reduces_diagonal_xy_area() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [4.0, 4.0, 0.0],
            [6.0, 6.0, 0.0],
            [8.0, 8.0, 0.0],
        ],
        dtype=float,
    )

    rotation, before_score, after_score = _find_min_xy_box_rotation(coords)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert after_score[0] < before_score[0] * 0.2


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


def test_run_auto_selects_anchor_atoms_when_omitted(
    monkeypatch, tmp_path: Path
) -> None:
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
    selected = [
        "resid 1 and name CA",
        "resid 2 and name CA",
        "resid 1 and name C",
    ]
    seen = {}

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

    def _fake_select_receptor_anchor_atoms(*args, **kwargs):
        seen["protein_dssp"] = kwargs.get("protein_dssp")
        return selected

    def _fake_find_anchor_atoms(*args, **kwargs):
        seen["anchor_atoms"] = args[3]
        return (1.0, 2.0, 3.0, ":1@CA", ":2@CA", ":1@C", 4.0)

    monkeypatch.setattr(_SystemPrepRunner, "_process_system", _fake_process_system)
    monkeypatch.setattr(
        _SystemPrepRunner, "_prepare_all_ligands", _fake_prepare_all_ligands
    )
    monkeypatch.setattr(
        system_prep_mod,
        "select_receptor_anchor_atoms",
        _fake_select_receptor_anchor_atoms,
    )
    monkeypatch.setattr(system_prep_mod, "find_anchor_atoms", _fake_find_anchor_atoms)

    manifest = runner.run(
        system_name="SYS",
        protein_input=str(protein),
        ligand_paths={"LIG1": str(ligand)},
        anchor_atoms=[],
    )

    assert seen["protein_dssp"] == fake_dssp["results"]
    assert seen["anchor_atoms"] == selected
    assert manifest["anchor_atom_selections"] == selected


def test_select_anchor_reference_ligand_prefers_real_ligand_when_apo_first(
    tmp_path: Path,
) -> None:
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("dummy\n")

    name, is_apo = _select_anchor_reference_ligand(
        ["APO", "PF06882961"],
        {
            "APO": apo_ligand_source_path(),
            "PF06882961": ligand,
        },
    )

    assert name == "PF06882961"
    assert is_apo is False


def test_run_uses_real_ligand_as_anchor_reference_when_apo_is_present(
    monkeypatch, tmp_path: Path
) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)

    protein = tmp_path / "protein.pdb"
    real_ligand = tmp_path / "pf06882961.sdf"
    _make_protein_pdb(protein)
    real_ligand.write_text("dummy\n")

    fake_dssp = {
        "npy": str(system.root / "all-ligands" / "protein_input_dssp.npy"),
        "json": str(system.root / "all-ligands" / "protein_input_dssp.json"),
        "shape": [1, 2],
        "results": [["H", "E"]],
    }
    selected = [
        "resid 1 and name CA",
        "resid 2 and name CA",
        "resid 1 and name C",
    ]
    seen = {}

    monkeypatch.setattr(
        _SystemPrepRunner, "_run_input_protein_dssp", lambda self: fake_dssp
    )
    monkeypatch.setattr(_SystemPrepRunner, "_get_alignment", lambda self: None)

    def _fake_process_system(self) -> None:
        self.ligands_folder.mkdir(parents=True, exist_ok=True)
        _make_protein_pdb(self.ligands_folder / "reference.pdb")
        _make_protein_pdb(self.ligands_folder / f"{self.system_name}.pdb")

    def _fake_prepare_all_ligands(self) -> None:
        apo_out = self.ligands_folder / "APO.pdb"
        real_out = self.ligands_folder / "PF06882961.pdb"
        _make_ligand_pdb_at(apo_out, -10.0)
        _make_ligand_pdb_at(real_out, 10.0)
        self.ligand_dict = {
            "APO": str(apo_out),
            "PF06882961": str(real_out),
        }

    def _fake_select_receptor_anchor_atoms(_u_prot, u_lig, lig_sdf, **kwargs):
        seen["lig_sdf"] = lig_sdf
        seen["ligand_first_x"] = float(u_lig.atoms.positions[0][0])
        seen["protein_dssp"] = kwargs.get("protein_dssp")
        return selected

    def _fake_find_anchor_atoms(_u_prot, u_lig, lig_sdf, anchor_atoms, *_args, **_kwargs):
        seen["find_lig_sdf"] = lig_sdf
        seen["find_ligand_first_x"] = float(u_lig.atoms.positions[0][0])
        seen["anchor_atoms"] = anchor_atoms
        return (1.0, 2.0, 3.0, ":1@CA", ":2@CA", ":1@C", 4.0)

    monkeypatch.setattr(_SystemPrepRunner, "_process_system", _fake_process_system)
    monkeypatch.setattr(
        _SystemPrepRunner, "_prepare_all_ligands", _fake_prepare_all_ligands
    )
    monkeypatch.setattr(
        system_prep_mod,
        "select_receptor_anchor_atoms",
        _fake_select_receptor_anchor_atoms,
    )
    monkeypatch.setattr(
        system_prep_mod,
        "select_apo_receptor_anchor_atoms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("apo anchor selector should not be used")
        ),
    )
    monkeypatch.setattr(system_prep_mod, "find_anchor_atoms", _fake_find_anchor_atoms)

    manifest = runner.run(
        system_name="SYS",
        protein_input=str(protein),
        ligand_paths={
            "APO": str(apo_ligand_source_path()),
            "PF06882961": str(real_ligand),
        },
        anchor_atoms=[],
    )

    assert seen["lig_sdf"] == str(real_ligand)
    assert seen["find_lig_sdf"] == str(real_ligand)
    assert seen["ligand_first_x"] == pytest.approx(10.0)
    assert seen["find_ligand_first_x"] == pytest.approx(10.0)
    assert seen["protein_dssp"] == fake_dssp["results"]
    assert seen["anchor_atoms"] == selected
    assert manifest["anchor_atom_selections"] == selected


def test_get_alignment_reduces_xy_area_and_rotates_ligand_without_system_input(
    tmp_path: Path,
) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)
    runner._system_name = "SYS"
    runner.protein_align = "resid 1 to 3"
    runner.ligands_folder.mkdir(parents=True, exist_ok=True)

    protein = tmp_path / "protein_diagonal.pdb"
    ligand = tmp_path / "ligand_offset.pdb"
    _make_diagonal_protein_pdb(protein)
    _make_offset_ligand_pdb(ligand)

    runner._protein_input = str(protein)
    runner._system_topology = None
    runner._system_input_pdb = str(protein)
    runner.ligand_dict = {"LIG1": str(ligand)}

    runner._get_alignment()
    runner._process_system()
    runner._prepare_all_ligands()

    u_prot_in = mda.Universe(str(protein))
    u_prot_out = mda.Universe(str(runner.ligands_folder / "reference.pdb"))
    u_lig_in = mda.Universe(str(ligand))
    u_lig_out = mda.Universe(str(runner.ligands_folder / "LIG1.pdb"))

    assert _xy_area(u_prot_out.select_atoms("protein")) < _xy_area(
        u_prot_in.select_atoms("protein")
    ) * 0.2

    ca_in = u_prot_in.select_atoms("resid 1 and name CA").positions[0]
    ca_out = u_prot_out.select_atoms("resid 1 and name CA").positions[0]
    lig_c1_in = u_lig_in.atoms.positions[0]
    lig_c1_out = u_lig_out.atoms.positions[0]
    lig_c2_in = u_lig_in.atoms.positions[1]
    lig_c2_out = u_lig_out.atoms.positions[1]

    assert np.linalg.norm(ca_out - lig_c1_out) == pytest.approx(
        np.linalg.norm(ca_in - lig_c1_in), abs=1e-3
    )
    assert np.linalg.norm(ca_out - lig_c2_out) == pytest.approx(
        np.linalg.norm(ca_in - lig_c2_in), abs=1e-3
    )


def test_get_alignment_skips_xy_optimization_when_system_input_is_present(
    monkeypatch, tmp_path: Path
) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)
    runner.protein_align = "resid 1 to 3"
    runner.ligands_folder.mkdir(parents=True, exist_ok=True)

    protein = tmp_path / "protein_diagonal.pdb"
    _make_diagonal_protein_pdb(protein)

    runner._protein_input = str(protein)
    runner._system_topology = str(protein)
    runner._system_input_pdb = str(protein)

    def _fail(*_args, **_kwargs):
        raise AssertionError("XY box optimization should be skipped when system_input is provided.")

    monkeypatch.setattr(system_prep_mod, "_find_min_xy_box_rotation", _fail)

    runner._get_alignment()


def test_get_alignment_normalizes_mixed_protein_segids_before_process_system(
    tmp_path: Path,
) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)
    runner._system_name = "SYS"
    runner.protein_align = "resid 1 to 2"
    runner.ligands_folder.mkdir(parents=True, exist_ok=True)

    protein = tmp_path / "protein_mixed_segid.pdb"
    _make_mixed_segid_protein_pdb(protein)

    runner._protein_input = str(protein)
    runner._system_topology = None
    runner._system_input_pdb = str(protein)

    runner._get_alignment()

    aligned = mda.Universe(str(runner._protein_aligned_pdb))
    assert len(aligned.select_atoms("protein").residues) == 2

    runner._process_system()

    reference = mda.Universe(str(runner.ligands_folder / "reference.pdb"))
    prot = reference.select_atoms("protein")
    assert len(prot.residues) == 2
    assert len(prot.segments) == 1


def _run_process_system_with_fragmented_protein(
    tmp_path: Path,
    monkeypatch,
    *,
    second_resid: int,
    second_ca_x: float,
) -> tuple[_SystemPrepRunner, list[str]]:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)
    runner._system_name = "SYS"
    runner.ligands_folder.mkdir(parents=True, exist_ok=True)

    protein = runner.ligands_folder / "protein_aligned.pdb"
    system_pdb = runner.ligands_folder / "system_aligned.pdb"
    _make_fragmented_protein_pdb(
        protein,
        second_resid=second_resid,
        second_ca_x=second_ca_x,
    )
    _make_fragmented_protein_pdb(
        system_pdb,
        second_resid=second_resid,
        second_ca_x=second_ca_x,
    )

    runner._protein_aligned_pdb = str(protein)
    runner._system_aligned_pdb = str(system_pdb)

    warnings: list[str] = []
    monkeypatch.setattr(system_prep_mod.logger, "warning", lambda message: warnings.append(str(message)))

    runner._process_system()
    return runner, warnings


def test_process_system_splits_protein_on_resid_gap(monkeypatch, tmp_path: Path) -> None:
    runner, warnings = _run_process_system_with_fragmented_protein(
        tmp_path,
        monkeypatch,
        second_resid=3,
        second_ca_x=3.5,
    )

    reference = mda.Universe(str(runner.ligands_folder / "reference.pdb"))
    residues = reference.select_atoms("protein").residues

    assert [int(residue.resid) for residue in residues] == [1, 3]
    assert [residue.atoms.chainIDs[0] for residue in residues] == ["A", "B"]
    assert any("resid discontinuity (1 -> 3)" in warning for warning in warnings)


def test_process_system_splits_protein_on_long_ca_distance(
    monkeypatch, tmp_path: Path
) -> None:
    runner, warnings = _run_process_system_with_fragmented_protein(
        tmp_path,
        monkeypatch,
        second_resid=2,
        second_ca_x=15.5,
    )

    reference = mda.Universe(str(runner.ligands_folder / "reference.pdb"))
    residues = reference.select_atoms("protein").residues

    assert [int(residue.resid) for residue in residues] == [1, 2]
    assert [residue.atoms.chainIDs[0] for residue in residues] == ["A", "B"]
    assert any("C-alpha distance 14.5 A > 10.0 A" in warning for warning in warnings)


def test_process_system_splits_6hty_into_three_segments(
    monkeypatch, tmp_path: Path
) -> None:
    system = SimSystem(name="SYS", root=tmp_path / "run")
    runner = _SystemPrepRunner(system, tmp_path)
    runner._system_name = "SYS"
    runner.ligands_folder.mkdir(parents=True, exist_ok=True)

    sixhty = Path(__file__).resolve().parent / "data" / "6hty.pdb"
    protein = runner.ligands_folder / "protein_aligned.pdb"
    system_pdb = runner.ligands_folder / "system_aligned.pdb"
    protein.write_text(sixhty.read_text())
    system_pdb.write_text(sixhty.read_text())

    runner._protein_aligned_pdb = str(protein)
    runner._system_aligned_pdb = str(system_pdb)

    warnings: list[str] = []
    monkeypatch.setattr(system_prep_mod.logger, "warning", lambda message: warnings.append(str(message)))

    runner._process_system()

    reference = mda.Universe(str(runner.ligands_folder / "reference.pdb"))
    prot = reference.select_atoms("protein")

    assert len(prot.segments) == 3
    assert sorted(set(prot.chainIDs)) == ["A", "B", "C"]
    assert any("resid discontinuity (177 -> 190)" in warning for warning in warnings)
    assert any("resid discontinuity (432 -> 441)" in warning for warning in warnings)
