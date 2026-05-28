from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter._internal.ops import build_complex as build_complex_mod


class _FakeAtom:
    def __init__(self, name: str, element: str = "") -> None:
        self.name = name
        self.element = element


class _FakeAtomGroup:
    def __init__(self, atoms: list[_FakeAtom]) -> None:
        self._atoms = atoms

    @property
    def names(self) -> list[str]:
        return [atom.name for atom in self._atoms]

    def __iter__(self):
        return iter(self._atoms)


def test_candidate_ligand_atom_name_string_uses_direct_indices_when_atom_counts_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sdf_file = tmp_path / "lig.sdf"
    sdf_file.write_text("")
    atoms = _FakeAtomGroup(
        [
            _FakeAtom("C1", "C"),
            _FakeAtom("H1", "H"),
            _FakeAtom("C2", "C"),
        ]
    )
    monkeypatch.setattr(
        build_complex_mod,
        "get_ligand_candidates",
        lambda path: [0, 2],
    )
    monkeypatch.setattr(
        build_complex_mod,
        "_sdf_heavy_atom_ordinals",
        lambda path: (3, {0: 0, 2: 1}),
    )

    names = build_complex_mod._candidate_ligand_atom_name_string(
        sdf_file,
        atoms,
        ligand_label="LIG",
        stage="equil",
    )

    assert names == "C1 C2"


def test_candidate_ligand_atom_name_string_maps_sdf_indices_to_heavy_atoms(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sdf_file = tmp_path / "lig.sdf"
    sdf_file.write_text("")
    atoms = _FakeAtomGroup(
        [
            _FakeAtom("C1", "C"),
            _FakeAtom("C2", "C"),
            _FakeAtom("C3", "C"),
        ]
    )
    monkeypatch.setattr(
        build_complex_mod,
        "get_ligand_candidates",
        lambda path: [0, 3, 5],
    )
    monkeypatch.setattr(
        build_complex_mod,
        "_sdf_heavy_atom_ordinals",
        lambda path: (6, {0: 0, 3: 1, 5: 2}),
    )

    names = build_complex_mod._candidate_ligand_atom_name_string(
        sdf_file,
        atoms,
        ligand_label="LIG",
        stage="equil",
    )

    assert names == "C1 C2 C3"


def test_is_apo_ligand_build_reads_param_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "APO.json"
    metadata.write_text(json.dumps({"apo": True}))

    assert build_complex_mod._is_apo_ligand_build(metadata, "APO", "APO")


def test_write_ligand_pdb_with_parameter_names_collapses_apo_dummy(
    tmp_path: Path,
) -> None:
    ligand_pdb = tmp_path / "APO.pdb"
    ligand_pdb.write_text(
        "\n".join(
            [
                "HETATM    1  DU1 lig L   1       0.000   0.000   0.000  0.00  0.00      LIG PB",
                "HETATM    2  DU2 lig L   1       4.000   0.000   0.000  0.00  0.00      LIG PB",
                "HETATM    3  DU3 lig L   1       0.000   4.000   0.000  0.00  0.00      LIG PB",
                "END",
                "",
            ]
        )
    )
    parameter_mol2 = tmp_path / "apo.mol2"
    parameter_mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "LIG",
                "    1     0     1     0     1",
                "SMALL",
                "USER_CHARGES",
                "@<TRIPOS>ATOM",
                "      1 DU1       0.000000    0.000000    0.000000 Pb        1 LIG       0.0000",
                "@<TRIPOS>BOND",
                "@<TRIPOS>SUBSTRUCTURE",
                "      1 LIG         1 ****               0 ****  ****",
                "",
            ]
        )
    )
    output_pdb = tmp_path / "apo_out.pdb"

    build_complex_mod._write_ligand_pdb_with_parameter_names(
        ligand_pdb,
        parameter_mol2,
        output_pdb,
        residue_name="apo",
        ligand_label="APO",
        apo_ligand=True,
    )

    output = output_pdb.read_text()
    assert output.count("HETATM") == 1
    assert "DU1" in output
    assert "DU2" not in output
    assert "DU3" not in output


def test_write_apo_anchor_outputs_tags_fixed_anchor_file(tmp_path: Path) -> None:
    (tmp_path / "equil-APO.pdb").write_text("ATOM\n")
    (tmp_path / "APO-noh.pdb").write_text("ATOM\n")
    (tmp_path / "dum.pdb").write_text(
        "ATOM      1  Pb  DUM D   1       0.000   0.000   0.000  0.00  0.00\nEND\n"
    )

    build_complex_mod._write_apo_anchor_outputs(
        tmp_path,
        ligand="APO",
        mol="APO",
        anchor_names=["DU1"],
    )

    assert not (tmp_path / "anchors.txt").exists()
    assert (tmp_path / "anchors-APO.txt").read_text() == "DU1\n"
    assert (tmp_path / "dum1.pdb").exists()
