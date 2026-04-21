from __future__ import annotations

from pathlib import Path

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
