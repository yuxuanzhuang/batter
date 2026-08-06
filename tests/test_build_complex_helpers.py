from __future__ import annotations

import json
from pathlib import Path

import pytest

mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

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


def test_candidate_ligand_atom_name_string_does_not_promote_charge_without_salt_bridge(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    rw_mol = Chem.RWMol()
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    nitrogen = Chem.Atom("N")
    nitrogen.SetFormalCharge(1)
    nitrogen.SetNoImplicit(True)
    nitrogen_idx = rw_mol.AddAtom(nitrogen)
    oxygen_idx = rw_mol.AddAtom(Chem.Atom("O"))
    rw_mol.AddBond(carbon_idx, nitrogen_idx, Chem.BondType.SINGLE)
    rw_mol.AddBond(carbon_idx, oxygen_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(3)
    conformer.SetAtomPosition(carbon_idx, Point3D(0.0, 0.0, 0.0))
    conformer.SetAtomPosition(nitrogen_idx, Point3D(1.0, 0.0, 0.0))
    conformer.SetAtomPosition(oxygen_idx, Point3D(0.0, 1.0, 0.0))
    mol.AddConformer(conformer)
    sdf_file = tmp_path / "charged.sdf"
    Chem.MolToMolFile(mol, str(sdf_file))
    atoms = _FakeAtomGroup(
        [
            _FakeAtom("C1", "C"),
            _FakeAtom("N1", "N"),
            _FakeAtom("O1", "O"),
        ]
    )

    names = build_complex_mod._candidate_ligand_atom_name_string(
        sdf_file,
        atoms,
        ligand_label="LIG",
        stage="equil",
    )

    assert names.split() == ["C1", "N1", "O1"]


def test_initial_pose_salt_bridge_ligand_atom_names_requires_contact(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    pdb = tmp_path / "salt_bridge.pdb"
    pdb.write_text(
        "".join(
            [
                "ATOM      1  OD1 ASP A  10       1.000   0.000   0.000  1.00  0.00           O\n",
                "ATOM      2  OD2 ASP A  10       2.000   0.000   0.000  1.00  0.00           O\n",
                "HETATM    3  N1  LIG L 300       2.500   0.000   0.000  1.00  0.00           N\n",
                "HETATM    4  C1  LIG L 300       8.000   0.000   0.000  1.00  0.00           C\n",
                "TER\n",
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    rw_mol = Chem.RWMol()
    nitrogen = Chem.Atom("N")
    nitrogen.SetFormalCharge(1)
    nitrogen.SetNoImplicit(True)
    nitrogen_idx = rw_mol.AddAtom(nitrogen)
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(nitrogen_idx, Point3D(2.5, 0.0, 0.0))
    conformer.SetAtomPosition(carbon_idx, Point3D(8.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    sdf = tmp_path / "LIG.sdf"
    Chem.MolToMolFile(mol, str(sdf))

    names = build_complex_mod._initial_pose_salt_bridge_ligand_atom_names(
        sdf_file=sdf,
        ligand_atoms=u.select_atoms("resname LIG"),
        protein_atoms=u.select_atoms("protein"),
    )
    no_contact_names = build_complex_mod._initial_pose_salt_bridge_ligand_atom_names(
        sdf_file=sdf,
        ligand_atoms=u.select_atoms("resname LIG"),
        protein_atoms=u.select_atoms("protein"),
        distance_cutoff=0.2,
    )

    assert names == ["N1"]
    assert no_contact_names == []


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


def _pdb_line(
    record: str,
    index: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"{record:<6}{index:5d} {name:^4s} {resname:>3s} {chain:1s}{resid:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def test_lipids_need_charmm_conversion_skips_amber_split_residues(
    tmp_path: Path,
) -> None:
    lipids_pdb = tmp_path / "lipids.pdb"
    lipids_pdb.write_text(
        "".join(
            [
                _pdb_line("HETATM", 1, "C31", "PC", "M", 1, 0.0, 0.0, 0.0),
                _pdb_line("HETATM", 2, "C2", "PA", "M", 2, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 3, "C21", "OL", "M", 3, 2.0, 0.0, 0.0),
                "END\n",
            ]
        )
    )

    assert not build_complex_mod._lipids_need_charmm_to_amber_conversion(
        lipids_pdb
    )


def test_lipids_need_charmm_conversion_detects_charmm_residue(
    tmp_path: Path,
) -> None:
    lipids_pdb = tmp_path / "lipids.pdb"
    lipids_pdb.write_text(
        "".join(
            [
                _pdb_line("HETATM", 1, "C31", "POPC", "M", 1, 0.0, 0.0, 0.0),
                "END\n",
            ]
        )
    )

    assert build_complex_mod._lipids_need_charmm_to_amber_conversion(lipids_pdb)


def test_guard_abfe_boresch_ligand_anchor_names_replaces_endpoint_frame(
    tmp_path: Path,
) -> None:
    fe_pdb = tmp_path / "fe-LIG.pdb"
    fe_pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, -1.0, 0.0, 0.0),
                _pdb_line("ATOM", 3, "CA", "SER", "A", 4, -1.0, 1.0, 0.0),
                _pdb_line("HETATM", 4, "C1", "LIG", "L", 12, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 5, "C2", "LIG", "L", 12, 1.0, 1.0, 0.0),
                _pdb_line("HETATM", 6, "C3", "LIG", "L", 12, 1.0, 1.0, 1.0),
                _pdb_line("HETATM", 7, "C4", "LIG", "L", 12, 0.2, 1.0, 1.1),
                _pdb_line("HETATM", 8, "C5", "LIG", "L", 12, 1.4, 2.1, 0.3),
                _pdb_line("HETATM", 9, "C6", "LIG", "L", 12, 0.6, 0.5, 2.2),
                "END\n",
            ]
        )
    )

    names = build_complex_mod._guard_abfe_boresch_ligand_anchor_names(
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="test",
        P1=":2@CA",
        P2=":3@CA",
        P3=":4@CA",
        lig_resid="12",
        selected_names=["C1", "C2", "C3"],
    )

    assert names == ["C4", "C5", "C6"]

    preferred_names = build_complex_mod._guard_abfe_boresch_ligand_anchor_names(
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="test",
        P1=":2@CA",
        P2=":3@CA",
        P3=":4@CA",
        lig_resid="12",
        selected_names=["C1", "C2", "C3"],
        preferred_first_names=["C4"],
    )

    assert preferred_names[0] == "C4"


def test_pick_ligand_anchor_names_prioritizes_salt_bridge_first_anchor(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, 0.0, 1.0, 0.0),
                _pdb_line("HETATM", 3, "C1", "LIG", "L", 10, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 4, "N1", "LIG", "L", 10, 2.0, 0.0, 0.0),
                _pdb_line("HETATM", 5, "C2", "LIG", "L", 10, 2.0, 1.0, 0.0),
                _pdb_line("HETATM", 6, "C3", "LIG", "L", 10, 2.0, 1.0, 1.0),
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    names = build_complex_mod._pick_ligand_anchor_names(
        u=u,
        mol="LIG",
        ligand_names=["C1", "N1", "C2", "C3"],
        preferred_l1_names=["N1"],
        p1_resid="2",
        p1_atom="CA",
        p2_resid="3",
        p2_atom="CA",
        l1_x=1.0,
        l1_y=0.0,
        l1_z=0.0,
        l1_range=3.0,
        min_adis=0.5,
        max_adis=2.0,
    )

    assert names[0] == "N1"


def test_guard_abfe_boresch_ligand_anchor_names_allows_pdb_resid_mismatch(
    tmp_path: Path,
) -> None:
    fe_pdb = tmp_path / "fe-LIG.pdb"
    fe_pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, -1.0, 0.0, 0.0),
                _pdb_line("ATOM", 3, "CA", "SER", "A", 4, -1.0, 1.0, 0.0),
                _pdb_line("HETATM", 4, "C1", "LIG", "L", 11, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 5, "C2", "LIG", "L", 11, 1.0, 1.0, 0.0),
                _pdb_line("HETATM", 6, "C3", "LIG", "L", 11, 1.0, 1.0, 1.0),
                _pdb_line("HETATM", 7, "C4", "LIG", "L", 11, 0.2, 1.0, 1.1),
                _pdb_line("HETATM", 8, "C5", "LIG", "L", 11, 1.4, 2.1, 0.3),
                _pdb_line("HETATM", 9, "C6", "LIG", "L", 11, 0.6, 0.5, 2.2),
                "END\n",
            ]
        )
    )

    names = build_complex_mod._guard_abfe_boresch_ligand_anchor_names(
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="test",
        P1=":2@CA",
        P2=":3@CA",
        P3=":4@CA",
        lig_resid="12",
        selected_names=["C1", "C2", "C3"],
    )

    assert names == ["C4", "C5", "C6"]


def test_ligand_residue_for_boresch_guard_allows_empty_ligand_resid(
    tmp_path: Path,
) -> None:
    fe_pdb = tmp_path / "fe-LIG.pdb"
    fe_pdb.write_text(
        "".join(
            [
                _pdb_line("HETATM", 1, "C1", "LIG", "L", 12, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 2, "C2", "LIG", "L", 12, 1.0, 1.0, 0.0),
                "END\n",
            ]
        )
    )
    universe = mda.Universe(str(fe_pdb))

    residue = build_complex_mod._ligand_residue_for_boresch_guard(
        universe,
        mol="LIG",
        lig_resid="",
    )

    assert residue is not None
    assert residue.resname == "LIG"
    assert int(residue.resid) == 12
