from __future__ import annotations

import json
from pathlib import Path

import pytest

mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter._internal.ops import build_complex as build_complex_mod
from batter._internal.ops import restraints as restraints_mod


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


def test_candidate_ligand_atom_name_string_prefers_heavy_ordinals_over_hydrogen_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sdf_file = tmp_path / "lig.sdf"
    sdf_file.write_text("")
    atoms = _FakeAtomGroup(
        [
            _FakeAtom("H10", "H"),
            _FakeAtom("C1", "C"),
            _FakeAtom("H7", "H"),
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
        lambda path: (4, {0: 0, 2: 1}),
    )

    names = build_complex_mod._candidate_ligand_atom_name_string(
        sdf_file,
        atoms,
        ligand_label="LIG",
        stage="equil",
    )

    assert names == "C1 C2"


def test_candidate_ligand_atom_name_string_fallback_uses_heavy_atoms(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sdf_file = tmp_path / "lig.sdf"
    sdf_file.write_text("")
    atoms = _FakeAtomGroup(
        [
            _FakeAtom("H10", "H"),
            _FakeAtom("C1", "C"),
            _FakeAtom("H7", "H"),
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
        lambda path: (4, {}),
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
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{index:5d} {name:^4s} {resname:>3s} {chain:1s}{resid:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def test_pdb4amber_is_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "protein_vmd.pdb"
    output_pdb = tmp_path / "protein.pdb"
    input_pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "GLU", "A", 30, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CB", "GLU", "A", 30, 1.0, 0.0, 0.0),
                _pdb_line("ATOM", 3, "CA", "VAL", "A", 31, 2.0, 0.0, 0.0),
                "END\n",
            ]
        )
    )
    monkeypatch.setattr(build_complex_mod, "_executable_path", lambda cmd: None)

    with pytest.raises(FileNotFoundError, match="pdb4amber is required"):
        build_complex_mod._run_pdb4amber(
            input_pdb,
            output_pdb,
            working_dir=tmp_path,
        )
    assert not output_pdb.exists()


def test_pdb4amber_resolves_from_active_python_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "protein_vmd.pdb"
    output_pdb = tmp_path / "protein.pdb"
    input_pdb.write_text("END\n")
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    python_exe = env_bin / "python"
    python_exe.write_text("")
    pdb4amber = env_bin / "pdb4amber"
    pdb4amber.write_text("#!/bin/sh\n")
    pdb4amber.chmod(0o755)
    commands: list[tuple[str, Path]] = []

    monkeypatch.setattr(build_complex_mod.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(build_complex_mod.sys, "executable", str(python_exe))
    monkeypatch.setattr(
        build_complex_mod,
        "run_with_log",
        lambda command, *, working_dir, **kwargs: commands.append((command, working_dir)),
    )

    build_complex_mod._run_pdb4amber(
        input_pdb,
        output_pdb,
        working_dir=tmp_path,
    )

    assert commands == [
        (f"{pdb4amber} -i protein_vmd.pdb -o protein.pdb -y", tmp_path)
    ]


def test_python_split_preserves_existing_parameter_ligand_pdb(
    tmp_path: Path,
) -> None:
    (tmp_path / "rec_file.pdb").write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "GLU", "A", 30, 0.0, 0.0, 0.0),
                _pdb_line("HETATM", 2, "C1", "lig", "L", 1, 1.0, 0.0, 0.0),
                "END\n",
            ]
        )
    )
    ligand_text = _pdb_line("HETATM", 1, "C1", "adr", "L", 1, 1.0, 0.0, 0.0)
    (tmp_path / "adr.pdb").write_text(ligand_text)

    build_complex_mod._python_split_rec_file(
        workdir=tmp_path,
        mol="adr",
        solv_shell=5.0,
        other_mol=[],
        lipid_mol=[],
    )

    assert (tmp_path / "adr.pdb").read_text() == ligand_text


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

    assert len(names) == 3
    assert names != ["C1", "C2", "C3"]

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


def test_guard_abfe_boresch_anchor_frame_reselects_p2_p3_to_keep_preferred_l1(
    tmp_path: Path,
) -> None:
    fe_pdb = tmp_path / "fe-LIG.pdb"
    fe_pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 2, 0.0, 8.0, 0.0),
                _pdb_line("ATOM", 3, "CA", "SER", "A", 3, 0.0, 16.0, 0.0),
                _pdb_line("ATOM", 4, "CA", "THR", "A", 4, 0.0, 8.0, 8.0),
                _pdb_line("HETATM", 5, "N1", "LIG", "L", 10, 4.0, 0.0, 0.0),
                _pdb_line("HETATM", 6, "C2", "LIG", "L", 10, 4.0, 1.0, 1.0),
                _pdb_line("HETATM", 7, "C3", "LIG", "L", 10, 4.0, 2.0, -1.0),
                _pdb_line("HETATM", 8, "C4", "LIG", "L", 10, 5.0, 0.0, 2.0),
                "END\n",
            ]
        )
    )

    p1, p2, p3, names = build_complex_mod._guard_abfe_boresch_anchor_frame(
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="test",
        P1=":1@CA",
        P2=":2@CA",
        P3=":3@CA",
        lig_resid="10",
        selected_names=["C2", "C3", "C4"],
        preferred_first_names=["N1"],
        allow_receptor_reselection=True,
    )

    assert (p1, p2, p3) == (":1@CA", ":2@CA", ":4@CA")
    assert names[0] == "N1"

    diagnostic = tmp_path / "boresch_anchor_guard.json"
    build_complex_mod._write_abfe_anchor_guard_diagnostic(
        path=diagnostic,
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="test",
        lig_resid="10",
        old_receptor_masks=(":1@CA", ":2@CA", ":3@CA"),
        new_receptor_masks=(p1, p2, p3),
        old_ligand_names=["C2", "C3", "C4"],
        new_ligand_names=names,
        preferred_first_names=["N1"],
        allow_receptor_reselection=True,
        user_anchor_triplet=False,
    )
    data = json.loads(diagnostic.read_text())
    assert data["receptor"]["reselected"]
    assert data["receptor"]["final"]["P3"]["amber_iat"] == 4
    assert data["ligand"]["final"]["L1"]["name"] == "N1"
    assert (
        data["boresch"]["final"]["torsion_margin_deg"]
        >= restraints_mod.BORESCH_MIN_TORSION_MARGIN_DEG
    )


def test_guard_abfe_exhausts_top_l1_before_lower_priority_current_frame(
    tmp_path: Path,
) -> None:
    fe_pdb = tmp_path / "fe-LIG.pdb"
    receptor = [
        ("ASP", 86, 41.537, 30.846, 69.419),
        ("ASP", 52, 37.714, 29.240, 60.074),
        ("ASN", 263, 32.912, 36.243, 60.977),
        ("GLU", 95, 45.368, 38.053, 58.042),
    ]
    ligand = [
        ("C1", 39.644, 33.848, 74.394),
        ("N1", 39.291, 34.463, 73.040),
        ("C2", 40.325, 35.375, 72.493),
        ("C3", 39.770, 36.282, 71.477),
        ("O1", 39.073, 35.613, 70.429),
        ("C4", 40.939, 37.064, 70.918),
        ("C5", 41.358, 36.960, 69.574),
        ("C6", 42.574, 37.597, 69.134),
        ("C7", 43.190, 38.541, 70.042),
        ("O2", 44.351, 39.220, 69.551),
        ("C8", 42.830, 38.596, 71.396),
        ("O3", 43.427, 39.465, 72.253),
        ("C9", 41.708, 37.852, 71.769),
    ]
    lines = [
        _pdb_line("ATOM", index, "CA", resname, "A", resid, x, y, z)
        for index, (resname, resid, x, y, z) in enumerate(receptor, start=1)
    ]
    lines.extend(
        _pdb_line("HETATM", index, name, "LIG", "L", 287, x, y, z)
        for index, (name, x, y, z) in enumerate(ligand, start=len(lines) + 1)
    )
    lines.append("END\n")
    fe_pdb.write_text("".join(lines))

    p1, p2, p3, names = build_complex_mod._guard_abfe_boresch_anchor_frame(
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="adrenaline-like",
        P1=":86@CA",
        P2=":52@CA",
        P3=":263@CA",
        lig_resid="287",
        selected_names=["O3", "C4", "C7"],
        preferred_first_names=["N1", "O1", "O3", "O2"],
        allow_receptor_reselection=True,
    )

    assert (p1, p2, p3) == (":86@CA", ":52@CA", ":95@CA")
    assert names == ["N1", "C4", "C8"]


def test_guard_abfe_boresch_anchor_frame_avoids_terminal_l2_l3(
    tmp_path: Path,
) -> None:
    fe_pdb = tmp_path / "fe-LIG.pdb"
    lines = [
        _pdb_line("ATOM", 1, "CA", "ASP", "A", 86, 41.478, 30.578, 70.156),
        _pdb_line("ATOM", 2, "CA", "ASP", "A", 52, 37.677, 28.284, 61.080),
        _pdb_line("ATOM", 3, "CA", "ASN", "A", 263, 32.855, 35.325, 61.566),
    ]
    ligand_coords = [
        ("C1", 38.491, 33.314, 75.423),
        ("N1", 38.411, 33.918, 74.061),
        ("C2", 39.481, 34.908, 73.686),
        ("C3", 39.297, 35.787, 72.485),
        ("O1", 38.600, 35.092, 71.496),
        ("C4", 40.618, 36.344, 71.919),
        ("C5", 41.007, 36.110, 70.606),
        ("C6", 42.063, 36.826, 70.075),
        ("C7", 42.864, 37.733, 70.931),
        ("O2", 43.958, 38.441, 70.512),
        ("C8", 42.421, 37.885, 72.298),
        ("O3", 43.060, 38.748, 73.186),
        ("C9", 41.277, 37.305, 72.731),
    ]
    for index, (name, x, y, z) in enumerate(ligand_coords, start=4):
        lines.append(_pdb_line("HETATM", index, name, "LIG", "L", 287, x, y, z))
    lines.append("END\n")
    fe_pdb.write_text("".join(lines))

    names = build_complex_mod._guard_abfe_boresch_ligand_anchor_names(
        fe_pdb=fe_pdb,
        mol="LIG",
        ligand_label="adrenaline-like",
        P1=":86@CA",
        P2=":52@CA",
        P3=":263@CA",
        lig_resid="287",
        selected_names=["N1", "C1", "O3"],
        preferred_first_names=["N1"],
    )

    assert names[0] == "N1"
    assert names[1] not in {"C1", "O1", "O2", "O3"}
    assert names[2] not in {"C1", "O1", "O2", "O3"}


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


def test_pick_ligand_anchor_names_relaxes_distances_for_compact_ligand(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "compact_ligand.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, 0.0, 1.0, 0.0),
                _pdb_line("HETATM", 3, "C1", "LIG", "L", 10, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 4, "C2", "LIG", "L", 10, 1.0, 1.0, 0.0),
                _pdb_line("HETATM", 5, "C3", "LIG", "L", 10, 1.0, 1.0, 1.0),
                "END\n",
            ]
        )
    )

    names = build_complex_mod._pick_ligand_anchor_names(
        u=mda.Universe(str(pdb)),
        mol="LIG",
        ligand_names=["C1", "C2", "C3"],
        preferred_l1_names=["C1"],
        p1_resid="2",
        p1_atom="CA",
        p2_resid="3",
        p2_atom="CA",
        l1_x=1.0,
        l1_y=0.0,
        l1_z=0.0,
        l1_range=3.0,
        min_adis=3.0,
        max_adis=7.0,
    )

    assert names == ["C1", "C2", "C3"]


def test_pick_ligand_anchor_names_ignores_hydrogen_candidates(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, 0.0, 1.0, 0.0),
                _pdb_line("HETATM", 3, "H1", "LIG", "L", 10, 1.0, 0.0, 0.0, "H"),
                _pdb_line("HETATM", 4, "C1", "LIG", "L", 10, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 5, "C2", "LIG", "L", 10, 1.0, 1.0, 0.0),
                _pdb_line("HETATM", 6, "C3", "LIG", "L", 10, 1.0, 1.0, 1.0),
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    names = build_complex_mod._pick_ligand_anchor_names(
        u=u,
        mol="LIG",
        ligand_names=["H1", "C1", "C2", "C3"],
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

    assert names == ["C1", "C2", "C3"]


def test_pick_ligand_anchor_names_prioritizes_nonterminal_l2(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, 0.0, 1.0, 0.0),
                _pdb_line("HETATM", 3, "C1", "LIG", "L", 10, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 4, "O1", "LIG", "L", 10, 1.0, 1.0, 0.0, "O"),
                _pdb_line("HETATM", 5, "C2", "LIG", "L", 10, 1.0, -1.0, 1.0),
                _pdb_line("HETATM", 6, "C3", "LIG", "L", 10, 1.0, -2.0, 1.0),
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    names = build_complex_mod._pick_ligand_anchor_names(
        u=u,
        mol="LIG",
        ligand_names=["C1", "O1", "C2", "C3"],
        preferred_l1_names=["C1"],
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

    assert names[0] == "C1"
    assert names[1] == "C2"


def test_pick_ligand_anchor_names_prioritizes_ring_l2_l3(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "aligned_amber.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
                _pdb_line("ATOM", 2, "CA", "GLY", "A", 3, 0.0, 1.0, 0.0),
                _pdb_line("HETATM", 3, "C1", "LIG", "L", 10, 1.0, 0.0, 0.0),
                _pdb_line("HETATM", 4, "B2", "LIG", "L", 10, 1.0, -1.45, 0.0),
                _pdb_line("HETATM", 5, "B3", "LIG", "L", 10, 1.0, -1.45, -1.45),
                _pdb_line("HETATM", 6, "B4", "LIG", "L", 10, 1.0, -1.45, -2.90),
                _pdb_line("HETATM", 7, "R2", "LIG", "L", 10, 1.0, 1.45, 0.0),
                _pdb_line("HETATM", 8, "R3", "LIG", "L", 10, 1.0, 1.45, 1.45),
                _pdb_line("HETATM", 9, "R4", "LIG", "L", 10, 1.0, 0.0, 1.45),
                "END\n",
            ]
        )
    )
    u = mda.Universe(str(pdb))

    names = build_complex_mod._pick_ligand_anchor_names(
        u=u,
        mol="LIG",
        ligand_names=["C1", "B2", "B3", "B4", "R2", "R3", "R4"],
        preferred_l1_names=["C1"],
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

    assert names == ["C1", "R2", "R3"]


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

    assert len(names) == 3
    assert names != ["C1", "C2", "C3"]


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
