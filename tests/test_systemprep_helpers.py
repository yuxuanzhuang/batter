from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter.systemprep.helpers import (
    find_anchor_atoms,
    select_apo_receptor_anchor_atoms,
    select_receptor_anchor_atoms,
)


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


def _make_protein(tmp_path: Path) -> mda.Universe:
    pdb = tmp_path / "protein.pdb"
    _write_pdb(
        pdb,
        [
            _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "CB", "ALA", "A", 11, 0.0, 3.0, 0.0, "C"),
            _atom_line(3, "CG", "ALA", "A", 12, 3.0, 0.0, 0.0, "C"),
        ],
    )
    return mda.Universe(str(pdb))


def _make_ligand(tmp_path: Path, xyz: tuple[float, float, float]) -> mda.Universe:
    pdb = tmp_path / "ligand.pdb"
    x, y, z = xyz
    _write_pdb(
        pdb,
        [
            _atom_line(1, "C1", "LIG", "L", 1, x, y, z, "C"),
            _atom_line(2, "C2", "LIG", "L", 1, x + 0.5, y, z, "C"),
        ],
    )
    return mda.Universe(str(pdb))


def test_find_anchor_atoms_checks_unbound_threshold_pass(tmp_path: Path) -> None:
    u_prot = _make_protein(tmp_path)
    u_lig = _make_ligand(tmp_path, (1.0, 1.0, 0.0))

    result = find_anchor_atoms(
        u_prot=u_prot,
        u_lig=u_lig,
        lig_sdf=None,
        anchor_atoms=[
            "resid 10 and name CA",
            "resid 11 and name CB",
            "resid 12 and name CG",
        ],
        unbound_threshold=5.0,
    )

    assert result[3] == ":10@CA"
    assert result[4] == ":11@CB"
    assert result[5] == ":12@CG"
    assert result[6] > 0.0


def test_find_anchor_atoms_checks_unbound_threshold_fail(tmp_path: Path) -> None:
    u_prot = _make_protein(tmp_path)
    u_lig = _make_ligand(tmp_path, (20.0, 20.0, 20.0))

    with pytest.raises(ValueError, match="Ligand appears unbound"):
        find_anchor_atoms(
            u_prot=u_prot,
            u_lig=u_lig,
            lig_sdf=None,
            anchor_atoms=[
                "resid 10 and name CA",
                "resid 11 and name CB",
                "resid 12 and name CG",
            ],
            unbound_threshold=8.0,
        )


def test_find_anchor_atoms_uses_synthetic_vector_for_apo_dummy(tmp_path: Path) -> None:
    u_prot = _make_protein(tmp_path)
    u_lig = _make_ligand(tmp_path, (-150.0, 25.0, -170.0))

    result = find_anchor_atoms(
        u_prot=u_prot,
        u_lig=u_lig,
        lig_sdf=None,
        anchor_atoms=[
            "resid 10 and name CA",
            "resid 11 and name CB",
            "resid 12 and name CG",
        ],
        apo_ligand=True,
        apo_ligand_distance=5.0,
    )

    vector = np.array(result[:3])
    assert np.linalg.norm(vector) == pytest.approx(5.0)
    assert np.linalg.norm(vector - np.array([-150.0, 25.0, -170.0])) > 100.0
    assert result[6] == pytest.approx(6.0)


def test_select_receptor_anchor_atoms_uses_ligand_pose_and_geometry(
    tmp_path: Path,
) -> None:
    protein = tmp_path / "protein_auto.pdb"
    _write_pdb(
        protein,
        [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "CA", "ALA", "A", 2, 0.0, 8.0, 0.0, "C"),
            _atom_line(3, "CA", "ALA", "A", 3, 8.0, 8.0, 0.0, "C"),
            _atom_line(4, "CA", "ALA", "A", 4, -8.0, 8.0, 0.0, "C"),
        ],
    )
    ligand = tmp_path / "ligand_auto.pdb"
    _write_pdb(
        ligand,
        [
            _atom_line(1, "C1", "LIG", "L", 1, 6.0, 0.0, 0.0, "C"),
            _atom_line(2, "C2", "LIG", "L", 1, 6.5, 0.0, 0.0, "C"),
        ],
    )

    selections = select_receptor_anchor_atoms(
        mda.Universe(str(protein)),
        mda.Universe(str(ligand)),
        host_min_distance=0.0,
        host_max_distance=20.0,
        max_candidates=10,
        max_p1_candidates=4,
    )

    assert selections == [
        "protein and resid 1 and name CA",
        "protein and resid 2 and name CA",
        "protein and resid 3 and name CA",
    ]


def test_select_apo_receptor_anchor_atoms_does_not_need_ligand(
    tmp_path: Path,
) -> None:
    protein = tmp_path / "protein_apo_auto.pdb"
    _write_pdb(
        protein,
        [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "CA", "ALA", "A", 2, 0.0, 8.0, 0.0, "C"),
            _atom_line(3, "CA", "ALA", "A", 3, 8.0, 8.0, 0.0, "C"),
            _atom_line(4, "CA", "ALA", "A", 4, -8.0, 8.0, 0.0, "C"),
        ],
    )

    selections = select_apo_receptor_anchor_atoms(
        mda.Universe(str(protein)),
        max_candidates=10,
        max_p1_candidates=4,
    )

    assert selections == [
        "protein and resid 1 and name CA",
        "protein and resid 2 and name CA",
        "protein and resid 3 and name CA",
    ]


def test_select_apo_receptor_anchor_atoms_relaxes_spacing_for_apo_md(
    tmp_path: Path,
) -> None:
    protein = tmp_path / "protein_apo_compact.pdb"
    _write_pdb(
        protein,
        [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "CA", "ALA", "A", 2, 0.0, 3.0, 0.0, "C"),
            _atom_line(3, "CA", "ALA", "A", 3, 3.0, 3.0, 0.0, "C"),
        ],
    )

    selections = select_apo_receptor_anchor_atoms(
        mda.Universe(str(protein)),
        min_anchor_distance=8.0,
        max_candidates=10,
        max_p1_candidates=4,
    )

    assert selections == [
        "protein and resid 1 and name CA",
        "protein and resid 2 and name CA",
        "protein and resid 3 and name CA",
    ]


def test_select_apo_receptor_anchor_atoms_ignores_short_peptide_chain(
    tmp_path: Path,
) -> None:
    protein = tmp_path / "protein_with_peptide.pdb"
    lines: list[str] = []
    serial = 1
    for idx, resid in enumerate(range(9, 40)):
        lines.append(
            _atom_line(
                serial,
                "CA",
                "ALA",
                "A",
                resid,
                -10.0,
                idx * 1.5,
                0.0,
                "C",
            )
        )
        serial += 1
    for idx, resid in enumerate(range(28, 128)):
        lines.append(
            _atom_line(
                serial,
                "CA",
                "ALA",
                "B",
                resid,
                float((idx % 10) * 4),
                float((idx // 10) * 4),
                0.0,
                "C",
            )
        )
        serial += 1
    _write_pdb(protein, lines)
    universe = mda.Universe(str(protein))

    selections = select_apo_receptor_anchor_atoms(
        universe,
        min_anchor_distance=8.0,
        max_candidates=200,
        max_p1_candidates=50,
    )

    selected_chains = [
        universe.select_atoms(selection)[0].chainID for selection in selections
    ]
    assert selected_chains == ["B", "B", "B"]


def test_select_receptor_anchor_atoms_prefers_salt_bridge_for_p1(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    Point3D = pytest.importorskip("rdkit.Geometry").Point3D

    protein = tmp_path / "protein_salt_bridge.pdb"
    _write_pdb(
        protein,
        [
            _atom_line(1, "CA", "LYS", "A", 1, 0.0, 0.0, 0.0, "C"),
            _atom_line(2, "NZ", "LYS", "A", 1, 5.4, 0.0, 0.0, "N"),
            _atom_line(3, "CA", "SER", "A", 2, 0.0, 8.0, 0.0, "C"),
            _atom_line(4, "OG", "SER", "A", 2, 6.0, 0.2, 0.0, "O"),
            _atom_line(5, "CA", "ALA", "A", 3, 8.0, 8.0, 0.0, "C"),
            _atom_line(6, "CA", "ALA", "A", 4, -8.0, 8.0, 0.0, "C"),
        ],
    )

    ligand_pdb = tmp_path / "ligand_salt_bridge.pdb"
    _write_pdb(
        ligand_pdb,
        [
            _atom_line(1, "O1", "LIG", "L", 1, 6.0, 0.0, 0.0, "O"),
            _atom_line(2, "C1", "LIG", "L", 1, 6.5, 0.0, 0.0, "C"),
        ],
    )

    rw_mol = Chem.RWMol()
    oxygen = Chem.Atom("O")
    oxygen.SetFormalCharge(-1)
    oxygen.SetNoImplicit(True)
    oxygen_idx = rw_mol.AddAtom(oxygen)
    carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(oxygen_idx, carbon_idx, Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(oxygen_idx, Point3D(6.0, 0.0, 0.0))
    conformer.SetAtomPosition(carbon_idx, Point3D(6.5, 0.0, 0.0))
    mol.AddConformer(conformer)
    ligand_sdf = tmp_path / "ligand_salt_bridge.sdf"
    Chem.MolToMolFile(mol, str(ligand_sdf))

    selections = select_receptor_anchor_atoms(
        mda.Universe(str(protein)),
        mda.Universe(str(ligand_pdb)),
        lig_sdf=ligand_sdf,
        host_min_distance=0.0,
        host_max_distance=20.0,
        max_candidates=10,
        max_p1_candidates=4,
    )

    assert selections[0] == "protein and resid 1 and name CA"
