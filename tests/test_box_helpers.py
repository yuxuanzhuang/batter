from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pmd = pytest.importorskip("parmed")
mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter._internal.ops import box
from batter._internal.ops.helpers import (
    merge_first_n_and_lipid_fragments_in_prmtop,
    revised_resids_for_lipid_fragments,
)


def _pdb_atom(
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
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def _format_prmtop_ints(values: list[int], per_line: int = 10) -> str:
    return "\n".join(
        "".join(f"{value:8d}" for value in values[i : i + per_line])
        for i in range(0, len(values), per_line)
    )


def _format_prmtop_names(values: list[str], per_line: int = 20) -> str:
    return "\n".join(
        "".join(f"{value:<4}" for value in values[i : i + per_line])
        for i in range(0, len(values), per_line)
    )


def test_first_atom_position_uses_first_atom_not_center_of_mass(tmp_path: Path) -> None:
    pdb = tmp_path / "lig.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "C1", "LIG", "A", 1, 1.0, 2.0, 3.0, "C"),
                _pdb_atom(2, "C2", "LIG", "A", 1, 7.0, 8.0, 9.0, "C"),
                "END\n",
            ]
        )
    )
    universe = mda.Universe(str(pdb))

    np.testing.assert_allclose(
        box._first_atom_position(universe.select_atoms("resname LIG")),
        [1.0, 2.0, 3.0],
    )


def _write_minimal_prmtop(
    path: Path,
    *,
    atoms_per_molecule: list[int],
    residue_labels: list[str],
    residue_pointers: list[int],
    solvent_pointers: list[int],
) -> None:
    path.write_text(
        "\n".join(
            [
                "%FLAG RESIDUE_LABEL",
                "%FORMAT(20a4)",
                _format_prmtop_names(residue_labels),
                "%FLAG RESIDUE_POINTER",
                "%FORMAT(10I8)",
                _format_prmtop_ints(residue_pointers),
                "%FLAG ATOMS_PER_MOLECULE",
                "%FORMAT(10I8)",
                _format_prmtop_ints(atoms_per_molecule),
                "%FLAG SOLVENT_POINTERS",
                "%FORMAT(3I8)",
                _format_prmtop_ints(solvent_pointers, per_line=3),
                "",
            ]
        )
    )


def _read_prmtop_int_flag(path: Path, flag: str) -> list[int]:
    lines = path.read_text().splitlines()
    start = lines.index(f"%FLAG {flag}") + 2
    end = start
    while end < len(lines) and not lines[end].startswith("%FLAG"):
        end += 1
    values: list[int] = []
    for line in lines[start:end]:
        for i in range(0, len(line), 8):
            chunk = line[i : i + 8].strip()
            if chunk:
                values.append(int(chunk))
    return values


def test_repair_parmed_molecule_table_handles_bad_standalone_ligand() -> None:
    data_dir = Path(__file__).resolve().parent / "data" / "ligand_params" / "ea7f6bcb5854"
    parm = pmd.load_file(str(data_dir / "lig.prmtop"), str(data_dir / "lig.pdb"))
    parm.parm_data["SOLVENT_POINTERS"] = [0, 0, 0]
    parm.parm_data["ATOMS_PER_MOLECULE"] = [0]

    repaired = box._repair_parmed_molecule_table_for_combine(parm)

    assert repaired is parm
    assert parm.parm_data["SOLVENT_POINTERS"] == [1, 1, 2]
    assert parm.parm_data["ATOMS_PER_MOLECULE"] == [len(parm.atoms)]
    assert len(copy.copy(parm).atoms) == len(parm.atoms)


def test_merge_first_n_and_lipid_fragments_groups_split_popc(tmp_path: Path) -> None:
    src = tmp_path / "full.prmtop"
    out = tmp_path / "full_merged.prmtop"
    _write_minimal_prmtop(
        src,
        atoms_per_molecule=[1, 1, 10, 5, 5, 46, 38, 50, 3],
        residue_labels=["DUM", "DUM", "PRO", "LIG", "LIG", "PA", "PC", "OL", "WAT"],
        residue_pointers=[1, 2, 3, 13, 18, 23, 69, 107, 157],
        solvent_pointers=[9, 9, 9],
    )

    merge_first_n_and_lipid_fragments_in_prmtop(
        src.as_posix(),
        5,
        ["POPC"],
        out.as_posix(),
    )

    assert _read_prmtop_int_flag(out, "ATOMS_PER_MOLECULE") == [22, 134, 3]
    assert _read_prmtop_int_flag(out, "SOLVENT_POINTERS") == [9, 3, 3]


def test_merge_first_n_and_lipid_fragments_leaves_grouped_popc_alone(
    tmp_path: Path,
) -> None:
    src = tmp_path / "full.prmtop"
    out = tmp_path / "full_merged.prmtop"
    _write_minimal_prmtop(
        src,
        atoms_per_molecule=[1, 1, 10, 5, 5, 134, 3],
        residue_labels=["DUM", "DUM", "PRO", "LIG", "LIG", "PA", "PC", "OL", "WAT"],
        residue_pointers=[1, 2, 3, 13, 18, 23, 69, 107, 157],
        solvent_pointers=[9, 7, 7],
    )

    merge_first_n_and_lipid_fragments_in_prmtop(
        src.as_posix(),
        5,
        ["POPC"],
        out.as_posix(),
    )

    assert _read_prmtop_int_flag(out, "ATOMS_PER_MOLECULE") == [22, 134, 3]
    assert _read_prmtop_int_flag(out, "SOLVENT_POINTERS") == [9, 3, 3]


def test_revised_resids_for_lipid_fragments_groups_split_popc() -> None:
    records = [
        ("ALA", "A", 10),
        ("LIG", "L", 220),
        ("PA", "X", 289),
        ("PC", "X", 290),
        ("OL", "X", 291),
        ("PA", "X", 292),
        ("PC", "X", 293),
        ("OL", "X", 294),
        ("WAT", "X", 295),
    ]

    assert revised_resids_for_lipid_fragments(records, ["POPC"]) == [
        1,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
    ]
    assert revised_resids_for_lipid_fragments(records, ["PC", "PA", "OL"]) == [
        1,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
    ]


def test_restore_reference_hydrogen_coordinates_repairs_existing_lipid_protons(
    tmp_path: Path,
) -> None:
    target = tmp_path / "full_pre.pdb"
    reference = tmp_path / "equil-reference.pdb"
    target.write_text(
        "".join(
            [
                _pdb_atom(1, "C31", "PC", "A", 365, 0.000, 0.000, 0.000, "C"),
                _pdb_atom(2, "C32", "PC", "A", 365, 1.500, 0.000, 0.000, "C"),
                _pdb_atom(3, "N31", "PC", "A", 365, 1.500, 1.400, 0.000, "N"),
                _pdb_atom(4, "C33", "PC", "A", 365, 1.500, 0.000, 1.400, "C"),
                _pdb_atom(5, "H2A", "PC", "A", 365, 0.560, 0.000, 0.000, "H"),
                _pdb_atom(6, "H2B", "PC", "A", 365, 0.600, 0.100, 0.000, "H"),
                _pdb_atom(7, "O", "WAT", "A", 366, 5.000, 0.000, 0.000, "O"),
                _pdb_atom(8, "H1", "WAT", "A", 366, 5.100, 0.000, 0.000, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )
    reference.write_text(
        "".join(
            [
                _pdb_atom(1, "C31", "PC", "X", 99, 0.000, 0.000, 0.000, "C"),
                _pdb_atom(2, "C32", "PC", "X", 99, 1.500, 0.000, 0.000, "C"),
                _pdb_atom(3, "N31", "PC", "X", 99, 1.500, 1.400, 0.000, "N"),
                _pdb_atom(4, "C33", "PC", "X", 99, 1.500, 0.000, 1.400, "C"),
                _pdb_atom(5, "H2A", "PC", "X", 99, 2.500, -0.300, -0.300, "H"),
                _pdb_atom(6, "H2B", "PC", "X", 99, 1.100, -0.800, 0.300, "H"),
                _pdb_atom(7, "O", "WAT", "X", 100, 7.000, 0.000, 0.000, "O"),
                _pdb_atom(8, "H1", "WAT", "X", 100, 7.900, 0.000, 0.000, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )

    restored = box._restore_reference_hydrogen_coordinates(target, reference)

    assert restored == 2
    atoms = {}
    for line in target.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() == "PC":
            atoms[line[12:16].strip()] = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    np.testing.assert_allclose(atoms["H2A"], [2.5, -0.3, -0.3], atol=1.0e-3)
    assert np.linalg.norm(atoms["H2A"] - atoms["C31"]) > 2.0
    assert np.linalg.norm(atoms["H2A"] - atoms["C32"]) == pytest.approx(1.086, abs=1.0e-3)

    water_h1 = next(
        line
        for line in target.read_text().splitlines()
        if line.startswith("ATOM") and line[17:20].strip() == "WAT" and line[12:16].strip() == "H1"
    )
    assert float(water_h1[30:38]) == pytest.approx(5.1)


def test_abfe_diff_charge_ligand_uses_second_pre_fe_ligand(tmp_path: Path) -> None:
    (tmp_path / "ref_vac.pdb").write_text(
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  LIG A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "TER\n"
        "ATOM      3  C1  LIG A   2      10.000  20.000  30.000  1.00  0.00           C\n"
        "ATOM      4  C2  LIG A   2      11.000  21.000  31.000  1.00  0.00           C\n"
        "TER\n"
        "END\n"
    )

    out = box._write_abfe_diff_charge_ligand_from_ref_vac(tmp_path, "LIG")

    u = mda.Universe(out.as_posix())
    np.testing.assert_allclose(
        u.atoms.positions,
        np.asarray([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]], dtype=float),
        atol=1.0e-3,
    )


def test_membrane_water_chunks_tile_reference_waters_to_cover_expanded_z(
    tmp_path: Path,
) -> None:
    build_pdb = tmp_path / "build.pdb"
    build_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "C1", "LIG", "A", 1, 0.000, 0.000, 0.000, "C"),
                "TER\n",
                _pdb_atom(2, "O", "WAT", "W", 2, 1.000, 1.000, 1.000, "O"),
                _pdb_atom(3, "H1", "WAT", "W", 2, 1.100, 1.000, 1.000, "H"),
                _pdb_atom(4, "H2", "WAT", "W", 2, 1.000, 1.100, 1.000, "H"),
                "TER\n",
                _pdb_atom(5, "O", "WAT", "W", 3, 2.000, 2.000, 8.000, "O"),
                _pdb_atom(6, "H1", "WAT", "W", 3, 2.100, 2.000, 8.000, "H"),
                _pdb_atom(7, "H2", "WAT", "W", 3, 2.000, 2.100, 8.000, "H"),
                "TER\n",
                "END\n",
            ]
        )
    )

    chunks = box._write_membrane_water_chunks_from_build(
        tmp_path,
        ligand_resname="LIG",
        box=[10.0, 10.0, 15.0],
        z_max=15.0,
        reference_z_period=10.0,
    )

    assert [path.name for path in chunks] == ["solvate_pre_wat_00.pdb"]
    oxygen_z = [
        float(line[46:54])
        for line in chunks[0].read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "O"
    ]
    assert sorted(oxygen_z) == pytest.approx([1.0, 8.0, 11.0])
    assert max(oxygen_z) <= 15.0


def test_make_residues_nonsteric_adds_private_zero_lj_type() -> None:
    data_dir = Path(__file__).resolve().parent / "data" / "ligand_params" / "ea7f6bcb5854"
    first = pmd.load_file(str(data_dir / "lig.prmtop"), str(data_dir / "lig.pdb"))
    second = pmd.load_file(str(data_dir / "lig.prmtop"), str(data_dir / "lig.pdb"))
    combined = first + second

    original_ntypes = combined.ptr("ntypes")
    original_charge = sum(atom.charge for atom in combined.residues[1].atoms)

    box._make_residues_nonsteric(combined, [1])

    first_atom = combined.residues[0].atoms[0]
    duplicate_atom = combined.residues[1].atoms[0]

    assert combined.ptr("ntypes") == original_ntypes + 1
    assert first_atom.epsilon > 0
    assert first_atom.rmin > 0
    assert duplicate_atom.epsilon == 0
    assert duplicate_atom.rmin == 0
    assert sum(atom.charge for atom in combined.residues[1].atoms) == pytest.approx(
        original_charge
    )


def test_split_structure_nonwater_then_water_keeps_ions_in_vacuum_prefix(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "full.pdb"
    pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "C1", "LIG", "A", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_atom(2, "NA", "Na+", "A", 2, 1.0, 0.0, 0.0, "Na"),
                _pdb_atom(3, "O", "WAT", "A", 3, 2.0, 0.0, 0.0, "O"),
                _pdb_atom(4, "H1", "WAT", "A", 3, 2.1, 0.0, 0.0, "H"),
                _pdb_atom(5, "CL", "Cl-", "A", 4, 3.0, 0.0, 0.0, "Cl"),
                "TER\n",
                "END\n",
            ]
        )
    )
    structure = pmd.load_file(str(pdb))

    nonwater, water, reordered = box._split_structure_nonwater_then_water(structure)

    assert [atom.residue.name for atom in nonwater.atoms] == ["LIG", "Na+", "Cl-"]
    assert [atom.residue.name for atom in water.atoms] == ["WAT", "WAT"]
    assert [atom.residue.name for atom in reordered.atoms] == [
        "LIG",
        "Na+",
        "Cl-",
        "WAT",
        "WAT",
    ]


def test_save_pre_ring_repair_snapshots_writes_unrepaired_coordinates(
    tmp_path: Path,
) -> None:
    class DummyStructure:
        def __init__(self, coordinates) -> None:
            self.coordinates = np.asarray(coordinates, dtype=float)
            self.saved = []

        def save(
            self,
            path: str,
            *,
            format: str | None = None,
            overwrite: bool = False,
        ) -> None:
            self.saved.append(
                {
                    "name": Path(path).name,
                    "format": format,
                    "overwrite": overwrite,
                    "coordinates": np.asarray(self.coordinates, dtype=float).copy(),
                }
            )

    vac_pre = np.asarray([[1.0, 2.0, 3.0]])
    vac_repaired = np.asarray([[4.0, 5.0, 6.0]])
    full_pre = np.asarray([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    full_repaired = np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    vac = DummyStructure(vac_repaired)
    full = DummyStructure(full_repaired)

    files = box._save_pre_ring_repair_snapshots(
        tmp_path,
        vac=vac,
        vac_coordinates=vac_pre,
        combined=full,
        combined_coordinates=full_pre,
    )

    assert files == {
        "full_inpcrd": "full.inpcrd.pre_ring_repair",
        "full_pdb": "full.pdb.pre_ring_repair",
        "vac_inpcrd": "vac.inpcrd.pre_ring_repair",
        "vac_pdb": "vac.pdb.pre_ring_repair",
    }
    assert [item["name"] for item in vac.saved] == [
        "vac.inpcrd.pre_ring_repair",
        "vac.pdb.pre_ring_repair",
    ]
    assert [item["name"] for item in full.saved] == [
        "full.inpcrd.pre_ring_repair",
        "full.pdb.pre_ring_repair",
    ]
    assert [item["format"] for item in vac.saved] == ["rst7", "pdb"]
    assert [item["format"] for item in full.saved] == ["rst7", "pdb"]
    assert all(item["overwrite"] for item in [*vac.saved, *full.saved])
    for item in vac.saved:
        np.testing.assert_allclose(item["coordinates"], vac_pre)
    for item in full.saved:
        np.testing.assert_allclose(item["coordinates"], full_pre)
    np.testing.assert_allclose(vac.coordinates, vac_repaired)
    np.testing.assert_allclose(full.coordinates, full_repaired)


def test_ligand_charge_from_metadata_rounds_and_handles_missing(tmp_path: Path) -> None:
    meta = tmp_path / "lig.json"
    meta.write_text(json.dumps({"ligand_charge": -1.6}))

    assert box._ligand_charge_from_metadata(meta) == -2
    assert box._ligand_charge_from_metadata(tmp_path / "missing.json") is None


def test_read_disulfide_pairs_deduplicates_and_ignores_empty_lines(tmp_path: Path) -> None:
    sslink = tmp_path / "build_amber_sslink"
    sslink.write_text("\n19 44\n44 19\n35 77\n")

    assert box._read_disulfide_pairs(sslink) == [(19, 44), (35, 77)]
    assert box._read_disulfide_pairs(tmp_path / "missing_sslink") == []


def test_map_disulfide_pairs_to_revised_resids() -> None:
    assert box._map_disulfide_pairs_to_resids(
        [(2, 4), (6, 7)], [10, 11, 12, 13, 14, 15, 16]
    ) == [(11, 13), (15, 16)]


def test_merge_disulfide_pairs_deduplicates_sorted_pairs() -> None:
    assert box._merge_disulfide_pairs(
        [(44, 19), (35, 77)], [(19, 44), (80, 90)]
    ) == [(19, 44), (35, 77), (80, 90)]


def test_infer_cyx_disulfide_pairs_from_atoms(tmp_path: Path) -> None:
    pdb = tmp_path / "cyx.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   CYX A  19       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  SG  CYX A  19       0.000   0.000   0.000  1.00  0.00           S",
                "ATOM      3  N   CYX A  44       1.900   0.000   0.000  1.00  0.00           N",
                "ATOM      4  SG  CYX A  44       2.000   0.000   0.000  1.00  0.00           S",
                "ATOM      5  N   CYX A  80      10.000   0.000   0.000  1.00  0.00           N",
                "ATOM      6  SG  CYX A  80      10.000   0.000   0.000  1.00  0.00           S",
                "ATOM      7  N   CYS A  81      12.000   0.000   0.000  1.00  0.00           N",
                "ATOM      8  SG  CYS A  81      12.000   0.000   0.000  1.00  0.00           S",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    universe = mda.Universe(str(pdb))

    assert box._infer_cyx_disulfide_pairs_from_atoms(universe.atoms) == [(19, 44)]


def test_write_leap_disulfide_bonds() -> None:
    class Sink:
        def __init__(self) -> None:
            self.text = ""

        def write(self, value: str) -> None:
            self.text += value

    sink = Sink()
    box._write_leap_disulfide_bonds(sink, "prot", [(19, 44), (35, 77)])

    assert sink.text == (
        "bond prot.19.SG prot.44.SG\n"
        "bond prot.35.SG prot.77.SG\n"
        "\n"
    )


def test_map_disulfide_pairs_to_leap_indices_after_inserted_cap(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ASP A   2       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ASP A   2       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   SER A   3       2.000   0.000   0.000  1.00  0.00           C",
                "ATOM      4  N   NHE A   9       3.000   0.000   0.000  1.00  0.00           N",
                "TER",
                "ATOM      5  N   CYX B   4       4.000   0.000   0.000  1.00  0.00           N",
                "ATOM      6  SG  CYX B   4       5.000   0.000   0.000  1.00  0.00           S",
                "ATOM      7  N   CYX B   5       6.000   0.000   0.000  1.00  0.00           N",
                "ATOM      8  SG  CYX B   5       7.000   0.000   0.000  1.00  0.00           S",
                "TER",
            ]
        )
        + "\n"
    )

    assert box._map_disulfide_pairs_to_leap_indices([(4, 5)], pdb) == [(5, 6)]


def test_sync_ligand_anchor_residue_with_pdb_updates_masks(tmp_path: Path) -> None:
    (tmp_path / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":198@CA",
                "P2": ":241@CA",
                "P3": ":190@CA",
                "L1": ":426@DU1",
                "L2": None,
                "L3": None,
                "lig_res": "426",
            }
        )
    )
    pdb = tmp_path / "vac.pdb"
    pdb.write_text(
        "HETATM    1  DU1 apo   427       0.000   0.000   0.000  0.00  0.00           P\n"
    )

    box._sync_ligand_anchor_residue_with_pdb(tmp_path, pdb, "apo")

    data = json.loads((tmp_path / "anchors.json").read_text())
    assert data["P1"] == ":198@CA"
    assert data["L1"] == ":427@DU1"
    assert data["lig_res"] == "427"


def test_sync_ligand_anchor_residue_updates_build_dir_after_cap_insert(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "z_build_files"
    window_dir = tmp_path / "z-1"
    build_dir.mkdir()
    window_dir.mkdir()
    (build_dir / "anchors.json").write_text(
        json.dumps(
            {
                "P1": ":351@CA",
                "P2": ":356@CA",
                "P3": ":111@CA",
                "L1": ":397@C4",
                "L2": ":397@C14",
                "L3": ":397@C21",
                "lig_res": "397",
            }
        )
    )
    (window_dir / "vac.pdb").write_text(
        "\n".join(
            [
                "ATOM      1  N   NME   397       0.000   0.000   0.000  1.00  0.00           N",
                "HETATM    2  C4  ooo   398       1.000   0.000   0.000  1.00  0.00           C",
                "HETATM    3  C14 ooo   398       2.000   0.000   0.000  1.00  0.00           C",
                "HETATM    4  C21 ooo   398       3.000   0.000   0.000  1.00  0.00           C",
                "HETATM    5  C4  ooo   399       4.000   0.000   0.000  1.00  0.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    box._sync_ligand_anchor_residue_with_pdb(
        build_dir, window_dir / "vac.pdb", "ooo"
    )

    data = json.loads((build_dir / "anchors.json").read_text())
    assert data["L1"] == ":398@C4"
    assert data["L2"] == ":398@C14"
    assert data["L3"] == ":398@C21"
    assert data["lig_res"] == "398"


def test_mark_disulfide_residue_names_and_filter_thiol_hydrogen() -> None:
    class Residue:
        def __init__(self, resid: int, resname: str) -> None:
            self.resid = resid
            self.resname = resname

    residues = [Residue(19, "CYS"), Residue(44, "CYX"), Residue(80, "CYS")]
    box._mark_disulfide_residue_names(residues, {19, 44})

    assert [res.resname for res in residues] == ["CYX", "CYX", "CYS"]

    line = "ATOM      1  HG  CYX A  19      -2.808 -21.114  19.366  1.00  0.00           H"
    assert box._is_disulfide_thiol_hydrogen_line(line, {19})
    assert not box._is_disulfide_thiol_hydrogen_line(line, {44})


def test_normalize_hybrid36_resids_for_mdanalysis(tmp_path: Path) -> None:
    pdb = tmp_path / "full_pre.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM  79139  O   WAT  A6VA     -20.656 -27.070 -58.884  1.00  0.00           O",
                "ATOM  79140  H1  WAT  A6VA     -20.022 -27.240 -58.310  1.00  0.00           H",
                "ATOM  79143  O   WAT  A6VB     -23.245 -26.959 -57.529  1.00  0.00           O",
                "TER   ",
                "END   ",
            ]
        )
        + "\n"
    )

    normalized = box._normalize_hybrid36_resids_for_mdanalysis(pdb)

    assert normalized is not None
    try:
        lines = normalized.read_text().splitlines()
        assert lines[0][22:26] == "8902"
        assert lines[1][22:26] == "8902"
        assert lines[2][22:26] == "8903"
        assert "A6VA" not in normalized.read_text()
        assert "A6VB" not in normalized.read_text()
    finally:
        normalized.unlink(missing_ok=True)


def test_normalize_hybrid36_resids_leaves_decimal_pdb_unchanged(tmp_path: Path) -> None:
    pdb = tmp_path / "full_pre.pdb"
    pdb.write_text(
        "ATOM  79330  O   WAT  18952    -23.297 -26.959 -57.529  1.00  0.00\n"
    )

    assert box._normalize_hybrid36_resids_for_mdanalysis(pdb) is None


def test_normalize_decimal_overflow_resids_for_mdanalysis(tmp_path: Path) -> None:
    pdb = tmp_path / "full_pre.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM  420846 O   WAT  100000    -45.500  -8.240 -66.839  1.00  0.00",
                "ATOM  420847 H1  WAT  100000    -45.069  -8.369 -67.586  1.00  0.00",
                "ATOM  420850 O   WAT  100001    -39.154 -14.676 -73.189  1.00  0.00",
                "ATOM  422910 O   WAT  100516    -35.697 -17.747-108.298  1.00  0.00",
                "TER   ",
                "END   ",
            ]
        )
        + "\n"
    )

    normalized = box._normalize_hybrid36_resids_for_mdanalysis(pdb)

    assert normalized is not None
    try:
        lines = normalized.read_text().splitlines()
        assert lines[0][22:26] == "0000"
        assert lines[1][22:26] == "0000"
        assert lines[2][22:26] == "0001"
        assert lines[3][22:26] == "0516"
        for line in lines[:4]:
            float(line[30:38])
            float(line[38:46])
            float(line[46:54])
        universe = mda.Universe(str(normalized))
        assert universe.atoms.n_atoms == 4
        assert universe.atoms.positions[0][1] == pytest.approx(-8.240)
        assert universe.atoms.positions[3][2] == pytest.approx(-108.298)
    finally:
        normalized.unlink(missing_ok=True)


def test_rewrite_terminal_amide_caps_for_leap(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   SER A   7       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  SER A   7       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   SER A   7       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   SER A   7       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  OXT SER A   7       2.500   1.000   0.000  1.00  0.00           O",
                "ATOM      6  N1  SER A   7       2.000   0.500   0.000  1.00  0.00           N",
                "ATOM      7  H1  SER A   7       2.500   0.000   0.000  1.00  0.00           H",
                "ATOM      8  H2  SER A   7       2.500   1.000   0.000  1.00  0.00           H",
                "TER",
                "ATOM      9  N   THR B   8       4.000   0.000   0.000  1.00  0.00           N",
                "ATOM     10  CA  THR B   8       5.000   0.000   0.000  1.00  0.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_terminal_amide_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " OXT " not in text
    assert " N1  SER" not in text
    assert " N   NHE A   9" in text
    assert " HN1 NHE A   9" in text
    assert " HN2 NHE A   9" in text
    assert text.index(" N   NHE") < text.index("TER")


def test_rewrite_embedded_terminal_methylamide_cap_for_leap(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   SER A   7       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  SER A   7       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   SER A   7       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   SER A   7       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  OXT SER A   7       2.500   1.000   0.000  1.00  0.00           O",
                "ATOM      6  N1  SER A   7       2.000   0.500   0.000  1.00  0.00           N",
                "ATOM      7  H1  SER A   7       2.500   0.000   0.000  1.00  0.00           H",
                "ATOM      8  C1  SER A   7       2.500   1.000   0.000  1.00  0.00           C",
                "ATOM      9  H2  SER A   7       3.000   0.500   0.000  1.00  0.00           H",
                "ATOM     10  H3  SER A   7       3.000   1.500   0.000  1.00  0.00           H",
                "ATOM     11  H4  SER A   7       2.500   1.000   1.000  1.00  0.00           H",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_terminal_amide_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " OXT " not in text
    assert " N1  SER" not in text
    assert " N   NME A   8" in text
    assert " H   NME A   8" in text
    assert " C   NME A   8" in text
    assert " H1  NME A   8" in text
    assert " H2  NME A   8" in text
    assert " H3  NME A   8" in text
    assert text.index(" N   NME") < text.index("TER")


def test_rewrite_terminal_nma_residue_as_nme_for_leap(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   LYS A 163       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  LYS A 163       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   LYS A 163       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   LYS A 163       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  OXT LYS A 163       2.500   1.000   0.000  1.00  0.00           O",
                "HETATM    6  N   NMA A 164       2.000   0.500   0.000  1.00  0.00           N",
                "HETATM    7  CA  NMA A 164       2.500   1.000   0.000  1.00  0.00           C",
                "HETATM    8  H   NMA A 164       2.500   0.000   0.000  1.00  0.00           H",
                "HETATM    9 1HA  NMA A 164       3.000   0.500   0.000  1.00  0.00           H",
                "HETATM   10 2HA  NMA A 164       3.000   1.500   0.000  1.00  0.00           H",
                "HETATM   11 3HA  NMA A 164       2.500   1.000   1.000  1.00  0.00           H",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_terminal_amide_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " OXT " not in text
    assert " NMA " not in text
    assert " N   NME A 164" in text
    assert " C   NME A 164" in text
    assert " H   NME A 164" in text
    assert " H1  NME A 164" in text
    assert " H2  NME A 164" in text
    assert " H3  NME A 164" in text


def test_rewrite_terminal_nme_drops_duplicate_methyl_aliases(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ARG B 426       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ARG B 426       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   ARG B 426       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   ARG B 426       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  N   NME B 426       2.000   0.500   0.000  1.00  0.00           N",
                "ATOM      6  H   NME B 426       2.500   0.000   0.000  1.00  0.00           H",
                "ATOM      7  C   NME B 426       2.500   1.000   0.000  1.00  0.00           C",
                "ATOM      8  H1  NME B 426       3.000   0.500   0.000  1.00  0.00           H",
                "ATOM      9  H2  NME B 426       3.000   1.500   0.000  1.00  0.00           H",
                "ATOM     10  H3  NME B 426       2.500   1.000   1.000  1.00  0.00           H",
                "ATOM     11  CH3 NME B 426       2.510   1.010   0.010  1.00  0.00           C",
                "ATOM     12 HH31 NME B 426       3.010   0.510   0.010  1.00  0.00           H",
                "ATOM     13 HH32 NME B 426       3.010   1.510   0.010  1.00  0.00           H",
                "ATOM     14 HH33 NME B 426       2.510   1.010   1.010  1.00  0.00           H",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_terminal_amide_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert text.count(" C   NME B 426") == 1
    assert text.count(" H1  NME B 426") == 1
    assert text.count(" H2  NME B 426") == 1
    assert text.count(" H3  NME B 426") == 1
    assert " CH3 NME" not in text
    assert "HH31 NME" not in text


def test_rewrite_ace_cap_aliases_for_leap(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CAY ACE A   3       0.000   0.000   0.000  1.00  0.00           C",
                "ATOM      2  HY1 ACE A   3       0.500   0.000   0.000  1.00  0.00           H",
                "ATOM      3  HY2 ACE A   3       0.000   0.500   0.000  1.00  0.00           H",
                "ATOM      4  HY3 ACE A   3       0.000   0.000   0.500  1.00  0.00           H",
                "ATOM      5  C   ACE A   3       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      6  OY  ACE A   3       1.500   0.000   0.000  1.00  0.00           O",
                "ATOM      7  N   GLU A   3       2.000   0.000   0.000  1.00  0.00           N",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_ace_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " CH3 ACE A   3" in text
    assert " H1  ACE A   3" in text
    assert " H2  ACE A   3" in text
    assert " H3  ACE A   3" in text
    assert " O   ACE A   3" in text
    assert " CAY ACE" not in text
    assert " HY1 ACE" not in text
    assert " OY  ACE" not in text


def test_rewrite_ace_cap_drops_duplicate_alias_atoms(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  H1  ACE A   3       0.500   0.000   0.000  1.00  0.00           H",
                "ATOM      2  CH3 ACE A   3       0.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  H2  ACE A   3       0.000   0.500   0.000  1.00  0.00           H",
                "ATOM      4  H3  ACE A   3       0.000   0.000   0.500  1.00  0.00           H",
                "ATOM      5  C   ACE A   3       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      6  O   ACE A   3       1.500   0.000   0.000  1.00  0.00           O",
                "ATOM      7  CAY ACE A   3       0.010   0.010   0.010  1.00  0.00           C",
                "ATOM      8  OY  ACE A   3       1.510   0.010   0.010  1.00  0.00           O",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_ace_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert text.count(" CH3 ACE A   3") == 1
    assert text.count(" O   ACE A   3") == 1
    assert " CAY ACE" not in text
    assert " OY  ACE" not in text


def test_rewrite_cterminal_oxygen_alias_drops_duplicate_o1(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ASP A 298       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ASP A 298       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   ASP A 298       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   ASP A 298       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  OXT ASP A 298       2.500   1.000   0.000  1.00  0.00           O",
                "ATOM      6  O1  ASP A 298       2.510   1.010   0.010  1.00  0.00           O",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_cterminal_oxygen_aliases_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " OXT ASP A 298" in text
    assert " O1  ASP A 298" not in text


def test_rewrite_cterminal_oxygen_alias_renames_ot_pair(tmp_path: Path) -> None:
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   SER B   7       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  SER B   7       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   SER B   7       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  OT1 SER B   7       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  OT2 SER B   7       2.500   1.000   0.000  1.00  0.00           O",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_cterminal_oxygen_aliases_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " O   SER B   7" in text
    assert " OXT SER B   7" in text
    assert " OT1 SER" not in text
    assert " OT2 SER" not in text


def test_rewrite_terminal_amide_cap_after_high_residues_uses_chain_local_id(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "build.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   SER A   7       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  SER A   7       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   SER A   7       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   SER A   7       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  N1  SER A   7       2.000   0.500   0.000  1.00  0.00           N",
                "ATOM      6  H1  SER A   7       2.500   0.000   0.000  1.00  0.00           H",
                "ATOM      7  H2  SER A   7       2.500   1.000   0.000  1.00  0.00           H",
                "TER",
                "ATOM      8  O   WAT W9999       5.000   0.000   0.000  1.00  0.00           O",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    assert box._rewrite_terminal_amide_caps_for_leap(pdb) == 1

    text = pdb.read_text()
    assert " N   NHE A   8" in text
    assert " HN1 NHE A   8" in text
    assert " HN2 NHE A   8" in text


def test_rewrite_terminal_amide_caps_ignores_ligand_n1_atom_with_full_chain(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "build.pdb"

    def atom_line(serial: int, atom: str, resname: str, chain: str, resid: int) -> str:
        return (
            f"ATOM  {serial:5d} {atom:>4} {resname:>3} {chain}{resid:4d}"
            "       0.000   0.000   0.000  1.00  0.00"
        )

    lines = [
        atom_line(1, "N1", "az1", "X", 398),
        atom_line(2, "C1", "az1", "X", 398),
        "TER",
    ]
    for resid in range(1, 10000):
        lines.extend([atom_line(resid + 2, "O", "WAT", "X", resid), "TER"])
    lines.append("END")
    pdb.write_text("\n".join(lines) + "\n")

    assert box._rewrite_terminal_amide_caps_for_leap(pdb) == 0

    text = pdb.read_text()
    assert " N1 az1 X 398" in text
    assert " NHE " not in text


def test_chain_id_from_renum_uses_amber_residue_ids() -> None:
    renum_df = pd.DataFrame(
        [
            {
                "old_resname": "NHE",
                "old_chain": "A",
                "old_resid": 33,
                "new_resname": "NHE",
                "new_resid": 33,
            },
            {
                "old_resname": "THR",
                "old_chain": "B",
                "old_resid": 33,
                "new_resname": "THR",
                "new_resid": 34,
            },
            {
                "old_resname": "NME",
                "old_chain": "B",
                "old_resid": 426,
                "new_resname": "NME",
                "new_resid": 427,
            },
        ]
    )

    assert box._chain_id_from_renum(renum_df, resid=34, resname="THR") == "B"
    assert box._chain_id_from_renum(renum_df, resid=427, resname="NME") == "B"
    assert box._chain_id_from_renum(renum_df, resid=33, resname="NHE") == "A"


def test_renum_chain_ids_for_shifted_fe_protein_uses_sequence(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "full_pre.pdb"
    lines = [
        _pdb_atom(1, "DU", "DUM", " ", 1, 0, 0, 0, "C"),
        _pdb_atom(2, "DU", "DUM", " ", 2, 0, 0, 0, "C"),
        _pdb_atom(3, "N", "LEU", " ", 3, 0, 0, 0, "N"),
        _pdb_atom(4, "CA", "LEU", " ", 3, 0, 0, 0, "C"),
        _pdb_atom(5, "N", "ALA", " ", 4, 0, 0, 0, "N"),
        _pdb_atom(6, "CA", "ALA", " ", 4, 0, 0, 0, "C"),
        _pdb_atom(7, "N", "GLU", " ", 5, 0, 0, 0, "N"),
        _pdb_atom(8, "CA", "GLU", " ", 5, 0, 0, 0, "C"),
        _pdb_atom(9, "N", "HID", " ", 6, 0, 0, 0, "N"),
        _pdb_atom(10, "CA", "HID", " ", 6, 0, 0, 0, "C"),
        _pdb_atom(11, "N", "LEU", " ", 7, 0, 0, 0, "N"),
        _pdb_atom(12, "CA", "LEU", " ", 7, 0, 0, 0, "C"),
        _pdb_atom(13, "N", "VAL", " ", 8, 0, 0, 0, "N"),
        _pdb_atom(14, "CA", "VAL", " ", 8, 0, 0, 0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    universe = mda.Universe(str(pdb))
    residues = universe.select_atoms(box._PROTEIN_WITH_TERMINAL_CAPS).residues
    renum_df = pd.DataFrame(
        [
            {
                "old_resname": "DUM",
                "old_chain": "D",
                "old_resid": 1,
                "new_resname": "DUM",
                "new_resid": 1,
            },
            {
                "old_resname": "DUM",
                "old_chain": "D",
                "old_resid": 2,
                "new_resname": "DUM",
                "new_resid": 2,
            },
            {
                "old_resname": "LEU",
                "old_chain": "A",
                "old_resid": 306,
                "new_resname": "LEU",
                "new_resid": 1,
            },
            {
                "old_resname": "ALA",
                "old_chain": "A",
                "old_resid": 307,
                "new_resname": "ALA",
                "new_resid": 2,
            },
            {
                "old_resname": "GLU",
                "old_chain": "B",
                "old_resid": 523,
                "new_resname": "GLU",
                "new_resid": 3,
            },
            {
                "old_resname": "HIE",
                "old_chain": "B",
                "old_resid": 524,
                "new_resname": "HIE",
                "new_resid": 4,
            },
            {
                "old_resname": "LEU",
                "old_chain": "B",
                "old_resid": 525,
                "new_resname": "LEU",
                "new_resid": 5,
            },
            {
                "old_resname": "VAL",
                "old_chain": "C",
                "old_resid": 534,
                "new_resname": "VAL",
                "new_resid": 6,
            },
            {
                "old_resname": "WAT",
                "old_chain": "W",
                "old_resid": 999,
                "new_resname": "WAT",
                "new_resid": 999,
            },
        ]
    )

    assert box._renum_chain_ids_for_residues(residues, renum_df) == [
        "A",
        "A",
        "B",
        "B",
        "B",
        "C",
    ]


def test_pdb4amber_is_required_for_box(tmp_path: Path, monkeypatch) -> None:
    input_pdb = tmp_path / "build.pdb"
    output_pdb = tmp_path / "build_amber.pdb"
    input_pdb.write_text("ATOM\n")
    monkeypatch.setattr(box, "_executable_path", lambda cmd: None)

    with pytest.raises(FileNotFoundError, match="pdb4amber is required"):
        box._run_pdb4amber_for_box(input_pdb, output_pdb, working_dir=tmp_path)

    assert not output_pdb.exists()
    assert not (tmp_path / "build_amber_renum.txt").exists()


def test_pdb4amber_for_box_resolves_from_active_python_environment(
    tmp_path: Path, monkeypatch
) -> None:
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    python = env_bin / "python"
    python.write_text("#!/bin/sh\n")
    pdb4amber = env_bin / "pdb4amber"
    pdb4amber.write_text("#!/bin/sh\n")
    pdb4amber.chmod(0o755)
    input_pdb = tmp_path / "build.pdb"
    output_pdb = tmp_path / "build_amber.pdb"
    input_pdb.write_text("ATOM\n")
    commands: list[tuple[str, Path | None]] = []

    monkeypatch.setattr(box.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(box.sys, "executable", str(python))
    monkeypatch.setattr(
        box,
        "run_with_log",
        lambda cmd, working_dir=None: commands.append((cmd, working_dir)),
    )

    box._run_pdb4amber_for_box(input_pdb, output_pdb, working_dir=tmp_path)

    assert commands == [
        (f"{pdb4amber} -i {input_pdb.name} -o {output_pdb.name} -y", tmp_path),
    ]


def test_collapse_terminal_cap_resid_values_uses_neighbor_resids() -> None:
    renum_df = pd.DataFrame(
        [
            {
                "old_resname": "ACE",
                "old_chain": "A",
                "old_resid": 9,
                "new_resname": "ACE",
                "new_resid": 1,
            },
            {
                "old_resname": "ALA",
                "old_chain": "A",
                "old_resid": 9,
                "new_resname": "ALA",
                "new_resid": 2,
            },
            {
                "old_resname": "SER",
                "old_chain": "B",
                "old_resid": 32,
                "new_resname": "SER",
                "new_resid": 3,
            },
            {
                "old_resname": "NHE",
                "old_chain": "B",
                "old_resid": 33,
                "new_resname": "NHE",
                "new_resid": 4,
            },
            {
                "old_resname": "ARG",
                "old_chain": "C",
                "old_resid": 421,
                "new_resname": "ARG",
                "new_resid": 5,
            },
            {
                "old_resname": "NME",
                "old_chain": "C",
                "old_resid": 422,
                "new_resname": "NME",
                "new_resid": 6,
            },
        ]
    )

    assert box._collapse_terminal_cap_resid_values(
        renum_df, [1, 2, 3, 4, 5, 6]
    ) == [2, 2, 3, 3, 5, 5]


def test_restore_protein_resids_collapses_synthetic_and_mapped_caps(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "full.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C",
                "ATOM      2  CH3 ACE A   1       0.500   0.000   0.000  1.00  0.00           C",
                "ATOM      3  N   ALA A   2       1.000   0.000   0.000  1.00  0.00           N",
                "ATOM      4  CA  ALA A   2       1.500   0.000   0.000  1.00  0.00           C",
                "ATOM      5  N   SER B   3       2.000   0.000   0.000  1.00  0.00           N",
                "ATOM      6  CA  SER B   3       2.500   0.000   0.000  1.00  0.00           C",
                "ATOM      7  N   NHE B   4       3.000   0.000   0.000  1.00  0.00           N",
                "ATOM      8  N   ARG C   5       4.000   0.000   0.000  1.00  0.00           N",
                "ATOM      9  CA  ARG C   5       4.500   0.000   0.000  1.00  0.00           C",
                "ATOM     10  N   NME C   6       5.000   0.000   0.000  1.00  0.00           N",
                "ATOM     11  CH3 NME C   6       5.500   0.000   0.000  1.00  0.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    universe = mda.Universe(str(pdb))
    renum_df = pd.DataFrame(
        [
            {
                "old_resname": "ACE",
                "old_chain": "A",
                "old_resid": 10,
                "new_resname": "ACE",
                "new_resid": 1,
            },
            {
                "old_resname": "ALA",
                "old_chain": "A",
                "old_resid": 10,
                "new_resname": "ALA",
                "new_resid": 2,
            },
            {
                "old_resname": "SER",
                "old_chain": "B",
                "old_resid": 32,
                "new_resname": "SER",
                "new_resid": 3,
            },
            {
                "old_resname": "ARG",
                "old_chain": "C",
                "old_resid": 421,
                "new_resname": "ARG",
                "new_resid": 4,
            },
            {
                "old_resname": "NMA",
                "old_chain": "C",
                "old_resid": 421,
                "new_resname": "NMA",
                "new_resid": 5,
            },
        ]
    )

    box._restore_protein_resids_from_renum(universe.atoms, renum_df)

    residues = universe.select_atoms(box._PROTEIN_WITH_TERMINAL_CAPS).residues
    assert list(residues.resnames) == ["ACE", "ALA", "SER", "NHE", "ARG", "NME"]
    assert list(residues.resids) == [10, 10, 32, 32, 421, 421]


def test_protein_with_terminal_caps_selection_includes_nhe(tmp_path: Path) -> None:
    pdb = tmp_path / "full_pre.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   SER    32       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  SER    32       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   SER    32       1.500   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   SER    32       1.500   2.000   0.000  1.00  0.00           O",
                "ATOM      5  N   NHE    33       2.000   0.500   0.000  1.00  0.00           N",
                "ATOM      6  HN1 NHE    33       2.500   0.000   0.000  1.00  0.00           H",
                "ATOM      7  HN2 NHE    33       2.500   1.000   0.000  1.00  0.00           H",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    universe = mda.Universe(str(pdb))

    assert "NHE" not in set(universe.select_atoms("protein").residues.resnames)
    assert "NHE" in set(
        universe.select_atoms(box._PROTEIN_WITH_TERMINAL_CAPS).residues.resnames
    )
