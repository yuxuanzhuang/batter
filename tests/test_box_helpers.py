from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("parmed")
mda = pytest.importorskip("MDAnalysis", exc_type=ImportError)

from batter._internal.ops import box


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
