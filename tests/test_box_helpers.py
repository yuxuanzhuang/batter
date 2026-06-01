from __future__ import annotations

import json
from pathlib import Path

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
