from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("parmed")
pytest.importorskip("MDAnalysis", exc_type=ImportError)

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
