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
