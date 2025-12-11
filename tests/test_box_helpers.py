from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("parmed")

from batter._internal.ops import box


def test_ligand_charge_from_metadata_rounds_and_handles_missing(tmp_path: Path) -> None:
    meta = tmp_path / "lig.json"
    meta.write_text(json.dumps({"ligand_charge": -1.6}))

    assert box._ligand_charge_from_metadata(meta) == -2
    assert box._ligand_charge_from_metadata(tmp_path / "missing.json") is None
