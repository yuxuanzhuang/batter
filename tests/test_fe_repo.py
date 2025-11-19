from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from batter.runtime.fe_repo import FEResultsRepository
from batter.runtime.portable import ArtifactStore


@pytest.mark.parametrize(
    "status,reason",
    [
        ("failed", "something went wrong"),
        ("unbound", "UNBOUND detected"),
    ],
)
def test_record_failure_creates_row_and_artifact(
    status: str, reason: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path)
    repo = FEResultsRepository(store)

    repo.record_failure(
        run_id="run1",
        ligand="lig1",
        system_name="sys",
        temperature=300.0,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        canonical_smiles="C",
        original_name="lig1",
        original_path="/some/path",
        protocol="abfe",
    )

    idx_path = tmp_path / "results" / "index.csv"
    assert idx_path.exists()
    df = pd.read_csv(idx_path)
    assert not df.empty
    row = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")].iloc[0]
    assert row["status"] == status
    assert row["failure_reason"] == reason

    failure_file = tmp_path / "results" / "run1" / "lig1" / "failure.json"
    assert failure_file.exists()
    data = json.loads(failure_file.read_text())
    assert data["status"] == status
    assert data["reason"] == reason
