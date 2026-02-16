from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from batter.runtime.fe_repo import FERecord, FEResultsRepository
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


def test_save_clears_stale_failure_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    repo = FEResultsRepository(store)

    # write a failure first
    repo.record_failure(
        run_id="run1",
        ligand="lig1",
        system_name="sys",
        temperature=300.0,
        status="failed",  # type: ignore[arg-type]
        reason="old failure",
    )
    failure_file = tmp_path / "results" / "run1" / "lig1" / "failure.json"
    assert failure_file.exists()

    # now save a success and ensure the failure marker is removed
    rec = FERecord(
        run_id="run1",
        ligand="lig1",
        mol_name="lig1",
        system_name="sys",
        fe_type="rest",
        temperature=300.0,
        method="mbar",
        total_dG=-1.0,
        total_se=0.1,
        components=["z"],
    )
    repo.save(rec)

    assert not failure_file.exists()
    df = pd.read_csv(tmp_path / "results" / "index.csv")
    row = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")].iloc[0]
    assert row["status"] == "success"
    assert pd.isna(row["failure_reason"]) or row["failure_reason"] == ""

    # repository view should also present a cleared reason
    idx = repo.index()
    row2 = idx[(idx["run_id"] == "run1") & (idx["ligand"] == "lig1")].iloc[0]
    assert row2["failure_reason"] == ""


def test_index_upsert_key_includes_analysis_start_step_and_n_bootstraps(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    repo = FEResultsRepository(store)

    def _mk_record(step: int, n_bootstraps: int, total_dg: float) -> FERecord:
        return FERecord(
            run_id="run1",
            ligand="lig1",
            mol_name="lig1",
            system_name="sys",
            fe_type="rest",
            temperature=300.0,
            method="mbar",
            total_dG=total_dg,
            total_se=0.1,
            components=["z"],
            analysis_start_step=step,
            n_bootstraps=n_bootstraps,
        )

    repo.save(_mk_record(step=0, n_bootstraps=0, total_dg=-1.0))
    repo.save(_mk_record(step=5000, n_bootstraps=0, total_dg=-2.0))
    repo.save(_mk_record(step=5000, n_bootstraps=64, total_dg=-3.0))
    repo.save(_mk_record(step=5000, n_bootstraps=64, total_dg=-4.0))

    df = pd.read_csv(tmp_path / "results" / "index.csv")
    rows = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")]
    assert "n_bootstraps" in df.columns
    assert len(rows) == 3
    keys = {
        (int(step), int(boots))
        for step, boots in zip(
            rows["analysis_start_step"].tolist(),
            rows["n_bootstraps"].tolist(),
        )
    }
    assert keys == {(0, 0), (5000, 0), (5000, 64)}
    updated = rows[
        (rows["analysis_start_step"] == 5000) & (rows["n_bootstraps"] == 64)
    ].iloc[0]
    assert updated["total_dG"] == pytest.approx(-4.0)
