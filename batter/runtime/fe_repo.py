from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
import json
import shutil

from .portable import ArtifactStore, Artifact


__all__ = ["WindowResult", "FERecord", "FEResultsRepository"]


class WindowResult(BaseModel):
    """
    Result for a single lambda window/component.

    Parameters
    ----------
    component : str
        Component key (e.g., 'e', 'v', 'z').
    lam : float
        Lambda value in [0, 1].
    dG : float
        Free-energy increment (kcal/mol).
    dG_se : float
        Standard error (kcal/mol).
    n_samples : int
        Samples (or effective sample size).
    meta : dict
        Extra metadata.
    """

    component: str
    lam: float
    dG: float
    dG_se: float = 0.0
    n_samples: int = 0
    meta: Dict[str, Any] = Field(default_factory=dict)


class FERecord(BaseModel):
    """
    A full FE result bundle (portable, versioned).

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    ligand : str
        Ligand identifier.
    mol_name : str
        Molecule resname.
    system_name : str
        Logical system name.
    fe_type : str
        Protocol type (e.g., 'uno_rest', 'asfe').
    temperature : float
        Simulation temperature (K).
    method : {"mbar","ti"}
        Integration method.
    total_dG : float
        Total free energy (kcal/mol).
    total_se : float
        Standard error (kcal/mol).
    components : list[str]
        Active components in this run.
    created_at : str
        ISO-8601 timestamp (UTC, Z-suffix).
    windows : list[WindowResult]
        Per-window results.
    """

    run_id: str
    ligand: str
    mol_name: str
    system_name: str
    fe_type: str
    temperature: float
    method: Literal["mbar", "ti"] = "mbar"
    total_dG: float
    total_se: float = 0.0
    components: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    windows: List[WindowResult] = Field(default_factory=list)
    canonical_smiles: str | None = None
    original_name: str | None = None
    original_path: str | None = None
    protocol: str = "abfe"
    status: Literal["success", "failed", "unbound"] = "success"


class FEResultsRepository:
    def __init__(self, store: "ArtifactStore") -> None:
        self.store = store
        self._root = store.root / "results"
        self._idx = self._root / "index.csv"

    def _lig_dir(self, run_id: str, ligand: str) -> Path:
        return self._root / run_id / ligand

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized.setdefault("temperature", pd.NA)
        normalized.setdefault("total_dG", pd.NA)
        normalized.setdefault("total_se", pd.NA)
        normalized.setdefault("canonical_smiles", "")
        normalized.setdefault("original_name", "")
        normalized.setdefault("original_path", "")
        normalized.setdefault("protocol", "")
        normalized.setdefault(
            "created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        normalized.setdefault("status", "success")
        normalized.setdefault("failure_reason", "")
        return normalized

    def _append_index_row(self, row: dict[str, Any]) -> None:
        row = self._normalize_row(row)
        if self._idx.exists():
            df = pd.read_csv(self._idx)
            df = df[~((df.run_id == row["run_id"]) & (df.ligand == row["ligand"]))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        cols = [
            "run_id",
            "ligand",
            "mol_name",
            "system_name",
            "temperature",
            "total_dG",
            "total_se",
            "canonical_smiles",
            "original_name",
            "original_path",
            "protocol",
            "status",
            "failure_reason",
            "created_at",
        ]
        if df.columns.tolist() != cols:
            for col in cols:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[cols]
        df.to_csv(self._idx, index=False)

    def save(self, rec: FERecord, copy_from: Path | None = None) -> None:
        lig_dir = self._lig_dir(rec.run_id, rec.ligand)
        lig_dir.mkdir(parents=True, exist_ok=True)
        # write JSON record
        (lig_dir / "record.json").write_text(json.dumps(rec.__dict__, indent=2))
        # optional: copy raw Results/ in
        if copy_from and copy_from.exists():
            # keep raw artifacts alongside the record
            shutil.rmtree(lig_dir / "Results", ignore_errors=True)
            shutil.copytree(copy_from, lig_dir / "Results")
        # update index table (append-or-upsert by (run_id, ligand))
        row = {
            "run_id": rec.run_id,
            "ligand": rec.ligand,
            "mol_name": rec.mol_name,
            "system_name": rec.system_name,
            "temperature": rec.temperature,
            "total_dG": rec.total_dG,
            "total_se": rec.total_se,
            "canonical_smiles": rec.canonical_smiles or "",
            "original_name": rec.original_name or "",
            "original_path": rec.original_path or "",
            "protocol": rec.protocol,
            "created_at": rec.created_at,
            "status": rec.status,
            "failure_reason": "",
        }
        self._append_index_row(row)

    def index(self) -> "pd.DataFrame":
        cols = [
            "run_id",
            "ligand",
            "mol_name",
            "system_name",
            "temperature",
            "total_dG",
            "total_se",
            "canonical_smiles",
            "original_name",
            "original_path",
            "protocol",
            "created_at",
        ]
        if self._idx.exists():
            df = pd.read_csv(self._idx)
        else:
            df = pd.DataFrame(columns=cols)
        # drop old columns if present
        for drop in ("fe_type", "components", "method"):
            if drop in df.columns:
                df = df.drop(columns=[drop])
        # ensure columns exist
        for key in ("status", "failure_reason"):
            if key not in df.columns:
                df[key] = pd.NA
        for col in cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df[cols + ["status", "failure_reason"]]

    def record_failure(
        self,
        run_id: str,
        ligand: str,
        system_name: str,
        temperature: float,
        *,
        status: Literal["failed", "unbound"],
        reason: str | None = None,
        canonical_smiles: str | None = None,
        original_name: str | None = None,
        original_path: str | None = None,
        protocol: str = "abfe",
    ) -> None:
        lig_dir = self._lig_dir(run_id, ligand)
        lig_dir.mkdir(parents=True, exist_ok=True)
        failure_detail = {
            "run_id": run_id,
            "ligand": ligand,
            "status": status,
            "reason": reason or "",
            "protocol": protocol,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (lig_dir / "failure.json").write_text(json.dumps(failure_detail, indent=2))
        row = {
            "run_id": run_id,
            "ligand": ligand,
            "mol_name": "",
            "system_name": system_name,
            "temperature": temperature,
            "total_dG": pd.NA,
            "total_se": pd.NA,
            "canonical_smiles": canonical_smiles or "",
            "original_name": original_name or "",
            "original_path": original_path or "",
            "protocol": protocol,
            "status": status,
            "failure_reason": reason or "",
            "created_at": failure_detail["timestamp"],
        }
        self._append_index_row(row)

    def load(self, run_id: str, ligand: str) -> FERecord:
        p = self._lig_dir(run_id, ligand) / "record.json"
        d = json.loads(p.read_text())
        d["components"] = (
            d.get("components", "").split(",")
            if isinstance(d.get("components"), str)
            else d.get("components", [])
        )
        return FERecord(**d)
