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
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    windows: List[WindowResult] = Field(default_factory=list)


class FEResultsRepository:
    def __init__(self, store: "ArtifactStore") -> None:
        self.store = store
        self._root = store.root / "results"
        self._idx = self._root / "index.csv"

    def _lig_dir(self, run_id: str, ligand: str) -> Path:
        return self._root / run_id / ligand

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
            "fe_type": rec.fe_type,
            "temperature": rec.temperature,
            "method": rec.method,
            "total_dG": rec.total_dG,
            "total_se": rec.total_se,
            "components": ",".join(rec.components),
            "created_at": rec.created_at,
        }
        if self._idx.exists():
            df = pd.read_csv(self._idx)
            df = df[~((df.run_id == rec.run_id) & (df.ligand == rec.ligand))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(self._idx, index=False)

    def index(self) -> "pd.DataFrame":
        if self._idx.exists():
            return pd.read_csv(self._idx)
        import pandas as pd
        return pd.DataFrame(columns=[
            "run_id","ligand","system_name","fe_type","temperature","method","total_dG","total_se","components","created_at"
        ])

    def load(self, run_id: str, ligand: str) -> FERecord:
        p = self._lig_dir(run_id, ligand) / "record.json"
        d = json.loads(p.read_text())
        d["components"] = d.get("components", "").split(",") if isinstance(d.get("components"), str) else d.get("components", [])
        return FERecord(**d)