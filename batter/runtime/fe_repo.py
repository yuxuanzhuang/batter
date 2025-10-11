from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

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
    system_name: str
    fe_type: str
    temperature: float
    method: Literal["mbar", "ti"] = "mbar"
    total_dG: float
    total_se: float = 0.0
    components: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    windows: List[WindowResult] = Field(default_factory=list)


class FEResultsRepository:
    """
    Repository that saves/loads FE records in a relocatable :class:`ArtifactStore`.

    Layout
    ------
    ``<store.root>/fe/``
      - ``index.parquet``          # tabular index of runs (one row per record)
      - ``<run_id>/record.json``   # full FERecord (JSON)
      - ``<run_id>/windows.parquet``  # per-window table (optional)

    Notes
    -----
    - Index appends; it is safe to call :meth:`save` many times with new runs.
    - All paths are relative to the store root and registered in the manifest.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.base = self.store.root / "fe"
        self.base.mkdir(parents=True, exist_ok=True)

    # ---------------- public ops ----------------

    def save(self, rec: FERecord) -> None:
        """
        Save a record (JSON + Parquet) and update the index.

        Parameters
        ----------
        rec : FERecord
            Free-energy record to persist.
        """
        run_dir = self.base / rec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSON record
        (run_dir / "record.json").write_text(rec.model_dump_json(indent=2))

        # Per-window parquet (optional)
        if rec.windows:
            df_w = pd.DataFrame([w.model_dump() for w in rec.windows])
            df_w.to_parquet(run_dir / "windows.parquet", index=False)

        # Append/update index
        idx_path = self.base / "index.parquet"
        row = {
            "run_id": rec.run_id,
            "system_name": rec.system_name,
            "fe_type": rec.fe_type,
            "temperature": rec.temperature,
            "method": rec.method,
            "total_dG": rec.total_dG,
            "total_se": rec.total_se,
            "components": ",".join(rec.components),
            "created_at": rec.created_at,
        }
        if idx_path.exists():
            idx = pd.read_parquet(idx_path)
            idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
        else:
            idx = pd.DataFrame([row])
        idx.to_parquet(idx_path, index=False)

        # Register artifacts for portability
        rel_run = run_dir.relative_to(self.store.root)
        self.store._manifest.add(Artifact(name=f"fe/{rec.run_id}/record", relpath=rel_run / "record.json"))
        if (run_dir / "windows.parquet").exists():
            self.store._manifest.add(Artifact(name=f"fe/{rec.run_id}/windows", relpath=rel_run / "windows.parquet"))
        self.store._manifest.add(Artifact(name="fe/index", relpath=Path("fe/index.parquet")))
        self.store.save_manifest()

    def load(self, run_id: str) -> FERecord:
        """
        Load a record by run id.

        Parameters
        ----------
        run_id : str
            Identifier to load.

        Returns
        -------
        FERecord
            Parsed record with windows (if present).
        """
        run_dir = self.base / run_id
        rec = FERecord.model_validate_json((run_dir / "record.json").read_text())
        wp = run_dir / "windows.parquet"
        if wp.exists():
            df_w = pd.read_parquet(wp)
            rec.windows = [WindowResult(**row) for row in df_w.to_dict(orient="records")]
        return rec

    def index(self) -> pd.DataFrame:
        """
        Return the run index as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Empty if no runs saved.
        """
        p = self.base / "index.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()