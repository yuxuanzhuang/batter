"""Helpers for converting BATTER RBFE results into Cinnabar ``FEMap`` objects."""

from __future__ import annotations

import html
import json
import re
import base64
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CinnabarConversionResult",
    "auto_write_rbfe_cinnabar_for_run",
    "build_batter_rbfe_cinnabar",
    "build_batter_rbfe_cinnabar_by_run",
    "dataframe_to_cinnabar",
    "load_batter_rbfe_results",
    "summarize_directionality",
    "write_cinnabar_outputs",
]


@dataclass
class CinnabarConversionResult:
    femap: Any
    edge_summary: pd.DataFrame
    raw_signed: pd.DataFrame
    merge_bidirectional: bool = True
    exp_summary: pd.DataFrame | None = None
    absolute_summary: pd.DataFrame | None = None
    absolute_warning: str | None = None
    ligand_assets: dict[str, dict[str, str]] = field(default_factory=dict)
    edge_assets: dict[str, dict[str, str]] = field(default_factory=dict)


def _import_networkx():
    try:
        import networkx as nx
    except Exception as exc:
        raise RuntimeError(
            "Cinnabar network rendering requires 'networkx'. "
            "Install it in the BATTER environment before using RBFE Cinnabar export."
        ) from exc
    return nx


def list_fe_runs(work_dir: str | Path) -> pd.DataFrame:
    """Lazy wrapper to avoid a hard import cycle with :mod:`batter.api`."""
    from batter.api import list_fe_runs as _list_fe_runs

    return _list_fe_runs(work_dir)


def summarize_directionality(edge_summary: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether an edge table contains reciprocal directional pairs."""
    if edge_summary is None or edge_summary.empty:
        return {
            "n_directional_edges": 0,
            "n_reciprocal_pairs": 0,
            "reciprocal_pairs": [],
        }

    directed_edges: set[tuple[str, str]] = set()
    for row in edge_summary.itertuples(index=False):
        label_a = str(getattr(row, "labelA", "") or "").strip()
        label_b = str(getattr(row, "labelB", "") or "").strip()
        if not label_a or not label_b:
            continue
        directed_edges.add((label_a, label_b))

    reciprocal_pairs = sorted(
        {
            tuple(sorted((label_a, label_b)))
            for label_a, label_b in directed_edges
            if label_a != label_b and (label_b, label_a) in directed_edges
        }
    )
    return {
        "n_directional_edges": int(len(directed_edges)),
        "n_reciprocal_pairs": int(len(reciprocal_pairs)),
        "reciprocal_pairs": [f"{label_a}~{label_b}" for label_a, label_b in reciprocal_pairs],
    }


def _rbfe_run_ids_for_replicate_note(
    work_dir: str | Path,
    current_run_id: str,
) -> list[str]:
    """Return RBFE run ids that look like replicate siblings of ``current_run_id``."""
    try:
        df = list_fe_runs(Path(work_dir)).copy()
    except Exception:
        return []
    if df.empty or "run_id" not in df.columns:
        return []

    protocol_series = (
        df.get("protocol", df.get("fe_type", pd.Series("", index=df.index)))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    rbfe_df = df.loc[protocol_series.eq("rbfe")].copy()
    if rbfe_df.empty:
        return []

    rbfe_df["run_id"] = rbfe_df["run_id"].astype(str)
    current_rows = rbfe_df.loc[rbfe_df["run_id"] == str(current_run_id)].copy()
    if current_rows.empty:
        return sorted(rbfe_df["run_id"].dropna().astype(str).unique().tolist())

    if "system_name" in rbfe_df.columns:
        system_name_series = rbfe_df["system_name"].fillna("").astype(str)
        current_system_names = (
            current_rows.get("system_name", pd.Series("", index=current_rows.index))
            .fillna("")
            .astype(str)
        )
        current_system_name = next(
            (name for name in current_system_names.tolist() if name),
            "",
        )
        if current_system_name:
            rbfe_df = rbfe_df.loc[system_name_series.eq(current_system_name)].copy()

    return sorted(rbfe_df["run_id"].dropna().astype(str).unique().tolist())


def _replicate_cinnabar_note(work_dir: str | Path, current_run_id: str) -> str | None:
    """Return a user-facing note for combining replicate RBFE runs."""
    run_ids = _rbfe_run_ids_for_replicate_note(work_dir, current_run_id)
    if len(run_ids) <= 1:
        return None
    cmd = " ".join(
        [
            "batter fe cinnabar",
            shlex.quote(str(Path(work_dir))),
            *[f"--run-id {shlex.quote(run_id)}" for run_id in run_ids],
        ]
    )
    return (
        "Multiple RBFE runs were detected for this work directory. "
        "To combine replicate runs into one Cinnabar bundle, run: "
        f"{cmd}"
    )


def _import_cinnabar_stack() -> tuple[Any, Any, Any]:
    try:
        from cinnabar.femap import FEMap
        from cinnabar import plotting
        from openff.units import unit
    except Exception as exc:  # pragma: no cover - exercised via caller-facing error handling
        raise RuntimeError(
            "Cinnabar conversion requires 'cinnabar' and 'openff.units'. "
            "Install them in the BATTER environment before using this command."
        ) from exc
    return FEMap, plotting, unit


def _combine_estimates(
    values: Sequence[float],
    ses: Sequence[float],
    uncertainty_mode: Literal["ivw", "sample", "max"] = "max",
) -> tuple[float, float]:
    values_arr = np.asarray(values, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)

    if len(values_arr) == 0:
        raise ValueError("No values to combine.")
    if np.any(~np.isfinite(values_arr)):
        raise ValueError("Non-finite values found.")
    if np.any(~np.isfinite(ses_arr)) or np.any(ses_arr <= 0):
        raise ValueError("All uncertainties must be finite and > 0.")

    if len(values_arr) == 1:
        return float(values_arr[0]), float(ses_arr[0])

    weights = 1.0 / np.square(ses_arr)
    mean = float(np.sum(weights * values_arr) / np.sum(weights))
    ivw_se = float(np.sqrt(1.0 / np.sum(weights)))
    sample_se = float(np.std(values_arr, ddof=1) / np.sqrt(len(values_arr)))

    if uncertainty_mode == "ivw":
        out_se = ivw_se
    elif uncertainty_mode == "sample":
        out_se = sample_se
    elif uncertainty_mode == "max":
        out_se = max(ivw_se, sample_se)
    else:  # pragma: no cover - guarded by Literal/click
        raise ValueError("uncertainty_mode must be 'ivw', 'sample', or 'max'.")

    return mean, out_se


def _normalize_energy_unit(unit_obj: Any, unit_module: Any) -> Any:
    if unit_obj is None:
        return unit_module.kilocalorie_per_mole
    if hasattr(unit_obj, "dimensionality"):
        return unit_obj

    text = str(unit_obj).strip().lower()
    mapping = {
        "kcal/mol": unit_module.kilocalorie_per_mole,
        "kilocalorie_per_mole": unit_module.kilocalorie_per_mole,
        "kilocalories_per_mole": unit_module.kilocalorie_per_mole,
        "kj/mol": unit_module.kilojoule_per_mole,
        "kilojoule_per_mole": unit_module.kilojoule_per_mole,
        "kilojoules_per_mole": unit_module.kilojoule_per_mole,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported unit: {unit_obj!r}")
    return mapping[text]


def _pick_edge_label(row: pd.Series, edge_separator: str) -> str:
    ligand = str(row.get("ligand", "") or "").strip()
    original_name = str(row.get("original_name", "") or "").strip()
    if edge_separator in original_name:
        return original_name
    return ligand


def _scan_rbfe_input_paths(
    work_dir: str | Path,
    run_ids: Sequence[str],
    ligand_labels: Sequence[str],
) -> dict[str, str]:
    """Best-effort map of ligand label -> staged RBFE input path."""
    labels = {str(label).strip() for label in ligand_labels if str(label).strip()}
    if not labels:
        return {}

    mapping: dict[str, str] = {}
    work_root = Path(work_dir)
    for run_id in run_ids:
        trans_root = work_root / "executions" / str(run_id) / "simulations" / "transformations"
        if not trans_root.is_dir():
            continue
        for inputs_dir in trans_root.glob("*~*/inputs"):
            if not inputs_dir.is_dir():
                continue
            for child in sorted(inputs_dir.iterdir()):
                if not child.is_file():
                    continue
                stem = child.stem.strip()
                if stem in labels and stem not in mapping:
                    mapping[stem] = str(child)
    return mapping


def _mol_from_any_path(path_str: str):
    """Load an RDKit molecule from a staged ligand path."""
    from rdkit import Chem

    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in supplier:
            if mol is not None:
                return mol
        return None
    if suffix == ".mol":
        return Chem.MolFromMolFile(str(path), removeHs=False)
    if suffix == ".mol2":
        return Chem.MolFromMol2File(str(path), removeHs=False)
    if suffix == ".pdb":
        return Chem.MolFromPDBFile(str(path), removeHs=False)
    return None


def _mol_to_svg_text(mol) -> str:
    """Render an RDKit molecule as a compact SVG string."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    draw_mol = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(draw_mol)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DSVG(260, 180)
    drawer.drawOptions().padding = 0.05
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, draw_mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


def _build_ligand_assets(
    rbfe_df: pd.DataFrame,
    *,
    work_dir: str | Path | None = None,
    edge_separator: str = "~",
) -> dict[str, dict[str, str]]:
    """Build ligand hover assets for HTML exports."""
    if rbfe_df is None or rbfe_df.empty:
        return {}

    labels: set[str] = set()
    smiles_by_label: dict[str, str] = {}
    path_by_label: dict[str, str] = {}

    edge_series = rbfe_df.get("edge_label", rbfe_df.get("ligand", pd.Series(dtype=str)))
    canonical_series = rbfe_df.get("canonical_smiles", pd.Series(index=rbfe_df.index, dtype=str))
    path_series = rbfe_df.get("original_path", pd.Series(index=rbfe_df.index, dtype=str))
    run_series = rbfe_df.get("run_id", pd.Series(index=rbfe_df.index, dtype=str))

    for edge_label, canonical_smiles, original_path in zip(
        edge_series.fillna("").astype(str),
        canonical_series.fillna("").astype(str),
        path_series.fillna("").astype(str),
    ):
        if edge_separator not in edge_label:
            continue
        left, right = (piece.strip() for piece in edge_label.split(edge_separator, 1))
        if left:
            labels.add(left)
        if right:
            labels.add(right)
        if left and canonical_smiles and left not in smiles_by_label:
            smiles_by_label[left] = canonical_smiles.strip()
        if left and original_path and left not in path_by_label:
            path_by_label[left] = original_path.strip()

    if work_dir is not None:
        scanned = _scan_rbfe_input_paths(
            work_dir,
            [str(run_id).strip() for run_id in run_series.dropna().astype(str).unique()],
            sorted(labels),
        )
        for label, input_path in scanned.items():
            path_by_label.setdefault(label, input_path)

    try:
        from rdkit import Chem
    except Exception:
        return {
            label: {
                "label": label,
                "smiles": smiles_by_label.get(label, ""),
                "input_path": path_by_label.get(label, ""),
                "svg": "",
            }
            for label in sorted(labels)
        }

    assets: dict[str, dict[str, str]] = {}
    for label in sorted(labels):
        smiles = smiles_by_label.get(label, "").strip()
        input_path = path_by_label.get(label, "").strip()
        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None and input_path:
            try:
                mol = _mol_from_any_path(input_path)
            except Exception:
                mol = None
        svg = ""
        if mol is not None:
            try:
                svg = _mol_to_svg_text(mol)
            except Exception:
                svg = ""
        assets[label] = {
            "label": label,
            "smiles": smiles,
            "input_path": input_path,
            "svg": svg,
        }
    return assets


def _file_to_data_uri(path: Path) -> str:
    """Encode a local file as a data URI."""
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _build_edge_assets(
    rbfe_df: pd.DataFrame,
    *,
    work_dir: str | Path,
    merge_bidirectional: bool,
    edge_separator: str = "~",
) -> dict[str, dict[str, str]]:
    """Build edge-click assets from stored RBFE mapping images."""
    if rbfe_df is None or rbfe_df.empty:
        return {}

    assets: dict[str, dict[str, str]] = {}
    work_root = Path(work_dir)

    for row in rbfe_df.itertuples(index=False):
        edge_label = str(getattr(row, "edge_label", "") or getattr(row, "ligand", "") or "").strip()
        if edge_separator not in edge_label:
            continue
        left, right = (part.strip() for part in edge_label.split(edge_separator, 1))
        if not left or not right:
            continue
        if merge_bidirectional:
            label_a, label_b = sorted((left, right))
        else:
            label_a, label_b = left, right
        edge_key = f"{label_a}~{label_b}"
        if edge_key in assets:
            continue

        run_id = str(getattr(row, "run_id", "") or "").strip()
        stored_pair_id = str(getattr(row, "ligand", "") or "").strip()
        if not run_id or not stored_pair_id:
            continue

        results_dir = work_root / "results" / run_id / stored_pair_id / "Results"
        image_path = None
        for candidate in ("mapping.png", "mapping.svg"):
            candidate_path = results_dir / candidate
            if candidate_path.is_file():
                image_path = candidate_path
                break
        if image_path is None:
            continue
        try:
            image_data_uri = _file_to_data_uri(image_path)
        except Exception:
            continue

        assets[edge_key] = {
            "edge_key": edge_key,
            "display_title": f"{label_a} → {label_b}",
            "run_id": run_id,
            "pair_id": stored_pair_id,
            "image_name": image_path.name,
            "image_data_uri": image_data_uri,
        }

    return assets


def load_batter_rbfe_results(
    work_dir: str | Path,
    *,
    run_ids: Sequence[str] | None = None,
    ligands: Sequence[str] | None = None,
    edge_separator: str = "~",
) -> pd.DataFrame:
    """Load stored BATTER FE records and keep only RBFE-like edge rows."""
    df = list_fe_runs(Path(work_dir)).copy()
    if df.empty:
        raise ValueError(f"No FE results found under {work_dir}.")

    edge_pattern = re.escape(edge_separator)
    ligand_mask = (
        df.get("ligand", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.contains(edge_pattern, regex=True)
    )
    original_mask = (
        df.get("original_name", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.contains(edge_pattern, regex=True)
    )
    protocol_mask = (
        df.get("protocol", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("rbfe")
    )

    work = df.loc[ligand_mask | original_mask | protocol_mask].copy()
    if work.empty:
        raise ValueError(f"No RBFE-like FE results found under {work_dir}.")

    if run_ids:
        requested = {str(v).strip() for v in run_ids if str(v).strip()}
        work = work.loc[work["run_id"].astype(str).isin(requested)].copy()
        if work.empty:
            raise ValueError(
                f"No RBFE rows remain after filtering for run_id(s): {sorted(requested)}."
            )

    work["edge_label"] = work.apply(
        lambda row: _pick_edge_label(row, edge_separator=edge_separator), axis=1
    )

    if ligands:
        requested_ligands = {str(v).strip() for v in ligands if str(v).strip()}
        work = work.loc[work["edge_label"].isin(requested_ligands)].copy()
        if work.empty:
            raise ValueError(
                "No RBFE rows remain after filtering for ligand(s): "
                + ", ".join(sorted(requested_ligands))
            )

    return work


def dataframe_to_cinnabar(
    rbfe_df: pd.DataFrame,
    *,
    ligand_column: str = "ligand",
    dg_column: str = "total_dG",
    se_column: str = "total_se",
    run_column: str = "run_id",
    status_column: str = "status",
    success_value: str = "success",
    temperature_column: str = "temperature",
    edge_separator: str = "~",
    source: str = "BATTER_RBFE",
    uncertainty_mode: Literal["ivw", "sample", "max"] = "max",
    combine_by_run_first: bool = True,
    merge_bidirectional: bool = True,
    experimental_df: pd.DataFrame | None = None,
    exp_ligand_column: str = "ligand",
    exp_abfe_column: str = "abfe",
    exp_error_column: str | None = None,
    exp_status_column: str | None = None,
    exp_success_value: str = "success",
    exp_temperature_column: str | None = None,
    exp_source: str = "experiment",
    exp_value_unit: Any = "kcal/mol",
    exp_error_unit: Any = None,
) -> CinnabarConversionResult:
    """Convert an RBFE dataframe into a Cinnabar ``FEMap`` and summary tables."""
    FEMap, _plotting, unit = _import_cinnabar_stack()

    exp_error_unit = exp_value_unit if exp_error_unit is None else exp_error_unit
    exp_value_unit = _normalize_energy_unit(exp_value_unit, unit)
    exp_error_unit = _normalize_energy_unit(exp_error_unit, unit)

    required = {ligand_column, dg_column, se_column}
    missing = required - set(rbfe_df.columns)
    if missing:
        raise ValueError(f"Missing RBFE columns: {sorted(missing)}")

    work = rbfe_df.copy()
    if status_column in work.columns:
        work = work.loc[work[status_column] == success_value].copy()

    work = work.dropna(subset=[ligand_column, dg_column, se_column]).copy()
    if work.empty:
        raise ValueError("No usable RBFE rows remain after filtering.")

    lig_split = work[ligand_column].astype(str).str.split(edge_separator, n=1, expand=True)
    if lig_split.shape[1] != 2:
        raise ValueError(
            f"Could not split '{ligand_column}' using separator '{edge_separator}'."
        )

    work["ligand_A_raw"] = lig_split[0].str.strip()
    work["ligand_B_raw"] = lig_split[1].str.strip()

    raw_dg = pd.to_numeric(work[dg_column], errors="raise").astype(float)
    if merge_bidirectional:
        forward_is_canonical = work["ligand_A_raw"] <= work["ligand_B_raw"]
        work["labelA"] = np.where(
            forward_is_canonical, work["ligand_A_raw"], work["ligand_B_raw"]
        )
        work["labelB"] = np.where(
            forward_is_canonical, work["ligand_B_raw"], work["ligand_A_raw"]
        )
        work["signed_dDG"] = np.where(forward_is_canonical, raw_dg, -raw_dg)
    else:
        work["labelA"] = work["ligand_A_raw"]
        work["labelB"] = work["ligand_B_raw"]
        work["signed_dDG"] = raw_dg
    work["input_se"] = pd.to_numeric(work[se_column], errors="raise").astype(float)

    if np.any(work["input_se"] <= 0):
        raise ValueError(f"Column '{se_column}' must contain only positive values.")

    if temperature_column in work.columns:
        work["temperature_K"] = pd.to_numeric(work[temperature_column], errors="coerce")
    else:
        work["temperature_K"] = 298.15

    raw_signed = work.copy()

    def summarize_rbfe_block(group: pd.DataFrame) -> dict[str, Any]:
        mean, out_se = _combine_estimates(
            group["signed_dDG"].values,
            group["input_se"].values,
            uncertainty_mode=uncertainty_mode,
        )
        return {
            "calc_DDG": mean,
            "calc_dDDG": out_se,
            "n_measurements": int(len(group)),
            "temperature_K": float(group["temperature_K"].dropna().mean())
            if group["temperature_K"].notna().any()
            else 298.15,
        }

    if combine_by_run_first and run_column in raw_signed.columns:
        per_run_records: list[dict[str, Any]] = []
        for (labelA, labelB, run_id), group in raw_signed.groupby(
            ["labelA", "labelB", run_column], sort=True
        ):
            rec = {"labelA": labelA, "labelB": labelB, run_column: run_id}
            rec.update(summarize_rbfe_block(group))
            per_run_records.append(rec)
        per_run = pd.DataFrame(per_run_records)

        edge_records: list[dict[str, Any]] = []
        for (labelA, labelB), group in per_run.groupby(["labelA", "labelB"], sort=True):
            mean, out_se = _combine_estimates(
                group["calc_DDG"].values,
                group["calc_dDDG"].values,
                uncertainty_mode=uncertainty_mode,
            )
            edge_records.append(
                {
                    "labelA": labelA,
                    "labelB": labelB,
                    "calc_DDG": mean,
                    "calc_dDDG": out_se,
                    "n_runs": int(len(group)),
                    "n_measurements": int(group["n_measurements"].sum()),
                    "temperature_K": float(group["temperature_K"].mean()),
                }
            )
        edge_summary = pd.DataFrame(edge_records)
    else:
        edge_records = []
        for (labelA, labelB), group in raw_signed.groupby(["labelA", "labelB"], sort=True):
            rec = {"labelA": labelA, "labelB": labelB}
            rec.update(summarize_rbfe_block(group))
            rec["n_runs"] = (
                int(group[run_column].nunique()) if run_column in group.columns else 1
            )
            edge_records.append(rec)
        edge_summary = pd.DataFrame(edge_records)

    femap = FEMap()
    for row in edge_summary.itertuples(index=False):
        femap.add_relative_calculation(
            labelA=row.labelA,
            labelB=row.labelB,
            value=float(row.calc_DDG) * unit.kilocalorie_per_mole,
            uncertainty=float(row.calc_dDDG) * unit.kilocalorie_per_mole,
            source=source,
            temperature=float(row.temperature_K) * unit.kelvin,
        )

    exp_summary = None
    if experimental_df is not None:
        exp_required = {exp_ligand_column, exp_abfe_column}
        exp_missing = exp_required - set(experimental_df.columns)
        if exp_missing:
            raise ValueError(f"Missing experimental columns: {sorted(exp_missing)}")

        exp_work = experimental_df.copy()
        if exp_status_column is not None and exp_status_column in exp_work.columns:
            exp_work = exp_work.loc[exp_work[exp_status_column] == exp_success_value].copy()

        drop_cols = [exp_ligand_column, exp_abfe_column]
        has_exp_error = bool(exp_error_column and exp_error_column in exp_work.columns)
        if has_exp_error:
            drop_cols.append(exp_error_column)

        exp_work = exp_work.dropna(subset=drop_cols).copy()
        if not exp_work.empty:
            exp_work["label"] = exp_work[exp_ligand_column].astype(str).str.strip()
            exp_work["exp_DG"] = pd.to_numeric(
                exp_work[exp_abfe_column], errors="raise"
            ).astype(float)

            if has_exp_error:
                exp_work["exp_uncertainty"] = pd.to_numeric(
                    exp_work[exp_error_column], errors="raise"
                ).astype(float)
                if np.any(exp_work["exp_uncertainty"] <= 0):
                    raise ValueError(
                        f"Experimental column '{exp_error_column}' must contain only positive values."
                    )
            else:
                exp_work["exp_uncertainty"] = np.nan

            if exp_temperature_column is not None and exp_temperature_column in exp_work.columns:
                exp_work["temperature_K"] = pd.to_numeric(
                    exp_work[exp_temperature_column], errors="coerce"
                )
            else:
                exp_work["temperature_K"] = 298.15

            exp_records: list[dict[str, Any]] = []
            for label, group in exp_work.groupby("label", sort=True):
                if has_exp_error:
                    mean, out_se = _combine_estimates(
                        group["exp_DG"].values,
                        group["exp_uncertainty"].values,
                        uncertainty_mode=uncertainty_mode,
                    )
                else:
                    mean = float(group["exp_DG"].mean())
                    out_se = np.nan

                exp_records.append(
                    {
                        "label": label,
                        "exp_DG": mean,
                        "exp_uncertainty": out_se,
                        "n_exp": int(len(group)),
                        "temperature_K": float(group["temperature_K"].dropna().mean())
                        if group["temperature_K"].notna().any()
                        else 298.15,
                    }
                )

            exp_summary = pd.DataFrame(exp_records)
            for row in exp_summary.itertuples(index=False):
                femap.add_experimental_measurement(
                    label=row.label,
                    value=float(row.exp_DG) * exp_value_unit,
                    uncertainty=(
                        float(row.exp_uncertainty) * exp_error_unit
                        if pd.notna(row.exp_uncertainty)
                        else 0 * exp_error_unit
                    ),
                    source=exp_source,
                    temperature=float(row.temperature_K) * unit.kelvin,
                )

    absolute_summary = None
    absolute_warning = None
    try:
        femap.generate_absolute_values()
        absolute_summary = femap.get_absolute_dataframe()
    except Exception as exc:
        absolute_summary = None
        absolute_warning = (
            "Could not build a full absolute ΔG solution from the RBFE network. "
            f"Continuing with relative-only outputs. Underlying error: {exc}"
        )

    return CinnabarConversionResult(
        femap=femap,
        edge_summary=edge_summary,
        raw_signed=raw_signed,
        merge_bidirectional=merge_bidirectional,
        exp_summary=exp_summary,
        absolute_summary=absolute_summary,
        absolute_warning=absolute_warning,
    )


def build_batter_rbfe_cinnabar(
    work_dir: str | Path,
    *,
    run_ids: Sequence[str] | None = None,
    ligands: Sequence[str] | None = None,
    edge_separator: str = "~",
    uncertainty_mode: Literal["ivw", "sample", "max"] = "max",
    combine_by_run_first: bool = True,
    merge_bidirectional: bool = True,
    experimental_df: pd.DataFrame | None = None,
    exp_ligand_column: str = "ligand",
    exp_abfe_column: str = "abfe",
    exp_error_column: str | None = None,
    exp_status_column: str | None = None,
    exp_success_value: str = "success",
    exp_temperature_column: str | None = None,
    source: str = "BATTER_RBFE",
    exp_source: str = "experiment",
    exp_value_unit: Any = "kcal/mol",
    exp_error_unit: Any = None,
) -> CinnabarConversionResult:
    work = load_batter_rbfe_results(
        work_dir,
        run_ids=run_ids,
        ligands=ligands,
        edge_separator=edge_separator,
    )
    result = dataframe_to_cinnabar(
        work,
        ligand_column="edge_label",
        edge_separator=edge_separator,
        uncertainty_mode=uncertainty_mode,
        combine_by_run_first=combine_by_run_first,
        merge_bidirectional=merge_bidirectional,
        experimental_df=experimental_df,
        exp_ligand_column=exp_ligand_column,
        exp_abfe_column=exp_abfe_column,
        exp_error_column=exp_error_column,
        exp_status_column=exp_status_column,
        exp_success_value=exp_success_value,
        exp_temperature_column=exp_temperature_column,
        source=source,
        exp_source=exp_source,
        exp_value_unit=exp_value_unit,
        exp_error_unit=exp_error_unit,
    )
    result.ligand_assets = _build_ligand_assets(
        work,
        work_dir=work_dir,
        edge_separator=edge_separator,
    )
    result.edge_assets = _build_edge_assets(
        work,
        work_dir=work_dir,
        merge_bidirectional=merge_bidirectional,
        edge_separator=edge_separator,
    )
    return result


def build_batter_rbfe_cinnabar_by_run(
    work_dir: str | Path,
    *,
    run_ids: Sequence[str] | None = None,
    ligands: Sequence[str] | None = None,
    edge_separator: str = "~",
    uncertainty_mode: Literal["ivw", "sample", "max"] = "max",
    combine_by_run_first: bool = True,
    merge_bidirectional: bool = True,
    experimental_df: pd.DataFrame | None = None,
    exp_ligand_column: str = "ligand",
    exp_abfe_column: str = "abfe",
    exp_error_column: str | None = None,
    exp_status_column: str | None = None,
    exp_success_value: str = "success",
    exp_temperature_column: str | None = None,
    source: str = "BATTER_RBFE",
    exp_source: str = "experiment",
    exp_value_unit: Any = "kcal/mol",
    exp_error_unit: Any = None,
) -> dict[str, CinnabarConversionResult]:
    work = load_batter_rbfe_results(
        work_dir,
        run_ids=run_ids,
        ligands=ligands,
        edge_separator=edge_separator,
    )
    out: dict[str, CinnabarConversionResult] = {}
    for run_id, group in work.groupby("run_id", sort=True):
        result = dataframe_to_cinnabar(
            group,
            ligand_column="edge_label",
            edge_separator=edge_separator,
            uncertainty_mode=uncertainty_mode,
            combine_by_run_first=combine_by_run_first,
            merge_bidirectional=merge_bidirectional,
            experimental_df=experimental_df,
            exp_ligand_column=exp_ligand_column,
            exp_abfe_column=exp_abfe_column,
            exp_error_column=exp_error_column,
            exp_status_column=exp_status_column,
            exp_success_value=exp_success_value,
            exp_temperature_column=exp_temperature_column,
            source=source,
            exp_source=exp_source,
            exp_value_unit=exp_value_unit,
            exp_error_unit=exp_error_unit,
        )
        result.ligand_assets = _build_ligand_assets(
            group,
            work_dir=work_dir,
            edge_separator=edge_separator,
        )
        result.edge_assets = _build_edge_assets(
            group,
            work_dir=work_dir,
            merge_bidirectional=merge_bidirectional,
            edge_separator=edge_separator,
        )
        out[str(run_id)] = result
    return out


def auto_write_rbfe_cinnabar_for_run(
    work_dir: str | Path,
    run_id: str,
    *,
    out_dir: str | Path | None = None,
    combine_by_run_first: bool = True,
    merge_bidirectional: bool = True,
    write_plots: bool = True,
    absolute_offset: float = 0.0,
) -> dict[str, Any]:
    """Write a per-run RBFE Cinnabar bundle plus a replicate-aware follow-up note."""
    work_root = Path(work_dir)
    output_dir = Path(out_dir) if out_dir is not None else (work_root / "results" / "cinnabar" / str(run_id))
    result = build_batter_rbfe_cinnabar(
        work_root,
        run_ids=[str(run_id)],
        combine_by_run_first=combine_by_run_first,
        merge_bidirectional=merge_bidirectional,
    )
    outputs = write_cinnabar_outputs(
        result,
        output_dir,
        method_name="BATTER",
        target_name=f"{work_root.name}:{run_id}",
        write_plots=write_plots,
        absolute_offset=absolute_offset,
    )
    return {
        "result": result,
        "outputs": outputs,
        "output_dir": output_dir,
        "replicate_note": _replicate_cinnabar_note(work_root, str(run_id)),
        "absolute_warning": getattr(result, "absolute_warning", None),
    }


def _node_color_mapping(
    graph: nx.DiGraph,
    absolute_summary: pd.DataFrame | None,
):
    """Return node colors and optional colorbar metadata."""
    try:
        from matplotlib import colors as mcolors
        from matplotlib import colormaps
    except Exception:
        mcolors = None
        colormaps = None

    node_degree = dict(graph.degree())
    node_order = list(graph.nodes)

    if absolute_summary is not None and not absolute_summary.empty and mcolors is not None:
        abs_df = absolute_summary.copy()
        dg_col = next((col for col in abs_df.columns if col.lower().startswith("dg")), None)
        label_col = "label" if "label" in abs_df.columns else None
        if dg_col and label_col:
            dg_map = (
                abs_df.dropna(subset=[label_col, dg_col])
                .drop_duplicates(subset=[label_col])
                .set_index(label_col)[dg_col]
                .astype(float)
                .to_dict()
            )
            if dg_map:
                node_values = [float(dg_map.get(node, np.nan)) for node in node_order]
                finite = [value for value in node_values if np.isfinite(value)]
                if finite:
                    limit = max(abs(min(finite)), abs(max(finite)), 1e-8)
                    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
                    cmap = colormaps["bwr_r"] if colormaps is not None else None
                    return {
                        "values": node_values,
                        "norm": norm,
                        "cmap": cmap,
                        "label": "MLE ΔG (kcal/mol)",
                        "mode": "absolute",
                    }

    if mcolors is not None:
        vmax = max(node_degree.values()) if node_degree else 1
        return {
            "values": [float(node_degree[node]) for node in node_order],
            "norm": mcolors.Normalize(vmin=0, vmax=max(1, vmax)),
            "cmap": colormaps["Blues"] if colormaps is not None else None,
            "label": "Node degree",
            "mode": "degree",
        }
    return {
        "values": [float(node_degree[node]) for node in node_order],
        "norm": None,
        "cmap": None,
        "label": "Node degree",
        "mode": "degree",
    }


def _rgba_to_hex(color_value: Any) -> str:
    try:
        from matplotlib import colors as mcolors

        return mcolors.to_hex(color_value, keep_alpha=False)
    except Exception:
        return "#5b8def"


def _layout_node_radii(graph) -> dict[str, float]:
    """Return conservative layout radii that leave room for labels and arrows."""
    node_degree = dict(graph.degree())
    return {
        str(node): 48.0 + 4.0 * float(node_degree.get(node, 0))
        for node in graph.nodes
    }


def _layout_bounds(
    positions: dict[str, np.ndarray],
    radii: dict[str, float],
    *,
    padding: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return bounding box including node radii."""
    min_x = min(float(positions[node][0]) - float(radii[node]) - padding for node in positions)
    max_x = max(float(positions[node][0]) + float(radii[node]) + padding for node in positions)
    min_y = min(float(positions[node][1]) - float(radii[node]) - padding for node in positions)
    max_y = max(float(positions[node][1]) + float(radii[node]) + padding for node in positions)
    return min_x, max_x, min_y, max_y


def _ensure_node_spacing(
    positions: dict[str, np.ndarray],
    radii: dict[str, float],
    *,
    padding: float = 24.0,
    iterations: int = 220,
) -> dict[str, np.ndarray]:
    """Repel overlapping nodes until their effective circles no longer collide."""
    nodes = list(positions)
    if len(nodes) < 2:
        return {node: np.asarray(pos, dtype=float).copy() for node, pos in positions.items()}

    adjusted = {node: np.asarray(pos, dtype=float).copy() for node, pos in positions.items()}
    anchors = {node: pos.copy() for node, pos in adjusted.items()}

    for _ in range(iterations):
        disp = {node: np.zeros(2, dtype=float) for node in nodes}
        moved = False
        for idx, node_a in enumerate(nodes):
            for node_b in nodes[idx + 1 :]:
                delta = adjusted[node_b] - adjusted[node_a]
                dist = float(np.linalg.norm(delta))
                target = float(radii[node_a] + radii[node_b] + padding)
                if dist >= target:
                    continue
                direction = _normalize_vec(
                    delta,
                    fallback=np.array([1.0 + 0.17 * idx, 0.35 + 0.11 * (idx + 1)], dtype=float),
                )
                push = 0.5 * (target - dist + 1e-3)
                disp[node_a] -= direction * push
                disp[node_b] += direction * push
                moved = True
        if not moved:
            break
        for node in nodes:
            adjusted[node] += 0.55 * disp[node] + 0.04 * (anchors[node] - adjusted[node])

    center = np.mean(np.stack([adjusted[node] for node in nodes]), axis=0)
    for node in nodes:
        adjusted[node] = adjusted[node] - center

    scale = 1.0
    for idx, node_a in enumerate(nodes):
        for node_b in nodes[idx + 1 :]:
            delta = adjusted[node_b] - adjusted[node_a]
            dist = float(np.linalg.norm(delta))
            target = float(radii[node_a] + radii[node_b] + padding)
            if dist <= 1e-6:
                scale = max(scale, 2.0)
            elif dist < target:
                scale = max(scale, target / dist)
    if scale > 1.0:
        for node in nodes:
            adjusted[node] = adjusted[node] * (scale * 1.05)

    return adjusted


def _initial_component_layout(component_graph) -> dict[str, np.ndarray]:
    """Create a stable initial layout for one connected component."""
    nx = _import_networkx()
    nodes = list(component_graph.nodes)
    n_nodes = len(nodes)
    if n_nodes == 1:
        return {str(nodes[0]): np.array([0.0, 0.0], dtype=float)}
    if n_nodes == 2:
        return {
            str(nodes[0]): np.array([-90.0, 0.0], dtype=float),
            str(nodes[1]): np.array([90.0, 0.0], dtype=float),
        }

    undirected = component_graph.to_undirected()
    degree = dict(undirected.degree())
    try:
        closeness = nx.closeness_centrality(undirected)
    except Exception:
        closeness = {str(node): 0.0 for node in undirected.nodes}
    center = max(
        undirected.nodes,
        key=lambda node: (
            float(closeness.get(node, 0.0)),
            float(degree.get(node, 0)),
            str(node),
        ),
    )
    distances = nx.single_source_shortest_path_length(undirected, center)
    shells: dict[int, list[str]] = {}
    for node, dist in distances.items():
        shells.setdefault(int(dist), []).append(str(node))
    nlist = [
        sorted(
            shell_nodes,
            key=lambda node: (
                -sum(
                    1
                    for nbr in undirected.neighbors(node)
                    if int(distances.get(nbr, 10**9)) < shell_idx
                ),
                -float(degree.get(node, 0)),
                str(node),
            ),
        )
        for shell_idx, shell_nodes in sorted(shells.items())
    ]
    shell_seed = nx.shell_layout(undirected, nlist=nlist, scale=1.0)
    refined = nx.spring_layout(
        undirected,
        pos=shell_seed,
        seed=42,
        iterations=400 if n_nodes <= 24 else 520,
        k=max(0.8, 2.8 / np.sqrt(float(n_nodes))),
        weight=None,
    )
    out = {str(node): np.asarray(refined[node], dtype=float) for node in refined}
    points = np.stack(list(out.values()))
    center_point = np.mean(points, axis=0)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-6)
    target_span = max(260.0, 180.0 * np.sqrt(float(n_nodes)))
    scale = target_span / span
    for node in out:
        out[node] = (out[node] - center_point) * scale
    return out


def _pack_component_layouts(
    component_layouts: Sequence[tuple[dict[str, np.ndarray], dict[str, float]]],
    *,
    gap: float = 180.0,
) -> dict[str, np.ndarray]:
    """Pack connected components into rows to avoid inter-component overlap."""
    if not component_layouts:
        return {}
    if len(component_layouts) == 1:
        positions, _ = component_layouts[0]
        return {node: np.asarray(pos, dtype=float).copy() for node, pos in positions.items()}

    records: list[dict[str, Any]] = []
    total_area = 0.0
    for positions, radii in component_layouts:
        min_x, max_x, min_y, max_y = _layout_bounds(positions, radii, padding=18.0)
        width = max_x - min_x
        height = max_y - min_y
        total_area += width * height
        records.append(
            {
                "positions": positions,
                "radii": radii,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "width": width,
                "height": height,
                "area": width * height,
            }
        )

    records.sort(key=lambda item: float(item["area"]), reverse=True)
    target_row_width = max(
        max(float(item["width"]) for item in records),
        np.sqrt(max(total_area, 1.0)) * 1.35,
    )

    packed: dict[str, np.ndarray] = {}
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for item in records:
        width = float(item["width"])
        height = float(item["height"])
        if cursor_x > 0.0 and cursor_x + width > target_row_width:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        shift = np.array(
            [cursor_x - float(item["min_x"]), cursor_y - float(item["min_y"])],
            dtype=float,
        )
        for node, pos in item["positions"].items():
            packed[str(node)] = np.asarray(pos, dtype=float) + shift
        cursor_x += width + gap
        row_height = max(row_height, height)

    min_x = min(float(pos[0]) for pos in packed.values())
    max_x = max(float(pos[0]) for pos in packed.values())
    min_y = min(float(pos[1]) for pos in packed.values())
    max_y = max(float(pos[1]) for pos in packed.values())
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)
    for node in packed:
        packed[node] = packed[node] - center
    return packed


def _network_graph_with_layout(edge_summary: pd.DataFrame) -> tuple[Any, dict[str, np.ndarray]]:
    """Build the directed graph and a packed non-overlapping layout for rendering."""
    nx = _import_networkx()
    graph = nx.DiGraph()
    for row in edge_summary.itertuples(index=False):
        graph.add_edge(
            str(row.labelA),
            str(row.labelB),
            calc_DDG=float(row.calc_DDG),
            calc_dDDG=float(row.calc_dDDG),
            n_runs=int(getattr(row, "n_runs", 1)),
            n_measurements=int(getattr(row, "n_measurements", 1)),
        )

    if graph.number_of_nodes() == 0:
        return graph, {}

    undirected = graph.to_undirected()
    component_layouts: list[tuple[dict[str, np.ndarray], dict[str, float]]] = []
    for component_nodes in nx.connected_components(undirected):
        subgraph = graph.subgraph(component_nodes).copy()
        radii = _layout_node_radii(subgraph)
        positions = _initial_component_layout(subgraph)
        positions = _ensure_node_spacing(
            positions,
            radii,
            padding=26.0,
            iterations=260 if subgraph.number_of_nodes() > 12 else 200,
        )
        component_layouts.append((positions, radii))

    packed = _pack_component_layouts(component_layouts)
    return graph, packed


def _png_layout_scale(graph) -> float:
    """Return an expansion factor for the static PNG network layout."""
    n_nodes = max(int(graph.number_of_nodes()), 1)
    n_edges = max(int(graph.number_of_edges()), 0)
    avg_degree = (2.0 * float(n_edges)) / float(n_nodes)
    scale = 1.12
    scale += min(0.42, 0.045 * np.sqrt(max(n_nodes - 3, 0)))
    scale += min(0.26, 0.035 * max(avg_degree - 1.5, 0.0))
    return float(scale)


def _label_rects_overlap(
    center_a: np.ndarray,
    size_a: tuple[float, float],
    center_b: np.ndarray,
    size_b: tuple[float, float],
    *,
    padding: float = 6.0,
) -> bool:
    """Return True when two center-positioned label boxes overlap."""
    return (
        abs(float(center_a[0]) - float(center_b[0]))
        < 0.5 * (float(size_a[0]) + float(size_b[0])) + padding
        and abs(float(center_a[1]) - float(center_b[1]))
        < 0.5 * (float(size_a[1]) + float(size_b[1])) + padding
    )


def _resolve_label_positions(
    label_specs: Sequence[dict[str, np.ndarray]],
    *,
    box_size: tuple[float, float],
) -> list[np.ndarray]:
    """Shift overlapping label anchors apart using tangent/normal offsets."""
    if not label_specs:
        return []

    step_normal = float(box_size[1]) * 0.95
    step_tangent = float(box_size[0]) * 0.55
    candidate_steps = [
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (0, 2),
        (0, -2),
        (2, 0),
        (-2, 0),
        (1, 2),
        (-1, 2),
        (1, -2),
        (-1, -2),
        (2, 1),
        (-2, 1),
        (2, -1),
        (-2, -1),
    ]

    placed: list[np.ndarray] = []
    resolved: list[np.ndarray] = []
    for spec in label_specs:
        base = np.asarray(spec["base"], dtype=float)
        tangent = np.asarray(spec["tangent"], dtype=float)
        normal = np.asarray(spec["normal"], dtype=float)

        tangent_norm = np.linalg.norm(tangent)
        normal_norm = np.linalg.norm(normal)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([1.0, 0.0])
        if normal_norm > 0:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 1.0])

        chosen = base
        for tangent_step, normal_step in candidate_steps:
            candidate = (
                base
                + tangent * tangent_step * step_tangent
                + normal * normal_step * step_normal
            )
            if all(
                not _label_rects_overlap(
                    candidate,
                    box_size,
                    other,
                    box_size,
                    padding=8.0,
                )
                for other in placed
            ):
                chosen = candidate
                break
        placed.append(chosen)
        resolved.append(chosen)

    return resolved


def _normalize_vec(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Return a unit vector, or a fallback direction when the norm is tiny."""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    if fallback is not None:
        fallback_arr = np.asarray(fallback, dtype=float)
        fallback_norm = float(np.linalg.norm(fallback_arr))
        if fallback_norm > 1e-12:
            return fallback_arr / fallback_norm
    return np.array([1.0, 0.0], dtype=float)


def _quadratic_bezier_tangent(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    t: float,
) -> np.ndarray:
    """Return the tangent vector of a quadratic Bezier curve at parameter t."""
    return (
        2.0 * (1.0 - float(t)) * (np.asarray(control, dtype=float) - np.asarray(start, dtype=float))
        + 2.0 * float(t) * (np.asarray(end, dtype=float) - np.asarray(control, dtype=float))
    )


def _render_network_png(
    edge_summary: pd.DataFrame,
    out_path: Path,
    *,
    absolute_summary: pd.DataFrame | None = None,
    title: str = "",
    merge_bidirectional: bool = True,
) -> bool:
    """Render a BATTER-styled RBFE network figure from summarized edge data."""
    if edge_summary.empty:
        return False
    nx = _import_networkx()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
        from matplotlib import colors as mcolors
        from matplotlib import cm, colormaps
    except Exception:
        return False

    graph, pos = _network_graph_with_layout(edge_summary)
    png_scale = _png_layout_scale(graph)
    plot_pos = {
        node: np.asarray(point, dtype=float) * png_scale
        for node, point in pos.items()
    }

    node_degree = dict(graph.degree())
    node_sizes = [1400 + 220 * node_degree[node] for node in graph.nodes]
    node_size_map = {node: size for node, size in zip(graph.nodes, node_sizes)}
    color_meta = _node_color_mapping(graph, absolute_summary)
    node_colors = color_meta["values"]
    norm = color_meta["norm"]
    cmap = color_meta["cmap"]
    colorbar_label = color_meta["label"]

    def _edge_curvature(node_a: str, node_b: str) -> float:
        if graph.has_edge(node_b, node_a) and node_a != node_b:
            # Use the same positive curvature magnitude for both directions.
            # Because the start/end points are reversed for the opposite edge,
            # reusing the same rad pushes the reciprocal edge onto the opposite
            # visual side. Using opposite signs here collapses both directions
            # back onto the same side.
            return 0.24
        return 0.0

    edge_metadata: list[tuple[str, str, dict[str, Any], float]] = []
    edge_magnitudes = []
    for node_a, node_b, data in graph.edges(data=True):
        curvature = _edge_curvature(str(node_a), str(node_b))
        edge_metadata.append((str(node_a), str(node_b), data, curvature))
        edge_magnitudes.append(abs(float(data.get("calc_DDG", 0.0))))
    if edge_magnitudes:
        edge_mag_min = min(edge_magnitudes)
        edge_mag_max = max(edge_magnitudes)
        if np.isclose(edge_mag_min, edge_mag_max):
            edge_mag_max = edge_mag_min + 1.0
    else:
        edge_mag_min = 0.0
        edge_mag_max = 1.0

    def _edge_width(abs_ddg: float) -> float:
        scaled = (abs_ddg - edge_mag_min) / max(edge_mag_max - edge_mag_min, 1e-12)
        return 2.8 + 4.2 * scaled

    edge_color = "#7c3aed"

    def _node_margin_points(node: str, *, arrow: bool) -> float:
        size = float(node_size_map.get(node, 1400.0))
        radius_points = np.sqrt(size / np.pi)
        return radius_points + (12.0 if arrow else 4.0)

    if plot_pos:
        xs = [float(point[0]) for point in plot_pos.values()]
        ys = [float(point[1]) for point in plot_pos.values()]
        layout_w = max(xs) - min(xs)
        layout_h = max(ys) - min(ys)
    else:
        layout_w = 800.0
        layout_h = 600.0
    fig_w = max(10.5, layout_w / 110.0 + 3.4)
    fig_h = max(8.0, layout_h / 110.0 + 3.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.035, right=0.88, top=0.93, bottom=0.145)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f7fb")
    ax.margins(x=0.10, y=0.12)

    for node_a, node_b, data, curvature in edge_metadata:
        nx.draw_networkx_edges(
            graph,
            plot_pos,
            ax=ax,
            edgelist=[(node_a, node_b)],
            width=_edge_width(abs(float(data.get("calc_DDG", 0.0)))),
            edge_color=[edge_color],
            alpha=0.95,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=24,
            min_source_margin=_node_margin_points(node_a, arrow=False),
            min_target_margin=_node_margin_points(node_b, arrow=True),
            connectionstyle=f"arc3,rad={curvature}",
        )
    node_artist = nx.draw_networkx_nodes(
        graph,
        plot_pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        linewidths=1.5,
        edgecolors="#243b53",
    )
    if norm is not None:
        node_artist.set_norm(norm)

    label_text = nx.draw_networkx_labels(
        graph,
        plot_pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
        font_color="#102a43",
    )
    for text in label_text.values():
        text.set_path_effects(
            [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)]
        )

    fig.canvas.draw()

    label_specs_display: list[dict[str, np.ndarray]] = []
    label_payloads: list[str] = []
    for node_a, node_b, data, curvature in edge_metadata:
        start = np.asarray(plot_pos[node_a], dtype=float)
        end = np.asarray(plot_pos[node_b], dtype=float)
        midpoint = 0.5 * (start + end)
        direction = end - start
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 0:
            perp = np.array([-direction[1], direction[0]]) / norm_dir
        else:
            perp = np.array([0.0, 0.0])
        base_data = midpoint + perp * curvature * 0.85
        base_disp = np.asarray(ax.transData.transform(base_data), dtype=float)
        start_disp = np.asarray(ax.transData.transform(start), dtype=float)
        end_disp = np.asarray(ax.transData.transform(end), dtype=float)
        tangent_disp = end_disp - start_disp
        tangent_norm = np.linalg.norm(tangent_disp)
        if tangent_norm > 0:
            tangent_disp = tangent_disp / tangent_norm
        else:
            tangent_disp = np.array([1.0, 0.0])
        normal_disp = np.array([-tangent_disp[1], tangent_disp[0]])
        label_specs_display.append(
            {"base": base_disp, "tangent": tangent_disp, "normal": normal_disp}
        )
        label_payloads.append(
            f"{float(data.get('calc_DDG', 0.0)):+.2f}\n"
            f"±{float(data.get('calc_dDDG', 0.0)):.2f}"
        )

    resolved_label_positions = _resolve_label_positions(
        label_specs_display,
        box_size=(66.0, 48.0),
    )

    for text_pos_disp, edge_label in zip(resolved_label_positions, label_payloads):
        text_pos = np.asarray(ax.transData.inverted().transform(text_pos_disp), dtype=float)
        ax.text(
            text_pos[0],
            text_pos[1],
            edge_label,
            ha="center",
            va="center",
            fontsize=8,
            color="#243b53",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "#cbd2d9",
                "alpha": 0.9,
            },
        )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color="#102a43", pad=14)

    node_scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    node_scalar.set_array([])
    cbar = fig.colorbar(node_scalar, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(colorbar_label, rotation=90)

    note_lines = [
        (
            "Direction mode: merged opposite directions"
            if merge_bidirectional
            else "Direction mode: split stored directions"
        ),
        "Arrows point from labelA to labelB",
        "Edge labels show ΔΔG ± s.e. (kcal/mol)",
        "Edge thickness scales with |ΔΔG|",
    ]
    fig.text(
        0.03,
        0.035,
        "\n".join(note_lines),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#486581",
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "white",
            "ec": "#cbd2d9",
            "alpha": 0.97,
        },
    )

    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path.exists()


def _render_network_html(
    edge_summary: pd.DataFrame,
    out_path: Path,
    *,
    absolute_summary: pd.DataFrame | None = None,
    title: str = "",
    merge_bidirectional: bool = True,
    ligand_assets: dict[str, dict[str, str]] | None = None,
) -> bool:
    """Render an interactive HTML RBFE network with ligand hover cards."""
    if edge_summary is None or edge_summary.empty:
        return False

    graph, pos = _network_graph_with_layout(edge_summary)
    color_meta = _node_color_mapping(graph, absolute_summary)
    node_values = color_meta["values"]
    norm = color_meta["norm"]
    cmap = color_meta["cmap"]

    canvas_w = 1100
    canvas_h = 760
    pad_x = 110
    pad_y = 90
    note_h = 120
    plot_h = canvas_h - note_h

    xs = [float(coord[0]) for coord in pos.values()]
    ys = [float(coord[1]) for coord in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-8)
    span_y = max(max_y - min_y, 1e-8)

    def _to_xy(point: np.ndarray) -> tuple[float, float]:
        x = pad_x + ((float(point[0]) - min_x) / span_x) * (canvas_w - 2 * pad_x)
        y = pad_y + ((max_y - float(point[1])) / span_y) * (plot_h - 2 * pad_y)
        return x, y

    def _edge_curvature(node_a: str, node_b: str) -> float:
        if graph.has_edge(node_b, node_a) and node_a != node_b:
            return 0.24
        return 0.0

    edge_magnitudes = [abs(float(data.get("calc_DDG", 0.0))) for _, _, data in graph.edges(data=True)]
    edge_mag_min = min(edge_magnitudes) if edge_magnitudes else 0.0
    edge_mag_max = max(edge_magnitudes) if edge_magnitudes else 1.0
    if np.isclose(edge_mag_min, edge_mag_max):
        edge_mag_max = edge_mag_min + 1.0

    def _edge_width(abs_ddg: float) -> float:
        scaled = (abs_ddg - edge_mag_min) / max(edge_mag_max - edge_mag_min, 1e-12)
        return 2.8 + 4.2 * scaled

    edge_color = "#7c3aed"
    assets = ligand_assets or {}

    node_degree = dict(graph.degree())
    node_radius = {node: 26.0 + 2.0 * node_degree[node] for node in graph.nodes}
    node_fill = {}
    for node, value in zip(graph.nodes, node_values):
        if norm is not None and cmap is not None and np.isfinite(value):
            node_fill[node] = _rgba_to_hex(cmap(norm(float(value))))
        else:
            node_fill[node] = "#88c0d0"

    edge_svg: list[str] = []
    label_svg: list[str] = []
    for node_a, node_b, data in graph.edges(data=True):
        curvature = _edge_curvature(str(node_a), str(node_b))
        start = np.asarray(_to_xy(pos[node_a]), dtype=float)
        end = np.asarray(_to_xy(pos[node_b]), dtype=float)
        direction = end - start
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 0:
            unit = direction / norm_dir
            perp = np.array([-unit[1], unit[0]])
        else:
            unit = np.array([0.0, 0.0])
            perp = np.array([0.0, 0.0])
        start2 = start + unit * (node_radius[node_a] + 4.0)
        end2 = end - unit * (node_radius[node_b] + 14.0)
        span = np.linalg.norm(end2 - start2)
        control = 0.5 * (start2 + end2) + perp * curvature * span * 0.75
        path_d = (
            f"M {start2[0]:.2f} {start2[1]:.2f} "
            f"Q {control[0]:.2f} {control[1]:.2f} {end2[0]:.2f} {end2[1]:.2f}"
        )
        edge_svg.append(
            f"<path d=\"{path_d}\" fill=\"none\" stroke=\"{edge_color}\" "
            f"stroke-width=\"{_edge_width(abs(float(data.get('calc_DDG', 0.0)))):.2f}\" "
            f"stroke-linecap=\"round\" stroke-opacity=\"0.96\" marker-end=\"url(#arrow)\" />"
        )

        text_pos = 0.25 * start2 + 0.5 * control + 0.25 * end2
        text_pos = text_pos + perp * curvature * span * 0.18
        edge_label = html.escape(
            f"{float(data.get('calc_DDG', 0.0)):+.2f}\n±{float(data.get('calc_dDDG', 0.0)):.2f}"
        )
        edge_label_lines = edge_label.split("\n")
        label_svg.append(
            "<g>"
            f"<rect x=\"{text_pos[0] - 30:.2f}\" y=\"{text_pos[1] - 22:.2f}\" width=\"60\" height=\"42\" "
            "rx=\"6\" ry=\"6\" fill=\"white\" fill-opacity=\"0.92\" stroke=\"#cbd2d9\" stroke-width=\"1.0\" />"
            f"<text x=\"{text_pos[0]:.2f}\" y=\"{text_pos[1] - 4:.2f}\" text-anchor=\"middle\" "
            "font-size=\"12\" fill=\"#243b53\">"
            f"{edge_label_lines[0]}</text>"
            f"<text x=\"{text_pos[0]:.2f}\" y=\"{text_pos[1] + 13:.2f}\" text-anchor=\"middle\" "
            "font-size=\"12\" fill=\"#243b53\">"
            f"{edge_label_lines[1]}</text>"
            "</g>"
        )

    node_svg: list[str] = []
    for node in graph.nodes:
        x, y = _to_xy(pos[node])
        label = html.escape(str(node))
        node_svg.append(
            "<g class=\"node\" "
            f"data-ligand=\"{label}\" transform=\"translate({x:.2f},{y:.2f})\">"
            f"<circle r=\"{node_radius[node]:.2f}\" fill=\"{node_fill[node]}\" stroke=\"#243b53\" stroke-width=\"3\" />"
            f"<text text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"18\" font-weight=\"700\" "
            "fill=\"#102a43\" paint-order=\"stroke\" stroke=\"white\" stroke-width=\"5\" stroke-linejoin=\"round\">"
            f"{label}</text>"
            "</g>"
        )

    note_lines = [
        (
            "Direction mode: merged opposite directions"
            if merge_bidirectional
            else "Direction mode: split stored directions"
        ),
        "Arrows point from labelA to labelB",
        "Edge labels show ΔΔG ± s.e. (kcal/mol)",
        "Edge thickness scales with |ΔΔG|",
        "Node colors use red → white → blue for negative → zero → positive ΔG",
    ]

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title or "BATTER RBFE network")}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7fb; color: #102a43; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 18px 18px 28px; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; text-align: center; }}
    .panel {{ background: white; border: 1px solid #d9e2ec; border-radius: 16px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); overflow: hidden; }}
    svg {{ width: 100%; height: auto; display: block; background: #f6f7fb; }}
    .notes {{ margin: 12px 14px 14px; padding: 10px 12px; border: 1px solid #cbd2d9; border-radius: 10px; background: rgba(255,255,255,0.96); color: #486581; white-space: pre-line; }}
    .tooltip {{ position: fixed; z-index: 1000; max-width: 320px; pointer-events: none; background: rgba(255,255,255,0.98); border: 1px solid #cbd2d9; border-radius: 12px; box-shadow: 0 14px 36px rgba(15, 23, 42, 0.16); padding: 10px; opacity: 0; transform: translate(12px, 12px); transition: opacity 0.08s ease-out; }}
    .tooltip.visible {{ opacity: 1; }}
    .tooltip .title {{ font-weight: 700; margin-bottom: 6px; }}
    .tooltip .smiles {{ margin-top: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #52606d; word-break: break-all; }}
    .tooltip .empty {{ font-size: 12px; color: #7b8794; }}
    .node {{ cursor: pointer; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title or "BATTER RBFE network")}</h1>
    <div class="panel">
      <svg viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="{html.escape(title or 'BATTER RBFE network')}">
        <defs>
          <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M 0 0 L 12 6 L 0 12 z" fill="{edge_color}" />
          </marker>
        </defs>
        {''.join(edge_svg)}
        {''.join(label_svg)}
        {''.join(node_svg)}
      </svg>
      <div class="notes">{html.escape(chr(10).join(note_lines))}</div>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const ligandAssets = {json.dumps(assets)};
    const tooltip = document.getElementById('tooltip');
    function renderTooltip(label) {{
      const asset = ligandAssets[label] || {{}};
      const svg = asset.svg || '<div class="empty">No 2D structure available</div>';
      const smiles = asset.smiles ? `<div class="smiles">${{asset.smiles}}</div>` : '';
      tooltip.innerHTML = `<div class="title">${{label}}</div>${{svg}}${{smiles}}`;
    }}
    document.querySelectorAll('.node').forEach((node) => {{
      node.addEventListener('mouseenter', (event) => {{
        const label = node.getAttribute('data-ligand') || '';
        renderTooltip(label);
        tooltip.classList.add('visible');
      }});
      node.addEventListener('mousemove', (event) => {{
        tooltip.style.left = `${{event.clientX + 14}}px`;
        tooltip.style.top = `${{event.clientY + 14}}px`;
      }});
      node.addEventListener('mouseleave', () => {{
        tooltip.classList.remove('visible');
      }});
    }});
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path.exists()


def _render_absolute_sorted_png(
    absolute_summary: pd.DataFrame,
    out_path: Path,
    *,
    exp_summary: pd.DataFrame | None = None,
    title: str = "",
    absolute_offset: float = 0.0,
    merge_bidirectional: bool = True,
) -> bool:
    """Render a sorted absolute free-energy ranking plot."""
    if absolute_summary is None or absolute_summary.empty:
        return False

    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colormaps
        from matplotlib import colors as mcolors
    except Exception:
        return False

    abs_df = absolute_summary.copy()
    label_col = "label" if "label" in abs_df.columns else None
    dg_col = next((col for col in abs_df.columns if col.lower().startswith("dg")), None)
    err_col = next(
        (
            col
            for col in abs_df.columns
            if "uncertainty" in col.lower() or col.lower().startswith("ddg")
        ),
        None,
    )
    if label_col is None or dg_col is None:
        return False

    abs_df = abs_df.dropna(subset=[label_col, dg_col]).copy()
    if abs_df.empty:
        return False

    abs_df["DG_raw"] = pd.to_numeric(abs_df[dg_col], errors="coerce")
    abs_df["DG_shifted"] = abs_df["DG_raw"] + float(absolute_offset)
    if err_col is not None:
        abs_df["DG_uncertainty"] = pd.to_numeric(abs_df[err_col], errors="coerce").fillna(0.0)
    else:
        abs_df["DG_uncertainty"] = 0.0
    abs_df = abs_df.sort_values("DG_shifted", ascending=True, kind="stable").reset_index(
        drop=True
    )

    exp_map: dict[str, tuple[float, float]] = {}
    if exp_summary is not None and not exp_summary.empty:
        exp_df = exp_summary.copy()
        if "label" in exp_df.columns and "exp_DG" in exp_df.columns:
            exp_df = exp_df.dropna(subset=["label", "exp_DG"]).copy()
            if not exp_df.empty:
                exp_df["exp_uncertainty"] = pd.to_numeric(
                    exp_df.get("exp_uncertainty", 0.0), errors="coerce"
                ).fillna(0.0)
                exp_map = {
                    str(row.label): (float(row.exp_DG), float(row.exp_uncertainty))
                    for row in exp_df.itertuples(index=False)
                }

    n_rows = len(abs_df)
    fig_w = max(8.0, 0.28 * n_rows + 7.0)
    fig_h = max(6.0, 0.42 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f7fb")

    y = np.arange(n_rows)
    calc_values = abs_df["DG_shifted"].to_numpy(dtype=float)
    calc_errs = abs_df["DG_uncertainty"].to_numpy(dtype=float)
    color_values = abs_df["DG_raw"].to_numpy(dtype=float)
    labels = abs_df[label_col].astype(str).tolist()

    finite_colors = color_values[np.isfinite(color_values)]
    if finite_colors.size:
        limit = max(abs(float(np.nanmin(finite_colors))), abs(float(np.nanmax(finite_colors))), 1e-8)
        bar_norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        bar_cmap = colormaps["bwr_r"]
        bar_colors = [bar_cmap(bar_norm(value)) if np.isfinite(value) else "#88c0d0" for value in color_values]
    else:
        bar_norm = None
        bar_cmap = None
        bar_colors = ["#88c0d0"] * len(calc_values)

    ax.barh(
        y,
        calc_values,
        xerr=calc_errs,
        height=0.66,
        color=bar_colors,
        edgecolor="#0b7285",
        linewidth=1.2,
        error_kw={
            "ecolor": "#0b7285",
            "elinewidth": 1.4,
            "capsize": 3,
            "capthick": 1.4,
        },
        label="BATTER MLE",
        zorder=2,
    )

    ax.axvline(0.0, color="#7b8794", linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)

    if exp_map:
        exp_values = []
        exp_errs = []
        for label in labels:
            value, uncertainty = exp_map.get(label, (np.nan, np.nan))
            exp_values.append(value)
            exp_errs.append(uncertainty)
        exp_values_arr = np.asarray(exp_values, dtype=float)
        exp_errs_arr = np.asarray(exp_errs, dtype=float)
        valid = np.isfinite(exp_values_arr)
        if np.any(valid):
            ax.errorbar(
                exp_values_arr[valid],
                y[valid],
                xerr=exp_errs_arr[valid],
                fmt="s",
                color="#bc6c25",
                ecolor="#bc6c25",
                elinewidth=1.2,
                capsize=3,
                markersize=5.5,
                label="Experiment",
                zorder=4,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Absolute ΔG (kcal/mol)", color="#102a43")
    ax.set_ylabel("Ligand", color="#102a43")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color="#102a43", pad=14)

    if bar_cmap is not None and bar_norm is not None:
        scalar = cm.ScalarMappable(norm=bar_norm, cmap=bar_cmap)
        scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=ax, shrink=0.86, pad=0.02)
        cbar.set_label("MLE ΔG (kcal/mol)", rotation=90)

    if not np.isclose(float(absolute_offset), 0.0):
        ax.text(
            0.99,
            0.01,
            f"Applied offset: {float(absolute_offset):+.2f} kcal/mol",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#486581",
        )
    ax.text(
        0.01,
        0.01,
        (
            "Direction mode: merged opposite directions"
            if merge_bidirectional
            else "Direction mode: split stored directions"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#486581",
    )

    if exp_map:
        ax.legend(frameon=False, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path.exists()


def _render_absolute_sorted_html(
    absolute_summary: pd.DataFrame,
    out_path: Path,
    *,
    exp_summary: pd.DataFrame | None = None,
    title: str = "",
    absolute_offset: float = 0.0,
    merge_bidirectional: bool = True,
    ligand_assets: dict[str, dict[str, str]] | None = None,
) -> bool:
    """Render an interactive HTML absolute-energy ranking plot."""
    if absolute_summary is None or absolute_summary.empty:
        return False

    abs_df = absolute_summary.copy()
    label_col = "label" if "label" in abs_df.columns else None
    dg_col = next((col for col in abs_df.columns if col.lower().startswith("dg")), None)
    err_col = next(
        (
            col
            for col in abs_df.columns
            if "uncertainty" in col.lower() or col.lower().startswith("ddg")
        ),
        None,
    )
    if label_col is None or dg_col is None:
        return False

    abs_df = abs_df.dropna(subset=[label_col, dg_col]).copy()
    if abs_df.empty:
        return False

    abs_df["DG_shifted"] = pd.to_numeric(abs_df[dg_col], errors="coerce") + float(
        absolute_offset
    )
    abs_df["DG_uncertainty"] = (
        pd.to_numeric(abs_df[err_col], errors="coerce").fillna(0.0)
        if err_col is not None
        else 0.0
    )
    abs_df = abs_df.sort_values("DG_shifted", ascending=True, kind="stable").reset_index(drop=True)

    exp_map: dict[str, tuple[float, float]] = {}
    if exp_summary is not None and not exp_summary.empty:
        exp_df = exp_summary.copy()
        if "label" in exp_df.columns and "exp_DG" in exp_df.columns:
            exp_df = exp_df.dropna(subset=["label", "exp_DG"]).copy()
            if not exp_df.empty:
                exp_df["exp_uncertainty"] = pd.to_numeric(
                    exp_df.get("exp_uncertainty", 0.0), errors="coerce"
                ).fillna(0.0)
                exp_map = {
                    str(row.label): (float(row.exp_DG), float(row.exp_uncertainty))
                    for row in exp_df.itertuples(index=False)
                }

    labels = abs_df[label_col].astype(str).tolist()
    calc_values = abs_df["DG_shifted"].to_numpy(dtype=float)
    calc_errs = abs_df["DG_uncertainty"].to_numpy(dtype=float)
    exp_values = np.asarray([exp_map.get(label, (np.nan, np.nan))[0] for label in labels], dtype=float)
    exp_errs = np.asarray([exp_map.get(label, (np.nan, np.nan))[1] for label in labels], dtype=float)

    xmin = min(np.nanmin(calc_values - calc_errs), np.nanmin(np.where(np.isfinite(exp_values), exp_values - exp_errs, np.nan))) if np.isfinite(exp_values).any() else np.nanmin(calc_values - calc_errs)
    xmax = max(np.nanmax(calc_values + calc_errs), np.nanmax(np.where(np.isfinite(exp_values), exp_values + exp_errs, np.nan))) if np.isfinite(exp_values).any() else np.nanmax(calc_values + calc_errs)
    if np.isclose(xmin, xmax):
        xmax = xmin + 1.0

    canvas_w = 1100
    row_h = 48
    top_pad = 70
    bottom_pad = 42
    left_pad = 210
    right_pad = 70
    canvas_h = top_pad + bottom_pad + row_h * len(labels)
    plot_w = canvas_w - left_pad - right_pad

    def _x(value: float) -> float:
        return left_pad + ((float(value) - xmin) / (xmax - xmin)) * plot_w

    zero_x = _x(0.0)
    assets = ligand_assets or {}
    rows_svg: list[str] = []
    for idx, label in enumerate(labels):
        y = top_pad + idx * row_h + row_h * 0.5
        value = float(calc_values[idx])
        err = float(calc_errs[idx])
        x0 = _x(min(0.0, value))
        x1 = _x(max(0.0, value))
        bar_x = min(x0, x1)
        bar_w = max(abs(x1 - x0), 1.5)
        err_l = _x(value - err)
        err_r = _x(value + err)
        rows_svg.append(
            f"<g class=\"bar-row\" data-ligand=\"{html.escape(label)}\">"
            f"<text x=\"{left_pad - 16:.2f}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-size=\"14\" fill=\"#102a43\">{html.escape(label)}</text>"
            f"<rect x=\"{bar_x:.2f}\" y=\"{y - 12:.2f}\" width=\"{bar_w:.2f}\" height=\"24\" rx=\"6\" ry=\"6\" fill=\"#88c0d0\" stroke=\"#0b7285\" stroke-width=\"1.2\" />"
            f"<line x1=\"{err_l:.2f}\" y1=\"{y:.2f}\" x2=\"{err_r:.2f}\" y2=\"{y:.2f}\" stroke=\"#0b7285\" stroke-width=\"1.4\" />"
            f"<line x1=\"{err_l:.2f}\" y1=\"{y - 7:.2f}\" x2=\"{err_l:.2f}\" y2=\"{y + 7:.2f}\" stroke=\"#0b7285\" stroke-width=\"1.4\" />"
            f"<line x1=\"{err_r:.2f}\" y1=\"{y - 7:.2f}\" x2=\"{err_r:.2f}\" y2=\"{y + 7:.2f}\" stroke=\"#0b7285\" stroke-width=\"1.4\" />"
        )
        if np.isfinite(exp_values[idx]):
            exp_x = _x(float(exp_values[idx]))
            exp_err = float(exp_errs[idx])
            rows_svg.append(
                f"<line x1=\"{_x(exp_values[idx] - exp_err):.2f}\" y1=\"{y:.2f}\" x2=\"{_x(exp_values[idx] + exp_err):.2f}\" y2=\"{y:.2f}\" stroke=\"#bc6c25\" stroke-width=\"1.2\" />"
                f"<rect x=\"{exp_x - 4:.2f}\" y=\"{y - 4:.2f}\" width=\"8\" height=\"8\" fill=\"#bc6c25\" />"
            )
        rows_svg.append(
            f"<text x=\"{right_pad + left_pad + plot_w - 4:.2f}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-size=\"12\" fill=\"#486581\">{value:+.2f} ± {err:.2f}</text></g>"
        )

    x_ticks = np.linspace(xmin, xmax, 6)
    grid_svg = []
    for tick in x_ticks:
        x = _x(float(tick))
        grid_svg.append(
            f"<line x1=\"{x:.2f}\" y1=\"{top_pad - 20:.2f}\" x2=\"{x:.2f}\" y2=\"{canvas_h - bottom_pad + 6:.2f}\" stroke=\"#d9e2ec\" stroke-width=\"1\" />"
            f"<text x=\"{x:.2f}\" y=\"{canvas_h - 10:.2f}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#52606d\">{tick:.1f}</text>"
        )

    note_lines = [
        (
            "Direction mode: merged opposite directions"
            if merge_bidirectional
            else "Direction mode: split stored directions"
        ),
        "Hover over a ligand row to view its 2D structure",
    ]
    if not np.isclose(float(absolute_offset), 0.0):
        note_lines.append(f"Applied offset: {float(absolute_offset):+.2f} kcal/mol")

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title or "BATTER absolute ranking")}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7fb; color: #102a43; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 18px 18px 28px; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; text-align: center; }}
    .panel {{ background: white; border: 1px solid #d9e2ec; border-radius: 16px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); overflow: hidden; }}
    svg {{ width: 100%; height: auto; display: block; background: #f6f7fb; }}
    .notes {{ margin: 12px 14px 14px; padding: 10px 12px; border: 1px solid #cbd2d9; border-radius: 10px; background: rgba(255,255,255,0.96); color: #486581; white-space: pre-line; }}
    .tooltip {{ position: fixed; z-index: 1000; max-width: 320px; pointer-events: none; background: rgba(255,255,255,0.98); border: 1px solid #cbd2d9; border-radius: 12px; box-shadow: 0 14px 36px rgba(15, 23, 42, 0.16); padding: 10px; opacity: 0; transform: translate(12px, 12px); transition: opacity 0.08s ease-out; }}
    .tooltip.visible {{ opacity: 1; }}
    .tooltip .title {{ font-weight: 700; margin-bottom: 6px; }}
    .tooltip .smiles {{ margin-top: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #52606d; word-break: break-all; }}
    .tooltip .empty {{ font-size: 12px; color: #7b8794; }}
    .bar-row {{ cursor: pointer; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title or "BATTER absolute ranking")}</h1>
    <div class="panel">
      <svg viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="{html.escape(title or 'BATTER absolute ranking')}">
        <line x1="{zero_x:.2f}" y1="{top_pad - 20:.2f}" x2="{zero_x:.2f}" y2="{canvas_h - bottom_pad + 6:.2f}" stroke="#7b8794" stroke-dasharray="4 4" stroke-width="1.2" />
        {''.join(grid_svg)}
        {''.join(rows_svg)}
        <text x="{left_pad + plot_w * 0.5:.2f}" y="{top_pad - 34:.2f}" text-anchor="middle" font-size="14" fill="#102a43">Absolute ΔG (kcal/mol)</text>
      </svg>
      <div class="notes">{html.escape(chr(10).join(note_lines))}</div>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const ligandAssets = {json.dumps(assets)};
    const tooltip = document.getElementById('tooltip');
    function renderTooltip(label) {{
      const asset = ligandAssets[label] || {{}};
      const svg = asset.svg || '<div class="empty">No 2D structure available</div>';
      const smiles = asset.smiles ? `<div class="smiles">${{asset.smiles}}</div>` : '';
      tooltip.innerHTML = `<div class="title">${{label}}</div>${{svg}}${{smiles}}`;
    }}
    document.querySelectorAll('.bar-row').forEach((row) => {{
      row.addEventListener('mouseenter', () => {{
        renderTooltip(row.getAttribute('data-ligand') || '');
        tooltip.classList.add('visible');
      }});
      row.addEventListener('mousemove', (event) => {{
        tooltip.style.left = `${{event.clientX + 14}}px`;
        tooltip.style.top = `${{event.clientY + 14}}px`;
      }});
      row.addEventListener('mouseleave', () => {{
        tooltip.classList.remove('visible');
      }});
    }});
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path.exists()


def _render_dashboard_html(
    edge_summary: pd.DataFrame,
    out_path: Path,
    *,
    absolute_summary: pd.DataFrame | None = None,
    exp_summary: pd.DataFrame | None = None,
    title: str = "",
    absolute_offset: float = 0.0,
    merge_bidirectional: bool = True,
    ligand_assets: dict[str, dict[str, str]] | None = None,
    edge_assets: dict[str, dict[str, str]] | None = None,
    absolute_warning: str | None = None,
) -> bool:
    """Render a single tabbed HTML dashboard for network and absolute plots."""
    if edge_summary is None or edge_summary.empty:
        return False

    graph, pos = _network_graph_with_layout(edge_summary)
    color_meta = _node_color_mapping(graph, absolute_summary)
    node_values = color_meta["values"]
    norm = color_meta["norm"]
    cmap = color_meta["cmap"]
    color_mode = color_meta.get("mode", "degree")
    assets = ligand_assets or {}
    mapping_assets = edge_assets or {}

    if pos:
        xs = [float(coord[0]) for coord in pos.values()]
        ys = [float(coord[1]) for coord in pos.values()]
        layout_min_x, layout_max_x = min(xs), max(xs)
        layout_min_y, layout_max_y = min(ys), max(ys)
        layout_span_x = max(layout_max_x - layout_min_x, 1.0)
        layout_span_y = max(layout_max_y - layout_min_y, 1.0)
    else:
        layout_min_x = layout_min_y = -300.0
        layout_max_x = layout_max_y = 300.0
        layout_span_x = layout_span_y = 600.0

    pad_x = max(110, int(0.10 * layout_span_x))
    pad_y = max(90, int(0.11 * layout_span_y))
    note_h = 120
    canvas_w = int(max(1100, layout_span_x + 2.0 * pad_x))
    plot_h = int(max(640, layout_span_y + 2.0 * pad_y))
    canvas_h = plot_h + note_h

    def _to_xy(point: np.ndarray) -> tuple[float, float]:
        x = pad_x + (float(point[0]) - layout_min_x)
        y = pad_y + (layout_max_y - float(point[1]))
        return x, y

    def _edge_curvature(node_a: str, node_b: str) -> float:
        if graph.has_edge(node_b, node_a) and node_a != node_b:
            return 0.24
        return 0.0

    edge_magnitudes = [abs(float(data.get("calc_DDG", 0.0))) for _, _, data in graph.edges(data=True)]
    edge_mag_min = min(edge_magnitudes) if edge_magnitudes else 0.0
    edge_mag_max = max(edge_magnitudes) if edge_magnitudes else 1.0
    if np.isclose(edge_mag_min, edge_mag_max):
        edge_mag_max = edge_mag_min + 1.0

    def _edge_width(abs_ddg: float) -> float:
        scaled = (abs_ddg - edge_mag_min) / max(edge_mag_max - edge_mag_min, 1e-12)
        return 2.8 + 4.2 * scaled

    edge_color = "#7c3aed"
    node_degree = dict(graph.degree())
    node_radius = {node: 26.0 + 2.0 * node_degree[node] for node in graph.nodes}
    node_fill = {}
    for node, value in zip(graph.nodes, node_values):
        if norm is not None and cmap is not None and np.isfinite(value):
            node_fill[node] = _rgba_to_hex(cmap(norm(float(value))))
        else:
            node_fill[node] = "#88c0d0"

    edge_svg: list[str] = []
    label_svg: list[str] = []
    label_specs_display: list[dict[str, np.ndarray]] = []
    label_payloads: list[tuple[str, str]] = []
    for node_a, node_b, data in graph.edges(data=True):
        edge_key = f"{node_a}~{node_b}"
        curvature = _edge_curvature(str(node_a), str(node_b))
        start = np.asarray(_to_xy(pos[node_a]), dtype=float)
        end = np.asarray(_to_xy(pos[node_b]), dtype=float)
        direction = end - start
        unit_dir = _normalize_vec(direction, fallback=np.array([1.0, 0.0]))
        perp = np.array([-unit_dir[1], unit_dir[0]])
        stroke_width = _edge_width(abs(float(data.get("calc_DDG", 0.0))))
        head_length = 11.0 + 1.6 * stroke_width
        head_half_width = 4.5 + 0.85 * stroke_width
        start2 = start + unit_dir * (node_radius[node_a] + 4.0 + 0.35 * stroke_width)
        tip = end - unit_dir * (node_radius[node_b] + 7.0 + 0.65 * stroke_width)
        span = np.linalg.norm(tip - start2)
        control = 0.5 * (start2 + tip) + perp * curvature * span * 0.75
        tip_tangent = _normalize_vec(
            _quadratic_bezier_tangent(start2, control, tip, 1.0),
            fallback=unit_dir,
        )
        tip_normal = np.array([-tip_tangent[1], tip_tangent[0]])
        shaft_end = tip - tip_tangent * head_length
        arrow_left = shaft_end + tip_normal * head_half_width
        arrow_right = shaft_end - tip_normal * head_half_width
        path_d = (
            f"M {start2[0]:.2f} {start2[1]:.2f} "
            f"Q {control[0]:.2f} {control[1]:.2f} {shaft_end[0]:.2f} {shaft_end[1]:.2f}"
        )
        hit_width = max(14.0, stroke_width + 10.0)
        edge_svg.append(
            f"<g class=\"edge-path\" data-edge=\"{html.escape(edge_key)}\">"
            f"<path d=\"{path_d}\" fill=\"none\" stroke=\"transparent\" stroke-width=\"{hit_width:.2f}\" "
            "stroke-linecap=\"round\" pointer-events=\"stroke\" />"
            f"<path d=\"{path_d}\" fill=\"none\" stroke=\"{edge_color}\" "
            f"stroke-width=\"{stroke_width:.2f}\" stroke-linecap=\"round\" stroke-opacity=\"0.96\" />"
            f"<polygon points=\"{tip[0]:.2f},{tip[1]:.2f} {arrow_left[0]:.2f},{arrow_left[1]:.2f} "
            f"{arrow_right[0]:.2f},{arrow_right[1]:.2f}\" fill=\"{edge_color}\" stroke=\"{edge_color}\" "
            "stroke-linejoin=\"round\" stroke-linecap=\"round\" />"
            "</g>"
        )

        text_pos = 0.25 * start2 + 0.5 * control + 0.25 * tip
        text_pos = text_pos + perp * curvature * span * 0.18
        label_specs_display.append(
            {"base": text_pos, "tangent": unit_dir, "normal": perp}
        )
        label_payloads.append(
            (
                edge_key,
                html.escape(
                    f"{float(data.get('calc_DDG', 0.0)):+.2f}\n±{float(data.get('calc_dDDG', 0.0)):.2f}"
                ),
            )
        )

    resolved_label_positions = _resolve_label_positions(
        label_specs_display,
        box_size=(60.0, 42.0),
    )

    for resolved_pos, (edge_key, edge_label) in zip(resolved_label_positions, label_payloads):
        edge_label_lines = edge_label.split("\n")
        label_svg.append(
            f"<g class=\"edge-label\" data-edge=\"{html.escape(edge_key)}\">"
            f"<rect x=\"{resolved_pos[0] - 30:.2f}\" y=\"{resolved_pos[1] - 22:.2f}\" width=\"60\" height=\"42\" "
            "rx=\"6\" ry=\"6\" fill=\"white\" fill-opacity=\"0.92\" stroke=\"#cbd2d9\" stroke-width=\"1.0\" />"
            f"<text x=\"{resolved_pos[0]:.2f}\" y=\"{resolved_pos[1] - 4:.2f}\" text-anchor=\"middle\" "
            "font-size=\"12\" fill=\"#243b53\">"
            f"{edge_label_lines[0]}</text>"
            f"<text x=\"{resolved_pos[0]:.2f}\" y=\"{resolved_pos[1] + 13:.2f}\" text-anchor=\"middle\" "
            "font-size=\"12\" fill=\"#243b53\">"
            f"{edge_label_lines[1]}</text>"
            "</g>"
        )

    node_svg: list[str] = []
    for node in graph.nodes:
        x, y = _to_xy(pos[node])
        label = html.escape(str(node))
        node_svg.append(
            "<g class=\"node\" "
            f"data-ligand=\"{label}\" transform=\"translate({x:.2f},{y:.2f})\">"
            f"<circle r=\"{node_radius[node]:.2f}\" fill=\"{node_fill[node]}\" stroke=\"#243b53\" stroke-width=\"3\" />"
            f"<text text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"18\" font-weight=\"700\" "
            "fill=\"#102a43\" paint-order=\"stroke\" stroke=\"white\" stroke-width=\"5\" stroke-linejoin=\"round\">"
            f"{label}</text>"
            "</g>"
        )

    network_notes = [
        (
            "Direction mode: merged opposite directions"
            if merge_bidirectional
            else "Direction mode: split stored directions"
        ),
        "Use mouse wheel to zoom and drag the background to pan",
        "Click a node to pin a ligand structure card",
        "Arrows point from labelA to labelB",
        "Edge labels show ΔΔG ± s.e. (kcal/mol)",
        "Edge thickness scales with |ΔΔG|",
        (
            "Node colors use red → white → blue for negative → zero → positive ΔG"
            if color_mode == "absolute"
            else "Node colors reflect degree because no absolute ΔG solution was available"
        ),
    ]

    network_svg_html = f"""
      <div class="network-toolbar">
        <button class="zoom-btn" id="network-zoom-in" type="button">+</button>
        <button class="zoom-btn" id="network-zoom-out" type="button">−</button>
        <button class="zoom-btn" id="network-fit" type="button">Fit</button>
        <button class="zoom-btn" id="network-reset" type="button">Reset</button>
      </div>
      <svg id="network-svg" viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="{html.escape(title or 'BATTER RBFE network')}">
        <rect id="network-pan-surface" x="0" y="0" width="{canvas_w}" height="{canvas_h}" fill="#f6f7fb" />
        <g id="network-viewport">
          {''.join(edge_svg)}
          {''.join(label_svg)}
          {''.join(node_svg)}
        </g>
      </svg>
    """

    absolute_panel_html = "<div class=\"empty-panel\">Absolute ΔG values are not available for this network.</div>"
    absolute_notes = [
        (
            "Direction mode: merged opposite directions"
            if merge_bidirectional
            else "Direction mode: split stored directions"
        ),
        "Absolute ΔG values are not available for this network",
    ]
    if absolute_warning:
        absolute_notes.append(absolute_warning)

    if absolute_summary is not None and not absolute_summary.empty:
        abs_df = absolute_summary.copy()
        label_col = "label" if "label" in abs_df.columns else None
        dg_col = next((col for col in abs_df.columns if col.lower().startswith("dg")), None)
        err_col = next(
            (
                col
                for col in abs_df.columns
                if "uncertainty" in col.lower() or col.lower().startswith("ddg")
            ),
            None,
        )
        if label_col is not None and dg_col is not None:
            abs_df = abs_df.dropna(subset=[label_col, dg_col]).copy()
            if not abs_df.empty:
                abs_df["DG_shifted"] = pd.to_numeric(abs_df[dg_col], errors="coerce") + float(
                    absolute_offset
                )
                abs_df["DG_uncertainty"] = (
                    pd.to_numeric(abs_df[err_col], errors="coerce").fillna(0.0)
                    if err_col is not None
                    else 0.0
                )
                abs_df = abs_df.sort_values("DG_shifted", ascending=True, kind="stable").reset_index(drop=True)

                exp_map: dict[str, tuple[float, float]] = {}
                if exp_summary is not None and not exp_summary.empty:
                    exp_df = exp_summary.copy()
                    if "label" in exp_df.columns and "exp_DG" in exp_df.columns:
                        exp_df = exp_df.dropna(subset=["label", "exp_DG"]).copy()
                        if not exp_df.empty:
                            exp_df["exp_uncertainty"] = pd.to_numeric(
                                exp_df.get("exp_uncertainty", 0.0), errors="coerce"
                            ).fillna(0.0)
                            exp_map = {
                                str(row.label): (float(row.exp_DG), float(row.exp_uncertainty))
                                for row in exp_df.itertuples(index=False)
                            }

                labels = abs_df[label_col].astype(str).tolist()
                calc_values = abs_df["DG_shifted"].to_numpy(dtype=float)
                calc_errs = abs_df["DG_uncertainty"].to_numpy(dtype=float)
                exp_values = np.asarray(
                    [exp_map.get(label, (np.nan, np.nan))[0] for label in labels], dtype=float
                )
                exp_errs = np.asarray(
                    [exp_map.get(label, (np.nan, np.nan))[1] for label in labels], dtype=float
                )

                calc_min = np.nanmin(calc_values - calc_errs)
                calc_max = np.nanmax(calc_values + calc_errs)
                if np.isfinite(exp_values).any():
                    exp_min = np.nanmin(np.where(np.isfinite(exp_values), exp_values - exp_errs, np.nan))
                    exp_max = np.nanmax(np.where(np.isfinite(exp_values), exp_values + exp_errs, np.nan))
                    xmin = min(calc_min, exp_min)
                    xmax = max(calc_max, exp_max)
                else:
                    xmin = calc_min
                    xmax = calc_max
                if np.isclose(xmin, xmax):
                    xmax = xmin + 1.0

                abs_canvas_w = 1100
                row_h = 48
                top_pad = 70
                bottom_pad = 42
                left_pad = 210
                right_pad = 70
                abs_canvas_h = top_pad + bottom_pad + row_h * len(labels)
                plot_w = abs_canvas_w - left_pad - right_pad

                def _x(value: float) -> float:
                    return left_pad + ((float(value) - xmin) / (xmax - xmin)) * plot_w

                zero_x = _x(0.0)

                limit = max(abs(float(np.nanmin(calc_values))), abs(float(np.nanmax(calc_values))), 1e-8)
                bar_norm = None
                bar_cmap = None
                if cmap is not None and norm is not None and color_mode == "absolute":
                    bar_norm = norm
                    bar_cmap = cmap

                rows_svg: list[str] = []
                for idx, label in enumerate(labels):
                    y = top_pad + idx * row_h + row_h * 0.5
                    value = float(calc_values[idx])
                    err = float(calc_errs[idx])
                    x0 = _x(min(0.0, value))
                    x1 = _x(max(0.0, value))
                    bar_x = min(x0, x1)
                    bar_w = max(abs(x1 - x0), 1.5)
                    err_l = _x(value - err)
                    err_r = _x(value + err)
                    if bar_cmap is not None and bar_norm is not None:
                        fill = _rgba_to_hex(bar_cmap(bar_norm(value)))
                    else:
                        fill = "#88c0d0"
                    rows_svg.append(
                        f"<g class=\"bar-row\" data-ligand=\"{html.escape(label)}\">"
                        f"<text x=\"{left_pad - 16:.2f}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-size=\"14\" fill=\"#102a43\">{html.escape(label)}</text>"
                        f"<rect x=\"{bar_x:.2f}\" y=\"{y - 12:.2f}\" width=\"{bar_w:.2f}\" height=\"24\" rx=\"6\" ry=\"6\" fill=\"{fill}\" stroke=\"#0b7285\" stroke-width=\"1.2\" />"
                        f"<line x1=\"{err_l:.2f}\" y1=\"{y:.2f}\" x2=\"{err_r:.2f}\" y2=\"{y:.2f}\" stroke=\"#0b7285\" stroke-width=\"1.4\" />"
                        f"<line x1=\"{err_l:.2f}\" y1=\"{y - 7:.2f}\" x2=\"{err_l:.2f}\" y2=\"{y + 7:.2f}\" stroke=\"#0b7285\" stroke-width=\"1.4\" />"
                        f"<line x1=\"{err_r:.2f}\" y1=\"{y - 7:.2f}\" x2=\"{err_r:.2f}\" y2=\"{y + 7:.2f}\" stroke=\"#0b7285\" stroke-width=\"1.4\" />"
                    )
                    if np.isfinite(exp_values[idx]):
                        exp_x = _x(float(exp_values[idx]))
                        exp_err = float(exp_errs[idx])
                        rows_svg.append(
                            f"<line x1=\"{_x(exp_values[idx] - exp_err):.2f}\" y1=\"{y:.2f}\" x2=\"{_x(exp_values[idx] + exp_err):.2f}\" y2=\"{y:.2f}\" stroke=\"#bc6c25\" stroke-width=\"1.2\" />"
                            f"<rect x=\"{exp_x - 4:.2f}\" y=\"{y - 4:.2f}\" width=\"8\" height=\"8\" fill=\"#bc6c25\" />"
                        )
                    rows_svg.append(
                        f"<text x=\"{right_pad + left_pad + plot_w - 4:.2f}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-size=\"12\" fill=\"#486581\">{value:+.2f} ± {err:.2f}</text></g>"
                    )

                x_ticks = np.linspace(xmin, xmax, 6)
                grid_svg = []
                for tick in x_ticks:
                    x = _x(float(tick))
                    grid_svg.append(
                        f"<line x1=\"{x:.2f}\" y1=\"{top_pad - 20:.2f}\" x2=\"{x:.2f}\" y2=\"{abs_canvas_h - bottom_pad + 6:.2f}\" stroke=\"#d9e2ec\" stroke-width=\"1\" />"
                        f"<text x=\"{x:.2f}\" y=\"{abs_canvas_h - 10:.2f}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#52606d\">{tick:.1f}</text>"
                    )

                absolute_panel_html = f"""
                  <svg viewBox=\"0 0 {abs_canvas_w} {abs_canvas_h}\" role=\"img\" aria-label=\"{html.escape(title or 'BATTER absolute ranking')}\">
                    <line x1=\"{zero_x:.2f}\" y1=\"{top_pad - 20:.2f}\" x2=\"{zero_x:.2f}\" y2=\"{abs_canvas_h - bottom_pad + 6:.2f}\" stroke=\"#7b8794\" stroke-dasharray=\"4 4\" stroke-width=\"1.2\" />
                    {''.join(grid_svg)}
                    {''.join(rows_svg)}
                    <text x=\"{left_pad + plot_w * 0.5:.2f}\" y=\"{top_pad - 34:.2f}\" text-anchor=\"middle\" font-size=\"14\" fill=\"#102a43\">Absolute ΔG (kcal/mol)</text>
                  </svg>
                """
                absolute_notes = [
                    (
                        "Direction mode: merged opposite directions"
                        if merge_bidirectional
                        else "Direction mode: split stored directions"
                    ),
                    "Click a ligand row to pin a ligand structure card",
                ]
                if exp_map:
                    absolute_notes.append("Experiment markers are shown as orange squares")
                if not np.isclose(float(absolute_offset), 0.0):
                    absolute_notes.append(f"Applied offset: {float(absolute_offset):+.2f} kcal/mol")

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title or "BATTER Cinnabar dashboard")}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7fb; color: #102a43; }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 18px 18px 28px; }}
    h1 {{ margin: 0 0 14px; font-size: 24px; text-align: center; }}
    .tabbar {{ display: flex; gap: 10px; margin-bottom: 14px; justify-content: center; }}
    .tab {{ border: 1px solid #cbd2d9; background: white; color: #334e68; border-radius: 999px; padding: 8px 16px; font-size: 14px; cursor: pointer; }}
    .tab.active {{ background: #7c3aed; border-color: #7c3aed; color: white; }}
    .panel {{ display: none; background: white; border: 1px solid #d9e2ec; border-radius: 16px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); overflow: hidden; }}
    .panel.active {{ display: block; }}
    .panel svg {{ width: 100%; height: auto; display: block; background: #f6f7fb; }}
    .network-toolbar {{ display: flex; justify-content: flex-end; gap: 8px; padding: 12px 14px 0; }}
    .zoom-btn {{ border: 1px solid #cbd2d9; background: white; color: #334e68; border-radius: 10px; padding: 6px 12px; font-size: 13px; cursor: pointer; }}
    .zoom-btn:hover {{ border-color: #9fb3c8; background: #f8fafc; }}
    .notes {{ margin: 12px 14px 14px; padding: 10px 12px; border: 1px solid #cbd2d9; border-radius: 10px; background: rgba(255,255,255,0.96); color: #486581; white-space: pre-line; }}
    .empty-panel {{ padding: 36px 24px; font-size: 15px; color: #52606d; text-align: center; }}
    .node, .bar-row, .edge-path, .edge-label {{ cursor: pointer; }}
    #network-svg {{ touch-action: none; user-select: none; }}
    #network-pan-surface {{ cursor: grab; }}
    #network-pan-surface.dragging {{ cursor: grabbing; }}
    #stickies {{ position: fixed; inset: 0; pointer-events: none; z-index: 1000; }}
    .sticky-note {{ position: fixed; width: 280px; min-height: 160px; background: #fff9c4; border: 1px solid #e0c56e; border-radius: 14px; box-shadow: 0 16px 38px rgba(15, 23, 42, 0.18); padding: 12px 12px 10px; pointer-events: auto; }}
    .sticky-note.edge-note {{ width: 360px; background: #eef2ff; border-color: #c7d2fe; }}
    .sticky-header {{ display: flex; align-items: center; justify-content: space-between; font-weight: 700; margin-bottom: 8px; color: #6b4f00; cursor: move; }}
    .sticky-note.edge-note .sticky-header {{ color: #3730a3; }}
    .sticky-close {{ border: 0; background: transparent; color: #6b4f00; font-size: 18px; line-height: 1; cursor: pointer; }}
    .sticky-note.edge-note .sticky-close {{ color: #3730a3; }}
    .sticky-body .smiles {{ margin-top: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #52606d; word-break: break-all; }}
    .sticky-body .empty {{ font-size: 12px; color: #7b8794; }}
    .sticky-body img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #cbd2d9; background: white; }}
    .sticky-meta {{ margin-top: 8px; font-size: 11px; color: #52606d; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title or "BATTER Cinnabar dashboard")}</h1>
    <div class="tabbar">
      <button class="tab active" data-panel="network-panel">Network</button>
      <button class="tab" data-panel="absolute-panel">Absolute</button>
    </div>
    <section id="network-panel" class="panel active">
      {network_svg_html}
      <div class="notes">{html.escape(chr(10).join(network_notes))}</div>
    </section>
    <section id="absolute-panel" class="panel">
      {absolute_panel_html}
      <div class="notes">{html.escape(chr(10).join(absolute_notes))}</div>
    </section>
  </div>
  <div id="stickies"></div>
  <script>
    const ligandAssets = {json.dumps(assets)};
    const edgeAssets = {json.dumps(mapping_assets)};
    const stickyRoot = document.getElementById('stickies');
    let zCounter = 1000;
    const networkSvg = document.getElementById('network-svg');
    const networkViewport = document.getElementById('network-viewport');
    const networkPanSurface = document.getElementById('network-pan-surface');
    let networkScale = 1.0;
    let networkPanX = 0.0;
    let networkPanY = 0.0;
    let networkDragging = false;
    let dragStartX = 0.0;
    let dragStartY = 0.0;
    let dragPanX = 0.0;
    let dragPanY = 0.0;

    function updateNetworkTransform() {{
      if (!networkViewport) return;
      networkViewport.setAttribute(
        'transform',
        `translate(${{networkPanX.toFixed(2)}} ${{networkPanY.toFixed(2)}}) scale(${{networkScale.toFixed(5)}})`
      );
    }}

    function fitNetworkViewport(extraScale = 1.0) {{
      if (!networkSvg || !networkViewport) return;
      const bbox = networkViewport.getBBox();
      const viewBox = networkSvg.viewBox.baseVal;
      if (!bbox || bbox.width <= 0 || bbox.height <= 0) return;
      const pad = 32.0;
      const scaleX = (viewBox.width - 2.0 * pad) / bbox.width;
      const scaleY = (viewBox.height - 2.0 * pad) / bbox.height;
      networkScale = Math.min(scaleX, scaleY) * extraScale;
      networkPanX = viewBox.x + (viewBox.width - bbox.width * networkScale) * 0.5 - bbox.x * networkScale;
      networkPanY = viewBox.y + (viewBox.height - bbox.height * networkScale) * 0.5 - bbox.y * networkScale;
      updateNetworkTransform();
    }}

    function zoomNetwork(factor, clientX = null, clientY = null) {{
      if (!networkSvg || !networkViewport) return;
      const viewBox = networkSvg.viewBox.baseVal;
      const rect = networkSvg.getBoundingClientRect();
      const anchorX = clientX === null ? rect.left + rect.width * 0.5 : clientX;
      const anchorY = clientY === null ? rect.top + rect.height * 0.5 : clientY;
      const svgX = viewBox.x + ((anchorX - rect.left) / rect.width) * viewBox.width;
      const svgY = viewBox.y + ((anchorY - rect.top) / rect.height) * viewBox.height;
      const nextScale = Math.min(8.0, Math.max(0.25, networkScale * factor));
      const localX = (svgX - networkPanX) / networkScale;
      const localY = (svgY - networkPanY) / networkScale;
      networkScale = nextScale;
      networkPanX = svgX - localX * networkScale;
      networkPanY = svgY - localY * networkScale;
      updateNetworkTransform();
    }}

    function stickyBodyHtml(label) {{
      const asset = ligandAssets[label] || {{}};
      const svg = asset.svg || '<div class="empty">No 2D structure available</div>';
      const smiles = asset.smiles ? `<div class="smiles">${{asset.smiles}}</div>` : '';
      return `<div class="sticky-body">${{svg}}${{smiles}}</div>`;
    }}

    function edgeBodyHtml(edgeKey) {{
      const asset = edgeAssets[edgeKey] || {{}};
      const image = asset.image_data_uri
        ? `<img src="${{asset.image_data_uri}}" alt="${{asset.display_title || edgeKey}} mapping graph" />`
        : '<div class="empty">No transformation mapping image available</div>';
      const meta = asset.run_id
        ? `<div class="sticky-meta">run_id: ${{asset.run_id}}<br />pair: ${{asset.pair_id || edgeKey}}</div>`
        : '';
      return `<div class="sticky-body">${{image}}${{meta}}</div>`;
    }}

    function bringToFront(note) {{
      zCounter += 1;
      note.style.zIndex = String(zCounter);
    }}

    function makeDraggable(note) {{
      const header = note.querySelector('.sticky-header');
      let startX = 0, startY = 0, startLeft = 0, startTop = 0, dragging = false;
      header.addEventListener('pointerdown', (event) => {{
        if (event.target && event.target.closest('.sticky-close')) {{
          return;
        }}
        dragging = true;
        bringToFront(note);
        startX = event.clientX;
        startY = event.clientY;
        startLeft = parseFloat(note.style.left || '0');
        startTop = parseFloat(note.style.top || '0');
        header.setPointerCapture(event.pointerId);
      }});
      header.addEventListener('pointermove', (event) => {{
        if (!dragging) return;
        note.style.left = `${{startLeft + event.clientX - startX}}px`;
        note.style.top = `${{startTop + event.clientY - startY}}px`;
      }});
      function endDrag(event) {{
        dragging = false;
        try {{ header.releasePointerCapture(event.pointerId); }} catch (_e) {{}}
      }}
      header.addEventListener('pointerup', endDrag);
      header.addEventListener('pointercancel', endDrag);
    }}

    function openSticky(label, event) {{
      const existing = document.querySelector(`.sticky-note[data-ligand="${{CSS.escape(label)}}"]`);
      if (existing) {{
        bringToFront(existing);
        return;
      }}
      const note = document.createElement('div');
      note.className = 'sticky-note';
      note.dataset.ligand = label;
      note.style.left = `${{Math.min(window.innerWidth - 320, Math.max(16, event.clientX + 12))}}px`;
      note.style.top = `${{Math.min(window.innerHeight - 260, Math.max(16, event.clientY + 12))}}px`;
      note.innerHTML = `
        <div class="sticky-header">
          <span>${{label}}</span>
          <button class="sticky-close" type="button" aria-label="Close">×</button>
        </div>
        ${{stickyBodyHtml(label)}}
      `;
      stickyRoot.appendChild(note);
      bringToFront(note);
      makeDraggable(note);
      note.addEventListener('pointerdown', () => bringToFront(note));
      const closeButton = note.querySelector('.sticky-close');
      closeButton.addEventListener('pointerdown', (event) => {{
        event.stopPropagation();
      }});
      closeButton.addEventListener('click', (event) => {{
        event.preventDefault();
        event.stopPropagation();
        note.remove();
      }});
    }}

    function openEdgeSticky(edgeKey, event) {{
      const existing = document.querySelector(`.sticky-note[data-edge="${{CSS.escape(edgeKey)}}"]`);
      if (existing) {{
        bringToFront(existing);
        return;
      }}
      const asset = edgeAssets[edgeKey] || {{}};
      const title = asset.display_title || edgeKey.replace('~', ' → ');
      const note = document.createElement('div');
      note.className = 'sticky-note edge-note';
      note.dataset.edge = edgeKey;
      note.style.left = `${{Math.min(window.innerWidth - 400, Math.max(16, event.clientX + 12))}}px`;
      note.style.top = `${{Math.min(window.innerHeight - 320, Math.max(16, event.clientY + 12))}}px`;
      note.innerHTML = `
        <div class="sticky-header">
          <span>${{title}}</span>
          <button class="sticky-close" type="button" aria-label="Close">×</button>
        </div>
        ${{edgeBodyHtml(edgeKey)}}
      `;
      stickyRoot.appendChild(note);
      bringToFront(note);
      makeDraggable(note);
      note.addEventListener('pointerdown', () => bringToFront(note));
      const closeButton = note.querySelector('.sticky-close');
      closeButton.addEventListener('pointerdown', (event) => {{
        event.stopPropagation();
      }});
      closeButton.addEventListener('click', (event) => {{
        event.preventDefault();
        event.stopPropagation();
        note.remove();
      }});
    }}

    document.querySelectorAll('.node, .bar-row').forEach((element) => {{
      element.addEventListener('click', (event) => {{
        const label = element.getAttribute('data-ligand') || '';
        if (label) {{
          openSticky(label, event);
        }}
      }});
    }});

    document.querySelectorAll('.edge-path, .edge-label').forEach((element) => {{
      element.addEventListener('click', (event) => {{
        const edgeKey = element.getAttribute('data-edge') || '';
        if (edgeKey) {{
          openEdgeSticky(edgeKey, event);
        }}
      }});
    }});

    document.getElementById('network-zoom-in')?.addEventListener('click', () => {{
      zoomNetwork(1.18);
    }});
    document.getElementById('network-zoom-out')?.addEventListener('click', () => {{
      zoomNetwork(1.0 / 1.18);
    }});
    document.getElementById('network-fit')?.addEventListener('click', () => {{
      fitNetworkViewport(1.0);
    }});
    document.getElementById('network-reset')?.addEventListener('click', () => {{
      fitNetworkViewport(0.96);
    }});

    networkSvg?.addEventListener('wheel', (event) => {{
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : (1.0 / 1.12);
      zoomNetwork(factor, event.clientX, event.clientY);
    }}, {{ passive: false }});

    networkSvg?.addEventListener('pointerdown', (event) => {{
      if (!networkPanSurface) return;
      if (event.target && event.target.closest('.node, .edge-path, .edge-label')) {{
        return;
      }}
      networkDragging = true;
      dragStartX = event.clientX;
      dragStartY = event.clientY;
      dragPanX = networkPanX;
      dragPanY = networkPanY;
      networkPanSurface.classList.add('dragging');
      networkSvg.setPointerCapture(event.pointerId);
    }});

    networkSvg?.addEventListener('pointermove', (event) => {{
      if (!networkDragging) return;
      networkPanX = dragPanX + (event.clientX - dragStartX);
      networkPanY = dragPanY + (event.clientY - dragStartY);
      updateNetworkTransform();
    }});

    function endNetworkDrag(event) {{
      if (!networkDragging) return;
      networkDragging = false;
      networkPanSurface?.classList.remove('dragging');
      try {{ networkSvg?.releasePointerCapture(event.pointerId); }} catch (_e) {{}}
    }}

    networkSvg?.addEventListener('pointerup', endNetworkDrag);
    networkSvg?.addEventListener('pointercancel', endNetworkDrag);
    networkSvg?.addEventListener('pointerleave', endNetworkDrag);

    document.querySelectorAll('.tab').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('active'));
        document.querySelectorAll('.panel').forEach((panel) => panel.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.panel).classList.add('active');
      }});
    }});

    fitNetworkViewport(0.96);
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path.exists()


def write_cinnabar_outputs(
    result: CinnabarConversionResult,
    out_dir: str | Path,
    *,
    method_name: str = "BATTER",
    target_name: str = "",
    write_plots: bool = True,
    absolute_offset: float = 0.0,
) -> dict[str, Path]:
    """Write stable on-disk outputs for a converted Cinnabar bundle."""
    _FEMap, plotting, _unit = _import_cinnabar_stack()
    directionality = summarize_directionality(result.edge_summary)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    raw_path = out_root / "raw_signed.csv"
    result.raw_signed.to_csv(raw_path, index=False)
    outputs["raw_signed_csv"] = raw_path

    edge_path = out_root / "edge_summary.csv"
    result.edge_summary.to_csv(edge_path, index=False)
    outputs["edge_summary_csv"] = edge_path

    rel_path = out_root / "cinnabar_relative.csv"
    result.femap.get_relative_dataframe().to_csv(rel_path, index=False)
    outputs["cinnabar_relative_csv"] = rel_path

    title = method_name if not target_name else f"{method_name}: {target_name}"

    if result.exp_summary is not None:
        exp_path = out_root / "experimental_summary.csv"
        result.exp_summary.to_csv(exp_path, index=False)
        outputs["experimental_summary_csv"] = exp_path

    if result.absolute_summary is not None:
        abs_path = out_root / "cinnabar_absolute.csv"
        result.absolute_summary.to_csv(abs_path, index=False)
        outputs["cinnabar_absolute_csv"] = abs_path
        abs_plot_path = out_root / "cinnabar_absolute_sorted.png"
        if _render_absolute_sorted_png(
            result.absolute_summary,
            abs_plot_path,
            exp_summary=result.exp_summary,
            title=title,
            absolute_offset=absolute_offset,
            merge_bidirectional=result.merge_bidirectional,
        ):
            outputs["absolute_sorted_png"] = abs_plot_path

    graph_path = out_root / "cinnabar_network.png"
    rendered = _render_network_png(
        result.edge_summary,
        graph_path,
        absolute_summary=result.absolute_summary,
        title=title,
        merge_bidirectional=result.merge_bidirectional,
    )
    if not rendered:
        try:
            result.femap.draw_graph(filename=str(graph_path), title=title)
            rendered = graph_path.exists()
        except Exception:
            rendered = False
    if rendered:
        outputs["network_png"] = graph_path
    dashboard_html_path = out_root / "cinnabar_dashboard.html"
    if _render_dashboard_html(
        result.edge_summary,
        dashboard_html_path,
        absolute_summary=result.absolute_summary,
        exp_summary=result.exp_summary,
        title=title,
        absolute_offset=absolute_offset,
        merge_bidirectional=result.merge_bidirectional,
        ligand_assets=result.ligand_assets,
        edge_assets=result.edge_assets,
        absolute_warning=result.absolute_warning,
    ):
        outputs["dashboard_html"] = dashboard_html_path

    if write_plots and result.exp_summary is not None:
        try:
            graph = result.femap.to_legacy_graph()
            dg_path = out_root / "cinnabar_dg.png"
            plotting.plot_DGs(
                graph,
                method_name=method_name,
                target_name=target_name,
                filename=str(dg_path),
            )
            if dg_path.exists():
                outputs["dg_png"] = dg_path
        except Exception:
            pass
        try:
            graph = result.femap.to_legacy_graph()
            ddg_path = out_root / "cinnabar_ddg.png"
            plotting.plot_DDGs(
                graph,
                method_name=method_name,
                target_name=target_name,
                filename=str(ddg_path),
            )
            if ddg_path.exists():
                outputs["ddg_png"] = ddg_path
        except Exception:
            pass

    manifest = {
        "n_edges": int(len(result.edge_summary)),
        "n_measurements": int(len(result.raw_signed)),
        "has_experimental": bool(result.exp_summary is not None),
        "has_absolute": bool(result.absolute_summary is not None),
        "absolute_warning": result.absolute_warning or "",
        "absolute_offset": float(absolute_offset),
        "direction_mode": "merged" if result.merge_bidirectional else "split",
        "n_directional_edges": directionality["n_directional_edges"],
        "n_reciprocal_pairs": directionality["n_reciprocal_pairs"],
        "reciprocal_pairs": directionality["reciprocal_pairs"],
        "outputs": {key: path.name for key, path in outputs.items()},
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs["manifest_json"] = manifest_path

    return outputs
