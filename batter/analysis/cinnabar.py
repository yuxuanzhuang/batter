"""Helpers for converting BATTER RBFE results into Cinnabar ``FEMap`` objects."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import networkx as nx

from batter.api import list_fe_runs

__all__ = [
    "CinnabarConversionResult",
    "build_batter_rbfe_cinnabar",
    "build_batter_rbfe_cinnabar_by_run",
    "dataframe_to_cinnabar",
    "load_batter_rbfe_results",
    "write_cinnabar_outputs",
]


@dataclass
class CinnabarConversionResult:
    femap: Any
    edge_summary: pd.DataFrame
    raw_signed: pd.DataFrame
    exp_summary: pd.DataFrame | None = None
    absolute_summary: pd.DataFrame | None = None


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
    try:
        femap.generate_absolute_values()
        absolute_summary = femap.get_absolute_dataframe()
    except Exception:
        absolute_summary = None

    return CinnabarConversionResult(
        femap=femap,
        edge_summary=edge_summary,
        raw_signed=raw_signed,
        exp_summary=exp_summary,
        absolute_summary=absolute_summary,
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
    return dataframe_to_cinnabar(
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
        out[str(run_id)] = dataframe_to_cinnabar(
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
    return out


def _render_network_png(
    edge_summary: pd.DataFrame,
    out_path: Path,
    *,
    absolute_summary: pd.DataFrame | None = None,
    title: str = "",
) -> bool:
    """Render a BATTER-styled RBFE network figure from summarized edge data."""
    if edge_summary.empty:
        return False

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
        from matplotlib import colors as mcolors
        from matplotlib import cm, colormaps
    except Exception:
        return False

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

    if graph.number_of_nodes() == 1:
        only = next(iter(graph.nodes))
        pos = {only: np.array([0.0, 0.0])}
    elif graph.number_of_nodes() == 2:
        nodes = list(graph.nodes)
        pos = {nodes[0]: np.array([-1.0, 0.0]), nodes[1]: np.array([1.0, 0.0])}
    else:
        pos = nx.kamada_kawai_layout(graph)

    node_degree = dict(graph.degree())
    node_sizes = [1400 + 220 * node_degree[node] for node in graph.nodes]

    node_colors = None
    colorbar_label = None
    norm = None
    cmap = None
    if absolute_summary is not None and not absolute_summary.empty:
        abs_df = absolute_summary.copy()
        dg_col = next(
            (col for col in abs_df.columns if col.lower().startswith("dg")),
            None,
        )
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
                node_colors = [dg_map.get(node, np.nan) for node in graph.nodes]
                finite_colors = [value for value in node_colors if np.isfinite(value)]
                if finite_colors:
                    vmin = min(finite_colors)
                    vmax = max(finite_colors)
                    if np.isclose(vmin, vmax):
                        vmax = vmin + 1.0
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    cmap = colormaps["viridis_r"]
                    colorbar_label = "MLE ΔG (kcal/mol)"
                else:
                    node_colors = None

    if node_colors is None:
        node_colors = [node_degree[node] for node in graph.nodes]
        vmax = max(node_colors) if node_colors else 1
        norm = mcolors.Normalize(vmin=0, vmax=max(1, vmax))
        cmap = colormaps["Blues"]
        colorbar_label = "Node degree"

    def _edge_curvature(node_a: str, node_b: str) -> float:
        if graph.has_edge(node_b, node_a) and node_a != node_b:
            ordered = tuple(sorted((str(node_a), str(node_b))))
            return 0.18 if (str(node_a), str(node_b)) == ordered else -0.18
        return 0.0

    edge_widths = []
    edge_colors = []
    edge_metadata: list[tuple[str, str, dict[str, Any], float]] = []
    edge_uncertainties = []
    for node_a, node_b, data in graph.edges(data=True):
        curvature = _edge_curvature(str(node_a), str(node_b))
        edge_metadata.append((str(node_a), str(node_b), data, curvature))
        edge_uncertainties.append(float(data.get("calc_dDDG", 0.0)))
    if edge_uncertainties:
        emin = min(edge_uncertainties)
        emax = max(edge_uncertainties)
        if np.isclose(emin, emax):
            emax = emin + 1.0
        edge_norm = mcolors.Normalize(vmin=emin, vmax=emax)
        edge_cmap = colormaps["magma_r"]
    else:
        edge_norm = mcolors.Normalize(vmin=0, vmax=1)
        edge_cmap = colormaps["magma_r"]

    for node_a, node_b, data, _curvature in edge_metadata:
        n_measurements = max(1, int(data.get("n_measurements", 1)))
        edge_widths.append(2.0 + 0.45 * np.log1p(n_measurements))
        edge_colors.append(edge_cmap(edge_norm(float(data.get("calc_dDDG", 0.0)))))

    fig_w = max(7.0, 1.8 * graph.number_of_nodes())
    fig_h = max(5.5, 1.5 * graph.number_of_nodes())
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f7fb")

    for (node_a, node_b, data, curvature), edge_width, edge_color in zip(
        edge_metadata, edge_widths, edge_colors
    ):
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=[(node_a, node_b)],
            width=edge_width,
            edge_color=[edge_color],
            alpha=0.95,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=24,
            min_source_margin=18,
            min_target_margin=18,
            connectionstyle=f"arc3,rad={curvature}",
        )
    node_artist = nx.draw_networkx_nodes(
        graph,
        pos,
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
        pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
        font_color="#102a43",
    )
    for text in label_text.values():
        text.set_path_effects(
            [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)]
        )

    for node_a, node_b, data, curvature in edge_metadata:
        start = np.asarray(pos[node_a], dtype=float)
        end = np.asarray(pos[node_b], dtype=float)
        midpoint = 0.5 * (start + end)
        direction = end - start
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 0:
            perp = np.array([-direction[1], direction[0]]) / norm_dir
        else:
            perp = np.array([0.0, 0.0])
        text_pos = midpoint + perp * curvature * 0.55
        edge_label = (
            f"{float(data.get('calc_DDG', 0.0)):+.2f}\n"
            f"±{float(data.get('calc_dDDG', 0.0)):.2f}"
        )
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

    ax.text(
        0.01,
        0.01,
        "Edge labels show ΔΔG ± s.e. (kcal/mol)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#486581",
    )
    ax.text(
        0.01,
        0.055,
        "Arrows point from labelA to labelB",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#486581",
    )

    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path.exists()


def _render_absolute_sorted_png(
    absolute_summary: pd.DataFrame,
    out_path: Path,
    *,
    exp_summary: pd.DataFrame | None = None,
    title: str = "",
    absolute_offset: float = 0.0,
) -> bool:
    """Render a sorted absolute free-energy ranking plot."""
    if absolute_summary is None or absolute_summary.empty:
        return False

    try:
        import matplotlib.pyplot as plt
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

    abs_df["DG_shifted"] = pd.to_numeric(abs_df[dg_col], errors="coerce") + float(
        absolute_offset
    )
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
    labels = abs_df[label_col].astype(str).tolist()

    ax.errorbar(
        calc_values,
        y,
        xerr=calc_errs,
        fmt="o",
        color="#0b7285",
        ecolor="#0b7285",
        elinewidth=1.4,
        capsize=3,
        markersize=6.5,
        label="BATTER MLE",
        zorder=3,
    )

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

    if exp_map:
        ax.legend(frameon=False, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
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
        ):
            outputs["absolute_sorted_png"] = abs_plot_path

    graph_path = out_root / "cinnabar_network.png"
    rendered = _render_network_png(
        result.edge_summary,
        graph_path,
        absolute_summary=result.absolute_summary,
        title=title,
    )
    if not rendered:
        try:
            result.femap.draw_graph(filename=str(graph_path), title=title)
            rendered = graph_path.exists()
        except Exception:
            rendered = False
    if rendered:
        outputs["network_png"] = graph_path

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
        "absolute_offset": float(absolute_offset),
        "outputs": {key: path.name for key, path in outputs.items()},
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs["manifest_json"] = manifest_path

    return outputs
