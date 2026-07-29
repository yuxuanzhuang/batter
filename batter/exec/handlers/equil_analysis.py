"""Analyse equilibration trajectories to determine binding status."""

from __future__ import annotations

import json
import inspect
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import MDAnalysis as mda
import numpy as np
import pandas as pd
from loguru import logger
from MDAnalysis.analysis import align

from batter.analysis.sim_validation import (
    STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
    SimValidator,
)
from batter._internal.ops.cleanup import cleanup_equil_after_analysis
from batter._internal.ops.box import _restore_protein_resids_from_renum
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils import cpptraj, run_with_log

PROLIF_INTERACTIONS_SCHEMA_VERSION = 3
PROLIF_OCCUPANCY_THRESHOLD = 0.30
PROLIF_CANDIDATE_EXCLUDED_INTERACTIONS = frozenset(
    {"hydrophobic", "vdwcontact", "vdwinteraction", "vdwinteractions"}
)
PROLIF_SALT_BRIDGE_INTERACTIONS = frozenset(
    {
        "anionic",
        "cationic",
        "ionic",
        "ionicinteraction",
        "ionicinteractions",
        "saltbridge",
        "saltbridges",
    }
)
PROLIF_HBOND_INTERACTIONS = frozenset(
    {
        "hbacceptor",
        "hbdonor",
        "hbond",
        "hbondacceptor",
        "hbonddonor",
    }
)
SALT_BRIDGE_DISTANCE_CUTOFF = 4.0
PROTEIN_POSITIVE_ATOMS = frozenset(
    {
        ("ARG", "NE"),
        ("ARG", "NH1"),
        ("ARG", "NH2"),
        ("LYS", "NZ"),
        ("HIP", "ND1"),
        ("HIP", "NE2"),
    }
)
PROTEIN_NEGATIVE_ATOMS = frozenset(
    {
        ("ASP", "OD1"),
        ("ASP", "OD2"),
        ("GLU", "OE1"),
        ("GLU", "OE2"),
    }
)
PROLIF_ARTIFACT_FILENAMES = {
    "timeseries_csv_gz": "prolif_interactions_timeseries.csv.gz",
    "barcode_png": "prolif_interactions_barcode.png",
    "occupancy_png": "prolif_interactions_occupancy.png",
    "lignetwork_html": "prolif_lignetwork.html",
    "interaction_diagram_png": "prolif_interaction_diagram.png",
}


def _paths(root: Path) -> dict[str, Path]:
    """Return commonly accessed equilibration paths under ``root``."""
    eq = root / "equil"
    prot_renum = eq / "q_build_files" / "protein_renum.txt"
    if not prot_renum.exists():
        prot_renum = eq / "protein_renum.txt"
    return {
        "equil_dir": eq,
        "finished": eq / "FINISHED",
        "failed": eq / "FAILED",
        "unbound": eq / "UNBOUND",
        "rep_pdb": eq / "representative.pdb",
        "rep_rst": eq / "representative.rst7",
        "stable_boresch_distance": eq / "stable_boresch_distance.json",
        "prolif_interactions": eq / "prolif_interactions.json",
        "prolif_timeseries": eq / PROLIF_ARTIFACT_FILENAMES["timeseries_csv_gz"],
        "prolif_barcode": eq / PROLIF_ARTIFACT_FILENAMES["barcode_png"],
        "prolif_occupancy": eq / PROLIF_ARTIFACT_FILENAMES["occupancy_png"],
        "prolif_lignetwork": eq / PROLIF_ARTIFACT_FILENAMES["lignetwork_html"],
        "prolif_interaction_diagram": eq
        / PROLIF_ARTIFACT_FILENAMES["interaction_diagram_png"],
        "build_files": eq / "q_build_files",
        "prot_renum": prot_renum,
        "full_pdb": eq / "full.pdb",
        "anchors_json": eq / "anchors.json",
    }


def _stable_boresch_distance_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    try:
        schema_version = int(data.get("schema_version", 0))
    except Exception:
        return False
    return schema_version >= STABLE_BORESCH_DISTANCE_SCHEMA_VERSION


def _prolif_interactions_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    try:
        schema_version = int(data.get("schema_version", 0))
    except Exception:
        return False
    return schema_version >= PROLIF_INTERACTIONS_SCHEMA_VERSION


_ANCHOR_MASK_RE = re.compile(r"^:?(?P<resid>-?\d+)@(?P<atom>[^,\s]+)$")


def _parse_anchor_mask(mask: str) -> tuple[int, str]:
    match = _ANCHOR_MASK_RE.match(str(mask).strip())
    if match is None:
        raise ValueError(f"Invalid prepared anchor mask in equil/anchors.json: {mask!r}")
    return int(match.group("resid")), match.group("atom")


def _load_equil_anchor_masks(equil_dir: Path) -> list[str]:
    anchors_path = Path(equil_dir) / "anchors.json"
    if not anchors_path.is_file():
        raise FileNotFoundError(f"Missing required prepared anchor file: {anchors_path}")
    try:
        data = json.loads(anchors_path.read_text())
    except Exception as exc:
        raise ValueError(f"Could not read prepared anchor file {anchors_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Prepared anchor file {anchors_path} must contain a JSON object.")
    masks: list[str] = []
    for key in ("P1", "P2", "P3"):
        value = str(data.get(key, "") or "").strip()
        if not value:
            raise ValueError(f"Prepared anchor file {anchors_path} is missing {key}.")
        _parse_anchor_mask(value)
        masks.append(value)
    return masks


def _load_protein_renum(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing required protein renumbering file: {path}")
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )


def _equil_anchor_masks_to_original_resids(
    masks: Sequence[str],
    protein_renum_path: Path,
) -> list[str]:
    renum = _load_protein_renum(protein_renum_path)
    out: list[str] = []
    for mask in masks:
        resid, atom = _parse_anchor_mask(mask)
        # Prepared equil systems include the leading DUM residue, so protein
        # anchor masks are one residue higher than protein_renum.txt new_resid.
        rows = renum[renum["new_resid"].astype(int) == int(resid) - 1]
        if len(rows) != 1:
            raise ValueError(
                f"Could not map prepared anchor mask {mask!r} through "
                f"{protein_renum_path}; matched {len(rows)} row(s)."
            )
        old_resid = int(rows.iloc[0]["old_resid"])
        out.append(f":{old_resid}@{atom}")
    return out


def _trailing_analysis_start_frame(n_frames: int, tail_fraction: float) -> int:
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in the interval (0, 1].")
    if n_frames <= 1:
        return 0
    start_frame = int(np.floor(n_frames * (1.0 - tail_fraction)))
    return min(max(start_frame, 0), n_frames - 1)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _dedupe_preserve_order(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _prolif_residue_metadata(value: Any) -> dict[str, Any]:
    label = str(value)
    resname = (
        str(getattr(value, "name", "") or getattr(value, "resname", "") or "").strip()
    )
    resid = _int_or_none(
        getattr(value, "number", None) or getattr(value, "resid", None)
    )
    chain = str(
        getattr(value, "chain", "") or getattr(value, "chainID", "") or ""
    ).strip()

    match = re.search(r"([A-Za-z]{1,4})(-?\d+)(?:[.:_-]?([A-Za-z0-9]+))?", label)
    if match:
        if not resname:
            resname = match.group(1)
        if resid is None:
            resid = _int_or_none(match.group(2))
        if not chain and match.group(3):
            chain = match.group(3)

    return {
        "label": label,
        "resname": resname,
        "resid": resid,
        "chainID": chain,
    }


def _prolif_column_parts(column: Any) -> tuple[Any, Any, Any] | None:
    parts = list(column) if isinstance(column, tuple) else [column]
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def _prolif_artifact_paths(prolif_path: Path) -> dict[str, Path]:
    return {
        key: prolif_path.parent / filename
        for key, filename in PROLIF_ARTIFACT_FILENAMES.items()
    }


def _shorten_label(value: Any, max_len: int = 64) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _prolif_interaction_id(column: Any) -> str:
    parsed = _prolif_column_parts(column)
    if parsed is None:
        return _shorten_label(column, 120)
    ligand_id, protein_id, interaction = parsed
    return "|".join(
        str(item).replace("|", "/")
        for item in (ligand_id, protein_id, interaction)
    )


def _prolif_interaction_display_label(column: Any) -> str:
    parsed = _prolif_column_parts(column)
    if parsed is None:
        return _shorten_label(column)
    _ligand_id, protein_id, interaction = parsed
    protein_meta = _prolif_residue_metadata(protein_id)
    protein_label = protein_meta.get("label") or str(protein_id)
    return _shorten_label(f"{protein_label} {interaction}")


def _bool_prolif_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    return df.fillna(False).astype(bool)


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _write_empty_prolif_plot(path: Path, *, title: str, message: str) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6.8, 2.8), dpi=150)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_prolif_timeseries_csv(df: pd.DataFrame, path: Path) -> None:
    bool_df = _bool_prolif_dataframe(df)
    out = pd.DataFrame({"frame": list(bool_df.index)})
    seen: dict[str, int] = {}
    for column in bool_df.columns:
        label = _prolif_interaction_id(column)
        count = seen.get(label, 0)
        seen[label] = count + 1
        if count:
            label = f"{label}#{count + 1}"
        out[label] = bool_df[column].astype(np.uint8).to_numpy()
    out.to_csv(path, index=False, compression="gzip")


def _prolif_columns_by_occupancy(
    bool_df: pd.DataFrame,
    *,
    max_interactions: int,
) -> list[Any]:
    if bool_df.empty or len(bool_df.columns) == 0:
        return []
    occupancy = bool_df.mean(axis=0).sort_values(ascending=False)
    active_columns = [column for column, value in occupancy.items() if float(value) > 0]
    if not active_columns:
        return []
    return active_columns[:max_interactions]


def _write_prolif_barcode_plot(
    df: pd.DataFrame,
    path: Path,
    *,
    max_interactions: int = 50,
) -> None:
    bool_df = _bool_prolif_dataframe(df)
    columns = _prolif_columns_by_occupancy(
        bool_df, max_interactions=max_interactions
    )
    if not columns:
        _write_empty_prolif_plot(
            path,
            title="ProLIF interaction barcode",
            message="No ligand-protein interactions were observed.",
        )
        return

    data = bool_df.loc[:, columns].T.astype(float).to_numpy()
    n_frames = int(data.shape[1])
    fig_h = min(max(3.2, 0.26 * len(columns) + 1.7), 14.0)
    fig_w = min(max(7.2, 0.015 * max(n_frames, 1) + 6.0), 14.0)
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    ax.set_title("ProLIF interaction barcode")
    ax.set_xlabel("Analyzed frame")
    ax.set_ylabel("Interaction")
    ax.set_yticks(np.arange(len(columns)))
    ax.set_yticklabels(
        [_prolif_interaction_display_label(column) for column in columns],
        fontsize=7,
    )
    tick_count = min(6, n_frames)
    if tick_count > 0:
        ticks = np.unique(np.linspace(0, n_frames - 1, tick_count, dtype=int))
        frame_labels = [str(bool_df.index[int(i)]) for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(frame_labels, rotation=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["absent", "present"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_prolif_occupancy_plot(
    interactions: Sequence[dict[str, Any]],
    path: Path,
    *,
    max_interactions: int = 30,
) -> None:
    active = [
        item
        for item in interactions
        if float(item.get("occupancy") or 0.0) > 0.0
    ][:max_interactions]
    if not active:
        _write_empty_prolif_plot(
            path,
            title="ProLIF interaction occupancy",
            message="No ligand-protein interactions were observed.",
        )
        return

    labels = []
    values = []
    for item in reversed(active):
        protein = item.get("protein") or {}
        protein_label = protein.get("label") or protein.get("resname") or "protein"
        labels.append(_shorten_label(f"{protein_label} {item.get('interaction', '')}"))
        values.append(float(item.get("occupancy") or 0.0))

    plt = _import_pyplot()
    fig_h = min(max(3.2, 0.27 * len(active) + 1.4), 12.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_h), dpi=150)
    ax.barh(np.arange(len(values)), values, color="#4C78A8")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Occupancy")
    ax.set_title("ProLIF interaction occupancy")
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_prolif_interaction_diagram(
    interactions: Sequence[dict[str, Any]],
    path: Path,
    *,
    ligand_label: str | None,
    max_residues: int = 24,
) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    for item in interactions:
        occupancy = float(item.get("occupancy") or 0.0)
        if occupancy <= 0.0:
            continue
        protein = item.get("protein") or {}
        protein_key = str(protein.get("label") or protein.get("resid") or "protein")
        entry = grouped.setdefault(
            protein_key,
            {
                "label": protein_key,
                "max_occupancy": 0.0,
                "interactions": [],
            },
        )
        entry["max_occupancy"] = max(float(entry["max_occupancy"]), occupancy)
        entry["interactions"].append(
            {
                "name": str(item.get("interaction") or "interaction"),
                "occupancy": occupancy,
            }
        )

    residues = sorted(
        grouped.values(),
        key=lambda item: (-float(item["max_occupancy"]), str(item["label"])),
    )[:max_residues]
    if not residues:
        _write_empty_prolif_plot(
            path,
            title="ProLIF interaction diagram",
            message="No ligand-protein interactions were observed.",
        )
        return

    plt = _import_pyplot()
    fig_h = min(max(4.0, 0.32 * len(residues) + 2.0), 12.0)
    fig, ax = plt.subplots(figsize=(8.8, fig_h), dpi=150)
    ax.axis("off")
    ax.set_title("ProLIF interaction diagram")

    ligand_name = _shorten_label(ligand_label or "ligand", 28)
    ligand_x, ligand_y = 0.12, 0.5
    ys = np.linspace(0.9, 0.1, len(residues))
    palette = {
        "Hydrophobic": "#4C78A8",
        "HBAcceptor": "#F58518",
        "HBDonor": "#54A24B",
        "PiStacking": "#B279A2",
        "Anionic": "#E45756",
        "Cationic": "#72B7B2",
        "CationPi": "#EECA3B",
        "VdWContact": "#9D755D",
    }

    ax.scatter([ligand_x], [ligand_y], s=1200, color="#343A40", zorder=3)
    ax.text(
        ligand_x,
        ligand_y,
        ligand_name,
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
    )

    for residue, y in zip(residues, ys):
        interactions_for_residue = sorted(
            residue["interactions"],
            key=lambda item: (-float(item["occupancy"]), str(item["name"])),
        )
        primary = interactions_for_residue[0]
        occupancy = float(residue["max_occupancy"])
        color = palette.get(str(primary["name"]), "#6B6B6B")
        protein_x = 0.78
        ax.plot(
            [ligand_x + 0.06, protein_x - 0.06],
            [ligand_y, y],
            color=color,
            linewidth=0.8 + 4.0 * occupancy,
            alpha=0.25 + 0.65 * occupancy,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [protein_x],
            [y],
            s=620,
            color="#F4F4F4",
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax.text(
            protein_x,
            y,
            _shorten_label(residue["label"], 20),
            ha="center",
            va="center",
            fontsize=7,
            color="#202020",
        )
        interaction_names = []
        for interaction in interactions_for_residue[:3]:
            interaction_names.append(
                f"{interaction['name']} {float(interaction['occupancy']):.2f}"
            )
        if len(interactions_for_residue) > 3:
            interaction_names.append(f"+{len(interactions_for_residue) - 3} more")
        ax.text(
            0.88,
            y,
            _shorten_label(", ".join(interaction_names), 44),
            ha="left",
            va="center",
            fontsize=7,
            color="#303030",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.02, 0.98)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_prolif_lignetwork_html(
    *,
    fingerprint: Any | None,
    ligand_selection: Any | None,
    prolif_module: Any | None,
    path: Path,
    threshold: float,
) -> None:
    if fingerprint is None or ligand_selection is None or prolif_module is None:
        path.write_text(
            "<!doctype html><html><body><p>"
            "ProLIF LigNetwork unavailable: fingerprint object was not provided."
            "</p></body></html>\n"
        )
        return

    ligand_mol = prolif_module.Molecule.from_mda(ligand_selection)
    view = fingerprint.plot_lignetwork(
        ligand_mol,
        kind="aggregate",
        threshold=float(threshold),
        height="650px",
        show_interaction_data=True,
    )
    view.save(path)


def _write_prolif_artifacts(
    *,
    prolif_path: Path,
    df: pd.DataFrame,
    interactions: Sequence[dict[str, Any]],
    ligand_label: str | None,
    fingerprint: Any | None = None,
    ligand_selection: Any | None = None,
    prolif_module: Any | None = None,
    occupancy_threshold: float = PROLIF_OCCUPANCY_THRESHOLD,
) -> tuple[dict[str, str], dict[str, str]]:
    paths = _prolif_artifact_paths(prolif_path)
    artifacts: dict[str, str] = {}
    errors: dict[str, str] = {}
    writers = {
        "timeseries_csv_gz": lambda p: _write_prolif_timeseries_csv(df, p),
        "barcode_png": lambda p: _write_prolif_barcode_plot(df, p),
        "occupancy_png": lambda p: _write_prolif_occupancy_plot(interactions, p),
        "lignetwork_html": lambda p: _write_prolif_lignetwork_html(
            fingerprint=fingerprint,
            ligand_selection=ligand_selection,
            prolif_module=prolif_module,
            path=p,
            threshold=occupancy_threshold,
        ),
        "interaction_diagram_png": lambda p: _write_prolif_interaction_diagram(
            interactions,
            p,
            ligand_label=ligand_label,
        ),
    }
    for key, writer in writers.items():
        path = paths[key]
        try:
            writer(path)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[equil_check:{}] Could not write ProLIF artifact {}: {}",
                ligand_label,
                path.name,
                exc,
            )
            continue
        if path.exists():
            artifacts[key] = path.name
    return artifacts, errors


def _run_prolif_fingerprint(
    fingerprint: Any,
    trajectory: Any,
    ligand: Any,
    protein: Any,
) -> None:
    kwargs = {}
    try:
        parameters = inspect.signature(fingerprint.run).parameters
    except Exception:
        parameters = {}
    if "progress" in parameters or any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    ):
        kwargs["progress"] = False
    try:
        fingerprint.run(trajectory, ligand, protein, **kwargs)
    except TypeError:
        if kwargs:
            fingerprint.run(trajectory, ligand, protein)
            return
        raise


def _prolif_interaction_allowed_for_candidates(interaction: Any) -> bool:
    return str(interaction).strip().lower() not in PROLIF_CANDIDATE_EXCLUDED_INTERACTIONS


def _normalized_prolif_interaction_name(interaction: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(interaction).strip().lower())


def _prolif_interaction_priority(interaction: Any) -> int:
    name = _normalized_prolif_interaction_name(interaction)
    if name in PROLIF_SALT_BRIDGE_INTERACTIONS:
        return 0
    if name in PROLIF_HBOND_INTERACTIONS:
        return 1
    if _prolif_interaction_allowed_for_candidates(interaction):
        return 2
    return 3


def _records_from_prolif_dataframe(
    df: pd.DataFrame,
    *,
    occupancy_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if df is None or df.empty:
        return [], []

    bool_df = _bool_prolif_dataframe(df)
    occupancy = bool_df.mean(axis=0)
    records: list[dict[str, Any]] = []
    persistent_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    n_frames = int(len(bool_df.index))

    for column, value in occupancy.items():
        parsed = _prolif_column_parts(column)
        if parsed is None:
            continue
        ligand_id, protein_id, interaction = parsed
        ligand_meta = _prolif_residue_metadata(ligand_id)
        protein_meta = _prolif_residue_metadata(protein_id)
        occ = float(value)
        active_frames = int(bool_df[column].sum())
        record = {
            "ligand": ligand_meta,
            "protein": protein_meta,
            "interaction": str(interaction),
            "occupancy": occ,
            "active_frames": active_frames,
            "n_frames": n_frames,
        }
        records.append(record)
        resid = protein_meta.get("resid")
        if (
            occ < float(occupancy_threshold)
            or resid is None
            or not _prolif_interaction_allowed_for_candidates(interaction)
        ):
            continue
        key = (
            int(resid),
            str(protein_meta.get("resname") or ""),
            str(protein_meta.get("chainID") or ""),
        )
        entry = persistent_by_key.setdefault(
            key,
            {
                "resid": int(resid),
                "resname": protein_meta.get("resname") or "",
                "chainID": protein_meta.get("chainID") or "",
                "max_occupancy": 0.0,
                "interactions": [],
            },
        )
        entry["max_occupancy"] = max(float(entry["max_occupancy"]), occ)
        entry["interactions"].append(
            {
                "interaction": str(interaction),
                "occupancy": occ,
                "active_frames": active_frames,
                "ligand": ligand_meta,
            }
        )

    records.sort(
        key=lambda item: (
            -float(item["occupancy"]),
            str(item["protein"].get("label") or ""),
            str(item["interaction"]),
        )
    )
    persistent = sorted(
        persistent_by_key.values(),
        key=lambda item: (
            -float(item["max_occupancy"]),
            int(item["resid"]),
            str(item.get("resname") or ""),
        ),
    )
    return records, persistent


def _persistent_prolif_residue_ids(prolif_record: dict[str, Any] | None) -> list[int]:
    if not isinstance(prolif_record, dict) or not prolif_record.get("usable", False):
        return []
    out: list[int] = []
    for item in prolif_record.get("persistent_protein_residues") or []:
        if not isinstance(item, dict):
            continue
        resid = _int_or_none(item.get("resid"))
        if resid is not None and resid not in out:
            out.append(resid)
    return out


def _persistent_prolif_residue_priorities(
    prolif_record: dict[str, Any] | None,
) -> dict[int, int]:
    if not isinstance(prolif_record, dict) or not prolif_record.get("usable", False):
        return {}
    priorities: dict[int, int] = {}
    for item in prolif_record.get("persistent_protein_residues") or []:
        if not isinstance(item, dict):
            continue
        resid = _int_or_none(item.get("resid"))
        if resid is None:
            continue
        best_priority = priorities.get(int(resid), 99)
        for interaction in item.get("interactions") or []:
            if not isinstance(interaction, dict):
                continue
            best_priority = min(
                best_priority,
                _prolif_interaction_priority(interaction.get("interaction")),
            )
        if best_priority != 99:
            priorities[int(resid)] = best_priority
    return priorities


def _persistent_prolif_salt_bridge_residues(
    prolif_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(prolif_record, dict) or not prolif_record.get("usable", False):
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for item in prolif_record.get("persistent_protein_residues") or []:
        if not isinstance(item, dict):
            continue
        resid = _int_or_none(item.get("resid"))
        if resid is None:
            continue
        has_salt_bridge = False
        max_occupancy = 0.0
        for interaction in item.get("interactions") or []:
            if not isinstance(interaction, dict):
                continue
            if _prolif_interaction_priority(interaction.get("interaction")) != 0:
                continue
            has_salt_bridge = True
            try:
                max_occupancy = max(max_occupancy, float(interaction.get("occupancy", 0.0)))
            except Exception:
                pass
        if not has_salt_bridge:
            continue
        key = (
            int(resid),
            str(item.get("resname") or ""),
            str(item.get("chainID") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "resid": int(resid),
                "resname": key[1],
                "chainID": key[2],
                "max_occupancy": max_occupancy,
            }
        )
    out.sort(key=lambda item: (-float(item["max_occupancy"]), int(item["resid"])))
    return out


def _protein_salt_bridge_atom_charge(atom: Any) -> int:
    key = (str(atom.resname).upper(), str(atom.name).upper())
    if key in PROTEIN_POSITIVE_ATOMS:
        return 1
    if key in PROTEIN_NEGATIVE_ATOMS:
        return -1
    return 0


def _write_prolif_interactions(
    *,
    prolif_path: Path,
    universe: mda.Universe,
    ligand_label: str | None,
    residue_name: str | None,
    tail_fraction: float,
    mode: str,
    occupancy_threshold: float = PROLIF_OCCUPANCY_THRESHOLD,
) -> dict[str, Any]:
    try:
        import prolif as plf

        if not residue_name:
            raise ValueError("No ligand residue name is available for ProLIF.")
        ligand = universe.select_atoms(f"resname {residue_name}")
        protein = universe.select_atoms("protein")
        if ligand.n_atoms == 0:
            raise ValueError(f"No ligand atoms found for resname {residue_name!r}.")
        if protein.n_atoms == 0:
            raise ValueError("No protein atoms found for ProLIF analysis.")

        n_frames_total = len(universe.trajectory)
        if n_frames_total == 0:
            raise ValueError("No trajectory frames available for ProLIF analysis.")
        start_frame = _trailing_analysis_start_frame(n_frames_total, tail_fraction)
        fp = plf.Fingerprint()
        _run_prolif_fingerprint(
            fp,
            universe.trajectory[start_frame:n_frames_total],
            ligand,
            protein,
        )
        df = fp.to_dataframe()
        interactions, persistent = _records_from_prolif_dataframe(
            df,
            occupancy_threshold=occupancy_threshold,
        )
        artifacts, artifact_errors = _write_prolif_artifacts(
            prolif_path=prolif_path,
            df=df,
            interactions=interactions,
            ligand_label=ligand_label or residue_name,
            fingerprint=fp,
            ligand_selection=ligand,
            prolif_module=plf,
            occupancy_threshold=occupancy_threshold,
        )
        record = {
            "schema_version": PROLIF_INTERACTIONS_SCHEMA_VERSION,
            "source": "equil_analysis",
            "mode": mode,
            "usable": True,
            "prolif_version": str(getattr(plf, "__version__", "")),
            "ligand": ligand_label or residue_name,
            "residue_name": residue_name,
            "tail_fraction": float(tail_fraction),
            "analysis_start_frame": int(start_frame),
            "n_frames": int(max(0, n_frames_total - start_frame)),
            "occupancy_threshold": float(occupancy_threshold),
            "candidate_interaction_filter": {
                "excluded_interactions": sorted(PROLIF_CANDIDATE_EXCLUDED_INTERACTIONS),
            },
            "persistent_protein_residues": persistent,
            "interactions": interactions,
            "artifacts": artifacts,
            "artifact_errors": artifact_errors,
        }
    except Exception as exc:
        record = {
            "schema_version": PROLIF_INTERACTIONS_SCHEMA_VERSION,
            "source": "equil_analysis",
            "mode": mode,
            "usable": False,
            "ligand": ligand_label,
            "residue_name": residue_name,
            "tail_fraction": float(tail_fraction),
            "occupancy_threshold": float(occupancy_threshold),
            "reason": f"{type(exc).__name__}: {exc}",
            "candidate_interaction_filter": {
                "excluded_interactions": sorted(PROLIF_CANDIDATE_EXCLUDED_INTERACTIONS),
            },
            "persistent_protein_residues": [],
            "interactions": [],
            "artifacts": {},
            "artifact_errors": {},
        }
        logger.warning("[equil_check:{}] ProLIF analysis unavailable: {}", ligand_label, exc)

    prolif_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _write_unusable_prolif_interactions(
    *,
    prolif_path: Path,
    ligand_label: str | None,
    residue_name: str | None,
    tail_fraction: float,
    mode: str,
    reason: Exception,
) -> dict[str, Any]:
    record = {
        "schema_version": PROLIF_INTERACTIONS_SCHEMA_VERSION,
        "source": "equil_analysis",
        "mode": mode,
        "usable": False,
        "ligand": ligand_label,
        "residue_name": residue_name,
        "tail_fraction": float(tail_fraction),
        "occupancy_threshold": float(PROLIF_OCCUPANCY_THRESHOLD),
        "reason": f"{type(reason).__name__}: {reason}",
        "candidate_interaction_filter": {
            "excluded_interactions": sorted(PROLIF_CANDIDATE_EXCLUDED_INTERACTIONS),
        },
        "persistent_protein_residues": [],
        "interactions": [],
        "artifacts": {},
        "artifact_errors": {},
    }
    prolif_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _sort_md_paths(paths: List[Path]) -> List[Path]:
    """Sort md-* files by their integer index (md-01, md01, etc.)."""

    def _idx(p: Path) -> int:
        stem = p.stem  # md-01 or md01
        for token in stem.split("-"):
            if token.isdigit():
                return int(token)
        try:
            return int("".join(filter(str.isdigit, stem)))
        except Exception:
            return -1

    return sorted(paths, key=_idx)


def _cpptraj_export_rep(
    rep_idx: int, prmtop: str, trajs: List[Path], workdir: Path
) -> None:
    """Export a representative frame to PDB/RST7 using cpptraj."""
    if not trajs:
        raise FileNotFoundError(
            "No md-*.nc trajectories found for equilibration analysis."
        )

    lines: List[str] = [f"parm {prmtop}"]
    for t in trajs:
        rel = t.name  # use local names; workdir is traj location
        lines.append(f"trajin {rel}")
    # cpptraj is 1-indexed for frames
    one_based_frame = rep_idx + 1
    lines.append(f"trajout representative.pdb pdb onlyframes {one_based_frame}")
    lines.append(f"trajout representative.rst7 restart onlyframes {one_based_frame}")

    script = "\n".join(lines) + "\n"
    (workdir / "rep.in").write_text(script)

    run_with_log(f"{cpptraj} -i rep.in", working_dir=workdir)


def _ligand_candidate_atom_names(
    *,
    system_root: Path,
    residue_name: str | None,
    ligand_label: str | None,
    universe: mda.Universe,
) -> list[str] | None:
    if not residue_name:
        return None
    sdf_file = system_root / "params" / f"{residue_name}.sdf"
    if not sdf_file.exists():
        return None
    lig_atoms = universe.select_atoms(f"resname {residue_name}")
    if lig_atoms.n_atoms == 0:
        return None
    try:
        from batter._internal.ops.build_complex import (
            _candidate_ligand_atom_name_string,
        )

        names = _candidate_ligand_atom_name_string(
            sdf_file,
            lig_atoms,
            ligand_label=ligand_label or residue_name,
            stage="equil-analysis",
        )
    except Exception as exc:
        logger.warning(
            "[equil_check:{}] Could not derive ligand anchor candidate names from {}: {}. "
            "Using all ligand heavy atoms for stable-distance search.",
            ligand_label,
            sdf_file,
            exc,
        )
        return None
    return [name for name in names.split() if name]


def _salt_bridge_ligand_atom_preference(
    *,
    system_root: Path,
    residue_name: str | None,
    ligand_label: str | None,
    universe: mda.Universe,
    tail_fraction: float,
    prolif_record: dict[str, Any] | None,
) -> dict[str, Any]:
    salt_bridge_residues = _persistent_prolif_salt_bridge_residues(prolif_record)
    empty = {
        "ligand_atom_names": [],
        "protein_residue_ids": [
            int(item["resid"]) for item in salt_bridge_residues
        ],
        "distance_cutoff": float(SALT_BRIDGE_DISTANCE_CUTOFF),
        "pairs": [],
    }
    if not salt_bridge_residues:
        return empty
    if not residue_name:
        return empty
    sdf_file = system_root / "params" / f"{residue_name}.sdf"
    if not sdf_file.exists():
        return empty
    lig_atoms = universe.select_atoms(f"resname {residue_name}")
    if lig_atoms.n_atoms == 0:
        return empty
    try:
        from batter._internal.ops.build_complex import (
            _sdf_formal_charge_by_ligand_atom_name,
        )

        ligand_charges = _sdf_formal_charge_by_ligand_atom_name(
            sdf_file,
            lig_atoms,
        )
    except Exception as exc:
        logger.debug(
            "[equil_check:{}] Could not derive ligand formal charges from {}: {}.",
            ligand_label,
            sdf_file,
            exc,
        )
        return empty
    if not ligand_charges:
        return empty

    ligand_atoms_by_name = {
        str(atom.name).strip(): atom
        for atom in lig_atoms
        if str(atom.name).strip() in ligand_charges
    }
    if not ligand_atoms_by_name:
        return empty

    protein_atoms = []
    for residue in salt_bridge_residues:
        resid = int(residue["resid"])
        atoms = universe.select_atoms(f"protein and resid {resid}")
        if atoms.n_atoms == 0:
            atoms = universe.select_atoms(f"resid {resid} and not resname {residue_name}")
        for atom in atoms:
            charge = _protein_salt_bridge_atom_charge(atom)
            if charge:
                protein_atoms.append((atom, charge))
    if not protein_atoms:
        return empty

    n_frames = len(universe.trajectory)
    if n_frames == 0:
        return empty
    start_frame = _trailing_analysis_start_frame(n_frames, tail_fraction)
    pair_distances: dict[tuple[int, str], list[float]] = {}
    protein_by_index: dict[int, Any] = {}
    for _ts in universe.trajectory[start_frame:n_frames]:
        for protein_atom, protein_charge in protein_atoms:
            protein_by_index[int(protein_atom.index)] = protein_atom
            for ligand_name, ligand_atom in ligand_atoms_by_name.items():
                ligand_charge = int(ligand_charges.get(ligand_name, 0))
                if ligand_charge == 0 or protein_charge * ligand_charge >= 0:
                    continue
                distance = float(
                    np.linalg.norm(
                        np.asarray(protein_atom.position, dtype=float)
                        - np.asarray(ligand_atom.position, dtype=float)
                    )
                )
                pair_distances.setdefault(
                    (int(protein_atom.index), ligand_name),
                    [],
                ).append(distance)

    pair_records: list[dict[str, Any]] = []
    for (protein_index, ligand_name), distances in pair_distances.items():
        if not distances:
            continue
        values = np.asarray(distances, dtype=float)
        contact_fraction = float(np.mean(values <= SALT_BRIDGE_DISTANCE_CUTOFF))
        if contact_fraction <= 0.0:
            continue
        protein_atom = protein_by_index[protein_index]
        ligand_atom = ligand_atoms_by_name[ligand_name]
        pair_records.append(
            {
                "protein": {
                    "index": int(protein_atom.index),
                    "resid": int(protein_atom.resid),
                    "resname": str(protein_atom.resname),
                    "name": str(protein_atom.name),
                    "mask": f":{int(protein_atom.resid)}@{protein_atom.name}",
                    "charge": int(_protein_salt_bridge_atom_charge(protein_atom)),
                },
                "ligand": {
                    "index": int(ligand_atom.index),
                    "resid": int(ligand_atom.resid),
                    "resname": str(ligand_atom.resname),
                    "name": str(ligand_name),
                    "mask": f":{int(ligand_atom.resid)}@{ligand_name}",
                    "charge": int(ligand_charges[ligand_name]),
                },
                "distance": {
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "contact_fraction": contact_fraction,
                },
            }
        )

    pair_records.sort(
        key=lambda item: (
            -float(item["distance"]["contact_fraction"]),
            float(item["distance"]["mean"]),
            int(item["protein"]["resid"]),
            str(item["protein"]["name"]),
            str(item["ligand"]["name"]),
        )
    )
    ligand_names = _dedupe_preserve_order(
        str(record["ligand"]["name"]) for record in pair_records
    )
    if ligand_names:
        logger.debug(
            "[equil_check:{}] Salt-bridge ligand atom preference from ProLIF: {}",
            ligand_label,
            " ".join(ligand_names),
        )
    return {
        **empty,
        "ligand_atom_names": ligand_names,
        "pairs": pair_records,
    }


def _stable_distance_validator(
    *,
    universe: mda.Universe,
    residue_name: str | None,
    directory: Path,
    protein_anchor_masks: list[str],
) -> SimValidator:
    validator = SimValidator.__new__(SimValidator)
    validator.universe = universe
    validator.workdir = directory.resolve()
    validator.ligand = residue_name
    validator.protein_anchor_masks = protein_anchor_masks
    validator.results = {}
    return validator


def _write_stable_boresch_distance(
    *,
    stable_path: Path,
    system_root: Path,
    sim: Any,
    sim_val: SimValidator,
    ligand_label: str | None,
    residue_name: str | None,
    universe: mda.Universe,
    tail_fraction: float,
    mode: str,
    prolif_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ligand_candidate_names = _ligand_candidate_atom_names(
        system_root=system_root,
        residue_name=residue_name,
        ligand_label=ligand_label,
        universe=universe,
    )
    salt_bridge_preference = _salt_bridge_ligand_atom_preference(
        system_root=system_root,
        residue_name=residue_name,
        ligand_label=ligand_label,
        universe=universe,
        tail_fraction=tail_fraction,
        prolif_record=prolif_record,
    )
    salt_bridge_ligand_names = list(
        salt_bridge_preference.get("ligand_atom_names") or []
    )
    salt_bridge_ligand_priorities = {
        name: idx for idx, name in enumerate(salt_bridge_ligand_names)
    }
    persistent_residue_ids = _persistent_prolif_residue_ids(prolif_record)
    persistent_residue_priorities = _persistent_prolif_residue_priorities(prolif_record)
    min_distance = float(getattr(sim, "min_adis", None) or 3.0)
    max_distance = float(getattr(sim, "max_adis", None) or 7.0)
    used_prolif_filter = False
    fallback_reason = None
    if persistent_residue_ids:
        try:
            stable_record = sim_val.find_stable_boresch_distance(
                tail_fraction=tail_fraction,
                min_distance=min_distance,
                max_distance=max_distance,
                ligand_atom_names=ligand_candidate_names,
                ligand_atom_priorities=salt_bridge_ligand_priorities,
                protein_residue_ids=persistent_residue_ids,
                protein_residue_priorities=persistent_residue_priorities,
            )
            used_prolif_filter = True
        except Exception as exc:
            fallback_reason = str(exc)
            logger.debug(
                "[equil_check:{}] Persistent ProLIF residues did not yield a "
                "stable CA-ligand Boresch distance; falling back to all CA "
                "candidates: {}",
                ligand_label,
                exc,
            )
            stable_record = sim_val.find_stable_boresch_distance(
                tail_fraction=tail_fraction,
                min_distance=min_distance,
                max_distance=max_distance,
                ligand_atom_names=ligand_candidate_names,
                ligand_atom_priorities=salt_bridge_ligand_priorities,
                protein_residue_priorities=persistent_residue_priorities,
            )
    else:
        stable_record = sim_val.find_stable_boresch_distance(
            tail_fraction=tail_fraction,
            min_distance=min_distance,
            max_distance=max_distance,
            ligand_atom_names=ligand_candidate_names,
            ligand_atom_priorities=salt_bridge_ligand_priorities,
            protein_residue_priorities=persistent_residue_priorities,
        )
    stable_record["mode"] = mode
    stable_record["usable"] = True
    stable_record["prolif_preference"] = {
        "usable": bool(isinstance(prolif_record, dict) and prolif_record.get("usable")),
        "occupancy_threshold": (
            float(prolif_record.get("occupancy_threshold"))
            if isinstance(prolif_record, dict)
            and prolif_record.get("occupancy_threshold") is not None
            else None
        ),
        "persistent_residue_ids": [int(x) for x in persistent_residue_ids],
        "persistent_residue_priorities": [
            {"resid": int(resid), "priority": int(priority)}
            for resid, priority in sorted(persistent_residue_priorities.items())
        ],
        "used_residue_filter": used_prolif_filter,
        "fallback_reason": fallback_reason,
    }
    stable_record["salt_bridge_preference"] = salt_bridge_preference
    stable_path.write_text(json.dumps(stable_record, indent=2) + "\n")
    logger.debug(
        "[equil_check:{}] stable Boresch pair: {} to {} "
        "(mean={:.2f} Å, std={:.2f} Å, frames={} from frame {}, "
        "ranked_pairs={}, mode={}).",
        ligand_label,
        stable_record["protein"]["mask"],
        stable_record["ligand"]["mask"],
        stable_record["distance"]["mean"],
        stable_record["distance"]["std"],
        stable_record["n_frames"],
        stable_record["analysis_start_frame"],
        len(stable_record.get("ranked_pairs") or []),
        mode,
    )
    return stable_record


def _write_unusable_stable_boresch_distance(
    *,
    stable_path: Path,
    mode: str,
    reason: Exception,
) -> None:
    stable_record = {
        "schema_version": STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
        "source": "equil_analysis",
        "mode": mode,
        "usable": False,
        "reason": str(reason),
    }
    stable_path.write_text(json.dumps(stable_record, indent=2) + "\n")


_EQUIL_ANALYSIS_ARTIFACT_FILES = (
    "representative.pdb",
    "representative.rst7",
    "representative_complex.pdb",
    "representative_pose.pdb",
    "initial_pose.pdb",
    "equilibration_analysis_results.npz",
    "stable_boresch_distance.json",
    "prolif_interactions.json",
    "prolif_interactions_timeseries.csv.gz",
    "prolif_interactions_barcode.png",
    "prolif_interactions_occupancy.png",
    "prolif_lignetwork.html",
    "prolif_interaction_diagram.png",
    "simulation_analysis.png",
    "dihed_hist.png",
)


def _copy_equil_analysis_artifacts(equil_dir: Path) -> None:
    artifacts_dir = equil_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in _EQUIL_ANALYSIS_ARTIFACT_FILES:
        src = equil_dir / name
        if src.exists():
            shutil.copy2(src, artifacts_dir / name)


def _add_existing_prolif_artifacts(
    artifacts: dict[str, Path],
    paths: dict[str, Path],
) -> None:
    keys = (
        "prolif_interactions",
        "prolif_timeseries",
        "prolif_barcode",
        "prolif_occupancy",
        "prolif_lignetwork",
        "prolif_interaction_diagram",
    )
    for key in keys:
        path = paths[key]
        if path.exists():
            artifacts[key] = path


def _maybe_cleanup_equil(payload: StepPayload, paths: dict[str, Path]) -> None:
    if bool(payload.get("store_debug_files", False)):
        return
    cleanup_equil_after_analysis(paths["equil_dir"])


def equil_analysis_handler(
    step: Step, system: SimSystem, params: Dict[str, Any]
) -> ExecResult:
    """Inspect equilibration trajectories and generate representative files.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system providing context and filesystem roots.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Artifacts describing the binding state (representative structures or
        ``UNBOUND`` sentinel).

    Raises
    ------
    FileNotFoundError
        When required inputs are missing.
    ValueError
        When the payload lacks a simulation configuration.
    """
    p = _paths(system.root)
    lig = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    logger.debug(
        f"Running equil_analysis_handler for ligand {lig} (residue {residue_name})"
    )

    rep_rel = p["rep_pdb"].relative_to(system.root).as_posix()
    unbound_rel = p["unbound"].relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "equil_analysis",
        required=[[rep_rel], [unbound_rel]],
        success=[[rep_rel], [unbound_rel]],
    )

    payload = StepPayload.model_validate(params)
    sim = payload.sim
    if sim is None:
        raise ValueError(
            "[equil_analysis] Missing simulation configuration in payload."
        )
    sys_params = payload.sys_params
    user_anchor_atoms = list(
        (sys_params.get("anchor_atoms", []) if sys_params is not None else []) or []
    )
    threshold = float(
        payload.get("unbound_threshold", getattr(sim, "unbound_threshold", 8.0))
    )
    hmr = str(sim.hmr)
    prmtop = "full.hmr.prmtop" if hmr == "yes" else "full.prmtop"

    # hard requirements
    if not p["finished"].exists():
        if p["failed"].exists():
            raise FileNotFoundError(f"[equil_check:{lig}] equil FAILED; cannot proceed")
        raise FileNotFoundError(f"[equil_check:{lig}] equil not FINISHED")

    if p["unbound"].exists():
        logger.warning(f"[equil_check:{lig}] previously marked UNBOUND — keeping as is")
        _maybe_cleanup_equil(payload, p)
        return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

    # if representative already exists, we're done (idempotent). For auto-anchor
    # runs, still allow a later invocation to backfill the stable-distance JSON.
    # Always allow a later invocation to backfill ProLIF interaction analysis.
    stable_distance_needed = not user_anchor_atoms
    prolif_needed = not _prolif_interactions_current(p["prolif_interactions"])
    if (
        stable_distance_needed
        and p["stable_boresch_distance"].exists()
        and not _stable_boresch_distance_current(p["stable_boresch_distance"])
    ):
        logger.debug(
            "[equil_check:{}] stable Boresch distance JSON is stale; "
            "removing it so the current selector can regenerate it.",
            lig,
        )
        try:
            p["stable_boresch_distance"].unlink()
        except OSError as exc:
            logger.warning(
                "[equil_check:{}] Could not remove stale stable Boresch distance "
                "JSON {}: {}",
                lig,
                p["stable_boresch_distance"],
                exc,
            )
    if p["prolif_interactions"].exists() and not _prolif_interactions_current(
        p["prolif_interactions"]
    ):
        logger.debug(
            "[equil_check:{}] ProLIF interaction JSON is stale; removing it.",
            lig,
        )
        try:
            p["prolif_interactions"].unlink()
        except OSError as exc:
            logger.warning(
                "[equil_check:{}] Could not remove stale ProLIF interaction "
                "JSON {}: {}",
                lig,
                p["prolif_interactions"],
                exc,
            )
    if (
        p["rep_pdb"].exists()
        and p["rep_rst"].exists()
        and not prolif_needed
        and (
            not stable_distance_needed
            or _stable_boresch_distance_current(p["stable_boresch_distance"])
        )
    ):
        logger.debug(
            f"[equil_check:{lig}] representative.* already present; skipping analysis"
        )
        artifacts = {
            "representative_pdb": p["rep_pdb"],
            "representative_rst7": p["rep_rst"],
        }
        if p["stable_boresch_distance"].exists():
            artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
        _add_existing_prolif_artifacts(artifacts, p)
        _maybe_cleanup_equil(payload, p)
        return ExecResult(job_ids=[], artifacts=artifacts)

    if not p["full_pdb"].exists():
        if p["rep_pdb"].exists() and p["rep_rst"].exists():
            logger.warning(
                f"[equil_check:{lig}] missing {p['full_pdb']}; cannot backfill "
                "stable Boresch distance, keeping existing representative.*"
            )
            if not _prolif_interactions_current(p["prolif_interactions"]):
                try:
                    u_prolif = mda.Universe(str(p["rep_pdb"]))
                    _write_prolif_interactions(
                        prolif_path=p["prolif_interactions"],
                        universe=u_prolif,
                        ligand_label=lig,
                        residue_name=residue_name,
                        tail_fraction=1.0,
                        mode="representative_only",
                    )
                except Exception as exc:
                    _write_unusable_prolif_interactions(
                        prolif_path=p["prolif_interactions"],
                        ligand_label=lig,
                        residue_name=residue_name,
                        tail_fraction=1.0,
                        mode="representative_only",
                        reason=exc,
                    )
            artifacts = {
                "representative_pdb": p["rep_pdb"],
                "representative_rst7": p["rep_rst"],
            }
            if p["stable_boresch_distance"].exists():
                artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
            _add_existing_prolif_artifacts(artifacts, p)
            _maybe_cleanup_equil(payload, p)
            return ExecResult(job_ids=[], artifacts=artifacts)
        raise FileNotFoundError(f"[equil_check:{lig}] missing {p['full_pdb']}")

    eq_steps = int(getattr(sim, "eq_steps", 0) or 0)
    if eq_steps == 0:
        eqnpt_appear = p["equil_dir"] / "eqnpt_appear.rst7"
        if not eqnpt_appear.exists():
            raise FileNotFoundError(
                f"[equil_check:{lig}] eq_steps=0 but missing {eqnpt_appear}"
            )
        shutil.copyfile(eqnpt_appear, p["rep_rst"])
        run_with_log(
            f"{cpptraj} -p {prmtop} -y representative.rst7 -x representative.pdb",
            working_dir=p["equil_dir"],
        )
        logger.debug(
            f"[equil_check:{lig}] eq_steps=0; copied {eqnpt_appear.name} as representative"
        )
        try:
            topology = p["equil_dir"] / prmtop
            if topology.exists() and p["rep_rst"].exists():
                u_prolif = mda.Universe(str(topology), str(p["rep_rst"]))
            else:
                u_prolif = mda.Universe(str(p["rep_pdb"]))
            prolif_record = _write_prolif_interactions(
                prolif_path=p["prolif_interactions"],
                universe=u_prolif,
                ligand_label=lig,
                residue_name=residue_name,
                tail_fraction=1.0,
                mode="single_frame_no_equil",
            )
        except Exception as exc:
            prolif_record = _write_unusable_prolif_interactions(
                prolif_path=p["prolif_interactions"],
                ligand_label=lig,
                residue_name=residue_name,
                tail_fraction=1.0,
                mode="single_frame_no_equil",
                reason=exc,
            )
        if user_anchor_atoms:
            logger.debug(
                "[equil_check:{}] explicit create.anchor_atoms were provided; "
                "skipping stable Boresch distance auto-anchor override.",
                lig,
            )
        else:
            try:
                u_static = mda.Universe(str(p["rep_pdb"]))
                anchor_masks = _load_equil_anchor_masks(p["equil_dir"])
                stable_val = _stable_distance_validator(
                    universe=u_static,
                    residue_name=residue_name,
                    directory=p["equil_dir"],
                    protein_anchor_masks=anchor_masks,
                )
                _write_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    system_root=system.root,
                    sim=sim,
                    sim_val=stable_val,
                    ligand_label=lig,
                    residue_name=residue_name,
                    universe=u_static,
                    tail_fraction=1.0,
                    mode="single_frame_no_equil",
                    prolif_record=prolif_record,
                )
            except Exception as exc:
                _write_unusable_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    mode="single_frame_no_equil",
                    reason=exc,
                )
                logger.warning(
                    "[equil_check:{}] Could not identify a single-frame "
                    "protein-ligand distance for automatic Boresch anchor "
                    "refinement: {}",
                    lig,
                    exc,
                )
        _copy_equil_analysis_artifacts(p["equil_dir"])
        # Skip trajectory-based validation/analysis when no equilibration steps ran.
        artifacts = {
            "representative_pdb": p["rep_pdb"],
            "representative_rst7": p["rep_rst"],
        }
        if p["stable_boresch_distance"].exists():
            artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
        _add_existing_prolif_artifacts(artifacts, p)
        _maybe_cleanup_equil(payload, p)
        return ExecResult(job_ids=[], artifacts=artifacts)

    # Run validation

    try:
        # Build trajectory list from completed equil segments
        trajs = _sort_md_paths(list(p["equil_dir"].glob("md-*.nc")))
        trajs = [t for t in trajs if t.exists()]
        # make sure each t is larger than 1 KB
        trajs = [t for t in trajs if t.stat().st_size > 1024]
        if not trajs:
            raise FileNotFoundError(
                f"[equil_check:{lig}] no md-*.nc trajectories found for analysis"
            )
        u = mda.Universe(str(p["full_pdb"]), [str(t) for t in trajs])
        anchor_masks = _equil_anchor_masks_to_original_resids(
            _load_equil_anchor_masks(p["equil_dir"]),
            p["prot_renum"],
        )
        sim_val = SimValidator(
            u,
            ligand=residue_name,
            directory=p["equil_dir"],
            protein_anchor_masks=anchor_masks,
        )
        sim_val.plot_analysis(savefig=True)

        # bound vs unbound
        ligand_bs_last = float(np.asarray(sim_val.results["ligand_bs"][-1]).item())
        if ligand_bs_last > threshold:
            logger.warning(
                f"[equil_check:{lig}] UNBOUND (ligand_bs={ligand_bs_last:.2f} Å) > {threshold:.2f} Å"
            )
            p["unbound"].write_text(f"UNBOUND with ligand_bs = {ligand_bs_last:.3f}\n")
            _maybe_cleanup_equil(payload, p)
            return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

        try:
            topology = p["equil_dir"] / prmtop
            u_prolif = (
                mda.Universe(str(topology), [str(t) for t in trajs])
                if topology.exists()
                else u
            )
            prolif_record = _write_prolif_interactions(
                prolif_path=p["prolif_interactions"],
                universe=u_prolif,
                ligand_label=lig,
                residue_name=residue_name,
                tail_fraction=0.25,
                mode="trajectory_tail",
            )
        except Exception as exc:
            prolif_record = _write_unusable_prolif_interactions(
                prolif_path=p["prolif_interactions"],
                ligand_label=lig,
                residue_name=residue_name,
                tail_fraction=0.25,
                mode="trajectory_tail",
                reason=exc,
            )

        if user_anchor_atoms:
            logger.debug(
                "[equil_check:{}] explicit create.anchor_atoms were provided; "
                "skipping stable Boresch distance auto-anchor override.",
                lig,
            )
        else:
            try:
                _write_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    system_root=system.root,
                    sim=sim,
                    sim_val=sim_val,
                    ligand_label=lig,
                    residue_name=residue_name,
                    universe=u,
                    tail_fraction=0.25,
                    mode="trajectory_tail",
                    prolif_record=prolif_record,
                )
            except Exception as exc:
                _write_unusable_stable_boresch_distance(
                    stable_path=p["stable_boresch_distance"],
                    mode="trajectory_tail",
                    reason=exc,
                )
                logger.warning(
                    "[equil_check:{}] Could not identify a stable protein-ligand "
                    "distance for automatic Boresch anchor refinement: {}",
                    lig,
                    exc,
                )
        rep_idx = int(sim_val.find_representative_snapshot())
        # pick representative frame and export using cpptraj
        _cpptraj_export_rep(rep_idx, prmtop, trajs, p["equil_dir"])
        sim_val.dump_results()

    # if traj doesn't exist
    # use the last frame as representative
    except Exception as e:
        logger.debug(f"[equil_check:{lig}] error during simulation validation: {e}")
        if p["rep_pdb"].exists() and p["rep_rst"].exists():
            logger.warning(
                f"[equil_check:{lig}] keeping existing representative.* after "
                "validation/backfill failure"
            )
        else:
            # copy last frame as representative
            last_rst = p["equil_dir"] / "md-current.rst7"
            if os.path.exists(last_rst):
                shutil.copyfile(last_rst, p["rep_rst"])
            else:
                raise FileNotFoundError(
                    f"[equil_check:{lig}] no md-current.rst7 found for fallback representative"
                )
            # convert to pdb
            run_with_log(
                f"{cpptraj} -p {prmtop} -y representative.rst7 -x representative.pdb",
                working_dir=p["equil_dir"],
            )

    # remap protein residue IDs back to original (protein_renum.txt)
    renum_txt = p["prot_renum"]
    if not renum_txt.exists():
        raise FileNotFoundError(
            f"[equil_check:{lig}] missing {renum_txt}; cannot renumber residues"
        )
    else:
        renum = pd.read_csv(
            renum_txt,
            sep=r"\s+",
            header=None,
            names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
        )
        uu = mda.Universe(str(p["rep_pdb"]))
        _restore_protein_resids_from_renum(uu.atoms, renum)
        uu.atoms.write(str(p["rep_pdb"]))

    # align representative to initial complex and extract poses
    protein_align = (getattr(sim, "protein_align", None) or "name CA").strip()
    if protein_align and p["rep_pdb"].exists() and p["full_pdb"].exists():
        try:
            aligned_rep_output = p["equil_dir"] / "representative_complex.pdb"
            u_rep = mda.Universe(str(p["rep_pdb"]))
            u_ref = mda.Universe(str(p["full_pdb"]))
            _ = align.alignto(
                mobile=u_rep.atoms,
                reference=u_ref.atoms,
                select=f"({protein_align}) and name CA and not resname NMA ACE",
            )
            u_rep.atoms.write(aligned_rep_output)
            if residue_name:
                u_ref.select_atoms(f"resname {residue_name}").write(
                    p["equil_dir"] / "initial_pose.pdb"
                )
                u_rep.select_atoms(f"resname {residue_name}").write(
                    p["equil_dir"] / "representative_pose.pdb"
                )
        except Exception as exc:
            logger.warning(
                f"[equil_check:{lig}] Failed to align representative complex: {exc}"
            )

    # copy key outputs into equil/artifacts for downstream use
    _copy_equil_analysis_artifacts(p["equil_dir"])

    logger.debug(f"[equil_check:{lig}] representative frame written")
    assert p["rep_pdb"].exists() and p["rep_rst"].exists()
    artifacts = {
        "representative_pdb": p["rep_pdb"],
        "representative_rst7": p["rep_rst"],
    }
    if p["stable_boresch_distance"].exists():
        artifacts["stable_boresch_distance"] = p["stable_boresch_distance"]
    _add_existing_prolif_artifacts(artifacts, p)
    _maybe_cleanup_equil(payload, p)
    return ExecResult(job_ids=[], artifacts=artifacts)
