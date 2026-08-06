"""Analyse equilibration trajectories to determine binding status."""

from __future__ import annotations

import html
import inspect
import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from batter._internal.ops.cleanup import cleanup_equil_after_analysis
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils import cpptraj, run_with_log

if TYPE_CHECKING:
    import MDAnalysis as mda
    from batter.analysis.sim_validation import SimValidator

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


def _mda():
    import MDAnalysis as mda

    return mda


def _mda_align():
    from MDAnalysis.analysis import align

    return align


def _load_no_equil_representative_universe(rep_pdb: Path):
    """Load the cpptraj-written PDB snapshot for eq_steps=0 analyses."""
    if not rep_pdb.exists():
        raise FileNotFoundError(
            f"Missing representative PDB for no-equil analysis: {rep_pdb}"
        )
    return _mda().Universe(str(rep_pdb))


def _sim_validator_cls():
    from batter.analysis.sim_validation import SimValidator

    return SimValidator


def _stable_boresch_distance_schema_version() -> int:
    from batter.analysis.sim_validation import STABLE_BORESCH_DISTANCE_SCHEMA_VERSION

    return STABLE_BORESCH_DISTANCE_SCHEMA_VERSION


def _restore_protein_resids_from_renum_fn():
    from batter._internal.ops.box import _restore_protein_resids_from_renum

    return _restore_protein_resids_from_renum


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
        "simulation_analysis": eq / "simulation_analysis.png",
        "build_files": eq / "q_build_files",
        "prot_renum": prot_renum,
        "full_pdb": eq / "full.pdb",
        "anchors_json": eq / "anchors.json",
    }


def _stable_boresch_distance_current(path: Path) -> bool:
    if not path.exists():
        return False
    sim_val = None
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
    return schema_version >= _stable_boresch_distance_schema_version()


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


def _npz_scalar_string(data: Any, key: str) -> str:
    if key not in data:
        return ""
    value = data[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return ""
    return str(value)


def _representative_selection_needs_refresh(equil_dir: Path) -> bool:
    """Return True for old last-frame fallbacks caused only by missing assign.in."""
    result_path = equil_dir / "equilibration_analysis_results.npz"
    if not result_path.exists() or not (equil_dir / "disang.rest").exists():
        return False
    try:
        with np.load(result_path, allow_pickle=True) as data:
            mode = _npz_scalar_string(data, "representative_selection_mode")
            reason = _npz_scalar_string(data, "representative_selection_reason")
    except Exception:
        return False
    return mode == "last_frame_fallback" and "assign.in" in reason


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

    if resid is not None:
        resid = int(resid)
    if resname and resid is not None:
        chain_suffix = ""
        if chain and chain not in {"0", "None", "none", "nan", "NaN"}:
            chain_suffix = f".{chain}"
        label = f"{resname}{resid}{chain_suffix}"

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
    ligand_label = _prolif_residue_metadata(ligand_id).get("label") or str(ligand_id)
    protein_label = _prolif_residue_metadata(protein_id).get("label") or str(protein_id)
    return "|".join(
        str(item).replace("|", "/")
        for item in (ligand_label, protein_label, interaction)
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
        _write_prolif_lignetwork_unavailable(
            path,
            "fingerprint object was not provided",
        )
        return

    try:
        ligand_mol = prolif_module.Molecule.from_mda(ligand_selection)
        view = fingerprint.plot_lignetwork(
            ligand_mol,
            kind="aggregate",
            threshold=float(threshold),
            height="650px",
            show_interaction_data=True,
        )
        view.save(path)
    except Exception as exc:
        logger.debug("ProLIF LigNetwork renderer unavailable: {}", exc)
        _write_prolif_lignetwork_unavailable(
            path,
            f"{type(exc).__name__}: {exc}",
        )


def _write_prolif_lignetwork_unavailable(path: Path, reason: str) -> None:
    safe_reason = html.escape(str(reason), quote=True)
    path.write_text(
        "<!doctype html><html><body><p>"
        f"ProLIF LigNetwork unavailable: {safe_reason}."
        "</p></body></html>\n"
    )


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
        "protein_residue_ids": [],
        "distance_cutoff": float(SALT_BRIDGE_DISTANCE_CUTOFF),
        "source": "prolif" if salt_bridge_residues else "charged_atom_distance",
        "used_prolif_residue_filter": bool(salt_bridge_residues),
        "pairs": [],
    }
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
    if salt_bridge_residues:
        residue_items: Sequence[Any] = salt_bridge_residues
    else:
        residue_items = [
            {"resid": int(residue.resid)}
            for residue in universe.select_atoms(
                f"protein and not resname {residue_name}"
            ).residues
        ]
    for residue in residue_items:
        resid = int(residue["resid"])
        atoms = universe.select_atoms(f"protein and resid {resid}")
        if atoms.n_atoms == 0:
            atoms = universe.select_atoms(
                f"resid {resid} and not resname {residue_name}"
            )
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
    protein_residue_ids = []
    seen_protein_resids: set[int] = set()
    for record in pair_records:
        resid = int(record["protein"]["resid"])
        if resid in seen_protein_resids:
            continue
        seen_protein_resids.add(resid)
        protein_residue_ids.append(resid)
    if ligand_names:
        logger.debug(
            "[equil_check:{}] Salt-bridge ligand atom preference from {}: {}",
            ligand_label,
            empty["source"],
            " ".join(ligand_names),
        )
    return {
        **empty,
        "ligand_atom_names": ligand_names,
        "protein_residue_ids": protein_residue_ids,
        "pairs": pair_records,
    }


def _stable_distance_validator(
    *,
    universe: mda.Universe,
    residue_name: str | None,
    directory: Path,
    protein_anchor_masks: list[str],
) -> SimValidator:
    sim_validator_cls = _sim_validator_cls()
    validator = sim_validator_cls.__new__(sim_validator_cls)
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
    preference_universe: mda.Universe | None = None,
    tail_fraction: float,
    mode: str,
    prolif_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preference_universe = preference_universe or universe
    ligand_candidate_names = _ligand_candidate_atom_names(
        system_root=system_root,
        residue_name=residue_name,
        ligand_label=ligand_label,
        universe=preference_universe,
    )
    salt_bridge_preference = _salt_bridge_ligand_atom_preference(
        system_root=system_root,
        residue_name=residue_name,
        ligand_label=ligand_label,
        universe=preference_universe,
        tail_fraction=tail_fraction,
        prolif_record=prolif_record,
    )
    salt_bridge_ligand_names = list(
        salt_bridge_preference.get("ligand_atom_names") or []
    )
    salt_bridge_ligand_priorities = {
        name: idx for idx, name in enumerate(salt_bridge_ligand_names)
    }
    salt_bridge_residue_ids = []
    for resid in salt_bridge_preference.get("protein_residue_ids") or []:
        value = _int_or_none(resid)
        if value is not None and value not in salt_bridge_residue_ids:
            salt_bridge_residue_ids.append(value)
    persistent_residue_ids = _persistent_prolif_residue_ids(prolif_record)
    persistent_residue_priorities = _persistent_prolif_residue_priorities(prolif_record)
    residue_filter_priorities = dict(persistent_residue_priorities)
    for idx, resid in enumerate(salt_bridge_residue_ids):
        residue_filter_priorities[int(resid)] = min(
            int(residue_filter_priorities.get(int(resid), 99)),
            idx,
        )
    min_distance = float(getattr(sim, "min_adis", None) or 3.0)
    max_distance = float(getattr(sim, "max_adis", None) or 7.0)
    used_prolif_filter = False
    used_salt_bridge_filter = False
    fallback_reason = None
    residue_filter_ids = persistent_residue_ids or salt_bridge_residue_ids
    if residue_filter_ids:
        residue_filter_source = (
            "ProLIF residues" if persistent_residue_ids else "salt-bridge geometry"
        )
        try:
            stable_record = sim_val.find_stable_boresch_distance(
                tail_fraction=tail_fraction,
                min_distance=min_distance,
                max_distance=max_distance,
                ligand_atom_names=ligand_candidate_names,
                ligand_atom_priorities=salt_bridge_ligand_priorities,
                protein_residue_ids=residue_filter_ids,
                protein_residue_priorities=residue_filter_priorities,
            )
            used_prolif_filter = bool(persistent_residue_ids)
            used_salt_bridge_filter = bool(
                salt_bridge_residue_ids and not persistent_residue_ids
            )
        except Exception as exc:
            fallback_reason = str(exc)
            logger.debug(
                "[equil_check:{}] {} did not yield a "
                "stable CA-ligand Boresch distance; falling back to all CA "
                "candidates: {}",
                ligand_label,
                residue_filter_source,
                exc,
            )
            stable_record = sim_val.find_stable_boresch_distance(
                tail_fraction=tail_fraction,
                min_distance=min_distance,
                max_distance=max_distance,
                ligand_atom_names=ligand_candidate_names,
                ligand_atom_priorities=salt_bridge_ligand_priorities,
                protein_residue_priorities=residue_filter_priorities,
            )
    else:
        stable_record = sim_val.find_stable_boresch_distance(
            tail_fraction=tail_fraction,
            min_distance=min_distance,
            max_distance=max_distance,
            ligand_atom_names=ligand_candidate_names,
            ligand_atom_priorities=salt_bridge_ligand_priorities,
            protein_residue_priorities=residue_filter_priorities,
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
        "salt_bridge_residue_ids": [int(x) for x in salt_bridge_residue_ids],
        "used_residue_filter": used_prolif_filter,
        "used_salt_bridge_residue_filter": used_salt_bridge_filter,
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
        "schema_version": _stable_boresch_distance_schema_version(),
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


def _write_equil_results_readme(results_dir: Path) -> None:
    (results_dir / "README.txt").write_text(
        "This directory contains equilibration analysis outputs for one BATTER simulation.\n\n"
        "Common files:\n"
        "- representative.pdb: representative equilibration snapshot used downstream.\n"
        "- representative.rst7: AMBER restart for the representative snapshot.\n"
        "- representative_complex.pdb: representative complex aligned to the initial complex.\n"
        "- representative_pose.pdb: ligand pose from the representative complex.\n"
        "- initial_pose.pdb: ligand pose from the initial complex.\n"
        "- equilibration_analysis_results.npz: NumPy archive of validation metrics.\n"
        "- simulation_analysis.png: ligand binding-site distance and RMSD over frame, with simulation time on the top axis when available.\n"
        "- dihed_hist.png: ligand dihedral distributions used to choose the representative snapshot.\n"
        "- stable_boresch_distance.json: automatically selected stable protein-ligand distance for Boresch anchor refinement.\n"
        "- prolif_interactions.json: raw ProLIF interaction summary and persistent interacting residues.\n"
        "- prolif_interactions_timeseries.csv.gz: per-frame ProLIF interaction barcode data.\n"
        "- prolif_interactions_barcode.png: per-frame ProLIF interaction barcode plot.\n"
        "- prolif_interactions_occupancy.png: ProLIF interaction occupancy plot.\n"
        "- prolif_interaction_diagram.png: residue-level ligand interaction diagram.\n"
        "- prolif_lignetwork.html: interactive ProLIF ligand interaction network when supported by ProLIF.\n",
    )


def _copy_equil_analysis_artifacts(equil_dir: Path) -> None:
    results_dir = equil_dir / "results"
    legacy_artifacts_dir = equil_dir / "artifacts"
    for dest_dir in (results_dir, legacy_artifacts_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in _EQUIL_ANALYSIS_ARTIFACT_FILES:
            src = equil_dir / name
            if src.exists():
                shutil.copy2(src, dest_dir / name)
        _write_equil_results_readme(dest_dir)


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


def discover_equil_analysis_targets(path: Path) -> list[Path]:
    """Return simulation roots accepted by the standalone simulation-analysis CLI."""
    root = Path(path).expanduser().resolve()
    targets: list[Path] = []

    def _add(candidate: Path) -> None:
        candidate = candidate.resolve()
        if (candidate / "equil").is_dir() and candidate not in targets:
            targets.append(candidate)

    if root.name == "equil" and root.is_dir():
        _add(root.parent)
    else:
        _add(root)
    simulation_dirs: list[Path] = []
    if (root / "simulations").is_dir():
        simulation_dirs.append(root / "simulations")
    if root.name == "simulations" and root.is_dir():
        simulation_dirs.append(root)
    for sim_dir in simulation_dirs:
        for equil_dir in sorted(sim_dir.glob("*/equil")):
            _add(equil_dir.parent)
        for equil_dir in sorted(sim_dir.glob("*/*/equil")):
            _add(equil_dir.parent)
    return sorted(targets)


_STANDALONE_REFRESH_FILES = (
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


def _read_standalone_run_config(system_root: Path) -> dict[str, Any]:
    for parent in (system_root, *system_root.parents):
        path = parent / "artifacts" / "config" / "run_config.normalized.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Could not read standalone run config {}: {}", path, exc)
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _standalone_config_sections(system_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _read_standalone_run_config(system_root)
    config = data.get("config") if isinstance(data.get("config"), dict) else data
    create = config.get("create") if isinstance(config.get("create"), dict) else {}
    fe_sim = config.get("fe_sim") if isinstance(config.get("fe_sim"), dict) else {}
    return create, fe_sim


def _infer_standalone_ligand_context(
    system_root: Path,
    *,
    residue_name: str | None,
    ligand_label: str | None,
) -> tuple[str | None, str]:
    if not ligand_label:
        for metadata_path in sorted((system_root / "params").glob("*.metadata.json")):
            try:
                data = json.loads(metadata_path.read_text())
            except Exception:
                continue
            title = str(data.get("title") or "").strip() if isinstance(data, dict) else ""
            if title:
                ligand_label = title
                break
    if not ligand_label:
        ligand_label = system_root.name

    if residue_name:
        return residue_name, ligand_label

    excluded_stems = {
        "full",
        "ligand",
        "vac_ligand",
        "anchors",
        "equil-reference",
        "extra_conf_restraints",
        "initial_pose",
        "ligand_dihedral_restraints",
        "ligand_dihedral_schedule",
        "metadata",
        "other_parts",
        "output",
        "representative",
        "representative_complex",
        "representative_pose",
        "stable_boresch_distance",
    }
    for search_dir, suffix in (
        (system_root / "params", "*.sdf"),
        (system_root / "equil", "*.sdf"),
        (system_root / "equil", "*.json"),
    ):
        if not search_dir.is_dir():
            continue
        for candidate in sorted(search_dir.glob(suffix)):
            stem = candidate.stem
            if stem in excluded_stems or stem.startswith("prolif_"):
                continue
            return stem, ligand_label
    return None, ligand_label


def _infer_standalone_hmr(
    equil_dir: Path,
    hmr: bool | str | None,
    fe_sim: dict[str, Any],
) -> str:
    value = hmr if hmr is not None else fe_sim.get("hmr")
    if value is None:
        return "yes" if (equil_dir / "full.hmr.prmtop").exists() else "no"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    return "yes" if text in {"1", "true", "yes", "y", "on"} else "no"


def _infer_standalone_eq_steps(equil_dir: Path, fe_sim: dict[str, Any]) -> int:
    value = fe_sim.get("eq_steps")
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    trajs = [path for path in equil_dir.glob("md-*.nc") if path.stat().st_size > 1024]
    if not trajs and (equil_dir / "eqnpt_appear.rst7").exists():
        return 0
    return 1_000_000


def _refresh_standalone_outputs(equil_dir: Path) -> None:
    for name in _STANDALONE_REFRESH_FILES:
        path = equil_dir / name
        if path.exists():
            path.unlink()


def _fallback_representative_frame_index(
    *,
    universe: mda.Universe,
    sim_val: SimValidator,
) -> int:
    frames = np.asarray(sim_val.results.get("frame_indices", []), dtype=int)
    if frames.size:
        return int(frames[-1])
    n_frames = len(universe.trajectory)
    if n_frames <= 0:
        return 0
    return int(n_frames - 1)


def run_equil_analysis_for_simulation(
    system_root: Path,
    *,
    residue_name: str | None = None,
    ligand_label: str | None = None,
    threshold: float | None = None,
    hmr: bool | str | None = None,
    force: bool = False,
) -> ExecResult:
    """Run the equil-analysis handler for one existing simulation directory."""
    from batter.config.simulation import SimulationConfig

    system_root = Path(system_root).expanduser().resolve()
    p = _paths(system_root)
    if not p["equil_dir"].is_dir():
        raise FileNotFoundError(f"Missing equil directory under {system_root}")
    if force:
        _refresh_standalone_outputs(p["equil_dir"])

    create, fe_sim = _standalone_config_sections(system_root)
    residue_name, ligand_label = _infer_standalone_ligand_context(
        system_root,
        residue_name=residue_name,
        ligand_label=ligand_label,
    )
    if threshold is None:
        threshold = float(fe_sim.get("unbound_threshold", 8.0) or 8.0)

    sim_data: dict[str, Any] = {
        "system_name": create.get("system_name") or system_root.name,
        "fe_type": fe_sim.get("fe_type") or "md",
        "hmr": _infer_standalone_hmr(p["equil_dir"], hmr, fe_sim),
        "eq_steps": _infer_standalone_eq_steps(p["equil_dir"], fe_sim),
        "unbound_threshold": float(threshold),
        "protein_align": create.get("protein_align")
        or fe_sim.get("protein_align")
        or "name CA",
    }
    for key in ("min_adis", "max_adis"):
        value = create.get(key, fe_sim.get(key))
        if value is not None:
            sim_data[key] = value

    sim = SimulationConfig.model_validate(sim_data)
    system = SimSystem(
        name=ligand_label or system_root.name,
        root=system_root,
        meta={"ligand": ligand_label, "residue_name": residue_name},
    )
    params = {
        "sim": sim,
        "sys_params": {"anchor_atoms": create.get("anchor_atoms") or []},
        "store_debug_files": True,
        "unbound_threshold": float(threshold),
    }
    return equil_analysis_handler(Step(name="equil_analysis"), system, params)


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
        _copy_equil_analysis_artifacts(p["equil_dir"])
        _maybe_cleanup_equil(payload, p)
        return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

    # if representative already exists, we're done (idempotent). For auto-anchor
    # runs, still allow a later invocation to backfill the stable-distance JSON.
    # Always allow a later invocation to backfill ProLIF interaction analysis.
    # Always backfill this record. FE preserves user-pinned receptor anchors, but
    # still reads salt_bridge_preference from this JSON to prioritize ligand L1.
    stable_distance_needed = True
    prolif_needed = not _prolif_interactions_current(p["prolif_interactions"])
    representative_refresh_needed = _representative_selection_needs_refresh(
        p["equil_dir"]
    )
    if representative_refresh_needed:
        logger.debug(
            "[equil_check:{}] representative.* came from an old missing-assign.in "
            "last-frame fallback; refreshing analysis from disang.rest.",
            lig,
        )
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
        and not representative_refresh_needed
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
        _copy_equil_analysis_artifacts(p["equil_dir"])
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
                    u_prolif = _mda().Universe(str(p["rep_pdb"]))
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
            _copy_equil_analysis_artifacts(p["equil_dir"])
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
            # eqnpt_appear.rst7 can be a NetCDF restart with a .rst7 suffix.
            # MDAnalysis cannot infer that reliably; cpptraj already wrote PDB.
            u_prolif = _load_no_equil_representative_universe(p["rep_pdb"])
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
        try:
            u_static = _load_no_equil_representative_universe(p["rep_pdb"])
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

    sim_val = None
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
        topology = p["equil_dir"] / prmtop
        analysis_topology = topology if topology.exists() else p["full_pdb"]
        u = _mda().Universe(str(analysis_topology), [str(t) for t in trajs])
        anchor_masks = _equil_anchor_masks_to_original_resids(
            _load_equil_anchor_masks(p["equil_dir"]),
            p["prot_renum"],
        )
        sim_val = _sim_validator_cls()(
            u,
            ligand=residue_name,
            directory=p["equil_dir"],
            protein_anchor_masks=anchor_masks,
        )
        # bound vs unbound
        ligand_bs_last = float(np.asarray(sim_val.results["ligand_bs"][-1]).item())
        if ligand_bs_last > threshold:
            logger.warning(
                f"[equil_check:{lig}] UNBOUND (ligand_bs={ligand_bs_last:.2f} Å) > {threshold:.2f} Å"
            )
            sim_val.plot_analysis(savefig=True)
            p["unbound"].write_text(f"UNBOUND with ligand_bs = {ligand_bs_last:.3f}\n")
            _copy_equil_analysis_artifacts(p["equil_dir"])
            _maybe_cleanup_equil(payload, p)
            return ExecResult(job_ids=[], artifacts={"unbound": p["unbound"]})

        stable_preference_universe = u
        try:
            u_prolif = u
            stable_preference_universe = u_prolif
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

        try:
            _write_stable_boresch_distance(
                stable_path=p["stable_boresch_distance"],
                system_root=system.root,
                sim=sim,
                sim_val=sim_val,
                ligand_label=lig,
                residue_name=residue_name,
                universe=u,
                preference_universe=stable_preference_universe,
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
        try:
            rep_idx = int(sim_val.find_representative_snapshot())
        except Exception as exc:
            rep_idx = _fallback_representative_frame_index(universe=u, sim_val=sim_val)
            sim_val.results["representative_frame_index"] = int(rep_idx)
            sim_val.results["representative_selection_mode"] = "last_frame_fallback"
            sim_val.results["representative_selection_reason"] = str(exc)
            logger.warning(
                "[equil_check:{}] Could not choose representative from ligand "
                "dihedrals; using last trajectory frame {}: {}",
                lig,
                rep_idx,
                exc,
            )
        sim_val.plot_analysis(savefig=True)
        sim_val.dump_results()
        # pick representative frame and export using cpptraj
        _cpptraj_export_rep(rep_idx, prmtop, trajs, p["equil_dir"])

    # if traj doesn't exist
    # use the last frame as representative
    except Exception as e:
        logger.debug(f"[equil_check:{lig}] error during simulation validation: {e}")
        if sim_val is not None and not p["simulation_analysis"].exists():
            try:
                sim_val.plot_analysis(savefig=True)
            except Exception as plot_exc:
                logger.debug(
                    "[equil_check:{}] Could not write fallback simulation_analysis.png: {}",
                    lig,
                    plot_exc,
                )
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
        uu = _mda().Universe(str(p["rep_pdb"]))
        _restore_protein_resids_from_renum_fn()(uu.atoms, renum)
        uu.atoms.write(str(p["rep_pdb"]))

    # align representative to initial complex and extract poses
    protein_align = (getattr(sim, "protein_align", None) or "name CA").strip()
    if protein_align and p["rep_pdb"].exists() and p["full_pdb"].exists():
        try:
            aligned_rep_output = p["equil_dir"] / "representative_complex.pdb"
            u_rep = _mda().Universe(str(p["rep_pdb"]))
            u_ref = _mda().Universe(str(p["full_pdb"]))
            _ = _mda_align().alignto(
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

    # copy key outputs into equil/results for user-facing inspection and into
    # equil/artifacts for compatibility with older downstream paths.
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
