"""State-function based free-energy correction for RBFE networks.

Acknowledgement
---------------
This module rewrites the matrix-based State-Function Based Free Energy
Correction (SFC) workflow from the MIT-licensed reference implementation for
BATTER's analysis API:
https://github.com/ZheLi-Lab/State-Function-based-free-energy-correction-SFC-

Reference
---------
Liu, R.; Lai, Y.; Yao, Y.; Huang, W.; Zhong, Y.; Luo, H.-B.; Li, Z.
State Function-Based Correction: A Simple and Efficient Free-Energy Correction
Algorithm for Large-Scale Relative Binding Free-Energy Calculations.
J. Phys. Chem. Lett. https://doi.org/10.1021/acs.jpclett.5c01119

The historical ``cycle_closure_*`` function names are kept for compatibility
with the existing BATTER Cinnabar integration. They now run SFC/WSFC rather
than the earlier cycle-enumeration WCC algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CycleClosureEdge",
    "CycleClosureResult",
    "StateFunctionCorrectionEdge",
    "StateFunctionCorrectionResult",
    "calculate_cycle_closure",
    "calculate_state_function_correction",
    "cycle_closure_from_dataframe",
    "cycle_closure_from_file",
    "read_cycle_closure_file",
    "read_state_function_correction_file",
    "state_function_correction_from_dataframe",
    "state_function_correction_from_file",
]

SFC_MIN_UNCERTAINTY = 1.0e-6


@dataclass(frozen=True)
class CycleClosureEdge:
    """One directed RBFE edge used as SFC input.

    Parameters
    ----------
    label_a, label_b
        Ligand labels defining the edge direction.
    ddg
        Relative free energy for ``label_a -> label_b``.
    uncertainties
        Optional standard-error columns. Each supplied column creates one WSFC
        estimate using uncertainty-derived weights.
    """

    label_a: str
    label_b: str
    ddg: float
    uncertainties: tuple[float, ...] = ()


@dataclass(frozen=True)
class CycleClosureResult:
    """SFC result tables and metadata."""

    reference: str
    reference_free_energy: float
    node_results: pd.DataFrame
    edge_results: pd.DataFrame
    cycles: tuple[tuple[str, ...], ...] = ()
    iterations: tuple[int, ...] = ()
    converged: tuple[bool, ...] = ()
    method: str = "sfc"
    schemes: tuple[str, ...] = ()


StateFunctionCorrectionEdge = CycleClosureEdge
StateFunctionCorrectionResult = CycleClosureResult


def _coerce_edges(
    edges: Iterable[CycleClosureEdge | Sequence[object]],
) -> tuple[CycleClosureEdge, ...]:
    coerced: list[CycleClosureEdge] = []
    for edge in edges:
        if isinstance(edge, CycleClosureEdge):
            candidate = edge
        else:
            if len(edge) < 3:
                raise ValueError("SFC edge sequences must contain at least 3 values.")
            label_a, label_b, ddg, *uncertainties = edge
            candidate = CycleClosureEdge(
                label_a=str(label_a),
                label_b=str(label_b),
                ddg=float(ddg),
                uncertainties=tuple(float(value) for value in uncertainties),
            )

        label_a = str(candidate.label_a).strip()
        label_b = str(candidate.label_b).strip()
        if not label_a or not label_b:
            raise ValueError("SFC edge labels cannot be empty.")
        if label_a == label_b:
            raise ValueError("SFC edges cannot connect a ligand to itself.")
        if not math.isfinite(float(candidate.ddg)):
            raise ValueError(f"Non-finite SFC ddG for {label_a}->{label_b}.")

        uncertainties = tuple(float(value) for value in candidate.uncertainties)
        if any((not math.isfinite(value)) or value < 0 for value in uncertainties):
            raise ValueError(
                f"Uncertainties for {label_a}->{label_b} must be finite and >= 0."
            )
        uncertainties = tuple(
            SFC_MIN_UNCERTAINTY if value == 0 else value
            for value in uncertainties
        )
        coerced.append(
            CycleClosureEdge(
                label_a=label_a,
                label_b=label_b,
                ddg=float(candidate.ddg),
                uncertainties=uncertainties,
            )
        )

    if not coerced:
        raise ValueError("SFC requires at least one edge.")

    uncertainty_counts = {len(edge.uncertainties) for edge in coerced}
    if len(uncertainty_counts) != 1:
        raise ValueError("All SFC edges must use the same uncertainty columns.")

    return tuple(coerced)


def _ordered_labels(
    edges: Sequence[CycleClosureEdge],
    reference: str | None,
) -> list[str]:
    labels: list[str] = []
    for edge in edges:
        for label in (edge.label_a, edge.label_b):
            if label not in labels:
                labels.append(label)

    if reference is None:
        return labels

    reference = str(reference).strip()
    if reference not in labels:
        raise ValueError(f"Reference ligand {reference!r} is not present in the graph.")
    return [reference, *[label for label in labels if label != reference]]


def _design_matrix(
    edges: Sequence[CycleClosureEdge],
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    n_edges = len(edges)
    n_labels = len(labels)
    n_uncertainty_cols = len(edges[0].uncertainties)

    a_matrix = np.zeros((n_edges, n_labels), dtype=float)
    b_vector = np.zeros(n_edges, dtype=float)
    uncertainty_matrix = np.zeros((n_edges, n_uncertainty_cols), dtype=float)

    for row_idx, edge in enumerate(edges):
        idx_a = label_index[edge.label_a]
        idx_b = label_index[edge.label_b]
        a_matrix[row_idx, idx_a] = -1.0
        a_matrix[row_idx, idx_b] = 1.0
        b_vector[row_idx] = edge.ddg
        for col_idx, uncertainty in enumerate(edge.uncertainties):
            uncertainty_matrix[row_idx, col_idx] = uncertainty

    return a_matrix, b_vector, uncertainty_matrix


def _validate_connected_system(a_matrix: np.ndarray, n_labels: int) -> None:
    rank = int(np.linalg.matrix_rank(a_matrix))
    if rank < n_labels - 1:
        raise ValueError(
            "SFC requires a connected RBFE graph; the design matrix is rank "
            f"deficient (rank={rank}, expected at least {n_labels - 1})."
        )


def _uncertainty_weights(uncertainties: np.ndarray) -> np.ndarray:
    total = float(np.sum(uncertainties))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("SFC uncertainty weights require positive uncertainties.")
    normalized = uncertainties / total
    return 1.0 / np.square(normalized)


def _solve_state_function(
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    *,
    weights: np.ndarray | None,
    reference_index: int,
    reference_free_energy: float,
    reference_weight: float,
) -> np.ndarray:
    n_labels = a_matrix.shape[1]
    ref_row = np.zeros((1, n_labels), dtype=float)
    ref_row[0, reference_index] = 1.0
    a_aug = np.vstack([a_matrix, ref_row])
    b_aug = np.concatenate([b_vector, [float(reference_free_energy)]])

    if weights is None:
        weights_aug = np.ones(a_aug.shape[0], dtype=float)
        weights_aug[-1] = float(reference_weight)
    else:
        if len(weights) != len(b_vector):
            raise ValueError("SFC weights must match the number of RBFE edges.")
        weights_aug = np.concatenate([np.asarray(weights, dtype=float), [reference_weight]])

    if np.any(~np.isfinite(weights_aug)) or np.any(weights_aug <= 0):
        raise ValueError("SFC weights must be finite and > 0.")

    sqrt_weights = np.sqrt(weights_aug)
    weighted_a = a_aug * sqrt_weights[:, None]
    weighted_b = b_aug * sqrt_weights
    solution, *_ = np.linalg.lstsq(weighted_a, weighted_b, rcond=None)

    # A constant shift leaves every predicted ddG unchanged and makes the
    # reported reference free energy exact instead of merely high-weighted.
    solution = solution + (float(reference_free_energy) - solution[reference_index])
    return solution


def _edge_dataframe(
    edges: Sequence[CycleClosureEdge],
    labels: Sequence[str],
    scheme_vectors: dict[str, np.ndarray],
    selected_scheme: str,
) -> pd.DataFrame:
    label_index = {label: idx for idx, label in enumerate(labels)}
    records: list[dict[str, float | str]] = []

    for edge in edges:
        idx_a = label_index[edge.label_a]
        idx_b = label_index[edge.label_b]
        record: dict[str, float | str] = {
            "labelA": edge.label_a,
            "labelB": edge.label_b,
        }
        for scheme, vector in scheme_vectors.items():
            predicted = float(vector[idx_b] - vector[idx_a])
            record[f"ddG_{scheme}"] = predicted
            record[f"pair_error_{scheme}"] = abs(float(edge.ddg) - predicted)
        record["pair_error"] = float(record[f"pair_error_{selected_scheme}"])
        records.append(record)

    return pd.DataFrame.from_records(records)


def _node_error_vectors(
    edges: Sequence[CycleClosureEdge],
    labels: Sequence[str],
    vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    incident_errors: list[list[float]] = [[] for _ in labels]
    for edge in edges:
        idx_a = label_index[edge.label_a]
        idx_b = label_index[edge.label_b]
        predicted = float(vector[idx_b] - vector[idx_a])
        residual = abs(float(edge.ddg) - predicted)
        incident_errors[idx_a].append(residual)
        incident_errors[idx_b].append(residual)

    max_error = np.zeros(len(labels), dtype=float)
    rms_error = np.zeros(len(labels), dtype=float)
    for idx, errors in enumerate(incident_errors):
        if not errors:
            continue
        arr = np.asarray(errors, dtype=float)
        max_error[idx] = float(np.max(arr))
        rms_error[idx] = float(np.sqrt(np.mean(np.square(arr))))
    return max_error, rms_error


def _node_dataframe(
    edges: Sequence[CycleClosureEdge],
    labels: Sequence[str],
    scheme_vectors: dict[str, np.ndarray],
    selected_scheme: str,
) -> pd.DataFrame:
    scheme_errors = {
        scheme: _node_error_vectors(edges, labels, vector)
        for scheme, vector in scheme_vectors.items()
    }

    records: list[dict[str, float | str]] = []
    for idx, label in enumerate(labels):
        record: dict[str, float | str] = {"label": label}
        for scheme, vector in scheme_vectors.items():
            max_error, rms_error = scheme_errors[scheme]
            record[f"dG_{scheme}"] = float(vector[idx])
            record[f"path_dependent_error_{scheme}"] = float(max_error[idx])
            record[f"path_independent_error_{scheme}"] = float(rms_error[idx])

        selected_max, selected_rms = scheme_errors[selected_scheme]
        record["path_dependent_error"] = float(selected_max[idx])
        record["path_independent_error"] = float(selected_rms[idx])
        records.append(record)

    return pd.DataFrame.from_records(records)


def calculate_cycle_closure(
    edges: Iterable[CycleClosureEdge | Sequence[object]],
    *,
    reference: str | None = None,
    reference_free_energy: float = 0.0,
    reference_weight: float = 1e6,
    require_cycles: bool | None = None,
    **_compat_kwargs,
) -> CycleClosureResult:
    """Run SFC/WSFC correction on an RBFE graph.

    ``require_cycles`` and extra keyword arguments are accepted for compatibility
    with the previous WCC implementation. SFC does not enumerate cycles and can
    operate on any connected RBFE graph.
    """

    coerced_edges = _coerce_edges(edges)
    labels = _ordered_labels(coerced_edges, reference)
    reference = labels[0]
    a_matrix, b_vector, uncertainty_matrix = _design_matrix(coerced_edges, labels)
    _validate_connected_system(a_matrix, len(labels))

    scheme_vectors: dict[str, np.ndarray] = {
        "sfc": _solve_state_function(
            a_matrix,
            b_vector,
            weights=None,
            reference_index=0,
            reference_free_energy=reference_free_energy,
            reference_weight=reference_weight,
        )
    }

    for col_idx in range(uncertainty_matrix.shape[1]):
        scheme = f"wsfc{col_idx + 1}"
        scheme_vectors[scheme] = _solve_state_function(
            a_matrix,
            b_vector,
            weights=_uncertainty_weights(uncertainty_matrix[:, col_idx]),
            reference_index=0,
            reference_free_energy=reference_free_energy,
            reference_weight=reference_weight,
        )

    selected_scheme = next(reversed(scheme_vectors))
    schemes = tuple(scheme_vectors.keys())
    return CycleClosureResult(
        reference=reference,
        reference_free_energy=float(reference_free_energy),
        node_results=_node_dataframe(coerced_edges, labels, scheme_vectors, selected_scheme),
        edge_results=_edge_dataframe(coerced_edges, labels, scheme_vectors, selected_scheme),
        cycles=(),
        iterations=tuple(1 for _ in schemes),
        converged=tuple(True for _ in schemes),
        method="sfc",
        schemes=schemes,
    )


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def cycle_closure_from_dataframe(
    df: pd.DataFrame,
    *,
    label_a_col: str = "labelA",
    label_b_col: str = "labelB",
    ddg_col: str | None = None,
    uncertainty_cols: Sequence[str] | None = None,
    reference: str | None = None,
    reference_free_energy: float = 0.0,
    **kwargs,
) -> CycleClosureResult:
    """Build SFC input from a dataframe and run the correction."""

    if ddg_col is None:
        ddg_col = _first_existing_column(
            df,
            ("calc_DDG", "DDG (kcal/mol)", "DDG", "ddG", "ddg", "dG", "total_dG"),
        )
    if ddg_col is None:
        raise ValueError("Could not infer the SFC ddG column.")

    if uncertainty_cols is None:
        uncertainty_col = _first_existing_column(
            df,
            (
                "calc_dDDG",
                "uncertainty (kcal/mol)",
                "uncertainty",
                "dDDG",
                "ddG_error",
                "total_se",
                "std",
            ),
        )
        uncertainty_cols = [uncertainty_col] if uncertainty_col is not None else []

    required = {label_a_col, label_b_col, ddg_col, *uncertainty_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing SFC dataframe columns: {sorted(missing)}")

    edges = [
        CycleClosureEdge(
            label_a=str(row[label_a_col]),
            label_b=str(row[label_b_col]),
            ddg=float(row[ddg_col]),
            uncertainties=tuple(float(row[column]) for column in uncertainty_cols),
        )
        for _, row in df.iterrows()
    ]
    return calculate_cycle_closure(
        edges,
        reference=reference,
        reference_free_energy=reference_free_energy,
        **kwargs,
    )


def read_cycle_closure_file(path: str | Path) -> pd.DataFrame:
    """Read a whitespace-delimited SFC input file.

    The first three columns are named ``labelA``, ``labelB``, and ``ddG``.
    Additional columns are treated as standard-error columns named ``std1``,
    ``std2``, etc.
    """

    input_path = Path(path)
    df = pd.read_csv(input_path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError("SFC input files need at least three columns.")

    columns = ["labelA", "labelB", "ddG"]
    columns.extend(f"std{idx}" for idx in range(1, df.shape[1] - 2))
    df.columns = columns
    return df


def cycle_closure_from_file(
    path: str | Path,
    *,
    reference: str | None = None,
    reference_free_energy: float = 0.0,
    **kwargs,
) -> CycleClosureResult:
    """Read an SFC-style input file and run state-function correction."""

    df = read_cycle_closure_file(path)
    uncertainty_cols = [column for column in df.columns if column.startswith("std")]
    return cycle_closure_from_dataframe(
        df,
        ddg_col="ddG",
        uncertainty_cols=uncertainty_cols,
        reference=reference,
        reference_free_energy=reference_free_energy,
        **kwargs,
    )


calculate_state_function_correction = calculate_cycle_closure
state_function_correction_from_dataframe = cycle_closure_from_dataframe
read_state_function_correction_file = read_cycle_closure_file
state_function_correction_from_file = cycle_closure_from_file
