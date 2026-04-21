"""Weighted cycle-closure correction for RBFE networks.

Acknowledgement
---------------
This module rewrites the cycle-closure workflow from the MIT-licensed
``zlisysu/Weighted_cc`` reference implementation for BATTER's analysis API:
https://github.com/zlisysu/Weighted_cc

For the WCC method, see Li et al., J. Chem. Inf. Model. 2022.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "CycleClosureEdge",
    "CycleClosureResult",
    "calculate_cycle_closure",
    "cycle_closure_from_dataframe",
    "cycle_closure_from_file",
    "read_cycle_closure_file",
]


@dataclass(frozen=True)
class CycleClosureEdge:
    """One directed RBFE edge used as cycle-closure input.

    Parameters
    ----------
    label_a, label_b
        Ligand labels defining the edge direction.
    ddg
        Relative free energy for ``label_a -> label_b``.
    uncertainties
        Optional standard-deviation columns. Each value creates one weighted
        cycle-closure estimate using variance weighting.
    """

    label_a: str
    label_b: str
    ddg: float
    uncertainties: tuple[float, ...] = ()


@dataclass(frozen=True)
class CycleClosureResult:
    """Cycle-closure result tables and metadata."""

    reference: str
    reference_free_energy: float
    node_results: pd.DataFrame
    edge_results: pd.DataFrame
    cycles: tuple[tuple[str, ...], ...]
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]


class _CycleClosureGraph:
    def __init__(self, edges: Sequence[CycleClosureEdge]) -> None:
        if not edges:
            raise ValueError("Cycle closure requires at least one edge.")

        uncertainty_counts = {len(edge.uncertainties) for edge in edges}
        if len(uncertainty_counts) != 1:
            raise ValueError("All cycle-closure edges must use the same uncertainty columns.")

        self.edges = tuple(edges)
        self.n_estimates = 1 + next(iter(uncertainty_counts))
        self.nodes: list[str] = []
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.values: dict[tuple[str, str], list[float]] = {}
        self.variances: dict[tuple[str, str], list[float]] = {}
        self.pair_errors: dict[tuple[str, str], float] = {}
        self.edge_order: list[tuple[str, str]] = []
        seen_pairs: set[frozenset[str]] = set()

        for edge in self.edges:
            label_a = str(edge.label_a).strip()
            label_b = str(edge.label_b).strip()
            if not label_a or not label_b:
                raise ValueError("Cycle-closure edge labels cannot be empty.")
            if label_a == label_b:
                raise ValueError("Cycle-closure edges cannot connect a ligand to itself.")

            pair_key = frozenset((label_a, label_b))
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate undirected cycle-closure edge: {label_a!r}, {label_b!r}."
                )
            seen_pairs.add(pair_key)

            ddg = float(edge.ddg)
            if not math.isfinite(ddg):
                raise ValueError(f"Non-finite cycle-closure ddG for {label_a}->{label_b}.")

            uncertainties = tuple(float(value) for value in edge.uncertainties)
            if any((not math.isfinite(value)) or value <= 0 for value in uncertainties):
                raise ValueError(
                    f"Uncertainties for {label_a}->{label_b} must be finite and > 0."
                )

            for label in (label_a, label_b):
                if label not in self.adjacency:
                    self.nodes.append(label)
                    self.adjacency[label] = []

            self.adjacency[label_a].append(label_b)
            self.adjacency[label_b].append(label_a)
            self.edge_order.append((label_a, label_b))

            values = [ddg for _ in range(self.n_estimates)]
            reverse_values = [-ddg for _ in range(self.n_estimates)]
            variances = [1.0, *(value * value for value in uncertainties)]
            initial_error = uncertainties[0] if uncertainties else 0.0

            self.values[(label_a, label_b)] = values
            self.values[(label_b, label_a)] = reverse_values
            self.variances[(label_a, label_b)] = list(variances)
            self.variances[(label_b, label_a)] = list(variances)
            self.pair_errors[(label_a, label_b)] = initial_error
            self.pair_errors[(label_b, label_a)] = initial_error

    def cycles(self) -> tuple[tuple[str, ...], ...]:
        """Enumerate simple cycles in the same deterministic spirit as WCC."""

        cycles: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        visited = {node: False for node in self.nodes}

        def dfs(start: str, current: str, path: list[str], depth: int) -> None:
            if depth > 0:
                visited[current] = True
            path.append(current)

            if depth > 0 and current == start:
                if len(path) > 3:
                    key = tuple(sorted(path[:-1]))
                    if key not in seen:
                        cycles.append(tuple(path))
                        seen.add(key)
            else:
                for neighbor in self.adjacency[current]:
                    if not visited[neighbor]:
                        dfs(start, neighbor, path, depth + 1)

            path.pop()
            visited[current] = False

        for node in self.nodes:
            dfs(node, node, [], 0)
            visited[node] = True

        return tuple(cycles)

    def cycle_delta(self, cycle: Sequence[str], estimate_index: int) -> tuple[float, int, float]:
        delta = 0.0
        variance_sum = 0.0
        edges = 0
        for label_a, label_b in zip(cycle, cycle[1:]):
            delta += self.values[(label_a, label_b)][estimate_index]
            variance_sum += self.variances[(label_a, label_b)][estimate_index]
            edges += 1
        return delta, edges, variance_sum

    def snapshot(self, estimate_index: int) -> dict[tuple[str, str], float]:
        return {key: values[estimate_index] for key, values in self.values.items()}

    def changed(
        self,
        previous: dict[tuple[str, str], float],
        estimate_index: int,
        tolerance: float,
    ) -> bool:
        return any(
            abs(previous[key] - values[estimate_index]) > tolerance
            for key, values in self.values.items()
        )

    def apply_cycle_closure(
        self,
        cycles: Sequence[Sequence[str]],
        estimate_index: int,
        *,
        edge_error_only: bool,
        max_error_cycle_edges: int,
    ) -> None:
        for cycle in cycles:
            delta, edge_count, variance_sum = self.cycle_delta(cycle, estimate_index)
            if edge_count == 0:
                continue

            if edge_error_only:
                if edge_count > max_error_cycle_edges:
                    continue
                cycle_error = abs(delta / math.sqrt(edge_count))
                for label_a, label_b in zip(cycle, cycle[1:]):
                    if cycle_error > self.pair_errors[(label_a, label_b)]:
                        self.pair_errors[(label_a, label_b)] = cycle_error
                        self.pair_errors[(label_b, label_a)] = cycle_error
                continue

            if variance_sum <= 0:
                raise ValueError("Cycle-closure variance sum must be positive.")

            for label_a, label_b in zip(cycle, cycle[1:]):
                scale = self.variances[(label_a, label_b)][estimate_index] / variance_sum
                corrected = self.values[(label_a, label_b)][estimate_index] - scale * delta
                self.values[(label_a, label_b)][estimate_index] = corrected
                self.values[(label_b, label_a)][estimate_index] = -corrected

    def edge_dataframe(self) -> pd.DataFrame:
        records: list[dict[str, float | str]] = []
        for label_a, label_b in self.edge_order:
            record: dict[str, float | str] = {
                "labelA": label_a,
                "labelB": label_b,
                "ddG_cc": self.values[(label_a, label_b)][0],
            }
            for estimate_index in range(1, self.n_estimates):
                record[f"ddG_wcc{estimate_index}"] = self.values[(label_a, label_b)][
                    estimate_index
                ]
            record["pair_error"] = self.pair_errors[(label_a, label_b)]
            records.append(record)
        return pd.DataFrame.from_records(records)


def _coerce_edges(
    edges: Iterable[CycleClosureEdge | Sequence[object]],
) -> tuple[CycleClosureEdge, ...]:
    coerced: list[CycleClosureEdge] = []
    for edge in edges:
        if isinstance(edge, CycleClosureEdge):
            coerced.append(edge)
            continue
        if len(edge) < 3:
            raise ValueError("Cycle-closure edge sequences must contain at least 3 values.")
        label_a, label_b, ddg, *uncertainties = edge
        coerced.append(
            CycleClosureEdge(
                label_a=str(label_a),
                label_b=str(label_b),
                ddg=float(ddg),
                uncertainties=tuple(float(value) for value in uncertainties),
            )
        )
    return tuple(coerced)


def _shortest_error_paths(
    graph: _CycleClosureGraph,
    reference: str,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    distances = {node: math.inf for node in graph.nodes}
    previous: dict[str, str | None] = {node: None for node in graph.nodes}
    distances[reference] = 0.0
    queue: list[tuple[float, str]] = [(0.0, reference)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for neighbor in graph.adjacency[node]:
            edge_error = graph.pair_errors[(node, neighbor)]
            candidate = distance + edge_error * edge_error
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    paths: dict[str, list[str]] = {}
    for node in graph.nodes:
        if not math.isfinite(distances[node]):
            raise ValueError(f"Reference ligand {reference!r} cannot reach ligand {node!r}.")
        path = [node]
        current = node
        while current != reference:
            parent = previous[current]
            if parent is None:
                raise ValueError(
                    f"Reference ligand {reference!r} cannot reach ligand {node!r}."
                )
            path.append(parent)
            current = parent
        paths[node] = path

    return distances, paths


def _path_independent_errors(graph: _CycleClosureGraph) -> dict[str, float]:
    errors: dict[str, float] = {}
    for node in graph.nodes:
        edge_errors = [graph.pair_errors[(node, neighbor)] for neighbor in graph.adjacency[node]]
        errors[node] = max(edge_errors, default=0.0)
    return errors


def _node_dataframe(
    graph: _CycleClosureGraph,
    reference: str,
    reference_free_energy: float,
) -> pd.DataFrame:
    path_dependent_variance, paths = _shortest_error_paths(graph, reference)
    path_independent = _path_independent_errors(graph)
    records: list[dict[str, float | str]] = []

    for node in graph.nodes:
        record: dict[str, float | str] = {"label": node}
        path = paths[node]
        for estimate_index in range(graph.n_estimates):
            free_energy = float(reference_free_energy)
            for label_a, label_b in zip(path, path[1:]):
                free_energy -= graph.values[(label_a, label_b)][estimate_index]
            column = "dG_cc" if estimate_index == 0 else f"dG_wcc{estimate_index}"
            record[column] = free_energy
        record["path_dependent_error"] = math.sqrt(path_dependent_variance[node])
        record["path_independent_error"] = path_independent[node]
        records.append(record)

    return pd.DataFrame.from_records(records)


def calculate_cycle_closure(
    edges: Iterable[CycleClosureEdge | Sequence[object]],
    *,
    reference: str | None = None,
    reference_free_energy: float = 0.0,
    tolerance: float = 0.001,
    minimum_iterations: int = 2,
    max_iterations: int = 10000,
    max_error_cycle_edges: int = 6,
    require_cycles: bool = True,
) -> CycleClosureResult:
    """Run cycle-closure correction on an RBFE graph.

    The first returned estimate, ``dG_cc``/``ddG_cc``, is the unweighted cycle
    closure. Each supplied uncertainty column adds a weighted estimate named
    ``dG_wcc1``, ``dG_wcc2``, and so on.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be > 0.")
    if minimum_iterations < 1:
        raise ValueError("minimum_iterations must be >= 1.")
    if max_iterations < minimum_iterations:
        raise ValueError("max_iterations must be >= minimum_iterations.")
    if max_error_cycle_edges < 1:
        raise ValueError("max_error_cycle_edges must be >= 1.")

    graph = _CycleClosureGraph(_coerce_edges(edges))
    cycles = graph.cycles()
    if require_cycles and not cycles:
        raise ValueError("Cycle closure requires at least one graph cycle.")

    if reference is None:
        reference = graph.nodes[0]
    reference = str(reference).strip()
    if reference not in graph.nodes:
        raise ValueError(f"Reference ligand {reference!r} is not present in the graph.")

    iteration_counts: list[int] = []
    convergence: list[bool] = []
    for estimate_index in range(graph.n_estimates):
        iteration = 0
        previous = graph.snapshot(estimate_index)
        converged = False
        while iteration < minimum_iterations or graph.changed(
            previous, estimate_index, tolerance
        ):
            if iteration >= max_iterations:
                break
            previous = graph.snapshot(estimate_index)
            graph.apply_cycle_closure(
                cycles,
                estimate_index,
                edge_error_only=(iteration == 0),
                max_error_cycle_edges=max_error_cycle_edges,
            )
            iteration += 1
        else:
            converged = True
        iteration_counts.append(iteration)
        convergence.append(converged)

    return CycleClosureResult(
        reference=reference,
        reference_free_energy=float(reference_free_energy),
        node_results=_node_dataframe(graph, reference, reference_free_energy),
        edge_results=graph.edge_dataframe(),
        cycles=cycles,
        iterations=tuple(iteration_counts),
        converged=tuple(convergence),
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
    """Build cycle-closure input from a dataframe and run the correction."""

    if ddg_col is None:
        ddg_col = _first_existing_column(
            df,
            ("calc_DDG", "DDG (kcal/mol)", "ddG", "ddg", "dG", "total_dG"),
        )
    if ddg_col is None:
        raise ValueError("Could not infer the cycle-closure ddG column.")

    if uncertainty_cols is None:
        uncertainty_col = _first_existing_column(
            df,
            (
                "calc_dDDG",
                "uncertainty (kcal/mol)",
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
        raise ValueError(f"Missing cycle-closure dataframe columns: {sorted(missing)}")

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
    """Read a whitespace-delimited WCC-style input file.

    The first three columns are named ``labelA``, ``labelB``, and ``ddG``.
    Additional columns are treated as standard-deviation columns named
    ``std1``, ``std2``, etc.
    """

    input_path = Path(path)
    df = pd.read_csv(input_path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError("Cycle-closure input files need at least three columns.")

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
    """Read a WCC-style input file and run cycle-closure correction."""

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
