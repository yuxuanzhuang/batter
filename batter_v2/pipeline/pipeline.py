from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List

from .step import Step, ExecResult


__all__ = ["Pipeline", "PipelineState"]


@dataclass(slots=True)
class PipelineState:
    """
    In-memory state of a pipeline execution.

    Attributes
    ----------
    results : dict[str, ExecResult]
        Per-step execution results.
    """
    results: Dict[str, ExecResult] = field(default_factory=dict)


class Pipeline:
    """
    Directed acyclic pipeline of :class:`Step` objects.

    Parameters
    ----------
    steps : list[Step]
        Steps that form a DAG. Dependencies are given by ``Step.requires``.

    Notes
    -----
    - A simple **topological sort** is performed before execution.
    - Backends must implement a ``run(step, system) -> ExecResult`` method.
    """

    def __init__(self, steps: List[Step]) -> None:
        self.steps = steps
        self._validate_unique_names()
        self._order = self._toposort()

    # ------------------- public -------------------

    def run(self, backend, system) -> Dict[str, ExecResult]:
        """
        Execute steps in topological order.

        Parameters
        ----------
        backend
            Object providing ``run(step, system) -> ExecResult``.
        system
            The :class:`~batter.systems.core.SimSystem` descriptor.

        Returns
        -------
        dict[str, ExecResult]
            Mapping from step name to execution result.

        Raises
        ------
        RuntimeError
            If a required dependency has not been produced.
        """
        state = PipelineState()
        for step in self._order:
            for req in step.requires:
                if req not in state.results:
                    raise RuntimeError(f"Dependency {req!r} missing before running {step.name!r}")
            res = backend.run(step, system, step.params)  # type: ignore[attr-defined]
            state.results[step.name] = res
        return state.results

    def ordered_steps(self) -> List[Step]:
        """Return steps in execution order."""
        return list(self._order)

    # ------------------- internals -------------------

    def _validate_unique_names(self) -> None:
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate step names: {sorted(dupes)}")

    def _toposort(self) -> List[Step]:
        graph = defaultdict(list)     # node -> children
        indeg = defaultdict(int)      # node -> indegree
        nodes = {s.name: s for s in self.steps}

        for s in self.steps:
            indeg.setdefault(s.name, 0)
            for r in s.requires:
                if r not in nodes:
                    raise ValueError(f"Unknown dependency {r!r} for step {s.name!r}")
                graph[r].append(s.name)
                indeg[s.name] += 1

        q = deque([nodes[n] for n, d in indeg.items() if d == 0])
        order: List[Step] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in graph[u.name]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(nodes[v])

        if len(order) != len(nodes):
            raise ValueError("Cycle detected in pipeline dependencies.")
        return order