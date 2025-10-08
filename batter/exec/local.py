from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from loguru import logger

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem
from .base import ExecBackend


__all__ = ["LocalBackend"]


Handler = Callable[[Step, SimSystem, Dict[str, Any]], ExecResult]


@dataclass(slots=True)
class LocalBackend(ExecBackend):
    """
    Local in-process backend with a pluggable handler registry.

    Notes
    -----
    - By default, unknown steps are treated as **no-ops** and return an empty
      :class:`ExecResult`. This lets you wire pieces incrementally.
    - You can register Python callables for step names via :meth:`register`.
    - Handlers receive (step, system, params) and must return :class:`ExecResult`.
    """
    name: str = "local"
    _handlers: Dict[str, Handler] = field(default_factory=dict)

    # ------------- public -------------

    def register(self, step_name: str, handler: Handler) -> None:
        """
        Register a handler for a specific step name.

        Parameters
        ----------
        step_name : str
            Name of the step to handle (e.g., ``"equil"``).
        handler : Callable
            Callable of type ``(step, system, params) -> ExecResult``.
        """
        self._handlers[step_name] = handler

    # ------------- ExecBackend -------------

    def run(self, step: Step, system: SimSystem, params: Dict) -> ExecResult:
        h = self._handlers.get(step.name)
        if h is None:
            logger.info("LOCAL: no handler for step {!r}; treating as no-op.", step.name)
            return ExecResult(job_ids=[], artifacts={})
        logger.info("LOCAL: executing step {!r}", step.name)
        return h(step, system, params)