# batter/exec/local.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from loguru import logger
from joblib import Parallel, delayed

from batter.pipeline.step import Step, ExecResult
from batter.pipeline.pipeline import Pipeline
from batter.systems.core import SimSystem
from batter.exec.base import ExecBackend

Handler = Callable[[Step, SimSystem, Mapping], ExecResult]


def _run_pipeline_task(
    pipeline: Pipeline,
    backend: "LocalBackend",
    sys: SimSystem,
) -> Tuple[str, Mapping[str, ExecResult] | None, BaseException | None]:
    """
    Top-level function so joblib can pickle it.
    Returns (system_name, results_or_None, error_or_None).
    """
    try:
        results = pipeline.run(backend, sys)
        return sys.name, results, None
    except BaseException as e:
        return sys.name, None, e


@dataclass
class LocalBackend(ExecBackend):
    """
    Local backend with pluggable handler registry.

    Now supports process-based parallel via joblib (loky).
    """

    name: str = "local"
    _handlers: Dict[str, Handler] = field(default_factory=dict)
    _max_workers: Optional[int] = None  # None = auto, 0/1 = serial

    def __init__(self, max_workers: Optional[int] = None):
        object.__setattr__(self, "name", "local")
        object.__setattr__(self, "_handlers", {})
        object.__setattr__(self, "_max_workers", max_workers)

    # ---------- registry ----------
    def register(self, step_name: str, handler: Handler) -> None:
        self._handlers[step_name] = handler

    # ---------- ExecBackend ----------
    def run(self, step: Step, system: SimSystem, params: Mapping) -> ExecResult:
        h = self._handlers.get(step.name)
        if h is None:
            logger.debug("LOCAL: no handler for step {!r}; treating as no-op.", step.name)
            return ExecResult(job_ids=[], artifacts={})
        logger.debug("LOCAL: executing step {!r}", step.name)
        return h(step, system, params)

    # ---------- parallel pipeline runner (process-based via joblib) ----------
    def run_parallel(
        self,
        pipeline: Pipeline,
        systems: Iterable[SimSystem],
        *,
        max_workers: Optional[int] = None,
        description: str = "",
        # joblib-specific knobs
        batch_size: str | int = "auto",
        verbose: int = 10,  # joblib progress logging
        prefer: str = "processes",  # enforce processes
        backend: Optional[str] = None,  # keep None to let joblib choose 'loky' for processes
    ) -> Dict[str, Mapping[str, ExecResult]]:
        """
        Run the Pipeline for many systems concurrently using joblib (processes).

        Returns
        -------
        dict: { system_name: { step_name: ExecResult, ... }, ... }
        """
        systems = list(systems)
        if not systems:
            return {}

        # resolve worker count
        mw = max_workers if max_workers is not None else self._max_workers
        if mw in (0, 1):
            logger.debug(
                "LOCAL(parallel): running serially for {} system(s) (max_workers={}) — {}",
                len(systems), mw, description,
            )
            out: Dict[str, Mapping[str, ExecResult]] = {}
            errors: Dict[str, BaseException] = {}
            for sys in systems:
                try:
                    out[sys.name] = pipeline.run(self, sys)
                except BaseException as e:
                    errors[sys.name] = e
                    raise RuntimeError(f"LOCAL(parallel-serial): {sys.name} failed") from e
            if errors:
                logger.warning(
                    "LOCAL(parallel-serial): {} system(s) failed: {}",
                    len(errors), ", ".join(errors.keys()),
                )
            return out

        # auto: cap by CPU count and number of systems
        if mw is None:
            cpu = os.cpu_count() or 1
            mw = min(len(systems), cpu)

        logger.debug(
            "LOCAL(parallel): joblib(loky) with n_jobs={} for {} system(s) — {}",
            mw, len(systems), description,
        )

        # IMPORTANT: self, pipeline, and systems must be picklable.
        # Ensure handlers are top-level callables.
        results: List[Tuple[str, Mapping[str, ExecResult] | None, BaseException | None]] = Parallel(
            n_jobs=mw,
            backend=backend,        # None → 'loky' for processes
            prefer=prefer,          # 'processes'
            batch_size=batch_size,
            verbose=verbose,
        )(
            delayed(_run_pipeline_task)(pipeline, self, sys)
            for sys in systems
        )

        out: Dict[str, Mapping[str, ExecResult]] = {}
        errors: Dict[str, BaseException] = {}

        for name, res, err in results:
            if err is None and res is not None:
                out[name] = res
                logger.debug("LOCAL(parallel): finished {}", name)
            else:
                errors[name] = err or RuntimeError("Unknown error")
                raise RuntimeError(f"LOCAL(parallel): {name} failed") from errors[name]

        if errors:
            logger.warning(
                "LOCAL(parallel): {} system(s) failed in parallel run: {}",
                len(errors), ", ".join(errors.keys()),
            )
        return out