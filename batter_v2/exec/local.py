# batter/exec/local.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from batter.pipeline.step import Step, ExecResult
from batter.pipeline.pipeline import Pipeline
from batter.systems.core import SimSystem
from batter.exec.base import ExecBackend  # whatever your base class path is

Handler = Callable[[Step, SimSystem, Mapping], ExecResult]


@dataclass
class LocalBackend(ExecBackend):
    """
    Local in-process backend with a pluggable handler registry.

    Features
    --------
    - `.run(step, system, params)` — execute a single step via a registered handler.
    - `.run_parallel(pipeline, systems, max_workers=None)` — run a Pipeline
      for many systems concurrently using ThreadPoolExecutor.
    - Unknown steps are treated as no-ops (returns empty ExecResult), so you
      can wire things incrementally.

    Notes
    -----
    ThreadPoolExecutor is used by default because most steps are IO/CLI-bound.
    If you *really* want process-based parallelism for CPU-bound handlers,
    it’s safer to implement a dedicated Process backend where handlers and
    their closures are guaranteed to be picklable.
    """

    name: str = "local"
    _handlers: Dict[str, Handler] = field(default_factory=dict)
    _max_workers: Optional[int] = None  # None = auto, 0/1 = serial

    def __init__(self, max_workers: Optional[int] = None):
        # dataclass init interop
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
            logger.info("LOCAL: no handler for step {!r}; treating as no-op.", step.name)
            return ExecResult(job_ids=[], artifacts={})
        logger.info("LOCAL: executing step {!r}", step.name)
        return h(step, system, params)

    # ---------- parallel pipeline runner ----------
    def run_parallel(
        self,
        pipeline: Pipeline,
        systems: Iterable[SimSystem],
        *,
        max_workers: Optional[int] = None,
        description: str = "",
    ) -> Dict[str, Mapping[str, ExecResult]]:
        """
        Run the given Pipeline once per system, concurrently.

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
            # serial execution
            logger.info(
                f"LOCAL(parallel): running serially for {len(systems)} system(s) (max_workers={mw}) — {description}",
            )
            out: Dict[str, Mapping[str, ExecResult]] = {}
            for sys in systems:
                out[sys.name] = pipeline.run(self, sys)
            return out

        # auto workers: min(#systems, CPU count) as a sensible default
        if mw is None:
            cpu = os.cpu_count() or 1
            mw = min(len(systems), cpu)
        logger.info(
            f"LOCAL(parallel): ThreadPoolExecutor with max_workers={mw} for {len(systems)} system(s) — {description}",
        )

        out: Dict[str, Mapping[str, ExecResult]] = {}
        errors: Dict[str, BaseException] = {}

        def _task(sys: SimSystem) -> tuple[str, Mapping[str, ExecResult]]:
            # Each pipeline.run uses this backend's registered handlers
            results = pipeline.run(self, sys)
            return sys.name, results

        with ThreadPoolExecutor(max_workers=mw) as ex:
            fut_map = {ex.submit(_task, sys): sys for sys in systems}
            for fut in as_completed(fut_map):
                sys = fut_map[fut]
                try:
                    name, results = fut.result()
                    out[name] = results
                    logger.info(f"LOCAL(parallel): finished {name}")
                except BaseException as e:
                    errors[sys.name] = e
                    logger.error(f"LOCAL(parallel): {sys.name} failed: {type(e).__name__}: {e}")

        if errors:
            # You can choose to raise, or just log and return partial results.
            # Here we *do not raise*, to let the orchestrator decide (prune/raise).
            logger.warning(f"LOCAL(parallel): {len(errors)} system(s) failed in parallel run: {', '.join(errors.keys())}")
        return out