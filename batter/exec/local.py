"""Execution backend for running pipelines locally."""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from loguru import logger
from joblib import Parallel, delayed

from batter.exec.base import ExecBackend
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem

Handler = Callable[[Step, SimSystem, Mapping], ExecResult]

_RECYCLED_WORKER_PHASES = frozenset({"pre_prepare_fe", "prepare_fe"})


def _slurm_task_limit() -> int | None:
    """Return the worker-process limit declared by the active Slurm job.

    BATTER manager jobs reserve multiple Slurm tasks and then use those slots
    for local ``joblib`` workers.  Respecting ``SLURM_NTASKS`` prevents a YAML
    ``max_workers`` value from oversubscribing both the manager's CPU and
    memory allocation.  Outside Slurm there is no additional limit.
    """
    raw_value = os.environ.get("SLURM_NTASKS")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _effective_worker_cap(requested: int | None, n_systems: int) -> int:
    """Resolve local parallelism without exceeding the Slurm allocation."""
    if requested is None:
        worker_cap = min(n_systems, os.cpu_count() or 1)
    else:
        worker_cap = min(requested, n_systems)

    slurm_limit = _slurm_task_limit()
    if slurm_limit is not None and worker_cap > slurm_limit:
        logger.info(
            "LOCAL(parallel): limiting workers from {} to {} to match "
            "SLURM_NTASKS={}",
            worker_cap,
            slurm_limit,
            slurm_limit,
        )
        worker_cap = slurm_limit
    return worker_cap


def _shutdown_reusable_process_pool() -> None:
    """Release loky workers so phase-local native allocations are returned.

    ``joblib`` intentionally keeps its loky executor alive between calls.
    Amber/MDAnalysis preparation can leave several gigabytes of native memory
    mapped in each worker, so reusing those processes across many ligands can
    exhaust a Slurm manager's cgroup even when the number of concurrent tasks
    is modest.  The executor reference is private joblib state, hence the
    defensive lookup and best-effort cleanup.
    """
    try:
        from joblib.externals.loky import reusable_executor

        executor = getattr(reusable_executor, "_executor", None)
        if executor is None:
            return
        terminate = getattr(executor, "terminate", None)
        if callable(terminate):
            terminate(kill_workers=True)
        else:  # pragma: no cover - compatibility with older loky versions
            executor.shutdown(wait=True, kill_workers=True)
    except Exception as exc:  # pragma: no cover - defensive cleanup
        logger.warning("Could not recycle local worker processes: {}", exc)


def _run_pipeline_task(
    pipeline: Pipeline,
    backend: "LocalBackend",
    sys: SimSystem,
) -> Tuple[str, Mapping[str, ExecResult] | None, str | None, str | None]:
    """Execute ``pipeline`` for a single system.

    Parameters
    ----------
    pipeline :
        Pipeline instance to execute.
    backend :
        Backend used to dispatch individual steps.
    sys :
        Simulation system descriptor.

    Returns
    -------
    tuple of (str, Mapping[str, ExecResult] or None, str or None, str or None)
        Tuple containing the system name, the step results if successful, the
        exception text otherwise, and formatted traceback text captured inside
        the worker process. The structure is joblib-friendly.
    """
    try:
        results = pipeline.run(backend, sys)
        return sys.name, results, None, None
    except BaseException as exc:  # pragma: no cover - propagated to parent
        return sys.name, None, _exception_text_for_worker(exc), traceback.format_exc()


def _exception_text_for_worker(exc: BaseException) -> str:
    """Return a picklable exception label for worker-to-parent reporting."""
    try:
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    except BaseException:
        try:
            text = repr(exc)
        except BaseException:
            text = type(exc).__name__
    return text or type(exc).__name__


def _format_failure_detail(exc: BaseException, tb_text: str | None = None) -> str:
    """Return useful failure detail for parent-process logging."""
    text = (tb_text or "").strip()
    if text:
        return text
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _failure_summary_line(exc: BaseException, tb_text: str | None = None) -> str:
    """Return a compact one-line failure summary."""
    detail = _format_failure_detail(exc, tb_text)
    for line in reversed(detail.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return repr(exc)


def _log_failures(
    prefix: str,
    errors: Mapping[str, BaseException],
    tracebacks: Mapping[str, str | None],
) -> None:
    """Log per-system failure details in the parent process."""
    logger.error(
        "{}: {} system(s) failed: {}",
        prefix,
        len(errors),
        ", ".join(errors.keys()),
    )
    for name, exc in errors.items():
        logger.error(
            "{}: failure details for {}\n{}",
            prefix,
            name,
            _format_failure_detail(exc, tracebacks.get(name)),
        )


def _parallel_failure_message(
    prefix: str,
    errors: Mapping[str, BaseException],
    tracebacks: Mapping[str, str | None],
) -> str:
    """Build an exception message that keeps per-system failure causes visible."""
    lines = [
        f"{prefix}: failures encountered for {', '.join(errors.keys())}",
        "Failure summaries:",
    ]
    for name, exc in errors.items():
        lines.append(f"- {name}: {_failure_summary_line(exc, tracebacks.get(name))}")
    return "\n".join(lines)


@dataclass
class LocalBackend(ExecBackend):
    """In-process execution backend with optional parallel orchestration.

    Parameters
    ----------
    max_workers : int, optional
        Maximum number of worker processes to use when :meth:`run_parallel`
        is invoked. ``None`` lets the backend auto-detect resources; ``0`` or
        ``1`` forces serial execution.
    """

    name: str = "local"
    _handlers: Dict[str, Handler] = field(default_factory=dict)
    _max_workers: Optional[int] = None

    def __init__(self, max_workers: Optional[int] = None):
        object.__setattr__(self, "name", "local")
        object.__setattr__(self, "_handlers", {})
        object.__setattr__(self, "_max_workers", max_workers)

    # ---------- registry ----------
    def register(self, step_name: str, handler: Handler) -> None:
        """Register a callable to execute ``step_name``.

        Parameters
        ----------
        step_name : str
            Identifier of the step (matches :class:`batter.pipeline.step.Step.name`).
        handler : Callable[[Step, SimSystem, Mapping], ExecResult]
            Function responsible for executing the step.
        """
        self._handlers[step_name] = handler

    # ---------- ExecBackend ----------
    def run(self, step: Step, system: SimSystem, params: Mapping) -> ExecResult:
        """Execute ``step`` for ``system`` on the local machine.

        Parameters
        ----------
        step :
            Pipeline step metadata.
        system :
            Simulation system descriptor.
        params :
            Step parameters, typically generated by the orchestration layer.

        Returns
        -------
        ExecResult
            Artifacts and job identifiers (empty for local execution).
        """
        handler = self._handlers.get(step.name)
        if handler is None:
            logger.debug("LOCAL: no handler for step {!r}; treating as no-op.", step.name)
            return ExecResult(job_ids=[], artifacts={})
        logger.debug("LOCAL: executing step {!r}", step.name)
        return handler(step, system, params)

    # ---------- parallel pipeline runner (process-based via joblib) ----------
    def run_parallel(
        self,
        pipeline: Pipeline,
        systems: Iterable[SimSystem],
        *,
        max_workers: Optional[int] = None,
        description: str = "",
        batch_size: str | int = "auto",
        verbose: int = 0,
        prefer: str = "processes",
        backend: Optional[str] = None,
    ) -> Dict[str, Mapping[str, ExecResult]]:
        """Execute ``pipeline`` for multiple systems in parallel.

        Parameters
        ----------
        pipeline :
            Pipeline object providing the sequence of steps to execute.
        systems : Iterable[SimSystem]
            Collection of systems to process.
        max_workers : int, optional
            Override the configured worker cap; ``None`` falls back to the
            value provided at construction time.
        description : str, optional
            Human-readable label used in debug logging.
        batch_size, verbose, prefer, backend :
            Joblib configuration knobs forwarded to :class:`joblib.Parallel`.

        Returns
        -------
        dict
            Mapping of ``system.name`` to per-step results.

        Raises
        ------
        RuntimeError
            When one or more systems fail.
        """
        systems = list(systems)
        if not systems:
            return {}

        requested_workers = (
            max_workers if max_workers is not None else self._max_workers
        )
        worker_cap = _effective_worker_cap(requested_workers, len(systems))
        if worker_cap in (0, 1):
            logger.debug(
                "LOCAL(parallel): running serially for {} system(s) (max_workers={}) — {}",
                len(systems),
                worker_cap,
                description,
            )
            out: Dict[str, Mapping[str, ExecResult]] = {}
            errors: Dict[str, BaseException] = {}
            traces: Dict[str, str | None] = {}
            for sys in systems:
                try:
                    out[sys.name] = pipeline.run(self, sys)
                except BaseException as exc:  # pragma: no cover - passthrough
                    errors[sys.name] = exc
                    traces[sys.name] = traceback.format_exc()
            if errors:
                prefix = "LOCAL(parallel-serial)"
                _log_failures(prefix, errors, traces)
                raise RuntimeError(
                    _parallel_failure_message(prefix, errors, traces)
                ) from next(iter(errors.values()))
            return out

        logger.debug(
            "LOCAL(parallel): joblib(loky) with n_jobs={} for {} system(s) — {}",
            worker_cap,
            len(systems),
            description,
        )

        def _execute_batch(batch_systems: List[SimSystem]):
            return Parallel(
                n_jobs=worker_cap,
                backend=backend,
                prefer=prefer,
                batch_size=batch_size,
                verbose=verbose,
                max_nbytes=None,
            )(
                delayed(_run_pipeline_task)(pipeline, self, sys)
                for sys in batch_systems
            )

        results: List[
            Tuple[str, Mapping[str, ExecResult] | None, str | None, str | None]
        ] = []
        recycle_workers = (
            description in _RECYCLED_WORKER_PHASES
            and prefer != "threads"
            and backend in (None, "loky")
        )
        if recycle_workers:
            logger.info(
                "LOCAL(parallel): recycling workers after each {}-system batch "
                "for memory-intensive phase {}",
                worker_cap,
                description,
            )
            # Do not carry allocations from an earlier parallel phase into FE
            # preparation.  Each subsequent worker handles at most one ligand.
            _shutdown_reusable_process_pool()
            for start in range(0, len(systems), worker_cap):
                try:
                    results.extend(_execute_batch(systems[start : start + worker_cap]))
                finally:
                    _shutdown_reusable_process_pool()
        else:
            results = _execute_batch(systems)

        out: Dict[str, Mapping[str, ExecResult]] = {}
        errors: Dict[str, BaseException] = {}
        traces: Dict[str, str | None] = {}

        for name, res, err, tb_text in results:
            if err is None and res is not None:
                out[name] = res
                logger.debug("LOCAL(parallel): finished {}", name)
            else:
                errors[name] = RuntimeError(err or "Unknown error")
                traces[name] = tb_text

        if errors:
            prefix = "LOCAL(parallel)"
            _log_failures(prefix, errors, traces)
            raise RuntimeError(
                _parallel_failure_message(prefix, errors, traces)
            ) from next(iter(errors.values()))

        return out
