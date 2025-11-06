from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping


__all__ = ["Step", "ExecResult"]


@dataclass(frozen=True, slots=True)
class Step:
    """
    One unit of work in the pipeline.

    Parameters
    ----------
    name : str
        Unique step name (e.g., ``"prepare_fe"``).
    requires : list[str]
        Names of steps that must complete before this step can run.
    payload : Any, optional
        Typed payload consumed by the backend. Typically a :class:`~batter.pipeline.payloads.StepPayload`.

    Notes
    -----
    - Steps are **immutable** descriptors. Execution is handled by a backend.
    - The backend decides how to interpret ``params`` (e.g., templates, flags).
    """
    name: str
    requires: List[str] = field(default_factory=list)
    payload: Any = None

    @property
    def params(self) -> Any:
        """Backwards-compatible alias for ``payload``."""
        return self.payload


@dataclass(slots=True)
class ExecResult:
    """
    Execution result returned by a backend.

    Parameters
    ----------
    job_ids : list[str]
        Scheduler or process identifiers (may be empty for local runs).
    artifacts : Mapping[str, Any]
        Named outputs (paths, metrics, small JSON blobs).
    """
    job_ids: List[str] = field(default_factory=list)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
