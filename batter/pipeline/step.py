from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


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
    params : dict
        Free-form, JSON-serializable parameters consumed by the backend.

    Notes
    -----
    - Steps are **immutable** descriptors. Execution is handled by a backend.
    - The backend decides how to interpret ``params`` (e.g., templates, flags).
    """
    name: str
    requires: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


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