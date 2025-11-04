from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Protocol

from batter.pipeline.step import Step, ExecResult
from batter.systems.core import SimSystem


__all__ = ["Resources", "ExecBackend"]


@dataclass(frozen=True, slots=True)
class Resources:
    """
    Resource hints for a step.

    Parameters
    ----------
    time : str, optional
        Walltime in Slurm format (e.g., ``"02:00:00"``).
    cpus : int, optional
        CPU cores per task.
    gpus : int, optional
        Number of GPUs.
    mem : str, optional
        Memory (e.g., ``"16G"``).
    partition : str, optional
        Cluster partition/queue.
    account : str, optional
        Slurm account.
    extra : Mapping[str, str]
        Backend-specific extra flags (e.g., constraint, qos).
    """
    time: Optional[str] = None
    cpus: Optional[int] = None
    gpus: Optional[int] = None
    mem: Optional[str] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    extra: Mapping[str, str] = field(default_factory=dict)


class ExecBackend(Protocol):
    """
    Execution backend interface.

    Methods
    -------
    run(step, system, params) -> ExecResult
        Execute ``step`` for ``system`` using ``params``. ``params`` can include
        a ``"resources"`` key that maps to :class:`Resources`-like values.
    """

    name: str

    def run(self, step: Step, system: SimSystem, params: Dict) -> ExecResult: ...