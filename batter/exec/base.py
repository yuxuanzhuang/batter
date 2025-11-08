"""Interfaces shared by execution backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Protocol

from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem

__all__ = ["Resources", "ExecBackend"]


@dataclass(frozen=True, slots=True)
class Resources:
    """Resource hints supplied to execution backends.

    Parameters
    ----------
    time : str, optional
        Walltime (e.g., ``"02:00:00"``).
    cpus : int, optional
        CPU cores per task.
    gpus : int, optional
        Number of GPUs required.
    mem : str, optional
        Memory request (e.g., ``"16G"``).
    partition : str, optional
        Scheduler partition or queue.
    account : str, optional
        Scheduler account.
    extra : Mapping[str, str], optional
        Backend-specific SBATCH-style flags.
    """

    time: Optional[str] = None
    cpus: Optional[int] = None
    gpus: Optional[int] = None
    mem: Optional[str] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    extra: Mapping[str, str] = field(default_factory=dict)


class ExecBackend(Protocol):
    """Protocol implemented by execution backends."""

    name: str

    def run(self, step: Step, system: SimSystem, params: Dict) -> ExecResult:
        """Execute ``step`` for ``system``.

        Parameters
        ----------
        step : Step
            Step metadata as produced by the pipeline.
        system : SimSystem
            Simulation system descriptor.
        params : dict
            Backend-specific parameters, potentially including ``resources``.

        Returns
        -------
        ExecResult
            Execution artifacts and job identifiers.
        """
        ...
