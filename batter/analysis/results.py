"""Parsing helpers for legacy component-wise free energy outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from loguru import logger

__all__ = ["ComponentFEResult", "FEResult"]


class ComponentFEResult:
    """
    Container for individual component free energies.

    Parameters
    ----------
    fe_value : float or None, optional
        Default value for every component prior to parsing.
    fe_std : float or None, optional
        Default uncertainty associated with ``fe_value``.
    """

    _component_keys = ["fe", "attach", "elec", "lj", "release", "boresch", "uno", "uno-rest"]

    def __init__(self, fe_value: float | None = None, fe_std: float | None = None) -> None:
        self._results: Dict[str, Tuple[float | str | None, float | str | None]] = {}
        for key in self._component_keys:
            self._results[key] = (fe_value, fe_std)

    @property
    def results(self) -> Dict[str, Tuple[float | str | None, float | str | None]]:
        """Return the internal component map."""
        return self._results

    def to_dict(self) -> Dict[str, Dict[str, float | str | None]]:
        """
        Convert results into a JSON-friendly dictionary.

        Returns
        -------
        dict
            ``{"component": {"value": float | str | None, "std": float | str | None}}``
            for each component listed in :attr:`_component_keys`.
        """
        json_dict: Dict[str, Dict[str, float | str | None]] = {}
        for key in self._component_keys:
            value, std = self._results.get(key, (None, None))
            json_dict[key] = {"value": value, "std": std}
        return json_dict

    def __repr__(self) -> str:  # pragma: no cover - string repr convenience only
        return f"FE: {self.fe if 'fe' in self._results else 'not calculated'} kcal/mol"


def _make_property(name: str, index: int):
    def getter(self):
        return self._results[name][index] if name in self._results else None

    return property(getter)


for key in ComponentFEResult._component_keys:
    setattr(ComponentFEResult, f"{key}", _make_property(key, 0))
    setattr(ComponentFEResult, f"{key}_std", _make_property(key, 1))


class FEResult(ComponentFEResult):
    """
    Representation of the ``Results/results.dat`` style output from BAT.py.

    Parameters
    ----------
    result_file : str or Path
        Path to the legacy text file.
    fe_timeseries : array-like, optional
        Optional time series of cumulative FE estimates to pack alongside the
        scalar results.
    """

    _LINE_MAP = {
        "Boresch": "boresch",
        "e": "elec",
        "v": "lj",
        "o": "uno",
        "n": "release",
        "m": "attach",
        "z": "uno-rest",
        "Total": "fe",
    }

    def __init__(self, result_file: str | os.PathLike[str], fe_timeseries=None):
        super().__init__()
        self.result_file = Path(result_file)
        self.fe_timeseries = fe_timeseries
        if not self.result_file.exists():
            raise FileNotFoundError(f"File {self.result_file} does not exist")
        if fe_timeseries is not None:
            self._results["fe_timeseries"] = fe_timeseries
        self._read_results()

    @classmethod
    def from_lines(cls, lines: Iterable[str], fe_timeseries=None) -> "FEResult":
        """
        Build an :class:`FEResult` from an iterable of lines.

        Parameters
        ----------
        lines : Iterable[str]
            Text lines formatted like the legacy results file.
        fe_timeseries : array-like, optional
            Optional time series to stash under ``fe_timeseries``.

        Returns
        -------
        FEResult
            Parsed object that behaves like the regular file-backed version.
        """
        obj = cls.__new__(cls)
        ComponentFEResult.__init__(obj)
        obj.result_file = Path("<in-memory>")
        obj.fe_timeseries = fe_timeseries
        if fe_timeseries is not None:
            obj._results["fe_timeseries"] = fe_timeseries
        obj._results.update(cls._parse_lines(lines))
        return obj

    @property
    def is_unbound(self) -> bool:
        """Return ``True`` if the analysis marked the complex as unbound."""
        value = self._results.get("fe")
        return isinstance(value, tuple) and value[0] == "unbound"

    def _read_results(self) -> None:
        with self.result_file.open("r") as handle:
            lines = handle.readlines()
        self._results.update(self._parse_lines(lines))

    @classmethod
    def _parse_lines(cls, lines: Iterable[str]) -> Dict[str, Tuple[float | str | None, float | str | None]]:
        """
        Parse legacy BAT.py analysis output.

        Parameters
        ----------
        lines : Iterable[str]
            File contents.

        Returns
        -------
        dict
            Mapping of component name to ``(value, std)``.
        """
        lines = list(lines)
        if not lines:
            return {}
        first = lines[0].strip()
        if "FAILED" in first:
            raise ValueError("Analysis failed")
        if "UNBOUND" in first:
            return {key: ("unbound", "unbound") for key in cls._component_keys}

        results: Dict[str, Tuple[float | str | None, float | str | None]] = {}
        for line in lines:
            comp = cls._component_from_line(line.strip())
            if comp is None:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            energy_token = tokens[-2].rstrip(",")
            std_token = tokens[-1]
            if energy_token.lower() == "na":
                results[comp] = (np.nan, np.nan)
                continue
            try:
                energy = float(energy_token)
                std = float(std_token)
            except ValueError:
                logger.warning("Could not parse FE line: %s", line.strip())
                continue
            results[comp] = (energy, std)
        return results

    @classmethod
    def _component_from_line(cls, line: str) -> str | None:
        for prefix, key in cls._LINE_MAP.items():
            if line.startswith(prefix):
                return key
        return None
