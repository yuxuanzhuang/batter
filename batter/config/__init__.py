"""
High-level helpers for loading and saving BATTER configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .run import RunConfig
from .simulation import SimulationConfig
from .utils import expand_env_vars

__all__ = [
    "load_run_config",
    "dump_run_config",
    "load_simulation_config",
    "dump_simulation_config",
]


def load_run_config(path: Path | str) -> RunConfig:
    """Read a run-level YAML file and return a validated:class:`RunConfig`."""
    file_path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(file_path.read_text()) or {}
    expanded = expand_env_vars(raw, base_dir=file_path.parent)
    return RunConfig.model_validate(expanded)


def dump_run_config(cfg: RunConfig, path: Path | str) -> None:
    """Serialize a :class:`RunConfig` to YAML."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=True)
    )


def load_simulation_config(path: Path | str) -> SimulationConfig:
    """Load a simulation config YAML file."""
    file_path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(file_path.read_text()) or {}
    expanded = expand_env_vars(raw, base_dir=file_path.parent)
    return SimulationConfig.model_validate(expanded)


def dump_simulation_config(cfg: SimulationConfig, path: Path | str) -> None:
    """Write a :class:`SimulationConfig` to YAML."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=True)
    )
