from __future__ import annotations
from pathlib import Path

from . import load_simulation_config, dump_simulation_config
from .simulation import SimulationConfig


def read_yaml_config(path: str | Path) -> SimulationConfig:
    return load_simulation_config(path)


def write_yaml_config(cfg: SimulationConfig, path: str | Path) -> None:
    dump_simulation_config(cfg, path)
