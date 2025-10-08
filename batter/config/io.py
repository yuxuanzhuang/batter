from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml
from .simulation import SimulationConfig
from batter.utils import COMPONENTS_LAMBDA_DICT

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())

def read_yaml_config(path: str | Path) -> SimulationConfig:
    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return SimulationConfig(**raw)

def write_yaml_config(cfg: SimulationConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=True))