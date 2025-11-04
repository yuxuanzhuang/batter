"""
Simulation configuration file.

This module provides a class to store essential simulation parameters.
"""
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger

from .input_process import SimulationConfig, FEP_COMPONENTS

def _coerce_yes_no(value: Any) -> str:
    """Coerce various truthy/falsey inputs to 'yes'/'no'."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "no"}:
            return v
        if v in {"true", "t", "1"}:
            return "yes"
        if v in {"false", "f", "0"}:
            return "no"
    if isinstance(value, bool):
        return "yes" if value else "no"
    raise ValueError(f"Invalid yes/no value: {value!r}")


def _normalize_legacy_flat_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build n_steps_dict and n_iter_dict from flat keys like 'c_steps1', 'c_itera2' if needed.
    Respect already-provided dicts and fill missing entries with 0.
    """
    # If user already provided dicts, trust them and fill missing entries.
    n_steps_dict = dict(params.get("n_steps_dict", {}))
    n_iter_dict = dict(params.get("n_iter_dict", {}))

    for comp in FEP_COMPONENTS:
        for ind in ("1", "2"):
            flat_steps_key = f"{comp}_steps{ind}"
            dict_steps_key = flat_steps_key
            if dict_steps_key not in n_steps_dict:
                if flat_steps_key in params:
                    n_steps_dict[dict_steps_key] = int(params[flat_steps_key])
                else:
                    # keep your previous behavior (default 0 -> will be validated later)
                    n_steps_dict[dict_steps_key] = 0

            flat_itera_key = f"{comp}_itera{ind}"
            dict_itera_key = flat_itera_key
            if dict_itera_key not in n_iter_dict:
                if flat_itera_key in params:
                    n_iter_dict[dict_itera_key] = int(params[flat_itera_key])
                else:
                    n_iter_dict[dict_itera_key] = 0

    params["n_steps_dict"] = n_steps_dict
    params["n_iter_dict"] = n_iter_dict
    return params


def _coerce_yaml_compat(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make YAML-friendly values compatible with the existing model:
    - Accept booleans for 'yes'/'no' fields.
    - Lowercase software/fe_type if strings for consistency.
    """
    def _lower_str(k: str):
        v = params.get(k)
        if isinstance(v, str):
            params[k] = v.lower()
    _lower_str("software")
    _lower_str("fe_type")
    _lower_str("dec_int")
    _lower_str("dec_method")

    for k in ("rec_bb", "neutralize_only", "hmr", "bb_equil"):
        if k in params:
            params[k] = _coerce_yes_no(params[k])

    # Rocklin correction kept as string in your model; coerce too.
    if "rocklin_correction" in params:
        params["rocklin_correction"] = _coerce_yes_no(params["rocklin_correction"])

    return params


def read_yaml_config(path: str | Path) -> SimulationConfig:
    """
    Read a YAML config and return a validated SimulationConfig.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    SimulationConfig
        Validated configuration object.

    Notes
    -----
    - Supports both legacy flat keys (e.g., ``c_steps1``) and the newer
      nested dictionaries ``n_steps_dict`` / ``n_iter_dict`` in YAML.
    - YAML booleans (``true``/``false``) are accepted for yes/no flags
      and coerced to the expected ``'yes'/'no'`` strings.
    """
    path = Path(path)
    with path.open("r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError("Top-level YAML must be a mapping (key: value).")

    raw = _coerce_yaml_compat(raw)
    raw = _normalize_legacy_flat_keys(raw)

    cfg = SimulationConfig(**raw)
    logger.info("Loaded YAML config: {}", path)
    return cfg


def write_yaml_config(cfg: SimulationConfig, path: str | Path) -> None:
    """
    Write a SimulationConfig back to YAML.

    Parameters
    ----------
    cfg : SimulationConfig
        Configuration object to serialize.
    path : str or Path
        Output YAML file path.

    Notes
    -----
    - Uses ``model_dump()`` from Pydantic v2.
    - Keeps your internal ``n_steps_dict`` and ``n_iter_dict``.
    """
    path = Path(path)
    data = cfg.model_dump()
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    logger.info("Wrote YAML config: {}", path)