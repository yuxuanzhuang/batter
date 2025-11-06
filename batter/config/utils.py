from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")


def coerce_yes_no(value: Any) -> str | None:
    """
    Normalise truthy flags into ``\"yes\"``/``\"no\"`` strings accepted by legacy code.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if value else "no"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "no"}:
            return text
        if text in {"true", "t", "1"}:
            return "yes"
        if text in {"false", "f", "0"}:
            return "no"
    raise ValueError(f"Expected yes/no (or boolean), got {value!r}")


def sanitize_ligand_name(name: str) -> str:
    """
    Convert arbitrary ligand identifiers into filesystem-safe uppercase tokens.
    """
    cleaned = _SANITIZE_RE.sub("_", name.strip())
    return cleaned.strip("_").upper()


def normalize_optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


def expand_env_vars(data: Any, *, base_dir: Path | None = None) -> Any:
    """
    Recursively expand environment variables in a YAML-derived structure.
    """
    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            expanded = os.path.expandvars(os.path.expanduser(value))
            if base_dir is not None and expanded.startswith("./"):
                return str((base_dir / expanded[2:]).resolve())
            return expanded
        if isinstance(value, Mapping):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_expand(v) for v in value]
        return value

    return _expand(data)
