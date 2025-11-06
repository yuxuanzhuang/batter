from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")


def coerce_yes_no(value: Any) -> str | None:
    """
    Normalize boolean-like values into ``\"yes\"`` or ``\"no\"``.

    Parameters
    ----------
    value :
        Input flag provided by the user. Supported types include ``bool``, numeric
        scalars, or strings such as ``\"true\"`` and ``\"0\"``.

    Returns
    -------
    str or None
        ``\"yes\"`` or ``\"no\"`` when the flag can be interpreted. ``None`` is
        returned unchanged to preserve optional semantics.

    Raises
    ------
    ValueError
        If the value cannot be coerced into a boolean switch.
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
    Convert a ligand identifier into a filesystem-safe token.

    Parameters
    ----------
    name : str
        Original ligand identifier, often derived from filenames or keys.

    Returns
    -------
    str
        Uppercase alphanumeric token with unsafe characters replaced by underscores.
    """
    cleaned = _SANITIZE_RE.sub("_", name.strip())
    return cleaned.strip("_").upper()


def normalize_optional_path(value: Any) -> Path | None:
    """
    Resolve optional path-like values into :class:`pathlib.Path` objects.

    Parameters
    ----------
    value :
        Path candidate that may be ``None`` or an empty string. Strings may
        contain environment variables or ``~``.

    Returns
    -------
    pathlib.Path or None
        Expanded path when provided; ``None`` if the value is empty.
    """
    if value in (None, ""):
        return None
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


def expand_env_vars(data: Any, *, base_dir: Path | None = None) -> Any:
    """
    Recursively expand environment variables in a YAML-derived structure.

    Parameters
    ----------
    data :
        Parsed YAML content to normalise.
    base_dir : Path, optional
        Base directory for resolving relative (``./``) paths.

    Returns
    -------
    Any
        Structure with string values expanded.
    """
    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            expanded = os.path.expandvars(os.path.expanduser(value))
            if base_dir is not None and not os.path.isabs(expanded):
                return str((base_dir / expanded).resolve())
            return expanded
        if isinstance(value, Mapping):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_expand(v) for v in value]
        return value

    return _expand(data)
