from __future__ import annotations

from pathlib import Path

__all__ = [
    "DEC_FOLDER_DICT",
    "COMPONENTS_LAMBDA_DICT",
    "FEP_COMPONENTS",
    "COMPONENTS_FOLDER_DICT",
    "COMPONENTS_DICT",
    "components_under",
]

DEC_FOLDER_DICT = {
    "dd": "dd",
    "sdr": "sdr",
    "exchange": "sdr",
}

COMPONENTS_LAMBDA_DICT = {
    "v": "lambdas",
    "e": "lambdas",
    "w": "lambdas",
    "f": "lambdas",
    "x": "lambdas",
    "o": "lambdas",
    "z": "lambdas",
    "s": "lambdas",
    "y": "lambdas",
    "a": "attach_rest",
    "l": "attach_rest",
    "t": "attach_rest",
    "r": "attach_rest",
    "c": "attach_rest",
    "m": "attach_rest",
    "n": "attach_rest",
}

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())

COMPONENTS_FOLDER_DICT = {
    "v": "sdr",
    "e": "sdr",
    "w": "sdr",
    "f": "sdr",
    "x": "sdr",
    "o": "sdr",
    "z": "sdr",
    "s": "sdr",
    "y": "sdr",
    "a": "rest",
    "l": "rest",
    "t": "rest",
    "r": "rest",
    "c": "rest",
    "m": "sdr",
    "n": "rest",
}

COMPONENTS_DICT = {
    "rest": ["a", "l", "t", "c", "r", "n"],
    "dd": ["e", "v", "f", "w", "x", "o", "s", "z", "y", "m"],
}


def components_under(root: Path) -> list[str]:
    """Return FE component folder names present under ``root``."""
    fe_root = root / "fe"
    if not fe_root.exists():
        return []
    return sorted(
        [
            p.name
            for p in fe_root.iterdir()
            if p.is_dir() and p.name in FEP_COMPONENTS
        ]
    )
