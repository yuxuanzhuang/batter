"""
Utilities for locating bundled test fixtures.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DATA_DIR",
    "data_path",
    "TWO_CANDIDATE_SDF",
    "THREE_CANDIDATE_SDF",
]

DATA_DIR = Path(__file__).resolve().parent


def data_path(*parts: str) -> Path:
    """
    Return an absolute path inside the ``tests/data`` directory.

    Parameters
    ----------
    *parts : str
        Relative path components appended to the data root.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested fixture.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    candidate = DATA_DIR.joinpath(*parts)
    if not candidate.exists():
        raise FileNotFoundError(f"No such test data: {candidate}")
    return candidate


TWO_CANDIDATE_SDF = data_path("ligands", "2_candidates.sdf")
THREE_CANDIDATE_SDF = data_path("ligands", "3_candidates.sdf")
