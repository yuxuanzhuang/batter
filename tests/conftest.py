from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _chdir_to_repo_root() -> Iterator[None]:
    """
    Autouse fixture to run tests from the repo root.

    This keeps relative paths (e.g., ``tests/data/*.yaml``) stable regardless of
    how pytest is invoked.

    Yields
    ------
    None
        Context manager effect only.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    old = Path.cwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(old)
