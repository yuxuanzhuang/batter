from pathlib import Path

import pytest

from batter.orchestrate.run import _load_stored_ligand_names, _store_ligand_names


@pytest.mark.parametrize("mapping", [{}, {"LIG1": "original-name"}])
def test_store_and_load_ligand_names(tmp_path: Path, mapping: dict[str, str]) -> None:
    run_dir = tmp_path / "run"
    _store_ligand_names(run_dir, mapping)
    loaded = _load_stored_ligand_names(run_dir)
    assert loaded == mapping


def test_load_missing_return_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = _load_stored_ligand_names(run_dir)
    assert result == {}
