from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from batter.orchestrate.run import _materialize_extra_conf_restraints


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "executions" / "rep1"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_materialize_extra_conf_restraints_copies_once(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    src = yaml_dir / "rest.json"
    src.write_text('{"atoms": []}')

    # First copy
    stored = _materialize_extra_conf_restraints(src, run_dir, yaml_dir)
    assert stored is not None
    assert stored.exists()
    assert stored.parent == run_dir / "artifacts" / "config"
    assert stored.read_text() == src.read_text()

    # Modify source and ensure existing stored copy is reused (not overwritten)
    src.write_text('{"atoms": [1]}')
    stored2 = _materialize_extra_conf_restraints(src, run_dir, yaml_dir)
    assert stored2 == stored
    assert stored2.read_text() != src.read_text()


def test_materialize_extra_conf_restraints_warns_when_missing(tmp_path: Path, capsys):
    run_dir = _make_run_dir(tmp_path)
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    missing = yaml_dir / "missing.json"

    stored = _materialize_extra_conf_restraints(missing, run_dir, yaml_dir)

    assert stored is None


def test_materialize_extra_conf_restraints_handles_relative(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    src = yaml_dir / "rel_rest.json"
    src.write_text("[]")

    stored = _materialize_extra_conf_restraints("rel_rest.json", run_dir, yaml_dir)
    assert stored is not None
    assert stored.exists()
    assert stored.read_text() == "[]"
