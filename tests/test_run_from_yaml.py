from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from batter.cli.run import cli as batter_cli


DATA_DIR = Path(__file__).parent / "data"
YAML_FILES = sorted(path for path in DATA_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_file", YAML_FILES)
def test_cli_run_dry_run(monkeypatch, yaml_file: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run_from_yaml(path, on_failure=None, system_overrides=None, run_overrides=None):
        calls["path"] = Path(path)
        calls["run_overrides"] = run_overrides or {}
        return None

    monkeypatch.setattr("batter.cli.run.run_from_yaml", fake_run_from_yaml)

    runner = CliRunner()
    result = runner.invoke(batter_cli, ["run", str(yaml_file), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert calls["path"] == yaml_file
    assert calls["run_overrides"].get("dry_run") is True
