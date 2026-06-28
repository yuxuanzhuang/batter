from __future__ import annotations

import importlib

import pytest

from batter.exec.local import LocalBackend
from batter.orchestrate.backend import register_local_handlers
from batter.pipeline.step import Step
from batter.systems.core import SimSystem


def test_register_local_handlers_defers_param_ligands_optional_import(
    monkeypatch,
    tmp_path,
) -> None:
    backend = LocalBackend()

    register_local_handlers(backend)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "batter.exec.handlers.param_ligands":
            raise ModuleNotFoundError("No module named 'gufe'", name="gufe")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="gufe"):
        backend.run(Step("param_ligands"), SimSystem("sys", tmp_path), {})
