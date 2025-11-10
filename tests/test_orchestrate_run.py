from pathlib import Path

from batter.orchestrate.run import (
    _builder_info_for_protocol,
    _resolve_signature_conflict,
    _select_system_builder,
)
from batter.systems.mabfe import MABFEBuilder
from batter.systems.masfe import MASFEBuilder
import pytest


def test_builder_info_for_md_protocol():
    builder_cls, expected = _builder_info_for_protocol("md")
    assert builder_cls is MABFEBuilder
    assert expected == "MABFE"


def test_builder_info_for_asfe_protocol():
    builder_cls, expected = _builder_info_for_protocol("asfe")
    assert builder_cls is MASFEBuilder
    assert expected == "MASFE"


def test_builder_info_unknown_protocol():
    with pytest.raises(ValueError):
        _builder_info_for_protocol("rbfe")


def test_select_system_builder_rejects_mismatch():
    with pytest.raises(ValueError, match="incompatible"):
        _select_system_builder("asfe", "MABFE")


def test_resolve_signature_conflict_auto_requests_new(tmp_path):
    run_dir = tmp_path
    assert (
        _resolve_signature_conflict(
            "abc",
            "def",
            "auto",
            False,
            run_id="runX",
            run_dir=run_dir,
        )
        is False
    )


def test_resolve_signature_conflict_explicit_denied():
    with pytest.raises(RuntimeError):
        _resolve_signature_conflict(
            "abc",
            "def",
            "manual",
            False,
            run_id="runX",
            run_dir=Path("dummy"),
        )


def test_resolve_signature_conflict_allows_override(tmp_path):
    assert (
        _resolve_signature_conflict(
            "abc",
            "def",
            "manual",
            True,
            run_id="runX",
            run_dir=tmp_path,
        )
        is True
    )
