from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from batter.config.simulation import SimulationConfig
from batter.orchestrate.run import save_fe_records
from batter.runtime.fe_repo import FEResultsRepository
from batter.runtime.portable import ArtifactStore
from batter.systems.core import SimSystem, SystemMeta
from batter.orchestrate import run as run_mod
from batter.orchestrate import run_support as rs


def _make_sim_cfg() -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "system_name": "sys",
            "fe_type": "rest",
            "dec_int": "mbar",
            "components": ["z"],
            "component_lambdas": {"z": [0.0, 1.0]},
            "lambdas": [0.0, 1.0],
            "temperature": 300.0,
            "analysis_fe_range": (0, -1),
            "buffer_x": 15.0,
            "buffer_y": 15.0,
            "buffer_z": 15.0,
        }
    )


@pytest.mark.parametrize("has_results", [False])
def test_save_fe_records_failure(tmp_path: Path, has_results: bool) -> None:
    run_dir = tmp_path / "run1"
    child_root = run_dir / "simulations" / "lig1"
    (child_root / "fe" / "Results").mkdir(parents=True, exist_ok=True)

    sim_cfg = _make_sim_cfg()
    child = SimSystem(
        name="sys:lig1:run1",
        root=child_root,
        meta=SystemMeta(ligand="lig1", residue_name="lig1"),
    )

    store = ArtifactStore(run_dir)
    repo = FEResultsRepository(store)

    failures = save_fe_records(
        run_dir=run_dir,
        run_id="run1",
        children_all=[child],
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol="abfe",
    )

    assert failures
    df = pd.read_csv(run_dir / "results" / "index.csv")
    row = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")].iloc[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "no_totals_found"
    failure_json = run_dir / "results" / "run1" / "lig1" / "failure.json"
    assert failure_json.exists()


def test_compute_run_signature_excludes_run_section(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
run:
  output_folder: out
create:
  system_name: sys
fe_sim: {}
protocol: abfe
"""
    )
    sig, payload = run_mod._compute_run_signature(yaml_path, {"override": 1})
    assert isinstance(sig, str) and len(sig) == 64
    assert "run" not in payload["config"]
    assert set(payload["config"].keys()) <= {"create", "fe_sim", "fe"}
    assert payload["run_overrides"] == {}


def test_stored_payload_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "exec"
    path = run_mod._payload_path(run_dir)
    payload = {"config": {"a": 1}, "run_overrides": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    assert run_mod._stored_payload(run_dir) == payload


def test_resolve_signature_conflict_reports_diffs(tmp_path: Path, caplog) -> None:
    stored_payload = {"config": {"a": 1}}
    current_payload = {"config": {"a": 2}}
    keep = run_mod._resolve_signature_conflict(
        "aaa",
        "bbb",
        requested_run_id="auto",
        allow_run_id_mismatch=False,
        run_id="rid",
        run_dir=tmp_path,
        stored_payload=stored_payload,
        current_payload=current_payload,
    )
    assert keep is False


def test_normalize_for_hash_strips_output_folder_and_paths(tmp_path: Path) -> None:
    payload = {
        "create": {"output_folder": "/tmp/out", "protein": tmp_path / "pdb.pdb"},
        "extra": [Path("/a/b"), {"c": "d"}],
    }
    normalized = rs.normalize_for_hash(payload)
    assert "output_folder" not in normalized["create"]
    assert normalized["create"]["protein"] == str(tmp_path / "pdb.pdb")
    assert normalized["extra"][0] == "/a/b"


def test_resolve_signature_conflict_allows_mismatch_when_flag(tmp_path: Path, caplog) -> None:
    keep = rs.resolve_signature_conflict(
        stored_sig="old",
        config_signature="new",
        requested_run_id="run1",
        allow_run_id_mismatch=True,
        run_id="run1",
        run_dir=tmp_path,
        stored_payload={"config": {"x": 1}},
        current_payload={"config": {"x": 2}},
    )
    assert keep is True


def test_resolve_signature_conflict_raises_on_mismatch(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        rs.resolve_signature_conflict(
            stored_sig="old",
            config_signature="new",
            requested_run_id="run1",
            allow_run_id_mismatch=False,
            run_id="run1",
            run_dir=tmp_path,
            stored_payload={"config": {"y": 1}},
            current_payload={"config": {"y": 2}},
        )


def test_select_system_builder_validates_system_type() -> None:
    builder = rs.select_system_builder("abfe", system_type=None)
    assert builder is not None
    with pytest.raises(ValueError):
        rs.select_system_builder("abfe", system_type="MASFE")


def test_select_run_id_reuses_latest(tmp_path: Path) -> None:
    exec_dir = tmp_path / "executions"
    old = exec_dir / "old"
    new = exec_dir / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    run_id, run_dir = rs.select_run_id(tmp_path, "abfe", "sys", requested=None)
    assert run_id == "new"
    assert run_dir == new


def _dummy_smtp(sent: dict[str, str | list[str]]):
    class DummySMTP:
        def __init__(self, host: str) -> None:
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def sendmail(self, sender: str, recipients: list[str], message: str) -> None:
            sent["sender"] = sender
            sent["recipients"] = recipients
            sent["message"] = message

    return DummySMTP


def _make_rc(tmp_path: Path, email_sender: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        protocol="abfe",
        create=SimpleNamespace(system_name="sys"),
        run=SimpleNamespace(
            email_on_completion="dest@example.com",
            email_sender=email_sender,
            output_folder=tmp_path,
        ),
    )


def test_notify_run_completion_prefers_config_sender(tmp_path: Path, monkeypatch) -> None:
    sent: dict[str, str | list[str]] = {}

    monkeypatch.setenv(run_mod.SENDER_ENV_VAR, "env@example.com")
    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender="config@example.com")

    run_mod._notify_run_completion(rc, "run1", tmp_path, [])

    assert sent["sender"] == "config@example.com"
    assert sent["recipients"] == ["dest@example.com"]
    assert "From: batter <config@example.com>" in sent["message"]


def test_notify_run_completion_uses_env_sender_when_set(tmp_path: Path, monkeypatch) -> None:
    sent: dict[str, str | list[str]] = {}
    monkeypatch.setenv(run_mod.SENDER_ENV_VAR, "env@example.com")
    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender=None)

    run_mod._notify_run_completion(rc, "run1", tmp_path, [])

    assert sent["sender"] == "env@example.com"
    assert "From: batter <env@example.com>" in sent["message"]


def test_notify_run_completion_warns_when_defaulting_sender(tmp_path: Path, monkeypatch) -> None:
    sent: dict[str, str | list[str]] = {}
    warnings: list[str] = []

    monkeypatch.delenv(run_mod.SENDER_ENV_VAR, raising=False)
    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))
    monkeypatch.setattr(run_mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))

    rc = _make_rc(tmp_path, email_sender=None)

    run_mod._notify_run_completion(rc, "run1", tmp_path, [])

    assert sent["sender"] == run_mod.DEFAULT_SENDER
    assert any("defaulting sender email" in w for w in warnings)
