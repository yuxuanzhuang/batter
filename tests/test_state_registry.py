from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batter.orchestrate.state_registry import (
    get_phase_state,
    read_phase_states,
    register_phase_state,
)


def test_register_phase_state_roundtrip(tmp_path):
    root = tmp_path / "workdir"
    root.mkdir()

    state = register_phase_state(
        root,
        "custom_phase",
        required=["sentinel.ok", ["alt_a.flag", "alt_b.flag"]],
        success=None,
        failure="failed.flag",
    )

    assert state.required == [
        ["sentinel.ok"],
        ["alt_a.flag", "alt_b.flag"],
    ]
    assert state.success == []
    assert state.failure == [["failed.flag"]]

    registry_file = root / "artifacts" / "phase_state.json"
    assert registry_file.exists()

    loaded = read_phase_states(root)["custom_phase"]
    assert loaded == state


def test_get_phase_state_legacy_default(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()

    legacy = get_phase_state(root, "equil")
    assert ["equil/FINISHED"] in legacy.required
    assert legacy.failure == [["equil/FAILED"]]

    unknown = get_phase_state(root, "nonexistent-phase")
    assert unknown.required == []
    assert unknown.success == []
    assert unknown.failure == []
