from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from batter.orchestrate import markers
from batter.orchestrate.markers import _spec_satisfied
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.pipeline import Pipeline
from batter.systems.core import SimSystem


def _make_system(root) -> SimSystem:
    return SimSystem(name=root.name, root=root, meta={"ligand": root.name})


def _register_example(root, phase: str) -> None:
    register_phase_state(
        root,
        phase,
        required=["marker.ok"],
        success=["marker.ok"],
        failure=["marker.fail"],
    )


def test_partition_children_by_status(tmp_path):
    phase = "example_phase"
    ok_root = tmp_path / "ok"
    fail_root = tmp_path / "fail"
    ok_root.mkdir()
    fail_root.mkdir()

    for root in (ok_root, fail_root):
        _register_example(root, phase)

    (ok_root / "marker.ok").write_text("done")
    (fail_root / "marker.fail").write_text("boom")

    systems = [_make_system(ok_root), _make_system(fail_root)]
    ok, bad = markers.partition_children_by_status(systems, phase)

    assert [s.name for s in ok] == ["ok"]
    assert [s.name for s in bad] == ["fail"]


def test_handle_phase_failures_prune_and_raise(tmp_path):
    phase = "example_phase"
    ok_root = tmp_path / "ok"
    fail_root = tmp_path / "fail"
    ok_root.mkdir()
    fail_root.mkdir()

    for root in (ok_root, fail_root):
        _register_example(root, phase)

    (ok_root / "marker.ok").write_text("done")
    (fail_root / "marker.fail").write_text("boom")

    systems = [_make_system(ok_root), _make_system(fail_root)]

    pruned = markers.handle_phase_failures(list(systems), phase, mode="prune")
    assert [s.name for s in pruned] == ["ok"]

    with pytest.raises(RuntimeError):
        markers.handle_phase_failures(list(systems), phase, mode="raise")


def test_filter_needing_phase_and_is_done(tmp_path):
    phase = "example_phase"
    done_root = tmp_path / "done"
    todo_root = tmp_path / "todo"
    done_root.mkdir()
    todo_root.mkdir()

    for root in (done_root, todo_root):
        _register_example(root, phase)

    (done_root / "marker.ok").write_text("done")

    done_sys = _make_system(done_root)
    todo_sys = _make_system(todo_root)

    assert markers.is_done(done_sys, phase) is True
    remaining = markers.filter_needing_phase([done_sys, todo_sys], phase)
    assert [s.name for s in remaining] == ["todo"]


@dataclass
class StubBackend:
    calls: List[str]

    def __init__(self):
        self.calls = []

    def run_parallel(self, pipeline, systems, description="", max_workers=None):
        self.calls.append(description or "run")


def test_run_phase_skipping_done_behavior(tmp_path):
    phase = "example_phase"

    todo_root = tmp_path / "todo"
    todo_root.mkdir()
    _register_example(todo_root, phase)
    todo_system = _make_system(todo_root)

    backend = StubBackend()
    pipeline = Pipeline([])

    skipped = markers.run_phase_skipping_done(pipeline, [todo_system], phase, backend)
    assert skipped is False
    assert backend.calls == [phase]

    done_root = tmp_path / "done"
    done_root.mkdir()
    _register_example(done_root, phase)
    (done_root / "marker.ok").write_text("done")
    done_system = _make_system(done_root)

    backend2 = StubBackend()
    skipped_done = markers.run_phase_skipping_done(pipeline, [done_system], phase, backend2)
    assert skipped_done is True
    assert backend2.calls == []


def test_run_phase_skipping_done_allows_prune_on_backend_error(tmp_path):
    phase = "example_phase"
    todo_root = tmp_path / "todo"
    todo_root.mkdir()
    _register_example(todo_root, phase)
    todo_system = _make_system(todo_root)
    pipeline = Pipeline([])

    class FailingBackend:
        def __init__(self):
            self.calls = 0

        def run_parallel(self, pipeline, systems, description="", max_workers=None):
            self.calls += 1
            raise RuntimeError("boom")

    backend = FailingBackend()
    skipped = markers.run_phase_skipping_done(
        pipeline,
        [todo_system],
        phase,
        backend,
        on_failure="prune",
    )
    assert skipped is False
    assert backend.calls == 1

    backend2 = FailingBackend()
    with pytest.raises(RuntimeError):
        markers.run_phase_skipping_done(
            pipeline,
            [todo_system],
            phase,
            backend2,
            on_failure="raise",
        )
    assert backend2.calls == 1


def test_spec_satisfied_writes_progress(tmp_path):
    run_root = tmp_path / "run"
    root = run_root / "simulations" / "LIG1"
    (run_root / "artifacts").mkdir(parents=True)
    (run_root / "simulations").mkdir(parents=True)
    target = root / "foo" / "FINISHED"
    target.parent.mkdir(parents=True)
    target.write_text("")
    spec = [["foo/FINISHED"]]

    assert _spec_satisfied(root, spec, "fe") is True

    progress = run_root / "artifacts" / "progress" / "fe" / "simulations__LIG1.csv"
    assert progress.exists()
    assert "foo/FINISHED" in progress.read_text()


def test_prepare_fe_progress_path(tmp_path):
    run_root = tmp_path / "run"
    root = run_root / "simulations" / "LIG1"
    (run_root / "artifacts").mkdir(parents=True)
    (run_root / "simulations").mkdir(parents=True)
    target = root / "foo" / "FINISHED"
    target.parent.mkdir(parents=True)
    target.write_text("")
    spec = [["foo/FINISHED"]]

    assert _spec_satisfied(root, spec, "prepare_fe") is True

    progress = (
        run_root
        / "artifacts"
        / "progress"
        / "prepare_fe"
        / "simulations__LIG1.csv"
    )
    assert progress.exists()
    assert "foo/FINISHED" in progress.read_text()
