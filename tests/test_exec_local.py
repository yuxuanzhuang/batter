import pytest

from batter.exec.local import LocalBackend
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem


def _dummy_handler(step: Step, system: SimSystem, params):
    return ExecResult(job_ids=[], artifacts={"name": system.name, "params": dict(params)})


def _failing_handler(step: Step, system: SimSystem, params):
    raise RuntimeError(f"boom-{system.name}")


def make_system(tmp_path, idx: int = 0) -> SimSystem:
    root = tmp_path / f"sys{idx}"
    root.mkdir()
    return SimSystem(name=f"sys{idx}", root=root)


def test_local_backend_run_invokes_registered_handler(tmp_path):
    backend = LocalBackend()
    backend.register("demo", _dummy_handler)

    step = Step(name="demo", payload={"value": 42})
    system = make_system(tmp_path)

    result = backend.run(step, system, step.params)
    assert result.artifacts["name"] == "sys0"
    assert result.artifacts["params"]["value"] == 42


def test_local_backend_run_parallel_success(tmp_path):
    backend = LocalBackend()
    backend.register("demo", _dummy_handler)

    steps = [Step(name="demo", payload={})]
    pipeline = Pipeline(steps)
    systems = [make_system(tmp_path, 0), make_system(tmp_path, 1)]

    results = backend.run_parallel(pipeline, systems, max_workers=2, prefer="threads")
    assert set(results.keys()) == {"sys0", "sys1"}
    assert all(r["demo"].artifacts["name"] in {"sys0", "sys1"} for r in results.values())


def test_local_backend_run_parallel_propagates_errors(tmp_path):
    backend = LocalBackend()
    backend.register("demo", _failing_handler)

    pipeline = Pipeline([Step(name="demo", payload={})])
    systems = [make_system(tmp_path, 0)]

    with pytest.raises(RuntimeError):
        backend.run_parallel(pipeline, systems, max_workers=2, prefer="threads")
