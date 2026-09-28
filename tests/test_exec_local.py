import pytest

import batter.exec.local as local_mod
from batter.exec.local import LocalBackend, _effective_worker_cap, _slurm_task_limit
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem


def _dummy_handler(step: Step, system: SimSystem, params):
    return ExecResult(job_ids=[], artifacts={"name": system.name, "params": dict(params)})


def _failing_handler(step: Step, system: SimSystem, params):
    raise RuntimeError(f"boom-{system.name}")


def _unpicklable_failing_handler(step: Step, system: SimSystem, params):
    exc = RuntimeError(f"unpicklable-boom-{system.name}")
    exc.callback = lambda: None
    raise exc


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

    with pytest.raises(RuntimeError, match="boom-sys0"):
        backend.run_parallel(pipeline, systems, max_workers=2, prefer="threads")


def test_local_backend_run_parallel_error_message_names_each_failure(tmp_path):
    backend = LocalBackend()
    backend.register("demo", _failing_handler)

    pipeline = Pipeline([Step(name="demo", payload={})])
    systems = [make_system(tmp_path, 0), make_system(tmp_path, 1)]

    with pytest.raises(RuntimeError) as exc_info:
        backend.run_parallel(pipeline, systems, max_workers=2, prefer="threads")

    message = str(exc_info.value)
    assert "sys0: RuntimeError: boom-sys0" in message
    assert "sys1: RuntimeError: boom-sys1" in message


def test_local_backend_run_parallel_process_errors_are_picklable(tmp_path):
    backend = LocalBackend()
    backend.register("demo", _unpicklable_failing_handler)

    pipeline = Pipeline([Step(name="demo", payload={})])
    systems = [make_system(tmp_path, 0)]

    with pytest.raises(RuntimeError) as exc_info:
        backend.run_parallel(pipeline, systems, max_workers=2)

    assert "sys0: RuntimeError: unpicklable-boom-sys0" in str(exc_info.value)


def test_slurm_task_limit_is_absent_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_NTASKS", raising=False)

    assert _slurm_task_limit() is None


@pytest.mark.parametrize("value", ["", "invalid", "0", "-2"])
def test_slurm_task_limit_ignores_invalid_values(monkeypatch, value):
    monkeypatch.setenv("SLURM_NTASKS", value)

    assert _slurm_task_limit() is None


def test_effective_worker_cap_respects_slurm_tasks(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "4")

    assert _effective_worker_cap(8, 87) == 4


def test_effective_worker_cap_does_not_raise_requested_limit(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "8")

    assert _effective_worker_cap(3, 87) == 3


def test_prepare_fe_recycles_workers_after_each_ligand_batch(monkeypatch, tmp_path):
    backend = LocalBackend()
    backend.register("demo", _dummy_handler)
    pipeline = Pipeline([Step(name="demo", payload={})])
    systems = [make_system(tmp_path, idx) for idx in range(5)]
    batch_sizes = []
    recycle_calls = []

    class InlineParallel:
        def __init__(self, **kwargs):
            assert kwargs["n_jobs"] == 2

        def __call__(self, calls):
            calls = list(calls)
            batch_sizes.append(len(calls))
            return [func(*args, **kwargs) for func, args, kwargs in calls]

    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setattr(local_mod, "Parallel", InlineParallel)
    monkeypatch.setattr(
        local_mod,
        "_shutdown_reusable_process_pool",
        lambda: recycle_calls.append(True),
    )

    results = backend.run_parallel(
        pipeline,
        systems,
        max_workers=2,
        description="prepare_fe",
    )

    assert batch_sizes == [2, 2, 1]
    assert len(recycle_calls) == 4  # before preparation and after every batch
    assert set(results) == {system.name for system in systems}


def test_regular_phase_keeps_single_parallel_pool(monkeypatch, tmp_path):
    backend = LocalBackend()
    backend.register("demo", _dummy_handler)
    pipeline = Pipeline([Step(name="demo", payload={})])
    systems = [make_system(tmp_path, idx) for idx in range(5)]
    batch_sizes = []

    class InlineParallel:
        def __init__(self, **kwargs):
            assert kwargs["n_jobs"] == 2

        def __call__(self, calls):
            calls = list(calls)
            batch_sizes.append(len(calls))
            return [func(*args, **kwargs) for func, args, kwargs in calls]

    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setattr(local_mod, "Parallel", InlineParallel)
    monkeypatch.setattr(
        local_mod,
        "_shutdown_reusable_process_pool",
        lambda: pytest.fail("regular phases must not recycle the process pool"),
    )

    backend.run_parallel(
        pipeline,
        systems,
        max_workers=2,
        description="prepare_equil",
    )

    assert batch_sizes == [5]
