from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from loguru import logger

from batter.exec.slurm_mgr import (
    SlurmJobManager,
    SlurmJobSpec,
    _atomic_append_jsonl_unique,
    _format_workdir_label,
)


def test_atomic_append_jsonl_unique(tmp_path):
    path = tmp_path / "registry.jsonl"
    rec = {"workdir": "/tmp/job1"}

    _atomic_append_jsonl_unique(path, rec)
    _atomic_append_jsonl_unique(path, rec)  # duplicate should no-op

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_slurm_job_manager_status(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    script.write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(registry_file=tmp_path / "queue.jsonl")
    manager.add(spec)

    # registry file should have exactly one entry
    manager.add(spec)
    lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_registry_filters_by_stage(tmp_path):
    wd_eq = tmp_path / "eq_job"
    wd_fe = tmp_path / "fe_job"
    wd_eq.mkdir()
    wd_fe.mkdir()

    manager = SlurmJobManager(registry_file=tmp_path / "queue.jsonl")
    manager.set_stage("equil")

    manager.add(SlurmJobSpec(workdir=wd_eq, stage="equil"))
    manager.add(SlurmJobSpec(workdir=wd_fe, stage="fe"))

    jobs_equil = manager.jobs()
    assert {j.workdir for j in jobs_equil} == {wd_eq}

    manager.set_stage("fe")
    jobs_fe = manager.jobs()
    assert {j.workdir for j in jobs_fe} == {wd_fe}



def test_submission_failure_raises(monkeypatch, tmp_path):
    workdir = tmp_path / "fail"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    script.write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=0,
        submit_retry_limit=3,
        submit_retry_delay_s=0.0,
    )

    attempts = {"count": 0}

    def fail_submit_once(spec: SlurmJobSpec) -> str:
        attempts["count"] += 1
        raise RuntimeError("QOS limit")

    monkeypatch.setattr(manager, "_submit_once", fail_submit_once)

    with pytest.raises(RuntimeError, match="after 4 attempt"):
        manager._wait_loop([spec])

    assert attempts["count"] == 4


def test_submit_waits_for_slot_before_each_submission(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, max_active_jobs=1)

    wait_calls = {"count": 0}

    def fake_wait_for_slot(*args, **kwargs):
        wait_calls["count"] += 1

    monkeypatch.setattr(manager, "wait_for_slot", fake_wait_for_slot)
    monkeypatch.setattr(manager, "_submit_once", lambda spec: "42")

    assert manager._submit(spec) == "42"
    assert manager._submit(spec) == "42"
    assert wait_calls["count"] == 2


def test_submission_rate_limit_uses_longer_backoff(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        submit_retry_limit=2,
        submit_retry_delay_s=60.0,
    )

    sleeps: list[float] = []

    monkeypatch.setattr(manager, "wait_for_slot", lambda *args, **kwargs: None)
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))

    def fail_submit_once(spec: SlurmJobSpec) -> str:
        raise RuntimeError("sbatch: error: Reached jobs per hour limit")

    monkeypatch.setattr(manager, "_submit_once", fail_submit_once)

    with pytest.raises(RuntimeError, match="after 3 attempt"):
        manager._submit(spec)

    assert sleeps == [900.0, 1800.0]


def test_submit_rebuilds_script_with_header(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#SBATCH -J old\nBODY\n")
    header_root = tmp_path / "headers"
    header_root.mkdir()
    (header_root / "SLURMM-Am.header").write_text("#HEADER\n")

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        header_name="SLURMM-Am.header",
        header_root=header_root,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, header_root=header_root)

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        class Dummy:
            returncode = 0
            stdout = "Submitted batch job 99"
            stderr = ""

        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    jobid = manager._submit_once(spec)
    assert jobid == "99"
    script_txt = (workdir / "SLURMM-run").read_text()
    assert script_txt.startswith("#HEADER")
    assert "BODY" in script_txt
    assert "SBATCH -J old" not in script_txt


def test_submit_rebuild_normalizes_stale_default_gpu_constraint(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("BODY\n")
    header_root = tmp_path / "headers"
    header_root.mkdir()
    (header_root / "SLURMM-Am.header").write_text(
        "#!/bin/bash\n#SBATCH -C \"GPU_GEN:AMP|GPU_GEN:PSC\"\n"
    )

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        header_name="SLURMM-Am.header",
        header_root=header_root,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, header_root=header_root)

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        class Dummy:
            returncode = 0
            stdout = "Submitted batch job 99"
            stderr = ""

        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    manager._submit_once(spec)
    script_txt = (workdir / "SLURMM-run").read_text()
    assert '#SBATCH -C "GPU_GEN:AMP"' in script_txt
    assert "GPU_GEN:PSC" not in script_txt


def test_submit_rebuild_does_not_duplicate_header_on_repeat(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n#SBATCH -J old\nBODY\n")
    header_root = tmp_path / "headers"
    header_root.mkdir()
    header_text = (
        "#!/bin/bash\n"
        "# SYSTEMNAME, STAGE, POSE are placeholders to be replaced when generating the script\n"
        "source /path/to/amber.sh\n"
    )
    (header_root / "SLURMM-Am.header").write_text(header_text)

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        header_name="SLURMM-Am.header",
        header_root=header_root,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, header_root=header_root)

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        class Dummy:
            returncode = 0
            stdout = "Submitted batch job 99"
            stderr = ""

        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    manager._submit_once(spec)
    manager._submit_once(spec)

    script_txt = (workdir / "SLURMM-run").read_text()
    assert script_txt.count("source /path/to/amber.sh") == 1
    assert script_txt.count("SYSTEMNAME, STAGE, POSE") == 1
    assert "BODY" in script_txt


def test_submit_rebuild_prefers_newer_body_only_script_over_stale_sidecar(
    monkeypatch,
    tmp_path,
):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    script = workdir / "SLURMM-run"
    sidecar = workdir / "SLURMM-run.body"
    script.write_text("NEW_BODY\n")
    sidecar.write_text("OLD_BODY\n")
    os.utime(sidecar, (1, 1))
    os.utime(script, (2, 2))
    header_root = tmp_path / "headers"
    header_root.mkdir()
    (header_root / "SLURMM-Am.header").write_text("#HEADER\n")

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel="SLURMM-run",
        header_name="SLURMM-Am.header",
        header_root=header_root,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0, header_root=header_root)

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        class Dummy:
            returncode = 0
            stdout = "Submitted batch job 99"
            stderr = ""

        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    manager._submit_once(spec)

    script_txt = script.read_text()
    assert script_txt.startswith("#HEADER")
    assert "NEW_BODY" in script_txt
    assert "OLD_BODY" not in script_txt
    assert sidecar.read_text() == "NEW_BODY\n"


def test_submit_uses_submit_dir(monkeypatch, tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    submit_dir = tmp_path / "batch"
    submit_dir.mkdir()
    script = submit_dir / "batch.sh"
    script.write_text("#!/bin/bash\necho hi\n")

    spec = SlurmJobSpec(
        workdir=workdir,
        script_rel=script.name,
        batch_script=script,
        submit_dir=submit_dir,
    )
    manager = SlurmJobManager(registry_file=None, poll_s=0.0)

    calls = {}

    class Dummy:
        returncode = 0
        stdout = "Submitted batch job 42"
        stderr = ""

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return Dummy()

    monkeypatch.setattr("subprocess.run", fake_run)

    jobid = manager._submit_once(spec)
    assert jobid == "42"
    assert calls["cwd"] == submit_dir
    assert script.name in calls["cmd"]


def test_format_workdir_label_prefers_ligand_stage_window_suffix(tmp_path: Path) -> None:
    workdir = tmp_path / "executions" / "rep1" / "simulations" / "G1I" / "fe" / "x" / "x01"
    workdir.mkdir(parents=True)

    assert _format_workdir_label(workdir) == "G1I/fe/x/x01"


def test_wait_loop_warning_uses_expanded_workdir_label(monkeypatch, tmp_path: Path) -> None:
    workdir = tmp_path / "executions" / "rep1" / "simulations" / "G1I" / "fe" / "x" / "x01"
    workdir.mkdir(parents=True)
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n")
    (workdir / "JOBID").write_text("21949383\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=3,
    )

    sentinel_calls = {"count": 0}

    def fake_sentinel_done(_spec: SlurmJobSpec):
        sentinel_calls["count"] += 1
        if sentinel_calls["count"] >= 3:
            return True, "FINISHED"
        return False, None

    monkeypatch.setattr(manager, "_sentinel_done", fake_sentinel_done)
    monkeypatch.setattr(manager, "_submit", lambda _spec: "21949384")
    monkeypatch.setattr("batter.exec.slurm_mgr._states_from_squeue", lambda _jobids: {})
    monkeypatch.setattr(
        "batter.exec.slurm_mgr._states_from_sacct",
        lambda jobids: {jid: "FAILED" for jid in jobids},
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(str(msg)), format="{message}")
    try:
        manager._wait_loop([spec])
    finally:
        logger.remove(sink_id)

    joined = "".join(messages)
    assert (
        "[SLURM] G1I/fe/x/x01: job 21949383 state=FAILED; resubmitting (1/3)"
        in joined
    )


def test_wait_loop_preempted_resubmits_without_retry_budget(
    monkeypatch, tmp_path: Path
) -> None:
    workdir = tmp_path / "executions" / "rep1" / "simulations" / "G1I" / "fe" / "x" / "x01"
    workdir.mkdir(parents=True)
    (workdir / "SLURMM-run").write_text("#!/bin/bash\n")
    (workdir / "JOBID").write_text("21949383\n")

    spec = SlurmJobSpec(workdir=workdir)
    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=0,
    )

    sentinel_calls = {"count": 0}
    submit_calls: list[Path] = []

    def fake_sentinel_done(_spec: SlurmJobSpec):
        sentinel_calls["count"] += 1
        if sentinel_calls["count"] >= 3:
            return True, "FINISHED"
        return False, None

    def fake_submit(_spec: SlurmJobSpec) -> str:
        submit_calls.append(_spec.workdir)
        (workdir / "JOBID").write_text("21949384\n")
        return "21949384"

    monkeypatch.setattr(manager, "_sentinel_done", fake_sentinel_done)
    monkeypatch.setattr(manager, "_submit", fake_submit)
    monkeypatch.setattr("batter.exec.slurm_mgr._states_from_squeue", lambda _jobids: {})
    monkeypatch.setattr(
        "batter.exec.slurm_mgr._states_from_sacct",
        lambda jobids: {jid: "PREEMPTED" for jid in jobids},
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    manager._wait_loop([spec])

    assert submit_calls == [workdir]
    assert not (workdir / "FAILED").exists()
    assert manager._retries.get(workdir, 0) == 0


def test_wait_loop_progress_starts_from_existing_sentinels(
    monkeypatch, tmp_path: Path
) -> None:
    finished_dir = tmp_path / "done"
    running_dir = tmp_path / "running"
    finished_dir.mkdir()
    running_dir.mkdir()
    (finished_dir / "FINISHED").write_text("FINISHED\n")
    (running_dir / "JOBID").write_text("1\n")

    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, iterable=None, total=None, initial=0, desc=None, **_kwargs):
            self.iterable = iterable
            self.total = total
            self.initial = initial
            self.desc = desc
            self.n = initial
            self.postfix = {}
            FakeTqdm.instances.append(self)

        def __iter__(self):
            return iter(self.iterable or [])

        def set_postfix(self, values):
            self.postfix = dict(values)

        def refresh(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=FakeTqdm))

    polls = {"count": 0}

    def fake_squeue(jobids):
        polls["count"] += 1
        if polls["count"] == 1:
            return {"1": "RUNNING"}
        (running_dir / "FINISHED").write_text("FINISHED\n")
        return {}

    monkeypatch.setattr("batter.exec.slurm_mgr._states_from_squeue", fake_squeue)
    monkeypatch.setattr("batter.exec.slurm_mgr._states_from_sacct", lambda _jobids: {})
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    manager = SlurmJobManager(
        registry_file=None,
        poll_s=0.0,
        resubmit_backoff_s=0.0,
        max_retries=0,
    )

    manager._wait_loop(
        [
            SlurmJobSpec(workdir=finished_dir),
            SlurmJobSpec(workdir=running_dir),
        ]
    )

    progress = next(inst for inst in FakeTqdm.instances if inst.desc == "SLURM jobs")
    assert progress.total == 2
    assert progress.initial == 1
    assert progress.n == 2
    assert progress.postfix["pending"] == 0


def test_wait_loop_failed_progress_is_red_and_logs_failed_folder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    failed_dir = (
        tmp_path
        / "executions"
        / "rep1"
        / "simulations"
        / "BAD"
        / "fe"
        / "z"
        / "z-1"
    )
    running_dir = (
        tmp_path
        / "executions"
        / "rep1"
        / "simulations"
        / "RUN"
        / "fe"
        / "z"
        / "z-1"
    )
    failed_dir.mkdir(parents=True)
    running_dir.mkdir(parents=True)
    (failed_dir / "FAILED").write_text("FAILED\n")
    (running_dir / "JOBID").write_text("1\n")

    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, iterable=None, total=None, initial=0, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.initial = initial
            self.desc = desc
            self.n = initial
            self.postfix = {}
            self.colour = kwargs.get("colour")
            FakeTqdm.instances.append(self)

        def __iter__(self):
            return iter(self.iterable or [])

        def set_postfix(self, values):
            self.postfix = dict(values)

        def refresh(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=FakeTqdm))

    def fake_squeue(_jobids):
        (running_dir / "FINISHED").write_text("FINISHED\n")
        return {}

    monkeypatch.setattr("batter.exec.slurm_mgr._states_from_squeue", fake_squeue)
    monkeypatch.setattr("batter.exec.slurm_mgr._states_from_sacct", lambda _jobids: {})
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(str(msg)), format="{message}")
    try:
        manager = SlurmJobManager(
            registry_file=None,
            poll_s=0.0,
            resubmit_backoff_s=0.0,
            max_retries=0,
        )
        manager._wait_loop(
            [
                SlurmJobSpec(workdir=failed_dir),
                SlurmJobSpec(workdir=running_dir),
            ]
        )
    finally:
        logger.remove(sink_id)

    progress = next(inst for inst in FakeTqdm.instances if inst.desc == "SLURM jobs")
    assert progress.colour == "red"
    assert progress.postfix["failed"] == 1
    joined = "".join(messages)
    assert "[SLURM] folder failed:" in joined
    assert str(failed_dir) in joined
    assert "BAD/fe/z/z-1" in joined
