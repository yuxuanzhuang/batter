from pathlib import Path

from batter.exec.handlers.batch import render_batch_slurm_script
from batter.exec.handlers.equil import _write_equil_batch_runner


def test_render_batch_slurm_script(tmp_path):
    batch_root = tmp_path / "batch_run"
    target = tmp_path / "work"
    target.mkdir()
    script = render_batch_slurm_script(
        batch_root=batch_root,
        target_dir=target,
        run_script="run-local.bash",
        env={"FOO": "bar"},
        system_name="sys",
        stage="equil",
        pose="lig",
        header_root=None,
    )

    text = script.read_text()
    assert script.exists()
    assert "#SBATCH" in text
    assert "cd" in text
    assert "FOO=bar" in text
    assert "run-local.bash" in text


def test_write_equil_batch_runner(tmp_path):
    run_root = tmp_path / "exec"
    sim = run_root / "simulations" / "lig1" / "equil"
    sim.mkdir(parents=True)
    (sim / "run-local.bash").write_text("#!/bin/bash\ntouch FINISHED\n")
    batch_root = tmp_path / "batch_run"

    helper = _write_equil_batch_runner(run_root, batch_root)
    content = helper.read_text()
    assert "run-local.bash" in content
    assert "equil_all.FINISHED" in content
