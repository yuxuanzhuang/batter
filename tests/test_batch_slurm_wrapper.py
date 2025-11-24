from pathlib import Path

from batter.exec.handlers.batch import render_batch_slurm_script


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
