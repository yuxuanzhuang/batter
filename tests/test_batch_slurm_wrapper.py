from pathlib import Path

from batter.exec.handlers.batch import render_batch_slurm_script
from batter.exec.handlers.fe import _write_ligand_fe_batch_runner


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
    assert "#SBATCH" not in text  # header added later at submission
    assert "cd" in text
    assert "FOO=bar" in text
    assert "run-local.bash" in text


def test_write_fe_batch_runner(tmp_path):
    system_root = tmp_path / "simulations" / "lig1"
    comp_dir = system_root / "fe" / "z"
    (comp_dir / "z-1").mkdir(parents=True)
    (comp_dir / "z00").mkdir(parents=True)
    (comp_dir / "z00" / "run-local.bash").write_text("#!/bin/bash\n")
    batch_root = tmp_path / "batch_run"

    helper = _write_ligand_fe_batch_runner(
        system_root=system_root,
        helper_root=batch_root / "lig1",
        ligand="lig1",
        batch_gpus=2,
        gpus_per_task=1,
    )
    text = helper.read_text()
    assert "fe_lig1.FINISHED" in text
    assert "--gpus-per-task" in text
    assert "srun" in text
