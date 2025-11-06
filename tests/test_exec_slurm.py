from pathlib import Path

import pytest

from batter.exec.slurm import SlurmBackend
from batter.pipeline.step import Step
from batter.systems.core import SimSystem


def fake_submit(script_path: Path) -> str:
    assert script_path.exists()
    return "4242"


def test_slurm_backend_writes_script_and_returns_job(monkeypatch, tmp_path):
    backend = SlurmBackend()
    monkeypatch.setattr(SlurmBackend, "_submit", staticmethod(fake_submit))

    system = SimSystem(name="sys", root=tmp_path)
    step = Step(name="fe_prepare", payload={})

    params = {
        "payload": "echo running",
        "resources": {
            "time": "01:00:00",
            "cpus": 4,
            "partition": "debug",
        },
        "env": {"FOO": "bar"},
    }

    result = backend.run(step, system, params)

    script_path = tmp_path / "sbatch" / "fe_prepare.sh"
    assert script_path.exists()
    content = script_path.read_text()
    assert "#SBATCH --job-name=fe_prepare" in content
    assert "#SBATCH --time=01:00:00" in content
    assert "export FOO=bar" in content
    assert "echo running" in content

    assert result.job_ids == ["4242"]
    assert result.artifacts["script"] == script_path

    stdout = result.artifacts["stdout"]
    stderr = result.artifacts["stderr"]
    assert Path(stdout).parent == tmp_path / "logs"
    assert Path(stderr).parent == tmp_path / "logs"
