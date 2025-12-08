from __future__ import annotations

from pathlib import Path
import subprocess


def _run_check_min_energy(output_file: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = (
        f"source '{repo_root}/batter/_internal/templates/run_files_orig/check_run.bash' "
        f"&& check_min_energy '{output_file}' -1000"
    )
    return subprocess.run(
        ["bash", "-lc", cmd],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def test_check_min_energy_prefers_eamber():
    data_dir = Path(__file__).resolve().parent / "data" / "md_output"
    result = _run_check_min_energy(data_dir / "mini.out")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "EAMBER energy: -64851.9795 kcal/mol" in result.stdout


def test_check_min_energy_falls_back_to_nstep_energy():
    data_dir = Path(__file__).resolve().parent / "data" / "md_output"
    result = _run_check_min_energy(data_dir / "mini.without.eamber.out")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ENERGY (NSTEP table): -64852.0000 kcal/mol" in result.stdout
