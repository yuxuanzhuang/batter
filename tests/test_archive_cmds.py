from __future__ import annotations

import json
import tarfile
from pathlib import Path

from click.testing import CliRunner

from batter.cli.run import cli
from batter.runtime.archive import create_execution_archive, discover_execution_paths


def _tar_names(path: Path) -> set[str]:
    with tarfile.open(path) as tar:
        return set(tar.getnames())


def test_create_execution_archive_keeps_results_and_reproducible_inputs(
    tmp_path: Path,
) -> None:
    execution = tmp_path / "executions" / "rep1"
    (tmp_path / "results" / "rep1").mkdir(parents=True)
    (tmp_path / "results" / "index.csv").write_text("run_id,ligand\n")
    (tmp_path / "results" / "rep1" / "record.json").write_text("{}\n")
    (execution / "results" / "rep1" / "LIG").mkdir(parents=True)
    (execution / "results" / "index.csv").write_text("local_index\n")
    (execution / "results" / "rep1" / "LIG" / "record.json").write_text("{}\n")
    (execution / "simulations" / "LIG" / "equil" / "results").mkdir(parents=True)
    (execution / "simulations" / "LIG" / "equil" / "results" / "README.txt").write_text(
        "equil report\n"
    )
    (execution / "simulations" / "LIG" / "fe" / "z" / "z-1" / "results").mkdir(
        parents=True
    )
    (
        execution
        / "simulations"
        / "LIG"
        / "fe"
        / "z"
        / "z-1"
        / "results"
        / "summary.csv"
    ).write_text("window,result\n")
    (execution / "simulations" / "LIG" / "equil" / "md-01.nc").write_text(
        "trajectory should not be archived\n"
    )
    (execution / "simulations" / "LIG" / "inputs").mkdir(parents=True)
    (execution / "simulations" / "LIG" / "inputs" / "ligand.sdf").write_text(
        "ligand\n"
    )
    (execution / "simulations" / "LIG" / "params").mkdir(parents=True)
    (execution / "simulations" / "LIG" / "params" / "lig.json").write_text("{}\n")
    (execution / "artifacts" / "config").mkdir(parents=True)
    (execution / "artifacts" / "config" / "run_config.yaml").write_text("run: {}\n")
    (execution / "batter.run.log").write_text("log\n")

    archive = tmp_path / "archive.tar.gz"
    progress_updates: list[str] = []
    result = create_execution_archive([execution], archive, progress=progress_updates.append)

    assert result.archive_path == archive.resolve()
    assert progress_updates == list(result.members)
    assert progress_updates[0] == "batter_archive/archive_manifest.json"
    names = _tar_names(archive)
    assert "batter_archive/rep1/archive_manifest.json" not in names
    assert "batter_archive/archive_manifest.json" in names
    assert "batter_archive/results/index.csv" in names
    assert "batter_archive/results/rep1/record.json" in names
    assert "batter_archive/rep1/results/index.csv" in names
    assert "batter_archive/rep1/results/rep1/LIG/record.json" in names
    assert "batter_archive/rep1/simulations/LIG/equil/results/README.txt" in names
    assert (
        "batter_archive/rep1/simulations/LIG/fe/z/z-1/results/summary.csv" in names
    )
    assert "batter_archive/rep1/simulations/LIG/inputs/ligand.sdf" in names
    assert "batter_archive/rep1/simulations/LIG/params/lig.json" in names
    assert "batter_archive/rep1/artifacts/config/run_config.yaml" in names
    assert "batter_archive/rep1/batter.run.log" in names
    assert "batter_archive/rep1/simulations/LIG/equil/md-01.nc" not in names

    with tarfile.open(archive) as tar:
        manifest_file = tar.extractfile("batter_archive/archive_manifest.json")
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read().decode("utf-8"))
    assert manifest["executions"][0]["archive_prefix"] == "rep1"
    assert "results" in manifest["executions"][0]["results_dirs"]
    assert "results/index.csv" in manifest["executions"][0]["associated_results"]
    assert "results/rep1" in manifest["executions"][0]["associated_results"]
    assert "simulations/LIG/equil/results" in manifest["executions"][0]["results_dirs"]
    assert (
        "simulations/LIG/fe/z/z-1/results"
        in manifest["executions"][0]["results_dirs"]
    )


def test_discover_execution_paths_expands_execution_parent(tmp_path: Path) -> None:
    rep1 = tmp_path / "work" / "executions" / "rep1"
    rep2 = tmp_path / "work" / "executions" / "rep2"
    rep1.mkdir(parents=True)
    rep2.mkdir(parents=True)

    assert discover_execution_paths([tmp_path / "work" / "executions"]) == [
        rep1.resolve(),
        rep2.resolve(),
    ]


def test_archive_cli_accepts_multiple_executions_and_extra_inputs(tmp_path: Path) -> None:
    exec_root = tmp_path / "work" / "executions"
    for run_id in ("rep1", "rep2"):
        execution = exec_root / run_id
        (execution / "results").mkdir(parents=True)
        (execution / "results" / "index.csv").write_text(f"{run_id}\n")
    yaml_path = tmp_path / "mabfe_eq.yaml"
    yaml_path.write_text("protocol: abfe\n")
    archive = tmp_path / "bundle.tar"

    result = CliRunner().invoke(
        cli,
        [
            "archive",
            str(exec_root),
            "--include",
            str(yaml_path),
            "-o",
            str(archive),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Archived 2 execution(s)." in result.output
    assert "Archive entries written:" in result.output
    names = _tar_names(archive)
    assert "batter_archive/rep1/results/index.csv" in names
    assert "batter_archive/rep2/results/index.csv" in names
    assert "batter_archive/extra_inputs/mabfe_eq.yaml" in names


def test_archive_cli_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    execution = tmp_path / "executions" / "rep1"
    (execution / "results").mkdir(parents=True)
    archive = tmp_path / "archive.tar.gz"
    archive.write_text("existing\n")

    result = CliRunner().invoke(
        cli,
        ["archive", str(execution), "-o", str(archive)],
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
