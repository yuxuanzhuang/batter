from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from batter.cli import param_cmds
from batter.cli.run import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_param_ligand_uses_standard_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: CliRunner,
) -> None:
    ligand_sdf = tmp_path / "example.sdf"
    ligand_sdf.write_text("test sdf\n")
    received: dict[str, object] = {}

    def fake_batch(ligand_paths, output_path, **kwargs):
        received["ligand_paths"] = ligand_paths
        received["output_path"] = output_path
        received.update(kwargs)
        return ["abc123def456"], {str(ligand_sdf): ("abc123def456", "C")}

    monkeypatch.setattr(param_cmds, "_batch_ligand_process", fake_batch)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["param-ligand", str(ligand_sdf)])

    assert result.exit_code == 0, result.output
    assert received == {
        "ligand_paths": {"example": str(ligand_sdf.resolve())},
        "output_path": (tmp_path / "ligand_param").resolve(),
        "retain_lig_prot": True,
        "ligand_ff": "openff-2.3.0",
        "charge_method": "openff-gnn-am1bcc-1.0.0.pt",
        "overwrite": False,
        "run_with_slurm": False,
        "on_failure": "raise",
    }
    assert str(tmp_path / "ligand_param" / "abc123def456") in result.output


def test_param_ligand_forwards_custom_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: CliRunner,
) -> None:
    ligand_sdf = tmp_path / "example.SDF"
    ligand_sdf.write_text("test sdf\n")
    output_dir = tmp_path / "custom_params"
    received: dict[str, object] = {}

    def fake_batch(ligand_paths, output_path, **kwargs):
        received["output_path"] = output_path
        received.update(kwargs)
        return ["fed654cba321"], {}

    monkeypatch.setattr(param_cmds, "_batch_ligand_process", fake_batch)

    result = runner.invoke(
        cli,
        [
            "param-ligand",
            str(ligand_sdf),
            "--output",
            str(output_dir),
            "--ligand-ff",
            "openff-2.3.0",
            "--charge-method",
            "openff-gnn-am1bcc-1.0.0.pt",
            "--no-retain-h",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["output_path"] == output_dir.resolve()
    assert received["ligand_ff"] == "openff-2.3.0"
    assert received["charge_method"] == "openff-gnn-am1bcc-1.0.0.pt"
    assert received["retain_lig_prot"] is False
    assert received["overwrite"] is True


def test_param_ligand_rejects_non_sdf(
    tmp_path: Path,
    runner: CliRunner,
) -> None:
    ligand_mol2 = tmp_path / "example.mol2"
    ligand_mol2.write_text("test mol2\n")

    result = runner.invoke(cli, ["param-ligand", str(ligand_mol2)])

    assert result.exit_code == 1
    assert "Ligand input must be an SDF file" in result.output


def test_param_ligand_reports_parameterization_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: CliRunner,
) -> None:
    ligand_sdf = tmp_path / "example.sdf"
    ligand_sdf.write_text("test sdf\n")

    def fail_batch(*args, **kwargs):
        raise RuntimeError("antechamber failed")

    monkeypatch.setattr(param_cmds, "_batch_ligand_process", fail_batch)

    result = runner.invoke(cli, ["param-ligand", str(ligand_sdf)])

    assert result.exit_code == 1
    assert "Error: antechamber failed" in result.output


def test_param_ligand_rejects_empty_backend_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: CliRunner,
) -> None:
    ligand_sdf = tmp_path / "example.sdf"
    ligand_sdf.write_text("test sdf\n")
    monkeypatch.setattr(
        param_cmds,
        "_batch_ligand_process",
        lambda *args, **kwargs: ([], {}),
    )

    result = runner.invoke(cli, ["param-ligand", str(ligand_sdf)])

    assert result.exit_code == 1
    assert "did not produce exactly one parameter set" in result.output


def test_param_ligand_is_registered(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "param-ligand" in result.output


def test_param_ligand_requires_input(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["param-ligand"])

    assert result.exit_code == 2
    assert "Missing argument 'LIGAND_SDF'" in result.output
