from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from click.testing import CliRunner

from batter.cli import analysis_cmds
from batter.cli.run import cli
from batter.exec.handlers.equil_analysis import (
    _fallback_representative_frame_index,
    _infer_standalone_ligand_context,
    _representative_selection_needs_refresh,
    discover_equil_analysis_targets,
)
from batter.pipeline.step import ExecResult


def test_discover_equil_analysis_targets_accepts_execution_and_ligand_dir(
    tmp_path: Path,
) -> None:
    exec_dir = tmp_path / "executions" / "rep1"
    ligand_dir = exec_dir / "simulations" / "L1"
    (ligand_dir / "equil").mkdir(parents=True)

    assert discover_equil_analysis_targets(exec_dir) == [ligand_dir.resolve()]
    assert discover_equil_analysis_targets(ligand_dir) == [ligand_dir.resolve()]
    assert discover_equil_analysis_targets(ligand_dir / "equil") == [
        ligand_dir.resolve()
    ]


def test_fallback_representative_frame_uses_last_recorded_frame() -> None:
    sim_val = SimpleNamespace(results={"frame_indices": np.array([4, 8, 12])})
    universe = SimpleNamespace(trajectory=range(100))

    assert _fallback_representative_frame_index(
        universe=universe,
        sim_val=sim_val,
    ) == 12


def test_missing_assign_fallback_representative_needs_refresh(tmp_path: Path) -> None:
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir()
    (equil_dir / "disang.rest").write_text("&rst iat=1,2,3,4, &end #Lig_D\n")
    np.savez_compressed(
        equil_dir / "equilibration_analysis_results.npz",
        representative_frame_index=39,
        representative_selection_mode="last_frame_fallback",
        representative_selection_reason=f"{equil_dir / 'assign.in'} not found",
    )

    assert _representative_selection_needs_refresh(equil_dir)


def test_normal_last_frame_fallback_representative_is_kept(tmp_path: Path) -> None:
    equil_dir = tmp_path / "equil"
    equil_dir.mkdir()
    (equil_dir / "disang.rest").write_text("&rst iat=1,2,3,4, &end #Lig_D\n")
    np.savez_compressed(
        equil_dir / "equilibration_analysis_results.npz",
        representative_frame_index=39,
        representative_selection_mode="last_frame_fallback",
        representative_selection_reason="no ligand dihedral definitions found",
    )

    assert not _representative_selection_needs_refresh(equil_dir)


def test_standalone_ligand_context_ignores_equil_sidecar_json(tmp_path: Path) -> None:
    ligand_dir = tmp_path / "simulations" / "L1"
    equil_dir = ligand_dir / "equil"
    equil_dir.mkdir(parents=True)
    (equil_dir / "extra_conf_restraints.json").write_text("{}\n")
    (equil_dir / "hmn.json").write_text("{}\n")

    residue_name, ligand_label = _infer_standalone_ligand_context(
        ligand_dir,
        residue_name=None,
        ligand_label=None,
    )

    assert residue_name == "hmn"
    assert ligand_label == "L1"


def test_simulation_analysis_cli_dispatches_each_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_dir = tmp_path / "executions" / "rep1"
    ligand_a = exec_dir / "simulations" / "L1"
    ligand_b = exec_dir / "simulations" / "L2"
    (ligand_a / "equil").mkdir(parents=True)
    (ligand_b / "equil").mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_run(target, **kwargs):
        calls.append({"target": Path(target), **kwargs})
        return ExecResult(job_ids=[], artifacts={"representative_pdb": Path(target) / "equil" / "representative.pdb"})

    monkeypatch.setattr(analysis_cmds, "run_equil_analysis_for_simulation", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "simulation-analysis",
            str(exec_dir),
            "--ligand-resname",
            "hmn",
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["target"] for call in calls] == [ligand_a.resolve(), ligand_b.resolve()]
    assert all(call["residue_name"] == "hmn" for call in calls)
    assert all(call["force"] is True for call in calls)
    assert "Found 2 simulation target(s)." in result.output
