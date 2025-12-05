from __future__ import annotations

from pathlib import Path
import shutil
import os
import subprocess
import sys
from typing import Iterable, List, Tuple

import pytest

# --- Project imports (adjust if your import path differs) ---
from batter.config.run import RunConfig
from batter.config import load_run_config  # if you expose it here
from batter.config.simulation import SimulationConfig  # where your class lives
from batter.orchestrate import run as run_mod
from batter.orchestrate.run import run_from_yaml
from batter.pipeline.step import ExecResult


DATA_DIR = Path(__file__).parent / "data"
YAML_FILES: List[Path] = sorted(DATA_DIR.glob("*.yaml"))


def _iter_yaml_files() -> Iterable[Path]:
    """
    Yield all YAML files to be tested.

    Returns
    -------
    Iterable[pathlib.Path]
        Paths to all YAML files under ``tests/data``.
    """
    return YAML_FILES


@pytest.mark.parametrize("yaml_path", _iter_yaml_files(), ids=lambda p: p.name)
def test_yaml_parse_and_validate(yaml_path: Path) -> None:
    """
    Parse every YAML config and validate it by constructing a SimulationConfig.

    This test exercises the configuration parsing layer end-to-end without
    actually launching simulations.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Path to a YAML configuration file.

    Raises
    ------
    AssertionError
        If required sections are missing, or validators raise.
    """
    assert yaml_path.exists(), f"Config file not found: {yaml_path}"

    # Load the high-level run config (adjust API if needed in your project).
    cfg: RunConfig = load_run_config(str(yaml_path))
    assert hasattr(cfg, "create"), "RunConfig missing 'create' section"
    assert hasattr(cfg, "fe_sim") or hasattr(
        cfg, "fe"
    ), "RunConfig missing 'fe_sim'/'fe' section"

    # Your code suggested using:
    #   SimulationConfig.from_sections(create, fe, partition=...)
    # Try both attribute names for compatibility.
    fe_section = getattr(cfg, "fe_sim", None) or getattr(cfg, "fe", None)

    # Build the merged model; this runs field + model validators (including _finalize).
    sim_cfg: SimulationConfig = SimulationConfig.from_sections(
        create=cfg.create,
        fe=fe_section,
        protocol=cfg.protocol,
    )

    # A few light sanity checks that derived fields were computed.
    assert isinstance(sim_cfg.component_lambdas, dict)
    assert isinstance(sim_cfg.components, list)
    assert sim_cfg.dec_int in {"mbar", "ti"}
    # If TI is forbidden, your validator will raise before this point.


@pytest.mark.parametrize("yaml_path", _iter_yaml_files(), ids=lambda p: p.name)
def test_yaml_has_minimal_keys(yaml_path: Path) -> None:
    """
    Ensure each YAML file is non-empty and parsable to a RunConfig.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Path to a YAML configuration file.
    """
    txt = yaml_path.read_text(encoding="utf-8").strip()
    assert txt, f"YAML is empty: {yaml_path}"

    # Basic round-trip via loader
    cfg: RunConfig = load_run_config(str(yaml_path))
    # Some minimal presence checks—tune to your schema.
    assert getattr(cfg, "create", None) is not None
    assert (
        getattr(cfg, "fe_sim", None) is not None or getattr(cfg, "fe", None) is not None
    )


@pytest.mark.heavy
@pytest.mark.parametrize("yaml_path", _iter_yaml_files(), ids=lambda p: p.name)
def test_cli_batter_run_dry(
    yaml_path: Path, tmp_path: Path,
) -> None:
    """
    Optional: smoke-test the CLI with a dry run for each YAML.

    This test is **skipped by default** to avoid heavy work. Enable it by setting:
        BATTER_TEST_RUN_CLI=1

    It looks for a ``--dry-run`` (or ``--check``) flag in ``batter run --help`` and uses it if present.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Path to a YAML configuration.
    tmp_path : pathlib.Path
        Temporary directory for CLI outputs, if needed.

    Notes
    -----
    - If ``batter`` is not on PATH, the test is skipped.
    - If no dry-run-like flag is detected, the test is skipped to avoid running full simulations.
    """
    # heavy marker handles skipping unless BATTER_TEST_RUN_CLI=1

    # Ensure 'batter' is invokable
    try:
        help_out = subprocess.run(
            ["batter", "run", "--help"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("'batter' CLI not found or not runnable in this environment.")

    # Detect a safe flag to avoid heavy execution
    has_dry = "--dry-run" in help_out
    has_check = "--check" in help_out
    if not (has_dry or has_check):
        pytest.skip("No --dry-run/--check flag; skipping heavy CLI invocation.")

    # Prefer --dry-run; fall back to --check
    safe_flag = "--dry-run" if has_dry else "--check"

    # Optional: direct output somewhere harmless
    out_dir = tmp_path / "cli_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the CLI.
    # Many CLIs accept the YAML as the positional argument after 'run'; adjust if needed.
    cmd: List[str] = [
        "batter",
        "run",
        safe_flag,
        str(yaml_path),
        "--output-folder",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Helpful debugging on failure
    if proc.returncode != 0:
        sys.stderr.write(f"\n--- STDERR ({yaml_path.name}) ---\n{proc.stderr}\n")
        sys.stderr.write(f"\n--- STDOUT ({yaml_path.name}) ---\n{proc.stdout}\n")

    assert proc.returncode == 0, f"CLI failed for {yaml_path.name} (see logs above)"


@pytest.mark.heavy
def test_runs_prepare_fe(tmp_path: Path) -> None:
    """
    test that --only-equil reuse of a pre-equilibrated folder drives only FE prep.

    This test is **skipped by default** to avoid heavy work. Enable it by setting:
        BATTER_TEST_RUN_CLI=1
    """
    # heavy marker handles skipping unless BATTER_TEST_RUN_CLI=1

    src = DATA_DIR / "equil_finished"
    work_dir = tmp_path / "equil_finished"
    shutil.copytree(src, work_dir)

    calls: list[str] = []

    yaml_path = DATA_DIR / "mabfe.yaml"
    run_from_yaml(
        yaml_path,
        on_failure="raise",
        run_overrides={"output_folder": work_dir, "run_id": "rep1", "dry_run": True},
    )

    # both ligands should have FE prep markers emitted
    for lig in ("CAU", "G1I"):
        lig_dir = (
            work_dir / "executions" / "rep1" / "simulations" / lig / "fe" / "artifacts"
        )
        assert (lig_dir / "prepare_fe.ok").exists()
        assert (lig_dir / "prepare_fe_windows.ok").exists()


def test_run_from_yaml_passes_max_active_jobs(monkeypatch, tmp_path: Path) -> None:
    """Ensure run.max_active_jobs is handed to the SlurmJobManager constructor."""

    run_dir = tmp_path / "out" / "executions" / "rep1"
    run_dir.mkdir(parents=True)
    lig_path = tmp_path / "lig.sdf"
    lig_path.write_text("dummy")

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        f"""
protocol: abfe
backend: slurm
run:
  output_folder: "{tmp_path / 'out'}"
  run_id: auto
  max_active_jobs: 7
create:
  system_name: example
  ligand_paths:
    LIG: "{lig_path}"
fe_sim:
  lambdas: [0.0, 1.0]
  num_equil_extends: 1
  eq_steps: 1000
  steps1: {{z: 50000}}
  steps2: {{z: 50000}}
"""
    )

    # Stub out components that would otherwise perform heavy work
    class DummyBuilder:
        def build(self, sys_exec, create_args):
            sys_exec.root.mkdir(parents=True, exist_ok=True)
            return sys_exec

    class Sentinel(Exception):
        pass

    seen: dict[str, int | None] = {}

    def fake_mgr(*args, **kwargs):
        seen["max_active_jobs"] = kwargs.get("max_active_jobs")
        raise Sentinel

    monkeypatch.setattr(run_mod, "SlurmJobManager", fake_mgr)
    monkeypatch.setattr(run_mod, "_select_system_builder", lambda *a, **k: DummyBuilder())
    monkeypatch.setattr(
        run_mod,
        "select_run_id",
        lambda *a, **k: ("rep1", run_dir),
    )
    monkeypatch.setattr(
        run_mod,
        "_compute_run_signature",
        lambda *a, **k: ("sig", {"payload": True}),
    )
    monkeypatch.setattr(run_mod, "_stored_signature", lambda rd: (None, rd / "sig"))
    monkeypatch.setattr(run_mod, "_stored_payload", lambda rd: None)
    monkeypatch.setattr(run_mod, "_resolve_signature_conflict", lambda *a, **k: True)
    monkeypatch.setattr(run_mod, "discover_staged_ligands", lambda run_dir: {})
    monkeypatch.setattr(
        run_mod,
        "resolve_ligand_map",
        lambda rc, yaml_dir: ({"LIG": lig_path}, {"LIG": "LIG"}),
    )
    monkeypatch.setattr(run_mod, "_store_ligand_names", lambda *a, **k: None)
    monkeypatch.setattr(
        run_mod, "_materialize_extra_conf_restraints", lambda *a, **k: None
    )

    with pytest.raises(Sentinel):
        run_from_yaml(run_yaml)

    assert seen["max_active_jobs"] == 7
