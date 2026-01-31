"""Run workflow commands."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple

import click

from batter.api import run_from_yaml
from batter.cli.root import cli
from batter.cli.shared import _upsert_sbatch_option, _which_batter
from batter.config.run import RunConfig
from batter.data import job_manager
from batter.orchestrate.run_support import (
    compute_run_signature,
    generate_run_id,
    resolve_signature_conflict,
    select_run_id,
    stored_payload,
    stored_signature,
)
from batter.utils.slurm_templates import render_slurm_with_header_body


def hash_run_input(yaml_path: Path, **options) -> str:
    """
    Return a stable hash for the YAML contents and CLI overrides.

    Parameters
    ----------
    yaml_path : Path
        Path to the run YAML file.
    **options
        CLI overrides that should affect the hash.

    Returns
    -------
    str
        First 12 characters of the SHA-256 digest.
    """
    p = Path(yaml_path)
    data = p.read_bytes()  # raw bytes to avoid newline normalization issues
    # freeze options dict deterministically (sorted keys)
    frozen = json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256()
    h.update(data)
    h.update(b"\0")  # separator byte to avoid accidental concatenation collisions
    h.update(frozen)
    return h.hexdigest()[:12]


def _resolve_run_dir_for_submission(
    cfg: RunConfig, yaml_path: Path, run_overrides: Dict[str, Any]
) -> Tuple[str, Path]:
    """
    Mirror the run_id resolution used by ``run_from_yaml`` to keep SLURM job names stable.
    """
    config_signature, config_payload = compute_run_signature(
        yaml_path, run_overrides or None
    )
    requested_run_id = getattr(cfg.run, "run_id", "auto")

    while True:
        run_id, run_dir = select_run_id(
            cfg.run.output_folder,
            cfg.protocol,
            cfg.create.system_name,
            requested_run_id,
        )
        stored_sig, _ = stored_signature(run_dir)
        prev_payload = stored_payload(run_dir)
        if resolve_signature_conflict(
            stored_sig,
            config_signature,
            requested_run_id,
            cfg.run.allow_run_id_mismatch,
            stored_payload=prev_payload,
            current_payload=config_payload,
            run_id=run_id,
            run_dir=run_dir,
        ):
            return run_id, run_dir

        # mismatch with auto run_id → generate a fresh execution directory
        requested_run_id = "auto"
        run_id = generate_run_id(cfg.protocol, cfg.create.system_name)
        run_dir = Path(cfg.run.output_folder) / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)


@cli.command("run")
@click.argument(
    "yaml_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--on-failure",
    type=click.Choice(["prune", "raise", "retry"], case_sensitive=False),
    default="raise",
    show_default=True,
)
@click.option(
    "--output-folder", type=click.Path(file_okay=False, path_type=Path), default=None
)
@click.option(
    "--run-id",
    default=None,
    help="Override run_id (e.g., rep1). Use 'auto' to reuse latest.",
)
@click.option(
    "--allow-run-id-mismatch/--no-allow-run-id-mismatch",
    default=None,
    help="Allow reusing a provided run-id even if the stored configuration hash differs.",
)
@click.option("--dry-run/--no-dry-run", default=None, help="Override YAML run.dry_run.")
@click.option(
    "--clean-failures/--no-clean-failures",
    default=None,
    help="Clear FAILED markers and progress caches before rerunning.",
)
@click.option(
    "--only-equil/--full", default=None, help="Run only equil steps; override YAML."
)
@click.option(
    "--slurm-submit/--local-run",
    default=False,
    help="Submit this run via SLURM (sbatch) instead of running locally.",
)
@click.option(
    "--slurm-manager-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a SLURM header/template to prepend to the generated script.",
)
def cmd_run(
    yaml_path: Path,
    on_failure: str,
    output_folder: Optional[Path],
    run_id: Optional[str],
    allow_run_id_mismatch: Optional[bool],
    dry_run: Optional[bool],
    clean_failures: Optional[bool],
    only_equil: Optional[bool],
    slurm_submit: bool,
    slurm_manager_path: Optional[Path],
) -> None:
    """
    Execute a BATTER workflow defined in ``YAML_PATH``.
    """
    run_over = {}
    if output_folder:
        run_over["output_folder"] = output_folder
    if run_id is not None:
        run_over["run_id"] = run_id
    if allow_run_id_mismatch is not None:
        run_over["allow_run_id_mismatch"] = allow_run_id_mismatch
    if dry_run is not None:
        run_over["dry_run"] = dry_run
    if clean_failures is not None:
        run_over["clean_failures"] = clean_failures
    if only_equil is not None:
        run_over["only_fe_preparation"] = only_equil

    # first do a basic validation of the YAML (with any CLI overrides applied)
    try:
        base_cfg = RunConfig.load(yaml_path)
        cfg_for_validation = base_cfg
        if run_over:
            cfg_for_validation = cfg_for_validation.model_copy(
                update={"run": cfg_for_validation.run.model_copy(update=run_over)}
            )
        # Force resolution so missing/invalid fields are surfaced before submitting
        cfg_for_validation.resolved_sim_config()
    except Exception as e:
        raise click.ClickException(f"Invalid SimulationConfig YAML: {e}")

    if slurm_submit:
        _, run_dir = _resolve_run_dir_for_submission(
            cfg_for_validation, yaml_path, run_over
        )
        run_dir_abs = run_dir.resolve()
        manager_job_name = (
            "fep_" + (PurePosixPath(run_dir_abs) / "simulations" / "manager").as_posix()
        )
        log_base = f"manager-{run_dir_abs.name or 'run'}"

        batter_cmd = _which_batter()
        parts = [batter_cmd, "run", shlex.quote(str(Path(yaml_path).resolve()))]
        parts += ["--on-failure", shlex.quote(on_failure)]

        if output_folder:
            parts += [
                "--output-folder",
                shlex.quote(str(Path(output_folder).resolve())),
            ]
        if run_id is not None:
            parts += ["--run-id", shlex.quote(run_id)]
        if allow_run_id_mismatch is not None:
            parts += [
                (
                    "--allow-run-id-mismatch"
                    if allow_run_id_mismatch
                    else "--no-allow-run-id-mismatch"
                )
            ]
        if dry_run is not None:
            parts += ["--dry-run" if dry_run else "--no-dry-run"]
        if clean_failures is not None:
            parts += [
                "--clean-failures" if clean_failures else "--no-clean-failures"
            ]
        if only_equil is not None:
            parts += ["--only-equil" if only_equil else "--full"]

        run_cmd = " ".join(parts)

        # create a hash based on contents of the yaml and options
        run_hash = hash_run_input(
            yaml_path,
            on_failure=on_failure.lower(),
            output_folder=str(Path(output_folder).resolve()) if output_folder else "",
            run_id=run_id or "",
            dry_run=("1" if dry_run else "0") if dry_run is not None else "",
            clean_failures=("1" if clean_failures else "0")
            if clean_failures is not None
            else "",
            only_equil=("1" if only_equil else "0") if only_equil is not None else "",
        )
        base_path = (
            Path(slurm_manager_path) if slurm_manager_path else Path(job_manager)
        )
        tpl_header = base_path.with_suffix(".header")
        tpl_body = base_path.with_suffix(".body")
        manager_code = render_slurm_with_header_body(
            "job_manager.header",
            tpl_header,
            tpl_body,
            {
                "__JOB_NAME__": manager_job_name,
                "__JOB_LOG_BASE__": log_base,
            },
            header_root=cfg_for_validation.run.slurm_header_dir,
        )
        manager_code = _upsert_sbatch_option(manager_code, "job-name", manager_job_name)
        with open(f"{run_hash}_job_manager.sbatch", "w") as f:
            f.write(manager_code)
            f.write("\n")
            f.write(run_cmd)
            f.write("\n")
            f.write("echo 'Job completed.'\n")
            f.write("\n")

        # submit slurm job
        result = subprocess.run(
            ["sbatch", f"{run_hash}_job_manager.sbatch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        click.echo(f"Submitted jobscript: {run_hash}_job_manager.sbatch")
        click.echo(f"STDOUT: {result.stdout}")
        click.echo(f"STDERR: {result.stderr}")
        return

    run_from_yaml(
        yaml_path,
        on_failure=on_failure.lower(),
        run_overrides=(run_over or None),
    )


@cli.command("run-exec")
@click.argument(
    "execution_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--on-failure",
    type=click.Choice(["prune", "raise", "retry"], case_sensitive=False),
    default="raise",
    show_default=True,
)
def cmd_run_exec(execution_dir: Path, on_failure: str) -> None:
    """
    Resume/extend a run using only an existing execution directory.
    """
    exec_dir = execution_dir.resolve()
    yaml_copy = exec_dir / "artifacts" / "config" / "run_config.yaml"
    if not yaml_copy.exists():
        raise click.ClickException(
            f"Could not find stored run_config.yaml under {yaml_copy}. "
            "Run once with `batter run` to seed artifacts/config."
        )

    run_overrides = {
        "output_folder": exec_dir.parent.parent,
        "run_id": exec_dir.name,
        "allow_run_id_mismatch": True,
    }
    try:
        run_from_yaml(
            yaml_copy,
            on_failure=on_failure.lower(),
            run_overrides=run_overrides,
        )
    except Exception as e:
        raise click.ClickException(str(e))
