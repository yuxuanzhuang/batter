from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from loguru import logger
import pandas as pd

# Import only from the public surface:
from batter.api import (
    run_from_yaml,
    load_sim_config,
    save_sim_config,
    list_fe_runs,
    load_fe_run,
    __version__,
)


# ----------------------------- Click groups -----------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="batter")
def cli() -> None:
    """
    BATTER command-line interface.

    Use `batter run` to execute a pipeline from a top-level YAML,
    and `batter fe` subcommands to query portable FE results.
    """


# -------------------------------- run ----------------------------------


@cli.command("run")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--on-failure",
    type=click.Choice(["prune", "raise"], case_sensitive=False),
    default="raise",
    show_default=True,
    help="How to handle a failing ligand pipeline: prune (skip ligand) or raise (abort run).",
)
def cmd_run(yaml_path: Path, on_failure: str) -> None:
    """
    Run an orchestration described by a YAML file.

    Parameters
    ----------
    yaml_path
        Path to a top-level run YAML (includes system/create/run/simulation).
    """
    logger.info("Starting BATTER run from {}", yaml_path)
    run_from_yaml(yaml_path, on_failure=on_failure.lower())
    logger.success("Run completed for {}", yaml_path)


# ------------------------------- config --------------------------------


@cli.group("cfg")
def cfg() -> None:
    """Configuration utilities."""


@cfg.command("validate")
@click.argument("sim_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def cfg_validate(sim_yaml: Path) -> None:
    """
    Validate a SimulationConfig YAML.

    Parameters
    ----------
    sim_yaml
        Path to a SimulationConfig YAML file.
    """
    _ = load_sim_config(sim_yaml)
    click.secho("OK: configuration is valid.", fg="green")


@cfg.command("resolve")
@click.argument("in_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("out_yaml", type=click.Path(dir_okay=False, path_type=Path))
def cfg_resolve(in_yaml: Path, out_yaml: Path) -> None:
    """
    Load a SimulationConfig and write a resolved YAML.

    Parameters
    ----------
    in_yaml
        Input SimulationConfig YAML.
    out_yaml
        Output path for the resolved YAML (directories are created).
    """
    cfg = load_sim_config(in_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    save_sim_config(cfg, out_yaml)
    click.secho(f"Wrote resolved config to {out_yaml}", fg="green")


# ---------------------------- free energy ------------------------------


@cli.group("fe")
def fe() -> None:
    """Query and inspect free-energy results."""


@fe.command("list")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def fe_list(work_dir: Path) -> None:
    """
    List FE runs in a work directory.

    Parameters
    ----------
    work_dir
        BATTER work directory (portable across clusters).
    """
    df = list_fe_runs(work_dir)
    if df.empty:
        click.secho("No FE runs found.", fg="yellow")
        return

    # pretty-ish print
    cols = ["run_id", "system_name", "fe_type", "temperature", "method", "total_dG", "total_se", "created_at"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""  # ensure column exists
    df = df[cols].sort_values("created_at")
    # limit width for terminal
    with pd.option_context("display.max_columns", None, "display.width", 120):
        click.echo(df.to_string(index=False))


@fe.command("show")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("run_id", type=str)
def fe_show(work_dir: Path, run_id: str) -> None:
    """
    Show a single FE record.

    Parameters
    ----------
    work_dir
        BATTER work directory (portable across clusters).
    run_id
        Identifier of the run (see `batter fe list`).
    """
    rec = load_fe_run(work_dir, run_id)

    click.secho("Summary", fg="cyan", bold=True)
    click.echo(
        f"- run_id     : {rec.run_id}\n"
        f"- system     : {rec.system_name}\n"
        f"- fe_type    : {rec.fe_type}\n"
        f"- method     : {rec.method}\n"
        f"- temperature: {rec.temperature}\n"
        f"- components : {', '.join(rec.components)}\n"
        f"- total_dG   : {rec.total_dG:.3f} kcal/mol\n"
        f"- total_se   : {rec.total_se:.3f} kcal/mol\n"
        f"- created_at : {rec.created_at}"
    )

    if rec.windows:
        click.secho("\nPer-window", fg="cyan", bold=True)
        df = pd.DataFrame([w.model_dump() for w in rec.windows])
        order = [c for c in ["component", "lam", "dG", "dG_se", "n_samples"] if c in df.columns]
        df = df[order + [c for c in df.columns if c not in order]]
        with pd.option_context("display.max_columns", None, "display.width", 120):
            click.echo(df.to_string(index=False))
    else:
        click.secho("\n(no per-window data saved)", fg="yellow")