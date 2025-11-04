from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import click
from loguru import logger
import pandas as pd
import yaml
import os
import re
import subprocess
import tempfile

# Import only from the public surface:
from batter.api import (
    run_from_yaml,
    load_sim_config,
    save_sim_config,
    list_fe_runs,
    load_fe_run,
    __version__,
    clone_execution
)


def _write_temp_yaml_with_run_overrides(
    yaml_path: Path,
    *,
    run_id: Optional[str] = None,
    dry_run_flag: Optional[bool] = None,
    only_equil: bool = False,
) -> Path:
    """
    Load the original YAML, apply run-level overrides, write to a temp file,
    and return the temp path. Leaves the original YAML untouched.
    """
    data = yaml.safe_load(yaml_path.read_text())

    # Ensure 'run' section exists
    run = dict(data.get("run") or {})
    if run_id is not None:
        run["run_id"] = run_id
    if dry_run_flag is not None:
        run["dry_run"] = bool(dry_run_flag)
    if only_equil:
        # harmless if not consumed yet by orchestrator; future-proof
        run["only_equil"] = True

    data["run"] = run

    # Write temp YAML
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return Path(tmp.name)

# ----------------------------- Click groups -----------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="batter")
def cli() -> None:
    """
    BATTER command-line interface.
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
@click.option(
    "--output-folder",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory to use instead of that specified in the YAML.",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Override run.run_id. Use 'auto' to let BATTER reuse the latest or create a fresh one.",
)
@click.option(
    "--dry-run/--no-dry-run",
    "dry_run_flag",
    default=None,  # only override if explicitly provided
    help="Override run.dry_run (no submissions / quick exit after first SLURM trigger).",
)
@click.option(
    "--only-equil",
    is_flag=True,
    default=False,
    help="Set run.only_equil=true in the config (run through equil and stop).",
)
def cmd_run(
    yaml_path: Path,
    on_failure: str,
    output_folder: Optional[Path],
    run_id: Optional[str],
    dry_run_flag: Optional[bool],
    only_equil: bool,
) -> None:
    """
    Run an orchestration described by a YAML file.

    Parameters
    ----------
    yaml_path
        Path to a top-level run YAML (includes system/create/run/simulation).
    """
    try:
        # Prepare a temp YAML if any run-level override is requested
        effective_yaml = yaml_path
        if (run_id is not None) or (dry_run_flag is not None) or only_equil:
            effective_yaml = _write_temp_yaml_with_run_overrides(
                yaml_path,
                run_id=run_id,
                dry_run_flag=dry_run_flag,
                only_equil=only_equil,
            )

        overrides = {"output_folder": output_folder} if output_folder else None
        run_from_yaml(effective_yaml, on_failure=on_failure.lower(), system_overrides=overrides)
    except Exception as e:
        raise click.ClickException(str(e))


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
    try:
        _ = load_sim_config(sim_yaml)
    except Exception as e:
        raise click.ClickException(str(e))
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
    try:
        cfg = load_sim_config(in_yaml)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        save_sim_config(cfg, out_yaml)
    except Exception as e:
        raise click.ClickException(str(e))
    click.secho(f"Wrote resolved config to {out_yaml}", fg="green")


# ---------------------------- free energy ------------------------------


@cli.group("fe")
def fe() -> None:
    """Query and inspect free-energy results."""


@fe.command("list")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "csv", "tsv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
def fe_list(work_dir: Path, fmt: str) -> None:
    """
    List FE runs in a work directory.

    Parameters
    ----------
    work_dir
        BATTER work directory (portable across clusters).
    """
    try:
        df = list_fe_runs(work_dir)
    except Exception as e:
        raise click.ClickException(str(e))

    if df.empty:
        click.secho("No FE runs found.", fg="yellow")
        return

    # ensure expected cols exist
    cols = ["run_id", "system_name", "fe_type", "temperature", "method", "total_dG", "total_se", "created_at"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # parse datetime for stable sort, but keep original text if non-parseable
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.assign(_created=created).sort_values("_created", na_position="last").drop(columns=["_created"])
    df = df[cols]

    if fmt.lower() == "json":
        click.echo(df.to_json(orient="records", date_unit="s"))
        return
    if fmt.lower() == "csv":
        click.echo(df.to_csv(index=False))
        return
    if fmt.lower() == "tsv":
        click.echo(df.to_csv(index=False, sep="\t"))
        return

    # pretty table
    with pd.option_context("display.max_columns", None, "display.width", 120):
        # format floats if present
        def _fmt(v):
            try:
                if pd.isna(v):
                    return ""
                if isinstance(v, (float, int)) and str(v) != "":
                    return f"{float(v):.3f}"
            except Exception:
                pass
            return v
        show = df.copy()
        if "total_dG" in show.columns:
            show["total_dG"] = show["total_dG"].map(_fmt)
        if "total_se" in show.columns:
            show["total_se"] = show["total_se"].map(_fmt)
        click.echo(show.to_string(index=False))


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
    try:
        rec = load_fe_run(work_dir, run_id)
    except FileNotFoundError:
        raise click.ClickException(f"Run '{run_id}' not found under {work_dir}.")
    except Exception as e:
        raise click.ClickException(str(e))

    def f3(x):
        try:
            if x is None:
                return "NA"
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    click.secho("Summary", fg="cyan", bold=True)
    click.echo(
        f"- run_id     : {rec.run_id}\n"
        f"- system     : {rec.system_name}\n"
        f"- fe_type    : {rec.fe_type}\n"
        f"- method     : {rec.method}\n"
        f"- temperature: {rec.temperature}\n"
        f"- components : {', '.join(rec.components) if rec.components else ''}\n"
        f"- total_dG   : {f3(getattr(rec, 'total_dG', None))} kcal/mol\n"
        f"- total_se   : {f3(getattr(rec, 'total_se', None))} kcal/mol\n"
        f"- created_at : {getattr(rec, 'created_at', '')}"
    )

    if getattr(rec, "windows", None):
        click.secho("\nPer-window", fg="cyan", bold=True)
        df = pd.DataFrame([w.model_dump() for w in rec.windows])
        # stable, readable column order
        order = [c for c in ["component", "lam", "dG", "dG_se", "n_samples"] if c in df.columns]
        df = df[order + [c for c in df.columns if c not in order]]
        # format numbers
        for col in ["dG", "dG_se"]:
            if col in df.columns:
                df[col] = df[col].map(lambda v: f"{float(v):.3f}" if pd.notna(v) else "")
        with pd.option_context("display.max_columns", None, "display.width", 120):
            click.echo(df.to_string(index=False))
    else:
        click.secho("\n(no per-window data saved)", fg="yellow")

# ------------------------ execution cloning ---------------------------
@cli.command("clone-exec")
@click.argument("work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("src_run_id", type=str)
@click.argument("dst_run_id", required=False)
@click.option(
    "--dst-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination work directory (defaults to WORK_DIR).",
)
@click.option(
    "--only-equil/--full",
    default=True,
    show_default=True,
    help="Clone only equilibration artifacts or the full FE layout.",
)
@click.option(
    "--symlink/--copy",
    default=True,
    show_default=True,
    help="Use symlinks instead of copying files where possible.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite destination execution folder if it exists.",
)
def cmd_clone_exec(
    work_dir: Path,
    src_run_id: str,
    dst_run_id: str | None,
    dst_root: Path | None,
    only_equil: bool,
    symlink: bool,
    force: bool,
) -> None:
    """
    Clone an existing execution (RUN_ID) to a new RUN_ID (and optionally a new WORK_DIR).

    WORK_DIR is the source work directory containing executions/<RUN_ID>.
    """
    dst_root = dst_root or work_dir
    if dst_run_id is None:
        dst_run_id = f"{src_run_id}-clone"

    # Basic sanity checks to give nice CLI errors before calling the underlying function
    src_exec = work_dir / "executions" / src_run_id
    if not src_exec.is_dir():
        raise click.ClickException(f"Source execution not found: {src_exec}")

    dst_exec = dst_root / "executions" / dst_run_id
    if dst_exec.exists() and not force:
        raise click.ClickException(
            f"Destination already exists: {dst_exec} (use --force to overwrite)"
        )

    # Delegate to your existing implementation
    try:
        clone_execution(
            src_root=work_dir,
            src_run_id=src_run_id,
            dst_root=dst_root,
            dst_run_id=dst_run_id,
            only_equil=only_equil,
            symlink=symlink,
            force=force,
        )
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.secho(
        f"Cloned execution '{src_run_id}' → '{dst_run_id}' under {dst_root}",
        fg="green",
    )


# ----------------------------- check status -------------------------------
KNOWN_STAGES = [
    "equil",
    "fe_equil",
    "fe",
]

__STAGE_RE = re.compile(rf"(?P<stage>{'|'.join(sorted(KNOWN_STAGES, key=len, reverse=True))})$")

_JOB_RE = re.compile(
    rf"""
    ^fep_
    (?P<abs>/.+?)                         # absolute root up to .../executions/<run_id>/simulations
    /executions/(?P<run_id>[^/]+)/simulations/
    (?P<ligand_stage>[^/]+)               # ligand+stage (concatenated)
    $
    """,
    re.VERBOSE,
)

def natural_keys(text: str):
    """Sort helper: natural order for strings with numbers."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(text))]


@cli.command("report-jobs")
@click.option(
    "--partition",
    "-p",
    default=None,
    help="SLURM partition to report jobs from; if not specified, report all partitions.",
)
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    default=False,
    help="Show detailed job information.",
)
def report_jobs(partition=None, detailed=False):
    """
    Report the status of SLURM jobs of BATTER runs (job names prefixed with 'fep_').
    """
    try:
        base_cmd = ["squeue", "--user", os.getenv("USER"), "--format=%i %j %T"]
        if partition:
            base_cmd = ["squeue", "--partition", partition, "--user", os.getenv("USER"), "--format=%i %j %T"]
        result = subprocess.run(
            base_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to get SLURM job list: {e.stderr}")
        return

    lines = result.stdout.strip().splitlines()
    if not lines:
        click.echo("No jobs found.")
        return

    # First line is header
    _, *jobs = lines
    job_info_list = []

    for job_line in jobs:
        parts = job_line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        jobid, jobname, status = parts

        if not jobname.startswith("fep_"):
            continue

        m = _JOB_RE.match(jobname)
        if not m:
            # Unknown layout—skip but show a hint in detailed mode
            if detailed:
                click.echo(f"Skipping unrecognized job name: {jobname}")
            continue

        abs_root = m.group("abs")
        run_id = m.group("run_id")
        ligand_stage = m.group("ligand_stage")

        # Split ligand and stage by matching a trailing KNOWN_STAGES token
        ms = __STAGE_RE.search(ligand_stage)
        if not ms:
            # No recognized stage suffix; treat whole token as ligand and stage unknown
            ligand = ligand_stage
            stage = "unknown"
        else:
            stage = ms.group("stage")
            ligand = ligand_stage[: -len(stage)]

        # Normalize ligand (strip incidental separators)
        ligand = ligand.rstrip("_-")

        # Try to extract comp/win for FE phases if present in the absolute path
        # This works for names like .../fe/<comp>/<comp>-1 or .../fe/<comp>/<comp>NN
        comp = ""
        win = ""
        fe_path_match = re.search(r"/fe/([^/]+)/([^/]+)$", abs_root)
        if fe_path_match:
            comp = fe_path_match.group(1)
            tail = fe_path_match.group(2)
            # tail examples: Z-1, Z00, Z07, etc.
            # Window is digits at the end; equil may be '-1'
            mwin = re.search(r"(-?\d+)$", tail)
            if mwin:
                win = mwin.group(1)

        # The "system" group: everything up to /executions/<run_id>
        system_root = f"{abs_root}"

        job_info = {
            "system": system_root,
            "run_id": run_id,
            "jobid": jobid,
            "ligand": ligand or "(unknown)",
            "comp": comp,
            "win": win,
            "stage": stage,
            "status": status,
        }
        job_info_list.append(job_info)

    job_df = pd.DataFrame(job_info_list)
    if job_df.empty:
        click.echo("No FEP jobs found.")
        return

    # Detect duplicates by (system, run_id, ligand, comp, win, stage)
    dup_keys = ["system", "run_id", "ligand", "comp", "win", "stage"]
    duplicates = job_df.duplicated(subset=dup_keys, keep=False)
    if duplicates.any():
        click.echo(click.style("Warning: Found duplicate jobs with the same keys.", fg="yellow"))
        for _, row in job_df[duplicates].iterrows():
            click.echo(
                f"Duplicate Job ID: {row['jobid']} "
                f"- System: {row['system']} "
                f"- Run: {row['run_id']} "
                f"- Ligand: {row['ligand']} "
                f"- Comp: {row['comp']} "
                f"- Win: {row['win']} "
                f"- Stage: {row['stage']} "
                f"- Status: {row['status']}"
            )

    total_jobs = len(job_df)
    running_jobs = (job_df["status"] == "RUNNING").sum()
    pending_jobs = (job_df["status"] == "PENDING").sum()
    click.echo(click.style(f"Total jobs: {total_jobs}, Running: {running_jobs}, Pending: {pending_jobs}", bold=True))

    # Group by system/run and summarize
    for (system, run_id), sys_df in job_df.groupby(["system", "run_id"], sort=False):
        stages = ", ".join(sorted(sys_df["stage"].unique(), key=natural_keys))
        click.echo(click.style(f"\nSystem: {system}\nRun: {run_id}\nStages: {stages}", bold=True))
        click.echo("-" * 60)
        click.echo(click.style("Ligand (PENDING, RUNNING):", bold=True))

        grouped = sys_df.groupby("ligand")["status"].value_counts().unstack(fill_value=0).reset_index()
        for col in ("RUNNING", "PENDING"):
            if col not in grouped.columns:
                grouped[col] = 0
        grouped = grouped.rename(columns={"RUNNING": "running_jobs", "PENDING": "pending_jobs"})
        grouped = grouped.sort_values(by="ligand", key=lambda x: x.map(natural_keys))

        # Make compact rows
        rows = []
        for _, row in grouped.iterrows():
            lig = row["ligand"]
            p = int(row.get("pending_jobs", 0))
            r = int(row.get("running_jobs", 0))
            if r > 0:
                r_str = click.style(f"{lig}(P={p},R={r})", fg="green", bold=True)
            else:
                r_str = click.style(f"{lig}(P={p},R={r})", fg="red")
            rows.append(r_str)

        # Print 4 per line
        for i in range(0, len(rows), 4):
            click.echo("   ".join(rows[i : i + 4]))

        if detailed:
            click.echo("")
            for _, row in sys_df.sort_values(by=["ligand", "comp", "win"], key=lambda s: s.map(natural_keys) if s.dtype == object else s).iterrows():
                base = f"Job ID: {row['jobid']} - Ligand: {row['ligand']} - Stage: {row['stage']}"
                extra = ""
                if row["comp"] or row["win"]:
                    extra = f" - Comp: {row['comp']} - Win: {row['win']}"
                if row["status"] == "RUNNING":
                    click.echo(click.style(base + extra + f" - Status: {row['status']}", fg="green", bold=True))
                elif row["status"] == "PENDING":
                    click.echo(click.style(base + extra + f" - Status: {row['status']}", fg="red"))
        click.echo("-" * 60)

    click.echo("If you want to cancel jobs, use 'batter cancel-jobs --name <substring>'.")


@cli.command("cancel-jobs")
@click.option("--name", "-n", required=True, help="Substring to match in job names (after 'fep_').")
def cancel_jobs(name):
    """
    Cancel all SLURM jobs whose names contain the given substring.
    """
    try:
        result = subprocess.run(
            ["squeue", "-u", os.getenv("USER"), "--format=%i %j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Error querying SLURM: {e.stderr}")
        return

    lines = result.stdout.strip().split("\n")[1:]
    matching_ids = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        jid, jname = parts
        # Only consider our jobs
        if not jname.startswith("fep_"):
            continue
        if name in jname:
            matching_ids.append(jid)

    if not matching_ids:
        click.echo(f"No jobs found containing '{name}' in job name.")
        return

    click.echo(f"Cancelling {len(matching_ids)} job(s)")
    for i in range(0, len(matching_ids), 30):
        batch = matching_ids[i : i + 30]
        try:
            subprocess.run(["scancel"] + batch, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"Failed to cancel jobs: {e.stderr}")
