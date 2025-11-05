from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
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
@click.option("--on-failure", type=click.Choice(["prune","raise"], case_sensitive=False),
              default="raise", show_default=True)
@click.option("--output-folder", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--run-id", default=None, help="Override run_id (e.g., rep1). Use 'auto' to reuse latest.")
@click.option("--dry-run/--no-dry-run", default=None, help="Override YAML run.dry_run.")
@click.option("--only-equil/--full", default=None, help="Run only equil steps; override YAML.")
def cmd_run(yaml_path: Path, on_failure: str, output_folder: Optional[Path],
            run_id: Optional[str], dry_run: Optional[bool], only_equil: Optional[bool]) -> None:
    overrides = {}
    if output_folder:
        overrides["output_folder"] = output_folder
    run_over = {}
    if run_id is not None:
        run_over["run_id"] = run_id
    if dry_run is not None:
        run_over["dry_run"] = dry_run
    if only_equil is not None:
        run_over["only_fe_preparation"] = only_equil
    run_from_yaml(
        yaml_path,
        on_failure=on_failure.lower(),
        system_overrides=(overrides or None),
        run_overrides=(run_over or None),
    )


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
_STAGE_SUFFIXES = ("_fe_equil", "_fe", "_eq")

#   LIG_COMP_fe_equil
_tail_re = re.compile(
    r"""
    ^
    (?P<lig>[A-Za-z0-9][A-Za-z0-9._-]*)
    (?:
        _(?:
            # fe-equil
            (?P<comp_feq>[A-Za-z]+)_fe_equil
          |
            # fe
            (?P<comp_fe>[A-Za-z]+)
            (?:
                _(?P<comptok>[A-Za-z]*)
                 (?P<win>\d+)
            )?
            _fe
          |
            # equil
            (?P<eq>eq)
        )
    )?
    $
    """,
    re.X,
)

def _endswith_stage(jobname: str):
    for s in _STAGE_SUFFIXES:
        if jobname.endswith(s):
            return s[1:], jobname[: -len(s)]  # stage (no leading _), basepath
    return None, jobname

def _split_after(parts, token):
    """Return index after first occurrence of token; -1 if not found."""
    try:
        i = parts.index(token)
        return i + 1
    except ValueError:
        return -1

def _parse_jobname(jobname: str):
    """
    Parse BATTER job names of forms:
      .../simulations/<LIGAND>_eq
      .../simulations/<LIGAND>_<COMP>_<COMP><WIN>_fe
      .../simulations/<LIGAND>_<COMP>_fe_equil
    Returns dict with: stage, run_id, system_root, ligand, comp, win(int|None)
    """
    if not jobname.startswith("fep_"):
        return None

    body = jobname[4:]  # strip 'fep_'

    # Split into prefix (system_root-ish) and tail (the last segment with stage info)
    if "/simulations/" in body:
        pre, tail = body.split("/simulations/", 1)
        tail = tail.rsplit("/", 1)[-1]  # keep only the last path component
    else:
        # Fallback: take last path component as tail
        parts = body.rsplit("/", 1)
        pre = parts[0] if len(parts) == 2 else ""
        tail = parts[-1]

    # Extract run_id from /executions/<RID>/
    run_id = None
    mrun = re.search(r"/executions/([^/]+)/", "/" + pre + "/")
    if mrun:
        run_id = mrun.group(1)

    m = _tail_re.match(tail)
    if not m:
        return {
            "stage": "unknown",
            "run_id": run_id,
            "system_root": pre,
            "ligand": None,
            "comp": None,
            "win": None,
        }

    gd = m.groupdict()
    ligand = gd.get("lig")
    comp = None
    win = None
    stage = "unknown"

    if gd.get("eq"):
        stage = "eq"
    elif gd.get("comp_feq"):
        stage = "fe_equil"
        comp = gd.get("comp_feq")
    elif gd.get("comp_fe"):
        stage = "fe"
        comp = gd.get("comp_fe")
        if gd.get("win"):
            try:
                win = int(gd["win"])
            except ValueError:
                win = None

    return {
        "stage": stage,
        "run_id": run_id,
        "system_root": pre,
        "ligand": ligand,
        "comp": comp,
        "win": win,
    }

_nat_split_rx = re.compile(r"(\d+)")

def _natural_keys(val: str):
    s = "" if val is None else str(val)
    parts = _nat_split_rx.split(s)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

def _natkey_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).map(_natural_keys)

@cli.command("report-jobs")
@click.option("--partition", "-p", default=None, help="SLURM partition filter.")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed job lines.")
def report_jobs(partition=None, detailed=False):
    """Report the status of SLURM jobs launched by BATTER (job names starting with 'fep_')."""
    try:
        cmd = ["squeue", "--user", os.getenv("USER"), "--format=%i %j %T"]
        if partition:
            cmd = ["squeue", "--partition", partition, "--user", os.getenv("USER"), "--format=%i %j %T"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to get SLURM job list: {e.stderr}")
        return

    lines = res.stdout.strip().splitlines()
    if not lines or len(lines) == 1:
        click.echo("No SLURM jobs found.")
        return
    _, *jobs = lines  # drop header

    rows = []
    for line in jobs:
        # robust split: job name may contain spaces → use maxsplit=2
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        jobid, jobname, status = parts
        if not jobname.startswith("fep_"):
            continue
        meta = _parse_jobname(jobname)
        rows.append({
            "jobid": jobid,
            "status": status,
            "stage": meta["stage"],
            "run_id": meta["run_id"],
            "system_root": meta["system_root"],
            "ligand": meta["ligand"],
            "comp": meta["comp"],
            "win": meta["win"],
            "jobname": jobname,
        })

    if not rows:
        click.echo("No BATTER jobs (fep_*) found.")
        return

    df = pd.DataFrame(rows)

    # topline
    total = len(df)
    running = (df["status"] == "RUNNING").sum()
    pending = (df["status"] == "PENDING").sum()
    click.echo(click.style(f"Total jobs: {total}, Running: {running}, Pending: {pending}", bold=True))

    # group by run_id (fallback to system_root)
    grp_key = df["run_id"]
    for gid, sub in df.groupby(grp_key):
        system_root = sub["system_root"].dropna().unique()[0]
        click.echo(click.style(f"\nRun: {system_root}", bold=True))
        stages = ", ".join(sorted(sub["stage"].dropna().unique()))
        click.echo(f"Stages present: {stages or '(unknown)'}")
        click.echo("-" * 70)

        label_col = "ligand"

        summary = sub.assign(label=sub[label_col]).groupby(["label"])["status"].value_counts().unstack(fill_value=0).reset_index()

        for need_col in ("RUNNING", "PENDING"):
            if need_col not in summary.columns:
                summary[need_col] = 0

        # natural sort labels
        summary = summary.sort_values(by="label", key=lambda s: s.map(_natural_keys))

        # print compact two columns
        line_buf = []
        for _, r in summary.iterrows():
            label = r["label"]
            p = int(r.get("PENDING", 0))
            r_ = int(r.get("RUNNING", 0))
            colored = click.style(f"{label}(P={p},R={r_})",
                                  fg=("green" if r_ > 0 else "yellow" if p > 0 else "red"),
                                  bold=(r_ > 0))
            line_buf.append(colored)

        for i in range(0, len(line_buf), 4):
            click.echo("   ".join(line_buf[i:i+4]))

        if detailed:
            click.echo(click.style("\nDetailed:", bold=True))
            det = (sub[["jobid", "status", "stage", "ligand", "comp", "win"]]
                .assign(win=sub["win"])
                .sort_values(["stage", "ligand", "comp", "win"], key=_natkey_series))
            with pd.option_context("display.width", 140, "display.max_columns", None):
                click.echo(det.to_string(index=False))
        click.echo("-" * 70)

    click.echo("To cancel, run: new_batter cancel-jobs --contains '<substring>'")


@click.command("cancel-jobs")
@click.option("--contains", "-c", required=True,
              help="Cancel all jobs whose SLURM job name contains this substring (match against full 'fep_...').")
def cancel_jobs(contains: str):
    """Cancel all SLURM jobs whose names contain the given substring."""
    try:
        res = subprocess.run(
            ["squeue", "--user", os.getenv("USER"), "--format=%i %j"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Error querying SLURM: {e.stderr}")
        return

    ids = []
    for line in res.stdout.strip().splitlines()[1:]:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        jobid, jobname = parts
        if contains in jobname:
            ids.append(jobid)

    if not ids:
        click.echo(f"No jobs found containing '{contains}'.")
        return

    click.echo(f"Cancelling {len(ids)} job(s)")
    for i in range(0, len(ids), 30):
        batch = ids[i:i+30]
        try:
            subprocess.run(["scancel"] + batch, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"Failed to cancel {batch}: {e.stderr}")