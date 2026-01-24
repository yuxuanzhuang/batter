"""SLURM job reporting and cancellation commands."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

import click
import pandas as pd

from batter.cli.root import cli
from batter.utils import natural_keys


def _parse_jobname(jobname: str) -> dict[str, Optional[object]] | None:
    """
    Parse BATTER job names emitted by SLURM handlers.
    """
    if not jobname.startswith("fep_"):
        return None

    body = jobname[4:]
    if "/simulations/" not in body:
        return None

    system_root, tail = body.split("/simulations/", 1)
    if "/" in tail:
        tail = tail.split("/", 1)[0]

    if not tail:
        return None

    stage = "unknown"
    comp = None
    win: int | None = None

    ligand = tail
    if tail == "manager":
        ligand = "MANAGER_JOB"
        stage = "manager"
    elif tail.endswith("_eq"):
        ligand = tail[: -len("_eq")]
        stage = "eq"
    elif tail.endswith("_fe_equil"):
        core = tail[: -len("_fe_equil")]
        parts = core.rsplit("_", 1)
        if len(parts) == 2:
            ligand, comp = parts
        stage = "fe_equil"
    elif tail.endswith("_fe"):
        core = tail[: -len("_fe")]
        m = re.match(r"(?P<lig>.+)_(?P<comp>[A-Za-z]+)_(?P<win>[A-Za-z]+\d+)$", core)
        if m:
            ligand = m.group("lig")
            comp = m.group("comp")
            win_match = re.search(r"(\d+)$", m.group("win"))
            if win_match:
                try:
                    win = int(win_match.group(1))
                except ValueError:
                    win = None
        stage = "fe"
    elif tail.endswith("_remd"):
        core = tail[: -len("_remd")]
        m = re.match(r"(?P<lig>.+)_(?P<comp>[A-Za-z]+)$", core)
        if m:
            ligand = m.group("lig")
            comp = m.group("comp")
        stage = "remd"

    run_id = None
    mrun = re.search(r"/executions/([^/]+)$", system_root)
    if mrun:
        run_id = mrun.group(1)

    return {
        "stage": stage,
        "run_id": run_id,
        "system_root": system_root,
        "ligand": ligand,
        "comp": comp,
        "win": win,
    }


def _natural_keys(val: str | None):
    """Return a tuple suitable for natural sorting of strings containing digits."""
    s = "" if val is None else str(val)
    parts = natural_keys(s)
    return tuple(p.lower() if isinstance(p, str) else p for p in parts)


def _natkey_series(s: pd.Series) -> pd.Series:
    """Vectorised version of :func:`_natural_keys` for pandas Series."""
    assert pd is not None  # for type-checkers
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).map(_natural_keys)


@cli.command("report-jobs")
@click.option("--partition", "-p", default=None, help="SLURM partition filter.")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed job lines.")
def report_jobs(partition=None, detailed=False):
    """Report SLURM job status for BATTER jobs prefixed with ``fep_``."""
    try:
        cmd = ["squeue", "--user", os.getenv("USER"), "--format=%i %j %T"]
        if partition:
            cmd = [
                "squeue",
                "--partition",
                partition,
                "--user",
                os.getenv("USER"),
                "--format=%i %j %T",
            ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
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
        if meta is None:
            continue
        rows.append(
            {
                "jobid": jobid,
                "status": status,
                "stage": meta["stage"],
                "run_id": meta["run_id"],
                "identifier": (
                    meta["system_root"] + "/" + meta["run_id"]
                    if meta["run_id"]
                    else meta["system_root"]
                ),
                "system_root": meta["system_root"],
                "ligand": meta["ligand"],
                "comp": meta["comp"],
                "win": meta["win"],
                "jobname": jobname,
            }
        )

    if not rows:
        click.echo("No BATTER jobs (fep_*) found.")
        return

    df = pd.DataFrame(rows)

    # topline
    total = len(df)
    running = (df["status"] == "RUNNING").sum()
    pending = (df["status"] == "PENDING").sum()
    click.echo(
        click.style(
            f"Total jobs: {total}, Running: {running}, Pending: {pending}", bold=True
        )
    )

    grp_key = df["identifier"].where(df["identifier"].notna())
    df = df.assign(_group=grp_key)
    for gid, sub in df.groupby("_group"):
        sys_root = sub["system_root"].dropna().unique()
        label = gid if gid is not None else "(unknown)"
        if sys_root.size > 0:
            label = sys_root[0]
        click.echo(click.style(f"\nRun: {label}", bold=True))
        stages = ", ".join(sorted(sub["stage"].dropna().unique()))
        click.echo(f"Stages present: {stages or '(unknown)'}")
        click.echo("-" * 70)

        label_col = "ligand"

        summary = (
            sub.assign(label=sub[label_col])
            .groupby(["label"])["status"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )

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
            colored = click.style(
                f"{label}(P={p},R={r_})",
                fg=("green" if r_ > 0 else "yellow" if p > 0 else "red"),
                bold=(r_ > 0),
            )
            line_buf.append(colored)

        for i in range(0, len(line_buf), 4):
            click.echo("   ".join(line_buf[i : i + 4]))

        if detailed:
            click.echo(click.style("\nDetailed:", bold=True))
            det = (
                sub[["jobid", "status", "stage", "ligand", "comp", "win"]]
                .assign(win=sub["win"])
                .sort_values(["stage", "ligand", "comp", "win"], key=_natkey_series)
            )
            with pd.option_context("display.width", 140, "display.max_columns", None):
                click.echo(det.to_string(index=False))
        click.echo("-" * 70)

    click.echo("To cancel, run: batter cancel-jobs --contains '<folder_listed_above>'")


@cli.command("cancel-jobs")
@click.option(
    "--contains",
    "-c",
    required=True,
    help="Cancel all jobs whose SLURM job name contains this substring (match against full 'fep_...').",
)
def cancel_jobs(contains: str):
    """Cancel all SLURM jobs whose names contain ``contains``."""
    try:
        res = subprocess.run(
            ["squeue", "--user", os.getenv("USER"), "--format=%i %j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
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
        batch = ids[i : i + 30]
        try:
            subprocess.run(["scancel"] + batch, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"Failed to cancel {batch}: {e.stderr}")
