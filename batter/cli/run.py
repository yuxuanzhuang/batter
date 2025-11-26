"""Command-line interface for BATTER."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional

import click
import pandas as pd
import yaml
from loguru import logger

from batter.api import (
    __version__,
    clone_execution,
    list_fe_runs,
    load_fe_run,
    run_analysis_from_execution,
    run_from_yaml,
)
from batter.config.run import RunConfig
from batter.data import job_manager
from batter.utils.slurm_templates import (
    render_slurm_with_header_body,
    seed_default_headers,
)
from batter.utils import natural_keys
from batter.cli.fek import fek_schedule


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="batter")
def cli() -> None:
    """Root command group for BATTER."""
    seed_default_headers()


@cli.command("seed-headers")
@click.option(
    "--dest",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Destination directory for Slurm headers (defaults to ~/.batter).",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Overwrite existing headers if present.",
)
def seed_headers(dest: Path | None, force: bool) -> None:
    """Copy packaged Slurm headers into dest (default: ~/.batter)."""
    copied = seed_default_headers(dest, overwrite=force)
    dest_dir = dest or Path.home() / ".batter"
    if copied:
        click.echo(f"Seeded headers into {dest_dir}:")
        for path in copied:
            click.echo(f"  - {path}")
    else:
        click.echo(f"No headers copied; existing headers already present under {dest_dir}.")
        if not force:
            click.echo("Use --force to overwrite existing header files.")


@cli.command("diff-headers")
@click.option(
    "--dest",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Location of Slurm headers (defaults to ~/.batter).",
)
def diff_headers_cmd(dest: Path | None) -> None:
    """Show differences between user headers and packaged defaults."""
    from batter.utils.slurm_templates import diff_headers as _diff_headers

    diffs = _diff_headers(dest)
    if not diffs:
        click.echo("No headers found.")
        return
    for name, diff in diffs.items():
        click.echo(f"=== {name} ===")
        if not diff:
            click.echo("No differences.")
            continue
        click.echo(diff)


# -------------------------------- run ----------------------------------


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


def _which_batter() -> str:
    """
    Resolve the executable used to invoke ``batter``.

    Returns
    -------
    str
        Shell-escaped token (``batter`` path or ``python -m batter.cli``).
    """
    import shutil

    exe = shutil.which("batter")
    if exe:
        return shlex.quote(exe)
    # last resort: run module (works inside editable installs)
    return shlex.quote(sys.executable) + " -m batter.cli"


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
            only_equil=("1" if only_equil else "0") if only_equil is not None else "",
        )
        base_path = Path(slurm_manager_path) if slurm_manager_path else Path(job_manager)
        tpl_header = base_path.with_suffix(".header")
        tpl_body = base_path.with_suffix(".body")
        manager_code = render_slurm_with_header_body(
            "job_manager.header",
            tpl_header,
            tpl_body,
            {},
            header_root=cfg_for_validation.run.slurm_header_dir,
        )
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


# ---------------------------- free energy results check ------------------------------


@cli.group("fe")
def fe() -> None:
    """Query and inspect free-energy results."""


@fe.command("list")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
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
    List free-energy runs stored within ``WORK_DIR``.

    Parameters
    ----------
    work_dir : Path
        Portable work directory containing the ``results/`` tree.
    fmt : {"table", "json", "csv", "tsv"}
        Output formatting option (defaults to ``"table"``).
    """
    try:
        df = list_fe_runs(work_dir)
    except Exception as e:
        raise click.ClickException(str(e))

    if df.empty:
        click.secho("No FE runs found.", fg="yellow")
        return

    # ensure expected cols exist
    cols = [
        "system_name",
        "run_id",
        "ligand",
        "mol_name",
        "temperature",
        "total_dG",
        "total_se",
        "original_name",
        "status",
        "protocol",
        "created_at",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # parse datetime for stable sort, but keep original text if non-parseable
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = (
        df.assign(_created=created)
        .sort_values("_created", na_position="last")
        .drop(columns=["_created"])
    )
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
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("run_id", type=str)
@click.option(
    "--ligand",
    "-l",
    type=str,
    default=None,
    help="Specify a ligand identifier when multiple records share the same run_id.",
)
def fe_show(work_dir: Path, run_id: str, ligand: str | None) -> None:
    """
    Display a single free-energy record from ``WORK_DIR``.

    Parameters
    ----------
    work_dir : Path
        Portable work directory.
    run_id : str
        Run identifier returned by :func:`fe_list`.
    ligand : str, optional
        Ligand identifier to disambiguate when multiple ligands are stored under the same run_id.
    """
    try:
        rec = load_fe_run(work_dir, run_id, ligand=ligand)
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
        order = [
            c
            for c in ["component", "lam", "dG", "dG_se", "n_samples"]
            if c in df.columns
        ]
        df = df[order + [c for c in df.columns if c not in order]]
        # format numbers
        for col in ["dG", "dG_se"]:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda v: f"{float(v):.3f}" if pd.notna(v) else ""
                )
        with pd.option_context("display.max_columns", None, "display.width", 120):
            click.echo(df.to_string(index=False))
    else:
        click.secho("\n(no per-window data saved)", fg="yellow")


@fe.command("analyze")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("run_id", type=str)
@click.option(
    "--ligand",
    "-l",
    type=str,
    default=None,
    help="Select a single ligand when multiple records exist for the run.",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="Number of local workers to pass to the FE analysis handler.",
)
@click.option(
    "--raise-on-error/--no-raise-on-error",
    default=True,
    help="Whether analysis failures should raise (default) or be logged and skipped.",
)
@click.option(
    "--sim-range",
    type=str,
    default=None,
    help="Subset of lambda windows to analyze, formatted as ``start,end``.",
)
def fe_analyze(
    work_dir: Path,
    run_id: str,
    ligand: str | None,
    workers: int | None,
    raise_on_error: bool,
    sim_range: str | None,
) -> None:
    """
    Re-run the FE analysis stage for a stored execution.
    """
    parsed_range: tuple[int, int] | None = None
    if sim_range:
        parts = sim_range.split(",")
        if len(parts) != 2:
            raise click.ClickException(
                "`--sim-range` expects two comma-separated integers."
            )
        try:
            parsed_range = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise click.ClickException(f"Invalid `--sim-range`: {exc}")

    try:
        run_analysis_from_execution(
            work_dir,
            run_id,
            ligand=ligand,
            n_workers=workers,
            sim_range=parsed_range,
            raise_on_error=raise_on_error,
        )
    except Exception as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"Analysis run finished for '{run_id}'"
        f"{' (ligand ' + ligand + ')' if ligand else ''}."
    )


# ------------------------ execution cloning ---------------------------
@cli.command("clone-exec")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
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
    "--mode",
    type=click.Choice(["copy", "hardlink", "symlink"], case_sensitive=False),
    default="symlink",
    show_default=True,
    help="Copy strategy for cloning files.",
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
    force: bool,
    mode: str,
) -> None:
    """
    Clone an existing execution directory.
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
            work_dir=work_dir,
            src_run_id=src_run_id,
            dst_root=dst_root,
            dst_run_id=dst_run_id,
            mode=mode,
            only_equil=only_equil,
            reset_states=True,
            overwrite=force,
        )
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.secho(
        f"Cloned execution '{src_run_id}' → '{dst_run_id}' under {dst_root}",
        fg="green",
    )


# ----------------------------- check status -------------------------------


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
    if tail.endswith("_eq"):
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


cli.add_command(fek_schedule)
