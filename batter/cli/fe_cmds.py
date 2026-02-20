"""Free-energy results commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from batter.api import list_fe_runs, load_fe_run, run_analysis_from_execution
from batter.cli.root import cli


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
@click.argument("run_id", type=str, required=False)
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
    "--analysis-start-step",
    type=int,
    default=None,
    help="First production step (per window) to include in analysis.",
)
@click.option(
    "--n-bootstrap",
    "--n-bootstraps",
    "n_bootstraps",
    type=int,
    default=None,
    help="Number of MBAR bootstrap resamples to use during analysis.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing analysis results when present.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Logging level for analysis stage.",
)
def fe_analyze(
    work_dir: Path,
    run_id: str | None,
    ligand: str | None,
    workers: int | None,
    raise_on_error: bool,
    analysis_start_step: int | None,
    n_bootstraps: int | None,
    overwrite: bool,
    log_level: str = "INFO",
) -> None:
    """
    Re-run the FE analysis stage for stored execution(s).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    if run_id:
        run_ids = [run_id]
    else:
        runs_root = work_dir / "executions"
        if not runs_root.is_dir():
            raise click.ClickException(f"No executions found under {work_dir}.")
        run_ids = sorted([p.name for p in runs_root.iterdir() if p.is_dir()])
        if not run_ids:
            raise click.ClickException(f"No executions found under {work_dir}.")
        logger.info(f"No run_id provided; analyzing all {len(run_ids)} available runs.")

    failed_runs: list[tuple[str, str]] = []
    for rid in run_ids:
        try:
            run_analysis_from_execution(
                work_dir,
                rid,
                ligand=ligand,
                n_workers=workers,
                analysis_start_step=analysis_start_step,
                n_bootstraps=n_bootstraps,
                overwrite=overwrite,
                raise_on_error=raise_on_error,
            )
        except Exception as exc:
            if raise_on_error:
                raise click.ClickException(str(exc))
            failed_runs.append((rid, str(exc)))
            logger.error(f"Analysis failed for run '{rid}': {exc}")

    if failed_runs:
        click.secho(
            "Analysis finished with failures: "
            + ", ".join(f"{rid} ({msg})" for rid, msg in failed_runs),
            fg="yellow",
        )
    else:
        target = run_id or f"{len(run_ids)} run(s)"
        click.echo(
            f"Analysis run finished for '{target}'"
            f"{' (ligand ' + ligand + ')' if ligand else ''}."
        )


def _resolve_ligand_analysis_path(
    ligand_dir: Path,
) -> tuple[Path, Path, str, Path | None, str | None]:
    """
    Resolve FE/ligand paths and optional execution context.

    Returns
    -------
    tuple
        ``(fe_dir, lig_root, ligand_name, work_dir, run_id)`` where ``work_dir`` and
        ``run_id`` are ``None`` when the folder is outside the portable execution layout.
    """
    target = ligand_dir.resolve()
    if target.name == "fe":
        fe_dir = target
        lig_root = target.parent
    else:
        fe_dir = target / "fe"
        lig_root = target

    if not fe_dir.is_dir():
        raise click.ClickException(
            f"Expected FE folder at '{fe_dir}'. "
            "Pass a ligand directory containing 'fe/' or the 'fe/' directory itself."
        )

    run_dir = next(
        (p for p in (lig_root, *lig_root.parents) if p.parent.name == "executions"),
        None,
    )

    if run_dir is None:
        return fe_dir, lig_root, lig_root.name, None, None

    sims_root = run_dir / "simulations"
    try:
        rel = lig_root.relative_to(sims_root)
    except ValueError:
        # Path is under executions/<run> but not simulations; run direct in-place.
        return fe_dir, lig_root, lig_root.name, None, None

    if not rel.parts:
        return fe_dir, lig_root, lig_root.name, None, None

    if rel.parts[0] == "transformations" and len(rel.parts) >= 2:
        ligand_name = rel.parts[1]
    else:
        ligand_name = rel.parts[0]

    work_dir = run_dir.parent.parent
    return fe_dir, lig_root, ligand_name, work_dir, run_dir.name


def _analysis_outputs_present(fe_root: Path) -> bool:
    return (fe_root / "Results" / "Results.dat").exists() and (
        fe_root / "analyze.ok"
    ).exists()


def _clear_analysis_outputs(fe_root: Path) -> None:
    import shutil

    shutil.rmtree(fe_root / "Results", ignore_errors=True)
    (fe_root / "analyze.ok").unlink(missing_ok=True)


def _run_in_place_ligand_analysis(system, params: dict[str, object]) -> None:
    from batter.exec.handlers.fe_analysis import analyze_handler
    from batter.pipeline.step import Step

    analyze_handler(Step(name="analyze"), system, params)


@fe.command("ligand-analyze")
@click.argument(
    "ligand_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
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
    "--analysis-start-step",
    type=int,
    default=None,
    help="First production step (per window) to include in analysis.",
)
@click.option(
    "--n-bootstrap",
    "--n-bootstraps",
    "n_bootstraps",
    type=int,
    default=None,
    help="Number of MBAR bootstrap resamples to use during analysis.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing analysis results when present.",
)
def fe_ligand_analyze(
    ligand_dir: Path,
    workers: int | None,
    raise_on_error: bool,
    analysis_start_step: int | None,
    n_bootstraps: int | None,
    overwrite: bool,
) -> None:
    """
    Re-run FE analysis for exactly one ligand folder.

    The folder must contain an ``fe/`` directory. When it is under
    ``.../executions/<run_id>/simulations/...`` BATTER also updates run-scoped
    FE records; otherwise analysis runs in-place for that ligand folder only.
    """
    fe_dir, lig_root, ligand_name, work_dir, run_id = _resolve_ligand_analysis_path(
        ligand_dir
    )

    if work_dir is not None and run_id is not None:
        try:
            run_analysis_from_execution(
                work_dir,
                run_id,
                ligand=ligand_name,
                n_workers=workers,
                analysis_start_step=analysis_start_step,
                n_bootstraps=n_bootstraps,
                overwrite=overwrite,
                raise_on_error=raise_on_error,
            )
        except Exception as exc:
            raise click.ClickException(str(exc))

        click.echo(
            f"Analysis run finished for ligand '{ligand_name}' in run '{run_id}'."
        )
        return

    # Fallback: run in-place when the folder is outside portable execution layout.
    from batter.systems.core import SimSystem, SystemMeta

    if not overwrite and _analysis_outputs_present(fe_dir):
        click.echo(
            f"Skipping analysis for '{ligand_name}' "
            "(results already exist; overwrite=False)."
        )
        return
    if overwrite:
        _clear_analysis_outputs(fe_dir)

    meta = {"ligand": ligand_name}
    if "~" in ligand_name or lig_root.parent.name == "transformations":
        meta["mode"] = "RBFE"

    params: dict[str, object] = {}
    if workers is not None:
        params["analysis_n_workers"] = workers
        params["n_workers"] = workers
    if analysis_start_step is not None:
        params["analysis_start_step"] = int(analysis_start_step)
    if n_bootstraps is not None:
        params["n_bootstraps"] = int(n_bootstraps)

    system = SimSystem(
        name=ligand_name,
        root=lig_root,
        meta=SystemMeta.from_mapping(meta),
    )

    try:
        _run_in_place_ligand_analysis(system, params)
    except Exception as exc:
        if raise_on_error:
            raise click.ClickException(str(exc))
        logger.error(f"Analysis failed for '{ligand_name}': {exc}")
        return

    click.echo(
        f"Analysis run finished for ligand '{ligand_name}' at '{lig_root}'."
    )
