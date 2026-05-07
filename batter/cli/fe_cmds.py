"""Free-energy results commands."""

from __future__ import annotations

import sys
import re
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from batter.analysis.cinnabar import (
    build_batter_rbfe_cinnabar,
    build_batter_rbfe_cinnabar_by_run,
    summarize_directionality,
    write_cinnabar_outputs,
)
from batter.api import list_fe_runs, load_fe_run, run_analysis_from_execution
from batter.cli.root import cli
from batter.runtime.fe_repo import FEResultsRepository
from batter.runtime.portable import ArtifactStore


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
        "include_in_analysis",
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


def _selection_indices(selection: str, *, max_index: int) -> list[int]:
    text = selection.strip().lower()
    if text == "all":
        return list(range(max_index))
    out: list[int] = []
    for part in re.split(r"[\s,]+", text):
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid selection '{part}'. Use row numbers or 'all'."
            ) from exc
        if idx < 1 or idx > max_index:
            raise click.ClickException(f"Selection {idx} is outside 1..{max_index}.")
        out.append(idx - 1)
    return sorted(set(out))


@fe.command("analysis-inclusion")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--run-id",
    "run_ids",
    multiple=True,
    help="Only show rows from the selected run id(s).",
)
def fe_analysis_inclusion(work_dir: Path, run_ids: tuple[str, ...]) -> None:
    """Interactively enable or disable FE rows for aggregate analysis."""
    repo = FEResultsRepository(ArtifactStore(work_dir))
    try:
        df = repo.index().copy()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if df.empty:
        click.secho("No FE runs found.", fg="yellow")
        return

    if run_ids:
        requested = {str(run_id) for run_id in run_ids}
        df = df.loc[df["run_id"].astype(str).isin(requested)].copy()
        if df.empty:
            raise click.ClickException(
                "No FE rows remain after filtering for run id(s): "
                + ", ".join(sorted(requested))
            )

    df = df.reset_index(drop=True)

    def _display() -> None:
        show_cols = [
            "row",
            "include_in_analysis",
            "run_id",
            "ligand",
            "original_name",
            "status",
            "protocol",
            "total_dG",
            "total_se",
        ]
        show = df.copy()
        show.insert(0, "row", range(1, len(show) + 1))
        for col in show_cols:
            if col not in show.columns:
                show[col] = ""
        for col in ("total_dG", "total_se"):
            show[col] = show[col].map(
                lambda value: ""
                if pd.isna(value) or str(value).strip() == ""
                else f"{float(value):.3f}"
            )
        click.echo(show[show_cols].to_string(index=False))

    click.echo(
        "Rows with include_in_analysis=False are skipped by Cinnabar and other aggregate analyses."
    )
    _display()
    click.echo("Commands: 'disable 1,3', 'enable 2', 'disable all', 'show', 'quit'.")

    while True:
        command = click.prompt("analysis-inclusion", default="quit", show_default=False)
        command = command.strip()
        if not command:
            continue
        lower = command.lower()
        if lower in {"q", "quit", "exit", "done"}:
            break
        if lower == "show":
            _display()
            continue
        parts = command.split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() not in {"enable", "disable"}:
            click.secho(
                "Expected 'enable <rows>', 'disable <rows>', 'show', or 'quit'.",
                fg="yellow",
            )
            continue
        include = parts[0].lower() == "enable"
        try:
            indices = _selection_indices(parts[1], max_index=len(df))
        except click.ClickException as exc:
            click.secho(str(exc), fg="red")
            continue
        updated = 0
        for idx in indices:
            row = df.iloc[idx]
            updated += repo.set_analysis_inclusion(
                run_id=str(row["run_id"]),
                ligand=str(row["ligand"]),
                include=include,
                analysis_start_step=FEResultsRepository._normalize_optional_int(
                    row.get("analysis_start_step")
                ),
                n_bootstraps=FEResultsRepository._normalize_n_bootstraps(
                    row.get("n_bootstraps")
                ),
            )
            df.loc[idx, "include_in_analysis"] = include
        click.echo(
            f"{'Enabled' if include else 'Disabled'} {updated} index row(s) for aggregate analysis."
        )
        _display()


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


@fe.command("cinnabar")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--run-id",
    "run_ids",
    multiple=True,
    help="Select one or more stored BATTER run IDs. Defaults to all available RBFE rows.",
)
@click.option(
    "--ligand",
    "ligands",
    multiple=True,
    help="Restrict to specific RBFE edge labels, e.g. 'LIGA~LIGB'.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for the converted Cinnabar bundle(s). Defaults to <WORK_DIR>/results/cinnabar.",
)
@click.option(
    "--combine-runs/--split-runs",
    default=True,
    help="Aggregate selected runs into one FEMap, or emit one bundle per run.",
)
@click.option(
    "--combine-by-run-first/--pool-all-measurements",
    default=True,
    help="Collapse repeated measurements within each run before combining across runs.",
)
@click.option(
    "--merge-directions/--split-directions",
    "merge_bidirectional",
    default=True,
    help="Merge A~B and B~A into one canonical edge, or keep them as separate directional transformations.",
)
@click.option(
    "--uncertainty-mode",
    type=click.Choice(["ivw", "sample", "max"], case_sensitive=False),
    default="max",
    show_default=True,
    help="How to combine repeated uncertainties.",
)
@click.option(
    "--edge-separator",
    default="~",
    show_default=True,
    help="Separator used in BATTER RBFE pair labels.",
)
@click.option(
    "--source",
    default="BATTER_RBFE",
    show_default=True,
    help="Source label stored on computational Cinnabar measurements.",
)
@click.option(
    "--experimental-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional CSV of experimental absolute affinities to merge into the FEMap.",
)
@click.option(
    "--exp-ligand-column",
    default="ligand",
    show_default=True,
    help="Ligand-label column in the experimental CSV.",
)
@click.option(
    "--exp-abfe-column",
    default="abfe",
    show_default=True,
    help="Absolute affinity column in the experimental CSV.",
)
@click.option(
    "--exp-error-column",
    default=None,
    help="Optional experimental uncertainty column.",
)
@click.option(
    "--exp-status-column",
    default=None,
    help="Optional experimental status column.",
)
@click.option(
    "--exp-success-value",
    default="success",
    show_default=True,
    help="Value in the experimental status column that marks usable rows.",
)
@click.option(
    "--exp-temperature-column",
    default=None,
    help="Optional experimental temperature column (Kelvin).",
)
@click.option(
    "--exp-source",
    default="experiment",
    show_default=True,
    help="Source label stored on experimental Cinnabar measurements.",
)
@click.option(
    "--exp-value-unit",
    default="kcal/mol",
    show_default=True,
    help="Unit for experimental absolute values.",
)
@click.option(
    "--exp-error-unit",
    default=None,
    help="Unit for experimental uncertainties. Defaults to --exp-value-unit.",
)
@click.option(
    "--write-plots/--no-write-plots",
    default=True,
    help="Attempt to write Cinnabar plots when plotting support is available.",
)
@click.option(
    "--cycle-closure/--no-cycle-closure",
    "write_cycle_closure",
    default=True,
    show_default=True,
    help="Also write RBFE cycle-closure correction tables into the Cinnabar bundle.",
)
@click.option(
    "--absolute-offset",
    type=float,
    default=0.0,
    show_default=True,
    help="Constant offset (kcal/mol) added to computed absolute ΔG values in the sorted absolute-energy plot.",
)
def fe_cinnabar(
    work_dir: Path,
    run_ids: tuple[str, ...],
    ligands: tuple[str, ...],
    out_dir: Path | None,
    combine_runs: bool,
    combine_by_run_first: bool,
    merge_bidirectional: bool,
    uncertainty_mode: str,
    edge_separator: str,
    source: str,
    experimental_csv: Path | None,
    exp_ligand_column: str,
    exp_abfe_column: str,
    exp_error_column: str | None,
    exp_status_column: str | None,
    exp_success_value: str,
    exp_temperature_column: str | None,
    exp_source: str,
    exp_value_unit: str,
    exp_error_unit: str | None,
    write_plots: bool,
    write_cycle_closure: bool,
    absolute_offset: float,
) -> None:
    """Convert stored BATTER RBFE results into Cinnabar FEMap-ready outputs."""
    exp_df = None
    if experimental_csv is not None:
        try:
            exp_df = pd.read_csv(experimental_csv)
        except Exception as exc:
            raise click.ClickException(
                f"Failed to read experimental CSV '{experimental_csv}': {exc}"
            ) from exc

    output_root = out_dir or (work_dir / "results" / "cinnabar")

    common_kwargs = {
        "work_dir": work_dir,
        "run_ids": run_ids or None,
        "ligands": ligands or None,
        "edge_separator": edge_separator,
        "uncertainty_mode": uncertainty_mode.lower(),
        "combine_by_run_first": combine_by_run_first,
        "merge_bidirectional": merge_bidirectional,
        "experimental_df": exp_df,
        "exp_ligand_column": exp_ligand_column,
        "exp_abfe_column": exp_abfe_column,
        "exp_error_column": exp_error_column,
        "exp_status_column": exp_status_column,
        "exp_success_value": exp_success_value,
        "exp_temperature_column": exp_temperature_column,
        "source": source,
        "exp_source": exp_source,
        "exp_value_unit": exp_value_unit,
        "exp_error_unit": exp_error_unit,
    }

    try:
        if combine_runs:
            result = build_batter_rbfe_cinnabar(**common_kwargs)
            outputs = write_cinnabar_outputs(
                result,
                output_root,
                method_name="BATTER",
                target_name=work_dir.name,
                write_plots=write_plots,
                write_cycle_closure=write_cycle_closure,
                absolute_offset=absolute_offset,
            )
            if getattr(result, "absolute_warning", None):
                click.secho(str(result.absolute_warning), fg="yellow")
            if not merge_bidirectional and hasattr(result, "edge_summary"):
                directionality = summarize_directionality(result.edge_summary)
                if directionality["n_reciprocal_pairs"] == 0:
                    click.secho(
                        "Split-direction export requested, but the stored RBFE results "
                        "contain no reciprocal A~B/B~A transformations. "
                        "The network will still show one arrow per stored transformation.",
                        fg="yellow",
                    )
                else:
                    click.echo(
                        "Split-direction export retained "
                        f"{directionality['n_directional_edges']} directional edges across "
                        f"{directionality['n_reciprocal_pairs']} reciprocal ligand pair(s)."
                    )
            click.echo(
                f"Wrote combined Cinnabar bundle to {output_root} "
                f"({len(outputs)} files tracked)."
            )
            return

        bundles = build_batter_rbfe_cinnabar_by_run(**common_kwargs)
        if not bundles:
            raise click.ClickException("No per-run RBFE bundles were generated.")

        split_direction_stats: list[dict[str, object]] = []
        for run_id, result in bundles.items():
            run_out_dir = output_root / run_id
            write_cinnabar_outputs(
                result,
                run_out_dir,
                method_name="BATTER",
                target_name=f"{work_dir.name}:{run_id}",
                write_plots=write_plots,
                write_cycle_closure=write_cycle_closure,
                absolute_offset=absolute_offset,
            )
            if getattr(result, "absolute_warning", None):
                click.secho(f"[{run_id}] {result.absolute_warning}", fg="yellow")
            if not merge_bidirectional and hasattr(result, "edge_summary"):
                stats = summarize_directionality(result.edge_summary)
                stats["run_id"] = run_id
                split_direction_stats.append(stats)
        if not merge_bidirectional and split_direction_stats:
            total_recip = sum(int(item["n_reciprocal_pairs"]) for item in split_direction_stats)
            total_dir = sum(int(item["n_directional_edges"]) for item in split_direction_stats)
            if total_recip == 0:
                click.secho(
                    "Split-direction export requested, but none of the selected runs contain "
                    "reciprocal A~B/B~A transformations. The network plots will remain one "
                    "arrow per stored transformation.",
                    fg="yellow",
                )
            else:
                click.echo(
                    "Split-direction export retained "
                    f"{total_dir} directional edges across {total_recip} reciprocal ligand pair(s)."
                )
        click.echo(
            f"Wrote {len(bundles)} per-run Cinnabar bundle(s) under {output_root}."
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


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


def _infer_analysis_timing_from_fe(fe_dir: Path) -> tuple[float, int, float]:
    """
    Best-effort timing inference for in-place analysis without run config.

    Returns ``(dt, ntwx, temperature)`` and falls back to
    ``(0.004, 0, 298.15)`` when unavailable.
    """
    import re

    dt: float | None = None
    ntwx: int | None = None
    temperature: float | None = None

    for comp_dir in sorted([p for p in fe_dir.iterdir() if p.is_dir()]):
        for win_dir in sorted([p for p in comp_dir.iterdir() if p.is_dir()]):
            if win_dir.name.endswith("-1"):
                continue
            candidates = [win_dir / fname for fname in (
                "mdin-template",
                "mdin.in",
                "mdin-00",
                "mdin-01",
                "mdin",
            )]
            candidates.extend(sorted(win_dir.glob("mdin-*.out")))
            candidates.extend(sorted(win_dir.glob("md-*.out")))

            for candidate in candidates:
                if not candidate.is_file():
                    continue
                text = candidate.read_text(errors="ignore")
                if dt is None:
                    m_dt = re.search(r"\bdt\s*=\s*([0-9]*\.?[0-9]+)", text)
                    if m_dt:
                        try:
                            dt = float(m_dt.group(1))
                        except ValueError:
                            pass
                if ntwx is None:
                    m_ntwx = re.search(r"\bntwx\s*=\s*([0-9]+)", text)
                    if m_ntwx:
                        try:
                            ntwx = int(m_ntwx.group(1))
                        except ValueError:
                            pass
                if temperature is None:
                    m_t = re.search(r"\btemp0\s*=\s*([0-9]*\.?[0-9]+)", text)
                    if m_t is None:
                        m_t = re.search(r"\btempi\s*=\s*([0-9]*\.?[0-9]+)", text)
                    if m_t:
                        try:
                            temperature = float(m_t.group(1))
                        except ValueError:
                            pass
                if dt is not None and ntwx is not None and temperature is not None:
                    return dt, ntwx, temperature

    return (
        (dt if dt is not None and dt > 0 else 0.004),
        (ntwx if ntwx is not None else 0),
        (temperature if temperature is not None and temperature > 0 else 298.15),
    )


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
    dt, ntwx, temperature = _infer_analysis_timing_from_fe(fe_dir)
    params["dt"] = dt
    params["ntwx"] = ntwx
    params["temperature"] = temperature

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
