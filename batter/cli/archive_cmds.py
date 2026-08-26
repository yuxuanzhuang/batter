"""Archive-oriented CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from batter.cli.root import cli
from batter.runtime.archive import (
    build_execution_archive_plan,
    write_execution_archive_plan,
)


@cli.command("archive")
@click.argument(
    "executions",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("batter_archive.tar.gz"),
    show_default=True,
    help="Output tar archive path. Compression is inferred from the suffix.",
)
@click.option(
    "--include",
    "include_paths",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Additional reproducibility input file or directory to include.",
)
@click.option(
    "--root-name",
    default="batter_archive",
    show_default=True,
    help="Top-level directory name inside the archive.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Overwrite the output archive if it already exists.",
)
def archive_cmd(
    executions: tuple[Path, ...],
    out: Path,
    include_paths: tuple[Path, ...],
    root_name: str,
    force: bool,
) -> None:
    """Create a compact archive from one or more BATTER executions."""
    try:
        click.echo("Planning archive contents...")
        plan = build_execution_archive_plan(
            executions,
            out,
            include=include_paths,
            root_name=root_name,
            overwrite=force,
        )
        total_members = len(plan.members) + 1
        with click.progressbar(
            length=total_members,
            label="Writing archive",
            item_show_func=lambda item: item or "",
        ) as bar:
            result = write_execution_archive_plan(
                plan,
                progress=lambda member: bar.update(1, current_item=member),
            )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Wrote {result.archive_path}")
    click.echo(f"Archived {len(result.executions)} execution(s).")
    missing = [
        summary
        for summary in result.executions
        if not summary.results_dirs and not summary.associated_results
    ]
    if missing:
        click.echo(
            f"Note: {len(missing)} execution(s) had no results/ directories."
        )
    click.echo(f"Archive entries written: {len(result.members)}")
