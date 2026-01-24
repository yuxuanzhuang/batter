"""Root CLI group and shared top-level commands."""

from __future__ import annotations

from pathlib import Path

import click

from batter.api import __version__
from batter.utils.slurm_templates import seed_default_headers


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
        click.echo(
            f"No headers copied; existing headers already present under {dest_dir}."
        )
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
