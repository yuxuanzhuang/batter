"""Analysis-oriented CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from batter.cli.root import cli
from batter.exec.handlers.equil_analysis import (
    discover_equil_analysis_targets,
    run_equil_analysis_for_simulation,
)


@cli.command("simulation-analysis")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--ligand-resname",
    default=None,
    help="Ligand residue name to analyze; inferred from params/equil files by default.",
)
@click.option(
    "--ligand-label",
    default=None,
    help="Ligand label stored in generated records; defaults to ligand metadata or folder name.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Unbound ligand binding-site distance threshold in Angstrom.",
)
@click.option(
    "--hmr/--no-hmr",
    default=None,
    help="Force use of full.hmr.prmtop or full.prmtop; inferred by default.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Refresh existing simulation-analysis, ProLIF, and stable-distance outputs.",
)
def simulation_analysis_cmd(
    path: Path,
    ligand_resname: str | None,
    ligand_label: str | None,
    threshold: float | None,
    hmr: bool | None,
    force: bool,
) -> None:
    """Run equilibration analysis for an execution or one ligand simulation folder."""
    targets = discover_equil_analysis_targets(path)
    if not targets:
        raise click.ClickException(
            "No simulation folders found. Pass an execution folder containing "
            "simulations/<ligand>/equil or one simulations/<ligand> folder."
        )

    click.echo(f"Found {len(targets)} simulation target(s).")
    failures: list[tuple[Path, Exception]] = []
    for target in targets:
        click.echo(f"Analyzing {target}")
        try:
            result = run_equil_analysis_for_simulation(
                target,
                residue_name=ligand_resname,
                ligand_label=ligand_label,
                threshold=threshold,
                hmr=hmr,
                force=force,
            )
        except Exception as exc:
            failures.append((target, exc))
            click.secho(f"  failed: {exc}", fg="red")
            continue
        artifact_names = ", ".join(sorted(result.artifacts)) or "none"
        click.echo(f"  artifacts: {artifact_names}")

    if failures:
        detail = "\n".join(f"- {target}: {exc}" for target, exc in failures)
        raise click.ClickException(
            f"Simulation analysis failed for {len(failures)} target(s):\n{detail}"
        )
