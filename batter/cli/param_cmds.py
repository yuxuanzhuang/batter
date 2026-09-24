"""Standalone ligand-parameterization CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from batter.cli.root import cli


def _batch_ligand_process(*args, **kwargs):
    """Load the optional ligand toolchain only when this command is used."""
    from batter.param.ligand import batch_ligand_process

    return batch_ligand_process(*args, **kwargs)


@cli.command("param-ligand")
@click.argument(
    "ligand_sdf",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("ligand_param"),
    show_default=True,
    help="Root directory for the content-addressed ligand parameter store.",
)
@click.option(
    "--ligand-ff",
    default="openff-2.3.0",
    show_default=True,
    help="Ligand force field (for example gaff2 or openff-2.3.0).",
)
@click.option(
    "--charge-method",
    default="openff-gnn-am1bcc-1.0.0.pt",
    show_default=True,
    help="Partial-charge method used by ligand parameterization.",
)
@click.option(
    "--retain-h/--no-retain-h",
    default=True,
    show_default=True,
    help="Retain explicit hydrogens from the input SDF.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Rebuild the matching parameter cache even if it is complete.",
)
def param_ligand_cmd(
    ligand_sdf: Path,
    output_dir: Path,
    ligand_ff: str,
    charge_method: str,
    retain_h: bool,
    overwrite: bool,
) -> None:
    """Parameterize one ligand SDF into a reusable ligand_param store.

    The generated AMBER-compatible files are written below OUTPUT in a
    content-addressed subdirectory. A protonated 3D SDF with explicit
    hydrogens is recommended when retaining input hydrogens.
    """
    if ligand_sdf.suffix.lower() != ".sdf":
        raise click.ClickException("Ligand input must be an SDF file (.sdf).")

    output_dir = output_dir.resolve()
    try:
        hashes, _metadata = _batch_ligand_process(
            {ligand_sdf.stem: str(ligand_sdf)},
            output_path=output_dir,
            retain_lig_prot=retain_h,
            ligand_ff=ligand_ff,
            charge_method=charge_method,
            overwrite=overwrite,
            run_with_slurm=False,
            on_failure="raise",
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if len(hashes) != 1:
        raise click.ClickException(
            "Ligand parameterization did not produce exactly one parameter set."
        )

    parameter_dir = output_dir / hashes[0]
    click.echo(f"Ligand parameters written to {parameter_dir}")
