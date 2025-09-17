"""
Run the ABFE pipeline for ligand–protein systems in a lipid bilayer.

Arguments:
  --input       Path to the system JSON file (from prepare_template.ipynb).
  --param       Path to the ABFE input configuration file.
  --replicate   Replica number or identifier for the simulation.
  --ligand-ff   Ligand force field (gaff2 or openff-2.2.1).
  --overwrite   Overwrite existing outputs.
  --only-equil  Only run the equilibration step (passed to only_fe_preparation).
  --dry-run     Prepare but don't actually run jobs.

Notes:
- Paths inside the system JSON (protein/system/ligand JSON) are resolved
  relative to the JSON file’s directory.
- This script is mainly run in local cluster to generate equilbration with --only-equil.
- if it is used without --only-equil, it will run the full pipeline.
"""

import json
from pathlib import Path
import sys

import click

from batter import MABFESystem  # , RBFESystem  # Uncomment if you need it


def _resolve(p: str | Path, base: Path) -> Path:
    """Resolve path p relative to base if not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (base / p)

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--input", "input_json",
              required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to the system JSON file.")
@click.option("--param", "param_file",
              required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to the ABFE input configuration file.")
@click.option("--replicate", required=True, type=str,
              help="Replica identifier (e.g., 1, A, r1).")
@click.option("--ligand-ff", default="gaff2",
              type=click.Choice(["gaff2", "openff-2.2.1"], case_sensitive=False),
              show_default=True, help="Ligand force field.")
@click.option("--overwrite", is_flag=True, default=False, show_default=True,
              help="Overwrite existing output folders/files.")
@click.option("--only-equil", is_flag=True, default=False, show_default=True,
              help="Only run the equilibration step (passed to only_fe_preparation).")
@click.option("--dry-run", is_flag=True, default=False, show_default=True,
              help="Prepare pipeline / submit jobs but do not execute runs.")
def run_pipeline(input_json: Path,
                 param_file: Path,
                 replicate: str,
                 ligand_ff: str,
                 overwrite: bool,
                 only_equil: bool,
                 dry_run: bool) -> None:
    """Run the ABFE pipeline."""
    base_dir = input_json.parent.resolve()

    # ---- Load system JSON ----
    try:
        system_data = json.loads(input_json.read_text())
    except Exception as e:
        click.echo(f"[ERROR] Failed to read system JSON: {input_json}\n{e}", err=True)
        sys.exit(1)

    # Required keys
    required = ["protein", "protein_input", "system_input", "anchor_atoms", "ligand_input"]
    missing = [k for k in required if k not in system_data]
    if missing:
        click.echo(f"[ERROR] System JSON missing keys: {', '.join(missing)}", err=True)
        sys.exit(1)

    protein_name = system_data["protein"]
    protein_file = _resolve(system_data["protein_input"], base_dir)
    system_file = _resolve(system_data["system_input"], base_dir)
    anchor_atoms = system_data["anchor_atoms"]
    extra_restraints = system_data.get("extra_restraints", None)
    conformational_restraints = system_data.get("conformational_restraints", None)

    ligand_json_path = _resolve(system_data["ligand_input"], base_dir)
    try:
        ligand_files = json.loads(ligand_json_path.read_text())
    except Exception as e:
        click.echo(f"[ERROR] Failed to read ligand JSON: {ligand_json_path}\n{e}", err=True)
        sys.exit(1)

    # Resolve ligand file paths relative to ligand_json_path
    ligand_base = ligand_json_path.parent
    ligand_files = {k: str(_resolve(v, ligand_base)) for k, v in ligand_files.items()}

    # ---- Output folder ----
    output_folder = Path(f"{protein_name}/{replicate}").resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # ---- Summary ----
    click.echo("Running ABFE pipeline…\n")
    click.echo(f" System JSON         : {input_json}")
    click.echo(f" ABFE param file     : {param_file}")
    click.echo(f" Protein name        : {protein_name}")
    click.echo(f" Replicate           : {replicate}")
    click.echo(f" Ligand force field  : {ligand_ff}")
    click.echo(f" Output folder       : {output_folder}")
    click.echo(f" Protein file        : {protein_file}")
    click.echo(f" System file         : {system_file}")
    click.echo(f" #Ligands            : {len(ligand_files)}")
    click.echo(f" Anchor atoms        : {anchor_atoms}")
    click.echo(f" Extra restraints    : {extra_restraints}")
    click.echo(f" Conform. restraints : {conformational_restraints}")
    click.echo(f" Overwrite           : {overwrite}")
    click.echo(f" Only equilibration  : {only_equil}")
    click.echo(f" Dry run             : {dry_run}\n")

    for pth, label in [(protein_file, "protein_input"),
                       (system_file, "system_input"),
                       (ligand_json_path, "ligand_input")]:
        if not Path(pth).exists():
            click.echo(f"[ERROR] {label} does not exist: {pth}", err=True)
            sys.exit(1)

    system = MABFESystem(folder=str(output_folder))

    # Create system if equilibration is not yet prepared or overwrite is requested
    if overwrite or not getattr(system, "_eq_prepared", False):
        system.create_system(
            system_name=protein_name,
            protein_input=str(protein_file),
            system_topology=str(system_file),
            ligand_paths=ligand_files,
            ligand_ff=ligand_ff,
            overwrite=overwrite,
            retain_lig_prot=True,
            lipid_mol=["POPC"],
            anchor_atoms=anchor_atoms,
        )

    # Run pipeline
    system.run_pipeline(
        input_file=str(param_file),
        only_fe_preparation=only_equil,
        dry_run=dry_run,
        extra_restraints=extra_restraints,
        extra_conformation_restraints=conformational_restraints,
    )


if __name__ == "__main__":
    run_pipeline()