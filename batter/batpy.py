import click
from loguru import logger
import glob as glob
import os as os
import shutil as shutil
import signal as signal
import sys as sys
from batter.batter import System
from functools import wraps
from batter.utils.slurm_job import SLURMJob

import tempfile

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def slurm_submit_or_run(default_partition="rondror",
                        default_project=None):
    def decorator(func):
        @click.option("--submit", is_flag=True, default=False, help="Submit as SLURM job instead of running locally.")
        @click.option("--partition", default=default_partition, help="SLURM partition (used only if --submit is set).")
        @click.option("--project", default=default_project, help="SLURM project (used only if --submit is set).")
        @wraps(func)
        def wrapper(*args, submit, partition, project, **kwargs):
            if submit:
                # Create a temporary script to run the same command
                args = sys.argv.copy()

                # Replace --submit and filter out --partition/--project
                filtered_args = []
                skip_next = False
                for i, arg in enumerate(args):
                    if arg == "--submit":
                        continue
                    elif arg in ("--partition", "--project"):
                        skip_next = True  # skip this and the next value
                        continue
                    elif skip_next:
                        skip_next = False
                        continue
                    elif args[i - 1] in ("--inputs", "-i"):
                        filtered_args.append(os.path.abspath(arg))
                    else:
                        filtered_args.append(arg)

                command = f"{sys.executable} {' '.join(filtered_args)}"
                command = command.replace(" --submit", "")
                
                if partition:
                    command = command.replace(f" --partition {partition}", "")
                if project:
                    command = command.replace(f" --project {project}", "")
                slurm_lines = [
                    f"#SBATCH --job-name={func.__name__}",
                    f"#SBATCH --partition={partition}",
                    f"#SBATCH --output=slurm-%j.out",
                    f"#SBATCH --error=slurm-%j.err",
                    "#SBATCH --time=02:00:00",
                    "#SBATCH --nodes=1",
                    "#SBATCH --cpus-per-task=1",
                    "#SBATCH --mem=200G",
                ]
                if project:
                    slurm_lines.append(f"#SBATCH -A {project}")

                # write to ~/.batter_job folder
                os.makedirs(os.path.expanduser("~/.batter_job"), exist_ok=True)

                with tempfile.NamedTemporaryFile(
                    mode='w', dir=os.path.expanduser("~/.batter_job"), delete=False, suffix=".sh"
                ) as f:
                    f.write(f"""#!/bin/bash
{os.linesep.join(slurm_lines)}
echo "Running submitted SLURM job: {func.__name__}"
{command}
""")
                    script_path = f.name

                logger.info(f"Created a SLURM job with script: {script_path}")
                slurm_job = SLURMJob(script_path, jobname=func.__name__)
                slurm_job.submit()
                click.echo(f"Submitted SLURM job for '{func.__name__}' with script: {script_path}")
            else:
                func(*args, **kwargs)
        return wrapper
    return decorator


@click.command(
    name="copy_system",
    help="Copy the system from system_path to new_system_path.",
)
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input system path",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output system path",
)
@click.option(
    "--only_equil/--no-only_equil",
    "-oe",
    default=True,
    help="Only copy the equilibration part of the system (default: enabled)",
)
@click.option(
    "--symlink/--no-symlink",
    "-s",
    default=True,
    help="Use symlinks instead of copying files (default: enabled)",
)
def copy_system(input, output, only_equil, symlink):
    """
    Copy the system from system_path to new_system_path.
    """
    system = System(input)
    system.copy_2_new_folder(output, only_equil=only_equil, symlink=symlink)
    logger.info(f"System copied to {output}")


@click.command(
    name="gather",
    help="Analyze systems and gather results, optionally submitting via SLURM."
)
@click.option(
    "--inputs", "-i",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    multiple=True,
    help="Paths to input systems (can be specified multiple times)."
)
@click.option(
    "--sim-range", "-r",
    type=(int, int),
    default=(0, -1),
    help="Simulation index range to analyze as (start, end). Default: (0, -1) for all."
)
@click.option(
    "--load", "-l",
    is_flag=True,
    default=False,
    help="If set, load existing results instead of performing new analysis."
)
@click.option(
    "--n-workers", "-n",
    type=int,
    default=64,
    help="Number of workers to use for analysis (default: 64)."
)
@slurm_submit_or_run()
def gather(inputs, sim_range=(0, -1), load=False, n_workers=64):
    """
    Analyze the specified systems and gather results.

    This command can either run locally or submit as a SLURM job using the --submit flag.
    """
    for input_path in inputs:
        logger.info(f"Processing system: {input_path}")
        system = System(input_path)
        system.n_workers = n_workers
        system.analysis_new(load=load, check_finished=False, sim_range=sim_range)
        logger.info(f"Finished processing system: {input_path}")