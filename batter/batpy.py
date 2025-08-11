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
from batter.utils.utils import natural_keys
import tempfile
import subprocess
import pandas as pd
from textwrap import wrap


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
        system.analysis(load=load, check_finished=False, sim_range=sim_range)
        logger.info(f"Finished processing system: {input_path}")

@click.command()
@click.option("--partition", "-p", default=None, help="SLURM partition to report jobs from; if not specified, report all partitions.")
@click.option("--detailed", "-d", is_flag=True, default=False, help="Show detailed job information.")
def report_jobs(partition=None, detailed=False):
    """
    Report the status of SLURM jobs of FEP simulations
    """
    try:
        # Get current user's SLURM jobs
        if partition is None:
            command = ["squeue", "--user", os.getenv("USER"), "--format=%i %j %T"]
        else:
            command = ["squeue", "--partition", partition, "--user", os.getenv("USER"), "--format=%i %j %T"]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to get SLURM job list: {e.stderr}")
        return

    lines = result.stdout.strip().split("\n")
    header, *jobs = lines


    job_info_list = []

    for job_line in jobs:
        jobid, jobname, status = job_line.strip().split(maxsplit=2)

        if jobname.startswith("fep_"):
            full_path = jobname[len("fep_"):]
            # last underscore is the stage
            stage = full_path.rsplit("_", 1)[-1]
            path = full_path.rsplit("_", 1)[0]
            comp_win = path.rsplit("/", 1)[-1]
            comp = comp_win[0]
            win = comp_win[1:]
            pose = path.split("/")[-3]
            system = path.split("/")[:-3]
            
            job_info = {
                "system": "/".join(system),
                "jobid": jobid,
                "pose": pose,
                "comp": comp,
                "win": win,
                "stage": stage,
                "status": status,
            }
            job_info_list.append(job_info)
    
    
    job_df = pd.DataFrame(job_info_list)
    if job_df.empty:
        click.echo("No FEP jobs found.")
        return

    # confirm no duplicate system + pose + comp + win + stage
    duplicates = job_df.duplicated(subset=["system", "pose", "comp", "win", "stage"], keep=False)
    if duplicates.any():
        click.echo(click.style("Warning: Found duplicate jobs with the same system, pose, comp, win, and stage.", fg="yellow"))
        for _, row in job_df[duplicates].iterrows():
            click.echo(f"Duplicate Job ID: {row['jobid']} - System: {row['system']} - Pose: {row['pose']} - Comp: {row['comp']} - Win: {row['win']} - Stage: {row['stage']} - Status: {row['status']}")
        logger.warning("Duplicate jobs detected. Please check the job list.")
        
    # print number of running/pending jobs
    total_jobs = len(job_df)
    running_jobs = job_df[job_df['status'] == 'RUNNING'].shape[0]
    pending_jobs = job_df[job_df['status'] == 'PENDING'].shape[0]
    click.echo(click.style(f"Total jobs: {total_jobs}, Running: {running_jobs}, Pending: {pending_jobs}", bold=True))
    # Group and count RUNNING/PENDING statuses
    for system, system_df in job_df.groupby("system"):
        stage = system_df["stage"].unique()
        click.echo(click.style(f"\nSystem: {system} - Stage: {', '.join(stage)}", bold=True))
        click.echo("-" * 60)
        click.echo(click.style("Pose (PENDING, RUNNING):", bold=True))
        grouped = system_df.groupby("pose")["status"].value_counts().unstack(fill_value=0).reset_index()

        # Ensure 'RUNNING' and 'PENDING' columns always exist
        for col in ["RUNNING", "PENDING"]:
            if col not in grouped.columns:
                grouped[col] = 0

        grouped = grouped.rename(columns={"RUNNING": "running_jobs", "PENDING": "pending_jobs"})
        # sort by pose
        grouped = grouped.sort_values(by="pose", key=lambda x: x.map(natural_keys))

        rows = []
        for _, row in grouped.iterrows():
            pose = row["pose"]
            p = row.get("pending_jobs", 0)
            r = row.get("running_jobs", 0)
            if r > 0:
                r_str = click.style(f"{pose}(P={p},R={r})", fg="green", bold=True)
            else:
                r_str = click.style(f"{pose}(P={p},R={r})", fg="red")

            rows.append(r_str)

        # Print 4 per line
        for i in range(0, len(rows), 4):
            click.echo("   ".join(rows[i:i+4]))

        if detailed:
            for _, row in system_df.iterrows():
                if row['status'] == 'RUNNING':
                    click.echo(click.style(f"Job ID: {row['jobid']} - Pose: {row['pose']} - Comp: {row['comp']} - Win: {row['win']} - Status: {row['status']}", fg="green", bold=True))
                elif row['status'] == 'PENDING':
                    click.echo(click.style(f"Job ID: {row['jobid']} - Pose: {row['pose']} - Comp: {row['comp']} - Win: {row['win']} - Status: {row['status']}", fg="red"))
        click.echo("-" * 60)
    click.echo("If you want to cancel jobs, use 'batter cancel-jobs --name <system_name>' command.")


@click.command()
@click.option("--name", "-n", required=True, help="Path to the system to cancel jobs for.")
def cancel_jobs(name):
    """
    Cancel all SLURM jobs that include the given system path in their job name.
    """
    try:
        # Run the equivalent of: squeue -u $USER --format="%i %j"
        result = subprocess.run(
            ["squeue", "-u", os.getenv("USER"), "--format=%i %j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Error querying SLURM: {e.stderr}")
        return

    lines = result.stdout.strip().split("\n")[1:]

    matching_ids = [
        line.split()[0]
        for line in lines
        if name in line
    ]

    if not matching_ids:
        click.echo(f"No jobs found containing '{name}' in job name.")
        return

    click.echo(f"Cancelling {len(matching_ids)} job(s)")

    # cancel 30 jobs at a time
    for i in range(0, len(matching_ids), 30):
        batch = matching_ids[i:i+30]
        #click.echo(f"Cancelling jobs: {', '.join(batch)}")
        try:
            subprocess.run(["scancel"] + batch, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"Failed to cancel jobs: {e.stderr}")