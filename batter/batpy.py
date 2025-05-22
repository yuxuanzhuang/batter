import click
from loguru import logger
import pickle
import pandas as pd
import glob as glob
import os as os
import re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
import numpy as np
import json
# from batter.utils.utils import run_with_log, antechamber, tleap, cpptraj
from batter.batter import System
import MDAnalysis as mda
# ignore UserWarning from MDAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
