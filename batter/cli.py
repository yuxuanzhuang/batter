import batter
from batter import batpy

import sys
import click
import glob as glob
import os as os
import re
import shutil as shutil
import signal as signal
import subprocess as sp
import numpy as np
#from batter.utils.utils import run_with_log, antechamber, tleap, cpptraj
#from batter.batter import System
from batter.input_process import get_configure_from_file
from batter.bat_lib import build, setup, analysis, scripts
from batter.data import run_files, openmm_files
from batter.utils.error_report import error_report
from batter.utils.align import aligning

import MDAnalysis as mda
# ignore UserWarning from MDAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import pickle

HELP_STR = f"""Batter project; a refactored version of BAT.py

Author: {batter.__author__}
Email: {batter.__email__}
Version: {batter.__version__}
"""

@click.group(help=HELP_STR)
def main(args=None):
    pass

main.add_command(batpy.batpy)
main.add_command(error_report)
main.add_command(aligning)

if __name__ == "__main__":
    main()  # pragma: no cover
