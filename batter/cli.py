import batter
from batter import batpy

import click
import glob as glob
import os as os
import shutil as shutil
import signal as signal

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

HELP_STR = f"""Batter project; a refactored version of BAT.py

Author: {batter.__author__}
Email: {batter.__email__}
Version: {batter.__version__}
"""

@click.group(help=HELP_STR)
def main(args=None):
    pass

from batter.batpy import copy_system, gather
from batter.utils.error_report import error_report
from batter.utils.align import aligning
from batter.run_in_batch import run_in_batch
from batter.analysis.preprocessing import preprocess

main.add_command(copy_system)
main.add_command(gather)
main.add_command(error_report)
main.add_command(aligning)
main.add_command(run_in_batch)
main.add_command(preprocess)

if __name__ == "__main__":
    main()  # pragma: no cover
