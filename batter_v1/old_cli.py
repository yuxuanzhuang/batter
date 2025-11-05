import batter_v1
from batter_v1 import batpy

import click
import glob as glob
import os as os
import shutil as shutil
import signal as signal

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

HELP_STR = f"""Batter project; a refactored version of BAT.py

Author: {batter_v1.__author__}
Email: {batter_v1.__email__}
Version: {batter_v1.__version__}
"""

@click.group(help=HELP_STR)
def main(args=None):
    pass

from batter_v1.batpy import copy_system, gather, report_jobs, cancel_jobs
from batter_v1.utils.error_report import error_report
from batter_v1.utils.align import aligning
from batter_v1.run_in_batch import run_in_batch
from batter_v1.analysis.preprocessing import preprocess

main.add_command(copy_system)
main.add_command(gather)
main.add_command(error_report)
main.add_command(aligning)
main.add_command(run_in_batch)
main.add_command(preprocess)
main.add_command(report_jobs)
main.add_command(cancel_jobs)

if __name__ == "__main__":
    main()  # pragma: no cover
