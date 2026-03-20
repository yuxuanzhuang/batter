"""BATTER package initialization."""

from __future__ import annotations

import logging
import sys
import warnings

from loguru import logger

from ._version import __version__

__author__ = "Yuxuan Zhuang"
__email__ = "yuxuan.zhuang@stanford.edu"
__version__ = __version__


def _seed_default_slurm_headers() -> None:
    try:
        from .utils.slurm_templates import seed_default_headers

        seed_default_headers()
    except Exception:
        pass


def _configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="MDAnalysis.coordinates.PDB",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="MDAnalysis.topology.PDBParser",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="MDAnalysis.topology.MOL2Parser",
    )

    try:
        from Bio.Application import BiopythonDeprecationWarning
    except ImportError:
        BiopythonDeprecationWarning = None

    if BiopythonDeprecationWarning is not None:
        warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)


def _configure_pymbar_logging() -> None:
    def _mute_timeseries(record):
        return "Warning on use of the timeseries module:" not in record.msg

    def _mute_jax(record):
        return "****** PyMBAR will use 64-bit JAX! *******" not in record.msg

    def _mute_jax_2(record):
        return "******* JAX 64-bit mode is now on! *******" not in record.msg

    def _mute_jax_3(record):
        return "PyMBAR can run faster with JAX" not in record.msg

    logging.getLogger("pymbar.timeseries").addFilter(_mute_timeseries)
    mbar_solvers_log = logging.getLogger("pymbar.mbar_solvers")
    mbar_solvers_log.addFilter(_mute_jax)
    mbar_solvers_log.addFilter(_mute_jax_2)
    mbar_solvers_log.addFilter(_mute_jax_3)
    logging.getLogger("MDAnalysis").setLevel(logging.WARNING)


def _configure_logger() -> None:
    logger.remove()
    logger.add(sys.stderr, format="{level} | <level>{message}</level> ", level="INFO")


_seed_default_slurm_headers()
_configure_warning_filters()
_configure_pymbar_logging()
_configure_logger()
