"""A python package that set up FEP simulations with bat.py"""

from ._version import __version__

__author__ = """Yuxuan Zhuang"""
__email__ = 'yuxuan.zhuang@stanford.edu'
__version__ = __version__

from loguru import logger
import sys

# silence pymbar logging warnings
# copy from openfe
import logging
def _mute_timeseries(record):
    return not "Warning on use of the timeseries module:" in record.msg
def _mute_jax(record):
    return not "****** PyMBAR will use 64-bit JAX! *******" in record.msg
def _mute_jax_2(record):
    return not "******* JAX 64-bit mode is now on! *******" in record.msg
_mbar_log = logging.getLogger("pymbar.timeseries")
_mbar_log.addFilter(_mute_timeseries)
_mbar_log = logging.getLogger("pymbar.mbar_solvers")
_mbar_log.addFilter(_mute_jax)
_mbar_log.addFilter(_mute_jax_2)

# Add imports here
from .batter import *

logger.remove()
logger_format = ('{level} | <level>{message}</level> ')
# format time to be human readable
logger.add(sys.stderr, format=logger_format, level="INFO")