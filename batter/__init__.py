"""A python package that set up FEP simulations with bat.py"""

__author__ = """Yuxuan Zhuang"""
__email__ = 'yuxuan.zhuang@stanford.edu'

# Add imports here
from .batter import *

from ._version import __version__

from loguru import logger
import sys

logger.remove()

logger_format = ('{level} | {message}')
# format time to be human readable
logger.add(sys.stderr, format=logger_format, level="INFO")