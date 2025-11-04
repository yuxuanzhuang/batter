import warnings
warnings.warn(
    "batter.results is deprecated; use batter.analysis.results",
    DeprecationWarning,
    stacklevel=2,
)
from .analysis.results import *  # re-export legacy names

class NewFEResult(FEResult):
    pass