import warnings
warnings.warn(
    "batter_v1.results is deprecated; use batter_v1.analysis.results",
    DeprecationWarning,
    stacklevel=2,
)
from .analysis.results import *  # re-export legacy names

class NewFEResult(FEResult):
    pass
