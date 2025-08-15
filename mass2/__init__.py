"""
Mass: a Microcalorimeter Analysis Software Suite

Python tools to analyze microcalorimeter data offline.


Joe Fowler, NIST Boulder Labs.  November 2010--
"""

# ruff: noqa: F403, F401

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from . import calibration
from . import common
from . import mathstat
from . import core
# at the top level lets import things we image people will want to use regularly, and try to avoid importing
# implementation details
from .calibration import STANDARD_FEATURES, spectra
from .core import LJHFile, TrueBqBin, Channel, Channels, NoiseChannel, NoisePSD, misc, show, FilterMaker, ChannelHeader, MultiFit, FitSpec
# from .mathstat import 
# from .common import *
# from .core import *

