try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .xrft import *  # noqa
from .detrend import detrend
from .padding import pad, unpad
