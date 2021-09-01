from ._version import __version__

__version__ = get_versions()["version"]
del get_versions

from .xrft import *  # noqa
from .detrend import detrend
