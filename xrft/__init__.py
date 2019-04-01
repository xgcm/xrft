from .version import __version__  # noqa
from .xrft import *  # noqa

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
