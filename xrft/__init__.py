from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # noqa: F401
    # package is not installed
    pass

del get_distribution, DistributionNotFound

# from .xrft import *  # noqa
# from .detrend import detrend
