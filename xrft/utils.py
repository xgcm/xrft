"""
Utility functions for xrft
"""
import numpy as np

from .xrft import _diff_coord


def get_spacing(coord):
    """
    Return the spacing of evenly spaced coordinates array
    """
    diff = _diff_coord(coord)
    if not np.allclose(diff, diff[0]):
        raise ValueError(
            f"Found unevenly spaced coordinates '{coord.name}'. "
            "These coordinates should be evenly spaced."
        )
    return diff[0]
