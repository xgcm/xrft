"""
Unit tests for padding functions
"""
import pytest
import numpy as np
import xarray as xr
import numpy.testing as npt

from ..padding import pad_coordinates


@pytest.fixture
def coords():
    """
    Define a dictionary with sample coordinates
    """
    x = np.linspace(-4, 5, 10)
    y = np.linspace(-1, 4, 11)
    coords = {
        "x": xr.DataArray(x, coords={"x": x}, dims=("x",)),
        "y": xr.DataArray(y, coords={"y": y}, dims=("y",)),
    }
    return coords


def test_pad_coordinates(coords):
    """
    Test pad_coordinates function
    """
    # Pad a single coordinate
    padded_coords = pad_coordinates(coords, {"x": 3})
    npt.assert_allclose(padded_coords["x"], np.linspace(-7, 8, 16))
    npt.assert_allclose(padded_coords["y"], coords["y"])
    # Pad two coordinates
    padded_coords = pad_coordinates(coords, {"x": 2, "y": 3})
    npt.assert_allclose(padded_coords["x"], np.linspace(-6, 7, 14))
    npt.assert_allclose(padded_coords["y"], np.linspace(-2.5, 5.5, 17))
    # Pad a single coordinate asymmetrically
    padded_coords = pad_coordinates(coords, {"x": (3, 2)})
    npt.assert_allclose(padded_coords["x"], np.linspace(-7, 7, 15))
    npt.assert_allclose(padded_coords["y"], coords["y"])
    # Pad two coordinates assymetrically
    padded_coords = pad_coordinates(coords, {"x": (2, 1), "y": (3, 4)})
    npt.assert_allclose(padded_coords["x"], np.linspace(-6, 6, 13))
    npt.assert_allclose(padded_coords["y"], np.linspace(-2.5, 6, 18))


def test_pad_coordinates_invalid(coords):
    """
    Test if pad_coordinates raises error after unevenly spaced coords
    """
    x = coords["x"]
    x[3] += 0.1
    with pytest.raises(ValueError):
        pad_coordinates(coords, pad_width={"x": 2})
