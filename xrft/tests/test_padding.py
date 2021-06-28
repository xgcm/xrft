"""
Unit tests for padding functions
"""
import pytest
import numpy as np
import xarray as xr
import numpy.testing as npt

from ..padding import pad, pad_coordinates


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


@pytest.fixture
def sample_da_2d():
    """
    Defines a 2D sample xarray.DataArray
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.arange(11 * 17, dtype=float).reshape(17, 11)
    return xr.DataArray(z, coords={"x": x, "y": y}, dims=("y", "x"))


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


def test_pad_with_kwargs(sample_da_2d):
    """
    Test pad function by passing pad_width as kwargs
    """
    # Check if the array is padded with nans by default
    padded_da = pad(sample_da_2d, x=2, y=1)
    assert padded_da.shape == (19, 15)
    npt.assert_allclose(padded_da.values[:1, :], np.nan)
    npt.assert_allclose(padded_da.values[-1:, :], np.nan)
    npt.assert_allclose(padded_da.values[:, :2], np.nan)
    npt.assert_allclose(padded_da.values[:, -2:], np.nan)
    npt.assert_allclose(padded_da.values[1:-1, 2:-2], sample_da_2d)
    npt.assert_allclose(padded_da.x, np.linspace(-2, 12, 15))
    npt.assert_allclose(padded_da.y, np.linspace(-4.5, 4.5, 19))


def test_pad_with_pad_width(sample_da_2d):
    """
    Test pad function by passing pad_width as argument
    """
    # Check if the array is padded with nans by default
    pad_width = {"x": (2, 3), "y": (1, 3)}
    padded_da = pad(sample_da_2d, pad_width)
    assert padded_da.shape == (21, 16)
    npt.assert_allclose(padded_da.values[:1, :], np.nan)
    npt.assert_allclose(padded_da.values[-3:, :], np.nan)
    npt.assert_allclose(padded_da.values[:, :2], np.nan)
    npt.assert_allclose(padded_da.values[:, -3:], np.nan)
    npt.assert_allclose(padded_da.values[1:-3, 2:-3], sample_da_2d)
    npt.assert_allclose(padded_da.x, np.linspace(-2, 13, 16))
    npt.assert_allclose(padded_da.y, np.linspace(-4.5, 5.5, 21))
