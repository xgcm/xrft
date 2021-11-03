"""
Unit tests for padding functions
"""
import pytest
import numpy as np
import xarray as xr
import numpy.testing as npt

from ..padding import pad, pad_coordinates


@pytest.fixture
def sample_da_2d():
    """
    Defines a 2D sample xarray.DataArray
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.arange(11 * 17, dtype=float).reshape(17, 11)
    # Create one xr.DataArray for each coordiante and add spacing and
    # direct_lag attributes to them
    dx, dy = x[1] - x[0], y[1] - y[0]
    x = xr.DataArray(
        x, coords={"x": x}, dims=("x",), attrs=dict(direct_lag=3.0, spacing=dx)
    )
    y = xr.DataArray(
        y, coords={"y": y}, dims=("y",), attrs=dict(direct_lag=-2.1, spacing=dy)
    )
    return xr.DataArray(z, coords={"x": x, "y": y}, dims=("y", "x"))


def test_pad_coordinates(sample_da_2d):
    """
    Test pad_coordinates function
    """
    coords = sample_da_2d.coords
    # Pad a single coordinate
    padded_coords = pad_coordinates(coords, {"x": 3})
    npt.assert_allclose(padded_coords["x"], np.linspace(-3, 13, 17))
    npt.assert_allclose(padded_coords["y"], coords["y"])
    # Pad two coordinates
    padded_coords = pad_coordinates(coords, {"x": 2, "y": 3})
    npt.assert_allclose(padded_coords["x"], np.linspace(-2, 12, 15))
    npt.assert_allclose(padded_coords["y"], np.linspace(-5.5, 5.5, 23))
    # Pad a single coordinate asymmetrically
    padded_coords = pad_coordinates(coords, {"x": (3, 2)})
    npt.assert_allclose(padded_coords["x"], np.linspace(-3, 12, 16))
    npt.assert_allclose(padded_coords["y"], coords["y"])
    # Pad two coordinates assymetrically
    padded_coords = pad_coordinates(coords, {"x": (2, 1), "y": (3, 4)})
    npt.assert_allclose(padded_coords["x"], np.linspace(-2, 11, 14))
    npt.assert_allclose(padded_coords["y"], np.linspace(-5.5, 6, 24))


def test_pad_coordinates_invalid(sample_da_2d):
    """
    Test if pad_coordinates raises error after unevenly spaced coords
    """
    x = sample_da_2d.coords["x"].values
    x[3] += 0.1
    sample_da_2d = sample_da_2d.assign_coords({"x": x})
    with pytest.raises(ValueError):
        pad_coordinates(sample_da_2d.coords, pad_width={"x": 2})


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


@pytest.mark.parametrize(
    "pad_width", ({"x": (2, 3), "y": (1, 3)}, {"x": (2, 3)}, {"y": (1, 3)})
)
def test_coordinates_attrs_after_pad(sample_da_2d, pad_width):
    """
    Test if the attributes of the coordinates are preserved after padding
    """
    padded_da = pad(sample_da_2d, pad_width)
    assert sample_da_2d.x.attrs == padded_da.x.attrs
    assert sample_da_2d.y.attrs == padded_da.y.attrs
