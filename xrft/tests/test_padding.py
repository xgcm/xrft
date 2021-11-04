"""
Unit tests for padding functions
"""
import pytest
import numpy as np
import xarray as xr
import xarray.testing as xrt
import numpy.testing as npt

from ..padding import pad, pad_coordinates, unpad, _pad_width_to_slice


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
    "pad_width",
    (
        {"x": 2, "y": 3},
        {"x": 2},
        {"y": 3},
        {"x": (2, 3), "y": 3},
        {"x": (2, 3), "y": (1, 3)},
        {"x": (2, 3)},
        {"y": (1, 3)},
    ),
)
def test_coordinates_attrs_after_pad(sample_da_2d, pad_width):
    """
    Test if the attributes of the coordinates are preserved after padding
    and if the pad_width has been added
    """
    padded_da = pad(sample_da_2d, pad_width)
    # Check if the attrs in sample_da_2d is a subset of the attrs in padded_da
    assert sample_da_2d.x.attrs.items() <= padded_da.x.attrs.items()
    assert sample_da_2d.y.attrs.items() <= padded_da.y.attrs.items()
    # Check if pad_width has been added to the attrs of each coordinate
    for coord, width in pad_width.items():
        assert padded_da.coords[coord].attrs["pad_width"] == width


@pytest.mark.parametrize(
    "pad_width",
    (
        {"x": 2, "y": 3},
        {"x": 2},
        {"y": 3},
        {"x": (2, 3), "y": 3},
        {"x": (2, 3), "y": (1, 3)},
        {"x": (2, 3)},
        {"y": (1, 3)},
    ),
)
def test_pad_unpad_round_trip(sample_da_2d, pad_width):
    """
    Test if applying pad and then unpad returns the original array
    """
    xrt.assert_allclose(sample_da_2d, unpad(pad(sample_da_2d, pad_width)))


def test_unpad_invalid_array(sample_da_2d):
    """
    Test if error is raised when a not padded array is passed to unpad
    """
    with pytest.raises(ValueError):
        unpad(sample_da_2d)


@pytest.mark.parametrize(
    "pad_width, size, expected_slice",
    (
        [(1, 1), 4, slice(1, 3)],
        [(1, 2), 5, slice(1, 3)],
        [(2, 3), 10, slice(2, 7)],
        [2, 10, slice(2, 8)],
    ),
)
def test_pad_width_to_slice(pad_width, size, expected_slice):
    """
    Test if _pad_width_to_slice work as expected
    """
    assert _pad_width_to_slice(pad_width, size) == expected_slice
