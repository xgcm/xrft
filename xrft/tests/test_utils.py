"""
Test utility functions of xrft
"""
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import numpy.testing as npt

from ..utils import get_spacing


@pytest.fixture
def sample_da_2d():
    """
    Defines a 2D sample xarray.DataArray
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.arange(11 * 17, dtype=float).reshape(17, 11)
    return xr.DataArray(z, coords={"x": x, "y": y}, dims=("y", "x"))


@pytest.fixture
def sample_da_time():
    """
    Defines a 1D sample xarray.DataArray with datetime coordinates
    """
    time = pd.date_range(
        "2021-01-01 00:00", "2021-01-01 23:00", periods=24
    ).to_pydatetime()
    values = np.arange(24, dtype=float)
    return xr.DataArray(values, coords={"time": time}, dims=("time",))


def test_get_spacing(sample_da_2d, sample_da_time):
    """
    Check if get_spacing function works as expected
    """
    npt.assert_allclose(get_spacing(sample_da_2d.x), 1)
    npt.assert_allclose(get_spacing(sample_da_2d.y), 0.5)
    npt.assert_allclose(get_spacing(sample_da_time.time), 60 * 60)


def test_get_spacing_unvenly_spaced(sample_da_2d):
    """
    Check if error is raised after unvenly spaced coordinates
    """
    # Make the x coordinate unevenly spaced
    x = sample_da_2d.x.values
    x[0] += 0.2
    da = sample_da_2d.assign_coords({"x": x})
    # Check if error is raised
    with pytest.raises(ValueError):
        get_spacing(da.x)
