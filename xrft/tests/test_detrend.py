import numpy as np
import xarray as xr
import scipy.signal as sps

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft
from xrft.detrend import detrend


def detrended_noise(N, amplitude=1.0):
    return sps.detrend(amplitude * np.random.rand(N))


def noise(dims, shape):
    assert len(dims) == len(shape)
    coords = {d: (d, np.arange(n)) for d, n in zip(dims, shape)}
    data = np.random.rand(*shape)
    for n in range(len(shape)):
        data = sps.detrend(data, n)
    da = xr.DataArray(data, dims=dims, coords=coords)
    return da


@pytest.mark.parametrize("dim", ["t", "time"])
@pytest.mark.parametrize("detrend_type", ["constant", "linear"])
def test_dim_format(dim, detrend_type):
    """Check that detrend can deal with dim in various formats"""
    data = xr.DataArray(
        np.random.random([10]),
        dims=[dim],
        coords={dim: range(10)},
    )
    dt = detrend(data, dim=dim, detrend_type=detrend_type)
    dt = detrend(data, dim=[dim], detrend_type=detrend_type)


@pytest.mark.parametrize(
    "array_dims, array_shape, detrend_dim, chunks, linear_error",
    (
        (["x"], [16], "x", None, None),
        (["y", "x"], [32, 16], "x", None, None),
        (["y", "x"], [32, 16], "x", {"y": 4}, None),
        (["y", "x"], [32, 16], "y", None, None),
        (["y", "x"], [32, 16], "y", {"x": 4}, None),
        (["time", "y", "x"], [4, 32, 16], "x", None, None),
        (["time", "y", "x"], [4, 32, 16], "x", {"y": 4}, None),
        (["time", "y", "x"], [4, 32, 16], "x", {"time": 1, "y": 4}, None),
        # error cases for linear detrending
        (["x"], [16], "x", {"x": 1}, ValueError),
        (["y", "x"], [32, 16], "x", {"x": 4}, ValueError),
    ),
)
@pytest.mark.parametrize("detrend_type", [None, "constant", "linear"])
@pytest.mark.parametrize("trend_amplitude", [0.01, 100])
def test_detrend_1D(
    array_dims,
    array_shape,
    detrend_dim,
    chunks,
    detrend_type,
    trend_amplitude,
    linear_error,
):
    da_original = noise(array_dims, array_shape)
    da_trend = da_original + trend_amplitude * da_original[detrend_dim]
    if chunks:
        da_trend = da_trend.chunk(chunks)

    # bail out if we are expecting an error
    if detrend_type == "linear" and linear_error:
        with pytest.raises(linear_error):
            detrend(da_trend, detrend_dim, detrend_type=detrend_type)
        return

    detrended = detrend(da_trend, detrend_dim, detrend_type=detrend_type)
    assert detrended.chunks == da_trend.chunks
    if detrend_type is None:
        xrt.assert_equal(detrended, da_trend)
    elif detrend_type == "constant":
        xrt.assert_allclose(detrended, da_trend - da_trend.mean(dim=detrend_dim))
    elif detrend_type == "linear":
        xrt.assert_allclose(detrended, da_original)


# always detrend on x y dims
@pytest.mark.parametrize(
    "array_dims, array_shape, chunks",
    (
        (["y", "x"], [32, 16], None),
        (["z", "y", "x"], [2, 32, 16], None),
        (["z", "y", "x"], [2, 32, 16], {"z": 1}),
    ),
)
@pytest.mark.parametrize("detrend_type", [None, "constant", "linear"])
@pytest.mark.parametrize(
    "trend_amplitude", [{"x": 0.1, "y": 0.1}, {"x": 10.0, "y": 0.01}]
)
def test_detrend_2D(array_dims, array_shape, chunks, detrend_type, trend_amplitude):
    da_original = noise(array_dims, array_shape)
    da_trend = (
        da_original
        + trend_amplitude["x"] * da_original["x"]
        + trend_amplitude["y"] * da_original["y"]
    )
    if chunks:
        da_trend = da_trend.chunk(chunks)

    detrend_dim = ["y", "x"]
    detrended = detrend(da_trend, detrend_dim, detrend_type=detrend_type)
    assert detrended.chunks == da_trend.chunks
    if detrend_type is None:
        xrt.assert_equal(detrended, da_trend)
    elif detrend_type == "constant":
        xrt.assert_allclose(detrended, da_trend - da_trend.mean(dim=detrend_dim))
    elif detrend_type == "linear":
        xrt.assert_allclose(detrended, da_original)
