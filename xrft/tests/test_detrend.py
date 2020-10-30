import numpy as np
import xarray as xr
import scipy.signal as sps

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft
from xrft.detrend import detrend

def detrended_noise(N, amplitude=1.0):
    return sps.detrend(amplitude*np.random.rand(N))

def noise(dims, shape):
    assert len(dims) == len(shape)
    coords = {d: (d, np.arange(n)) for d, n in zip(dims, shape)}
    data = np.random.rand(*shape)
    for n in range(len(shape)):
        data = sps.detrend(data, n)
    da = xr.DataArray(data, dims=dims, coords=coords)
    return da

@pytest.mark.parametrize(
    'array_dims, array_shape, detrend_dim, chunks',
    (
        (['x'], [16], 'x', None),
        (['y', 'x'], [32, 16], 'x', None),
        (['y', 'x'], [32, 16], 'x', {'y': 4}),
        (['y', 'x'], [32, 16], 'y', None),
        (['y', 'x'], [32, 16], 'y', {'x': 4}),
    )
)
@pytest.mark.parametrize('detrend_type', [None, 'constant', 'linear'])
@pytest.mark.parametrize('trend_amplitude', [0.01, 100])
def test_detrend_one_dim(array_dims, array_shape, detrend_dim, chunks, detrend_type, trend_amplitude):
    da_original = noise(array_dims, array_shape)
    da_trend = da_original + trend_amplitude * da_original[detrend_dim]
    if chunks:
        da_original = da_original.chunk(chunks)
    detrended = detrend(da_trend, detrend_dim, detrend_type=detrend_type)
    if detrend_type is None:
        xrt.assert_equal(detrended, da_trend)
    elif detrend_type == 'constant':
        xrt.assert_allclose(detrended, da_trend - da_trend.mean(dim=detrend_dim))
    elif detrend_type == 'linear':
        xrt.assert_allclose(detrended, da_original)
