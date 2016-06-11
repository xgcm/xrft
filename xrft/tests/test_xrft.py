import numpy as np
import pandas as pd
import xarray as xr
import numpy.testing as npt
import pytest
import xrft

@pytest.fixture()
def sample_data_3d():
    """Create three dimensional test data."""
    pass

@pytest.fixture()
def sample_data_1d():
    """Create one dimensional test DataArray."""
    pass

def test_dft_1d():
    """Test the discrete Fourier transform function on one-dimensional data."""
    Nx = 16
    Lx = 1.0
    x = np.linspace(0, Lx, Nx)
    dx = x[1] - x[0]
    da = xr.DataArray(np.random.rand(Nx), coords=[x], dims=['x'])

    # defaults with no keyword args
    ft = xrft.dft(da)
    # check that the frequency dimension was created properly
    assert ft.dims == ('freq_x',)
    # check that the coords are correct
    freq_x_expected = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    npt.assert_allclose(ft['freq_x'], freq_x_expected)
    # check that a spacing variable was created
    assert ft['freq_x_spacing'] == freq_x_expected[1] - freq_x_expected[0]
    # check that the Fourier transform itself is correct
    data = (da - da.mean()).values
    ft_data_expected = np.fft.fftshift(np.fft.fft(data))
    # because the zero frequency component is zero, there is a numerical
    # precision issue. Fixed by setting atol
    npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)

    # redo without removing mean
    ft = xrft.dft(da, remove_mean=False)
    ft_data_expected = np.fft.fftshift(np.fft.fft(da))
    npt.assert_allclose(ft_data_expected, ft.values)

    # modify data to be non-evenly spaced
    da['x'][-1] *= 2
    with pytest.raises(ValueError):
        ft = xrft.dft(da)

def test_dft_1d_time():
    """Test the discrete Fourier transform function on timeseries data."""
    time = pd.date_range('2000-01-01', '2001-01-01', closed='left')
    Nt = len(time)
    da = xr.DataArray(np.random.rand(Nt), coords=[time], dims=['time'])

    ft = xrft.dft(da)

    # check that frequencies are correct
    dt = (time[1] - time[0]).total_seconds()
    freq_time_expected = np.fft.fftshift(np.fft.fftfreq(Nt, dt))
    npt.assert_allclose(ft['freq_time'], freq_time_expected)
