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
    da['x'].values[-1] *= 2
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

def synthetic_field(N, dL, amp, s):
    """
    Generate a synthetic series of size N by N
    with a spectral slope of s.
    """

    k = np.fft.fftshift(np.fft.fftfreq(N, dL))
    l = np.fft.fftshift(np.fft.fftfreq(N, dL))
    kk, ll = np.meshgrid(k, l)
    K = np.sqrt(kk**2+ll**2)

    ########
    # amplitude
    ########
    r_kl = np.ma.masked_invalid(np.sqrt(amp*.5*(np.pi)**(-1)*K**(s-1.))).filled(0.)
    #r = np.ma.masked_invalid(np.abs(k)**(-slope/2.)).filled(0.)
    ########
    # phase
    ########
    phi = np.zeros((N, N))

    N_2 = int(N/2)
    phi_upper_right = 2.*np.pi*np.random.random((N_2-1,
                                                 N_2-1)) - np.pi
    phi[N_2+1:,N_2+1:] = phi_upper_right.copy()
    phi[1:N_2, 1:N_2] = -phi_upper_right[::-1, ::-1].copy()


    phi_upper_left = 2.*np.pi*np.random.random((N_2-1,
                                                N_2-1)) - np.pi
    phi[N_2+1:,1:N_2] = phi_upper_left.copy()
    phi[1:N_2, N_2+1:] = -phi_upper_left[::-1, ::-1].copy()


    phi_upper_middle = 2.*np.pi*np.random.random(N_2) - np.pi
    phi[N_2:, N_2] = phi_upper_middle.copy()
    phi[1:N_2, N_2] = -phi_upper_middle[1:][::-1].copy()


    phi_right_middle = 2.*np.pi*np.random.random(N_2-1) - np.pi
    phi[N_2, N_2+1:] = phi_right_middle.copy()
    phi[N_2, 1:N_2] = -phi_right_middle[::-1].copy()


    phi_edge_upperleft = 2.*np.pi*np.random.random(N_2) - np.pi
    phi[N_2:, 0] = phi_edge_upperleft.copy()
    phi[1:N_2, 0] = -phi_edge_upperleft[1:][::-1].copy()


    phi_bot_right = 2.*np.pi*np.random.random(N_2) - np.pi
    phi[0, N_2:] = phi_bot_right.copy()
    phi[0, 1:N_2] = -phi_bot_right[1:][::-1].copy()


    phi_corner_leftbot = 2.*np.pi*np.random.random() - np.pi


#     print(phi[N/2-1,N-1], phi[N/2+1,1])
#     print(phi[N/2+1,N/2+1], phi[N/2-1,N/2-1])


#     phi[:N/2, :] = -np.rot90(np.rot90(phi[N/2:, :]))
#     phi[:N/2, :] = -phi[N/2:, :][::-1,::-1]
#     i, j = 25, 40
#     print(phi[N/2+j,N/2+i], -phi[N/2-j,N/2-i])

    for i in range(1, N_2):
        for j in range(1, N_2):
            assert (phi[N_2+j, N_2+i] == -phi[N_2-j, N_2-i])

    for i in range(1, N_2):
        for j in range(1, N_2):
            assert (phi[N_2+j, N_2-i] == -phi[N_2-j, N_2+i])

    for i in range(1, N_2):
        assert (phi[N_2, N-i] == -phi[N_2, i])
        assert (phi[N-i, N_2] == -phi[i, N_2])
        assert (phi[N-i, 0] == -phi[i, 0])
        assert (phi[0, i] == -phi[0, N-i])
    #########
    # complex fourier amplitudes
    #########
    #a = r + 1j*th
    F_theta = r_kl * np.exp(1j * phi)

    # check that symmetry of FT is satisfied
    #np.testing.assert_almost_equal(a[1:N/2], a[-1:-N/2:-1].conj())

    theta = np.fft.ifft2(np.fft.ifftshift(F_theta))
    return np.real(theta)

def fit_loglog(x, y):
    """Fit a line to isotropic spectra in log-log space"""
    # fig log vs log
    p = np.polyfit(np.log2(x), np.log2(y), 1)
    y_fit = 2**(np.log2(x)*p[0] + p[1])
    #A = np.vstack([np.log2(x), np.ones(len(x))]).T
    #a, b = np.linalg.lstsq(A, np.log2(y))[0]
    #y_fit = 2**(np.log2(x)*a + b)

    return y_fit, p[0], p[1]

def test_dft_2d_slope(N=512, dL=1., amp=1e1, s=-3.):
    """Test the spectral slope of synthetic data."""
    theta = synthetic_field(N, dL, amp, s)
    theta = xr.DataArray(theta, dims=['k', 'l'],
                        coords={'k':range(N), 'l':range(N)})
    f, iso_f = xrft.dft(theta, remove_mean=True,
                  density=True, iso=True)
    y_fit, a, b = fit_loglog(iso_f.freq_kr.values[4:],
                            iso_f.values[4:])

    npt.assert_allclose(a, s, atol=.1)
