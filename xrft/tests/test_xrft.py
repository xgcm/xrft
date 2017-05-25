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
    da2 = da.copy()
    da2[-1] = np.nan
    with pytest.raises(ValueError):
        ft = xrft.dft(da2)

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

def test_dft_2d():
    """Test the discrete Fourier transform on 2D data"""
    N = 16
    da = xr.DataArray(np.random.rand(N,N), dims=['x','y'],
                    coords={'x':range(N),'y':range(N)}
                     )
    daft = xrft.dft(da, shift=False, remove_mean=False)
    npt.assert_almost_equal(daft.values, np.fft.fftn(da.values))

    daft = xrft.dft(da, shift=False, window=True)
    dim = da.dims
    window = np.hanning(N) * np.hanning(N)[:, np.newaxis]
    da_prime = (da - da.mean(dim=dim)).values
    npt.assert_almost_equal(daft.values, np.fft.fftn(da_prime*window))

def test_dft_3d():
    """Test the discrete Fourier transform on 3D dask array data"""
    N=16
    da = xr.DataArray(np.random.rand(2,N,N), dims=['time','x','y'],
                      coords={'time':range(2),'x':range(N),
                              'y':range(N)}).chunk({'time': 1}
                     )
    daft = xrft.dft(da, dim=['x','y'], shift=False, remove_mean=False)
    npt.assert_almost_equal(daft.values, np.fft.fftn(da.values, axes=[1,2]))

def test_power_spectrum():
    """Test the power spectrum function"""
    N = 16
    da = xr.DataArray(np.random.rand(N,N), dims=['x','y'],
                    coords={'x':range(N),'y':range(N)}
                     )
    ps = xrft.power_spectrum(da, window=True, density=False)
    daft = xrft.dft(da,
                    dim=None, shift=True, remove_mean=True,
                    window=True)
    npt.assert_almost_equal(ps.values, np.real(daft*np.conj(daft)))
    npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.)

    ### Normalized
    dim = da.dims
    ps = xrft.power_spectrum(da, window=True)
    coord = list(daft.coords)
    daft = xrft.dft(da,
                    dim=None, shift=True, remove_mean=True,
                    window=True)
    test = np.real(daft*np.conj(daft))/N**4
    for i in range(len(dim)):
        test /= daft[coord[-i-1]].values
    npt.assert_almost_equal(ps.values, test)
    npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.)

def test_cross_spectrum():
    """Test the cross spectrum function"""
    N = 16
    da = xr.DataArray(np.random.rand(N,N), dims=['x','y'],
                    coords={'x':range(N),'y':range(N)}
                     )
    da2 = xr.DataArray(np.random.rand(N,N), dims=['x','y'],
                    coords={'x':range(N),'y':range(N)}
                     )
    cs = xrft.cross_spectrum(da, da2, window=True, density=False)
    daft = xrft.dft(da,
                    dim=None, shift=True, remove_mean=True,
                    window=True)
    daft2 = xrft.dft(da2,
                    dim=None, shift=True, remove_mean=True,
                    window=True)
    npt.assert_almost_equal(cs.values, np.real(daft*np.conj(daft2)))

    npt.assert_almost_equal(np.ma.masked_invalid(cs).mask.sum(), 0.)

def test_parseval():
    """Test whether the Parseval's relation is satisfied."""

    N = 16
    da = xr.DataArray(np.random.rand(N,N),
                    dims=['x','y'], coords={'x':range(N), 'y':range(N)})
    da2 = xr.DataArray(np.random.rand(N,N),
                    dims=['x','y'], coords={'x':range(N), 'y':range(N)})

    dim = da.dims
    delta_x = []
    for d in dim:
        coord = da[d]
        diff = np.diff(coord)
        # if pd.core.common.is_timedelta64_dtype(diff):
        #     # convert to seconds so we get hertz
        #     diff = diff.astype('timedelta64[s]').astype('f8')
        delta = diff[0]
        delta_x.append(delta)

    window = np.hanning(N) * np.hanning(N)[:, np.newaxis]
    ps = xrft.power_spectrum(da, window=True)
    da_prime = (da - da.mean(dim=dim)).values
    npt.assert_almost_equal(ps.values.sum(),
                            (np.asarray(delta_x).prod()
                            * ((da_prime*window)**2).sum()
                            ), decimal=5
                            )

    cs = xrft.cross_spectrum(da, da2, window=True)
    da2_prime = (da2 - da2.mean(dim=dim)).values
    npt.assert_almost_equal(cs.values.sum(),
                            (np.asarray(delta_x).prod()
                            * ((da_prime*window)
                            * (da2_prime*window)).sum()
                            ), decimal=5
                            )

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
    r_kl = np.ma.masked_invalid(np.sqrt(amp*.5*(np.pi)**(-1)
                                *K**(s-1.))).filled(0.)
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

def test_isotropic_ps_slope(N=512, dL=1., amp=1e1, s=-3.):
    """Test the spectral slope of isotropic power spectrum."""

    theta = xr.DataArray(synthetic_field(N, dL, amp, s),
                        dims=['x', 'y'],
                        coords={'x':range(N), 'y':range(N)})
    iso_ps = xrft.isotropic_powerspectrum(theta, remove_mean=True,
                                        density=True)
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps[1:]).mask.sum(), 0.)
    y_fit, a, b = xrft.fit_loglog(iso_ps.freq_r.values[4:],
                                iso_ps.values[4:])

    npt.assert_allclose(a, s, atol=.1)

def test_isotropic_cs():
    """Test isotropic cross spectrum"""
    N = 16
    da = xr.DataArray(np.random.rand(N,N),
                    dims=['x','y'], coords={'x':range(N), 'y':range(N)})
    da2 = xr.DataArray(np.random.rand(N,N),
                    dims=['x','y'], coords={'x':range(N), 'y':range(N)})

    dim = da.dims
    delta_x = []
    for d in dim:
        coord = da[d]
        diff = np.diff(coord)
        # if pd.core.common.is_timedelta64_dtype(diff):
        #     # convert to seconds so we get hertz
        #     diff = diff.astype('timedelta64[s]').astype('f8')
        delta = diff[0]
        delta_x.append(delta)

    iso_cs = xrft.isotropic_crossspectrum(da, da2, window=True, nbins=int(N/4))
    npt.assert_almost_equal(np.ma.masked_invalid(iso_cs[1:]).mask.sum(), 0.)

    da2 = xr.DataArray(np.random.rand(N,N),
                    dims=['lon','lat'], coords={'lon':range(N), 'lat':range(N)})
    with pytest.raises(ValueError):
        xrft.isotropic_crossspectrum(da, da2)
