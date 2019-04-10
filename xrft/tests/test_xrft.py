import numpy as np
import pandas as pd
import xarray as xr
import dask.array as dsar

import scipy.signal as sps
import scipy.linalg as spl

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft


@pytest.fixture()
def sample_data_3d():
    """Create three dimensional test data."""
    pass

@pytest.fixture(params=['numpy', 'dask', 'nocoords'])
def test_data_1d(request):
    """Create one dimensional test DataArray."""
    Nx = 16
    Lx = 1.0
    x = np.linspace(0, Lx, Nx)
    dx = x[1] - x[0]
    coords = None if request.param == 'nocoords' else [x]
    da = xr.DataArray(np.random.rand(Nx), coords=coords, dims=['x'])
    if request.param == 'dask':
        da = da.chunk()
    return da

def numpy_detrend(da):
    """
    Detrend a 2D field by subtracting out the least-square plane fit.

    Parameters
    ----------
    da : `numpy.array`
        The data to be detrended

    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """
    N = da.shape

    G = np.ones((N[0]*N[1],3))
    for i in range(N[0]):
        G[N[1]*i:N[1]*i+N[1], 1] = i+1
        G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)

    d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    lin_trend = np.reshape(d_est, N)

    return da - lin_trend

def test_detrend():
    N = 16
    x = np.arange(N+1)
    y = np.arange(N-1)
    t = np.linspace(-int(N/2), int(N/2), N-6)
    z = np.arange(int(N/2))
    d4d = (t[:,np.newaxis,np.newaxis,np.newaxis]
            + z[np.newaxis,:,np.newaxis,np.newaxis]
            + y[np.newaxis,np.newaxis,:,np.newaxis]
            + x[np.newaxis,np.newaxis,np.newaxis,:]
          )
    da4d = xr.DataArray(d4d, dims=['time','z','y','x'],
                     coords={'time':range(len(t)),'z':range(len(z)),'y':range(len(y)),
                             'x':range(len(x))}
                     )

    func = xrft.detrend_wrap(xrft.detrendn)

    #########
    # Chunk along the `time` axis
    #########
    da = da4d.chunk({'time': 1})
    with pytest.raises(ValueError):
        func(da.data, axes=[0]).compute
    with pytest.raises(ValueError):
        func(da.data, axes=[0,1,2,3]).compute()
    da_prime = func(da.data, axes=[2]).compute()
    npt.assert_allclose(da_prime[0,0], sps.detrend(d4d[0,0], axis=0))
    da_prime = func(da.data, axes=[1,2,3]).compute()
    npt.assert_allclose(da_prime[0],
                        xrft.detrendn(d4d[0], axes=[0,1,2]))

    #########
    # Chunk along the `time` and `z` axes
    #########
    da = da4d.chunk({'time':1, 'z':1})
    with pytest.raises(ValueError):
        func(da.data, axes=[1,2]).compute()
    with pytest.raises(ValueError):
        func(da.data, axes=[2,2]).compute()
    da_prime = func(da.data, axes=[2,3]).compute()
    npt.assert_allclose(da_prime[0,0],
                        xrft.detrendn(d4d[0,0], axes=[0,1]))

class TestDFTImag(object):
    def test_dft_1d(self, test_data_1d):
        """Test the discrete Fourier transform function on one-dimensional data."""

        da = test_data_1d
        Nx = len(da)
        dx = float(da.x[1] - da.x[0]) if 'x' in da.dims else 1

        # defaults with no keyword args
        ft = xrft.dft(da, detrend='constant')
        # check that the frequency dimension was created properly
        assert ft.dims == ('freq_x',)
        # check that the coords are correct
        freq_x_expected = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
        npt.assert_allclose(ft['freq_x'], freq_x_expected)
        # check that a spacing variable was created
        assert ft['freq_x_spacing'] == freq_x_expected[1] - freq_x_expected[0]
        # make sure the function is lazy
        assert isinstance(ft.data, type(da.data))
        # check that the Fourier transform itself is correct
        data = (da - da.mean()).values
        ft_data_expected = np.fft.fftshift(np.fft.fft(data))
        # because the zero frequency component is zero, there is a numerical
        # precision issue. Fixed by setting atol
        npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)


        # redo without removing mean
        ft = xrft.dft(da)
        ft_data_expected = np.fft.fftshift(np.fft.fft(da))
        npt.assert_allclose(ft_data_expected, ft.values)

        # redo with detrending linear least-square fit
        ft = xrft.dft(da, detrend='linear')
        da_prime = sps.detrend(da.values)
        ft_data_expected = np.fft.fftshift(np.fft.fft(da_prime))
        npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)

        if 'x' in da and not da.chunks:
            da['x'].values[-1] *= 2
            with pytest.raises(ValueError):
                ft = xrft.dft(da)

    def test_dft_1d_time(self):
        """Test the discrete Fourier transform function on timeseries data."""
        time = pd.date_range('2000-01-01', '2001-01-01', closed='left')
        Nt = len(time)
        da = xr.DataArray(np.random.rand(Nt), coords=[time], dims=['time'])

        ft = xrft.dft(da)

        # check that frequencies are correct
        dt = (time[1] - time[0]).total_seconds()
        freq_time_expected = np.fft.fftshift(np.fft.fftfreq(Nt, dt))
        npt.assert_allclose(ft['freq_time'], freq_time_expected)

    def test_dft_2d(self):
        """Test the discrete Fourier transform on 2D data"""
        N = 16
        da = xr.DataArray(np.random.rand(N,N), dims=['x','y'],
                        coords={'x':range(N),'y':range(N)}
                         )
        ft = xrft.dft(da, shift=False)
        npt.assert_almost_equal(ft.values, np.fft.fftn(da.values))

        ft = xrft.dft(da, shift=False, window=True, detrend='constant')
        dim = da.dims
        window = np.hanning(N) * np.hanning(N)[:, np.newaxis]
        da_prime = (da - da.mean(dim=dim)).values
        npt.assert_almost_equal(ft.values, np.fft.fftn(da_prime*window))

    @pytest.mark.parametrize("dask", [False, True])
    def test_dft_3d_dask(self, dask):
        """Test the discrete Fourier transform on 3D dask array data"""
        N=16
        da = xr.DataArray(np.random.rand(N,N,N), dims=['time','x','y'],
                          coords={'time':range(N),'x':range(N),
                                  'y':range(N)}
                         )
        if dask:
            da = da.chunk({'time': 1})
            daft = xrft.dft(da, dim=['x','y'], shift=False)
            npt.assert_almost_equal(daft.values,
                                   np.fft.fftn(da.chunk({'time': 1}).values,
                                              axes=[1,2])
                                   )
            da = da.chunk({'x': 1})
            with pytest.raises(ValueError):
                xrft.dft(da, dim=['x'])

            da = da.chunk({'time':N})
            daft = xrft.dft(da, dim=['time'],
                            shift=False, detrend='linear')
            da_prime = sps.detrend(da, axis=0)
            npt.assert_almost_equal(daft.values,
                                   np.fft.fftn(da_prime, axes=[0])
                                   )

    def test_dft_4d(self):
        """Test the discrete Fourier transform on 2D data"""
        N = 16
        da = xr.DataArray(np.random.rand(N,N,N,N),
                         dims=['time','z','y','x'],
                         coords={'time':range(N),'z':range(N),
                                'y':range(N),'x':range(N)}
                         )
        with pytest.raises(ValueError):
            xrft.dft(da.chunk({'time':8}), dim=['y','x'], detrend='linear')
        ft = xrft.dft(da, shift=False)
        npt.assert_almost_equal(ft.values, np.fft.fftn(da.values))

        da_prime = xrft.detrendn(da[:,0].values, [0,1,2]) # cubic detrend over time, y, and x
        npt.assert_almost_equal(xrft.dft(da[:,0].drop('z'),
                                        dim=['time','y','x'],
                                        shift=False, detrend='linear'
                                        ).values,
                                np.fft.fftn(da_prime))


class TestDFTReal(object):
    def test_dft_real_1d(self, test_data_1d):
        """
        Test the discrete Fourier transform function on one-dimensional data.
        """
        da = test_data_1d
        Nx = len(da)
        dx = float(da.x[1] - da.x[0]) if 'x' in da.dims else 1

        # defaults with no keyword args
        ft = xrft.dft(da, real='x', detrend='constant')
        # check that the frequency dimension was created properly
        assert ft.dims == ('freq_x',)
        # check that the coords are correct
        freq_x_expected = np.fft.rfftfreq(Nx, dx)
        npt.assert_allclose(ft['freq_x'], freq_x_expected)
        # check that a spacing variable was created
        assert ft['freq_x_spacing'] == freq_x_expected[1] - freq_x_expected[0]
        # make sure the function is lazy
        assert isinstance(ft.data, type(da.data))
        # check that the Fourier transform itself is correct
        data = (da - da.mean()).values
        ft_data_expected = np.fft.rfft(data)
        # because the zero frequency component is zero, there is a numerical
        # precision issue. Fixed by setting atol
        npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)

        with pytest.raises(ValueError):
            xrft.dft(da, real='y', detrend='constant')

    def test_dft_real_2d(self):
        """
        Test the real discrete Fourier transform function on one-dimensional
        data. Non-trivial because we need to keep only some of the negative
        frequencies.
        """
        Nx, Ny = 16, 32
        da = xr.DataArray(np.random.rand(Nx, Ny), dims=['x', 'y'],
                          coords={'x': range(Nx), 'y': range(Ny)})
        dx = float(da.x[1] - da.x[0])
        dy = float(da.y[1] - da.y[0])

        daft = xrft.dft(da, real='x')
        npt.assert_almost_equal(daft.values,
                               np.fft.rfftn(da.transpose('y','x')).transpose())
        npt.assert_almost_equal(daft.values,
                               xrft.dft(da, dim=['y'], real='x'))

        actual_freq_x = daft.coords['freq_x'].values
        expected_freq_x = np.fft.rfftfreq(Nx, dx)
        npt.assert_almost_equal(actual_freq_x, expected_freq_x)

        actual_freq_y = daft.coords['freq_y'].values
        expected_freq_y = np.fft.fftfreq(Ny, dy)
        npt.assert_almost_equal(actual_freq_y, expected_freq_y)


def test_chunks_to_segments():
    N = 32
    da = xr.DataArray(np.random.rand(N,N,N),
                     dims=['time','y','x'],
                     coords={'time':range(N),'y':range(N),'x':range(N)}
                     )

    with pytest.raises(ValueError):
        xrft.dft(da.chunk(chunks=((20,N,N),(N-20,N,N))), dim=['time'],
                detrend='linear', chunks_to_segments=True)

    ft = xrft.dft(da.chunk({'time':16}), dim=['time'], shift=False,
                 chunks_to_segments=True)
    assert ft.dims == ('time_segment','freq_time','y','x')
    data = da.chunk({'time':16}).data.reshape((2,16,N,N))
    npt.assert_almost_equal(ft.values, dsar.fft.fftn(data, axes=[1]),
                           decimal=7)
    ft = xrft.dft(da.chunk({'y':16,'x':16}), dim=['y','x'], shift=False,
                 chunks_to_segments=True)
    assert ft.dims == ('time','y_segment','freq_y','x_segment','freq_x')
    data = da.chunk({'y':16,'x':16}).data.reshape((N,2,16,2,16))
    npt.assert_almost_equal(ft.values, dsar.fft.fftn(data, axes=[2,4]),
                           decimal=7)
    ps = xrft.power_spectrum(da.chunk({'y':16,'x':16}), dim=['y','x'],
                            shift=False, density=False,
                            chunks_to_segments=True)
    npt.assert_almost_equal(ps.values,
                           (ft*np.conj(ft)).real.values,
                           )
    da2 = xr.DataArray(np.random.rand(N,N,N),
                      dims=['time','y','x'],
                      coords={'time':range(N),'y':range(N),'x':range(N)}
                      )
    ft2 = xrft.dft(da2.chunk({'y':16,'x':16}), dim=['y','x'], shift=False,
                  chunks_to_segments=True)
    cs = xrft.cross_spectrum(da.chunk({'y':16,'x':16}),
                            da2.chunk({'y':16,'x':16}),
                            dim=['y','x'], shift=False, density=False,
                            chunks_to_segments=True)
    npt.assert_almost_equal(cs.values,
                           (ft*np.conj(ft2)).real.values,
                           )


def test_dft_nocoords():
    # Julius' example
    # https://github.com/rabernat/xrft/issues/17
    data = xr.DataArray(np.random.random([20,30,100]),dims=['time','lat','lon'])
    dft = xrft.dft(data,dim=['time'])
    ps = xrft.power_spectrum(data,dim=['time'])


def test_window_single_dim():
    # Julius' example
    # https://github.com/rabernat/xrft/issues/16
    data = xr.DataArray(np.random.random([20,30,100]),
                    dims=['time','lat','lon'],
                    coords={'time':range(20),'lat':range(30),'lon':range(100)})
    ps = xrft.power_spectrum(data, dim=['time'], window=True)
    # make sure it works with dask data
    ps = xrft.power_spectrum(data.chunk(), dim=['time'], window=True)
    ps.load()


class TestSpectrum(object):
    @pytest.mark.parametrize("dask", [False, True])
    def test_power_spectrum(self, dask):
        """Test the power spectrum function"""
        N = 16
        da = xr.DataArray(np.random.rand(2,N,N), dims=['time','y','x'],
                         coords={'time':np.array(['2019-04-18', '2019-04-19'],
                                                dtype='datetime64'),
                                'y':range(N),'x':range(N)}
                         )
        if dask:
            da = da.chunk({'time': 1})
        ps = xrft.power_spectrum(da, dim=['y','x'], window=True, density=False,
                                detrend='constant')
        daft = xrft.dft(da, dim=['y','x'], detrend='constant', window=True)
        npt.assert_almost_equal(ps.values, np.real(daft*np.conj(daft)))
        npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.)

        ps = xrft.power_spectrum(da, dim=['y'], real='x', window=True,
                                density=False, detrend='constant')
        daft = xrft.dft(da, dim=['y'], real='x', detrend='constant', window=True)
        npt.assert_almost_equal(ps.values, np.real(daft*np.conj(daft)))

        ### Normalized
        ps = xrft.power_spectrum(da, dim=['y','x'], window=True, detrend='constant')
        daft = xrft.dft(da, dim=['y','x'], window=True, detrend='constant')
        test = np.real(daft*np.conj(daft))/N**4
        dk = np.diff(np.fft.fftfreq(N, 1.))[0]
        test /= dk**2
        npt.assert_almost_equal(ps.values, test)
        npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.)

        ### Remove least-square fit
        ps = xrft.power_spectrum(da, dim=['y','x'],
                                window=True, density=False, detrend='linear'
                                )
        daft = xrft.dft(da, dim=['y','x'], window=True, detrend='linear')
        npt.assert_almost_equal(ps.values, np.real(daft*np.conj(daft)))
        npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.)

    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_spectrum(self, dask):
        """Test the cross spectrum function"""
        N = 16
        dim = ['x','y']
        da = xr.DataArray(np.random.rand(2,N,N), dims=['time','x','y'],
                          coords={'time':np.array(['2019-04-18', '2019-04-19'],
                                                 dtype='datetime64'),
                                 'x':range(N), 'y':range(N)})
        da2 = xr.DataArray(np.random.rand(2,N,N), dims=['time','x','y'],
                          coords={'time':np.array(['2019-04-18', '2019-04-19'],
                                                 dtype='datetime64'),
                                  'x':range(N), 'y':range(N)})
        if dask:
            da = da.chunk({'time': 1})
            da2 = da2.chunk({'time': 1})

        daft = xrft.dft(da, dim=dim, shift=True, detrend='constant',
                        window=True)
        daft2 = xrft.dft(da2, dim=dim, shift=True, detrend='constant',
                        window=True)
        cs = xrft.cross_spectrum(da, da2, dim=dim, window=True, density=False,
                                detrend='constant')
        npt.assert_almost_equal(cs.values, np.real(daft*np.conj(daft2)))
        npt.assert_almost_equal(np.ma.masked_invalid(cs).mask.sum(), 0.)

        cs = xrft.cross_spectrum(da, da2, dim=dim, shift=True, window=True,
                                detrend='constant')
        test = (daft * np.conj(daft2)).real.values/N**4

        dk = np.diff(np.fft.fftfreq(N, 1.))[0]
        test /= dk**2
        npt.assert_almost_equal(cs.values, test)
        npt.assert_almost_equal(np.ma.masked_invalid(cs).mask.sum(), 0.)


class TestCrossPhase(object):
    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_phase_1d(self, dask):
        N = 32
        x = np.linspace(0, 1, num=N, endpoint=False)
        f = 6
        phase_offset = np.pi/2
        signal1 = np.cos(2*np.pi*f*x)  # frequency = 1/(2*pi)
        signal2 = np.cos(2*np.pi*f*x - phase_offset)
        da1 = xr.DataArray(data=signal1, name='a', dims=['x'], coords={'x': x})
        da2 = xr.DataArray(data=signal2, name='b', dims=['x'], coords={'x': x})

        if dask:
            da1 = da1.chunk({'x': 32})
            da2 = da2.chunk({'x': 32})
        cp = xrft.cross_phase(da1, da2, dim=['x'])

        actual_phase_offset = cp.sel(freq_x=f).values
        npt.assert_almost_equal(actual_phase_offset, phase_offset)
        assert cp.name == 'a_b_phase'

        xrt.assert_equal(xrft.cross_phase(da1, da2), cp)

        with pytest.raises(ValueError):
            xrft.cross_phase(da1, da2.isel(x=0).drop('x'))

        with pytest.raises(ValueError):
            xrft.cross_phase(da1, da2.rename({'x':'y'}))

    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_phase_2d(self, dask):
        Ny, Nx = (32, 16)
        x = np.linspace(0, 1, num=Nx, endpoint=False)
        y = np.ones(Ny)
        f = 6
        phase_offset = np.pi/2
        signal1 = np.cos(2*np.pi*f*x)  # frequency = 1/(2*pi)
        signal2 = np.cos(2*np.pi*f*x - phase_offset)
        da1 = xr.DataArray(data=signal1*y[:,np.newaxis], name='a',
                          dims=['y','x'], coords={'y':y, 'x':x})
        da2 = xr.DataArray(data=signal2*y[:,np.newaxis], name='b',
                          dims=['y','x'], coords={'y':y, 'x':x})
        with pytest.raises(ValueError):
            xrft.cross_phase(da1, da2, dim=['y','x'])

        if dask:
            da1 = da1.chunk({'x': 16})
            da2 = da2.chunk({'x': 16})
        cp = xrft.cross_phase(da1, da2, dim=['x'])
        actual_phase_offset = cp.sel(freq_x=f).values
        npt.assert_almost_equal(actual_phase_offset, phase_offset)


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
        delta = diff[0]
        delta_x.append(delta)

    window = np.hanning(N) * np.hanning(N)[:, np.newaxis]
    ps = xrft.power_spectrum(da, window=True, detrend='constant')
    da_prime = da.values - da.mean(dim=dim).values
    npt.assert_almost_equal(ps.values.sum(),
                            (np.asarray(delta_x).prod()
                            * ((da_prime*window)**2).sum()
                            ), decimal=5
                            )

    cs = xrft.cross_spectrum(da, da2, window=True, detrend='constant')
    da2_prime = da2.values - da2.mean(dim=dim).values
    npt.assert_almost_equal(cs.values.sum(),
                            (np.asarray(delta_x).prod()
                            * ((da_prime*window)
                            * (da2_prime*window)).sum()
                            ), decimal=5
                            )

    d3d = xr.DataArray(np.random.rand(N,N,N),
                    dims=['time','y','x'],
                    coords={'time':range(N), 'y':range(N), 'x':range(N)}
                      ).chunk({'time':1})
    ps = xrft.power_spectrum(d3d, dim=['x','y'], window=True, detrend='linear')
    npt.assert_almost_equal(ps[0].values.sum(),
                            (np.asarray(delta_x).prod()
                            * ((numpy_detrend(d3d[0].values)*window)**2).sum()
                            ), decimal=5
                           )

def _synthetic_field(N, dL, amp, s):
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
    F_theta = r_kl * np.exp(1j * phi)

    # check that symmetry of FT is satisfied
    theta = np.fft.ifft2(np.fft.ifftshift(F_theta))
    return np.real(theta)

def test_isotropic_ps_slope(N=512, dL=1., amp=1e1, s=-3.):
    """Test the spectral slope of isotropic power spectrum."""

    theta = xr.DataArray(_synthetic_field(N, dL, amp, s),
                        dims=['y', 'x'],
                        coords={'y':range(N), 'x':range(N)})
    iso_ps = xrft.isotropic_powerspectrum(theta, detrend='constant',
                                         density=True)
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps[1:]).mask.sum(), 0.)
    y_fit, a, b = xrft.fit_loglog(iso_ps.freq_r.values[4:],
                                 iso_ps.values[4:])

    npt.assert_allclose(a, s, atol=.1)

def test_isotropic_ps():
    """Test data with extra coordinates"""
    da = xr.DataArray(np.random.rand(2,5,5,16,32),
                  dims=['time','zz','z','y', 'x'],
                  coords={'time': np.array(['2019-04-18', '2019-04-19'],
                                          dtype='datetime64'),
                         'zz': np.arange(5), 'z': np.arange(5),
                         'y': np.arange(16), 'x': np.arange(32)})
    with pytest.raises(ValueError):
        xrft.isotropic_powerspectrum(da, dim=['y','x'])
    da = da[:,0,:,:,:].drop(['zz'])
    with pytest.raises(ValueError):
        xrft.isotropic_powerspectrum(da, dim=['z','y','x'])
    iso_ps = xrft.isotropic_powerspectrum(da, dim=['y','x']).values
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps[:,:,1:]).mask.sum(), 0.)

def test_isotropic_cs():
    """Test isotropic cross spectrum"""
    N = 16
    da = xr.DataArray(np.random.rand(N,N),
                    dims=['y','x'], coords={'y':range(N),'x':range(N)})
    da2 = xr.DataArray(np.random.rand(N,N),
                    dims=['y','x'], coords={'y':range(N),'x':range(N)})

    iso_cs = xrft.isotropic_crossspectrum(da, da2, window=True)
    npt.assert_almost_equal(np.ma.masked_invalid(iso_cs[1:]).mask.sum(), 0.)

    da2 = xr.DataArray(np.random.rand(N,N),
                    dims=['lat','lon'],
                    coords={'lat':range(N),'lon':range(N)})
    with pytest.raises(ValueError):
        xrft.isotropic_crossspectrum(da, da2)

    da = xr.DataArray(np.random.rand(2,5,5,16,32),
                  dims=['time','zz','z','y', 'x'],
                  coords={'time': np.array(['2019-04-18', '2019-04-19'],
                                          dtype='datetime64'),
                         'zz': np.arange(5), 'z': np.arange(5),
                         'y': np.arange(16), 'x': np.arange(32)})
    da2 = xr.DataArray(np.random.rand(2,5,5,16,32),
                  dims=['time','zz','z','y', 'x'],
                  coords={'time': np.array(['2019-04-18', '2019-04-19'],
                                          dtype='datetime64'),
                         'zz': np.arange(5), 'z': np.arange(5),
                         'y': np.arange(16), 'x': np.arange(32)})
    with pytest.raises(ValueError):
        xrft.isotropic_crossspectrum(da, da2, dim=['y','x'])
    da = da[:,0,:,:,:].drop(['zz'])
    da2 = da2[:,0,:,:,:].drop(['zz'])
    with pytest.raises(ValueError):
        xrft.isotropic_crossspectrum(da, da2, dim=['z','y','x'])

    iso_cs = xrft.isotropic_crossspectrum(da, da2, dim=['y','x'],
                                         window=True).values
    npt.assert_almost_equal(np.ma.masked_invalid(iso_cs[:,:,1:]).mask.sum(), 0.)

def test_spacing_tol(test_data_1d):
    da = test_data_1d
    da2 = da.copy().load()

    # Create improperly spaced data
    Nx = 16
    Lx = 1.0
    x  = np.linspace(0, Lx, Nx)
    x[-1] = x[-1] + .001
    da3 = xr.DataArray(np.random.rand(Nx), coords=[x], dims=['x'])

    # This shouldn't raise an error
    xrft.dft(da3, spacing_tol=1e-1)
    # But this should
    with pytest.raises(ValueError):
        xrft.dft(da3, spacing_tol=1e-4)

def test_spacing_tol_float_value(test_data_1d):
    da = test_data_1d
    with pytest.raises(TypeError):
        xrft.dft(da, spacing_tol='string')
