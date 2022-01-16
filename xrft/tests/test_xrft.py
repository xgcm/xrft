import numpy as np
import pandas as pd
import xarray as xr
import cftime
import dask.array as dsar

import scipy.signal as sps
import scipy.linalg as spl

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft
from ..xrft import _apply_window


@pytest.fixture()
def sample_data_3d():
    """Create three dimensional test data."""
    temp = 10 * np.random.rand(2, 2, 10)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    ds = xr.Dataset(
        {"temp": (["x", "y", "time"], temp)},
        coords={
            "lon": (["x", "y"], lon),
            "lat": (["x", "y"], lat),
            "time": np.arange(10),
        },
    )
    return ds


@pytest.fixture(params=["numpy", "dask", "nocoords"])
def test_data_1d(request):
    """Create one dimensional test DataArray."""
    Nx = 16
    Lx = 1.0
    x = np.linspace(0, Lx, Nx)
    dx = x[1] - x[0]
    coords = None if request.param == "nocoords" else [x]
    da = xr.DataArray(np.random.rand(Nx), coords=coords, dims=["x"])
    if request.param == "dask":
        da = da.chunk()
    return da


@pytest.fixture(params=["pandas", "standard", "julian", "365_day", "360_day"])
def time_data(request):
    if request.param == "pandas":
        return pd.date_range("2000-01-01", "2001-01-01", closed="left")
    else:
        units = "days since 2000-01-01 00:00:00"
        return cftime.num2date(np.arange(0, 10 * 365), units, request.param)


class TestFFTImag(object):
    def test_fft_1d(self, test_data_1d):
        """Test the discrete Fourier transform function on one-dimensional data."""

        da = test_data_1d
        Nx = len(da)
        dx = float(da.x[1] - da.x[0]) if "x" in da.dims else 1

        # defaults with no keyword args
        ft = xrft.fft(da, detrend="constant")
        # check that the frequency dimension was created properly
        assert ft.dims == ("freq_x",)
        # check that the coords are correct
        freq_x_expected = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
        npt.assert_allclose(ft["freq_x"], freq_x_expected)
        # check that a spacing variable was created
        assert ft["freq_x"].spacing == freq_x_expected[1] - freq_x_expected[0]
        # make sure the function is lazy
        assert isinstance(ft.data, type(da.data))
        # check that the Fourier transform itself is correct
        data = (da - da.mean()).values
        ft_data_expected = np.fft.fftshift(np.fft.fft(data))
        # because the zero frequency component is zero, there is a numerical
        # precision issue. Fixed by setting atol
        npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)

        # redo without removing mean
        ft = xrft.fft(da)
        ft_data_expected = np.fft.fftshift(np.fft.fft(da))
        npt.assert_allclose(ft_data_expected, ft.values)

        # redo with detrending linear least-square fit
        ft = xrft.fft(da, detrend="linear")
        da_prime = sps.detrend(da.values)
        ft_data_expected = np.fft.fftshift(np.fft.fft(da_prime))
        npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)

        if "x" in da and not da.chunks:
            da["x"].values[-1] *= 2
            with pytest.raises(ValueError):
                ft = xrft.fft(da)

    def test_fft_1d_time(self, time_data):
        """Test the discrete Fourier transform function on timeseries data."""
        time = time_data
        Nt = len(time)
        da = xr.DataArray(np.random.rand(Nt), coords=[time], dims=["time"])

        ft = xrft.fft(da, shift=False)

        # check that frequencies are correct
        if pd.api.types.is_datetime64_dtype(time):
            dt = (time[1] - time[0]).total_seconds()
        else:
            dt = np.diff(time)[0].total_seconds()
        freq_time_expected = np.fft.fftfreq(Nt, dt)
        npt.assert_allclose(ft["freq_time"], freq_time_expected)

    def test_fft_2d(self):
        """Test the discrete Fourier transform on 2D data"""
        N = 16
        da = xr.DataArray(
            np.random.rand(N, N), dims=["x", "y"], coords={"x": range(N), "y": range(N)}
        )
        ft = xrft.fft(da, shift=False)
        npt.assert_almost_equal(ft.values, np.fft.fftn(da.values))

        ft = xrft.fft(da, shift=False, window="hann", detrend="constant")
        dim = da.dims
        window = (
            sps.windows.hann(N, sym=False)
            * sps.windows.hann(N, sym=False)[:, np.newaxis]
        )
        da_prime = (da - da.mean(dim=dim)).values
        npt.assert_almost_equal(ft.values, np.fft.fftn(da_prime * window))

        da = xr.DataArray(
            np.random.rand(N, N),
            dims=["x", "y"],
            coords={"x": range(N, 0, -1), "y": range(N, 0, -1)},
        )
        assert (xrft.power_spectrum(da, shift=False, density=True) >= 0.0).all()

    def test_dim_fft(self):
        N = 16
        da = xr.DataArray(
            np.random.rand(N, N), dims=["x", "y"], coords={"x": range(N), "y": range(N)}
        )
        npt.assert_array_equal(
            xrft.fft(da, dim="y", shift=False).values,
            xrft.fft(da, dim=["y"], shift=False).values,
        )
        assert xrft.fft(da, dim="y").dims == ("x", "freq_y")

    @pytest.mark.parametrize("dask", [False, True])
    def test_fft_3d_dask(self, dask):
        """Test the discrete Fourier transform on 3D dask array data"""
        N = 16
        da = xr.DataArray(
            np.random.rand(N, N, N),
            dims=["time", "x", "y"],
            coords={"time": range(N), "x": range(N), "y": range(N)},
        )
        if dask:
            da = da.chunk({"time": 1})
            daft = xrft.fft(da, dim=["x", "y"], shift=False)
            npt.assert_almost_equal(
                daft.values, np.fft.fftn(da.chunk({"time": 1}).values, axes=[1, 2])
            )
            da = da.chunk({"x": 1})
            with pytest.raises(ValueError):
                xrft.fft(da, dim=["x"])
            with pytest.raises(ValueError):
                xrft.fft(da, dim="x")

            da = da.chunk({"time": N})
            daft = xrft.fft(da, dim=["time"], shift=False, detrend="linear")
            da_prime = sps.detrend(da, axis=0)
            npt.assert_almost_equal(daft.values, np.fft.fftn(da_prime, axes=[0]))
            npt.assert_array_equal(
                daft.values, xrft.fft(da, dim="time", shift=False, detrend="linear")
            )

    @pytest.mark.skip(reason="3D detrending not implemented")
    def test_fft_4d(self):
        """Test the discrete Fourier transform on 2D data"""
        N = 16
        da = xr.DataArray(
            np.random.rand(N, N, N, N),
            dims=["time", "z", "y", "x"],
            coords={"time": range(N), "z": range(N), "y": range(N), "x": range(N)},
        )
        ft = xrft.fft(da, shift=False)
        npt.assert_almost_equal(ft.values, np.fft.fftn(da.values))

        dim = ["time", "y", "x"]
        da_prime = xrft.detrend(da[:, 0], dim)  # cubic detrend over time, y, and x
        npt.assert_almost_equal(
            xrft.fft(
                da[:, 0].drop("z"),
                dim=dim,
                shift=False,
                detrend="linear",
            ).values,
            np.fft.fftn(da_prime),
        )


class TestfftReal(object):
    def test_fft_real_1d(self, test_data_1d):
        """
        Test the discrete Fourier transform function on one-dimensional data.
        """
        da = test_data_1d
        Nx = len(da)
        dx = float(da.x[1] - da.x[0]) if "x" in da.dims else 1

        # defaults with no keyword args
        ft = xrft.fft(da, real_dim="x", detrend="constant")
        # check that the frequency dimension was created properly
        assert ft.dims == ("freq_x",)
        # check that the coords are correct
        freq_x_expected = np.fft.rfftfreq(Nx, dx)
        npt.assert_allclose(ft["freq_x"], freq_x_expected)
        # check that a spacing variable was created
        assert ft["freq_x"].spacing == freq_x_expected[1] - freq_x_expected[0]
        # make sure the function is lazy
        assert isinstance(ft.data, type(da.data))
        # check that the Fourier transform itself is correct
        data = (da - da.mean()).values
        ft_data_expected = np.fft.rfft(data)
        # because the zero frequency component is zero, there is a numerical
        # precision issue. Fixed by setting atol
        npt.assert_allclose(ft_data_expected, ft.values, atol=1e-14)

        with pytest.raises(ValueError):
            xrft.fft(da, real_dim="y", detrend="constant")

    def test_fft_real_2d(self):
        """
        Test the real discrete Fourier transform function on one-dimensional
        data. Non-trivial because we need to keep only some of the negative
        frequencies.
        """
        Nx, Ny = 16, 32
        da = xr.DataArray(
            np.random.rand(Nx, Ny),
            dims=["x", "y"],
            coords={"x": range(Nx), "y": range(Ny)},
        )
        dx = float(da.x[1] - da.x[0])
        dy = float(da.y[1] - da.y[0])

        daft = xrft.fft(da, real_dim="x")
        npt.assert_almost_equal(
            daft.values, np.fft.rfftn(da.transpose("y", "x")).transpose()
        )
        npt.assert_almost_equal(daft.values, xrft.fft(da, dim=["y"], real_dim="x"))

        actual_freq_x = daft.coords["freq_x"].values
        expected_freq_x = np.fft.rfftfreq(Nx, dx)
        npt.assert_almost_equal(actual_freq_x, expected_freq_x)

        actual_freq_y = daft.coords["freq_y"].values
        expected_freq_y = np.fft.fftfreq(Ny, dy)
        npt.assert_almost_equal(actual_freq_y, expected_freq_y)


def test_chunks_to_segments():
    N = 32
    da = xr.DataArray(
        np.random.rand(N, N, N),
        dims=["time", "y", "x"],
        coords={"time": range(N), "y": range(N), "x": range(N)},
    )

    with pytest.raises(ValueError):
        xrft.fft(
            da.chunk(chunks=((20, N, N), (N - 20, N, N))),
            dim=["time"],
            detrend="linear",
            chunks_to_segments=True,
        )

    ft = xrft.fft(
        da.chunk({"time": 16}), dim=["time"], shift=False, chunks_to_segments=True
    )
    assert ft.dims == ("time_segment", "freq_time", "y", "x")
    data = da.chunk({"time": 16}).data.reshape((2, 16, N, N))
    npt.assert_almost_equal(ft.values, dsar.fft.fftn(data, axes=[1]), decimal=7)
    ft = xrft.fft(
        da.chunk({"y": 16, "x": 16}),
        dim=["y", "x"],
        shift=False,
        chunks_to_segments=True,
    )
    assert ft.dims == ("time", "y_segment", "freq_y", "x_segment", "freq_x")
    data = da.chunk({"y": 16, "x": 16}).data.reshape((N, 2, 16, 2, 16))
    npt.assert_almost_equal(ft.values, dsar.fft.fftn(data, axes=[2, 4]), decimal=7)
    ps = xrft.power_spectrum(
        da.chunk({"y": 16, "x": 16}),
        dim=["y", "x"],
        shift=False,
        density=False,
        chunks_to_segments=True,
    )
    npt.assert_almost_equal(
        ps.values,
        (ft * np.conj(ft)).values,
    )
    da2 = xr.DataArray(
        np.random.rand(N, N, N),
        dims=["time", "y", "x"],
        coords={"time": range(N), "y": range(N), "x": range(N)},
    )
    ft2 = xrft.fft(
        da2.chunk({"y": 16, "x": 16}),
        dim=["y", "x"],
        shift=False,
        chunks_to_segments=True,
    )
    cs = xrft.cross_spectrum(
        da.chunk({"y": 16, "x": 16}),
        da2.chunk({"y": 16, "x": 16}),
        dim=["y", "x"],
        shift=False,
        density=False,
        chunks_to_segments=True,
    )
    npt.assert_almost_equal(
        cs.values,
        (ft * np.conj(ft2)).values,
    )


def test_fft_nocoords():
    # Julius' example
    # https://github.com/rabernat/xrft/issues/17
    data = xr.DataArray(np.random.random([20, 30, 100]), dims=["time", "lat", "lon"])
    dft = xrft.fft(data, dim=["time"])
    ps = xrft.power_spectrum(data, dim=["time"])


def test_window_single_dim():
    # Julius' example
    # https://github.com/rabernat/xrft/issues/16
    data = xr.DataArray(
        np.random.random([20, 30, 100]),
        dims=["time", "lat", "lon"],
        coords={"time": range(20), "lat": range(30), "lon": range(100)},
    )
    ps = xrft.power_spectrum(data, dim=["time"], window="hann")
    # make sure it works with dask data
    ps = xrft.power_spectrum(data.chunk(), dim=["time"], window="hann")
    ps.load()


class TestSpectrum(object):
    @pytest.mark.parametrize("dim", ["t", "time"])
    @pytest.mark.parametrize("window_correction", [True, False])
    @pytest.mark.parametrize("detrend", ["constant", "linear"])
    def test_dim_format(self, dim, window_correction, detrend):
        """Check that can deal with dim in various formats"""
        data = xr.DataArray(
            np.random.random([10]),
            dims=[dim],
            coords={dim: range(10)},
        )
        ps = xrft.power_spectrum(
            data,
            dim=dim,
            window="hann",
            window_correction=window_correction,
            detrend=detrend,
        )
        ps = xrft.power_spectrum(
            data,
            dim=[dim],
            window="hann",
            window_correction=window_correction,
            detrend=detrend,
        )

    @pytest.mark.parametrize("dask", [False, True])
    def test_power_spectrum(self, dask):
        """Test the power spectrum function"""

        N = 16
        da = xr.DataArray(
            np.random.rand(N),
            dims=["x"],
            coords={
                "x": range(N),
            },
        )
        f_scipy, p_scipy = sps.periodogram(
            da.values, window="rectangular", return_onesided=True
        )
        ps = xrft.power_spectrum(da, dim="x", real_dim="x", detrend="constant")
        npt.assert_almost_equal(ps.values, p_scipy)

        A = 20
        fs = 1e4
        n_segments = int(fs // 10)
        fsig = 300
        ii = int(fsig * n_segments // fs)  # Freq index of fsig

        tt = np.arange(fs) / fs
        x = A * np.sin(2 * np.pi * fsig * tt)
        for window_type in ["hann", "bartlett", "tukey", "flattop"]:
            # see https://github.com/scipy/scipy/blob/master/scipy/signal/tests/test_spectral.py#L485

            x_da = xr.DataArray(x, coords=[tt], dims=["t"]).chunk({"t": n_segments})
            ps = xrft.power_spectrum(
                x_da,
                dim="t",
                window=window_type,
                chunks_to_segments=True,
                window_correction=True,
            ).mean("t_segment")
            # Check the energy correction
            npt.assert_allclose(
                np.sqrt(np.trapz(ps.values, ps.freq_t.values)),
                A * np.sqrt(2) / 2,
                rtol=1e-3,
            )

            ps = xrft.power_spectrum(
                x_da,
                dim="t",
                window=window_type,
                chunks_to_segments=True,
                scaling="spectrum",
                window_correction=True,
            ).mean("t_segment")
            # Check the amplitude correction
            # The factor of 0.5 is there because we're checking the two-sided spectrum
            npt.assert_allclose(ps.sel(freq_t=fsig), 0.5 * A ** 2 / 2.0)

        da = xr.DataArray(
            np.random.rand(2, N, N),
            dims=["time", "y", "x"],
            coords={
                "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
                "y": range(N),
                "x": range(N),
            },
        )
        if dask:
            da = da.chunk({"time": 1})
        ps = xrft.power_spectrum(
            da, dim=["y", "x"], window="hann", density=False, detrend="constant"
        )
        daft = xrft.fft(da, dim=["y", "x"], detrend="constant", window="hann")
        npt.assert_almost_equal(ps.values, np.real(daft * np.conj(daft)))
        npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.0)

        ps = xrft.power_spectrum(
            da,
            dim=["y"],
            real_dim="x",
            window="hann",
            density=False,
            detrend="constant",
        )
        daft = xrft.fft(da, dim=["y"], real_dim="x", detrend="constant", window="hann")
        ps_test = np.real(daft * np.conj(daft))
        f = np.full(ps_test.sizes["freq_x"], 2.0)
        f[0], f[-1] = 1.0, 1.0
        ps_test = ps_test * xr.DataArray(f, dims="freq_x")
        npt.assert_almost_equal(ps.values, ps_test.values)

        ### Normalized
        ps = xrft.power_spectrum(da, dim=["y", "x"], window="hann", detrend="constant")
        daft = xrft.fft(da, dim=["y", "x"], window="hann", detrend="constant")
        test = np.real(daft * np.conj(daft)) / N ** 4
        dk = np.diff(np.fft.fftfreq(N, 1.0))[0]
        test /= dk ** 2
        npt.assert_almost_equal(ps.values, test)
        npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.0)

        ### Remove least-square fit
        ps = xrft.power_spectrum(
            da, dim=["y", "x"], window="hann", density=False, detrend="linear"
        )
        daft = xrft.fft(da, dim=["y", "x"], window="hann", detrend="linear")
        npt.assert_almost_equal(ps.values, np.real(daft * np.conj(daft)))
        npt.assert_almost_equal(np.ma.masked_invalid(ps).mask.sum(), 0.0)

        with pytest.raises(ValueError):
            xrft.power_spectrum(da, dim=["y", "x"], window=None, window_correction=True)

    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_spectrum(self, dask):
        """Test the cross spectrum function"""
        N = 16
        dim = ["x", "y"]
        da = xr.DataArray(
            np.random.rand(2, N, N),
            dims=["time", "x", "y"],
            coords={
                "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
                "x": range(N),
                "y": range(N),
            },
        )
        da2 = xr.DataArray(
            np.random.rand(2, N, N),
            dims=["time", "x", "y"],
            coords={
                "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
                "x": range(N),
                "y": range(N),
            },
        )
        if dask:
            da = da.chunk({"time": 1})
            da2 = da2.chunk({"time": 1})

        daft = xrft.fft(da, dim=dim, shift=True, detrend="constant", window="hann")
        daft2 = xrft.fft(da2, dim=dim, shift=True, detrend="constant", window="hann")
        cs = xrft.cross_spectrum(
            da, da2, dim=dim, window="hann", density=False, detrend="constant"
        )
        npt.assert_almost_equal(cs.values, daft * np.conj(daft2))
        npt.assert_almost_equal(np.ma.masked_invalid(cs).mask.sum(), 0.0)

        cs = xrft.cross_spectrum(
            da, da2, dim=dim, shift=True, window="hann", detrend="constant"
        )
        test = (daft * np.conj(daft2)) / N ** 4

        dk = np.diff(np.fft.fftfreq(N, 1.0))[0]
        test /= dk ** 2
        npt.assert_almost_equal(cs.values, test)
        npt.assert_almost_equal(np.ma.masked_invalid(cs).mask.sum(), 0.0)

        cs = xrft.cross_spectrum(
            da,
            da2,
            dim=dim,
            shift=True,
            window="hann",
            detrend="constant",
            window_correction=True,
        )
        test = (daft * np.conj(daft2)) / N ** 4
        window, _ = _apply_window(da, dim, window_type="hann")
        dk = np.diff(np.fft.fftfreq(N, 1.0))[0]
        test /= dk ** 2 * (window ** 2).mean()

        npt.assert_almost_equal(cs.values, test)
        npt.assert_almost_equal(np.ma.masked_invalid(cs).mask.sum(), 0.0)

        with pytest.raises(ValueError):
            xrft.cross_spectrum(da, da2, dim=dim, window=None, window_correction=True)

    def test_spectrum_dim(self):
        N = 16
        da = xr.DataArray(
            np.random.rand(2, N, N),
            dims=["time", "y", "x"],
            coords={
                "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
                "y": range(N),
                "x": range(N),
            },
        )

        ps = xrft.power_spectrum(
            da, dim="y", real_dim="x", window="hann", detrend="constant"
        )
        npt.assert_array_equal(
            ps.values,
            xrft.power_spectrum(
                da, dim=["y"], real_dim="x", window="hann", detrend="constant"
            ).values,
        )
        assert ps.dims == ("time", "freq_y", "freq_x")

        da2 = xr.DataArray(
            np.random.rand(2, N, N),
            dims=["time", "y", "x"],
            coords={
                "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
                "y": range(N),
                "x": range(N),
            },
        )
        cs = xrft.cross_spectrum(
            da, da2, dim="y", shift=True, window="hann", detrend="constant"
        )
        npt.assert_array_equal(
            xrft.cross_spectrum(
                da, da2, dim=["y"], shift=True, window="hann", detrend="constant"
            ).values,
            cs.values,
        )
        assert cs.dims == ("time", "freq_y", "x")


class TestCrossPhase(object):
    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_phase_1d(self, dask):
        N = 32
        x = np.linspace(0, 1, num=N, endpoint=False)
        f = 6
        phase_offset = np.pi / 2
        signal1 = np.cos(2 * np.pi * f * x)  # frequency = 1/(2*pi)
        signal2 = np.cos(2 * np.pi * f * x - phase_offset)
        da1 = xr.DataArray(data=signal1, name="a", dims=["x"], coords={"x": x})
        da2 = xr.DataArray(data=signal2, name="b", dims=["x"], coords={"x": x})

        if dask:
            da1 = da1.chunk({"x": 32})
            da2 = da2.chunk({"x": 32})
        cp = xrft.cross_phase(da1, da2, dim=["x"])

        actual_phase_offset = cp.sel(freq_x=f).values
        npt.assert_almost_equal(actual_phase_offset, phase_offset)
        assert cp.name == "a_b_phase"

        xrt.assert_equal(xrft.cross_phase(da1, da2), cp)

        with pytest.raises(ValueError):
            xrft.cross_phase(da1, da2.isel(x=0).drop("x"))

        with pytest.raises(ValueError):
            xrft.cross_phase(da1, da2.rename({"x": "y"}))

    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_phase_2d(self, dask):
        dx = 0.1
        dy = 0.14
        x = np.arange(-10, 10, dx)
        y = np.arange(-18, 18, dy)
        x = xr.DataArray(x, dims="x", coords={"x": x})
        y = xr.DataArray(y, dims="y", coords={"y": y})
        fx = np.random.choice(np.fft.fftfreq(len(x), dx))
        fy = np.random.choice(np.fft.fftfreq(len(y), dy))
        phase_offset = 2 * (np.random.rand() - 0.5) * np.pi
        da1 = np.cos(2 * np.pi * fx * x + 2 * np.pi * fy * y)
        da2 = np.cos(2 * np.pi * fx * x + 2 * np.pi * fy * y - phase_offset)
        if dask:
            da1 = da1.chunk()
            da2 = da2.chunk()
        cp = xrft.cross_phase(da1, da2)
        offset = cp[
            {
                "freq_x": (np.abs(cp["freq_x"] - fx)).argmin(),
                "freq_y": (np.abs(cp["freq_y"] - fy)).argmin(),
            }
        ].data
        npt.assert_almost_equal(offset, phase_offset)

    @pytest.mark.parametrize("dask", [False, True])
    def test_cross_phase_true_phase_2d(self, dask):
        """With true_phase = True, a lag on the coordinates should be recovered in cross_phase"""
        dx = 0.1
        dy = 0.14
        x = np.arange(-10, 10, dx)
        y = np.arange(-18, 18, dy)
        x = xr.DataArray(x, dims="x", coords={"x": x})
        y = xr.DataArray(y, dims="y", coords={"y": y})
        fx = np.random.choice(np.fft.fftfreq(len(x), dx))
        fy = np.random.choice(np.fft.fftfreq(len(y), dy))
        da1 = np.cos(2 * np.pi * fx * x + 2 * np.pi * fy * y)
        np.random.seed(0)
        lagx = np.random.rand() * x.max().data
        lagy = np.random.rand() * y.max().data
        da2 = da1.assign_coords(x=da1["x"] + lagx, y=da1["y"] + lagy)
        if dask:
            da1 = da1.chunk()
            da2 = da2.chunk()
        cp = xrft.cross_phase(da1, da2, true_phase=True)
        offset = cp[
            {
                "freq_x": (np.abs(cp["freq_x"] - fx)).argmin(),
                "freq_y": (np.abs(cp["freq_y"] - fy)).argmin(),
            }
        ].data
        phase_offset = 2 * np.pi * (fx * lagx + fy * lagy)
        phase_offset = np.arctan2(
            np.sin(phase_offset), np.cos(phase_offset)
        )  # Offset in [-pi, pi]
        npt.assert_almost_equal(np.float(offset), phase_offset)


@pytest.mark.parametrize("chunks_to_segments", [False, True])
def test_parseval(chunks_to_segments):
    """Test whether the Parseval's relation is satisfied."""

    N = 16  # Must be divisible by n_segments (below)
    da = xr.DataArray(
        np.random.rand(N, N), dims=["x", "y"], coords={"x": range(N), "y": range(N)}
    )
    da2 = xr.DataArray(
        np.random.rand(N, N), dims=["x", "y"], coords={"x": range(N), "y": range(N)}
    )

    if chunks_to_segments:
        n_segments = 2
        # Chunk da and da2 into n_segments
        da = da.chunk({"x": N / n_segments, "y": N / n_segments})
        da2 = da2.chunk({"x": N / n_segments, "y": N / n_segments})
    else:
        n_segments = 1

    dim = da.dims
    fftdim = [f"freq_{d}" for d in da.dims]
    delta_x = []
    for d in dim:
        coord = da[d]
        diff = np.diff(coord)
        delta = diff[0]
        delta_x.append(delta)
    delta_xy = np.asarray(delta_x).prod()  # Area of the spacings

    ### Test Parseval's theorem for power_spectrum with `window=False` and detrend=None
    ps = xrft.power_spectrum(da, chunks_to_segments=chunks_to_segments)
    # If n_segments > 1, use xrft._stack_chunks() to stack each segment along a new dimension
    da_seg = xrft.xrft._stack_chunks(da, dim).squeeze() if chunks_to_segments else da
    da_prime = da_seg
    # Check that the (rectangular) integral of the spectrum matches the energy
    npt.assert_almost_equal(
        (1 / delta_xy) * ps.mean(fftdim).values,
        (da_prime ** 2).mean(dim).values,
        decimal=5,
    )

    ### Test Parseval's theorem for power_spectrum with `window=True` and detrend='constant'
    # Note that applying a window weighting reduces the energy in a signal and we have to account
    # for this reduction when testing Parseval's theorem.
    ps = xrft.power_spectrum(
        da, window="hann", detrend="constant", chunks_to_segments=chunks_to_segments
    )
    # If n_segments > 1, use xrft._stack_chunks() to stack each segment along a new dimension
    da_seg = xrft.xrft._stack_chunks(da, dim).squeeze() if chunks_to_segments else da
    da_prime = da_seg - da_seg.mean(dim=dim)
    # Generate the window weightings for each segment
    window = xr.DataArray(
        np.tile(
            # np.hanning(N / n_segments) * np.hanning(N / n_segments)[:, np.newaxis],
            sps.windows.hann(int(N / n_segments), sym=False)
            * sps.windows.hann(int(N / n_segments), sym=False)[:, np.newaxis],
            (n_segments, n_segments),
        ),
        dims=dim,
        coords=da.coords,
    )

    # Check that the (rectangular) integral of the spectrum matches the windowed variance
    npt.assert_almost_equal(
        (1 / delta_xy) * ps.mean(fftdim).values,
        ((da_prime * window) ** 2).mean(dim).values,
        decimal=5,
    )

    ### Test Parseval's theorem for cross_spectrum with `window=True` and detrend='constant'
    cs = xrft.cross_spectrum(
        da,
        da2,
        window="hann",
        detrend="constant",
        chunks_to_segments=chunks_to_segments,
    )
    # If n_segments > 1, use xrft._stack_chunks() to stack each segment along a new dimension
    da2_seg = xrft.xrft._stack_chunks(da2, dim).squeeze() if chunks_to_segments else da2
    da2_prime = da2_seg - da2_seg.mean(dim=dim)
    # Check that the (rectangular) integral of the cross-spectrum matches the windowed co-variance
    npt.assert_almost_equal(
        (1 / delta_xy) * cs.mean(fftdim).values,
        ((da_prime * window) * (da2_prime * window)).mean(dim).values,
        decimal=5,
    )

    ### Test Parseval's theorem for a 3D case with `window=True` and `detrend='linear'`
    if not chunks_to_segments:
        d3d = xr.DataArray(
            np.random.rand(N, N, N),
            dims=["time", "y", "x"],
            coords={"time": range(N), "y": range(N), "x": range(N)},
        ).chunk({"time": 1})
        dim = ["x", "y"]
        ps = xrft.power_spectrum(d3d, dim=dim, window="hann", detrend="linear")
        npt.assert_almost_equal(
            (1 / delta_xy) * ps[0].values.mean(),
            (
                (xrft.detrend(d3d, dim, detrend_type="linear")[0].values * window)
                ** 2
                # (xrft.detrend(d3d, dim, detrend_type="linear")[0].values) ** 2
            ).mean(),
            decimal=5,
        )

    ###Testing parseval identity in 1D with dft###
    Nx = 40
    dx = np.random.rand()
    s = xr.DataArray(
        np.random.rand(Nx) + 1j * np.random.rand(Nx),
        dims="x",
        coords={
            "x": dx
            * (
                np.arange(-Nx // 2, -Nx // 2 + Nx)
                + np.random.randint(-Nx // 2, Nx // 2)
            )
        },
    )
    FTs = xrft.dft(s, dim="x", true_phase=True, true_amplitude=True)
    npt.assert_almost_equal(
        (np.abs(s) ** 2).sum() * dx, (np.abs(FTs) ** 2).sum() * FTs["freq_x"].spacing
    )

    ###Testing parseval identity in 2D with dft###
    Nx, Ny = 40, 60
    dx, dy = np.random.rand(), np.random.rand()
    s = xr.DataArray(
        np.random.rand(Nx, Ny) + 1j * np.random.rand(Nx, Ny),
        dims=("x", "y"),
        coords={
            "x": dx
            * (
                np.arange(-Nx // 2, -Nx // 2 + Nx)
                + np.random.randint(-Nx // 2, Nx // 2)
            ),
            "y": dy
            * (
                np.arange(-Ny // 2, -Ny // 2 + Ny)
                + np.random.randint(-Ny // 2, Ny // 2)
            ),
        },
    )
    FTs = xrft.dft(s, dim=("x", "y"), true_phase=True, true_amplitude=True)
    npt.assert_almost_equal(
        (np.abs(s) ** 2).sum() * dx * dy,
        (np.abs(FTs) ** 2).sum() * FTs["freq_x"].spacing * FTs["freq_y"].spacing,
    )


def synthetic_field(N, dL, amp, s):
    """
    Generate a synthetic series of size N by N
    with a spectral slope of s.
    """

    k = np.fft.fftshift(np.fft.fftfreq(N, dL))
    l = np.fft.fftshift(np.fft.fftfreq(N, dL))
    kk, ll = np.meshgrid(k, l)
    K = np.sqrt(kk ** 2 + ll ** 2)

    ########
    # amplitude
    ########
    r_kl = np.ma.masked_invalid(
        np.sqrt(amp * 0.5 * (np.pi) ** (-1) * K ** (s - 1.0))
    ).filled(0.0)
    ########
    # phase
    ########
    phi = np.zeros((N, N))

    N_2 = int(N / 2)
    phi_upper_right = 2.0 * np.pi * np.random.random((N_2 - 1, N_2 - 1)) - np.pi
    phi[N_2 + 1 :, N_2 + 1 :] = phi_upper_right.copy()
    phi[1:N_2, 1:N_2] = -phi_upper_right[::-1, ::-1].copy()

    phi_upper_left = 2.0 * np.pi * np.random.random((N_2 - 1, N_2 - 1)) - np.pi
    phi[N_2 + 1 :, 1:N_2] = phi_upper_left.copy()
    phi[1:N_2, N_2 + 1 :] = -phi_upper_left[::-1, ::-1].copy()

    phi_upper_middle = 2.0 * np.pi * np.random.random(N_2) - np.pi
    phi[N_2:, N_2] = phi_upper_middle.copy()
    phi[1:N_2, N_2] = -phi_upper_middle[1:][::-1].copy()

    phi_right_middle = 2.0 * np.pi * np.random.random(N_2 - 1) - np.pi
    phi[N_2, N_2 + 1 :] = phi_right_middle.copy()
    phi[N_2, 1:N_2] = -phi_right_middle[::-1].copy()

    phi_edge_upperleft = 2.0 * np.pi * np.random.random(N_2) - np.pi
    phi[N_2:, 0] = phi_edge_upperleft.copy()
    phi[1:N_2, 0] = -phi_edge_upperleft[1:][::-1].copy()

    phi_bot_right = 2.0 * np.pi * np.random.random(N_2) - np.pi
    phi[0, N_2:] = phi_bot_right.copy()
    phi[0, 1:N_2] = -phi_bot_right[1:][::-1].copy()

    phi_corner_leftbot = 2.0 * np.pi * np.random.random() - np.pi

    for i in range(1, N_2):
        for j in range(1, N_2):
            assert phi[N_2 + j, N_2 + i] == -phi[N_2 - j, N_2 - i]

    for i in range(1, N_2):
        for j in range(1, N_2):
            assert phi[N_2 + j, N_2 - i] == -phi[N_2 - j, N_2 + i]

    for i in range(1, N_2):
        assert phi[N_2, N - i] == -phi[N_2, i]
        assert phi[N - i, N_2] == -phi[i, N_2]
        assert phi[N - i, 0] == -phi[i, 0]
        assert phi[0, i] == -phi[0, N - i]
    #########
    # complex fourier amplitudes
    #########
    F_theta = r_kl * np.exp(1j * phi)

    # check that symmetry of FT is satisfied
    theta = np.fft.ifft2(np.fft.ifftshift(F_theta))
    return np.real(theta)


def synthetic_field_xr(
    N, dL, amp, s, other_dim_sizes=None, dim_order=True, chunks=None
):

    theta = xr.DataArray(
        synthetic_field(N, dL, amp, s),
        dims=["y", "x"],
        coords={"y": range(N), "x": range(N)},
    )

    if other_dim_sizes:
        _da = xr.DataArray(
            np.ones(other_dim_sizes),
            dims=["d%d" % i for i in range(len(other_dim_sizes))],
        )
        if dim_order:
            theta = theta + _da
        else:
            theta = _da + theta

    if chunks:
        theta = theta.chunk(chunks)

    return theta


@pytest.mark.parametrize("truncate", [False, True])
def test_isotropize(truncate, N=512):
    """Test the isotropization of a power spectrum."""

    # generate synthetic 2D spectrum, isotropize and check values
    dL, amp, s = 1.0, 1e1, -3.0
    dims = ["x", "y"]
    fftdim = ["freq_x", "freq_y"]
    spacing_tol = 1e-3
    nfactor = 4

    def _test_iso(theta):
        ps = xrft.power_spectrum(theta, spacing_tol=spacing_tol, dim=dims)
        ps = np.sqrt(ps.freq_x ** 2 + ps.freq_y ** 2)
        ps_iso = xrft.isotropize(ps, fftdim, nfactor=nfactor, truncate=truncate)
        assert len(ps_iso.dims) == 1
        assert ps_iso.dims[0] == "freq_r"
        npt.assert_allclose(ps_iso, ps_iso.freq_r ** 2, atol=0.02)

    # np data
    theta = synthetic_field_xr(N, dL, amp, s)
    _test_iso(theta)
    # np with other dim
    theta = synthetic_field_xr(N, dL, amp, s, other_dim_sizes=[10], dim_order=True)
    _test_iso(theta)
    # da chunked, order 1
    theta = synthetic_field_xr(
        N,
        dL,
        amp,
        s,
        chunks={"y": None, "x": None, "d0": 2},
        other_dim_sizes=[10],
        dim_order=True,
    )
    _test_iso(theta)
    # da chunked, order 2
    theta = synthetic_field_xr(
        N,
        dL,
        amp,
        s,
        chunks={"y": None, "x": None, "d0": 2},
        other_dim_sizes=[10],
        dim_order=False,
    )
    _test_iso(theta)


@pytest.mark.parametrize("chunk", [False, True])
def test_isotropic_ps_slope(chunk, N=512, dL=1.0, amp=1e1, s=-3.0):
    """Test the spectral slope of isotropic power spectrum."""

    theta = synthetic_field_xr(
        N,
        dL,
        amp,
        s,
        other_dim_sizes=[10],
        dim_order=True,
    )

    if chunk:
        theta = theta.chunk({"d0": 2})

    iso_ps = xrft.isotropic_power_spectrum(
        theta, dim=["y", "x"], detrend="constant", density=True
    ).mean("d0")
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
    y_fit, a, b = xrft.fit_loglog(iso_ps.freq_r.values[4:], iso_ps.values[4:])
    npt.assert_allclose(a, s, atol=0.1)

    iso_ps_sequal = np.zeros((len(theta.d0), int(N / 4)))
    for i in range(len(theta.d0)):
        iso_ps_sequal[i] = xrft.isotropic_power_spectrum(
            theta.isel(d0=i), detrend="constant", density=True
        )
    npt.assert_almost_equal(iso_ps.values, iso_ps_sequal.mean(axis=0))

    iso_ps = xrft.isotropic_power_spectrum(
        theta, dim=["y", "x"], detrend="constant", scaling="density"
    ).mean("d0")
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
    y_fit, a, b = xrft.fit_loglog(iso_ps.freq_r.values[4:], iso_ps.values[4:])
    npt.assert_allclose(a, s, atol=0.1)
    npt.assert_almost_equal(iso_ps.values, iso_ps_sequal.mean(axis=0))


@pytest.mark.parametrize("chunk", [False, True])
def test_isotropic_ps(chunk):
    """Test data with extra coordinates"""
    da = xr.DataArray(
        np.random.rand(2, 5, 16, 32),
        dims=["time", "z", "y", "x"],
        coords={
            "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
            "zz": ("z", np.arange(5)),
            "z": np.arange(5),
            "y": np.arange(16),
            "x": np.arange(32),
        },
    )
    with pytest.raises(ValueError):
        xrft.isotropic_power_spectrum(da, dim=["z", "y", "x"])

    if chunk:
        da = da.chunk({"time": 1, "z": 1})

    iso_ps = xrft.isotropic_power_spectrum(da, dim=["y", "x"])
    npt.assert_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)


@pytest.mark.parametrize("chunk", [False, True])
def test_isotropic_cs(chunk):
    """Test isotropic cross spectrum"""
    N = 16
    da = xr.DataArray(
        np.random.rand(N, N), dims=["y", "x"], coords={"y": range(N), "x": range(N)}
    )
    da2 = xr.DataArray(
        np.random.rand(N, N), dims=["y", "x"], coords={"y": range(N), "x": range(N)}
    )

    iso_cs = xrft.isotropic_cross_spectrum(da, da2, window="hann")
    npt.assert_equal(np.ma.masked_invalid(iso_cs).mask.sum(), 0.0)

    da2 = xr.DataArray(
        np.random.rand(N, N),
        dims=["lat", "lon"],
        coords={"lat": range(N), "lon": range(N)},
    )
    with pytest.raises(ValueError):
        xrft.isotropic_cross_spectrum(da, da2)

    da = xr.DataArray(
        np.random.rand(2, 5, 16, 32),
        dims=["time", "z", "y", "x"],
        coords={
            "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
            "zz": ("z", np.arange(5)),
            "z": np.arange(5),
            "y": np.arange(16),
            "x": np.arange(32),
        },
    )
    da2 = xr.DataArray(
        np.random.rand(2, 5, 16, 32),
        dims=["time", "z", "y", "x"],
        coords={
            "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
            "zz": ("z", np.arange(5)),
            "z": np.arange(5),
            "y": np.arange(16),
            "x": np.arange(32),
        },
    )

    with pytest.raises(ValueError):
        xrft.isotropic_cross_spectrum(da, da2, dim=["z", "y", "x"])

    if chunk:
        da = da.chunk({"time": 1})
        da2 = da2.chunk({"time": 1})

    iso_cs = xrft.isotropic_cross_spectrum(da, da2, dim=["y", "x"], window="hann")
    npt.assert_equal(np.ma.masked_invalid(iso_cs).mask.sum(), 0.0)


def test_spacing_tol(test_data_1d):
    da = test_data_1d
    da2 = da.copy().load()

    # Create improperly spaced data
    Nx = 16
    Lx = 1.0
    x = np.linspace(0, Lx, Nx)
    x[-1] = x[-1] + 0.001
    da3 = xr.DataArray(np.random.rand(Nx), coords=[x], dims=["x"])

    # This shouldn't raise an error
    xrft.fft(da3, spacing_tol=1e-1)
    # But this should
    with pytest.raises(ValueError):
        xrft.fft(da3, spacing_tol=1e-4)


def test_spacing_tol_float_value(test_data_1d):
    da = test_data_1d
    with pytest.raises(TypeError):
        xrft.fft(da, spacing_tol="string")


@pytest.mark.parametrize("func", ("fft", "power_spectrum"))
@pytest.mark.parametrize("dim", ["time"])
def test_keep_coords(sample_data_3d, func, dim):
    """Test whether xrft keeps multi-dim coords from rasm sample data."""
    ds = sample_data_3d.temp
    ps = getattr(xrft, func)(ds, dim=dim)
    # check that all coords except dim from ds are kept in ps
    for c in ds.drop(dim).coords:
        assert c in ps.coords


@pytest.mark.parametrize("chunk", [False, True])
def test_true_phase_preservation(chunk):
    """Test if dft is (phase-) preserved when signal is at same place but coords range is changed"""
    x = np.arange(-15, 15)
    y = np.random.rand(len(x))

    N1 = np.random.randint(30) + 5
    N2 = np.random.randint(30) + 5
    l = np.arange(-N1, 0) + np.min(x)
    r = np.arange(1, N2 + 1) + np.max(x)
    s1 = xr.DataArray(
        np.concatenate([np.zeros(N1), y, np.zeros(N2)]),
        dims=("x",),
        coords={"x": np.concatenate([l, x, r])},
    )
    if chunk:
        s1 = s1.chunk()

    S1 = xrft.dft(s1, dim="x", true_phase=True)
    assert s1.chunks == S1.chunks

    N3 = N1
    while N3 == N1:
        N3 = np.minimum(np.random.randint(30), N1 + N2)
    N4 = N1 + N2 - N3

    l = np.arange(-N3, 0) + np.min(x)
    r = np.arange(1, N4 + 1) + np.max(x)
    s2 = xr.DataArray(
        np.concatenate([np.zeros(N3), y, np.zeros(N4)]),
        dims=("x",),
        coords={"x": np.concatenate([l, x, r])},
    )
    if chunk:
        s2 = s2.chunk()

    S2 = xrft.dft(s2, dim="x", true_phase=True)
    assert s2.chunks == S2.chunks

    xrt.assert_allclose(S1, S2)


def test_true_phase():
    """Test if true phase"""
    f0 = 2.0
    T = 4.0
    dx = 0.02
    x = np.arange(-8 * T, 5 * T + dx, dx)  # uncentered and odd number of points
    y = np.cos(2 * np.pi * f0 * x)
    y[np.abs(x) >= (T / 2.0)] = 0.0
    s = xr.DataArray(y, dims=("x",), coords={"x": x})
    lag = x[len(x) // 2]
    f = np.fft.fftfreq(len(x), dx)
    expected = np.fft.fft(np.fft.ifftshift(y)) * np.exp(-1j * 2.0 * np.pi * f * lag)
    expected = xr.DataArray(expected, dims="freq_x", coords={"freq_x": f})
    output = xrft.dft(
        s, dim="x", true_phase=True, true_amplitude=False, shift=False, prefix="freq_"
    )
    xrt.assert_allclose(expected, output)


def test_theoretical_matching(rtol=1e-8, atol=1e-3):
    """Test dft against theoretical results"""
    f0 = 2.0
    T = 4.0
    dx = 1e-4
    x = np.arange(-6 * T, 5 * T, dx)
    y = np.cos(2.0 * np.pi * f0 * x)
    y[np.abs(x) >= (T / 2.0)] = 0.0
    s = xr.DataArray(y, dims=("x",), coords={"x": x})
    S = xrft.dft(
        s, dim="x", true_phase=True, true_amplitude=True
    )  # Fast Fourier Transform of original signal
    f = S.freq_x  # Frequency axis
    TF_s = xr.DataArray(
        (T / 2 * (np.sinc(T * (f - f0)) + np.sinc(T * (f + f0)))).astype(np.complex),
        dims=("freq_x",),
        coords={"freq_x": f},
    )  # Theoretical expression of the Fourier transform
    xrt.assert_allclose(S, TF_s, rtol=rtol, atol=atol)


def test_real_dft_true_phase():
    """Test if real transform is half the total transform when signal is real and true_phase=True"""
    Nx = 40
    dx = np.random.rand()
    s = xr.DataArray(
        np.random.rand(Nx),
        dims="x",
        coords={
            "x": dx
            * (
                np.arange(-Nx // 2, -Nx // 2 + Nx)
                + np.random.randint(-Nx // 2, Nx // 2)
            )
        },
    )
    s1 = xrft.dft(s, dim="x", true_phase=True, shift=True)
    s2 = xrft.dft(s, real_dim="x", true_phase=True, shift=True)
    s1 = np.conj(s1[{"freq_x": slice(None, s1.sizes["freq_x"] // 2 + 1)}])
    s1 = s1.assign_coords(freq_x=-s1["freq_x"]).sortby("freq_x")
    xrt.assert_allclose(s1, s2)


def test_ifft_fft():
    """
    Testing ifft(fft(s.data)) == s.data
    """
    N = 20
    s = xr.DataArray(
        np.random.rand(N) + 1j * np.random.rand(N),
        dims="x",
        coords={"x": np.arange(0, N)},
    )
    FTs = xrft.fft(s)
    IFTs = xrft.ifft(FTs, shift=True)  # Shift=True is mandatory for the assestion below
    npt.assert_allclose(s.data, IFTs.data)


def test_idft_dft():
    """
    Testing idft(dft(s)) == s
    """
    N = 40
    dx = np.random.rand()
    s = xr.DataArray(
        np.random.rand(N) + 1j * np.random.rand(N),
        dims="x",
        coords={
            "x": dx
            * (np.arange(-N // 2, -N // 2 + N) + np.random.randint(-N // 2, N // 2))
        },
    )
    FTs = xrft.dft(s, true_phase=True, true_amplitude=True)
    mean_lag = float(
        s["x"][{"x": s.sizes["x"] // 2}]
    )  # lag ensure IFTs to be on the same coordinate range than s

    # lag is set manually
    IFTs = xrft.idft(
        FTs, shift=True, true_phase=True, true_amplitude=True, lag=mean_lag
    )
    xrt.assert_allclose(s, IFTs)

    # lag is set automatically
    IFTs = xrft.idft(FTs, shift=True, true_phase=True, true_amplitude=True)
    xrt.assert_allclose(s, IFTs)


def test_idft_centered_coordinates():
    """error should be raised if coordinates are not centered on zero in idft"""
    N = 20
    s = xr.DataArray(
        np.random.rand(N) + 1j * np.random.rand(N),
        dims="freq_x",
        coords={"freq_x": np.arange(-N // 2, N // 2) + 2},
    )
    with pytest.raises(ValueError):
        xrft.idft(s)


def test_constant_coordinates():
    """error should be raised if coordinates are constant"""
    N = 20
    s = xr.DataArray(
        np.random.rand(N) + 1j * np.random.rand(N),
        dims="freq_x",
        coords={"freq_x": np.zeros(N)},
    )
    with pytest.raises(ValueError):
        xrft.dft(s)

        with pytest.raises(ValueError):
            xrft.idft(s)


def test_reversed_coordinates():
    """Reversed coordinates should not impact dft with true_phase = True"""
    N = 20
    s = xr.DataArray(
        np.random.rand(N) + 1j * np.random.rand(N),
        dims="x",
        coords={"x": np.arange(N // 2, -N // 2, -1) + 2},
    )
    s2 = s.sortby("x")
    xrt.assert_allclose(
        xrft.dft(s, dim="x", true_phase=True), xrft.dft(s2, dim="x", true_phase=True)
    )


def test_nondim_coords():
    """Error should be raised if there are non-dimensional coordinates attached to the dimension(s) over which the FFT is being taken"""
    N = 16
    da = xr.DataArray(
        np.random.rand(2, N, N),
        dims=["time", "x", "y"],
        coords={
            "time": np.array(["2019-04-18", "2019-04-19"], dtype="datetime64"),
            "x": range(N),
            "y": range(N),
            "x_nondim": ("x", np.arange(N)),
        },
    )

    with pytest.raises(ValueError):
        xrft.power_spectrum(da)

    xrft.power_spectrum(da, dim=["time", "y"])
