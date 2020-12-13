import warnings
import operator
import sys
import functools as ft
from functools import reduce

import numpy as np
import xarray as xr
import pandas as pd

import dask.array as dsar
from dask import delayed

import scipy.signal as sps
import scipy.linalg as spl

from .detrend import detrend as _detrend


__all__ = [
    "fft",
    "ifft",
    "dft",
    "idft",
    "power_spectrum",
    "cross_spectrum",
    "cross_phase",
    "isotropize",
    "isotropic_power_spectrum",
    "isotropic_cross_spectrum",
    "isotropic_powerspectrum",
    "isotropic_crossspectrum",
    "fit_loglog",
]


def _fft_module(da):
    if da.chunks:
        return dsar.fft
    else:
        return np.fft


def _apply_window(da, dims, window_type="hanning"):
    """Creating windows in dimensions dims."""

    if window_type not in ["hanning"]:
        raise NotImplementedError("Only hanning window is supported for now.")

    numpy_win_func = getattr(np, window_type)

    if da.chunks:

        def dask_win_func(n):
            return dsar.from_delayed(delayed(numpy_win_func, pure=True)(n), (n,), float)

        win_func = dask_win_func
    else:
        win_func = numpy_win_func

    windows = [
        xr.DataArray(win_func(len(da[d])), dims=da[d].dims, coords=da[d].coords)
        for d in dims
    ]

    return da * reduce(operator.mul, windows[::-1])


def _stack_chunks(da, dim, suffix="_segment"):
    """Reshape a DataArray so there is only one chunk along dimension `dim`"""
    data = da.data
    attr = da.attrs
    newdims = []
    newcoords = {}
    newshape = []
    for d in da.dims:
        if d in dim:
            axis_num = da.get_axis_num(d)
            if np.diff(da.chunks[axis_num]).sum() != 0:
                raise ValueError("Chunk lengths need to be the same.")
            n = len(da[d])
            chunklen = da.chunks[axis_num][0]
            coord_rs = da[d].data.reshape((int(n / chunklen), int(chunklen)))
            newdims.append(d + suffix)
            newdims.append(d)
            newshape.append(int(n / chunklen))
            newshape.append(int(chunklen))
            newcoords[d + suffix] = range(int(n / chunklen))
            newcoords[d] = coord_rs[0]
        else:
            newdims.append(d)
            newshape.append(len(da[d]))
            newcoords[d] = da[d].data

    da = xr.DataArray(
        data.reshape(newshape), dims=newdims, coords=newcoords, attrs=attr
    )

    return da


def _freq(N, delta_x, real, shift):
    # calculate frequencies from coordinates
    # coordinates are always loaded eagerly, so we use numpy
    if real is None:
        fftfreq = [np.fft.fftfreq] * len(N)
    else:
        # Discard negative frequencies from transform along last axis to be
        # consistent with np.fft.rfftn
        fftfreq = [np.fft.fftfreq] * (len(N) - 1)
        fftfreq.append(np.fft.rfftfreq)

    k = [fftfreq(Nx, dx) for (fftfreq, Nx, dx) in zip(fftfreq, N, delta_x)]

    if shift:
        k = [np.fft.fftshift(l) for l in k]

    return k


def _ifreq(N, delta_x, real, shift):
    # calculate frequencies from coordinates
    # coordinates are always loaded eagerly, so we use numpy
    if real is None:
        fftfreq = [np.fft.fftfreq] * len(N)
    else:
        irfftfreq = lambda Nx, dx: np.fft.fftfreq(
            2 * (Nx - 1), dx
        )  # Not in standard numpy !
        fftfreq = [np.fft.fftfreq] * (len(N) - 1)
        fftfreq.append(irfftfreq)

    k = [fftfreq(Nx, dx) for (fftfreq, Nx, dx) in zip(fftfreq, N, delta_x)]

    if shift:
        k = [np.fft.fftshift(l) for l in k]

    return k


def _new_dims_and_coords(da, dim, wavenm, prefix):
    # set up new dimensions and coordinates for dataarray
    swap_dims = dict()
    new_coords = dict()
    wavenm = dict(zip(dim, wavenm))

    for d in dim:
        k = wavenm[d]
        new_name = prefix + d if d[: len(prefix)] != prefix else d[len(prefix) :]
        new_dim = xr.DataArray(k, dims=new_name, coords={new_name: k}, name=new_name)
        new_dim.attrs.update({"spacing": k[1] - k[0]})
        new_coords[new_name] = new_dim
        swap_dims[d] = new_name

    return new_coords, swap_dims


def _diff_coord(coord):
    """Returns the difference as a xarray.DataArray."""

    v0 = coord.values[0]
    calendar = getattr(v0, "calendar", None)
    if calendar:
        import cftime

        ref_units = "seconds since 1800-01-01 00:00:00"
        decoded_time = cftime.date2num(coord, ref_units, calendar)
        coord = xr.DataArray(decoded_time, dims=coord.dims, coords=coord.coords)
        return np.diff(coord)
    elif pd.api.types.is_datetime64_dtype(v0):
        return np.diff(coord).astype("timedelta64[s]").astype("f8")
    else:
        return np.diff(coord)


def _lag_coord(coord):
    """Returns the coordinate lag"""

    v0 = coord.values[0]
    calendar = getattr(v0, "calendar", None)
    lag = coord[(len(coord.data)) // 2]
    if calendar:
        import cftime

        ref_units = "seconds since 1800-01-01 00:00:00"
        decoded_time = cftime.date2num(lag, ref_units, calendar)
        return decoded_time
    elif pd.api.types.is_datetime64_dtype(v0):
        return lag.astype("timedelta64[s]").astype("f8").data
    else:
        return lag.data


def _calc_normalization_factor(da, axis_num, chunks_to_segments):
    """Return the signal length, N, to be used in the normalisation of spectra"""

    if chunks_to_segments:
        # Use chunk sizes for normalisation
        return [da.chunks[n][0] for n in axis_num]
    else:
        return [da.shape[n] for n in axis_num]


def fft(da, **kwargs):
    """
    See xrft.dft for argument list
    """
    if kwargs.pop("true_phase", False):
        warnings.warn("true_phase argument is ignored in xrft.fft")
    if kwargs.pop("true_amplitude", False):
        warnings.warn("true_amplitude argument is ignored in xrft.fft")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dft(da, true_phase=False, true_amplitude=False, **kwargs)


def ifft(daft, **kwargs):
    """
    See xrft.idft for argument list
    """
    if kwargs.pop("true_phase", False):
        warnings.warn("true_phase argument is ignored in xrft.ifft")
    if kwargs.pop("true_amplitude", False):
        warnings.warn("true_amplitude argument is ignored in xrft.ifft")
    if kwargs.pop("lag", False):
        warnings.warn("lag argument is ignored in xrft.ifft")
    msg = "xrft.ifft does not guarantee correct coordinate phasing for its output. We recommend xrft.dft and xrft.idft as forward and backward Fourier Transforms with true_phase flags set to True for accurate coordinate handling."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return idft(daft, true_phase=False, true_amplitude=False, **kwargs)


def dft(
    da,
    spacing_tol=1e-3,
    dim=None,
    real=None,
    shift=True,
    detrend=None,
    window=False,
    true_phase=False,
    true_amplitude=False,
    chunks_to_segments=False,
    prefix="freq_",
):
    """
    Perform discrete Fourier transform of xarray data-array `da` along the
    specified dimensions.

    .. math::
        daft = \mathbb{F}(da - \overline{da})

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    real : str, optional
        Real Fourier transform will be taken along this dimension.
    shift : bool, default
        Whether to shift the fft output. Default is `True`, unless `real=True`,
        in which case shift will be set to False always.
    detrend : {None, 'constant', 'linear'}
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit will be subtracted before
        the FT. For `linear`, only dims of length 1 and 2 are supported.
    window : bool, optional
        Whether to apply a Hann window to the data before the Fourier
        transform is taken. A window will be applied to all the dimensions in
        dim.
    true_phase : bool, optional
        If set to False, standard fft algorithm is applied on signal without consideration of coordinates.
        If set to True, coordinates location are correctly taken into account to evaluate Fourier Tranforrm phase and
        fftshift is applied on input signal prior to fft  (fft algorithm intrinsically considers that input signal is on fftshifted grid).
    true_amplitude : bool, optional
        If set to True, output is multiplied by the spacing of the transformed variables to match theoretical FT amplitude.
        If set to False, amplitude regularisation by spacing is not applied (as in numpy.fft)
    chunks_to_segments : bool, optional
        Whether the data is chunked along the axis to take FFT.
    prefix : str
        The prefix for the new transformed dimensions.

    Returns
    -------
    daft : `xarray.DataArray`
        The output of the Fourier transformation, with appropriate dimensions.
    """

    if not true_phase and not true_amplitude:
        msg = "Flags true_phase and true_amplitude will be set to True in future versions of xrft to preserve the theoretical phasing and amplitude of FT. Consider using xrft.fft to ensure future compatibility with numpy.fft like behavior and to deactivate this warning."
        warnings.warn(msg, FutureWarning)

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if real is not None:
        if real not in da.dims:
            raise ValueError(
                "The dimension along which real FT is taken must be one of the existing dimensions."
            )
        else:
            dim = [d for d in dim if d != real] + [
                real
            ]  # real dim has to be moved or added at the end !

    if chunks_to_segments:
        da = _stack_chunks(da, dim)

    rawdims = da.dims  # take care of segmented dimesions, if any

    if real is not None:
        da = da.transpose(
            *[d for d in da.dims if d not in [real]] + [real]
        )  # dimension for real transformed is moved at the end

    fft = _fft_module(da)

    if real is None:
        fft_fn = fft.fftn
    else:
        shift = False
        fft_fn = fft.rfftn

    # the axes along which to take ffts
    axis_num = [
        da.get_axis_num(d) for d in dim
    ]  # if there is a real dim , it has to be the last one

    N = [da.shape[n] for n in axis_num]

    # verify even spacing of input coordinates
    delta_x = []
    lag_x = []
    for d in dim:
        diff = _diff_coord(da[d])
        delta = np.abs(diff[0])
        lag = _lag_coord(da[d])
        if not np.allclose(diff, diff[0], rtol=spacing_tol):
            raise ValueError(
                "Can't take Fourier transform because "
                "coodinate %s is not evenly spaced" % d
            )
        delta_x.append(delta)
        lag_x.append(lag)

    if detrend:
        da = _detrend(da, dim, detrend_type=detrend)

    if window:
        da = _apply_window(da, dim)

    if true_phase:
        f = fft_fn(fft.ifftshift(da.data, axes=axis_num), axes=axis_num)
    else:
        f = fft_fn(da.data, axes=axis_num)

    if shift:
        f = fft.fftshift(f, axes=axis_num)

    k = _freq(N, delta_x, real, shift)

    newcoords, swap_dims = _new_dims_and_coords(da, dim, k, prefix)
    daft = xr.DataArray(
        f, dims=da.dims, coords=dict([c for c in da.coords.items() if c[0] not in dim])
    )
    daft = daft.swap_dims(swap_dims).assign_coords(newcoords)
    daft = daft.drop([d for d in dim if d in daft.coords])

    updated_dims = [
        daft.dims[i] for i in da.get_axis_num(dim)
    ]  # List of transformed dimensions

    if true_phase:
        for up_dim, lag in zip(updated_dims, lag_x):
            daft = daft * xr.DataArray(
                np.exp(-1j * 2.0 * np.pi * newcoords[up_dim] * lag),
                dims=up_dim,
                coords={up_dim: newcoords[up_dim]},
            )  # taking advantage of xarray broadcasting and ordered coordinates

    if true_amplitude:
        daft = daft * np.prod(delta_x)

    return daft.transpose(
        *[swap_dims.get(d, d) for d in rawdims]
    )  # Do nothing if da was not transposed


def idft(
    daft,
    spacing_tol=1e-3,
    dim=None,
    real=None,
    shift=True,
    detrend=None,
    true_phase=False,
    true_amplitude=False,
    window=False,
    chunks_to_segments=False,
    prefix="freq_",
    lag=None,
):
    """
    Perform inverse discrete Fourier transform of xarray data-array `daft` along the
    specified dimensions.

    .. math::
        da = \mathbb{F}(daft - \overline{daft})

    Parameters
    ----------
    daft : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    real : str, optional
        Real Fourier transform will be taken along this dimension.
    shift : bool, default
        Whether to shift the fft output. Default is `True`.
    detrend : str, optional
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit will be subtracted before
        the FT.
    window : bool, optional
        Whether to apply a Hann window to the data before the Fourier
        transform is taken. A window will be applied to all the dimensions in
        dim.
    chunks_to_segments : bool, optional
        Whether the data is chunked along the axis to take FFT.
    prefix : str
        The prefix for the new transformed dimensions.
    true_phase : bool, optional
        If set to False, standard ifft algorithm is applied on signal without consideration of coordinates order.
        If set to True, coordinates are correctly taken into account to evaluate Inverse Fourier Tranforrm phase and
        fftshift is applied on input signal prior to ifft (ifft algorithm intrinsically considers that input signal is on fftshifted grid).
    true_amplitude : bool, optional
        If set to True, output is divided by the spacing of the transformed variables to match theoretical IFT amplitude.
        If set to False, amplitude regularisation by spacing is not applied (as in numpy.ifft)
    lag : float or sequence of float, optional
        If lag is None or zero, output coordinates are centered on zero.
        If defined, lag must have same length as dim.
        Output coordinates corresponding to transformed dimensions will be shifted by corresponding lag values.
        Correct signal phasing will be preserved if true_phase is set to True.

    Returns
    -------
    da : `xarray.DataArray`
        The output of the Inverse Fourier transformation, with appropriate dimensions.
    """

    if not true_phase and not true_amplitude:
        msg = "xrft.idft default behaviour will be modified in future versions of xrft. Use xrft.ifft to ensure future compatibility and deactivate this warning"
        warnings.warn(msg, FutureWarning)

    if dim is None:
        dim = list(daft.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if real is not None:
        if real not in daft.dims:
            raise ValueError(
                "The dimension along which real IFT is taken must be one of the existing dimensions."
            )
        else:
            dim = [d for d in dim if d != real] + [
                real
            ]  # real dim has to be moved or added at the end !

    if lag is not None:
        if isinstance(lag, float) or isinstance(lag, int):
            lag = [lag]
        if len(dim) != len(lag):
            raise ValueError("dim and lag must have the same length.")
        if not true_phase:
            msg = "Setting lag with true_phase=False does not guarantee accurate idft."
            warnings.warn(msg, Warning)

        for d, l in zip(dim, lag):
            daft = daft * np.exp(1j * 2.0 * np.pi * daft[d] * l)

    if chunks_to_segments:
        daft = _stack_chunks(daft, dim)

    rawdims = daft.dims  # take care of segmented dimesions, if any

    if real is not None:
        daft = daft.transpose(
            *[d for d in daft.dims if d not in [real]] + [real]
        )  # dimension for real transformed is moved at the end

    fftm = _fft_module(daft)

    if real is None:
        fft_fn = fftm.ifftn
    else:
        fft_fn = fftm.irfftn

    # the axes along which to take ffts
    axis_num = [daft.get_axis_num(d) for d in dim]

    N = [daft.shape[n] for n in axis_num]

    # verify even spacing of input coordinates (It handle fftshifted grids)
    delta_x = []
    for d in dim:
        diff = _diff_coord(daft[d])
        delta = np.abs(diff[0])
        l = _lag_coord(daft[d]) if d is not real else daft[d][0].data
        if not np.allclose(
            diff, diff[0], rtol=spacing_tol
        ):  # means that input is not on regular increasing grid
            reordered_coord = daft[d].copy()
            reordered_coord = reordered_coord.sortby(d)
            diff = _diff_coord(reordered_coord)
            l = _lag_coord(reordered_coord)
            if np.allclose(
                diff, diff[0], rtol=spacing_tol
            ):  # means that input is on fftshifted grid
                daft = daft.sortby(d)  # reordering the input
            else:
                raise ValueError(
                    "Can't take Fourier transform because "
                    "coodinate %s is not evenly spaced" % d
                )
        if np.abs(l) > spacing_tol:
            raise ValueError(
                "Inverse Fourier Transform can not be computed because coordinate %s is not centered on zero frequency"
                % d
            )
        delta_x.append(delta)

    if detrend:
        daft = _apply_detrend(daft, dim, axis_num, detrend)

    if window:
        daft = _apply_window(daft, dim)

    axis_shift = [
        daft.get_axis_num(d) for d in dim if d is not real
    ]  # remove real dim of the list

    f = fftm.ifftshift(
        daft.data, axes=axis_shift
    )  # Force to be on fftshift grid before Fourier Transform
    f = fft_fn(f, axes=axis_num)

    if not true_phase:
        f = fftm.ifftshift(f, axes=axis_num)

    if shift:
        f = fftm.fftshift(f, axes=axis_num)

    k = _ifreq(N, delta_x, real, shift)

    newcoords, swap_dims = _new_dims_and_coords(daft, dim, k, prefix)
    da = xr.DataArray(
        f,
        dims=daft.dims,
        coords=dict([c for c in daft.coords.items() if c[0] not in dim]),
    )
    da = da.swap_dims(swap_dims).assign_coords(newcoords)
    da = da.drop([d for d in dim if d in da.coords])

    if lag is not None:
        with xr.set_options(
            keep_attrs=True
        ):  # This line ensures keeping spacing attribute in output coordinates
            for d, l in zip(dim, lag):
                tfd = swap_dims[d]
                da = da.assign_coords({tfd: da[tfd] + l})

    if true_amplitude:
        da = da / np.prod([float(da[up_dim].spacing) for up_dim in swap_dims.values()])

    return da.transpose(
        *[swap_dims.get(d, d) for d in rawdims]
    )  # Do nothing if daft was not transposed


def power_spectrum(
    da,
    spacing_tol=1e-3,
    dim=None,
    real=None,
    shift=True,
    detrend=None,
    window=False,
    chunks_to_segments=False,
    density=True,
    prefix="freq_",
):
    """
    Calculates the power spectrum of da.

    .. math::
        da' = da - \overline{da}
    .. math::
        ps = \mathbb{F}(da') {\mathbb{F}(da')}^*

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    real : str, optional
        Real Fourier transform will be taken along this dimension.
    shift : bool, optional
        Whether to shift the fft output.
    detrend : str, optional
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit will be subtracted before
        the FT.
    density : bool, optional
        If true, it will normalize the spectrum to spectral density
    window : bool, optional
        Whether to apply a Hann window to the data before the Fourier
        transform is taken
    chunks_to_segments : bool, optional
        Whether the data is chunked along the axis to take FFT.

    Returns
    -------
    ps : `xarray.DataArray`
        Two-dimensional power spectrum
    """

    daft = fft(
        da,
        spacing_tol=spacing_tol,
        dim=dim,
        real=real,
        shift=shift,
        detrend=detrend,
        window=window,
        chunks_to_segments=chunks_to_segments,
        prefix=prefix,
    )

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [
                dim,
            ]
    if real is not None and real not in dim:
        dim += [real]

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = _calc_normalization_factor(da, axis_num, chunks_to_segments)

    return _power_spectrum(daft, dim, N, density)


def _power_spectrum(daft, dim, N, density):

    ps = (daft * np.conj(daft)).real

    if density:
        ps /= (np.asarray(N).prod()) ** 2
        for i in dim:
            ps /= daft["freq_" + i].spacing
            # ps /= daft["freq_" + i + "_spacing"]

    return ps


def cross_spectrum(
    da1,
    da2,
    spacing_tol=1e-3,
    dim=None,
    shift=True,
    detrend=None,
    window=False,
    chunks_to_segments=False,
    density=True,
    prefix="freq_",
):
    """
    Calculates the cross spectra of da1 and da2.

    .. math::
        da1' = da1 - \overline{da1};\ \ da2' = da2 - \overline{da2}
    .. math::
        cs = \mathbb{F}(da1') {\mathbb{F}(da2')}^*

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool, optional
        Whether to shift the fft output.
    detrend : str, optional
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit along one axis will be
        subtracted before the FT. It will give an error if the length of
        `dim` is longer than one.
    density : bool, optional
        If true, it will normalize the spectrum to spectral density
    window : bool, optional
        Whether to apply a Hann window to the data before the Fourier
        transform is taken

    Returns
    -------
    cs : `xarray.DataArray`
        Two-dimensional cross spectrum
    """

    daft1 = fft(
        da1,
        spacing_tol=spacing_tol,
        dim=dim,
        shift=shift,
        detrend=detrend,
        window=window,
        chunks_to_segments=chunks_to_segments,
        prefix=prefix,
    )
    daft2 = fft(
        da2,
        spacing_tol=spacing_tol,
        dim=dim,
        shift=shift,
        detrend=detrend,
        window=window,
        chunks_to_segments=chunks_to_segments,
        prefix=prefix,
    )

    if dim is None:
        dim = da1.dims
        dim2 = da2.dims
        if dim != dim2:
            raise ValueError("The two datasets have different dimensions")
    else:
        if isinstance(dim, str):
            dim = [
                dim,
            ]
            dim2 = [
                dim,
            ]

    # the axes along which to take ffts
    axis_num = [da1.get_axis_num(d) for d in dim]

    N = _calc_normalization_factor(da1, axis_num, chunks_to_segments)

    return _cross_spectrum(daft1, daft2, dim, N, density)


def _cross_spectrum(daft1, daft2, dim, N, density):
    cs = (daft1 * np.conj(daft2)).real

    if density:
        cs /= (np.asarray(N).prod()) ** 2
        for i in dim:
            # cs /= daft1["freq_" + i + "_spacing"]
            cs /= daft1["freq_" + i].spacing

    return cs


def cross_phase(
    da1,
    da2,
    spacing_tol=1e-3,
    dim=None,
    detrend=None,
    window=False,
    chunks_to_segments=False,
):
    """
    Calculates the cross-phase between da1 and da2.

    Returned values are in [-pi, pi].

    .. math::
        da1' = da1 - \overline{da1};\ \ da2' = da2 - \overline{da2}
    .. math::
        cp = \text{Arg} [\mathbb{F}(da1')^*, \mathbb{F}(da2')]

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : list, optional
        The dimension along which to take the real Fourier transformation.
        If `None`, all dimensions will be transformed.
    shift : bool, optional
        Whether to shift the fft output.
    detrend : str, optional
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit along one axis will be
        subtracted before the FT. It will give an error if the length of
        `dim` is longer than one.
    window : bool, optional
        Whether to apply a Hann window to the data before the Fourier
        transform is taken

    Returns
    -------
    cp : `xarray.DataArray`
        Cross-phase as a function of frequency.
    """

    if dim is None:
        dim = da1.dims
        dim2 = da2.dims
        if dim != dim2:
            raise ValueError("The two datasets have different dimensions")
    elif not isinstance(dim, list):
        dim = [dim]
    if len(dim) > 1:
        raise ValueError(
            "Cross phase calculation should only be done along " "a single dimension."
        )

    daft1 = fft(
        da1,
        spacing_tol=spacing_tol,
        dim=dim,
        real=dim[0],
        shift=False,
        detrend=detrend,
        window=window,
        chunks_to_segments=chunks_to_segments,
    )
    daft2 = fft(
        da2,
        spacing_tol=spacing_tol,
        dim=dim,
        real=dim[0],
        shift=False,
        detrend=detrend,
        window=window,
        chunks_to_segments=chunks_to_segments,
    )

    if daft1.chunks and daft2.chunks:
        _cross_phase = lambda a, b: dsar.angle(a * dsar.conj(b))
    else:
        _cross_phase = lambda a, b: np.angle(a * np.conj(b))
    cp = xr.apply_ufunc(_cross_phase, daft1, daft2, dask="allowed")

    if da1.name and da2.name:
        cp.name = "{}_{}_phase".format(da1.name, da2.name)

    return cp


def _radial_wvnum(k, l, N, nfactor):
    """Creates a radial wavenumber based on two horizontal wavenumbers
    along with the appropriate index map
    """

    # compute target wavenumbers
    k = k.values
    l = l.values
    K = np.sqrt(k[np.newaxis, :] ** 2 + l[:, np.newaxis] ** 2)
    nbins = int(N / nfactor)
    if k.max() > l.max():
        ki = np.linspace(0.0, l.max(), nbins)
    else:
        ki = np.linspace(0.0, k.max(), nbins)

    # compute bin index
    kidx = np.digitize(np.ravel(K), ki)
    # compute number of points for each wavenumber
    area = np.bincount(kidx)
    # compute the average radial wavenumber for each bin
    kr = np.bincount(kidx, weights=K.ravel()) / np.ma.masked_where(area == 0, area)

    return ki, kr[1:-1]


def isotropize(ps, fftdim, nfactor=4):
    """
    Isotropize a 2D power spectrum or cross spectrum
    by taking an azimuthal average.

    .. math::
        \text{iso}_{ps} = k_r N^{-1} \sum_{N} |\mathbb{F}(da')|^2

    where :math:`N` is the number of azimuthal bins.

    Parameters
    ----------
    ps : `xarray.DataArray`
        The power spectrum or cross spectrum to be isotropized.
    fftdim : list
        The fft dimensions overwhich the isotropization must be performed.
    nfactor : int, optional
        Ratio of number of bins to take the azimuthal averaging with the
        data size. Default is 4.
    """

    # compute radial wavenumber bins
    k = ps[fftdim[1]]
    l = ps[fftdim[0]]
    N = [k.size, l.size]
    ki, kr = _radial_wvnum(k, l, min(N), nfactor)

    # average azimuthally
    ps = ps.assign_coords(freq_r=np.sqrt(k ** 2 + l ** 2))
    iso_ps = (
        ps.groupby_bins("freq_r", bins=ki, labels=kr)
        .mean()
        .rename({"freq_r_bins": "freq_r"})
    )
    return iso_ps * iso_ps.freq_r


def isotropic_powerspectrum(*args, **kwargs):  # pragma: no cover
    """
    Deprecated function. See isotropic_power_spectrum doc
    """
    msg = (
        "This function has been renamed and will disappear in the future."
        + " Please use isotropic_power_spectrum instead"
    )
    warnings.warn(msg, Warning)
    return isotropic_power_spectrum(*args, **kwargs)


def isotropic_power_spectrum(
    da,
    spacing_tol=1e-3,
    dim=None,
    shift=True,
    detrend=None,
    density=True,
    window=False,
    nfactor=4,
):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrum by taking the
    azimuthal average.

    .. math::
        \text{iso}_{ps} = k_r N^{-1} \sum_{N} |\mathbb{F}(da')|^2

    where :math:`N` is the number of azimuthal bins.

    Note: the method is not lazy does trigger computations.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : list, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool, optional
        Whether to shift the fft output.
    detrend : str, optional
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit will be subtracted before
        the FT.
    density : list, optional
        If true, it will normalize the spectrum to spectral density
    window : bool, optional
        Whether to apply a Hann window to the data before the Fourier
        transform is taken
    nfactor : int, optional
        Ratio of number of bins to take the azimuthal averaging with the
        data size. Default is 4.

    Returns
    -------
    iso_ps : `xarray.DataArray`
        Isotropic power spectrum
    """

    if dim is None:
        dim = da.dims
    if len(dim) != 2:
        raise ValueError("The Fourier transform should be two dimensional")

    ps = power_spectrum(
        da,
        spacing_tol=spacing_tol,
        dim=dim,
        shift=shift,
        detrend=detrend,
        density=density,
        window=window,
    )

    fftdim = ["freq_" + d for d in dim]

    return isotropize(ps, fftdim, nfactor=nfactor)


def isotropic_crossspectrum(*args, **kwargs):  # pragma: no cover
    """
    Deprecated function. See isotropic_cross_spectrum doc
    """
    msg = (
        "This function has been renamed and will disappear in the future."
        + " Please use isotropic_cross_spectrum instead"
    )
    warnings.warn(msg, Warning)
    return isotropic_cross_spectrum(*args, **kwargs)


def isotropic_cross_spectrum(
    da1,
    da2,
    spacing_tol=1e-3,
    dim=None,
    shift=True,
    detrend=None,
    density=True,
    window=False,
    nfactor=4,
):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrumby taking the
    azimuthal average.

    .. math::
        \text{iso}_{cs} = k_r N^{-1} \sum_{N} (\mathbb{F}(da1') {\mathbb{F}(da2')}^*)

    where :math:`N` is the number of azimuthal bins.

    Note: the method is not lazy does trigger computations.

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
    spacing_tol: float (default)
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : list (optional)
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool (optional)
        Whether to shift the fft output.
    detrend : str (optional)
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit will be subtracted before
        the FT.
    density : list (optional)
        If true, it will normalize the spectrum to spectral density
    window : bool (optional)
        Whether to apply a Hann window to the data before the Fourier
        transform is taken
    nfactor : int (optional)
        Ratio of number of bins to take the azimuthal averaging with the
        data size. Default is 4.

    Returns
    -------
    iso_cs : `xarray.DataArray`
        Isotropic cross spectrum
    """

    if dim is None:
        dim = da1.dims
        dim2 = da2.dims
        if dim != dim2:
            raise ValueError("The two datasets have different dimensions")
    if len(dim) != 2:
        raise ValueError("The Fourier transform should be two dimensional")

    cs = cross_spectrum(
        da1,
        da2,
        spacing_tol=spacing_tol,
        dim=dim,
        shift=shift,
        detrend=detrend,
        density=density,
        window=window,
    )

    fftdim = ["freq_" + d for d in dim]

    return isotropize(cs, fftdim, nfactor=nfactor)


def fit_loglog(x, y):
    """
    Fit a line to isotropic spectra in log-log space

    Parameters
    ----------
    x : `numpy.array`
        Coordinate of the data
    y : `numpy.array`
        data

    Returns
    -------
    y_fit : `numpy.array`
        The linear fit
    a : float64
        Slope of the fit
    b : float64
        Intercept of the fit
    """
    # fig log vs log
    p = np.polyfit(np.log2(x), np.log2(y), 1)
    y_fit = 2 ** (np.log2(x) * p[0] + p[1])

    return y_fit, p[0], p[1]
