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


def _apply_window(da, dims, window_type="hann"):
    """Creating windows in dimensions dims."""

    if window_type == True:
        window_type = "hann"
        warnings.warn(
            "Please provide the name of window adhering to scipy.signal.windows. The boolean option will be deprecated in future releases.",
            FutureWarning,
        )
    elif window_type not in [
        "hann",
        "hamming",
        "kaiser",
        "tukey",
        "parzen",
        "taylor",
        "boxcar",
        "barthann",
        "bartlett",
        "blackman",
        "blackmanharris",
        "bohman",
        "chebwin",
        "cosine",
        "dpss",
        "exponential",
        "flattop",
        "gaussian",
        "general_cosine",
        "general_gaussian",
        "general_hamming",
        "triang",
        "nuttall",
    ]:
        raise NotImplementedError(
            "Window type {window_type} not supported. Please adhere to scipy.signal.windows for naming convention."
        )

    if dims is None:
        dims = list(da.dims)
    else:
        if isinstance(dims, str):
            dims = [dims]

    scipy_win_func = getattr(sps.windows, window_type)

    if da.chunks:

        def dask_win_func(n, sym=False):
            return dsar.from_delayed(
                delayed(scipy_win_func, pure=True)(n, sym=sym), (n,), float
            )

        win_func = dask_win_func
    else:
        win_func = scipy_win_func

    windows = [
        xr.DataArray(
            win_func(len(da[d]), sym=False), dims=da[d].dims, coords=da[d].coords
        )
        for d in dims
    ]

    return reduce(operator.mul, windows[::-1]), da * reduce(operator.mul, windows[::-1])


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
    if coord[-1] > coord[0]:
        coord_data = coord.data
    else:
        coord_data = np.flip(coord.data, axis=-1)
    lag = coord_data[len(coord.data) // 2]
    if calendar:
        import cftime

        ref_units = "seconds since 1800-01-01 00:00:00"
        decoded_time = cftime.date2num(lag, ref_units, calendar)
        return decoded_time
    elif pd.api.types.is_datetime64_dtype(v0):
        return lag.astype("timedelta64[s]").astype("f8").data
    else:
        return lag.data


def dft(
    da, dim=None, true_phase=False, true_amplitude=False, **kwargs
):  # pragma: no cover
    """
    Deprecated function. See fft doc
    """
    msg = (
        "This function has been renamed and will disappear in the future."
        + " Please use `fft` instead"
    )
    warnings.warn(msg, FutureWarning)
    return fft(
        da, dim=dim, true_phase=true_phase, true_amplitude=true_amplitude, **kwargs
    )


def idft(
    daft, dim=None, true_phase=False, true_amplitude=False, **kwargs
):  # pragma: no cover
    """
    Deprecated function. See ifft doc
    """
    msg = (
        "This function has been renamed and will disappear in the future."
        + " Please use `ifft` instead"
    )
    warnings.warn(msg, FutureWarning)
    return ifft(
        daft, dim=dim, true_phase=true_phase, true_amplitude=true_amplitude, **kwargs
    )


def fft(
    da,
    spacing_tol=1e-3,
    dim=None,
    real_dim=None,
    shift=True,
    detrend=None,
    window=None,
    true_phase=False,
    true_amplitude=False,
    chunks_to_segments=False,
    prefix="freq_",
    **kwargs,
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
    real_dim : str, optional
        Real Fourier transform will be taken along this dimension.
    shift : bool, default
        Whether to shift the fft output. Default is `True`, unless `real_dim is not None`,
        in which case shift will be set to False always.
    detrend : {None, 'constant', 'linear'}
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit will be subtracted before
        the FT. For `linear`, only dims of length 1 and 2 are supported.
    window : str, optional
        Whether to apply a window to the data before the Fourier
        transform is taken. A window will be applied to all the dimensions in
        dim. Please follow `scipy.signal.windows`' naming convention.
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
        msg = "Flags true_phase and true_amplitude will be set to True in future versions of xrft.dft to preserve the theoretical phasing and amplitude of Fourier Transform. Consider using xrft.fft to ensure future compatibility with numpy.fft like behavior and to deactivate this warning."
        warnings.warn(msg, FutureWarning)

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if "real" in kwargs:
        real_dim = kwargs.get("real")
        msg = "`real` flag will be deprecated in future version of xrft.dft and replaced by `real_dim` flag."
        warnings.warn(msg, FutureWarning)

    if real_dim is not None:
        if real_dim not in da.dims:
            raise ValueError(
                "The dimension along which real FT is taken must be one of the existing dimensions."
            )
        else:
            dim = [d for d in dim if d != real_dim] + [
                real_dim
            ]  # real dim has to be moved or added at the end !

    if chunks_to_segments:
        da = _stack_chunks(da, dim)

    rawdims = da.dims  # take care of segmented dimesions, if any

    if real_dim is not None:
        da = da.transpose(
            *[d for d in da.dims if d not in [real_dim]] + [real_dim]
        )  # dimension for real transformed is moved at the end

    fftm = _fft_module(da)

    if real_dim is None:
        fft_fn = fftm.fftn
    else:
        shift = False
        fft_fn = fftm.rfftn

    # the axes along which to take ffts
    axis_num = [
        da.get_axis_num(d) for d in dim
    ]  # if there is a real dim , it has to be the last one

    N = [da.shape[n] for n in axis_num]

    # raise error if there are multiple coordinates attached to the dimension(s) over which the FFT is taken
    for d in dim:
        bad_coords = [
            cname for cname in da.coords if cname != d and d in da[cname].dims
        ]
        if bad_coords:
            raise ValueError(
                f"The input array contains coordinate variable(s) ({bad_coords}) whose dims include the transform dimension(s) `{d}`. "
                f"Please drop these coordinates (`.drop({bad_coords}`) before invoking xrft."
            )

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
        if delta == 0.0:
            raise ValueError(
                "Can't take Fourier transform because spacing in coordinate %s is zero"
                % d
            )
        delta_x.append(delta)
        lag_x.append(lag)

    if detrend is not None:
        if detrend == "linear":
            orig_dims = da.dims
            da = _detrend(da, dim, detrend_type=detrend).transpose(*orig_dims)
        else:
            da = _detrend(da, dim, detrend_type=detrend)

    if window is not None:
        _, da = _apply_window(da, dim, window_type=window)

    if true_phase:
        reversed_axis = [
            da.get_axis_num(d) for d in dim if da[d][-1] < da[d][0]
        ]  # handling decreasing coordinates
        f = fft_fn(
            fftm.ifftshift(np.flip(da, axis=reversed_axis), axes=axis_num),
            axes=axis_num,
        )
    else:
        f = fft_fn(da.data, axes=axis_num)

    if shift:
        f = fftm.fftshift(f, axes=axis_num)

    k = _freq(N, delta_x, real_dim, shift)

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
            daft[up_dim].attrs.update({"direct_lag": lag.obj})

    if true_amplitude:
        daft = daft * np.prod(delta_x)

    return daft.transpose(
        *[swap_dims.get(d, d) for d in rawdims]
    )  # Do nothing if da was not transposed


def ifft(
    daft,
    spacing_tol=1e-3,
    dim=None,
    real_dim=None,
    shift=True,
    true_phase=False,
    true_amplitude=False,
    chunks_to_segments=False,
    prefix="freq_",
    lag=None,
    **kwargs,
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
    real_dim : str, optional
        Real Fourier transform will be taken along this dimension.
    shift : bool, default
        Whether to shift the fft output. Default is `True`.
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
    lag : None, float or sequence of float and/or None, optional
        Output coordinates of transformed dimensions will be shifted by corresponding lag values and correct signal phasing will be preserved if true_phase is set to True.
        If lag is None (default), 'direct_lag' attributes of each dimension is used (or set to zero if not found).
        If defined, lag must have same length as dim.
        If lag is a sequence, a None element means that 'direct_lag' attribute will be used for the corresponding dimension
        Manually set lag to zero to get output coordinates centered on zero.


    Returns
    -------
    da : `xarray.DataArray`
        The output of the Inverse Fourier transformation, with appropriate dimensions.
    """

    if not true_phase and not true_amplitude:
        msg = "Flags true_phase and true_amplitude will be set to True in future versions of xrft.idft to preserve the theoretical phasing and amplitude of Inverse Fourier Transform. Consider using xrft.ifft to ensure future compatibility with numpy.ifft like behavior and to deactivate this warning."
        warnings.warn(msg, FutureWarning)

    if dim is None:
        dim = list(daft.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if "real" in kwargs:
        real_dim = kwargs.get("real")
        msg = "`real` flag will be deprecated in future version of xrft.idft and replaced by `real_dim` flag."
        warnings.warn(msg, FutureWarning)
    if real_dim is not None:
        if real_dim not in daft.dims:
            raise ValueError(
                "The dimension along which real IFT is taken must be one of the existing dimensions."
            )
        else:
            dim = [d for d in dim if d != real_dim] + [
                real_dim
            ]  # real dim has to be moved or added at the end !
    if lag is None:
        lag = [daft[d].attrs.get("direct_lag", 0.0) for d in dim]
        msg = "Default idft's behaviour (lag=None) changed! Default value of lag was zero (centered output coordinates) and is now set to transformed coordinate's attribute: 'direct_lag'."
        warnings.warn(msg, FutureWarning)
    else:
        if isinstance(lag, float) or isinstance(lag, int):
            lag = [lag]
        if len(dim) != len(lag):
            raise ValueError("dim and lag must have the same length.")
        if not true_phase:
            msg = "Setting lag with true_phase=False does not guarantee accurate idft."
            warnings.warn(msg, Warning)
        lag = [
            daft[d].attrs.get("direct_lag") if l is None else l
            for d, l in zip(dim, lag)
        ]  # enable lag of the form [3.2, None, 7]

    if true_phase:
        for d, l in zip(dim, lag):
            daft = daft * np.exp(1j * 2.0 * np.pi * daft[d] * l)

    if chunks_to_segments:
        daft = _stack_chunks(daft, dim)

    rawdims = daft.dims  # take care of segmented dimensions, if any

    if real_dim is not None:
        daft = daft.transpose(
            *[d for d in daft.dims if d not in [real_dim]] + [real_dim]
        )  # dimension for real transformed is moved at the end

    fftm = _fft_module(daft)

    if real_dim is None:
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
        l = _lag_coord(daft[d]) if d is not real_dim else daft[d][0].data
        if not np.allclose(
            diff, delta, rtol=spacing_tol
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
        if delta == 0.0:
            raise ValueError(
                "Can't take Inverse Fourier transform because spacing in coordinate %s is zero"
                % d
            )
        delta_x.append(delta)

    axis_shift = [
        daft.get_axis_num(d) for d in dim if d is not real_dim
    ]  # remove real dim of the list

    f = fftm.ifftshift(
        daft.data, axes=axis_shift
    )  # Force to be on fftshift grid before Fourier Transform
    f = fft_fn(f, axes=axis_num)

    if not true_phase:
        f = fftm.ifftshift(f, axes=axis_num)

    if shift:
        f = fftm.fftshift(f, axes=axis_num)

    k = _ifreq(N, delta_x, real_dim, shift)

    newcoords, swap_dims = _new_dims_and_coords(daft, dim, k, prefix)
    da = xr.DataArray(
        f,
        dims=daft.dims,
        coords=dict([c for c in daft.coords.items() if c[0] not in dim]),
    )
    da = da.swap_dims(swap_dims).assign_coords(newcoords)
    da = da.drop([d for d in dim if d in da.coords])

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
    da, dim=None, real_dim=None, scaling="density", window_correction=False, **kwargs
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
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    real_dim : str, optional
        Real Fourier transform will be taken along this dimension.
    scaling : str, optional
        If 'density', it will normalize the output to power spectral density
        If 'spectrum', it will normalize the output to power spectrum
    window_correction : boolean
        If True, it will correct for the energy reduction resulting from applying a non-uniform window.
        This is the default behaviour of many tools for computing power spectrum (e.g scipy.signal.welch and scipy.signal.periodogram).
        If scaling = 'spectrum', correct the amplitude of peaks in the spectrum. This ensures, for example, that the peak in the one-sided power spectrum of a 10 Hz sine wave with RMS**2 = 10 has a magnitude of 10.
        If scaling = 'density', correct for the energy (integral) of the spectrum. This ensures, for example, that the power spectral density integrates to the square of the RMS of the signal (ie that Parseval's theorem is satisfied). Note that in most cases, Parseval's theorem will only be approximately satisfied with this correction as it assumes that the signal being windowed is independent of the window. The correction becomes more accurate as the width of the window gets large in comparison with any noticeable period in the signal.
        If False, the spectrum gives a representation of the power in the windowed signal.
        Note that when True, Parseval's theorem may only be approximately satisfied.
    kwargs : dict : see xrft.dft for argument list
    """

    if "density" in kwargs:
        density = kwargs.pop("density")
        msg = (
            "density flag will be deprecated in future version of xrft.power_spectrum and replaced by scaling flag. "
            + 'density=True should be replaced by scaling="density" and '
            + "density=False will not be maintained.\nscaling flag is ignored !"
        )
        warnings.warn(msg, FutureWarning)
        scaling = "density" if density else "false_density"

    if "real" in kwargs:
        real_dim = kwargs.get("real")
        msg = "`real` flag will be deprecated in future version of xrft.power_spectrum and replaced by `real_dim` flag."
        warnings.warn(msg, FutureWarning)

    kwargs.update(
        {"true_amplitude": True, "true_phase": False}
    )  # true_phase do not matter in power_spectrum

    daft = fft(da, dim=dim, real_dim=real_dim, **kwargs)
    updated_dims = [
        d for d in daft.dims if (d not in da.dims and "segment" not in d)
    ]  # Transformed dimensions
    ps = np.abs(daft) ** 2

    if real_dim is not None:
        real = [d for d in updated_dims if real_dim == d[-len(real_dim) :]][
            0
        ]  # find transformed real dimension
        f = np.full(ps.sizes[real], 2.0)
        if len(da[real_dim]) % 2 == 0:
            f[0], f[-1] = 1.0, 1.0
        else:
            f[0] = 1.0
        ps = ps * xr.DataArray(f, dims=real, coords=ps[real].coords)

    if scaling == "density":
        if window_correction:
            if kwargs.get("window") == None:
                raise ValueError(
                    "window_correction can only be applied when windowing is turned on."
                )
            else:
                windows, _ = _apply_window(da, dim, window_type=kwargs.get("window"))
                ps = ps / (windows ** 2).mean()
        fs = np.prod([float(ps[d].spacing) for d in updated_dims])
        ps *= fs
    elif scaling == "spectrum":
        if window_correction:
            if kwargs.get("window") == None:
                raise ValueError(
                    "window_correction can only be applied when windowing is turned on."
                )
            else:
                windows, _ = _apply_window(da, dim, window_type=kwargs.get("window"))
                ps = ps / windows.mean() ** 2
        fs = np.prod([float(ps[d].spacing) for d in updated_dims])
        ps *= fs ** 2
    elif scaling == "false_density":  # Corresponds to density=False
        pass
    else:
        raise ValueError("Unknown {} scaling flag".format(scaling))
    return ps


def cross_spectrum(
    da1,
    da2,
    dim=None,
    real_dim=None,
    scaling="density",
    window_correction=False,
    true_phase=False,
    **kwargs,
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
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    real_dim : str, optional
        Real Fourier transform will be taken along this dimension.
    scaling : str, optional
        If 'density', it will normalize the output to power spectral density
        If 'spectrum', it will normalize the output to power spectrum
    window_correction : boolean
        If True, it will correct for the energy reduction resulting from applying a non-uniform window.
        This is the default behaviour of many tools for computing power spectrum (e.g scipy.signal.welch and scipy.signal.periodogram).
        If scaling = 'spectrum', correct the amplitude of peaks in the spectrum. This ensures, for example, that the peak in the one-sided power spectrum of a 10 Hz sine wave with RMS**2 = 10 has a magnitude of 10.
        If scaling = 'density', correct for the energy (integral) of the spectrum. This ensures, for example, that the power spectral density integrates to the square of the RMS of the signal (ie that Parseval's theorem is satisfied). Note that in most cases, Parseval's theorem will only be approximately satisfied with this correction as it assumes that the signal being windowed is independent of the window. The correction becomes more accurate as the width of the window gets large in comparison with any noticeable period in the signal.
        If False, the spectrum gives a representation of the power in the windowed signal.
        Note that when True, Parseval's theorem may only be approximately satisfied.
    kwargs : dict : see xrft.dft for argument list
    """

    if not true_phase:
        msg = (
            "true_phase flag will be set to True in future version of xrft.dft possibly impacting cross_spectrum output. "
            + "Set explicitely true_phase = False in cross_spectrum arguments list to ensure future compatibility "
            + "with numpy-like behavior where the coordinates are disregarded."
        )
        warnings.warn(msg, FutureWarning)

    if "real" in kwargs:
        real_dim = kwargs.get("real")
        msg = "`real` flag will be deprecated in future version of xrft.cross_spectrum and replaced by `real_dim` flag."
        warnings.warn(msg, FutureWarning)

    if "density" in kwargs:
        density = kwargs.pop("density")
        msg = (
            "density flag will be deprecated in future version of xrft.cross_spectrum and replaced by scaling flag. "
            + 'density=True should be replaced by scaling="density" and '
            + "density=False will not be maintained.\nscaling flag is ignored !"
        )
        warnings.warn(msg, FutureWarning)

        scaling = "density" if density else "false_density"

    kwargs.update({"true_amplitude": True})

    daft1 = fft(da1, dim=dim, real_dim=real_dim, true_phase=true_phase, **kwargs)
    daft2 = fft(da2, dim=dim, real_dim=real_dim, true_phase=true_phase, **kwargs)

    if daft1.dims != daft2.dims:
        raise ValueError("The two datasets have different dimensions")

    updated_dims = [
        d for d in daft1.dims if (d not in da1.dims and "segment" not in d)
    ]  # Transformed dimensions
    cs = daft1 * np.conj(daft2)

    if real_dim is not None:
        real = [d for d in updated_dims if real_dim == d[-len(real_dim) :]][
            0
        ]  # find transformed real dimension
        f = np.full(cs.sizes[real], 2.0)
        if len(da1[real_dim]) % 2 == 0:
            f[0], f[-1] = 1.0, 1.0
        else:
            f[0] = 1.0
        cs = cs * xr.DataArray(f, dims=real, coords=cs[real].coords)

    if scaling == "density":
        if window_correction:
            if kwargs.get("window") == None:
                raise ValueError(
                    "window_correction can only be applied when windowing is turned on."
                )
            else:
                windows, _ = _apply_window(da1, dim, window_type=kwargs.get("window"))
                cs = cs / (windows ** 2).mean()
        fs = np.prod([float(cs[d].spacing) for d in updated_dims])
        cs *= fs
    elif scaling == "spectrum":
        if window_correction:
            if kwargs.get("window") == None:
                raise ValueError(
                    "window_correction can only be applied when windowing is turned on."
                )
            else:
                windows, _ = _apply_window(da1, dim, window_type=kwargs.get("window"))
                cs = cs / windows.mean() ** 2
        fs = np.prod([float(cs[d].spacing) for d in updated_dims])
        cs *= fs ** 2
    elif scaling == "false_density":  # Corresponds to density=False
        pass
    else:
        raise ValueError("Unknown {} scaling flag".format(scaling))
    return cs


def cross_phase(da1, da2, dim=None, true_phase=False, **kwargs):
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
    kwargs : dict : see xrft.dft for argument list
    """
    if not true_phase:
        msg = (
            "true_phase flag will be set to True in future version of xrft.dft possibly impacting cross_phase output. "
            + "Set explicitely true_phase = False in cross_spectrum arguments list to ensure future compatibility "
            + "with numpy-like behavior where the coordinates are disregarded."
        )
        warnings.warn(msg, FutureWarning)

    cp = xr.ufuncs.angle(
        cross_spectrum(da1, da2, dim=dim, true_phase=true_phase, **kwargs)
    )

    if da1.name and da2.name:
        cp.name = "{}_{}_phase".format(da1.name, da2.name)

    return cp


def _binned_agg(
    array: np.ndarray,
    indices: np.ndarray,
    num_bins: int,
    *,
    func,
    fill_value,
    dtype,
) -> np.ndarray:
    """NumPy helper function for aggregating over bins."""

    try:
        import numpy_groupies
    except ImportError:
        raise ImportError(
            "This function requires the `numpy_groupies` package to be installed. Please install it with pip or conda."
        )

    mask = np.logical_not(np.isnan(indices))
    int_indices = indices[mask].astype(int)
    shape = array.shape[: -indices.ndim] + (num_bins,)
    result = numpy_groupies.aggregate(
        int_indices,
        array[..., mask],
        func=func,
        size=num_bins,
        fill_value=fill_value,
        dtype=dtype,
        axis=-1,
    )
    return result


def _groupby_bins_agg(
    array: xr.DataArray,
    group: xr.DataArray,
    bins,
    func="sum",
    fill_value=0,
    dtype=None,
    **cut_kwargs,
) -> xr.DataArray:
    """Faster equivalent of Xarray's groupby_bins(...).sum()."""
    # https://github.com/pydata/xarray/issues/4473
    binned = pd.cut(np.ravel(group), bins, **cut_kwargs)
    new_dim_name = group.name + "_bins"
    indices = group.copy(data=binned.codes.reshape(group.shape))

    result = xr.apply_ufunc(
        _binned_agg,
        array,
        indices,
        input_core_dims=[indices.dims, indices.dims],
        output_core_dims=[[new_dim_name]],
        output_dtypes=[array.dtype],
        dask_gufunc_kwargs=dict(
            allow_rechunk=True,
            output_sizes={new_dim_name: binned.categories.size},
        ),
        kwargs={
            "num_bins": binned.categories.size,
            "func": func,
            "fill_value": fill_value,
            "dtype": dtype,
        },
        dask="parallelized",
    )
    result.coords[new_dim_name] = binned.categories
    return result


def isotropize(ps, fftdim, nfactor=4, truncate=False):
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
    truncate : bool, optional
        If True, the spectrum will be truncated for wavenumbers larger than
        the Nyquist wavenumber.
    """

    # compute radial wavenumber bins
    k = ps[fftdim[1]]
    l = ps[fftdim[0]]

    N = [k.size, l.size]
    nbins = int(min(N) / nfactor)
    freq_r = np.sqrt(k ** 2 + l ** 2).rename("freq_r")
    kr = _groupby_bins_agg(freq_r, freq_r, bins=nbins, func="mean")

    if truncate:
        if k.max() > l.max():
            kmax = l.max()
        else:
            kmax = k.max()
        kr = kr.where(kr <= kmax)
    else:
        msg = (
            "The flag `truncate` will be set to True by default in future version "
            + "in order to truncate the isotropic wavenumber larger than the "
            + "Nyquist wavenumber."
        )
        warnings.warn(msg, FutureWarning)

    iso_ps = (
        _groupby_bins_agg(ps, freq_r, bins=nbins, func="mean")
        .rename({"freq_r_bins": "freq_r"})
        .drop_vars("freq_r")
    )
    iso_ps.coords["freq_r"] = kr.data
    if truncate:
        return (iso_ps * iso_ps.freq_r).dropna("freq_r")
    else:
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
    scaling="density",
    window=None,
    window_correction=False,
    nfactor=4,
    truncate=False,
    **kwargs,
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
    window : str, optional
        Whether to apply a window to the data before the Fourier
        transform is taken. Please adhere to scipy.signal.windows for naming convention.
    nfactor : int, optional
        Ratio of number of bins to take the azimuthal averaging with the
        data size. Default is 4.
    truncate : bool, optional
        If True, the spectrum will be truncated for wavenumbers larger than
        the Nyquist wavenumber.

    Returns
    -------
    iso_ps : `xarray.DataArray`
        Isotropic power spectrum
    """
    if "density" in kwargs:
        density = kwargs.pop("density")
        scaling = "density" if density else "false_density"

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
        scaling=scaling,
        window_correction=window_correction,
        window=window,
        **kwargs,
    )

    fftdim = ["freq_" + d for d in dim]

    return isotropize(ps, fftdim, nfactor=nfactor, truncate=truncate)


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
    scaling="density",
    window=None,
    window_correction=False,
    nfactor=4,
    truncate=False,
    **kwargs,
):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrum by taking the
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
    window : str (optional)
        Whether to apply a window to the data before the Fourier
        transform is taken. Please adhere to scipy.signal.windows for naming convention.
    nfactor : int (optional)
        Ratio of number of bins to take the azimuthal averaging with the
        data size. Default is 4.
    truncate : bool, optional
        If True, the spectrum will be truncated for wavenumbers larger than
        the Nyquist wavenumber.

    Returns
    -------
    iso_cs : `xarray.DataArray`
        Isotropic cross spectrum
    """
    if "density" in kwargs:
        density = kwargs.pop("density")
        scaling = "density" if density else "false_density"

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
        scaling=scaling,
        window_correction=window_correction,
        window=window,
        **kwargs,
    )

    fftdim = ["freq_" + d for d in dim]

    return isotropize(cs, fftdim, nfactor=nfactor, truncate=truncate)


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
