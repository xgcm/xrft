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


__all__ = ["detrendn", "detrend_wrap",
           "dft","power_spectrum", "cross_spectrum", "cross_phase",
           "isotropize",
           "isotropic_power_spectrum", "isotropic_cross_spectrum",
           "isotropic_powerspectrum", "isotropic_crossspectrum",
           "fit_loglog"]

def _fft_module(da):
    if da.chunks:
        return dsar.fft
    else:
        return np.fft

def _apply_window(da, dims, window_type='hanning'):
    """Creating windows in dimensions dims."""

    if window_type not in ['hanning']:
        raise NotImplementedError("Only hanning window is supported for now.")

    numpy_win_func = getattr(np, window_type)

    if da.chunks:
        def dask_win_func(n):
            return dsar.from_delayed(
                delayed(numpy_win_func, pure=True)(n),
                (n,), float)
        win_func = dask_win_func
    else:
        win_func = numpy_win_func

    windows = [xr.DataArray(win_func(len(da[d])),
               dims=da[d].dims, coords=da[d].coords) for d in dims]

    return da * reduce(operator.mul, windows[::-1])

def detrendn(da, axes=None):
    """
    Detrend by subtracting out the least-square plane or least-square cubic fit
    depending on the number of axis.

    Parameters
    ----------
    da : `dask.array`
        The data to be detrended

    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """
    N = [da.shape[n] for n in axes]
    M = []
    for n in range(da.ndim):
        if n not in axes:
            M.append(da.shape[n])

    if len(N) == 2:
        G = np.ones((N[0]*N[1],3))
        for i in range(N[0]):
            G[N[1]*i:N[1]*i+N[1], 1] = i+1
            G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)
        if type(da) == xr.DataArray:
            d_obs = np.reshape(da.copy().values, (N[0]*N[1],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
    elif len(N) == 3:
        if type(da) == xr.DataArray:
            if da.ndim > 3:
                raise NotImplementedError("Cubic detrend is not implemented "
                                         "for 4-dimensional `xarray.DataArray`."
                                         " We suggest converting it to "
                                         "`dask.array`.")
            else:
                d_obs = np.reshape(da.copy().values, (N[0]*N[1]*N[2],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1]*N[2],1))

        G = np.ones((N[0]*N[1]*N[2],4))
        G[:,3] = np.tile(np.arange(1,N[2]+1), N[0]*N[1])
        ys = np.zeros(N[1]*N[2])
        for i in range(N[1]):
            ys[N[2]*i:N[2]*i+N[2]] = i+1
        G[:,2] = np.tile(ys, N[0])
        for i in range(N[0]):
            G[len(ys)*i:len(ys)*i+len(ys),1] = i+1
    else:
        raise NotImplementedError("Detrending over more than 4 axes is "
                                 "not implemented.")

    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    lin_trend = np.reshape(d_est, da.shape)

    return da - lin_trend

def detrend_wrap(detrend_func):
    """
    Wrapper function for `xrft.detrendn`.
    """
    def func(a, axes=None):
        if len(axes) > 3:
            raise ValueError("Detrending is only supported up to "
                            "3 dimensions.")
        if axes is None:
            axes = tuple(range(a.ndim))
        else:
            if len(set(axes)) < len(axes):
                raise ValueError("Duplicate axes are not allowed.")

        for each_axis in axes:
            if len(a.chunks[each_axis]) != 1:
                raise ValueError('The axis along the detrending is upon '
                                'cannot be chunked.')

        if len(axes) == 1:
            return dsar.map_blocks(sps.detrend, a, axis=axes[0],
                                   chunks=a.chunks, dtype=a.dtype
                                  )
        else:
            for each_axis in range(a.ndim):
                if each_axis not in axes:
                    if len(a.chunks[each_axis]) != a.shape[each_axis]:
                        raise ValueError("The axes other than ones to detrend "
                                        "over should have a chunk length of 1.")
            return dsar.map_blocks(detrend_func, a, axes,
                                   chunks=a.chunks, dtype=a.dtype
                                  )

    return func

def _apply_detrend(da, axis_num):
    """Wrapper function for applying detrending"""
    if da.chunks:
        func = detrend_wrap(detrendn)
        da = xr.DataArray(func(da.data, axes=axis_num),
                         dims=da.dims, coords=da.coords)
    else:
        if da.ndim == 1:
            da = xr.DataArray(sps.detrend(da),
                             dims=da.dims, coords=da.coords)
        else:
            da = detrendn(da, axes=axis_num)

    return da

def _stack_chunks(da, dim, suffix='_segment'):
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
            coord_rs = da[d].data.reshape((int(n/chunklen),int(chunklen)))
            newdims.append(d + suffix)
            newdims.append(d)
            newshape.append(int(n/chunklen))
            newshape.append(int(chunklen))
            newcoords[d+suffix] = range(int(n/chunklen))
            newcoords[d] = coord_rs[0]
        else:
            newdims.append(d)
            newshape.append(len(da[d]))
            newcoords[d] = da[d].data

    da = xr.DataArray(data.reshape(newshape), dims=newdims, coords=newcoords,
                     attrs=attr)

    return da

def _transpose(da, real, trans=False):
    if real is not None:
        transdim = list(da.dims)
        if real not in transdim:
            raise ValueError("The dimension along real FT is taken must "
                            "be one of the existing dimensions.")
        elif real != transdim[-1]:
            transdim.remove(real)
            transdim += [real]
            da = da.transpose(*transdim)
            trans = True
    return da, trans

def _freq(N, delta_x, real, shift):
    # calculate frequencies from coordinates
    # coordinates are always loaded eagerly, so we use numpy
    if real is None:
        fftfreq = [np.fft.fftfreq]*len(N)
    else:
        # Discard negative frequencies from transform along last axis to be
        # consistent with np.fft.rfftn
        fftfreq = [np.fft.fftfreq]*(len(N)-1)
        fftfreq.append(np.fft.rfftfreq)

    k = [fftfreq(Nx, dx) for (fftfreq, Nx, dx) in zip(fftfreq, N, delta_x)]

    if shift:
        k = [np.fft.fftshift(l) for l in k]

    return k

def _new_dims_and_coords(da, axis_num, dim, wavenm, prefix):
    # set up new dimensions and coordinates for dataarray
    newdims = list(da.dims)
    for anum, d in zip(axis_num, dim):
        newdims[anum] = prefix + d if d[:len(prefix)]!=prefix else d[len(prefix):]

    k_names = [prefix + d for d in dim]
    k_coords = {key: val for (key,val) in zip(k_names, wavenm)}

    newcoords = {}
    # keep former coords
    if len(da.coords) > 1:
        for c in da.drop(dim).coords:
            newcoords[c] = da[c]
    for d in newdims:
        if d in k_coords:
            newcoords[d] = k_coords[d]
        elif d in da.coords:
            newcoords[d] = da[d].data

    dk = [l[1] - l[0] for l in wavenm]
    for this_dk, d in zip(dk, dim):
        newcoords[prefix + d + '_spacing'] = this_dk

    return newdims, newcoords

def _diff_coord(coord):
    """Returns the difference as a xarray.DataArray."""

    v0 = coord.values[0]
    calendar = getattr(v0, 'calendar', None)
    if calendar:
        import cftime
        ref_units = 'seconds since 1800-01-01 00:00:00'
        decoded_time = cftime.date2num(coord, ref_units, calendar)
        coord = xr.DataArray(decoded_time, dims=coord.dims, coords=coord.coords)
        return np.diff(coord)
    elif pd.api.types.is_datetime64_dtype(v0):
        return np.diff(coord).astype('timedelta64[s]').astype('f8')
    else:
        return np.diff(coord)

def _calc_normalization_factor(da, axis_num, chunks_to_segments):
    """Return the signal length, N, to be used in the normalisation of spectra"""
    
    if chunks_to_segments:
        # Use chunk sizes for normalisation
        return [da.chunks[n][0] for n in axis_num]
    else:
        return [da.shape[n] for n in axis_num]
    
def dft(da, spacing_tol=1e-3, dim=None, real=None, shift=True, detrend=None,
        window=False, chunks_to_segments=False, prefix='freq_'):
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
        dimensions will be transformed.
    real : str, optional
        Real Fourier transform will be taken along this dimension.
    shift : bool, default
        Whether to shift the fft output. Default is `True`, unless `real=True`,
        in which case shift will be set to False always.
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

    Returns
    -------
    daft : `xarray.DataArray`
        The output of the Fourier transformation, with appropriate dimensions.
    """
    # check for proper spacing tolerance input
    if not isinstance(spacing_tol, float):
        raise TypeError("Please provide a float argument")

    # check for xr.da input
    if not isinstance(da, xr.DataArray):
        raise TypeError("Please provide xr.DataArray, found", type(da))

    rawdims = da.dims
    da, trans = _transpose(da, real)
    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim,]
    if real is not None and real not in dim:
        dim += [real]

    if not da.chunks:
        if np.isnan(da.values).any():
            raise ValueError("Data cannot take Nans")
    else:
        if detrend=='linear' and len(dim)>1:
            for d in da.dims:
                a_n = da.get_axis_num(d)
                if d not in dim and da.chunks[a_n][0]>1:
                    raise ValueError("Linear detrending utilizes the `dask.map_blocks` "
                                    "API so the dimensions not being detrended "
                                    "must have the chunk length of 1.")

    fft = _fft_module(da)

    if real is None:
        fft_fn = fft.fftn
    else:
        shift = False
        fft_fn = fft.rfftn

    if chunks_to_segments:
        da = _stack_chunks(da, dim)

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]

    # verify even spacing of input coordinates
    delta_x = []
    for d in dim:
        diff = _diff_coord(da[d])
        delta = np.abs(diff[0])
        if not np.allclose(diff, diff[0], rtol=spacing_tol):
            raise ValueError("Can't take Fourier transform because "
                             "coodinate %s is not evenly spaced" % d)
        delta_x.append(delta)

    if detrend == 'constant':
        da = da - da.mean(dim=dim)
    elif detrend == 'linear':
        for d in da.dims:
            if d not in dim:
                da = da.chunk({d:1})
        da = _apply_detrend(da, axis_num)

    if window:
        da = _apply_window(da, dim)

    f = fft_fn(da.data, axes=axis_num)

    if shift:
        f = fft.fftshift(f, axes=axis_num)

    k = _freq(N, delta_x, real, shift)

    newdims, newcoords = _new_dims_and_coords(da, axis_num, dim, k, prefix)

    daft = xr.DataArray(f, dims=newdims, coords=newcoords)
    if trans:
        enddims = [d for d in rawdims if d not in dim]
        enddims += [prefix + d for d in rawdims if d in dim]
        return daft.transpose(*enddims)
    else:
        return daft


def power_spectrum(da, spacing_tol=1e-3, dim=None, real=None, shift=True,
                   detrend=None, window=False, chunks_to_segments=False,
                   density=True, prefix='freq_'):
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

    daft = dft(da, spacing_tol,
              dim=dim, real=real, shift=shift, detrend=detrend, window=window,
              chunks_to_segments=chunks_to_segments, prefix=prefix)

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim,]
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
            ps /= daft['freq_' + i + '_spacing']

    return ps


def cross_spectrum(da1, da2, spacing_tol=1e-3, dim=None, shift=True,
                  detrend=None, window=False, chunks_to_segments=False,
                  density=True, prefix='freq_'):
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

    daft1 = dft(da1, spacing_tol,
               dim=dim, shift=shift, detrend=detrend, window=window,
               chunks_to_segments=chunks_to_segments,
               prefix=prefix)
    daft2 = dft(da2, spacing_tol,
               dim=dim, shift=shift, detrend=detrend, window=window,
               chunks_to_segments=chunks_to_segments,
               prefix=prefix)

    if dim is None:
        dim = da1.dims
        dim2 = da2.dims
        if dim != dim2:
            raise ValueError('The two datasets have different dimensions')
    else:
        if isinstance(dim, str):
            dim = [dim,]
            dim2 = [dim,]

    # the axes along which to take ffts
    axis_num = [da1.get_axis_num(d) for d in dim]

    N = _calc_normalization_factor(da1, axis_num, chunks_to_segments)

    return _cross_spectrum(daft1, daft2, dim, N, density)


def _cross_spectrum(daft1, daft2, dim, N, density):
    cs = (daft1 * np.conj(daft2)).real

    if density:
        cs /= (np.asarray(N).prod())**2
        for i in dim:
            cs /= daft1['freq_' + i + '_spacing']

    return cs


def cross_phase(da1, da2, spacing_tol=1e-3, dim=None, detrend=None,
                window=False, chunks_to_segments=False):
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
            raise ValueError('The two datasets have different dimensions')
    elif not isinstance(dim, list):
        dim = [dim]
    if len(dim)>1:
        raise ValueError('Cross phase calculation should only be done along '
                        'a single dimension.')

    daft1 = dft(da1, spacing_tol,
                dim=dim, real=dim[0], shift=False, detrend=detrend,
                window=window, chunks_to_segments=chunks_to_segments)
    daft2 = dft(da2, spacing_tol,
                dim=dim, real=dim[0], shift=False, detrend=detrend,
                window=window, chunks_to_segments=chunks_to_segments)

    if daft1.chunks and daft2.chunks:
        _cross_phase = lambda a, b: dsar.angle(a * dsar.conj(b))
    else:
        _cross_phase = lambda a, b: np.angle(a * np.conj(b))
    cp = xr.apply_ufunc(_cross_phase, daft1, daft2, dask='allowed')

    if da1.name and da2.name:
        cp.name = "{}_{}_phase".format(da1.name, da2.name)

    return cp


def _radial_wvnum(k, l, N, nfactor):
    """ Creates a radial wavenumber based on two horizontal wavenumbers
    along with the appropriate index map
    """

    # compute target wavenumbers
    k = k.values
    l = l.values
    K = np.sqrt(k[np.newaxis,:]**2 + l[:,np.newaxis]**2)
    nbins = int(N/nfactor)
    if k.max() > l.max():
        ki = np.linspace(0., l.max(), nbins)
    else:
        ki = np.linspace(0., k.max(), nbins)

    # compute bin index
    kidx = np.digitize(np.ravel(K), ki)
    # compute number of points for each wavenumber
    area = np.bincount(kidx)
    # compute the average radial wavenumber for each bin
    kr = (np.bincount(kidx, weights=K.ravel())
          / np.ma.masked_where(area==0, area))

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
    ps = ps.assign_coords(freq_r=np.sqrt(k**2+l**2))
    iso_ps = (ps.groupby_bins('freq_r', bins=ki, labels=kr).mean()
              .rename({'freq_r_bins': 'freq_r'})
             )
    return iso_ps * iso_ps.freq_r

def isotropic_powerspectrum(*args, **kwargs): # pragma: no cover
    """
    Deprecated function. See isotropic_power_spectrum doc
    """
    import warnings
    msg = "This function has been renamed and will disappear in the future."\
          +" Please use isotropic_power_spectrum instead"
    warnings.warn(msg, Warning)
    return isotropic_power_spectrum(*args, **kwargs)

def isotropic_power_spectrum(da, spacing_tol=1e-3, dim=None, shift=True,
                           detrend=None, density=True, window=False, nfactor=4):
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
        raise ValueError('The Fourier transform should be two dimensional')

    ps = power_spectrum(da, spacing_tol, dim=dim, shift=shift,
                       detrend=detrend, density=density,
                       window=window)

    fftdim = ['freq_' + d for d in dim]

    return isotropize(ps, fftdim, nfactor=nfactor)

def isotropic_crossspectrum(*args, **kwargs): # pragma: no cover
    """
    Deprecated function. See isotropic_cross_spectrum doc
    """
    import warnings
    msg = "This function has been renamed and will disappear in the future."\
          +" Please use isotropic_cross_spectrum instead"
    warnings.warn(msg, Warning)
    return isotropic_cross_spectrum(*args, **kwargs)

def isotropic_cross_spectrum(da1, da2, spacing_tol=1e-3,
                           dim=None, shift=True, detrend=None,
                           density=True, window=False, nfactor=4):
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
            raise ValueError('The two datasets have different dimensions')
    if len(dim) != 2:
        raise ValueError('The Fourier transform should be two dimensional')

    cs = cross_spectrum(da1, da2, spacing_tol, dim=dim, shift=shift,
                       detrend=detrend, density=density,
                       window=window)

    fftdim = ['freq_' + d for d in dim]

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
    y_fit = 2**(np.log2(x)*p[0] + p[1])

    return y_fit, p[0], p[1]
