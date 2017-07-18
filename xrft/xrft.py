import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import dask.array as dsar
import scipy.signal as sps
import scipy.linalg as spl
import warnings

__all__ = ["detrend2","_detrend_wrap",
            "dft","power_spectrum","cross_spectrum",
            "isotropic_powerspectrum","isotropic_crossspectrum",
            "fit_loglog"]


def _hanning(da, N, dim):
    """Apply Hanning window"""

    if len(N)==1:
        window = xr.DataArray(np.hanning(N[0]), dims=dim)
    elif len(N)==2:
        window = xr.DataArray(np.hanning(N[-1])
                            * np.hanning(N[-2])[:,np.newaxis],
                            dims=dim)
    elif len(N)==3:
        window = xr.DataArray(np.hanning(N[-3])[:,np.newaxis,np.newaxis]
                            * np.hanning(N[-2])[np.newaxis,:,np.newaxis]
                            * np.hanning(N[-1]),
                            dims=dim)
    elif len(N)==4:
        window = xr.DataArray(np.hanning(N[0])[:,np.newaxis,
                                            np.newaxis,np.newaxis]
                            * np.hanning(N[1])[np.newaxis,:,
                                            np.newaxis,np.newaxis]
                            * np.hanning(N[2])[np.newaxis,np.newaxis,
                                            :,np.newaxis]
                            * np.hanning(N[3]),
                            dims=dim)
    else:
        raise ValueError("Windowing can only be applied up to four dimesnions "
                        "(time, z, x, y).")

    # dim = da.dims
    # coord = da.coords

    # if len(dim) == 3:
    #     N1 = da.shape[0]
    #     if da[0].shape != window.shape:
    #         raise ValueError('The spatial dimensions do not match up')
    #     for i in range(N1):
    #         da[i] *= window
    # elif len(dim) == 4:
    #     N1, N2 = da.shape[:2]
    #     if da[0,0].shape != window.shape:
    #         raise ValueError('The spatial dimensions do not match up')
    #     for j in range(N1):
    #         for i in range(N2):
    #             da[j,i] *= window
    # elif len(dim) == 2:
    #     da *= window
    # else:
    #     raise ValueError('Data has too many dimensions')
    da *= window

    return da

def detrend2(da, axes=None):
    """
    Detrend a 2D field by subtracting out the least-square plane fit.

    Parameters
    ----------
    da : `dask.array`
        The data to be detrended

    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """
#     if da.ndim > 2:
#         raise ValueError('The data should only have two dimensions')
#     print(da.shape)
    N = [da.shape[n] for n in axes]
    M = []
    for n in range(da.ndim):
        if n not in axes:
            M.append(da.shape[n])

    G = np.ones((N[0]*N[1],3))
    for i in range(N[0]):
        G[N[1]*i:N[1]*i+N[1], 1] = i+1
        G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)
    d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    if len(M) == 1:
        lin_trend = np.reshape(d_est, da.shape)
    else:
        lin_trend = np.reshape(d_est, da.shape)

    return da - lin_trend

def _detrend_wrap(detrend_func):
    """
    Wrapper function for `xrft.detrend2`.
    """
    def func(a, axes=None):
        if a.ndim > 4:
            raise ValueError("This function only expects up to 4 dimensions")
        if axes is None:
            axes = tuple(range(a.ndim))
        else:
            if len(set(axes)) < len(axes):
                raise ValueError("Duplicate axes are not allowed.")

        for each_axis in axes:
            if len(a.chunks[each_axis]) != 1:
                raise ValueError('The axis along the detrending is upon'
                                'cannot be chunked.')

        if len(axes) == 1:
            return dsar.map_blocks(sps.detrend, a, axis=axes[0],
                                chunks=a.chunks, dtype=a.dtype)
        elif len(axes) == 2:
            for each_axis in range(a.ndim):
                if each_axis not in axes:
                    if len(a.chunks[each_axis]) != a.shape[each_axis]:
                        raise ValueError('The axes other than ones to detrend'
                                        'over should have a chunk length of 1')
            return dsar.map_blocks(detrend_func, a, axes,
                                chunks=a.chunks, dtype=a.dtype)
        else:
            raise ValueError("Too many dimensions to detrend over.")

    return func

def dft(da, dim=None, shift=True, detrend=None, window=False):
    """
    Perform discrete Fourier transform of xarray data-array `da` along the
    specified dimensions.

    .. math::

     daft = \mathbb{F}(da - \overline{da})

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
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
    window : bool (optional)
        Whether to apply a Hann window to the data before the Fourier
        transform is taken

    Returns
    -------
    daft : `xarray.DataArray`
        The output of the Fourier transformation, with appropriate dimensions.
    """
    if np.isnan(da.values).any():
        raise ValueError("Data cannot take Nans")

    if dim is None:
        dim = da.dims

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]

    # verify even spacing of input coordinates
    delta_x = []
    for d in dim:
        coord = da[d]
        diff = np.diff(coord)
        if pd.core.common.is_timedelta64_dtype(diff):
            # convert to seconds so we get hertz
            diff = diff.astype('timedelta64[s]').astype('f8')
        delta = diff[0]
        if not np.allclose(diff, diff[0]):
            raise ValueError("Can't take Fourier transform because"
                             "coodinate %s is not evenly spaced" % d)
        delta_x.append(delta)
    # calculate frequencies from coordinates
    k = [ np.fft.fftfreq(Nx, dx) for (Nx, dx) in zip(N, delta_x) ]

    if detrend == 'constant':
        da = da - da.mean(dim=dim)
    elif detrend == 'linear':
        if hasattr(da.data, 'dask'):
            func = _detrend_wrap(detrend2)
            da = xr.DataArray(func(da.data, axes=axis_num).compute(),
                            dims=da.dims, coords=da.coords)
        else:
            if da.ndim == 1:
                da = xr.DataArray(sps.detrend(da),
                                dims=da.dims, coords=da.coords)
            else:
                raise ValueError("Data should be dask array.")

    if window:
        da = _hanning(da, N, dim)

    # the hard work
    #f = np.fft.fftn(da.values, axes=axis_num)
    # need special path for dask
    # is this the best way to check for dask?
    data = da.data
    if hasattr(data, 'dask'):
        f = dsar.fft.fftn(data, axes=axis_num)
    else:
        f = np.fft.fftn(data, axes=axis_num)

    if shift:
        f = np.fft.fftshift(f, axes=axis_num)
        k = [np.fft.fftshift(l) for l in k]

    # set up new coordinates for dataarray
    prefix = 'freq_'
    k_names = [prefix + d for d in dim]
    k_coords = {key: val for (key,val) in zip(k_names, k)}

    newdims = list(da.dims)
    for anum, d in zip(axis_num, dim):
        newdims[anum] = prefix + d

    newcoords = {}
    for d in newdims:
        if d in da.coords:
            newcoords[d] = da.coords[d].values
        else:
            newcoords[d] = k_coords[d]

    dk = [l[1] - l[0] for l in k]
    for this_dk, d in zip(dk, dim):
        newcoords[prefix + d + '_spacing'] = this_dk

    return xr.DataArray(f, dims=newdims, coords=newcoords)

def power_spectrum(da, dim=None, shift=True, detrend=None, density=True,
                window=False):
    """
    Calculates the power spectrum of da.

    .. math::

     da' = da - \overline{da}
     ps = \mathbb{F}(da') * {\mathbb{F}(da')}^*

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
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

    Returns
    -------
    ps : `xarray.DataArray`
        Two-dimensional power spectrum
    """

    if dim is None:
        dim = da.dims

    # the axes along which to take ffts
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]

    daft = dft(da,
            dim=dim, shift=shift, detrend=detrend,
            window=window)

    coord = list(daft.coords)

    ps = (daft * np.conj(daft)).real

    if density:
        ps /= (np.asarray(N).prod())**2
        for i in dim:
            ps /= daft['freq_' + i + '_spacing']

    return ps

def cross_spectrum(da1, da2, dim=None,
                   shift=True, detrend=None, density=True, window=False):
    """
    Calculates the cross spectra of da1 and da2.

    .. math::

     da1' = da1 - \overline{da1}; da2' = da2 - \overline{da2}
     cs = \mathbb{F}(da1') * {\mathbb{F}(da2')}^*

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
    dim : list (optional)
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed.
    shift : bool (optional)
        Whether to shift the fft output.
    detrend : str (optional)
        If `constant`, the mean across the transform dimensions will be
        subtracted before calculating the Fourier transform (FT).
        If `linear`, the linear least-square fit along one axis will be
        subtracted before the FT. It will give an error if the length of
        `dim` is longer than one.
    density : list (optional)
        If true, it will normalize the spectrum to spectral density
    window : bool (optional)
        Whether to apply a Hann window to the data before the Fourier
        transform is taken

    Returns
    -------
    cs : `xarray.DataArray`
        Two-dimensional cross spectrum
    """

    if dim is None:
        dim = da1.dims
        dim2 = da2.dims
        if dim != dim2:
            raise ValueError('The two datasets have different dimensions')

    # the axes along which to take ffts
    axis_num = [da1.get_axis_num(d) for d in dim]

    N = [da1.shape[n] for n in axis_num]

    daft1 = dft(da1, dim=dim,
                shift=shift, detrend=detrend, window=window)
    daft2 = dft(da2, dim=dim,
                shift=shift, detrend=detrend, window=window)

    coord = list(daft1.coords)

    cs = (daft1 * np.conj(daft2)).real

    if density:
        cs /= (np.asarray(N).prod())**2
        for i in dim:
            cs /= daft1['freq_' + i + '_spacing']

    return cs

def _azimuthal_avg(k, l, f, fftdim, N, nfactor):
    """
    Takes the azimuthal average of a given field.
    """
    k = k.values; l = l.values
    kk, ll = np.meshgrid(k, l)
    K = np.sqrt(kk**2 + ll**2)
    nbins = int(N/nfactor)
    if k.max() > l.max():
        ki = np.linspace(0., l.max(), nbins)
    else:
        ki = np.linspace(0., k.max(), nbins)

    kidx = np.digitize(np.ravel(K), ki)
    area = np.bincount(kidx)

    kr = np.bincount(kidx, weights=K.ravel()) / area

    if f.ndim == 2:
        iso_f = np.ma.masked_invalid(np.bincount(kidx,
                                    weights=f.data.ravel())
                                    / area) * kr
    else:
        raise ValueError('The data has too many or few dimensions.'
                        'The input should only have the two dimensions'
                        'to take the azimuthal averaging over.')

    return kr, iso_f

def isotropic_powerspectrum(da, dim=None, shift=True, detrend=None,
                       density=True, window=False, nfactor=4):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrum by taking the
    azimuthal average.

    ..math::

     iso_ps = k_r \frac{1}{N_{\theta}} \sum_{N_{\theta}} |\mathbb{F}(da')|^2

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed
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
    iso_ps : `xarray.DataArray`
        Isotropic power spectrum
    """

    if dim is None:
        dim = da.dims[-2:]
    if len(dim) != 2:
        raise ValueError('The azimuthal averaging can only be applied to '
                        'two dimensions.')

    ps = power_spectrum(da, dim=dim, shift=shift,
                       detrend=detrend, density=density,
                       window=window)

    fftdim = ['freq_' + d for d in dim]
    k = ps[fftdim[1]]
    l = ps[fftdim[0]]

    axis_num = [da.get_axis_num(d) for d in dim]
    N = [da.shape[n] for n in axis_num]
    kr, iso_ps = _azimuthal_avg(k, l, ps, fftdim,
                                np.asarray(N).min(), nfactor)

    k_coords = {'freq_r': kr}

    newdims = []
    for i in range(ps.ndim-1):
        if i not in axis_num:
            newdims.append(ps.dims[i])
    newdims.append('freq_r')

    newcoords = {}
    for d in newdims:
        if d in da.coords:
            newcoords[d] = da.coords[d].values
        else:
            newcoords[d] = k_coords[d]

    # dk = [l[1] - l[0] for l in kr]
    # for this_dk, d in zip(dk, dim):
    #     newcoords[prefix + d + '_spacing'] = this_dk

    return xr.DataArray(iso_ps, dims=newdims, coords=newcoords)

def isotropic_crossspectrum(da1, da2,
                        dim=None, shift=True, detrend=None,
                        density=True, window=False, nfactor=4):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrumby taking the
    azimuthal average.

    ..math::

     iso_ps = k_r \frac{1}{N_{\theta}} \sum_{N_{\theta}} \\
            (\mathbb{F}(da1') \times {\mathbb{F}(da2')}^* )

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
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
        dim = da1.dims[-2:]
        dim2 = da2.dims[-2:]
        if dim != dim2:
            raise ValueError('The two datasets have different dimensions')
    if len(dim) != 2:
        raise ValueError('The azimuthal averaging can only be applied to '
                        'two dimensions.')

    cs = cross_spectrum(da1, da2, dim=dim, shift=shift,
                       detrend=detrend, density=density,
                       window=window)
    # if len(cs.dims) > 2:
    #     raise ValueError('The data set has too many dimensions')

    fftdim = ['freq_' + d for d in dim]
    k = cs[fftdim[1]]
    l = cs[fftdim[0]]

    axis_num = [da1.get_axis_num(d) for d in dim]
    N = [da1.shape[n] for n in axis_num]
    kr, iso_cs = _azimuthal_avg(k, l, cs, fftdim,
                                np.asarray(N).min(), nfactor)

    k_coords = {'freq_r': kr}

    newdims = []
    for i in range(cs.ndim-1):
        if i not in axis_num:
            newdims.append(cs.dims[i])
    newdims.append('freq_r')

    newcoords = {}
    for d in newdims:
        if d in da1.coords:
            newcoords[d] = da1.coords[d].values
        else:
            newcoords[d] = k_coords[d]

    return xr.DataArray(iso_cs, dims=newdims, coords=newcoords)

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
    #A = np.vstack([np.log2(x), np.ones(len(x))]).T
    #a, b = np.linalg.lstsq(A, np.log2(y))[0]
    #y_fit = 2**(np.log2(x)*a + b)

    return y_fit, p[0], p[1]
