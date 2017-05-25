import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import dask.array as dsar

__all__ = ["dft","power_spectrum","cross_spectrum",
            "isotropic_powerspectrum","isotropic_crossspectrum",
            "fit_loglog"]


def _hanning(da, N):
    """Apply Hanning window"""

    window = np.hanning(N[-1]) * np.hanning(N[-2])[:, np.newaxis]

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

def dft(da, dim=None, shift=True, remove_mean=True, window=False):
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
    remove_mean : bool (optional)
        If `True`, the mean across the transform dimensions will be subtracted
        before calculating the Fourier transform.

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

    if remove_mean:
        da = da - da.mean(dim=dim)

    if window:
        da = _hanning(da, N)

    # the hard work
    #f = np.fft.fftn(da.values, axes=axis_num)
    # need special path for dask
    # is this the best way to check for dask?
    data = da.data
    if hasattr(data, 'dask'):
        assert len(axis_num)==1
        f = dsar.fft.fft(data, axis=axis_num[0])
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
            newcoords[d] = da.coords[d]
        else:
            newcoords[d] = k_coords[d]

    dk = [l[1] - l[0] for l in k]
    for this_dk, d in zip(dk, dim):
        newcoords[prefix + d + '_spacing'] = this_dk

    return xr.DataArray(f, dims=newdims, coords=newcoords)

def power_spectrum(da, dim=None, shift=True, remove_mean=True, density=True,
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
    remove_mean : bool (optional)
        If `True`, the mean across the transform dimensions will be subtracted
        before calculating the Fourier transform.
    density : list (optional)
        If true, it will normalize the spectrum to spectral density

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
            dim=dim, shift=shift, remove_mean=remove_mean,
            window=window)

    coord = list(daft.coords)

    ps = np.real(daft*np.conj(daft))
    if density:
        ps /= (np.asarray(N).prod())**2
        for i in range(len(dim)):
            ps /= daft[coord[-i-1]].values

    return xr.DataArray(ps, coords=daft.coords, dims=daft.dims)

def cross_spectrum(da1, da2, dim=None,
                   shift=True, remove_mean=True, density=True, window=False):
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
    remove_mean : bool (optional)
        If `True`, the mean across the transform dimensions will be subtracted
        before calculating the Fourier transform.
    density : list (optional)
        If true, it will normalize the spectrum to spectral density

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
                shift=shift, remove_mean=remove_mean, window=window)
    daft2 = dft(da2, dim=dim,
                shift=shift, remove_mean=remove_mean, window=window)

    coord = list(daft1.coords)

    cs = np.real(daft1 * np.conj(daft2))
    if density:
        cs /= (np.asarray(N).prod())**2
        for i in range(int(len(dim))):
            cs /= daft1[coord[-i-1]].values

    return xr.DataArray(cs, coords=daft1.coords, dims=daft1.dims)

def _azimuthal_avg(k, l, f, nbins=64):
    """
    Takes the azimuthal average of a given field.
    """

    kk, ll = np.meshgrid(k, l)
    K = np.sqrt(kk**2 + ll**2)
    if k.max() > l.max():
        ki = np.linspace(0., l.max(), nbins)
    else:
        ki = np.linspace(0., k.max(), nbins)

    kidx = np.digitize(K.ravel(), ki)
    area = np.bincount(kidx)

    kr = np.ma.masked_invalid(np.bincount(kidx,
                                          weights=K.ravel()) / area
                                         )
    iso_f = np.ma.masked_invalid(np.bincount(kidx,
                                            weights=f.values.ravel())
                                            / area
                                            ) * kr
    return kr, iso_f

def isotropic_powerspectrum(da, dim=None, shift=True, remove_mean=True,
                       density=True, window=False, nbins=64):
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
    remove_mean : bool (optional)
        If `True`, the mean across the transform dimensions will be subtracted
        before calculating the Fourier transform.
    density : list (optional)
        If true, it will normalize the spectrum to spectral density

    Returns
    -------
    iso_ps : `xarray.DataArray`
        Isotropic power spectrum
    """

    if dim is None:
        dim = da.dims

    ps = power_spectrum(da, dim=dim, shift=shift,
                       remove_mean=remove_mean, density=density,
                       window=window)
    if len(ps.dims) > 2:
        raise ValueError('The data set has too many dimensions')

    k = ps[ps.dims[0]].values
    l = ps[ps.dims[1]].values

    kr, iso_ps = _azimuthal_avg(k, l, ps, nbins=nbins)

    return xr.DataArray(iso_ps, dims=['freq_r'],
                        coords={'freq_r':kr})

def isotropic_crossspectrum(da1, da2,
                        dim=None, shift=True, remove_mean=True,
                        density=True, window=False, nbins=64):
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
    remove_mean : bool (optional)
        If `True`, the mean across the transform dimensions will be subtracted
        before calculating the Fourier transform.
    density : list (optional)
        If true, it will normalize the spectrum to spectral density

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

    cs = cross_spectrum(da1, da2, dim=dim, shift=shift,
                       remove_mean=remove_mean, density=density,
                       window=window)
    if len(cs.dims) > 2:
        raise ValueError('The data set has too many dimensions')

    k = cs[cs.dims[0]].values
    l = cs[cs.dims[1]].values

    kr, iso_cs = _azimuthal_avg(k, l, cs, nbins=nbins)

    return xr.DataArray(iso_cs, dims=['freq_r'],
                        coords={'freq_r':kr})

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
