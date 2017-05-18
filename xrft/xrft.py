import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import dask.array as dsar

__all__ = ["dft","power_spectrum","cross_spectrum",
            "isotropic_powerspectrum","isotropic_crossspectrum",
            "fit_loglog"]


def dft(da, dim=None, shift=True, remove_mean=True):
    """
    Perform discrete Fourier transform of xarray data-array `da` along the
    specified dimensions.

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
    k = [ np.fft.fftfreq(Nx, dx) for (Nx, dx) in zip(N, delta_x)]

    if remove_mean:
        da = da - da.mean(dim=dim)

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

def power_spectrum(da, dim=None, shift=True, remove_mean=True, density=True):
    """
    Calculates the power spectrum of da.

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

    daft = dft(da, dim=dim, shift=shift, remove_mean=remove_mean)
    coord = list(daft.coords)

    ps = np.real(daft*np.conj(daft))
    if density:
        ps /= (np.asarray(N).prod())**2
        for i in range(int(len(dim))):
            ps /= daft[coord[-i-1]].values

    if density:
        delta_x = []
        for d in dim:
            coord = da[d]
            diff = np.diff(coord)
            if pd.core.common.is_timedelta64_dtype(diff):
                # convert to seconds so we get hertz
                diff = diff.astype('timedelta64[s]').astype('f8')
            delta = diff[0]
            delta_x.append(delta)
        np.testing.assert_almost_equal(ps.sum()
                                       / (np.asarray(delta_x).prod()
                                          * (da.values**2).sum()
                                         ), 1., decimal=5
                                      )

    return xr.DataArray(ps, coords=daft.coords, dims=daft.dims)

def cross_spectrum(da1, da2, a1=1., a2=1., dim=None,
                   shift=True, remove_mean=True, density=True):
    """
    Calculates the cross spectra of da1 and da2.

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
    a1 : float64
        Coefficient of da1
    a2 : float64
        Coefficient of da2
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
        if len(dim) != len(dim2):
            raise ValueError('The two datasets have different dimensions')

    # the axes along which to take ffts
    axis_num = [da1.get_axis_num(d) for d in dim]

    N = [da1.shape[n] for n in axis_num]

    daft1 = dft(da1, dim=dim,
                shift=shift, remove_mean=remove_mean)
    daft2 = dft(da2, dim=dim,
                shift=shift, remove_mean=remove_mean)
    coord = list(daft1.coords)

    cs = np.real(a1*daft1 * a2*np.conj(daft2))
    if density:
        cs /= (np.asarray(N).prod())**2
        for i in range(int(len(dim))):
            cs /= daft1[coord[-i-1]].values

    if density:
        delta_x = []
        for d in dim:
            coord = da1[d]
            diff = np.diff(coord)
            if pd.core.common.is_timedelta64_dtype(diff):
                # convert to seconds so we get hertz
                diff = diff.astype('timedelta64[s]').astype('f8')
            delta = diff[0]
            delta_x.append(delta)
        np.testing.assert_almost_equal(cs.sum()
                                       / (np.asarray(delta_x).prod()
                                          * (a1*a2*da1.values*da2.values).sum()
                                         ), 1., decimal=5
                                      )

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
                       density=True, nbins=64):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrum.

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
                       remove_mean=remove_mean, density=density)
    if len(ps.dims) > 2:
        raise ValueError('The data set has too many dimensions')

    k = ps[ps.dims[0]].values
    l = ps[ps.dims[1]].values

    kr, iso_ps = _azimuthal_avg(k, l, ps, nbins=nbins)

    return xr.DataArray(iso_ps, dims=['freq_r'],
                        coords={'freq_r':kr})

def isotropic_crossspectrum(da1, da2, a1=1., a2=1.,
                        dim=None, shift=True, remove_mean=True,
                        density=True, nbins=64):
    """
    Calculates the isotropic spectrum from the
    two-dimensional power spectrum.

    Parameters
    ----------
    da1 : `xarray.DataArray`
        The data to be transformed
    da2 : `xarray.DataArray`
        The data to be transformed
    a1 : float64
        Coefficient of da1
    a2 : float64
        Coefficient of da2
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
        if len(dim) != len(dim2):
            raise ValueError('The two datasets have different dimensions')

    cs = cross_spectrum(da1, da2, a1=a1, a2=a2, dim=dim, shift=shift,
                       remove_mean=remove_mean, density=density)
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
