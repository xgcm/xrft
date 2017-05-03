import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import dask.array as dsar

__all__ = ["dft"]


def dft(da, dim=None, shift=True, remove_mean=True, density=False,
        iso=False, nbins=64):
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
    density : bool (optional)
        If `True`, the output will be normalized to give spectral density.
    iso : bool (optional)
        If `True`, the isotropic spectra will be calculated.
    bins : integer
        Defines the number of radial bins to take the azimuthal average over.

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

    dk = [l[1] - l[0] for l in k]

    if density:
        try:
            spectral_volume = reduce(lambda x, y: x*y, dk)
        except:
            spectral_volume = ft.reduce(lambda x, y: x*y, dk)
        f /= spectral_volume

        if iso:
            kk, ll = np.meshgrid(np.asarray(k[0]),
                                np.asarray(k[1]))
            K = np.sqrt(kk**2 + ll**2)
            ki = np.linspace(0., np.asarray(k).max(), nbins)
            # if k.max() > l.max():
            #     ki = np.linspace(0., l.max(), nbins)
            # else:
            #     ki = np.linspace(0., k.max(), nbins)
            kidx = np.digitize(K.ravel(), ki)
            invalid = kidx[-1]
            area = np.bincount(kidx)

            kr = np.ma.masked_invalid(np.bincount(kidx,
                                                weights=K.ravel()) / area)
            iso_f = np.ma.masked_invalid(np.bincount(kidx,
                                                    weights=np.real(f
                                                    *np.conj(f)).ravel())
                                                    / area) * kr

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

    for this_dk, d in zip(dk, dim):
        newcoords[prefix + d + '_spacing'] = this_dk

    if density:
        return xr.DataArray(f, dims=newdims, coords=newcoords),\
            xr.DataArray(iso_f, dims=['freq_kr'],
                        coords={'freq_kr':kr})
    else:
        return xr.DataArray(f, dims=newdims, coords=newcoords)
