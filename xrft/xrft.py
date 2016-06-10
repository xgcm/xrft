import numpy as np
import xarray as xr
from .due import due, Doi

__all__ = ["dft"]


def dft(da, dim=None, shift=True, remove_mean=True, density=False):
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
        diff = np.diff(da[d])
        if not np.allclose(diff, diff[0]):
            raise ValueError("Can't take Fourier transform because"
                             "coodinate %s is not evenly spaced" % d)
        delta_x.append(diff[0])
    # calculate frequencies from coordinates
    k = [ np.fft.fftfreq(Nx, dx) for (Nx, dx) in zip(N, delta_x)]

    if remove_mean:
        da = da - da.mean(dim=dim)

    # the hard work
    f = np.fft.fftn(da.values, axes=axis_num)

    if shift:
        f = np.fft.fftshift(f, axes=axis_num)
        k = [np.fft.fftshift(l) for l in k]

    dk = [l[1] - l[0] for l in k]

    if density:
        spectral_volume = reduce(lambda x, y: x*y, dk)
        f /= spectral_volume

    # set up new dimensions for dataarray
    k_names = ['freq_'+d for d in dim]
    k_coords = { key: val for (key,val) in zip(k_names, k)}

    newdims = list(da.dims)
    for anum, d in zip(axis_num, dim):
        newdims[anum] = 'freq_' + d

    newcoords = {}
    for d in newdims:
        if d in da.coords:
            newcoords[d] = da.coords[d]
        else:
            newcoords[d] = k_coords[d]

    for this_dk, d in zip(dk, dim):
        newcoords['freq_' + d + '_spacing'] = this_dk
