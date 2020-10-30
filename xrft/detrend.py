"""
Functions for detrending xarray data.
"""

import xarray as xr
import scipy.signal as sps


def detrend(da, dim, detrend_type='constant'):
    """Detrend a DataArray along dimensions described by dim"""

    if detrend_type not in ['constant','linear', None]:
        raise NotImplementedError("%s is not a valid detrending option. Valid "
                                  "options are: 'constant','linear', or None."
                                  % detrend_type)

    if detrend_type is None:
        return da
    elif detrend_type == 'constant':
        return da - da.mean(dim=dim)
    elif detrend_type == 'linear':
        data = da.data
        axis_num = [da.get_axis_num(d) for d in dim]
        chunks = getattr(data, 'chunks', None)
        if chunks:
            axis_chunks = [data.chunks[a] for a in axis_num]
            assert all([len(ac)==1 for ac in axis_chunks]), 'Contiguous chunks required for detrending'
        if len(dim) == 1:
            da_detrend = xr.apply_ufunc(sps.detrend, da, axis_num[0],
                                        output_dtypes=[da.dtype],
                                        dask='parallelized')
            return da_detrend

#
#         elif len(dim) > 3:
#             raise NotImplementedError("Detrending over more than 4 axes is "
#                                  "not implemented.")
#
#         # If taking FFT over all dimensions don't need to check for chunking
#         if len(dim) == len(da.dims):
#             da = detrendn(da, axes=axis_num)
#
#         else:
#             if da.chunks == None:
#                 raise ValueError("Linear detrending utilizes the "
#                                  "`dask.map_blocks` API so the dimensions "
#                                  "not being detrended must have a chunk "
#                                  "length of 1. Please chunk your data "
#                                  "first by calling, e.g., `da.chunk('dim': 1)`.")
#
#             for d in da.dims:
#                 if d not in dim:
#                     a_n = da.get_
#
#
# def detrendn(da, axes=None):
#     """
#     Detrend by subtracting out the least-square plane or least-square cubic fit
#     depending on the number of axis.
#
#     Parameters
#     ----------
#     da : `dask.array`
#         The data to be detrended
#
#     Returns
#     -------
#     da : `numpy.array`
#         The detrended input data
#     """
#     N = [da.shape[n] for n in axes]
#     M = []
#     for n in range(da.ndim):
#         if n not in axes:
#             M.append(da.shape[n])
#
#     if len(N) == 2:
#         G = np.ones((N[0]*N[1],3))
#         for i in range(N[0]):
#             G[N[1]*i:N[1]*i+N[1], 1] = i+1
#             G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)
#         if type(da) == xr.DataArray:
#             d_obs = np.reshape(da.copy().values, (N[0]*N[1],1))
#         else:
#             d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
#     elif len(N) == 3:
#         if type(da) == xr.DataArray:
#             if da.ndim > 3:
#                 raise NotImplementedError("Cubic detrend is not implemented "
#                                          "for 4-dimensional `xarray.DataArray`."
#                                          " We suggest converting it to "
#                                          "`dask.array`.")
#             else:
#                 d_obs = np.reshape(da.copy().values, (N[0]*N[1]*N[2],1))
#         else:
#             d_obs = np.reshape(da.copy(), (N[0]*N[1]*N[2],1))
#
#         G = np.ones((N[0]*N[1]*N[2],4))
#         G[:,3] = np.tile(np.arange(1,N[2]+1), N[0]*N[1])
#         ys = np.zeros(N[1]*N[2])
#         for i in range(N[1]):
#             ys[N[2]*i:N[2]*i+N[2]] = i+1
#         G[:,2] = np.tile(ys, N[0])
#         for i in range(N[0]):
#             G[len(ys)*i:len(ys)*i+len(ys),1] = i+1
#     else:
#         raise NotImplementedError("Detrending over more than 4 axes is "
#                                  "not implemented.")
#
#     m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
#     d_est = np.dot(G, m_est)
#
#     linear_fit = np.reshape(d_est, da.shape)
#
#     return da - linear_fit
#
# def detrend_wrap(detrend_func):
#     """
#     Wrapper function for `xrft.detrendn`.
#     """
#     def func(a, axes=None):
#
#         if len(set(axes)) < len(axes):
#             raise ValueError("Duplicate axes are not allowed.")
#
#         return dsar.map_blocks(detrend_func, a, axes,
#                                    chunks=a.chunks, dtype=a.dtype
#                                   )
#
#     return func
#
# def _apply_detrend(da, dim, axis_num, detrend_type):
#     """Wrapper function for applying detrending"""
#
#     if detrend_type not in ['constant','linear',None]:
#         raise NotImplementedError("%s is not a valid detrending option. Valid "
#                                   "options are: 'constant','linear', or None."
#                                   % detrend_type)
#
#     if detrend_type == 'constant':
#         return da - da.mean(dim=dim)
#
#     elif detrend_type == 'linear':
#         if len(dim) == 1:
#             p = da.polyfit(dim=dim[0], deg=1)
#             linear_fit = xr.polyval(da[dim[0]], p.polyfit_coefficients)
#             return da - linear_fit
#
#         elif len(dim) > 3:
#             raise NotImplementedError("Detrending over more than 4 axes is "
#                                  "not implemented.")
#
#         # If taking FFT over all dimensions don't need to check for chunking
#         if len(dim) == len(da.dims):
#             da = detrendn(da, axes=axis_num)
#
#         else:
#             if da.chunks == None:
#                 raise ValueError("Linear detrending utilizes the "
#                                  "`dask.map_blocks` API so the dimensions "
#                                  "not being detrended must have a chunk "
#                                  "length of 1. Please chunk your data "
#                                  "first by calling, e.g., `da.chunk('dim': 1)`.")
#
#             for d in da.dims:
#                 if d not in dim:
#                     a_n = da.get_axis_num(d)
#                     if len(da.chunks[a_n]) != len(da[str(d)]):
#                         raise ValueError("Linear detrending utilizes the "
#                                          "`dask.map_blocks` API so the dimensions "
#                                          "not being detrended must have a chunk "
#                                          "length of 1. Please rechunk your data "
#                                          "first by calling, e.g., `da.chunk('%s': 1)`. " %d)
#
#             func = detrend_wrap(detrendn)
#             da = xr.DataArray(func(da.data, axes=axis_num),
#                          dims=da.dims, coords=da.coords)
#
#         return da
