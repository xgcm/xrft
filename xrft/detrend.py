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
