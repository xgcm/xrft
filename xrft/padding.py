"""
Functions to pad and unpad a N-dimensional regular grid
"""
import numpy as np
from xarray.core.utils import either_dict_or_kwargs

from .utils import get_spacing


def pad(
    da,
    pad_width=None,
    mode="constant",
    stat_length=None,
    constant_values=None,
    end_values=None,
    reflect_type=None,
    **pad_width_kwargs,
):
    """
    Pad array with evenly spaced coordinates

    Wraps the :meth:`xarray.DataArray.pad` method but also pads the evenly
    spaced coordinates by extrapolation using the same coordinate spacing.

    Parameters
    ----------
    da : :class:`xarray.DataArray`
        Array to be padded. The coordinates along which the array will be
        padded must be evenly spaced.
    pad_width : mapping of hashable to tuple of int
        Mapping with the form of {dim: (pad_before, pad_after)}
        describing the number of values padded along each dimension.
        {dim: pad} is a shortcut for pad_before = pad_after = pad
    mode : str, default: "constant"
        One of the following string values (taken from numpy docs).
        - constant: Pads with a constant value.
        - edge: Pads with the edge values of array.
        - linear_ramp: Pads with the linear ramp between end_value and the
          array edge value.
        - maximum: Pads with the maximum value of all or part of the
          vector along each axis.
        - mean: Pads with the mean value of all or part of the
          vector along each axis.
        - median: Pads with the median value of all or part of the
          vector along each axis.
        - minimum: Pads with the minimum value of all or part of the
          vector along each axis.
        - reflect: Pads with the reflection of the vector mirrored on
          the first and last values of the vector along each axis.
        - symmetric: Pads with the reflection of the vector mirrored
          along the edge of the array.
        - wrap: Pads with the wrap of the vector along the axis.
          The first values are used to pad the end and the
          end values are used to pad the beginning.
    stat_length : int, tuple or mapping of hashable to tuple, default: None
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.
        {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique
        statistic lengths along each dimension.
        ((before, after),) yields same before and after statistic lengths
        for each dimension.
        (stat_length,) or int is a shortcut for before = after = statistic
        length for all axes.
        Default is ``None``, to use the entire axis.
    constant_values : scalar, tuple or mapping of hashable to tuple, default: 0
        Used in 'constant'.  The values to set the padded values for each
        axis.
        ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
        pad constants along each dimension.
        ``((before, after),)`` yields same before and after constants for each
        dimension.
        ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
        all dimensions.
        Default is 0.
    end_values : scalar, tuple or mapping of hashable to tuple, default: 0
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.
        ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
        end values along each dimension.
        ``((before, after),)`` yields same before and after end values for each
        axis.
        ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
        all axes.
        Default is 0.
    reflect_type : {"even", "odd"}, optional
        Used in "reflect", and "symmetric".  The "even" style is the
        default with an unaltered reflection around the edge value.  For
        the "odd" style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.
    **pad_width_kwargs
        The keyword arguments form of ``pad_width``.
        One of ``pad_width`` or ``pad_width_kwargs`` must be provided.

    Returns
    -------
    da_padded : :class:`xarray.DataArray`

    Examples
    --------

    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     coords={"x": [0, 1, 2], "y": [-5, -4, -3]},
    ...     dims=("y", "x"),
    ... )
    >>> pad(da, x=2, y=1, constant_values=0)
    <xarray.DataArray (y: 5, x: 7)>
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3, 0, 0],
           [0, 0, 4, 5, 6, 0, 0],
           [0, 0, 7, 8, 9, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    Coordinates:
      * x        (x) int64 -2 -1 0 1 2 3 4
      * y        (y) int64 -6 -5 -4 -3 -2

    """
    # Redefine pad_width if pad_width_kwargs were passed
    pad_width = either_dict_or_kwargs(pad_width, pad_width_kwargs, "pad")
    # Pad the array using the xarray.DataArray.pad method
    padded_da = da.pad(
        pad_width,
        mode,
        stat_length,
        constant_values,
        end_values,
        reflect_type,
    )
    # Pad the coordinates selected in pad_width
    padded_coords = pad_coordinates(da.coords, pad_width)
    # Assign the padded coordinates to the padded array
    padded_da = padded_da.assign_coords(padded_coords)
    # Add attrs of the original coords to the padded array
    for dim in pad_width.keys():
        padded_da.coords[dim].attrs.update(da.coords[dim].attrs)
    # Return padded array
    return padded_da


def pad_coordinates(coords, pad_width):
    """
    Pad coordinates arrays according to the passed pad_width

    Parameters
    ----------
    coords : dict-like object
        Dictionary with coordinates as :class:`xarray.DataArray`.
        Only the coordinates specified through ``pad_width`` will be padded.
        Every coordinate that will be padded should be evenly spaced.
    pad_width : dict-like object
        Dictionary with the same keys as ``coords``. The coordinates specified
        through ``pad_width`` are returned as padded.

    Returns
    -------
    padded_coords : dict-like object
        Dictionary with 1d-arrays corresponding to the padded coordinates.

    Examples
    --------

    >>> import numpy as np
    >>> import xarray as xr
    >>> x = np.linspace(-4, -1, 4)
    >>> y = np.linspace(-1, 4, 6)
    >>> coords = {
    ... "x": xr.DataArray(x, coords={"x": x}, dims=("x",)),
    ... "y": xr.DataArray(y, coords={"y": y}, dims=("y",)),
    ... }
    >>> pad_width = {"x": 2}
    >>> padded_coords = pad_coordinates(coords, pad_width)
    >>> padded_coords["x"]
    array([-6., -5., -4., -3., -2., -1.,  0.,  1.])
    >>> padded_coords["y"]
    <xarray.DataArray (y: 6)>
    array([-1.,  0.,  1.,  2.,  3.,  4.])
    Coordinates:
      * y        (y) float64 -1.0 0.0 1.0 2.0 3.0 4.0


    """
    # Generate a dictionary with the original coordinates
    padded_coords = {dim: coords[dim] for dim in coords}
    # Start padding the coordinates that appear in pad_width
    for dim in pad_width:
        # Get the spacing of the selected coordinate
        # (raises an error if not evenly spaced)
        spacing = get_spacing(padded_coords[dim])
        # Pad the coordinates using numpy.pad with the _pad_coordinates callback
        padded_coords[dim] = np.pad(
            padded_coords[dim],
            pad_width=pad_width[dim],
            mode=_pad_coordinates,
            spacing=spacing,  # spacing is passed as a kwarg to the callback
        )
    return padded_coords


def _pad_coordinates(vector, iaxis_pad_width, iaxis, kwargs):
    """
    Callback for padding coordinates

    This function is not intended to be called, but to be passed as the
    ``mode`` method to the :func:`numpy.pad`

    Parameters
    ----------
    vector : 1d-array
    iaxis_pad_width : tuple
    iaxis : int
    kwargs : dict
    """
    assert iaxis == 0
    spacing = kwargs["spacing"]
    n_start, n_end = iaxis_pad_width[:]
    vmin, vmax = vector[n_start], vector[-(n_end + 1)]
    vector[:n_start] = np.arange(vmin - spacing * n_start, vmin, spacing)
    vector[-n_end:] = np.arange(vmax + spacing, vmax + spacing * (n_end + 1), spacing)
    return vector
