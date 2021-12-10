"""
Functions to pad and unpad an *N*\-dimensional regular grid.
"""
import numpy as np
from xarray.core.utils import either_dict_or_kwargs

from .utils import get_spacing


def pad(
    da,
    pad_width=None,
    mode="constant",
    stat_length=None,
    constant_values=0,
    end_values=None,
    reflect_type=None,
    **pad_width_kwargs,
):
    """
    Pad array with evenly spaced coordinates.

    Wraps the :meth:`xarray.DataArray.pad` method but also pads the evenly
    spaced coordinates by extrapolation using the same coordinate spacing.
    The ``pad_width`` used for each coordinate is stored as one of its
    attributes.

    Parameters
    ----------
    da : xarray.DataArray
        Array to be padded. The coordinates along which the array will be
        padded must be evenly spaced.
    pad_width : mapping of hashable to tuple of int
        Mapping with the form of {dim: (pad_before, pad_after)}
        describing the number of values padded along each dimension.
        {dim: pad} is a shortcut for pad_before = pad_after = pad
    mode : str, default: 'constant'
        One of the following string values (taken from :func:`numpy.pad` doc).

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
    da_padded : xarray.DataArray

    See Also
    --------
    xrft.padding.unpad

    Examples
    --------

    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     coords={"x": [0, 1, 2], "y": [-5, -4, -3]},
    ...     dims=("y", "x"),
    ... )
    >>> da_padded = pad(da, x=2, y=1)
    >>> da_padded
    <xarray.DataArray (y: 5, x: 7)>
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3, 0, 0],
           [0, 0, 4, 5, 6, 0, 0],
           [0, 0, 7, 8, 9, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    Coordinates:
      * x        (x) int64 -2 -1 0 1 2 3 4
      * y        (y) int64 -6 -5 -4 -3 -2
    >>> da_padded.x
    <xarray.DataArray 'x' (x: 7)>
    array([-2, -1,  0,  1,  2,  3,  4])
    Coordinates:
      * x        (x) int64 -2 -1 0 1 2 3 4
    Attributes:
        pad_width:  2
    >>> da_padded.y
    <xarray.DataArray 'y' (y: 5)>
    array([-6, -5, -4, -3, -2])
    Coordinates:
      * y        (y) int64 -6 -5 -4 -3 -2
    Attributes:
        pad_width:  1

    Asymmetric padding

    >>> da_padded = pad(da, x=(1, 4))
    >>> da_padded
    <xarray.DataArray (y: 3, x: 8)>
    array([[0, 1, 2, 3, 0, 0, 0, 0],
           [0, 4, 5, 6, 0, 0, 0, 0],
           [0, 7, 8, 9, 0, 0, 0, 0]])
    Coordinates:
      * x        (x) int64 -1 0 1 2 3 4 5 6
      * y        (y) int64 -5 -4 -3
    >>> da_padded.x
    <xarray.DataArray 'x' (x: 8)>
    array([-1,  0,  1,  2,  3,  4,  5,  6])
    Coordinates:
      * x        (x) int64 -1 0 1 2 3 4 5 6
    Attributes:
        pad_width:  (1, 4)

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
    padded_coords = _pad_coordinates(da.coords, pad_width)
    # Assign the padded coordinates to the padded array
    padded_da = padded_da.assign_coords(padded_coords)
    # Edit the attributes of the padded array
    for dim in pad_width.keys():
        # Add attrs of the original coords to the padded array
        padded_da.coords[dim].attrs.update(da.coords[dim].attrs)
        # Add the pad_width used for this coordinate
        padded_da.coords[dim].attrs.update({"pad_width": pad_width[dim]})
    # Return padded array
    return padded_da


def _pad_coordinates(coords, pad_width):
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
            mode=_pad_coordinates_callback,
            spacing=spacing,  # spacing is passed as a kwarg to the callback
        )
    return padded_coords


def _pad_coordinates_callback(vector, iaxis_pad_width, iaxis, kwargs):
    """
    Callback for padding coordinates

    This function is not intended to be called, but to be passed as the
    ``mode`` method to the :func:`numpy.pad`

    Parameters
    ----------
    vector : 1d-array
        A rank 1 array already padded with zeros. Padded values are
        ``vector[:iaxis_pad_width[0]]`` and ``vector[-iaxis_pad_width[1]:]``.
    iaxis_pad_width : tuple
        A 2-tuple of ints, ``iaxis_pad_width[0]`` represents the number of
        values padded at the beginning of vector where ``iaxis_pad_width[1]``
        represents the number of values padded at the end of vector.
    iaxis : int
        The axis currently being calculated. This parameter is not used, but
        the function will check if it's equal to zero. It exists for
        compatibility with the ``padding_func`` callback that :func:`numpy.pad`
        needs.
    kwargs : dict
        Any keyword arguments the function requires. The kwargs are ignored in
        this function, they exist for compatibility with the ``padding_func``
        callback that :func:`numpy.pad` needs.

    Returns
    -------
    vector : 1d-array
        Padded vector.
    """
    assert iaxis == 0
    spacing = kwargs["spacing"]
    n_start, n_end = iaxis_pad_width[:]
    vmin, vmax = vector[n_start], vector[-(n_end + 1)]
    vector[:n_start] = np.arange(vmin - spacing * n_start, vmin, spacing)
    vector[-n_end:] = np.arange(vmax + spacing, vmax + spacing * (n_end + 1), spacing)
    return vector


def unpad(da, pad_width=None, **pad_width_kwargs):
    """
    Unpad an array and its coordinates.

    Undo the padding process of the :func:`xrft.pad` function by slicing the
    passed :class:`xarray.DataArray` `da` and its coordinates.

    Parameters
    ----------
    da : xarray.DataArray
        Padded array. The coordinates along which the array will be
        padded must be evenly spaced.
    pad_width : mapping of hashable to tuple of int (optional)
        Mapping with the form of {dim: (pad_before, pad_after)}
        describing the number of values padded along each dimension.
        {dim: pad} is a shortcut for pad_before = pad_after = pad.
        If ``None``, then the *pad_width* for each coordinate is read from
        their ``pad_width`` attribute.
    **pad_width_kwargs : dict, optional
        The keyword arguments form of ``pad_width``.
        Pass ``pad_width`` or ``pad_width_kwargs``.

    Returns
    -------
    da_unpadded : xarray.DataArray
        Unpadded array.

    See Also
    --------
    xrft.padding.pad

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     coords={"x": [0, 1, 2], "y": [-5, -4, -3]},
    ...     dims=("y", "x"),
    ... )
    >>> da_padded = pad(da, x=2, y=1)
    >>> da_padded
    <xarray.DataArray (y: 5, x: 7)>
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3, 0, 0],
           [0, 0, 4, 5, 6, 0, 0],
           [0, 0, 7, 8, 9, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    Coordinates:
      * x        (x) int64 -2 -1 0 1 2 3 4
      * y        (y) int64 -6 -5 -4 -3 -2
    >>> unpad(da_padded)
    <xarray.DataArray (y: 3, x: 3)>
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 -5 -4 -3

    Custom ``pad_width``

    >>> unpad(da_padded, x=1, y=1)
    <xarray.DataArray (y: 3, x: 5)>
    array([[0, 1, 2, 3, 0],
           [0, 4, 5, 6, 0],
           [0, 7, 8, 9, 0]])
    Coordinates:
      * x        (x) int64 -1 0 1 2 3
      * y        (y) int64 -5 -4 -3


    """
    # Generate the pad_width dictionary
    if pad_width is None and not pad_width_kwargs:
        # Read the pad_width from each coordinate if pad_width is None and
        # no pad_width_kwargs has been passed
        pad_width = {
            dim: coord.attrs["pad_width"]
            for dim, coord in da.coords.items()
            if "pad_width" in coord.attrs
        }
        # Raise error if there's no pad_width attribute in the coordinates
        if not pad_width:
            raise ValueError(
                "The passed array doesn't seem to be a padded one: the 'pad_width' "
                + "attribute was missing on every one of its coordinates. "
            )
    else:
        # Redefine pad_width if pad_width_kwargs were passed
        pad_width = either_dict_or_kwargs(pad_width, pad_width_kwargs, "pad")
    # Transform every pad_width into a tuple with indices
    slices = {}
    for dim in pad_width:
        slices[dim] = _pad_width_to_slice(pad_width[dim], da.coords[dim].size)
    # Slice the padded array
    unpadded_da = da.isel(indexers=slices)
    # Remove the pad_width attribute from coords since it's no longer necessary
    for dim in pad_width:
        if "pad_width" in unpadded_da.coords[dim].attrs:
            unpadded_da.coords[dim].attrs.pop("pad_width")
    return unpadded_da


def _pad_width_to_slice(pad_width, size):
    """
    Create a slice for removing the padded elements of a coordinate array

    Parameters
    ----------
    pad_width : int or tuple
        Integer or tuples with the width of the padding at each side of the
        coordinate array. An integer means an equal padding at each side equal
        to its value.
    size : int
        Number of elements of the coordinates array.

    Returns
    -------
    coord_slice : slice
        A slice object for removing the padded elements of the coordinate
        array.
    """
    if type(pad_width) == int:
        pad_width = (pad_width, pad_width)
    return slice(pad_width[0], size - pad_width[1])
