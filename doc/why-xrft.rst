Overview: Why xrft?
===================

For robustness and efficiency
-----------------------------

In the field of Earth Science, we often take Fourier transforms of the variable of interest.
There has, however, not been an universal algorithm in which we calculate the transforms
and our aim is to stream line this process.

We utilize the dask_ API to parallelize the computation to make it efficient for large data sets.

For usability and simplicity
----------------------------

The arguments in xrft rely on well-estabilished standards
(dask and numpy), so users don't need to learn a bunch of new syntaxes or even a new software stack.

xrft can track the metadata in ``xarray.DataArray`` (:doc:`example <./MITgcm_example>`),
which makes it easy for large data sets.

The choice of Python and Anaconda also makes xrft :ref:`extremely easy to install <installation-label>`.


.. _dask: http://dask.pydata.org/en/latest/array-api.html
