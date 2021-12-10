Overview: Why xrft?
===================

For robustness and efficiency
-----------------------------

In the field of Earth Science, we often take Fourier transforms of the variable of interest.
There has, however, not been a universal algorithm with which we calculate the transforms,
and our aim is to streamline this process.

We utilize the Dask_ API to parallelize the computation to make it efficient for large data sets.

For usability and simplicity
----------------------------

The arguments in xrft rely on well-established standards
(Dask and NumPy), so users don't need to learn a bunch of new syntaxes or even a new software stack.

xrft can track the metadata in :class:`xarray.DataArray`\s (:doc:`example <./MITgcm_example>`),
which makes it easy to use large data sets.

The choice of Python and Anaconda also makes xrft :ref:`extremely easy to install <installation-label>`.


.. _Dask: http://dask.pydata.org/en/latest/array-api.html
