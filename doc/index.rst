.. xrft documentation master file, created by
   sphinx-quickstart on Wed Aug 22 12:19:33 2018.


xrft: Fourier transforms for xarray data
==============================================

**xrft** is a Python package for
taking the discrete Fourier transform (DFT) on xarray_ and dask_ arrays.
It is:

- **Powerful**: It keeps the metadata and coordinates of the original xarray dataset and provides a clean work flow of DFT.
- **Easy-to-use**: It uses the native arguments of numpy FFT and provides a simple, high-level API.
- **Fast**: It uses the dask API of FFT and map_blocks to allow parallelization of DFT.

.. note::

    xrft is at early stage of development and will keep improving in the future.
    The discrete Fourier transform API should be quite stable,
    but minor utilities could change in the next version.
    If you find any bugs or would like to request any enhancements,
    please `raise an issue on GitHub <https://github.com/xgcm/xrft/issues>`_.

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   why-xrft
   limitations
   installation
   contributor_guide

.. toctree::
   :maxdepth: 1
   :caption: Examples

   DFT-iDFT_example
   Parseval_example
   chunk_example
   MITgcm_example

.. toctree::
   :maxdepth: 1
   :caption: Help & reference

   whats-new
   api


.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.pydata.org/en/latest/array-api.html
