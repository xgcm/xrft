xrft: Fourier transforms for xarray data
=========================================

.. image:: https://travis-ci.org/xgcm/xrft.svg?branch=master
   :target: https://travis-ci.org/xgcm/xrft
.. image:: https://codecov.io/gh/xgcm/xrft/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/xgcm/xrft
.. image:: https://readthedocs.org/projects/xrft/badge/?version=latest
   :target: https://xrft.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1402636.svg
   :target: https://doi.org/10.5281/zenodo.1402636

**xrft** is an open-source Python package for
taking the discrete Fourier transform (DFT) on xarray_ and dask_ arrays.

.. _xarray: http://xarray.pydata.org/en/stable/
.. _dask: https://dask.org

It is:

- **Powerful**: It keeps the metadata and coordinates of the original xarray dataset and provides a clean work flow of DFT.
- **Easy-to-use**: It uses the native arguments of `numpy FFT`_ and provides a simple, high-level API.
- **Fast**: It uses the `dask API of FFT`_ and `map_blocks`_ to allow parallelization of DFT.

.. _numpy FFT: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
.. _dask API of FFT: http://docs.dask.org/en/latest/array-api.html?highlight=fft#fast-fourier-transforms
.. _map_blocks: http://docs.dask.org/en/latest/array-api.html?highlight=map_blocks#dask.array.core.map_blocks

Get in touch
------------

- Report bugs, suggest features or view the source code `on GitHub`_.

.. _on GitHub: https://github.com/xgcm/xrft
