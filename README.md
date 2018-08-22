# xrft : Fourier transforms for xarray data

[![Build Status](https://travis-ci.org/rabernat/xrft.svg?branch=master)](https://travis-ci.org/rabernat/xrft)

[![codecov](https://codecov.io/gh/rabernat/xrft/branch/master/graph/badge.svg)](https://codecov.io/gh/rabernat/xrft)


xrft is a Python package for
taking the discrete Fourier transform (DFT) on xarray_ and dask_ arrays.
It is

- **Powerful**: It keeps the metadata and coordinates of the original xarray dataset and provides a clean work flow of DFT.
- **Easy-to-use**: It uses the native arguments of numpy fft and provides a simple, high-level API.
- **Fast**: It uses the dask API of fft and map_blocks to allow parallelization of DFT.
