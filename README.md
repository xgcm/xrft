# xrft : Fourier transforms for xarray data

[![Build Status](https://travis-ci.org/xgcm/xrft.svg?branch=master)](https://travis-ci.org/xgcm/xrft)

[![codecov](https://codecov.io/gh/xgcm/xrft/branch/master/graph/badge.svg)](https://codecov.io/gh/xgcm/xrft)

[![Documentation Status](https://readthedocs.org/projects/xrft/badge/?version=latest)](https://xrft.readthedocs.io/en/latest/?badge=latest)

[![DOI](https://zenodo.org/badge/60866797.svg)](https://zenodo.org/badge/latestdoi/60866797)


xrft is a Python package for
taking the discrete Fourier transform (DFT) on xarray and dask arrays.
It is

- **Powerful**: It keeps the metadata and coordinates of the original xarray dataset and provides a clean work flow of DFT.
- **Easy-to-use**: It uses the native arguments of numpy FFT and provides a simple, high-level API.
- **Fast**: It uses the dask API of FFT and map_blocks to allow parallelization of DFT.
