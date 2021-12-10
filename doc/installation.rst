.. _installation-label:

Installation
============

The quickest way
----------------

xrft is compatible both with Python 2 and 3. The major dependencies are xarray_ and Dask_.
The best way to install them is using Anaconda_.

.. prompt:: bash

   conda install -c conda-forge xarray dask xrft

It is also possible to install from PyPI_ by

.. prompt:: bash

   pip install xrft

Install xrft from GitHub repo
-----------------------------
To get the latest version

.. prompt:: bash

   git clone https://github.com/xgcm/xrft.git
   cd xrft
   python setup.py install .

Developers can track source code changes by

.. prompt:: bash

   git clone https://github.com/xgcm/xrft.git
   cd xrft
   python setup.py develop .

.. _xarray: http://xarray.pydata.org
.. _Dask: http://dask.pydata.org/en/latest/
.. _Anaconda: https://www.continuum.io/downloads
.. _PyPI: https://pypi.org/
