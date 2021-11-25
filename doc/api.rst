.. currentmodule:: xrft

#############
API reference
#############

This page provides a summary of xrft's top-level API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.


.. note::

  None of xrft's functions will work correctly in the presence of NaNs or
  missing data. It's the user's responsibility to ensure data are free of NaN
  or that NaNs have been filled somehow.

FFT routines
============

.. autosummary::
   :toctree: api/

   fft
   ifft
   power_spectrum
   cross_spectrum
   cross_phase
   isotropize
   isotropic_power_spectrum
   isotropic_cross_spectrum
  

Deprecated names:

.. autosummary::
   :toctree: api/

   dft
   idft
   isotropic_powerspectrum
   isotropic_crossspectrum


Detrending
==========

You may wish to use xrft's detrend function on its own.

.. autosummary::
   :toctree: api/

   detrend

Padding
=======

Pad and unpad arrays and its coordinates so they can be used for computing
FFTs.

.. autosummary::
   :toctree: api/

   pad
   unpad

Misc.
=====

.. autosummary::
   :toctree: api/

   fit_loglog
