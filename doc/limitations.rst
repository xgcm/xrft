Current limitations
===================

.. _limitations-label:

Windowing
---------

xrft currently only supports the Hanning window.

Discrete sinusoid transform
---------------------------

xrft currently only supports discrete fourier transforms. We plan to implement
discrete sinusoid tranforms in the near future.

Parallel isotropic spectrum
---------------------------

Isotropic spectral calculations in xrft currently only run in serial along the axes
that are not being transformed.
Parallel options are being investigated.
