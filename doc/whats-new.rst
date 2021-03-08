.. currentmodule:: xrft

What's New
==========

.. _whats-new.0.3.0:

v0.3.0 (18 February 2021)
----------------------

Enhancements
~~~~~~~~~~~~

- Implemented the inverse discrete Fourier transform ``idft``. By `Frederic Nouguier <https://github.com/lanougue>`_

- Allowed windowing other than the Hann function. By `Takaya Uchida <https://github.com/roxyboy>`_

- Allowed parallelization of isotropizing the spectrum via ``numpy_groupies``.
  By `Takaya Uchida <https://github.com/roxyboy>`_

- Implemented proper amplitude correction for real Fourier transform and windowed data. By `Dougie Squire <https://github.com/dougiesquire>`_

.. _whats-new.0.2.0:

v0.2.0 (10 April 2019)
----------------------

Enhancements
~~~~~~~~~~~~

- Allowed ``dft`` and ``power_spectrum`` functions to support real Fourier transforms. (:issue:`57`)
  By `Takaya Uchida <https://github.com/roxyboy>`_ and
  `Tom Nicholas <https://github.com/TomNicholas>`_.

- Implemented ``cross_phase`` function to calculate the phase difference between two signals as a function of frequency.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

- Allowed ``isotropic_powerspectrum`` function to support arrays with up to four dimensions. (:issue:`9`)
  By `Takaya Uchida <https://github.com/roxyboy>`_

.. warning::

  Python 2.7 is no longer supported in ``xrft``. For the more details, see:

  - `Python 3 Statement <http://www.python3statement.org/>`__
  - `Tips on porting to Python 3 <https://docs.python.org/3/howto/pyporting.html>`__
