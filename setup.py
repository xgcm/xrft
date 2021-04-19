import os
import versioneer
from setuptools import setup, find_packages

PACKAGES = find_packages()

DISTNAME = "xrft"
LICENSE = "MIT"
AUTHOR = "xrft Developers"
AUTHOR_EMAIL = "takaya@ldeo.columbia.edu"
URL = "https://github.com/xgcm/xrft"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = ["xarray", "dask", "numpy", "pandas", "scipy"]
EXTRAS_REQUIRE = ["cftime", "numpy_groupies"]
SETUP_REQUIRES = ["pytest-runner"]
TESTS_REQUIRE = ["pytest >= 2.8", "coverage"]

DESCRIPTION = "Discrete Fourier Transform with xarray"


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=readme(),
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    packages=find_packages(),
)
