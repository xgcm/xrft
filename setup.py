import os
import versioneer
from setuptools import setup, find_packages
PACKAGES = find_packages()

VERSION = 'v0.1'
DISTNAME = 'xrft'
LICENSE = 'MIT'
AUTHOR = 'xrft Developers'
AUTHOR_EMAIL = 'takaya@ldeo.columbia.edu'
URL = 'https://github.com/xgcm/xrft'
CLASSIFIERS = [
    'Development Status :: 3 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['xarray', 'dask', 'numpy', 'future', 'docrep']
SETUP_REQUIRES = ['pytest-runner']
TESTS_REQUIRE = ['pytest >= 2.8', 'coverage']

DESCRIPTION = "Discrete Fourier Transform with xarray"
def readme():
    with open('README.md') as f:
        return f.read()

setup(name=DISTNAME,
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
      packages=find_packages())

# # Get version and release info, which is all stored in xrft/version.py
# ver_file = os.path.join('xrft', 'version.py')
# with open(ver_file) as f:
#     exec(f.read())

# opts = dict(name=NAME,
#             maintainer=MAINTAINER,
#             maintainer_email=MAINTAINER_EMAIL,
#             description=DESCRIPTION,
#             long_description=LONG_DESCRIPTION,
#             url=URL,
#             download_url=DOWNLOAD_URL,
#             license=LICENSE,
#             classifiers=CLASSIFIERS,
#             author=AUTHOR,
#             author_email=AUTHOR_EMAIL,
#             platforms=PLATFORMS,
#             version=VERSION,
#             packages=PACKAGES,
#             package_data=PACKAGE_DATA,
#             requires=REQUIRES)
#
#
# if __name__ == '__main__':
#     setup(**opts)
