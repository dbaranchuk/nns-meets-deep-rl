#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy
import os

os.environ['CC'] = 'g++ -std=c++11'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# extension module
_search_hnsw = Extension("_search_hnsw",
                         ["search_hnsw.i","search_hnsw.cc"],
                        include_dirs = [numpy_include],
                        extra_compile_args=["-Ofast", "-fopenmp", "-march=native",
                                            "-ftree-vectorize", "-ftree-vectorizer-verbose=0"],
                        extra_link_args=['-lgomp'],
                        swig_opts=['-threads']
                        )

# setup
setup(  name        = "Search in HNSW graph",
        description = "Function that performs a beam search in HNSW graph (numpy.i: a SWIG Interface File for NumPy)",
        author      = "Dmitry Baranchuk",
        version     = "1.0",
        ext_modules = [_search_hnsw]
        )
