from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules=cythonize("_lda_helpers.pyx")
# )

setup(
    ext_modules=[
        Extension("_lda_helpers", ["_lda_helpers.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)
