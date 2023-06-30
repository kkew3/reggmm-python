import sys
from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np

if sys.platform == 'darwin':
    openmp_args = ['-Xpreprocessor', '-fopenmp']
else:
    openmp_args = ['-fopenmp']

extensions = [
    Extension(
        name='c_reggmm',
        sources=['_reggmm.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=openmp_args,
        extra_link_args=openmp_args,
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level='3'),
    zip_safe=False,
)
