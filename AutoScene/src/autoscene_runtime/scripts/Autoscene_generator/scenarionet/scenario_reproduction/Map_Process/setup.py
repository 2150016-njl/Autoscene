from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("skewed_rectangle_cython.pyx"),
)
