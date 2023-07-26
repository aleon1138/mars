from distutils.core import setup, Extension
import pybind11

module = Extension(
    "marslib",
    extra_compile_args=["-fopenmp", "-march=native", "-std=c++11"],
    extra_link_args=["-fopenmp"],
    include_dirs=[pybind11.get_include(), "/usr/include/eigen3"],
    sources=["marslib.cc"],
)

setup(
    name="marslib",
    version="0.1",
    author="Arnaldo Leon",
    description="Multivariate Adaptive Regression Splines",
    py_modules=["mars"],
    ext_modules=[module],
)
