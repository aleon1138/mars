from setuptools import setup, Extension
import pybind11

extra_compile_args = [
    "-O3",
    "-Wall",
    "-std=c++14",
    "-march=native",
    "-fopenmp",
]

define_macros = [
    ("NDEBUG", "1"),
    ("EIGEN_DONT_PARALLELIZE", "1"),
]

module = Extension(
    "marslib",
    extra_compile_args=extra_compile_args,
    extra_link_args=["-fopenmp"],
    include_dirs=[pybind11.get_include(), "/usr/include/eigen3"],
    sources=["marslib.cc"],
    define_macros=define_macros,
    language="c++",
)

setup(
    name="mars",
    version="0.1",
    author="Arnaldo Leon",
    description="Multivariate Adaptive Regression Splines",
    py_modules=["mars"],
    ext_modules=[module],
)
