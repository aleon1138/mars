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


setup(
    ext_modules=[
        Extension(
            "marslib",
            sources=["marslib.cc"],
            include_dirs=[pybind11.get_include(), "/usr/include/eigen3"],
            extra_compile_args=extra_compile_args,
            extra_link_args=["-fopenmp"],
            define_macros=define_macros,
            language="c++",
        )
    ],
)
