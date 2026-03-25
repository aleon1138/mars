import os
import subprocess
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def find_eigen_include():
    """Locate Eigen3 include directory across platforms."""
    # Explicit override
    eigen_dir = os.environ.get("EIGEN3_INCLUDE_DIR")
    if eigen_dir and os.path.isdir(eigen_dir):
        return eigen_dir

    # Try pkg-config (works on most Linux, Homebrew, conda)
    try:
        out = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "eigen3"],
            stderr=subprocess.DEVNULL,
        )
        for token in out.decode().split():
            if token.startswith("-I"):
                path = token[2:]
                if os.path.isdir(path):
                    return path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Common fallback paths
    candidates = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path

    print(
        "WARNING: Eigen3 not found. Set EIGEN3_INCLUDE_DIR or install "
        "eigen3 via your package manager.",
        file=sys.stderr,
    )
    return "/usr/include/eigen3"  # optimistic fallback


ext_modules = [
    Pybind11Extension(
        "marslib",
        sources=["marslib.cc"],
        include_dirs=[find_eigen_include()],
        extra_compile_args=[
            "-O3",
            "-Wall",
            "-march=native",
            "-fopenmp",
        ],
        extra_link_args=["-fopenmp"],
        define_macros=[
            ("NDEBUG", "1"),
            ("EIGEN_DONT_PARALLELIZE", "1"),
        ],
        cxx_std=17,
    ),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
