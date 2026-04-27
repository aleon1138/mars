import os
import platform
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


def find_libomp_prefix():
    """Locate Homebrew libomp prefix on macOS."""
    prefix = os.environ.get("LIBOMP_PREFIX")
    if prefix and os.path.isdir(prefix):
        return prefix
    try:
        out = subprocess.check_output(
            ["brew", "--prefix", "libomp"],
            stderr=subprocess.DEVNULL,
        )
        path = out.decode().strip()
        if os.path.isdir(path):
            return path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    for path in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.isdir(path):
            return path
    print(
        "WARNING: libomp not found. Install with `brew install libomp` "
        "or set LIBOMP_PREFIX.",
        file=sys.stderr,
    )
    return None


is_darwin = sys.platform == "darwin"
is_arm64 = platform.machine() in ("arm64", "aarch64")

include_dirs = [find_eigen_include()]
library_dirs = []
extra_compile_args = ["-O3", "-Wall", "-march=native"]
extra_link_args = []

if is_darwin:
    # -mfma is x86-only; not valid on Apple Silicon.
    if not is_arm64:
        extra_compile_args.append("-mfma")
    # Apple Clang requires libomp (Homebrew) and -Xpreprocessor -fopenmp.
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args.append("-lomp")
    libomp_prefix = find_libomp_prefix()
    if libomp_prefix:
        include_dirs.append(os.path.join(libomp_prefix, "include"))
        library_dirs.append(os.path.join(libomp_prefix, "lib"))
else:
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")


ext_modules = [
    Pybind11Extension(
        "marslib",
        sources=["marslib.cc", "marsalgo.cc"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
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
