# Thin wrapper around CMake (the real build driver is CMakeLists.txt).
#
#   make                 CPU-only build; drops marslib*.so at the source root
#   make cuda            one-shot GPU build (configure with CUDA + build)
#   make test            build, then run the C++ unit tests and pytest
#   make configure-cuda  reconfigure with the GPU kernels only (no build)
#   make format          astyle all .h/.cc
#   make clean           remove build/ and the built .so
#
# CUDA is opt-in and OFF by default: `make` never touches nvcc. `make cuda` does
# a one-shot GPU build. For a CUDA wheel instead of a dev build, use
# `pip install . -C cmake.define.USE_CUDA=ON` (see README.md).
#
# For memchecks, configure by hand with:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
#         -DCMAKE_CXX_FLAGS="-O0 -g -fsanitize=address"

BUILD_DIR ?= build

.PHONY: all cuda configure configure-cuda test format clean

all: configure
	cmake --build $(BUILD_DIR) --parallel

configure: $(BUILD_DIR)/CMakeCache.txt

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON

# One-shot GPU build: reconfigure with CUDA, then build.
cuda: configure-cuda
	cmake --build $(BUILD_DIR) --parallel

# Opt-in CUDA build (drops the GPU kernels into marslib.so + builds test_cuda).
# Only *reconfigures* — `make cuda` (or a follow-up `make`) does the build.
# Reuses the existing build dir, so run `make clean` first to switch a CPU build
# over (and again to switch back to CPU-only).
configure-cuda:
	cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON -DUSE_CUDA=ON

test: all
	$(BUILD_DIR)/unittest
	python3 -m pytest tests/ -v

format:
	astyle -A4 -S -z2 -n -j *.h *.cc

clean:
	rm -rf $(BUILD_DIR) __pycache__/ mars.egg-info/ dist marslib*.so .ruff_cache .pytest_cache
