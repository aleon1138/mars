# Thin wrapper around CMake (the real build driver is CMakeLists.txt).
#
#   make                 build; drops marslib*.so at the source root. The GPU
#                        kernels are built in when nvcc and an NVIDIA GPU are
#                        both detected (USE_CUDA=AUTO), CPU-only otherwise
#   NOCUDA=1 make        CPU-only build, no probing
#   make cuda            GPU build, required (configure error without nvcc)
#   make test            build, then run the C++ unit tests (+ CUDA tests if
#                        built), then pytest
#   make configure-cuda  reconfigure with the GPU kernels required (no build)
#   make format          astyle all .h/.cc
#   make clean           remove build/ and the built .so
#
# The CUDA decision is made at configure time and cached in build/, so run
# `make clean` before switching between a CPU-only and a GPU build. A wheel
# makes the same AUTO decision: `pip install .`, with `NOCUDA=1 pip install .`
# and `pip install . -C cmake.define.USE_CUDA=ON` to force it (see README.md).
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

# Required CUDA build (drops the GPU kernels into marslib.so + builds
# test_cuda; a configure error if nvcc is missing). Only *reconfigures* --
# `make cuda` (or a follow-up `make`) does the build. Reuses the existing build
# dir, so run `make clean` first to switch a CPU build over (and again to switch
# back to CPU-only).
configure-cuda:
	cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON -DUSE_CUDA=ON

# test_cuda only exists in a CUDA configure (auto-detected, or `make cuda`);
# run it when it's there. The binary itself GTEST_SKIPs every case if no GPU
# is visible.
test: all
	$(BUILD_DIR)/unittest
	@if [ -x $(BUILD_DIR)/test_cuda ]; then \
		$(BUILD_DIR)/test_cuda; \
	else \
		echo "[make test] $(BUILD_DIR)/test_cuda not built - skipping CUDA tests"; \
	fi
	python3 -m pytest tests/ -v

format:
	astyle -A4 -S -z2 -n -j *.h *.cc

clean:
	rm -rf $(BUILD_DIR) __pycache__/ mars.egg-info/ dist marslib*.so .ruff_cache .pytest_cache
