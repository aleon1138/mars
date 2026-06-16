# Thin wrapper around CMake. For memchecks, configure with:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
#         -DCMAKE_CXX_FLAGS="-O0 -g -fsanitize=address"

BUILD_DIR ?= build

.PHONY: all configure configure-cuda test format clean

all: configure
	cmake --build $(BUILD_DIR) --parallel

configure: $(BUILD_DIR)/CMakeCache.txt

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON

# Opt-in CUDA build (drops the GPU kernels into marslib.so + builds test_cuda).
# Reconfigures the existing build dir; run `make clean` first to switch back.
configure-cuda:
	cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON -DUSE_CUDA=ON

test: all
	$(BUILD_DIR)/unittest
	python3 -m pytest tests/ -v

format:
	astyle -A4 -S -z2 -n -j *.h *.cc

clean:
	rm -rf $(BUILD_DIR) __pycache__/ mars.egg-info/ dist marslib*.so .ruff_cache .pytest_cache
