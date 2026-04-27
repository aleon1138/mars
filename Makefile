# Thin wrapper around CMake. The python extension is also buildable via
# `pip install .` which uses the same CMakeLists.txt via scikit-build-core.
#
# For a sanitizer build, configure CMake directly (then `make` to build):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
#       -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
#       -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address \
#       -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address \
#       -DMARS_BUILD_TESTS=ON

BUILD_DIR  ?= build
BUILD_TYPE ?= Release

.PHONY: all test format clean configure

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DMARS_BUILD_TESTS=ON

all: configure
	cmake --build $(BUILD_DIR) -j
	cp $(BUILD_DIR)/marslib*.so .

test: all
	python3 -m pytest tests/ -v
	$(BUILD_DIR)/unittest

format:
	astyle -A4 -S -z2 -n -j *.h *.cc

clean:
	rm -rf __pycache__/ $(BUILD_DIR)/ mars.egg-info/ marslib*.so dist
