cmake_minimum_required(VERSION 3.20)
project(marslib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# On macOS with Homebrew, help find_package() locate Eigen3 and libomp by
# adding brew prefixes to CMAKE_PREFIX_PATH. User-set CMAKE_PREFIX_PATH wins.
if(APPLE)
    execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE _BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_BREW_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH
            "${_BREW_PREFIX}"
            "${_BREW_PREFIX}/opt/libomp"
            "${_BREW_PREFIX}/opt/eigen"
        )
    endif()
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# pybind11 is typically pip-installed into the active Python env, not via a
# system package manager. Ask the interpreter for its CMake config dir so
# direct `cmake` invocations work; `pip install .` already sets pybind11_DIR.
if(NOT pybind11_DIR)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
find_package(pybind11 CONFIG REQUIRED)

add_compile_options(-O3 -Wall -march=native)
if(APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # -mfma is x86-only and not implied by -march=native on Apple Clang
    add_compile_options(-mfma)
endif()

pybind11_add_module(marslib MODULE marslib.cc marsalgo.cc)
target_link_libraries(marslib PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_definitions(marslib PRIVATE NDEBUG EIGEN_DONT_PARALLELIZE)

# Wheel layout: drop marslib.<abi>.so and mars.py at the wheel root.
install(TARGETS marslib DESTINATION .)
install(FILES mars.py DESTINATION .)

# C++ gtest binary — only when explicitly requested (skipped by pip/wheel build).
option(MARS_BUILD_TESTS "Build the C++ unit test binary" OFF)
if(MARS_BUILD_TESTS)
    find_package(GTest REQUIRED)
    add_executable(unittest tests/unittest.cc marsalgo.cc)
    target_link_libraries(unittest PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        GTest::gtest
        GTest::gtest_main
    )
endif()
