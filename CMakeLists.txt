cmake_minimum_required(VERSION 3.18)
project(mars LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Python + pybind11 -------------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# --- Eigen3 (prefer CMake config, fall back to pkg-config) -------------------
find_package(Eigen3 3.3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 IMPORTED_TARGET eigen3)
    endif()
endif()
if(NOT Eigen3_FOUND AND NOT EIGEN3_FOUND)
    message(FATAL_ERROR
        "Eigen3 not found. Install eigen3 (e.g. apt install libeigen3-dev, "
        "brew install eigen) or set Eigen3_DIR / CMAKE_PREFIX_PATH.")
endif()

# --- OpenMP (with Homebrew libomp fallback on macOS) -------------------------
if(APPLE AND NOT DEFINED OpenMP_ROOT)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE _brew_libomp
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_brew_libomp AND IS_DIRECTORY "${_brew_libomp}")
        set(OpenMP_ROOT "${_brew_libomp}")
    endif()
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

# --- Shared compile/link settings -------------------------------------------
add_library(mars_common INTERFACE)
target_compile_features(mars_common INTERFACE cxx_std_17)
target_compile_options(mars_common INTERFACE -O3 -Wall -march=native -fvisibility=hidden)
target_compile_definitions(mars_common INTERFACE NDEBUG EIGEN_DONT_PARALLELIZE)
target_link_libraries(mars_common INTERFACE OpenMP::OpenMP_CXX)
if(Eigen3_FOUND)
    target_link_libraries(mars_common INTERFACE Eigen3::Eigen)
else()
    target_link_libraries(mars_common INTERFACE PkgConfig::EIGEN3)
endif()
# -march=native does not imply -mfma on x86 macOS — add it explicitly.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    target_compile_options(mars_common INTERFACE -mfma)
endif()

# --- pybind11 module ---------------------------------------------------------
pybind11_add_module(marslib marslib.cc marsalgo.cc kernels.cc)
target_link_libraries(marslib PRIVATE mars_common)

# For standalone dev builds, drop marslib*.so next to mars.py so `import
# marslib` works from the source tree. scikit-build-core sets SKBUILD when it
# drives the build, in which case we leave the file in the build dir and let
# the install() rule place it in the wheel.
if(NOT DEFINED SKBUILD)
    set_target_properties(marslib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

install(TARGETS marslib LIBRARY DESTINATION .)
install(FILES mars.py DESTINATION .)

# --- C++ unit tests (opt-in) -------------------------------------------------
option(BUILD_TESTING "Build C++ unit tests" OFF)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    find_package(Threads REQUIRED)
    add_executable(unittest tests/unittest.cc tests/test_kernels.cc
                            marsalgo.cc kernels.cc)
    target_link_libraries(unittest PRIVATE
        mars_common GTest::gtest GTest::gtest_main Threads::Threads)
endif()
