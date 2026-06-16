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

# --- CUDA (opt-in) -----------------------------------------------------------
# Build the GPU orthonormalize kernels into a static lib and link it into
# marslib.so. OFF by default so CPU-only builds / CI without nvcc still work.
# Enable with -DUSE_CUDA=ON (or, for a wheel, pip install . -C cmake.define.USE_CUDA=ON).
option(USE_CUDA "Build CUDA orthonormalize kernels" OFF)
if(USE_CUDA)
    # Must be set BEFORE enable_language(CUDA): enabling the language initializes
    # CMAKE_CUDA_ARCHITECTURES (from the build host's detected GPU), after which a
    # `NOT DEFINED` default would never fire. 75 = RTX 2080 Ti (dev), 120 = RTX
    # PRO 6000 Blackwell (research server). Override with
    # -DCMAKE_CUDA_ARCHITECTURES=... to pin a single target.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;120")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(mars_cuda STATIC cuda/orthonormalize.cu)
    set_target_properties(mars_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON)   # linked into a shared object (marslib.so)
    target_include_directories(mars_cuda PRIVATE "${CMAKE_SOURCE_DIR}")  # kernels.h
    # mars_cuda must NOT inherit mars_common (nvcc rejects -march=native). Keep
    # the project's hidden-visibility hygiene by passing it to the host compiler.
    target_compile_options(mars_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
    target_link_libraries(mars_cuda PRIVATE CUDA::cublas CUDA::cudart)
endif()

# Eigen3 is used only by the C++ unit tests (as the correctness oracle); the
# shipped library is Eigen-free. find_package(Eigen3) lives in the BUILD_TESTING
# block below, so a library/wheel build does not require Eigen at all.

# --- OpenMP ------------------------------------------------------------------
# On macOS, statically link Homebrew's libomp into marslib.so so the OpenMP
# runtime stays private to this extension. Avoids the "OMP: Error #15:
# Initializing libomp.dylib, but found libomp.dylib already initialized"
# crash that fires whenever another extension in the same Python interpreter
# (numpy/scipy/sklearn/...) loads its own libomp.dylib first. Mac's two-level
# namespace + the project-wide -fvisibility=hidden keep the static OMP from
# leaking out of marslib.so.
#
# On Linux, find_package(OpenMP) is fine -- the system loader resolves a
# single OMP runtime for the whole process and conflicts don't arise.
add_library(mars_openmp INTERFACE)
if(APPLE)
    if(NOT DEFINED OpenMP_ROOT)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE _brew_libomp
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_brew_libomp AND IS_DIRECTORY "${_brew_libomp}")
            set(OpenMP_ROOT "${_brew_libomp}")
        endif()
    endif()
    find_library(LIBOMP_STATIC NAMES libomp.a HINTS "${OpenMP_ROOT}/lib")
    find_path(LIBOMP_INCLUDE NAMES omp.h     HINTS "${OpenMP_ROOT}/include")
    if(NOT LIBOMP_STATIC OR NOT LIBOMP_INCLUDE)
        message(FATAL_ERROR
            "libomp.a / omp.h not found under '${OpenMP_ROOT}'. "
            "Install with 'brew install libomp' or pass "
            "-DOpenMP_ROOT=<prefix> to cmake.")
    endif()
    target_compile_options(mars_openmp INTERFACE -Xpreprocessor -fopenmp)
    target_include_directories(mars_openmp INTERFACE "${LIBOMP_INCLUDE}")
    target_link_options(mars_openmp INTERFACE
        "LINKER:-force_load,${LIBOMP_STATIC}")
else()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(mars_openmp INTERFACE OpenMP::OpenMP_CXX)
endif()

# --- Shared compile/link settings -------------------------------------------
add_library(mars_common INTERFACE)
target_compile_features(mars_common INTERFACE cxx_std_17)
target_compile_options(mars_common INTERFACE -O3 -Wall -march=native -fvisibility=hidden)
target_compile_definitions(mars_common INTERFACE NDEBUG)
target_link_libraries(mars_common INTERFACE mars_openmp)
# -march=native does not imply -mfma on x86 macOS — add it explicitly.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    target_compile_options(mars_common INTERFACE -mfma)
endif()

# --- pybind11 module ---------------------------------------------------------
pybind11_add_module(marslib marslib.cc marsalgo.cc kernels.cc)
target_link_libraries(marslib PRIVATE mars_common)
if(USE_CUDA)
    target_link_libraries(marslib PRIVATE mars_cuda)
    target_compile_definitions(marslib PRIVATE MARS_USE_CUDA)
endif()

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
    # Eigen is the oracle the kernel/algorithm tests check against; only the
    # tests need it (the library itself is Eigen-free).
    find_package(Eigen3 3.3 CONFIG QUIET)
    if(NOT Eigen3_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(EIGEN3 IMPORTED_TARGET eigen3)
        endif()
    endif()
    if(NOT Eigen3_FOUND AND NOT EIGEN3_FOUND)
        message(FATAL_ERROR
            "Eigen3 not found (required for the unit tests). Install eigen3 "
            "(e.g. apt install libeigen3-dev, brew install eigen) or set "
            "Eigen3_DIR / CMAKE_PREFIX_PATH.")
    endif()

    find_package(GTest REQUIRED)
    find_package(Threads REQUIRED)
    add_executable(unittest tests/unittest.cc tests/test_kernels.cc
                            marsalgo.cc kernels.cc)
    target_compile_definitions(unittest PRIVATE EIGEN_DONT_PARALLELIZE)
    target_link_libraries(unittest PRIVATE
        mars_common GTest::gtest GTest::gtest_main Threads::Threads)
    if(Eigen3_FOUND)
        target_link_libraries(unittest PRIVATE Eigen3::Eigen)
    else()
        target_link_libraries(unittest PRIVATE PkgConfig::EIGEN3)
    endif()

    # CUDA orthonormalize tests: validate the GPU path against the same Eigen
    # oracle and against the CPU kernel. Host-compiled (calls only the
    # mars::cuda::* host API), linked against the CUDA static lib.
    if(USE_CUDA)
        add_executable(test_cuda tests/test_cuda.cc kernels.cc)
        target_include_directories(test_cuda PRIVATE "${CMAKE_SOURCE_DIR}")
        target_link_libraries(test_cuda PRIVATE
            mars_common mars_cuda CUDA::cudart
            GTest::gtest GTest::gtest_main Threads::Threads)
        if(Eigen3_FOUND)
            target_link_libraries(test_cuda PRIVATE Eigen3::Eigen)
        else()
            target_link_libraries(test_cuda PRIVATE PkgConfig::EIGEN3)
        endif()
    endif()
endif()
