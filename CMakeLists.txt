cmake_minimum_required(VERSION 3.18)
project(cuda-poker-solver CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(ENABLE_CUDA "Build the CUDA kernels (requires nvcc + NVIDIA GPU)" ON)

# Core CPU library (always built).
add_library(ppcore STATIC
  src/cards.cpp
  src/common.cpp
  src/eval.cpp
  src/fgs.cpp
  src/hand169.cpp
  src/icm.cpp
  src/poker.cpp
  src/postflop_cfr.cpp
  src/range.cpp
  src/showdown.cpp)
target_include_directories(ppcore PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(ppcore PUBLIC Threads::Threads)
# The Python bindings link ppcore into a shared module.
set_property(TARGET ppcore PROPERTY POSITION_INDEPENDENT_CODE ON)

# Solver CLI.
add_executable(ppsolve src/main.cpp)
target_link_libraries(ppsolve PRIVATE ppcore)

# Tests.
add_executable(ppsolver_tests tests/test_main.cpp)
target_link_libraries(ppsolver_tests PRIVATE ppcore)
target_include_directories(ppsolver_tests PRIVATE tests)
enable_testing()
add_test(NAME unit_tests COMMAND ppsolver_tests)

# Hand-computed tiny-turn ground truth (see tools/quality/gpu_postflop_parity.sh).
add_executable(tiny_check tools/tiny_check.cpp)
target_link_libraries(tiny_check PRIVATE ppcore)

# Optional CUDA kernels: batched ICM and board sampling for the solver's
# hot loops. Not exercised by the CPU solver; integrate via the launch
# helpers in cuda/poker_kernels.cu.
if(ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    add_library(ppcuda STATIC cuda/poker_kernels.cu)
    target_include_directories(ppcuda PUBLIC src)
    target_link_libraries(ppcuda PUBLIC CUDA::cudart)
    message(STATUS "CUDA kernels enabled (unvalidated without an NVIDIA GPU)")
    # On-hardware validation of the kernels against the CPU reference
    # (src/icm.cpp). Not linked against ppcuda: the harness includes
    # cuda/poker_kernels.cu directly.
    add_executable(cuda_validate tools/cuda_validate.cu)
    target_include_directories(cuda_validate PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(cuda_validate PRIVATE ppcore CUDA::cudart)
    # GPU-CFR river solver (compile-to-dataflow + CUDA graph replay).
    add_executable(gpu_pfflop tools/gpu_pfflop.cu cuda/gpu_cfr.cu)
    target_include_directories(gpu_pfflop PRIVATE ${CMAKE_SOURCE_DIR}/cuda)
    target_link_libraries(gpu_pfflop PRIVATE ppcore CUDA::cudart)
    # Python bindings (the pps package, python/). Requires pybind11 in
    # the target interpreter's env: python -m pip install pybind11 numpy.
    # Point CMake at the right interpreter with -DPython3_EXECUTABLE=...
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
    if(Python3_Development.Module_FOUND)
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKEDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RES)
      if(PYBIND11_RES EQUAL 0)
        find_package(pybind11 CONFIG QUIET HINTS ${PYBIND11_CMAKEDIR})
        if(pybind11_FOUND)
          pybind11_add_module(pps_native python/pps_native.cpp cuda/gpu_cfr.cu)
          target_include_directories(pps_native PRIVATE ${CMAKE_SOURCE_DIR}/cuda)
          target_link_libraries(pps_native PRIVATE ppcore CUDA::cudart)
          set_target_properties(pps_native PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/pps)
          message(STATUS "Python bindings enabled (pps_native)")
        else()
          message(STATUS "pybind11 cmake config not found; skipping Python bindings")
        endif()
      else()
        message(STATUS "${Python3_EXECUTABLE} cannot import pybind11; skipping Python bindings")
      endif()
    else()
      message(STATUS "Python3 Development.Module not found; skipping Python bindings")
    endif()
  else()
    message(STATUS "CUDA toolkit not found; building CPU-only")
  endif()
endif()
