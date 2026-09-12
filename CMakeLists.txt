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
  src/fgs.cpp
  src/hand169.cpp
  src/icm.cpp
  src/poker.cpp
  src/showdown.cpp)
target_include_directories(ppcore PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(ppcore PUBLIC Threads::Threads)

# Solver CLI.
add_executable(ppsolve src/main.cpp)
target_link_libraries(ppsolve PRIVATE ppcore)

# Tests.
add_executable(ppsolver_tests tests/test_main.cpp)
target_link_libraries(ppsolver_tests PRIVATE ppcore)
target_include_directories(ppsolver_tests PRIVATE tests)
enable_testing()
add_test(NAME unit_tests COMMAND ppsolver_tests)

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
  else()
    message(STATUS "CUDA toolkit not found; building CPU-only")
  endif()
endif()
