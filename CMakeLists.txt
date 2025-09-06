cmake_minimum_required(VERSION 3.18)
project(MultiTestSolver LANGUAGES CXX CUDA)

# Let CMake detect the native GPU architectures automatically
set(CMAKE_CUDA_ARCHITECTURES "native")

# Or specify a list of architectures to target (optional)
# Comment out one of these two if you want to use only one method
# set(CMAKE_CUDA_ARCHITECTURES "native")  
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)  # e.g., for Turing (7.5), Ampere (8.6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOLVER_SOURCES
    src/solver.cu
    src/solver.h
)

add_executable(test1 src/test-random-float.cu ${SOLVER_SOURCES})
# add_executable(test2 src/main_test2.cpp ${SOLVER_SOURCES})

set_target_properties(test1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(test2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA runtime library directly (no CUDA::cudart target)
target_link_libraries(test1 cudart)
# target_link_libraries(test2 cudart)
