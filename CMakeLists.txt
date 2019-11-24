cmake_minimum_required(VERSION 3.8)
project(sgemm)

set(CMAKE_CXX_COMPILER nvcc)
set(CMAKE_CXX_FLAGS "--generate-code arch=compute_61,code=sm_61")

add_subdirectory(matrix)
add_subdirectory(cuda_matrix)

include_directories(
    ${CMAKE_SOURCE_DIR}/matrix
)

add_executable(benchmark
    main.cpp
)

link_directories(
    ${CMAKE_SOURCE_DIR}/matrix
    ${CMAKE_SOURCE_DIR}/cuda_matrix
)

target_link_libraries(benchmark
    matrix
    cuda_matrix
)