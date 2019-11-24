cmake_minimum_required(VERSION 3.8)
project(sgemm)

add_subdirectory(matrix)

include_directories(
    ${CMAKE_SOURCE_DIR}/matrix
)

add_executable(benchmark
    main.cpp
)

link_directories(
    ${CMAKE_SOURCE_DIR}/matrix
)

target_link_libraries(benchmark
    matrix
)