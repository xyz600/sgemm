
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -Wall")

add_library(matrix
    matrix_cpu.cpp
)