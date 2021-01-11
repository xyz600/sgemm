cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

project(benchmark_main CUDA CXX)

# main program
add_executable(benchmark_main main.cu)
target_link_libraries(benchmark_main
                      PRIVATE matrix_cpu matrix_gpu)
target_include_directories(benchmark_main
                      PRIVATE matrix_cpu matrix_gpu)

set_target_properties(benchmark_main PROPERTIES CUDA_ARCHITECTURES 61)