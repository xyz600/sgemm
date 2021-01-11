cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

project(matrix_cpu CXX)

# Static library, therefore STATIC
file(GLOB hs_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.c*)
file(GLOB hh_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.h*)
add_library(matrix_cpu STATIC ${hs_sources} ${hh_headers})

# PUBLIC: target's dependencies of it's user and itself
target_include_directories(matrix_cpu
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(matrix_cpu PROPERTIES CUDA_ARCHITECTURES 61)