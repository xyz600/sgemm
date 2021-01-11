cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

project(matrix_gpu CUDA CXX)

# FLAGS

set(
  CMAKE_CUDA_FLAGS
  "-lineinfo -use_fast_math -Xptxas=-v -Xcompiler=\"${CMAKE_CXX_FLAGS}\""
  )
set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"${CMAKE_CXX_FLAGS_DEBUG}\"")
set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"${CMAKE_CXX_FLAGS_RELEASE}\"")

# CUDA SOURCES
file(GLOB CU_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.c*)
file(GLOB CU_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.h*)
add_library(matrix_gpu ${CU_SOURCES} ${CU_HEADERS})

target_include_directories(matrix_gpu SYSTEM
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(matrix_gpu
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/matrix_cpu)

set_target_properties(matrix_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_gpu PROPERTIES CUDA_ARCHITECTURES 61)