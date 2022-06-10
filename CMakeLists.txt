cmake_minimum_required(VERSION 3.21)
project(opencl_lab1 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fopenmp")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(kt_3-StannisMod main.c select.c opencl_lab2.h)

target_include_directories (kt_3-StannisMod PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (kt_3-StannisMod ${OpenCL_LIBRARY})

#target_link_libraries(opencl_lab1 ${OPENCL_LIBRARY})
