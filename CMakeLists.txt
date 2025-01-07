
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/ncnn" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

project(ncnn)

cmake_minimum_required(VERSION 2.8.10)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug CACHE STRING "Choose the type of build" FORCE)
endif()

# Note: option会缓存在cmakefile中
option(NCNN_SHARED_LIB "shared library support" OFF)
option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_OPENCV "minimal opencv structure emulation" OFF)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" OFF)
option(NCNN_ASAN "build for address sanitizer" OFF)
option(NCNN_COVERAGE "build for coverage" OFF)

if(NCNN_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# warning, all and extra
add_definitions(-Wall -Wextra -Wno-unused-function)

# Position-Independent Code，PIC
add_definitions(-fPIC)
if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
    add_definitions(-Ofast)
    add_definitions(-ffast-math)
endif()
# add_definitions(-march=native) # 将目标架构设置为当前主机的架构

# Link Time Optimization，LTO
# add_definitions(-flto)

# 将所有符号的默认可见性设置为隐藏，包含内联函数的符号
add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)

##############################################

add_subdirectory(examples)
# add_subdirectory(benchmark)
add_subdirectory(src)
add_subdirectory(tools)