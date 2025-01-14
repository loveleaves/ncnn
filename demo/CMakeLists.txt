# 设置最低 CMake 版本要求
cmake_minimum_required(VERSION 2.8.12)

# 设置项目名称
project(NCNN_Demo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

add_definitions(-Wall -Wextra)
add_definitions(-O2 -g)
# add_definitions(-O3)
# set(CMAKE_BUILD_TYPE release)

# 查找 NCNN 库和头文件
set(NCNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/ncnn/include)
set(NCNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/ncnn/lib)

# 包含 NCNN 的头文件
include_directories(${NCNN_INCLUDE_DIR})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

## ncnn macro
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_tools.cmake)

option(NCNN_ASAN "build for address sanitizer" ON)

set(CMAKE_BUILD_TYPE debug CACHE STRING "Choose the type of build" FORCE)

ncnn_get_file_define(${NCNN_INCLUDE_DIR}/platform.h NCNN_SIMPLEOCV)
if(NOT NCNN_SIMPLEOCV)
    find_package(OpenCV QUIET COMPONENTS opencv_world)
    # for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
    if("${OpenCV_LIBS}" STREQUAL "")
        set(OpenCV_FOUND FALSE)
    endif()
    if(NOT OpenCV_FOUND)
        find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
    endif()
    if(NOT OpenCV_FOUND)
        find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
    endif()
endif()

macro(ncnn_add_demo name)
    add_executable(${name} ${name}.cpp)
    if(NCNN_SIMPLEOCV)
        # define USE_NCNN_SIMPLEOCV into specified file to use simpleocv
        target_compile_definitions(${name} PUBLIC USE_NCNN_SIMPLEOCV)
        target_link_libraries(${name} PRIVATE ${NCNN_LIB_DIR}/libncnn.a)
        message(STATUS "Demo: using simpleocv")
    elseif(OpenCV_FOUND)
        target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ${NCNN_LIB_DIR}/libncnn.a ${OpenCV_LIBS})
    endif()

    if(NCNN_ASAN)
        message(STATUS "Enabling AddressSanitizer (ASan)")
        target_compile_options(${name} PUBLIC -fsanitize=address)
        target_link_libraries(${name} PUBLIC -fsanitize=address)
    endif()
endmacro()

ncnn_add_demo(squeezenet)
