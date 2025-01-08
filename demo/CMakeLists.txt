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

# 链接 NCNN 静态库 libncnn.a
find_package(OpenCV REQUIRED core highgui imgproc)

macro(ncnn_add_demo name)
    add_executable(${name} ${name}.cpp)
    if(OpenCV_FOUND)
        target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ${NCNN_LIB_DIR}/libncnn.a ${OpenCV_LIBS})
    elseif(NCNN_SIMPLEOCV)
        # define USE_NCNN_SIMPLEOCV to use simpleocv
        target_compile_definitions(${name} PUBLIC USE_NCNN_SIMPLEOCV)
        target_link_libraries(${name} PRIVATE ${NCNN_LIB_DIR}/libncnn.a)
    endif()
    configure_file(demo_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/demo_config.h)
endmacro()

ncnn_add_demo(squeezenet)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
