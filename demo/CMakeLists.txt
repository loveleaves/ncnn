# 设置最低 CMake 版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(NCNN_Test)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_definitions(-Wall -Wextra)
add_definitions(-g)

# 查找 NCNN 库和头文件
set(NCNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/ncnn/include)
set(NCNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/ncnn/lib)

# 包含 NCNN 的头文件
include_directories(${NCNN_INCLUDE_DIR})

# 定义源文件
set(SOURCE_FILES test.cpp)

# 创建可执行文件
add_executable(test_infer ${SOURCE_FILES})

# 链接 NCNN 静态库 libncnn.a
find_package(OpenCV REQUIRED core highgui imgproc)
target_link_libraries(test_infer ${NCNN_LIB_DIR}/libncnn.a ${OpenCV_LIBS})

link_directories(${NCNN_LIB_DIR})
