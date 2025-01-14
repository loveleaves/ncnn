set(BINARY ${CMAKE_PROJECT_NAME}_test)
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/llt/test_*.cpp")

include_directories(../src)
include_directories(../src/layer)
include_directories(../src/layer/${NCNN_TARGET_ARCH})

set(TEST_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
file(GLOB_RECURSE TEST_COMMON_SOURCES
    "${TEST_COMMON_DIR}/*.c"
    "${TEST_COMMON_DIR}/*.cpp"
)
list(APPEND TEST_SOURCES ${TEST_COMMON_SOURCES})
include_directories(llt)
include_directories(common)

# 添加 googletest 子目录
add_subdirectory(thirdParty/googletest)
add_executable(${BINARY} ${TEST_SOURCES})

find_package(OpenCV REQUIRED core highgui imgproc)
target_include_directories(${BINARY} PRIVATE ${OpenCV_INCLUDE_DIRS})

add_test(NAME ${BINARY} COMMAND ${BINARY})
# 链接src生成的lib库和gtest库
target_link_libraries(${BINARY} PUBLIC gtest ncnn ${OpenCV_LIBS})