set(BINARY ${CMAKE_PROJECT_NAME}_test)

include_directories(../src)
include_directories(../src/layer)
include_directories(../src/layer/${NCNN_TARGET_ARCH})

set(TEST_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(TEST_LLT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llt)
file(GLOB_RECURSE TEST_SOURCES "${TEST_LLT_DIR}/test_*.cpp")
file(GLOB_RECURSE TEST_COMMON_SOURCES
    "${TEST_COMMON_DIR}/*.c"
    "${TEST_COMMON_DIR}/*.cpp"
)
list(APPEND TEST_SOURCES ${TEST_COMMON_SOURCES})
include_directories(llt)
include_directories(common)

# 添加 googletest 子目录
add_subdirectory(thirdParty/googletest)

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

if(NOT OpenCV_FOUND)
    # 没有opencv跳过相关用例
    message(WARNING "OpenCV not found, test_squeezenet won't be tested")
    list(REMOVE_ITEM TEST_SOURCES "${TEST_LLT_DIR}/test_squeezenet.cpp")
endif()

message(STATUS "src: ${TEST_SOURCES}")
add_executable(${BINARY} ${TEST_SOURCES})
add_test(NAME ${BINARY} COMMAND ${BINARY})
# 链接src生成的lib库和gtest库

if(OpenCV_FOUND)
    target_include_directories(${BINARY} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(${BINARY} PUBLIC gtest ncnn ${OpenCV_LIBS})