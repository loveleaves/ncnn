
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

# Add source file to list, and add to special visual folder
function(ncnn_src_group ncnn_src_string folder)
    string(REPLACE " " ";" _ncnn_src_list ${ncnn_src_string})

    string(REGEX REPLACE "/" "\\\\" _target_folder "${folder}")

    foreach(_file IN LISTS ${_ncnn_src_list})
        source_group ("${_target_folder}" FILES "${_file}")
    endforeach ()
endfunction()

set(ncnn_SRCS
    allocator.cpp
    benchmark.cpp
    blob.cpp
    cpu.cpp
    datareader.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    mat_pixel_drawing.cpp
    modelbin.cpp
    option.cpp
    net.cpp
    simpleocv.cpp
    paramdict.cpp
)

ncnn_src_group(ncnn_SRCS "sources")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}")
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_tools.cmake)

# remove old
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)

# add notification
ncnn_add_declaration_header(layer_declaration.h ON)
ncnn_add_declaration_header(layer_registry.h ON)
ncnn_add_declaration_header(layer_type_enum.h ON)

macro(ncnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    message("WITH_LAYER_${name} = ${WITH_LAYER_${name}}")

    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp")

        # look for arch specific implementation and append source
        set(LAYER_ARCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.cpp)
        if(EXISTS ${LAYER_ARCH_SRC})
            set(WITH_LAYER_${name}_${NCNN_TARGET_ARCH} 1)
            list(APPEND ncnn_SRCS ${LAYER_ARCH_SRC})
        endif()

        # vulkan, TODO
    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        if(WITH_LAYER_${name}_arm)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_arm_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_arm_layer_creator},\n#else\n{${class}_arm_layer_creator},\n#endif\n")
        elseif(WITH_LAYER_${name}_x86)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_x86_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_x86_layer_creator},\n#else\n{${class}_x86_layer_creator},\n#endif\n")
        else()
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
        endif()
    else()
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()
set(__LAYER_TYPE_ENUM_INDEX 0)
# layer implementation
ncnn_add_layer(BinaryOp)
ncnn_add_layer(Concat)
ncnn_add_layer(Convolution)
ncnn_add_layer(ConvolutionDepthWise)
ncnn_add_layer(DeconvolutionDepthWise)
ncnn_add_layer(Dropout)
ncnn_add_layer(Eltwise)
ncnn_add_layer(InnerProduct)
ncnn_add_layer(Input)
ncnn_add_layer(Pooling)
ncnn_add_layer(ReLU)
ncnn_add_layer(Softmax)
ncnn_add_layer(Split)
ncnn_add_layer(Yolov3DetectionOutput)

# end of notification
ncnn_add_declaration_header(layer_declaration.h OFF)
ncnn_add_declaration_header(layer_registry.h OFF)
ncnn_add_declaration_header(layer_type_enum.h OFF)

if(NCNN_SHARED_LIB)
    add_library(ncnn SHARED ${ncnn_SRCS})
else()
    add_library(ncnn STATIC ${ncnn_SRCS})
endif()

# link math lib explicitly
target_link_libraries(ncnn PUBLIC m)

target_include_directories(ncnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/ncnn>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/layer>
)

# Debug 模式其他选项
if(CMAKE_BUILD_TYPE MATCHES "debug|Debug|DEBUG")
    # 启用 ASan
    if(NCNN_ASAN)
        message(STATUS "Enabling AddressSanitizer (ASan)")
        target_compile_options(ncnn PUBLIC -fsanitize=address)
        target_link_libraries(ncnn PUBLIC -fsanitize=address)
    endif()

    # 启用 lcov
    if(NCNN_COVERAGE)
        message(STATUS "Enabling line converage（lcov）")
        target_compile_options(ncnn PUBLIC -coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(ncnn PUBLIC -coverage -lgcov)
    endif()
endif()

install(TARGETS ncnn EXPORT ncnn
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES
    allocator.h
    benchmark.h
    blob.h
    cpu.h
    datareader.h
    layer.h
    layer_type.h
    mat.h
    modelbin.h
    option.h
    net.h
    simpleocv.h
    paramdict.h
    ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
    ${CMAKE_CURRENT_BINARY_DIR}/platform.h
    DESTINATION include
)
