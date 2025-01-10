
macro(ncnn_get_file_define header_file define_name)
    get_filename_component(header_file_abs ${header_file} ABSOLUTE)

    # 创建一个临时测试源文件来检查宏是否定义
    file(WRITE "${CMAKE_BINARY_DIR}/check_macro.cpp" "
        #include \"${header_file_abs}\"
        int main() {
            #if defined(${define_name}) && ${define_name}
                return 0;
            #else
                error;
            #endif
        }
    ")

    try_compile(define_name ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/check_macro.cpp)
    if(NOT define_name)
        set(${define_name} 0)
    else()
        set(${define_name} 1)
    endif()
    file(REMOVE "${CMAKE_BINARY_DIR}/check_macro.cpp")
endmacro()
