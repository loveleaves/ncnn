
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

macro(ncnn_add_declaration_header file_name beginning_of_file)
    string(TOUPPER ${file_name} upper_file_name)
    string(REPLACE "." "_" upper_file_name ${upper_file_name})

    message("beginning_of_file = ${beginning_of_file}")
    if(${beginning_of_file})
        set(NCNN_FILE_DECLARATION_HEADER "#ifndef ${upper_file_name}\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER}#define ${upper_file_name}\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER}\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER}/**\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER} * @file ${file_name}\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER} * @brief This file is automatically generated by CMake.\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER} * @note Do not edit directly unless you understand CMake and the project structure.\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER} * \n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER} * @version 1.0\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER} */\n")
        set(NCNN_FILE_DECLARATION_HEADER "${NCNN_FILE_DECLARATION_HEADER}\n")
    else()
        set(NCNN_FILE_DECLARATION_HEADER "\n#endif // ${upper_file_name}")
    endif()
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${file_name} ${NCNN_FILE_DECLARATION_HEADER})
endmacro()