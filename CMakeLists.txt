
if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/ncnn" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

project(ncnn)

cmake_minimum_required(VERSION 2.8.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug CACHE STRING "Choose the type of build" FORCE)
endif()

# Note: option会缓存在cmakefile中
option(NCNN_SHARED_LIB "shared library support" OFF)
option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_SIMPLEOCV "minimal opencv structure emulation" OFF)
option(NCNN_PIXEL_DRAWING "draw basic figure and text" ON)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_BUILD_TESTS "build tests" ON)
option(NCNN_THREADS "build with threads" OFF)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_ASAN "build for address sanitizer" OFF)
option(NCNN_COVERAGE "build for coverage" OFF)
option(NCNN_FORCE_INLINE "force inline some function" ON)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# gnu inline assembly in clang msvc does not work actually
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")))
    check_cxx_source_compiles("int main() { int a = 0; asm volatile(\"\" : \"=r\"(a) : \"0\"(a) : \"memory\"); return 0; }" NCNN_COMPILER_SUPPORT_GNU_INLINE_ASM)
    if(NCNN_COMPILER_SUPPORT_GNU_INLINE_ASM)
        option(NCNN_GNU_INLINE_ASM "optimize platform with gnu style inline assembly" ON)
    else()
        message(WARNING "The compiler does not support gnu style inline assembly. NCNN_GNU_INLINE_ASM will be OFF.")
    endif()
endif()

if((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm")
    OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
    OR (CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "(ARMV7|ARM64)")
    OR ((CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")) AND (${CMAKE_GENERATOR_PLATFORM} MATCHES "^(arm|arm64)")))
    set(NCNN_TARGET_ARCH arm)

    if(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64_32")
        set(NCNN_TARGET_ILP32 TRUE)
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND NOT NCNN_TARGET_ILP32)
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s, _a, _b; _s = vmlaq_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM_NEON)

        if(NCNN_COMPILER_SUPPORT_ARM_NEON)
            if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
                set(CMAKE_REQUIRED_FLAGS "/arch:VFPv4")
                check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _a; float16x4_t _s = vcvt_f16_f32(_a); return 0; }" NCNN_COMPILER_SUPPORT_ARM_VFPV4)

                unset(CMAKE_REQUIRED_FLAGS)
            else()
                set(CMAKE_REQUIRED_FLAGS "-mfpu=neon-vfpv4")
                check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _a; float16x4_t _s = vcvt_f16_f32(_a); return 0; }" NCNN_COMPILER_SUPPORT_ARM_VFPV4)

                if(NOT NCNN_COMPILER_SUPPORT_ARM_VFPV4)
                    set(CMAKE_REQUIRED_FLAGS "-mfpu=neon-vfpv4 -mfp16-format=ieee")
                    check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _a; float16x4_t _s = vcvt_f16_f32(_a); return 0; }" NCNN_COMPILER_SUPPORT_ARM_VFPV4_FP16)
                endif()

                unset(CMAKE_REQUIRED_FLAGS)
            endif()
        endif()

        if(NCNN_COMPILER_SUPPORT_ARM_VFPV4 OR NCNN_COMPILER_SUPPORT_ARM_VFPV4_FP16)
            option(NCNN_VFPV4 "optimize armv7 platform with vfpv4" ON)
        else()
            message(WARNING "The compiler does not support arm vfpv4. NCNN_VFPV4 will be OFF.")
        endif()
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8 OR NCNN_TARGET_ILP32)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.0")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _a; float16x4_t _s = vcvt_f16_f32(_a); return 0; }" NCNN_COMPILER_SUPPORT_ARM_VFPV4)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.2")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16x8_t _s, _a, _b; _s = vfmaq_f16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_FP16)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.2")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int32x4_t _s; int8x16_t _a, _b; _s = vdotq_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_DOTPROD)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.2")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s; float16x8_t _a, _b; _s = vfmlalq_low_f16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_FP16FML)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.4")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s; bfloat16x8_t _a, _b; _s = vcvt_f32_bf16(vcvt_bf16_f32(vbfmmlaq_f32(_s, _a, _b))); return 0; }" NCNN_COMPILER_SUPPORT_ARM84_BF16)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.4")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int32x4_t _s; int8x16_t _a, _b; _s = vmmlaq_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM84_I8MM)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat16_t _s, _a, _b; svbool_t bp; _s = svmla_f16_z(bp, _s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVE)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svint16_t _s; svint8_t _a, _b; _s = svmlslb_s16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVE2)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat32_t _s; svbfloat16_t _a, _b; _s = svbfmmla_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEBF16)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svint32_t _s; svint8_t _a, _b; _s = svmmla_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEI8MM)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat32_t _s, _a, _b; _s = svmmla_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEF32MM)

            unset(CMAKE_REQUIRED_FLAGS)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.0")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _a; float16x4_t _s = vcvt_f16_f32(_a); return 0; }" NCNN_COMPILER_SUPPORT_ARM_VFPV4)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.2 -march=armv8.2-a+fp16")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16x8_t _s, _a, _b; _s = vfmaq_f16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_FP16)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.2 -march=armv8.2-a+dotprod")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int32x4_t _s; int8x16_t _a, _b; _s = vdotq_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_DOTPROD)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.2 -march=armv8.2-a+fp16fml")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s; float16x8_t _a, _b; _s = vfmlalq_low_f16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_FP16FML)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.4 -march=armv8.4-a+bf16")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s; bfloat16x8_t _a, _b; _s = vcvt_f32_bf16(vcvt_bf16_f32(vbfmmlaq_f32(_s, _a, _b))); return 0; }" NCNN_COMPILER_SUPPORT_ARM84_BF16)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.4 -march=armv8.4-a+i8mm")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int32x4_t _s; int8x16_t _a, _b; _s = vmmlaq_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM84_I8MM)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6 -march=armv8.6-a+sve")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat16_t _s, _a, _b; svbool_t bp; _s = svmla_f16_z(bp, _s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVE)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6 -march=armv8.6-a+sve2")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svint16_t _s; svint8_t _a, _b; _s = svmlslb_s16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVE2)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6 -march=armv8.6-a+sve+bf16")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat32_t _s; svbfloat16_t _a, _b; _s = svbfmmla_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEBF16)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6 -march=armv8.6-a+sve+i8mm")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svint32_t _s; svint8_t _a, _b; _s = svmmla_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEI8MM)

            set(CMAKE_REQUIRED_FLAGS "/arch:armv8.6 -march=armv8.6-a+sve+f32mm")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat32_t _s, _a, _b; _s = svmmla_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEF32MM)

            unset(CMAKE_REQUIRED_FLAGS)
        else()
            set(CMAKE_REQUIRED_FLAGS "-march=armv8-a")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _a; float16x4_t _s = vcvt_f16_f32(_a); return 0; }" NCNN_COMPILER_SUPPORT_ARM_VFPV4)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+fp16")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16x8_t _s, _a, _b; _s = vfmaq_f16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_FP16)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+dotprod")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int32x4_t _s; int8x16_t _a, _b; _s = vdotq_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_DOTPROD)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+fp16fml")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s; float16x8_t _a, _b; _s = vfmlalq_low_f16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM82_FP16FML)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.4-a+bf16")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t _s; bfloat16x8_t _a, _b; _s = vcvt_f32_bf16(vcvt_bf16_f32(vbfmmlaq_f32(_s, _a, _b))); return 0; }" NCNN_COMPILER_SUPPORT_ARM84_BF16)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.4-a+i8mm")
            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int32x4_t _s; int8x16_t _a, _b; _s = vmmlaq_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM84_I8MM)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sve")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat16_t _s, _a, _b; svbool_t bp; _s = svmla_f16_z(bp, _s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVE)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sve2")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svint16_t _s; svint8_t _a, _b; _s = svmlslb_s16(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVE2)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sve+bf16")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat32_t _s; svbfloat16_t _a, _b; _s = svbfmmla_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEBF16)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sve+i8mm")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svint32_t _s; svint8_t _a, _b; _s = svmmla_s32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEI8MM)

            set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sve+f32mm")
            check_cxx_source_compiles("#include <arm_sve.h>\nint main() { svfloat32_t _s, _a, _b; _s = svmmla_f32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_ARM86_SVEF32MM)

            unset(CMAKE_REQUIRED_FLAGS)
        endif()

        if(NCNN_COMPILER_SUPPORT_ARM_VFPV4)
            option(NCNN_VFPV4 "optimize aarch64 platform with vfpv4" ON)
        else()
            message(WARNING "The compiler does not support arm vfpv4. NCNN_VFPV4 will be OFF.")
        endif()

        if(NCNN_COMPILER_SUPPORT_ARM82_FP16)
            option(NCNN_ARM82 "optimize aarch64 platform with armv8.2 fp16" ON)
            if(NCNN_COMPILER_SUPPORT_ARM82_DOTPROD)
                if(NCNN_ARM82)
                    option(NCNN_ARM82DOT "optimize aarch64 platform with armv8.2 dotprod" ON)
                endif()
            else()
                message(WARNING "The compiler does not support armv8.2 dotprod. NCNN_ARM82DOT will be OFF.")
            endif()
            if(NCNN_COMPILER_SUPPORT_ARM82_FP16FML)
                if(NCNN_ARM82)
                    option(NCNN_ARM82FP16FML "optimize aarch64 platform with armv8.2 fp16fml" ON)
                endif()
            else()
                message(WARNING "The compiler does not support armv8.2 fp16fml. NCNN_ARM82FP16FML will be OFF.")
            endif()
            if(NCNN_COMPILER_SUPPORT_ARM84_BF16)
                if(NCNN_ARM82DOT AND NCNN_ARM82FP16FML)
                    option(NCNN_ARM84BF16 "optimize aarch64 platform with armv8.4 bf16" ON)
                endif()
            else()
                message(WARNING "The compiler does not support armv8.4 bf16. NCNN_ARM86BF16 will be OFF.")
            endif()
            if(NCNN_COMPILER_SUPPORT_ARM84_I8MM)
                if(NCNN_ARM82DOT AND NCNN_ARM82FP16FML)
                    option(NCNN_ARM84I8MM "optimize aarch64 platform with armv8.4 i8mm" ON)
                endif()
            else()
                message(WARNING "The compiler does not support armv8.4 i8mm. NCNN_ARM84I8MM will be OFF.")
            endif()
            if(NCNN_COMPILER_SUPPORT_ARM86_SVE)
                if(NCNN_ARM84BF16 AND NCNN_ARM84I8MM)
                    option(NCNN_ARM86SVE "optimize aarch64 platform with armv8.6 sve" ON)
                    if(NCNN_COMPILER_SUPPORT_ARM86_SVE2)
                        if(NCNN_ARM86SVE)
                            option(NCNN_ARM86SVE2 "optimize aarch64 platform with armv8.6 sve2" ON)
                        endif()
                    else()
                        message(WARNING "The compiler does not support armv8.6 sve2. NCNN_ARM86SVE2 will be OFF.")
                    endif()
                    if(NCNN_COMPILER_SUPPORT_ARM86_SVEBF16)
                        if(NCNN_ARM86SVE)
                            option(NCNN_ARM86SVEBF16 "optimize aarch64 platform with armv8.6 sve bf16" ON)
                        endif()
                    else()
                        message(WARNING "The compiler does not support armv8.6 sve bf16. NCNN_ARM86SVEBF16 will be OFF.")
                    endif()
                    if(NCNN_COMPILER_SUPPORT_ARM86_SVEI8MM)
                        if(NCNN_ARM86SVE)
                            option(NCNN_ARM86SVEI8MM "optimize aarch64 platform with armv8.6 sve i8mm" ON)
                        endif()
                    else()
                        message(WARNING "The compiler does not support armv8.6 sve i8mm. NCNN_ARM86SVEI8MM will be OFF.")
                    endif()
                    if(NCNN_COMPILER_SUPPORT_ARM86_SVEF32MM)
                        if(NCNN_ARM86SVE)
                            option(NCNN_ARM86SVEF32MM "optimize aarch64 platform with armv8.6 sve f32mm" ON)
                        endif()
                    else()
                        message(WARNING "The compiler does not support armv8.6 sve f32mm. NCNN_ARM86SVEF32MM will be OFF.")
                    endif()
                endif()
            else()
                message(WARNING "The compiler does not support armv8.6 sve. NCNN_ARM86SVE will be OFF.")
            endif()
        else()
            message(WARNING "The compiler does not support armv8.2 fp16. NCNN_ARM82 will be OFF.")
        endif()
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    set(NCNN_TARGET_ARCH riscv)

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv")
        check_cxx_source_compiles("#include <riscv_vector.h>\nint main() { vfloat32m8_t _s, _w; float _v; size_t vl; _s = __riscv_vfmacc_vf_f32m8(_s, _v, _w, vl); vfloat32m1_t _x; vfloat32m1x2_t _xx = __riscv_vcreate_v_f32m1x2(_x, _x); return 0; }" NCNN_COMPILER_SUPPORT_RISCV_V)

        set(CMAKE_REQUIRED_FLAGS "-march=rv64gc_zfh -D__fp16=_Float16")
        check_cxx_source_compiles("int main() { __fp16 s, v; s = v * v; return 0; }" NCNN_COMPILER_SUPPORT_RISCV_ZFH)

        set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv_zfh_zvfh -D__fp16=_Float16")
        check_cxx_source_compiles("#include <riscv_vector.h>\nint main() { vfloat16m8_t _s, _w; __fp16 _v; size_t vl; _s = __riscv_vfmacc_vf_f16m8(_s, _v, _w, vl); return 0; }" NCNN_COMPILER_SUPPORT_RISCV_ZVFH)

        set(CMAKE_REQUIRED_FLAGS "-march=rv64gc_zfh_xtheadvector -D__fp16=_Float16")
        check_cxx_source_compiles("#include <riscv_vector.h>\nint main() { vfloat16m8_t _s, _w; __fp16 _v; size_t vl; _s = __riscv_vfmacc_vf_f16m8(_s, _v, _w, vl); vfloat32m1_t _x; vfloat32m1x2_t _xx = __riscv_vcreate_v_f32m1x2(_x, _x); return 0; }" NCNN_COMPILER_SUPPORT_RISCV_XTHEADVECTOR)

        unset(CMAKE_REQUIRED_FLAGS)

        if(NCNN_COMPILER_SUPPORT_RISCV_V OR NCNN_COMPILER_SUPPORT_RISCV_XTHEADVECTOR)
            option(NCNN_RVV "optimize risc-v platform with v extension" ON)
        else()
            message(WARNING "The compiler does not support risc-v v or xtheadvector extension. NCNN_RVV will be OFF.")
        endif()

        if(NCNN_COMPILER_SUPPORT_RISCV_XTHEADVECTOR)
            option(NCNN_XTHEADVECTOR "optimize risc-v platform with xtheadvector extension" ON)
        else()
            message(WARNING "The compiler does not support risc-v xtheadvector extension. NCNN_XTHEADVECTOR will be OFF.")
        endif()

        if(NCNN_COMPILER_SUPPORT_RISCV_ZFH)
            option(NCNN_ZFH "optimize risc-v platform with zfh extension" ON)
            if(NCNN_COMPILER_SUPPORT_RISCV_ZVFH OR NCNN_COMPILER_SUPPORT_RISCV_XTHEADVECTOR)
                if(NCNN_RVV AND NCNN_ZFH)
                    option(NCNN_ZVFH "optimize risc-v platform with zvfh extension" ON)
                endif()
            else()
                message(WARNING "The compiler does not support zvfh extension. NCNN_ZVFH will be OFF.")
            endif()
        else()
            message(WARNING "The compiler does not support risc-v zfh extension. NCNN_ZFH will be OFF.")
        endif()

    endif()

else()
    set(NCNN_TARGET_ARCH x86)

    option(NCNN_SSE2 "optimize x86 platform with sse2 extension" ON)

    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        check_cxx_compiler_flag("/arch:AVX" NCNN_COMPILER_SUPPORT_X86_AVX)
        check_cxx_compiler_flag("/arch:AVX" NCNN_COMPILER_SUPPORT_X86_FMA)
        check_cxx_compiler_flag("/arch:AVX" NCNN_COMPILER_SUPPORT_X86_XOP)
        check_cxx_compiler_flag("/arch:AVX" NCNN_COMPILER_SUPPORT_X86_F16C)
        check_cxx_compiler_flag("/arch:AVX2" NCNN_COMPILER_SUPPORT_X86_AVX2)
        check_cxx_compiler_flag("/arch:AVX512" NCNN_COMPILER_SUPPORT_X86_AVX512)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpwssd_avx_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpbssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT8)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpwsud_avx_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT16)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256 _a; __m128bh _s = _mm256_cvtneps_avx_pbh(_a); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_NE_CONVERT)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512i _s, _a, _b; _s = _mm512_dpwssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX512_VNNI)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256bh _s; __m512bh _a, _b; _s = _mm512_cvtneps_pbh(_mm512_dpbf16_ps(_mm512_cvtpbh_ps(_s), _a, _b)); return 0; }\n__m512i t(__m512 a) { __m256i _a = (__m256i)_mm512_cvtneps_pbh(a); return _mm512_inserti32x8(_mm512_castsi256_si512(_a), _a, 1); }" NCNN_COMPILER_SUPPORT_X86_AVX512_BF16)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512h _s, _a, _b; _s = _mm512_fmadd_ph(_s, _a, _b); __m512 _s2; _s2 = _mm512_cvtxph_ps(_mm512_cvtxps_ph(_s2)); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX512_FP16)

        unset(CMAKE_REQUIRED_FLAGS)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
        check_cxx_compiler_flag("-mrecip=none" NCNN_COMPILER_SUPPORT_X86_RECIP_NONE)

        check_cxx_compiler_flag("/arch:AVX" NCNN_COMPILER_SUPPORT_X86_AVX)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX -mfma -mf16c")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256 _s, _a, _b; _s = _mm256_fmadd_ps(_a, _b, _s); return 0; }" NCNN_COMPILER_SUPPORT_X86_FMA)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX -mxop")
        check_cxx_source_compiles("#include <x86intrin.h>\nint main() { __m128 _s, _a, _b; _s = _mm_maddd_epi16(_a, _b, _s); return 0; }" NCNN_COMPILER_SUPPORT_X86_XOP)

        check_cxx_compiler_flag("/arch:AVX -mf16c" NCNN_COMPILER_SUPPORT_X86_F16C)
        check_cxx_compiler_flag("/arch:AVX2 -mfma -mf16c" NCNN_COMPILER_SUPPORT_X86_AVX2)
        check_cxx_compiler_flag("/arch:AVX512 -mfma -mf16c -mavx512cd -mavx512bw -mavx512dq -mavx512vl" NCNN_COMPILER_SUPPORT_X86_AVX512)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 -mfma -mf16c -mavxvnni")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpwssd_avx_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 -mfma -mf16c -mavxvnni -mavxvnniint8")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpbssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT8)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 -mfma -mf16c -mavxvnni -mavxvnniint16")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpwsud_avx_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT16)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 -mfma -mf16c -mavxneconvert")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256 _a; __m128bh _s = _mm256_cvtneps_avx_pbh(_a); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_NE_CONVERT)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512 -mfma -mf16c -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512i _s, _a, _b; _s = _mm512_dpwssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX512_VNNI)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512 -mfma -mf16c -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512bf16")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256bh _s; __m512bh _a, _b; _s = _mm512_cvtneps_pbh(_mm512_dpbf16_ps(_mm512_cvtpbh_ps(_s), _a, _b)); return 0; }\n__m512i t(__m512 a) { __m256i _a = (__m256i)_mm512_cvtneps_pbh(a); return _mm512_inserti32x8(_mm512_castsi256_si512(_a), _a, 1); }" NCNN_COMPILER_SUPPORT_X86_AVX512_BF16)

        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512 -mfma -mf16c -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512fp16")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512h _s, _a, _b; _s = _mm512_fmadd_ph(_s, _a, _b); __m512 _s2; _s2 = _mm512_cvtxph_ps(_mm512_cvtxps_ph(_s2)); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX512_FP16)

        unset(CMAKE_REQUIRED_FLAGS)
    else()
        check_cxx_compiler_flag("-mrecip=none" NCNN_COMPILER_SUPPORT_X86_RECIP_NONE)

        check_cxx_compiler_flag("-mavx" NCNN_COMPILER_SUPPORT_X86_AVX)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256 _s, _a, _b; _s = _mm256_fmadd_ps(_a, _b, _s); return 0; }" NCNN_COMPILER_SUPPORT_X86_FMA)

        check_cxx_compiler_flag("-mxop" NCNN_COMPILER_SUPPORT_X86_XOP)
        check_cxx_compiler_flag("-mf16c" NCNN_COMPILER_SUPPORT_X86_F16C)
        check_cxx_compiler_flag("-mfma -mf16c -mavx2" NCNN_COMPILER_SUPPORT_X86_AVX2)
        check_cxx_compiler_flag("-mfma -mf16c -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl" NCNN_COMPILER_SUPPORT_X86_AVX512)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx2 -mavxvnni")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpwssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx2 -mavxvnni -mavxvnniint8")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpbssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT8)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx2 -mavxvnni -mavxvnniint16")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256i _s, _a, _b; _s = _mm256_dpwsud_avx_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT16)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx2 -mavxneconvert")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256 _a; __m128bh _s = _mm256_cvtneps_avx_pbh(_a); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX_NE_CONVERT)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512i _s, _a, _b; _s = _mm512_dpwssd_epi32(_s, _a, _b); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX512_VNNI)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512bf16")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m256bh _s; __m512bh _a, _b; _s = _mm512_cvtneps_pbh(_mm512_dpbf16_ps(_mm512_cvtpbh_ps(_s), _a, _b)); return 0; }\n__m512i t(__m512 a) { __m256i _a = (__m256i)_mm512_cvtneps_pbh(a); return _mm512_inserti32x8(_mm512_castsi256_si512(_a), _a, 1); }" NCNN_COMPILER_SUPPORT_X86_AVX512_BF16)

        set(CMAKE_REQUIRED_FLAGS "-mfma -mf16c -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512fp16")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512h _s, _a, _b; _s = _mm512_fmadd_ph(_s, _a, _b); __m512 _s2; _s2 = _mm512_cvtxph_ps(_mm512_cvtxps_ph(_s2)); return 0; }" NCNN_COMPILER_SUPPORT_X86_AVX512_FP16)

        unset(CMAKE_REQUIRED_FLAGS)
    endif()

    if(NOT CMAKE_SYSTEM_NAME MATCHES "Emscripten|WASI" AND NCNN_COMPILER_SUPPORT_X86_AVX)
        option(NCNN_AVX "optimize x86 platform with avx extension" ON)
        if(NCNN_COMPILER_SUPPORT_X86_FMA)
            if(NCNN_AVX)
                option(NCNN_FMA "optimize x86 platform with fma extension" ON)
            endif()
        else()
            message(WARNING "The compiler does not support fma extension. NCNN_FMA will be OFF.")
        endif()
        if(NCNN_COMPILER_SUPPORT_X86_XOP)
            if(NCNN_AVX)
                option(NCNN_XOP "optimize x86 platform with xop extension" ON)
            endif()
        else()
            message(WARNING "The compiler does not support xop extension. NCNN_XOP will be OFF.")
        endif()
        if(NCNN_COMPILER_SUPPORT_X86_F16C)
            if(NCNN_AVX)
                option(NCNN_F16C "optimize x86 platform with f16c extension" ON)
            endif()
        else()
            message(WARNING "The compiler does not support f16c extension. NCNN_F16C will be OFF.")
        endif()
        if(NCNN_COMPILER_SUPPORT_X86_AVX2)
            if(NCNN_AVX)
                option(NCNN_AVX2 "optimize x86 platform with avx2 extension" ON)
            endif()
            if(NCNN_COMPILER_SUPPORT_X86_AVX_VNNI)
                if(NCNN_AVX2)
                    option(NCNN_AVXVNNI "optimize x86 platform with avx vnni extension" ON)
                endif()
                if(NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT8)
                    if(NCNN_AVXVNNI)
                        option(NCNN_AVXVNNIINT8 "optimize x86 platform with avx vnni int8 extension" ON)
                    endif()
                else()
                    message(WARNING "The compiler does not support avx vnni int8 extension. NCNN_AVXVNNIINT8 will be OFF.")
                endif()
                if(NCNN_COMPILER_SUPPORT_X86_AVX_VNNI_INT16)
                    if(NCNN_AVXVNNI)
                        option(NCNN_AVXVNNIINT16 "optimize x86 platform with avx vnni int16 extension" ON)
                    endif()
                else()
                    message(WARNING "The compiler does not support avx vnni int16 extension. NCNN_AVXVNNIINT16 will be OFF.")
                endif()
            else()
                message(WARNING "The compiler does not support avx vnni extension. NCNN_AVXVNNI will be OFF.")
            endif()
            if(NCNN_COMPILER_SUPPORT_X86_AVX_NE_CONVERT)
                if(NCNN_AVX2)
                    option(NCNN_AVXNECONVERT "optimize x86 platform with avx ne convert extension" ON)
                endif()
            else()
                message(WARNING "The compiler does not support avx ne convert extension. NCNN_AVXNECONVERT will be OFF.")
            endif()
            if(NCNN_COMPILER_SUPPORT_X86_AVX512)
                if(NCNN_AVX2)
                    option(NCNN_AVX512 "optimize x86 platform with avx512 extension" ON)
                endif()
                if(NCNN_COMPILER_SUPPORT_X86_AVX512_VNNI)
                    if(NCNN_AVX512)
                        option(NCNN_AVX512VNNI "optimize x86 platform with avx512 vnni extension" ON)
                    endif()
                else()
                    message(WARNING "The compiler does not support avx512 vnni extension. NCNN_AVX512VNNI will be OFF.")
                endif()
                if(NCNN_COMPILER_SUPPORT_X86_AVX512_BF16)
                    if(NCNN_AVX512)
                        option(NCNN_AVX512BF16 "optimize x86 platform with avx512 bf16 extension" ON)
                    endif()
                else()
                    message(WARNING "The compiler does not support avx512 bf16 extension. NCNN_AVX512BF16 will be OFF.")
                endif()
                if(NCNN_COMPILER_SUPPORT_X86_AVX512_FP16)
                    if(NCNN_AVX512)
                        option(NCNN_AVX512FP16 "optimize x86 platform with avx512 fp16 extension" ON)
                    endif()
                else()
                    message(WARNING "The compiler does not support avx512 fp16 extension. NCNN_AVX512FP16 will be OFF.")
                endif()
            else()
                message(WARNING "The compiler does not support avx512 extension. NCNN_AVX512 will be OFF.")
            endif()
        else()
            message(WARNING "The compiler does not support avx2 extension. NCNN_AVX2 will be OFF.")
        endif()
    else()
        message(WARNING "The compiler does not support avx extension. NCNN_AVX will be OFF.")
    endif()
endif()

message(STATUS "NCNN_TARGET_ARCH = ${NCNN_TARGET_ARCH}")
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
if(NCNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()