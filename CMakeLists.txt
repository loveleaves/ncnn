
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
# still has bug on simpleocv
option(NCNN_SIMPLEOCV "minimal opencv structure emulation" OFF)
option(NCNN_PIXEL_DRAWING "draw basic figure and text" ON)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
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

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    set(NCNN_TARGET_ARCH riscv)

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