# Standalone build of the twin math tests (not part of the root project yet).
#
#   cmake -S tests/twin/math -B <dir> -G Ninja -DTW_MATH_ISA=avx2 [-DCMAKE_CXX_COMPILER=clang++]
#   ninja -C <dir> && ctest --test-dir <dir> -L quick
#
# TW_MATH_ISA: sse2 | avx2 | avx512 (= -mavx512f -mavx512vl) | native
# TW_MATH_OPT: -O2 (default) | -O0 | -O3 ...
# TW_MATH_CONTRACT: off (required) | fast (negative control: -ffp-contract=fast -mfma,
#                   the canary and the hashes must then fail)
cmake_minimum_required(VERSION 3.16)
project(tw_math_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TW_MATH_ISA "sse2" CACHE STRING "sse2|avx2|avx512|native")
set(TW_MATH_CONTRACT "off" CACHE STRING "off|fast")
set(TW_MATH_OPT "-O2" CACHE STRING "optimisation flag (the hashes must not depend on it)")

get_filename_component(TW_MATH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/twin/math" ABSOLUTE)

set(tw_flags ${TW_MATH_OPT} -fno-fast-math -Wall -Wextra)
if(TW_MATH_CONTRACT STREQUAL "fast")
  list(APPEND tw_flags -ffp-contract=fast -mfma)
else()
  list(APPEND tw_flags -ffp-contract=off)
endif()
if(TW_MATH_ISA STREQUAL "sse2")
  list(APPEND tw_flags -msse2)
elseif(TW_MATH_ISA STREQUAL "avx2")
  list(APPEND tw_flags -mavx2)
elseif(TW_MATH_ISA STREQUAL "avx512")
  list(APPEND tw_flags -mavx512f -mavx512vl)
elseif(TW_MATH_ISA STREQUAL "native")
  list(APPEND tw_flags -march=native)
else()
  message(FATAL_ERROR "unknown TW_MATH_ISA ${TW_MATH_ISA}")
endif()

find_package(Threads REQUIRED)

add_executable(tw_math_tests tw_math_tests.cpp)
add_executable(tw_math_bench tw_math_bench.cpp)
add_library(tw_math_audit OBJECT tw_math_audit.cpp)  # object-code audit (audit_objects.sh)
foreach(t tw_math_tests tw_math_bench tw_math_audit)
  target_include_directories(${t} PRIVATE ${TW_MATH_SRC})
  target_compile_options(${t} PRIVATE ${tw_flags})
endforeach()
foreach(t tw_math_tests tw_math_bench)
  target_link_libraries(${t} PRIVATE Threads::Threads)
  if(WIN32)
    target_link_options(${t} PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endforeach()
# The bench measures the scalar form as scalar code.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(tw_math_bench PRIVATE -fno-vectorize -fno-slp-vectorize)
else()
  target_compile_options(tw_math_bench PRIVATE -fno-tree-vectorize)
endif()

enable_testing()
add_test(NAME tw_math_canary COMMAND tw_math_tests canary)
add_test(NAME tw_math_identity COMMAND tw_math_tests identity)
add_test(NAME tw_math_atan2 COMMAND tw_math_tests atan2 30000000)
add_test(NAME tw_math_equal_sin_small COMMAND tw_math_tests equal sin sincos_small.sin -0.785 0.785)
add_test(NAME tw_math_equal_cos_small COMMAND tw_math_tests equal cos sincos_small.cos -0.785 0.785)
add_test(NAME tw_math_equal_asin_small COMMAND tw_math_tests equal asin asin_small -0.5 0.5)
add_test(NAME tw_math_audit COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/audit_objects.sh ${CMAKE_BINARY_DIR})
set_tests_properties(tw_math_canary tw_math_identity tw_math_atan2 tw_math_audit tw_math_equal_sin_small
                     tw_math_equal_cos_small tw_math_equal_asin_small PROPERTIES LABELS quick TIMEOUT 60)
foreach(fn sin cos sincos.sin sincos.cos sincos_small.sin sincos_small.cos asin asin_small acos atan exp
           sqrt rsqrt trunc floor ceil rint
           fmod_pow2.32 to_i32_trunc_sat to_u32_trunc_wrap from_u32 from_i32)
  add_test(NAME tw_math_exhaustive_${fn} COMMAND tw_math_tests exhaustive ${fn})
  set_tests_properties(tw_math_exhaustive_${fn} PROPERTIES LABELS exhaustive TIMEOUT 60)
endforeach()
