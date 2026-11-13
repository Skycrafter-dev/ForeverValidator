# Twin contact response (M5a): standalone build, does not touch the root CMakeLists.
#
#   cmake -S src/twin/resp -B <build> -G Ninja [-DFV_BUILD_DIR=<non-IPO FV build with libforevervalidator_{core,native}.a>]
#   ninja -C <build>
#
# Sources: kernel src/twin/model/{response.h, response.cpp} (+ impulse.h, vec3.h, car_force_consts.h, tw_state.h),
# tools tools/twin/twin_resp_{record,check}.cpp. twin_resp_check is FV-free; twin_resp_record links FV and wraps
# four FV symbols with GNU ld --wrap (needs -DFOREVERVALIDATOR_ENABLE_RELEASE_IPO=OFF for the FV build).
cmake_minimum_required(VERSION 3.20)
project(tw_resp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(TW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(FV_BUILD_DIR "" CACHE PATH "Out-of-tree non-IPO FV build holding libforevervalidator_{core,native}.a")
set(TW_ISA "native" CACHE STRING "-march for the kernel and tools (native | x86-64-v3 | x86-64)")

# Determinism (docs/twin/math.md §4): no contraction, no fast-math.
set(TW_FLAGS -O2 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra)
set(TW_INC "${TW_REPO_ROOT}/src/twin/model" "${TW_REPO_ROOT}/src/twin/math" "${TW_REPO_ROOT}/src/twin/fv" "${TW_REPO_ROOT}")

add_library(tw_resp STATIC "${TW_REPO_ROOT}/src/twin/model/response.cpp")
target_include_directories(tw_resp PUBLIC ${TW_INC})
target_compile_options(tw_resp PRIVATE ${TW_FLAGS} -march=${TW_ISA})



if(FV_BUILD_DIR)
endif()
