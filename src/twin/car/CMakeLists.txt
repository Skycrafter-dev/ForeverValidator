# Twin StadiumCar force kernel (M3/M4 forces): standalone build, does not touch the root CMakeLists.
#
#   cmake -S src/twin/car -B <build> -G Ninja [-DFV_BUILD_DIR=<non-IPO FV build with libforevervalidator_{core,native}.a>]
#   ninja -C <build>
#
# Sources: kernel src/twin/model/{forces.h, forces_io.h, impulse.h, vec3.h, car_force_consts.h, forces.cpp},
# FV adapter src/twin/fv/compile_car.cpp, tools tools/twin/twin_car_{record,check,bench}.cpp.
# twin_car_check and twin_car_bench are FV-free; twin_car_record links FV and wraps three FV symbols
# with GNU ld --wrap (needs -DFOREVERVALIDATOR_ENABLE_RELEASE_IPO=OFF for the FV build).
cmake_minimum_required(VERSION 3.20)
project(tw_car CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(TW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(FV_BUILD_DIR "" CACHE PATH "Out-of-tree non-IPO FV build holding libforevervalidator_{core,native}.a")
set(TW_ISA "native" CACHE STRING "-march for the kernel and bench (native | x86-64-v3 | x86-64)")

# Determinism (docs/twin/math.md §4): no contraction, no fast-math.
set(TW_FLAGS -O3 -g -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra)
set(TW_INC "${TW_REPO_ROOT}/src/twin/model" "${TW_REPO_ROOT}/src/twin/math" "${TW_REPO_ROOT}/src/twin/fv" "${TW_REPO_ROOT}")

add_library(tw_car STATIC "${TW_REPO_ROOT}/src/twin/model/forces.cpp")
target_include_directories(tw_car PUBLIC ${TW_INC})
target_compile_options(tw_car PRIVATE ${TW_FLAGS} -march=${TW_ISA})





if(FV_BUILD_DIR)
endif()


find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    set(TW_SRC "${TW_REPO_ROOT}/src/twin")
    set(TW_REF_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/twx_generated")
    set(TW_REF_GEN_INPUTS
        "${TW_SRC}/model/vec3.h" "${TW_SRC}/model/lane3.h" "${TW_SRC}/model/car_force_consts.h"
        "${TW_SRC}/model/impulse.h" "${TW_SRC}/model/forces_io.h" "${TW_SRC}/model/forces.h"
        "${TW_SRC}/model/tw_state.h" "${TW_SRC}/model/state_fields.inc" "${TW_SRC}/model/response.h"
        "${TW_SRC}/race/tw_race.h")
    add_custom_command(
        OUTPUT "${TW_REF_GEN_DIR}/twx_forces.h" "${TW_REF_GEN_DIR}/twx_widen.inc"
        COMMAND "${Python3_EXECUTABLE}" "${TW_REF_DIR}/twx_gen.py" "${TW_SRC}" "${TW_REF_GEN_DIR}"
        DEPENDS "${TW_REF_DIR}/twx_gen.py" ${TW_REF_GEN_INPUTS}
        VERBATIM)
    add_custom_target(tw_air_ref_generated DEPENDS "${TW_REF_GEN_DIR}/twx_forces.h" "${TW_REF_GEN_DIR}/twx_widen.inc")
    set(TW_AIR_REF_SOURCES "${TW_REF_DIR}/twx_ref_double.cpp" "${TW_REF_DIR}/twx_ref_sr.cpp"
        "${TW_REF_DIR}/twx_ref_rn0.cpp" "${TW_REF_DIR}/twx_ref_rn1.cpp" "${TW_REF_DIR}/twx_ref_rn2.cpp")
    set(TW_AIR_INC "${TW_REF_GEN_DIR}" "${TW_REF_DIR}" "${TW_SRC}/ref" "${TW_SRC}/api" "${TW_SRC}/step" "${TW_SRC}/race"
        "${TW_SRC}/model" "${TW_SRC}/world" "${TW_SRC}/collide_c" "${TW_SRC}/math" "${TW_SRC}/fv" "${TW_SRC}/collide"
        "${TW_SRC}/isa" "${TW_REPO_ROOT}")
    set_source_files_properties("${TW_REF_DIR}/twx_ref_rn1.cpp" PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast;-mfma")
    set_source_files_properties("${TW_REF_DIR}/twx_ref_rn2.cpp" PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=fast;-mfma;-fassociative-math;-fno-signed-zeros;-fno-trapping-math")
endif()
