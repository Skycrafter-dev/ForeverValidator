# Architect C collision prototype (standalone; does not touch the root build).
#
#   cmake -S src/twin/collide_c -B <build> -G Ninja
#   ninja -C <build>
#
# Reads B's tw_record streams (src/twin/world, *.twrec); reuses B's recording
# format, exact test and BVH4 (reference pipeline) from src/twin/world.
cmake_minimum_required(VERSION 3.20)
project(cc_collide CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(TW_WORLD "${CMAKE_CURRENT_SOURCE_DIR}/../world")
# Determinism flags (docs/twin/design-b.md §5, docs/twin/math.md): no contraction, no fast-math.
set(CC_OPT "-O3" CACHE STRING "optimisation level")
add_compile_options(${CC_OPT} -march=native -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra)

set(CC_SRC cc_bench.cpp cc_core.cpp cc_patches.cpp ${TW_WORLD}/tw_world.cpp ${TW_WORLD}/tw_accel.cpp)

add_executable(cc_bench ${CC_SRC})
target_include_directories(cc_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}")
target_compile_definitions(cc_bench PRIVATE CC_STATS=1)

add_executable(cc_bench_lean ${CC_SRC})
target_include_directories(cc_bench_lean PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}")
target_compile_definitions(cc_bench_lean PRIVATE CC_STATS=0)

add_executable(cc_bench_split ${CC_SRC})
target_include_directories(cc_bench_split PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}")
target_compile_definitions(cc_bench_split PRIVATE CC_STATS=0 CC_SPLIT=1)

add_executable(cc_bench_micro ${CC_SRC})
target_include_directories(cc_bench_micro PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}")
target_compile_definitions(cc_bench_micro PRIVATE CC_STATS=0 CC_MICROB=1)

add_executable(cc_bench_fine ${CC_SRC})
target_include_directories(cc_bench_fine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}")
target_compile_definitions(cc_bench_fine PRIVATE CC_STATS=0 CC_MICROB=1 CC_FINE=1)

# Block-cell candidate index (§9) against B's BVH4 on the world package
set(CC_GRID_SRC cc_gridbench.cpp cc_grid.cpp ${TW_WORLD}/tw_world_pack.cpp ${TW_WORLD}/tw_world.cpp ${TW_WORLD}/tw_accel.cpp)
add_executable(cc_gridbench ${CC_GRID_SRC})
target_include_directories(cc_gridbench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}" "${CMAKE_CURRENT_SOURCE_DIR}/../race" "${CMAKE_CURRENT_SOURCE_DIR}/../model")
add_executable(cc_gridbench_stats ${CC_GRID_SRC})
target_include_directories(cc_gridbench_stats PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_WORLD}" "${CMAKE_CURRENT_SOURCE_DIR}/../race" "${CMAKE_CURRENT_SOURCE_DIR}/../model")
target_compile_definitions(cc_gridbench_stats PRIVATE CC_GRID_STATS=1)

# Drop-in check: B's M3 stage (src/twin/collide) with its BVH4 replaced by the CellGrid, against B's own tw_m3check
set(TW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
file(READ "${TW_ROOT}/collide/tw_collide.h" CC_M3_H)
file(READ "${TW_ROOT}/collide/tw_collide.cpp" CC_M3_CPP)
string(REPLACE "#include \"tw_accel.h\"" "#include \"tw_accel.h\"\n#include \"cc_grid.h\"" CC_M3_H "${CC_M3_H}")
string(REPLACE "    Bvh4 bvh_;" "    cc::CellGrid bvh_;  // architect C: block-cell index in place of the BVH4" CC_M3_H "${CC_M3_H}")
string(REPLACE "    bvh_.Build(w);" "    (void)w;\n    bvh_.Build(view, cc::CellGrid::Options{});" CC_M3_CPP "${CC_M3_CPP}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/m3grid/tw_collide.h" "${CC_M3_H}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/m3grid/tw_collide.cpp" "${CC_M3_CPP}")
set(CC_M3_INC "${TW_WORLD}" "${TW_ROOT}/step" "${TW_ROOT}/model" "${TW_ROOT}/race" "${TW_ROOT}/math" "${TW_ROOT}/common")
add_executable(cc_m3check_bvh ${TW_WORLD}/tw_m3check.cpp ${TW_WORLD}/tw_world.cpp ${TW_WORLD}/tw_accel.cpp ${TW_WORLD}/tw_world_pack.cpp ${TW_WORLD}/tw_car_query.cpp
               ${TW_WORLD}/tw_exact_fast.cpp ${TW_WORLD}/tw_patches.cpp ${TW_ROOT}/collide/tw_collide.cpp)
target_include_directories(cc_m3check_bvh PRIVATE "${TW_ROOT}/collide" ${CC_M3_INC})
add_executable(cc_m3check_grid ${TW_WORLD}/tw_m3check.cpp ${TW_WORLD}/tw_world.cpp ${TW_WORLD}/tw_accel.cpp ${TW_WORLD}/tw_world_pack.cpp
               ${TW_WORLD}/tw_car_query.cpp ${TW_WORLD}/tw_exact_fast.cpp ${TW_WORLD}/tw_patches.cpp "${CMAKE_CURRENT_BINARY_DIR}/m3grid/tw_collide.cpp" cc_grid.cpp)
target_include_directories(cc_m3check_grid BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/m3grid" "${CMAKE_CURRENT_SOURCE_DIR}" ${CC_M3_INC})

# Rival exact collision stage (cc_stage.h) against B's stage on engine pose streams.
set(CC_STAGE_SRC cc_stagebench.cpp cc_stage.cpp cc_memo.cpp ${TW_ROOT}/collide/tw_collide.cpp ${TW_WORLD}/tw_car_query.cpp ${TW_WORLD}/tw_exact_fast.cpp ${TW_WORLD}/tw_patches.cpp
    ${TW_WORLD}/tw_world_pack.cpp ${TW_WORLD}/tw_world.cpp ${TW_WORLD}/tw_accel.cpp)
add_executable(cc_stagebench ${CC_STAGE_SRC})
target_include_directories(cc_stagebench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_ROOT}/collide" ${CC_M3_INC})
# the production baseline ISA (FOREVERVALIDATOR_TWIN_ISA=default: x86-64, SSE2)
add_executable(cc_stagebench_base ${CC_STAGE_SRC})
target_include_directories(cc_stagebench_base PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_ROOT}/collide" ${CC_M3_INC})
target_compile_options(cc_stagebench_base PRIVATE -march=x86-64 -mtune=generic)
add_executable(cc_stagebench_fine ${CC_STAGE_SRC})
target_include_directories(cc_stagebench_fine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_ROOT}/collide" ${CC_M3_INC})
target_compile_definitions(cc_stagebench_fine PRIVATE CC_FINE=1)

# The rival stage in the real all-twin engine (cc_enginebench.cpp): needs an FV twin build to link
# against (FOREVERVALIDATOR_TWIN on), e.g. -DCC_FVTWIN_BUILD=$HOME/fv-twin-wc-build/fvtwin. Twin flags
# (src/twin/CMakeLists.txt): baseline ISA as the world's model tick, and -march=native.
set(CC_FVTWIN_BUILD "" CACHE PATH "FV twin build directory to link the engine bench against")
if(CC_FVTWIN_BUILD)
    set(FV_ROOT "${TW_ROOT}/../..")
    set(CC_ENGINE_INC "${FV_ROOT}" "${FV_ROOT}/include" "${FV_ROOT}/src" "${TW_ROOT}/api" "${TW_ROOT}/step" "${TW_ROOT}/race" "${TW_ROOT}/model"
        "${TW_ROOT}/world" "${TW_ROOT}/math" "${TW_ROOT}/fv" "${TW_ROOT}/collide" "${TW_ROOT}/isa" "${CC_FVTWIN_BUILD}/src/twin/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}" "${FV_ROOT}/src/twin/common")
    set(CC_ENGINE_LIBS "${CC_FVTWIN_BUILD}/libforevervalidator_native.a" "${CC_FVTWIN_BUILD}/libforevervalidator_core.a"
        "${CC_FVTWIN_BUILD}/src/twin/libtw_model.a" crypto z "${CC_FVTWIN_BUILD}/src/twin/libtw_isa.a" "${CC_FVTWIN_BUILD}/src/twin/libtw_engine.a"
        "${CC_FVTWIN_BUILD}/src/twin/libtw_collide.a" "${CC_FVTWIN_BUILD}/src/twin/libtw_car.a" "${CC_FVTWIN_BUILD}/src/twin/libtw_world.a"
        "${CC_FVTWIN_BUILD}/src/twin/libtw_race.a" "${CC_FVTWIN_BUILD}/src/twin/libtw_model.a" pthread)
    # B's current sources under renamed namespaces (cc_bnative.cpp), via wrappers so the defines stay off
    # the other targets' copies
    set(CC_BNAT_SRC)
    foreach(f collide/tw_collide.cpp world/tw_car_query.cpp world/tw_exact_fast.cpp world/tw_patches.cpp collide_c/cc_stage.cpp collide_c/cc_memo.cpp)
        get_filename_component(nm "${f}" NAME_WE)
        set(w "${CMAKE_CURRENT_BINARY_DIR}/bnat/bnat_${nm}.cpp")
        file(WRITE "${w}.tmp" "#include \"${TW_ROOT}/${f}\"\n")
        configure_file("${w}.tmp" "${w}" COPYONLY)
        list(APPEND CC_BNAT_SRC "${w}")
    endforeach()
    list(APPEND CC_BNAT_SRC cc_bnative.cpp)
    option(CC_BNAT "build B's current sources into the engine bench (needs them to compile)" ON)
    if(NOT CC_BNAT)
        set(CC_BNAT_SRC cc_bnative_stub.cpp)
    endif()
    if(CC_BNAT)
        set_source_files_properties(${CC_BNAT_SRC} PROPERTIES COMPILE_DEFINITIONS "collide=bnat_collide;cq=bnat_cq;exact=bnat_exact;collide_c=bnat_collide_c")
    endif()
    foreach(v base native)
        add_executable(cc_enginebench_${v} cc_enginebench.cpp cc_stage.cpp cc_memo.cpp ${CC_BNAT_SRC})
        target_include_directories(cc_enginebench_${v} PRIVATE ${CC_ENGINE_INC})
        target_compile_definitions(cc_enginebench_${v} PRIVATE FOREVERVALIDATOR_HAS_HIP=0 FOREVERVALIDATOR_HAS_TWIN=1 FOREVERVALIDATOR_HAS_VULKAN=0
                                   NOMINMAX OPENSSL_SUPPRESS_DEPRECATED NDEBUG)
        target_compile_options(cc_enginebench_${v} PRIVATE -Wno-deprecated-declarations)
        target_link_libraries(cc_enginebench_${v} PRIVATE ${CC_ENGINE_LIBS})
    endforeach()
    target_compile_options(cc_enginebench_base PRIVATE -march=x86-64 -mtune=generic)
endif()

# B's tw_m3check with the double-precision reference (M3_REFD=1, M3_REFD_NOISE=1: B's stage + C's noise narrow)
add_executable(cc_m3check_refd ${TW_WORLD}/tw_m3check.cpp ${TW_WORLD}/tw_world.cpp ${TW_WORLD}/tw_accel.cpp ${TW_WORLD}/tw_world_pack.cpp
               ${TW_WORLD}/tw_car_query.cpp ${TW_WORLD}/tw_exact_fast.cpp ${TW_WORLD}/tw_patches.cpp ${TW_ROOT}/collide/tw_collide.cpp
               ${TW_ROOT}/collide/tw_collide_ref.cpp cc_stage.cpp cc_memo.cpp)
target_include_directories(cc_m3check_refd PRIVATE "${TW_ROOT}/collide" "${CMAKE_CURRENT_SOURCE_DIR}" ${CC_M3_INC})
