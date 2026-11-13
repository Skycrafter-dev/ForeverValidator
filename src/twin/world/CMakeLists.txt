# Twin static world + M2 collision bake-off (standalone; does not touch the root build).
#
#   cmake -S src/twin/world -B <build> -G Ninja -DFV_BUILD_DIR=<out-of-tree FV build>
#   ninja -C <build>
#
# tw_world (library) and tw_bench are FV-independent. tw_record links FV's
# static libraries and hooks three FV symbols with GNU ld --wrap; it needs an
# FV build configured with -DFOREVERVALIDATOR_ENABLE_RELEASE_IPO=OFF (LTO would
# inline across the wrapped call sites).
cmake_minimum_required(VERSION 3.20)
project(tw_world CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(FV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "ForeverValidator source root")
set(FV_BUILD_DIR "" CACHE PATH "Out-of-tree FV build holding libforevervalidator_{core,native}.a")

# Determinism flags (docs/twin/design-b.md §5): no contraction, no fast-math.
add_compile_options(-O2 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra)

add_library(tw_world STATIC tw_world.cpp tw_accel.cpp tw_blocks.cpp)
target_include_directories(tw_world PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(tw_bench tw_bench.cpp)
target_link_libraries(tw_bench PRIVATE tw_world)
# Same bench built for the host ISA (AVX-512 on the 9600X): shows what auto-vectorisation buys.
add_executable(tw_bench_native tw_bench.cpp tw_world.cpp tw_accel.cpp tw_blocks.cpp)
target_include_directories(tw_bench_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(tw_bench_native PRIVATE -march=native)
add_executable(tw_x1bench tw_x1bench.cpp tw_world.cpp tw_accel.cpp tw_blocks.cpp)
target_include_directories(tw_x1bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(tw_x1bench PRIVATE -mavx512f -mavx512vl -mavx512dq -mavx512bw -mavx2)
add_executable(tw_blockbench tw_blockbench.cpp tw_world.cpp tw_accel.cpp tw_blocks.cpp)
target_include_directories(tw_blockbench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(tw_blockbench PRIVATE -march=native)

add_executable(tw_voxbench tw_voxbench.cpp tw_voxels.cpp tw_world.cpp tw_accel.cpp tw_blocks.cpp)
target_include_directories(tw_voxbench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(tw_voxbench PRIVATE -march=native)
# same bench without the per-phase rdtsc reads (end-to-end wall time only)
add_executable(tw_voxbench_lean tw_voxbench.cpp tw_voxels.cpp tw_world.cpp tw_accel.cpp tw_blocks.cpp)
target_include_directories(tw_voxbench_lean PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(tw_voxbench_lean PRIVATE -march=native)
target_compile_definitions(tw_voxbench_lean PRIVATE TW_NOSPLIT)

# M3 collision stage (src/twin/collide/) checked against FV's recorded merged contacts
set(TW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_executable(tw_m3check tw_m3check.cpp tw_world.cpp tw_accel.cpp tw_world_pack.cpp tw_car_query.cpp tw_exact_fast.cpp tw_patches.cpp ${TW_ROOT}/collide/tw_collide.cpp ${TW_ROOT}/collide/tw_collide_ref.cpp
               ${TW_ROOT}/collide_c/cc_stage.cpp ${TW_ROOT}/collide_c/cc_memo.cpp)
target_include_directories(tw_m3check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_ROOT}/collide" "${TW_ROOT}/collide_c" "${TW_ROOT}/step" "${TW_ROOT}/model"
                           "${TW_ROOT}/race" "${TW_ROOT}/math")
target_compile_options(tw_m3check PRIVATE -march=native -O3)

if(FV_BUILD_DIR)
    add_executable(tw_record tw_record.cpp tw_world_compile.cpp ${TW_ROOT}/fv/compile_car_shapes.cpp ${TW_ROOT}/collide/tw_collide.cpp tw_car_query.cpp tw_exact_fast.cpp tw_patches.cpp tw_world_pack.cpp)
    target_include_directories(tw_record PRIVATE "${FV_SOURCE_DIR}/include" "${FV_SOURCE_DIR}/src" "${FV_SOURCE_DIR}"
                               "${TW_ROOT}/fv" "${TW_ROOT}/collide" "${TW_ROOT}/step" "${TW_ROOT}/model" "${TW_ROOT}/race" "${TW_ROOT}/math")
    target_compile_definitions(tw_record PRIVATE NOMINMAX OPENSSL_SUPPRESS_DEPRECATED)
    target_compile_options(tw_record PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
    target_link_libraries(tw_record PRIVATE tw_world
        "${FV_BUILD_DIR}/libforevervalidator_native.a"
        "${FV_BUILD_DIR}/libforevervalidator_core.a"
        crypto z pthread)
    find_library(TW_VULKAN_LIB vulkan)   # FV core built with its Vulkan backend (the default)
    if(TW_VULKAN_LIB)
        target_link_libraries(tw_record PRIVATE "${TW_VULKAN_LIB}")
    endif()
    target_link_options(tw_record PRIVATE
        "-Wl,--wrap=_ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj"
        "-Wl,--wrap=_ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection"
        "-Wl,--wrap=_ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver"
        "-Wl,--wrap=_ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus"
        "-Wl,--wrap=_Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer")
endif()
