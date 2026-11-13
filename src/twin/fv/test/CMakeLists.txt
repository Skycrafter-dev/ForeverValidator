# M2 world package: FV-linked acceptance check (standalone; the root build
# compiles fv/compile_world.cpp into forevervalidator_core and
# world/tw_world_pack.cpp into tw_world, src/twin/CMakeLists.txt).
#
#   cmake -S src/twin/fv/test -B <build> -G Ninja -DFV_BUILD_DIR=<non-IPO FV build>
#   ninja -C <build> tw_world_fvcheck
#   <build>/tw_world_fvcheck ~/tmuf/Packs [--out DIR] REPLAY...
#
# tw_world_fvcheck hooks four FV symbols with GNU ld --wrap; it needs an FV
# build configured with -DFOREVERVALIDATOR_ENABLE_RELEASE_IPO=OFF.
cmake_minimum_required(VERSION 3.20)
project(tw_world_fvcheck CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
get_filename_component(TW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(TW "${TW_REPO_ROOT}/src/twin")
set(FV_BUILD_DIR "" CACHE PATH "Out-of-tree non-IPO FV build holding libforevervalidator_{core,native}.a")
add_compile_options(-O2 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra)

# FV-free part: the package format, alone (proves it builds without FV headers).
add_library(tw_world_pack STATIC "${TW}/world/tw_world_pack.cpp" "${TW}/world/tw_world.cpp")
target_include_directories(tw_world_pack PUBLIC "${TW}/world" "${TW}/race" "${TW}/model")
add_executable(tw_world_pack_test tw_world_pack_test.cpp)
target_link_libraries(tw_world_pack_test PRIVATE tw_world_pack)

if(FV_BUILD_DIR)
    add_executable(tw_world_fvcheck tw_world_fvcheck.cpp
        "${TW}/fv/compile_world.cpp"
        "${TW}/fv/compile_car.cpp"
        "${TW}/world/tw_world_compile.cpp"
        "${TW_REPO_ROOT}/src/twin/common/twin_fv_state.cpp")
    target_include_directories(tw_world_fvcheck PRIVATE "${TW}/fv" "${TW}/math" "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/src"
        "${TW_REPO_ROOT}/include")
    target_compile_definitions(tw_world_fvcheck PRIVATE NOMINMAX OPENSSL_SUPPRESS_DEPRECATED)
    target_compile_options(tw_world_fvcheck PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
    target_link_libraries(tw_world_fvcheck PRIVATE tw_world_pack
        "${FV_BUILD_DIR}/libforevervalidator_native.a"
        "${FV_BUILD_DIR}/libforevervalidator_core.a"
        crypto z pthread)
    # FV core compiles its Vulkan backend unless configured with -DFOREVERVALIDATOR_ENABLE_VULKAN=OFF
    find_library(TW_VULKAN_LIB vulkan)
    if(TW_VULKAN_LIB)
        target_link_libraries(tw_world_fvcheck PRIVATE "${TW_VULKAN_LIB}")
    endif()
    target_link_options(tw_world_fvcheck PRIVATE
        "-Wl,--wrap=_ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection"
        "-Wl,--wrap=_ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver"
        "-Wl,--wrap=_ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj"
        "-Wl,--wrap=_ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus")
endif()
