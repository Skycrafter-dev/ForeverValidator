# Twin race layer (M6a): standalone build, does not touch the root CMakeLists.
#   cmake -S src/twin/race -B build-race [-DFV_BUILD_DIR=<non-IPO FV build with libforevervalidator_{core,native}.a>]
#
# tw_race_test is FV-independent. tw_race_fvcheck links FV and hooks seven FV
# symbols with GNU ld --wrap (FV itself unmodified); it needs an FV build with
# -DFOREVERVALIDATOR_ENABLE_RELEASE_IPO=OFF (LTO would inline the wrapped calls).
cmake_minimum_required(VERSION 3.21)
project(tw_race CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(TW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(TW_FLAGS -O2 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra)
set(TW_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model")

add_library(tw_race STATIC tw_controls.cpp)
target_include_directories(tw_race PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${TW_MODEL_DIR}")
target_compile_options(tw_race PRIVATE ${TW_FLAGS})

add_executable(tw_race_test test/tw_race_test.cpp)
target_compile_options(tw_race_test PRIVATE ${TW_FLAGS})
target_link_libraries(tw_race_test PRIVATE tw_race)

if(FV_BUILD_DIR)
  add_executable(tw_race_fvcheck test/tw_race_fvcheck.cpp
    "${TW_MODEL_DIR}/tw_compile_car.cpp"
    "${TW_REPO_ROOT}/src/twin/common/twin_fv_state.cpp")
  target_include_directories(tw_race_fvcheck PRIVATE "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/src" "${TW_REPO_ROOT}/include")
  target_compile_options(tw_race_fvcheck PRIVATE ${TW_FLAGS} -Wno-unused-parameter -Wno-deprecated-declarations)
  target_compile_definitions(tw_race_fvcheck PRIVATE NOMINMAX OPENSSL_SUPPRESS_DEPRECATED)
  target_link_libraries(tw_race_fvcheck PRIVATE tw_race
    "${FV_BUILD_DIR}/libforevervalidator_native.a"
    "${FV_BUILD_DIR}/libforevervalidator_core.a"
    crypto z pthread)
  target_link_options(tw_race_fvcheck PRIVATE
    "-Wl,--wrap=_ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection"
    "-Wl,--wrap=_ZNK10CHmsCorpus36RunAbsorbContactForCollisionResponseEP19CHmsPhysicalContact"
    "-Wl,--wrap=_ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver"
    "-Wl,--wrap=_ZN18ReplayPhysicsWorld4StepEv"
    "-Wl,--wrap=_ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus"
    "-Wl,--wrap=_ZN23ReplayVehicleSimulation7RespawnER17ReplayVehicleBody"
    "-Wl,--wrap=_ZN23ReplayVehicleSimulation11PrepareStepERK17ReplayControlTickR17ReplayVehicleBody")
endif()
