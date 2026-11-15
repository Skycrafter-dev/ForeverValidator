# Fast exact ellipsoid-triangle test (src/twin/world/tw_exact_fast.*) and its checker/bench,
# standalone; does not touch the root build or src/twin/world's.
#
#   cmake -S src/twin/exact -B <build> -G Ninja [-DTW_EXACT_OPT=-O2]
#   ninja -C <build>
#   twin_exact_fast PACK REC [PACK REC ...]     (packs and recordings: ~/fv-twin-world-build)
cmake_minimum_required(VERSION 3.20)
project(tw_exact CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(TW_EXACT_OPT "-O2" CACHE STRING "optimisation level (the M3 stage builds with -O2)")

set(TW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(TW_REPO "${TW_ROOT}/../..")
set(TW_FLAGS ${TW_EXACT_OPT} -g -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra -march=native)

add_library(tw_exact_fast STATIC "${TW_ROOT}/world/tw_exact_fast.cpp")
target_include_directories(tw_exact_fast PUBLIC "${TW_ROOT}/world")
target_compile_options(tw_exact_fast PRIVATE ${TW_FLAGS})



# Candidate memo + world-space prefilter (src/twin/world/tw_car_query.*) in the stage's loop, against the stage.
set(TW_CQ_SOURCES "${TW_ROOT}/world/tw_car_query.cpp" "${TW_ROOT}/world/tw_exact_fast.cpp" "${TW_ROOT}/world/tw_world.cpp"
    "${TW_ROOT}/world/tw_accel.cpp" "${TW_ROOT}/world/tw_world_pack.cpp" "${TW_ROOT}/collide/tw_collide.cpp")
set(TW_CQ_INCLUDES "${TW_ROOT}/world" "${TW_ROOT}/collide" "${TW_ROOT}/step" "${TW_ROOT}/model" "${TW_ROOT}/race" "${TW_ROOT}/math")

