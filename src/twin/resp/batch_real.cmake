# twin_batch_real (M10, real world packs + M3 collision): added to a root build
# without editing the root or src/twin CMakeLists:
#
#   cmake -S . -B <b> -G Ninja -DCMAKE_BUILD_TYPE=Release -DFOREVERVALIDATOR_BUILD_TWIN=ON \
#         -DFOREVERVALIDATOR_ENABLE_RELEASE_IPO=OFF \
#         -DCMAKE_PROJECT_ForeverValidator_INCLUDE=<repo>/src/twin/resp/batch_real.cmake
#   ninja -C <b> twin_batch_real
#
# The target is defined when the root directory has been processed (DEFER), so
# forevervalidator_core and the tw_* libraries exist. The tool and the batch
# runner headers compile with -march=native (TW_BATCH_REAL_ISA); the libraries
# keep FOREVERVALIDATOR_TWIN_ISA. Bits do not depend on either (math.md §4).
set(TW_BATCH_REAL_ISA "native" CACHE STRING "-march for twin_batch_real")

function(tw_batch_real_targets)
    if(NOT TARGET forevervalidator_core OR NOT TARGET tw_engine OR NOT TARGET tw_collide)
        message(WARNING "twin_batch_real: needs -DFOREVERVALIDATOR_BUILD_TWIN=ON")
        return()
    endif()
    set(R "${CMAKE_SOURCE_DIR}")
    set(T "${R}/src/twin")
    add_executable(twin_batch_real "${R}/tools/twin/twin_batch_real.cpp" "${T}/collide/tw_collide_lanes.cpp")   # + the batch-lane collision
    target_include_directories(twin_batch_real PRIVATE
        "${R}" "${R}/include" "${R}/src"
        "${T}/api" "${T}/step" "${T}/race" "${T}/model" "${T}/world" "${T}/math" "${T}/fv" "${T}/collide" "${T}/isa" "${T}/batch")
    target_link_libraries(twin_batch_real PRIVATE forevervalidator_native forevervalidator_core
        tw_engine tw_collide tw_race tw_car tw_world tw_model pthread)
    if(TARGET tw_isa)
        target_link_libraries(twin_batch_real PRIVATE tw_isa)   # ISA dispatch of the tick (src/twin/isa)
    endif()
    forevervalidator_configure_target(twin_batch_real)
    target_compile_options(twin_batch_real PRIVATE -O2 -g -ffp-contract=off -fno-fast-math -fno-strict-aliasing
        -fno-math-errno -march=${TW_BATCH_REAL_ISA} -Wno-unused-parameter)
endfunction()
cmake_language(DEFER CALL tw_batch_real_targets)
