# Contact atlas (gen2 S1, architect E): standalone build of the stage and its checks. The root build
# compiles atlas_build.cpp / atlas_stage.cpp into tw_collide (src/twin/CMakeLists.txt).
#
#   cmake -S src/twin/atlas -B <build> -G Ninja && ninja -C <build>
#   <build>/atlas_check PACK.twpack REC.twrec
cmake_minimum_required(VERSION 3.20)
project(tw_atlas CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(TW "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(TW_ATLAS_ISA "-march=native" CACHE STRING "ISA flags")
add_compile_options(-O3 ${TW_ATLAS_ISA} -ffp-contract=off -fno-fast-math -fno-strict-aliasing -fno-math-errno -Wall -Wextra)
set(TW_ATLAS_SRC
    atlas_build.cpp atlas_stage.cpp atlas_colcache.cpp tw_atlas_fb.cpp
    ${TW}/collide/tw_collide.cpp
    ${TW}/world/tw_world.cpp ${TW}/world/tw_accel.cpp ${TW}/world/tw_world_pack.cpp
    ${TW}/world/tw_exact_fast.cpp ${TW}/world/tw_car_query.cpp ${TW}/world/tw_patches.cpp
    ${TW}/collide_c/cc_stage.cpp ${TW}/collide_c/cc_memo.cpp)
add_library(tw_atlas_lib STATIC ${TW_ATLAS_SRC})
target_include_directories(tw_atlas_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${TW}/collide" "${TW}/collide_c" "${TW}/step" "${TW}/world" "${TW}/model"
                           "${TW}/race" "${TW}/math" "${TW}/common")
add_executable(atlas_check atlas_check.cpp)
target_link_libraries(atlas_check PRIVATE tw_atlas_lib)
add_executable(atlas_dump atlas_dump.cpp)
target_link_libraries(atlas_dump PRIVATE tw_atlas_lib)
