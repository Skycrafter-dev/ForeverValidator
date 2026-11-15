# Twin model (state + car pack): standalone build, does not touch the root CMakeLists.
#   cmake -S src/twin/model -B build-model [-DFV_BUILD_DIR=<FV build with libforevervalidator_core.a>]
cmake_minimum_required(VERSION 3.21)
project(tw_model CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(TW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(TW_FLAGS -O2 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra)

add_library(tw_model STATIC tw_compile_car.cpp)
target_include_directories(tw_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(tw_model PRIVATE ${TW_FLAGS})

add_executable(tw_m1_test test/tw_m1_test.cpp)
target_include_directories(tw_m1_test PRIVATE "${TW_REPO_ROOT}")
target_compile_options(tw_m1_test PRIVATE ${TW_FLAGS})
target_link_libraries(tw_m1_test PRIVATE tw_model)

if(FV_BUILD_DIR)
  add_executable(tw_curve_fvcheck test/tw_curve_fvcheck.cpp)
  target_include_directories(tw_curve_fvcheck PRIVATE "${TW_REPO_ROOT}/src" "${TW_REPO_ROOT}/include")
  target_compile_options(tw_curve_fvcheck PRIVATE ${TW_FLAGS} -Wno-unused-parameter)
  target_compile_definitions(tw_curve_fvcheck PRIVATE NOMINMAX)
  target_link_libraries(tw_curve_fvcheck PRIVATE tw_model "${FV_BUILD_DIR}/libforevervalidator_core.a" crypto z)
endif()
