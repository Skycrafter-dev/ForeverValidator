cmake_minimum_required(VERSION 3.21)

project(ForeverValidator LANGUAGES CXX)

option(FOREVERVALIDATOR_BUILD_CLI "Build the ForeverValidator CLI"
       ${PROJECT_IS_TOP_LEVEL})
option(FOREVERVALIDATOR_BUILD_TESTS "Build ForeverValidator public tests" OFF)
option(FOREVERVALIDATOR_BUILD_BENCHMARKS
       "Build ForeverValidator developer benchmarks" OFF)
option(FOREVERVALIDATOR_ENABLE_RELEASE_IPO
       "Enable supported GCC interprocedural optimization in Release builds"
       ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32 AND (NOT MINGW OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Windows builds require MinGW-w64 GCC")
endif()

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(ZLIB REQUIRED)

set(FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE OFF)
if(FOREVERVALIDATOR_ENABLE_RELEASE_IPO AND
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    include(CheckIPOSupported)
    check_ipo_supported(
        RESULT FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE
        OUTPUT FOREVERVALIDATOR_RELEASE_IPO_ERROR
        LANGUAGES CXX)
    if(NOT FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE)
        message(WARNING
            "GCC Release IPO is unavailable; using the portable non-IPO "
            "build: ${FOREVERVALIDATOR_RELEASE_IPO_ERROR}")
    endif()
elseif(FOREVERVALIDATOR_ENABLE_RELEASE_IPO)
    message(STATUS
        "Release IPO is currently enabled only for GCC; using the portable "
        "non-IPO build")
endif()

file(GLOB_RECURSE FOREVERVALIDATOR_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/format/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/validation/api/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/validation/evaluation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/validation/planning/*.cpp")
file(GLOB_RECURSE FOREVERVALIDATOR_JSON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/validation/serialization/*.cpp")
file(GLOB_RECURSE FOREVERVALIDATOR_NATIVE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/native/*.cpp")
file(GLOB_RECURSE FOREVERVALIDATOR_CLI_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/app/cli/*.cpp")

function(forevervalidator_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        NOMINMAX
        OPENSSL_SUPPRESS_DEPRECATED)
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wno-deprecated-declarations
        -ffp-contract=off
        -frandom-seed=forevervalidator
        "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.")
endfunction()

add_library(forevervalidator_core STATIC ${FOREVERVALIDATOR_CORE_SOURCES})
add_library(ForeverValidator::core ALIAS forevervalidator_core)
target_include_directories(forevervalidator_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(forevervalidator_core
    PUBLIC OpenSSL::Crypto ZLIB::ZLIB)
forevervalidator_configure_target(forevervalidator_core)

if(FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE)
    set_property(TARGET forevervalidator_core PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    target_compile_definitions(forevervalidator_core PRIVATE
        "$<$<CONFIG:Release>:FOREVERVALIDATOR_RELEASE_IPO_BUILD=1>")
    # Keep Reference routing and its observable mesh boundary as ordinary
    # externally visible code while the OptimizedCpu hot graph uses GCC IPA.
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/reference/reference_backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/physics/geometry/mesh_collision_queries.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/runtime/replay_simulation_runtime.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/runtime/replay_simulation_session.cpp"
        PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:Release>:-fno-lto>")
endif()

add_library(forevervalidator_json STATIC ${FOREVERVALIDATOR_JSON_SOURCES})
add_library(ForeverValidator::json ALIAS forevervalidator_json)
target_include_directories(forevervalidator_json
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(forevervalidator_json PUBLIC forevervalidator_core)
forevervalidator_configure_target(forevervalidator_json)

add_library(forevervalidator_native STATIC ${FOREVERVALIDATOR_NATIVE_SOURCES})
add_library(ForeverValidator::native ALIAS forevervalidator_native)
target_include_directories(forevervalidator_native
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(forevervalidator_native PUBLIC forevervalidator_core)
forevervalidator_configure_target(forevervalidator_native)

if(FOREVERVALIDATOR_BUILD_CLI)
    add_executable(forevervalidator ${FOREVERVALIDATOR_CLI_SOURCES})
    target_include_directories(forevervalidator PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(forevervalidator PRIVATE
        forevervalidator_native
        forevervalidator_json
        forevervalidator_core)
    if(WIN32)
        target_link_libraries(forevervalidator PRIVATE
            ws2_32
            gdi32
            crypt32)
    endif()
    if(MINGW)
        target_link_options(forevervalidator PRIVATE
            -static
            -Wl,--no-insert-timestamp)
    endif()
    forevervalidator_configure_target(forevervalidator)
    if(FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE)
        set_property(TARGET forevervalidator PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

if(FOREVERVALIDATOR_BUILD_BENCHMARKS)
    add_executable(optimized_cpu_benchmark
        tools/optimized_cpu_benchmark.cpp)
    target_include_directories(optimized_cpu_benchmark PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(optimized_cpu_benchmark PRIVATE
        forevervalidator_native
        forevervalidator_core)
    forevervalidator_configure_target(optimized_cpu_benchmark)
    if(FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE)
        set_property(TARGET optimized_cpu_benchmark PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(optimized_cpu_differential
        tools/optimized_cpu_differential.cpp)
    target_include_directories(optimized_cpu_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(optimized_cpu_differential PRIVATE
        forevervalidator_native
        forevervalidator_core)
    forevervalidator_configure_target(optimized_cpu_differential)

    add_executable(optimized_cpu_sphere_mesh_differential
        tools/optimized_cpu_sphere_mesh_differential.cpp)
    target_include_directories(optimized_cpu_sphere_mesh_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(optimized_cpu_sphere_mesh_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_sphere_mesh_differential)

    add_executable(optimized_cpu_ellipsoid_mesh_differential
        tools/optimized_cpu_ellipsoid_mesh_differential.cpp)
    target_include_directories(optimized_cpu_ellipsoid_mesh_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(optimized_cpu_ellipsoid_mesh_differential PRIVATE
        forevervalidator_core)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
        target_compile_definitions(
            optimized_cpu_ellipsoid_mesh_differential PRIVATE
            FOREVERVALIDATOR_HAS_GNU_LD_WRAP=1)
        target_link_options(optimized_cpu_ellipsoid_mesh_differential PRIVATE
            "LINKER:--wrap=_Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer")
    endif()
    forevervalidator_configure_target(
        optimized_cpu_ellipsoid_mesh_differential)

    add_executable(optimized_cpu_static_transform_cache_differential
        tools/optimized_cpu_static_transform_cache_differential.cpp)
    target_include_directories(
        optimized_cpu_static_transform_cache_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_static_transform_cache_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_static_transform_cache_differential)

    add_executable(optimized_cpu_static_triangle_sidecar_differential
        tools/optimized_cpu_static_triangle_sidecar_differential.cpp)
    target_include_directories(
        optimized_cpu_static_triangle_sidecar_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_static_triangle_sidecar_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_static_triangle_sidecar_differential)

    add_executable(optimized_cpu_static_scene_lifecycle_differential
        tools/optimized_cpu_static_scene_lifecycle_differential.cpp)
    target_include_directories(
        optimized_cpu_static_scene_lifecycle_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_static_scene_lifecycle_differential PRIVATE
        forevervalidator_native
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_static_scene_lifecycle_differential)

    add_executable(optimized_cpu_model3_vehicle_differential
        tools/optimized_cpu_model3_vehicle_differential.cpp)
    target_include_directories(
        optimized_cpu_model3_vehicle_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_model3_vehicle_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_model3_vehicle_differential)

    add_executable(optimized_cpu_static_bounds_overlap_differential
        tools/optimized_cpu_static_bounds_overlap_differential.cpp)
    target_include_directories(
        optimized_cpu_static_bounds_overlap_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_static_bounds_overlap_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_static_bounds_overlap_differential)
endif()
