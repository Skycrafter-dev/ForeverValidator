cmake_minimum_required(VERSION 3.21)

project(ForeverValidator VERSION 0.2.5 LANGUAGES CXX)

option(FOREVERVALIDATOR_BUILD_CLI "Build the ForeverValidator CLI"
       ${PROJECT_IS_TOP_LEVEL})
option(FOREVERVALIDATOR_BUILD_TESTS "Build ForeverValidator public tests" OFF)
option(FOREVERVALIDATOR_BUILD_BENCHMARKS
       "Build ForeverValidator developer benchmarks" OFF)
option(FOREVERVALIDATOR_ENABLE_RELEASE_IPO
       "Enable supported GCC interprocedural optimization in Release builds"
       ON)
option(FOREVERVALIDATOR_ENABLE_CUDA
       "Build the optional NVIDIA CUDA simulation backend" OFF)
option(FOREVERVALIDATOR_ENABLE_HIP
       "Build the optional HIP simulation backend" OFF)
option(FOREVERVALIDATOR_ENABLE_VULKAN
       "Build the portable Vulkan Compute simulation backend" ON)
set(FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS "1" CACHE STRING
    "Parallel NVCC optimization jobs per CUDA translation unit; may change code generation (1 disables)")
set(FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT "" CACHE FILEPATH
    "Exact prebuilt cuda_search_executor object for release/CI reuse")

if(FOREVERVALIDATOR_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "FOREVERVALIDATOR_ENABLE_CUDA is ON, but no CUDA compiler "
            "was found")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT ("${FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS}"
            MATCHES "^[1-9][0-9]*$"))
        message(FATAL_ERROR
            "FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS must be a positive "
            "integer")
    endif()
    if(NOT ("${FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS}" STREQUAL "1"))
        include(CheckCompilerFlag)
        check_compiler_flag(
            CUDA
            "--split-compile=${FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS}"
            FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_SUPPORTED)
    endif()
endif()

if(FOREVERVALIDATOR_ENABLE_HIP)
    if(CMAKE_VERSION VERSION_LESS 3.30)
        message(FATAL_ERROR "HIP requires CMake 3.30 or newer")
    endif()
    enable_language(HIP)
    find_path(FOREVERVALIDATOR_HIP_INCLUDE_DIR hip/hip_runtime.h
        HINTS "$ENV{HIP_PATH}/include" "$ENV{ROCM_PATH}/include"
        REQUIRED)
    if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        find_package(CUDAToolkit REQUIRED)
    endif()
endif()

if(FOREVERVALIDATOR_ENABLE_VULKAN)
    find_package(Vulkan 1.2 REQUIRED)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
list(FILTER FOREVERVALIDATOR_CORE_SOURCES EXCLUDE REGEX
    "/simulation/backends/cuda/(cuda_driver_shim|cuda_session_specialization)\.cpp$")
list(FILTER FOREVERVALIDATOR_CORE_SOURCES EXCLUDE REGEX
    "/simulation/backends/hip/.*\.cpp$")
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
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" OR MSVC)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>"
            "$<$<COMPILE_LANGUAGE:CXX>:/fp:strict>")
    else()
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall>"
            "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>"
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>"
            "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>"
            "$<$<COMPILE_LANGUAGE:CXX>:-frandom-seed=forevervalidator>"
            "$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.>")
    endif()
endfunction()

add_library(forevervalidator_core STATIC ${FOREVERVALIDATOR_CORE_SOURCES})
add_library(ForeverValidator::core ALIAS forevervalidator_core)
target_include_directories(forevervalidator_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(forevervalidator_core
    PUBLIC OpenSSL::Crypto ZLIB::ZLIB)
forevervalidator_configure_target(forevervalidator_core)
target_sources(forevervalidator_core PRIVATE
    src/simulation/backends/hip/generated/hip_backend.cpp)
if(FOREVERVALIDATOR_ENABLE_HIP)
    file(GLOB FOREVERVALIDATOR_HIP_DEVICE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/hip/generated/*.cu")
    file(GLOB FOREVERVALIDATOR_HIP_HOST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/hip/generated/*.cpp")
    list(REMOVE_ITEM FOREVERVALIDATOR_HIP_HOST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/hip/generated/hip_backend.cpp")
    set_source_files_properties(${FOREVERVALIDATOR_HIP_DEVICE_SOURCES}
        PROPERTIES LANGUAGE HIP)
    target_sources(forevervalidator_core PRIVATE
        ${FOREVERVALIDATOR_HIP_DEVICE_SOURCES}
        ${FOREVERVALIDATOR_HIP_HOST_SOURCES}
        src/simulation/backends/hip/hip_state_bridge.cpp
        src/simulation/backends/hip/hip_session_specialization_adapter.cpp)
    target_include_directories(forevervalidator_core PRIVATE
        "${FOREVERVALIDATOR_HIP_INCLUDE_DIR}")
    target_compile_definitions(forevervalidator_core PUBLIC
        FOREVERVALIDATOR_HAS_HIP=1)
    if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        target_compile_options(forevervalidator_core PRIVATE
            "$<$<COMPILE_LANGUAGE:HIP>:--expt-relaxed-constexpr>"
            "$<$<COMPILE_LANGUAGE:HIP>:--fmad=false>"
            "$<$<COMPILE_LANGUAGE:HIP>:--prec-div=true>"
            "$<$<COMPILE_LANGUAGE:HIP>:--prec-sqrt=true>"
            "$<$<COMPILE_LANGUAGE:HIP>:--ftz=false>")
        target_compile_definitions(forevervalidator_core PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_NVIDIA__=1>")
        target_link_libraries(forevervalidator_core PUBLIC CUDA::cudart)
        if(NOT FOREVERVALIDATOR_ENABLE_CUDA)
            target_sources(forevervalidator_core PRIVATE
                src/simulation/backends/cuda/cuda_driver_shim.cpp)
        endif()
    else()
        target_compile_definitions(forevervalidator_core PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>")
        target_compile_options(forevervalidator_core PRIVATE
            "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>"
            "$<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>")
    endif()
else()
    target_compile_definitions(forevervalidator_core PUBLIC
        FOREVERVALIDATOR_HAS_HIP=0)
endif()
if(FOREVERVALIDATOR_ENABLE_VULKAN)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(FOREVERVALIDATOR_VULKAN_TIMELINE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_timeline.spv")
    set(FOREVERVALIDATOR_VULKAN_TIMELINE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_timeline_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_finish_refinement.spv")
    set(FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_finish_refinement_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_FINISH_PROBE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_finish_probe.spv")
    set(FOREVERVALIDATOR_VULKAN_FINISH_PROBE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_finish_probe_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_generate.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_generate_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_initialize.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_initialize_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_prepare.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_prepare_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_physics.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_physics_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_physics_steady_velocity_water.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_physics_steady_velocity_water_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_FORCE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_physics_steady_velocity_water_force.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_FORCE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_physics_steady_velocity_water_force_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_DETECT_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_physics_steady_velocity_water_detect.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_DETECT_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_physics_steady_velocity_water_detect_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_RESPOND_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_physics_steady_velocity_water_respond.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_RESPOND_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_physics_steady_velocity_water_respond_spirv.h")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_SPIRV
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/vulkan/vulkan_search_evaluate.spv")
    set(FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_vulkan_search_evaluate_spirv.h")
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_TIMELINE_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_TIMELINE_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_TIMELINE_HEADER}"
                --symbol ForeverValidatorVulkanTimelineSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_TIMELINE_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_HEADER}"
                --symbol ForeverValidatorVulkanFinishRefinementSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_FINISH_PROBE_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_FINISH_PROBE_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_FINISH_PROBE_HEADER}"
                --symbol ForeverValidatorVulkanFinishProbeSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_FINISH_PROBE_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_HEADER}"
                --symbol ForeverValidatorVulkanSearchGenerateSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_HEADER}"
                --symbol ForeverValidatorVulkanSearchInitializeSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_HEADER}"
                --symbol ForeverValidatorVulkanSearchPrepareSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_HEADER}"
                --symbol ForeverValidatorVulkanSearchEvaluateSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_HEADER}"
                --symbol ForeverValidatorVulkanSearchPhysicsSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_SPIRV}"
                --output "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_HEADER}"
                --symbol ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterSpirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    foreach(PHASE IN ITEMS FORCE DETECT RESPOND)
        add_custom_command(
            OUTPUT "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_${PHASE}_HEADER}"
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                    --input "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_${PHASE}_SPIRV}"
                    --output "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_${PHASE}_HEADER}"
                    --symbol "ForeverValidatorVulkanSearchPhysicsSteadyVelocityWater${PHASE}Spirv"
            DEPENDS
                    "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_${PHASE}_SPIRV}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
            VERBATIM)
    endforeach()
    add_custom_target(forevervalidator_vulkan_timeline_spirv
        DEPENDS
                "${FOREVERVALIDATOR_VULKAN_TIMELINE_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_FINISH_REFINEMENT_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_FINISH_PROBE_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_GENERATE_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_INITIALIZE_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PREPARE_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_FORCE_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_DETECT_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_PHYSICS_STEADY_VELOCITY_WATER_RESPOND_HEADER}"
                "${FOREVERVALIDATOR_VULKAN_SEARCH_EVALUATE_HEADER}")
    add_dependencies(forevervalidator_core
        forevervalidator_vulkan_timeline_spirv)
    target_include_directories(forevervalidator_core PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(forevervalidator_core PUBLIC Vulkan::Vulkan)
    target_compile_definitions(forevervalidator_core PUBLIC
        FOREVERVALIDATOR_HAS_VULKAN=1)
    message(STATUS "ForeverValidator Vulkan Compute backend enabled")
else()
    target_compile_definitions(forevervalidator_core PUBLIC
        FOREVERVALIDATOR_HAS_VULKAN=0)
    message(STATUS "ForeverValidator Vulkan Compute backend disabled")
endif()
if(FOREVERVALIDATOR_ENABLE_CUDA)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(FOREVERVALIDATOR_CUDA_SEARCH_LTO_IR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_cuda_search.ltoir")
    set(FOREVERVALIDATOR_CUDA_SEARCH_LTO_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/forevervalidator_cuda_search_lto_ir.h")
    file(GLOB_RECURSE FOREVERVALIDATOR_CUDA_SPECIALIZATION_HEADERS
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/*.cuh")
    set(FOREVERVALIDATOR_CUDA_COMPILER_COMMAND
        "${CMAKE_CUDA_COMPILER}")
    if(CMAKE_CUDA_COMPILER_LAUNCHER)
        set(FOREVERVALIDATOR_CUDA_COMPILER_COMMAND
            ${CMAKE_CUDA_COMPILER_LAUNCHER}
            "${CMAKE_CUDA_COMPILER}")
    endif()
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_IR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND ${FOREVERVALIDATOR_CUDA_COMPILER_COMMAND}
                -O3 -DNDEBUG -std=c++17 -arch=sm_75
                --expt-relaxed-constexpr
                --fmad=false --prec-div=true --prec-sqrt=true --ftz=false
                -DFOREVERVALIDATOR_HAS_CUDA=1
                -DFOREVERVALIDATOR_RELEASE_IPO_BUILD=1
                -DFOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO=1
                -DNOMINMAX -DOPENSSL_SUPPRESS_DEPRECATED
                "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
                -dlto -ltoir
                "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_search_executor.cu"
                -o "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_IR}"
        DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_search_executor.cu"
                ${FOREVERVALIDATOR_CUDA_SPECIALIZATION_HEADERS}
        VERBATIM)
    add_custom_command(
        OUTPUT "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_HEADER}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
                --input "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_IR}"
                --output "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_HEADER}"
                --symbol ForeverValidatorCudaSearchLtoIr
        DEPENDS
                "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_IR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
        VERBATIM)
    add_custom_target(forevervalidator_cuda_search_lto
        DEPENDS "${FOREVERVALIDATOR_CUDA_SEARCH_LTO_HEADER}")
    add_dependencies(forevervalidator_core
        forevervalidator_cuda_search_lto)
    target_sources(forevervalidator_core PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_backend.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_arithmetic_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_dynamics_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_collision_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_physics_step_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_race_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_vehicle_prefix_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_vehicle_force_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_stunt_certification.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_scene_storage.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_driver_shim.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_session_specialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_timeline_executor.cu")
    if(FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT)
        if(NOT EXISTS "${FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT}")
            message(FATAL_ERROR
                "FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT does not exist: "
                "${FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT}")
        endif()
        set_source_files_properties(
            "${FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT}"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(forevervalidator_core PRIVATE
            "${FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT}")
        message(STATUS
            "ForeverValidator using prebuilt CUDA search object: "
            "${FOREVERVALIDATOR_CUDA_SEARCH_PREBUILT_OBJECT}")
    else()
        target_sources(forevervalidator_core PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/backends/cuda/cuda_search_executor.cu")
    endif()
    target_include_directories(forevervalidator_core PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(forevervalidator_core PUBLIC
        CUDA::cudart CUDA::nvrtc CUDA::nvJitLink)
    target_compile_definitions(forevervalidator_core PUBLIC
        FOREVERVALIDATOR_HAS_CUDA=1)
    target_compile_options(forevervalidator_core PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>")
    if(FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_SUPPORTED AND
       NOT ("${FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS}" STREQUAL "1"))
        target_compile_options(forevervalidator_core PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:--split-compile=${FOREVERVALIDATOR_CUDA_SPLIT_COMPILE_JOBS}>")
    endif()
    set_target_properties(forevervalidator_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set_property(TARGET forevervalidator_core PROPERTY
                CUDA_ARCHITECTURES native)
        else()
            set_property(TARGET forevervalidator_core PROPERTY
                CUDA_ARCHITECTURES "75-real;80-real;86")
        endif()
    endif()
    message(STATUS "ForeverValidator CUDA backend enabled")
else()
    message(STATUS
        "ForeverValidator CUDA backend disabled "
        "(set FOREVERVALIDATOR_ENABLE_CUDA=ON to enable)")
endif()

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
    enable_testing()

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

    if(FOREVERVALIDATOR_ENABLE_VULKAN)
        add_executable(vulkan_search_benchmark
            tools/cuda_search_benchmark.cpp)
        target_include_directories(
            vulkan_search_benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            vulkan_search_benchmark PRIVATE
            forevervalidator_native
            forevervalidator_core)
        target_compile_definitions(
            vulkan_search_benchmark PRIVATE
            FOREVERVALIDATOR_SEARCH_BENCHMARK_VULKAN=1)
        forevervalidator_configure_target(vulkan_search_benchmark)
    endif()

    if(FOREVERVALIDATOR_ENABLE_CUDA)

        add_executable(cuda_backend_benchmark
            tools/cuda_backend_benchmark.cpp)
        target_include_directories(
            cuda_backend_benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            cuda_backend_benchmark PRIVATE
            forevervalidator_native
            forevervalidator_core)
        forevervalidator_configure_target(
            cuda_backend_benchmark)

        add_executable(cuda_search_benchmark
            tools/cuda_search_benchmark.cpp)
        target_include_directories(
            cuda_search_benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            cuda_search_benchmark PRIVATE
            forevervalidator_native
            forevervalidator_core)
        forevervalidator_configure_target(
            cuda_search_benchmark)

        add_executable(cuda_modifier_event_ops_benchmark
            tools/cuda_modifier_event_ops_benchmark.cpp)
        target_include_directories(
            cuda_modifier_event_ops_benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        forevervalidator_configure_target(
            cuda_modifier_event_ops_benchmark)

        add_executable(cuda_replay_parity
            tools/cuda_replay_parity.cpp)
        target_include_directories(
            cuda_replay_parity PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            cuda_replay_parity PRIVATE
            forevervalidator_native
            forevervalidator_core)
        forevervalidator_configure_target(
            cuda_replay_parity)
    endif()

    if(FOREVERVALIDATOR_ENABLE_VULKAN)
        add_executable(vulkan_replay_parity
            tools/vulkan_replay_parity.cpp)
        target_include_directories(
            vulkan_replay_parity PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            vulkan_replay_parity PRIVATE
            forevervalidator_native
            forevervalidator_core)
        forevervalidator_configure_target(
            vulkan_replay_parity)
    endif()

    add_executable(optimized_cpu_ellipsoid_mesh_kernel_benchmark
        tools/optimized_cpu_ellipsoid_mesh_kernel_benchmark.cpp)
    target_include_directories(
        optimized_cpu_ellipsoid_mesh_kernel_benchmark PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_ellipsoid_mesh_kernel_benchmark PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_ellipsoid_mesh_kernel_benchmark)
    if(FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE)
        set_property(
            TARGET optimized_cpu_ellipsoid_mesh_kernel_benchmark
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(optimized_cpu_compiled_tuning_curve_benchmark
        tools/optimized_cpu_compiled_tuning_curve_benchmark.cpp)
    target_include_directories(
        optimized_cpu_compiled_tuning_curve_benchmark PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_compiled_tuning_curve_benchmark PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_compiled_tuning_curve_benchmark)
    if(FOREVERVALIDATOR_RELEASE_IPO_AVAILABLE)
        set_property(
            TARGET optimized_cpu_compiled_tuning_curve_benchmark
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
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

    add_executable(optimized_cpu_vehicle_forces_differential
        tools/optimized_cpu_vehicle_forces_differential.cpp)
    target_include_directories(
        optimized_cpu_vehicle_forces_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_vehicle_forces_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_vehicle_forces_differential)

    add_executable(optimized_cpu_vehicle_collision_bounds_plan_differential
        tools/optimized_cpu_vehicle_collision_bounds_plan_differential.cpp)
    target_include_directories(
        optimized_cpu_vehicle_collision_bounds_plan_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_vehicle_collision_bounds_plan_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_vehicle_collision_bounds_plan_differential)

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

    add_executable(optimized_cpu_static_bvh_differential
        tools/optimized_cpu_static_bvh_differential.cpp)
    target_include_directories(
        optimized_cpu_static_bvh_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_static_bvh_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_static_bvh_differential)

    add_executable(optimized_cpu_ellipsoid_mesh_packet_differential
        tools/optimized_cpu_ellipsoid_mesh_packet_differential.cpp)
    target_include_directories(
        optimized_cpu_ellipsoid_mesh_packet_differential PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        optimized_cpu_ellipsoid_mesh_packet_differential PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        optimized_cpu_ellipsoid_mesh_packet_differential)

    add_test(NAME optimized_cpu_sphere_mesh
        COMMAND optimized_cpu_sphere_mesh_differential)
    add_test(NAME optimized_cpu_ellipsoid_mesh
        COMMAND optimized_cpu_ellipsoid_mesh_differential)
    add_test(NAME optimized_cpu_static_transform_cache
        COMMAND optimized_cpu_static_transform_cache_differential)
    add_test(NAME optimized_cpu_static_triangle_sidecar
        COMMAND optimized_cpu_static_triangle_sidecar_differential)
    add_test(NAME optimized_cpu_vehicle_forces
        COMMAND optimized_cpu_vehicle_forces_differential)
    add_test(NAME optimized_cpu_vehicle_collision_bounds_plan
        COMMAND optimized_cpu_vehicle_collision_bounds_plan_differential)
    add_test(NAME optimized_cpu_static_bounds_overlap
        COMMAND optimized_cpu_static_bounds_overlap_differential)
    add_test(NAME optimized_cpu_static_bvh
        COMMAND optimized_cpu_static_bvh_differential)
    add_test(NAME optimized_cpu_ellipsoid_mesh_packet
        COMMAND optimized_cpu_ellipsoid_mesh_packet_differential)
endif()

if(FOREVERVALIDATOR_BUILD_TESTS)
    enable_testing()

    add_executable(forevervalidator-custom-volume-tests tests/cuda_custom_volume_tests.cpp)
    target_include_directories(forevervalidator-custom-volume-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    forevervalidator_configure_target(forevervalidator-custom-volume-tests)
    add_test(NAME forevervalidator-custom-volume COMMAND forevervalidator-custom-volume-tests)

    add_executable(forevervalidator-search-event-capacity-tests
        tests/search_event_capacity_tests.cpp)
    add_executable(forevervalidator-search-segments-tests
        tests/cuda_search_segments_tests.cpp)
    target_include_directories(forevervalidator-search-segments-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    forevervalidator_configure_target(forevervalidator-search-segments-tests)
    add_test(NAME forevervalidator-search-segments
             COMMAND forevervalidator-search-segments-tests)
    target_include_directories(forevervalidator-search-event-capacity-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    forevervalidator_configure_target(forevervalidator-search-event-capacity-tests)
    add_test(NAME forevervalidator-search-event-capacity
             COMMAND forevervalidator-search-event-capacity-tests)

    add_executable(forevervalidator-race-camera-tests
        tests/race_camera_tests.cpp)
    target_include_directories(forevervalidator-race-camera-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(forevervalidator-race-camera-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(forevervalidator-race-camera-tests)
    add_test(NAME forevervalidator-race-camera
             COMMAND forevervalidator-race-camera-tests)

    add_executable(forevervalidator-race-camera-pack-tests
        tests/race_camera_pack_tests.cpp)
    target_include_directories(forevervalidator-race-camera-pack-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(forevervalidator-race-camera-pack-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(forevervalidator-race-camera-pack-tests)
    add_test(NAME forevervalidator-race-camera-pack
             COMMAND forevervalidator-race-camera-pack-tests)

    add_executable(forevervalidator-finish-time-tests
        tests/replay_finish_time_estimator_tests.cpp)
    target_include_directories(forevervalidator-finish-time-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(forevervalidator-finish-time-tests PRIVATE
        forevervalidator_json
        forevervalidator_core)
    forevervalidator_configure_target(forevervalidator-finish-time-tests)
    add_test(NAME forevervalidator-finish-time
             COMMAND forevervalidator-finish-time-tests)

    add_executable(forevervalidator-backend-routing-tests
        tests/backend_routing_tests.cpp)
    target_include_directories(forevervalidator-backend-routing-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(forevervalidator-backend-routing-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(forevervalidator-backend-routing-tests)
    add_test(NAME forevervalidator-backend-routing
             COMMAND forevervalidator-backend-routing-tests)

    add_executable(forevervalidator-cuda-state-layout-tests
        tests/cuda_state_layout_tests.cpp)
    target_include_directories(
        forevervalidator-cuda-state-layout-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        forevervalidator-cuda-state-layout-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        forevervalidator-cuda-state-layout-tests)
    add_test(NAME forevervalidator-cuda-state-layout
             COMMAND forevervalidator-cuda-state-layout-tests)

    add_executable(forevervalidator-cuda-candidate-events-tests
        tests/cuda_candidate_events_tests.cpp)
    target_include_directories(
        forevervalidator-cuda-candidate-events-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    forevervalidator_configure_target(
        forevervalidator-cuda-candidate-events-tests)
    add_test(NAME forevervalidator-cuda-candidate-events
             COMMAND forevervalidator-cuda-candidate-events-tests)

    add_executable(forevervalidator-cuda-static-configuration-tests
        tests/cuda_static_configuration_tests.cpp)
    target_include_directories(
        forevervalidator-cuda-static-configuration-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        forevervalidator-cuda-static-configuration-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        forevervalidator-cuda-static-configuration-tests)
    add_test(NAME forevervalidator-cuda-static-configuration
             COMMAND forevervalidator-cuda-static-configuration-tests)

    if(FOREVERVALIDATOR_ENABLE_VULKAN)
        add_executable(forevervalidator-vulkan-timeline-tests
            tests/vulkan_timeline_executor_tests.cpp)
        target_include_directories(
            forevervalidator-vulkan-timeline-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-vulkan-timeline-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-vulkan-timeline-tests)
        add_test(NAME forevervalidator-vulkan-timeline
                 COMMAND forevervalidator-vulkan-timeline-tests)
    endif()

    if(FOREVERVALIDATOR_ENABLE_HIP)
        add_executable(forevervalidator-hip-arithmetic-tests
            tests/hip_arithmetic_tests.cpp)
        target_include_directories(
            forevervalidator-hip-arithmetic-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-hip-arithmetic-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-hip-arithmetic-tests)
        add_test(NAME forevervalidator-hip-arithmetic
            COMMAND forevervalidator-hip-arithmetic-tests)
        if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
            add_test(NAME forevervalidator-hip-unavailable
                COMMAND forevervalidator-hip-arithmetic-tests --unavailable)
            set_tests_properties(forevervalidator-hip-unavailable PROPERTIES
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1")
        endif()

        add_executable(forevervalidator-hip-timeline-tests
            tests/generated/hip_timeline_executor_tests.cpp)
        target_include_directories(
            forevervalidator-hip-timeline-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${FOREVERVALIDATOR_HIP_INCLUDE_DIR}")
        if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
            target_compile_definitions(
                forevervalidator-hip-timeline-tests PRIVATE
                __HIP_PLATFORM_NVIDIA__=1)
        else()
            target_compile_definitions(
                forevervalidator-hip-timeline-tests PRIVATE
                __HIP_PLATFORM_AMD__=1)
        endif()
        target_link_libraries(
            forevervalidator-hip-timeline-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-hip-timeline-tests)
        add_test(NAME forevervalidator-hip-timeline
            COMMAND forevervalidator-hip-timeline-tests)

        foreach(hip_test IN ITEMS
                search_branch_state
                search_winner_reduction
                custom_volume
                modifier_event_ops
                candidate_events
                state_layout
                static_configuration
                dynamics
                stunt
                collision
                race
                finish_time_origin)
            set(hip_test_source
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/generated/hip_${hip_test}_tests.cpp")
            if(hip_test STREQUAL "search_branch_state" OR
               hip_test STREQUAL "search_winner_reduction" OR
               hip_test STREQUAL "finish_time_origin")
                set(hip_test_source
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/generated/hip_${hip_test}_tests.cu")
                set_source_files_properties("${hip_test_source}"
                    PROPERTIES LANGUAGE HIP)
            endif()
            add_executable(forevervalidator-hip-${hip_test}-tests
                "${hip_test_source}")
            target_include_directories(
                forevervalidator-hip-${hip_test}-tests PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
                "${FOREVERVALIDATOR_HIP_INCLUDE_DIR}")
            if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
                target_compile_definitions(
                    forevervalidator-hip-${hip_test}-tests PRIVATE
                    "$<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_NVIDIA__=1>")
                target_compile_options(
                    forevervalidator-hip-${hip_test}-tests PRIVATE
                    "$<$<COMPILE_LANGUAGE:HIP>:--fmad=false>"
                    "$<$<COMPILE_LANGUAGE:HIP>:--prec-div=true>"
                    "$<$<COMPILE_LANGUAGE:HIP>:--prec-sqrt=true>"
                    "$<$<COMPILE_LANGUAGE:HIP>:--ftz=false>")
            else()
                target_compile_definitions(
                    forevervalidator-hip-${hip_test}-tests PRIVATE
                    "$<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>")
                target_compile_options(
                    forevervalidator-hip-${hip_test}-tests PRIVATE
                    "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>"
                    "$<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>")
            endif()
            target_link_libraries(
                forevervalidator-hip-${hip_test}-tests PRIVATE
                forevervalidator_core)
            forevervalidator_configure_target(
                forevervalidator-hip-${hip_test}-tests)
            add_test(NAME forevervalidator-hip-${hip_test}
                COMMAND forevervalidator-hip-${hip_test}-tests)
        endforeach()
    endif()

    if(FOREVERVALIDATOR_ENABLE_HIP AND FOREVERVALIDATOR_ENABLE_CUDA AND
       CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        add_executable(forevervalidator-hip-cuda-timeline-parity-tests
            tests/hip_cuda_timeline_parity_tests.cpp)
        target_include_directories(
            forevervalidator-hip-cuda-timeline-parity-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${FOREVERVALIDATOR_HIP_INCLUDE_DIR}")
        target_compile_definitions(
            forevervalidator-hip-cuda-timeline-parity-tests PRIVATE
            __HIP_PLATFORM_NVIDIA__=1)
        target_link_libraries(
            forevervalidator-hip-cuda-timeline-parity-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-hip-cuda-timeline-parity-tests)
        add_test(NAME forevervalidator-hip-cuda-timeline-parity
            COMMAND forevervalidator-hip-cuda-timeline-parity-tests)

        add_executable(forevervalidator-hip-cuda-search-parity-tests
            tests/hip_cuda_search_parity_tests.cpp)
        target_include_directories(
            forevervalidator-hip-cuda-search-parity-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${FOREVERVALIDATOR_HIP_INCLUDE_DIR}")
        target_compile_definitions(
            forevervalidator-hip-cuda-search-parity-tests PRIVATE
            __HIP_PLATFORM_NVIDIA__=1)
        target_link_libraries(
            forevervalidator-hip-cuda-search-parity-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-hip-cuda-search-parity-tests)
        add_test(NAME forevervalidator-hip-cuda-search-parity
            COMMAND forevervalidator-hip-cuda-search-parity-tests)
    endif()

    if(FOREVERVALIDATOR_ENABLE_CUDA)
        add_executable(forevervalidator-cuda-arithmetic-tests
            tests/cuda_arithmetic_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-arithmetic-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-arithmetic-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-cuda-arithmetic-tests)
        add_test(NAME forevervalidator-cuda-arithmetic
                 COMMAND forevervalidator-cuda-arithmetic-tests)

        add_executable(forevervalidator-cuda-driver-shim-tests
            src/simulation/backends/cuda/cuda_driver_shim.cpp
            tests/cuda_driver_shim_tests.cpp)
        target_include_directories(forevervalidator-cuda-driver-shim-tests PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS})
        forevervalidator_configure_target(forevervalidator-cuda-driver-shim-tests)
        add_test(NAME forevervalidator-cuda-driver-shim
            COMMAND forevervalidator-cuda-driver-shim-tests)

        add_executable(forevervalidator-cuda-dynamics-tests
            tests/cuda_dynamics_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-dynamics-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-dynamics-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-cuda-dynamics-tests)
        add_test(NAME forevervalidator-cuda-dynamics
                 COMMAND forevervalidator-cuda-dynamics-tests)

        add_executable(forevervalidator-cuda-timeline-tests
            tests/cuda_timeline_executor_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-timeline-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-timeline-tests PRIVATE
            forevervalidator_core CUDA::cudart)
        forevervalidator_configure_target(
            forevervalidator-cuda-timeline-tests)
        add_test(NAME forevervalidator-cuda-timeline
                 COMMAND forevervalidator-cuda-timeline-tests)

        add_executable(forevervalidator-cuda-stunt-tests
            tests/cuda_stunt_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-stunt-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-stunt-tests PRIVATE
            forevervalidator_core)
        forevervalidator_configure_target(
            forevervalidator-cuda-stunt-tests)
        add_test(NAME forevervalidator-cuda-stunts
                 COMMAND forevervalidator-cuda-stunt-tests)

        add_executable(forevervalidator-cuda-collision-tests
            tests/cuda_collision_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-collision-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-collision-tests PRIVATE
            forevervalidator_core CUDA::cudart)
        forevervalidator_configure_target(
            forevervalidator-cuda-collision-tests)
        add_test(NAME forevervalidator-cuda-collision
                 COMMAND forevervalidator-cuda-collision-tests)

        add_executable(forevervalidator-cuda-race-tests
            tests/cuda_race_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-race-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-race-tests PRIVATE
            forevervalidator_core CUDA::cudart)
        forevervalidator_configure_target(
            forevervalidator-cuda-race-tests)
        add_test(NAME forevervalidator-cuda-race
                 COMMAND forevervalidator-cuda-race-tests)

        add_executable(forevervalidator-cuda-finish-time-origin-tests
            tests/cuda_finish_time_origin_tests.cu)
        target_include_directories(
            forevervalidator-cuda-finish-time-origin-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-finish-time-origin-tests PRIVATE
            CUDA::cudart)
        forevervalidator_configure_target(
            forevervalidator-cuda-finish-time-origin-tests)
        add_test(NAME forevervalidator-cuda-finish-time-origin
                 COMMAND forevervalidator-cuda-finish-time-origin-tests)

        add_executable(forevervalidator-cuda-search-winner-reduction-tests
            tests/cuda_search_winner_reduction_tests.cu)
        target_include_directories(
            forevervalidator-cuda-search-winner-reduction-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        forevervalidator_configure_target(
            forevervalidator-cuda-search-winner-reduction-tests)
        add_test(NAME forevervalidator-cuda-search-winner-reduction
                 COMMAND
                 forevervalidator-cuda-search-winner-reduction-tests)

        add_executable(forevervalidator-cuda-modifier-event-ops-tests
            tests/cuda_modifier_event_ops_tests.cpp)
        target_include_directories(
            forevervalidator-cuda-modifier-event-ops-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        forevervalidator_configure_target(
            forevervalidator-cuda-modifier-event-ops-tests)
        add_test(NAME forevervalidator-cuda-modifier-event-ops
                 COMMAND
                 forevervalidator-cuda-modifier-event-ops-tests)

        add_executable(forevervalidator-cuda-search-branch-state-tests
            tests/cuda_search_branch_state_tests.cu)
        target_include_directories(
            forevervalidator-cuda-search-branch-state-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(
            forevervalidator-cuda-search-branch-state-tests PRIVATE
            CUDA::cudart)
        forevervalidator_configure_target(
            forevervalidator-cuda-search-branch-state-tests)
        add_test(NAME forevervalidator-cuda-search-branch-state
                 COMMAND forevervalidator-cuda-search-branch-state-tests)
    endif()

    add_executable(forevervalidator-render-scene-tests
        tests/render_scene_tests.cpp)
    target_include_directories(forevervalidator-render-scene-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(forevervalidator-render-scene-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(forevervalidator-render-scene-tests)
    add_test(NAME forevervalidator-render-scene
             COMMAND forevervalidator-render-scene-tests)

    add_executable(forevervalidator-replay-control-plan-tests
        tests/replay_control_plan_tests.cpp)
    target_include_directories(
        forevervalidator-replay-control-plan-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        forevervalidator-replay-control-plan-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        forevervalidator-replay-control-plan-tests)
    add_test(NAME forevervalidator-replay-control-plan
        COMMAND forevervalidator-replay-control-plan-tests)

    add_executable(forevervalidator-checkpoint-freewheel-tests
        tests/checkpoint_freewheel_tests.cpp)
    target_include_directories(
        forevervalidator-checkpoint-freewheel-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        forevervalidator-checkpoint-freewheel-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        forevervalidator-checkpoint-freewheel-tests)
    add_test(NAME forevervalidator-checkpoint-freewheel
        COMMAND forevervalidator-checkpoint-freewheel-tests)

    add_executable(forevervalidator-replay-decoration-size-tests
        tests/replay_decoration_size_tests.cpp)
    target_include_directories(
        forevervalidator-replay-decoration-size-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        forevervalidator-replay-decoration-size-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        forevervalidator-replay-decoration-size-tests)
    add_test(NAME forevervalidator-replay-decoration-size
        COMMAND forevervalidator-replay-decoration-size-tests)

    add_executable(forevervalidator-replay-scene-unresolved-blocks-tests
        tests/replay_scene_unresolved_blocks_tests.cpp)
    target_include_directories(forevervalidator-replay-scene-unresolved-blocks-tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(forevervalidator-replay-scene-unresolved-blocks-tests PRIVATE
        forevervalidator_core)
    forevervalidator_configure_target(
        forevervalidator-replay-scene-unresolved-blocks-tests)
    add_test(NAME forevervalidator-replay-scene-unresolved-blocks
        COMMAND forevervalidator-replay-scene-unresolved-blocks-tests)
endif()
