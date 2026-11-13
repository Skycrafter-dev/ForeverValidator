# Twin engine build (docs/twin/integration.md).
#
#   Root build:  cmake -S . -B <b> -DFOREVERVALIDATOR_BUILD_TWIN=ON [-DFOREVERVALIDATOR_BUILD_TWIN_TOOLS=ON]
#                twin libraries + the FV adapter compiled into forevervalidator_core
#                (FOREVERVALIDATOR_HAS_TWIN=1, backend "twin", experimental/twin.h)
#                + CTest tests + engine "twin" in twin_harness.
#   Standalone:  cmake -S src/twin -B <b> -G Ninja [-DFOREVERVALIDATOR_TWIN_ISA=avx2] [-DCMAKE_CXX_COMPILER=clang++]
#                FV-free libraries and tests only (the determinism matrix, tests/twin/ci/).
#
# The modules' own CMakeLists (math tests, model, world, car, race) stay
# standalone and are not read here; this file compiles their library
# sources directly with one flag set.
#
# Options:
#   FOREVERVALIDATOR_TWIN_ISA       default (compiler baseline) | sse2 | avx2 | avx512 | native
#   FOREVERVALIDATOR_TWIN_CONTRACT  off (default: -ffp-contract=off, no FMA) | fast (measurement only:
#                                   FMA contraction; the FMA audit and the identity gates then fail)
#   FOREVERVALIDATOR_TWIN_MARCH     ON: the AVX2 / AVX-512 tick variants use -march=x86-64-v3 / -v4
#                                   (OFF: -mavx2 / -mavx512f -mavx512vl, plus -mfma when contracting)
#   FOREVERVALIDATOR_TWIN_LTO       ON: each tick variant is one partial LTO link of its sources
#   FOREVERVALIDATOR_TWIN_EXHAUSTIVE_TESTS  register the 2^32-input math tests (label exhaustive,
#                                   3-10 s each; run them one at a time)
#
# Floating point (user decisions 2026-11-08 and 2026-11-09): no FMA (-ffp-contract=off on every twin
# TU; the object audit fails on any FMA / rcp instruction); -march and LTO
# per ISA variant are allowed. Gates: the same binary on the same CPU repeats bit for bit
# (tests/twin/ci/repeat.sh, CTest tw_repeat_*), the FMA audit, and, with contraction off, the pinned
# hashes and every ISA variant equal to the baseline. The acceptance test (docs/twin/scoreboard.md)
# is the correctness bar. FMA gained ~4% (docs/twin/integration.md §5b) and -ffast-math failed it.
cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(tw_twin CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    set(TW_STANDALONE ON)
    enable_testing()
else()
    set(TW_STANDALONE OFF)
endif()

get_filename_component(TW_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TW_SRC "${CMAKE_CURRENT_SOURCE_DIR}")

set(FOREVERVALIDATOR_TWIN_ISA "default" CACHE STRING
    "Twin kernel ISA: default | sse2 | avx2 | avx512 | native")
set(FOREVERVALIDATOR_TWIN_CONTRACT "off" CACHE STRING
    "Twin FP contraction: off (no FMA, default) | fast (measurement only)")
option(FOREVERVALIDATOR_TWIN_MARCH "Twin AVX2 / AVX-512 tick variants with -march=x86-64-v3 / -v4" ON)
option(FOREVERVALIDATOR_TWIN_LTO "Twin tick variants as one partial LTO link each" ON)
option(FOREVERVALIDATOR_TWIN_EXHAUSTIVE_TESTS
       "Register the exhaustive (2^32-input) twin math tests" OFF)
option(FOREVERVALIDATOR_TWIN_ISA_DISPATCH
       "Build AVX2 and AVX-512 variants of the twin's tick path, chosen at session creation (x86-64, GNU binutils)" ON)

# ---- floating-point and ISA flags ------------------------------------------------
# TW_FP_STRICT: no contraction, so builds agree bit for bit (the old determinism flags,
# docs/twin/math.md §4); otherwise FMA contraction is allowed (the default).
set(TW_FP_STRICT 0)
if(FOREVERVALIDATOR_TWIN_CONTRACT STREQUAL "off")
    set(TW_FP_STRICT 1)
elseif(NOT FOREVERVALIDATOR_TWIN_CONTRACT STREQUAL "fast")
    message(FATAL_ERROR "FOREVERVALIDATOR_TWIN_CONTRACT must be fast or off")
endif()
if(MSVC)
    if(TW_FP_STRICT)
        set(TW_FP_FLAGS /fp:strict)
    else()
        set(TW_FP_FLAGS /fp:precise /fp:contract)
    endif()
    set(TW_ISA_FLAGS "")
    if(FOREVERVALIDATOR_TWIN_ISA STREQUAL "avx2")
        set(TW_ISA_FLAGS /arch:AVX2)
    elseif(FOREVERVALIDATOR_TWIN_ISA STREQUAL "avx512")
        set(TW_ISA_FLAGS /arch:AVX512)
    endif()
else()
    set(TW_FP_FLAGS -fno-fast-math -fno-strict-aliasing -fno-math-errno)
    if(TW_FP_STRICT)
        list(APPEND TW_FP_FLAGS -ffp-contract=off)
    else()
        list(APPEND TW_FP_FLAGS -ffp-contract=fast)
    endif()
    # The AVX2 / AVX-512 instruction sets as the tick variants (and the standalone matrix) use them:
    # x86-64-v3 / -v4 (tw_isa.cpp checks the same levels), or the bare feature flags (+ FMA when contracting).
    if(FOREVERVALIDATOR_TWIN_MARCH)
        set(TW_ISA_FLAGS_avx2 -march=x86-64-v3)
        set(TW_ISA_FLAGS_avx512 -march=x86-64-v4)
    else()
        set(TW_ISA_FLAGS_avx2 -mavx2)
        set(TW_ISA_FLAGS_avx512 -mavx512f -mavx512vl)
        if(NOT TW_FP_STRICT)
            list(APPEND TW_ISA_FLAGS_avx2 -mfma)
            list(APPEND TW_ISA_FLAGS_avx512 -mfma)
        endif()
    endif()
    if(FOREVERVALIDATOR_TWIN_ISA STREQUAL "default")
        set(TW_ISA_FLAGS "")
    elseif(FOREVERVALIDATOR_TWIN_ISA STREQUAL "sse2")
        set(TW_ISA_FLAGS -msse2)
    elseif(FOREVERVALIDATOR_TWIN_ISA STREQUAL "avx2")
        set(TW_ISA_FLAGS ${TW_ISA_FLAGS_avx2})
    elseif(FOREVERVALIDATOR_TWIN_ISA STREQUAL "avx512")
        set(TW_ISA_FLAGS ${TW_ISA_FLAGS_avx512})
    elseif(FOREVERVALIDATOR_TWIN_ISA STREQUAL "native")
        set(TW_ISA_FLAGS -march=native)
    else()
        message(FATAL_ERROR "unknown FOREVERVALIDATOR_TWIN_ISA ${FOREVERVALIDATOR_TWIN_ISA}")
    endif()
endif()
set(TW_FLAGS ${TW_FP_FLAGS} ${TW_ISA_FLAGS})
if(NOT MSVC)
    list(APPEND TW_FLAGS -Wall -Wextra)
endif()

# Flags go PRIVATE on each twin target, never INTERFACE: consumers (FV core)
# keep their own flags.
function(tw_configure target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE ${TW_FLAGS})
endfunction()

find_package(Threads REQUIRED)

# ---- FV-free libraries -----------------------------------------------------------
add_library(tw_math INTERFACE)                       # header-only (math agent, done)
target_include_directories(tw_math INTERFACE "${TW_SRC}/math")

add_library(tw_model STATIC "${TW_SRC}/model/tw_compile_car.cpp")    # state + car pack (M1)
target_include_directories(tw_model PUBLIC "${TW_SRC}/model")
target_link_libraries(tw_model PUBLIC tw_math)
tw_configure(tw_model)

add_library(tw_car STATIC "${TW_SRC}/model/forces.cpp")              # StadiumCar forces (M4, in progress)
target_include_directories(tw_car PUBLIC "${TW_SRC}/model")
target_link_libraries(tw_car PUBLIC tw_math)
tw_configure(tw_car)

add_library(tw_race STATIC "${TW_SRC}/race/tw_controls.cpp")         # race layer (M6a)
target_include_directories(tw_race PUBLIC "${TW_SRC}/race")
target_link_libraries(tw_race PUBLIC tw_model)
tw_configure(tw_race)

add_library(tw_world STATIC                                          # static world (W-BROAD/X1) + world pack (M2)
    "${TW_SRC}/world/tw_world.cpp"
    "${TW_SRC}/world/tw_accel.cpp"
    "${TW_SRC}/world/tw_blocks.cpp"
    "${TW_SRC}/world/tw_world_pack.cpp")
target_include_directories(tw_world PUBLIC "${TW_SRC}/world")
target_link_libraries(tw_world PUBLIC tw_race)                       # tw_world_pack.h uses tw_race.h
tw_configure(tw_world)

add_library(tw_collide STATIC "${TW_SRC}/collide/tw_collide.cpp"   # M3 exact collision stage (step::CollisionStage)
    "${TW_SRC}/world/tw_exact_fast.cpp"                              # + the fast exact test's packet path
    "${TW_SRC}/world/tw_car_query.cpp"                               # + the candidate memo and world-space prefilter
    "${TW_SRC}/world/tw_patches.cpp"                                 # + coplanar patches (merged patch semantics)
    "${TW_SRC}/collide/tw_collide_ref.cpp"                           # + the gate 1' double-precision collision reference
    "${TW_SRC}/collide_c/cc_stage.cpp" "${TW_SRC}/collide_c/cc_memo.cpp"   # + architect C's stage (FOREVERVALIDATOR_TWIN_COLLIDE=c)
    "${TW_SRC}/atlas/atlas_build.cpp" "${TW_SRC}/atlas/atlas_stage.cpp"   # + architect E's atlas stage (FOREVERVALIDATOR_TWIN_COLLIDE=atlas)
    "${TW_SRC}/atlas/tw_atlas_fb.cpp"                                        #   and architect D's per-shape fallback (S2)
    "${TW_SRC}/atlas/atlas_colcache.cpp")                                    #   and architect C's lazy column cache
target_include_directories(tw_collide PUBLIC "${TW_SRC}/collide" "${TW_SRC}/collide_c" "${TW_SRC}/step" "${TW_SRC}/atlas")
target_link_libraries(tw_collide PUBLIC tw_world tw_model tw_race tw_math)
tw_configure(tw_collide)

add_library(tw_engine STATIC "${TW_SRC}/step/tw_step.cpp"           # tick entry points (M5: step/tick.h)
    "${TW_SRC}/step/policy.cpp")
target_include_directories(tw_engine PUBLIC "${TW_SRC}/step")
target_link_libraries(tw_engine PUBLIC tw_model tw_car tw_world tw_race tw_math)
tw_configure(tw_engine)

# ---- GPU engine (src/twin/gpu, docs/twin/gpu.md) -------------------------------------------
# The scalar (L = 1) path as CUDA device code, one candidate per thread: no FMA, IEEE division and
# square root, no flush-to-zero, so the device computes the CPU twin's bits.
# HIP: the same sources through gpu/hip (tw_hip_compat.h): one module per kernel file, each with the device stages
# it calls (no device link), the twin's floating point (no contraction, IEEE division and square root, denormals
# kept) and 32-lane waves.
set(TW_GPU_HIP_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>"
    "$<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>"
    "$<$<COMPILE_LANGUAGE:HIP>:-fhip-fp32-correctly-rounded-divide-sqrt>"
    "$<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64>")
if(CMAKE_HIP_FLAGS MATCHES "driver-mode=cl")
    list(APPEND TW_GPU_HIP_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:/FI${TW_SRC}/gpu/hip/tw_hip_compat.h>")
else()
    list(APPEND TW_GPU_HIP_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${TW_SRC}/gpu/hip/tw_hip_compat.h>")
endif()
if(FOREVERVALIDATOR_BUILD_TWIN_GPU AND TW_GPU_HIP)
    add_library(tw_gpu STATIC "${TW_SRC}/gpu/hip/tw_gpu_car_hip.hip" "${TW_SRC}/gpu/hip/tw_gpu_collide_hip.hip"
        "${TW_SRC}/gpu/hip/tw_gpu_queue_hip.hip" "${TW_SRC}/gpu/hip/tw_gpu_session_hip.hip" "${TW_SRC}/gpu/tw_gpu_world.cpp")
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS tw_gpu_pool=2)
    set_target_properties(tw_gpu PROPERTIES JOB_POOL_COMPILE tw_gpu_pool HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
        HIP_EXTENSIONS OFF)
    target_include_directories(tw_gpu BEFORE PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:${TW_SRC}/gpu/hip/include>"
        "$<$<COMPILE_LANGUAGE:HIP>:${TW_SRC}/gpu/hip>")
    target_include_directories(tw_gpu PUBLIC "${TW_SRC}/gpu")
    target_link_libraries(tw_gpu PUBLIC tw_collide tw_model tw_car tw_race tw_math)
    target_compile_options(tw_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${TW_FLAGS}>" ${TW_GPU_HIP_OPTIONS})
    target_compile_definitions(tw_gpu PRIVATE TW_CAR_NO_SSE3V=1)
elseif(FOREVERVALIDATOR_BUILD_TWIN_GPU)
    add_library(tw_gpu STATIC "${TW_SRC}/gpu/tw_gpu_car.cu" "${TW_SRC}/gpu/tw_gpu_collide.cu" "${TW_SRC}/gpu/tw_gpu_queue.cu"
        "${TW_SRC}/gpu/tw_gpu_k_forces.cu" "${TW_SRC}/gpu/tw_gpu_k_detect.cu" "${TW_SRC}/gpu/tw_gpu_k_respond.cu"
        "${TW_SRC}/gpu/tw_gpu_session.cu" "${TW_SRC}/gpu/tw_gpu_world.cpp")
    # the stages are separate modules (tw_gpu_stages.cuh): device-linked in the library; one CUDA compile at a
    # time (ptxas takes several GB per module)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS tw_gpu_pool=1)
    set_target_properties(tw_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON JOB_POOL_COMPILE tw_gpu_pool)
    target_include_directories(tw_gpu PUBLIC "${TW_SRC}/gpu")
    # with FV's CUDA backend in the same process (the twin search session's GPU engine next to FV's CUDA search) one
    # runtime: FV's shared cudart
    if(FOREVERVALIDATOR_ENABLE_CUDA)
        target_link_libraries(tw_gpu PUBLIC tw_collide tw_model tw_car tw_race tw_math CUDA::cudart)
    else()
        target_link_libraries(tw_gpu PUBLIC tw_collide tw_model tw_car tw_race tw_math CUDA::cudart_static)
    endif()
    target_compile_options(tw_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${TW_FLAGS}>")
    set_target_properties(tw_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_property(TARGET tw_gpu PROPERTY CUDA_ARCHITECTURES native)
    endif()
    target_compile_options(tw_gpu PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>")
    target_compile_definitions(tw_gpu PRIVATE TW_CAR_NO_SSE3V=1)
    # nvcc with cl.exe (Windows): the GCC builtins and attributes the twin spells, for both passes
    set(TW_CUDA_MSVC_OPTIONS "")
    if(CMAKE_CUDA_SIMULATE_ID STREQUAL "MSVC")
        set(TW_CUDA_MSVC_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${TW_SRC}/gpu/tw_cuda_msvc_compat.h>")
    endif()
    target_compile_options(tw_gpu PRIVATE ${TW_CUDA_MSVC_OPTIONS})
    # 128 (512 threads an SM) is the measured best on real sessions with the stages inlined in the lane runner's pass
    # (gpu-a2; 80 before, 64 before detection v5): re-measure 96 / 128 / 168 when the collision or the pass changes
    set(FOREVERVALIDATOR_TWIN_GPU_MAXRREG "128" CACHE STRING "Twin GPU: registers per thread cap (-maxrregcount; empty: none)")
    if(FOREVERVALIDATOR_TWIN_GPU_MAXRREG)
        target_compile_options(tw_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=${FOREVERVALIDATOR_TWIN_GPU_MAXRREG}>")
    endif()
endif()

if(FOREVERVALIDATOR_BUILD_TWIN_GPU)
    # the session judge's FV targets (gpu/tw_gpu_judge.cuh): FV's evaluator code with its CUDA qualifiers, extracted at
    # build time (FV source only, no FV header)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(TW_JUDGE_EXTRACT "${CMAKE_CURRENT_BINARY_DIR}/generated/twin_cuda_judge_extract.inc")
    set(TW_CUDA_BACKEND_DIR "${TW_REPO_ROOT}/src/simulation/backends/cuda")
    add_custom_command(
        OUTPUT "${TW_JUDGE_EXTRACT}"
        COMMAND "${Python3_EXECUTABLE}" "${TW_SRC}/fv/extract_cuda_search.py" --judge
                "${TW_CUDA_BACKEND_DIR}/cuda_search_executor.cu" "${TW_JUDGE_EXTRACT}"
        DEPENDS "${TW_SRC}/fv/extract_cuda_search.py" "${TW_CUDA_BACKEND_DIR}/cuda_search_executor.cu"
                "${TW_CUDA_BACKEND_DIR}/cuda_search_executor.h" "${TW_CUDA_BACKEND_DIR}/cuda_search_winner_selection.cuh"
                "${TW_CUDA_BACKEND_DIR}/cuda_custom_volume.cuh"
        COMMENT "Extracting FV CUDA search evaluator code for the twin GPU judge"
        VERBATIM)
    add_custom_target(tw_judge_extract DEPENDS "${TW_JUDGE_EXTRACT}")
    add_dependencies(tw_gpu tw_judge_extract)
    target_include_directories(tw_gpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated" "${TW_REPO_ROOT}/include")
endif()

# ---- runtime ISA dispatch (isa/tw_isa.h) -----------------------------------------------
# The tick path again, as isolated AVX2 / AVX-512 variant objects; the
# baseline is the libraries above. A session picks one by CPUID.
set(TW_ISA_TICK_SOURCES
    "${TW_SRC}/isa/tw_isa_tick_entry.cpp"
    "${TW_SRC}/step/tw_step.cpp"
    "${TW_SRC}/step/policy.cpp"
    "${TW_SRC}/collide/tw_collide.cpp"
    "${TW_SRC}/world/tw_exact_fast.cpp"
    "${TW_SRC}/world/tw_car_query.cpp"
    "${TW_SRC}/world/tw_patches.cpp"
    "${TW_SRC}/collide_c/cc_stage.cpp"
    "${TW_SRC}/collide_c/cc_memo.cpp"
    "${TW_SRC}/atlas/atlas_build.cpp"
    "${TW_SRC}/atlas/atlas_stage.cpp"
    "${TW_SRC}/atlas/tw_atlas_fb.cpp"
    "${TW_SRC}/atlas/atlas_colcache.cpp"
    "${TW_SRC}/model/forces.cpp")
set(TW_ISA_INCLUDE_DIRS "${TW_SRC}/isa" "${TW_SRC}/batch" "${TW_SRC}/step" "${TW_SRC}/collide" "${TW_SRC}/collide_c" "${TW_SRC}/atlas" "${TW_SRC}/world" "${TW_SRC}/race"
                        "${TW_SRC}/model" "${TW_SRC}/math")
set(TW_ISA_VARIANT_OBJECTS "")
set(TW_ISA_DEFINES "")
find_program(TW_READELF NAMES readelf "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu-readelf")
set(TW_ISA_CAN_DISPATCH OFF)
if(FOREVERVALIDATOR_TWIN_ISA_DISPATCH AND NOT MSVC AND NOT WIN32 AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND FOREVERVALIDATOR_TWIN_ISA STREQUAL "default" AND
   CMAKE_LINKER AND CMAKE_OBJCOPY AND CMAKE_NM AND TW_READELF)
    set(TW_ISA_CAN_DISPATCH ON)
endif()
if(TW_ISA_CAN_DISPATCH)
    set(TW_ISA_KIND_avx2 1)
    set(TW_ISA_KIND_avx512 2)
    # LTO: the variant's sources compile to IR and the partial link (-r) optimizes them as one unit
    # (GCC: one partition, native relocatable output; Clang: full LTO). Without it: ld -r.
    set(TW_ISA_LTO OFF)
    if(FOREVERVALIDATOR_TWIN_LTO AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(TW_ISA_LTO ON)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" TW_BUILD_TYPE_UPPER)
        separate_arguments(TW_LTO_OPT UNIX_COMMAND "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${TW_BUILD_TYPE_UPPER}}")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(TW_LTO_COMPILE -flto)
            set(TW_LTO_LINK -flto -flto-partition=one -r -nostdlib -flinker-output=nolto-rel)
        else()
            set(TW_LTO_COMPILE -flto)
            set(TW_LTO_LINK -flto -r -nostdlib)
        endif()
    endif()
    set(TW_ISA_FP_FLAGS ${TW_FP_FLAGS})
    # Extra flags for both variants, compile and LTO link (code layout experiments; no FP flags here)
    set(FOREVERVALIDATOR_TWIN_VARIANT_FLAGS "" CACHE STRING "Extra compile + link flags of the twin tick variants")
    separate_arguments(TW_ISA_EXTRA_FLAGS UNIX_COMMAND "${FOREVERVALIDATOR_TWIN_VARIANT_FLAGS}")
    foreach(isa avx2 avx512)
        add_library(tw_isa_${isa}_objects OBJECT ${TW_ISA_TICK_SOURCES})
        target_include_directories(tw_isa_${isa}_objects PRIVATE ${TW_ISA_INCLUDE_DIRS})
        target_compile_options(tw_isa_${isa}_objects PRIVATE ${TW_ISA_FP_FLAGS} ${TW_ISA_FLAGS_${isa}} ${TW_ISA_EXTRA_FLAGS} -Wall -Wextra)
        target_compile_definitions(tw_isa_${isa}_objects PRIVATE
            TW_ISA_KIND=${TW_ISA_KIND_${isa}} TW_ISA_ENTRY=tw_isa_tick_table_${isa})
        target_compile_features(tw_isa_${isa}_objects PRIVATE cxx_std_17)
        if(TW_ISA_LTO)
            target_compile_options(tw_isa_${isa}_objects PRIVATE ${TW_LTO_COMPILE})
            set(TW_ISA_LINK_${isa} "${CMAKE_CXX_COMPILER}" ${TW_LTO_OPT} ${TW_ISA_FP_FLAGS} ${TW_ISA_FLAGS_${isa}} ${TW_ISA_EXTRA_FLAGS} ${TW_LTO_LINK})
        else()
            set(TW_ISA_LINK_${isa} "${CMAKE_LINKER}" -r)
        endif()
        set(out "${CMAKE_CURRENT_BINARY_DIR}/tw_isa_${isa}.o")
        add_custom_command(
            OUTPUT "${out}"
            COMMAND "${TW_SRC}/isa/isolate_variant.sh" "${out}" tw_isa_tick_table_${isa} ".tw_isa_${isa}"
                    "${CMAKE_OBJCOPY}" "${CMAKE_NM}" "${TW_READELF}" -- ${TW_ISA_LINK_${isa}} --
                    "$<TARGET_OBJECTS:tw_isa_${isa}_objects>"
            DEPENDS tw_isa_${isa}_objects "${TW_SRC}/isa/isolate_variant.sh" $<TARGET_OBJECTS:tw_isa_${isa}_objects>
            COMMAND_EXPAND_LISTS
            COMMENT "Isolating the twin ${isa} tick variant"
            VERBATIM)
        list(APPEND TW_ISA_VARIANT_OBJECTS "${out}")
    endforeach()
    list(APPEND TW_ISA_DEFINES TW_ISA_HAVE_AVX2=1 TW_ISA_HAVE_AVX512=1)
endif()
set_source_files_properties(${TW_ISA_VARIANT_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
# Baseline table + selection. The baseline entry compiles the same entry
# source at the libraries' ISA and links against them.
add_library(tw_isa STATIC "${TW_SRC}/isa/tw_isa.cpp" "${TW_SRC}/isa/tw_isa_tick_entry.cpp"
    ${TW_ISA_VARIANT_OBJECTS})
target_include_directories(tw_isa PUBLIC "${TW_SRC}/isa")
target_include_directories(tw_isa PRIVATE "${TW_SRC}/batch")
target_compile_definitions(tw_isa PRIVATE TW_ISA_KIND=0 TW_ISA_ENTRY=tw_isa_tick_table_baseline ${TW_ISA_DEFINES})
target_link_libraries(tw_isa PUBLIC tw_engine tw_collide tw_car tw_world tw_race tw_model tw_math)
tw_configure(tw_isa)
message(STATUS "twin ISA dispatch: ${TW_ISA_CAN_DISPATCH} (contraction ${FOREVERVALIDATOR_TWIN_CONTRACT}, "
               "march ${FOREVERVALIDATOR_TWIN_MARCH}, LTO ${TW_ISA_LTO})")

set(TW_LIBRARIES tw_isa tw_engine tw_collide tw_race tw_car tw_world tw_model)
set(TW_INCLUDE_DIRS "${TW_SRC}/api" "${TW_SRC}/step" "${TW_SRC}/race" "${TW_SRC}/model" "${TW_SRC}/world" "${TW_SRC}/collide_c" "${TW_SRC}/atlas"
                    "${TW_SRC}/math" "${TW_SRC}/fv" "${TW_SRC}/collide" "${TW_SRC}/isa")
if(FOREVERVALIDATOR_BUILD_TWIN_GPU)
    list(APPEND TW_INCLUDE_DIRS "${TW_SRC}/gpu")
endif()

# ---- FV-free tests -----------------------------------------------------------------
set(TW_MATH_TESTS "${TW_REPO_ROOT}/tests/twin/math")
# The math library's own tests check its algorithms bit for bit against references, so they
# compile without contraction whatever the product's flags (the product is judged by the
# acceptance test).
add_executable(tw_math_tests "${TW_MATH_TESTS}/tw_math_tests.cpp")
target_link_libraries(tw_math_tests PRIVATE tw_math Threads::Threads)
tw_configure(tw_math_tests)
add_library(tw_math_audit OBJECT "${TW_MATH_TESTS}/tw_math_audit.cpp")
target_link_libraries(tw_math_audit PRIVATE tw_math)
tw_configure(tw_math_audit)
if(NOT MSVC)
    target_compile_options(tw_math_tests PRIVATE -ffp-contract=off)
    target_compile_options(tw_math_audit PRIVATE -ffp-contract=off)
endif()

add_executable(tw_m1_test "${TW_SRC}/model/test/tw_m1_test.cpp")
target_include_directories(tw_m1_test PRIVATE "${TW_REPO_ROOT}")
target_link_libraries(tw_m1_test PRIVATE tw_model)
tw_configure(tw_m1_test)

add_executable(tw_race_test "${TW_SRC}/race/test/tw_race_test.cpp")
target_link_libraries(tw_race_test PRIVATE tw_race)
tw_configure(tw_race_test)

add_executable(tw_world_pack_test "${TW_SRC}/fv/test/tw_world_pack_test.cpp")   # M2 synthetic checks
target_link_libraries(tw_world_pack_test PRIVATE tw_world)
tw_configure(tw_world_pack_test)

add_executable(tw_golden_hash "${TW_REPO_ROOT}/tests/twin/golden/tw_golden_hash.cpp")
target_link_libraries(tw_golden_hash PRIVATE ${TW_LIBRARIES})
target_compile_definitions(tw_golden_hash PRIVATE
    TW_GOLDEN_FILE="${TW_REPO_ROOT}/tests/twin/golden/goldens.txt" TW_FP_STRICT=${TW_FP_STRICT})
tw_configure(tw_golden_hash)

add_test(NAME tw_math_canary COMMAND tw_math_tests canary)
add_test(NAME tw_math_identity COMMAND tw_math_tests identity)
add_test(NAME tw_math_atan2 COMMAND tw_math_tests atan2 30000000)
add_test(NAME tw_math_equal_sin_small COMMAND tw_math_tests equal sin sincos_small.sin -0.785 0.785)
add_test(NAME tw_math_equal_cos_small COMMAND tw_math_tests equal cos sincos_small.cos -0.785 0.785)
add_test(NAME tw_math_equal_asin_small COMMAND tw_math_tests equal asin asin_small -0.5 0.5)
add_test(NAME tw_math_audit COMMAND "${TW_MATH_TESTS}/audit_objects.sh" "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME tw_model_m1 COMMAND tw_m1_test --json "${TW_REPO_ROOT}/src/twin/model/stadiumcar-values.json")
add_test(NAME tw_race COMMAND tw_race_test)
add_test(NAME tw_world_pack COMMAND tw_world_pack_test)
set(TW_REPEAT "${TW_REPO_ROOT}/tests/twin/ci/repeat.sh")
# The determinism gate: the same binary on the same CPU repeats bit for bit.
add_test(NAME tw_repeat_components COMMAND "${TW_REPEAT}" $<TARGET_FILE:tw_golden_hash> run components)
add_test(NAME tw_repeat_isa COMMAND "${TW_REPEAT}" $<TARGET_FILE:tw_golden_hash> run isa)
add_test(NAME tw_golden_engine COMMAND tw_golden_hash check engine)
set(TW_QUICK_TESTS tw_math_canary tw_math_identity tw_math_atan2 tw_math_equal_sin_small
    tw_math_equal_cos_small tw_math_equal_asin_small tw_math_audit tw_model_m1 tw_race tw_world_pack
    tw_repeat_components tw_repeat_isa tw_golden_engine)
if(TW_FP_STRICT)
    # Without contraction the cross-build gates hold: the pinned hashes (tests/twin/golden/goldens.txt)
    # and every ISA variant equal to the baseline.
    add_test(NAME tw_golden_components COMMAND tw_golden_hash check components)
    add_test(NAME tw_golden_isa COMMAND tw_golden_hash check isa)
    list(APPEND TW_QUICK_TESTS tw_golden_components tw_golden_isa)
endif()
# The FMA audit: no FMA / rcp / rsqrt instruction in any twin object (gate); libm calls are listed.
add_test(NAME tw_object_audit COMMAND "${TW_REPO_ROOT}/tests/twin/ci/audit_twin_objects.sh" "${CMAKE_CURRENT_BINARY_DIR}")
list(APPEND TW_QUICK_TESTS tw_object_audit)
set_tests_properties(${TW_QUICK_TESTS} PROPERTIES LABELS "twin;quick" TIMEOUT 60)
# The engine suite skips (exit 77) while TwinStep is a stub.
set_tests_properties(tw_golden_engine PROPERTIES SKIP_RETURN_CODE 77)
if(FOREVERVALIDATOR_TWIN_EXHAUSTIVE_TESTS)
    foreach(fn sin cos sincos.sin sincos.cos sincos_small.sin sincos_small.cos asin asin_small acos atan exp
               sqrt rsqrt trunc floor ceil rint
               fmod_pow2.32 to_i32_trunc_sat to_u32_trunc_wrap from_u32 from_i32)
        add_test(NAME tw_math_exhaustive_${fn} COMMAND tw_math_tests exhaustive ${fn})
        set_tests_properties(tw_math_exhaustive_${fn} PROPERTIES LABELS "twin;exhaustive" TIMEOUT 60)
    endforeach()
endif()

if(TW_STANDALONE)
    return()
endif()

# ---- FV integration (root build only) -------------------------------------------------
# The FV-dependent twin code (public API, FV adapters) is compiled into
# forevervalidator_core, as the CUDA sources are, so there is no library cycle.
# The twin libraries are linked link-only: their include directories and flags
# do not reach FV's own sources.
set(TW_CORE_SOURCES
    "${TW_SRC}/api/twin_api.cpp"
    "${TW_SRC}/api/twin_bridge.cpp"
    "${TW_SRC}/fv/twin_timeline.cpp"
    "${TW_SRC}/fv/twin_search_candidates.cpp"
    "${TW_SRC}/fv/twin_search_executor.cpp"
    "${TW_SRC}/fv/compile_world.cpp"
    "${TW_SRC}/fv/compile_car.cpp"
    "${TW_SRC}/fv/compile_car_shapes.cpp"
    "${TW_SRC}/fv/twin_world_build.cpp"
    "${TW_SRC}/fv/import.cpp"
    # FV runtime clone <-> state words (A's import recipe). twin_harness also
    # compiles its own copy and links core: the linker never pulls this member there.
    "${TW_REPO_ROOT}/src/twin/common/twin_fv_state.cpp")
enable_testing()
# FV's CUDA search mutation and evaluator functions, copied for the host at
# build time so the twin search keeps generating FV's GPU candidates
# (docs/twin/m13-search.md).
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(TW_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(TW_SEARCH_EXTRACT "${TW_GENERATED_DIR}/twin_cuda_search_extract.inc")
set(TW_CUDA_SEARCH_SOURCE "${TW_REPO_ROOT}/src/simulation/backends/cuda/cuda_search_executor.cu")
add_custom_command(
    OUTPUT "${TW_SEARCH_EXTRACT}"
    COMMAND "${Python3_EXECUTABLE}" "${TW_SRC}/fv/extract_cuda_search.py"
            "${TW_CUDA_SEARCH_SOURCE}" "${TW_SEARCH_EXTRACT}"
    DEPENDS "${TW_SRC}/fv/extract_cuda_search.py" "${TW_CUDA_SEARCH_SOURCE}"
    COMMENT "Extracting FV CUDA search mutation code for the twin search"
    VERBATIM)
add_custom_target(tw_search_extract DEPENDS "${TW_SEARCH_EXTRACT}")
add_dependencies(forevervalidator_core tw_search_extract)
list(APPEND TW_INCLUDE_DIRS "${TW_GENERATED_DIR}")
target_sources(forevervalidator_core PRIVATE ${TW_CORE_SOURCES})
# FV core already compiles with -ffp-contract=off (MSVC: /fp:strict); add the
# rest of the twin set. No ISA flags: core stays baseline.
set(TW_CORE_FLAGS "")
if(NOT MSVC)
    set(TW_CORE_FLAGS -fno-fast-math -fno-strict-aliasing)
endif()
set_source_files_properties(${TW_CORE_SOURCES} TARGET_DIRECTORY forevervalidator_core PROPERTIES
    INCLUDE_DIRECTORIES "${TW_INCLUDE_DIRS};${TW_REPO_ROOT}"
    COMPILE_OPTIONS "${TW_CORE_FLAGS}")
target_link_libraries(forevervalidator_core PRIVATE "$<LINK_ONLY:tw_isa>" "$<LINK_ONLY:tw_engine>" "$<LINK_ONLY:tw_collide>" "$<LINK_ONLY:tw_race>"
    "$<LINK_ONLY:tw_car>" "$<LINK_ONLY:tw_world>" "$<LINK_ONLY:tw_model>")
if(FOREVERVALIDATOR_BUILD_TWIN_GPU)
    # the GPU session's per-candidate work on the device (fv/twin_search_device.cu): FV's mutation functions with their
    # CUDA qualifiers, extracted at build time like the host copy
    set(TW_SEARCH_EXTRACT_DEVICE "${TW_GENERATED_DIR}/twin_cuda_search_extract_device.inc")
    add_custom_command(
        OUTPUT "${TW_SEARCH_EXTRACT_DEVICE}"
        COMMAND "${Python3_EXECUTABLE}" "${TW_SRC}/fv/extract_cuda_search.py" --device
                "${TW_CUDA_SEARCH_SOURCE}" "${TW_SEARCH_EXTRACT_DEVICE}"
        DEPENDS "${TW_SRC}/fv/extract_cuda_search.py" "${TW_CUDA_SEARCH_SOURCE}"
        COMMENT "Extracting FV CUDA search mutation code for the twin search on the device"
        VERBATIM)
    add_library(tw_gpu_search STATIC "${TW_SRC}/fv/twin_search_device.cu" "${TW_SEARCH_EXTRACT_DEVICE}")
    target_include_directories(tw_gpu_search PRIVATE "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src"
        "${TW_GENERATED_DIR}" "${TW_SRC}/fv")
    target_link_libraries(tw_gpu_search PUBLIC tw_gpu)
    if(NOT TW_GPU_HIP)
        target_compile_options(tw_gpu_search PRIVATE ${TW_CUDA_MSVC_OPTIONS})
    endif()
    if(TW_GPU_HIP)
        set_source_files_properties("${TW_SRC}/fv/twin_search_device.cu" PROPERTIES LANGUAGE HIP)
        set_target_properties(tw_gpu_search PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF)
        target_include_directories(tw_gpu_search BEFORE PRIVATE "${TW_SRC}/gpu/hip/include" "${TW_SRC}/gpu/hip")
        target_compile_options(tw_gpu_search PRIVATE ${TW_GPU_HIP_OPTIONS})
    endif()
    set_target_properties(tw_gpu_search PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF JOB_POOL_COMPILE tw_gpu_pool)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_property(TARGET tw_gpu_search PROPERTY CUDA_ARCHITECTURES native)
    endif()
    target_compile_options(tw_gpu_search PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>" "$<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>" "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>")
    target_link_libraries(forevervalidator_core PRIVATE "$<LINK_ONLY:tw_gpu_search>")
    # the twin search session's GPU engine (tw_gpu_session.h SearchEngine; FOREVERVALIDATOR_TWIN_SEARCH_ENGINE=gpu)
    target_link_libraries(forevervalidator_core PRIVATE "$<LINK_ONLY:tw_gpu>")
    set_property(SOURCE "${TW_SRC}/fv/twin_search_executor.cpp" TARGET_DIRECTORY forevervalidator_core APPEND PROPERTY
        COMPILE_DEFINITIONS "FOREVERVALIDATOR_TWIN_GPU=1")
endif()
target_compile_definitions(forevervalidator_core PUBLIC FOREVERVALIDATOR_HAS_TWIN=1)
message(STATUS "ForeverValidator twin backend enabled (ISA ${FOREVERVALIDATOR_TWIN_ISA})")

# FV-side tests (no game assets needed).
add_executable(forevervalidator-twin-api-tests "${TW_REPO_ROOT}/tests/twin/twin_api_tests.cpp")
target_include_directories(forevervalidator-twin-api-tests PRIVATE
    "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src" ${TW_INCLUDE_DIRS})
target_link_libraries(forevervalidator-twin-api-tests PRIVATE forevervalidator_core)
forevervalidator_configure_target(forevervalidator-twin-api-tests)
add_test(NAME forevervalidator-twin-api COMMAND forevervalidator-twin-api-tests)

add_executable(forevervalidator-twin-search-tests "${TW_REPO_ROOT}/tests/twin/twin_search_tests.cpp")
target_include_directories(forevervalidator-twin-search-tests PRIVATE
    "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src" ${TW_INCLUDE_DIRS})
target_link_libraries(forevervalidator-twin-search-tests PRIVATE forevervalidator_core)
forevervalidator_configure_target(forevervalidator-twin-search-tests)
add_test(NAME forevervalidator-twin-search COMMAND forevervalidator-twin-search-tests)

# Session creation from an FV-loaded replay up to the first twin tick (needs
# the game packs and replays: run by hand, or set both cache variables to
# register it as a CTest with label twin;assets).
add_executable(forevervalidator-twin-handoff-tests "${TW_REPO_ROOT}/tests/twin/twin_handoff_tests.cpp")
target_include_directories(forevervalidator-twin-handoff-tests PRIVATE
    "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src" ${TW_INCLUDE_DIRS})
target_link_libraries(forevervalidator-twin-handoff-tests PRIVATE forevervalidator_native forevervalidator_core tw_model)
target_compile_definitions(forevervalidator-twin-handoff-tests PRIVATE
    TW_STADIUMCAR_JSON="${TW_REPO_ROOT}/src/twin/model/stadiumcar-values.json")
forevervalidator_configure_target(forevervalidator-twin-handoff-tests)
if(FOREVERVALIDATOR_ENABLE_HIP AND NOT CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    # FV's internal headers it reads include HIP's (as forevervalidator_core's sources do)
    target_compile_definitions(forevervalidator-twin-handoff-tests PRIVATE __HIP_PLATFORM_AMD__=1)
endif()
set(FOREVERVALIDATOR_TWIN_TEST_PACKS "" CACHE PATH "Game pack directory for the twin asset tests")
set(FOREVERVALIDATOR_TWIN_TEST_REPLAYS "" CACHE STRING "Replays (;-list) for the twin asset tests")
if(FOREVERVALIDATOR_TWIN_TEST_PACKS AND FOREVERVALIDATOR_TWIN_TEST_REPLAYS)
    add_test(NAME forevervalidator-twin-handoff COMMAND forevervalidator-twin-handoff-tests
        "${FOREVERVALIDATOR_TWIN_TEST_PACKS}" ${FOREVERVALIDATOR_TWIN_TEST_REPLAYS})
    set_tests_properties(forevervalidator-twin-handoff PROPERTIES LABELS "twin;assets" TIMEOUT 60)
endif()

# The per-candidate collision cache is result-invariant: replays tick by tick
# without a cache, with one, flushed at every call, and shared stale (assets).
add_executable(forevervalidator-twin-collide-cache-tests "${TW_REPO_ROOT}/tests/twin/twin_collide_cache_tests.cpp")
target_include_directories(forevervalidator-twin-collide-cache-tests PRIVATE
    "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src" ${TW_INCLUDE_DIRS})
target_link_libraries(forevervalidator-twin-collide-cache-tests PRIVATE forevervalidator_native forevervalidator_core tw_model)
forevervalidator_configure_target(forevervalidator-twin-collide-cache-tests)
if(FOREVERVALIDATOR_TWIN_TEST_PACKS AND FOREVERVALIDATOR_TWIN_TEST_REPLAYS)
    add_test(NAME forevervalidator-twin-collide-cache COMMAND forevervalidator-twin-collide-cache-tests
        "${FOREVERVALIDATOR_TWIN_TEST_PACKS}" ${FOREVERVALIDATOR_TWIN_TEST_REPLAYS})
    set_tests_properties(forevervalidator-twin-collide-cache PROPERTIES LABELS "twin;assets" TIMEOUT 60)
endif()


# All-twin tick cost over replays (M3 wired); a bench, not a test (no add_test).
add_executable(forevervalidator-twin-step-bench "${TW_REPO_ROOT}/tests/twin/twin_step_bench.cpp")
target_include_directories(forevervalidator-twin-step-bench PRIVATE
    "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src" ${TW_INCLUDE_DIRS})
target_link_libraries(forevervalidator-twin-step-bench PRIVATE forevervalidator_native forevervalidator_core tw_model)
forevervalidator_configure_target(forevervalidator-twin-step-bench)
# The same bench with the tick sources compiled for AVX2 / AVX-512 into the executable (what such a
# session runs; the bench drives tw::step directly, so it cannot go through the ISA tables).
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(TW_BENCH_TICK_SOURCES ${TW_ISA_TICK_SOURCES})
    list(REMOVE_ITEM TW_BENCH_TICK_SOURCES "${TW_SRC}/isa/tw_isa_tick_entry.cpp")
    set(TW_BENCH_FLAGS_avx2 ${TW_ISA_FLAGS_avx2})
    set(TW_BENCH_FLAGS_avx512 ${TW_ISA_FLAGS_avx512})
    foreach(isa avx2 avx512)
        add_library(tw_bench_${isa}_objects OBJECT ${TW_BENCH_TICK_SOURCES})
        target_include_directories(tw_bench_${isa}_objects PRIVATE ${TW_ISA_INCLUDE_DIRS})
        target_compile_options(tw_bench_${isa}_objects PRIVATE ${TW_FP_FLAGS} ${TW_BENCH_FLAGS_${isa}})
        target_compile_features(tw_bench_${isa}_objects PRIVATE cxx_std_17)
        add_executable(forevervalidator-twin-step-bench-${isa} "${TW_REPO_ROOT}/tests/twin/twin_step_bench.cpp"
            $<TARGET_OBJECTS:tw_bench_${isa}_objects>)
        target_include_directories(forevervalidator-twin-step-bench-${isa} PRIVATE
            "${TW_REPO_ROOT}" "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src" ${TW_INCLUDE_DIRS})
        target_link_libraries(forevervalidator-twin-step-bench-${isa} PRIVATE forevervalidator_native forevervalidator_core tw_model)
        forevervalidator_configure_target(forevervalidator-twin-step-bench-${isa})
    endforeach()
endif()

add_executable(forevervalidator-twin-backend-routing-tests
    "${TW_REPO_ROOT}/tests/twin/twin_backend_routing_tests.cpp")
target_include_directories(forevervalidator-twin-backend-routing-tests PRIVATE
    "${TW_REPO_ROOT}/include" "${TW_REPO_ROOT}/src")
target_link_libraries(forevervalidator-twin-backend-routing-tests PRIVATE forevervalidator_core)
forevervalidator_configure_target(forevervalidator-twin-backend-routing-tests)
add_test(NAME forevervalidator-twin-backend-routing COMMAND forevervalidator-twin-backend-routing-tests)
set_tests_properties(forevervalidator-twin-api forevervalidator-twin-backend-routing forevervalidator-twin-search
    PROPERTIES LABELS "twin;quick" TIMEOUT 60)




# A build without tests (an application's, BUILD_TESTING OFF) leaves the twin's test and tool programs out of
# "all"; each still builds by name.
if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    get_property(TW_DIRECTORY_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(tw_target IN LISTS TW_DIRECTORY_TARGETS)
        get_target_property(tw_type ${tw_target} TYPE)
        if(tw_type STREQUAL "EXECUTABLE")
            set_target_properties(${tw_target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()
    endforeach()
endif()
