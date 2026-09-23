// Generated from src/simulation/backends/cuda/cuda_rng.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_RNG_CUH
#define FOREVERVALIDATOR_HIP_RNG_CUH

#include "simulation/backends/hip/generated/hip_exact_math.cuh"

namespace forevervalidator::simulation::hip::rng {

__device__ inline std::uint32_t Next15Bits(
        std::uint32_t &state) {
    state = state * 214013u + 2531011u;
    return (state >> 16u) & 0x7fffu;
}

__device__ inline std::uint32_t Natural(
        std::uint32_t &state,
        std::uint32_t minimum,
        std::uint32_t maximum) {
    const std::uint32_t randomValue = Next15Bits(state);
    const std::uint32_t range = maximum - minimum + 1u;
    const float unitRandom =
            exact::FromUnsignedInteger(randomValue) / 32768.0f;
    const float scaledRange =
            unitRandom * exact::FromUnsignedInteger(range);
    const float result =
            scaledRange + exact::FromUnsignedInteger(minimum);
    return static_cast<std::uint32_t>(result);
}

}  // namespace forevervalidator::simulation::hip::rng

#endif
