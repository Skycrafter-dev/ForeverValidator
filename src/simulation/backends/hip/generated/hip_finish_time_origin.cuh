// Generated from src/simulation/backends/cuda/cuda_finish_time_origin.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_FINISH_TIME_ORIGIN_CUH
#define FOREVERVALIDATOR_HIP_FINISH_TIME_ORIGIN_CUH

#include <cstdint>

namespace forevervalidator::simulation::hip::finish {

__host__ __device__ constexpr std::uint64_t TickStartNanoseconds(
        std::uint32_t tickTimeMs) noexcept {
    return static_cast<std::uint64_t>(tickTimeMs) * 1000000u;
}

}  // namespace forevervalidator::simulation::hip::finish

#endif
