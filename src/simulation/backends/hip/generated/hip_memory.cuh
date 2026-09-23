// Generated from src/simulation/backends/cuda/cuda_memory.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_MEMORY_CUH
#define FOREVERVALIDATOR_HIP_MEMORY_CUH

#include <cstddef>

namespace forevervalidator::simulation::hip::memory {

template<std::size_t Count, typename Destination, typename Source>
__device__ inline void CopyBytes(Destination *destination,
                                 const Source *source) {
    static_assert(Count <= sizeof(Destination));
    static_assert(Count <= sizeof(Source));
    auto *destinationBytes =
            reinterpret_cast<unsigned char *>(destination);
    const auto *sourceBytes =
            reinterpret_cast<const unsigned char *>(source);
#pragma unroll
    for (std::size_t index = 0u; index < Count; ++index) {
        destinationBytes[index] = sourceBytes[index];
    }
}

}  // namespace forevervalidator::simulation::hip::memory

#endif
