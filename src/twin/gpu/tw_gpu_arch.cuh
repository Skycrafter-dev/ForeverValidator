// Twin GPU engine on GPUs older than what it uses (it is built for every CUDA
// generation from 5.0). HIP: gpu/hip/tw_hip_compat.h.
#ifndef TW_GPU_ARCH_CUH
#define TW_GPU_ARCH_CUH

// __grid_constant__ (sm 7.0): a parameter read where it is; earlier GPUs get a
// copy of it, same values.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define TW_GRID_CONSTANT
#else
#define TW_GRID_CONSTANT __grid_constant__
#endif

// The warp reductions of sm 8.0 (__reduce_{add,min,max}_sync). Integer
// results, so the same as the hardware ones whatever the order. Lanes are
// walked by the mask, not by a butterfly: callers pass partial masks.

#if defined(__CUDACC__) && !defined(__HIPCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

namespace tw::gpu::warp {

template <typename T, typename Op>
__device__ inline T Reduce(unsigned mask, T value, Op op) {
    T result = value;
    bool first = true;
    for (unsigned rest = mask; rest != 0u; rest &= rest - 1u) {
        const T lane = __shfl_sync(mask, value, __ffs(static_cast<int>(rest)) - 1);
        result = first ? lane : op(result, lane);
        first = false;
    }
    return result;
}

}  // namespace tw::gpu::warp

__device__ inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {
    return tw::gpu::warp::Reduce(mask, value, [](unsigned a, unsigned b) { return a + b; });
}
__device__ inline int __reduce_add_sync(unsigned mask, int value) {
    return tw::gpu::warp::Reduce(mask, value, [](int a, int b) { return a + b; });
}
__device__ inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {
    return tw::gpu::warp::Reduce(mask, value, [](unsigned a, unsigned b) { return a < b ? a : b; });
}
__device__ inline int __reduce_min_sync(unsigned mask, int value) {
    return tw::gpu::warp::Reduce(mask, value, [](int a, int b) { return a < b ? a : b; });
}
__device__ inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {
    return tw::gpu::warp::Reduce(mask, value, [](unsigned a, unsigned b) { return a > b ? a : b; });
}
__device__ inline int __reduce_max_sync(unsigned mask, int value) {
    return tw::gpu::warp::Reduce(mask, value, [](int a, int b) { return a > b ? a : b; });
}

#endif
#endif
