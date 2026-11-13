#ifndef TW_GPU_HIP_TW_HIP_COMPAT_H
#define TW_GPU_HIP_TW_HIP_COMPAT_H

// The twin's GPU engine (written in CUDA) compiled as HIP: force-included into every HIP translation
// unit of tw_gpu and tw_gpu_search, with include/ (cuda_runtime.h, cub/...) ahead of the system's.
// The CUDA runtime names map to HIP's; warp operations given CUDA's 32-bit masks call HIP's 64-bit
// ones (the engine runs one wave of 32 lanes: built with -mno-wavefrontsize64).

#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaFuncAttributes hipFuncAttributes
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
// CUDA takes the kernel itself; HIP its address.
template <typename Kernel>
inline hipError_t TwHipFuncGetAttributes(hipFuncAttributes *attributes, Kernel kernel) {
    return hipFuncGetAttributes(attributes, reinterpret_cast<const void *>(kernel));
}
#define cudaFuncGetAttributes TwHipFuncGetAttributes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
// CUDA's warp operations with a 32-bit mask: HIP's take a 64-bit one.
template <typename T>
__device__ inline T __shfl_sync(unsigned mask, T var, int lane, int width = warpSize) {
    return __shfl_sync(static_cast<unsigned long long>(mask), var, lane, width);
}
template <typename T>
__device__ inline T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize) {
    return __shfl_up_sync(static_cast<unsigned long long>(mask), var, delta, width);
}
template <typename T>
__device__ inline T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize) {
    return __shfl_down_sync(static_cast<unsigned long long>(mask), var, delta, width);
}
template <typename T>
__device__ inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize) {
    return __shfl_xor_sync(static_cast<unsigned long long>(mask), var, laneMask, width);
}
__device__ inline unsigned __ballot_sync(unsigned mask, int predicate) {
    return static_cast<unsigned>(__ballot_sync(static_cast<unsigned long long>(mask), predicate));
}
__device__ inline int __any_sync(unsigned mask, int predicate) {
    return __any_sync(static_cast<unsigned long long>(mask), predicate);
}
__device__ inline int __all_sync(unsigned mask, int predicate) {
    return __all_sync(static_cast<unsigned long long>(mask), predicate);
}
__device__ inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {
    return __reduce_add_sync(static_cast<unsigned long long>(mask), value);
}
__device__ inline int __reduce_add_sync(unsigned mask, int value) {
    return __reduce_add_sync(static_cast<unsigned long long>(mask), value);
}
__device__ inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {
    return __reduce_max_sync(static_cast<unsigned long long>(mask), value);
}
__device__ inline int __reduce_max_sync(unsigned mask, int value) {
    return __reduce_max_sync(static_cast<unsigned long long>(mask), value);
}
__device__ inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {
    return __reduce_min_sync(static_cast<unsigned long long>(mask), value);
}
__device__ inline int __reduce_min_sync(unsigned mask, int value) {
    return __reduce_min_sync(static_cast<unsigned long long>(mask), value);
}
// CUDA's cache hints: ordinary accesses. The lockstep queue reads its entries after
// __threadfence(), whose device-scope acquire also invalidates AMD's non-coherent caches.
template <typename T> __device__ inline T __ldcg(const T *address) { return *address; }
template <typename T> __device__ inline void __stcg(T *address, T value) { *address = value; }
template <typename T> __device__ inline void __stcs(T *address, T value) { *address = value; }
__device__ inline void __syncwarp(unsigned mask) {
    __syncwarp(static_cast<unsigned long long>(mask));
}
#endif

// A kernel parameter in CUDA's constant bank: an ordinary parameter here.
#define __grid_constant__

#endif
