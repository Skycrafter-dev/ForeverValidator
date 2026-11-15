// Twin GPU engine: host-side CUDA helpers (error strings, owned device buffers).
#ifndef TW_GPU_UTIL_CUH
#define TW_GPU_UTIL_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace tw::gpu {

inline bool Check(cudaError_t e, const char *what, std::string *error) {
    if (e == cudaSuccess) return true;
    if (error != nullptr) *error = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

// One device allocation, freed with its owner.
struct DevBuffer {
    void *ptr = nullptr;
    std::size_t bytes = 0;
    DevBuffer() = default;
    DevBuffer(const DevBuffer &) = delete;
    DevBuffer &operator=(const DevBuffer &) = delete;
    ~DevBuffer() { Free(); }
    void Free() {
        if (ptr != nullptr) cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
    }
    bool Alloc(std::size_t n, std::string *error) {
        Free();
        if (n == 0u) return true;
        if (!Check(cudaMalloc(&ptr, n), "cudaMalloc", error)) return ptr = nullptr, false;
        bytes = n;
        return true;
    }
    bool Put(const void *src, std::size_t n, std::string *error) {
        return Alloc(n, error) && (n == 0u || Check(cudaMemcpy(ptr, src, n, cudaMemcpyHostToDevice), "cudaMemcpy H2D", error));
    }
    bool Get(void *dst, std::size_t n, std::string *error) const {
        return n == 0u || Check(cudaMemcpy(dst, ptr, n, cudaMemcpyDeviceToHost), "cudaMemcpy D2H", error);
    }
};

}  // namespace tw::gpu

#endif
