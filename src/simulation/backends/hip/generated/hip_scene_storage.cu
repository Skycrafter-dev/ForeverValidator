// Generated from src/simulation/backends/cuda/cuda_scene_storage.cu by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_scene_storage.h"

#include <hip/hip_runtime_api.h>

#include <chrono>

namespace forevervalidator::simulation {

bool UploadHipSceneBytes(const std::byte *source,
                          std::size_t size,
                          void **destination,
                          double *milliseconds,
                          std::string *diagnostic) noexcept {
    if (source == nullptr || size == 0u || destination == nullptr) {
        if (diagnostic != nullptr) {
            *diagnostic = "invalid HIP scene upload request";
        }
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    hipError_t error = hipMalloc(destination, size);
    if (error == hipSuccess) {
        error = hipMemcpy(
                *destination, source, size, hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        error = hipDeviceSynchronize();
    }
    const auto end = std::chrono::steady_clock::now();
    if (milliseconds != nullptr) {
        *milliseconds =
                std::chrono::duration<double, std::milli>(
                        end - start).count();
    }
    if (error != hipSuccess) {
        if (*destination != nullptr) {
            hipFree(*destination);
            *destination = nullptr;
        }
        if (diagnostic != nullptr) {
            *diagnostic = std::string("HIP scene upload failed: ") +
                    hipGetErrorString(error);
        }
        return false;
    }
    return true;
}

void ReleaseHipSceneBytes(void *allocation) noexcept {
    if (allocation != nullptr) {
        hipFree(allocation);
    }
}

}  // namespace forevervalidator::simulation
