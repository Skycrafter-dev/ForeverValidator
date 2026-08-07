#include "simulation/backends/cuda/cuda_backend.h"

#include <cuda_runtime_api.h>

#include <cstdio>

namespace forevervalidator::simulation {
namespace {

const char *CudaErrorText(cudaError_t error) noexcept {
    const char *text = cudaGetErrorString(error);
    return text != nullptr ? text : "unknown CUDA runtime error";
}

void SetCudaFailure(CudaBackendDiagnostics &result,
                    CudaBackendStatus status,
                    const char *operation,
                    cudaError_t error) noexcept {
    result.status = status;
    char buffer[320];
    std::snprintf(buffer, sizeof(buffer), "%s failed: %s (CUDA error %d)",
                  operation, CudaErrorText(error), static_cast<int>(error));
    result.diagnostic = buffer;
}

}  // namespace

CudaBackendDiagnostics QueryCompiledCudaRuntimeDiagnostics() noexcept {
    CudaBackendDiagnostics result;
    cudaError_t error = cudaRuntimeGetVersion(&result.runtimeVersion);
    if (error != cudaSuccess) {
        SetCudaFailure(result, CudaBackendStatus::RuntimeUnavailable,
                       "cudaRuntimeGetVersion", error);
        return result;
    }
    error = cudaDriverGetVersion(&result.driverVersion);
    if (error != cudaSuccess) {
        SetCudaFailure(result, CudaBackendStatus::RuntimeUnavailable,
                       "cudaDriverGetVersion", error);
        return result;
    }
    error = cudaGetDeviceCount(&result.deviceCount);
    if (error != cudaSuccess) {
        SetCudaFailure(result, CudaBackendStatus::RuntimeUnavailable,
                       "cudaGetDeviceCount", error);
        return result;
    }
    if (result.deviceCount == 0) {
        result.status = CudaBackendStatus::NoDevice;
        result.diagnostic = "CUDA runtime reported no CUDA-capable devices";
        return result;
    }
    error = cudaGetDevice(&result.selectedDevice);
    if (error != cudaSuccess) {
        SetCudaFailure(result, CudaBackendStatus::InitializationFailed,
                       "cudaGetDevice", error);
        return result;
    }
    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, result.selectedDevice);
    if (error != cudaSuccess) {
        SetCudaFailure(result, CudaBackendStatus::InitializationFailed,
                       "cudaGetDeviceProperties", error);
        return result;
    }
    result.computeCapabilityMajor = properties.major;
    result.computeCapabilityMinor = properties.minor;
    result.totalGlobalMemoryBytes =
            static_cast<std::uint64_t>(properties.totalGlobalMem);
    result.deviceName = properties.name;
    if (!IsCudaComputeCapabilitySupported(properties.major,
                                          properties.minor)) {
        result.status = CudaBackendStatus::UnsupportedDevice;
        result.diagnostic =
                "CUDA backend requires compute capability 6.1 or newer";
        return result;
    }
    error = cudaFree(nullptr);
    if (error != cudaSuccess) {
        SetCudaFailure(result, CudaBackendStatus::InitializationFailed,
                       "CUDA primary-context initialization", error);
        return result;
    }
    result.status = CudaBackendStatus::Ready;
    char buffer[512];
    const char *const specializationStatus =
            IsCudaSessionSpecializationSupported(properties.major,
                                                 properties.minor)
            ? ""
            : "; optional per-map fast kernel disabled on this device; "
              "using the generic CUDA search kernel";
    std::snprintf(
            buffer, sizeof(buffer),
            "CUDA device %d ready: %s, compute capability %d.%d, "
            "driver %d, runtime %d, %llu bytes global memory%s",
            result.selectedDevice, properties.name, properties.major,
            properties.minor, result.driverVersion, result.runtimeVersion,
            static_cast<unsigned long long>(result.totalGlobalMemoryBytes),
            specializationStatus);
    result.diagnostic = buffer;
    return result;
}

}  // namespace forevervalidator::simulation
