// Generated from src/simulation/backends/cuda/cuda_backend.cu by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_backend.h"

#include <hip/hip_runtime_api.h>

#include <cstdio>

namespace forevervalidator::simulation {
namespace {

const char *HipErrorText(hipError_t error) noexcept {
    const char *text = hipGetErrorString(error);
    return text != nullptr ? text : "unknown HIP runtime error";
}

void SetHipFailure(HipBackendDiagnostics &result,
                    HipBackendStatus status,
                    const char *operation,
                    hipError_t error) noexcept {
    result.status = status;
    char buffer[320];
    std::snprintf(buffer, sizeof(buffer), "%s failed: %s (HIP error %d)",
                  operation, HipErrorText(error), static_cast<int>(error));
    result.diagnostic = buffer;
}

}  // namespace

HipBackendDiagnostics QueryCompiledHipRuntimeDiagnostics() noexcept {
    HipBackendDiagnostics result;
    hipError_t error = hipRuntimeGetVersion(&result.runtimeVersion);
    if (error != hipSuccess) {
        SetHipFailure(result, HipBackendStatus::RuntimeUnavailable,
                       "hipRuntimeGetVersion", error);
        return result;
    }
    error = hipDriverGetVersion(&result.driverVersion);
    if (error != hipSuccess) {
        SetHipFailure(result, HipBackendStatus::RuntimeUnavailable,
                       "hipDriverGetVersion", error);
        return result;
    }
    error = hipGetDeviceCount(&result.deviceCount);
    if (error == hipErrorNoDevice) {
        result.status = HipBackendStatus::NoDevice;
        result.diagnostic = "HIP runtime reported no HIP-capable devices";
        return result;
    }
    if (error != hipSuccess) {
        SetHipFailure(result, HipBackendStatus::RuntimeUnavailable,
                       "hipGetDeviceCount", error);
        return result;
    }
    if (result.deviceCount == 0) {
        result.status = HipBackendStatus::NoDevice;
        result.diagnostic = "HIP runtime reported no HIP-capable devices";
        return result;
    }
    error = hipGetDevice(&result.selectedDevice);
    if (error != hipSuccess) {
        SetHipFailure(result, HipBackendStatus::InitializationFailed,
                       "hipGetDevice", error);
        return result;
    }
    hipDeviceProp_t properties{};
    error = hipGetDeviceProperties(&properties, result.selectedDevice);
    if (error != hipSuccess) {
        SetHipFailure(result, HipBackendStatus::InitializationFailed,
                       "hipGetDeviceProperties", error);
        return result;
    }
    result.computeCapabilityMajor = properties.major;
    result.computeCapabilityMinor = properties.minor;
    result.totalGlobalMemoryBytes =
            static_cast<std::uint64_t>(properties.totalGlobalMem);
    result.deviceName = properties.name;
#if defined(__HIP_PLATFORM_NVIDIA__)
    if (!HipBackendDiagnostics::SupportsComputeCapability(
            properties.major, properties.minor)) {
        result.status = HipBackendStatus::UnsupportedDevice;
        result.diagnostic =
                "HIP backend requires compute capability 5.0 or newer";
        return result;
    }
#endif
    error = hipFree(nullptr);
    if (error != hipSuccess) {
        SetHipFailure(result, HipBackendStatus::InitializationFailed,
                       "HIP primary-context initialization", error);
        return result;
    }
    result.status = HipBackendStatus::Ready;
    char buffer[384];
#if defined(__HIP_PLATFORM_NVIDIA__)
    std::snprintf(
            buffer, sizeof(buffer),
            "HIP device %d ready: %s, compute capability %d.%d, "
            "driver %d, runtime %d, %llu bytes global memory",
            result.selectedDevice, properties.name, properties.major,
            properties.minor, result.driverVersion, result.runtimeVersion,
            static_cast<unsigned long long>(result.totalGlobalMemoryBytes));
#else
    std::snprintf(buffer, sizeof(buffer),
                  "HIP device %d ready: %s, driver %d, runtime %d, "
                  "%llu bytes global memory",
                  result.selectedDevice, properties.name,
                  result.driverVersion, result.runtimeVersion,
                  static_cast<unsigned long long>(result.totalGlobalMemoryBytes));
#endif
    result.diagnostic = buffer;
    return result;
}

}  // namespace forevervalidator::simulation
