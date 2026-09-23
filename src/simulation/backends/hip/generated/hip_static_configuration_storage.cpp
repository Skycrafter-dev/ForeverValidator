// Generated from src/simulation/backends/cuda/cuda_static_configuration_storage.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_static_configuration_storage.h"

#include <chrono>
#include <utility>
#include <vector>

namespace forevervalidator::simulation {

#if FOREVERVALIDATOR_HAS_HIP
bool UploadHipSceneBytes(const std::byte *source,
                          std::size_t size,
                          void **destination,
                          double *milliseconds,
                          std::string *diagnostic) noexcept;
void ReleaseHipSceneBytes(void *allocation) noexcept;
#endif

HipDeviceStaticConfiguration::~HipDeviceStaticConfiguration() {
    Reset();
}

HipDeviceStaticConfiguration::HipDeviceStaticConfiguration(
        HipDeviceStaticConfiguration &&other) noexcept
    : deviceData_(std::exchange(other.deviceData_, nullptr)),
      deviceBytes_(std::exchange(other.deviceBytes_, 0u)),
      configurationHash_(
              std::exchange(other.configurationHash_, 0u)) {}

HipDeviceStaticConfiguration &
HipDeviceStaticConfiguration::operator=(
        HipDeviceStaticConfiguration &&other) noexcept {
    if (this != &other) {
        Reset();
        deviceData_ = std::exchange(other.deviceData_, nullptr);
        deviceBytes_ = std::exchange(other.deviceBytes_, 0u);
        configurationHash_ =
                std::exchange(other.configurationHash_, 0u);
    }
    return *this;
}

HipStaticConfigurationTransferMetrics
HipDeviceStaticConfiguration::Upload(
        const HipHostStaticConfiguration &source) noexcept {
    Reset();
    HipStaticConfigurationTransferMetrics result;
    const auto packStart = std::chrono::steady_clock::now();
    std::vector<std::byte> bytes;
    if (!PackHipStaticConfiguration(source, &bytes)) {
        result.diagnostic =
                "HIP static configuration packing failed";
        return result;
    }
    const auto packEnd = std::chrono::steady_clock::now();
    result.packMilliseconds =
            std::chrono::duration<double, std::milli>(
                    packEnd - packStart).count();
    result.hostPackedBytes = bytes.size();
#if FOREVERVALIDATOR_HAS_HIP
    if (!UploadHipSceneBytes(
                bytes.data(), bytes.size(), &deviceData_,
                &result.uploadMilliseconds, &result.diagnostic)) {
        Reset();
        return result;
    }
    deviceBytes_ = bytes.size();
    configurationHash_ = source.deterministicHash;
    result.deviceBytes = deviceBytes_;
    result.success = true;
    result.diagnostic =
            "HIP immutable vehicle and environment configuration uploaded";
#else
    result.diagnostic =
            "HIP static configuration upload unavailable in a CPU-only build";
#endif
    return result;
}

void HipDeviceStaticConfiguration::Reset() noexcept {
#if FOREVERVALIDATOR_HAS_HIP
    ReleaseHipSceneBytes(deviceData_);
#endif
    deviceData_ = nullptr;
    deviceBytes_ = 0u;
    configurationHash_ = 0u;
}

}  // namespace forevervalidator::simulation
