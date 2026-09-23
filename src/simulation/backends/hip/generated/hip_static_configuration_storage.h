// Generated from src/simulation/backends/cuda/cuda_static_configuration_storage.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_STATIC_CONFIGURATION_STORAGE_H
#define FOREVERVALIDATOR_HIP_STATIC_CONFIGURATION_STORAGE_H

#include <cstdint>
#include <string>

#include "simulation/backends/hip/generated/hip_static_configuration.h"

namespace forevervalidator::simulation {

struct HipStaticConfigurationTransferMetrics {
    bool success = false;
    std::uint64_t hostPackedBytes = 0u;
    std::uint64_t deviceBytes = 0u;
    double packMilliseconds = 0.0;
    double uploadMilliseconds = 0.0;
    std::string diagnostic;
};

class HipDeviceStaticConfiguration {
public:
    HipDeviceStaticConfiguration() = default;
    ~HipDeviceStaticConfiguration();
    HipDeviceStaticConfiguration(
            HipDeviceStaticConfiguration &&other) noexcept;
    HipDeviceStaticConfiguration &operator=(
            HipDeviceStaticConfiguration &&other) noexcept;

    HipDeviceStaticConfiguration(
            const HipDeviceStaticConfiguration &) = delete;
    HipDeviceStaticConfiguration &operator=(
            const HipDeviceStaticConfiguration &) = delete;

    HipStaticConfigurationTransferMetrics Upload(
            const HipHostStaticConfiguration &source) noexcept;
    void Reset() noexcept;
    bool Ready() const noexcept { return deviceData_ != nullptr; }
    std::uint64_t ConfigurationHash() const noexcept {
        return configurationHash_;
    }
    std::uint64_t DeviceBytes() const noexcept { return deviceBytes_; }
    const void *DeviceData() const noexcept { return deviceData_; }

private:
    void *deviceData_ = nullptr;
    std::uint64_t deviceBytes_ = 0u;
    std::uint64_t configurationHash_ = 0u;
};

}  // namespace forevervalidator::simulation

#endif
