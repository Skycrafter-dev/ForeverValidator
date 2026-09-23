#ifndef FOREVERVALIDATOR_VULKAN_SCENE_STORAGE_H
#define FOREVERVALIDATOR_VULKAN_SCENE_STORAGE_H

#include <cstdint>
#include <cstddef>
#include <string>

#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_static_configuration_storage.h"
#include "simulation/backends/vulkan/vulkan_compute_runtime.h"

namespace forevervalidator::simulation {

struct CudaCandidatePhysicsState;

struct VulkanTransferMetrics {
    bool success = false;
    std::uint64_t hostPackedBytes = 0u;
    std::uint64_t deviceBytes = 0u;
    double packMilliseconds = 0.0;
    double uploadMilliseconds = 0.0;
    std::string diagnostic;
};

class VulkanDeviceScene {
public:
    VulkanTransferMetrics Upload(const CudaHostScene &source) noexcept;
    VulkanTransferMetrics UploadPacked(
            const std::byte *data, std::size_t size) noexcept;
    void Reset() noexcept;
    bool Ready() const noexcept { return buffer_ != nullptr; }
    std::uint64_t SceneHash() const noexcept { return sceneHash_; }
    std::uint64_t DeviceBytes() const noexcept;

private:
    friend class VulkanTimelineExecutorAccess;
    friend class VulkanSearchExecutorAccess;
    vulkan::BufferHandle buffer_;
    std::uint64_t sceneHash_ = 0u;
};

class VulkanDeviceStaticConfiguration {
public:
    VulkanTransferMetrics Upload(
            const CudaHostStaticConfiguration &source) noexcept;
    VulkanTransferMetrics UploadPacked(
            const std::byte *data, std::size_t size) noexcept;
    void Reset() noexcept;
    bool Ready() const noexcept { return buffer_ != nullptr; }
    std::uint64_t ConfigurationHash() const noexcept {
        return configurationHash_;
    }
    CudaHandlingSpecialization HandlingSpecialization() const noexcept {
        return handlingSpecialization_;
    }
    bool SteadyVelocityFactsMatch(
            const CudaCandidatePhysicsState &state) const noexcept;
    std::uint64_t DeviceBytes() const noexcept;

private:
    friend class VulkanTimelineExecutorAccess;
    friend class VulkanSearchExecutorAccess;
    vulkan::BufferHandle buffer_;
    std::uint64_t configurationHash_ = 0u;
    CudaHandlingSpecialization handlingSpecialization_ =
            CudaHandlingSpecialization::Generic;
    VehicleWheelSetDefinition wheels_{};
    bool steadyVelocityStaticFacts_ = false;
};

}  // namespace forevervalidator::simulation

#endif
