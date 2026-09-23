#ifndef FOREVERVALIDATOR_VULKAN_TIMELINE_EXECUTOR_H
#define FOREVERVALIDATOR_VULKAN_TIMELINE_EXECUTOR_H

#include "simulation/backends/cuda/cuda_timeline_executor.h"
#include "simulation/backends/vulkan/vulkan_scene_storage.h"

namespace forevervalidator::simulation {

using VulkanControlTick = CudaControlTick;
using VulkanTimelineObservation = CudaTimelineObservation;
using VulkanTimelineStatus = CudaTimelineStatus;
using VulkanCandidateTimelineInput = CudaCandidateTimelineInput;
using VulkanCandidateTimelineOutput = CudaCandidateTimelineOutput;
using VulkanTimelineExecutionMetrics = CudaTimelineExecutionMetrics;
using VulkanTimelineBatchResult = CudaTimelineBatchResult;

VulkanTimelineBatchResult ExecuteVulkanTimelineBatch(
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &configuration,
        const std::vector<VulkanCandidateTimelineInput> &candidates,
        bool cancellationRequested = false) noexcept;

VulkanTimelineBatchResult ExecuteVulkanFinishTimelineBatch(
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &configuration,
        const std::vector<VulkanCandidateTimelineInput> &candidates,
        bool cancellationRequested = false) noexcept;

const char *VulkanTimelineStatusName(VulkanTimelineStatus status) noexcept;

}  // namespace forevervalidator::simulation

#endif
