#ifndef FOREVERVALIDATOR_VULKAN_SEARCH_EXECUTOR_H
#define FOREVERVALIDATOR_VULKAN_SEARCH_EXECUTOR_H

#include <functional>
#include <memory>
#include <string>

#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/vulkan/vulkan_scene_storage.h"

namespace forevervalidator::simulation {

class VulkanSearchExecutor {
public:
    static std::unique_ptr<VulkanSearchExecutor> Create(
            const CudaSearchExecutorConfiguration &configuration,
            const VulkanDeviceScene &scene,
            const VulkanDeviceStaticConfiguration &staticConfiguration,
            std::string *diagnostic) noexcept;

    ~VulkanSearchExecutor();
    VulkanSearchExecutor(VulkanSearchExecutor &&) noexcept;
    VulkanSearchExecutor &operator=(VulkanSearchExecutor &&) noexcept;
    VulkanSearchExecutor(const VulkanSearchExecutor &) = delete;
    VulkanSearchExecutor &operator=(const VulkanSearchExecutor &) = delete;

    CudaSearchBatchExecution EvaluateBaseline() noexcept;
    CudaSearchBatchExecution EvaluateBaseline(
            const std::function<bool()> &cancellationRequested) noexcept;
    CudaSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool cancellationRequested) noexcept;
    CudaSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            const std::function<bool()> &cancellationRequested) noexcept;
    bool ReserveBatchCapacity(
            std::uint32_t candidateCount,
            std::string *diagnostic) noexcept;
    bool UpdateConditionTimes(
            double lastImprovementTimeSeconds,
            double lastRestartTimeSeconds) noexcept;
    std::uint32_t BatchCapacity() const noexcept;

private:
    struct Impl;
    explicit VulkanSearchExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace forevervalidator::simulation

#endif
