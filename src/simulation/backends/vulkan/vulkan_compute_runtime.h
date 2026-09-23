#ifndef FOREVERVALIDATOR_VULKAN_COMPUTE_RUNTIME_H
#define FOREVERVALIDATOR_VULKAN_COMPUTE_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace forevervalidator::simulation::vulkan {

enum class RuntimeStatus : std::uint8_t {
    NotCompiled,
    LoaderUnavailable,
    NoDevice,
    UnsupportedDevice,
    InitializationFailed,
    Ready,
};

struct RuntimeDiagnostics {
    RuntimeStatus status = RuntimeStatus::NotCompiled;
    std::uint32_t apiVersion = 0u;
    std::uint32_t driverVersion = 0u;
    std::uint32_t vendorId = 0u;
    std::uint32_t deviceId = 0u;
    std::uint64_t deviceLocalMemoryBytes = 0u;
    std::uint32_t subgroupSize = 0u;
    std::string deviceName;
    std::string driverName;
    std::string diagnostic;

    bool IsReady() const noexcept { return status == RuntimeStatus::Ready; }
};

class Buffer;
using BufferHandle = std::shared_ptr<Buffer>;

RuntimeDiagnostics QueryRuntimeDiagnostics() noexcept;
bool SupportsExactSearchPhysics() noexcept;

bool UploadImmutableBuffer(
        const std::byte *data,
        std::size_t size,
        BufferHandle *destination,
        double *milliseconds,
        std::string *diagnostic) noexcept;

std::uint64_t BufferSize(const BufferHandle &buffer) noexcept;

struct TimelineKernelRequest {
    BufferHandle scene;
    BufferHandle configuration;
    const void *states = nullptr;
    std::size_t stateBytes = 0u;
    const void *descriptors = nullptr;
    std::size_t descriptorBytes = 0u;
    const void *ticks = nullptr;
    std::size_t tickBytes = 0u;
    const void *initialResults = nullptr;
    std::size_t resultBytes = 0u;
    const void *cancellation = nullptr;
    std::size_t cancellationBytes = 0u;
    std::size_t observationBytes = 0u;
    std::size_t scratchBytes = 0u;
    std::uint32_t candidateCount = 0u;
    std::uint32_t stateStride = 0u;
    bool fullState = true;
    bool finishRefinement = false;

    void *outputStates = nullptr;
    void *outputResults = nullptr;
    void *outputObservations = nullptr;
};

struct TimelineKernelMetrics {
    std::uint64_t uploadBytes = 0u;
    std::uint64_t downloadBytes = 0u;
    std::uint64_t peakDeviceBytes = 0u;
    double allocationMilliseconds = 0.0;
    double transferMilliseconds = 0.0;
    double kernelMilliseconds = 0.0;
    double synchronizationMilliseconds = 0.0;
};

bool ExecuteTimelineKernel(
        const TimelineKernelRequest &request,
        TimelineKernelMetrics *metrics,
        std::string *diagnostic) noexcept;

enum class ComputeKernel : std::uint8_t {
    SearchGenerate,
    SearchInitialize,
    SearchPrepare,
    SearchPhysics,
    SearchPhysicsSteadyVelocityWater,
    SearchPhysicsForcePhase,
    SearchPhysicsDetectPhase,
    SearchPhysicsRespondPhase,
    Timeline,
    SearchEvaluate,
};

struct ComputeRegion {
    const void *upload = nullptr;
    void *download = nullptr;
    std::size_t bytes = 0u;
};

struct ComputeRegionAddressBinding {
    std::size_t parameterOffset = 0u;
    std::size_t regionIndex = 0u;
};

struct ComputeExternalAddressBinding {
    std::size_t parameterOffset = 0u;
    BufferHandle buffer;
};

struct ComputeDispatch {
    ComputeKernel kernel = ComputeKernel::SearchGenerate;
    std::uint32_t workItemCount = 0u;
    std::uint32_t tickIndex = 0u;
    std::uint32_t candidateBase = 0u;
    std::uint32_t tickCount = 0u;
};

struct ComputeTimelineBindings {
    BufferHandle scene;
    BufferHandle configuration;
    std::size_t statesRegion = static_cast<std::size_t>(-1);
    std::size_t descriptorsRegion = static_cast<std::size_t>(-1);
    std::size_t ticksRegion = static_cast<std::size_t>(-1);
    std::size_t observationsRegion = static_cast<std::size_t>(-1);
    std::size_t resultsRegion = static_cast<std::size_t>(-1);
    std::size_t scratchRegion = static_cast<std::size_t>(-1);
    std::size_t cancellationRegion = static_cast<std::size_t>(-1);
    std::uint32_t stateStride = 0u;
    bool fullState = true;
};

struct ComputeKernelRequest {
    const void *parameters = nullptr;
    std::size_t parameterBytes = 0u;
    std::vector<ComputeRegion> regions;
    std::vector<ComputeRegionAddressBinding> regionBindings;
    std::vector<ComputeExternalAddressBinding> externalBindings;
    std::vector<ComputeDispatch> dispatches;
    std::size_t maxDispatchesPerSubmission = 0u;
    ComputeTimelineBindings timeline;
    std::size_t dynamicCancellationRegion =
            static_cast<std::size_t>(-1);
    std::function<bool()> cancellationRequested;
};

struct ComputeKernelMetrics {
    std::uint64_t uploadBytes = 0u;
    std::uint64_t downloadBytes = 0u;
    std::uint64_t peakDeviceBytes = 0u;
    double allocationMilliseconds = 0.0;
    double transferMilliseconds = 0.0;
    double kernelMilliseconds = 0.0;
    double synchronizationMilliseconds = 0.0;
};

bool ExecuteComputeKernels(
        const ComputeKernelRequest &request,
        ComputeKernelMetrics *metrics,
        std::string *diagnostic) noexcept;

}  // namespace forevervalidator::simulation::vulkan

#endif
