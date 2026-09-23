#include "simulation/backends/vulkan/vulkan_timeline_executor.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "simulation/backends/cuda/cuda_collision_layout.h"

namespace forevervalidator::simulation {
namespace {

constexpr std::uint64_t MaximumTotalTicks = 100000000u;
constexpr std::uint64_t MaximumTotalObservations = 10000000u;
// Keep individual submissions comfortably below desktop GPU watchdog limits.
// Candidate state carries every cross-tick value, so dispatch chunking is
// semantically identical to one long-running kernel invocation.
constexpr std::size_t MaximumTicksPerDispatch = 256u;
constexpr std::size_t FinishExactTicksPerDispatch = 1u;

struct VulkanTimelineDescriptor {
    std::uint64_t firstTick = 0u;
    std::uint32_t tickCount = 0u;
    std::uint32_t padding0 = 0u;
    std::uint64_t firstObservation = 0u;
    std::uint32_t observationCapacity = 0u;
    std::uint32_t padding1 = 0u;
};

struct VulkanTimelineDeviceResult {
    VulkanTimelineStatus status = VulkanTimelineStatus::InvalidArgument;
    std::uint32_t failureTick = UINT32_MAX;
    std::uint32_t failureDetail = 0u;
    std::uint32_t executedTickCount = 0u;
    std::uint32_t executedRespawnCount = 0u;
    std::uint32_t observationCount = 0u;
};

struct VulkanFinishRefinementWork {
    std::uint32_t status = 0u;
    float fullDt = 0.0f;
    double substepStartNs = 0.0;
    FinishTimeEstimate estimate{};
};

static_assert(sizeof(VulkanTimelineDescriptor) == 32u);
static_assert(sizeof(VulkanTimelineDeviceResult) == 24u);
static_assert(sizeof(VulkanFinishRefinementWork) == 40u);

template<typename T>
bool ByteSize(std::size_t count, std::size_t *result) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    *result = count * sizeof(T);
    return true;
}

}  // namespace

class VulkanTimelineExecutorAccess {
public:
    static vulkan::BufferHandle Scene(const VulkanDeviceScene &value) {
        return value.buffer_;
    }
    static vulkan::BufferHandle Configuration(
            const VulkanDeviceStaticConfiguration &value) {
        return value.buffer_;
    }
};

const char *VulkanTimelineStatusName(
        VulkanTimelineStatus status) noexcept {
    return CudaTimelineStatusName(status);
}

static VulkanTimelineBatchResult ExecuteVulkanTimelineSingleDispatch(
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &configuration,
        const std::vector<VulkanCandidateTimelineInput> &candidates,
        bool cancellationRequested) noexcept {
    VulkanTimelineBatchResult result;
    if (!scene.Ready() || !configuration.Ready() ||
        candidates.empty() ||
        candidates.size() >
                std::numeric_limits<std::uint32_t>::max()) {
        result.status = VulkanTimelineStatus::InvalidArgument;
        result.diagnostic = "invalid Vulkan timeline batch";
        return result;
    }
    try {
        std::vector<CudaCandidateState> states;
        std::vector<VulkanTimelineDescriptor> descriptors;
        std::vector<VulkanControlTick> ticks;
        std::uint64_t totalTicks = 0u;
        std::uint64_t totalObservations = 0u;
        states.reserve(candidates.size());
        descriptors.reserve(candidates.size());
        for (const VulkanCandidateTimelineInput &candidate : candidates) {
            if (candidate.ticks.size() > MaximumTotalTicks - totalTicks ||
                candidate.ticks.size() >
                        std::numeric_limits<std::uint32_t>::max()) {
                result.status = VulkanTimelineStatus::CapacityExceeded;
                result.diagnostic =
                        "Vulkan timeline tick capacity exceeded";
                return result;
            }
            const std::uint64_t candidateObservations =
                    static_cast<std::uint64_t>(std::count_if(
                            candidate.ticks.begin(),
                            candidate.ticks.end(),
                            [](const VulkanControlTick &tick) {
                                return tick.observe;
                            }));
            if (candidateObservations >
                    MaximumTotalObservations - totalObservations) {
                result.status = VulkanTimelineStatus::CapacityExceeded;
                result.diagnostic =
                        "Vulkan observation capacity exceeded";
                return result;
            }
            VulkanTimelineDescriptor descriptor;
            descriptor.firstTick = totalTicks;
            descriptor.tickCount = static_cast<std::uint32_t>(
                    candidate.ticks.size());
            descriptor.firstObservation = totalObservations;
            descriptor.observationCapacity =
                    static_cast<std::uint32_t>(candidateObservations);
            states.push_back(candidate.initialState);
            descriptors.push_back(descriptor);
            ticks.insert(ticks.end(), candidate.ticks.begin(),
                         candidate.ticks.end());
            totalTicks += candidate.ticks.size();
            totalObservations += candidateObservations;
        }

        std::vector<VulkanTimelineObservation> observations(
                totalObservations);
        std::vector<VulkanTimelineDeviceResult> deviceResults(
                candidates.size());

        std::size_t stateBytes = 0u;
        std::size_t descriptorBytes = 0u;
        std::size_t tickBytes = 0u;
        std::size_t observationBytes = 0u;
        std::size_t resultBytes = 0u;
        std::size_t scratchBytes = 0u;
        if (!ByteSize<CudaCandidateState>(states.size(), &stateBytes) ||
            !ByteSize<VulkanTimelineDescriptor>(
                    descriptors.size(), &descriptorBytes) ||
            !ByteSize<VulkanControlTick>(ticks.size(), &tickBytes) ||
            !ByteSize<VulkanTimelineObservation>(
                    observations.size(), &observationBytes) ||
            !ByteSize<VulkanTimelineDeviceResult>(
                    deviceResults.size(), &resultBytes) ||
            !ByteSize<cuda::collision::CudaCollisionScratch>(
                    candidates.size(), &scratchBytes)) {
            result.status = VulkanTimelineStatus::CapacityExceeded;
            result.diagnostic = "Vulkan timeline byte size overflow";
            return result;
        }

        const std::uint32_t cancellation =
                cancellationRequested ? 1u : 0u;
        vulkan::TimelineKernelRequest request;
        request.scene = VulkanTimelineExecutorAccess::Scene(scene);
        request.configuration =
                VulkanTimelineExecutorAccess::Configuration(configuration);
        request.states = states.data();
        request.stateBytes = stateBytes;
        request.descriptors = descriptors.data();
        request.descriptorBytes = descriptorBytes;
        request.ticks = ticks.data();
        request.tickBytes = tickBytes;
        request.initialResults = deviceResults.data();
        request.resultBytes = resultBytes;
        request.cancellation = &cancellation;
        request.cancellationBytes = sizeof(cancellation);
        request.observationBytes = observationBytes;
        request.scratchBytes = scratchBytes;
        request.candidateCount =
                static_cast<std::uint32_t>(candidates.size());
        request.stateStride = sizeof(CudaCandidateState);
        request.fullState = true;
        request.outputStates = states.data();
        request.outputResults = deviceResults.data();
        request.outputObservations = observations.data();

        vulkan::TimelineKernelMetrics kernelMetrics;
        if (!vulkan::ExecuteTimelineKernel(
                    request, &kernelMetrics, &result.diagnostic)) {
            result.status = VulkanTimelineStatus::DeviceFailure;
            return result;
        }
        result.metrics.candidateCount = candidates.size();
        result.metrics.tickCount = totalTicks;
        result.metrics.observationCapacity = totalObservations;
        result.metrics.hostToDeviceBytes = kernelMetrics.uploadBytes;
        result.metrics.deviceToHostBytes = kernelMetrics.downloadBytes;
        result.metrics.peakDeviceBytes = kernelMetrics.peakDeviceBytes;
        result.metrics.allocationMilliseconds =
                kernelMetrics.allocationMilliseconds;
        result.metrics.transferMilliseconds =
                kernelMetrics.transferMilliseconds;
        result.metrics.kernelMilliseconds =
                kernelMetrics.kernelMilliseconds;
        result.metrics.synchronizationMilliseconds =
                kernelMetrics.synchronizationMilliseconds;

        std::vector<std::uint8_t> finishRefinementRequired(
                candidates.size(), 0u);
        bool anyFinishRefinement = false;
        for (std::size_t index = 0u;
             index < candidates.size(); ++index) {
            const bool required =
                    deviceResults[index].status ==
                            VulkanTimelineStatus::Success &&
                    !candidates[index].initialState.
                            race.progress.raceCompleted &&
                    states[index].race.progress.raceCompleted &&
                    !states[index].finishTime.present;
            finishRefinementRequired[index] = required ? 1u : 0u;
            anyFinishRefinement = anyFinishRefinement || required;
        }
        if (anyFinishRefinement) {
            std::vector<CudaCandidateState> finalStates = states;
            std::vector<VulkanTimelineDeviceResult> finalDeviceResults =
                    deviceResults;
            std::vector<VulkanTimelineDescriptor> refinementDescriptors =
                    descriptors;
            std::vector<VulkanTimelineDeviceResult> refinementResults(
                    candidates.size());
            std::vector<VulkanFinishRefinementWork> refinementOutputs(
                    candidates.size());
            if (candidates.size() >
                std::numeric_limits<std::size_t>::max() / 3u) {
                result.status = VulkanTimelineStatus::CapacityExceeded;
                result.diagnostic =
                        "Vulkan refinement state capacity exceeded";
                return result;
            }
            std::vector<CudaCandidateState> refinementStates(
                    candidates.size() * 3u);
            for (std::size_t index = 0u;
                 index < candidates.size(); ++index) {
                refinementStates[index] = candidates[index].initialState;
                if (!finishRefinementRequired[index]) {
                    refinementDescriptors[index].tickCount = 0u;
                }
            }
            std::size_t refinementStateBytes = 0u;
            std::size_t refinementOutputBytes = 0u;
            if (!ByteSize<CudaCandidateState>(
                        refinementStates.size(),
                        &refinementStateBytes) ||
                !ByteSize<VulkanFinishRefinementWork>(
                        refinementOutputs.size(),
                        &refinementOutputBytes)) {
                result.status = VulkanTimelineStatus::CapacityExceeded;
                result.diagnostic =
                        "Vulkan refinement state byte size overflow";
                return result;
            }

            request.states = refinementStates.data();
            request.stateBytes = refinementStateBytes;
            request.descriptors = refinementDescriptors.data();
            request.initialResults = refinementResults.data();
            request.observationBytes = refinementOutputBytes;
            request.finishRefinement = true;
            request.outputStates = refinementStates.data();
            request.outputResults = refinementResults.data();
            request.outputObservations = refinementOutputs.data();
            vulkan::TimelineKernelMetrics refinementMetrics;
            if (!vulkan::ExecuteTimelineKernel(
                        request, &refinementMetrics,
                        &result.diagnostic)) {
                result.status = VulkanTimelineStatus::DeviceFailure;
                return result;
            }
            result.metrics.hostToDeviceBytes +=
                    refinementMetrics.uploadBytes;
            result.metrics.deviceToHostBytes +=
                    refinementMetrics.downloadBytes;
            result.metrics.peakDeviceBytes = std::max(
                    result.metrics.peakDeviceBytes,
                    refinementMetrics.peakDeviceBytes);
            result.metrics.allocationMilliseconds +=
                    refinementMetrics.allocationMilliseconds;
            result.metrics.transferMilliseconds +=
                    refinementMetrics.transferMilliseconds;
            result.metrics.kernelMilliseconds +=
                    refinementMetrics.kernelMilliseconds;
            result.metrics.synchronizationMilliseconds +=
                    refinementMetrics.synchronizationMilliseconds;

            for (std::size_t index = 0u;
                 index < candidates.size(); ++index) {
                if (!finishRefinementRequired[index]) continue;
                if (refinementResults[index].status !=
                            VulkanTimelineStatus::Success ||
                    !refinementStates[index].finishTime.present ||
                    !refinementStates[index].finishTime.value.IsValid()) {
                    result.status = VulkanTimelineStatus::DeviceFailure;
                    result.diagnostic =
                            "Vulkan finish refinement failed: status=" +
                            std::to_string(static_cast<unsigned>(
                                    refinementResults[index].status)) +
                            " tick=" + std::to_string(
                                    refinementResults[index].failureTick) +
                            " detail=" + std::to_string(
                                    refinementResults[index].failureDetail) +
                            " present=" + std::to_string(
                                    refinementStates[index].finishTime.present) +
                            " estimate=" + std::to_string(
                                    refinementStates[index].finishTime.value.lowerBoundNs) +
                            "," + std::to_string(
                                    refinementStates[index].finishTime.value.upperBoundNs) +
                            "," + std::to_string(
                                    refinementStates[index].finishTime.value.estimatedNs);
                    return result;
                }
                finalStates[index].finishTime =
                        refinementStates[index].finishTime;
            }
            states = std::move(finalStates);
            deviceResults = std::move(finalDeviceResults);
        }

        result.candidates.resize(candidates.size());
        result.status = VulkanTimelineStatus::Success;
        for (std::size_t index = 0u;
             index < candidates.size(); ++index) {
            VulkanCandidateTimelineOutput &output =
                    result.candidates[index];
            const VulkanTimelineDeviceResult &deviceOutput =
                    deviceResults[index];
            output.status = deviceOutput.status;
            output.failureTick = deviceOutput.failureTick;
            output.failureDetail = deviceOutput.failureDetail;
            output.executedTickCount = deviceOutput.executedTickCount;
            output.executedRespawnCount =
                    deviceOutput.executedRespawnCount;
            output.finalState = states[index];
            const VulkanTimelineDescriptor &descriptor =
                    descriptors[index];
            const std::uint32_t count = std::min(
                    deviceOutput.observationCount,
                    descriptor.observationCapacity);
            output.observations.assign(
                    observations.begin() +
                            descriptor.firstObservation,
                    observations.begin() +
                            descriptor.firstObservation + count);
            if (output.status != VulkanTimelineStatus::Success &&
                result.status == VulkanTimelineStatus::Success) {
                result.status = output.status;
            }
        }
        result.winnerCandidateIndex =
                SelectCudaTimelineWinner(result.candidates);
        if (result.winnerCandidateIndex.has_value()) {
            result.winnerCandidateId =
                    result.candidates[*result.winnerCandidateIndex].
                            finalState.candidateId;
        }
        result.diagnostic =
                std::string("Vulkan timeline kernel completed with status ") +
                VulkanTimelineStatusName(result.status);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = VulkanTimelineStatus::CapacityExceeded;
        result.diagnostic = "Vulkan timeline host allocation failed";
        return result;
    } catch (...) {
        result.status = VulkanTimelineStatus::DeviceFailure;
        result.diagnostic =
                "unexpected Vulkan timeline execution failure";
        return result;
    }
}

static VulkanTimelineBatchResult ExecuteVulkanTimelineBatchBounded(
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &configuration,
        const std::vector<VulkanCandidateTimelineInput> &candidates,
        bool cancellationRequested,
        std::size_t maximumTicksPerDispatch,
        bool exactFinishTransitions) noexcept {
    VulkanTimelineBatchResult result;
    if (!scene.Ready() || !configuration.Ready() || candidates.empty() ||
        maximumTicksPerDispatch == 0u) {
        result.status = VulkanTimelineStatus::InvalidArgument;
        result.diagnostic = "invalid Vulkan timeline batch";
        return result;
    }

    const bool needsChunking = std::any_of(
            candidates.begin(), candidates.end(),
            [maximumTicksPerDispatch](
                    const VulkanCandidateTimelineInput &candidate) {
                return candidate.ticks.size() > maximumTicksPerDispatch;
            });
    const bool needsFinishDetection = exactFinishTransitions &&
            std::any_of(
                    candidates.begin(), candidates.end(),
                    [](const VulkanCandidateTimelineInput &candidate) {
                        return candidate.ticks.size() >
                               FinishExactTicksPerDispatch;
                    });
    if (!needsChunking && !needsFinishDetection) {
        return ExecuteVulkanTimelineSingleDispatch(
                scene, configuration, candidates, cancellationRequested);
    }

    try {
        result.status = VulkanTimelineStatus::Success;
        result.candidates.resize(candidates.size());
        std::vector<std::size_t> cursors(candidates.size(), 0u);
        std::vector<bool> active(candidates.size(), true);
        for (std::size_t index = 0u; index < candidates.size(); ++index) {
            result.candidates[index].finalState =
                    candidates[index].initialState;
            result.candidates[index].failureTick = UINT32_MAX;
            result.metrics.tickCount += candidates[index].ticks.size();
            result.metrics.observationCapacity +=
                    static_cast<std::uint64_t>(std::count_if(
                            candidates[index].ticks.begin(),
                            candidates[index].ticks.end(),
                            [](const VulkanControlTick &tick) {
                                return tick.observe;
                            }));
        }
        result.metrics.candidateCount = candidates.size();

        for (;;) {
            std::vector<std::size_t> indices;
            std::vector<VulkanCandidateTimelineInput> wave;
            indices.reserve(candidates.size());
            wave.reserve(candidates.size());
            for (std::size_t index = 0u; index < candidates.size(); ++index) {
                if (!active[index] ||
                    cursors[index] >= candidates[index].ticks.size()) {
                    continue;
                }
                const std::size_t end = std::min(
                        cursors[index] + maximumTicksPerDispatch,
                        candidates[index].ticks.size());
                VulkanCandidateTimelineInput input;
                input.initialState = result.candidates[index].finalState;
                input.ticks.assign(
                        candidates[index].ticks.begin() + cursors[index],
                        candidates[index].ticks.begin() + end);
                indices.push_back(index);
                wave.push_back(std::move(input));
            }
            if (wave.empty()) break;

            VulkanTimelineBatchResult executed =
                    ExecuteVulkanTimelineSingleDispatch(
                            scene, configuration, wave,
                            cancellationRequested);
            result.metrics.hostToDeviceBytes +=
                    executed.metrics.hostToDeviceBytes;
            result.metrics.deviceToHostBytes +=
                    executed.metrics.deviceToHostBytes;
            result.metrics.peakDeviceBytes = std::max(
                    result.metrics.peakDeviceBytes,
                    executed.metrics.peakDeviceBytes);
            result.metrics.allocationMilliseconds +=
                    executed.metrics.allocationMilliseconds;
            result.metrics.transferMilliseconds +=
                    executed.metrics.transferMilliseconds;
            result.metrics.kernelMilliseconds +=
                    executed.metrics.kernelMilliseconds;
            result.metrics.synchronizationMilliseconds +=
                    executed.metrics.synchronizationMilliseconds;

            if (executed.candidates.size() != wave.size()) {
                result.status = executed.status ==
                                        VulkanTimelineStatus::Success
                        ? VulkanTimelineStatus::DeviceFailure
                        : executed.status;
                result.diagnostic = executed.diagnostic;
                return result;
            }
            if (exactFinishTransitions) {
                std::vector<std::size_t> exactWaveIndices;
                std::vector<VulkanCandidateTimelineInput> exactWave;
                exactWaveIndices.reserve(wave.size());
                exactWave.reserve(wave.size());
                for (std::size_t waveIndex = 0u;
                     waveIndex < wave.size(); ++waveIndex) {
                    if (wave[waveIndex].ticks.size() >
                                    FinishExactTicksPerDispatch &&
                        executed.candidates[waveIndex].status ==
                                    VulkanTimelineStatus::Success &&
                        !wave[waveIndex].initialState.race.progress.
                                    raceCompleted &&
                        executed.candidates[waveIndex].finalState.race.
                                    progress.raceCompleted) {
                        exactWaveIndices.push_back(waveIndex);
                        exactWave.push_back(wave[waveIndex]);
                    }
                }
                if (!exactWave.empty()) {
                    VulkanTimelineBatchResult exact =
                            ExecuteVulkanTimelineBatchBounded(
                                    scene, configuration, exactWave,
                                    cancellationRequested,
                                    FinishExactTicksPerDispatch, false);
                    result.metrics.candidateCount +=
                            exact.metrics.candidateCount;
                    result.metrics.tickCount += exact.metrics.tickCount;
                    result.metrics.observationCapacity +=
                            exact.metrics.observationCapacity;
                    result.metrics.hostToDeviceBytes +=
                            exact.metrics.hostToDeviceBytes;
                    result.metrics.deviceToHostBytes +=
                            exact.metrics.deviceToHostBytes;
                    result.metrics.peakDeviceBytes = std::max(
                            result.metrics.peakDeviceBytes,
                            exact.metrics.peakDeviceBytes);
                    result.metrics.allocationMilliseconds +=
                            exact.metrics.allocationMilliseconds;
                    result.metrics.transferMilliseconds +=
                            exact.metrics.transferMilliseconds;
                    result.metrics.kernelMilliseconds +=
                            exact.metrics.kernelMilliseconds;
                    result.metrics.synchronizationMilliseconds +=
                            exact.metrics.synchronizationMilliseconds;
                    if (exact.candidates.size() != exactWave.size()) {
                        result.status = exact.status ==
                                                VulkanTimelineStatus::Success
                                ? VulkanTimelineStatus::DeviceFailure
                                : exact.status;
                        result.diagnostic = exact.diagnostic;
                        return result;
                    }
                    for (std::size_t index = 0u;
                         index < exactWaveIndices.size(); ++index) {
                        executed.candidates[exactWaveIndices[index]] =
                                std::move(exact.candidates[index]);
                    }
                }
            }
            for (std::size_t waveIndex = 0u;
                 waveIndex < wave.size(); ++waveIndex) {
                const std::size_t index = indices[waveIndex];
                VulkanCandidateTimelineOutput &destination =
                        result.candidates[index];
                VulkanCandidateTimelineOutput &source =
                        executed.candidates[waveIndex];
                const std::size_t waveBegin = cursors[index];
                destination.finalState = source.finalState;
                destination.executedTickCount += source.executedTickCount;
                destination.executedRespawnCount +=
                        source.executedRespawnCount;
                destination.observations.insert(
                        destination.observations.end(),
                        source.observations.begin(),
                        source.observations.end());
                cursors[index] += source.executedTickCount;
                if (source.status != VulkanTimelineStatus::Success) {
                    destination.status = source.status;
                    destination.failureDetail = source.failureDetail;
                    destination.failureTick = source.failureTick == UINT32_MAX
                            ? UINT32_MAX
                            : static_cast<std::uint32_t>(
                                      waveBegin + source.failureTick);
                    active[index] = false;
                    if (result.status == VulkanTimelineStatus::Success) {
                        result.status = source.status;
                    }
                } else {
                    destination.status = VulkanTimelineStatus::Success;
                    cursors[index] = waveBegin + wave[waveIndex].ticks.size();
                }
            }
        }

        result.winnerCandidateIndex =
                SelectCudaTimelineWinner(result.candidates);
        if (result.winnerCandidateIndex.has_value()) {
            result.winnerCandidateId =
                    result.candidates[*result.winnerCandidateIndex]
                            .finalState.candidateId;
        }
        result.diagnostic =
                std::string("Vulkan timeline completed in bounded dispatches ") +
                "with status " + VulkanTimelineStatusName(result.status);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = VulkanTimelineStatus::CapacityExceeded;
        result.diagnostic = "Vulkan timeline chunk allocation failed";
        return result;
    } catch (...) {
        result.status = VulkanTimelineStatus::DeviceFailure;
        result.diagnostic = "unexpected Vulkan chunked timeline failure";
        return result;
    }
}

VulkanTimelineBatchResult ExecuteVulkanFinishTimelineBatch(
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &configuration,
        const std::vector<VulkanCandidateTimelineInput> &candidates,
        bool cancellationRequested) noexcept {
    return ExecuteVulkanTimelineBatchBounded(
            scene, configuration, candidates, cancellationRequested,
            FinishExactTicksPerDispatch, false);
}

VulkanTimelineBatchResult ExecuteVulkanTimelineBatch(
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &configuration,
        const std::vector<VulkanCandidateTimelineInput> &candidates,
        bool cancellationRequested) noexcept {
    return ExecuteVulkanTimelineBatchBounded(
            scene, configuration, candidates, cancellationRequested,
            MaximumTicksPerDispatch, true);
}

}  // namespace forevervalidator::simulation
