#include "simulation/backends/cuda/cuda_timeline_executor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_finish_time_refinement.cuh"
#include "simulation/backends/cuda/cuda_physics_step.cuh"
#include "simulation/backends/cuda/cuda_stunts.cuh"
#include "simulation/backends/cuda/cuda_vehicle_powertrain.cuh"
#include "simulation/backends/cuda/cuda_vehicle_transitions.cuh"
#include "simulation/backends/cuda/cuda_vehicle_wheels.cuh"

namespace forevervalidator::simulation {
namespace {

constexpr std::uint64_t MaximumTotalTicks = 100000000u;
constexpr std::uint64_t MaximumTotalObservations = 10000000u;

struct DeviceTimelineDescriptor {
    std::uint64_t firstTick = 0u;
    std::uint32_t tickCount = 0u;
    std::uint64_t firstObservation = 0u;
    std::uint32_t observationCapacity = 0u;
};

struct DeviceTimelineResult {
    CudaTimelineStatus status =
            CudaTimelineStatus::InvalidArgument;
    std::uint32_t failureTick = UINT32_MAX;
    std::uint32_t failureDetail = 0u;
    std::uint32_t executedTickCount = 0u;
    std::uint32_t executedRespawnCount = 0u;
    std::uint32_t observationCount = 0u;
};

using DeviceFinishRefinement = cuda::finish::Refinement;

template<typename T>
class DeviceAllocation {
public:
    ~DeviceAllocation() { Reset(); }
    DeviceAllocation() = default;
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;

    bool Allocate(std::size_t count) {
        Reset();
        count_ = count;
        if (count == 0u) {
            return true;
        }
        if (count > std::numeric_limits<std::size_t>::max() /
                            sizeof(T)) {
            return false;
        }
        if (cudaMalloc(
                    reinterpret_cast<void **>(&data_),
                    count * sizeof(T)) != cudaSuccess) {
            data_ = nullptr;
            count_ = 0u;
            return false;
        }
        return true;
    }

    void Reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        count_ = 0u;
    }

    T *Get() const { return data_; }
    std::size_t Bytes() const { return count_ * sizeof(T); }

private:
    T *data_ = nullptr;
    std::size_t count_ = 0u;
};

class Event {
public:
    Event() {
        valid_ = cudaEventCreate(&event_) == cudaSuccess;
    }
    ~Event() {
        if (valid_) cudaEventDestroy(event_);
    }
    bool Valid() const { return valid_; }
    cudaEvent_t Get() const { return event_; }

private:
    cudaEvent_t event_{};
    bool valid_ = false;
};

__device__ bool ValidPackedInputs(
        const void *sceneData,
        const void *configurationData) {
    if (sceneData == nullptr || configurationData == nullptr) {
        return false;
    }
    const auto *scene =
            static_cast<const CudaPackedSceneHeader *>(sceneData);
    const auto *configuration =
            static_cast<
                    const CudaPackedStaticConfigurationHeader *>(
                    configurationData);
    return ValidCudaPackedSceneHeader(*scene) &&
           configuration->magic ==
                   CudaPackedStaticConfigurationHeader::Magic &&
           configuration->schemaVersion ==
                   CudaPackedStaticConfigurationHeader::SchemaVersion;
}

__device__ void ApplyControlAndTimingPrefix(
        CudaCandidateState &state,
        const CudaControlTick &tick,
        bool applyControls) {
    state.world.schemePeriodMs = tick.periodMs;
    state.world.tickTimeMs = tick.timeMs;
    if (!applyControls) return;
    state.vehicle.controls.lowSpeedGateA =
            tick.controls.lowSpeedGateA;
    state.vehicle.controls.lowSpeedGateB =
            tick.controls.lowSpeedGateB;
    state.vehicle.controls.steeringControl =
            tick.controls.steering;
    state.vehicle.frameHistory.physicsCurrent.lowSpeedGateA =
            tick.controls.lowSpeedGateA;
    state.vehicle.frameHistory.physicsCurrent.lowSpeedGateB =
            tick.controls.lowSpeedGateB;
    state.vehicle.frameHistory.physicsCurrent.steeringControl =
            tick.controls.steering;
}

__device__ void RecordObservation(
        const CudaCandidateState &state,
        const CudaControlTick &tick,
        CudaTimelineObservation &observation) {
    observation.simulatedPosition =
            state.body.current.position;
    observation.writePosition = state.body.write.position;
    observation.hasComparison = tick.hasComparisonTarget;
    observation.comparisonTarget = tick.comparisonTarget;
    if (tick.hasComparisonTarget) {
        observation.comparisonDelta = {
                state.body.write.position.x -
                        tick.comparisonTarget.x,
                state.body.write.position.y -
                        tick.comparisonTarget.y,
                state.body.write.position.z -
                        tick.comparisonTarget.z,
        };
        const GmVec3 &delta = observation.comparisonDelta;
        observation.comparisonDistance = cuda::exact::Sqrt(
                (delta.x * delta.x + delta.y * delta.y) +
                delta.z * delta.z);
    }
    observation.hasFinishTick =
            state.race.progress.raceCompleted;
    observation.finishTickMs =
            state.race.progress.lastPrepareTimeMs;
}

__global__ void RefineFinishTimesKernel(
        const void *sceneData,
        const void *configurationData,
        CudaCandidateState *states,
        const DeviceTimelineDescriptor *descriptors,
        const CudaControlTick *ticks,
        const std::uint8_t *required,
        DeviceFinishRefinement *outputs,
        cuda::collision::CudaCollisionScratch *scratch,
        std::uint32_t candidateCount) {
    const std::uint32_t candidate =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidateCount || !required[candidate]) {
        return;
    }
    auto *scene = static_cast<const CudaPackedSceneHeader *>(sceneData);
    auto *configuration =
            static_cast<const CudaPackedStaticConfigurationHeader *>(
                    configurationData);
    CudaCandidateState &state = states[candidate];
    const DeviceTimelineDescriptor descriptor = descriptors[candidate];
    for (std::uint32_t index = 0u;
         index < descriptor.tickCount; ++index) {
        const CudaControlTick &tick =
                ticks[descriptor.firstTick + index];
        ApplyControlAndTimingPrefix(state, tick, false);
        if (!state.firstStep) {
            cuda::transition::PrepareStep(
                    state, tick, configuration);
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        state.vehicle.mobil.physicsUpdatesEnabled =
                (tick.actionFlags &
                 CudaControlActionSuppressVehicleForceCallbacks) == 0u;
        for (std::uint32_t respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn) {
            if (cuda::transition::Respawn(state, configuration)) {
                ++state.incrementalRespawnCount;
            }
        }
        const cuda::physics::Status status =
                cuda::finish::StepAndRefine(
                scene, configuration, state, tick,
                scratch[candidate], outputs[candidate]);
        if (status != cuda::physics::Status::Success) {
            outputs[candidate].failed = true;
            return;
        }
        if (outputs[candidate].present ||
            outputs[candidate].failed) {
            return;
        }
        state.firstStep = false;
    }
    outputs[candidate].failed = true;
}

__global__ void ExecuteTimelineKernel(
        const void *sceneData,
        const void *configurationData,
        CudaCandidateState *states,
        const DeviceTimelineDescriptor *descriptors,
        const CudaControlTick *ticks,
        CudaTimelineObservation *observations,
        DeviceTimelineResult *results,
        cuda::collision::CudaCollisionScratch *scratch,
        const std::uint32_t *cancellation,
        std::uint32_t candidateCount) {
    const std::uint32_t candidate =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidateCount) {
        return;
    }
    DeviceTimelineResult &result = results[candidate];
    CudaCandidateState &state = states[candidate];
    if (!ValidPackedInputs(sceneData, configurationData)) {
        result.status = CudaTimelineStatus::InvalidArgument;
        return;
    }
    if (state.schemaVersion != CudaCandidateState::SchemaVersion) {
        result.status = CudaTimelineStatus::SchemaMismatch;
        return;
    }
    const DeviceTimelineDescriptor descriptor =
            descriptors[candidate];
    if (descriptor.tickCount == 0u) {
        result.status = CudaTimelineStatus::Success;
        result.failureTick = UINT32_MAX;
        return;
    }
    for (std::uint32_t index = 0u;
         index < descriptor.tickCount; ++index) {
        if (atomicAdd(
                    const_cast<std::uint32_t *>(cancellation),
                    0u) != 0u) {
            result.status = CudaTimelineStatus::Cancelled;
            result.failureTick = index;
            return;
        }
        const CudaControlTick &tick =
                ticks[descriptor.firstTick + index];
        ApplyControlAndTimingPrefix(state, tick, false);
        if (!state.firstStep) {
            cuda::transition::PrepareStep(
                    state, tick,
                    static_cast<const
                            CudaPackedStaticConfigurationHeader *>(
                            configurationData));
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        state.vehicle.mobil.physicsUpdatesEnabled =
                (tick.actionFlags &
                 CudaControlActionSuppressVehicleForceCallbacks) == 0u;
        for (std::uint32_t respawn = 0u;
             respawn < tick.respawnAtCheckpointCount;
             ++respawn) {
            if (cuda::transition::Respawn(
                        state,
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData))) {
                ++result.executedRespawnCount;
                ++state.incrementalRespawnCount;
                cuda::stunts::ApplyRespawnPenalty(
                        state.stunts);
            }
        }
        const cuda::physics::Status physicsStatus =
                cuda::physics::Step(
                        static_cast<
                                const CudaPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, scratch[candidate]);
        if (physicsStatus != cuda::physics::Status::Success) {
            result.status =
                    CudaTimelineStatus::
                            UnsupportedPhysicsTransition;
            result.failureTick = index;
            result.failureDetail =
                    static_cast<std::uint32_t>(physicsStatus) +
                    1000u * static_cast<std::uint32_t>(
                            scratch[candidate].overflowReason) +
                    100000u * scratch[candidate].collisionCount;
            return;
        }
        cuda::collision::detail::CaptureReplacementOverflow(
                scratch[candidate],
                state.collisionReplacementOverflow);
        if (state.stuntsEnabled) {
            const cuda::stunts::Status stuntStatus =
                    cuda::stunts::Update(state, tick);
            if (stuntStatus != cuda::stunts::Status::Success) {
                result.status =
                        CudaTimelineStatus::CapacityExceeded;
                result.failureTick = index;
                result.failureDetail =
                        static_cast<std::uint32_t>(stuntStatus);
                return;
            }
        }
        state.firstStep = false;
        ++state.controlCursor;
        ++result.executedTickCount;
        if (tick.observe) {
            if (result.observationCount >=
                descriptor.observationCapacity) {
                result.status =
                        CudaTimelineStatus::CapacityExceeded;
                result.failureTick = index;
                return;
            }
            RecordObservation(
                    state, tick,
                    observations[
                            descriptor.firstObservation +
                            result.observationCount]);
            ++result.observationCount;
        }
    }
    result.status = CudaTimelineStatus::Success;
    result.failureTick = UINT32_MAX;
}

std::string CudaFailure(const char *operation, cudaError_t error) {
    return std::string(operation) + " failed: " +
           cudaGetErrorName(error) + " (" +
           cudaGetErrorString(error) + ")";
}

}  // namespace

CudaTimelineBatchResult ExecuteCudaTimelineBatch(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const std::vector<CudaCandidateTimelineInput> &candidates,
        bool cancellationRequested) noexcept {
    CudaTimelineBatchResult result;
    if (deviceScene == nullptr ||
        deviceStaticConfiguration == nullptr ||
        candidates.empty() ||
        candidates.size() >
                std::numeric_limits<std::uint32_t>::max()) {
        result.status = CudaTimelineStatus::InvalidArgument;
        result.diagnostic = "invalid CUDA timeline batch";
        return result;
    }
    try {
        std::vector<CudaCandidateState> states;
        std::vector<DeviceTimelineDescriptor> descriptors;
        std::vector<CudaControlTick> ticks;
        std::uint64_t observationCount = 0u;
        std::uint64_t tickCount = 0u;
        states.reserve(candidates.size());
        descriptors.reserve(candidates.size());
        for (const CudaCandidateTimelineInput &candidate :
             candidates) {
            if (candidate.ticks.size() >
                        MaximumTotalTicks - tickCount) {
                result.status =
                        CudaTimelineStatus::CapacityExceeded;
                result.diagnostic =
                        "CUDA timeline tick capacity exceeded";
                return result;
            }
            const std::uint64_t candidateObservations =
                    static_cast<std::uint64_t>(std::count_if(
                            candidate.ticks.begin(),
                            candidate.ticks.end(),
                            [](const CudaControlTick &tick) {
                                return tick.observe;
                            }));
            if (candidateObservations >
                        MaximumTotalObservations - observationCount) {
                result.status =
                        CudaTimelineStatus::CapacityExceeded;
                result.diagnostic =
                        "CUDA observation capacity exceeded";
                return result;
            }
            DeviceTimelineDescriptor descriptor;
            descriptor.firstTick = tickCount;
            descriptor.tickCount =
                    static_cast<std::uint32_t>(
                            candidate.ticks.size());
            descriptor.firstObservation = observationCount;
            descriptor.observationCapacity =
                    static_cast<std::uint32_t>(
                            candidateObservations);
            states.push_back(candidate.initialState);
            descriptors.push_back(descriptor);
            ticks.insert(ticks.end(), candidate.ticks.begin(),
                         candidate.ticks.end());
            tickCount += candidate.ticks.size();
            observationCount += candidateObservations;
        }
        std::vector<CudaTimelineObservation> observations(
                observationCount);
        std::vector<DeviceTimelineResult> deviceResults(
                candidates.size());

        result.metrics.candidateCount = candidates.size();
        result.metrics.tickCount = tickCount;
        result.metrics.observationCapacity = observationCount;

        const auto allocationStart =
                std::chrono::steady_clock::now();
        DeviceAllocation<CudaCandidateState> deviceStates;
        DeviceAllocation<DeviceTimelineDescriptor> deviceDescriptors;
        DeviceAllocation<CudaControlTick> deviceTicks;
        DeviceAllocation<CudaTimelineObservation> deviceObservations;
        DeviceAllocation<DeviceTimelineResult> deviceResultsAllocation;
        DeviceAllocation<cuda::collision::CudaCollisionScratch>
                deviceScratch;
        DeviceAllocation<std::uint32_t> deviceCancellation;
        if (!deviceStates.Allocate(states.size()) ||
            !deviceDescriptors.Allocate(descriptors.size()) ||
            !deviceTicks.Allocate(ticks.size()) ||
            !deviceObservations.Allocate(observations.size()) ||
            !deviceResultsAllocation.Allocate(deviceResults.size()) ||
            !deviceScratch.Allocate(candidates.size()) ||
            !deviceCancellation.Allocate(1u)) {
            result.status = CudaTimelineStatus::DeviceFailure;
            result.diagnostic =
                    "CUDA timeline device allocation failed";
            return result;
        }
        const auto allocationEnd =
                std::chrono::steady_clock::now();
        result.metrics.allocationMilliseconds =
                std::chrono::duration<double, std::milli>(
                        allocationEnd - allocationStart).count();
        result.metrics.peakDeviceBytes =
                deviceStates.Bytes() + deviceDescriptors.Bytes() +
                deviceTicks.Bytes() + deviceObservations.Bytes() +
                deviceResultsAllocation.Bytes() +
                deviceScratch.Bytes() +
                deviceCancellation.Bytes();

        const auto transferStart =
                std::chrono::steady_clock::now();
#define COPY_TO_DEVICE(allocation, source)                                    \
        do {                                                                   \
            if (!(source).empty()) {                                           \
                const cudaError_t copyError = cudaMemcpy(                      \
                        (allocation).Get(), (source).data(),                    \
                        (allocation).Bytes(), cudaMemcpyHostToDevice);          \
                if (copyError != cudaSuccess) {                                \
                    result.status = CudaTimelineStatus::DeviceFailure;          \
                    result.diagnostic = CudaFailure(                            \
                            "CUDA host-to-device transfer", copyError);         \
                    return result;                                             \
                }                                                              \
            }                                                                  \
        } while (false)
        COPY_TO_DEVICE(deviceStates, states);
        COPY_TO_DEVICE(deviceDescriptors, descriptors);
        COPY_TO_DEVICE(deviceTicks, ticks);
        COPY_TO_DEVICE(deviceResultsAllocation, deviceResults);
#undef COPY_TO_DEVICE
        const std::uint32_t cancellation =
                cancellationRequested ? 1u : 0u;
        cudaError_t cudaResult = cudaMemcpy(
                deviceCancellation.Get(), &cancellation,
                sizeof(cancellation), cudaMemcpyHostToDevice);
        if (cudaResult != cudaSuccess) {
            result.status = CudaTimelineStatus::DeviceFailure;
            result.diagnostic = CudaFailure(
                    "CUDA cancellation transfer", cudaResult);
            return result;
        }
        const auto transferEnd =
                std::chrono::steady_clock::now();
        result.metrics.transferMilliseconds =
                std::chrono::duration<double, std::milli>(
                        transferEnd - transferStart).count();
        result.metrics.hostToDeviceBytes =
                deviceStates.Bytes() + deviceDescriptors.Bytes() +
                deviceTicks.Bytes() + deviceResultsAllocation.Bytes() +
                sizeof(cancellation);

        Event kernelStart;
        Event kernelEnd;
        if (!kernelStart.Valid() || !kernelEnd.Valid()) {
            result.status = CudaTimelineStatus::DeviceFailure;
            result.diagnostic = "CUDA event creation failed";
            return result;
        }
        cudaEventRecord(kernelStart.Get());
        // Each candidate owns a long, register-heavy serial timeline.
        // A one-thread block lets the scheduler spread independent candidates
        // across SMs instead of pinning an entire batch to one SM.
        constexpr std::uint32_t Threads = 1u;
        const std::uint32_t blocks =
                (static_cast<std::uint32_t>(candidates.size()) +
                 Threads - 1u) /
                Threads;
        ExecuteTimelineKernel<<<blocks, Threads>>>(
                deviceScene, deviceStaticConfiguration,
                deviceStates.Get(), deviceDescriptors.Get(),
                deviceTicks.Get(), deviceObservations.Get(),
                deviceResultsAllocation.Get(),
                deviceScratch.Get(),
                deviceCancellation.Get(),
                static_cast<std::uint32_t>(candidates.size()));
        cudaResult = cudaGetLastError();
        if (cudaResult != cudaSuccess) {
            result.status = CudaTimelineStatus::DeviceFailure;
            result.diagnostic =
                    CudaFailure("CUDA timeline launch", cudaResult);
            return result;
        }
        cudaEventRecord(kernelEnd.Get());
        const auto synchronizationStart =
                std::chrono::steady_clock::now();
        cudaResult = cudaEventSynchronize(kernelEnd.Get());
        const auto synchronizationEnd =
                std::chrono::steady_clock::now();
        if (cudaResult != cudaSuccess) {
            result.status = CudaTimelineStatus::DeviceFailure;
            result.diagnostic =
                    CudaFailure("CUDA timeline synchronization",
                                cudaResult);
            return result;
        }
        float kernelMilliseconds = 0.0f;
        cudaEventElapsedTime(
                &kernelMilliseconds, kernelStart.Get(),
                kernelEnd.Get());
        result.metrics.kernelMilliseconds = kernelMilliseconds;
        result.metrics.synchronizationMilliseconds =
                std::chrono::duration<double, std::milli>(
                        synchronizationEnd -
                        synchronizationStart).count();

        const auto downloadStart =
                std::chrono::steady_clock::now();
        cudaResult = cudaMemcpy(
                states.data(), deviceStates.Get(),
                deviceStates.Bytes(), cudaMemcpyDeviceToHost);
        if (cudaResult == cudaSuccess) {
            cudaResult = cudaMemcpy(
                    deviceResults.data(),
                    deviceResultsAllocation.Get(),
                    deviceResultsAllocation.Bytes(),
                    cudaMemcpyDeviceToHost);
        }
        if (cudaResult == cudaSuccess && !observations.empty()) {
            cudaResult = cudaMemcpy(
                    observations.data(), deviceObservations.Get(),
                    deviceObservations.Bytes(),
                    cudaMemcpyDeviceToHost);
        }
        if (cudaResult != cudaSuccess) {
            result.status = CudaTimelineStatus::DeviceFailure;
            result.diagnostic = CudaFailure(
                    "CUDA device-to-host transfer", cudaResult);
            return result;
        }
        const auto downloadEnd =
                std::chrono::steady_clock::now();
        result.metrics.transferMilliseconds +=
                std::chrono::duration<double, std::milli>(
                        downloadEnd - downloadStart).count();
        result.metrics.deviceToHostBytes =
                deviceStates.Bytes() +
                deviceResultsAllocation.Bytes() +
                deviceObservations.Bytes();

        std::vector<std::uint8_t> finishRefinementRequired(
                candidates.size(), 0u);
        bool anyFinishRefinement = false;
        for (std::size_t index = 0u;
             index < candidates.size(); ++index) {
            const bool required =
                    deviceResults[index].status ==
                            CudaTimelineStatus::Success &&
                    !candidates[index].initialState.
                            race.progress.raceCompleted &&
                    states[index].race.progress.raceCompleted &&
                    !states[index].finishTime.present;
            finishRefinementRequired[index] = required ? 1u : 0u;
            anyFinishRefinement = anyFinishRefinement || required;
        }
        if (anyFinishRefinement) {
            std::vector<CudaCandidateState> refinementStates;
            std::vector<DeviceFinishRefinement> finishRefinements(
                    candidates.size());
            refinementStates.reserve(candidates.size());
            for (const CudaCandidateTimelineInput &candidate :
                 candidates) {
                refinementStates.push_back(candidate.initialState);
            }
            DeviceAllocation<std::uint8_t> deviceFinishRequired;
            DeviceAllocation<DeviceFinishRefinement>
                    deviceFinishRefinements;
            if (!deviceFinishRequired.Allocate(candidates.size()) ||
                !deviceFinishRefinements.Allocate(candidates.size())) {
                result.status = CudaTimelineStatus::DeviceFailure;
                result.diagnostic =
                        "CUDA finish refinement allocation failed";
                return result;
            }
            result.metrics.peakDeviceBytes +=
                    deviceFinishRequired.Bytes() +
                    deviceFinishRefinements.Bytes();
            const auto refinementTransferStart =
                    std::chrono::steady_clock::now();
            cudaResult = cudaMemcpy(
                    deviceStates.Get(), refinementStates.data(),
                    deviceStates.Bytes(), cudaMemcpyHostToDevice);
            if (cudaResult == cudaSuccess) {
                cudaResult = cudaMemcpy(
                        deviceFinishRequired.Get(),
                        finishRefinementRequired.data(),
                        deviceFinishRequired.Bytes(),
                        cudaMemcpyHostToDevice);
            }
            if (cudaResult == cudaSuccess) {
                cudaResult = cudaMemcpy(
                        deviceFinishRefinements.Get(),
                        finishRefinements.data(),
                        deviceFinishRefinements.Bytes(),
                        cudaMemcpyHostToDevice);
            }
            if (cudaResult != cudaSuccess) {
                result.status = CudaTimelineStatus::DeviceFailure;
                result.diagnostic = CudaFailure(
                        "CUDA finish refinement upload",
                        cudaResult);
                return result;
            }
            result.metrics.hostToDeviceBytes +=
                    deviceStates.Bytes() +
                    deviceFinishRequired.Bytes() +
                    deviceFinishRefinements.Bytes();

            Event refinementStart;
            Event refinementEnd;
            if (!refinementStart.Valid() ||
                !refinementEnd.Valid()) {
                result.status = CudaTimelineStatus::DeviceFailure;
                result.diagnostic =
                        "CUDA finish refinement event creation failed";
                return result;
            }
            cudaEventRecord(refinementStart.Get());
            RefineFinishTimesKernel<<<blocks, Threads>>>(
                    deviceScene, deviceStaticConfiguration,
                    deviceStates.Get(), deviceDescriptors.Get(),
                    deviceTicks.Get(), deviceFinishRequired.Get(),
                    deviceFinishRefinements.Get(),
                    deviceScratch.Get(),
                    static_cast<std::uint32_t>(
                            candidates.size()));
            cudaResult = cudaGetLastError();
            if (cudaResult == cudaSuccess) {
                cudaEventRecord(refinementEnd.Get());
                cudaResult =
                        cudaEventSynchronize(refinementEnd.Get());
            }
            if (cudaResult != cudaSuccess) {
                result.status = CudaTimelineStatus::DeviceFailure;
                result.diagnostic = CudaFailure(
                        "CUDA finish refinement", cudaResult);
                return result;
            }
            float refinementMilliseconds = 0.0f;
            cudaEventElapsedTime(
                    &refinementMilliseconds,
                    refinementStart.Get(),
                    refinementEnd.Get());
            result.metrics.kernelMilliseconds +=
                    refinementMilliseconds;
            cudaResult = cudaMemcpy(
                    finishRefinements.data(),
                    deviceFinishRefinements.Get(),
                    deviceFinishRefinements.Bytes(),
                    cudaMemcpyDeviceToHost);
            if (cudaResult != cudaSuccess) {
                result.status = CudaTimelineStatus::DeviceFailure;
                result.diagnostic = CudaFailure(
                        "CUDA finish refinement download",
                        cudaResult);
                return result;
            }
            const auto refinementTransferEnd =
                    std::chrono::steady_clock::now();
            result.metrics.transferMilliseconds +=
                    std::chrono::duration<double, std::milli>(
                            refinementTransferEnd -
                            refinementTransferStart).count();
            result.metrics.deviceToHostBytes +=
                    deviceFinishRefinements.Bytes();
            for (std::size_t index = 0u;
                 index < candidates.size(); ++index) {
                if (!finishRefinementRequired[index]) {
                    continue;
                }
                if (finishRefinements[index].failed ||
                    !finishRefinements[index].present ||
                    !finishRefinements[index].estimate.IsValid()) {
                    result.status =
                            CudaTimelineStatus::DeviceFailure;
                    result.diagnostic =
                            "CUDA finish refinement failed";
                    return result;
                }
                states[index].finishTime.present = true;
                states[index].finishTime.value =
                        finishRefinements[index].estimate;
            }
        }

        result.candidates.resize(candidates.size());
        result.status = CudaTimelineStatus::Success;
        for (std::size_t index = 0u;
             index < candidates.size(); ++index) {
            CudaCandidateTimelineOutput &output =
                    result.candidates[index];
            const DeviceTimelineResult &deviceOutput =
                    deviceResults[index];
            output.status = deviceOutput.status;
            output.failureTick = deviceOutput.failureTick;
            output.failureDetail = deviceOutput.failureDetail;
            output.executedTickCount =
                    deviceOutput.executedTickCount;
            output.executedRespawnCount =
                    deviceOutput.executedRespawnCount;
            output.finalState = states[index];
            const DeviceTimelineDescriptor &descriptor =
                    descriptors[index];
            const std::uint32_t outputObservationCount =
                    std::min(
                            deviceOutput.observationCount,
                            descriptor.observationCapacity);
            output.observations.assign(
                    observations.begin() +
                            descriptor.firstObservation,
                    observations.begin() +
                            descriptor.firstObservation +
                            outputObservationCount);
            if (output.status != CudaTimelineStatus::Success &&
                result.status == CudaTimelineStatus::Success) {
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
                std::string("CUDA timeline kernel completed with status ") +
                CudaTimelineStatusName(result.status);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = CudaTimelineStatus::CapacityExceeded;
        result.diagnostic =
                "CUDA timeline host allocation failed";
        return result;
    } catch (...) {
        result.status = CudaTimelineStatus::DeviceFailure;
        result.diagnostic =
                "unexpected CUDA timeline execution failure";
        return result;
    }
}

}  // namespace forevervalidator::simulation
