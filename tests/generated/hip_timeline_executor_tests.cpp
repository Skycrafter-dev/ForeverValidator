// Generated from tests/cuda_timeline_executor_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_scene_storage.h"
#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_timeline_executor.h"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

template<typename T>
class DeviceValue {
public:
    explicit DeviceValue(const T &value) {
        if (hipMalloc(&data_, sizeof(T)) == hipSuccess) {
            if (hipMemcpy(
                        data_, &value, sizeof(T),
                        hipMemcpyHostToDevice) != hipSuccess) {
                hipFree(data_);
                data_ = nullptr;
            }
        }
    }
    ~DeviceValue() {
        if (data_ != nullptr) hipFree(data_);
    }
    DeviceValue(const DeviceValue &) = delete;
    DeviceValue &operator=(const DeviceValue &) = delete;
    const void *Get() const { return data_; }

private:
    void *data_ = nullptr;
};

}  // namespace

int main() {
    using namespace forevervalidator::simulation;
    std::vector<HipCandidateTimelineOutput> ranked(3u);
    for (std::size_t index = 0u; index < ranked.size(); ++index) {
        ranked[index].status = HipTimelineStatus::Success;
        ranked[index].finalState.candidateId =
                static_cast<std::uint32_t>(10u + index);
    }
    ranked[0].finalState.race.progress.completedLapCount = 1u;
    ranked[1].finalState.race.progress.raceCompleted = true;
    ranked[1].finalState.race.progress.completedLapCount = 1u;
    ranked[1].finalState.race.progress.lastPrepareTimeMs = 1200u;
    ranked[2].finalState.race.progress.raceCompleted = true;
    ranked[2].finalState.race.progress.completedLapCount = 1u;
    ranked[2].finalState.race.progress.lastPrepareTimeMs = 1100u;
    if (SelectHipTimelineWinner(ranked) != 2u) {
        std::cerr << "race candidate winner ordering is not exact\n";
        return 1;
    }
    ranked[1].finalState.race.progress.lastPrepareTimeMs = 1100u;
    ranked[1].finalState.finishTime.value =
            forevervalidator::FinishTimeEstimate{
                    1099999998u, 1099999999u, 1099999999u};
    ranked[1].finalState.finishTime.present = true;
    ranked[2].finalState.finishTime.value =
            forevervalidator::FinishTimeEstimate{
                    1099999999u, 1100000000u, 1100000000u};
    ranked[2].finalState.finishTime.present = true;
    if (SelectHipTimelineWinner(ranked) != 1u) {
        std::cerr << "race candidate winner ignored nanosecond estimate\n";
        return 1;
    }
    for (HipCandidateTimelineOutput &candidate : ranked) {
        candidate.finalState.race.replayPlayMode =
                static_cast<std::uint32_t>(
                        EChallengePlayMode::Stunts);
    }
    ranked[0].finalState.stunts.stuntsScore = 100u;
    ranked[1].finalState.stunts.stuntsScore = 250u;
    ranked[2].finalState.stunts.stuntsScore = 200u;
    if (SelectHipTimelineWinner(ranked) != 1u) {
        std::cerr << "stunt candidate winner ordering is not exact\n";
        return 1;
    }

    HipPackedSceneHeader scene;
    scene.totalSize = sizeof(scene);
    HipPackedStaticConfigurationHeader configuration;
    configuration.totalSize = sizeof(configuration);
    DeviceValue<HipPackedSceneHeader> deviceScene(scene);
    DeviceValue<HipPackedStaticConfigurationHeader>
            deviceConfiguration(configuration);
    if (deviceScene.Get() == nullptr ||
        deviceConfiguration.Get() == nullptr) {
        std::cerr << "test device input allocation failed\n";
        return 1;
    }

    ReplayControlTick sourceTick;
    sourceTick.periodMs = 10u;
    sourceTick.timeMs = 1234u;
    sourceTick.controls = {0.25f, 0.75f, -0.5f};
    sourceTick.actions.enableRaceSimulation = true;
    sourceTick.observe = true;
    sourceTick.comparisonTarget = GmVec3{1.0f, 2.0f, 3.0f};
    HipControlTick tick = FlattenHipControlTick(sourceTick);
    if ((tick.actionFlags &
         HipControlActionEnableRaceSimulation) == 0u ||
        !tick.hasComparisonTarget) {
        std::cerr << "control flattening lost timeline inputs\n";
        return 1;
    }

    std::vector<HipCandidateTimelineInput> batch(256u);
    for (std::size_t index = 0u; index < batch.size(); ++index) {
        batch[index].initialState.candidateId =
                static_cast<std::uint32_t>(index);
        batch[index].initialState.firstStep = false;
        batch[index].ticks.push_back(tick);
    }
    HipTimelineBatchResult executed = ExecuteHipTimelineBatch(
            deviceScene.Get(), deviceConfiguration.Get(), batch);
    if (executed.status != HipTimelineStatus::Success ||
        executed.candidates.size() != batch.size() ||
        executed.metrics.candidateCount != batch.size() ||
        executed.metrics.tickCount != batch.size() ||
        executed.metrics.kernelMilliseconds < 0.0) {
        std::cerr << "batched HIP timeline launch failed: "
                  << executed.diagnostic << '\n';
        return 1;
    }
    for (std::size_t index = 0u;
         index < executed.candidates.size(); ++index) {
        const HipCandidateTimelineOutput &candidate =
                executed.candidates[index];
        if (candidate.status != HipTimelineStatus::Success ||
            candidate.failureTick != UINT32_MAX ||
            candidate.executedTickCount != 1u ||
            candidate.finalState.candidateId != index ||
            candidate.finalState.world.schemePeriodMs != 10u ||
            candidate.finalState.world.tickTimeMs != 1234u ||
            candidate.finalState.vehicle.controls.steeringControl !=
                    -0.5f) {
            std::cerr
                    << "candidate-owned HIP transition state changed: "
                    << "status="
                    << HipTimelineStatusName(candidate.status)
                    << " failure_tick=" << candidate.failureTick
                    << " executed=" << candidate.executedTickCount
                    << " candidate_id="
                    << candidate.finalState.candidateId
                    << " period="
                    << candidate.finalState.world.schemePeriodMs
                    << " time="
                    << candidate.finalState.world.tickTimeMs
                    << " steering="
                    << candidate.finalState.vehicle.controls.
                            steeringControl
                    << '\n';
            return 1;
        }
    }

    std::vector<HipCandidateTimelineInput> largeBatch(4097u);
    for (std::size_t index = 0u; index < largeBatch.size(); ++index) {
        largeBatch[index].initialState.candidateId =
                static_cast<std::uint32_t>(index);
        largeBatch[index].initialState.firstStep = false;
        largeBatch[index].ticks.push_back(tick);
    }
    const HipTimelineBatchResult large =
            ExecuteHipTimelineBatch(
                    deviceScene.Get(), deviceConfiguration.Get(),
                    largeBatch);
    if (large.status != HipTimelineStatus::Success ||
        large.candidates.size() != largeBatch.size() ||
        large.metrics.candidateCount != largeBatch.size() ||
        large.winnerCandidateId != 0u) {
        std::cerr << "HIP timeline retained an arbitrary 4096-candidate "
                     "limit: "
                  << large.diagnostic << '\n';
        return 1;
    }

    HipTimelineBatchResult cancelled = ExecuteHipTimelineBatch(
            deviceScene.Get(), deviceConfiguration.Get(),
            {batch.front()}, true);
    if (cancelled.status != HipTimelineStatus::Cancelled ||
        cancelled.candidates.size() != 1u ||
        cancelled.candidates[0].failureTick != 0u ||
        cancelled.candidates[0].finalState.world.tickTimeMs != 0u) {
        std::cerr << "HIP cancellation was not deterministic\n";
        return 1;
    }

    HipTimelineBatchResult invalid = ExecuteHipTimelineBatch(
            nullptr, deviceConfiguration.Get(), {batch.front()});
    if (invalid.status != HipTimelineStatus::InvalidArgument) {
        std::cerr << "invalid HIP device inputs were not rejected\n";
        return 1;
    }
    HipCandidateTimelineInput wrongSchema = batch.front();
    ++wrongSchema.initialState.schemaVersion;
    const HipTimelineBatchResult schema =
            ExecuteHipTimelineBatch(
                    deviceScene.Get(), deviceConfiguration.Get(),
                    {wrongSchema});
    if (schema.status != HipTimelineStatus::SchemaMismatch ||
        schema.candidates.size() != 1u ||
        schema.candidates[0].status !=
                HipTimelineStatus::SchemaMismatch) {
        std::cerr << "HIP candidate schema mismatch was not explicit\n";
        return 1;
    }
    HipPackedSceneHeader corruptScene = scene;
    corruptScene.magic = 0u;
    DeviceValue<HipPackedSceneHeader> deviceCorruptScene(corruptScene);
    const HipTimelineBatchResult corrupt =
            ExecuteHipTimelineBatch(
                    deviceCorruptScene.Get(),
                    deviceConfiguration.Get(), {batch.front()});
    if (corrupt.status != HipTimelineStatus::InvalidArgument ||
        corrupt.candidates.size() != 1u ||
        corrupt.candidates[0].status !=
                HipTimelineStatus::InvalidArgument) {
        std::cerr << "corrupt HIP scene header was not rejected\n";
        return 1;
    }
    if (ExecuteHipTimelineBatch(
                deviceScene.Get(), deviceConfiguration.Get(), {}).
                    status != HipTimelineStatus::InvalidArgument) {
        std::cerr << "empty HIP batch was not rejected\n";
        return 1;
    }
    return 0;
}
