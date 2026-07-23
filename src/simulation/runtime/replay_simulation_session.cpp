#include "simulation/runtime/replay_simulation_session.h"
#include <new>
#include <utility>

#include "engine/core/binary32_math.h"
#include "simulation/replay/replay_map_scene.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"
#include "simulation/runtime/replay_environment.h"
#include "simulation/runtime/replay_physics_world.h"
#include "simulation/runtime/replay_simulation_runtime.h"
#include "simulation/runtime/replay_vehicle_body.h"
#include "simulation/runtime/replay_vehicle_simulation.h"
#include "engine/game/trackmania_race.h"
#include "simulation/runtime/replay_deterministic_execution.h"
namespace {

ReplayTrajectoryObservation ObserveReplayTrajectory(
        const ReplaySimulationStepExecution &execution,
        const ReplayControlTick &tick) {
    ReplayTrajectoryObservation observation;
    observation.simulatedPosition = execution.simulatedFrame.position;
    observation.writePosition = execution.writeFrame.position;
    observation.finishTickMs = execution.finishTickMs;
    if (!tick.comparisonTarget.has_value()) {
        return observation;
    }

    ReplayTrajectoryDeviation comparison;
    comparison.targetPosition = *tick.comparisonTarget;
    comparison.delta = {
            observation.writePosition.x - comparison.targetPosition.x,
            observation.writePosition.y - comparison.targetPosition.y,
            observation.writePosition.z - comparison.targetPosition.z};
    const float horizontalDistanceSquared =
            comparison.delta.x * comparison.delta.x +
            comparison.delta.y * comparison.delta.y;
    comparison.distance = CIsqrt(
            horizontalDistanceSquared +
            comparison.delta.z * comparison.delta.z);
    observation.comparison = comparison;
    return observation;
}

}  // namespace

struct ReplaySimulationInstance {
    CTrackManiaRace race;
    std::unique_ptr<ReplaySimulationRuntime> runtime;
    std::uint32_t incrementalRespawnCount = 0u;

    void ResetRuntime() {
        runtime.reset();
        incrementalRespawnCount = 0u;
    }
};

struct ReplaySimulationSession::Impl {
    explicit Impl(forevervalidator::SimulationBackend requestedBackend)
        : backend(requestedBackend) {}

    forevervalidator::SimulationBackend backend;
    ReplayMapScene mapScene;
    ReplaySimulationInstance instance;

    void ResetRuntime() { instance.ResetRuntime(); }
};

ReplaySimulationSession::ReplaySimulationSession(
        forevervalidator::SimulationBackend backend)
    : impl(std::make_unique<Impl>(backend)) {}

ReplaySimulationSession::~ReplaySimulationSession() = default;

void ReplaySimulationSession::Reset() {
    impl->ResetRuntime();
    impl->mapScene.Reset(impl->instance.race);
}

bool ReplaySimulationSession::PreloadChallenge(
        CGameCtnChallengeConstruction &construction) {
    return impl->mapScene.PreloadChallenge(construction) ==
           ReplayMapSceneResult::Ready;
}

bool ReplaySimulationSession::InstallStaticScene(
        StaticSceneModelCollection models) {
    return impl->mapScene.InstallModels(std::move(models)) ==
           ReplayMapSceneResult::Ready;
}

void ReplaySimulationSession::ActivateStaticScene() {
    impl->mapScene.Activate();
}

void ReplaySimulationSession::ConfigureReplayRace(
        EChallengePlayMode playMode,
        bool isLapRace,
        std::uint32_t lapCount) {
    impl->instance.race.SetReplayChallengePlayMode(playMode);
    impl->instance.race.InitNbLapsAndCheckpoints(
            isLapRace ? lapCount : 1u);
}

ReplaySimulationTimelineResult ReplaySimulationSession::SimulateTimeline(
        const ReplaySimulationDefinition &simulationDefinition,
        const std::vector<ReplayControlTick> &controlTicks,
        std::uint32_t validationSeed) {
    ReplaySimulationTimelineResult result;
    if (controlTicks.empty()) {
        return result;
    }

    if (!tmnf::simulation::DeterministicExecutionScope::IsActive()) {
        result.result =
                ReplaySimulationRunResult::DeterministicExecutionUnavailable;
        return result;
    }
    impl->ResetRuntime();

    const ReplayMapSceneResult readyResult =
            impl->mapScene.EnsureReady(impl->instance.race);
    if (readyResult != ReplayMapSceneResult::Ready) {
        result.result = MapReplaySceneResult(readyResult);
        return result;
    }
    GmIso4 startLocation;
    if (!impl->mapScene.FirstStartLineSpawnLocation(startLocation)) {
        result.result = ReplaySimulationRunResult::MapStartUnavailable;
        return result;
    }

    impl->instance.runtime = std::make_unique<ReplaySimulationRuntime>(
            impl->instance.race, impl->backend);
    result.result = impl->instance.runtime->Start(
            simulationDefinition,
            impl->mapScene,
            startLocation,
            controlTicks.front(),
            validationSeed);
    if (result.result != ReplaySimulationRunResult::Success) {
        return result;
    }

    if (impl->backend == forevervalidator::SimulationBackend::OptimizedCpu) {
        impl->instance.runtime->PrepareOptimizedCpuStaticTransforms();
        impl->instance.runtime->
                CertifyOptimizedCpuStaticTransformsForAdvance();
        const forevervalidator::simulation::OptimizedCpuBinary32MathPath
                binary32MathPath = forevervalidator::simulation::
                        SelectOptimizedCpuBinary32MathPathForActiveExecution();
        if (binary32MathPath == forevervalidator::simulation::
                                        OptimizedCpuBinary32MathPath::X86Sse2) {
            for (const ReplayControlTick &tick : controlTicks) {
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->
                                StepOptimizedCpuNativeBinary32(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                result.executedRespawnCount +=
                        execution.respawnExecutedCount;

                if (tick.observe) {
                    try {
                        result.observations.push_back(
                                ObserveReplayTrajectory(execution, tick));
                    } catch (const std::bad_alloc &) {
                        result.result =
                                ReplaySimulationRunResult::ObservationAllocationFailed;
                        return result;
                    }
                }
            }
        } else {
            for (const ReplayControlTick &tick : controlTicks) {
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->StepOptimizedCpu(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                result.executedRespawnCount +=
                        execution.respawnExecutedCount;

                if (tick.observe) {
                    try {
                        result.observations.push_back(
                                ObserveReplayTrajectory(execution, tick));
                    } catch (const std::bad_alloc &) {
                        result.result =
                                ReplaySimulationRunResult::ObservationAllocationFailed;
                        return result;
                    }
                }
            }
        }
    } else {
        for (const ReplayControlTick &tick : controlTicks) {
            const ReplaySimulationStepExecution execution =
                    impl->instance.runtime->Step(tick);
            if (execution.result != ReplaySimulationRunResult::Success) {
                result.result = execution.result;
                return result;
            }
            result.executedRespawnCount +=
                    execution.respawnExecutedCount;

            if (tick.observe) {
                try {
                    result.observations.push_back(
                            ObserveReplayTrajectory(execution, tick));
                } catch (const std::bad_alloc &) {
                    result.result =
                            ReplaySimulationRunResult::ObservationAllocationFailed;
                    return result;
                }
            }
        }
    }
    result.finishTimeMs = impl->instance.runtime->FinishTimeMs();
    result.stuntsScore = impl->instance.runtime->StuntsScore();
    result.raceCompleted = result.finishTimeMs.has_value();
    result.result = ReplaySimulationRunResult::Success;
    return result;
}

ReplaySimulationRunResult ReplaySimulationSession::StartIncremental(
        const ReplaySimulationDefinition &simulationDefinition,
        const ReplayControlTick &firstTick,
        std::uint32_t validationSeed) {
    if (!tmnf::simulation::DeterministicExecutionScope::IsActive()) {
        return ReplaySimulationRunResult::DeterministicExecutionUnavailable;
    }
    impl->ResetRuntime();
    const ReplayMapSceneResult readyResult =
            impl->mapScene.EnsureReady(impl->instance.race);
    if (readyResult != ReplayMapSceneResult::Ready) {
        return MapReplaySceneResult(readyResult);
    }
    GmIso4 startLocation;
    if (!impl->mapScene.FirstStartLineSpawnLocation(startLocation)) {
        return ReplaySimulationRunResult::MapStartUnavailable;
    }
    impl->instance.runtime = std::make_unique<ReplaySimulationRuntime>(
            impl->instance.race, impl->backend);
    const ReplaySimulationRunResult result = impl->instance.runtime->Start(
            simulationDefinition,
            impl->mapScene,
            startLocation,
            firstTick,
            validationSeed);
    if (result == ReplaySimulationRunResult::Success &&
        impl->backend == forevervalidator::SimulationBackend::OptimizedCpu) {
        impl->instance.runtime->PrepareOptimizedCpuStaticTransforms();
    }
    return result;
}

ReplaySimulationTimelineResult ReplaySimulationSession::AdvanceIncremental(
        const std::vector<ReplayControlTick> &controlTicks,
        std::size_t begin,
        std::size_t count) {
    ReplaySimulationTimelineResult result;
    if (!impl->instance.runtime || begin > controlTicks.size() ||
        count > controlTicks.size() - begin) {
        return result;
    }
    if (impl->backend == forevervalidator::SimulationBackend::OptimizedCpu) {
        impl->instance.runtime->
                CertifyOptimizedCpuStaticTransformsForAdvance();
        const forevervalidator::simulation::OptimizedCpuBinary32MathPath
                binary32MathPath = forevervalidator::simulation::
                        SelectOptimizedCpuBinary32MathPathForActiveExecution();
        if (binary32MathPath == forevervalidator::simulation::
                                        OptimizedCpuBinary32MathPath::X86Sse2) {
            for (std::size_t index = begin; index < begin + count; ++index) {
                const ReplayControlTick &tick = controlTicks[index];
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->
                                StepOptimizedCpuNativeBinary32(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                impl->instance.incrementalRespawnCount +=
                        execution.respawnExecutedCount;
            }
        } else {
            for (std::size_t index = begin; index < begin + count; ++index) {
                const ReplayControlTick &tick = controlTicks[index];
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->StepOptimizedCpu(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                impl->instance.incrementalRespawnCount +=
                        execution.respawnExecutedCount;
            }
        }
    } else {
        for (std::size_t index = begin; index < begin + count; ++index) {
            const ReplayControlTick &tick = controlTicks[index];
            const ReplaySimulationStepExecution execution =
                    impl->instance.runtime->Step(tick);
            if (execution.result != ReplaySimulationRunResult::Success) {
                result.result = execution.result;
                return result;
            }
            impl->instance.incrementalRespawnCount +=
                    execution.respawnExecutedCount;
        }
    }
    result.finishTimeMs = impl->instance.runtime->FinishTimeMs();
    result.stuntsScore = impl->instance.runtime->StuntsScore();
    result.raceCompleted = result.finishTimeMs.has_value();
    result.executedRespawnCount = impl->instance.incrementalRespawnCount;
    result.result = ReplaySimulationRunResult::Success;
    return result;
}

std::optional<ReplaySimulationStateView>
ReplaySimulationSession::CurrentState() const {
    if (!impl->instance.runtime) {
        return std::nullopt;
    }
    ReplaySimulationStateView result;
    result.frame = impl->instance.runtime->CurrentFrame();
    result.controls = impl->instance.runtime->CurrentControls();
    result.race = impl->instance.runtime->RaceProgress();
    result.finishTimeMs = impl->instance.runtime->FinishTimeMs();
    result.stuntsScore = impl->instance.runtime->StuntsScore();
    result.respawnCount = impl->instance.incrementalRespawnCount;
    return result;
}

std::optional<std::uint32_t>
ReplaySimulationSession::ApplyReplayStuntTimePenalty(
        std::uint32_t overtimeMs) {
    if (!impl->instance.runtime) {
        return std::nullopt;
    }
    return impl->instance.runtime->ApplyReplayStuntTimePenalty(overtimeMs);
}

std::shared_ptr<const ReplaySimulationInstanceClone>
ReplaySimulationSession::CaptureRuntimeClone() const {
    if (!impl->instance.runtime ||
        impl->instance.runtime->CurrentPhase() !=
                                  ReplaySimulationRuntime::Phase::Idle) {
        return {};
    }
    std::optional<ReplaySimulationRuntime::RuntimeClone> runtime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!runtime.has_value()) {
        return {};
    }
    auto clone = std::make_shared<ReplaySimulationInstanceClone>();
    clone->race = impl->instance.race.CaptureRuntimeClone();
    clone->runtime = std::move(*runtime);
    clone->incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    return clone;
}

bool ReplaySimulationSession::PrepareRuntimeCloneRestore(
        const ReplaySimulationInstanceClone &clone) {
    return impl->instance.runtime &&
           impl->instance.race.PrepareRuntimeCloneRestore(clone.race) &&
           impl->instance.runtime->PrepareRuntimeCloneRestore(clone.runtime);
}

void ReplaySimulationSession::RestoreRuntimeClone(
        ReplaySimulationInstanceClone clone) noexcept {
    impl->instance.race.RestoreRuntimeClone(std::move(clone.race));
    impl->instance.runtime->RestoreRuntimeClone(std::move(clone.runtime));
    impl->instance.incrementalRespawnCount = clone.incrementalRespawnCount;
}

std::optional<OptimizedCpuStaticSceneFingerprint>
ReplaySimulationSession::
        CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                void) const noexcept {
    if (!impl->instance.runtime) {
        return std::nullopt;
    }
    return impl->instance.runtime->
            CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                    impl->mapScene.PersistentCollisionZoneForTesting());
}
