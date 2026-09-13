#ifndef TMNF_REPLAY_SIMULATION_SESSION_H
#define TMNF_REPLAY_SIMULATION_SESSION_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <forevervalidator/experimental/physics_sandbox.h>
#include "engine/game/game_ctn_types.h"
#include "simulation/replay/replay_challenge_construction.h"
#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_simulation_definition.h"
#include "simulation/runtime/replay_simulation_result.h"
#include "simulation/runtime/replay_simulation_runtime.h"
#include "simulation/runtime/replay_trajectory_observation.h"
#include "simulation/runtime/replay_dyna_frame_state.h"
#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_static_configuration_storage.h"
#include "simulation/backends/cuda/cuda_timeline_executor.h"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/cuda/cuda_execution_context.h"
#include "engine/scene/static_scene_model.h"
#include "engine/game/trackmania_race.h"
struct ReplaySimulationTimelineResult {
    ReplaySimulationRunResult result =
            ReplaySimulationRunResult::InvalidControlTimeline;
    std::vector<ReplayTrajectoryObservation> observations;
    bool raceCompleted = false;
    std::optional<std::uint32_t> finishTimeMs;
    std::optional<forevervalidator::FinishTimeEstimate> finishTime;
    std::optional<std::uint32_t> stuntsScore;
    std::uint32_t executedRespawnCount = 0u;
};

struct ReplaySimulationStateView {
    ReplayDynaFrameState frame{};
    ReplayVehicleControlState controls{};
    ReplayRaceProgress race{};
    std::optional<std::uint32_t> finishTimeMs;
    std::optional<forevervalidator::FinishTimeEstimate> finishTime;
    std::optional<std::uint32_t> stuntsScore;
    std::uint32_t respawnCount = 0u;
    float signedSpeed = 0.0f;
    float turbo = 0.0f;
    float cameraFlightTransition = 0.0f;
    bool burning = false;
    bool gearChanged = false;
    std::array<bool, 4> wheelContact{{true, true, true, true}};
    std::array<bool, 4> wheelHasSurface{{true, true, true, true}};
    GmVec3 cameraSupportUp{0.0f, 1.0f, 0.0f};
    GmVec3 localSpeed{};
    bool freeWheeling = false;
    bool lateralContact = false;
    bool sliding = false;
    std::int32_t gear = 0;
    float rpm = 0.0f;
    float turningRate = 0.0f;
    std::uint32_t turboType = 0u;
    float turboBoostFactor = 0.0f;
    std::array<bool, 4> wheelSliding{{false, false, false, false}};
    std::array<std::uint16_t, 4> wheelSurface{{0u, 0u, 0u, 0u}};
};

struct ReplayCudaVehiclePrefixDifferential {
    bool success = false;
    std::size_t checkedBytes = 0u;
    std::size_t firstMismatchByte = SIZE_MAX;
    std::uint8_t cpuByte = 0u;
    std::uint8_t gpuByte = 0u;
    std::string diagnostic;
};

struct ReplayStaticCollisionTriangle {
    GmVec3 a{};
    GmVec3 b{};
    GmVec3 c{};
};

struct ReplaySimulationInstanceClone {
    CTrackManiaRace::RuntimeClone race;
    ReplaySimulationRuntime::RuntimeClone runtime;
    std::uint32_t incrementalRespawnCount = 0u;
    std::uint32_t randomState = 1u;
};

std::uint64_t ReplaySimulationInstanceSemanticHash(
        const ReplaySimulationInstanceClone &clone);

void ClassifyPhysicsSandboxRenderLayers(
        forevervalidator::experimental::PhysicsSandboxRenderScene &scene);

std::uint32_t PhysicsSandboxRenderLodLevelForVisualMip(
        std::uint32_t parentLodLevel,
        std::uint32_t mipLevel,
        std::uint32_t mipLevelCount);

class ReplaySimulationSession {
public:
    explicit ReplaySimulationSession(
            forevervalidator::SimulationBackend backend);
    ~ReplaySimulationSession();

    ReplaySimulationSession(const ReplaySimulationSession &) = delete;
    ReplaySimulationSession &operator=(const ReplaySimulationSession &) = delete;

    std::unique_ptr<ReplaySimulationSession> ClonePrepared() const;
    std::optional<forevervalidator::simulation::CudaExecutionContext>
    CaptureCudaExecutionContext(std::uint64_t cursor) const;

    void Reset();
    bool PreloadChallenge(CGameCtnChallengeConstruction &construction);
    bool InstallStaticScene(StaticSceneModelCollection models);
    void ActivateStaticScene();
    void ConfigureReplayRace(EChallengePlayMode playMode,
                             bool isLapRace,
                             std::uint32_t lapCount);

    ReplaySimulationTimelineResult SimulateTimeline(
            const ReplaySimulationDefinition &simulationDefinition,
            const std::vector<ReplayControlTick> &controlTicks,
            std::uint32_t validationSeed);
    ReplaySimulationRunResult StartIncremental(
            const ReplaySimulationDefinition &simulationDefinition,
            const ReplayControlTick &firstTick,
            std::uint32_t validationSeed);
    ReplaySimulationTimelineResult AdvanceIncremental(
            const std::vector<ReplayControlTick> &controlTicks,
            std::size_t begin,
            std::size_t count);
    std::optional<ReplaySimulationStateView> CurrentState() const;
    std::optional<OptimizedCpuStaticSceneFingerprint>
            CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                    void) const noexcept;
    std::optional<
            forevervalidator::simulation::CudaSceneTransferMetrics>
            CudaSceneTransferMetricsForTesting(void) const;
    std::optional<forevervalidator::simulation::
                          CudaStaticConfigurationTransferMetrics>
            CudaStaticConfigurationTransferMetricsForTesting(void) const;
    const std::string &CudaInitializationDiagnostic() const noexcept;
    std::optional<forevervalidator::simulation::
                          CudaTimelineExecutionMetrics>
            CudaTimelineMetricsForTesting(void) const;
    forevervalidator::simulation::CudaTimelineBatchResult
            ExecuteCudaCandidateBatchForTesting(
                    const std::vector<ReplayControlTick> &ticks,
                    std::uint32_t candidateCount,
                    bool mutateControls,
                    std::uint64_t initialControlCursor,
                    bool cancellationRequested = false);
    ReplayCudaVehiclePrefixDifferential
            RunCudaCandidateBatchDifferentialForTesting(
                    const std::vector<ReplayControlTick> &ticks,
                    std::uint32_t candidateCount,
                    bool mutateControls,
                    std::uint64_t initialControlCursor);
    ReplayCudaVehiclePrefixDifferential
            RunCudaVehiclePrefixDifferentialForTesting(float dt);
    ReplayCudaVehiclePrefixDifferential
            RunCudaVehicleForceDifferentialForTesting(float dt);
    ReplayCudaVehiclePrefixDifferential
            RunCudaCollisionDifferentialForTesting(void);
    ReplayCudaVehiclePrefixDifferential
            RunCudaPhysicsStepDifferentialForTesting(void);
    ReplayCudaVehiclePrefixDifferential
            RunCudaCollisionSubstepDifferentialForTesting(float dt);
    ReplayCudaVehiclePrefixDifferential
            RunCudaPreCollisionDifferentialForTesting(float dt);
    ReplayCudaVehiclePrefixDifferential
            RunCudaTimelineTickDifferentialForTesting(
                    const ReplayControlTick &tick);
    bool StageCudaTimelinePrefixForTesting(
            const ReplayControlTick &tick);
    bool StageCollisionSubstepForTesting(float dt);
    bool StageCudaPreCollisionForTesting(float dt = 0.0f);
    const std::vector<ReplayStaticCollisionTriangle> &
            StaticCollisionTriangles() const noexcept;
    forevervalidator::experimental::PhysicsSandboxRenderSceneHandle
            StaticRenderScene() const noexcept;
    std::optional<std::uint32_t> ApplyReplayStuntTimePenalty(
            std::uint32_t overtimeMs);
    std::shared_ptr<const ReplaySimulationInstanceClone>
            CaptureRuntimeClone() const;
    bool PrepareCudaSearchSpecialization(std::string *diagnostic);
    std::shared_ptr<const forevervalidator::simulation::cuda::specialization::
                            SessionModule>
            CudaSearchSpecialization() const noexcept;
    const std::string &CudaSearchSpecializationDiagnostic() const noexcept;
    std::unique_ptr<forevervalidator::simulation::CudaSearchExecutor>
            CreateCudaSearchExecutor(
                    forevervalidator::simulation::
                            CudaSearchExecutorConfiguration configuration,
                    std::uint64_t initialControlCursor,
                    std::string *diagnostic) const;
    bool PrepareRuntimeCloneRestore(
            const ReplaySimulationInstanceClone &clone);
    void RestoreRuntimeClone(ReplaySimulationInstanceClone clone) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif
