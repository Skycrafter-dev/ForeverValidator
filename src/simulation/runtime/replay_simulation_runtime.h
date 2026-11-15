#ifndef TMNF_REPLAY_SIMULATION_RUNTIME_H
#define TMNF_REPLAY_SIMULATION_RUNTIME_H

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <forevervalidator/validation.h>

#include "simulation/backends/optimized_cpu/optimized_cpu_static_scene_fingerprint.h"
#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_dyna_frame_state.h"
#include "simulation/runtime/replay_physics_world.h"
#include "simulation/runtime/replay_simulation_result.h"
#include "simulation/runtime/replay_vehicle_body.h"
#include "simulation/runtime/replay_vehicle_simulation.h"
#include "engine/game/trackmania_race.h"
#include "engine/scene/scene_vehicle_car.h"
class CTrackManiaRace;
class ReplayMapScene;
struct ReplaySimulationDefinition;

struct ReplaySimulationStepExecution {
    ReplaySimulationRunResult result = ReplaySimulationRunResult::Success;
    ReplayDynaFrameState simulatedFrame{};
    ReplayDynaFrameState writeFrame{};
    std::optional<std::uint32_t> finishTickMs;
    std::optional<forevervalidator::FinishTimeEstimate> finishTime;
    std::uint32_t respawnExecutedCount = 0u;
};

struct ReplayRaceCameraVehicleState {
    float signedSpeed = 0.0f;
    float turbo = 0.0f;
    float cameraFlightTransition = 0.0f;
    bool burning = false;
    bool gearChanged = false;
    std::array<bool, 4> wheelContact{{true, true, true, true}};
    std::array<bool, 4> wheelHasSurface{{true, true, true, true}};
    GmVec3 cameraSupportUp{0.0f, 1.0f, 0.0f};
};


ReplayStuntSimulationState BuildReplayStuntSimulationState(
        const ReplaySimulationStepExecution &execution,
        const CSceneVehicleCar::SVehicleCarState &physics,
        const ReplayControlTick &tick);

class ReplaySimulationRuntime {
public:
    enum class Phase : std::uint8_t {
        Detached,
        Idle,
        Stepping,
    };

    struct RuntimeClone {
        ReplayPhysicsWorld::RuntimeClone world{};
        ReplayVehicleBody::RuntimeClone body{};
        ReplayVehicleSimulation::RuntimeClone vehicle{};
        std::optional<forevervalidator::FinishTimeEstimate> finishTime;
        bool firstStep = true;
        bool stuntsEnabled = false;
    };
    ReplaySimulationRuntime(
            CTrackManiaRace &race,
            forevervalidator::SimulationBackend backend);
    ~ReplaySimulationRuntime();

    ReplaySimulationRuntime(const ReplaySimulationRuntime &) = delete;
    ReplaySimulationRuntime &operator=(const ReplaySimulationRuntime &) = delete;

    ReplaySimulationRunResult Start(
            const ReplaySimulationDefinition &definition,
            ReplayMapScene &mapScene,
            const GmIso4 &spawnLocation,
            const ReplayControlTick &firstTick,
            std::uint32_t validationSeed);
    ReplaySimulationStepExecution Step(
            const ReplayControlTick &tick);
    ReplaySimulationStepExecution StepOptimizedCpu(
            const ReplayControlTick &tick);
    ReplaySimulationStepExecution StepOptimizedCpuNativeBinary32(
            const ReplayControlTick &tick);
    void PrepareOptimizedCpuStaticTransforms(void) noexcept;
    void CertifyOptimizedCpuStaticTransformsForAdvance(void) noexcept;
    std::optional<OptimizedCpuStaticSceneFingerprint>
            CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                    const CHmsCollisionManagerSZone &expectedPersistentZone)
                    const noexcept;
    std::optional<std::uint32_t> FinishTimeMs() const;
    std::optional<forevervalidator::FinishTimeEstimate> FinishTime() const;
    std::optional<std::uint32_t> StuntsScore() const;
    ReplayDynaFrameState CurrentFrame() const;
    ReplayVehicleControlState CurrentControls() const;
    ReplayRaceCameraVehicleState CurrentRaceCameraState() const;
    // The fraction along start..start+delta of the first static collision
    // face the segment enters (the race cameras' query), if any.
    std::optional<float> IntersectStaticSegment(
            const GmVec3 &start, const GmVec3 &delta) const;
    CSceneVehicleCar::SConditionState CurrentConditionState() const;
    const ReplayRaceProgress &RaceProgress() const;
    std::optional<std::uint32_t> ApplyReplayStuntTimePenalty(
            std::uint32_t overtimeMs);
    std::optional<RuntimeClone> CaptureRuntimeClone() const;
    bool CaptureRuntimeClone(RuntimeClone &clone) const;
    std::optional<RuntimeClone>
            CaptureVehiclePrefixReferenceForTesting(float dt);
    std::optional<RuntimeClone>
            CaptureVehicleForceReferenceForTesting(float dt);
    std::optional<RuntimeClone>
            CaptureCollisionSubstepReferenceForTesting(float dt);
    std::optional<RuntimeClone>
            CapturePreCollisionReferenceForTesting(float dt);
    std::optional<RuntimeClone>
            CaptureForcePassReferenceForTesting(float dt);
    std::optional<std::vector<GmCollision>>
            CaptureCollisionReferenceForTesting(void);
    bool ApplyCollisionResponseReferenceForTesting(void);
    std::uint32_t DynamicCollisionCorpusCountForTesting(void) const;
    bool StepPhysicsKernelReferenceForTesting(
            const ReplayControlTick &tick);
    bool PrepareStepForTesting(const ReplayControlTick &tick);
    bool PrepareRuntimeCloneRestore(const RuntimeClone &clone);
    void RestoreRuntimeClone(RuntimeClone clone) noexcept;
    Phase CurrentPhase() const noexcept;

private:
    void EstimateFinishTime(
            const ReplayControlTick &tick,
            std::uint8_t physicsPath,
            RuntimeClone preTickRuntime,
            CTrackManiaRace::RuntimeClone preTickRace);
    bool ProbeFinishSubstep(float dt, std::uint8_t physicsPath);

    struct State;
    std::unique_ptr<State> state_;
};

std::uint64_t ReplaySimulationRuntimeSemanticHash(
        const ReplaySimulationRuntime::RuntimeClone &clone);
std::uint64_t ReplayRaceRuntimeSemanticHash(
        const CTrackManiaRace::RuntimeClone &clone);

#endif
