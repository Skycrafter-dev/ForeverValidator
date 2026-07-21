#ifndef TMNF_REPLAY_SIMULATION_SESSION_H
#define TMNF_REPLAY_SIMULATION_SESSION_H

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "engine/game/game_ctn_types.h"
#include "simulation/replay/replay_challenge_construction.h"
#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_simulation_definition.h"
#include "simulation/runtime/replay_simulation_result.h"
#include "simulation/runtime/replay_simulation_runtime.h"
#include "simulation/runtime/replay_trajectory_observation.h"
#include "simulation/runtime/replay_dyna_frame_state.h"
#include "engine/scene/static_scene_model.h"
#include "engine/game/trackmania_race.h"
struct ReplaySimulationTimelineResult {
    ReplaySimulationRunResult result =
            ReplaySimulationRunResult::InvalidControlTimeline;
    std::vector<ReplayTrajectoryObservation> observations;
    bool raceCompleted = false;
    std::optional<std::uint32_t> finishTimeMs;
    std::optional<std::uint32_t> stuntsScore;
    std::uint32_t executedRespawnCount = 0u;
};

struct ReplaySimulationStateView {
    ReplayDynaFrameState frame{};
    ReplayVehicleControlState controls{};
    ReplayRaceProgress race{};
    std::optional<std::uint32_t> finishTimeMs;
    std::optional<std::uint32_t> stuntsScore;
    std::uint32_t respawnCount = 0u;
};

struct ReplaySimulationInstanceClone {
    CTrackManiaRace::RuntimeClone race;
    ReplaySimulationRuntime::RuntimeClone runtime;
    std::uint32_t incrementalRespawnCount = 0u;
};

class ReplaySimulationSession {
public:
    ReplaySimulationSession();
    ~ReplaySimulationSession();

    ReplaySimulationSession(const ReplaySimulationSession &) = delete;
    ReplaySimulationSession &operator=(const ReplaySimulationSession &) = delete;

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
    std::optional<std::uint32_t> ApplyReplayStuntTimePenalty(
            std::uint32_t overtimeMs);
    std::shared_ptr<const ReplaySimulationInstanceClone>
            CaptureRuntimeClone() const;
    bool PrepareRuntimeCloneRestore(
            const ReplaySimulationInstanceClone &clone);
    void RestoreRuntimeClone(ReplaySimulationInstanceClone clone) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif
