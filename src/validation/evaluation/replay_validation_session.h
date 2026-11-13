#ifndef TMNF_REPLAY_VALIDATION_SESSION_H
#define TMNF_REPLAY_VALIDATION_SESSION_H

#include <cstddef>
#include <cstdint>
#include <optional>

#include <forevervalidator/result.h>

#include "format/replay/replay_ghost_trajectory.h"
#include "format/replay/replay_input_timeline.h"
#include "validation/evaluation/replay_validation_model.h"
struct ReplaySimulationDefinition;
class ReplayFile;
class ReplaySimulationSession;

class ReplayValidationReplay {
public:
    ReplayValidationReplay(const ReplayInputTimeline &inputTimeline,
                           const ReplayGhostTrajectory &ghostTrajectory,
                           std::optional<std::uint32_t> stuntsTimeLimitMs =
                                   std::nullopt)
        : inputTimeline_(inputTimeline),
          ghostTrajectory_(ghostTrajectory),
          stuntsTimeLimitMs_(stuntsTimeLimitMs) {}

    ReplayValidationReplay(const ReplayValidationReplay &) = delete;
    ReplayValidationReplay &operator=(const ReplayValidationReplay &) = delete;
    ReplayValidationReplay(ReplayValidationReplay &&) = delete;
    ReplayValidationReplay &operator=(ReplayValidationReplay &&) = delete;

    const ReplayGhostTrajectory &Trajectory() const {
        return ghostTrajectory_;
    }

    const ReplayInputTimeline &InputTimeline() const {
        return inputTimeline_;
    }

    std::optional<std::int32_t> ExpectedRaceTimeMs() const;
    std::optional<std::int32_t> ExpectedStuntsScore() const;
    std::optional<std::int32_t> ExpectedRespawns() const;
    std::size_t ExpectedSamples() const;
    std::uint32_t RequestedSamples(
            const ReplayValidationConfiguration &configuration) const;
    ReplayValidationPlan BuildStreamPlan(
            const ReplayValidationConfiguration &configuration,
            ReplayValidationMode validationMode) const;
    ReplayValidationMetadata BuildMetadata(
            const ReplayValidationConfiguration &configuration) const;

private:
    const ReplayInputTimeline &inputTimeline_;
    const ReplayGhostTrajectory &ghostTrajectory_;
    std::optional<std::uint32_t> stuntsTimeLimitMs_;

    static std::size_t ExpectedSampleCount(
            const ReplayGhostTrajectory &trajectory,
            std::int32_t inputDurationMs,
            std::optional<std::int32_t> expectedRaceTimeMs);
};

enum class ReplayValidationExecutionResult {
    Success,
    MissingInput,
    InvalidPlan,
    ControlPlanInvalidRequest,
    ControlTargetAllocationFailed,
    ControlTargetTimeOutOfRange,
    ControlTargetNonFinite,
    ControlTickReservationFailed,
    ControlTickAllocationFailed,
    ControlTargetMissing,
    ControlOutputMissing,
    InvalidControlPlan,
    PhysicsInputInvalid,
    MapStartUnavailable,
    ObservationAllocationFailed,
    DeterministicExecutionUnavailable,
    CudaUnavailable,
    CudaInitializationFailed,
    CudaExecutionFailed,
    TwinUnavailable,
    TwinNotImplemented,
    TwinExecutionFailed,
};

ReplayValidationExecutionResult ExecuteReplayValidation(
        ReplayValidationExecutionOutput *out,
        ReplaySimulationSession &simulationSession,
        const ReplayValidationPlan &plan,
        const ReplaySimulationDefinition &simulationDefinition,
        const ReplayGhostTrajectory &trajectory,
        const ReplayInputTimeline &inputTimeline);

struct ReplayFileValidationResult {
    ReplayValidationMetadata metadata{};
    ReplayValidationResult validation{};
    ReplayRaceOutcome raceOutcome{};
};

using ReplayFileValidationBuild =
        forevervalidator::DiscriminatedResult<
                ReplayFileValidationResult,
                ReplayValidationExecutionResult>;

std::optional<ReplayFileValidationResult> ClassifyReplayCompatibility(
        const ReplayFile &replay,
        const ReplayValidationConfiguration &configuration);

std::optional<ReplayFileValidationResult> ClassifyReplayInputAvailability(
        const ReplayFile &replay,
        const ReplayValidationConfiguration &configuration);

ReplayFileValidationBuild ValidateReplayFile(
        const ReplayFile &replay,
        ReplayValidationMode validationMode,
        ReplaySimulationSession &simulationSession,
        const ReplaySimulationDefinition &simulationDefinition,
        const ReplayValidationConfiguration &configuration);

#endif
