// Generated from src/simulation/backends/cuda/cuda_timeline_executor.cpp by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_timeline_executor.h"

#include <cmath>

namespace forevervalidator::simulation {

HipControlTick FlattenHipControlTick(
        const ReplayControlTick &source) noexcept {
    HipControlTick result;
    result.periodMs = source.periodMs;
    result.timeMs = source.timeMs;
    result.stuntsTimeLimitMs = source.actions.stuntsTimeLimitMs;
    result.respawnAtCheckpointCount =
            source.actions.respawnAtCheckpointCount;
    result.controls = source.controls;
    result.stuntsInput = source.stuntsInput;
    result.observe = source.observe;
    result.hasComparisonTarget =
            source.comparisonTarget.has_value();
    if (source.comparisonTarget.has_value()) {
        result.comparisonTarget = *source.comparisonTarget;
    }
#define SET_ACTION(field, flag)                                                \
    if (source.actions.field) result.actionFlags |= flag
    SET_ACTION(establishRaceSpawn,
               HipControlActionEstablishRaceSpawn);
    SET_ACTION(suppressVehicleForceCallbacks,
               HipControlActionSuppressVehicleForceCallbacks);
    SET_ACTION(enableRaceSimulation,
               HipControlActionEnableRaceSimulation);
    SET_ACTION(resetAtRaceStart,
               HipControlActionResetAtRaceStart);
    SET_ACTION(enableStuntsSimulation,
               HipControlActionEnableStuntsSimulation);
    SET_ACTION(finishRace, HipControlActionFinishRace);
#undef SET_ACTION
    return result;
}

const char *HipTimelineStatusName(HipTimelineStatus status) noexcept {
    switch (status) {
    case HipTimelineStatus::Success: return "success";
    case HipTimelineStatus::InvalidArgument:
        return "invalid_argument";
    case HipTimelineStatus::SchemaMismatch:
        return "schema_mismatch";
    case HipTimelineStatus::CapacityExceeded:
        return "capacity_exceeded";
    case HipTimelineStatus::Cancelled: return "cancelled";
    case HipTimelineStatus::DeviceFailure: return "device_failure";
    case HipTimelineStatus::UnsupportedPhysicsTransition:
        return "unsupported_physics_transition";
    }
    return "unknown";
}

std::optional<std::size_t> SelectHipTimelineWinner(
        const std::vector<HipCandidateTimelineOutput> &candidates) noexcept {
    const auto finalDistance = [](const HipCandidateTimelineOutput &candidate)
            -> std::optional<float> {
        for (std::size_t index = candidate.observations.size();
             index != 0u; --index) {
            const HipTimelineObservation &observation =
                    candidate.observations[index - 1u];
            if (observation.hasComparison &&
                std::isfinite(observation.comparisonDistance)) {
                return observation.comparisonDistance;
            }
        }
        return std::nullopt;
    };
    const auto isBetter = [&](const HipCandidateTimelineOutput &left,
                              const HipCandidateTimelineOutput &right) {
        const bool stunts =
                left.finalState.race.replayPlayMode ==
                static_cast<std::uint32_t>(
                        EChallengePlayMode::Stunts);
        if (stunts &&
            left.finalState.stunts.stuntsScore !=
                    right.finalState.stunts.stuntsScore) {
            return left.finalState.stunts.stuntsScore >
                   right.finalState.stunts.stuntsScore;
        }
        const ReplayRaceProgress &leftRace =
                left.finalState.race.progress;
        const ReplayRaceProgress &rightRace =
                right.finalState.race.progress;
        if (leftRace.raceCompleted != rightRace.raceCompleted) {
            return leftRace.raceCompleted;
        }
        if (leftRace.completedLapCount != rightRace.completedLapCount) {
            return leftRace.completedLapCount >
                   rightRace.completedLapCount;
        }
        if (leftRace.totalCheckpointEventCount !=
            rightRace.totalCheckpointEventCount) {
            return leftRace.totalCheckpointEventCount >
                   rightRace.totalCheckpointEventCount;
        }
        if (leftRace.raceCompleted &&
            left.finalState.finishTime.present &&
            right.finalState.finishTime.present &&
            left.finalState.finishTime.value.estimatedNs !=
                    right.finalState.finishTime.value.estimatedNs) {
            return left.finalState.finishTime.value.estimatedNs <
                   right.finalState.finishTime.value.estimatedNs;
        }
        if (leftRace.raceCompleted &&
            leftRace.lastPrepareTimeMs != rightRace.lastPrepareTimeMs) {
            return leftRace.lastPrepareTimeMs <
                   rightRace.lastPrepareTimeMs;
        }
        const std::optional<float> leftDistance = finalDistance(left);
        const std::optional<float> rightDistance = finalDistance(right);
        if (leftDistance.has_value() != rightDistance.has_value()) {
            return leftDistance.has_value();
        }
        if (leftDistance.has_value() &&
            *leftDistance != *rightDistance) {
            return *leftDistance < *rightDistance;
        }
        if (left.executedRespawnCount != right.executedRespawnCount) {
            return left.executedRespawnCount <
                   right.executedRespawnCount;
        }
        return left.finalState.candidateId <
               right.finalState.candidateId;
    };

    std::optional<std::size_t> winner;
    for (std::size_t index = 0u; index < candidates.size(); ++index) {
        if (candidates[index].status != HipTimelineStatus::Success) {
            continue;
        }
        if (!winner.has_value() ||
            isBetter(candidates[index], candidates[*winner])) {
            winner = index;
        }
    }
    return winner;
}

#if !FOREVERVALIDATOR_HAS_HIP
HipTimelineBatchResult ExecuteHipTimelineBatch(
        const void *,
        const void *,
        const std::vector<HipCandidateTimelineInput> &,
        bool) noexcept {
    HipTimelineBatchResult result;
    result.status = HipTimelineStatus::DeviceFailure;
    result.diagnostic =
            "HIP timeline execution unavailable in a CPU-only build";
    return result;
}
#endif

}  // namespace forevervalidator::simulation
