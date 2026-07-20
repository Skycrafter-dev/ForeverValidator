#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "validation/evaluation/replay_result_comparator.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "validation/evaluation/replay_validation_evaluation.h"
#include "validation/evaluation/replay_validation_session.h"
namespace {

using u32 = std::uint32_t;
using s32 = std::int32_t;

constexpr float ReplayValidationThreshold = 0.1f;
constexpr u32 ReplayRespawnCountLimit = 999u;

u32 SaturateReplayRespawnCount(u32 respawnCount) {
    return respawnCount > ReplayRespawnCountLimit
            ? ReplayRespawnCountLimit
            : respawnCount;
}

ReplayValidationOutcome OutcomeFromComparison(
        ReplayValidationComparisonResult result) {
    switch (result) {
    case ReplayValidationComparisonResult::Invalid:
        return ReplayValidationOutcome::Invalid;
    case ReplayValidationComparisonResult::Valid:
        return ReplayValidationOutcome::Valid;
    case ReplayValidationComparisonResult::WrongSimulation:
        return ReplayValidationOutcome::WrongSimulation;
    }
    return ReplayValidationOutcome::Error;
}

struct ReplayEvaluationMetadata {
    std::optional<bool> raceCompleted;
    std::optional<s32> raceTimeMs;
    std::optional<s32> stuntsScore;
    std::optional<s32> respawns;
};

ReplayValidationStatus StatusFromComparison(
        const ReplayValidationComparison &comparison,
        const ReplayEvaluationMetadata &expected,
        const ReplayEvaluationMetadata &actual) {
    if (comparison.result == ReplayValidationComparisonResult::Valid) {
        return ReplayValidationStatus::Valid;
    }
    if (comparison.result == ReplayValidationComparisonResult::WrongSimulation) {
        return ReplayValidationStatus::WrongSimulation;
    }
    if (expected.raceTimeMs.has_value() &&
        (!actual.raceTimeMs.has_value() ||
         std::abs(*expected.raceTimeMs - *actual.raceTimeMs) > 10)) {
        return ReplayValidationStatus::RaceTimeMismatch;
    }
    if (expected.stuntsScore.has_value() &&
        expected.stuntsScore != actual.stuntsScore) {
        return ReplayValidationStatus::StuntsScoreMismatch;
    }
    if (expected.respawns.has_value() && expected.respawns != actual.respawns) {
        return ReplayValidationStatus::RespawnCountMismatch;
    }
    return ReplayValidationStatus::ExpectingCompletedRace;
}

ReplayValidationDeviation MakeDeviation(
        u32 comparisonOrdinal,
        u32 samplePeriodMs,
        u32 validationPrestartMs,
        float distance,
        const ReplayTrajectoryObservation &observation) {
    ReplayValidationDeviation deviation;
    deviation.comparisonOrdinal = comparisonOrdinal;
    deviation.ghostTimeMs = static_cast<s32>(
            comparisonOrdinal * samplePeriodMs);
    deviation.simulationTimeMs =
            static_cast<s32>(validationPrestartMs) +
            deviation.ghostTimeMs;
    deviation.distance = distance;
    deviation.simulatedPosition = observation.simulatedPosition;
    deviation.writePosition = observation.writePosition;
    if (observation.comparison.has_value()) {
        deviation.targetPosition =
                observation.comparison->targetPosition;
    }
    return deviation;
}

}  // namespace

std::optional<s32> EvaluateTrajectory(
        const ReplayValidationPlan &plan,
        const std::vector<ReplayTrajectoryObservation> &observations,
        ReplayValidationResult &output) {
    std::optional<s32> simulatedFinishRaceTimeMs;
    for (const ReplayTrajectoryObservation &observation : observations) {
        if (!simulatedFinishRaceTimeMs.has_value() &&
            observation.finishTickMs.has_value() &&
            *observation.finishTickMs >= plan.validationPrestartMs) {
            simulatedFinishRaceTimeMs = static_cast<s32>(
                    *observation.finishTickMs - plan.validationPrestartMs);
        }
        if (!observation.comparison.has_value()) {
            continue;
        }

        const u32 comparisonOrdinal = output.measuredSamples++;
        const ReplayTrajectoryDeviation &comparison = *observation.comparison;
        if (!std::isfinite(comparison.distance)) {
            output.status = ReplayValidationStatus::ObservationError;
            output.outcome = ReplayValidationOutcome::Error;
            output.observationError = ReplayObservationError::NonFiniteDistance;
            output.firstDivergence = MakeDeviation(
                    comparisonOrdinal, plan.samplePeriodMs,
                    plan.validationPrestartMs, 0.0f, observation);
            break;
        }

        ++output.comparedExactGhostStateCount;
        if (!output.firstExactDeviation.has_value() && comparison.distance > 0.0f) {
            output.firstExactDeviation = MakeDeviation(
                    comparisonOrdinal, plan.samplePeriodMs,
                    plan.validationPrestartMs, comparison.distance, observation);
        }
        if (output.maxDeviation < comparison.distance) {
            output.maxDeviation = comparison.distance;
            output.maxDeviationTimeMs = static_cast<s32>(
                    comparisonOrdinal * plan.samplePeriodMs);
            output.maxDeviationDistance = comparison.distance;
        }
        if (comparison.distance >= ReplayValidationThreshold) {
            output.status = ReplayValidationStatus::WrongSimulation;
            output.outcome = ReplayValidationOutcome::WrongSimulation;
            output.wrongSimulation = true;
            output.firstDivergence = MakeDeviation(
                    comparisonOrdinal, plan.samplePeriodMs,
                    plan.validationPrestartMs, comparison.distance, observation);
            break;
        }
    }
    return simulatedFinishRaceTimeMs;
}

ReplayEvaluationMetadata ResolveActualMetadata(
        const ReplayValidationPlan &plan,
        const ReplaySimulationTimelineResult &simulationResult,
        const std::optional<s32> &simulatedFinishRaceTimeMs) {
    ReplayEvaluationMetadata actual{
        simulationResult.raceCompleted,
        std::nullopt,
        std::nullopt,
        static_cast<s32>(SaturateReplayRespawnCount(
                simulationResult.executedRespawnCount)),
    };
    if (simulationResult.stuntsScore.has_value() &&
        *simulationResult.stuntsScore <=
                static_cast<u32>(std::numeric_limits<s32>::max())) {
        actual.stuntsScore = static_cast<s32>(
                *simulationResult.stuntsScore);
    }
    if (simulationResult.finishTimeMs.has_value() &&
        *simulationResult.finishTimeMs >= plan.validationPrestartMs) {
        actual.raceCompleted = true;
        actual.raceTimeMs = static_cast<s32>(
                *simulationResult.finishTimeMs - plan.validationPrestartMs);
    } else if (plan.validationMode != ReplayValidationMode::Platform &&
               plan.validationMode != ReplayValidationMode::Puzzle &&
               plan.validationMode != ReplayValidationMode::Stunts &&
               simulatedFinishRaceTimeMs.has_value()) {
        actual.raceCompleted = true;
        actual.raceTimeMs = simulatedFinishRaceTimeMs;
    }
    return actual;
}

void FinalizeValidation(
        const ReplayValidationPlan &plan,
        const ReplayEvaluationMetadata &actual,
        ReplayValidationResult &output) {
    if (output.status != ReplayValidationStatus::Valid) {
        return;
    }
    if (output.measuredSamples < plan.expectedSamples) {
        output.status = ReplayValidationStatus::IncompleteStandaloneRun;
        output.outcome = ReplayValidationOutcome::Unavailable;
        return;
    }
    if ((plan.validationMode == ReplayValidationMode::Platform ||
         plan.validationMode == ReplayValidationMode::Puzzle) &&
        (!plan.expectedRaceTimeMs.has_value() ||
         *plan.expectedRaceTimeMs < 0)) {
        output.status = ReplayValidationStatus::ObservationError;
        output.outcome = ReplayValidationOutcome::Error;
        output.observationError =
                ReplayObservationError::ReplayMetadataUnavailable;
        return;
    }
    if (plan.validationMode == ReplayValidationMode::Stunts &&
        (!plan.expectedStuntsScore.has_value() ||
         *plan.expectedStuntsScore < 0)) {
        output.status = ReplayValidationStatus::ObservationError;
        output.outcome = ReplayValidationOutcome::Error;
        output.observationError =
                ReplayObservationError::ReplayMetadataUnavailable;
        return;
    }
    if (plan.validationMode == ReplayValidationMode::Stunts &&
        !plan.expectedRespawns.has_value()) {
        output.status = ReplayValidationStatus::RespawnExpectationUnavailable;
        output.outcome = ReplayValidationOutcome::Unavailable;
        return;
    }
    if ((plan.validationMode == ReplayValidationMode::Platform ||
         plan.validationMode == ReplayValidationMode::Puzzle) &&
        !plan.expectedRespawns.has_value()) {
        output.status = ReplayValidationStatus::RespawnExpectationUnavailable;
        output.outcome = ReplayValidationOutcome::Unavailable;
        return;
    }
    if (!actual.raceCompleted.has_value()) {
        output.status = ReplayValidationStatus::RaceCompletionUnavailable;
        output.outcome = ReplayValidationOutcome::Unavailable;
        return;
    }
    if ((plan.validationMode == ReplayValidationMode::Platform ||
         plan.validationMode == ReplayValidationMode::Puzzle) &&
        !*actual.raceCompleted) {
        output.status = ReplayValidationStatus::ExpectingCompletedRace;
        output.outcome = ReplayValidationOutcome::Invalid;
        return;
    }
    ReplayValidationExpectations expectedComparison{
        plan.expectedRaceTimeMs,
        plan.expectedStuntsScore,
        plan.expectedRespawns,
    };
    const s32 respawnCount = actual.respawns.value_or(-1);
    ReplayValidationObservation observation{
        output.maxDeviation,
        respawnCount < 0 ? 0u : static_cast<u32>(respawnCount),
        output.maxDeviationTimeMs,
        output.maxDeviationDistance,
        actual.raceTimeMs,
        actual.stuntsScore,
        *actual.raceCompleted,
    };
    const ReplayValidationComparison comparison =
            ReplayValidationCompareResult(expectedComparison, observation);
    const ReplayEvaluationMetadata expected{
        std::nullopt,
        plan.expectedRaceTimeMs,
        plan.expectedStuntsScore,
        plan.expectedRespawns,
    };
    output.outcome = OutcomeFromComparison(comparison.result);
    output.status = StatusFromComparison(comparison, expected, actual);
    output.wrongSimulation =
            comparison.result == ReplayValidationComparisonResult::WrongSimulation;
}

ReplayValidationExecutionOutput EvaluateReplayValidation(
        const ReplayValidationPlan &plan,
        const std::vector<ReplayTrajectoryObservation> &observations,
        const ReplaySimulationTimelineResult &simulationResult) {
    ReplayValidationExecutionOutput output;
    output.validation.expectedSamples = static_cast<u32>(plan.expectedSamples);
    const auto simulatedFinish = EvaluateTrajectory(
            plan, observations, output.validation);
    const ReplayEvaluationMetadata actual = ResolveActualMetadata(
            plan, simulationResult, simulatedFinish);
    FinalizeValidation(plan, actual, output.validation);
    output.raceOutcome.raceCompleted = actual.raceCompleted;
    output.raceOutcome.raceTimeMs = actual.raceTimeMs;
    output.raceOutcome.stuntsScore = actual.stuntsScore;
    output.raceOutcome.respawnCount = SaturateReplayRespawnCount(
            simulationResult.executedRespawnCount);
    return output;
}
