#include <cstdint>
#include <functional>
#include <limits>

#include "validation/evaluation/replay_validation_session.h"
#include "simulation/control/replay_control_plan.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "validation/evaluation/replay_recorded_input_outcome.h"
#include "validation/evaluation/replay_validation_evaluation.h"
namespace {

ReplayValidationExecutionResult FromControlPlanResult(
        ReplayControlPlanBuildResult result) {
    switch (result) {
    case ReplayControlPlanBuildResult::Success:
        return ReplayValidationExecutionResult::Success;
    case ReplayControlPlanBuildResult::InvalidRequest:
        return ReplayValidationExecutionResult::ControlPlanInvalidRequest;
    case ReplayControlPlanBuildResult::TargetAllocationFailed:
        return ReplayValidationExecutionResult::ControlTargetAllocationFailed;
    case ReplayControlPlanBuildResult::TargetTimeOutOfRange:
        return ReplayValidationExecutionResult::ControlTargetTimeOutOfRange;
    case ReplayControlPlanBuildResult::NonFiniteTarget:
        return ReplayValidationExecutionResult::ControlTargetNonFinite;
    case ReplayControlPlanBuildResult::TickReservationFailed:
        return ReplayValidationExecutionResult::ControlTickReservationFailed;
    case ReplayControlPlanBuildResult::TickAllocationFailed:
        return ReplayValidationExecutionResult::ControlTickAllocationFailed;
    case ReplayControlPlanBuildResult::MissingComparisonTarget:
        return ReplayValidationExecutionResult::ControlTargetMissing;
    case ReplayControlPlanBuildResult::MissingOutput:
        return ReplayValidationExecutionResult::ControlOutputMissing;
    }
    return ReplayValidationExecutionResult::ControlPlanInvalidRequest;
}

ReplayValidationExecutionResult FromSimulationResult(
        ReplaySimulationRunResult result) {
    switch (result) {
    case ReplaySimulationRunResult::Success:
        return ReplayValidationExecutionResult::Success;
    case ReplaySimulationRunResult::InvalidControlTimeline:
        return ReplayValidationExecutionResult::PhysicsInputInvalid;
    case ReplaySimulationRunResult::MapStartUnavailable:
        return ReplayValidationExecutionResult::MapStartUnavailable;
    case ReplaySimulationRunResult::ObservationAllocationFailed:
        return ReplayValidationExecutionResult::ObservationAllocationFailed;
    case ReplaySimulationRunResult::DeterministicExecutionUnavailable:
        return ReplayValidationExecutionResult::
                DeterministicExecutionUnavailable;
    case ReplaySimulationRunResult::CudaUnavailable:
        return ReplayValidationExecutionResult::CudaUnavailable;
    case ReplaySimulationRunResult::CudaInitializationFailed:
        return ReplayValidationExecutionResult::CudaInitializationFailed;
    case ReplaySimulationRunResult::CudaExecutionFailed:
        return ReplayValidationExecutionResult::CudaExecutionFailed;
    case ReplaySimulationRunResult::TwinUnavailable:
        return ReplayValidationExecutionResult::TwinUnavailable;
    case ReplaySimulationRunResult::TwinNotImplemented:
        return ReplayValidationExecutionResult::TwinNotImplemented;
    case ReplaySimulationRunResult::TwinExecutionFailed:
        return ReplayValidationExecutionResult::TwinExecutionFailed;
    case ReplaySimulationRunResult::StaticSceneConstructionFailed:
    case ReplaySimulationRunResult::StaticSceneSourcesMissing:
    case ReplaySimulationRunResult::StaticSceneCorpusCountMismatch:
    case ReplaySimulationRunResult::StaticSceneInstallFailed:
    case ReplaySimulationRunResult::VehicleCollisionModelFailed:
        return ReplayValidationExecutionResult::PhysicsInputInvalid;
    }
    return ReplayValidationExecutionResult::PhysicsInputInvalid;
}

}  // namespace

ReplayValidationExecutionResult ExecuteReplayValidation(
        ReplayValidationExecutionOutput *out,
        ReplaySimulationSession &simulationSession,
        const ReplayValidationPlan &plan,
        const ReplaySimulationDefinition &simulationDefinition,
        const ReplayGhostTrajectory &trajectory,
        const ReplayInputTimeline &inputTimeline) {
    if (out == nullptr) {
        return ReplayValidationExecutionResult::MissingInput;
    }
    if (plan.sampleCount != 0u && trajectory.Empty()) {
        return ReplayValidationExecutionResult::MissingInput;
    }
    if (plan.controlTickMs == 0u ||
        plan.expectedSamples >
                std::numeric_limits<std::uint32_t>::max()) {
        return ReplayValidationExecutionResult::InvalidPlan;
    }
    if (plan.sampleCount == 0u) {
        if (plan.samplePeriodMs != 0u ||
            plan.requestedSamples != 0u ||
            plan.expectedSamples != 0u ||
            !trajectory.Empty()) {
            return ReplayValidationExecutionResult::InvalidPlan;
        }
    } else if (trajectory.SampleCount() != plan.sampleCount ||
               trajectory.FixedStepMs() != plan.samplePeriodMs ||
               plan.requestedSamples == 0u ||
               plan.requestedSamples > plan.sampleCount) {
        return ReplayValidationExecutionResult::InvalidPlan;
    }

    ReplayControlPlanRequest controlRequest(inputTimeline);
    controlRequest.controlTickMs = plan.controlTickMs;
    controlRequest.requestedSamples = plan.requestedSamples;
    controlRequest.validationDurationMs = plan.validationDurationMs;
    controlRequest.validationPrestartMs = plan.validationPrestartMs;
    controlRequest.inputTimeBaseMs = plan.inputTimeBaseMs;
    controlRequest.baseActions = plan.baseActions;
    controlRequest.enableRaceSimulationAfterMs =
            plan.enableRaceSimulationAfterMs;
    controlRequest.establishRaceSpawnAtMs = plan.establishRaceSpawnAtMs;
    controlRequest.appendUnobservedTrailingTick =
            plan.validationMode == ReplayValidationMode::Race;
    if (plan.sampleCount != 0u) {
        controlRequest.trajectory = std::cref(trajectory);
    }

    ReplayControlPlan controlPlan;
    const ReplayControlPlanBuildResult controlResult =
            BuildReplayControlPlan(controlRequest, &controlPlan);
    if (controlResult != ReplayControlPlanBuildResult::Success) {
        return FromControlPlanResult(controlResult);
    }
    const std::uint32_t expectedTargetCount =
            plan.requestedSamples != 0u ? plan.requestedSamples : 1u;
    if (controlPlan.ticks.empty() ||
        controlPlan.comparisonTargetCount != expectedTargetCount ||
        controlPlan.ticks.size() >
                std::numeric_limits<std::uint32_t>::max()) {
        return ReplayValidationExecutionResult::InvalidControlPlan;
    }

    simulationSession.ConfigureReplayRace(
            plan.playMode, plan.isLapRace, plan.lapCount);
    ReplaySimulationTimelineResult simulationResult =
            simulationSession.SimulateTimeline(
                    simulationDefinition,
                    controlPlan.ticks,
                    plan.validationSeed);
    if (simulationResult.result != ReplaySimulationRunResult::Success) {
        return FromSimulationResult(simulationResult.result);
    }
    // Platform, Puzzle, and Stunts carry their recorded outcome in replay
    // events. Race completion remains authoritative only when the standalone
    // checkpoint simulation reaches a valid finish.
    const ReplayRecordedInputOutcome recordedOutcome =
            ObserveReplayRecordedInputOutcome(
                    inputTimeline, plan.inputTimeBaseMs);
    if (plan.validationMode == ReplayValidationMode::Stunts &&
        recordedOutcome.finishRaceTimeMs.has_value() &&
        static_cast<std::uint32_t>(*recordedOutcome.finishRaceTimeMs) >
                plan.baseActions.stuntsTimeLimitMs) {
        const std::uint32_t overtimeMs =
                static_cast<std::uint32_t>(
                        *recordedOutcome.finishRaceTimeMs) -
                plan.baseActions.stuntsTimeLimitMs;
        simulationResult.stuntsScore =
                simulationSession.ApplyReplayStuntTimePenalty(overtimeMs);
        if (!simulationResult.stuntsScore.has_value()) {
            return ReplayValidationExecutionResult::InvalidPlan;
        }
    }
    if (!ApplyReplayRecordedInputOutcome(
                plan, recordedOutcome, &simulationResult)) {
        return ReplayValidationExecutionResult::InvalidPlan;
    }

    *out = EvaluateReplayValidation(
            plan,
            simulationResult.observations,
            simulationResult);
    return ReplayValidationExecutionResult::Success;
}
