#include <forevervalidator/validation.h>
#include <forevervalidator/experimental/physics_sandbox.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "format/assets/replay_asset_repository.h"
#include "format/pack/default_vehicle_pack_archive.h"
#include "format/pack/installed/installed_pack_key_catalog.h"
#include "format/pack/installed/plug_file_pack.h"
#include "format/pack/installed_vehicle_asset_graph.h"
#include "format/pack/replay_vehicle_source_bundle.h"
#include "format/replay/replay_file.h"
#include "format/static_solid/default_vehicle_solid_archive.h"
#include "simulation/runtime/replay_deterministic_execution.h"
#include "simulation/runtime/replay_simulation_definition.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "simulation/backends/simulation_backend.h"
#include "simulation/control/replay_control_plan.h"
#include "validation/evaluation/replay_validation_session.h"
#include "validation/api/physics_sandbox_static_scene_test_access.h"
#include "validation/api/physics_sandbox_cuda_test_access.h"
#include "validation/planning/replay_asset_route.h"
#include "validation/planning/replay_challenge_map_preload.h"

namespace forevervalidator {

namespace {

struct CachedInstalledPack {
    std::string packName;
    AssetBytes bytes;
};

struct CachedPackAssets {
    std::string packName;
    std::unique_ptr<ReplayAssetRepository> repository;
};

struct CachedVehicleAssets {
    std::string packName;
    ::ReplayVehicleModel vehicleModel = ::ReplayVehicleModel::Unknown;
    InstalledVehicleAssetGraph assetGraph;
    ReplayVehicleSourceBundle vehicleSources;
};

struct PreparedAssets {
    ReplayAssetRepository *mapAssets = nullptr;
    ReplayAssetRepository *decorationAssets = nullptr;
    const ReplayVehicleSourceBundle *vehicleSources = nullptr;
};

struct ValidationState {
    explicit ValidationState(AssetProvider value)
        : provider(std::move(value)) {}

    AssetProvider provider;
    AssetBytes packlistBytes;
    std::unique_ptr<InstalledPackKeyCatalog> packKeys;
    std::vector<std::unique_ptr<CachedInstalledPack>> installedPacks;
    std::vector<std::unique_ptr<CachedPackAssets>> assetRepositories;
    std::vector<std::unique_ptr<CachedVehicleAssets>> vehicleAssets;
};

}  // namespace

struct AssetSource::Impl {
    explicit Impl(AssetProvider value) : provider(std::move(value)) {}
    AssetProvider provider;
};

namespace detail {

struct PhysicsSandboxAssetSourceAccess {
    static AssetProvider Take(AssetSource &source) {
        if (source.impl_ == nullptr) {
            return {};
        }
        AssetProvider provider = std::move(source.impl_->provider);
        source.impl_.reset();
        return provider;
    }
};

}  // namespace detail

struct ValidationContext::Impl {
    explicit Impl(AssetProvider value) : state(std::move(value)) {}
    ValidationState state;
};

namespace {

ValidationError MakeError(
        ValidationErrorCategory category,
        ValidationErrorCode code,
        ValidationStage stage,
        ValidationFailureReason reason,
        const ReplayIdentity &identity,
        const char *diagnostic) {
    ValidationError error;
    error.category = category;
    error.code = code;
    error.stage = stage;
    error.reason = reason;
    error.replay = identity;
    error.diagnostic = diagnostic == nullptr ? "" : diagnostic;
    return error;
}

ValidationError AllocationError(
        ValidationStage stage,
        const ReplayIdentity &identity,
        const char *diagnostic) {
    return MakeError(
            ValidationErrorCategory::Allocation,
            ValidationErrorCode::AllocationFailed,
            stage,
            ValidationFailureReason::AllocationFailed,
            identity,
            diagnostic);
}

bool IsCudaSupportedRoute(const ReplayAssetRoute &route) noexcept {
    const bool stadium =
            route.mapEnvironment == ReplayMapEnvironment::Stadium &&
            route.vehicleModel == ReplayVehicleModel::StadiumCar;
    const bool speed =
            route.mapEnvironment == ReplayMapEnvironment::Speed &&
            route.vehicleModel == ReplayVehicleModel::DesertCar;
    return route.mapEnvironment == route.decorationEnvironment &&
           (stadium || speed);
}

ValidationError CudaScopeError(
        const ReplayAssetRoute &route,
        const ReplayIdentity &identity) {
    ValidationError error = MakeError(
            ValidationErrorCategory::Simulation,
            ValidationErrorCode::CudaUnavailable,
            ValidationStage::SimulationStartup,
            ValidationFailureReason::CudaUnsupportedSimulationScope,
            identity,
            "CUDA supports only certified map and vehicle combinations");
    error.relatedAsset =
            std::string(ReplayMapEnvironmentName(route.mapEnvironment)) +
            "/" + ReplayVehicleModelName(route.vehicleModel);
    return error;
}

ValidationStatus ToPublicStatus(ReplayValidationStatus status) {
    switch (status) {
    case ReplayValidationStatus::Valid: return ValidationStatus::Valid;
    case ReplayValidationStatus::ValidPrefix:
        return ValidationStatus::ValidPrefix;
    case ReplayValidationStatus::WrongSimulation:
        return ValidationStatus::WrongSimulation;
    case ReplayValidationStatus::IncompleteStandaloneRun:
        return ValidationStatus::IncompleteStandaloneRun;
    case ReplayValidationStatus::RaceCompletionUnavailable:
        return ValidationStatus::RaceCompletionUnavailable;
    case ReplayValidationStatus::ExpectingCompletedRace:
        return ValidationStatus::ExpectingCompletedRace;
    case ReplayValidationStatus::RaceTimeMismatch:
        return ValidationStatus::RaceTimeMismatch;
    case ReplayValidationStatus::StuntsScoreMismatch:
        return ValidationStatus::StuntsScoreMismatch;
    case ReplayValidationStatus::RespawnCountMismatch:
        return ValidationStatus::RespawnCountMismatch;
    case ReplayValidationStatus::RespawnExpectationUnavailable:
        return ValidationStatus::RespawnExpectationUnavailable;
    case ReplayValidationStatus::ObservationError:
        return ValidationStatus::ObservationError;
    case ReplayValidationStatus::IncompatibleReplayVersion:
        return ValidationStatus::IncompatibleReplayVersion;
    case ReplayValidationStatus::InputUnavailable:
        return ValidationStatus::InputUnavailable;
    }
    return ValidationStatus::ObservationError;
}

ValidationOutcome ToPublicOutcome(ReplayValidationOutcome outcome) {
    switch (outcome) {
    case ReplayValidationOutcome::Invalid: return ValidationOutcome::Invalid;
    case ReplayValidationOutcome::Valid: return ValidationOutcome::Valid;
    case ReplayValidationOutcome::WrongSimulation:
        return ValidationOutcome::WrongSimulation;
    case ReplayValidationOutcome::Unavailable:
        return ValidationOutcome::Unavailable;
    case ReplayValidationOutcome::Error: return ValidationOutcome::Error;
    }
    return ValidationOutcome::Error;
}

Vector3 ToPublicVector(const GmVec3 &value) {
    return {value.x, value.y, value.z};
}

ValidationDeviation ToPublicDeviation(
        const ReplayValidationDeviation &value) {
    ValidationDeviation result;
    result.comparisonOrdinal = value.comparisonOrdinal;
    result.ghostTimeMs = value.ghostTimeMs;
    result.simulationTimeMs = value.simulationTimeMs;
    result.distance = value.distance;
    result.simulatedPosition = ToPublicVector(value.simulatedPosition);
    result.writePosition = ToPublicVector(value.writePosition);
    result.targetPosition = ToPublicVector(value.targetPosition);
    return result;
}

std::optional<ObservationError> ToPublicObservationError(
        const std::optional<ReplayObservationError> &error) {
    if (!error.has_value()) {
        return std::nullopt;
    }
    switch (*error) {
    case ReplayObservationError::NonFiniteDistance:
        return ObservationError::NonFiniteDistance;
    case ReplayObservationError::ReplayMetadataUnavailable:
        return ObservationError::ReplayMetadataUnavailable;
    }
    return ObservationError::ReplayMetadataUnavailable;
}

MapEnvironment ToPublicMapEnvironment(
        ::ReplayMapEnvironment environment) noexcept {
    switch (environment) {
    case ::ReplayMapEnvironment::Alpine: return MapEnvironment::Alpine;
    case ::ReplayMapEnvironment::Speed: return MapEnvironment::Speed;
    case ::ReplayMapEnvironment::Rally: return MapEnvironment::Rally;
    case ::ReplayMapEnvironment::Island: return MapEnvironment::Island;
    case ::ReplayMapEnvironment::Coast: return MapEnvironment::Coast;
    case ::ReplayMapEnvironment::Bay: return MapEnvironment::Bay;
    case ::ReplayMapEnvironment::Stadium: return MapEnvironment::Stadium;
    case ::ReplayMapEnvironment::Unknown: return MapEnvironment::Unknown;
    }
    return MapEnvironment::Unknown;
}

VehicleModel ToPublicVehicleModel(::ReplayVehicleModel vehicle) noexcept {
    switch (vehicle) {
    case ::ReplayVehicleModel::SnowCar: return VehicleModel::SnowCar;
    case ::ReplayVehicleModel::DesertCar: return VehicleModel::DesertCar;
    case ::ReplayVehicleModel::RallyCar: return VehicleModel::RallyCar;
    case ::ReplayVehicleModel::IslandCar: return VehicleModel::IslandCar;
    case ::ReplayVehicleModel::CoastCar: return VehicleModel::CoastCar;
    case ::ReplayVehicleModel::BayCar: return VehicleModel::BayCar;
    case ::ReplayVehicleModel::StadiumCar: return VehicleModel::StadiumCar;
    case ::ReplayVehicleModel::Unknown: return VehicleModel::Unknown;
    }
    return VehicleModel::Unknown;
}

PlayMode ToPublicPlayMode(EChallengePlayMode mode) noexcept {
    switch (mode) {
    case EChallengePlayMode::Race: return PlayMode::Race;
    case EChallengePlayMode::Platform: return PlayMode::Platform;
    case EChallengePlayMode::Puzzle: return PlayMode::Puzzle;
    case EChallengePlayMode::Crazy: return PlayMode::Crazy;
    case EChallengePlayMode::Shortcut: return PlayMode::Shortcut;
    case EChallengePlayMode::Stunts: return PlayMode::Stunts;
    }
    return PlayMode::Race;
}

ReplayProvenance ToPublicReplayProvenance(
        ReplayInputProvenance provenance) noexcept {
    switch (provenance) {
    case ReplayInputProvenance::Unmarked:
        return ReplayProvenance::Unmarked;
    case ReplayInputProvenance::Scripted:
        return ReplayProvenance::Scripted;
    }
    return ReplayProvenance::Unmarked;
}

ValidationReport ToPublicReport(
        const ReplayIdentity &identity,
        const ReplayFileValidationResult &source,
        const ReplayFile &replay,
        const ReplayAssetRoute &route) {
    ValidationReport report;
    report.replay = identity;
    report.valid = source.validation.status == ReplayValidationStatus::Valid;
    report.status = ToPublicStatus(source.validation.status);
    report.outcome = ToPublicOutcome(source.validation.outcome);
    report.measuredSamples = source.validation.measuredSamples;
    report.expectedSamples = source.validation.expectedSamples;
    report.comparedExactGhostStateCount =
            source.validation.comparedExactGhostStateCount;
    report.wrongSimulation = source.validation.wrongSimulation;
    if (source.validation.firstDivergence.has_value()) {
        report.firstDivergence =
                ToPublicDeviation(*source.validation.firstDivergence);
    }
    if (source.validation.firstExactDeviation.has_value()) {
        report.firstExactDeviation =
                ToPublicDeviation(*source.validation.firstExactDeviation);
    }
    report.maxDeviation = source.validation.maxDeviation;
    report.maxDeviationTimeMs = source.validation.maxDeviationTimeMs;
    report.maxDeviationDistance = source.validation.maxDeviationDistance;
    report.observationError =
            ToPublicObservationError(source.validation.observationError);
    report.metadata.replayProvenance = ToPublicReplayProvenance(
            replay.InputTimeline().Provenance());
    if (report.metadata.replayProvenance == ReplayProvenance::Scripted) {
        if (source.validation.status ==
            ReplayValidationStatus::WrongSimulation) {
            report.inputGhostMatch = InputGhostMatch::Mismatch;
        } else if (source.validation.expectedSamples > 0u &&
                   source.validation.measuredSamples ==
                           source.validation.expectedSamples) {
            report.inputGhostMatch = InputGhostMatch::Match;
        }
        if (source.validation.status == ReplayValidationStatus::Valid ||
            source.validation.status == ReplayValidationStatus::ValidPrefix) {
            report.valid = false;
            report.status = ValidationStatus::ScriptedReplay;
            report.outcome = ValidationOutcome::Invalid;
            report.wrongSimulation = false;
        }
    }
    report.metadata.mapEnvironment =
            ToPublicMapEnvironment(route.mapEnvironment);
    report.metadata.vehicleModel = ToPublicVehicleModel(route.vehicleModel);
    report.metadata.playMode = ToPublicPlayMode(route.playMode);
    if (route.validationMode == ReplayValidationMode::Stunts) {
        report.metadata.expectedStuntsScore =
                replay.InputTimeline().Metadata().stuntScore;
    }
    report.metadata.sampleCount = source.metadata.sampleCount;
    report.metadata.samplePeriodMs = source.metadata.samplePeriodMs;
    report.metadata.encodedGhostSampleByteCount =
            source.metadata.encodedGhostSampleByteCount;
    report.metadata.encodedGhostStateByteCount =
            source.metadata.encodedGhostStateByteCount;
    report.metadata.inputDurationMs = source.metadata.inputDurationMs;
    report.metadata.expectedRaceTimeMs = source.metadata.expectedRaceTimeMs;
    report.metadata.expectedRespawns = source.metadata.expectedRespawns;
    report.metadata.requestedSamples = source.metadata.requestedSamples;
    report.metadata.expectedSamples = source.metadata.expectedSamples;
    report.metadata.actionCount = source.metadata.actionCount;
    report.metadata.eventCount = source.metadata.eventCount;
    report.simulation.raceCompleted = source.raceOutcome.raceCompleted;
    report.simulation.raceTimeMs = source.raceOutcome.raceTimeMs;
    report.simulation.raceTime = source.raceOutcome.raceTime;
    report.simulation.stuntsScore = source.raceOutcome.stuntsScore;
    report.simulation.respawnCount = source.raceOutcome.respawnCount;
    return report;
}

ValidationError ReplayRouteError(
        ReplayAssetRouteResult routeResult,
        const ReplayIdentity &identity,
        const ReplayFile &replay) {
    ValidationFailureReason reason = ValidationFailureReason::UnsupportedPlayMode;
    std::string relatedIdentifier;
    switch (routeResult) {
    case ReplayAssetRouteResult::UnsupportedMapEnvironment:
        reason = ValidationFailureReason::UnsupportedMapEnvironmentIdentifier;
        relatedIdentifier = replay.MapInput().DefaultCollectionName();
        break;
    case ReplayAssetRouteResult::UnsupportedDecorationEnvironment:
        reason = ValidationFailureReason::UnsupportedMapEnvironmentIdentifier;
        relatedIdentifier =
                replay.MapInput().DecorationCollection().Name();
        break;
    case ReplayAssetRouteResult::UnsupportedVehicleIdentifier:
        reason = ValidationFailureReason::UnsupportedVehicleIdentifier;
        relatedIdentifier = replay.VehicleIdentifier().id;
        break;
    case ReplayAssetRouteResult::MissingPlayMode:
    case ReplayAssetRouteResult::UnsupportedPlayMode:
        reason = ValidationFailureReason::UnsupportedPlayMode;
        if (replay.ChallengeMetadata().playMode.has_value()) {
            relatedIdentifier = EChallengePlayModeName(
                    *replay.ChallengeMetadata().playMode);
        }
        break;
    case ReplayAssetRouteResult::Success:
    case ReplayAssetRouteResult::MissingOutput:
        reason = ValidationFailureReason::UnexpectedFailure;
        break;
    }
    ValidationError error = MakeError(
            ValidationErrorCategory::Replay,
            ValidationErrorCode::ReplayDecodingFailed,
            ValidationStage::ReplayDecoding,
            reason,
            identity,
            ReplayAssetRouteResultName(routeResult));
    error.relatedAsset = std::move(relatedIdentifier);
    return error;
}

ValidationFailureReason ReplayReadReason(ReplayFileReadError error) {
    switch (error) {
    case ReplayFileReadError::Success: return ValidationFailureReason::None;
    case ReplayFileReadError::InvalidRequest:
        return ValidationFailureReason::ReplayInvalidRequest;
    case ReplayFileReadError::FileReadFailed:
        return ValidationFailureReason::ReplayFileReadFailed;
    case ReplayFileReadError::InvalidContainer:
        return ValidationFailureReason::ReplayInvalidContainer;
    case ReplayFileReadError::AllocationFailed:
        return ValidationFailureReason::AllocationFailed;
    case ReplayFileReadError::RootBodyDecompressionFailed:
        return ValidationFailureReason::ReplayRootBodyDecompressionFailed;
    case ReplayFileReadError::MissingGhostBuffer:
        return ValidationFailureReason::ReplayMissingGhostBuffer;
    case ReplayFileReadError::MissingInputStream:
        return ValidationFailureReason::ReplayMissingInputStream;
    case ReplayFileReadError::TooManyInputActions:
        return ValidationFailureReason::ReplayTooManyInputActions;
    case ReplayFileReadError::InvalidGhostMetadata:
        return ValidationFailureReason::ReplayInvalidGhostMetadata;
    case ReplayFileReadError::InvalidInputHeader:
        return ValidationFailureReason::ReplayInvalidInputHeader;
    case ReplayFileReadError::InvalidInputActions:
        return ValidationFailureReason::ReplayInvalidInputActions;
    case ReplayFileReadError::InvalidInputEventHeader:
        return ValidationFailureReason::ReplayInvalidInputEventHeader;
    case ReplayFileReadError::InvalidInputEvents:
        return ValidationFailureReason::ReplayInvalidInputEvents;
    case ReplayFileReadError::InvalidInputTimeline:
        return ValidationFailureReason::ReplayInvalidInputTimeline;
    case ReplayFileReadError::MissingGhostState:
        return ValidationFailureReason::ReplayMissingGhostState;
    case ReplayFileReadError::GhostStateDecompressionFailed:
        return ValidationFailureReason::ReplayGhostStateDecompressionFailed;
    case ReplayFileReadError::InvalidGhostState:
        return ValidationFailureReason::ReplayInvalidGhostState;
    case ReplayFileReadError::GhostTrajectoryAllocationFailed:
        return ValidationFailureReason::ReplayGhostTrajectoryAllocationFailed;
    case ReplayFileReadError::MissingEmbeddedChallenge:
        return ValidationFailureReason::ReplayMissingEmbeddedChallenge;
    case ReplayFileReadError::InvalidEmbeddedChallenge:
        return ValidationFailureReason::ReplayInvalidEmbeddedChallenge;
    case ReplayFileReadError::InvalidMap:
        return ValidationFailureReason::ReplayInvalidMap;
    }
    return ValidationFailureReason::UnexpectedFailure;
}

ValidationError ReplayDecodeError(
        ReplayFileReadError readError,
        const ReplayIdentity &identity) {
    const bool allocation =
            readError == ReplayFileReadError::AllocationFailed ||
            readError == ReplayFileReadError::GhostTrajectoryAllocationFailed;
    ValidationError error = MakeError(
            allocation ? ValidationErrorCategory::Allocation
                       : ValidationErrorCategory::Replay,
            allocation ? ValidationErrorCode::AllocationFailed
                       : ValidationErrorCode::ReplayDecodingFailed,
            ValidationStage::ReplayDecoding,
            ReplayReadReason(readError),
            identity,
            ReplayFileReadErrorName(readError));
    return error;
}

ValidationFailureReason PreloadReason(ReplayChallengePreloadResult result) {
    switch (result) {
    case ReplayChallengePreloadResult::Success:
        return ValidationFailureReason::None;
    case ReplayChallengePreloadResult::InvalidRequest:
        return ValidationFailureReason::ChallengeInvalidRequest;
    case ReplayChallengePreloadResult::AllocationFailed:
        return ValidationFailureReason::AllocationFailed;
    case ReplayChallengePreloadResult::WaterDefinitionFailed:
        return ValidationFailureReason::ChallengeWaterDefinitionFailed;
    case ReplayChallengePreloadResult::SceneDefinitionFailed:
        return ValidationFailureReason::ChallengeSceneDefinitionFailed;
    case ReplayChallengePreloadResult::ChallengeConstructionFailed:
        return ValidationFailureReason::ChallengeConstructionFailed;
    case ReplayChallengePreloadResult::StaticSceneFailed:
        return ValidationFailureReason::ChallengeStaticSceneFailed;
    }
    return ValidationFailureReason::UnexpectedFailure;
}

ValidationError PreloadError(
        ReplayChallengePreloadResult result,
        const ReplayIdentity &identity) {
    const bool allocation =
            result == ReplayChallengePreloadResult::AllocationFailed;
    return MakeError(
            allocation ? ValidationErrorCategory::Allocation
                       : ValidationErrorCategory::Simulation,
            allocation ? ValidationErrorCode::AllocationFailed
                       : ValidationErrorCode::ChallengePreloadFailed,
            result == ReplayChallengePreloadResult::ChallengeConstructionFailed
                    ? ValidationStage::MapConstruction
                    : ValidationStage::ChallengePreload,
            PreloadReason(result),
            identity,
            "replay challenge preload failed");
}

ValidationFailureReason DefinitionReason(
        ReplaySimulationDefinitionBuildResult result) {
    switch (result) {
    case ReplaySimulationDefinitionBuildResult::Success:
        return ValidationFailureReason::None;
    case ReplaySimulationDefinitionBuildResult::MissingVehicleDefinition:
        return ValidationFailureReason::SimulationMissingVehicleDefinition;
    case ReplaySimulationDefinitionBuildResult::InvalidVehiclePhysics:
        return ValidationFailureReason::SimulationInvalidVehiclePhysics;
    case ReplaySimulationDefinitionBuildResult::InvalidInitialParameters:
        return ValidationFailureReason::SimulationInvalidInitialParameters;
    case ReplaySimulationDefinitionBuildResult::AllocationFailure:
        return ValidationFailureReason::AllocationFailed;
    case ReplaySimulationDefinitionBuildResult::InvalidEnvironment:
        return ValidationFailureReason::SimulationInvalidEnvironment;
    case ReplaySimulationDefinitionBuildResult::InvalidMaterials:
        return ValidationFailureReason::SimulationInvalidMaterials;
    }
    return ValidationFailureReason::UnexpectedFailure;
}

ValidationError DefinitionError(
        ReplaySimulationDefinitionBuildResult result,
        const ReplayIdentity &identity) {
    const bool allocation =
            result == ReplaySimulationDefinitionBuildResult::AllocationFailure;
    return MakeError(
            allocation ? ValidationErrorCategory::Allocation
                       : ValidationErrorCategory::Simulation,
            allocation ? ValidationErrorCode::AllocationFailed
                       : ValidationErrorCode::SimulationDefinitionFailed,
            ValidationStage::SimulationStartup,
            DefinitionReason(result),
            identity,
            "simulation definition build failed");
}

ValidationFailureReason ExecutionReason(
        ReplayValidationExecutionResult result) {
    switch (result) {
    case ReplayValidationExecutionResult::Success:
        return ValidationFailureReason::None;
    case ReplayValidationExecutionResult::MissingInput:
        return ValidationFailureReason::SimulationMissingInput;
    case ReplayValidationExecutionResult::InvalidPlan:
        return ValidationFailureReason::SimulationInvalidPlan;
    case ReplayValidationExecutionResult::ControlPlanInvalidRequest:
        return ValidationFailureReason::SimulationControlPlanInvalidRequest;
    case ReplayValidationExecutionResult::ControlTargetAllocationFailed:
        return ValidationFailureReason::SimulationControlTargetAllocationFailed;
    case ReplayValidationExecutionResult::ControlTargetTimeOutOfRange:
        return ValidationFailureReason::SimulationControlTargetTimeOutOfRange;
    case ReplayValidationExecutionResult::ControlTargetNonFinite:
        return ValidationFailureReason::SimulationControlTargetNonFinite;
    case ReplayValidationExecutionResult::ControlTickReservationFailed:
        return ValidationFailureReason::SimulationControlTickReservationFailed;
    case ReplayValidationExecutionResult::ControlTickAllocationFailed:
        return ValidationFailureReason::SimulationControlTickAllocationFailed;
    case ReplayValidationExecutionResult::ControlTargetMissing:
        return ValidationFailureReason::SimulationControlTargetMissing;
    case ReplayValidationExecutionResult::ControlOutputMissing:
        return ValidationFailureReason::SimulationControlOutputMissing;
    case ReplayValidationExecutionResult::InvalidControlPlan:
        return ValidationFailureReason::SimulationInvalidControlPlan;
    case ReplayValidationExecutionResult::PhysicsInputInvalid:
        return ValidationFailureReason::SimulationPhysicsInputInvalid;
    case ReplayValidationExecutionResult::MapStartUnavailable:
        return ValidationFailureReason::SimulationMapStartUnavailable;
    case ReplayValidationExecutionResult::ObservationAllocationFailed:
        return ValidationFailureReason::ObservationAllocationFailed;
    case ReplayValidationExecutionResult::DeterministicExecutionUnavailable:
        return ValidationFailureReason::DeterministicExecutionUnavailable;
    case ReplayValidationExecutionResult::CudaUnavailable: {
        const CudaBackendDiagnostics diagnostics =
                QueryCudaBackendDiagnostics();
        switch (diagnostics.status) {
        case CudaBackendStatus::NotCompiled:
            return ValidationFailureReason::CudaNotCompiled;
        case CudaBackendStatus::RuntimeUnavailable:
            return ValidationFailureReason::CudaRuntimeUnavailable;
        case CudaBackendStatus::NoDevice:
            return ValidationFailureReason::CudaDeviceUnavailable;
        case CudaBackendStatus::UnsupportedDevice:
            return ValidationFailureReason::CudaDeviceUnsupported;
        case CudaBackendStatus::InitializationFailed:
        case CudaBackendStatus::Ready:
            return ValidationFailureReason::CudaInitializationFailed;
        }
        return ValidationFailureReason::CudaInitializationFailed;
    }
    case ReplayValidationExecutionResult::CudaInitializationFailed:
        return ValidationFailureReason::CudaInitializationFailed;
    case ReplayValidationExecutionResult::CudaExecutionFailed:
        return ValidationFailureReason::CudaExecutionFailed;
    }
    return ValidationFailureReason::UnexpectedFailure;
}

ValidationError ExecutionError(
        ReplayValidationExecutionResult result,
        const ReplayIdentity &identity) {
    ValidationErrorCategory category = ValidationErrorCategory::Simulation;
    ValidationErrorCode code = ValidationErrorCode::SimulationFailed;
    ValidationStage stage = ValidationStage::SimulationStep;
    const char *diagnostic = "replay simulation failed";
    switch (result) {
    case ReplayValidationExecutionResult::ControlTargetAllocationFailed:
    case ReplayValidationExecutionResult::ControlTickReservationFailed:
    case ReplayValidationExecutionResult::ControlTickAllocationFailed:
        category = ValidationErrorCategory::Allocation;
        code = ValidationErrorCode::AllocationFailed;
        break;
    case ReplayValidationExecutionResult::MapStartUnavailable:
        stage = ValidationStage::SimulationStartup;
        break;
    case ReplayValidationExecutionResult::ObservationAllocationFailed:
        category = ValidationErrorCategory::Observation;
        code = ValidationErrorCode::ObservationFailed;
        stage = ValidationStage::Observation;
        break;
    case ReplayValidationExecutionResult::DeterministicExecutionUnavailable:
        code = ValidationErrorCode::DeterministicExecutionUnavailable;
        stage = ValidationStage::SimulationStartup;
        diagnostic = "deterministic execution mode unavailable";
        break;
    case ReplayValidationExecutionResult::CudaUnavailable:
        code = ValidationErrorCode::CudaUnavailable;
        stage = ValidationStage::SimulationStartup;
        diagnostic = "CUDA backend unavailable";
        break;
    case ReplayValidationExecutionResult::CudaInitializationFailed:
        code = ValidationErrorCode::CudaInitializationFailed;
        stage = ValidationStage::SimulationStartup;
        diagnostic = "CUDA backend initialization or certification failed";
        break;
    case ReplayValidationExecutionResult::CudaExecutionFailed:
        code = ValidationErrorCode::CudaExecutionFailed;
        stage = ValidationStage::SimulationStep;
        diagnostic = "CUDA backend execution failed";
        break;
    case ReplayValidationExecutionResult::Success:
    case ReplayValidationExecutionResult::MissingInput:
    case ReplayValidationExecutionResult::InvalidPlan:
    case ReplayValidationExecutionResult::ControlPlanInvalidRequest:
    case ReplayValidationExecutionResult::ControlTargetTimeOutOfRange:
    case ReplayValidationExecutionResult::ControlTargetNonFinite:
    case ReplayValidationExecutionResult::ControlTargetMissing:
    case ReplayValidationExecutionResult::ControlOutputMissing:
    case ReplayValidationExecutionResult::InvalidControlPlan:
    case ReplayValidationExecutionResult::PhysicsInputInvalid:
        break;
    }
    ValidationError error = MakeError(
            category, code, stage, ExecutionReason(result), identity, diagnostic);
    if (result == ReplayValidationExecutionResult::CudaUnavailable ||
        result == ReplayValidationExecutionResult::CudaInitializationFailed ||
        result == ReplayValidationExecutionResult::CudaExecutionFailed) {
        const CudaBackendDiagnostics cuda = QueryCudaBackendDiagnostics();
        if (!cuda.diagnostic.empty()) {
            error.diagnostic = cuda.diagnostic;
        }
    }
    return error;
}

Result<AssetBytes> LoadRequiredAsset(
        ValidationState &context,
        const char *identifier,
        ValidationFailureReason missingReason,
        const ReplayIdentity &identity) {
    const AssetRequest request{identifier};
    Result<AssetBytes> loaded = [&]() -> Result<AssetBytes> {
        try {
            return context.provider(request);
        } catch (const std::bad_alloc &) {
            ValidationError error = AllocationError(
                    ValidationStage::AssetLoading,
                    identity,
                    "allocation failed in asset provider");
            error.relatedAsset = identifier;
            return Result<AssetBytes>::Failure(std::move(error));
        } catch (...) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    ValidationFailureReason::AssetProviderFailed,
                    identity,
                    "asset provider threw an unexpected exception");
            error.relatedAsset = identifier;
            return Result<AssetBytes>::Failure(std::move(error));
        }
    }();
    if (!loaded) {
        ValidationError error = std::move(loaded).Error();
        error.stage = ValidationStage::AssetLoading;
        error.replay = identity;
        if (error.category == ValidationErrorCategory::Internal &&
            error.code == ValidationErrorCode::UnexpectedFailure &&
            error.reason == ValidationFailureReason::UnexpectedFailure) {
            error.category = ValidationErrorCategory::Asset;
            error.code = ValidationErrorCode::AssetLoadingFailed;
            error.reason = ValidationFailureReason::AssetProviderFailed;
        }
        if (error.relatedAsset.empty()) {
            error.relatedAsset = identifier;
        }
        return Result<AssetBytes>::Failure(std::move(error));
    }
    AssetBytes bytes = std::move(loaded).Value();
    if (bytes.empty()) {
        ValidationError error = MakeError(
                ValidationErrorCategory::Asset,
                ValidationErrorCode::AssetLoadingFailed,
                ValidationStage::AssetLoading,
                missingReason,
                identity,
                "required installed-pack asset is empty or unavailable");
        error.relatedAsset = identifier;
        return Result<AssetBytes>::Failure(std::move(error));
    }
    return Result<AssetBytes>::Success(std::move(bytes));
}

ValidationFailureReason MissingPackReason(std::string_view packName) noexcept {
    return packName == "Stadium"
            ? ValidationFailureReason::StadiumPackMissing
            : ValidationFailureReason::InstalledPackMissing;
}

ValidationFailureReason InvalidPackReason(std::string_view packName) noexcept {
    return packName == "Stadium"
            ? ValidationFailureReason::StadiumPackInvalid
            : ValidationFailureReason::InstalledPackInvalid;
}

Result<InstalledPackKeyCatalog *> PreparePackKeys(
        ValidationState &context,
        const ReplayIdentity &identity) {
    if (context.packKeys != nullptr) {
        return Result<InstalledPackKeyCatalog *>::Success(
                context.packKeys.get());
    }
    Result<AssetBytes> packlist = LoadRequiredAsset(
            context,
            "packlist.dat",
            ValidationFailureReason::PacklistMissing,
            identity);
    if (!packlist) {
        return Result<InstalledPackKeyCatalog *>::Failure(
                std::move(packlist).Error());
    }
    auto keys = std::make_unique<InstalledPackKeyCatalog>();
    context.packlistBytes = std::move(packlist).Value();
    if (!keys->LoadFromMemory(
                context.packlistBytes.data(),
                context.packlistBytes.size())) {
        context.packlistBytes.clear();
        ValidationError error = MakeError(
                ValidationErrorCategory::Asset,
                ValidationErrorCode::AssetLoadingFailed,
                ValidationStage::AssetLoading,
                ValidationFailureReason::InstalledPackInvalid,
                identity,
                "could not authenticate or decode packlist.dat");
        error.relatedAsset = "packlist.dat";
        return Result<InstalledPackKeyCatalog *>::Failure(std::move(error));
    }
    context.packKeys = std::move(keys);
    return Result<InstalledPackKeyCatalog *>::Success(context.packKeys.get());
}

Result<CachedInstalledPack *> PrepareInstalledPack(
        ValidationState &context,
        std::string_view packName,
        const ReplayIdentity &identity) {
    for (const auto &cached : context.installedPacks) {
        if (cached->packName == packName) {
            return Result<CachedInstalledPack *>::Success(cached.get());
        }
    }

    Result<InstalledPackKeyCatalog *> keyResult =
            PreparePackKeys(context, identity);
    if (!keyResult) {
        return Result<CachedInstalledPack *>::Failure(
                std::move(keyResult).Error());
    }
    try {
        const std::string packNameText(packName);
        const std::string identifier = packNameText + ".pak";
        Result<AssetBytes> loaded = LoadRequiredAsset(
                context,
                identifier.c_str(),
                MissingPackReason(packName),
                identity);
        if (!loaded) {
            return Result<CachedInstalledPack *>::Failure(
                    std::move(loaded).Error());
        }
        auto cached = std::make_unique<CachedInstalledPack>();
        cached->packName = packNameText;
        cached->bytes = std::move(loaded).Value();
        CPlugFilePack probe;
        if (!probe.OpenFromMemory(
                    cached->bytes.data(),
                    cached->bytes.size(),
                    *keyResult.Value(),
                    cached->packName.c_str())) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    InvalidPackReason(packName),
                    identity,
                    "could not decode the selected installed pack");
            error.relatedAsset = identifier;
            return Result<CachedInstalledPack *>::Failure(std::move(error));
        }
        CachedInstalledPack *result = cached.get();
        context.installedPacks.push_back(std::move(cached));
        return Result<CachedInstalledPack *>::Success(result);
    } catch (const std::bad_alloc &) {
        return Result<CachedInstalledPack *>::Failure(AllocationError(
                ValidationStage::AssetLoading,
                identity,
                "allocation failed while caching an installed pack"));
    }
}

Result<CachedPackAssets *> PreparePackAssets(
        ValidationState &context,
        std::string_view packName,
        const ReplayIdentity &identity) {
    for (const auto &cached : context.assetRepositories) {
        if (cached->packName == packName) {
            return Result<CachedPackAssets *>::Success(cached.get());
        }
    }
    Result<CachedInstalledPack *> packResult =
            PrepareInstalledPack(context, packName, identity);
    if (!packResult) {
        return Result<CachedPackAssets *>::Failure(
                std::move(packResult).Error());
    }
    try {
        CachedInstalledPack &pack = *packResult.Value();
        auto cached = std::make_unique<CachedPackAssets>();
        cached->packName = pack.packName;
        cached->repository = OpenReplayAssetRepository(
                pack.bytes.data(),
                pack.bytes.size(),
                *context.packKeys,
                pack.packName.c_str());
        if (!cached->repository) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    ValidationFailureReason::AssetRepositoryUnavailable,
                    identity,
                    "could not create the routed pack asset repository");
            error.relatedAsset = pack.packName + ".pak";
            return Result<CachedPackAssets *>::Failure(std::move(error));
        }
        CachedPackAssets *result = cached.get();
        context.assetRepositories.push_back(std::move(cached));
        return Result<CachedPackAssets *>::Success(result);
    } catch (const std::bad_alloc &) {
        return Result<CachedPackAssets *>::Failure(AllocationError(
                ValidationStage::AssetLoading,
                identity,
                "allocation failed while caching routed pack assets"));
    }
}

Result<CachedVehicleAssets *> PrepareVehicleAssets(
        ValidationState &context,
        ::ReplayVehicleModel vehicleModel,
        std::string_view packName,
        const ReplayIdentity &identity) {
    for (const auto &cached : context.vehicleAssets) {
        if (cached->vehicleModel == vehicleModel &&
            cached->packName == packName) {
            return Result<CachedVehicleAssets *>::Success(cached.get());
        }
    }
    Result<CachedInstalledPack *> packResult =
            PrepareInstalledPack(context, packName, identity);
    if (!packResult) {
        return Result<CachedVehicleAssets *>::Failure(
                std::move(packResult).Error());
    }
    try {
        CachedInstalledPack &installed = *packResult.Value();
        CPlugFilePack pack;
        if (!pack.OpenFromMemory(
                    installed.bytes.data(),
                    installed.bytes.size(),
                    *context.packKeys,
                    installed.packName.c_str())) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    InvalidPackReason(packName),
                    identity,
                    "could not reopen the routed vehicle pack");
            error.relatedAsset = installed.packName + ".pak";
            return Result<CachedVehicleAssets *>::Failure(std::move(error));
        }
        std::optional<InstalledVehicleAssetGraph> assetGraph =
                InstalledVehicleAssetGraph::ResolveFromPack(pack);
        if (!assetGraph.has_value()) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    ValidationFailureReason::DefaultVehicleUnavailable,
                    identity,
                    "could not resolve the routed vehicle asset graph");
            error.relatedAsset = installed.packName + ".pak";
            return Result<CachedVehicleAssets *>::Failure(std::move(error));
        }
        std::optional<DefaultVehiclePackData> vehicle =
                DefaultVehiclePackArchive::LoadFromPack(pack, *assetGraph);
        std::optional<ReplayVehicleSolidDefinition> solid =
                DefaultVehicleSolidArchive::LoadFromPack(pack, *assetGraph);
        if (!vehicle.has_value() || !solid.has_value()) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    ValidationFailureReason::DefaultVehicleUnavailable,
                    identity,
                    "could not load the routed vehicle definitions");
            error.relatedAsset = installed.packName + ".pak";
            return Result<CachedVehicleAssets *>::Failure(std::move(error));
        }
        auto cached = std::make_unique<CachedVehicleAssets>();
        cached->packName = installed.packName;
        cached->vehicleModel = vehicleModel;
        cached->assetGraph = std::move(*assetGraph);
        cached->vehicleSources = ReplayVehicleSourceBundle{
                std::move(*solid),
                std::move(vehicle->tuning),
                std::move(vehicle->vehicle)};
        if (!cached->vehicleSources.IsComplete()) {
            ValidationError error = MakeError(
                    ValidationErrorCategory::Asset,
                    ValidationErrorCode::AssetLoadingFailed,
                    ValidationStage::AssetLoading,
                    ValidationFailureReason::DefaultVehicleUnavailable,
                    identity,
                    "routed vehicle definitions are incomplete");
            error.relatedAsset = installed.packName + ".pak";
            return Result<CachedVehicleAssets *>::Failure(std::move(error));
        }
        CachedVehicleAssets *result = cached.get();
        context.vehicleAssets.push_back(std::move(cached));
        return Result<CachedVehicleAssets *>::Success(result);
    } catch (const std::bad_alloc &) {
        return Result<CachedVehicleAssets *>::Failure(AllocationError(
                ValidationStage::AssetLoading,
                identity,
                "allocation failed while caching routed vehicle assets"));
    }
}

Result<PreparedAssets> PrepareAssets(
        ValidationState &context,
        const ReplayAssetRoute &route,
        const ReplayIdentity &identity) {
    Result<CachedPackAssets *> mapResult =
            PreparePackAssets(context, route.mapPackName, identity);
    if (!mapResult) {
        return Result<PreparedAssets>::Failure(std::move(mapResult).Error());
    }
    Result<CachedPackAssets *> decorationResult = PreparePackAssets(
            context, route.decorationPackName, identity);
    if (!decorationResult) {
        return Result<PreparedAssets>::Failure(
                std::move(decorationResult).Error());
    }
    Result<CachedVehicleAssets *> vehicleResult = PrepareVehicleAssets(
            context,
            route.vehicleModel,
            route.vehiclePackName,
            identity);
    if (!vehicleResult) {
        return Result<PreparedAssets>::Failure(
                std::move(vehicleResult).Error());
    }
    return Result<PreparedAssets>::Success(PreparedAssets{
            mapResult.Value()->repository.get(),
            decorationResult.Value()->repository.get(),
            &vehicleResult.Value()->vehicleSources,
    });
}

Result<ValidationReport> RunReplayValidation(
        ValidationState &context,
        ByteView replayBytes,
        const ReplayIdentity &identity,
        const ValidationOptions &options) {
    ReplayFile replayFile;
    const ReplayFileReadError readError = ReadReplayBytes(
            reinterpret_cast<const std::uint8_t *>(replayBytes.data),
            replayBytes.size,
            &replayFile);
    if (readError != ReplayFileReadError::Success) {
        return Result<ValidationReport>::Failure(
                ReplayDecodeError(readError, identity));
    }

    ReplayAssetRoute route;
    const ReplayAssetRouteResult routeResult =
            BuildReplayAssetRoute(replayFile, &route);
    if (routeResult != ReplayAssetRouteResult::Success) {
        return Result<ValidationReport>::Failure(
                ReplayRouteError(routeResult, identity, replayFile));
    }
    if (options.backend == SimulationBackend::Cuda &&
        !IsCudaSupportedRoute(route)) {
        return Result<ValidationReport>::Failure(
                CudaScopeError(route, identity));
    }

    const ReplayValidationConfiguration configuration{
            options.requestedSamples,
            options.controlTickMs,
            options.validationPrestartMs,
            {},
            100000u,
    };
    std::optional<ReplayFileValidationResult> compatibility =
            ClassifyReplayCompatibility(replayFile, configuration);
    if (compatibility.has_value()) {
        return Result<ValidationReport>::Success(ToPublicReport(
                identity, *compatibility, replayFile, route));
    }
    std::optional<ReplayFileValidationResult> inputAvailability =
            ClassifyReplayInputAvailability(replayFile, configuration);
    if (inputAvailability.has_value()) {
        return Result<ValidationReport>::Success(ToPublicReport(
                identity, *inputAvailability, replayFile, route));
    }

    Result<PreparedAssets> preparedResult =
            PrepareAssets(context, route, identity);
    if (!preparedResult) {
        return Result<ValidationReport>::Failure(
                std::move(preparedResult).Error());
    }
    const PreparedAssets prepared = preparedResult.Value();

    ReplaySimulationSession simulationSession(options.backend);
    CGameCtnReplayChallengeMapPreload preload;
    const ReplayChallengePreloadResult preloadResult = preload.Preload(
            replayFile.MapInput(),
            *prepared.mapAssets,
            *prepared.decorationAssets,
            simulationSession);
    if (preloadResult != ReplayChallengePreloadResult::Success) {
        ValidationError error = PreloadError(preloadResult, identity);
        if (options.backend == SimulationBackend::Cuda &&
            !simulationSession.CudaInitializationDiagnostic().empty()) {
            error.diagnostic =
                    simulationSession.CudaInitializationDiagnostic();
        }
        return Result<ValidationReport>::Failure(
                std::move(error));
    }

    ReplaySimulationDefinitionBuild definition =
            BuildReplaySimulationDefinition(
                    *prepared.vehicleSources, preload.WaterDefinition());
    if (!definition) {
        return Result<ValidationReport>::Failure(
                DefinitionError(definition.Error(), identity));
    }
    definition.Value().optimizedCpuStadiumSpecializationsEnabled =
            route.vehicleModel == ::ReplayVehicleModel::StadiumCar;

    simulationSession.ActivateStaticScene();
    ReplayFileValidationBuild validation = ValidateReplayFile(
            replayFile,
            route.validationMode,
            simulationSession,
            definition.Value(),
            configuration);
    if (!validation) {
        ValidationError error =
                ExecutionError(validation.Error(), identity);
        if (options.backend == SimulationBackend::Cuda &&
            !simulationSession.CudaInitializationDiagnostic().empty()) {
            error.diagnostic =
                    simulationSession.CudaInitializationDiagnostic();
        }
        return Result<ValidationReport>::Failure(
                std::move(error));
    }
    return Result<ValidationReport>::Success(
            ToPublicReport(identity, validation.Value(), replayFile, route));
}

}  // namespace

namespace experimental {

namespace {

constexpr std::uint32_t SandboxInputTimeBaseMs = 100000u;
constexpr std::uint32_t SandboxRuntimeCloneSchema = 1u;

PhysicsSandboxError SandboxError(
        PhysicsSandboxErrorCode code,
        const char *diagnostic,
        ValidationError validationError = {}) {
    PhysicsSandboxError error;
    error.code = code;
    error.validationError = std::move(validationError);
    error.diagnostic = diagnostic == nullptr ? "" : diagnostic;
    return error;
}

PhysicsSandboxInputAction ToSandboxAction(ReplayInputActionKind action) {
    switch (action) {
    case ReplayInputActionKind::Accelerate:
        return PhysicsSandboxInputAction::Accelerate;
    case ReplayInputActionKind::Gas: return PhysicsSandboxInputAction::Gas;
    case ReplayInputActionKind::Brake: return PhysicsSandboxInputAction::Brake;
    case ReplayInputActionKind::Steer: return PhysicsSandboxInputAction::Steer;
    case ReplayInputActionKind::SteerLeft:
        return PhysicsSandboxInputAction::SteerLeft;
    case ReplayInputActionKind::SteerRight:
        return PhysicsSandboxInputAction::SteerRight;
    case ReplayInputActionKind::RaceRunning:
        return PhysicsSandboxInputAction::RaceRunning;
    case ReplayInputActionKind::FinishLine:
        return PhysicsSandboxInputAction::FinishLine;
    case ReplayInputActionKind::Respawn:
        return PhysicsSandboxInputAction::Respawn;
    case ReplayInputActionKind::Unmapped:
        return PhysicsSandboxInputAction::Unmapped;
    }
    return PhysicsSandboxInputAction::Unmapped;
}

ReplayInputActionKind FromSandboxAction(PhysicsSandboxInputAction action) {
    switch (action) {
    case PhysicsSandboxInputAction::Accelerate:
        return ReplayInputActionKind::Accelerate;
    case PhysicsSandboxInputAction::Gas: return ReplayInputActionKind::Gas;
    case PhysicsSandboxInputAction::Brake: return ReplayInputActionKind::Brake;
    case PhysicsSandboxInputAction::Steer: return ReplayInputActionKind::Steer;
    case PhysicsSandboxInputAction::SteerLeft:
        return ReplayInputActionKind::SteerLeft;
    case PhysicsSandboxInputAction::SteerRight:
        return ReplayInputActionKind::SteerRight;
    case PhysicsSandboxInputAction::RaceRunning:
        return ReplayInputActionKind::RaceRunning;
    case PhysicsSandboxInputAction::FinishLine:
        return ReplayInputActionKind::FinishLine;
    case PhysicsSandboxInputAction::Respawn:
        return ReplayInputActionKind::Respawn;
    case PhysicsSandboxInputAction::Unmapped:
        return ReplayInputActionKind::Unmapped;
    }
    return ReplayInputActionKind::Unmapped;
}

PhysicsSandboxInputValue ToSandboxValue(
        const ReplayInputActionValue &value) {
    PhysicsSandboxInputValue result;
    switch (value.Kind()) {
    case ReplayInputActionValueKind::None:
        result.kind = PhysicsSandboxInputValueKind::None;
        break;
    case ReplayInputActionValueKind::Analog:
        result.kind = PhysicsSandboxInputValueKind::Analog;
        result.analog = value.AnalogValue();
        break;
    case ReplayInputActionValueKind::Switch:
        result.kind = PhysicsSandboxInputValueKind::Switch;
        if (!value.IsActive()) {
            result.switchState = PhysicsSandboxSwitchState::Released;
        } else if (value.IsCanonicalPress()) {
            result.switchState = PhysicsSandboxSwitchState::Pressed;
        } else {
            result.switchState =
                    PhysicsSandboxSwitchState::NonCanonicalActive;
        }
        break;
    }
    return result;
}

PhysicsSandboxSceneView BuildSandboxScene(
        const ReplaySimulationSession &session,
        const ReplaySimulationDefinition &definition) {
    PhysicsSandboxSceneView scene;
    const std::vector<ReplayStaticCollisionTriangle> &triangles =
            session.StaticCollisionTriangles();
    scene.collisionTriangles.reserve(triangles.size());
    for (const ReplayStaticCollisionTriangle &triangle : triangles) {
        scene.collisionTriangles.push_back({
                ToPublicVector(triangle.a),
                ToPublicVector(triangle.b),
                ToPublicVector(triangle.c)});
    }

    const std::vector<VehicleCollisionShapeEntry> &shapes =
            definition.vehicle.collisionModel.ShapesInArchiveOrder();
    std::vector<GmIso4> absolutePoses;
    absolutePoses.reserve(shapes.size());
    scene.carEllipsoids.reserve(shapes.size());
    for (const VehicleCollisionShapeEntry &entry : shapes) {
        GmIso4 absolutePose = entry.shape.localPose;
        if (entry.parentShapeIndex.has_value()) {
            absolutePose.Mult(absolutePoses[*entry.parentShapeIndex]);
        }
        absolutePoses.push_back(absolutePose);

        GmVec3 center;
        center.SetMult(entry.shape.localBounds.center, absolutePose);
        GmQuat rotation;
        rotation.Set(absolutePose.rotation);
        rotation.Normalize();
        const GmVec3 &halfExtents = entry.shape.localBounds.halfExtents;
        scene.carEllipsoids.push_back({
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
                ToPublicVector(center),
                {std::fabs(halfExtents.x),
                 std::fabs(halfExtents.y),
                 std::fabs(halfExtents.z)}});
    }
    return scene;
}

ReplayInputActionValue FromSandboxValue(
        const PhysicsSandboxInputValue &value) {
    switch (value.kind) {
    case PhysicsSandboxInputValueKind::None:
        return ReplayInputActionValue::None();
    case PhysicsSandboxInputValueKind::Analog:
        return ReplayInputActionValue::Analog(value.analog);
    case PhysicsSandboxInputValueKind::Switch:
        switch (value.switchState) {
        case PhysicsSandboxSwitchState::Released:
            return ReplayInputActionValue::Switch(
                    ReplayInputSwitchState::Released);
        case PhysicsSandboxSwitchState::Pressed:
            return ReplayInputActionValue::Switch(
                    ReplayInputSwitchState::Pressed);
        case PhysicsSandboxSwitchState::NonCanonicalActive:
            return ReplayInputActionValue::Switch(
                    ReplayInputSwitchState::NonCanonicalActive);
        }
    }
    return ReplayInputActionValue::None();
}

bool SameInputEvent(const PhysicsSandboxInputEvent &left,
                    const PhysicsSandboxInputEvent &right) {
    return left.timeMs == right.timeMs && left.action == right.action &&
           left.value.kind == right.value.kind &&
           left.value.switchState == right.value.switchState &&
           left.value.analog == right.value.analog;
}

std::uint64_t Fingerprint(ByteView bytes) {
    constexpr std::uint64_t Offset = 1469598103934665603ull;
    constexpr std::uint64_t Prime = 1099511628211ull;
    std::uint64_t result = Offset;
    for (std::size_t index = 0u; index < bytes.size; ++index) {
        result ^= static_cast<std::uint8_t>(bytes.data[index]);
        result *= Prime;
    }
    return result;
}

}  // namespace

namespace {

struct SandboxInputWindow {
    std::int64_t minimumTimeMs = 0;
    std::int64_t maximumTimeMs = 0;
    std::vector<PhysicsSandboxInputEvent> events;
};

struct SandboxInputStorage {
    std::shared_ptr<const std::vector<PhysicsSandboxInputEvent>> base;
    std::optional<SandboxInputWindow> window;

    static std::shared_ptr<const SandboxInputStorage> Full(
            std::vector<PhysicsSandboxInputEvent> events) {
        auto storage = std::make_shared<SandboxInputStorage>();
        storage->base = std::make_shared<const std::vector<
                PhysicsSandboxInputEvent>>(std::move(events));
        return storage;
    }

    std::size_t Size() const noexcept {
        if (!base) return 0u;
        if (!window) return base->size();
        const auto first = std::lower_bound(
                base->begin(), base->end(), window->minimumTimeMs,
                [](const PhysicsSandboxInputEvent &event,
                   std::int64_t timeMs) {
                    return event.timeMs < timeMs;
                });
        const auto last = std::upper_bound(
                first, base->end(), window->maximumTimeMs,
                [](std::int64_t timeMs,
                   const PhysicsSandboxInputEvent &event) {
                    return timeMs < event.timeMs;
                });
        return base->size() - static_cast<std::size_t>(last - first) +
                window->events.size();
    }

    std::vector<PhysicsSandboxInputEvent> Materialize() const {
        if (!base) return {};
        if (!window) return *base;
        const auto first = std::lower_bound(
                base->begin(), base->end(), window->minimumTimeMs,
                [](const PhysicsSandboxInputEvent &event,
                   std::int64_t timeMs) {
                    return event.timeMs < timeMs;
                });
        const auto last = std::upper_bound(
                first, base->end(), window->maximumTimeMs,
                [](std::int64_t timeMs,
                   const PhysicsSandboxInputEvent &event) {
                    return timeMs < event.timeMs;
                });
        std::vector<PhysicsSandboxInputEvent> result;
        result.reserve(Size());
        result.insert(result.end(), base->begin(), first);
        result.insert(result.end(), window->events.begin(),
                      window->events.end());
        result.insert(result.end(), last, base->end());
        return result;
    }

    std::vector<PhysicsSandboxInputEvent> MaterializeThrough(
            std::int64_t maximumTimeMs) const {
        std::vector<PhysicsSandboxInputEvent> result;
        if (!base) return result;
        const auto baseEnd = std::upper_bound(
                base->begin(), base->end(), maximumTimeMs,
                [](std::int64_t timeMs,
                   const PhysicsSandboxInputEvent &event) {
                    return timeMs < event.timeMs;
                });
        if (!window || window->minimumTimeMs > maximumTimeMs) {
            result.assign(base->begin(), baseEnd);
            return result;
        }
        const auto first = std::lower_bound(
                base->begin(), baseEnd, window->minimumTimeMs,
                [](const PhysicsSandboxInputEvent &event,
                   std::int64_t timeMs) {
                    return event.timeMs < timeMs;
                });
        const auto last = std::upper_bound(
                first, baseEnd,
                std::min(window->maximumTimeMs, maximumTimeMs),
                [](std::int64_t timeMs,
                   const PhysicsSandboxInputEvent &event) {
                    return timeMs < event.timeMs;
                });
        result.reserve(static_cast<std::size_t>(first - base->begin()) +
                       window->events.size() +
                       static_cast<std::size_t>(baseEnd - last));
        result.insert(result.end(), base->begin(), first);
        for (const PhysicsSandboxInputEvent &event : window->events) {
            if (event.timeMs <= maximumTimeMs) result.push_back(event);
        }
        result.insert(result.end(), last, baseEnd);
        return result;
    }
};

struct SandboxControlPlanStorage {
    std::shared_ptr<const ReplayControlPlan> base;
    std::size_t replacementBegin = 0u;
    std::vector<ReplayControlTick> replacement;

    static std::shared_ptr<const SandboxControlPlanStorage> Full(
            ReplayControlPlan plan) {
        auto storage = std::make_shared<SandboxControlPlanStorage>();
        storage->base =
                std::make_shared<const ReplayControlPlan>(std::move(plan));
        return storage;
    }

    std::size_t Size() const noexcept {
        return base ? base->ticks.size() : 0u;
    }

    const ReplayControlTick &Tick(std::size_t index) const {
        if (!replacement.empty() && index >= replacementBegin &&
            index - replacementBegin < replacement.size()) {
            return replacement[index - replacementBegin];
        }
        return base->ticks[index];
    }

    ReplayControlPlan Materialize() const {
        ReplayControlPlan result = base ? *base : ReplayControlPlan{};
        if (!replacement.empty()) {
            std::copy(replacement.begin(), replacement.end(),
                      result.ticks.begin() + replacementBegin);
        }
        return result;
    }

    std::vector<ReplayControlTick> CopyRange(
            std::size_t begin, std::size_t count) const {
        std::vector<ReplayControlTick> result;
        result.reserve(count);
        for (std::size_t index = begin; index < begin + count; ++index) {
            result.push_back(Tick(index));
        }
        return result;
    }
};

}  // namespace

struct PhysicsSandboxState::Impl {
    PhysicsSandboxStateView view{};
    std::shared_ptr<const ReplaySimulationInstanceClone> runtimeClone;
    std::shared_ptr<const SandboxInputStorage> inputs;
    std::shared_ptr<const SandboxControlPlanStorage> controlPlan;
    std::uint64_t scenarioFingerprint = 0u;
    std::uint32_t validationSeed = 0u;
    SimulationBackend backend = SimulationBackend::Reference;
    std::uint32_t tickDurationMs = 0u;
    std::uint32_t prestartDurationMs = 0u;
    std::uint32_t simulationHorizonMs = 0u;
    PhysicsSandboxTimelineMode timelineMode =
            PhysicsSandboxTimelineMode::RecordedReplay;
    std::size_t cursor = 0u;
    std::uint32_t runtimeCloneSchema = 0u;
};

struct PhysicsSandbox::Impl {
    Impl(AssetProvider provider, PhysicsSandboxOptions sandboxOptions)
        : validationState(std::move(provider)), options(sandboxOptions) {}

    ValidationState validationState;
    PhysicsSandboxOptions options{};
    std::unique_ptr<ReplaySimulationSession> session;
    ReplaySimulationDefinition definition{};
    std::shared_ptr<const SandboxControlPlanStorage> controlPlan;
    ReplayInputMetadata inputMetadata{};
    std::vector<ReplayInputActionKind> definedActions;
    ReplayInputProvenance provenance = ReplayInputProvenance::Unmarked;
    std::shared_ptr<const SandboxInputStorage> inputs;
    std::uint32_t simulationHorizonMs = 0u;
    ReplayChallengeMetadata challengeMetadata{};
    ReplayAssetRoute route{};
    ReplayIdentity identity{};
    std::uint64_t scenarioFingerprint = 0u;
    PhysicsSandboxSceneView scene{};
    PhysicsSandboxRenderSceneHandle renderScene;
    std::size_t cursor = 0u;
    std::size_t prestartTicks = 0u;
    bool loaded = false;

    PhysicsSandboxResult<ReplayControlPlan> BuildControlPlan(
            const std::vector<PhysicsSandboxInputEvent> &source,
            std::optional<std::int64_t> durationOverrideMs =
                    std::nullopt) const {
        std::vector<ReplayInputEvent> events;
        try {
            events.reserve(source.size());
        } catch (const std::bad_alloc &) {
            return PhysicsSandboxResult<ReplayControlPlan>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                                 "could not allocate sandbox inputs"));
        }
        for (const PhysicsSandboxInputEvent &event : source) {
            const std::int64_t absoluteTime =
                    static_cast<std::int64_t>(SandboxInputTimeBaseMs) +
                    event.timeMs;
            if (absoluteTime < 0 || absoluteTime >
                    std::numeric_limits<std::uint32_t>::max() ||
                (event.value.kind == PhysicsSandboxInputValueKind::Analog &&
                 !IsAnalogInputStateValid(event.value.analog))) {
                return PhysicsSandboxResult<ReplayControlPlan>::Failure(
                        SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                     "sandbox input is out of range"));
            }
            events.push_back({
                    static_cast<std::uint32_t>(absoluteTime),
                    FromSandboxAction(event.action),
                    FromSandboxValue(event.value)});
        }

        ReplayInputTimeline timeline;
        const ReplayInputTimelineCreateResult timelineResult =
                ReplayInputTimeline::Create(
                        inputMetadata,
                        definedActions,
                        std::move(events),
                        &timeline,
                        provenance);
        if (timelineResult != ReplayInputTimelineCreateResult::Success) {
            return PhysicsSandboxResult<ReplayControlPlan>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "sandbox input timeline is invalid"));
        }

        ReplayControlPlanRequest request(timeline);
        request.controlTickMs = options.tickDurationMs;
        const std::int64_t durationMs = durationOverrideMs.value_or(
                static_cast<std::int64_t>(simulationHorizonMs));
        if (durationMs < 0 ||
            durationMs > std::numeric_limits<std::int32_t>::max()) {
            return PhysicsSandboxResult<ReplayControlPlan>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "sandbox control duration is out of range"));
        }
        request.validationDurationMs =
                static_cast<std::int32_t>(durationMs);
        request.validationPrestartMs = options.prestartDurationMs;
        request.inputTimeBaseMs = SandboxInputTimeBaseMs;
        request.enableRaceSimulationAfterMs =
                static_cast<std::int32_t>(options.prestartDurationMs);
        request.establishRaceSpawnAtMs = 0;
        request.baseActions.enableStuntsSimulation =
                route.validationMode == ReplayValidationMode::Stunts;
        request.baseActions.stuntsTimeLimitMs =
                request.baseActions.enableStuntsSimulation
                        ? challengeMetadata.stuntsTimeLimitMs.value_or(
                                  DefaultChallengeStuntsTimeLimitMs)
                        : 0u;
        ReplayControlPlan plan;
        if (BuildReplayControlPlan(request, &plan) !=
                ReplayControlPlanBuildResult::Success || plan.ticks.empty()) {
            return PhysicsSandboxResult<ReplayControlPlan>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "could not build sandbox control ticks"));
        }
        return PhysicsSandboxResult<ReplayControlPlan>::Success(
                std::move(plan));
    }

    ReplaySimulationTimelineResult AdvanceControlPlan(
            std::size_t begin, std::size_t count) {
        ReplaySimulationTimelineResult aggregate;
        aggregate.result = ReplaySimulationRunResult::Success;
        if (!controlPlan || !session || begin > controlPlan->Size() ||
            count > controlPlan->Size() - begin) {
            aggregate.result = ReplaySimulationRunResult::InvalidControlTimeline;
            return aggregate;
        }
        std::size_t cursor = begin;
        std::size_t remaining = count;
        while (remaining != 0u) {
            const bool inReplacement =
                    !controlPlan->replacement.empty() &&
                    cursor >= controlPlan->replacementBegin &&
                    cursor - controlPlan->replacementBegin <
                            controlPlan->replacement.size();
            const std::vector<ReplayControlTick> *ticks = nullptr;
            std::size_t localBegin = 0u;
            std::size_t chunk = remaining;
            if (inReplacement) {
                ticks = &controlPlan->replacement;
                localBegin = cursor - controlPlan->replacementBegin;
                chunk = std::min(
                        chunk,
                        controlPlan->replacement.size() - localBegin);
            } else {
                ticks = &controlPlan->base->ticks;
                localBegin = cursor;
                if (!controlPlan->replacement.empty() &&
                    cursor < controlPlan->replacementBegin) {
                    chunk = std::min(
                            chunk,
                            controlPlan->replacementBegin - cursor);
                }
            }
            const ReplaySimulationTimelineResult advanced =
                    session->AdvanceIncremental(*ticks, localBegin, chunk);
            if (advanced.result != ReplaySimulationRunResult::Success) {
                return advanced;
            }
            aggregate = advanced;
            cursor += chunk;
            remaining -= chunk;
        }
        return aggregate;
    }

    PhysicsSandboxResult<PhysicsSandboxStateView> Restart(
            std::uint64_t raceTick) {
        if (!loaded || !session || raceTick >
                std::numeric_limits<std::size_t>::max() - prestartTicks) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "sandbox state cannot be restored"));
        }
        const std::size_t targetCursor = prestartTicks +
                static_cast<std::size_t>(raceTick);
        if (!controlPlan || targetCursor > controlPlan->Size()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "sandbox state exceeds the input timeline"));
        }

        session->ConfigureReplayRace(
                challengeMetadata.playMode.value_or(EChallengePlayMode::Race),
                challengeMetadata.isLapRace,
                challengeMetadata.isLapRace ? challengeMetadata.lapCount : 1u);
        tmnf::simulation::DeterministicExecutionScope deterministicScope;
        if (!deterministicScope.Established()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::SimulationFailed,
                            "deterministic execution mode is unavailable"));
        }
        ReplaySimulationRunResult start = session->StartIncremental(
                definition,
                controlPlan->Tick(0u),
                inputMetadata.validationSeed);
        if (start != ReplaySimulationRunResult::Success) {
            deterministicScope.Restore();
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::SimulationFailed,
                                 "sandbox simulation could not start"));
        }
        const ReplaySimulationTimelineResult advanced =
                AdvanceControlPlan(0u, targetCursor);
        if (advanced.result != ReplaySimulationRunResult::Success ||
            !deterministicScope.Restore()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::SimulationFailed,
                                 "sandbox simulation could not be restored"));
        }
        cursor = targetCursor;
        return ReadView();
    }

    PhysicsSandboxResult<PhysicsSandboxStateView> ReadView() const {
        if (!loaded || !session || cursor < prestartTicks) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no loaded scenario"));
        }
        const std::optional<ReplaySimulationStateView> state =
                session->CurrentState();
        if (!state.has_value()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox simulation is not running"));
        }
        PhysicsSandboxStateView view;
        view.tick = cursor - prestartTicks;
        view.timeMs = view.tick * options.tickDurationMs;
        view.durationMs = simulationHorizonMs;
        view.mapEnvironment = ToPublicMapEnvironment(route.mapEnvironment);
        view.vehicleModel = ToPublicVehicleModel(route.vehicleModel);
        view.playMode = ToPublicPlayMode(
                challengeMetadata.playMode.value_or(EChallengePlayMode::Race));
        const ReplayDynaFrameState &frame = state->frame;
        view.car.rotationX = frame.rotationQuaternion.x;
        view.car.rotationY = frame.rotationQuaternion.y;
        view.car.rotationZ = frame.rotationQuaternion.z;
        view.car.rotationW = frame.rotationQuaternion.w;
        view.car.position = ToPublicVector(frame.position);
        view.car.linearSpeed = ToPublicVector(frame.linearSpeed);
        view.car.angularSpeed = ToPublicVector(frame.angularSpeed);
        view.car.force = ToPublicVector(frame.force);
        view.car.torque = ToPublicVector(frame.torque);
        view.car.signedSpeed = state->signedSpeed;
        view.car.turbo = state->turbo;
        view.car.cameraFlightTransition = state->cameraFlightTransition;
        view.car.burning = state->burning;
        view.car.gearChanged = state->gearChanged;
        view.car.wheelContact = state->wheelContact;
        view.car.wheelHasSurface = state->wheelHasSurface;
        view.car.cameraSupportUp = ToPublicVector(state->cameraSupportUp);
        view.car.localSpeed = ToPublicVector(state->localSpeed);
        view.car.freeWheeling = state->freeWheeling;
        view.car.lateralContact = state->lateralContact;
        view.car.sliding = state->sliding;
        view.car.gear = state->gear;
        view.car.rpm = state->rpm;
        view.car.turningRate = state->turningRate;
        view.car.turboType = state->turboType;
        view.car.turboBoostFactor = state->turboBoostFactor;
        view.car.wheelSliding = state->wheelSliding;
        view.car.wheelSurface = state->wheelSurface;
        view.accelerate = state->controls.lowSpeedGateA;
        view.brake = state->controls.lowSpeedGateB;
        view.steering = state->controls.steering;
        view.checkpointsCollected = state->race.checkpointCount;
        view.checkpointsTotal = state->race.requiredCheckpointCount;
        view.completedLaps = state->race.completedLapCount;
        view.totalLaps = state->race.requiredLapCount;
        view.raceCompleted = state->race.raceCompleted;
        if (state->finishTime.has_value()) {
            const std::uint64_t prestartNs =
                    static_cast<std::uint64_t>(
                            options.prestartDurationMs) *
                    1000000u;
            view.finishTime = *state->finishTime;
            view.finishTime->lowerBoundNs =
                    view.finishTime->lowerBoundNs >= prestartNs
                    ? view.finishTime->lowerBoundNs - prestartNs
                    : 0u;
            view.finishTime->upperBoundNs =
                    view.finishTime->upperBoundNs >= prestartNs
                    ? view.finishTime->upperBoundNs - prestartNs
                    : 0u;
            view.finishTime->estimatedNs =
                    view.finishTime->estimatedNs >= prestartNs
                    ? view.finishTime->estimatedNs - prestartNs
                    : 0u;
            view.finishTimeMs = static_cast<std::uint32_t>(
                    view.finishTime->estimatedNs / 1000000u);
        } else if (state->finishTimeMs.has_value()) {
            view.finishTimeMs = *state->finishTimeMs >=
                            options.prestartDurationMs
                    ? *state->finishTimeMs - options.prestartDurationMs
                    : 0u;
        }
        view.respawnCount = state->respawnCount;
        view.stuntsScore = state->stuntsScore;
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Success(view);
    }
};

struct PhysicsSandboxCudaSearchSession::Impl {
    std::unique_ptr<simulation::CudaSearchExecutor> executor;
    std::shared_ptr<const SandboxInputStorage> inputs;
    std::vector<PhysicsSandboxInputEvent> lateInputs;
    std::shared_ptr<const SandboxControlPlanStorage> controlPlan;
    std::uint64_t scenarioFingerprint = 0u;
    std::uint32_t validationSeed = 0u;
    std::uint32_t tickDurationMs = 0u;
    std::uint32_t prestartDurationMs = 0u;
    std::uint64_t durationMs = 0u;
    PhysicsSandboxTimelineMode timelineMode =
            PhysicsSandboxTimelineMode::RecordedReplay;
    std::size_t prestartTicks = 0u;
    MapEnvironment mapEnvironment = MapEnvironment::Unknown;
    VehicleModel vehicleModel = VehicleModel::Unknown;
    std::optional<PlayMode> playMode;

    PhysicsSandboxResult<PhysicsSandboxCudaSearchBatch> Convert(
            simulation::CudaSearchBatchExecution execution);
};

namespace {

simulation::CudaSearchWindow CudaWindow(
        const PhysicsSandboxCudaModifierWindow &source) {
    return {source.minimumTimeMs,
            source.maximumTimeMs,
            source.seed};
}

simulation::CudaSearchModifierConfiguration CudaModifier(
        const PhysicsSandboxCudaModifier &source) {
    return std::visit(
            [](const auto &modifier) {
                using T = std::decay_t<decltype(modifier)>;
                simulation::CudaSearchModifierConfiguration result;
                result.window = CudaWindow(modifier.window);
                if constexpr (std::is_same_v<
                                      T,
                                      PhysicsSandboxCudaRandomSteeringModifier>) {
                    result.kind = simulation::CudaSearchModifierKind::
                            RandomSteering;
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaExistingEventModifier>) {
                    result.kind = simulation::CudaSearchModifierKind::
                            ExistingEvent;
                    result.minimumCount = modifier.minimumCount;
                    result.maximumCount = modifier.maximumCount;
                    result.timeParameterMs = modifier.maximumTimeShiftMs;
                    result.analogMinimum =
                            modifier.steeringDeltaMinimum;
                    result.analogMaximum =
                            modifier.steeringDeltaMaximum;
                    result.secondaryAnalogMinimum =
                            modifier.steeringAbsoluteMinimum;
                    result.secondaryAnalogMaximum =
                            modifier.steeringAbsoluteMaximum;
                    result.optionFlags =
                            (modifier.absoluteSteering ? 1u : 0u) |
                            (modifier.toggleAccelerate ? 2u : 0u) |
                            (modifier.toggleBrake ? 4u : 0u);
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaSmoothSteeringModifier>) {
                    result.kind = simulation::CudaSearchModifierKind::
                            SmoothSteering;
                    result.minimumCount = modifier.deformationCount;
                    result.timeParameterMs = modifier.radiusMs;
                    result.analogMinimum = modifier.amplitudeMinimum;
                    result.analogMaximum = modifier.amplitudeMaximum;
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaInputInsertionModifier>) {
                    result.kind = simulation::CudaSearchModifierKind::
                            InputInsertion;
                    const auto channel = [](const auto &value) {
                        return simulation::CudaSearchChannel{
                                value.enabled ? 1u : 0u,
                                value.minimumCount,
                                value.maximumCount,
                                value.maximumHoldMs};
                    };
                    result.steering = channel(modifier.steering);
                    result.accelerate = channel(modifier.accelerate);
                    result.brake = channel(modifier.brake);
                    result.analogMinimum =
                            modifier.steeringAbsoluteMinimum;
                    result.analogMaximum =
                            modifier.steeringAbsoluteMaximum;
                    result.secondaryAnalogMinimum =
                            modifier.steeringOffsetMinimum;
                    result.secondaryAnalogMaximum =
                            modifier.steeringOffsetMaximum;
                    result.optionFlags =
                            modifier.steeringOffset ? 1u : 0u;
                } else {
                    result.kind = simulation::CudaSearchModifierKind::
                            InputDeletion;
                    const auto channel = [](const auto &value) {
                        return simulation::CudaSearchChannel{
                                value.enabled ? 1u : 0u,
                                0u,
                                value.maximumCount,
                                0};
                    };
                    result.steering = channel(modifier.steering);
                    result.accelerate = channel(modifier.accelerate);
                    result.brake = channel(modifier.brake);
                }
                return result;
            },
            source);
}

simulation::CudaSearchEvaluatorConfiguration CudaEvaluator(
        const PhysicsSandboxCudaEvaluator &source) {
    return std::visit(
            [](const auto &evaluator) {
                using T = std::decay_t<decltype(evaluator)>;
                simulation::CudaSearchEvaluatorConfiguration result;
                if constexpr (std::is_same_v<
                                      T,
                                      PhysicsSandboxCudaVelocityEvaluator>) {
                    result.kind =
                            simulation::CudaSearchEvaluatorKind::Velocity;
                    result.optionFlags =
                            (evaluator.projected ? 1u : 0u) |
                            (evaluator.alignmentEnabled ? 2u : 0u);
                    result.values[0] = evaluator.direction.x;
                    result.values[1] = evaluator.direction.y;
                    result.values[2] = evaluator.direction.z;
                    result.values[3] = evaluator.minimumAlignment;
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaPointEvaluator>) {
                    result.kind =
                            simulation::CudaSearchEvaluatorKind::Point;
                    result.values[0] = evaluator.target.x;
                    result.values[1] = evaluator.target.y;
                    result.values[2] = evaluator.target.z;
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaPoseEvaluator>) {
                    result.kind =
                            simulation::CudaSearchEvaluatorKind::Pose;
                    result.values[0] = evaluator.targetPosition.x;
                    result.values[1] = evaluator.targetPosition.y;
                    result.values[2] = evaluator.targetPosition.z;
                    result.values[3] = evaluator.targetRotationX;
                    result.values[4] = evaluator.targetRotationY;
                    result.values[5] = evaluator.targetRotationZ;
                    result.values[6] = evaluator.targetRotationW;
                    result.values[7] = evaluator.rotationWeight;
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaVolumeEntryEvaluator>) {
                    result.kind =
                            simulation::CudaSearchEvaluatorKind::VolumeEntry;
                    result.values[0] = evaluator.minimum.x;
                    result.values[1] = evaluator.minimum.y;
                    result.values[2] = evaluator.minimum.z;
                    result.values[3] = evaluator.maximum.x;
                    result.values[4] = evaluator.maximum.y;
                    result.values[5] = evaluator.maximum.z;
                } else if constexpr (std::is_same_v<
                                             T,
                                             PhysicsSandboxCudaStuntPointsEvaluator>) {
                    result.kind =
                            simulation::CudaSearchEvaluatorKind::StuntPoints;
                } else {
                    result.kind =
                            simulation::CudaSearchEvaluatorKind::FinishTime;
                }
                return result;
            },
            source);
}

simulation::CudaSearchInputEvent CudaInput(
        const PhysicsSandboxInputEvent &source) {
    simulation::CudaSearchInputEvent result;
    result.timeMs = source.timeMs;
    result.action = static_cast<std::uint32_t>(source.action);
    result.valueKind = static_cast<std::uint32_t>(source.value.kind);
    if (source.value.kind == PhysicsSandboxInputValueKind::Analog) {
        result.value = source.value.analog;
    } else {
        result.value =
                static_cast<std::int32_t>(source.value.switchState);
    }
    return result;
}

PhysicsSandboxInputEvent PublicInput(
        const simulation::CudaSearchInputEvent &source) {
    PhysicsSandboxInputEvent result;
    result.timeMs = source.timeMs;
    result.action =
            static_cast<PhysicsSandboxInputAction>(source.action);
    result.value.kind =
            static_cast<PhysicsSandboxInputValueKind>(source.valueKind);
    if (result.value.kind == PhysicsSandboxInputValueKind::Analog) {
        result.value.analog = source.value;
    } else {
        result.value.switchState =
                static_cast<PhysicsSandboxSwitchState>(source.value);
    }
    return result;
}

bool AddEventCapacity(std::size_t amount, std::size_t *capacity) {
    constexpr std::size_t MaximumSearchEvents = 1024u * 1024u;
    if (amount > MaximumSearchEvents - *capacity) {
        return false;
    }
    *capacity += amount;
    return true;
}

bool MaximumEventCapacity(
        std::size_t baselineCount,
        const std::vector<PhysicsSandboxCudaModifier> &modifiers,
        std::uint32_t tickDurationMs,
        std::size_t *capacity) {
    *capacity = baselineCount;
    for (const PhysicsSandboxCudaModifier &modifier : modifiers) {
        bool valid = std::visit(
                [&](const auto &value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<
                                          T,
                                          PhysicsSandboxCudaSmoothSteeringModifier>) {
                        if (value.radiusMs < 0 ||
                            value.radiusMs %
                                            static_cast<std::int64_t>(
                                                    tickDurationMs) !=
                                    0) {
                            return false;
                        }
                        const std::uint64_t perDeformation =
                                static_cast<std::uint64_t>(
                                        value.radiusMs /
                                        tickDurationMs) *
                                        2u +
                                2u;
                        if (value.deformationCount != 0u &&
                            perDeformation >
                                    std::numeric_limits<std::size_t>::max() /
                                            value.deformationCount) {
                            return false;
                        }
                        return AddEventCapacity(
                                static_cast<std::size_t>(
                                        perDeformation *
                                        value.deformationCount),
                                capacity);
                    } else if constexpr (std::is_same_v<
                                                 T,
                                                 PhysicsSandboxCudaInputInsertionModifier>) {
                        const std::uint64_t operations =
                                (value.steering.enabled
                                         ? value.steering.maximumCount
                                         : 0u) +
                                (value.accelerate.enabled
                                         ? value.accelerate.maximumCount
                                         : 0u) +
                                (value.brake.enabled
                                         ? value.brake.maximumCount
                                         : 0u);
                        if (operations >
                            std::numeric_limits<std::size_t>::max() / 2u) {
                            return false;
                        }
                        return AddEventCapacity(
                                static_cast<std::size_t>(operations * 2u),
                                capacity);
                    } else {
                        return true;
                    }
                },
                modifier);
        if (!valid) {
            return false;
        }
    }
    return true;
}

PhysicsSandboxError SearchError(
        PhysicsSandboxErrorCode code,
        const std::string &diagnostic) {
    PhysicsSandboxError result;
    result.code = code;
    result.diagnostic = diagnostic;
    return result;
}

}  // namespace

PhysicsSandboxState::PhysicsSandboxState(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}
PhysicsSandboxState::PhysicsSandboxState(const PhysicsSandboxState &) = default;
PhysicsSandboxState &PhysicsSandboxState::operator=(
        const PhysicsSandboxState &) = default;
PhysicsSandboxState::PhysicsSandboxState(PhysicsSandboxState &&) noexcept =
        default;
PhysicsSandboxState &PhysicsSandboxState::operator=(
        PhysicsSandboxState &&) noexcept = default;
PhysicsSandboxState::~PhysicsSandboxState() = default;

const PhysicsSandboxStateView &PhysicsSandboxState::View() const noexcept {
    static const PhysicsSandboxStateView empty;
    return impl_ ? impl_->view : empty;
}

PhysicsSandbox::PhysicsSandbox(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PhysicsSandbox::PhysicsSandbox(PhysicsSandbox &&) noexcept = default;
PhysicsSandbox &PhysicsSandbox::operator=(PhysicsSandbox &&) noexcept = default;
PhysicsSandbox::~PhysicsSandbox() = default;

SimulationBackend PhysicsSandbox::Backend() const noexcept {
    return impl_ ? impl_->options.backend : SimulationBackend::Reference;
}

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::LoadReplay(
        ByteView replayBytes,
        const ReplayIdentity &identity) noexcept {
    return LoadScenarioFile(replayBytes, identity, false);
}

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::LoadScenario(
        ByteView scenarioBytes,
        const ReplayIdentity &identity) noexcept {
    return LoadScenarioFile(scenarioBytes, identity, true);
}

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::LoadScenarioFile(
        ByteView replayBytes,
        const ReplayIdentity &identity,
        bool acceptStandaloneChallenge) noexcept {
    try {
        if (!impl_ || !replayBytes.IsValid() || replayBytes.size == 0u ||
            identity.name.empty()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "invalid sandbox scenario request"));
        }
        ReplayFile replay;
        bool standaloneChallenge = false;
        ReplayFileReadError readError = ReadReplayBytes(
                reinterpret_cast<const std::uint8_t *>(replayBytes.data),
                replayBytes.size,
                &replay);
        if (readError == ReplayFileReadError::InvalidContainer &&
            acceptStandaloneChallenge) {
            readError = ReadChallengeBytes(
                    reinterpret_cast<const std::uint8_t *>(
                            replayBytes.data),
                    replayBytes.size,
                    &replay);
            standaloneChallenge =
                    readError == ReplayFileReadError::Success;
        }
        if (readError != ReplayFileReadError::Success) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::ReplayLoadingFailed,
                            "sandbox scenario could not be decoded",
                            ReplayDecodeError(readError, identity)));
        }
        if (standaloneChallenge &&
            impl_->options.timelineMode !=
                    PhysicsSandboxTimelineMode::Canonical) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "standalone challenges require the canonical "
                            "timeline mode"));
        }
        if (impl_->options.timelineMode ==
                    PhysicsSandboxTimelineMode::RecordedReplay &&
            !replay.HasValidationInput()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::ReplayLoadingFailed,
                                 "sandbox replay has no playable input"));
        }
        ReplayAssetRoute route;
        const ReplayAssetRouteResult routeResult =
                BuildReplayAssetRoute(replay, &route);
        if (routeResult != ReplayAssetRouteResult::Success) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::ReplayLoadingFailed,
                            "sandbox scenario route is unsupported",
                            ReplayRouteError(routeResult, identity, replay)));
        }
        if (impl_->options.backend == SimulationBackend::Cuda &&
            !IsCudaSupportedRoute(route)) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::SimulationFailed,
                            "CUDA simulation scope is unsupported",
                            CudaScopeError(route, identity)));
        }
        Result<PreparedAssets> prepared = PrepareAssets(
                impl_->validationState, route, identity);
        if (!prepared) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::MapLoadingFailed,
                            "sandbox assets could not be prepared",
                            std::move(prepared).Error()));
        }

        auto session = std::make_unique<ReplaySimulationSession>(
                impl_->options.backend);
        CGameCtnReplayChallengeMapPreload preload;
        const ReplayChallengePreloadResult preloadResult = preload.Preload(
                replay.MapInput(),
                *prepared.Value().mapAssets,
                *prepared.Value().decorationAssets,
                *session);
        if (preloadResult != ReplayChallengePreloadResult::Success) {
            ValidationError error = PreloadError(preloadResult, identity);
            if (impl_->options.backend == SimulationBackend::Cuda &&
                !session->CudaInitializationDiagnostic().empty()) {
                error.diagnostic =
                        session->CudaInitializationDiagnostic();
            }
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::MapLoadingFailed,
                            "sandbox map could not be loaded",
                            std::move(error)));
        }
        ReplaySimulationDefinitionBuild definition =
                BuildReplaySimulationDefinition(
                        *prepared.Value().vehicleSources,
                        preload.WaterDefinition());
        if (!definition) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::MapLoadingFailed,
                            "sandbox vehicle definition could not be built",
                            DefinitionError(definition.Error(), identity)));
        }
        definition.Value().optimizedCpuStadiumSpecializationsEnabled =
                route.vehicleModel == ::ReplayVehicleModel::StadiumCar;
        session->ActivateStaticScene();

        std::vector<PhysicsSandboxInputEvent> inputs;
        if (impl_->options.timelineMode ==
            PhysicsSandboxTimelineMode::Canonical) {
            inputs.push_back({
                    0,
                    PhysicsSandboxInputAction::RaceRunning,
                    {PhysicsSandboxInputValueKind::Switch,
                     PhysicsSandboxSwitchState::Pressed,
                     0}});
        } else {
            inputs.reserve(replay.InputTimeline().Events().size());
            for (const ReplayInputEvent &event :
                 replay.InputTimeline().Events()) {
                const std::int64_t relative =
                        static_cast<std::int64_t>(event.timeMs) -
                        SandboxInputTimeBaseMs;
                if (relative < std::numeric_limits<std::int32_t>::min() ||
                    relative > std::numeric_limits<std::int32_t>::max()) {
                    return PhysicsSandboxResult<
                            PhysicsSandboxStateView>::Failure(
                            SandboxError(
                                    PhysicsSandboxErrorCode::ReplayLoadingFailed,
                                    "sandbox replay input time is out of range"));
                }
                inputs.push_back({
                        static_cast<std::int32_t>(relative),
                        ToSandboxAction(event.action),
                        ToSandboxValue(event.value)});
            }
        }

        PhysicsSandboxSceneView scene = BuildSandboxScene(
                *session, definition.Value());
        PhysicsSandboxRenderSceneHandle renderScene =
                session->StaticRenderScene();
        if (!renderScene) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::MapLoadingFailed,
                            "sandbox visual scene could not be built"));
        }
        impl_->session = std::move(session);
        impl_->definition = std::move(definition).Value();
        impl_->inputMetadata = replay.InputTimeline().Metadata();
        if (impl_->options.timelineMode ==
            PhysicsSandboxTimelineMode::Canonical) {
            impl_->simulationHorizonMs =
                    *impl_->options.simulationHorizonMs;
            impl_->inputMetadata.durationMs = impl_->simulationHorizonMs;
            impl_->inputMetadata.raceTimeMs.reset();
            impl_->inputMetadata.respawnCount.reset();
            impl_->inputMetadata.stuntScore.reset();
            impl_->definedActions = {
                    ReplayInputActionKind::Accelerate,
                    ReplayInputActionKind::Gas,
                    ReplayInputActionKind::Brake,
                    ReplayInputActionKind::Steer,
                    ReplayInputActionKind::SteerLeft,
                    ReplayInputActionKind::SteerRight,
                    ReplayInputActionKind::RaceRunning,
                    ReplayInputActionKind::Respawn};
            impl_->provenance = ReplayInputProvenance::Scripted;
        } else {
            impl_->simulationHorizonMs = impl_->inputMetadata.durationMs;
            impl_->definedActions = replay.InputTimeline().DefinedActions();
            impl_->provenance = replay.InputTimeline().Provenance();
        }
        impl_->challengeMetadata = replay.ChallengeMetadata();
        impl_->route = route;
        impl_->identity = identity;
        impl_->scenarioFingerprint = Fingerprint(replayBytes);
        impl_->scene = std::move(scene);
        impl_->renderScene = std::move(renderScene);
        impl_->inputs = SandboxInputStorage::Full(std::move(inputs));
        impl_->prestartTicks =
                impl_->options.prestartDurationMs /
                impl_->options.tickDurationMs;
        PhysicsSandboxResult<ReplayControlPlan> plan =
                impl_->BuildControlPlan(*impl_->inputs->base);
        if (!plan) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    std::move(plan).Error());
        }
        impl_->controlPlan = SandboxControlPlanStorage::Full(
                std::move(plan).Value());
        impl_->loaded = true;
        PhysicsSandboxResult<PhysicsSandboxStateView> restarted =
                impl_->Restart(0u);
        if (restarted &&
            impl_->options.backend == SimulationBackend::Cuda &&
            impl_->options.prepareCudaSearchSpecialization) {
            static_cast<void>(
                    impl_->session->PrepareCudaSearchSpecialization(
                            nullptr));
        }
        return restarted;
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while loading sandbox"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox replay loading failure"));
    }
}

PhysicsSandboxResult<std::string> PhysicsSandbox::ReadMapName()
        const noexcept {
    try {
        if (!impl_ || !impl_->loaded) {
            return PhysicsSandboxResult<std::string>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no loaded map name"));
        }
        return PhysicsSandboxResult<std::string>::Success(
                impl_->challengeMetadata.mapName);
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<std::string>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "could not copy sandbox map name"));
    } catch (...) {
        return PhysicsSandboxResult<std::string>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox map name read failure"));
    }
}

PhysicsSandboxResult<std::vector<PhysicsSandboxInputEvent>>
PhysicsSandbox::ReadInputs() const noexcept {
    try {
        if (!impl_ || !impl_->loaded) {
            return PhysicsSandboxResult<
                    std::vector<PhysicsSandboxInputEvent>>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no loaded inputs"));
        }
        return PhysicsSandboxResult<
                std::vector<PhysicsSandboxInputEvent>>::Success(
                        impl_->inputs->Materialize());
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<
                std::vector<PhysicsSandboxInputEvent>>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "could not copy sandbox inputs"));
    } catch (...) {
        return PhysicsSandboxResult<
                std::vector<PhysicsSandboxInputEvent>>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox input read failure"));
    }
}

PhysicsSandboxResult<std::size_t> PhysicsSandbox::ReplaceInputs(
        std::vector<PhysicsSandboxInputEvent> events) noexcept {
    try {
        if (!impl_ || !impl_->loaded) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no loaded scenario"));
        }
        const std::int64_t currentTime = static_cast<std::int64_t>(
                impl_->cursor - impl_->prestartTicks) *
                impl_->options.tickDurationMs;
        const std::vector<PhysicsSandboxInputEvent> currentInputs =
                impl_->inputs->Materialize();
        std::vector<PhysicsSandboxInputEvent> oldPast;
        std::vector<PhysicsSandboxInputEvent> newPast;
        for (const PhysicsSandboxInputEvent &event : currentInputs) {
            if (event.timeMs < currentTime) oldPast.push_back(event);
        }
        for (const PhysicsSandboxInputEvent &event : events) {
            if (event.timeMs < currentTime) newPast.push_back(event);
        }
        if (oldPast.size() != newPast.size() ||
            !std::equal(oldPast.begin(), oldPast.end(), newPast.begin(),
                        SameInputEvent)) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "past sandbox inputs are immutable"));
        }
        PhysicsSandboxResult<ReplayControlPlan> plan =
                impl_->BuildControlPlan(events);
        if (!plan) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    std::move(plan).Error());
        }
        if (impl_->cursor > plan.Value().ticks.size()) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "input replacement ends before current tick"));
        }
        impl_->inputs = SandboxInputStorage::Full(std::move(events));
        impl_->controlPlan = SandboxControlPlanStorage::Full(
                std::move(plan).Value());
        return PhysicsSandboxResult<std::size_t>::Success(
                impl_->inputs->Size());
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<std::size_t>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while replacing inputs"));
    } catch (...) {
        return PhysicsSandboxResult<std::size_t>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox input replacement failure"));
    }
}

PhysicsSandboxResult<std::size_t> PhysicsSandbox::ReplaceInputWindow(
        std::int64_t minimumTimeMs,
        std::int64_t maximumTimeMs,
        std::vector<PhysicsSandboxInputEvent> events) noexcept {
    try {
        if (!impl_ || !impl_->loaded || !impl_->inputs ||
            !impl_->controlPlan || minimumTimeMs < 0 ||
            maximumTimeMs < minimumTimeMs ||
            minimumTimeMs % impl_->options.tickDurationMs != 0 ||
            maximumTimeMs % impl_->options.tickDurationMs != 0 ||
            maximumTimeMs >
                    static_cast<std::int64_t>(
                            impl_->simulationHorizonMs)) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "invalid sandbox input window"));
        }
        const std::int64_t currentTimeMs =
                static_cast<std::int64_t>(
                        impl_->cursor - impl_->prestartTicks) *
                impl_->options.tickDurationMs;
        if (minimumTimeMs < currentTimeMs) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "past sandbox inputs are immutable"));
        }

        std::int32_t previousTimeMs = std::numeric_limits<std::int32_t>::min();
        for (const PhysicsSandboxInputEvent &event : events) {
            if (event.timeMs < minimumTimeMs ||
                event.timeMs > maximumTimeMs) {
                return PhysicsSandboxResult<std::size_t>::Failure(
                        SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                     "sandbox input window contains an event outside its bounds"));
            }
            if (event.timeMs < previousTimeMs) {
                return PhysicsSandboxResult<std::size_t>::Failure(
                        SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                     "sandbox input window is not time ordered"));
            }
            if (event.value.kind == PhysicsSandboxInputValueKind::Analog &&
                !IsAnalogInputStateValid(event.value.analog)) {
                return PhysicsSandboxResult<std::size_t>::Failure(
                        SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                     "sandbox input window contains an invalid analog value"));
            }
            previousTimeMs = event.timeMs;
        }

        std::shared_ptr<const std::vector<PhysicsSandboxInputEvent>> baseInputs;
        if (impl_->inputs->window) {
            baseInputs = std::make_shared<const std::vector<
                    PhysicsSandboxInputEvent>>(
                    impl_->inputs->Materialize());
        } else {
            baseInputs = impl_->inputs->base;
        }
        if (!baseInputs) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox inputs are unavailable"));
        }

        bool affectsSteer = false;
        bool affectsAccelerate = false;
        bool affectsBrake = false;
        const auto markAffected = [&](PhysicsSandboxInputAction action) {
            affectsSteer |= action == PhysicsSandboxInputAction::Steer;
            affectsAccelerate |=
                    action == PhysicsSandboxInputAction::Accelerate ||
                    action == PhysicsSandboxInputAction::Gas;
            affectsBrake |= action == PhysicsSandboxInputAction::Brake;
        };
        const auto baselineFirst = std::lower_bound(
                baseInputs->begin(), baseInputs->end(), minimumTimeMs,
                [](const PhysicsSandboxInputEvent &event,
                   std::int64_t timeMs) {
                    return event.timeMs < timeMs;
                });
        const auto baselineLast = std::upper_bound(
                baselineFirst, baseInputs->end(), maximumTimeMs,
                [](std::int64_t timeMs,
                   const PhysicsSandboxInputEvent &event) {
                    return timeMs < event.timeMs;
                });
        for (auto it = baselineFirst; it != baselineLast; ++it) {
            markAffected(it->action);
        }
        for (const PhysicsSandboxInputEvent &event : events) {
            markAffected(event.action);
        }

        std::int64_t replacementEndTimeMs = maximumTimeMs;
        const auto extendToReset = [&](PhysicsSandboxInputAction action,
                                       bool affected) {
            if (!affected || replacementEndTimeMs >=
                                     static_cast<std::int64_t>(
                                             impl_->simulationHorizonMs)) {
                return;
            }
            const auto next = std::find_if(
                    baselineLast, baseInputs->end(),
                    [action](const PhysicsSandboxInputEvent &event) {
                        if (action == PhysicsSandboxInputAction::Accelerate) {
                            return event.action ==
                                           PhysicsSandboxInputAction::Accelerate ||
                                    event.action ==
                                           PhysicsSandboxInputAction::Gas;
                        }
                        return event.action == action;
                    });
            replacementEndTimeMs = std::max(
                    replacementEndTimeMs,
                    next == baseInputs->end()
                            ? static_cast<std::int64_t>(
                                      impl_->simulationHorizonMs)
                            : std::min<std::int64_t>(
                                      next->timeMs,
                                      impl_->simulationHorizonMs));
        };
        extendToReset(PhysicsSandboxInputAction::Steer, affectsSteer);
        extendToReset(
                PhysicsSandboxInputAction::Accelerate, affectsAccelerate);
        extendToReset(PhysicsSandboxInputAction::Brake, affectsBrake);
        replacementEndTimeMs =
                (replacementEndTimeMs /
                 impl_->options.tickDurationMs) *
                impl_->options.tickDurationMs;

        auto inputStorage = std::make_shared<SandboxInputStorage>();
        inputStorage->base = std::move(baseInputs);
        inputStorage->window = SandboxInputWindow{
                minimumTimeMs, maximumTimeMs, std::move(events)};
        const std::vector<PhysicsSandboxInputEvent> controlInputs =
                inputStorage->MaterializeThrough(replacementEndTimeMs);
        PhysicsSandboxResult<ReplayControlPlan> partial =
                impl_->BuildControlPlan(
                        controlInputs, replacementEndTimeMs);
        if (!partial) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    std::move(partial).Error());
        }

        std::shared_ptr<const ReplayControlPlan> basePlan;
        if (impl_->controlPlan->replacement.empty()) {
            basePlan = impl_->controlPlan->base;
        } else {
            basePlan = std::make_shared<const ReplayControlPlan>(
                    impl_->controlPlan->Materialize());
        }
        ReplayControlPlan partialPlan = std::move(partial).Value();
        const std::uint64_t firstClockTimeMs =
                static_cast<std::uint64_t>(
                        impl_->options.prestartDurationMs) +
                static_cast<std::uint64_t>(minimumTimeMs);
        const auto firstTick = std::lower_bound(
                partialPlan.ticks.begin(), partialPlan.ticks.end(),
                firstClockTimeMs,
                [](const ReplayControlTick &tick, std::uint64_t timeMs) {
                    return tick.timeMs < timeMs;
                });
        const std::size_t replacementBegin =
                static_cast<std::size_t>(
                        firstTick - partialPlan.ticks.begin());
        if (!basePlan || partialPlan.ticks.size() > basePlan->ticks.size() ||
            replacementBegin > partialPlan.ticks.size()) {
            return PhysicsSandboxResult<std::size_t>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "sandbox input window exceeds the control plan"));
        }

        auto controlStorage =
                std::make_shared<SandboxControlPlanStorage>();
        controlStorage->base = std::move(basePlan);
        controlStorage->replacementBegin = replacementBegin;
        controlStorage->replacement.assign(
                firstTick, partialPlan.ticks.end());
        impl_->inputs = std::move(inputStorage);
        impl_->controlPlan = std::move(controlStorage);
        return PhysicsSandboxResult<std::size_t>::Success(
                impl_->inputs->Size());
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<std::size_t>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while replacing input window"));
    } catch (...) {
        return PhysicsSandboxResult<std::size_t>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox input window failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::AdvanceTicks(
        std::uint32_t count) noexcept {
    try {
        if (!impl_ || !impl_->loaded || count == 0u ||
            !impl_->controlPlan ||
            impl_->cursor > impl_->controlPlan->Size() ||
            count > impl_->controlPlan->Size() - impl_->cursor) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "invalid sandbox tick advance"));
        }
        tmnf::simulation::DeterministicExecutionScope deterministicScope;
        if (!deterministicScope.Established()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::SimulationFailed,
                                 "deterministic execution mode is unavailable"));
        }
        const ReplaySimulationTimelineResult result =
                impl_->AdvanceControlPlan(impl_->cursor, count);
        if (result.result != ReplaySimulationRunResult::Success ||
            !deterministicScope.Restore()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::SimulationFailed,
                                 "sandbox tick advance failed"));
        }
        impl_->cursor += count;
        return impl_->ReadView();
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed during sandbox advance"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox advance failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxStateView>
PhysicsSandbox::SetSimulationHorizonMs(
        std::uint32_t simulationHorizonMs) noexcept {
    try {
        constexpr std::uint32_t maximumHorizonMs =
                static_cast<std::uint32_t>(
                        std::numeric_limits<std::int32_t>::max());
        const std::uint32_t maximumCanonicalHorizonMs =
                impl_ && impl_->options.prestartDurationMs <= maximumHorizonMs
                ? maximumHorizonMs - impl_->options.prestartDurationMs
                : 0u;
        if (!impl_ || !impl_->loaded ||
            impl_->options.timelineMode !=
                    PhysicsSandboxTimelineMode::Canonical ||
            simulationHorizonMs < impl_->options.tickDurationMs ||
            simulationHorizonMs > maximumCanonicalHorizonMs ||
            simulationHorizonMs % impl_->options.tickDurationMs != 0u ||
            impl_->cursor > impl_->prestartTicks +
                    simulationHorizonMs / impl_->options.tickDurationMs) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "invalid canonical simulation horizon"));
        }
        PhysicsSandboxResult<ReplayControlPlan> rebuilt =
                impl_->BuildControlPlan(
                        impl_->inputs->Materialize(),
                        simulationHorizonMs);
        if (!rebuilt) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    std::move(rebuilt).Error());
        }
        impl_->simulationHorizonMs = simulationHorizonMs;
        impl_->options.simulationHorizonMs = simulationHorizonMs;
        impl_->inputMetadata.durationMs = simulationHorizonMs;
        impl_->controlPlan = SandboxControlPlanStorage::Full(
                std::move(rebuilt).Value());
        return impl_->ReadView();
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "could not resize canonical simulation horizon"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected canonical horizon resize failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxState> PhysicsSandbox::CaptureState()
        const noexcept {
    try {
        if (!impl_ || !impl_->loaded) {
            return PhysicsSandboxResult<PhysicsSandboxState>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no state to capture"));
        }
        PhysicsSandboxResult<PhysicsSandboxStateView> view = impl_->ReadView();
        if (!view) {
            return PhysicsSandboxResult<PhysicsSandboxState>::Failure(
                    std::move(view).Error());
        }
        auto state = std::make_shared<PhysicsSandboxState::Impl>();
        state->runtimeClone = impl_->session->CaptureRuntimeClone();
        if (!state->runtimeClone) {
            return PhysicsSandboxResult<PhysicsSandboxState>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox runtime is not at a capture boundary"));
        }
        state->view = view.Value();
        state->inputs = impl_->inputs;
        state->controlPlan = impl_->controlPlan;
        state->scenarioFingerprint = impl_->scenarioFingerprint;
        state->validationSeed = impl_->inputMetadata.validationSeed;
        state->backend = impl_->options.backend;
        state->tickDurationMs = impl_->options.tickDurationMs;
        state->prestartDurationMs = impl_->options.prestartDurationMs;
        state->simulationHorizonMs = impl_->simulationHorizonMs;
        state->timelineMode = impl_->options.timelineMode;
        state->cursor = impl_->cursor;
        state->runtimeCloneSchema = SandboxRuntimeCloneSchema;
        return PhysicsSandboxResult<PhysicsSandboxState>::Success(
                PhysicsSandboxState(std::move(state)));
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxState>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while capturing state"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxState>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox state capture failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::RestoreState(
        const PhysicsSandboxState &state) noexcept {
    try {
        if (!impl_ || !impl_->loaded || !state.impl_ ||
            state.impl_->runtimeCloneSchema != SandboxRuntimeCloneSchema ||
            state.impl_->scenarioFingerprint != impl_->scenarioFingerprint ||
            state.impl_->validationSeed != impl_->inputMetadata.validationSeed ||
            state.impl_->backend != impl_->options.backend ||
            state.impl_->tickDurationMs != impl_->options.tickDurationMs ||
            state.impl_->prestartDurationMs !=
                    impl_->options.prestartDurationMs ||
            (state.impl_->simulationHorizonMs !=
                     impl_->simulationHorizonMs &&
             impl_->options.timelineMode !=
                     PhysicsSandboxTimelineMode::Canonical) ||
            state.impl_->timelineMode != impl_->options.timelineMode) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state is incompatible"));
        }
        if (!state.impl_->runtimeClone) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state has no runtime clone"));
        }
        if (!state.impl_->inputs ||
            state.impl_->cursor < impl_->prestartTicks) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state cursor is incompatible"));
        }
        std::shared_ptr<const SandboxControlPlanStorage> restoredControlPlan =
                state.impl_->simulationHorizonMs ==
                                impl_->simulationHorizonMs
                ? state.impl_->controlPlan
                : nullptr;
        if (!restoredControlPlan) {
            PhysicsSandboxResult<ReplayControlPlan> rebuilt =
                    impl_->BuildControlPlan(
                            state.impl_->inputs->Materialize());
            if (!rebuilt) {
                return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                        std::move(rebuilt).Error());
            }
            restoredControlPlan = SandboxControlPlanStorage::Full(
                    std::move(rebuilt).Value());
        }
        if (state.impl_->cursor > restoredControlPlan->Size()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state cursor is incompatible"));
        }
        ReplaySimulationInstanceClone runtimeClone =
                *state.impl_->runtimeClone;
        if (!impl_->session->PrepareRuntimeCloneRestore(runtimeClone)) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                                 "sandbox runtime clone could not be prepared"));
        }

        impl_->session->RestoreRuntimeClone(std::move(runtimeClone));
        impl_->inputs = state.impl_->inputs;
        impl_->controlPlan = std::move(restoredControlPlan);
        impl_->cursor = state.impl_->cursor;
        return impl_->ReadView();
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while restoring state"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox state restore failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::ReadState()
        const noexcept {
    try {
        if (!impl_) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox is moved-from"));
        }
        return impl_->ReadView();
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox state read failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxSceneView> PhysicsSandbox::ReadScene()
        const noexcept {
    try {
        if (!impl_ || !impl_->loaded) {
            return PhysicsSandboxResult<PhysicsSandboxSceneView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no loaded scene"));
        }
        return PhysicsSandboxResult<PhysicsSandboxSceneView>::Success(
                impl_->scene);
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxSceneView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "could not copy sandbox scene"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxSceneView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox scene read failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxRenderSceneHandle>
PhysicsSandbox::ReadRenderScene() const noexcept {
    try {
        if (!impl_ || !impl_->loaded || !impl_->renderScene) {
            return PhysicsSandboxResult<
                    PhysicsSandboxRenderSceneHandle>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox has no loaded render scene"));
        }
        return PhysicsSandboxResult<
                PhysicsSandboxRenderSceneHandle>::Success(
                impl_->renderScene);
    } catch (...) {
        return PhysicsSandboxResult<
                PhysicsSandboxRenderSceneHandle>::Failure(
                SandboxError(
                        PhysicsSandboxErrorCode::UnexpectedFailure,
                        "unexpected sandbox render scene read failure"));
    }
}

PhysicsSandboxResult<PhysicsSandbox> CreatePhysicsSandbox(
        AssetSource source,
        const PhysicsSandboxOptions &options) noexcept {
    try {
        AssetProvider provider =
                detail::PhysicsSandboxAssetSourceAccess::Take(source);
        const bool canonical = options.timelineMode ==
                PhysicsSandboxTimelineMode::Canonical;
        const std::uint32_t signedTimeLimit =
                static_cast<std::uint32_t>(
                        std::numeric_limits<std::int32_t>::max());
        const std::uint32_t maximumHorizon =
                options.prestartDurationMs <= signedTimeLimit
                ? signedTimeLimit - options.prestartDurationMs
                : 0u;
        if (!provider || options.tickDurationMs == 0u ||
            options.prestartDurationMs == 0u ||
            options.prestartDurationMs % options.tickDurationMs != 0u ||
            options.prestartDurationMs > signedTimeLimit ||
            (canonical &&
             (!options.simulationHorizonMs.has_value() ||
              *options.simulationHorizonMs < options.tickDurationMs ||
              *options.simulationHorizonMs % options.tickDurationMs != 0u ||
              *options.simulationHorizonMs > maximumHorizon)) ||
            (!canonical && options.simulationHorizonMs.has_value()) ||
            !simulation::IsSimulationBackendSupported(options.backend)) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "invalid sandbox creation request"));
        }
        return PhysicsSandboxResult<PhysicsSandbox>::Success(PhysicsSandbox(
                std::make_unique<PhysicsSandbox::Impl>(
                        std::move(provider), options)));
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while creating sandbox"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox creation failure"));
    }
}

PhysicsSandboxResult<PhysicsSandbox> ClonePhysicsSandbox(
        const PhysicsSandbox &source) noexcept {
    try {
        if (!source.impl_ || !source.impl_->loaded ||
            !source.impl_->session) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "source sandbox is not loaded"));
        }
        if (source.impl_->options.backend !=
                SimulationBackend::OptimizedCpu) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "only optimized CPU sandboxes can be cloned"));
        }
        std::unique_ptr<ReplaySimulationSession> session =
                source.impl_->session->ClonePrepared();
        if (!session) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                                 "prepared sandbox scene could not be cloned"));
        }

        auto impl = std::make_unique<PhysicsSandbox::Impl>(
                source.impl_->validationState.provider,
                source.impl_->options);
        impl->session = std::move(session);
        impl->definition = source.impl_->definition;
        impl->controlPlan = source.impl_->controlPlan;
        impl->inputMetadata = source.impl_->inputMetadata;
        impl->definedActions = source.impl_->definedActions;
        impl->provenance = source.impl_->provenance;
        impl->inputs = source.impl_->inputs;
        impl->simulationHorizonMs = source.impl_->simulationHorizonMs;
        impl->challengeMetadata = source.impl_->challengeMetadata;
        impl->route = source.impl_->route;
        impl->identity = source.impl_->identity;
        impl->scenarioFingerprint = source.impl_->scenarioFingerprint;
        impl->scene = source.impl_->scene;
        impl->renderScene = source.impl_->renderScene;
        impl->cursor = 0u;
        impl->prestartTicks = source.impl_->prestartTicks;
        impl->loaded = true;

        PhysicsSandbox clone(std::move(impl));
        PhysicsSandboxResult<PhysicsSandboxStateView> started =
                clone.impl_->Restart(0u);
        if (!started) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    std::move(started).Error());
        }
        PhysicsSandboxResult<PhysicsSandboxState> state =
                source.CaptureState();
        if (!state) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    std::move(state).Error());
        }
        PhysicsSandboxResult<PhysicsSandboxStateView> restored =
                clone.RestoreState(state.Value());
        if (!restored) {
            return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                    std::move(restored).Error());
        }
        return PhysicsSandboxResult<PhysicsSandbox>::Success(
                std::move(clone));
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while cloning sandbox"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandbox>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox clone failure"));
    }
}

PhysicsSandboxCudaSearchSession::PhysicsSandboxCudaSearchSession(
        std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PhysicsSandboxCudaSearchSession::~PhysicsSandboxCudaSearchSession() =
        default;
PhysicsSandboxCudaSearchSession::PhysicsSandboxCudaSearchSession(
        PhysicsSandboxCudaSearchSession &&) noexcept = default;
PhysicsSandboxCudaSearchSession &
PhysicsSandboxCudaSearchSession::operator=(
        PhysicsSandboxCudaSearchSession &&) noexcept = default;

PhysicsSandboxResult<PhysicsSandboxCudaSearchBatch>
PhysicsSandboxCudaSearchSession::Impl::Convert(
        simulation::CudaSearchBatchExecution execution) {
    if (execution.status != simulation::CudaSearchStatus::Success &&
        execution.status != simulation::CudaSearchStatus::Cancelled) {
        const PhysicsSandboxErrorCode code =
                execution.status ==
                                simulation::CudaSearchStatus::CapacityExceeded
                        ? PhysicsSandboxErrorCode::AllocationFailed
                        : execution.status ==
                                          simulation::CudaSearchStatus::
                                                  InvalidArgument ||
                                  execution.status ==
                                          simulation::CudaSearchStatus::
                                                  UnsupportedConfiguration
                        ? PhysicsSandboxErrorCode::InvalidRequest
                        : PhysicsSandboxErrorCode::SimulationFailed;
        std::string diagnostic = "CUDA search batch failed";
        if (!execution.diagnostic.empty()) {
            diagnostic += ": " + execution.diagnostic;
        }
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Failure(
                SearchError(code, diagnostic));
    }

    PhysicsSandboxCudaSearchBatch result;
    result.firstCandidateId = execution.firstCandidateId;
    result.candidateCount = execution.candidateCount;
    result.evaluatedCandidateCount =
            execution.evaluatedCandidateCount;
    result.evaluatorCalls = execution.evaluatorCalls;
    result.totalMutationCount = execution.totalMutationCount;
    result.mutationImprovementCount =
            execution.mutationImprovementCount;
    result.cancelled =
            execution.status == simulation::CudaSearchStatus::Cancelled;
    result.bestChanged = execution.bestChanged;
    result.bestValid = execution.best.valid;
    result.metrics.residentDeviceBytes =
            execution.residentDeviceBytes;
    result.metrics.mutationDeviceBytes =
            execution.mutationDeviceBytes;
    result.metrics.candidateInputDeviceBytes =
            execution.candidateInputDeviceBytes;
    result.metrics.mutationScratchDeviceBytes =
            execution.mutationScratchDeviceBytes;
    result.metrics.winnerSelectionDeviceBytes =
            execution.winnerSelectionDeviceBytes;
    result.metrics.hostToDeviceBytes = execution.hostToDeviceBytes;
    result.metrics.deviceToHostBytes = execution.deviceToHostBytes;
    result.metrics.kernelMilliseconds = execution.kernelMilliseconds;
    result.metrics.scoreInitializationKernelMilliseconds =
            execution.scoreInitializationKernelMilliseconds;
    result.metrics.mutationKernelMilliseconds =
            execution.mutationKernelMilliseconds;
    result.metrics.simulationKernelMilliseconds =
            execution.simulationKernelMilliseconds;
    result.metrics.finishRefinementKernelMilliseconds =
            execution.finishRefinementKernelMilliseconds;
    result.metrics.winnerKernelMilliseconds =
            execution.winnerKernelMilliseconds;
    result.metrics.winnerReductionKernelMilliseconds =
            execution.winnerReductionKernelMilliseconds;
    result.metrics.winnerStateCaptureKernelMilliseconds =
            execution.winnerStateCaptureKernelMilliseconds;
    result.metrics.finalizationKernelMilliseconds =
            execution.finalizationKernelMilliseconds;
    result.metrics.simulationThreadsPerBlock =
            execution.simulationThreadsPerBlock;
    result.metrics.simulationRegistersPerThread =
            execution.simulationRegistersPerThread;
    result.metrics.simulationLocalBytesPerThread =
            execution.simulationLocalBytesPerThread;
    result.metrics.simulationActiveBlocksPerMultiprocessor =
            execution.simulationActiveBlocksPerMultiprocessor;
    result.metrics.simulationTheoreticalOccupancy =
            execution.simulationTheoreticalOccupancy;
    if (!execution.best.valid) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Success(
                std::move(result));
    }

    const simulation::CudaSearchBest &best = execution.best;
    result.bestIsMutation = best.mutation;
    if (best.mutation) {
        result.bestCandidateId = best.candidateId;
    }
    result.bestMutationCount = best.mutationCount;
    result.bestEvaluationTick = best.evaluationTick;
    result.bestScore = best.score;
    result.bestTimeMs = best.timeMs;
    result.bestDetail0 = best.detail0;
    result.bestDetail1 = best.detail1;
    result.bestInputs.reserve(best.inputs.size());
    for (const simulation::CudaSearchInputEvent &input : best.inputs) {
        result.bestInputs.push_back(PublicInput(input));
    }
    result.bestInputs.insert(
            result.bestInputs.end(), lateInputs.begin(), lateInputs.end());

    if (!best.stateCaptured) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Success(std::move(result));
    }

    ReplaySimulationInstanceClone clone;
    if (simulation::DecodeCudaCandidateState(
                best.state, &clone) !=
        simulation::CudaStateConversionResult::Success ||
        best.state.controlCursor < prestartTicks ||
        best.state.controlCursor >
                std::numeric_limits<std::size_t>::max()) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::SimulationFailed,
                        "CUDA winning state conversion failed"));
    }

    PhysicsSandboxStateView view;
    view.tick = best.state.controlCursor - prestartTicks;
    view.timeMs = view.tick * tickDurationMs;
    view.durationMs = durationMs;
    view.mapEnvironment = mapEnvironment;
    view.vehicleModel = vehicleModel;
    view.playMode = playMode;
    const CHmsDyna::CHmsStateDyna &frame = best.state.body.current;
    view.car.rotationX = frame.rotationQuat.x;
    view.car.rotationY = frame.rotationQuat.y;
    view.car.rotationZ = frame.rotationQuat.z;
    view.car.rotationW = frame.rotationQuat.w;
    view.car.position = ToPublicVector(frame.position);
    view.car.linearSpeed = ToPublicVector(frame.linearSpeed);
    view.car.angularSpeed = ToPublicVector(frame.angularSpeed);
    view.car.force = ToPublicVector(frame.force);
    view.car.torque = ToPublicVector(frame.torque);
    view.accelerate = best.state.vehicle.controls.lowSpeedGateA;
    view.brake = best.state.vehicle.controls.lowSpeedGateB;
    view.steering = best.state.vehicle.controls.steeringControl;
    const ReplayRaceProgress &race = best.state.race.progress;
    view.checkpointsCollected = race.checkpointCount;
    view.checkpointsTotal = race.requiredCheckpointCount;
    view.completedLaps = race.completedLapCount;
    view.totalLaps = race.requiredLapCount;
    view.raceCompleted = race.raceCompleted;
    if (race.raceCompleted) {
        if (best.state.finishTime.present) {
            const std::uint64_t prestartNs =
                    static_cast<std::uint64_t>(
                            prestartDurationMs) *
                    1000000u;
            view.finishTime = best.state.finishTime.value;
            view.finishTime->lowerBoundNs =
                    view.finishTime->lowerBoundNs >= prestartNs
                    ? view.finishTime->lowerBoundNs - prestartNs
                    : 0u;
            view.finishTime->upperBoundNs =
                    view.finishTime->upperBoundNs >= prestartNs
                    ? view.finishTime->upperBoundNs - prestartNs
                    : 0u;
            view.finishTime->estimatedNs =
                    view.finishTime->estimatedNs >= prestartNs
                    ? view.finishTime->estimatedNs - prestartNs
                    : 0u;
            view.finishTimeMs = static_cast<std::uint32_t>(
                    view.finishTime->estimatedNs / 1000000u);
        } else {
            view.finishTimeMs =
                    race.lastPrepareTimeMs >= prestartDurationMs
                    ? race.lastPrepareTimeMs - prestartDurationMs
                    : 0u;
        }
    }
    view.respawnCount = best.state.incrementalRespawnCount;
    if (best.state.stuntsEnabled) {
        view.stuntsScore = best.state.stunts.stuntsScore;
    }
    result.bestState = view;

    auto state = std::make_shared<PhysicsSandboxState::Impl>();
    state->view = view;
    state->runtimeClone =
            std::make_shared<ReplaySimulationInstanceClone>(
                    std::move(clone));
    state->inputs = SandboxInputStorage::Full(result.bestInputs);
    state->controlPlan.reset();
    state->scenarioFingerprint = scenarioFingerprint;
    state->validationSeed = validationSeed;
    state->backend = SimulationBackend::Cuda;
    state->tickDurationMs = tickDurationMs;
    state->prestartDurationMs = prestartDurationMs;
    state->simulationHorizonMs = static_cast<std::uint32_t>(durationMs);
    state->timelineMode = timelineMode;
    state->cursor =
            static_cast<std::size_t>(best.state.controlCursor);
    state->runtimeCloneSchema = SandboxRuntimeCloneSchema;
    PhysicsSandboxState snapshot(std::move(state));
    result.bestSnapshot = std::move(snapshot);
    return PhysicsSandboxResult<
            PhysicsSandboxCudaSearchBatch>::Success(std::move(result));
}

PhysicsSandboxResult<PhysicsSandboxCudaSearchBatch>
PhysicsSandboxCudaSearchSession::EvaluateBaseline() noexcept {
    return EvaluateBaseline(std::function<bool()>{});
}

PhysicsSandboxResult<PhysicsSandboxCudaSearchBatch>
PhysicsSandboxCudaSearchSession::EvaluateBaseline(
        const std::function<bool()> &cancellationRequested) noexcept {
    try {
        if (!impl_ || !impl_->executor) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchBatch>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidSandbox,
                            "CUDA search session is invalid"));
        }
        return impl_->Convert(
                impl_->executor->EvaluateBaseline(
                        cancellationRequested));
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::AllocationFailed,
                        "CUDA baseline result allocation failed"));
    } catch (...) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::UnexpectedFailure,
                        "unexpected CUDA baseline evaluation failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxCudaSearchBatch>
PhysicsSandboxCudaSearchSession::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool cancellationRequested) noexcept {
    const std::function<bool()> probe = cancellationRequested
            ? std::function<bool()>([] { return true; })
            : std::function<bool()>{};
    return RunBatch(firstCandidateId, candidateCount, probe);
}

PhysicsSandboxResult<PhysicsSandboxCudaSearchBatch>
PhysicsSandboxCudaSearchSession::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        const std::function<bool()> &cancellationRequested) noexcept {
    try {
        if (!impl_ || !impl_->executor) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchBatch>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidSandbox,
                            "CUDA search session is invalid"));
        }
        return impl_->Convert(impl_->executor->RunBatch(
                firstCandidateId,
                candidateCount,
                cancellationRequested));
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::AllocationFailed,
                        "CUDA batch result allocation failed"));
    } catch (...) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchBatch>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::UnexpectedFailure,
                        "unexpected CUDA search batch failure"));
    }
}

PhysicsSandboxResult<std::uint32_t>
PhysicsSandboxCudaSearchSession::ReserveBatchCapacity(
        std::uint32_t candidateCount) noexcept {
    try {
        if (!impl_ || !impl_->executor || candidateCount == 0u) {
            return PhysicsSandboxResult<std::uint32_t>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "invalid CUDA calibration batch capacity"));
        }
        std::string diagnostic;
        if (!impl_->executor->ReserveBatchCapacity(
                    candidateCount, &diagnostic)) {
            return PhysicsSandboxResult<std::uint32_t>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::AllocationFailed,
                            diagnostic.empty()
                                    ? "CUDA calibration capacity allocation failed"
                                    : diagnostic));
        }
        return PhysicsSandboxResult<std::uint32_t>::Success(
                impl_->executor->BatchCapacity());
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<std::uint32_t>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::AllocationFailed,
                        "CUDA calibration capacity allocation failed"));
    } catch (...) {
        return PhysicsSandboxResult<std::uint32_t>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::UnexpectedFailure,
                        "unexpected CUDA calibration capacity failure"));
    }
}

PhysicsSandboxResult<bool>
PhysicsSandboxCudaSearchSession::UpdateConditionTimes(
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds) noexcept {
    if (!impl_ || !impl_->executor) {
        return PhysicsSandboxResult<bool>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::InvalidSandbox,
                        "CUDA search session is invalid"));
    }
    if (!impl_->executor->UpdateConditionTimes(
                lastImprovementTimeSeconds,
                lastRestartTimeSeconds)) {
        return PhysicsSandboxResult<bool>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::InvalidRequest,
                        "CUDA condition times are unavailable or invalid"));
    }
    return PhysicsSandboxResult<bool>::Success(true);
}

PhysicsSandboxResult<PhysicsSandboxCudaSearchSession>
CreatePhysicsSandboxCudaSearchSession(
        PhysicsSandbox &sandbox,
        const PhysicsSandboxCudaSearchConfiguration &configuration) noexcept {
    try {
        if (!sandbox.impl_ || !sandbox.impl_->loaded ||
            !sandbox.impl_->session ||
            sandbox.impl_->options.backend != SimulationBackend::Cuda ||
            configuration.maximumBatchSize == 0u ||
            configuration.modifiers.empty()) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "invalid CUDA search session request"));
        }
        const PhysicsSandbox::Impl &source = *sandbox.impl_;
        const std::uint32_t tickDurationMs =
                source.options.tickDurationMs;
        const PhysicsSandboxResult<PhysicsSandboxStateView> current =
                source.ReadView();
        if (!current) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    current.Error());
        }
        const std::int64_t branchTimeMs =
                static_cast<std::int64_t>(current.Value().timeMs);
        if (configuration.earliestMutationTimeMs !=
                        branchTimeMs + tickDurationMs ||
            configuration.evaluationStartTimeMs <
                    configuration.earliestMutationTimeMs ||
            configuration.evaluationEndTimeMs <
                    configuration.evaluationStartTimeMs ||
            configuration.evaluationEndTimeMs >
                    static_cast<std::int64_t>(
                            current.Value().durationMs) ||
            configuration.evaluationStartTimeMs % tickDurationMs != 0 ||
            configuration.evaluationEndTimeMs % tickDurationMs != 0) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "CUDA search times are invalid or unaligned"));
        }

        const std::uint64_t endRaceTicks =
                static_cast<std::uint64_t>(
                        configuration.evaluationEndTimeMs) /
                tickDurationMs;
        if (endRaceTicks >
                    std::numeric_limits<std::size_t>::max() -
                            source.prestartTicks ||
            source.cursor >
                    source.prestartTicks +
                            static_cast<std::size_t>(endRaceTicks)) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "CUDA search timeline is out of range"));
        }
        const std::size_t endCursor =
                source.prestartTicks +
                static_cast<std::size_t>(endRaceTicks);
        if (!source.controlPlan || endCursor > source.controlPlan->Size()) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "CUDA search timeline exceeds the Simulation horizon"));
        }

        const std::vector<PhysicsSandboxInputEvent> allSourceInputs =
                source.inputs->Materialize();
        const auto lateBegin = std::upper_bound(
                allSourceInputs.begin(),
                allSourceInputs.end(),
                source.simulationHorizonMs,
                [](std::int64_t timeMs,
                   const PhysicsSandboxInputEvent &event) {
                    return timeMs < event.timeMs;
                });
        const std::vector<PhysicsSandboxInputEvent> sourceInputs(
                allSourceInputs.begin(), lateBegin);
        std::size_t maximumEventCount = 0u;
        if (!MaximumEventCapacity(
                    sourceInputs.size(),
                    configuration.modifiers,
                    tickDurationMs,
                    &maximumEventCount)) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::InvalidRequest,
                            "CUDA modifier pipeline event capacity is unsupported"));
        }

        simulation::CudaSearchExecutorConfiguration internal;
        internal.maximumBatchSize = configuration.maximumBatchSize;
        internal.tickDurationMs = tickDurationMs;
        internal.prestartDurationMs =
                source.options.prestartDurationMs;
        internal.branchTimeMs = branchTimeMs;
        internal.evaluationStartTimeMs =
                configuration.evaluationStartTimeMs;
        internal.evaluationEndTimeMs =
                configuration.evaluationEndTimeMs;
        internal.maximumEventCount = maximumEventCount;
        internal.useLegacyMutationPipelineForTesting =
                configuration.useLegacyMutationPipelineForTesting;
        internal.captureBestState = configuration.captureBestState;
        if (configuration.incumbent) {
            if (configuration.incumbent->mutationCount >
                        std::numeric_limits<std::uint32_t>::max() ||
                (configuration.incumbent->mutation &&
                 !configuration.incumbent->candidateId)) {
                return PhysicsSandboxResult<
                        PhysicsSandboxCudaSearchSession>::Failure(
                        SearchError(
                                PhysicsSandboxErrorCode::InvalidRequest,
                                "invalid CUDA incumbent seed"));
            }
            simulation::CudaSearchIncumbent incumbent;
            incumbent.mutation = configuration.incumbent->mutation;
            incumbent.candidateId =
                    configuration.incumbent->candidateId.value_or(0u);
            incumbent.mutationCount = static_cast<std::uint32_t>(
                    configuration.incumbent->mutationCount);
            incumbent.evaluationTick =
                    configuration.incumbent->evaluationTick;
            incumbent.score = configuration.incumbent->score;
            incumbent.timeMs = configuration.incumbent->timeMs;
            incumbent.detail0 = configuration.incumbent->detail0;
            incumbent.detail1 = configuration.incumbent->detail1;
            incumbent.preciseFinish =
                    configuration.incumbent->preciseFinish;
            internal.incumbent = incumbent;
        }
        if (configuration.useSessionSpecialization) {
            internal.sessionSpecialization =
                    source.session->CudaSearchSpecialization();
            if (!internal.sessionSpecialization) {
                std::string specializationDiagnostic;
                if (!source.session->PrepareCudaSearchSpecialization(
                            &specializationDiagnostic)) {
                    return PhysicsSandboxResult<
                            PhysicsSandboxCudaSearchSession>::Failure(
                            SearchError(
                                    PhysicsSandboxErrorCode::
                                            SimulationFailed,
                                    specializationDiagnostic.empty()
                                            ? "The optional fast CUDA kernel is unavailable"
                                            : std::move(
                                                      specializationDiagnostic)));
                }
                internal.sessionSpecialization =
                        source.session->CudaSearchSpecialization();
            }
        }
        internal.baselineTicks.reserve(endCursor - source.cursor);
        for (std::size_t index = source.cursor;
             index < endCursor; ++index) {
            internal.baselineTicks.push_back(
                    simulation::FlattenCudaControlTick(
                            source.controlPlan->Tick(index)));
        }
        internal.baselineInputs.reserve(sourceInputs.size());
        for (const PhysicsSandboxInputEvent &input : sourceInputs) {
            internal.baselineInputs.push_back(CudaInput(input));
        }
        internal.modifiers.reserve(configuration.modifiers.size());
        for (const PhysicsSandboxCudaModifier &modifier :
             configuration.modifiers) {
            simulation::CudaSearchModifierConfiguration converted =
                    CudaModifier(modifier);
            if (const auto *smooth = std::get_if<
                        PhysicsSandboxCudaSmoothSteeringModifier>(
                        &modifier)) {
                converted.weightOffset = static_cast<std::uint32_t>(
                        internal.smoothWeights.size());
                const std::uint64_t radiusTicks =
                        static_cast<std::uint64_t>(smooth->radiusMs) /
                        tickDurationMs;
                if (radiusTicks >
                    std::numeric_limits<std::uint32_t>::max() -
                            internal.smoothWeights.size()) {
                    return PhysicsSandboxResult<
                            PhysicsSandboxCudaSearchSession>::Failure(
                            SearchError(
                                    PhysicsSandboxErrorCode::InvalidRequest,
                                    "CUDA smooth-steering weight table is too large"));
                }
                constexpr double pi = 3.14159265358979323846;
                internal.smoothWeights.reserve(
                        internal.smoothWeights.size() +
                        static_cast<std::size_t>(radiusTicks + 1u));
                for (std::uint64_t distance = 0u;
                     distance <= radiusTicks; ++distance) {
                    internal.smoothWeights.push_back(
                            smooth->radiusMs == 0
                                    ? 1.0
                                    : 0.5 *
                                              (1.0 +
                                               std::cos(
                                                       pi *
                                                       static_cast<double>(
                                                               distance *
                                                               tickDurationMs) /
                                                       static_cast<double>(
                                                               smooth->radiusMs))));
                }
            }
            internal.modifiers.push_back(converted);
        }
        internal.evaluator = CudaEvaluator(configuration.evaluator);
        if (configuration.condition) {
            if (configuration.condition->instructions.empty() ||
                configuration.condition->instructions.size() > 256u) {
                return PhysicsSandboxResult<
                        PhysicsSandboxCudaSearchSession>::Failure(
                        SearchError(
                                PhysicsSandboxErrorCode::InvalidRequest,
                                "CUDA condition program must contain between 1 and 256 instructions"));
            }
            simulation::CudaSearchConditionConfiguration condition;
            condition.lastImprovementTimeSeconds =
                    configuration.condition->lastImprovementTimeSeconds;
            condition.lastRestartTimeSeconds =
                    configuration.condition->lastRestartTimeSeconds;
            condition.instructions.reserve(
                    configuration.condition->instructions.size());
            for (const PhysicsSandboxCudaConditionInstruction &instruction :
                 configuration.condition->instructions) {
                condition.instructions.push_back({
                        static_cast<simulation::CudaSearchConditionOpcode>(
                                instruction.opcode),
                        static_cast<simulation::CudaSearchConditionValue>(
                                instruction.value),
                        instruction.x,
                        instruction.y,
                        instruction.z});
            }
            internal.condition = std::move(condition);
        }

        std::string diagnostic;
        std::unique_ptr<simulation::CudaSearchExecutor> executor =
                source.session->CreateCudaSearchExecutor(
                        std::move(internal),
                        source.cursor,
                        &diagnostic);
        if (!executor) {
            return PhysicsSandboxResult<
                    PhysicsSandboxCudaSearchSession>::Failure(
                    SearchError(
                            PhysicsSandboxErrorCode::SimulationFailed,
                            diagnostic.empty()
                                    ? "CUDA search session creation failed"
                                    : diagnostic));
        }
        auto impl =
                std::make_unique<PhysicsSandboxCudaSearchSession::Impl>();
        impl->executor = std::move(executor);
        impl->inputs = source.inputs;
        impl->lateInputs.assign(lateBegin, allSourceInputs.end());
        impl->controlPlan = source.controlPlan;
        impl->scenarioFingerprint = source.scenarioFingerprint;
        impl->validationSeed = source.inputMetadata.validationSeed;
        impl->tickDurationMs = tickDurationMs;
        impl->prestartDurationMs = source.options.prestartDurationMs;
        impl->durationMs = source.simulationHorizonMs;
        impl->timelineMode = source.options.timelineMode;
        impl->prestartTicks = source.prestartTicks;
        impl->mapEnvironment =
                ToPublicMapEnvironment(source.route.mapEnvironment);
        impl->vehicleModel =
                ToPublicVehicleModel(source.route.vehicleModel);
        impl->playMode = ToPublicPlayMode(
                source.challengeMetadata.playMode.value_or(
                        EChallengePlayMode::Race));
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchSession>::Success(
                PhysicsSandboxCudaSearchSession(std::move(impl)));
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchSession>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::AllocationFailed,
                        "CUDA search session allocation failed"));
    } catch (...) {
        return PhysicsSandboxResult<
                PhysicsSandboxCudaSearchSession>::Failure(
                SearchError(
                        PhysicsSandboxErrorCode::UnexpectedFailure,
                        "unexpected CUDA search session creation failure"));
    }
}

std::vector<PhysicsSandboxResult<PhysicsSandboxStateView>>
AdvancePhysicsSandboxes(
        const std::vector<PhysicsSandbox *> &sandboxes,
        std::uint32_t count) noexcept {
    std::vector<PhysicsSandboxResult<PhysicsSandboxStateView>> results;
    try {
        results.reserve(sandboxes.size());
        simulation::ExecuteBatched(
                sandboxes.size(),
                [&](std::size_t index) {
                    PhysicsSandbox *sandbox = sandboxes[index];
                    if (sandbox == nullptr || !sandbox->impl_ ||
                        sandbox->Backend() != SimulationBackend::Batched) {
                        results.push_back(PhysicsSandboxResult<
                                PhysicsSandboxStateView>::Failure(
                                SandboxError(
                                        PhysicsSandboxErrorCode::InvalidRequest,
                                        "batched advance requires batched sandboxes")));
                        return;
                    }
                    results.push_back(sandbox->AdvanceTicks(count));
                });
    } catch (...) {
        results.clear();
    }
    return results;
}

forevervalidator::simulation::CudaTimelineBatchResult
cuda_test::PhysicsSandboxCudaTestAccess::RunCandidateBatch(
        PhysicsSandbox &sandbox,
        std::uint32_t candidateCount,
        std::uint32_t tickCount,
        bool mutateControls,
        bool cancellationRequested) {
    forevervalidator::simulation::CudaTimelineBatchResult result;
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session || tickCount == 0u ||
        !sandbox.impl_->controlPlan ||
        sandbox.impl_->cursor > sandbox.impl_->controlPlan->Size() ||
        tickCount >
                sandbox.impl_->controlPlan->Size() -
                        sandbox.impl_->cursor) {
        result.status = forevervalidator::simulation::
                CudaTimelineStatus::InvalidArgument;
        result.diagnostic =
                "sandbox CUDA candidate batch request is invalid";
        return result;
    }
    try {
        std::vector<ReplayControlTick> ticks =
                sandbox.impl_->controlPlan->CopyRange(
                        sandbox.impl_->cursor, tickCount);
        return sandbox.impl_->session->
                ExecuteCudaCandidateBatchForTesting(
                        ticks, candidateCount, mutateControls,
                        sandbox.impl_->cursor,
                        cancellationRequested);
    } catch (const std::bad_alloc &) {
        result.status = forevervalidator::simulation::
                CudaTimelineStatus::CapacityExceeded;
        result.diagnostic =
                "sandbox CUDA candidate batch allocation failed";
        return result;
    }
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::
        RunCandidateBatchDifferential(
                PhysicsSandbox &sandbox,
                std::uint32_t candidateCount,
                std::uint32_t tickCount,
                bool mutateControls) {
    ReplayCudaVehiclePrefixDifferential result;
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session || tickCount == 0u ||
        !sandbox.impl_->controlPlan ||
        sandbox.impl_->cursor >
                sandbox.impl_->controlPlan->Size() ||
        tickCount >
                sandbox.impl_->controlPlan->Size() -
                        sandbox.impl_->cursor) {
        result.diagnostic =
                "sandbox CUDA candidate batch differential request is invalid";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    try {
        std::vector<ReplayControlTick> ticks =
                sandbox.impl_->controlPlan->CopyRange(
                        sandbox.impl_->cursor, tickCount);
        result = sandbox.impl_->session->
                RunCudaCandidateBatchDifferentialForTesting(
                        ticks, candidateCount, mutateControls,
                        sandbox.impl_->cursor);
    } catch (const std::bad_alloc &) {
        result.diagnostic =
                "sandbox CUDA candidate batch differential allocation failed";
    }
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

std::optional<forevervalidator::simulation::CudaSceneTransferMetrics>
cuda_test::PhysicsSandboxCudaTestAccess::SceneTransfer(
        const PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->session) {
        return std::nullopt;
    }
    return sandbox.impl_->session->
            CudaSceneTransferMetricsForTesting();
}

std::optional<forevervalidator::simulation::
                      CudaStaticConfigurationTransferMetrics>
cuda_test::PhysicsSandboxCudaTestAccess::ConfigurationTransfer(
        const PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->session) {
        return std::nullopt;
    }
    return sandbox.impl_->session->
            CudaStaticConfigurationTransferMetricsForTesting();
}

std::optional<forevervalidator::simulation::CudaCandidateState>
cuda_test::PhysicsSandboxCudaTestAccess::CaptureCandidateState(
        const PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        return std::nullopt;
    }
    const auto clone = sandbox.impl_->session->CaptureRuntimeClone();
    if (!clone) {
        return std::nullopt;
    }
    forevervalidator::simulation::CudaCandidateState result;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                *clone, sandbox.impl_->inputMetadata.validationSeed,
                sandbox.impl_->cursor, 0u, clone->randomState,
                &result) != forevervalidator::simulation::
                            CudaStateConversionResult::Success) {
        return std::nullopt;
    }
    return result;
}

std::size_t cuda_test::PhysicsSandboxCudaTestAccess::TimelineSize(
        const PhysicsSandbox &sandbox) noexcept {
    return sandbox.impl_ && sandbox.impl_->loaded
            ? sandbox.impl_->controlPlan->Size()
            : 0u;
}

std::size_t cuda_test::PhysicsSandboxCudaTestAccess::Cursor(
        const PhysicsSandbox &sandbox) noexcept {
    return sandbox.impl_ && sandbox.impl_->loaded
            ? sandbox.impl_->cursor
            : 0u;
}

std::optional<OptimizedCpuStaticSceneFingerprint>
static_scene_test::PhysicsSandboxStaticSceneTestAccess::
        CaptureStaticSceneFingerprint(
                const PhysicsSandbox &sandbox) noexcept {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        return std::nullopt;
    }
    return sandbox.impl_->session->
            CaptureOptimizedCpuStaticSceneFingerprintForTesting();
}

PhysicsSandboxResult<PhysicsSandboxStateView>
static_scene_test::PhysicsSandboxStaticSceneTestAccess::RestartAtRaceTick(
        PhysicsSandbox &sandbox,
        std::uint64_t raceTick) noexcept {
    try {
        if (!sandbox.impl_) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidSandbox,
                                 "sandbox is moved-from"));
        }
        return sandbox.impl_->Restart(raceTick);
    } catch (const std::bad_alloc &) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                             "allocation failed while restarting sandbox"));
    } catch (...) {
        return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                SandboxError(PhysicsSandboxErrorCode::UnexpectedFailure,
                             "unexpected sandbox restart failure"));
    }
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunVehiclePrefix(
        PhysicsSandbox &sandbox,
        float dt) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no loaded CUDA simulation";
        return result;
    }
    return sandbox.impl_->session->
            RunCudaVehiclePrefixDifferentialForTesting(dt);
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunVehicleForce(
        PhysicsSandbox &sandbox,
        float dt) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no loaded CUDA simulation";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    ReplayCudaVehiclePrefixDifferential result =
            sandbox.impl_->session->
                    RunCudaVehicleForceDifferentialForTesting(dt);
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunCollision(
        PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no loaded CUDA simulation";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    ReplayCudaVehiclePrefixDifferential result =
            sandbox.impl_->session->
                    RunCudaCollisionDifferentialForTesting();
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunPhysicsStep(
        PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no loaded CUDA simulation";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    ReplayCudaVehiclePrefixDifferential result =
            sandbox.impl_->session->
                    RunCudaPhysicsStepDifferentialForTesting();
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunCollisionSubstep(
        PhysicsSandbox &sandbox,
        float dt) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no loaded CUDA simulation";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    ReplayCudaVehiclePrefixDifferential result =
            sandbox.impl_->session->
                    RunCudaCollisionSubstepDifferentialForTesting(dt);
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunPreCollision(
        PhysicsSandbox &sandbox,
        float dt) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no loaded CUDA simulation";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    ReplayCudaVehiclePrefixDifferential result =
            sandbox.impl_->session->
                    RunCudaPreCollisionDifferentialForTesting(dt);
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

ReplayCudaVehiclePrefixDifferential
cuda_test::PhysicsSandboxCudaTestAccess::RunNextTimelineTick(
        PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->loaded ||
        !sandbox.impl_->session ||
        !sandbox.impl_->controlPlan ||
        sandbox.impl_->cursor >=
                sandbox.impl_->controlPlan->Size()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "sandbox has no next CUDA timeline tick";
        return result;
    }
    tmnf::simulation::DeterministicExecutionScope scope;
    if (!scope.Established()) {
        ReplayCudaVehiclePrefixDifferential result;
        result.diagnostic =
                "deterministic execution mode is unavailable";
        return result;
    }
    ReplayCudaVehiclePrefixDifferential result =
            sandbox.impl_->session->
                    RunCudaTimelineTickDifferentialForTesting(
                            sandbox.impl_->controlPlan->Tick(
                                    sandbox.impl_->cursor));
    if (!scope.Restore()) {
        result.success = false;
        result.diagnostic =
                "deterministic execution restoration failed";
    }
    return result;
}

bool cuda_test::PhysicsSandboxCudaTestAccess::StageNextTimelinePrefix(
        PhysicsSandbox &sandbox) {
    return sandbox.impl_ && sandbox.impl_->loaded &&
            sandbox.impl_->session &&
            sandbox.impl_->controlPlan &&
            sandbox.impl_->cursor <
                    sandbox.impl_->controlPlan->Size() &&
            sandbox.impl_->session->
                    StageCudaTimelinePrefixForTesting(
                            sandbox.impl_->controlPlan->Tick(
                                    sandbox.impl_->cursor));
}

bool cuda_test::PhysicsSandboxCudaTestAccess::StageCollisionSubstep(
        PhysicsSandbox &sandbox,
        float dt) {
    return sandbox.impl_ && sandbox.impl_->loaded &&
            sandbox.impl_->session &&
            sandbox.impl_->session->
                    StageCollisionSubstepForTesting(dt);
}

bool cuda_test::PhysicsSandboxCudaTestAccess::StagePreCollision(
        PhysicsSandbox &sandbox,
        float dt) {
    return sandbox.impl_ && sandbox.impl_->loaded &&
            sandbox.impl_->session &&
            sandbox.impl_->session->
                    StageCudaPreCollisionForTesting(dt);
}

std::string cuda_test::PhysicsSandboxCudaTestAccess::Diagnostic(
        const PhysicsSandbox &sandbox) {
    if (!sandbox.impl_ || !sandbox.impl_->session) {
        return "sandbox has no simulation session";
    }
    return sandbox.impl_->session->CudaInitializationDiagnostic();
}

}  // namespace experimental

bool IsNormalizedAssetIdentifier(std::string_view identifier) noexcept {
    if (identifier.empty() || identifier.front() == '/' ||
        identifier.back() == '/' ||
        identifier.find('\\') != std::string_view::npos ||
        identifier.find('\0') != std::string_view::npos) {
        return false;
    }
    const bool asciiDrivePrefix = identifier.size() >= 2u &&
            ((identifier[0] >= 'A' && identifier[0] <= 'Z') ||
             (identifier[0] >= 'a' && identifier[0] <= 'z')) &&
            identifier[1] == ':';
    if (asciiDrivePrefix) {
        return false;
    }
    std::size_t start = 0u;
    while (start < identifier.size()) {
        const std::size_t end = identifier.find('/', start);
        const std::size_t count =
                (end == std::string_view::npos ? identifier.size() : end) - start;
        if (count == 0u ||
            (count == 1u && identifier[start] == '.') ||
            (count == 2u && identifier[start] == '.' &&
             identifier[start + 1u] == '.')) {
            return false;
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1u;
    }
    return true;
}

AssetSource::AssetSource(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
AssetSource::AssetSource(AssetSource &&) noexcept = default;
AssetSource &AssetSource::operator=(AssetSource &&) noexcept = default;
AssetSource::~AssetSource() = default;

Result<AssetSource> CreateAssetSource(AssetProvider provider) noexcept {
    try {
        if (!provider) {
            return Result<AssetSource>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::InvalidArgument,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidAssetProvider,
                    {},
                    "asset provider is empty"));
        }
        return Result<AssetSource>::Success(AssetSource(
                std::make_unique<AssetSource::Impl>(std::move(provider))));
    } catch (const std::bad_alloc &) {
        return Result<AssetSource>::Failure(AllocationError(
                ValidationStage::ContextCreation, {},
                "allocation failed while creating asset source"));
    } catch (...) {
        return Result<AssetSource>::Failure(MakeError(
                ValidationErrorCategory::Internal,
                ValidationErrorCode::UnexpectedFailure,
                ValidationStage::ContextCreation,
                ValidationFailureReason::UnexpectedFailure,
                {},
                "unexpected asset source creation failure"));
    }
}

ValidationContext::ValidationContext(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
ValidationContext::ValidationContext(ValidationContext &&) noexcept = default;
ValidationContext &ValidationContext::operator=(ValidationContext &&) noexcept =
        default;
ValidationContext::~ValidationContext() = default;

Result<ValidationContext> CreateValidationContext(
        AssetSource source) noexcept {
    try {
        if (source.impl_ == nullptr || !source.impl_->provider) {
            return Result<ValidationContext>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::AssetSourceUnavailable,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidAssetSource,
                    {},
                    "asset source is moved-from or invalid"));
        }
        AssetProvider provider = std::move(source.impl_->provider);
        source.impl_.reset();
        return Result<ValidationContext>::Success(ValidationContext(
                std::make_unique<ValidationContext::Impl>(
                        std::move(provider))));
    } catch (const std::bad_alloc &) {
        return Result<ValidationContext>::Failure(AllocationError(
                ValidationStage::ContextCreation, {},
                "allocation failed while creating validation context"));
    } catch (...) {
        return Result<ValidationContext>::Failure(MakeError(
                ValidationErrorCategory::Internal,
                ValidationErrorCode::UnexpectedFailure,
                ValidationStage::ContextCreation,
                ValidationFailureReason::UnexpectedFailure,
                {},
                "unexpected validation context creation failure"));
    }
}

Result<ValidationReport> ValidateReplay(
        ValidationContext &context,
        ByteView replayBytes,
        const ReplayIdentity &identity,
        const ValidationOptions &options) noexcept {
    try {
        const ValidationOptions immutableOptions = options;
        if (context.impl_ == nullptr || !context.impl_->state.provider) {
            return Result<ValidationReport>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::InvalidArgument,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidValidationContext,
                    identity,
                    "validation context is moved-from or invalid"));
        }
        if (!replayBytes.IsValid() || replayBytes.size == 0u ||
            identity.name.empty() || immutableOptions.requestedSamples == 0u ||
            immutableOptions.controlTickMs == 0u ||
            immutableOptions.validationPrestartMs == 0u ||
            !simulation::IsSimulationBackendSupported(
                    immutableOptions.backend)) {
            return Result<ValidationReport>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::InvalidArgument,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidValidationRequest,
                    identity,
                    "invalid replay validation request"));
        }
        if (immutableOptions.backend == SimulationBackend::Batched) {
            std::vector<ReplayValidationRequest> requests;
            requests.push_back({replayBytes, identity});
            Result<ReplayBatchReport> batch = ValidateReplayBatch(
                    context, requests, immutableOptions);
            if (!batch) {
                return Result<ValidationReport>::Failure(
                        std::move(batch).Error());
            }
            ReplayBatchReport report = std::move(batch).Value();
            if (report.attempts.size() != 1u) {
                return Result<ValidationReport>::Failure(MakeError(
                        ValidationErrorCategory::Internal,
                        ValidationErrorCode::UnexpectedFailure,
                        ValidationStage::ValidationEvaluation,
                        ValidationFailureReason::UnexpectedFailure,
                        identity,
                        "batched backend returned an invalid result count"));
            }
            return std::move(report.attempts.front());
        }

        tmnf::simulation::DeterministicExecutionScope deterministicScope;
        if (!deterministicScope.Established()) {
            return Result<ValidationReport>::Failure(MakeError(
                    ValidationErrorCategory::Simulation,
                    ValidationErrorCode::DeterministicExecutionUnavailable,
                    ValidationStage::SimulationStartup,
                    ValidationFailureReason::DeterministicExecutionUnavailable,
                    identity,
                    "deterministic execution mode unavailable"));
        }

        Result<ValidationReport> result = RunReplayValidation(
                context.impl_->state, replayBytes, identity,
                immutableOptions);
        if (!deterministicScope.Restore()) {
            return Result<ValidationReport>::Failure(MakeError(
                    ValidationErrorCategory::Simulation,
                    ValidationErrorCode::DeterministicExecutionUnavailable,
                    ValidationStage::SimulationStep,
                    ValidationFailureReason::DeterministicStateRestoreFailed,
                    identity,
                    "deterministic execution state could not be restored"));
        }
        return result;
    } catch (const std::bad_alloc &) {
        return Result<ValidationReport>::Failure(AllocationError(
                ValidationStage::ValidationEvaluation,
                identity,
                "allocation failed during replay validation"));
    } catch (...) {
        return Result<ValidationReport>::Failure(MakeError(
                ValidationErrorCategory::Internal,
                ValidationErrorCode::UnexpectedFailure,
                ValidationStage::ValidationEvaluation,
                ValidationFailureReason::UnexpectedFailure,
                identity,
                "unexpected replay validation failure"));
    }
}

Result<ReplayBatchReport> ValidateReplayBatch(
        ValidationContext &context,
        const std::vector<ReplayValidationRequest> &requests,
        const ValidationOptions &options) noexcept {
    try {
        if (context.impl_ == nullptr || !context.impl_->state.provider) {
            return Result<ReplayBatchReport>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::InvalidArgument,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidValidationContext,
                    {},
                    "validation context is moved-from or invalid"));
        }
        if (requests.empty() || options.requestedSamples == 0u ||
            options.controlTickMs == 0u ||
            options.validationPrestartMs == 0u ||
            !simulation::IsSimulationBackendSupported(options.backend)) {
            return Result<ReplayBatchReport>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::InvalidArgument,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidValidationRequest,
                    {},
                    "invalid replay batch request"));
        }

        ValidationOptions leafOptions = options;
        leafOptions.backend = simulation::ResolveLeafBackend(options.backend);
        ReplayBatchReport report;
        report.attempts.reserve(requests.size());
        simulation::ExecuteBatched(
                requests.size(),
                [&](std::size_t index) {
                    const ReplayValidationRequest &request = requests[index];
                    report.attempts.push_back(ValidateReplay(
                            context,
                            request.replayBytes,
                            request.identity,
                            leafOptions));
                });
        return Result<ReplayBatchReport>::Success(std::move(report));
    } catch (const std::bad_alloc &) {
        return Result<ReplayBatchReport>::Failure(AllocationError(
                ValidationStage::ValidationEvaluation,
                {},
                "allocation failed during replay batch validation"));
    } catch (...) {
        return Result<ReplayBatchReport>::Failure(MakeError(
                ValidationErrorCategory::Internal,
                ValidationErrorCode::UnexpectedFailure,
                ValidationStage::ValidationEvaluation,
                ValidationFailureReason::UnexpectedFailure,
                {},
                "unexpected replay batch validation failure"));
    }
}

}  // namespace forevervalidator
