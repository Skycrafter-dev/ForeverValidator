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
    case ReplayInputProvenance::TMInterface:
        return ReplayProvenance::TMInterface;
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
    if (report.metadata.replayProvenance == ReplayProvenance::TMInterface) {
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
            report.status = ValidationStatus::TMInterfaceReplay;
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
    return MakeError(
            category, code, stage, ExecutionReason(result), identity, diagnostic);
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
        return Result<ValidationReport>::Failure(
                PreloadError(preloadResult, identity));
    }

    ReplaySimulationDefinitionBuild definition =
            BuildReplaySimulationDefinition(
                    *prepared.vehicleSources, preload.WaterDefinition());
    if (!definition) {
        return Result<ValidationReport>::Failure(
                DefinitionError(definition.Error(), identity));
    }

    simulationSession.ActivateStaticScene();
    ReplayFileValidationBuild validation = ValidateReplayFile(
            replayFile,
            route.validationMode,
            simulationSession,
            definition.Value(),
            configuration);
    if (!validation) {
        return Result<ValidationReport>::Failure(
                ExecutionError(validation.Error(), identity));
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

struct PhysicsSandboxState::Impl {
    PhysicsSandboxStateView view{};
    std::shared_ptr<const ReplaySimulationInstanceClone> runtimeClone;
    std::vector<PhysicsSandboxInputEvent> inputs;
    std::uint64_t scenarioFingerprint = 0u;
    std::uint32_t validationSeed = 0u;
    SimulationBackend backend = SimulationBackend::Reference;
    std::uint32_t tickDurationMs = 0u;
    std::uint32_t prestartDurationMs = 0u;
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
    ReplayControlPlan controlPlan{};
    ReplayInputMetadata inputMetadata{};
    std::vector<ReplayInputActionKind> definedActions;
    ReplayInputProvenance provenance = ReplayInputProvenance::Unmarked;
    std::vector<PhysicsSandboxInputEvent> inputs;
    ReplayChallengeMetadata challengeMetadata{};
    ReplayAssetRoute route{};
    ReplayIdentity identity{};
    std::uint64_t scenarioFingerprint = 0u;
    std::size_t cursor = 0u;
    std::size_t prestartTicks = 0u;
    bool loaded = false;

    PhysicsSandboxResult<ReplayControlPlan> BuildControlPlan(
            const std::vector<PhysicsSandboxInputEvent> &source) const {
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
                 !std::isfinite(event.value.analog))) {
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
        request.validationDurationMs =
                static_cast<std::int32_t>(inputMetadata.durationMs);
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
        if (targetCursor > controlPlan.ticks.size()) {
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
                controlPlan.ticks.front(),
                inputMetadata.validationSeed);
        if (start != ReplaySimulationRunResult::Success) {
            deterministicScope.Restore();
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::SimulationFailed,
                                 "sandbox simulation could not start"));
        }
        const ReplaySimulationTimelineResult advanced =
                session->AdvanceIncremental(
                        controlPlan.ticks, 0u, targetCursor);
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
        view.accelerate = state->controls.lowSpeedGateA;
        view.brake = state->controls.lowSpeedGateB;
        view.steering = state->controls.steering;
        view.checkpointsCollected = state->race.checkpointCount;
        view.checkpointsTotal = state->race.requiredCheckpointCount;
        view.completedLaps = state->race.completedLapCount;
        view.totalLaps = state->race.requiredLapCount;
        view.raceCompleted = state->race.raceCompleted;
        if (state->finishTimeMs.has_value()) {
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
    try {
        if (!impl_ || !replayBytes.IsValid() || replayBytes.size == 0u ||
            identity.name.empty()) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::InvalidRequest,
                                 "invalid sandbox replay request"));
        }
        ReplayFile replay;
        const ReplayFileReadError readError = ReadReplayBytes(
                reinterpret_cast<const std::uint8_t *>(replayBytes.data),
                replayBytes.size,
                &replay);
        if (readError != ReplayFileReadError::Success) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::ReplayLoadingFailed,
                            "sandbox replay could not be decoded",
                            ReplayDecodeError(readError, identity)));
        }
        if (!replay.HasValidationInput()) {
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
                            "sandbox replay route is unsupported",
                            ReplayRouteError(routeResult, identity, replay)));
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
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(
                            PhysicsSandboxErrorCode::MapLoadingFailed,
                            "sandbox map could not be loaded",
                            PreloadError(preloadResult, identity)));
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
        session->ActivateStaticScene();

        std::vector<PhysicsSandboxInputEvent> inputs;
        inputs.reserve(replay.InputTimeline().Events().size());
        for (const ReplayInputEvent &event : replay.InputTimeline().Events()) {
            const std::int64_t relative =
                    static_cast<std::int64_t>(event.timeMs) -
                    SandboxInputTimeBaseMs;
            if (relative < std::numeric_limits<std::int32_t>::min() ||
                relative > std::numeric_limits<std::int32_t>::max()) {
                return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                        SandboxError(
                                PhysicsSandboxErrorCode::ReplayLoadingFailed,
                                "sandbox replay input time is out of range"));
            }
            inputs.push_back({
                    static_cast<std::int32_t>(relative),
                    ToSandboxAction(event.action),
                    ToSandboxValue(event.value)});
        }

        impl_->session = std::move(session);
        impl_->definition = std::move(definition).Value();
        impl_->inputMetadata = replay.InputTimeline().Metadata();
        impl_->definedActions = replay.InputTimeline().DefinedActions();
        impl_->provenance = replay.InputTimeline().Provenance();
        impl_->challengeMetadata = replay.ChallengeMetadata();
        impl_->route = route;
        impl_->identity = identity;
        impl_->scenarioFingerprint = Fingerprint(replayBytes);
        impl_->inputs = std::move(inputs);
        impl_->prestartTicks =
                impl_->options.prestartDurationMs /
                impl_->options.tickDurationMs;
        PhysicsSandboxResult<ReplayControlPlan> plan =
                impl_->BuildControlPlan(impl_->inputs);
        if (!plan) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    std::move(plan).Error());
        }
        impl_->controlPlan = std::move(plan).Value();
        impl_->loaded = true;
        return impl_->Restart(0u);
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
                std::vector<PhysicsSandboxInputEvent>>::Success(impl_->inputs);
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
        std::vector<PhysicsSandboxInputEvent> oldPast;
        std::vector<PhysicsSandboxInputEvent> newPast;
        for (const PhysicsSandboxInputEvent &event : impl_->inputs) {
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
        impl_->inputs = std::move(events);
        impl_->controlPlan = std::move(plan).Value();
        return PhysicsSandboxResult<std::size_t>::Success(
                impl_->inputs.size());
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

PhysicsSandboxResult<PhysicsSandboxStateView> PhysicsSandbox::AdvanceTicks(
        std::uint32_t count) noexcept {
    try {
        if (!impl_ || !impl_->loaded || count == 0u ||
            impl_->cursor > impl_->controlPlan.ticks.size() ||
            count > impl_->controlPlan.ticks.size() - impl_->cursor) {
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
                impl_->session->AdvanceIncremental(
                        impl_->controlPlan.ticks, impl_->cursor, count);
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
        state->scenarioFingerprint = impl_->scenarioFingerprint;
        state->validationSeed = impl_->inputMetadata.validationSeed;
        state->backend = impl_->options.backend;
        state->tickDurationMs = impl_->options.tickDurationMs;
        state->prestartDurationMs = impl_->options.prestartDurationMs;
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
                    impl_->options.prestartDurationMs) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state is incompatible"));
        }
        if (!state.impl_->runtimeClone) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state has no runtime clone"));
        }
        PhysicsSandboxResult<ReplayControlPlan> plan =
                impl_->BuildControlPlan(state.impl_->inputs);
        if (!plan) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    std::move(plan).Error());
        }
        ReplayControlPlan restoredPlan = std::move(plan).Value();
        if (state.impl_->cursor > restoredPlan.ticks.size() ||
            state.impl_->cursor < impl_->prestartTicks) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::IncompatibleState,
                                 "sandbox state cursor is incompatible"));
        }
        std::vector<PhysicsSandboxInputEvent> restoredInputs =
                state.impl_->inputs;
        ReplaySimulationInstanceClone runtimeClone =
                *state.impl_->runtimeClone;
        if (!impl_->session->PrepareRuntimeCloneRestore(runtimeClone)) {
            return PhysicsSandboxResult<PhysicsSandboxStateView>::Failure(
                    SandboxError(PhysicsSandboxErrorCode::AllocationFailed,
                                 "sandbox runtime clone could not be prepared"));
        }

        impl_->session->RestoreRuntimeClone(std::move(runtimeClone));
        impl_->inputs.swap(restoredInputs);
        impl_->controlPlan = std::move(restoredPlan);
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

PhysicsSandboxResult<PhysicsSandbox> CreatePhysicsSandbox(
        AssetSource source,
        const PhysicsSandboxOptions &options) noexcept {
    try {
        AssetProvider provider =
                detail::PhysicsSandboxAssetSourceAccess::Take(source);
        if (!provider || options.tickDurationMs == 0u ||
            options.prestartDurationMs == 0u ||
            options.prestartDurationMs % options.tickDurationMs != 0u ||
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
