#include <forevervalidator/validation.h>

#include <cstdint>
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
#include "validation/evaluation/replay_validation_session.h"
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

    ReplaySimulationSession simulationSession;
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
            identity.name.empty() || options.requestedSamples == 0u ||
            options.controlTickMs == 0u ||
            options.validationPrestartMs == 0u) {
            return Result<ValidationReport>::Failure(MakeError(
                    ValidationErrorCategory::InvalidInput,
                    ValidationErrorCode::InvalidArgument,
                    ValidationStage::ContextCreation,
                    ValidationFailureReason::InvalidValidationRequest,
                    identity,
                    "invalid replay validation request"));
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
                context.impl_->state, replayBytes, identity, options);
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

}  // namespace forevervalidator
