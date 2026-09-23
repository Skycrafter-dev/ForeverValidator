#include <forevervalidator/validation.h>

namespace forevervalidator {

const char *ValidationErrorCategoryName(
        ValidationErrorCategory category) noexcept {
    switch (category) {
    case ValidationErrorCategory::InvalidInput: return "invalid_input";
    case ValidationErrorCategory::Allocation: return "allocation";
    case ValidationErrorCategory::Asset: return "asset";
    case ValidationErrorCategory::Replay: return "replay";
    case ValidationErrorCategory::Simulation: return "simulation";
    case ValidationErrorCategory::Observation: return "observation";
    case ValidationErrorCategory::Serialization: return "serialization";
    case ValidationErrorCategory::Internal: return "internal";
    }
    return "internal";
}

const char *ValidationErrorCodeName(ValidationErrorCode code) noexcept {
    switch (code) {
    case ValidationErrorCode::None: return "none";
    case ValidationErrorCode::InvalidArgument: return "invalid_argument";
    case ValidationErrorCode::AllocationFailed: return "allocation_failed";
    case ValidationErrorCode::AssetSourceUnavailable:
        return "asset_source_unavailable";
    case ValidationErrorCode::AssetLoadingFailed: return "asset_loading_failed";
    case ValidationErrorCode::ReplayDecodingFailed:
        return "replay_decoding_failed";
    case ValidationErrorCode::ChallengePreloadFailed:
        return "challenge_preload_failed";
    case ValidationErrorCode::SimulationDefinitionFailed:
        return "simulation_definition_failed";
    case ValidationErrorCode::SimulationFailed: return "simulation_failed";
    case ValidationErrorCode::ObservationFailed: return "observation_failed";
    case ValidationErrorCode::EvaluationFailed: return "evaluation_failed";
    case ValidationErrorCode::DeterministicExecutionUnavailable:
        return "deterministic_execution_unavailable";
    case ValidationErrorCode::SerializationFailed:
        return "serialization_failed";
    case ValidationErrorCode::UnexpectedFailure: return "unexpected_failure";
    case ValidationErrorCode::CudaUnavailable: return "cuda_unavailable";
    case ValidationErrorCode::CudaInitializationFailed:
        return "cuda_initialization_failed";
    case ValidationErrorCode::CudaExecutionFailed:
        return "cuda_execution_failed";
    case ValidationErrorCode::HipUnavailable: return "hip_unavailable";
    case ValidationErrorCode::HipInitializationFailed:
        return "hip_initialization_failed";
    case ValidationErrorCode::HipExecutionFailed:
        return "hip_execution_failed";
    }
    return "unexpected_failure";
}

const char *ValidationStageName(ValidationStage stage) noexcept {
    switch (stage) {
    case ValidationStage::ContextCreation: return "context_creation";
    case ValidationStage::AssetLoading: return "asset_loading";
    case ValidationStage::ReplayDecoding: return "replay_decoding";
    case ValidationStage::ChallengePreload: return "challenge_preload";
    case ValidationStage::MapConstruction: return "map_construction";
    case ValidationStage::SimulationStartup: return "simulation_startup";
    case ValidationStage::SimulationStep: return "simulation_step";
    case ValidationStage::Observation: return "observation";
    case ValidationStage::ValidationEvaluation:
        return "validation_evaluation";
    case ValidationStage::Serialization: return "serialization";
    }
    return "context_creation";
}

const char *ValidationFailureReasonName(
        ValidationFailureReason reason) noexcept {
#define REASON_NAME(value, text) case ValidationFailureReason::value: return text
    switch (reason) {
    REASON_NAME(None, "none");
    REASON_NAME(InvalidAssetProvider, "invalid_asset_provider");
    REASON_NAME(InvalidAssetIdentifier, "invalid_asset_identifier");
    REASON_NAME(InvalidAssetSource, "invalid_asset_source");
    REASON_NAME(InvalidValidationContext, "invalid_validation_context");
    REASON_NAME(InvalidValidationRequest, "invalid_validation_request");
    REASON_NAME(EmptyInstalledPackDirectory, "empty_installed_pack_directory");
    REASON_NAME(EmptyReplayPath, "empty_replay_path");
    REASON_NAME(AllocationFailed, "allocation_failed");
    REASON_NAME(AssetProviderFailed, "asset_provider_failed");
    REASON_NAME(RequiredAssetMissing, "required_asset_missing");
    REASON_NAME(StadiumPackMissing, "stadium_pack_missing");
    REASON_NAME(PacklistMissing, "packlist_missing");
    REASON_NAME(StadiumPackInvalid, "stadium_pack_invalid");
    REASON_NAME(DefaultVehicleUnavailable, "default_vehicle_unavailable");
    REASON_NAME(AssetRepositoryUnavailable, "asset_repository_unavailable");
    REASON_NAME(AssetPathEscapesRoot, "asset_path_escapes_root");
    REASON_NAME(InstalledPackMissing, "installed_pack_missing");
    REASON_NAME(InstalledPackInvalid, "installed_pack_invalid");
    REASON_NAME(UnsupportedMapEnvironmentIdentifier,
                "unsupported_map_environment_identifier");
    REASON_NAME(UnsupportedVehicleIdentifier,
                "unsupported_vehicle_identifier");
    REASON_NAME(UnsupportedPlayMode, "unsupported_play_mode");
    REASON_NAME(ReplayFileOpenFailed, "replay_file_open_failed");
    REASON_NAME(ReplayFileLengthInvalid, "replay_file_length_invalid");
    REASON_NAME(ReplayFileReadFailed, "replay_file_read_failed");
    REASON_NAME(ReplayInvalidRequest, "replay_invalid_request");
    REASON_NAME(ReplayInvalidContainer, "replay_invalid_container");
    REASON_NAME(ReplayRootBodyDecompressionFailed,
                "replay_root_body_decompression_failed");
    REASON_NAME(ReplayMissingGhostBuffer, "replay_missing_ghost_buffer");
    REASON_NAME(ReplayMissingInputStream, "replay_missing_input_stream");
    REASON_NAME(ReplayTooManyInputActions, "replay_too_many_input_actions");
    REASON_NAME(ReplayInvalidGhostMetadata, "replay_invalid_ghost_metadata");
    REASON_NAME(ReplayInvalidInputHeader, "replay_invalid_input_header");
    REASON_NAME(ReplayInvalidInputActions, "replay_invalid_input_actions");
    REASON_NAME(ReplayInvalidInputEventHeader,
                "replay_invalid_input_event_header");
    REASON_NAME(ReplayInvalidInputEvents, "replay_invalid_input_events");
    REASON_NAME(ReplayInvalidInputTimeline, "replay_invalid_input_timeline");
    REASON_NAME(ReplayMissingGhostState, "replay_missing_ghost_state");
    REASON_NAME(ReplayGhostStateDecompressionFailed,
                "replay_ghost_state_decompression_failed");
    REASON_NAME(ReplayInvalidGhostState, "replay_invalid_ghost_state");
    REASON_NAME(ReplayGhostTrajectoryAllocationFailed,
                "replay_ghost_trajectory_allocation_failed");
    REASON_NAME(ReplayMissingEmbeddedChallenge,
                "replay_missing_embedded_challenge");
    REASON_NAME(ReplayInvalidEmbeddedChallenge,
                "replay_invalid_embedded_challenge");
    REASON_NAME(ReplayInvalidMap, "replay_invalid_map");
    REASON_NAME(ChallengeInvalidRequest, "challenge_invalid_request");
    REASON_NAME(ChallengeWaterDefinitionFailed,
                "challenge_water_definition_failed");
    REASON_NAME(ChallengeSceneDefinitionFailed,
                "challenge_scene_definition_failed");
    REASON_NAME(ChallengeConstructionFailed,
                "challenge_construction_failed");
    REASON_NAME(ChallengeStaticSceneFailed,
                "challenge_static_scene_failed");
    REASON_NAME(SimulationMissingVehicleDefinition,
                "simulation_missing_vehicle_definition");
    REASON_NAME(SimulationInvalidVehiclePhysics,
                "simulation_invalid_vehicle_physics");
    REASON_NAME(SimulationInvalidInitialParameters,
                "simulation_invalid_initial_parameters");
    REASON_NAME(SimulationInvalidEnvironment,
                "simulation_invalid_environment");
    REASON_NAME(SimulationInvalidMaterials,
                "simulation_invalid_materials");
    REASON_NAME(SimulationMissingInput, "simulation_missing_input");
    REASON_NAME(SimulationInvalidPlan, "simulation_invalid_plan");
    REASON_NAME(SimulationControlPlanInvalidRequest,
                "simulation_control_plan_invalid_request");
    REASON_NAME(SimulationControlTargetAllocationFailed,
                "simulation_control_target_allocation_failed");
    REASON_NAME(SimulationControlTargetTimeOutOfRange,
                "simulation_control_target_time_out_of_range");
    REASON_NAME(SimulationControlTargetNonFinite,
                "simulation_control_target_non_finite");
    REASON_NAME(SimulationControlTickReservationFailed,
                "simulation_control_tick_reservation_failed");
    REASON_NAME(SimulationControlTickAllocationFailed,
                "simulation_control_tick_allocation_failed");
    REASON_NAME(SimulationControlTargetMissing,
                "simulation_control_target_missing");
    REASON_NAME(SimulationControlOutputMissing,
                "simulation_control_output_missing");
    REASON_NAME(SimulationInvalidControlPlan,
                "simulation_invalid_control_plan");
    REASON_NAME(SimulationPhysicsInputInvalid,
                "simulation_physics_input_invalid");
    REASON_NAME(SimulationMapStartUnavailable,
                "simulation_map_start_unavailable");
    REASON_NAME(ObservationAllocationFailed,
                "observation_allocation_failed");
    REASON_NAME(DeterministicExecutionUnavailable,
                "deterministic_execution_unavailable");
    REASON_NAME(DeterministicStateRestoreFailed,
                "deterministic_state_restore_failed");
    REASON_NAME(CudaNotCompiled, "cuda_not_compiled");
    REASON_NAME(CudaRuntimeUnavailable, "cuda_runtime_unavailable");
    REASON_NAME(CudaDeviceUnavailable, "cuda_device_unavailable");
    REASON_NAME(CudaDeviceUnsupported, "cuda_device_unsupported");
    REASON_NAME(CudaInitializationFailed, "cuda_initialization_failed");
    REASON_NAME(CudaExecutionFailed, "cuda_execution_failed");
    REASON_NAME(CudaUnsupportedSimulationScope,
                "cuda_unsupported_simulation_scope");
    REASON_NAME(HipNotCompiled, "hip_not_compiled");
    REASON_NAME(HipRuntimeUnavailable, "hip_runtime_unavailable");
    REASON_NAME(HipDeviceUnavailable, "hip_device_unavailable");
    REASON_NAME(HipDeviceUnsupported, "hip_device_unsupported");
    REASON_NAME(HipInitializationFailed, "hip_initialization_failed");
    REASON_NAME(HipExecutionFailed, "hip_execution_failed");
    REASON_NAME(HipUnsupportedSimulationScope,
                "hip_unsupported_simulation_scope");
    REASON_NAME(SerializationFailed, "serialization_failed");
    REASON_NAME(UnexpectedFailure, "unexpected_failure");
    }
#undef REASON_NAME
    return "unexpected_failure";
}

const char *MapEnvironmentName(MapEnvironment environment) noexcept {
    switch (environment) {
    case MapEnvironment::Alpine: return "Alpine";
    case MapEnvironment::Speed: return "Speed";
    case MapEnvironment::Rally: return "Rally";
    case MapEnvironment::Island: return "Island";
    case MapEnvironment::Coast: return "Coast";
    case MapEnvironment::Bay: return "Bay";
    case MapEnvironment::Stadium: return "Stadium";
    case MapEnvironment::Unknown: return "Unknown";
    }
    return "Unknown";
}

const char *VehicleModelName(VehicleModel vehicle) noexcept {
    switch (vehicle) {
    case VehicleModel::SnowCar: return "SnowCar";
    case VehicleModel::DesertCar: return "DesertCar";
    case VehicleModel::RallyCar: return "RallyCar";
    case VehicleModel::IslandCar: return "IslandCar";
    case VehicleModel::CoastCar: return "CoastCar";
    case VehicleModel::BayCar: return "BayCar";
    case VehicleModel::StadiumCar: return "StadiumCar";
    case VehicleModel::Unknown: return "Unknown";
    }
    return "Unknown";
}

const char *PlayModeName(PlayMode mode) noexcept {
    switch (mode) {
    case PlayMode::Race: return "Race";
    case PlayMode::Platform: return "Platform";
    case PlayMode::Puzzle: return "Puzzle";
    case PlayMode::Crazy: return "Crazy";
    case PlayMode::Shortcut: return "Shortcut";
    case PlayMode::Stunts: return "Stunts";
    }
    return "Unknown";
}

int ValidationLegacyParserCode(ValidationFailureReason reason) noexcept {
    switch (reason) {
    case ValidationFailureReason::ReplayInvalidRequest: return 1;
    case ValidationFailureReason::ReplayFileReadFailed: return 2;
    case ValidationFailureReason::ReplayInvalidContainer: return 3;
    case ValidationFailureReason::AllocationFailed: return 4;
    case ValidationFailureReason::ReplayRootBodyDecompressionFailed: return 5;
    case ValidationFailureReason::ReplayMissingGhostBuffer: return 6;
    case ValidationFailureReason::ReplayMissingInputStream: return 7;
    case ValidationFailureReason::ReplayTooManyInputActions: return 8;
    case ValidationFailureReason::ReplayInvalidGhostMetadata: return 9;
    case ValidationFailureReason::ReplayInvalidInputHeader: return 10;
    case ValidationFailureReason::ReplayInvalidInputActions: return 11;
    case ValidationFailureReason::ReplayInvalidInputEventHeader: return 12;
    case ValidationFailureReason::ReplayInvalidInputEvents: return 13;
    case ValidationFailureReason::ReplayInvalidInputTimeline: return 14;
    case ValidationFailureReason::ReplayMissingGhostState: return 15;
    case ValidationFailureReason::ReplayGhostStateDecompressionFailed: return 16;
    case ValidationFailureReason::ReplayInvalidGhostState: return 17;
    case ValidationFailureReason::ReplayGhostTrajectoryAllocationFailed: return 18;
    case ValidationFailureReason::ReplayMissingEmbeddedChallenge: return 19;
    case ValidationFailureReason::ReplayInvalidEmbeddedChallenge: return 20;
    case ValidationFailureReason::ReplayInvalidMap: return 21;
    default: return -1;
    }
}

int ValidationLegacySimulationCode(ValidationFailureReason) noexcept {
    return -1;
}

int ValidationLegacyCauseCode(ValidationFailureReason reason) noexcept {
    switch (reason) {
    case ValidationFailureReason::ChallengeInvalidRequest: return 1;
    case ValidationFailureReason::AllocationFailed: return 2;
    case ValidationFailureReason::ChallengeWaterDefinitionFailed: return 3;
    case ValidationFailureReason::ChallengeSceneDefinitionFailed: return 4;
    case ValidationFailureReason::ChallengeConstructionFailed: return 5;
    case ValidationFailureReason::ChallengeStaticSceneFailed: return 6;
    case ValidationFailureReason::SimulationMissingVehicleDefinition: return 2;
    case ValidationFailureReason::SimulationInvalidVehiclePhysics: return 6;
    case ValidationFailureReason::SimulationInvalidInitialParameters: return 27;
    case ValidationFailureReason::SimulationInvalidEnvironment: return 47;
    case ValidationFailureReason::SimulationInvalidMaterials: return 67;
    case ValidationFailureReason::SimulationMissingInput: return 1;
    case ValidationFailureReason::SimulationInvalidPlan: return 2;
    case ValidationFailureReason::SimulationControlPlanInvalidRequest: return 3;
    case ValidationFailureReason::SimulationControlTargetAllocationFailed: return 4;
    case ValidationFailureReason::SimulationControlTargetTimeOutOfRange: return 5;
    case ValidationFailureReason::SimulationControlTargetNonFinite: return 6;
    case ValidationFailureReason::SimulationControlTickReservationFailed: return 7;
    case ValidationFailureReason::SimulationControlTickAllocationFailed: return 8;
    case ValidationFailureReason::SimulationControlTargetMissing: return 9;
    case ValidationFailureReason::SimulationControlOutputMissing: return 10;
    case ValidationFailureReason::SimulationInvalidControlPlan: return 11;
    case ValidationFailureReason::SimulationPhysicsInputInvalid: return 12;
    case ValidationFailureReason::SimulationMapStartUnavailable: return 13;
    case ValidationFailureReason::ObservationAllocationFailed: return 14;
    case ValidationFailureReason::DeterministicExecutionUnavailable: return 15;
    default: return -1;
    }
}

int ValidationErrorExitCode(const ValidationError &error) noexcept {
    switch (error.code) {
    case ValidationErrorCode::None: return 0;
    case ValidationErrorCode::InvalidArgument: return 64;
    case ValidationErrorCode::AssetSourceUnavailable:
    case ValidationErrorCode::AssetLoadingFailed: return 69;
    case ValidationErrorCode::ReplayDecodingFailed: return 2;
    case ValidationErrorCode::ChallengePreloadFailed: return 24;
    case ValidationErrorCode::SimulationDefinitionFailed:
        switch (error.reason) {
        case ValidationFailureReason::SimulationMissingVehicleDefinition: return 2;
        case ValidationFailureReason::SimulationInvalidVehiclePhysics: return 6;
        case ValidationFailureReason::SimulationInvalidInitialParameters: return 27;
        case ValidationFailureReason::SimulationInvalidEnvironment: return 47;
        case ValidationFailureReason::SimulationInvalidMaterials: return 67;
        default: return 5;
        }
    case ValidationErrorCode::SimulationFailed:
    case ValidationErrorCode::ObservationFailed:
    case ValidationErrorCode::EvaluationFailed:
        switch (error.reason) {
        case ValidationFailureReason::SimulationMissingInput: return 2;
        case ValidationFailureReason::SimulationInvalidPlan:
        case ValidationFailureReason::SimulationControlPlanInvalidRequest:
        case ValidationFailureReason::ObservationAllocationFailed: return 3;
        case ValidationFailureReason::SimulationControlTargetAllocationFailed:
        case ValidationFailureReason::SimulationPhysicsInputInvalid:
        case ValidationFailureReason::DeterministicExecutionUnavailable: return 5;
        case ValidationFailureReason::SimulationControlTargetTimeOutOfRange:
        case ValidationFailureReason::SimulationInvalidControlPlan: return 7;
        case ValidationFailureReason::SimulationControlTargetNonFinite: return 8;
        case ValidationFailureReason::SimulationMapStartUnavailable: return 9;
        case ValidationFailureReason::SimulationControlTickReservationFailed: return 14;
        case ValidationFailureReason::SimulationControlTickAllocationFailed: return 15;
        case ValidationFailureReason::SimulationControlTargetMissing: return 16;
        case ValidationFailureReason::SimulationControlOutputMissing: return 18;
        default: return 3;
        }
    case ValidationErrorCode::DeterministicExecutionUnavailable: return 5;
    case ValidationErrorCode::CudaUnavailable: return 69;
    case ValidationErrorCode::CudaInitializationFailed: return 70;
    case ValidationErrorCode::CudaExecutionFailed: return 71;
    case ValidationErrorCode::HipUnavailable: return 69;
    case ValidationErrorCode::HipInitializationFailed: return 70;
    case ValidationErrorCode::HipExecutionFailed: return 71;
    case ValidationErrorCode::SerializationFailed: return 22;
    case ValidationErrorCode::AllocationFailed:
        return error.stage == ValidationStage::Serialization ? 22 : 5;
    case ValidationErrorCode::UnexpectedFailure: return 70;
    }
    return 70;
}

}  // namespace forevervalidator
