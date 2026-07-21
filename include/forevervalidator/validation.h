#ifndef FOREVERVALIDATOR_VALIDATION_H
#define FOREVERVALIDATOR_VALIDATION_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <forevervalidator/result.h>

namespace forevervalidator {

struct ByteView {
    const std::byte *data = nullptr;
    std::size_t size = 0u;
    bool IsValid() const noexcept { return data != nullptr || size == 0u; }
};

using AssetBytes = std::vector<std::byte>;

struct ReplayIdentity { std::string name; };

struct ValidationOptions {
    std::uint32_t requestedSamples = 0xffffffffu;
    std::uint32_t controlTickMs = 10u;
    std::uint32_t validationPrestartMs = 2600u;
};

enum class ValidationStage : std::uint16_t {
    ContextCreation = 1,
    AssetLoading = 2,
    ReplayDecoding = 3,
    ChallengePreload = 4,
    MapConstruction = 5,
    SimulationStartup = 6,
    SimulationStep = 7,
    Observation = 8,
    ValidationEvaluation = 9,
    Serialization = 10,
};

enum class ValidationErrorCategory : std::uint16_t {
    InvalidInput = 1,
    Allocation = 2,
    Asset = 3,
    Replay = 4,
    Simulation = 5,
    Observation = 6,
    Serialization = 7,
    Internal = 8,
};

enum class ValidationErrorCode : std::uint16_t {
    None = 0,
    InvalidArgument = 1,
    AllocationFailed = 2,
    AssetSourceUnavailable = 3,
    AssetLoadingFailed = 4,
    ReplayDecodingFailed = 5,
    ChallengePreloadFailed = 6,
    SimulationDefinitionFailed = 7,
    SimulationFailed = 8,
    ObservationFailed = 9,
    EvaluationFailed = 10,
    DeterministicExecutionUnavailable = 11,
    SerializationFailed = 12,
    UnexpectedFailure = 13,
};

enum class ValidationFailureReason : std::uint16_t {
    None = 0,
    InvalidAssetProvider = 100,
    InvalidAssetIdentifier = 101,
    InvalidAssetSource = 102,
    InvalidValidationContext = 103,
    InvalidValidationRequest = 104,
    EmptyInstalledPackDirectory = 105,
    EmptyReplayPath = 106,
    AllocationFailed = 200,
    AssetProviderFailed = 300,
    RequiredAssetMissing = 301,
    StadiumPackMissing = 302,
    PacklistMissing = 303,
    StadiumPackInvalid = 304,
    DefaultVehicleUnavailable = 305,
    AssetRepositoryUnavailable = 306,
    AssetPathEscapesRoot = 307,
    InstalledPackMissing = 308,
    InstalledPackInvalid = 309,
    UnsupportedMapEnvironmentIdentifier = 310,
    UnsupportedVehicleIdentifier = 311,
    UnsupportedPlayMode = 312,
    ReplayFileOpenFailed = 400,
    ReplayFileLengthInvalid = 401,
    ReplayFileReadFailed = 402,
    ReplayInvalidRequest = 410,
    ReplayInvalidContainer = 411,
    ReplayRootBodyDecompressionFailed = 412,
    ReplayMissingGhostBuffer = 413,
    ReplayMissingInputStream = 414,
    ReplayTooManyInputActions = 415,
    ReplayInvalidGhostMetadata = 416,
    ReplayInvalidInputHeader = 417,
    ReplayInvalidInputActions = 418,
    ReplayInvalidInputEventHeader = 419,
    ReplayInvalidInputEvents = 420,
    ReplayInvalidInputTimeline = 421,
    ReplayMissingGhostState = 422,
    ReplayGhostStateDecompressionFailed = 423,
    ReplayInvalidGhostState = 424,
    ReplayGhostTrajectoryAllocationFailed = 425,
    ReplayMissingEmbeddedChallenge = 426,
    ReplayInvalidEmbeddedChallenge = 427,
    ReplayInvalidMap = 428,
    ChallengeInvalidRequest = 500,
    ChallengeWaterDefinitionFailed = 501,
    ChallengeSceneDefinitionFailed = 502,
    ChallengeConstructionFailed = 503,
    ChallengeStaticSceneFailed = 504,
    SimulationMissingVehicleDefinition = 600,
    SimulationInvalidVehiclePhysics = 601,
    SimulationInvalidInitialParameters = 602,
    SimulationInvalidEnvironment = 603,
    SimulationInvalidMaterials = 604,
    SimulationMissingInput = 700,
    SimulationInvalidPlan = 701,
    SimulationControlPlanInvalidRequest = 702,
    SimulationControlTargetAllocationFailed = 703,
    SimulationControlTargetTimeOutOfRange = 704,
    SimulationControlTargetNonFinite = 705,
    SimulationControlTickReservationFailed = 706,
    SimulationControlTickAllocationFailed = 707,
    SimulationControlTargetMissing = 708,
    SimulationControlOutputMissing = 709,
    SimulationInvalidControlPlan = 710,
    SimulationPhysicsInputInvalid = 711,
    SimulationMapStartUnavailable = 712,
    ObservationAllocationFailed = 713,
    DeterministicExecutionUnavailable = 714,
    DeterministicStateRestoreFailed = 715,
    SerializationFailed = 900,
    UnexpectedFailure = 1000,
};

struct ValidationError {
    ValidationErrorCategory category = ValidationErrorCategory::Internal;
    ValidationErrorCode code = ValidationErrorCode::UnexpectedFailure;
    ValidationStage stage = ValidationStage::ContextCreation;
    ValidationFailureReason reason = ValidationFailureReason::UnexpectedFailure;
    ReplayIdentity replay;
    std::string relatedAsset;
    std::string diagnostic;
};

template<typename T>
using Result = DiscriminatedResult<T, ValidationError>;

struct AssetRequest {
    std::string logicalIdentifier;
};

using AssetProvider =
        std::function<Result<AssetBytes>(const AssetRequest &request)>;

bool IsNormalizedAssetIdentifier(std::string_view identifier) noexcept;

enum class ValidationStatus {
    Valid, ValidPrefix, WrongSimulation, IncompleteStandaloneRun,
    RaceCompletionUnavailable, ExpectingCompletedRace, RaceTimeMismatch,
    StuntsScoreMismatch, RespawnCountMismatch,
    RespawnExpectationUnavailable, ObservationError,
    IncompatibleReplayVersion = 11,
    InputUnavailable = 12,
    TMInterfaceReplay = 13,
};

enum class ValidationOutcome { Invalid, Valid, WrongSimulation, Unavailable, Error };
enum class ObservationError { NonFiniteDistance, ReplayMetadataUnavailable };

enum class ReplayProvenance : std::uint8_t {
    Unmarked,
    TMInterface,
};

enum class InputGhostMatch : std::uint8_t {
    Unavailable,
    Match,
    Mismatch,
};

enum class MapEnvironment : std::uint8_t {
    Unknown,
    Alpine,
    Speed,
    Rally,
    Island,
    Coast,
    Bay,
    Stadium,
};

enum class VehicleModel : std::uint8_t {
    Unknown,
    SnowCar,
    DesertCar,
    RallyCar,
    IslandCar,
    CoastCar,
    BayCar,
    StadiumCar,
};

enum class PlayMode : std::uint8_t {
    Race,
    Platform,
    Puzzle,
    Crazy,
    Shortcut,
    Stunts,
};

struct Vector3 { float x = 0.0f; float y = 0.0f; float z = 0.0f; };
struct ValidationDeviation {
    std::uint32_t comparisonOrdinal = 0u;
    std::int32_t ghostTimeMs = 0;
    std::int32_t simulationTimeMs = 0;
    float distance = 0.0f;
    Vector3 simulatedPosition;
    Vector3 writePosition;
    Vector3 targetPosition;
};
struct ValidationMetadata {
    MapEnvironment mapEnvironment = MapEnvironment::Unknown;
    VehicleModel vehicleModel = VehicleModel::Unknown;
    std::optional<PlayMode> playMode;
    std::optional<std::uint32_t> expectedStuntsScore;
    std::size_t sampleCount = 0u;
    std::uint32_t samplePeriodMs = 0u;
    std::size_t encodedGhostSampleByteCount = 0u;
    std::size_t encodedGhostStateByteCount = 0u;
    std::int32_t inputDurationMs = 0;
    std::optional<std::int32_t> expectedRaceTimeMs;
    std::optional<std::int32_t> expectedRespawns;
    std::uint32_t requestedSamples = 0u;
    std::size_t expectedSamples = 0u;
    std::size_t actionCount = 0u;
    std::size_t eventCount = 0u;
    ReplayProvenance replayProvenance = ReplayProvenance::Unmarked;
};
struct SimulationOutcome {
    std::optional<bool> raceCompleted;
    std::optional<std::int32_t> raceTimeMs;
    std::optional<std::int32_t> stuntsScore;
    std::uint32_t respawnCount = 0u;
};
struct ValidationReport {
    ReplayIdentity replay;
    bool valid = false;
    ValidationStatus status = ValidationStatus::Valid;
    ValidationOutcome outcome = ValidationOutcome::Valid;
    std::uint32_t measuredSamples = 0u;
    std::uint32_t expectedSamples = 0u;
    std::uint32_t comparedExactGhostStateCount = 0u;
    bool wrongSimulation = false;
    std::optional<ValidationDeviation> firstDivergence;
    std::optional<ValidationDeviation> firstExactDeviation;
    float maxDeviation = 0.0f;
    std::optional<std::int32_t> maxDeviationTimeMs;
    float maxDeviationDistance = 0.0f;
    std::optional<ObservationError> observationError;
    ValidationMetadata metadata;
    SimulationOutcome simulation;
    InputGhostMatch inputGhostMatch = InputGhostMatch::Unavailable;
};

class ValidationContext;

class AssetSource {
public:
    AssetSource(AssetSource &&) noexcept;
    AssetSource &operator=(AssetSource &&) noexcept;
    ~AssetSource();
    AssetSource(const AssetSource &) = delete;
    AssetSource &operator=(const AssetSource &) = delete;
private:
    struct Impl;
    explicit AssetSource(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend Result<AssetSource> CreateAssetSource(AssetProvider provider) noexcept;
    friend Result<ValidationContext> CreateValidationContext(
            AssetSource source) noexcept;
};

Result<AssetSource> CreateAssetSource(AssetProvider provider) noexcept;

class ValidationContext {
public:
    ValidationContext(ValidationContext &&) noexcept;
    ValidationContext &operator=(ValidationContext &&) noexcept;
    ~ValidationContext();
    ValidationContext(const ValidationContext &) = delete;
    ValidationContext &operator=(const ValidationContext &) = delete;
private:
    struct Impl;
    explicit ValidationContext(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend Result<ValidationContext> CreateValidationContext(
            AssetSource source) noexcept;
    friend Result<ValidationReport> ValidateReplay(
            ValidationContext &context, ByteView replayBytes,
            const ReplayIdentity &identity,
            const ValidationOptions &options) noexcept;
};

Result<ValidationContext> CreateValidationContext(AssetSource source) noexcept;
Result<ValidationReport> ValidateReplay(
        ValidationContext &context, ByteView replayBytes,
        const ReplayIdentity &identity,
        const ValidationOptions &options = {}) noexcept;

const char *ValidationErrorCategoryName(ValidationErrorCategory category) noexcept;
const char *ValidationErrorCodeName(ValidationErrorCode code) noexcept;
const char *ValidationStageName(ValidationStage stage) noexcept;
const char *ValidationFailureReasonName(ValidationFailureReason reason) noexcept;
const char *MapEnvironmentName(MapEnvironment environment) noexcept;
const char *VehicleModelName(VehicleModel vehicle) noexcept;
const char *PlayModeName(PlayMode mode) noexcept;
int ValidationErrorExitCode(const ValidationError &error) noexcept;
int ValidationLegacyParserCode(ValidationFailureReason reason) noexcept;
int ValidationLegacySimulationCode(ValidationFailureReason reason) noexcept;
int ValidationLegacyCauseCode(ValidationFailureReason reason) noexcept;

}  // namespace forevervalidator

#endif
