#ifndef FOREVERVALIDATOR_CUDA_STATE_LAYOUT_H
#define FOREVERVALIDATOR_CUDA_STATE_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <forevervalidator/finish_time.h>

#include "engine/game/trackmania_race.h"
#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_physics_world.h"

struct ReplaySimulationInstanceClone;

namespace forevervalidator::simulation {

enum class CudaStateConversionResult : std::uint8_t {
    Success,
    InvalidArgument,
    SchemaMismatch,
    WheelOverflow,
    CollisionReplacementOverflow,
    CheckpointOverflow,
    StuntEventOverflow,
    AllocationFailed,
};

template<typename T, std::size_t Capacity>
struct CudaFixedArray {
    std::uint32_t count = 0u;
    T values[Capacity]{};
};

constexpr std::size_t CudaCollisionReplacementInlineCapacity = 1u;
constexpr std::size_t CudaCollisionReplacementOverflowCapacity = 511u;
constexpr std::size_t CudaCollisionReplacementCapacity =
        CudaCollisionReplacementInlineCapacity +
        CudaCollisionReplacementOverflowCapacity;
constexpr std::size_t CudaCheckpointSlotCapacity = 1024u;

struct CudaCheckpointSlots {
    static constexpr std::size_t WordBits = 32u;
    static constexpr std::size_t WordCount =
            CudaCheckpointSlotCapacity / WordBits;

    std::uint32_t count = 0u;
    std::uint32_t words[WordCount]{};

#if defined(__CUDACC__)
    __host__ __device__ bool Get(std::uint32_t index) const {
        return (words[index / WordBits] &
                (1u << (index % WordBits))) != 0u;
    }

    __host__ __device__ void Set(std::uint32_t index) {
        words[index / WordBits] |= 1u << (index % WordBits);
    }

    __host__ __device__ void Clear() {
        for (std::size_t index = 0u; index < WordCount; ++index) {
            words[index] = 0u;
        }
    }
#endif
};

template<typename T>
struct CudaOptional {
    bool present = false;
    T value{};
};

struct CudaWheelState {
    bool killsLateralSpeedOnContact = false;
    std::uint32_t axle = 0u;
    float rollingRadius = 0.0f;
    GmIso4 restPose{};
    GmIso4 currentPose{};
    GmVec3 forceApplicationPoint{};
    CSceneVehicleCar::SSimulationWheel::SRealTimeState realTime{};
    CSceneVehicleCar::SSimulationWheel::SState previousPhysics{};
    CSceneVehicleCar::SSimulationWheel::SState currentPhysics{};
    bool surfaceMovedByUpdate = false;
};

struct CudaVehicleCarFrameState {
    float forwardSpeed = 0.0f;
    float sideSpeed = 0.0f;
    float steeringControl = 0.0f;
    float lowSpeedGateA = 0.0f;
    float lowSpeedGateB = 0.0f;
    std::uint32_t turboActive = 0u;
    float turboProgressRatio = 0.0f;
    std::uint32_t wheelSpeedOverrideActive = 0u;
    float surfaceFeedbackAccumulator = 0.0f;
    float feedbackSideSpringValue = 0.0f;
    float feedbackForwardSpringValue = 0.0f;
    float feedbackRamp1 = 0.0f;
    float feedbackRamp0 = 0.0f;
    GmIso4 corpusIso{};
    std::uint32_t vehicleEvent0Value = 1u;
    std::uint32_t waterSplashEventCounter = 1u;
    GmVec3 localLinearSpeed{};
    float materialFeedbackSpeed = 0.0f;
    float materialFeedbackIntensity = 0.0f;
    float engineInputMemory = 0.0f;
    bool airControlRefreshMemory = false;
    std::uint32_t engineControlState = 0u;
    std::uint32_t shiftDirection = 0u;
    bool hasWheelContact = false;
    bool hasBodyContact = false;
    float bodyContactVerticalAngle = 0.0f;
    bool bodyContactZPositive = false;
    float bodyContactHorizontalAngle = 0.0f;
    bool noGroundFrictionGuard = false;
};

struct CudaFrameHistory {
    CudaVehicleCarFrameState physicsPrevious{};
    CudaVehicleCarFrameState physicsCurrent{};
};

struct CudaVehicleState {
    CSceneMobil::RuntimeClone mobil{};
    CSceneVehicle::SEventSlot vehicleEvents[2]{};
    CSceneVehicle::SWaterState water{};

    CudaFixedArray<CudaWheelState, 4u> wheels{};
    CSceneVehicleCar::SControls controls{};
    CSceneVehicleCar::SFeedback feedback{};
    float linearSpeedCap = 0.0f;
    CSceneVehicleCar::SIntegration integration{};
    CudaFrameHistory frameHistory{};
    CSceneVehicleCar::SEngine engine{};
    float reverseGearSpeedThreshold = 0.0f;
    CSceneVehicleCar::STurbo turbo{};
    CSceneVehicleCar::SAirControl airControl{};
    CSceneVehicleCar::SContacts contacts{};
    CSceneVehicleCar::SRadiusSteeringState radiusSteering{};
    CSceneVehicleCar::SSlipMemoryState slipMemory{};
    CSceneVehicleCar::SGearedDriveState gearedDrive{};
    std::uint32_t lastComputeForcesTick = 0u;
    GmSpring<float> dynaPartSprings[4]{};
    CSceneVehicleCar::SForceAccumulators forceAccumulators{};
};

#if defined(__CUDACC__)
namespace cuda::facts {

__device__ inline std::uint32_t WheelCount(
        const CudaVehicleState &vehicle) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_FOUR_WHEELS)
    return 4u;
#else
    return vehicle.wheels.count;
#endif
}

}  // namespace cuda::facts
#endif

struct CudaWheelPassthroughState {
    CSceneVehicleCar::SSimulationWheel::SState previousAsync{};
    CSceneVehicleCar::SSimulationWheel::SState currentAsync{};
};

struct CudaVehiclePassthroughState {
    bool updateAsync = true;
    bool networked = false;
    std::uint32_t predictionDelayTicks = 0u;
    CudaOptional<CSceneVehicle::SStateSampleWindow> stateSampleWindow{};
    float asyncPeriodSeconds = 0.0f;
    CudaVehicleCarFrameState asyncCurrent{};
    CudaVehicleCarFrameState asyncPrevious{};
    CudaWheelPassthroughState wheels[4]{};
};

struct CudaDynamicBodyState {
    CudaOptional<float> maxAngularSpeed{};
    CHmsDynaParams parameters{};
    CPlugPhysicalParameters physicalParameters{};
    GmIso4 corpusLocalIso{};
    CHmsDyna::CHmsStateDyna temporary{};
    CHmsDyna::CHmsStateDyna write{};
    CHmsDyna::CHmsStateDyna current{};
    CudaFixedArray<
            GmVec3,
            CudaCollisionReplacementInlineCapacity>
            collisionReplacements{};
    bool dynamicActive = false;
    std::uint32_t dynamicType = 0u;
};

struct CudaRacePhysicsState {
    CTrackManiaPlayer::RuntimeClone player{};
    CudaCheckpointSlots checkpointSlotsPassed{};
    CudaOptional<GmIso4> playerSpawnLocation{};
    CudaOptional<GmIso4> lastAcceptedSpawnLocation{};
    bool currentSpawnLocationInitialized = false;
    std::uint32_t preparedEventTimeMs = 0u;
    std::uint32_t replayPlayMode = 0u;
    std::uint32_t replayNbLaps = 1u;
    ReplayRaceProgress progress{};
};

struct CudaStuntState {
    bool replayStuntsEnabled = false;
    bool replayStuntStateAvailable = false;
    std::uint32_t replayStuntsTimeLimitMs = 0u;
    std::uint32_t replayStuntsRaceStartTimeMs = 0u;
    ReplayStuntSimulationState replayStuntState{};
    CTrackManiaRace::ReplayStuntInputSnapshot stuntInputHistory[32]{};
    std::uint32_t stuntInputHistorySize = 0u;
    GmIso4 stuntLocationHistory[20]{};
    std::uint32_t stuntLocationHistorySize = 0u;
    GmIso4 stuntPreviousLocation{};
    GmIso4 stuntTakeoffLocation{};
    GmVec3 stuntRotation{};
    float stuntLandingDirection = 0.0f;
    std::uint32_t stuntTakeoffTick = UINT32_MAX;
    std::uint32_t stuntLandingTick = UINT32_MAX;
    std::uint32_t stuntPreviousLandingTick = UINT32_MAX;
    std::uint32_t stuntChain = 0u;
    std::uint32_t stuntComboWindowMs = 0u;
    bool stuntInProgress = false;
    bool stuntMasterJump = false;
    bool stuntBadLanding = false;
    CudaOptional<std::uint32_t> stuntScoreAtTimeLimit{};
    std::uint32_t stuntFigureScores[39]{};
    std::uint32_t stuntsScore = 0u;
};

struct CudaRaceState : CudaRacePhysicsState {
    CudaStuntState stunts{};
    CudaFixedArray<ReplayStuntEvent, 2048u> stuntEvents{};
};

struct CudaCandidatePhysicsState {
    static constexpr std::uint32_t SchemaVersion = 11u;

    std::uint32_t schemaVersion = SchemaVersion;
    std::uint32_t candidateId = 0u;
    std::uint32_t validationSeed = 0u;
    std::uint32_t randomState = 1u;
    std::uint64_t controlCursor = 0u;
    ReplayPhysicsWorld::RuntimeClone world{};
    CudaDynamicBodyState body{};
    CudaVehicleState vehicle{};
    CudaRacePhysicsState race{};
    std::uint32_t incrementalRespawnCount = 0u;
    CudaOptional<forevervalidator::FinishTimeEstimate> finishTime{};
    bool firstStep = true;
    bool stuntsEnabled = false;
    std::uint8_t reserved[5]{};
};

struct CudaCandidateState : CudaCandidatePhysicsState {
    CudaVehiclePassthroughState vehiclePassthrough{};
    CudaStuntState stunts{};
    CudaFixedArray<ReplayStuntEvent, 2048u> stuntEvents{};
    CudaFixedArray<
            GmVec3,
            CudaCollisionReplacementOverflowCapacity>
            collisionReplacementOverflow{};
};

static_assert(std::is_standard_layout_v<CudaCandidatePhysicsState>);
static_assert(std::is_trivially_copyable_v<CudaCandidatePhysicsState>);
static_assert(std::is_trivially_copyable_v<CudaRaceState>);
static_assert(sizeof(CudaCandidatePhysicsState) < 8u * 1024u);
static_assert(std::is_trivially_copyable_v<CudaCandidateState>);
static_assert(sizeof(CudaCandidateState) < 192u * 1024u);

CudaStateConversionResult EncodeCudaCandidateState(
        const ReplaySimulationInstanceClone &source,
        std::uint32_t validationSeed,
        std::uint64_t controlCursor,
        std::uint32_t candidateId,
        std::uint32_t randomState,
        CudaCandidateState *destination) noexcept;

CudaStateConversionResult EncodeCudaRaceState(
        const CTrackManiaRace::RuntimeClone &source,
        CudaRaceState *destination) noexcept;

CudaStateConversionResult DecodeCudaRaceState(
        const CudaRaceState &source,
        CTrackManiaRace::RuntimeClone *destination) noexcept;

CudaStateConversionResult DecodeCudaCandidateState(
        const CudaCandidateState &source,
        ReplaySimulationInstanceClone *destination) noexcept;

}  // namespace forevervalidator::simulation

#endif
