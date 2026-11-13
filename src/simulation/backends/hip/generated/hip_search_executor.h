// Generated from src/simulation/backends/cuda/cuda_search_executor.h by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_SEARCH_EXECUTOR_H
#define FOREVERVALIDATOR_HIP_SEARCH_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "simulation/backends/hip/generated/hip_state_layout.h"
#include "simulation/backends/hip/generated/hip_timeline_executor.h"

namespace forevervalidator::simulation {

namespace hip::specialization {
class SessionModule;
}

enum class HipSearchModifierKind : std::uint32_t {
    RandomSteering,
    ExistingEvent,
    SmoothSteering,
    InputInsertion,
    InputDeletion,
};

struct HipSearchWindow {
    std::int64_t minimumTimeMs = 0;
    std::int64_t maximumTimeMs = 0;
    std::uint32_t seed = 0u;
};

struct HipSearchChannel {
    std::uint32_t enabled = 0u;
    std::uint32_t minimumCount = 0u;
    std::uint32_t maximumCount = 0u;
    std::int64_t maximumHoldMs = 0;
};

struct HipSearchModifierConfiguration {
    HipSearchModifierKind kind =
            HipSearchModifierKind::RandomSteering;
    HipSearchWindow window{};
    std::uint32_t minimumCount = 0u;
    std::uint32_t maximumCount = 0u;
    std::int64_t timeParameterMs = 0;
    std::int32_t analogMinimum = 0;
    std::int32_t analogMaximum = 0;
    std::int32_t secondaryAnalogMinimum = 0;
    std::int32_t secondaryAnalogMaximum = 0;
    std::uint32_t optionFlags = 0u;
    std::uint32_t weightOffset = 0u;
    HipSearchChannel steering{};
    HipSearchChannel accelerate{};
    HipSearchChannel brake{};
};

enum class HipSearchEvaluatorKind : std::uint32_t {
    Velocity,
    Point,
    Pose,
    VolumeEntry,
    StuntPoints,
    FinishTime,
    Scripted,
    ConditionTimeEarliest,
    ConditionTimeLatest,
    CheckpointEvent,
    CustomVolumeEntry,
    Distance,
};

enum class HipSearchConditionOpcode : std::uint32_t {
    Constant,
    ConstantVector,
    Scalar,
    Vector,
    Add,
    Subtract,
    Multiply,
    Divide,
    KilometersPerHour,
    Degrees,
    Distance,
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    LogicalAnd,
    NotEqual,
    LogicalOr,
};

enum class HipSearchConditionValue : std::uint32_t {
    Position,
    PreviousPosition,
    Velocity,
    PreviousVelocity,
    LocalVelocity,
    PreviousLocalVelocity,
    AngularVelocity,
    PreviousAngularVelocity,
    Yaw,
    Pitch,
    Roll,
    PreviousYaw,
    PreviousPitch,
    PreviousRoll,
    Speed,
    PreviousSpeed,
    LocalSpeed,
    PreviousLocalSpeed,
    FreeWheeling,
    LateralContact,
    Sliding,
    Gear,
    Rpm,
    TurningRate,
    TurboType,
    TurboBoostFactor,
    WheelGroundContact0,
    WheelGroundContact1,
    WheelGroundContact2,
    WheelGroundContact3,
    WheelSliding0,
    WheelSliding1,
    WheelSliding2,
    WheelSliding3,
    WheelSurface0,
    WheelSurface1,
    WheelSurface2,
    WheelSurface3,
    Iterations,
    LastImprovementTime,
    LastRestartTime,
    CurrentTime,
    CheckpointCount,
    CompletedLaps,
};

struct HipSearchConditionInstruction {
    HipSearchConditionOpcode opcode = HipSearchConditionOpcode::Constant;
    HipSearchConditionValue value = HipSearchConditionValue::Speed;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline constexpr std::uint32_t HipSearchMaximumScriptedObjectives = 16u;
inline constexpr std::uint32_t HipSearchMaximumScriptedInstructions = 256u;
inline constexpr std::uint32_t HipSearchMaximumCustomVolumeVertices = 256u;
inline constexpr std::uint32_t HipSearchMaximumDistanceTriangles = 1024u;
inline constexpr std::uint32_t HipSearchMaximumDistanceShapes = 16u;
inline constexpr std::uint32_t HipSearchMaximumDistanceTriggers = 128u;
// optionFlags bits beyond each kind's own.
inline constexpr std::uint32_t HipSearchVelocityTowardsPoint = 4u;
inline constexpr std::uint32_t HipSearchEntrySpeed = 0x100u;

struct HipSearchScriptedObjective {
    std::uint32_t kind = 0u;
    std::uint32_t firstInstruction = 0u;
    std::uint32_t instructionCount = 0u;
    double target = 0.0;
};

struct HipSearchEvaluatorConfiguration {
    HipSearchEvaluatorKind kind = HipSearchEvaluatorKind::FinishTime;
    std::uint32_t optionFlags = 0u;
    // Velocity: direction 0-2, minimum alignment 3, point 4-6.
    // Pose: position 0-2, rotation 3-6, rotation weight 7, axis weights
    // 8-13. Distance: the CheckpointEvent selectors 0-5, car radius 6.
    double values[16]{};
    std::uint32_t scriptedObjectiveCount = 0u;
    HipSearchScriptedObjective
            scriptedObjectives[HipSearchMaximumScriptedObjectives]{};
    HipSearchConditionInstruction
            scriptedInstructions[HipSearchMaximumScriptedInstructions]{};
    std::uint32_t customVolumeVertexCount = 0u;
    double customVolumeVertices[HipSearchMaximumCustomVolumeVertices][2]{};
    // Distance: triggers (map slot or -1 for a finish line, first
    // triangle, triangle count, box min xyz, box max xyz), their triangles
    // (a, b, c) in world space, and the car's collision shapes, car-local
    // (centre, rotation x y z w, radii).
    std::uint32_t distanceTriggerCount = 0u;
    std::uint32_t distanceTriangleCount = 0u;
    std::uint32_t distanceShapeCount = 0u;
    double distanceTriggers[HipSearchMaximumDistanceTriggers][9]{};
    double distanceTriangles[HipSearchMaximumDistanceTriangles][9]{};
    double distanceShapes[HipSearchMaximumDistanceShapes][10]{};
    // Distance around walls: the tables (device memory, set when uploaded;
    // null measures straight), each trigger's byte offset into them, and
    // the car's body sphere (car-local centre, radius) the tables were
    // built for.
    const unsigned char *aroundWalls = nullptr;
    std::uint64_t aroundWallsOffsets[HipSearchMaximumDistanceTriggers]{};
    double bodyCenter[3]{};
    double bodyRadius = 0.0;
};

struct HipSearchConditionConfiguration {
    std::vector<HipSearchConditionInstruction> instructions;
    double lastImprovementTimeSeconds = 0.0;
    double lastRestartTimeSeconds = 0.0;
};

struct alignas(16) HipSearchInputEvent {
    std::int32_t timeMs = 0;
    std::uint32_t action = 0u;
    std::uint32_t valueKind = 0u;
    std::int32_t value = 0;
};

struct HipSearchIncumbent {
    bool mutation = false;
    std::uint64_t candidateId = 0u;
    std::uint32_t mutationCount = 0u;
    std::uint32_t evaluationTick = 0u;
    double score = 0.0;
    double timeMs = 0.0;
    double detail0 = 0.0;
    double detail1 = 0.0;
    bool preciseFinish = false;
    std::uint32_t scriptedObjectiveCount = 0u;
    double objectiveScores[HipSearchMaximumScriptedObjectives]{};
    double metricValues[HipSearchMaximumScriptedObjectives]{};
};

struct HipSearchMutationSegment {
    std::int64_t minimumTimeMs = 0;
    std::int64_t maximumTimeMs = 0;
};

struct HipSearchExecutorConfiguration {
    const void *deviceScene = nullptr;
    const void *deviceStaticConfiguration = nullptr;
    HipCandidateState branchState{};
    std::vector<HipControlTick> baselineTicks;
    std::vector<HipSearchInputEvent> baselineInputs;
    std::vector<HipSearchModifierConfiguration> modifiers;
    std::vector<HipSearchMutationSegment> mutationSegments;
    std::uint32_t changedSegmentCount = 1u;
    std::vector<double> smoothWeights;
    HipSearchEvaluatorConfiguration evaluator{};
    // The evaluator's around-walls tables, uploaded with it. Corners the
    // batches need are filled on demand (around_walls_table.h); after each
    // fill, aroundWallsFilled (if set) gets the table's index and bytes.
    std::vector<unsigned char> aroundWallsTables;
    std::function<void(std::uint32_t, const unsigned char *)> aroundWallsFilled;
    std::optional<HipSearchConditionConfiguration> condition;
    std::uint32_t maximumBatchSize = 1u;
    std::uint32_t tickDurationMs = 10u;
    std::uint32_t prestartDurationMs = 0u;
    std::int64_t branchTimeMs = 0;
    std::int64_t evaluationStartTimeMs = 0;
    std::int64_t evaluationEndTimeMs = 0;
    std::size_t maximumEventCount = 0u;
    bool useLegacyMutationPipelineForTesting = false;
    bool captureBestState = true;
    std::optional<HipSearchIncumbent> incumbent;
    std::shared_ptr<const hip::specialization::SessionModule>
            sessionSpecialization;
};

enum class HipSearchStatus : std::uint32_t {
    Success,
    InvalidArgument,
    UnsupportedConfiguration,
    CapacityExceeded,
    Cancelled,
    DeviceFailure,
    UnsupportedPhysicsTransition,
};

struct HipSearchBest {
    bool valid = false;
    bool mutation = false;
    bool stateCaptured = false;
    std::uint64_t candidateId = 0u;
    std::uint32_t mutationCount = 0u;
    std::uint32_t evaluationTick = 0u;
    double score = 0.0;
    double timeMs = 0.0;
    double detail0 = 0.0;
    double detail1 = 0.0;
    HipCandidateState state{};
    std::vector<HipSearchInputEvent> inputs;
    std::uint32_t scriptedObjectiveCount = 0u;
    double objectiveScores[HipSearchMaximumScriptedObjectives]{};
    double metricValues[HipSearchMaximumScriptedObjectives]{};
};

struct HipSearchBatchExecution {
    HipSearchStatus status = HipSearchStatus::InvalidArgument;
    std::uint64_t firstCandidateId = 0u;
    double evaluationCurrentTimeSeconds = 0.0;
    std::uint32_t candidateCount = 0u;
    std::uint32_t evaluatedCandidateCount = 0u;
    std::uint64_t evaluatorCalls = 0u;
    std::uint64_t totalMutationCount = 0u;
    // Candidate-best samples that strictly improved the incumbent in
    // logical candidate order.
    std::uint64_t mutationImprovementCount = 0u;
    bool bestChanged = false;
    HipSearchBest best{};
    std::uint64_t residentDeviceBytes = 0u;
    std::uint64_t mutationDeviceBytes = 0u;
    std::uint64_t candidateInputDeviceBytes = 0u;
    std::uint64_t mutationScratchDeviceBytes = 0u;
    // Candidate-best samples, the incumbent/reduction output, and CUB
    // temporary storage. This is independent of evaluationTickCount.
    std::uint64_t winnerSelectionDeviceBytes = 0u;
    std::uint64_t hostToDeviceBytes = 0u;
    std::uint64_t deviceToHostBytes = 0u;
    double kernelMilliseconds = 0.0;
    // Retained name for compatibility; now measures the O(1) incumbent seed.
    double scoreInitializationKernelMilliseconds = 0.0;
    double mutationKernelMilliseconds = 0.0;
    double simulationKernelMilliseconds = 0.0;
    double finishRefinementKernelMilliseconds = 0.0;
    double winnerKernelMilliseconds = 0.0;
    double winnerReductionKernelMilliseconds = 0.0;
    double winnerStateCaptureKernelMilliseconds = 0.0;
    double finalizationKernelMilliseconds = 0.0;
    // Around walls: times the batch ran again after filling the corners it
    // needed (its time includes those runs and fills).
    std::uint32_t aroundWallsReruns = 0u;
    std::uint32_t simulationThreadsPerBlock = 0u;
    std::uint32_t simulationRegistersPerThread = 0u;
    std::uint64_t simulationLocalBytesPerThread = 0u;
    std::uint32_t simulationActiveBlocksPerMultiprocessor = 0u;
    double simulationTheoreticalOccupancy = 0.0;
    std::string diagnostic;
    std::uint64_t reservationBytesPerCandidate = 0u;
};

class HipSearchExecutor {
public:
    static std::unique_ptr<HipSearchExecutor> Create(
            const HipSearchExecutorConfiguration &configuration,
            std::string *diagnostic) noexcept;

    ~HipSearchExecutor();
    HipSearchExecutor(HipSearchExecutor &&) noexcept;
    HipSearchExecutor &operator=(HipSearchExecutor &&) noexcept;
    HipSearchExecutor(const HipSearchExecutor &) = delete;
    HipSearchExecutor &operator=(const HipSearchExecutor &) = delete;

    HipSearchBatchExecution EvaluateBaseline() noexcept;
    HipSearchBatchExecution EvaluateBaseline(
            const std::function<bool()> &cancellationRequested) noexcept;
    HipSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool cancellationRequested) noexcept;
    HipSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            const std::function<bool()> &cancellationRequested) noexcept;
    bool ReserveBatchCapacity(
            std::uint32_t candidateCount,
            std::string *diagnostic) noexcept;
    bool UpdateConditionTimes(
            double lastImprovementTimeSeconds,
            double lastRestartTimeSeconds) noexcept;
    bool UpdateChangedSegmentCount(
            std::uint32_t changedSegmentCount) noexcept;
    std::uint32_t BatchCapacity() const noexcept;

private:
    struct Impl;
    explicit HipSearchExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

const char *HipSearchStatusName(HipSearchStatus status) noexcept;

}  // namespace forevervalidator::simulation

#endif
