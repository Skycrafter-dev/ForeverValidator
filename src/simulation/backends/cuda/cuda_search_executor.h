#ifndef FOREVERVALIDATOR_CUDA_SEARCH_EXECUTOR_H
#define FOREVERVALIDATOR_CUDA_SEARCH_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/backends/cuda/cuda_timeline_executor.h"

namespace forevervalidator::simulation {

namespace cuda::specialization {
class SessionModule;
}

enum class CudaSearchModifierKind : std::uint32_t {
    RandomSteering,
    ExistingEvent,
    SmoothSteering,
    InputInsertion,
    InputDeletion,
};

struct CudaSearchWindow {
    std::int64_t minimumTimeMs = 0;
    std::int64_t maximumTimeMs = 0;
    std::uint32_t seed = 0u;
};

struct CudaSearchChannel {
    std::uint32_t enabled = 0u;
    std::uint32_t minimumCount = 0u;
    std::uint32_t maximumCount = 0u;
    std::int64_t maximumHoldMs = 0;
};

struct CudaSearchModifierConfiguration {
    CudaSearchModifierKind kind =
            CudaSearchModifierKind::RandomSteering;
    CudaSearchWindow window{};
    std::uint32_t minimumCount = 0u;
    std::uint32_t maximumCount = 0u;
    std::int64_t timeParameterMs = 0;
    std::int32_t analogMinimum = 0;
    std::int32_t analogMaximum = 0;
    std::int32_t secondaryAnalogMinimum = 0;
    std::int32_t secondaryAnalogMaximum = 0;
    std::uint32_t optionFlags = 0u;
    std::uint32_t weightOffset = 0u;
    CudaSearchChannel steering{};
    CudaSearchChannel accelerate{};
    CudaSearchChannel brake{};
};

enum class CudaSearchEvaluatorKind : std::uint32_t {
    Velocity,
    Point,
    Pose,
    VolumeEntry,
    StuntPoints,
    FinishTime,
};

struct CudaSearchEvaluatorConfiguration {
    CudaSearchEvaluatorKind kind = CudaSearchEvaluatorKind::FinishTime;
    std::uint32_t optionFlags = 0u;
    double values[10]{};
};

enum class CudaSearchConditionOpcode : std::uint32_t {
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
};

enum class CudaSearchConditionValue : std::uint32_t {
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
};

struct CudaSearchConditionInstruction {
    CudaSearchConditionOpcode opcode = CudaSearchConditionOpcode::Constant;
    CudaSearchConditionValue value = CudaSearchConditionValue::Speed;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct CudaSearchConditionConfiguration {
    std::vector<CudaSearchConditionInstruction> instructions;
    double lastImprovementTimeSeconds = 0.0;
    double lastRestartTimeSeconds = 0.0;
};

struct CudaSearchInputEvent {
    std::int32_t timeMs = 0;
    std::uint32_t action = 0u;
    std::uint32_t valueKind = 0u;
    std::int32_t value = 0;
};

struct CudaSearchIncumbent {
    bool mutation = false;
    std::uint64_t candidateId = 0u;
    std::uint32_t mutationCount = 0u;
    std::uint32_t evaluationTick = 0u;
    double score = 0.0;
    double timeMs = 0.0;
    double detail0 = 0.0;
    double detail1 = 0.0;
    bool preciseFinish = false;
};

struct CudaSearchExecutorConfiguration {
    const void *deviceScene = nullptr;
    const void *deviceStaticConfiguration = nullptr;
    CudaCandidateState branchState{};
    std::vector<CudaControlTick> baselineTicks;
    std::vector<CudaSearchInputEvent> baselineInputs;
    std::vector<CudaSearchModifierConfiguration> modifiers;
    std::vector<double> smoothWeights;
    CudaSearchEvaluatorConfiguration evaluator{};
    std::optional<CudaSearchConditionConfiguration> condition;
    std::uint32_t maximumBatchSize = 1u;
    std::uint32_t tickDurationMs = 10u;
    std::uint32_t prestartDurationMs = 0u;
    std::int64_t branchTimeMs = 0;
    std::int64_t evaluationStartTimeMs = 0;
    std::int64_t evaluationEndTimeMs = 0;
    std::size_t maximumEventCount = 0u;
    bool useLegacyMutationPipelineForTesting = false;
    bool sortCandidatesByLocality = true;
    bool reuseBaselinePrefixes = true;
    bool deduplicateLowEntropyCandidateInputs = true;
    std::uint32_t simulationMinimumBlocksPerMultiprocessorForTesting = 0u;
    bool captureBestState = true;
    std::optional<CudaSearchIncumbent> incumbent;
    std::shared_ptr<const cuda::specialization::SessionModule>
            sessionSpecialization;
};

struct CudaSearchPrefixReusePlan {
    bool enabled = false;
    std::uint64_t lowEntropyChoiceCount = 0u;

    bool DeduplicationStorageEligible(
            std::uint32_t candidateCapacity) const noexcept {
        return enabled && lowEntropyChoiceCount != 0u &&
                lowEntropyChoiceCount <=
                        static_cast<std::uint64_t>(candidateCapacity) / 4u;
    }
};

// Prefix reuse keeps one physics state, winner sample, and target-progress
// value per baseline tick. Bound that cache so unusually long horizons fall
// back to the allocation-free full-timeline path instead of consuming an
// unbounded share of device memory. DeviceSample is fixed at 64 bytes.
inline constexpr std::size_t CudaSearchMaximumBaselinePrefixDeviceBytes =
        256u * 1024u * 1024u;
inline constexpr std::size_t CudaSearchBaselinePrefixSampleBytes = 64u;
inline constexpr std::size_t CudaSearchBaselinePrefixBytesPerTick =
        sizeof(CudaCandidatePhysicsState) +
        CudaSearchBaselinePrefixSampleBytes + sizeof(double);
inline constexpr std::size_t CudaSearchMaximumBaselinePrefixTickCount =
        CudaSearchMaximumBaselinePrefixDeviceBytes /
        CudaSearchBaselinePrefixBytesPerTick;

inline CudaSearchPrefixReusePlan PlanCudaSearchPrefixReuse(
        bool reuseBaselinePrefixes,
        bool stuntsEnabled,
        bool hasConditionInstructions,
        CudaSearchEvaluatorKind evaluatorKind,
        bool deduplicateLowEntropyCandidateInputs,
        std::size_t baselineTickCount,
        std::uint32_t tickDurationMs,
        const CudaSearchModifierConfiguration *modifiers,
        std::size_t modifierCount) noexcept {
    CudaSearchPrefixReusePlan plan;
    plan.enabled = reuseBaselinePrefixes && !stuntsEnabled &&
            !hasConditionInstructions &&
            evaluatorKind != CudaSearchEvaluatorKind::FinishTime &&
            baselineTickCount != 0u &&
            baselineTickCount <=
                    CudaSearchMaximumBaselinePrefixTickCount;
    if (!plan.enabled ||
        !deduplicateLowEntropyCandidateInputs ||
        tickDurationMs == 0u || modifierCount != 1u ||
        modifiers == nullptr) {
        return plan;
    }
    const CudaSearchModifierConfiguration &modifier =
            modifiers[0];
    const bool lowEntropyInsertion =
            modifier.kind == CudaSearchModifierKind::InputInsertion &&
            modifier.steering.enabled != 0u &&
            modifier.steering.minimumCount == 1u &&
            modifier.steering.maximumCount == 1u &&
            modifier.steering.maximumHoldMs == 0 &&
            modifier.accelerate.enabled == 0u &&
            modifier.brake.enabled == 0u &&
            (modifier.optionFlags & 1u) != 0u &&
            modifier.secondaryAnalogMinimum ==
                    modifier.secondaryAnalogMaximum;
    if (!lowEntropyInsertion) {
        return plan;
    }
    const std::int64_t firstChoice =
            modifier.window.minimumTimeMs /
            tickDurationMs;
    const std::int64_t lastChoice =
            modifier.window.maximumTimeMs /
            tickDurationMs;
    if (lastChoice >= firstChoice) {
        plan.lowEntropyChoiceCount =
                static_cast<std::uint64_t>(lastChoice - firstChoice) + 1u;
    }
    return plan;
}

inline CudaSearchPrefixReusePlan PlanCudaSearchPrefixReuse(
        const CudaSearchExecutorConfiguration &configuration) noexcept {
    return PlanCudaSearchPrefixReuse(
            configuration.reuseBaselinePrefixes,
            configuration.branchState.stuntsEnabled,
            configuration.condition &&
                    !configuration.condition->instructions.empty(),
            configuration.evaluator.kind,
            configuration.deduplicateLowEntropyCandidateInputs,
            configuration.baselineTicks.size(),
            configuration.tickDurationMs,
            configuration.modifiers.data(),
            configuration.modifiers.size());
}

inline bool IsCudaSearchSimulationMinimumBlocksValid(
        std::uint32_t minimumBlocks) noexcept {
    return minimumBlocks == 0u || minimumBlocks == 16u ||
            minimumBlocks == 12u || minimumBlocks == 8u;
}

enum class CudaSearchStatus : std::uint32_t {
    Success,
    InvalidArgument,
    UnsupportedConfiguration,
    CapacityExceeded,
    Cancelled,
    DeviceFailure,
    UnsupportedPhysicsTransition,
};

struct CudaSearchBest {
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
    CudaCandidateState state{};
    std::vector<CudaSearchInputEvent> inputs;
};

struct CudaSearchBatchExecution {
    CudaSearchStatus status = CudaSearchStatus::InvalidArgument;
    std::uint64_t firstCandidateId = 0u;
    std::uint32_t candidateCount = 0u;
    std::uint32_t evaluatedCandidateCount = 0u;
    std::uint64_t evaluatorCalls = 0u;
    std::uint64_t qualifyingCandidateCount = 0u;
    std::optional<double> closestTargetDistance;
    std::uint64_t totalMutationCount = 0u;
    // Candidate-best samples that strictly improved the incumbent in
    // logical candidate order.
    std::uint64_t mutationImprovementCount = 0u;
    bool bestChanged = false;
    CudaSearchBest best{};
    std::uint64_t residentDeviceBytes = 0u;
    std::uint64_t mutationDeviceBytes = 0u;
    std::uint64_t candidateInputDeviceBytes = 0u;
    std::uint64_t mutationScratchDeviceBytes = 0u;
    std::uint64_t baselinePrefixDeviceBytes = 0u;
    std::uint64_t candidatePrefixDeviceBytes = 0u;
    std::uint64_t candidateDeduplicationDeviceBytes = 0u;
    // Candidate-best and prefix-best samples, the scan output, and CUB scan
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
    // The minimum-blocks-per-SM launch-bounds variant actually dispatched.
    std::uint32_t simulationSelectedMinimumBlocksPerMultiprocessor = 0u;
    std::uint32_t simulationThreadsPerBlock = 0u;
    std::uint32_t simulationRegistersPerThread = 0u;
    std::uint64_t simulationLocalBytesPerThread = 0u;
    std::uint32_t simulationActiveBlocksPerMultiprocessor = 0u;
    double simulationTheoreticalOccupancy = 0.0;
    std::string diagnostic;
};

class CudaSearchExecutor {
public:
    static std::unique_ptr<CudaSearchExecutor> Create(
            const CudaSearchExecutorConfiguration &configuration,
            std::string *diagnostic) noexcept;

    ~CudaSearchExecutor();
    CudaSearchExecutor(CudaSearchExecutor &&) noexcept;
    CudaSearchExecutor &operator=(CudaSearchExecutor &&) noexcept;
    CudaSearchExecutor(const CudaSearchExecutor &) = delete;
    CudaSearchExecutor &operator=(const CudaSearchExecutor &) = delete;

    CudaSearchBatchExecution EvaluateBaseline() noexcept;
    CudaSearchBatchExecution EvaluateBaseline(
            const std::function<bool()> &cancellationRequested) noexcept;
    CudaSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool cancellationRequested) noexcept;
    CudaSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            const std::function<bool()> &cancellationRequested) noexcept;
    bool ReserveBatchCapacity(
            std::uint32_t candidateCount,
            std::string *diagnostic) noexcept;
    bool UpdateConditionTimes(
            double lastImprovementTimeSeconds,
            double lastRestartTimeSeconds) noexcept;
    std::uint32_t BatchCapacity() const noexcept;

private:
    struct Impl;
    explicit CudaSearchExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

const char *CudaSearchStatusName(CudaSearchStatus status) noexcept;

}  // namespace forevervalidator::simulation

#endif
