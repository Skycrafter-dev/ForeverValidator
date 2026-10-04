#include "simulation/backends/vulkan/vulkan_search_executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#include "simulation/backends/cuda/cuda_collision_layout.h"
#include "simulation/backends/cuda/cuda_search_branch_state.cuh"
#include "simulation/backends/cuda/cuda_search_winner_selection.cuh"
#include "simulation/backends/vulkan/vulkan_compute_runtime.h"
#include "simulation/backends/vulkan/vulkan_timeline_executor.h"

namespace forevervalidator::simulation {
namespace {

using cuda_search_detail::DeviceControlState;
using cuda_search_detail::DeviceSample;
using cuda_search_detail::InvalidCandidateSlot;
using cuda_search_detail::StrictlyBetter;

// Keep each long-running search invocation comfortably below common GPU
// watchdog limits. The push-constant tick index carries the chunk base for
// the generate, initialize, and combined search kernels.
constexpr std::uint32_t SearchCandidateDispatchSize = 1024u;
constexpr std::uint32_t GenericSearchTickDispatchSize = 8u;
constexpr std::uint32_t SteadyVelocityTickDispatchSize = 100u;

struct VulkanCollisionSearchScratch {
    std::uint32_t collisionCount = 0u;
    std::uint32_t shapeCollisionCount = 0u;
    std::uint32_t surfaceHitCount = 0u;
    std::uint8_t overflow = 0u;
    std::uint8_t surfaceCacheEnabled = 0u;
    std::uint8_t padding0[2]{};
    std::uint64_t collisionStorage = 0u;
    std::uint64_t shapeCollisionStorage = 0u;
    std::uint64_t shapeWorldStorage = 0u;
    std::uint64_t movingBoundsStorage = 0u;
    std::uint64_t unifiedMovingBoundsStorage = 0u;
    std::uint64_t surfaceHitStorage = 0u;
    std::uint64_t meshRangeStorage = 0u;
    std::uint64_t meshCellStorage = 0u;
    std::uint32_t slot = 0u;
    std::uint32_t stride = 0u;
    std::uint32_t shapeCapacity = 0u;
    std::uint32_t overflowReason = 0u;
    std::uint8_t surfaceCacheValid = 0u;
    std::uint8_t padding1[3]{};
    std::uint32_t meshCellCount = 0u;
    std::uint8_t meshCacheValid = 0u;
    std::uint8_t padding2[3]{};
    std::uint32_t replacementOverflowCount = 0u;
    float replacementSumX = 0.0f;
    float replacementSumY = 0.0f;
    float replacementSumZ = 0.0f;
    std::uint8_t padding3[4]{};
    std::uint64_t responseOrderStorage = 0u;
};

static_assert(offsetof(VulkanCollisionSearchScratch, collisionStorage) == 16u);
static_assert(offsetof(VulkanCollisionSearchScratch, slot) == 80u);
static_assert(offsetof(VulkanCollisionSearchScratch, responseOrderStorage) == 128u);
static_assert(sizeof(VulkanCollisionSearchScratch) == 136u);

struct VulkanSearchParameters {
    std::uint64_t scene = 0u;
    std::uint64_t configuration = 0u;
    std::uint64_t branchState = 0u;
    std::uint64_t baselineTicks = 0u;
    std::uint64_t evaluator = 0u;
    std::uint64_t condition = 0u;
    std::uint64_t candidateStates = 0u;
    std::uint64_t candidateBestStates = 0u;
    std::uint64_t collisionScratch = 0u;
    std::uint64_t collisionStorage = 0u;
    std::uint64_t shapeCollisionStorage = 0u;
    std::uint64_t shapeWorldStorage = 0u;
    std::uint64_t movingBoundsStorage = 0u;
    std::uint64_t unifiedMovingBoundsStorage = 0u;
    std::uint64_t surfaceHitStorage = 0u;
    std::uint64_t meshRangeStorage = 0u;
    std::uint64_t meshCellStorage = 0u;
    std::uint64_t responseOrderStorage = 0u;
    std::uint64_t controlStates = 0u;
    std::uint64_t eventCursors = 0u;
    std::uint64_t simulationActive = 0u;
    std::uint64_t evaluatorReported = 0u;
    std::uint64_t previousPositions = 0u;
    std::uint64_t evaluationIndices = 0u;
    std::uint64_t timelineDescriptors = 0u;
    std::uint64_t candidateTicks = 0u;
    std::uint64_t timelineResults = 0u;
    std::uint64_t baselineInputs = 0u;
    std::uint64_t modifiers = 0u;
    std::uint64_t smoothWeights = 0u;
    std::uint64_t mutableBoundaryControls = 0u;
    std::uint64_t candidateBestSamples = 0u;
    std::uint64_t randomStateWords = 0u;
    std::uint64_t candidateEvents = 0u;
    std::uint64_t temporaryEvents = 0u;
    std::uint64_t passBaselineEvents = 0u;
    std::uint64_t eligibleIndices = 0u;
    std::uint64_t eventCounts = 0u;
    std::uint64_t mutationCounts = 0u;
    std::uint64_t statuses = 0u;
    std::uint64_t activeCandidates = 0u;
    std::uint64_t cancellation = 0u;
    std::uint64_t firstCandidateId = 0u;
    std::int64_t branchTimeMs = 0;
    std::int64_t mutableFromTimeMs = 0;
    std::int64_t evaluationStartTimeMs = 0;
    double lastImprovementTimeSeconds = 0.0;
    double lastRestartTimeSeconds = 0.0;
    double currentTimeSeconds = 0.0;
    std::uint32_t baselineInputCount = 0u;
    std::uint32_t immutableTailInputCount = 0u;
    std::uint32_t modifierCount = 0u;
    std::uint32_t tickDurationMs = 0u;
    std::uint32_t candidateCount = 0u;
    std::uint32_t eventCapacity = 0u;
    std::uint32_t baseline = 0u;
    std::uint32_t legacyMutationPipeline = 0u;
    std::uint32_t baselineInputsCanonical = 0u;
    std::uint32_t timelineTickCount = 0u;
    std::uint32_t conditionInstructionCount = 0u;
    std::uint32_t prestartDurationMs = 0u;
    std::uint32_t evaluationTickCount = 0u;
    std::uint32_t simulateStunts = 0u;
    std::uint32_t stateWordCount = 0u;
    std::uint32_t scratchStride = 0u;
    std::uint32_t shapeCapacity = 0u;
    std::uint32_t padding0 = 0u;
    std::uint64_t winnerSummary = 0u;
    std::uint64_t winnerEvents = 0u;
};

static_assert(offsetof(VulkanSearchParameters, firstCandidateId) == 336u);
static_assert(offsetof(VulkanSearchParameters, baselineInputCount) == 392u);
static_assert(offsetof(VulkanSearchParameters, winnerSummary) == 464u);
static_assert(sizeof(VulkanSearchParameters) == 480u);

struct VulkanSearchBatchSummary {
    DeviceSample winner{};
    std::uint64_t totalMutationCount = 0u;
    std::uint64_t mutationImprovementCount = 0u;
    std::uint32_t winnerSlot = InvalidCandidateSlot;
    std::uint32_t winnerEventCount = 0u;
    std::uint32_t winnerMutationCount = 0u;
    std::uint32_t evaluatedCandidateCount = 0u;
    CudaSearchStatus status = CudaSearchStatus::Success;
    std::uint32_t bestChanged = 0u;
};

static_assert(sizeof(DeviceSample) == 328u);
static_assert(offsetof(VulkanSearchBatchSummary, totalMutationCount) == 328u);
static_assert(sizeof(VulkanSearchBatchSummary) == 368u);

struct VulkanSearchTimelineDescriptor {
    std::uint64_t firstTick = 0u;
    std::uint32_t tickCount = 0u;
    std::uint32_t padding0 = 0u;
    std::uint64_t firstObservation = 0u;
    std::uint32_t observationCapacity = 0u;
    std::uint32_t padding1 = 0u;
};

struct VulkanSearchTimelineResult {
    VulkanTimelineStatus status = VulkanTimelineStatus::InvalidArgument;
    std::uint32_t failureTick = UINT32_MAX;
    std::uint32_t failureDetail = 0u;
    std::uint32_t executedTickCount = 0u;
    std::uint32_t executedRespawnCount = 0u;
    std::uint32_t observationCount = 0u;
};

static_assert(sizeof(VulkanSearchTimelineDescriptor) == 32u);
static_assert(sizeof(VulkanSearchTimelineResult) == 24u);
static_assert(sizeof(CudaCandidateState) % sizeof(std::uint32_t) == 0u);

bool CanonicalInputs(const std::vector<CudaSearchInputEvent> &inputs) {
    std::int32_t previousTime = std::numeric_limits<std::int32_t>::min();
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const CudaSearchInputEvent &input = inputs[index];
        if (input.timeMs < 0 || input.timeMs < previousTime ||
            (input.valueKind == 2u &&
             (input.value < -65536 || input.value > 65536)) ||
            (input.valueKind == 1u &&
             input.value != 0 && input.value != 1)) {
            return false;
        }
        for (std::size_t duplicate = index;
             duplicate != 0u &&
             inputs[duplicate - 1u].timeMs == input.timeMs;
             --duplicate) {
            if (inputs[duplicate - 1u].action == input.action) {
                return false;
            }
        }
        previousTime = input.timeMs;
    }
    return true;
}

template <typename T>
std::size_t ByteSize(const std::vector<T> &values) {
    return values.size() * sizeof(T);
}

template <typename T>
const void *UploadPointer(const std::vector<T> &values) {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
void *DownloadPointer(std::vector<T> &values) {
    return values.empty() ? nullptr : values.data();
}

}  // namespace

class VulkanSearchExecutorAccess {
public:
    static vulkan::BufferHandle Scene(const VulkanDeviceScene &value) {
        return value.buffer_;
    }
    static vulkan::BufferHandle Configuration(
            const VulkanDeviceStaticConfiguration &value) {
        return value.buffer_;
    }
};

struct VulkanSearchExecutor::Impl {
    CudaSearchExecutorConfiguration configuration;
    VulkanDeviceScene deviceScene;
    VulkanDeviceStaticConfiguration deviceConfiguration;
    vulkan::BufferHandle scene;
    vulkan::BufferHandle staticConfiguration;
    std::vector<CudaSearchInputEvent> immutableInputPrefix;
    std::vector<CudaSearchInputEvent> immutableInputTail;
    DeviceControlState mutableBoundaryControls{};
    std::int64_t mutableFromTimeMs = 0;
    std::uint32_t evaluationTickCount = 0u;
    bool baselineInputsCanonical = false;
    bool baselineEvaluated = false;
    bool simulateStunts = false;
    vulkan::ComputeKernel physicsKernel = vulkan::ComputeKernel::SearchPhysics;
    DeviceSample globalBestSample{};
    CudaCandidateState globalBestState{};
    std::vector<CudaSearchInputEvent> globalBestInputs;
    std::uint32_t globalBestMutationCount = 0u;

    bool BuildTimelineInput(
            std::uint64_t candidateId,
            std::uint32_t evaluationTick,
            const CudaSearchInputEvent *events,
            std::uint32_t eventCount,
            VulkanCandidateTimelineInput *input,
            std::string *diagnostic) const {
        const std::int64_t evaluationStartTick =
                (configuration.evaluationStartTimeMs -
                 (configuration.branchTimeMs +
                  configuration.tickDurationMs)) /
                configuration.tickDurationMs;
        if (evaluationStartTick < 0 ||
            static_cast<std::uint64_t>(evaluationStartTick) +
                            evaluationTick >=
                    configuration.baselineTicks.size()) {
            if (diagnostic) {
                *diagnostic =
                        "Vulkan winning evaluation tick " +
                        std::to_string(evaluationTick) +
                        " is out of range for " +
                        std::to_string(configuration.baselineTicks.size()) +
                        " timeline ticks";
            }
            return false;
        }
        const std::size_t targetTick =
                static_cast<std::size_t>(evaluationStartTick) +
                evaluationTick;
        input->initialState = configuration.branchState;
        input->initialState.candidateId =
                static_cast<std::uint32_t>(candidateId);
        input->ticks.clear();
        input->ticks.reserve(targetTick + 1u);

        DeviceControlState controls = mutableBoundaryControls;
        std::uint32_t eventCursor = 0u;
        for (std::size_t tickIndex = 0u;
             tickIndex <= targetTick; ++tickIndex) {
            const std::int64_t publicTime =
                    configuration.branchTimeMs +
                    static_cast<std::int64_t>(tickIndex + 1u) *
                            configuration.tickDurationMs;
            const std::int64_t suffixTime =
                    publicTime - mutableFromTimeMs;
            while (eventCursor < eventCount &&
                   events[eventCursor].timeMs <= suffixTime) {
                cuda_search_detail::ApplyControlEvent(
                        controls, events[eventCursor], mutableFromTimeMs);
                ++eventCursor;
            }
            CudaControlTick tick = configuration.baselineTicks[tickIndex];
            tick.controls = cuda_search_detail::ControlsFromState(controls);
            tick.stuntsInput = cuda_search_detail::StuntsFromState(
                    controls, configuration.prestartDurationMs);
            input->ticks.push_back(tick);
        }
        return true;
    }

    static void AddTimelineMetrics(
            const VulkanTimelineExecutionMetrics &source,
            CudaSearchBatchExecution *destination) {
        destination->hostToDeviceBytes += source.hostToDeviceBytes;
        destination->deviceToHostBytes += source.deviceToHostBytes;
        destination->residentDeviceBytes = std::max(
                destination->residentDeviceBytes,
                source.peakDeviceBytes);
        destination->kernelMilliseconds += source.kernelMilliseconds;
        destination->simulationKernelMilliseconds +=
                source.kernelMilliseconds;
    }

    CudaSearchBatchExecution Execute(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool baseline,
            const std::function<bool()> &cancellationRequested,
            bool allowSplitWater = true) noexcept {
        CudaSearchBatchExecution result;
        result.firstCandidateId = firstCandidateId;
        result.candidateCount = candidateCount;
        if ((!baseline && !baselineEvaluated) || candidateCount == 0u ||
            candidateCount > configuration.maximumBatchSize) {
            result.status = CudaSearchStatus::InvalidArgument;
            result.diagnostic = !baseline && !baselineEvaluated
                    ? "Vulkan baseline must be evaluated before mutation batches"
                    : "invalid Vulkan search batch size";
            return result;
        }

        bool cancelled = false;
        if (cancellationRequested) {
            try {
                cancelled = cancellationRequested();
            } catch (...) {
                cancelled = true;
            }
        }

        try {
            const vulkan::ComputeKernel batchPhysicsKernel = physicsKernel;
            constexpr std::uint32_t SplitCollisionLanes = 4u;
            const bool splitWater =
                    allowSplitWater &&
                    batchPhysicsKernel == vulkan::ComputeKernel::
                            SearchPhysicsSteadyVelocityWater;
            const std::size_t scratchMultiplier =
                    splitWater ? SplitCollisionLanes : 1u;
            const std::size_t eventCapacity =
                    configuration.maximumEventCount;
            constexpr std::size_t ShapeCapacity = 8u;
            if ((eventCapacity != 0u && candidateCount >
                        std::numeric_limits<std::size_t>::max() /
                                eventCapacity) ||
                (eventCapacity != 0u &&
                 candidateCount * eventCapacity >
                        std::numeric_limits<std::size_t>::max() /
                                sizeof(CudaSearchInputEvent)) ||
                candidateCount >
                        std::numeric_limits<std::size_t>::max() / 624u ||
                candidateCount >
                        std::numeric_limits<std::size_t>::max() /
                                scratchMultiplier /
                                cuda::collision::CollisionCapacity ||
                candidateCount >
                        std::numeric_limits<std::size_t>::max() /
                                scratchMultiplier /
                                cuda::collision::ShapeCollisionCapacity ||
                candidateCount >
                        std::numeric_limits<std::size_t>::max() /
                                scratchMultiplier /
                                cuda::collision::SurfaceHitCapacity ||
                candidateCount >
                        std::numeric_limits<std::size_t>::max() /
                                scratchMultiplier /
                                cuda::collision::MeshCellHitCapacity ||
                candidateCount >
                        std::numeric_limits<std::size_t>::max() /
                                scratchMultiplier / ShapeCapacity ||
                (splitWater &&
                 candidateCount > UINT32_MAX / SplitCollisionLanes)) {
                result.status = CudaSearchStatus::CapacityExceeded;
                result.diagnostic = "Vulkan search workspace size overflow";
                return result;
            }
            const std::size_t eventSlots =
                    static_cast<std::size_t>(candidateCount) * eventCapacity;
            const std::size_t eventStorageSlots =
                    std::max<std::size_t>(eventSlots, 1u);
            const std::size_t scratchStride =
                    static_cast<std::size_t>(candidateCount) *
                    scratchMultiplier;
            const std::size_t collisionSlots =
                    scratchStride *
                    cuda::collision::CollisionCapacity;
            const std::size_t collisionTileStride =
                    (scratchStride +
                     cuda::collision::CudaCollisionSearchTileWidth - 1u) /
                    cuda::collision::CudaCollisionSearchTileWidth;
            const std::size_t collisionTileSlots =
                    collisionTileStride * cuda::collision::CollisionCapacity;
            const std::size_t shapeCollisionTileSlots =
                    collisionTileStride *
                    cuda::collision::ShapeCollisionCapacity;
            const std::size_t shapeQuerySlots =
                    scratchStride * ShapeCapacity;
            const std::size_t surfaceHitSlots =
                    scratchStride *
                    cuda::collision::SurfaceHitCapacity;
            const std::size_t meshCellSlots =
                    scratchStride *
                    cuda::collision::MeshCellHitCapacity;

            const bool simulateStunts = this->simulateStunts;
            const bool useDeviceSummary =
                    configuration.evaluator.kind !=
                    CudaSearchEvaluatorKind::FinishTime;
            const std::size_t stateStride = simulateStunts
                    ? sizeof(CudaCandidateState)
                    : sizeof(CudaCandidatePhysicsState);
            const std::uint64_t observationPlaceholder = 0u;
            std::vector<DeviceSample> samples(
                    static_cast<std::size_t>(candidateCount) + 1u);
            samples[0] = globalBestSample;
            samples[0].logicalOrder = 0u;
            samples[0].candidateSlot = InvalidCandidateSlot;
            std::vector<CudaSearchInputEvent> candidateEvents(
                    useDeviceSummary ? 0u : eventStorageSlots);
            std::vector<std::uint32_t> eventCounts(
                    useDeviceSummary ? 0u : candidateCount);
            std::vector<std::uint32_t> mutationCounts(
                    useDeviceSummary ? 0u : candidateCount);
            std::vector<std::uint32_t> statuses(
                    useDeviceSummary ? 0u : candidateCount);
            std::vector<std::uint8_t> activeCandidates(
                    useDeviceSummary ? 0u : candidateCount);
            VulkanSearchBatchSummary batchSummary;
            std::vector<CudaSearchInputEvent> winnerEvents(
                    std::max<std::size_t>(eventCapacity, 1u));
            std::uint32_t cancellation = cancelled ? 1u : 0u;

            const CudaSearchInputEvent dummyInput{};
            const double dummyWeight = 0.0;
            const CudaSearchConditionInstruction dummyCondition{};
            const std::vector<CudaSearchInputEvent> *baselineInputs =
                    &configuration.baselineInputs;
            const std::size_t baselineBytes = std::max<std::size_t>(
                    sizeof(dummyInput), ByteSize(*baselineInputs));
            const std::size_t weightBytes = std::max<std::size_t>(
                    sizeof(dummyWeight), ByteSize(configuration.smoothWeights));
            const std::size_t conditionBytes = std::max<std::size_t>(
                    sizeof(dummyCondition),
                    configuration.condition
                            ? ByteSize(configuration.condition->instructions)
                            : 0u);

            VulkanSearchParameters parameters;
            parameters.firstCandidateId = firstCandidateId;
            parameters.branchTimeMs = configuration.branchTimeMs;
            parameters.mutableFromTimeMs = mutableFromTimeMs;
            parameters.evaluationStartTimeMs =
                    configuration.evaluationStartTimeMs;
            parameters.lastImprovementTimeSeconds =
                    configuration.condition
                    ? configuration.condition->lastImprovementTimeSeconds
                    : 0.0;
            parameters.lastRestartTimeSeconds =
                    configuration.condition
                    ? configuration.condition->lastRestartTimeSeconds
                    : 0.0;
            parameters.currentTimeSeconds =
                    std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
            result.evaluationCurrentTimeSeconds =
                    parameters.currentTimeSeconds;
            parameters.baselineInputCount = static_cast<std::uint32_t>(
                    configuration.baselineInputs.size());
            parameters.immutableTailInputCount =
                    static_cast<std::uint32_t>(immutableInputTail.size());
            parameters.modifierCount = static_cast<std::uint32_t>(
                    configuration.modifiers.size());
            parameters.tickDurationMs = configuration.tickDurationMs;
            parameters.candidateCount = candidateCount;
            parameters.eventCapacity =
                    static_cast<std::uint32_t>(eventCapacity);
            parameters.baseline = baseline ? 1u : 0u;
            parameters.legacyMutationPipeline =
                    configuration.useLegacyMutationPipelineForTesting
                    ? 1u : 0u;
            parameters.baselineInputsCanonical =
                    baselineInputsCanonical ? 1u : 0u;
            parameters.timelineTickCount = static_cast<std::uint32_t>(
                    configuration.baselineTicks.size());
            parameters.conditionInstructionCount =
                    configuration.condition
                    ? static_cast<std::uint32_t>(
                              configuration.condition->instructions.size())
                    : 0u;
            parameters.prestartDurationMs =
                    configuration.prestartDurationMs;
            parameters.evaluationTickCount = evaluationTickCount;
            parameters.simulateStunts = simulateStunts ? 1u : 0u;
            parameters.stateWordCount = static_cast<std::uint32_t>(
                    stateStride / sizeof(std::uint32_t));
            parameters.scratchStride =
                    static_cast<std::uint32_t>(scratchStride);
            parameters.shapeCapacity =
                    static_cast<std::uint32_t>(ShapeCapacity);

            enum Region : std::size_t {
                BranchState,
                BaselineTicks,
                Evaluator,
                Condition,
                CandidateStates,
                CollisionScratch,
                CollisionStorage,
                ShapeCollisionStorage,
                ShapeWorldStorage,
                MovingBoundsStorage,
                UnifiedMovingBoundsStorage,
                SurfaceHitStorage,
                MeshRangeStorage,
                MeshCellStorage,
                ResponseOrderStorage,
                ControlStates,
                EventCursors,
                SimulationActive,
                EvaluatorReported,
                PreviousPositions,
                EvaluationIndices,
                TimelineDescriptors,
                CandidateTicks,
                TimelineResults,
                TimelineObservations,
                BaselineInputs,
                Modifiers,
                SmoothWeights,
                BoundaryControls,
                Samples,
                RandomStateWords,
                CandidateEvents,
                TemporaryEvents,
                PassBaselineEvents,
                EligibleIndices,
                EventCounts,
                MutationCounts,
                Statuses,
                ActiveCandidates,
                Cancellation,
                WinnerSummary,
                WinnerEvents,
            };
            vulkan::ComputeKernelRequest request;
            request.parameters = &parameters;
            request.parameterBytes = sizeof(parameters);
            request.regions = {
                    {&configuration.branchState, nullptr,
                     sizeof(configuration.branchState)},
                    {configuration.baselineTicks.data(), nullptr,
                     ByteSize(configuration.baselineTicks)},
                    {&configuration.evaluator, nullptr,
                     sizeof(configuration.evaluator)},
                    {configuration.condition &&
                                     !configuration.condition->instructions.empty()
                             ? configuration.condition->instructions.data()
                             : &dummyCondition,
                     nullptr, conditionBytes},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) * stateStride},
                    {nullptr, nullptr,
                     scratchStride * sizeof(VulkanCollisionSearchScratch)},
                    {nullptr, nullptr,
                     collisionTileSlots *
                             sizeof(cuda::collision::CudaCollisionSearchTile)},
                    {nullptr, nullptr,
                     shapeCollisionTileSlots *
                             sizeof(cuda::collision::CudaCollisionSearchTile)},
                    {nullptr, nullptr,
                     shapeQuerySlots * sizeof(GmIso4)},
                    {nullptr, nullptr,
                     shapeQuerySlots * sizeof(GmBoxAligned)},
                    {nullptr, nullptr,
                     scratchStride * sizeof(GmBoxAligned)},
                    {nullptr, nullptr,
                     surfaceHitSlots *
                             sizeof(cuda::collision::CudaCollisionSurfaceHit)},
                    {nullptr, nullptr,
                     surfaceHitSlots *
                             sizeof(cuda::collision::CudaCollisionMeshRange)},
                    {nullptr, nullptr,
                     meshCellSlots * sizeof(std::uint32_t)},
                    {nullptr, nullptr,
                     collisionSlots * sizeof(std::uint16_t)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(DeviceControlState)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint32_t)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint8_t)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint8_t)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) * sizeof(GmVec3)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint32_t)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(VulkanSearchTimelineDescriptor)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(CudaControlTick)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(VulkanSearchTimelineResult)},
                    {&observationPlaceholder, nullptr,
                     sizeof(observationPlaceholder)},
                    {configuration.baselineInputs.empty()
                             ? &dummyInput
                             : configuration.baselineInputs.data(),
                     nullptr, baselineBytes},
                    {configuration.modifiers.data(), nullptr,
                     ByteSize(configuration.modifiers)},
                    {configuration.smoothWeights.empty()
                             ? &dummyWeight
                             : configuration.smoothWeights.data(),
                     nullptr, weightBytes},
                    {&mutableBoundaryControls, nullptr,
                     sizeof(mutableBoundaryControls)},
                    {samples.data(),
                     useDeviceSummary ? nullptr : samples.data(),
                     ByteSize(samples)},
                    {nullptr, nullptr,
                     static_cast<std::size_t>(candidateCount) * 624u *
                             sizeof(std::uint32_t)},
                    {nullptr,
                     useDeviceSummary
                             ? nullptr
                             : DownloadPointer(candidateEvents),
                     eventStorageSlots * sizeof(CudaSearchInputEvent)},
                    {nullptr, nullptr,
                     eventStorageSlots * sizeof(CudaSearchInputEvent)},
                    {nullptr, nullptr,
                     eventStorageSlots * sizeof(CudaSearchInputEvent)},
                    {nullptr, nullptr,
                     eventStorageSlots * sizeof(std::uint32_t)},
                    {nullptr,
                     useDeviceSummary ? nullptr : eventCounts.data(),
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint32_t)},
                    {nullptr,
                     useDeviceSummary ? nullptr : mutationCounts.data(),
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint32_t)},
                    {nullptr,
                     useDeviceSummary ? nullptr : statuses.data(),
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint32_t)},
                    {nullptr,
                     useDeviceSummary ? nullptr : activeCandidates.data(),
                     static_cast<std::size_t>(candidateCount) *
                             sizeof(std::uint8_t)},
                    {&cancellation, nullptr, sizeof(cancellation)},
                    {nullptr,
                     useDeviceSummary ? &batchSummary : nullptr,
                     sizeof(batchSummary)},
                    {nullptr,
                     useDeviceSummary
                             ? DownloadPointer(winnerEvents)
                             : nullptr,
                     ByteSize(winnerEvents)},
            };
            const auto bind = [&](std::size_t parameterOffset,
                                  Region region) {
                request.regionBindings.push_back(
                        {parameterOffset, static_cast<std::size_t>(region)});
            };
            bind(offsetof(VulkanSearchParameters, branchState), BranchState);
            bind(offsetof(VulkanSearchParameters, baselineTicks), BaselineTicks);
            bind(offsetof(VulkanSearchParameters, evaluator), Evaluator);
            bind(offsetof(VulkanSearchParameters, condition), Condition);
            bind(offsetof(VulkanSearchParameters, candidateStates), CandidateStates);
            bind(offsetof(VulkanSearchParameters, collisionScratch), CollisionScratch);
            bind(offsetof(VulkanSearchParameters, collisionStorage),
                 CollisionStorage);
            bind(offsetof(VulkanSearchParameters, shapeCollisionStorage),
                 ShapeCollisionStorage);
            bind(offsetof(VulkanSearchParameters, shapeWorldStorage),
                 ShapeWorldStorage);
            bind(offsetof(VulkanSearchParameters, movingBoundsStorage),
                 MovingBoundsStorage);
            bind(offsetof(VulkanSearchParameters, unifiedMovingBoundsStorage),
                 UnifiedMovingBoundsStorage);
            bind(offsetof(VulkanSearchParameters, surfaceHitStorage),
                 SurfaceHitStorage);
            bind(offsetof(VulkanSearchParameters, meshRangeStorage),
                 MeshRangeStorage);
            bind(offsetof(VulkanSearchParameters, meshCellStorage),
                 MeshCellStorage);
            bind(offsetof(VulkanSearchParameters, responseOrderStorage),
                 ResponseOrderStorage);
            bind(offsetof(VulkanSearchParameters, controlStates), ControlStates);
            bind(offsetof(VulkanSearchParameters, eventCursors), EventCursors);
            bind(offsetof(VulkanSearchParameters, simulationActive), SimulationActive);
            bind(offsetof(VulkanSearchParameters, evaluatorReported), EvaluatorReported);
            bind(offsetof(VulkanSearchParameters, previousPositions), PreviousPositions);
            bind(offsetof(VulkanSearchParameters, evaluationIndices), EvaluationIndices);
            bind(offsetof(VulkanSearchParameters, timelineDescriptors), TimelineDescriptors);
            bind(offsetof(VulkanSearchParameters, candidateTicks), CandidateTicks);
            bind(offsetof(VulkanSearchParameters, timelineResults), TimelineResults);
            bind(offsetof(VulkanSearchParameters, baselineInputs), BaselineInputs);
            bind(offsetof(VulkanSearchParameters, modifiers), Modifiers);
            bind(offsetof(VulkanSearchParameters, smoothWeights), SmoothWeights);
            bind(offsetof(VulkanSearchParameters, mutableBoundaryControls),
                 BoundaryControls);
            bind(offsetof(VulkanSearchParameters, candidateBestSamples), Samples);
            bind(offsetof(VulkanSearchParameters, randomStateWords), RandomStateWords);
            bind(offsetof(VulkanSearchParameters, candidateEvents), CandidateEvents);
            bind(offsetof(VulkanSearchParameters, temporaryEvents), TemporaryEvents);
            bind(offsetof(VulkanSearchParameters, passBaselineEvents), PassBaselineEvents);
            bind(offsetof(VulkanSearchParameters, eligibleIndices), EligibleIndices);
            bind(offsetof(VulkanSearchParameters, eventCounts), EventCounts);
            bind(offsetof(VulkanSearchParameters, mutationCounts), MutationCounts);
            bind(offsetof(VulkanSearchParameters, statuses), Statuses);
            bind(offsetof(VulkanSearchParameters, activeCandidates), ActiveCandidates);
            bind(offsetof(VulkanSearchParameters, cancellation), Cancellation);
            bind(offsetof(VulkanSearchParameters, winnerSummary), WinnerSummary);
            bind(offsetof(VulkanSearchParameters, winnerEvents), WinnerEvents);
            request.externalBindings.push_back(
                    {offsetof(VulkanSearchParameters, scene), scene});
            request.externalBindings.push_back(
                    {offsetof(VulkanSearchParameters, configuration),
                     staticConfiguration});
            if (cancellationRequested) {
                request.dynamicCancellationRegion = Cancellation;
                request.cancellationRequested = cancellationRequested;
            }
            request.maxDispatchesPerSubmission = 16u;
            const std::uint32_t tickDispatchSize =
                    batchPhysicsKernel == vulkan::ComputeKernel::SearchPhysics
                    ? GenericSearchTickDispatchSize
                    : (splitWater ? 1u : SteadyVelocityTickDispatchSize);
            const std::uint32_t dispatchCount =
                    (candidateCount + SearchCandidateDispatchSize - 1u) /
                    SearchCandidateDispatchSize;
            const std::uint32_t tickDispatchCount =
                    (parameters.timelineTickCount +
                     tickDispatchSize - 1u) /
                    tickDispatchSize;
            request.dispatches.reserve(
                    static_cast<std::size_t>(dispatchCount) *
                            (2u + tickDispatchCount) +
                    (useDeviceSummary ? 1u : 0u));
            const auto appendCandidateDispatches =
                    [&](vulkan::ComputeKernel kernel) {
                for (std::uint32_t base = 0u; base < candidateCount;
                     base += SearchCandidateDispatchSize) {
                    request.dispatches.push_back(
                            {kernel,
                             std::min(SearchCandidateDispatchSize,
                                      candidateCount - base),
                             0u,
                             base,
                             0u});
                }
            };
            appendCandidateDispatches(vulkan::ComputeKernel::SearchGenerate);
            appendCandidateDispatches(vulkan::ComputeKernel::SearchInitialize);
            for (std::uint32_t tickBase = 0u;
                 tickBase < parameters.timelineTickCount;
                 tickBase += tickDispatchSize) {
                const std::uint32_t tickCount =
                        std::min(tickDispatchSize,
                                 parameters.timelineTickCount - tickBase);
                for (std::uint32_t candidateBase = 0u;
                     candidateBase < candidateCount;
                     candidateBase += SearchCandidateDispatchSize) {
                    const std::uint32_t dispatchCandidates =
                            std::min(SearchCandidateDispatchSize,
                                     candidateCount - candidateBase);
                    if (splitWater) {
                        for (std::uint32_t substep = 0u;
                             substep < 2u; ++substep) {
                            request.dispatches.push_back(
                                    {vulkan::ComputeKernel::
                                             SearchPhysicsForcePhase,
                                     dispatchCandidates,
                                     tickBase, candidateBase, substep});
                            request.dispatches.push_back(
                                    {vulkan::ComputeKernel::
                                             SearchPhysicsDetectPhase,
                                     dispatchCandidates * SplitCollisionLanes,
                                     tickBase, candidateBase, substep});
                            request.dispatches.push_back(
                                    {vulkan::ComputeKernel::
                                             SearchPhysicsRespondPhase,
                                     dispatchCandidates,
                                     tickBase, candidateBase, substep});
                        }
                    } else {
                        request.dispatches.push_back(
                                {batchPhysicsKernel, dispatchCandidates,
                                 tickBase, candidateBase, tickCount});
                    }
                }
            }
            if (useDeviceSummary) {
                request.dispatches.push_back(
                        {splitWater
                                 ? vulkan::ComputeKernel::
                                           SearchPhysicsRespondPhase
                                 : batchPhysicsKernel,
                         1u, UINT32_MAX, 0u, 0u});
            }

            vulkan::ComputeKernelMetrics metrics;
            // Workspace and staging have identical region sizes. Report their
            // scalable part separately from retained fixed storage. Tile slack
            // and alignment are covered by the measured resident-byte allowance.
            std::uint64_t reservation = 0;
            for (std::size_t i = 0; i < request.regions.size(); ++i) {
                if (!((i >= CandidateStates && i <= TimelineResults) ||
                      (i >= Samples && i <= ActiveCandidates))) continue;
                const auto bytes = request.regions[i].bytes;
                const std::uint64_t perCandidate =
                        i == CollisionStorage || i == ShapeCollisionStorage
                        ? (bytes / collisionTileStride) * scratchMultiplier /
                                cuda::collision::CudaCollisionSearchTileWidth
                        : bytes / candidateCount + (bytes % candidateCount != 0u ? 1u : 0u);
                if (reservation > UINT64_MAX / 2u ||
                    perCandidate > UINT64_MAX / 2u - reservation) {
                    result.status = CudaSearchStatus::DeviceFailure;
                    result.diagnostic = "Vulkan search reservation estimate overflow";
                    return result;
                }
                reservation += perCandidate;
            }
            result.reservationBytesPerCandidate = reservation * 2u;
            if (!vulkan::ExecuteComputeKernels(
                        request, &metrics, &result.diagnostic)) {
                result.status = CudaSearchStatus::DeviceFailure;
                return result;
            }
            result.hostToDeviceBytes = metrics.uploadBytes;
            result.deviceToHostBytes = metrics.downloadBytes;
            result.residentDeviceBytes = metrics.peakDeviceBytes;
            result.kernelMilliseconds = metrics.kernelMilliseconds;
            result.simulationKernelMilliseconds =
                    metrics.kernelMilliseconds;
            if (useDeviceSummary) {
                result.status = batchSummary.status;
                result.evaluatedCandidateCount =
                        batchSummary.evaluatedCandidateCount;
                result.evaluatorCalls =
                        static_cast<std::uint64_t>(
                                batchSummary.evaluatedCandidateCount) *
                        evaluationTickCount;
                result.totalMutationCount =
                        batchSummary.totalMutationCount;
                result.mutationImprovementCount = baseline
                        ? 0u
                        : batchSummary.mutationImprovementCount;
            } else {
                result.status = CudaSearchStatus::Success;
                for (std::uint32_t slot = 0u;
                     slot < candidateCount; ++slot) {
                    if (statuses[slot] == 2u) {
                        result.status = CudaSearchStatus::Cancelled;
                    } else if (statuses[slot] == 1u) {
                        result.status = CudaSearchStatus::CapacityExceeded;
                    } else if (statuses[slot] == 4u) {
                        result.status = CudaSearchStatus::
                                UnsupportedPhysicsTransition;
                    }
                    if (activeCandidates[slot] != 0u) {
                        ++result.evaluatedCandidateCount;
                        result.evaluatorCalls += evaluationTickCount;
                    }
                    result.totalMutationCount += mutationCounts[slot];
                }
            }

            if (splitWater &&
                result.status ==
                        CudaSearchStatus::UnsupportedPhysicsTransition) {
                CudaSearchBatchExecution fallback = Execute(
                        firstCandidateId, candidateCount, baseline,
                        cancellationRequested, false);
                fallback.hostToDeviceBytes += result.hostToDeviceBytes;
                fallback.deviceToHostBytes += result.deviceToHostBytes;
                fallback.residentDeviceBytes = std::max(
                        fallback.residentDeviceBytes,
                        result.residentDeviceBytes);
                fallback.kernelMilliseconds += result.kernelMilliseconds;
                fallback.simulationKernelMilliseconds +=
                        result.simulationKernelMilliseconds;
                return fallback;
            }

            std::vector<CudaCandidateState> refinedFinishStates;
            std::vector<std::uint8_t> refinedFinishStateValid;
            if (configuration.evaluator.kind ==
                    CudaSearchEvaluatorKind::FinishTime) {
                refinedFinishStates.resize(candidateCount);
                refinedFinishStateValid.resize(candidateCount);
                std::vector<std::uint32_t> refinementSlots;
                std::vector<VulkanCandidateTimelineInput>
                        refinementInputs;
                refinementSlots.reserve(candidateCount);
                refinementInputs.reserve(candidateCount);
                for (std::uint32_t slot = 0u;
                     slot < candidateCount; ++slot) {
                    if (!samples[slot + 1u].valid ||
                        statuses[slot] != 0u) {
                        continue;
                    }
                    VulkanCandidateTimelineInput input;
                    const std::size_t eventOffset =
                            static_cast<std::size_t>(slot) * eventCapacity;
                    if (!BuildTimelineInput(
                                samples[slot + 1u].candidateId,
                                samples[slot + 1u].evaluationTick,
                                candidateEvents.data() + eventOffset,
                                eventCounts[slot], &input,
                                &result.diagnostic)) {
                        result.status = CudaSearchStatus::DeviceFailure;
                        return result;
                    }
                    refinementSlots.push_back(slot);
                    refinementInputs.push_back(std::move(input));
                }
                if (!refinementInputs.empty()) {
                    VulkanTimelineBatchResult refinement =
                            ExecuteVulkanFinishTimelineBatch(
                                    deviceScene, deviceConfiguration,
                                    refinementInputs, false);
                    AddTimelineMetrics(refinement.metrics, &result);
                    if (refinement.candidates.size() !=
                            refinementInputs.size()) {
                        result.status = CudaSearchStatus::DeviceFailure;
                        result.diagnostic = refinement.diagnostic;
                        return result;
                    }
                    const std::uint64_t prestartNs =
                            static_cast<std::uint64_t>(
                                    configuration.prestartDurationMs) *
                            1000000u;
                    for (std::size_t index = 0u;
                         index < refinementSlots.size(); ++index) {
                        const std::uint32_t slot = refinementSlots[index];
                        const VulkanCandidateTimelineOutput &output =
                                refinement.candidates[index];
                        if (output.status != VulkanTimelineStatus::Success ||
                            !output.finalState.finishTime.present ||
                            !output.finalState.finishTime.value.IsValid() ||
                            output.finalState.finishTime.value.upperBoundNs <
                                    prestartNs) {
                            samples[slot + 1u] = {};
                            statuses[slot] = 4u;
                            result.status = CudaSearchStatus::
                                    UnsupportedPhysicsTransition;
                            continue;
                        }
                        DeviceSample &sample = samples[slot + 1u];
                        sample.score = static_cast<double>(
                                output.finalState.finishTime.value.
                                        upperBoundNs - prestartNs);
                        sample.timeMs = sample.score / 1000000.0;
                        sample.preciseFinish = true;
                        refinedFinishStates[slot] = output.finalState;
                        refinedFinishStateValid[slot] = 1u;
                    }
                }
            }

            const bool maximize =
                    configuration.evaluator.kind ==
                            CudaSearchEvaluatorKind::Velocity ||
                    configuration.evaluator.kind ==
                            CudaSearchEvaluatorKind::StuntPoints;
            DeviceSample incumbent = useDeviceSummary
                    ? batchSummary.winner
                    : samples[0];
            std::uint32_t winnerSlot = useDeviceSummary
                    ? batchSummary.winnerSlot
                    : InvalidCandidateSlot;
            if (!useDeviceSummary && !baseline) {
                for (std::uint32_t slot = 0u;
                     slot < candidateCount; ++slot) {
                    if (StrictlyBetter(
                                samples[slot + 1u], incumbent,
                                maximize)) {
                        ++result.mutationImprovementCount;
                        incumbent = samples[slot + 1u];
                        winnerSlot = slot;
                    }
                }
            } else if (!useDeviceSummary && samples[1].valid) {
                incumbent = samples[1];
                winnerSlot = 0u;
            }

            const std::uint32_t winnerEventCount = useDeviceSummary
                    ? batchSummary.winnerEventCount
                    : (winnerSlot != InvalidCandidateSlot
                       ? eventCounts[winnerSlot] : 0u);
            if (incumbent.scriptedObjectiveCount >
                        CudaSearchMaximumScriptedObjectives ||
                (winnerSlot != InvalidCandidateSlot &&
                 (winnerSlot >= candidateCount ||
                  winnerEventCount > eventCapacity))) {
                result.status = CudaSearchStatus::DeviceFailure;
                result.diagnostic =
                        "Vulkan search returned invalid winner metadata";
                return result;
            }
            const std::uint32_t winnerMutationCount = useDeviceSummary
                    ? batchSummary.winnerMutationCount
                    : (winnerSlot != InvalidCandidateSlot
                       ? mutationCounts[winnerSlot] : 0u);
            const CudaSearchInputEvent *winnerEventData = useDeviceSummary
                    ? winnerEvents.data()
                    : (winnerSlot != InvalidCandidateSlot
                       ? candidateEvents.data() +
                                 static_cast<std::size_t>(winnerSlot) *
                                         eventCapacity
                       : nullptr);

            if (winnerSlot != InvalidCandidateSlot && incumbent.valid &&
                (useDeviceSummary
                         ? batchSummary.bestChanged != 0u
                         : (baseline || StrictlyBetter(
                                                incumbent, globalBestSample,
                                                maximize)))) {
                CudaCandidateState capturedState{};
                if (configuration.captureBestState) {
                    if (configuration.evaluator.kind ==
                                CudaSearchEvaluatorKind::FinishTime &&
                        refinedFinishStateValid[winnerSlot] != 0u) {
                        capturedState = refinedFinishStates[winnerSlot];
                    } else {
                        VulkanCandidateTimelineInput captureInput;
                        if (!BuildTimelineInput(
                                    incumbent.candidateId,
                                    incumbent.evaluationTick,
                                    winnerEventData,
                                    winnerEventCount, &captureInput,
                                    &result.diagnostic)) {
                            result.status = CudaSearchStatus::DeviceFailure;
                            return result;
                        }
                        VulkanTimelineBatchResult capture =
                                ExecuteVulkanTimelineBatch(
                                        deviceScene, deviceConfiguration,
                                        {captureInput}, false);
                        AddTimelineMetrics(capture.metrics, &result);
                        if (capture.candidates.size() != 1u ||
                            capture.candidates[0].status !=
                                    VulkanTimelineStatus::Success) {
                            result.status = CudaSearchStatus::DeviceFailure;
                            result.diagnostic = capture.diagnostic;
                            return result;
                        }
                        capturedState = capture.candidates[0].finalState;
                    }
                }
                globalBestSample = incumbent;
                if (configuration.captureBestState) {
                    globalBestState = capturedState;
                }
                globalBestMutationCount = winnerMutationCount;
                globalBestInputs = immutableInputPrefix;
                globalBestInputs.reserve(
                        globalBestInputs.size() + winnerEventCount +
                        immutableInputTail.size());
                for (std::uint32_t index = 0u;
                     index < winnerEventCount; ++index) {
                    globalBestInputs.push_back(
                            cuda_search_detail::AbsoluteSuffixEvent(
                                    winnerEventData[index],
                                    mutableFromTimeMs));
                }
                for (CudaSearchInputEvent input : immutableInputTail) {
                    globalBestInputs.push_back(
                            cuda_search_detail::AbsoluteSuffixEvent(
                                    input, mutableFromTimeMs));
                }
                result.bestChanged = true;
            }
            if (globalBestSample.valid) {
                result.best.valid = true;
                result.best.mutation = globalBestSample.mutation;
                result.best.stateCaptured =
                        configuration.captureBestState;
                result.best.candidateId = globalBestSample.candidateId;
                result.best.mutationCount = globalBestMutationCount;
                result.best.evaluationTick =
                        globalBestSample.evaluationTick;
                result.best.score = globalBestSample.score;
                result.best.timeMs = globalBestSample.timeMs;
                result.best.detail0 = globalBestSample.detail0;
                result.best.detail1 = globalBestSample.detail1;
                result.best.scriptedObjectiveCount =
                        globalBestSample.scriptedObjectiveCount;
                for (std::uint32_t i = 0u;
                     i < globalBestSample.scriptedObjectiveCount; ++i) {
                    result.best.objectiveScores[i] =
                            globalBestSample.objectiveScores[i];
                    result.best.metricValues[i] =
                            globalBestSample.metricValues[i];
                }
                if (configuration.captureBestState) {
                    result.best.state = globalBestState;
                }
                result.best.inputs = globalBestInputs;
            }
            if (baseline && result.status == CudaSearchStatus::Success) {
                baselineEvaluated = true;
            }
            if (result.status != CudaSearchStatus::Success) {
                result.diagnostic =
                        std::string("Vulkan search batch status: ") +
                        CudaSearchStatusName(result.status);
            } else {
                result.diagnostic = "Vulkan search batch completed";
            }
            return result;
        } catch (const std::bad_alloc &) {
            result.status = CudaSearchStatus::CapacityExceeded;
            result.diagnostic = "Vulkan search host allocation failed";
            return result;
        } catch (...) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic =
                    "unexpected Vulkan search execution failure";
            return result;
        }
    }
};

VulkanSearchExecutor::VulkanSearchExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
VulkanSearchExecutor::~VulkanSearchExecutor() = default;
VulkanSearchExecutor::VulkanSearchExecutor(
        VulkanSearchExecutor &&) noexcept = default;
VulkanSearchExecutor &VulkanSearchExecutor::operator=(
        VulkanSearchExecutor &&) noexcept = default;

std::unique_ptr<VulkanSearchExecutor> VulkanSearchExecutor::Create(
        const CudaSearchExecutorConfiguration &configuration,
        const VulkanDeviceScene &scene,
        const VulkanDeviceStaticConfiguration &staticConfiguration,
        std::string *diagnostic) noexcept {
    try {
        if (!scene.Ready() || !staticConfiguration.Ready() ||
            configuration.maximumBatchSize == 0u ||
            configuration.tickDurationMs == 0u ||
            configuration.maximumEventCount <
                    configuration.baselineInputs.size() ||
            configuration.maximumEventCount > UINT32_MAX ||
            configuration.baselineTicks.empty() ||
            configuration.baselineTicks.size() > UINT32_MAX ||
            configuration.modifiers.empty() ||
            configuration.modifiers.size() > UINT32_MAX ||
            configuration.branchTimeMs < 0 ||
            configuration.branchTimeMs >
                    INT32_MAX - configuration.tickDurationMs ||
            configuration.evaluationStartTimeMs <
                    configuration.branchTimeMs +
                            configuration.tickDurationMs ||
            configuration.evaluationEndTimeMs <
                    configuration.evaluationStartTimeMs ||
            (configuration.condition &&
             (configuration.condition->instructions.empty() ||
              configuration.condition->instructions.size() > 256u))) {
            if (diagnostic) {
                *diagnostic =
                        "invalid Vulkan search executor configuration";
            }
            return {};
        }
        const std::uint64_t evaluationTicks =
                static_cast<std::uint64_t>(
                        configuration.evaluationEndTimeMs -
                        configuration.evaluationStartTimeMs) /
                        configuration.tickDurationMs +
                1u;
        if (evaluationTicks == 0u || evaluationTicks > UINT32_MAX) {
            if (diagnostic) {
                *diagnostic = "Vulkan evaluation timeline is too large";
            }
            return {};
        }
        const std::int64_t mutableFromTimeMs =
                configuration.branchTimeMs +
                configuration.tickDurationMs;
        cuda_search_detail::SearchInputPartition partition;
        if (!cuda_search_detail::PartitionSearchInputs(
                    configuration.baselineInputs,
                    configuration.branchTimeMs,
                    mutableFromTimeMs, &partition)) {
            if (diagnostic) {
                std::int32_t previous = INT32_MIN;
                std::size_t invalidIndex = configuration.baselineInputs.size();
                for (std::size_t index = 0u;
                     index < configuration.baselineInputs.size(); ++index) {
                    const CudaSearchInputEvent &input =
                            configuration.baselineInputs[index];
                    if (input.timeMs < previous ||
                        (input.valueKind == 2u &&
                         (input.value < -65536 || input.value > 65536)) ||
                        (input.valueKind == 1u &&
                         input.value != 0 && input.value != 1)) {
                        invalidIndex = index;
                        break;
                    }
                    previous = input.timeMs;
                }
                *diagnostic = invalidIndex ==
                                      configuration.baselineInputs.size()
                        ? "Vulkan search mutable boundary is invalid"
                        : "Vulkan search input " +
                                  std::to_string(invalidIndex) +
                                  " is not canonical (time=" +
                                  std::to_string(
                                          configuration.baselineInputs[
                                                  invalidIndex].timeMs) +
                                  ", kind=" +
                                  std::to_string(
                                          configuration.baselineInputs[
                                                  invalidIndex].valueKind) +
                                  ", value=" +
                                  std::to_string(
                                          configuration.baselineInputs[
                                                  invalidIndex].value) +
                                  ")";
            }
            return {};
        }
        CudaSearchExecutorConfiguration prepared = configuration;
        prepared.baselineInputs = partition.mutableSuffix;
        prepared.maximumEventCount -= partition.immutablePrefix.size();
        if (prepared.maximumEventCount < prepared.baselineInputs.size()) {
            if (diagnostic) {
                *diagnostic = "Vulkan search event capacity is invalid";
            }
            return {};
        }
        for (CudaSearchModifierConfiguration &modifier :
             prepared.modifiers) {
            if (modifier.window.maximumTimeMs < mutableFromTimeMs) {
                if (diagnostic) {
                    *diagnostic =
                            "Vulkan modifier window does not intersect the mutable suffix";
                }
                return {};
            }
            modifier.window.minimumTimeMs =
                    std::max(
                            modifier.window.minimumTimeMs,
                            mutableFromTimeMs) -
                    mutableFromTimeMs;
            modifier.window.maximumTimeMs -= mutableFromTimeMs;
        }

        std::vector<CudaSearchInputEvent> immutableInputTail;
        if (!prepared.useLegacyMutationPipelineForTesting) {
            const std::int64_t evaluationEndRelativeMs =
                    prepared.evaluationEndTimeMs - mutableFromTimeMs;
            std::int64_t materializationEndTimeMs =
                    evaluationEndRelativeMs;
            for (const CudaSearchModifierConfiguration &modifier :
                 prepared.modifiers) {
                materializationEndTimeMs = std::max(
                        materializationEndTimeMs,
                        modifier.window.maximumTimeMs);
            }
            cuda_search_detail::SearchInputWindow inputWindow;
            if (!cuda_search_detail::PartitionSearchInputWindow(
                        prepared.baselineInputs,
                        materializationEndTimeMs, &inputWindow) ||
                inputWindow.immutableTail.size() >
                        prepared.maximumEventCount) {
                if (diagnostic) {
                    *diagnostic =
                            "Vulkan search inputs cannot be partitioned at "
                            "the materialization horizon";
                }
                return {};
            }
            prepared.maximumEventCount -=
                    inputWindow.immutableTail.size();
            prepared.baselineInputs =
                    std::move(inputWindow.materialized);
            immutableInputTail =
                    std::move(inputWindow.immutableTail);
            if (prepared.maximumEventCount <
                        prepared.baselineInputs.size()) {
                if (diagnostic) {
                    *diagnostic =
                            "Vulkan search materialized event capacity is "
                            "invalid";
                }
                return {};
            }
        }

        auto impl = std::make_unique<Impl>();
        impl->configuration = std::move(prepared);
        impl->deviceScene = scene;
        impl->deviceConfiguration = staticConfiguration;
        impl->scene = VulkanSearchExecutorAccess::Scene(scene);
        impl->staticConfiguration =
                VulkanSearchExecutorAccess::Configuration(
                        staticConfiguration);
        impl->immutableInputPrefix =
                std::move(partition.immutablePrefix);
        impl->immutableInputTail =
                std::move(immutableInputTail);
        impl->mutableBoundaryControls =
                partition.mutableBoundaryControls;
        impl->mutableFromTimeMs = mutableFromTimeMs;
        impl->evaluationTickCount =
                static_cast<std::uint32_t>(evaluationTicks);
        impl->simulateStunts =
                impl->configuration.branchState.stuntsEnabled &&
                impl->configuration.evaluator.kind !=
                        CudaSearchEvaluatorKind::FinishTime;
        const auto &branchIntegration =
                impl->configuration.branchState.vehicle.integration;
        const bool raceSimulationAlreadyEnabled =
                branchIntegration.updateWheelVisuals &&
                branchIntegration.integrateWheels &&
                branchIntegration.integrateEngine &&
                !branchIntegration.zeroHorizontalSpeed &&
                !branchIntegration.speedBlocked &&
                !branchIntegration.speedBlockedSecondary;
        const std::uint32_t steadyActionMask =
                raceSimulationAlreadyEnabled
                ? CudaControlActionEnableRaceSimulation
                : 0u;
        const bool steadyTimeline = std::all_of(
                impl->configuration.baselineTicks.begin(),
                impl->configuration.baselineTicks.end(),
                [steadyActionMask](const CudaControlTick &tick) {
                    return (tick.actionFlags & ~steadyActionMask) == 0u &&
                           tick.respawnAtCheckpointCount == 0u;
                });
        const bool steadyFacts =
                staticConfiguration.SteadyVelocityFactsMatch(
                        impl->configuration.branchState);
        if (!impl->simulateStunts &&
            impl->configuration.evaluator.kind ==
                    CudaSearchEvaluatorKind::Velocity &&
            !impl->configuration.condition &&
            steadyTimeline && steadyFacts) {
            switch (staticConfiguration.HandlingSpecialization()) {
            case CudaHandlingSpecialization::GearedDriveWater:
                impl->physicsKernel =
                        vulkan::ComputeKernel::SearchPhysicsSteadyVelocityWater;
                break;
            default:
                break;
            }
        }
        if (!vulkan::SupportsExactSearchPhysics()) {
            if (diagnostic) {
                *diagnostic =
                        "Vulkan search requires VK_KHR_shader_float_controls2 "
                        "with exact FP32 controls";
            }
            return nullptr;
        }
        impl->baselineInputsCanonical =
                CanonicalInputs(impl->configuration.baselineInputs);
        if (configuration.incumbent) {
            const CudaSearchIncumbent &source = *configuration.incumbent;
            impl->globalBestSample.score = source.score;
            impl->globalBestSample.timeMs = source.timeMs;
            impl->globalBestSample.detail0 = source.detail0;
            impl->globalBestSample.detail1 = source.detail1;
            impl->globalBestSample.candidateId = source.candidateId;
            impl->globalBestSample.logicalOrder = 0u;
            impl->globalBestSample.candidateSlot = InvalidCandidateSlot;
            impl->globalBestSample.evaluationTick =
                    source.evaluationTick;
            impl->globalBestSample.eventCount =
                    static_cast<std::uint32_t>(
                            impl->configuration.baselineInputs.size());
            impl->globalBestSample.valid = true;
            impl->globalBestSample.mutation = source.mutation;
            impl->globalBestSample.preciseFinish =
                    source.preciseFinish;
            impl->globalBestSample.scriptedObjectiveCount =
                    source.scriptedObjectiveCount;
            for (std::uint32_t i = 0u;
                 i < impl->globalBestSample.scriptedObjectiveCount; ++i) {
                impl->globalBestSample.objectiveScores[i] =
                        source.objectiveScores[i];
                impl->globalBestSample.metricValues[i] =
                        source.metricValues[i];
            }
            impl->globalBestMutationCount = source.mutationCount;
            impl->globalBestInputs = configuration.baselineInputs;
            impl->baselineEvaluated = true;
        }
        if (diagnostic) diagnostic->clear();
        return std::unique_ptr<VulkanSearchExecutor>(
                new VulkanSearchExecutor(std::move(impl)));
    } catch (const std::bad_alloc &) {
        if (diagnostic) *diagnostic =
                "Vulkan search host allocation failed";
        return {};
    } catch (...) {
        if (diagnostic) *diagnostic =
                "unexpected Vulkan search creation failure";
        return {};
    }
}

CudaSearchBatchExecution VulkanSearchExecutor::EvaluateBaseline() noexcept {
    return EvaluateBaseline(std::function<bool()>{});
}

CudaSearchBatchExecution VulkanSearchExecutor::EvaluateBaseline(
        const std::function<bool()> &cancellationRequested) noexcept {
    if (!impl_ || impl_->baselineEvaluated) {
        CudaSearchBatchExecution result;
        result.status = CudaSearchStatus::InvalidArgument;
        result.diagnostic = !impl_
                ? "Vulkan search executor is invalid"
                : "Vulkan baseline was already evaluated";
        return result;
    }
    return impl_->Execute(0u, 1u, true, cancellationRequested);
}

CudaSearchBatchExecution VulkanSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool cancellationRequested) noexcept {
    return RunBatch(
            firstCandidateId, candidateCount,
            cancellationRequested
                    ? std::function<bool()>([] { return true; })
                    : std::function<bool()>{});
}

CudaSearchBatchExecution VulkanSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        const std::function<bool()> &cancellationRequested) noexcept {
    if (!impl_ ||
        (candidateCount != 0u &&
         firstCandidateId >
                 std::numeric_limits<std::uint64_t>::max() -
                         (candidateCount - 1u))) {
        CudaSearchBatchExecution result;
        result.status = CudaSearchStatus::InvalidArgument;
        result.firstCandidateId = firstCandidateId;
        result.candidateCount = candidateCount;
        result.diagnostic = !impl_
                ? "Vulkan search executor is invalid"
                : "Vulkan candidate ID range overflow";
        return result;
    }
    return impl_->Execute(
            firstCandidateId, candidateCount, false,
            cancellationRequested);
}

bool VulkanSearchExecutor::ReserveBatchCapacity(
        std::uint32_t candidateCount,
        std::string *diagnostic) noexcept {
    if (!impl_ || candidateCount == 0u) {
        if (diagnostic) {
            *diagnostic = !impl_
                    ? "Vulkan search executor is invalid"
                    : "Vulkan batch capacity must be positive";
        }
        return false;
    }
    impl_->configuration.maximumBatchSize =
            std::max(impl_->configuration.maximumBatchSize,
                     candidateCount);
    if (diagnostic) diagnostic->clear();
    return true;
}

bool VulkanSearchExecutor::UpdateConditionTimes(
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds) noexcept {
    if (!impl_ || !impl_->configuration.condition ||
        !std::isfinite(lastImprovementTimeSeconds) ||
        !std::isfinite(lastRestartTimeSeconds)) {
        return false;
    }
    impl_->configuration.condition->lastImprovementTimeSeconds =
            lastImprovementTimeSeconds;
    impl_->configuration.condition->lastRestartTimeSeconds =
            lastRestartTimeSeconds;
    return true;
}

std::uint32_t VulkanSearchExecutor::BatchCapacity() const noexcept {
    return impl_ ? impl_->configuration.maximumBatchSize : 0u;
}

}  // namespace forevervalidator::simulation
