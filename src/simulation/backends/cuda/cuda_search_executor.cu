#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
#define FOREVERVALIDATOR_CUDA_RESEARCH_WHEEL_FORCE_MODE 2u
#define FOREVERVALIDATOR_CUDA_RESEARCH_STEADY_INTEGRATION
#define FOREVERVALIDATOR_CUDA_RESEARCH_FULL_ANGULAR_DYNAMICS
#define FOREVERVALIDATOR_CUDA_RESEARCH_ONE_UNIFORM_FORCE_FIELD
#define FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES
#define FOREVERVALIDATOR_CUDA_RESEARCH_FOUR_WHEELS
#define FOREVERVALIDATOR_CUDA_RESEARCH_CANONICAL_WHEEL_FACTS
#define FOREVERVALIDATOR_CUDA_RESEARCH_WATER_ONLY
#endif

#include "simulation/backends/cuda/cuda_search_executor.h"

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#include "simulation/backends/cuda/cuda_candidate_events.cuh"
#include "simulation/backends/cuda/cuda_sparse_candidate_events.cuh"
#include "simulation/backends/cuda/cuda_exact_math.cuh"
#include "simulation/backends/cuda/cuda_finish_time_refinement.cuh"
#include "simulation/backends/cuda/cuda_modifier_event_ops.cuh"
#include "simulation/backends/cuda/cuda_physics_step.cuh"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/backends/cuda/cuda_session_specialization.h"
#include "simulation/backends/cuda/cuda_search_branch_state.cuh"
#include "simulation/backends/cuda/cuda_search_progress.cuh"
#include "simulation/backends/cuda/cuda_search_winner_selection.cuh"
#include "simulation/backends/cuda/cuda_stunts.cuh"
#include "simulation/backends/cuda/cuda_vehicle_transitions.cuh"

namespace forevervalidator::simulation {

#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
extern "C" __device__
CudaPackedStaticConfigurationHeader
ForeverValidatorSessionConfiguration();
extern "C" __device__
CudaPackedSceneHeader ForeverValidatorSessionScene();
#endif

namespace {

constexpr std::uint32_t SimulationBlockSize = 32u;
constexpr std::uint32_t BatchSummaryBlockSize = 128u;
constexpr std::uint32_t ThroughputKernelMinimumBlocksPerSm = 16u;
constexpr std::uint32_t TailKernelMinimumBlocksPerSm = 12u;
constexpr std::uint32_t DenseTailKernelMinimumBlocksPerSm = 8u;
// Keep the state immediately before the most recent unfinished tick. Exact
// finish refinement then replays one tick instead of a long timeline suffix.
constexpr std::uint32_t FinishCheckpointInvalidTick = UINT32_MAX;

template<typename... Arguments, std::size_t... Indices>
CUresult LaunchDriverKernelImpl(
        CUfunction function,
        std::uint32_t blocks,
        std::tuple<Arguments...> &arguments,
        std::index_sequence<Indices...>) {
    std::array<void *, sizeof...(Arguments)> pointers{
            static_cast<void *>(&std::get<Indices>(arguments))...};
    return cuLaunchKernel(
            function,
            blocks, 1u, 1u,
            SimulationBlockSize, 1u, 1u,
            0u, nullptr, pointers.data(), nullptr);
}

template<typename... Arguments>
CUresult LaunchDriverKernel(
        CUfunction function,
        std::uint32_t blocks,
        Arguments... arguments) {
    std::tuple<Arguments...> storage(arguments...);
    return LaunchDriverKernelImpl(
            function,
            blocks,
            storage,
            std::index_sequence_for<Arguments...>{});
}

template<typename... Arguments, std::size_t... Indices>
cudaError_t LaunchRuntimeKernelImpl(
        const void *function,
        std::uint32_t blocks,
        std::tuple<Arguments...> &arguments,
        std::index_sequence<Indices...>) {
    std::array<void *, sizeof...(Arguments)> pointers{
            static_cast<void *>(&std::get<Indices>(arguments))...};
    return cudaLaunchKernel(
            function,
            dim3(blocks, 1u, 1u),
            dim3(SimulationBlockSize, 1u, 1u),
            pointers.data(),
            0u,
            nullptr);
}

template<typename... Arguments>
cudaError_t LaunchRuntimeKernel(
        const void *function,
        std::uint32_t blocks,
        Arguments... arguments) {
    std::tuple<Arguments...> storage(arguments...);
    return LaunchRuntimeKernelImpl(
            function,
            blocks,
            storage,
            std::index_sequence_for<Arguments...>{});
}

enum class DeviceCandidateStatus : std::uint32_t {
    Success,
    Cancelled,
    CapacityExceeded,
    UnsupportedPhysicsTransition,
};

using cuda_search_detail::BetterSample;
using cuda_search_detail::BindSampleToCandidate;
using cuda_search_detail::ControlsFromState;
using cuda_search_detail::DeviceControlState;
using cuda_search_detail::DeviceSample;
using cuda_search_detail::InvalidCandidateSlot;
using cuda_search_detail::ApplyControlEvent;
using cuda_search_detail::StuntsFromState;
using cuda_search_detail::StrictlyBetter;

static_assert(
        sizeof(DeviceSample) == CudaSearchBaselinePrefixSampleBytes,
        "update the CUDA prefix-cache memory bound when DeviceSample changes");
namespace modifier_ops = cuda_search_modifier_detail;
namespace sparse_events = cuda::sparse_candidate_events;

struct DeviceBatchSummary {
    CudaSearchStatus status = CudaSearchStatus::Success;
    std::uint32_t evaluatedCandidateCount = 0u;
    std::uint32_t simulatedCandidateCount = 0u;
    std::uint64_t evaluatorCalls = 0u;
    std::uint64_t qualifyingCandidateCount = 0u;
    double closestTargetDistanceSquared =
            cuda_search_progress_detail::
                    InvalidClosestTargetDistanceSquared;
    std::uint64_t totalMutationCount = 0u;
    std::uint64_t mutationImprovementCount = 0u;
    std::uint32_t globalEventCount = 0u;
    bool bestChanged = false;
    bool bestValid = false;
    bool bestMutation = false;
    std::uint64_t bestCandidateId = 0u;
    std::uint32_t bestMutationCount = 0u;
    std::uint64_t lastFailure = 0u;
};

struct DeviceHotPathRecord {
    std::uint64_t physicallySimulatedCandidateCount = 0u;
    std::uint64_t firstSimulationTick = 0u;
    std::uint64_t executedTickCount = 0u;
    std::uint64_t completedTickCount = 0u;
    cuda::collision::CudaHotPathCounters physics{};
};

template<bool CollectHotPathMetrics>
struct DeviceHotPathStorage {};

template<>
struct DeviceHotPathStorage<true> {
    DeviceHotPathRecord record{};
};

static_assert(std::is_empty_v<DeviceHotPathStorage<false>>);
static_assert(!std::is_empty_v<DeviceHotPathStorage<true>>);

__host__ __device__ constexpr std::size_t AlignUp(
        std::size_t value,
        std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

__host__ __device__ constexpr std::size_t HotPathRecordsOffset(
        std::uint32_t candidateCapacity) {
    const std::size_t blockCapacity =
            (static_cast<std::size_t>(candidateCapacity) +
             SimulationBlockSize - 1u) /
            SimulationBlockSize;
    return AlignUp(
            blockCapacity * sizeof(double),
            alignof(DeviceHotPathRecord));
}

__host__ __device__ constexpr std::size_t HotPathScratchBytes(
        std::uint32_t candidateCapacity) {
    return HotPathRecordsOffset(candidateCapacity) +
            static_cast<std::size_t>(candidateCapacity) *
                    sizeof(DeviceHotPathRecord);
}

void AccumulateHotPathRecord(
        CudaSearchHotPathMetrics *result,
        const DeviceHotPathRecord &record) {
    if (record.physicallySimulatedCandidateCount != 0u) {
        if (result->physicallySimulatedCandidateCount == 0u) {
            result->firstSimulationTickMinimum =
                    record.firstSimulationTick;
        } else {
            result->firstSimulationTickMinimum = std::min(
                    result->firstSimulationTickMinimum,
                    record.firstSimulationTick);
        }
        result->firstSimulationTickMaximum = std::max(
                result->firstSimulationTickMaximum,
                record.firstSimulationTick);
        result->firstSimulationTickSum += record.firstSimulationTick;
        result->physicallySimulatedCandidateCount +=
                record.physicallySimulatedCandidateCount;
    }
    result->executedTickCount += record.executedTickCount;
    result->completedTickCount += record.completedTickCount;
    const cuda::collision::CudaHotPathCounters &physics = record.physics;
    result->physicsSubstepCount += physics.physicsSubstepCount;
    result->maximumSubstepsPerTick = std::max(
            result->maximumSubstepsPerTick,
            physics.maximumSubstepsPerTick);
    result->collisionDetectCount += physics.collisionDetectCount;
    result->surfaceCacheEligibleCount += physics.surfaceCacheEligibleCount;
    result->surfaceCacheReuseCount += physics.surfaceCacheReuseCount;
    result->surfaceCacheRefreshCount += physics.surfaceCacheRefreshCount;
    result->surfaceCacheRefreshFailureCount +=
            physics.surfaceCacheRefreshFailureCount;
    result->meshCacheReuseCount += physics.meshCacheReuseCount;
    result->accelerationCellVisitCount +=
            physics.accelerationCellVisitCount;
    result->accelerationSurfaceVisitCount +=
            physics.accelerationSurfaceVisitCount;
    result->octreeCellVisitCount += physics.octreeCellVisitCount;
    result->cachedTriangleLeafVisitCount +=
            physics.cachedTriangleLeafVisitCount;
    result->triangleTestCount += physics.triangleTestCount;
    result->triangleHitCount += physics.triangleHitCount;
    result->rawContactCount += physics.rawContactCount;
    result->responseSortCallCount += physics.responseSortCallCount;
    result->responseSortItemCount += physics.responseSortItemCount;
    result->maximumResponseSortItemCount = std::max(
            result->maximumResponseSortItemCount,
            physics.maximumResponseSortItemCount);
    result->groundForcePassCount += physics.groundForcePassCount;
    result->airForcePassCount += physics.airForcePassCount;
    result->waterForcePassCount += physics.waterForcePassCount;
    result->physicsCallbackDisabledForcePassCount +=
            physics.physicsCallbackDisabledForcePassCount;
    result->zeroDynamicsForcePassCount +=
            physics.zeroDynamicsForcePassCount;
}

template<typename T>
class DeviceAllocation {
public:
    DeviceAllocation() = default;
    ~DeviceAllocation() { Reset(); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0u)) {}
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
        if (this != &other) {
            Reset();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0u);
        }
        return *this;
    }

    bool Allocate(std::size_t count) {
        Reset();
        if (count == 0u) {
            return true;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return false;
        }
        if (cudaMalloc(
                    reinterpret_cast<void **>(&data_),
                    count * sizeof(T)) != cudaSuccess) {
            data_ = nullptr;
            return false;
        }
        count_ = count;
        return true;
    }

    void Reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        count_ = 0u;
    }

    T *Get() const { return data_; }
    std::size_t Count() const { return count_; }
    std::size_t Bytes() const { return count_ * sizeof(T); }

private:
    T *data_ = nullptr;
    std::size_t count_ = 0u;
};

class MappedCancellation {
public:
    MappedCancellation() = default;
    ~MappedCancellation() { Reset(); }
    MappedCancellation(const MappedCancellation &) = delete;
    MappedCancellation &operator=(const MappedCancellation &) = delete;

    bool Allocate() {
        Reset();
        if (cudaHostAlloc(
                    reinterpret_cast<void **>(&host_),
                    sizeof(*host_),
                    cudaHostAllocMapped | cudaHostAllocPortable) !=
            cudaSuccess) {
            return false;
        }
        if (cudaHostGetDevicePointer(
                    reinterpret_cast<void **>(&device_),
                    host_, 0u) != cudaSuccess) {
            Reset();
            return false;
        }
        *host_ = 0u;
        return true;
    }

    void Reset() {
        if (host_ != nullptr) {
            cudaFreeHost(host_);
        }
        host_ = nullptr;
        device_ = nullptr;
    }

    std::uint32_t *Host() const { return host_; }
    std::uint32_t *Get() const { return device_; }
    std::size_t Bytes() const { return sizeof(std::uint32_t); }

private:
    std::uint32_t *host_ = nullptr;
    std::uint32_t *device_ = nullptr;
};

class Event {
public:
    Event() {
        valid_ = cudaEventCreate(&event_) == cudaSuccess;
    }
    ~Event() {
        if (valid_) {
            cudaEventDestroy(event_);
        }
    }
    bool Valid() const { return valid_; }
    cudaEvent_t Get() const { return event_; }

private:
    cudaEvent_t event_{};
    bool valid_ = false;
};

std::string CudaFailure(const char *operation, cudaError_t error) {
    return std::string(operation) + " failed: " +
            cudaGetErrorName(error) + " (" +
            cudaGetErrorString(error) + ")";
}

bool CanonicalBaselineInputs(
        const std::vector<CudaSearchInputEvent> &inputs,
        std::int64_t mutableFromTimeMs) {
    bool mutableInputSeen = false;
    std::int32_t previousMutableTime = INT32_MIN;
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const CudaSearchInputEvent &input = inputs[index];
        if (input.timeMs < mutableFromTimeMs) {
            if (mutableInputSeen) {
                return false;
            }
            continue;
        }
        mutableInputSeen = true;
        if (input.timeMs < 0 ||
            input.timeMs < previousMutableTime ||
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
        previousMutableTime = input.timeMs;
    }
    return true;
}

std::uint32_t CompactOutputEditCapacity(
        const CudaSearchExecutorConfiguration &configuration) {
    std::uint64_t baselineEdits = 0u;
    std::uint64_t insertedEdits = 0u;
    const auto baselineMatches =
            [&](const CudaSearchModifierConfiguration &modifier,
                const auto &selected) {
                return static_cast<std::uint64_t>(std::count_if(
                        configuration.baselineInputs.begin(),
                        configuration.baselineInputs.end(),
                        [&](const CudaSearchInputEvent &event) {
                            return event.timeMs >=
                                            modifier.window.minimumTimeMs &&
                                    event.timeMs <=
                                            modifier.window.maximumTimeMs &&
                                    selected(event);
                        }));
            };
    for (const CudaSearchModifierConfiguration &modifier :
         configuration.modifiers) {
        switch (modifier.kind) {
        case CudaSearchModifierKind::RandomSteering:
            baselineEdits += baselineMatches(
                    modifier,
                    [](const CudaSearchInputEvent &event) {
                        return event.action == 4u &&
                                event.valueKind == 2u;
                    });
            break;
        case CudaSearchModifierKind::ExistingEvent:
            baselineEdits += modifier.maximumCount;
            break;
        case CudaSearchModifierKind::SmoothSteering: {
            const std::uint64_t windowTicks =
                    static_cast<std::uint64_t>(
                            modifier.window.maximumTimeMs -
                            modifier.window.minimumTimeMs) /
                            configuration.tickDurationMs +
                    1u;
            const std::uint64_t radiusTicks =
                    static_cast<std::uint64_t>(
                            modifier.timeParameterMs) *
                            2u /
                            configuration.tickDurationMs +
                    1u;
            insertedEdits +=
                    static_cast<std::uint64_t>(
                            modifier.minimumCount) *
                    std::min(windowTicks, radiusTicks);
            break;
        }
        case CudaSearchModifierKind::InputInsertion:
            for (const CudaSearchChannel *channel :
                 {&modifier.steering,
                  &modifier.accelerate,
                  &modifier.brake}) {
                if (channel->enabled != 0u) {
                    insertedEdits +=
                            static_cast<std::uint64_t>(
                                    channel->maximumCount) *
                            (channel->maximumHoldMs > 0 ? 2u : 1u);
                }
            }
            break;
        case CudaSearchModifierKind::InputDeletion:
            break;
        }
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
            configuration.maximumEventCount,
            std::min<std::uint64_t>(
                    configuration.baselineInputs.size(),
                    baselineEdits) +
                    insertedEdits));
}

__device__ bool IsAnalog(const CudaSearchInputEvent &event) {
    return event.valueKind == 2u;
}

__device__ bool IsSwitch(const CudaSearchInputEvent &event) {
    return event.valueKind == 1u;
}

__device__ bool IsSteerAction(std::uint32_t action) {
    return action == 4u;
}

__device__ std::int32_t SaturateAnalog(std::int64_t value) {
    if (value < -65536) {
        return -65536;
    }
    if (value > 65536) {
        return 65536;
    }
    return static_cast<std::int32_t>(value);
}

__device__ bool SameEvent(const CudaSearchInputEvent &left,
                          const CudaSearchInputEvent &right) {
    return left.timeMs == right.timeMs &&
            left.action == right.action &&
            left.valueKind == right.valueKind &&
            left.value == right.value;
}

__device__ CudaSearchInputEvent CandidateInputAt(
        const CudaSearchInputEvent *baselineInputs,
        const CudaSearchInputEvent *materializedInputs,
        const std::int32_t *compactValues,
        const std::uint32_t *compactOffsets,
        bool compact,
        std::uint32_t index,
        std::uint32_t candidateSlot,
        std::uint32_t candidateStride) {
    if (!compact) {
        return materializedInputs[index];
    }
    CudaSearchInputEvent result = baselineInputs[index];
    const std::uint32_t offset = compactOffsets[index];
    if (offset != UINT32_MAX) {
        result.value = compactValues[
                static_cast<std::uint64_t>(offset) *
                        candidateStride +
                candidateSlot];
    }
    return result;
}

class CandidateInputCursor {
public:
    __device__ CandidateInputCursor(
            const CudaSearchInputEvent *baselineInputs,
            std::uint32_t baselineInputCount,
            const CudaSearchInputEvent *materializedInputs,
            const std::int32_t *compactValues,
            const std::uint32_t *compactOffsets,
            bool compactRandom,
            bool compactEdits,
            bool sparseEvents,
            cuda::candidate_events::CoalescedEditStorage edits,
            sparse_events::Storage sparseStorage,
            std::uint32_t finalCount,
            std::uint32_t candidateSlot,
            std::uint32_t candidateStride)
        : baselineInputs_(baselineInputs),
          materializedInputs_(materializedInputs),
          compactValues_(compactValues),
          compactOffsets_(compactOffsets),
          compactRandom_(compactRandom),
          compactEdits_(compactEdits),
          sparseEvents_(sparseEvents),
          candidateSlot_(candidateSlot),
          candidateStride_(candidateStride),
          finalCount_(finalCount),
          editCursor_({
                  {baselineInputs, baselineInputCount, 0},
                  edits,
                  candidateSlot,
                  finalCount}),
          sparseCursor_(
                  baselineInputs, sparseStorage,
                  candidateSlot, finalCount) {}

    __device__ bool Next(CudaSearchInputEvent *event) {
        if (sparseEvents_) {
            return sparseCursor_.Next(event);
        }
        if (compactEdits_) {
            return editCursor_.Next(event);
        }
        if (index_ >= finalCount_) {
            return false;
        }
        *event = CandidateInputAt(
                baselineInputs_, materializedInputs_,
                compactValues_, compactOffsets_, compactRandom_,
                index_++, candidateSlot_, candidateStride_);
        return true;
    }

private:
    const CudaSearchInputEvent *baselineInputs_ = nullptr;
    const CudaSearchInputEvent *materializedInputs_ = nullptr;
    const std::int32_t *compactValues_ = nullptr;
    const std::uint32_t *compactOffsets_ = nullptr;
    bool compactRandom_ = false;
    bool compactEdits_ = false;
    bool sparseEvents_ = false;
    std::uint32_t candidateSlot_ = 0u;
    std::uint32_t candidateStride_ = 0u;
    std::uint32_t finalCount_ = 0u;
    std::uint32_t index_ = 0u;
    cuda::candidate_events::CandidateCursor editCursor_;
    sparse_events::Cursor sparseCursor_;
};

// Reuses the common prefix already discovered while finding the first
// divergent event. The tail cursor begins immediately after the first
// unmatched candidate event, so locality sampling does not rescan the
// candidate's entire input stream.
class CandidateLocalityCursor {
public:
    __device__ CandidateLocalityCursor(
            const CudaSearchInputEvent *commonEvents,
            std::uint32_t commonCount,
            CandidateInputCursor *tail,
            bool hasPendingEvent,
            const CudaSearchInputEvent &pendingEvent)
        : commonEvents_(commonEvents),
          commonCount_(commonCount),
          tail_(tail),
          hasPendingEvent_(hasPendingEvent),
          pendingEvent_(pendingEvent) {}

    __device__ bool Next(CudaSearchInputEvent *event) {
        if (commonIndex_ < commonCount_) {
            *event = commonEvents_[commonIndex_++];
            return true;
        }
        if (hasPendingEvent_) {
            *event = pendingEvent_;
            hasPendingEvent_ = false;
            return true;
        }
        return tail_->Next(event);
    }

private:
    const CudaSearchInputEvent *commonEvents_ = nullptr;
    std::uint32_t commonCount_ = 0u;
    std::uint32_t commonIndex_ = 0u;
    CandidateInputCursor *tail_ = nullptr;
    bool hasPendingEvent_ = false;
    CudaSearchInputEvent pendingEvent_{};
};

class DeviceMt19937 {
public:
    __device__ DeviceMt19937(std::uint32_t *stateWords,
                             std::uint32_t slot,
                             std::uint32_t stride)
        : stateWords_(stateWords), slot_(slot), stride_(stride) {}

    __device__ void Seed(std::uint32_t seed,
                         std::uint64_t candidateId,
                         std::uint32_t passIndex) {
        const std::uint32_t seeds[4]{
                seed,
                static_cast<std::uint32_t>(candidateId),
                static_cast<std::uint32_t>(candidateId >> 32u),
                passIndex};
        for (std::uint32_t index = 0u; index < 624u; ++index) {
            State(index) = 0x8b8b8b8bu;
        }
        constexpr std::uint32_t n = 624u;
        constexpr std::uint32_t s = 4u;
        constexpr std::uint32_t p = 306u;
        constexpr std::uint32_t q = 317u;
        {
            const std::uint32_t r1 = 1371501266u;
            const std::uint32_t r2 = r1 + s;
            State(p) += r1;
            State(q) += r2;
            State(0u) = r2;
        }
        for (std::uint32_t k = 1u; k <= s; ++k) {
            const std::uint32_t kn = k % n;
            const std::uint32_t kpn = (k + p) % n;
            const std::uint32_t kqn = (k + q) % n;
            const std::uint32_t argument =
                    State(kn) ^ State(kpn) ^ State((k - 1u) % n);
            const std::uint32_t r1 =
                    1664525u * (argument ^ (argument >> 27u));
            const std::uint32_t r2 = r1 + kn + seeds[k - 1u];
            State(kpn) += r1;
            State(kqn) += r2;
            State(kn) = r2;
        }
        for (std::uint32_t k = s + 1u; k < n; ++k) {
            const std::uint32_t kn = k % n;
            const std::uint32_t kpn = (k + p) % n;
            const std::uint32_t kqn = (k + q) % n;
            const std::uint32_t argument =
                    State(kn) ^ State(kpn) ^ State((k - 1u) % n);
            const std::uint32_t r1 =
                    1664525u * (argument ^ (argument >> 27u));
            const std::uint32_t r2 = r1 + kn;
            State(kpn) += r1;
            State(kqn) += r2;
            State(kn) = r2;
        }
        for (std::uint32_t k = n; k < 2u * n; ++k) {
            const std::uint32_t kn = k % n;
            const std::uint32_t kpn = (k + p) % n;
            const std::uint32_t kqn = (k + q) % n;
            const std::uint32_t argument =
                    State(kn) + State(kpn) + State((k - 1u) % n);
            const std::uint32_t r3 =
                    1566083941u * (argument ^ (argument >> 27u));
            const std::uint32_t r4 = r3 - kn;
            State(kpn) ^= r3;
            State(kqn) ^= r4;
            State(kn) = r4;
        }
        cursor_ = n;
    }

    __device__ std::uint32_t Next() {
        if (cursor_ >= 624u) {
            Twist();
        }
        std::uint32_t value = State(cursor_++);
        value ^= value >> 11u;
        value ^= (value << 7u) & 0x9d2c5680u;
        value ^= (value << 15u) & 0xefc60000u;
        value ^= value >> 18u;
        return value;
    }

    __device__ std::uint64_t UniformUnsigned(std::uint64_t minimum,
                                             std::uint64_t maximum) {
        if (minimum > maximum) {
            const std::uint64_t swap = minimum;
            minimum = maximum;
            maximum = swap;
        }
        const std::uint64_t range = maximum - minimum;
        std::uint64_t result = 0u;
        if (range < UINT32_MAX) {
            const std::uint32_t extendedRange =
                    static_cast<std::uint32_t>(range + 1u);
            std::uint64_t product =
                    static_cast<std::uint64_t>(Next()) * extendedRange;
            std::uint32_t low = static_cast<std::uint32_t>(product);
            if (low < extendedRange) {
                const std::uint32_t threshold =
                        static_cast<std::uint32_t>(-extendedRange) %
                        extendedRange;
                while (low < threshold) {
                    product =
                            static_cast<std::uint64_t>(Next()) *
                            extendedRange;
                    low = static_cast<std::uint32_t>(product);
                }
            }
            result = product >> 32u;
        } else if (range == UINT32_MAX) {
            result = Next();
        } else {
            do {
                constexpr std::uint64_t generatorRange =
                        UINT64_C(1) << 32u;
                const std::uint64_t high = UniformUnsigned(
                        0u, range / generatorRange);
                const std::uint64_t temporary =
                        generatorRange * high;
                result = temporary + Next();
                if (result <= range && result >= temporary) {
                    break;
                }
            } while (true);
        }
        return result + minimum;
    }

    __device__ std::uint32_t UniformU32(std::uint32_t minimum,
                                       std::uint32_t maximum) {
        return static_cast<std::uint32_t>(
                UniformUnsigned(minimum, maximum));
    }

    __device__ std::int32_t UniformS32(std::int32_t minimum,
                                      std::int32_t maximum) {
        if (minimum > maximum) {
            const std::int32_t swap = minimum;
            minimum = maximum;
            maximum = swap;
        }
        const std::uint32_t unsignedMinimum =
                static_cast<std::uint32_t>(minimum);
        const std::uint32_t range =
                static_cast<std::uint32_t>(maximum) - unsignedMinimum;
        return static_cast<std::int32_t>(
                static_cast<std::uint32_t>(
                        UniformUnsigned(0u, range)) +
                unsignedMinimum);
    }

    __device__ std::int64_t UniformS64(std::int64_t minimum,
                                      std::int64_t maximum) {
        if (minimum > maximum) {
            const std::int64_t swap = minimum;
            minimum = maximum;
            maximum = swap;
        }
        const std::uint64_t unsignedMinimum =
                static_cast<std::uint64_t>(minimum);
        const std::uint64_t range =
                static_cast<std::uint64_t>(maximum) - unsignedMinimum;
        return static_cast<std::int64_t>(
                UniformUnsigned(0u, range) + unsignedMinimum);
    }

private:
    __device__ std::uint32_t &State(std::uint32_t index) {
        return stateWords_[
                static_cast<std::uint64_t>(index) * stride_ + slot_];
    }

    __device__ void Twist() {
        constexpr std::uint32_t upperMask = 0x80000000u;
        constexpr std::uint32_t lowerMask = 0x7fffffffu;
        constexpr std::uint32_t coefficient = 0x9908b0dfu;
        for (std::uint32_t index = 0u; index < 227u; ++index) {
            const std::uint32_t value =
                    (State(index) & upperMask) |
                    (State(index + 1u) & lowerMask);
            State(index) = State(index + 397u) ^
                    (value >> 1u) ^
                    ((value & 1u) ? coefficient : 0u);
        }
        for (std::uint32_t index = 227u; index < 623u; ++index) {
            const std::uint32_t value =
                    (State(index) & upperMask) |
                    (State(index + 1u) & lowerMask);
            State(index) = State(index - 227u) ^
                    (value >> 1u) ^
                    ((value & 1u) ? coefficient : 0u);
        }
        const std::uint32_t value =
                (State(623u) & upperMask) |
                (State(0u) & lowerMask);
        State(623u) = State(396u) ^ (value >> 1u) ^
                ((value & 1u) ? coefficient : 0u);
        cursor_ = 0u;
    }

    std::uint32_t *stateWords_ = nullptr;
    std::uint32_t slot_ = 0u;
    std::uint32_t stride_ = 0u;
    std::uint32_t cursor_ = 624u;
};

__device__ void ShuffleIndices(std::uint32_t *indices,
                               std::uint32_t count,
                               DeviceMt19937 &random) {
    if (count <= 1u) {
        return;
    }
    const std::uint64_t generatorRange = UINT32_MAX;
    const std::uint64_t range = count;
    if (generatorRange / range >= range) {
        std::uint32_t index = 1u;
        if ((range % 2u) == 0u) {
            const std::uint32_t selected =
                    random.UniformU32(0u, 1u);
            const std::uint32_t swap = indices[index];
            indices[index] = indices[selected];
            indices[selected] = swap;
            ++index;
        }
        while (index != count) {
            const std::uint64_t swapRange =
                    static_cast<std::uint64_t>(index) + 1u;
            const std::uint64_t position = random.UniformUnsigned(
                    0u, swapRange * (swapRange + 1u) - 1u);
            const std::uint32_t first =
                    static_cast<std::uint32_t>(
                            position / (swapRange + 1u));
            const std::uint32_t second =
                    static_cast<std::uint32_t>(
                            position % (swapRange + 1u));
            std::uint32_t swap = indices[index];
            indices[index] = indices[first];
            indices[first] = swap;
            ++index;
            swap = indices[index];
            indices[index] = indices[second];
            indices[second] = swap;
            ++index;
        }
        return;
    }
    for (std::uint32_t index = 1u; index < count; ++index) {
        const std::uint32_t selected =
                random.UniformU32(0u, index);
        const std::uint32_t swap = indices[index];
        indices[index] = indices[selected];
        indices[selected] = swap;
    }
}

__device__ std::uint32_t NormalizeEvents(
        CudaSearchInputEvent *events,
        std::uint32_t count,
        CudaSearchInputEvent *temporary,
        const CudaSearchInputEvent *passBaseline,
        std::uint32_t passBaselineCount,
        std::int64_t mutableFromTimeMs,
        bool legacyMutationPipeline,
        std::uint32_t capacity) {
    (void)legacyMutationPipeline;
    return cuda::candidate_events::NormalizeWithPrefix(
            events, count, temporary,
            passBaseline, passBaselineCount,
            mutableFromTimeMs, capacity);
}

__device__ std::uint32_t EffectiveChangeCount(
        const CudaSearchInputEvent *baseline,
        std::uint32_t baselineCount,
        const CudaSearchInputEvent *events,
        std::uint32_t eventCount) {
    const std::uint32_t common =
            baselineCount < eventCount ? baselineCount : eventCount;
    std::uint32_t result = baselineCount > eventCount
            ? baselineCount - eventCount
            : eventCount - baselineCount;
    for (std::uint32_t index = 0u; index < common; ++index) {
        if (!SameEvent(baseline[index], events[index])) {
            ++result;
        }
    }
    return result;
}

__device__ bool EncodeCandidateEdits(
        const CudaSearchInputEvent *baseline,
        std::uint32_t baselineCount,
        const CudaSearchInputEvent *events,
        std::uint32_t eventCount,
        std::uint32_t *sourceStates,
        cuda::candidate_events::CoalescedEditStorage storage,
        std::uint32_t slot) {
    storage.counts[slot] = 0u;
    storage.erasedCounts[slot] = 0u;
    for (std::uint32_t source = 0u;
         source < baselineCount; ++source) {
        sourceStates[source] = 0u;
    }

    cuda::candidate_events::EditWriter writer(storage, slot);
    std::uint32_t nextSource = 0u;
    for (std::uint32_t output = 0u;
         output < eventCount; ++output) {
        std::uint32_t source = nextSource;
        while (source < baselineCount &&
               !SameEvent(events[output], baseline[source])) {
            ++source;
        }
        if (source < baselineCount) {
            sourceStates[source] = 1u;
            nextSource = source + 1u;
        } else if (!writer.Insert(output, events[output])) {
            return false;
        }
    }
    for (std::uint32_t source = 0u;
         source < baselineCount; ++source) {
        if (sourceStates[source] != 1u &&
            !writer.Erase(source)) {
            return false;
        }
    }
    return true;
}

__device__ std::int32_t SteeringStateAt(
        const CudaSearchInputEvent *events,
        std::uint32_t count,
        std::int64_t timeMs,
        std::int32_t initialState,
        bool sortedByTime = false) {
    return modifier_ops::ChannelStateAt(
            events, count, 4u, 2u, timeMs,
            sortedByTime, initialState);
}

__device__ bool SwitchStateAt(
        const CudaSearchInputEvent *events,
        std::uint32_t count,
        std::uint32_t action,
        std::int64_t timeMs,
        bool initialState,
        bool sortedByTime = false) {
    return modifier_ops::ChannelStateAt(
                   events, count, action, 1u,
                   timeMs, sortedByTime,
                   initialState ? 1 : 0) != 0;
}

__device__ bool PushEvent(CudaSearchInputEvent *events,
                          std::uint32_t *count,
                          std::uint32_t capacity,
                          CudaSearchInputEvent event) {
    if (*count >= capacity) {
        return false;
    }
    events[*count] = event;
    ++*count;
    return true;
}

__device__ CudaSearchInputEvent AnalogEvent(
        std::int64_t timeMs,
        std::uint32_t action,
        std::int32_t value) {
    return {static_cast<std::int32_t>(timeMs), action, 2u, value};
}

__device__ CudaSearchInputEvent SwitchEvent(
        std::int64_t timeMs,
        std::uint32_t action,
        bool value) {
    return {static_cast<std::int32_t>(timeMs),
            action,
            1u,
            value ? 1 : 0};
}

__device__ std::uint32_t CollectEligible(
        const CudaSearchInputEvent *events,
        std::uint32_t count,
        std::uint32_t *eligible,
        const CudaSearchModifierConfiguration &modifier,
        bool sortedByTime) {
    return modifier_ops::CollectExistingEventEligible(
            events, count, eligible,
            modifier.window.minimumTimeMs,
            modifier.window.maximumTimeMs,
            modifier.optionFlags, sortedByTime);
}

__device__ bool ApplyModifier(
        const CudaSearchModifierConfiguration &modifier,
        std::uint32_t passIndex,
        std::uint64_t candidateId,
        DeviceMt19937 &random,
        std::uint32_t tickDurationMs,
        std::int64_t mutableFromTimeMs,
        const DeviceControlState &initialControls,
        const CudaSearchInputEvent *globalBaseline,
        std::uint32_t globalBaselineCount,
        CudaSearchInputEvent *events,
        std::uint32_t *eventCount,
        std::uint32_t eventCapacity,
        CudaSearchInputEvent *temporary,
        CudaSearchInputEvent *passBaseline,
        std::uint32_t *eligible,
        const double *smoothWeights,
        bool legacyMutationPipeline,
        bool *normalized) {
    const std::uint32_t passBaselineCount = *eventCount;
    const bool passBaselineCanonical = *normalized;
    const bool insertionRestoresHeldState =
            modifier.kind == CudaSearchModifierKind::InputInsertion &&
            ((modifier.steering.enabled != 0u &&
              modifier.steering.maximumHoldMs > 0) ||
             (modifier.accelerate.enabled != 0u &&
              modifier.accelerate.maximumHoldMs > 0) ||
             (modifier.brake.enabled != 0u &&
              modifier.brake.maximumHoldMs > 0));
    const bool needsPassSnapshot =
            legacyMutationPipeline ||
            insertionRestoresHeldState ||
            modifier.window.minimumTimeMs < mutableFromTimeMs;
    if (needsPassSnapshot) {
        for (std::uint32_t index = 0u;
             index < passBaselineCount; ++index) {
            passBaseline[index] = events[index];
        }
    }
    const CudaSearchInputEvent *normalizationBaseline =
            needsPassSnapshot ? passBaseline : globalBaseline;
    const std::uint32_t normalizationBaselineCount =
            needsPassSnapshot ? passBaselineCount : globalBaselineCount;
    random.Seed(modifier.window.seed, candidateId, passIndex);

    switch (modifier.kind) {
    case CudaSearchModifierKind::RandomSteering: {
        const std::uint32_t begin = *normalized
                ? modifier_ops::LowerBoundTime(
                          events, *eventCount,
                          modifier.window.minimumTimeMs)
                : 0u;
        const std::uint32_t end = *normalized
                ? modifier_ops::UpperBoundTime(
                          events, *eventCount,
                          modifier.window.maximumTimeMs)
                : *eventCount;
        for (std::uint32_t index = begin; index < end; ++index) {
            CudaSearchInputEvent &event = events[index];
            if ((!*normalized &&
                 (event.timeMs < modifier.window.minimumTimeMs ||
                  event.timeMs > modifier.window.maximumTimeMs)) ||
                event.action != 4u || !IsAnalog(event)) {
                continue;
            }
            std::int32_t value =
                    random.UniformS32(-65536, 65536);
            if (value == event.value) {
                value = value == 65536 ? -65536 : 65536;
            }
            event.value = value;
        }
        if (!legacyMutationPipeline &&
            modifier.window.minimumTimeMs >= mutableFromTimeMs &&
            *normalized) {
            break;
        }
        *eventCount = NormalizeEvents(
                events, *eventCount, temporary,
                normalizationBaseline, normalizationBaselineCount,
                mutableFromTimeMs, legacyMutationPipeline,
                eventCapacity);
        if (*eventCount == UINT32_MAX) {
            return false;
        }
        *normalized = true;
        break;
    }
    case CudaSearchModifierKind::ExistingEvent: {
        const std::uint32_t eligibleCount = CollectEligible(
                events, *eventCount, eligible, modifier, *normalized);
        if (eligibleCount == 0u) {
            break;
        }
        ShuffleIndices(eligible, eligibleCount, random);
        const std::uint32_t requested = random.UniformU32(
                modifier.minimumCount, modifier.maximumCount);
        const std::uint32_t count =
                requested < eligibleCount ? requested : eligibleCount;
        const std::int64_t maximumShiftTicks =
                modifier.timeParameterMs /
                static_cast<std::int64_t>(tickDurationMs);
        for (std::uint32_t index = 0u; index < count; ++index) {
            CudaSearchInputEvent &event = events[eligible[index]];
            const std::int64_t shiftTicks = random.UniformS64(
                    -maximumShiftTicks, maximumShiftTicks);
            std::int64_t time =
                    static_cast<std::int64_t>(event.timeMs) +
                    shiftTicks * tickDurationMs;
            if (time < modifier.window.minimumTimeMs) {
                time = modifier.window.minimumTimeMs;
            }
            if (time > modifier.window.maximumTimeMs) {
                time = modifier.window.maximumTimeMs;
            }
            event.timeMs = static_cast<std::int32_t>(time);
            if (IsSteerAction(event.action)) {
                if ((modifier.optionFlags & 1u) != 0u) {
                    event.value = random.UniformS32(
                            modifier.secondaryAnalogMinimum,
                            modifier.secondaryAnalogMaximum);
                } else {
                    const std::int32_t delta = random.UniformS32(
                            modifier.analogMinimum,
                            modifier.analogMaximum);
                    event.value = SaturateAnalog(
                            static_cast<std::int64_t>(event.value) +
                            delta);
                }
            } else if (IsSwitch(event)) {
                event.value = event.value != 0 ? 0 : 1;
            }
        }
        if (!legacyMutationPipeline &&
            modifier.timeParameterMs == 0 &&
            modifier.window.minimumTimeMs >= mutableFromTimeMs &&
            *normalized) {
            break;
        }
        *eventCount = NormalizeEvents(
                events, *eventCount, temporary,
                normalizationBaseline, normalizationBaselineCount,
                mutableFromTimeMs, legacyMutationPipeline,
                eventCapacity);
        if (*eventCount == UINT32_MAX) {
            return false;
        }
        *normalized = true;
        break;
    }
    case CudaSearchModifierKind::SmoothSteering:
        for (std::uint32_t deformation = 0u;
             deformation < modifier.minimumCount; ++deformation) {
            const std::uint32_t deformationBaselineCount =
                    *eventCount;
            const bool deformationBaselineCanonical =
                    *normalized;
            const std::int64_t minimumTick =
                    modifier.window.minimumTimeMs / tickDurationMs;
            const std::int64_t maximumTick =
                    modifier.window.maximumTimeMs / tickDurationMs;
            const std::int64_t center =
                    random.UniformS64(minimumTick, maximumTick) *
                    tickDurationMs;
            const std::int32_t amplitude = random.UniformS32(
                    modifier.analogMinimum, modifier.analogMaximum);
            std::int64_t start =
                    center - modifier.timeParameterMs;
            if (start < modifier.window.minimumTimeMs) {
                start = modifier.window.minimumTimeMs;
            }
            std::int64_t end = center + modifier.timeParameterMs;
            if (end > modifier.window.maximumTimeMs) {
                end = modifier.window.maximumTimeMs;
            }
            start = start <= 0
                    ? 0
                    : (start / tickDurationMs) * tickDurationMs;
            for (std::int64_t time = start;
                 time <= end; time += tickDurationMs) {
                const std::uint64_t distance =
                        static_cast<std::uint64_t>(
                                time > center ? time - center
                                              : center - time);
                const std::uint32_t weightIndex =
                        modifier.weightOffset +
                        static_cast<std::uint32_t>(
                                distance / tickDurationMs);
                const std::int64_t delta = static_cast<std::int64_t>(
                        llround(static_cast<double>(amplitude) *
                                smoothWeights[weightIndex]));
                const std::int32_t steeringState =
                        deformationBaselineCanonical
                        ? modifier_ops::
                                  ChannelStateAtWithAppendedRun(
                                          events,
                                          deformationBaselineCount,
                                          *eventCount,
                                          4u, 2u, time,
                                          initialControls.steerValue)
                        : SteeringStateAt(
                                  events, *eventCount, time,
                                  initialControls.steerValue);
                const std::int32_t value = SaturateAnalog(
                        static_cast<std::int64_t>(steeringState) +
                        delta);
                if (!PushEvent(
                            events, eventCount, eventCapacity,
                            AnalogEvent(time, 4u, value))) {
                    return false;
                }
            }
            *eventCount = NormalizeEvents(
                    events, *eventCount, temporary,
                    normalizationBaseline, normalizationBaselineCount,
                    mutableFromTimeMs, legacyMutationPipeline,
                    eventCapacity);
            if (*eventCount == UINT32_MAX) {
                return false;
            }
            *normalized = true;
        }
        break;
    case CudaSearchModifierKind::InputInsertion: {
        const auto randomTime = [&]() {
            return random.UniformS64(
                           modifier.window.minimumTimeMs /
                                   tickDurationMs,
                           modifier.window.maximumTimeMs /
                                   tickDurationMs) *
                    tickDurationMs;
        };
        const auto randomHold = [&](std::int64_t maximum) {
            return maximum <= 0
                    ? INT64_C(0)
                    : random.UniformS64(
                                      0, maximum / tickDurationMs) *
                              tickDurationMs;
        };
        if (modifier.steering.enabled != 0u) {
            const std::uint32_t count = random.UniformU32(
                    modifier.steering.minimumCount,
                    modifier.steering.maximumCount);
            for (std::uint32_t index = 0u; index < count; ++index) {
                const std::int64_t start = randomTime();
                std::int64_t end =
                        start + randomHold(
                                        modifier.steering.maximumHoldMs);
                if (end > modifier.window.maximumTimeMs) {
                    end = modifier.window.maximumTimeMs;
                }
                const std::int32_t previous =
                        modifier_ops::RemoveActionRangeAndReadState(
                                events, eventCount, 4u, 2u,
                                start, end,
                                initialControls.steerValue);
                const std::int32_t value =
                        (modifier.optionFlags & 1u) != 0u
                        ? SaturateAnalog(
                                  static_cast<std::int64_t>(previous) +
                                  random.UniformS32(
                                          modifier.secondaryAnalogMinimum,
                                          modifier.secondaryAnalogMaximum))
                        : random.UniformS32(
                                  modifier.analogMinimum,
                                  modifier.analogMaximum);
                if (!PushEvent(
                            events, eventCount, eventCapacity,
                            AnalogEvent(start, 4u, value))) {
                    return false;
                }
                if (end > start &&
                    !PushEvent(
                            events, eventCount, eventCapacity,
                            AnalogEvent(
                                    end, 4u,
                                    SteeringStateAt(
                                            passBaseline,
                                            passBaselineCount,
                                            end,
                                            initialControls.steerValue,
                                            passBaselineCanonical)))) {
                    return false;
                }
            }
        }
        const auto insertSwitch =
                [&](const CudaSearchChannel &channel,
                    std::uint32_t action) {
                    if (channel.enabled == 0u) {
                        return true;
                    }
                    const std::uint32_t count = random.UniformU32(
                            channel.minimumCount,
                            channel.maximumCount);
                    for (std::uint32_t index = 0u;
                         index < count; ++index) {
                        const std::int64_t start = randomTime();
                        std::int64_t end =
                                start + randomHold(
                                                channel.maximumHoldMs);
                        if (end > modifier.window.maximumTimeMs) {
                            end = modifier.window.maximumTimeMs;
                        }
                        const bool previous =
                                modifier_ops::
                                        RemoveActionRangeAndReadState(
                                                events, eventCount,
                                                action, 1u,
                                                start, end,
                                                action == 1u
                                                        ? initialControls.
                                                                  accelerate
                                                        : initialControls.
                                                                  brake) != 0;
                        if (!PushEvent(
                                    events, eventCount,
                                    eventCapacity,
                                    SwitchEvent(
                                            start, action, !previous))) {
                            return false;
                        }
                        if (end > start &&
                            !PushEvent(
                                    events, eventCount,
                                    eventCapacity,
                                    SwitchEvent(
                                            end, action,
                                            SwitchStateAt(
                                                    passBaseline,
                                                    passBaselineCount,
                                                    action, end,
                                                    action == 1u
                                                            ? initialControls.
                                                                      accelerate
                                                            : initialControls.
                                                                      brake,
                                                    passBaselineCanonical)))) {
                            return false;
                        }
                    }
                    return true;
                };
        if (!insertSwitch(modifier.accelerate, 1u) ||
            !insertSwitch(modifier.brake, 3u)) {
            return false;
        }
        *eventCount = NormalizeEvents(
                events, *eventCount, temporary,
                normalizationBaseline, normalizationBaselineCount,
                mutableFromTimeMs, legacyMutationPipeline,
                eventCapacity);
        if (*eventCount == UINT32_MAX) {
            return false;
        }
        *normalized = true;
        break;
    }
    case CudaSearchModifierKind::InputDeletion: {
        const auto deleteChannel =
                [&](const CudaSearchChannel &channel,
                    std::uint32_t kind) {
                    if (channel.enabled == 0u) {
                        return;
                    }
                    const std::uint32_t requested =
                            random.UniformU32(
                                    0u, channel.maximumCount);
                    if (requested == 0u) {
                        return;
                    }
                    const std::uint32_t initialEligibleCount =
                            modifier_ops::CollectDeletionEligible(
                                    events, *eventCount, eligible,
                                    modifier.window.minimumTimeMs,
                                    modifier.window.maximumTimeMs,
                                    kind, *normalized);
                    std::uint32_t eligibleCount =
                            initialEligibleCount;
                    for (std::uint32_t removal = 0u;
                         removal < requested; ++removal) {
                        if (eligibleCount == 0u) {
                            break;
                        }
                        modifier_ops::SelectDeletionRank(
                                eligible, &eligibleCount,
                                random.UniformU32(
                                        0u, eligibleCount - 1u));
                    }
                    if (eligibleCount != initialEligibleCount) {
                        modifier_ops::CompactSelectedDeletionTail(
                                events, eventCount, eligible,
                                eligibleCount, initialEligibleCount);
                    }
                };
        deleteChannel(modifier.steering, 0u);
        deleteChannel(modifier.accelerate, 1u);
        deleteChannel(modifier.brake, 2u);
        if (!legacyMutationPipeline &&
            modifier.window.minimumTimeMs >= mutableFromTimeMs &&
            *normalized) {
            break;
        }
        *eventCount = NormalizeEvents(
                events, *eventCount, temporary,
                normalizationBaseline, normalizationBaselineCount,
                mutableFromTimeMs, legacyMutationPipeline,
                eventCapacity);
        if (*eventCount == UINT32_MAX) {
            return false;
        }
        *normalized = true;
        break;
    }
    }
    return true;
}

__device__ bool SparseExistingEventEligible(
        const CudaSearchInputEvent &event,
        const CudaSearchModifierConfiguration &modifier) {
    return event.timeMs >= modifier.window.minimumTimeMs &&
            event.timeMs <= modifier.window.maximumTimeMs &&
            ((event.action == 4u && event.valueKind == 2u) ||
             ((modifier.optionFlags & 2u) != 0u &&
              (event.action == 1u || event.action == 2u)) ||
             ((modifier.optionFlags & 4u) != 0u &&
              event.action == 3u));
}

__device__ bool SparseDeletionEligible(
        const CudaSearchInputEvent &event,
        const CudaSearchModifierConfiguration &modifier,
        std::uint32_t group) {
    return event.timeMs >= modifier.window.minimumTimeMs &&
            event.timeMs <= modifier.window.maximumTimeMs &&
            modifier_ops::ActionInGroup(event.action, group);
}

template<typename Predicate>
__device__ std::uint32_t CollectSparseEligible(
        const sparse_events::Candidate &candidate,
        std::uint32_t *eligible,
        const std::uint32_t *sharedEligible,
        std::uint32_t sharedEligibleCount,
        bool identityReferences,
        Predicate predicate) {
    std::uint32_t result = 0u;
    if (identityReferences && sharedEligible != nullptr) {
        for (std::uint32_t index = 0u;
             index < sharedEligibleCount; ++index) {
            const std::uint32_t ordinal = sharedEligible[index];
            if (ordinal < candidate.Count() &&
                predicate(candidate.EventAt(ordinal))) {
                eligible[result++] = ordinal;
            }
        }
        return result;
    }
    for (std::uint32_t ordinal = 0u;
         ordinal < candidate.Count(); ++ordinal) {
        if (predicate(candidate.EventAt(ordinal))) {
            eligible[result++] = ordinal;
        }
    }
    return result;
}

__device__ void EraseSparseSelectedTail(
        sparse_events::Candidate *candidate,
        std::uint32_t *eligible,
        std::uint32_t remaining,
        std::uint32_t initialEligibleCount) {
    for (std::uint32_t index = remaining + 1u;
         index < initialEligibleCount; ++index) {
        const std::uint32_t value = eligible[index];
        std::uint32_t insertion = index;
        while (insertion > remaining &&
               eligible[insertion - 1u] > value) {
            eligible[insertion] = eligible[insertion - 1u];
            --insertion;
        }
        eligible[insertion] = value;
    }
    candidate->EraseSortedOrdinals(
            eligible + remaining,
            initialEligibleCount - remaining);
}

__device__ std::uint32_t SparseEffectiveChangeCount(
        const CudaSearchInputEvent *baseline,
        std::uint32_t baselineCount,
        const sparse_events::Candidate &candidate) {
    const std::uint32_t eventCount = candidate.Count();
    const std::uint32_t common =
            baselineCount < eventCount ? baselineCount : eventCount;
    std::uint32_t result = baselineCount > eventCount
            ? baselineCount - eventCount
            : eventCount - baselineCount;
    for (std::uint32_t index = 0u; index < common; ++index) {
        if (!SameEvent(baseline[index], candidate.EventAt(index))) {
            ++result;
        }
    }
    return result;
}

__device__ bool ApplySparseModifier(
        const CudaSearchModifierConfiguration &modifier,
        std::uint32_t passIndex,
        std::uint64_t candidateId,
        DeviceMt19937 &random,
        std::uint32_t tickDurationMs,
        const DeviceControlState &initialControls,
        sparse_events::Candidate *candidate,
        std::uint32_t *eligible,
        const std::uint32_t *sharedEligible,
        std::uint32_t sharedEligibleCount,
        bool identityReferences,
        const double *smoothWeights) {
    random.Seed(modifier.window.seed, candidateId, passIndex);

    switch (modifier.kind) {
    case CudaSearchModifierKind::RandomSteering:
        for (std::uint32_t index = 0u;
             index < candidate->Count(); ++index) {
            CudaSearchInputEvent event = candidate->EventAt(index);
            if (event.timeMs < modifier.window.minimumTimeMs ||
                event.timeMs > modifier.window.maximumTimeMs ||
                event.action != 4u || !IsAnalog(event)) {
                continue;
            }
            std::int32_t value =
                    random.UniformS32(-65536, 65536);
            if (value == event.value) {
                value = value == 65536 ? -65536 : 65536;
            }
            event.value = value;
            if (!candidate->SetAt(index, event)) {
                return false;
            }
        }
        break;
    case CudaSearchModifierKind::ExistingEvent: {
        const std::uint32_t eligibleCount = CollectSparseEligible(
                *candidate, eligible, sharedEligible,
                sharedEligibleCount, identityReferences,
                [&](const CudaSearchInputEvent &event) {
                    return SparseExistingEventEligible(event, modifier);
                });
        if (eligibleCount == 0u) {
            break;
        }
        ShuffleIndices(eligible, eligibleCount, random);
        const std::uint32_t requested = random.UniformU32(
                modifier.minimumCount, modifier.maximumCount);
        const std::uint32_t count =
                requested < eligibleCount ? requested : eligibleCount;
        const std::int64_t maximumShiftTicks =
                modifier.timeParameterMs /
                static_cast<std::int64_t>(tickDurationMs);
        for (std::uint32_t index = 0u; index < count; ++index) {
            const std::uint32_t ordinal = eligible[index];
            CudaSearchInputEvent event = candidate->EventAt(ordinal);
            const std::int64_t shiftTicks = random.UniformS64(
                    -maximumShiftTicks, maximumShiftTicks);
            std::int64_t time =
                    static_cast<std::int64_t>(event.timeMs) +
                    shiftTicks * tickDurationMs;
            if (time < modifier.window.minimumTimeMs) {
                time = modifier.window.minimumTimeMs;
            }
            if (time > modifier.window.maximumTimeMs) {
                time = modifier.window.maximumTimeMs;
            }
            event.timeMs = static_cast<std::int32_t>(time);
            if (IsSteerAction(event.action)) {
                if ((modifier.optionFlags & 1u) != 0u) {
                    event.value = random.UniformS32(
                            modifier.secondaryAnalogMinimum,
                            modifier.secondaryAnalogMaximum);
                } else {
                    const std::int32_t delta = random.UniformS32(
                            modifier.analogMinimum,
                            modifier.analogMaximum);
                    event.value = SaturateAnalog(
                            static_cast<std::int64_t>(event.value) +
                            delta);
                }
            } else if (IsSwitch(event)) {
                event.value = event.value != 0 ? 0 : 1;
            }
            if (!candidate->SetAt(ordinal, event)) {
                return false;
            }
        }
        candidate->Canonicalize();
        break;
    }
    case CudaSearchModifierKind::SmoothSteering:
        for (std::uint32_t deformation = 0u;
             deformation < modifier.minimumCount; ++deformation) {
            const std::int64_t minimumTick =
                    modifier.window.minimumTimeMs / tickDurationMs;
            const std::int64_t maximumTick =
                    modifier.window.maximumTimeMs / tickDurationMs;
            const std::int64_t center =
                    random.UniformS64(minimumTick, maximumTick) *
                    tickDurationMs;
            const std::int32_t amplitude = random.UniformS32(
                    modifier.analogMinimum, modifier.analogMaximum);
            std::int64_t start =
                    center - modifier.timeParameterMs;
            if (start < modifier.window.minimumTimeMs) {
                start = modifier.window.minimumTimeMs;
            }
            std::int64_t end = center + modifier.timeParameterMs;
            if (end > modifier.window.maximumTimeMs) {
                end = modifier.window.maximumTimeMs;
            }
            start = start <= 0
                    ? 0
                    : (start / tickDurationMs) * tickDurationMs;
            if (!candidate->ApplySmoothSteeringRun(
                        start, end, tickDurationMs, center, amplitude,
                        smoothWeights, modifier.weightOffset,
                        initialControls.steerValue)) {
                return false;
            }
        }
        break;
    case CudaSearchModifierKind::InputInsertion: {
        const std::uint32_t snapshotCount =
                candidate->BeginInsertionBatch();
        auto *operationTimes =
                reinterpret_cast<std::int32_t *>(eligible);
        std::uint32_t operationCount = 0u;
        const auto randomTime = [&]() {
            return random.UniformS64(
                           modifier.window.minimumTimeMs /
                                   tickDurationMs,
                           modifier.window.maximumTimeMs /
                                   tickDurationMs) *
                    tickDurationMs;
        };
        const auto randomHold = [&](std::int64_t maximum) {
            return maximum <= 0
                    ? INT64_C(0)
                    : random.UniformS64(
                                      0, maximum / tickDurationMs) *
                              tickDurationMs;
        };
        std::uint32_t steeringOperationCount = 0u;
        if (modifier.steering.enabled != 0u) {
            steeringOperationCount = random.UniformU32(
                    modifier.steering.minimumCount,
                    modifier.steering.maximumCount);
            for (std::uint32_t index = 0u;
                 index < steeringOperationCount; ++index) {
                const std::int64_t start = randomTime();
                std::int64_t end =
                        start + randomHold(
                                        modifier.steering.maximumHoldMs);
                if (end > modifier.window.maximumTimeMs) {
                    end = modifier.window.maximumTimeMs;
                }
                const std::int32_t previous =
                        candidate->InsertionBatchChannelStateAt(
                                snapshotCount, operationTimes,
                                0u, index, 4u, 2u, start,
                                initialControls.steerValue);
                const std::int32_t value =
                        (modifier.optionFlags & 1u) != 0u
                        ? SaturateAnalog(
                                  static_cast<std::int64_t>(previous) +
                                  random.UniformS32(
                                          modifier.secondaryAnalogMinimum,
                                          modifier.secondaryAnalogMaximum))
                        : random.UniformS32(
                                  modifier.analogMinimum,
                                  modifier.analogMaximum);
                const CudaSearchInputEvent startEvent =
                        AnalogEvent(start, 4u, value);
                const CudaSearchInputEvent endEvent =
                        AnalogEvent(
                                end, 4u,
                                candidate->SnapshotChannelStateAt(
                                        snapshotCount, 4u, 2u, end,
                                        initialControls.steerValue));
                if (!candidate->AppendInsertionBatchOperation(
                            operationTimes, operationCount++,
                            startEvent, endEvent, end > start)) {
                    return false;
                }
            }
        }
        std::uint32_t accelerateOperationCount = 0u;
        std::uint32_t brakeOperationCount = 0u;
        const auto insertSwitch =
                [&](const CudaSearchChannel &channel,
                    std::uint32_t action,
                    std::uint32_t firstOperation,
                    std::uint32_t *channelOperationCount) {
                    if (channel.enabled == 0u) {
                        return true;
                    }
                    *channelOperationCount = random.UniformU32(
                            channel.minimumCount,
                            channel.maximumCount);
                    for (std::uint32_t index = 0u;
                         index < *channelOperationCount; ++index) {
                        const std::int64_t start = randomTime();
                        std::int64_t end =
                                start + randomHold(
                                                channel.maximumHoldMs);
                        if (end > modifier.window.maximumTimeMs) {
                            end = modifier.window.maximumTimeMs;
                        }
                        const std::int32_t initialState =
                                action == 1u
                                ? initialControls.accelerate
                                : initialControls.brake;
                        const bool previous =
                                candidate->InsertionBatchChannelStateAt(
                                        snapshotCount, operationTimes,
                                        firstOperation, index,
                                        action, 1u, start,
                                        initialState) != 0;
                        const CudaSearchInputEvent startEvent =
                                SwitchEvent(start, action, !previous);
                        const CudaSearchInputEvent endEvent =
                                SwitchEvent(
                                        end, action,
                                        candidate->SnapshotChannelStateAt(
                                                snapshotCount,
                                                action, 1u, end,
                                                initialState) != 0);
                        if (!candidate->AppendInsertionBatchOperation(
                                    operationTimes, operationCount++,
                                    startEvent, endEvent,
                                    end > start)) {
                            return false;
                        }
                    }
                    return true;
                };
        if (!insertSwitch(
                    modifier.accelerate, 1u,
                    steeringOperationCount,
                    &accelerateOperationCount) ||
            !insertSwitch(
                    modifier.brake, 3u,
                    steeringOperationCount +
                            accelerateOperationCount,
                    &brakeOperationCount) ||
            !candidate->FinishInsertionBatch(
                    snapshotCount, operationTimes,
                    steeringOperationCount,
                    accelerateOperationCount,
                    brakeOperationCount)) {
            return false;
        }
        break;
    }
    case CudaSearchModifierKind::InputDeletion: {
        bool sharedIdentityReferences = identityReferences;
        const auto deleteChannel =
                [&](const CudaSearchChannel &channel,
                    std::uint32_t group) {
                    if (channel.enabled == 0u) {
                        return;
                    }
                    const std::uint32_t requested =
                            random.UniformU32(0u, channel.maximumCount);
                    if (requested == 0u) {
                        return;
                    }
                    const std::uint32_t initialEligibleCount =
                            CollectSparseEligible(
                                    *candidate, eligible,
                                    sharedEligible, sharedEligibleCount,
                                    sharedIdentityReferences,
                                    [&](const CudaSearchInputEvent &event) {
                                        return SparseDeletionEligible(
                                                event, modifier, group);
                                    });
                    std::uint32_t eligibleCount =
                            initialEligibleCount;
                    for (std::uint32_t removal = 0u;
                         removal < requested; ++removal) {
                        if (eligibleCount == 0u) {
                            break;
                        }
                        modifier_ops::SelectDeletionRank(
                                eligible, &eligibleCount,
                                random.UniformU32(
                                        0u, eligibleCount - 1u));
                    }
                    if (eligibleCount != initialEligibleCount) {
                        EraseSparseSelectedTail(
                                candidate, eligible, eligibleCount,
                                initialEligibleCount);
                        sharedIdentityReferences = false;
                    }
                };
        deleteChannel(modifier.steering, 0u);
        deleteChannel(modifier.accelerate, 1u);
        deleteChannel(modifier.brake, 2u);
        break;
    }
    }
    return true;
}

__device__ void ApplyControlPrefix(CudaCandidatePhysicsState &state,
                                   const CudaControlTick &tick) {
    state.world.schemePeriodMs = tick.periodMs;
    state.world.tickTimeMs = tick.timeMs;
}

__device__ bool ValidPackedInputs(
        const void *sceneData,
        const void *configurationData) {
    if (sceneData == nullptr || configurationData == nullptr) {
        return false;
    }
    const auto *scene =
            static_cast<const CudaPackedSceneHeader *>(sceneData);
    const auto *configuration =
            static_cast<const CudaPackedStaticConfigurationHeader *>(
                    configurationData);
    return ValidCudaPackedSceneHeader(*scene) &&
            configuration->magic ==
                    CudaPackedStaticConfigurationHeader::Magic &&
            configuration->schemaVersion ==
                    CudaPackedStaticConfigurationHeader::SchemaVersion;
}

__device__ bool ContainsVolume(
        const CudaSearchEvaluatorConfiguration &evaluator,
        const GmVec3 &position) {
    return static_cast<double>(position.x) >= evaluator.values[0] &&
            static_cast<double>(position.y) >= evaluator.values[1] &&
            static_cast<double>(position.z) >= evaluator.values[2] &&
            static_cast<double>(position.x) <= evaluator.values[3] &&
            static_cast<double>(position.y) <= evaluator.values[4] &&
            static_cast<double>(position.z) <= evaluator.values[5];
}

__device__ bool SegmentEntry(
        const CudaSearchEvaluatorConfiguration &evaluator,
        const GmVec3 &from,
        const GmVec3 &to,
        double *fraction) {
    const double fromValues[3]{from.x, from.y, from.z};
    const double toValues[3]{to.x, to.y, to.z};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double sweptMinimum =
                fromValues[axis] < toValues[axis]
                ? fromValues[axis] : toValues[axis];
        const double sweptMaximum =
                fromValues[axis] > toValues[axis]
                ? fromValues[axis] : toValues[axis];
        if (sweptMaximum < evaluator.values[axis] ||
            sweptMinimum > evaluator.values[axis + 3u]) {
            return false;
        }
    }

    double enter = 0.0;
    double leave = 1.0;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double delta = toValues[axis] - fromValues[axis];
        if (fabs(delta) <= 1e-12) {
            if (fromValues[axis] < evaluator.values[axis] ||
                fromValues[axis] > evaluator.values[axis + 3u]) {
                return false;
            }
            continue;
        }
        double near =
                (evaluator.values[axis] - fromValues[axis]) / delta;
        double far =
                (evaluator.values[axis + 3u] - fromValues[axis]) / delta;
        if (near > far) {
            const double swap = near;
            near = far;
            far = swap;
        }
        enter = enter > near ? enter : near;
        leave = leave < far ? leave : far;
        if (enter > leave) {
            return false;
        }
    }
    if (enter < 0.0 || enter > 1.0) {
        return false;
    }
    *fraction = enter;
    return true;
}

struct DeviceConditionValue {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool vector = false;
};

__device__ double ConditionLength(const GmVec3 &value) {
    return sqrt(static_cast<double>(value.x) * value.x +
                static_cast<double>(value.y) * value.y +
                static_cast<double>(value.z) * value.z);
}

__device__ GmVec3 ConditionLocalSpeed(
        const CHmsDyna::CHmsStateDyna &body) {
    const auto dot = [](const GmVec3 &a, const GmVec3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    return {
            dot(body.linearSpeed, body.rotation.basisX),
            dot(body.linearSpeed, body.rotation.basisY),
            dot(body.linearSpeed, body.rotation.basisZ)};
}

__device__ DeviceConditionValue ConditionAngles(
        const GmQuat &q) {
    const double sinrCosp = 2.0 *
            (static_cast<double>(q.w) * q.x +
             static_cast<double>(q.y) * q.z);
    const double cosrCosp = 1.0 - 2.0 *
            (static_cast<double>(q.x) * q.x +
             static_cast<double>(q.y) * q.y);
    const double sinp = 2.0 *
            (static_cast<double>(q.w) * q.y -
             static_cast<double>(q.z) * q.x);
    const double sinyCosp = 2.0 *
            (static_cast<double>(q.w) * q.z +
             static_cast<double>(q.x) * q.y);
    const double cosyCosp = 1.0 - 2.0 *
            (static_cast<double>(q.y) * q.y +
             static_cast<double>(q.z) * q.z);
    const double pitch = fabs(sinp) >= 1.0
            ? copysign(1.57079632679489661923, sinp)
            : asin(sinp);
    return {atan2(sinyCosp, cosyCosp), pitch,
            atan2(sinrCosp, cosrCosp), true};
}

__device__ DeviceConditionValue ConditionSource(
        CudaSearchConditionValue source,
        const CudaCandidatePhysicsState &state,
        double iterationCount,
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds,
        double currentTimeSeconds) {
    const CHmsDyna::CHmsStateDyna &current = state.body.current;
    const CHmsDyna::CHmsStateDyna &previous = state.body.temporary;
    const GmVec3 currentLocal =
            state.vehicle.frameHistory.physicsCurrent.localLinearSpeed;
    const GmVec3 previousLocal = ConditionLocalSpeed(previous);
    const auto vector = [](const GmVec3 &value) {
        return DeviceConditionValue{value.x, value.y, value.z, true};
    };
    switch (source) {
    case CudaSearchConditionValue::Position: return vector(current.position);
    case CudaSearchConditionValue::PreviousPosition: return vector(previous.position);
    case CudaSearchConditionValue::Velocity: return vector(current.linearSpeed);
    case CudaSearchConditionValue::PreviousVelocity: return vector(previous.linearSpeed);
    case CudaSearchConditionValue::LocalVelocity: return vector(currentLocal);
    case CudaSearchConditionValue::PreviousLocalVelocity: return vector(previousLocal);
    case CudaSearchConditionValue::AngularVelocity: return vector(current.angularSpeed);
    case CudaSearchConditionValue::PreviousAngularVelocity: return vector(previous.angularSpeed);
    case CudaSearchConditionValue::Yaw: return {ConditionAngles(current.rotationQuat).x};
    case CudaSearchConditionValue::Pitch: return {ConditionAngles(current.rotationQuat).y};
    case CudaSearchConditionValue::Roll: return {ConditionAngles(current.rotationQuat).z};
    case CudaSearchConditionValue::PreviousYaw: return {ConditionAngles(previous.rotationQuat).x};
    case CudaSearchConditionValue::PreviousPitch: return {ConditionAngles(previous.rotationQuat).y};
    case CudaSearchConditionValue::PreviousRoll: return {ConditionAngles(previous.rotationQuat).z};
    case CudaSearchConditionValue::Speed: return {ConditionLength(current.linearSpeed)};
    case CudaSearchConditionValue::PreviousSpeed: return {ConditionLength(previous.linearSpeed)};
    case CudaSearchConditionValue::LocalSpeed: return {ConditionLength(currentLocal)};
    case CudaSearchConditionValue::PreviousLocalSpeed: return {ConditionLength(previousLocal)};
    case CudaSearchConditionValue::FreeWheeling: return {state.vehicle.controls.forcedLowSpeedFriction ? 1.0 : 0.0};
    case CudaSearchConditionValue::LateralContact: return {state.vehicle.contacts.lateralSlowDownContactActive ? 1.0 : 0.0};
    case CudaSearchConditionValue::Sliding: {
        bool sliding = false;
        for (std::uint32_t i = 0u; i < cuda::facts::WheelCount(state.vehicle); ++i) {
            sliding = sliding || (state.vehicle.wheels.values[i].realTime.contactPresent &&
                                  state.vehicle.wheels.values[i].realTime.slipping);
        }
        return {sliding ? 1.0 : 0.0};
    }
    case CudaSearchConditionValue::Gear:
        return {state.vehicle.engine.useLowSpeedGateB ? -1.0 :
                static_cast<double>(state.vehicle.engine.gearIndex)};
    case CudaSearchConditionValue::Rpm: return {state.vehicle.engine.engineInputMemory};
    case CudaSearchConditionValue::TurningRate: return {state.vehicle.radiusSteering.steerAngle};
    case CudaSearchConditionValue::TurboType: return {static_cast<double>(state.vehicle.turbo.type)};
    case CudaSearchConditionValue::TurboBoostFactor:
        return {state.vehicle.turbo.type == CSceneVehicleCar::ETurboType_Roulette
                ? static_cast<double>(state.vehicle.turbo.type2Phase) + 1.0
                : state.vehicle.turbo.impulseScale};
    case CudaSearchConditionValue::Iterations: return {iterationCount};
    case CudaSearchConditionValue::LastImprovementTime: return {lastImprovementTimeSeconds};
    case CudaSearchConditionValue::LastRestartTime: return {lastRestartTimeSeconds};
    case CudaSearchConditionValue::CurrentTime: return {currentTimeSeconds};
    default: break;
    }
    const std::uint32_t raw = static_cast<std::uint32_t>(source);
    const std::uint32_t ground0 = static_cast<std::uint32_t>(CudaSearchConditionValue::WheelGroundContact0);
    const std::uint32_t sliding0 = static_cast<std::uint32_t>(CudaSearchConditionValue::WheelSliding0);
    const std::uint32_t surface0 = static_cast<std::uint32_t>(CudaSearchConditionValue::WheelSurface0);
    std::uint32_t wheel = 0u;
    if (raw >= ground0 && raw < ground0 + 4u) {
        wheel = raw - ground0;
        return {wheel < cuda::facts::WheelCount(state.vehicle) &&
                state.vehicle.wheels.values[wheel].realTime.contactPresent ? 1.0 : 0.0};
    }
    if (raw >= sliding0 && raw < sliding0 + 4u) {
        wheel = raw - sliding0;
        const auto &value = state.vehicle.wheels.values[wheel].realTime;
        return {wheel < cuda::facts::WheelCount(state.vehicle) &&
                value.contactPresent && value.slipping ? 1.0 : 0.0};
    }
    wheel = raw - surface0;
    if (raw >= surface0 && raw < surface0 + 4u &&
        wheel < cuda::facts::WheelCount(state.vehicle)) {
        const auto &value = state.vehicle.wheels.values[wheel].realTime;
        return {value.contactPresent
                ? static_cast<double>(value.contactMaterial)
                : 65535.0};
    }
    return {};
}

__device__ __noinline__ bool EvaluateCondition(
        const CudaSearchConditionInstruction *instructions,
        std::uint32_t instructionCount,
        const CudaCandidatePhysicsState &state,
        double iterationCount,
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds,
        double currentTimeSeconds) {
    if (instructionCount == 0u) return true;
    DeviceConditionValue stack[32];
    std::uint32_t size = 0u;
    for (std::uint32_t index = 0u; index < instructionCount; ++index) {
        const CudaSearchConditionInstruction instruction = instructions[index];
        if (instruction.opcode == CudaSearchConditionOpcode::Constant) {
            if (size >= 32u) return false;
            stack[size++] = {instruction.x};
            continue;
        }
        if (instruction.opcode == CudaSearchConditionOpcode::ConstantVector) {
            if (size >= 32u) return false;
            stack[size++] = {instruction.x, instruction.y, instruction.z, true};
            continue;
        }
        if (instruction.opcode == CudaSearchConditionOpcode::Scalar ||
            instruction.opcode == CudaSearchConditionOpcode::Vector) {
            if (size >= 32u) return false;
            DeviceConditionValue value = ConditionSource(
                    instruction.value, state, iterationCount,
                    lastImprovementTimeSeconds,
                    lastRestartTimeSeconds, currentTimeSeconds);
            if (instruction.opcode == CudaSearchConditionOpcode::Scalar &&
                value.vector) {
                const int component = static_cast<int>(instruction.x);
                value = {component == 1 ? value.x
                         : component == 2 ? value.y
                         : component == 3 ? value.z : 0.0};
            }
            if ((instruction.opcode == CudaSearchConditionOpcode::Scalar &&
                 value.vector) ||
                (instruction.opcode == CudaSearchConditionOpcode::Vector &&
                 !value.vector)) return false;
            stack[size++] = value;
            continue;
        }
        if (instruction.opcode == CudaSearchConditionOpcode::KilometersPerHour ||
            instruction.opcode == CudaSearchConditionOpcode::Degrees) {
            if (size == 0u || stack[size - 1u].vector) return false;
            stack[size - 1u].x *= instruction.opcode ==
                    CudaSearchConditionOpcode::KilometersPerHour
                    ? 3.6 : 57.2957795130823208768;
            continue;
        }
        if (size < 2u) return false;
        const DeviceConditionValue right = stack[--size];
        DeviceConditionValue &left = stack[size - 1u];
        switch (instruction.opcode) {
        case CudaSearchConditionOpcode::Distance:
            if (!left.vector || !right.vector) return false;
            left = {sqrt((left.x-right.x)*(left.x-right.x) +
                         (left.y-right.y)*(left.y-right.y) +
                         (left.z-right.z)*(left.z-right.z))};
            break;
        case CudaSearchConditionOpcode::Add: left.x += right.x; break;
        case CudaSearchConditionOpcode::Subtract: left.x -= right.x; break;
        case CudaSearchConditionOpcode::Multiply: left.x *= right.x; break;
        case CudaSearchConditionOpcode::Divide: left.x = right.x == 0.0 ? 0.0 : left.x / right.x; break;
        case CudaSearchConditionOpcode::Greater: left = {left.x > right.x ? 1.0 : 0.0}; break;
        case CudaSearchConditionOpcode::Less: left = {left.x < right.x ? 1.0 : 0.0}; break;
        case CudaSearchConditionOpcode::GreaterOrEqual: left = {left.x >= right.x ? 1.0 : 0.0}; break;
        case CudaSearchConditionOpcode::LessOrEqual: left = {left.x <= right.x ? 1.0 : 0.0}; break;
        case CudaSearchConditionOpcode::Equal: left = {left.x == right.x ? 1.0 : 0.0}; break;
        case CudaSearchConditionOpcode::LogicalAnd: left = {left.x != 0.0 && right.x != 0.0 ? 1.0 : 0.0}; break;
        default: return false;
        }
    }
    return size == 1u && !stack[0].vector && stack[0].x != 0.0;
}

__device__ DeviceSample EvaluateState(
        const CudaSearchEvaluatorConfiguration &evaluator,
        const CudaCandidatePhysicsState &state,
        const GmVec3 &previousPosition,
        double previousTimeMs,
        double currentTimeMs,
        std::uint32_t stuntsScore,
        bool *reported) {
    DeviceSample result;
    result.timeMs = currentTimeMs;
    const GmVec3 &position = state.body.current.position;
    switch (evaluator.kind) {
    case CudaSearchEvaluatorKind::Velocity: {
        const GmVec3 &velocity = state.body.current.linearSpeed;
        const double x = velocity.x;
        const double y = velocity.y;
        const double z = velocity.z;
        const double speed = sqrt((x * x + y * y) + z * z);
        double alignment = 1.0;
        if ((evaluator.optionFlags & 3u) != 0u) {
            alignment = speed <= 1e-12
                    ? 0.0
                    : (x * evaluator.values[0] +
                       y * evaluator.values[1] +
                       z * evaluator.values[2]) /
                              speed;
            if (alignment < evaluator.values[3]) {
                return result;
            }
        }
        result.score = (evaluator.optionFlags & 1u) != 0u
                ? x * evaluator.values[0] +
                          y * evaluator.values[1] +
                          z * evaluator.values[2]
                : speed;
        result.detail0 = speed;
        result.detail1 = alignment;
        result.valid = true;
        break;
    }
    case CudaSearchEvaluatorKind::Point: {
        const double x = static_cast<double>(position.x) -
                evaluator.values[0];
        const double y = static_cast<double>(position.y) -
                evaluator.values[1];
        const double z = static_cast<double>(position.z) -
                evaluator.values[2];
        result.score = sqrt((x * x + y * y) + z * z);
        result.valid = true;
        break;
    }
    case CudaSearchEvaluatorKind::Pose: {
        const double x = static_cast<double>(position.x) -
                evaluator.values[0];
        const double y = static_cast<double>(position.y) -
                evaluator.values[1];
        const double z = static_cast<double>(position.z) -
                evaluator.values[2];
        const double positionError =
                sqrt((x * x + y * y) + z * z);
        const GmQuat &rotation = state.body.current.rotationQuat;
        double dot = fabs(
                evaluator.values[3] * rotation.x +
                evaluator.values[4] * rotation.y +
                evaluator.values[5] * rotation.z +
                evaluator.values[6] * rotation.w);
        dot = dot < 0.0 ? 0.0 : (dot > 1.0 ? 1.0 : dot);
        const double rotationError = 2.0 * acos(dot);
        result.score =
                (1.0 - evaluator.values[7]) * positionError +
                evaluator.values[7] * rotationError;
        result.detail0 = positionError;
        result.detail1 = rotationError;
        result.valid = true;
        break;
    }
    case CudaSearchEvaluatorKind::VolumeEntry:
        if (*reported || ContainsVolume(evaluator, previousPosition)) {
            return result;
        } else {
            double fraction = 0.0;
            if (!SegmentEntry(
                        evaluator, previousPosition, position,
                        &fraction)) {
                return result;
            }
            *reported = true;
            result.timeMs = previousTimeMs +
                    fraction * (currentTimeMs - previousTimeMs);
            result.score = result.timeMs;
            result.valid = true;
        }
        break;
    case CudaSearchEvaluatorKind::StuntPoints:
        result.score = static_cast<double>(stuntsScore);
        result.valid = true;
        break;
    case CudaSearchEvaluatorKind::FinishTime:
        if (*reported || !state.race.progress.raceCompleted) {
            return result;
        }
        *reported = true;
        result.timeMs =
                state.race.progress.lastPrepareTimeMs;
        result.score = result.timeMs;
        result.valid = true;
        break;
    }
    return result;
}

__host__ __device__ bool MaximizesScore(
        CudaSearchEvaluatorKind kind) {
    return kind == CudaSearchEvaluatorKind::Velocity ||
            kind == CudaSearchEvaluatorKind::StuntPoints;
}

__global__ void SeedCandidateBestSamplesKernel(
        DeviceSample *candidateBestSamples,
        const DeviceSample *incumbent) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        DeviceSample seed = *incumbent;
        seed.logicalOrder = 0u;
        seed.candidateSlot = InvalidCandidateSlot;
        candidateBestSamples[0] = seed;
    }
}

__global__ void InitializeSearchBatchSummaryKernel(
        DeviceBatchSummary *summary,
        double *summaryBlockClosestTargetDistanceSquared,
        std::uint32_t summaryBlockCount) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *summary = DeviceBatchSummary{};
    }
    for (std::uint32_t blockIndex = threadIdx.x;
         blockIndex < summaryBlockCount;
         blockIndex += blockDim.x) {
        summaryBlockClosestTargetDistanceSquared[blockIndex] =
                cuda_search_progress_detail::
                        InvalidClosestTargetDistanceSquared;
    }
}

__global__ void GenerateSearchCandidatesKernel(
        const CudaSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        std::uint32_t immutableTailInputCount,
        const CudaSearchModifierConfiguration *modifiers,
        std::uint32_t modifierCount,
        const double *smoothWeights,
        const DeviceControlState *mutableBoundaryControls,
        std::uint32_t tickDurationMs,
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool baseline,
        bool legacyMutationPipeline,
        bool baselineInputsCanonical,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        bool directDeletionPipeline,
        bool directExistingEventPipeline,
        std::uint32_t eventCapacity,
        const std::uint32_t *compactInputIndices,
        std::uint32_t compactInputCount,
        std::int32_t *candidateInputValues,
        DeviceSample *candidateBestSamples,
        std::uint32_t *randomStateWords,
        CudaSearchInputEvent *candidateEvents,
        CudaSearchInputEvent *temporaryEvents,
        CudaSearchInputEvent *passBaselineEvents,
        std::uint32_t *eligibleIndices,
        const std::uint32_t *sharedEligibleIndices,
        std::uint32_t sharedEligibleCount,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        std::uint32_t *eventCounts,
        std::uint32_t *mutationCounts,
        DeviceCandidateStatus *statuses,
        bool *activeCandidates,
        const std::uint32_t *cancellation) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount) {
        return;
    }
    const std::uint64_t candidateId = firstCandidateId + slot;
    CudaSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    CudaSearchInputEvent *temporary =
            temporaryEvents == nullptr
            ? nullptr
            : temporaryEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    CudaSearchInputEvent *passBaseline =
            passBaselineEvents == nullptr
            ? nullptr
            : passBaselineEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    std::uint32_t *eligible =
            eligibleIndices == nullptr
            ? nullptr
            : eligibleIndices +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    DeviceMt19937 random(randomStateWords, slot, candidateCount);
    std::uint32_t eventCount = baselineInputCount;
    sparse_events::Candidate sparseCandidate(
            baselineInputs, baselineInputCount,
            sparseCandidateEvents, slot);
    const bool fusedSparseRandomInitialization =
            sparseMutationPipeline && !baseline && modifierCount != 0u &&
            modifiers[0].kind ==
                    CudaSearchModifierKind::RandomSteering;
    if (sparseMutationPipeline &&
        !fusedSparseRandomInitialization) {
        sparseCandidate.Initialize();
    }
    if (compactRandomSteeringPipeline) {
        for (std::uint32_t index = 0u;
             index < compactInputCount; ++index) {
            candidateInputValues[
                    static_cast<std::uint64_t>(index) *
                            candidateCount +
                    slot] =
                    baselineInputs[compactInputIndices[index]].value;
        }
    } else if (events != nullptr) {
        for (std::uint32_t index = 0u;
             index < baselineInputCount; ++index) {
            events[index] = baselineInputs[index];
        }
    }
    if (compactEditPipeline &&
        (directDeletionPipeline ||
         directExistingEventPipeline)) {
        candidateEdits.counts[slot] = 0u;
        candidateEdits.erasedCounts[slot] = 0u;
    }
    statuses[slot] = DeviceCandidateStatus::Success;
    candidateBestSamples[slot + 1u] = {};
    if (*reinterpret_cast<volatile const std::uint32_t *>(
                cancellation) != 0u) {
        statuses[slot] = DeviceCandidateStatus::Cancelled;
        activeCandidates[slot] = false;
        eventCounts[slot] = eventCount;
        mutationCounts[slot] = 0u;
        return;
    }
    if ((compactRandomSteeringPipeline || compactEditPipeline ||
         sparseMutationPipeline) &&
        baseline) {
        eventCounts[slot] = baselineInputCount;
        mutationCounts[slot] = 0u;
        activeCandidates[slot] = true;
        return;
    }
    if (!baseline) {
        if (compactRandomSteeringPipeline) {
            for (std::uint32_t pass = 0u;
                 pass < modifierCount; ++pass) {
                const CudaSearchModifierConfiguration modifier =
                        modifiers[pass];
                random.Seed(
                        modifier.window.seed, candidateId, pass);
                for (std::uint32_t compactIndex = 0u;
                     compactIndex < compactInputCount;
                     ++compactIndex) {
                    const CudaSearchInputEvent &input =
                            baselineInputs[
                                    compactInputIndices[compactIndex]];
                    if (input.timeMs <
                                    modifier.window.minimumTimeMs ||
                        input.timeMs >
                                    modifier.window.maximumTimeMs) {
                        continue;
                    }
                    std::int32_t value =
                            random.UniformS32(
                                    -65536, 65536);
                    std::int32_t &compactValue =
                            candidateInputValues[
                                    static_cast<std::uint64_t>(
                                            compactIndex) *
                                            candidateCount +
                                    slot];
                    if (value == compactValue) {
                        value = value == 65536 ? -65536 : 65536;
                    }
                    compactValue = value;
                }
            }
            std::uint32_t mutationCount = 0u;
            for (std::uint32_t index = 0u;
                 index < compactInputCount; ++index) {
                if (candidateInputValues[
                            static_cast<std::uint64_t>(index) *
                                    candidateCount +
                            slot] !=
                    baselineInputs[compactInputIndices[index]].value) {
                    ++mutationCount;
                }
            }
            eventCounts[slot] = baselineInputCount;
            mutationCounts[slot] = mutationCount;
            activeCandidates[slot] = mutationCount != 0u;
            return;
        }
        if (sparseMutationPipeline) {
            std::uint32_t firstPass = 0u;
            bool identityReferences = true;
            if (fusedSparseRandomInitialization) {
                const CudaSearchModifierConfiguration modifier =
                        modifiers[0];
                random.Seed(
                        modifier.window.seed, candidateId, 0u);
                sparseCandidate.BeginInitialize();
                for (std::uint32_t index = 0u;
                     index < baselineInputCount; ++index) {
                    CudaSearchInputEvent event =
                            baselineInputs[index];
                    if (event.timeMs <
                                    modifier.window.minimumTimeMs ||
                        event.timeMs >
                                    modifier.window.maximumTimeMs ||
                        event.action != 4u || !IsAnalog(event)) {
                        sparseCandidate.InitializeBaselineAt(index);
                        continue;
                    }
                    std::int32_t value =
                            random.UniformS32(-65536, 65536);
                    if (value == event.value) {
                        value = value == 65536 ? -65536 : 65536;
                    }
                    event.value = value;
                    if (!sparseCandidate.InitializeEditAt(
                                index, event)) {
                        statuses[slot] =
                                DeviceCandidateStatus::CapacityExceeded;
                        activeCandidates[slot] = false;
                        eventCounts[slot] = baselineInputCount;
                        mutationCounts[slot] = 0u;
                        return;
                    }
                }
                firstPass = 1u;
            }
            for (std::uint32_t pass = firstPass;
                 pass < modifierCount; ++pass) {
                if (!ApplySparseModifier(
                            modifiers[pass], pass, candidateId,
                            random, tickDurationMs,
                            *mutableBoundaryControls,
                            &sparseCandidate, eligible,
                            sharedEligibleIndices,
                            sharedEligibleCount,
                            identityReferences, smoothWeights)) {
                    statuses[slot] =
                            DeviceCandidateStatus::CapacityExceeded;
                    activeCandidates[slot] = false;
                    eventCounts[slot] = sparseCandidate.Count();
                    mutationCounts[slot] = 0u;
                    return;
                }
                const CudaSearchModifierConfiguration modifier =
                        modifiers[pass];
                if ((modifier.kind ==
                             CudaSearchModifierKind::ExistingEvent &&
                     modifier.timeParameterMs != 0) ||
                    modifier.kind ==
                            CudaSearchModifierKind::SmoothSteering ||
                    modifier.kind ==
                            CudaSearchModifierKind::InputInsertion ||
                    modifier.kind ==
                            CudaSearchModifierKind::InputDeletion) {
                    identityReferences = false;
                }
            }
            eventCount = sparseCandidate.Count();
            std::uint32_t mutationCount =
                    SparseEffectiveChangeCount(
                            baselineInputs, baselineInputCount,
                            sparseCandidate);
            if (eventCount != baselineInputCount) {
                mutationCount += immutableTailInputCount;
            }
            eventCounts[slot] = eventCount;
            mutationCounts[slot] = mutationCount;
            activeCandidates[slot] = mutationCount != 0u;
            return;
        }
        if (directExistingEventPipeline) {
            const CudaSearchModifierConfiguration modifier =
                    modifiers[0];
            random.Seed(
                    modifier.window.seed, candidateId, 0u);
            const std::uint32_t eligibleCount =
                    CollectEligible(
                            baselineInputs, baselineInputCount,
                            eligible, modifier, true);
            if (eligibleCount == 0u) {
                eventCounts[slot] = baselineInputCount;
                mutationCounts[slot] = 0u;
                activeCandidates[slot] = false;
                return;
            }
            ShuffleIndices(eligible, eligibleCount, random);
            const std::uint32_t requested = random.UniformU32(
                    modifier.minimumCount, modifier.maximumCount);
            const std::uint32_t count =
                    requested < eligibleCount
                    ? requested : eligibleCount;
            cuda::candidate_events::EditWriter writer(
                    candidateEdits, slot);
            std::uint32_t mutationCount = 0u;
            for (std::uint32_t index = 0u;
                 index < count; ++index) {
                const std::uint32_t source = eligible[index];
                CudaSearchInputEvent event =
                        baselineInputs[source];
                static_cast<void>(random.UniformS64(0, 0));
                if (IsSteerAction(event.action)) {
                    if ((modifier.optionFlags & 1u) != 0u) {
                        event.value = random.UniformS32(
                                modifier.secondaryAnalogMinimum,
                                modifier.secondaryAnalogMaximum);
                    } else {
                        const std::int32_t delta =
                                random.UniformS32(
                                        modifier.analogMinimum,
                                        modifier.analogMaximum);
                        event.value = SaturateAnalog(
                                static_cast<std::int64_t>(
                                        event.value) +
                                delta);
                    }
                } else if (IsSwitch(event)) {
                    event.value = event.value != 0 ? 0 : 1;
                }
                if (SameEvent(
                            event, baselineInputs[source])) {
                    continue;
                }
                if (!writer.Insert(source, event) ||
                    !writer.Erase(source)) {
                    statuses[slot] =
                            DeviceCandidateStatus::CapacityExceeded;
                    activeCandidates[slot] = false;
                    eventCounts[slot] = baselineInputCount;
                    mutationCounts[slot] = 0u;
                    return;
                }
                ++mutationCount;
            }
            cuda::candidate_events::SortOutputEdits(
                    candidateEdits, slot);
            cuda::candidate_events::SortErasedSources(
                    candidateEdits, slot);
            eventCounts[slot] = baselineInputCount;
            mutationCounts[slot] = mutationCount;
            activeCandidates[slot] = mutationCount != 0u;
            return;
        }
        if (directDeletionPipeline) {
            const CudaSearchModifierConfiguration modifier =
                    modifiers[0];
            random.Seed(
                    modifier.window.seed, candidateId, 0u);
            cuda::candidate_events::EditWriter writer(
                    candidateEdits, slot);
            const auto deleteChannel =
                    [&](const CudaSearchChannel &channel,
                        std::uint32_t kind) {
                        if (channel.enabled == 0u) {
                            return true;
                        }
                        const std::uint32_t requested =
                                random.UniformU32(
                                        0u, channel.maximumCount);
                        if (requested == 0u) {
                            return true;
                        }
                        const std::uint32_t initialEligibleCount =
                                modifier_ops::CollectDeletionEligible(
                                        baselineInputs,
                                        baselineInputCount,
                                        eligible,
                                        modifier.window.minimumTimeMs,
                                        modifier.window.maximumTimeMs,
                                        kind, true);
                        std::uint32_t eligibleCount =
                                initialEligibleCount;
                        for (std::uint32_t removal = 0u;
                             removal < requested; ++removal) {
                            if (eligibleCount == 0u) {
                                break;
                            }
                            modifier_ops::SelectDeletionRank(
                                    eligible, &eligibleCount,
                                    random.UniformU32(
                                            0u,
                                            eligibleCount - 1u));
                        }
                        for (std::uint32_t selected = eligibleCount;
                             selected < initialEligibleCount;
                             ++selected) {
                            if (!writer.Erase(eligible[selected])) {
                                return false;
                            }
                        }
                        return true;
                    };
            if (!deleteChannel(modifier.steering, 0u) ||
                !deleteChannel(modifier.accelerate, 1u) ||
                !deleteChannel(modifier.brake, 2u)) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                activeCandidates[slot] = false;
                eventCounts[slot] = baselineInputCount;
                mutationCounts[slot] = 0u;
                return;
            }
            cuda::candidate_events::SortErasedSources(
                    candidateEdits, slot);
            eventCount = baselineInputCount -
                    candidateEdits.erasedCounts[slot];
            cuda::candidate_events::CandidateCursor cursor({
                    {baselineInputs, baselineInputCount, 0},
                    candidateEdits,
                    slot,
                    eventCount});
            std::uint32_t mutationCount =
                    baselineInputCount - eventCount;
            for (std::uint32_t output = 0u;
                 output < eventCount; ++output) {
                CudaSearchInputEvent event{};
                if (!cursor.Next(&event) ||
                    !SameEvent(baselineInputs[output], event)) {
                    ++mutationCount;
                }
            }
            if (eventCount != baselineInputCount) {
                mutationCount += immutableTailInputCount;
            }
            eventCounts[slot] = eventCount;
            mutationCounts[slot] = mutationCount;
            activeCandidates[slot] = mutationCount != 0u;
            return;
        }
        bool normalized = baselineInputsCanonical;
        for (std::uint32_t pass = 0u; pass < modifierCount; ++pass) {
            if (!ApplyModifier(
                        modifiers[pass], pass, candidateId,
                        random,
                        tickDurationMs, 0,
                        *mutableBoundaryControls,
                        baselineInputs, baselineInputCount,
                        events, &eventCount, eventCapacity,
                        temporary,
                        passBaseline, eligible,
                        smoothWeights, legacyMutationPipeline,
                        &normalized)) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                activeCandidates[slot] = false;
                eventCounts[slot] = eventCount;
                mutationCounts[slot] = 0u;
                return;
            }
        }
        if (legacyMutationPipeline || !normalized) {
            for (std::uint32_t index = 0u;
                 index < baselineInputCount; ++index) {
                passBaseline[index] = baselineInputs[index];
            }
            eventCount = NormalizeEvents(
                    events, eventCount, temporary,
                    passBaseline, baselineInputCount,
                    0,
                    legacyMutationPipeline, eventCapacity);
            if (eventCount == UINT32_MAX) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                activeCandidates[slot] = false;
                eventCounts[slot] = eventCapacity;
                mutationCounts[slot] = 0u;
                return;
            }
        }
    }
    std::uint32_t mutationCount = baseline
            ? 0u
            : EffectiveChangeCount(
                      baselineInputs, baselineInputCount,
                      events, eventCount);
    if (!baseline && eventCount != baselineInputCount) {
        mutationCount += immutableTailInputCount;
    }
    eventCounts[slot] = eventCount;
    mutationCounts[slot] = mutationCount;
    const bool active = baseline || mutationCount != 0u;
    activeCandidates[slot] = active;
}

__global__ void EncodeSearchCandidateEditsKernel(
        const CudaSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *candidateEvents,
        std::uint32_t eventCapacity,
        const std::uint32_t *eventCounts,
        std::uint32_t *sourceStates,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        DeviceCandidateStatus *statuses,
        bool *activeCandidates,
        std::uint32_t candidateCount) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount) {
        return;
    }
    candidateEdits.counts[slot] = 0u;
    candidateEdits.erasedCounts[slot] = 0u;
    if (statuses[slot] != DeviceCandidateStatus::Success) {
        return;
    }
    const CudaSearchInputEvent *events =
            candidateEvents +
            static_cast<std::uint64_t>(slot) * eventCapacity;
    std::uint32_t *states =
            sourceStates +
            static_cast<std::uint64_t>(slot) * eventCapacity;
    if (!EncodeCandidateEdits(
                baselineInputs, baselineInputCount,
                events, eventCounts[slot], states,
                candidateEdits, slot)) {
        statuses[slot] = DeviceCandidateStatus::CapacityExceeded;
        activeCandidates[slot] = false;
    }
}

__device__ std::uint32_t ResolvedControlLocalityCode(
        const DeviceControlState &state) {
    const ReplayVehicleControlState controls = ControlsFromState(state);
    std::int32_t steeringFixed = static_cast<std::int32_t>(
            controls.steering * 65536.0f);
    if (steeringFixed < -65536) steeringFixed = -65536;
    if (steeringFixed > 65536) steeringFixed = 65536;
    const std::uint32_t quantizedSteering =
            static_cast<std::uint32_t>(
                    (static_cast<std::int64_t>(steeringFixed + 65536) *
                             63 +
                     65536) /
                    131072);
    return (controls.lowSpeedGateA != 0.0f ? 1u << 7u : 0u) |
            (controls.lowSpeedGateB != 0.0f ? 1u << 6u : 0u) |
            quantizedSteering;
}

__global__ void ComputeSearchCandidateSimulationKeysKernel(
        const CudaSearchInputEvent *__restrict__ baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *__restrict__ candidateEvents,
        std::uint32_t eventCapacity,
        const std::int32_t *__restrict__ candidateInputValues,
        const std::uint32_t *__restrict__ compactInputOffsets,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *__restrict__ eventCounts,
        const bool *__restrict__ activeCandidates,
        const DeviceControlState *__restrict__ mutableBoundaryControls,
        std::int64_t mutableFromTimeMs,
        std::int64_t branchTimeMs,
        std::uint32_t tickDurationMs,
        std::uint32_t timelineTickCount,
        bool sortCandidatesByLocality,
        std::uint64_t *__restrict__ simulationKeys,
        std::uint32_t *__restrict__ candidateSlots,
        std::uint32_t candidateCount) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount) {
        return;
    }
    candidateSlots[slot] = slot;
    if (!activeCandidates[slot]) {
        simulationKeys[slot] = UINT64_MAX;
        return;
    }
    const std::uint32_t eventCount = eventCounts[slot];
    const CudaSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    CandidateInputCursor cursor(
            baselineInputs, baselineInputCount, events,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount, slot, candidateCount);
    const std::uint32_t commonCount =
            eventCount < baselineInputCount
            ? eventCount : baselineInputCount;
    std::int64_t firstDifferentTimeMs = INT64_MAX;
    CudaSearchInputEvent firstUnmatchedCandidateEvent{};
    bool hasFirstUnmatchedCandidateEvent = false;
    std::uint32_t common = 0u;
    for (; common < commonCount; ++common) {
        CudaSearchInputEvent candidateEvent{};
        if (!cursor.Next(&candidateEvent)) {
            firstDifferentTimeMs = baselineInputs[common].timeMs;
            break;
        }
        const CudaSearchInputEvent baselineEvent =
                baselineInputs[common];
        if (!SameEvent(candidateEvent, baselineEvent)) {
            firstUnmatchedCandidateEvent = candidateEvent;
            hasFirstUnmatchedCandidateEvent = true;
            firstDifferentTimeMs =
                    candidateEvent.timeMs < baselineEvent.timeMs
                    ? candidateEvent.timeMs
                    : baselineEvent.timeMs;
            break;
        }
    }
    if (firstDifferentTimeMs == INT64_MAX &&
        common == commonCount && eventCount != baselineInputCount) {
        if (eventCount > baselineInputCount) {
            CudaSearchInputEvent candidateEvent{};
            if (cursor.Next(&candidateEvent)) {
                firstUnmatchedCandidateEvent = candidateEvent;
                hasFirstUnmatchedCandidateEvent = true;
                firstDifferentTimeMs = candidateEvent.timeMs;
            }
        } else {
            firstDifferentTimeMs = baselineInputs[commonCount].timeMs;
        }
    }
    std::uint32_t firstSimulationTick = 0u;
    if (firstDifferentTimeMs != INT64_MAX) {
        const std::int64_t firstDifferentPublicTimeMs =
                mutableFromTimeMs + firstDifferentTimeMs;
        const std::int64_t deltaMs =
                firstDifferentPublicTimeMs - branchTimeMs;
        if (deltaMs > 0) {
            firstSimulationTick = static_cast<std::uint32_t>(
                    (deltaMs - 1) / tickDurationMs);
            if (firstSimulationTick > timelineTickCount) {
                firstSimulationTick = timelineTickCount;
            }
        }
    }
    std::uint32_t localityCode = 0u;
    if (sortCandidatesByLocality &&
        firstSimulationTick < timelineTickCount) {
        CandidateLocalityCursor localityCursor(
                baselineInputs, common, &cursor,
                hasFirstUnmatchedCandidateEvent,
                firstUnmatchedCandidateEvent);
        CudaSearchInputEvent nextEvent{};
        bool hasNextEvent = localityCursor.Next(&nextEvent);
        DeviceControlState controlState = *mutableBoundaryControls;
        constexpr std::uint32_t SignatureTickOffsets[] = {
                0u, 1u, 4u, 16u};
#pragma unroll
        for (std::uint32_t signatureIndex = 0u;
             signatureIndex < 4u; ++signatureIndex) {
            const std::uint32_t remainingTicks =
                    timelineTickCount - 1u - firstSimulationTick;
            const std::uint32_t signatureTick =
                    SignatureTickOffsets[signatureIndex] > remainingTicks
                    ? timelineTickCount - 1u
                    : firstSimulationTick +
                              SignatureTickOffsets[signatureIndex];
            const std::int64_t publicTime =
                    branchTimeMs +
                    static_cast<std::int64_t>(signatureTick + 1u) *
                            tickDurationMs;
            const std::int64_t suffixTime =
                    publicTime - mutableFromTimeMs;
            while (hasNextEvent && nextEvent.timeMs <= suffixTime) {
                ApplyControlEvent(
                        controlState, nextEvent, mutableFromTimeMs);
                hasNextEvent = localityCursor.Next(&nextEvent);
            }
            localityCode =
                    (localityCode << 8u) |
                    ResolvedControlLocalityCode(controlState);
        }
    }
    simulationKeys[slot] =
            static_cast<std::uint64_t>(firstSimulationTick) << 32u |
            localityCode;
}

__device__ std::uint64_t HashCandidateInputEvent(
        std::uint64_t hash,
        const CudaSearchInputEvent &event) {
    constexpr std::uint64_t Prime = UINT64_C(1099511628211);
    const std::uint32_t words[] = {
            static_cast<std::uint32_t>(event.timeMs),
            event.action,
            event.valueKind,
            static_cast<std::uint32_t>(event.value),
    };
#pragma unroll
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        hash ^= words[index];
        hash *= Prime;
        hash ^= hash >> 32u;
    }
    return hash;
}

__global__ void HashSearchCandidateInputsKernel(
        const CudaSearchInputEvent *__restrict__ baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *__restrict__ candidateEvents,
        std::uint32_t eventCapacity,
        const std::int32_t *__restrict__ candidateInputValues,
        const std::uint32_t *__restrict__ compactInputOffsets,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *__restrict__ eventCounts,
        const bool *__restrict__ activeCandidates,
        std::uint64_t *__restrict__ hashes,
        std::uint32_t candidateCount) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount) return;
    if (!activeCandidates[slot]) {
        hashes[slot] = UINT64_MAX - slot;
        return;
    }
    const std::uint32_t eventCount = eventCounts[slot];
    const CudaSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    CandidateInputCursor cursor(
            baselineInputs, baselineInputCount, events,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount, slot, candidateCount);
    std::uint64_t hash =
            UINT64_C(1469598103934665603) ^ eventCount;
    for (std::uint32_t index = 0u; index < eventCount; ++index) {
        CudaSearchInputEvent event{};
        if (!cursor.Next(&event)) {
            hash ^= UINT64_C(0xd6e8feb86659fd93);
            break;
        }
        hash = HashCandidateInputEvent(hash, event);
    }
    hashes[slot] = hash;
}

__device__ bool SameCandidateInputs(
        std::uint32_t leftSlot,
        std::uint32_t rightSlot,
        const CudaSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *candidateEvents,
        std::uint32_t eventCapacity,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *eventCounts,
        std::uint32_t candidateCount) {
    const std::uint32_t eventCount = eventCounts[leftSlot];
    if (eventCount != eventCounts[rightSlot]) return false;
    const CudaSearchInputEvent *leftEvents =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(leftSlot) *
                              eventCapacity;
    const CudaSearchInputEvent *rightEvents =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(rightSlot) *
                              eventCapacity;
    CandidateInputCursor left(
            baselineInputs, baselineInputCount, leftEvents,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount,
            leftSlot, candidateCount);
    CandidateInputCursor right(
            baselineInputs, baselineInputCount, rightEvents,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount,
            rightSlot, candidateCount);
    for (std::uint32_t index = 0u; index < eventCount; ++index) {
        CudaSearchInputEvent leftEvent{};
        CudaSearchInputEvent rightEvent{};
        if (!left.Next(&leftEvent) || !right.Next(&rightEvent) ||
            !SameEvent(leftEvent, rightEvent)) {
            return false;
        }
    }
    return true;
}

__global__ void MarkUniqueSearchCandidateInputsKernel(
        const std::uint64_t *__restrict__ sortedHashes,
        const std::uint32_t *__restrict__ hashSortedSlots,
        const std::uint64_t *__restrict__ simulationKeys,
        const CudaSearchInputEvent *__restrict__ baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *__restrict__ candidateEvents,
        std::uint32_t eventCapacity,
        const std::int32_t *__restrict__ candidateInputValues,
        const std::uint32_t *__restrict__ compactInputOffsets,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *__restrict__ eventCounts,
        const bool *__restrict__ activeCandidates,
        std::uint64_t *__restrict__ deduplicatedSimulationKeys,
        std::uint32_t *__restrict__ deduplicatedSlots,
        std::uint32_t *__restrict__ representativeSlots,
        std::uint32_t replicaLimit,
        std::uint32_t candidateCount) {
    const std::uint32_t position =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= candidateCount) return;
    const std::uint32_t slot = hashSortedSlots[position];
    deduplicatedSlots[position] = slot;
    representativeSlots[slot] = slot;
    if (!activeCandidates[slot]) {
        deduplicatedSimulationKeys[position] = UINT64_MAX;
        return;
    }
    const std::uint64_t hash = sortedHashes[position];
    std::uint32_t low = 0u;
    std::uint32_t high = position;
    while (low < high) {
        const std::uint32_t middle = low + (high - low) / 2u;
        if (sortedHashes[middle] < hash) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    const std::uint32_t firstSlot = hashSortedSlots[low];
    if (firstSlot != slot && activeCandidates[firstSlot] &&
        SameCandidateInputs(
                firstSlot, slot,
                baselineInputs, baselineInputCount,
                candidateEvents, eventCapacity,
                candidateInputValues, compactInputOffsets,
                compactRandomSteeringPipeline,
                compactEditPipeline, sparseMutationPipeline,
                candidateEdits, sparseCandidateEvents,
                eventCounts, candidateCount)) {
        if (position - low < replicaLimit) {
            deduplicatedSimulationKeys[position] =
                    simulationKeys[slot];
            return;
        }
        representativeSlots[slot] = firstSlot;
        deduplicatedSimulationKeys[position] = UINT64_MAX;
        return;
    }
    deduplicatedSimulationKeys[position] = simulationKeys[slot];
}

__global__ void ExpandDeduplicatedSearchSamplesKernel(
        DeviceSample *__restrict__ candidateBestSamples,
        DeviceCandidateStatus *__restrict__ statuses,
        const std::uint32_t *__restrict__ representativeSlots,
        std::uint64_t firstCandidateId,
        std::uint32_t evaluationTickCount,
        std::uint32_t candidateCount) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount) return;
    const std::uint32_t representative = representativeSlots[slot];
    if (representative == slot) return;
    const DeviceSample sample = BindSampleToCandidate(
            candidateBestSamples[representative + 1u],
            firstCandidateId, slot, evaluationTickCount);
    candidateBestSamples[slot + 1u] = sample;
    statuses[slot] = statuses[representative];
}

template <typename State, bool SimulateStunts>
__device__ State LoadSearchState(
        const CudaCandidateState *branchState) {
    if constexpr (SimulateStunts) {
        return *branchState;
    } else {
        return static_cast<const CudaCandidatePhysicsState &>(
                *branchState);
    }
}

template <
        typename State,
        bool SimulateStunts,
        bool SteadyTimeline,
        CudaHandlingSpecialization Handling,
        std::uint32_t MinimumBlocksPerSm,
        bool CollectHotPathMetrics = false>
__global__ __launch_bounds__(
        SimulationBlockSize,
        MinimumBlocksPerSm) void SimulateSearchCandidatesKernel(
        const void *__restrict__ sceneData,
        const void *__restrict__ configurationData,
        const CudaCandidateState *__restrict__ branchState,
        const DeviceControlState *__restrict__ mutableBoundaryControls,
        const CudaControlTick *__restrict__ baselineTicks,
        std::uint32_t timelineTickCount,
        const CudaSearchEvaluatorConfiguration *__restrict__ evaluator,
        const CudaSearchConditionInstruction *__restrict__ condition,
        std::uint32_t conditionInstructionCount,
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds,
        double currentTimeSeconds,
        std::uint32_t tickDurationMs,
        std::uint32_t prestartDurationMs,
        std::int64_t branchTimeMs,
        std::int64_t mutableFromTimeMs,
        std::int64_t evaluationStartTimeMs,
        std::uint32_t evaluationTickCount,
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool baseline,
        CudaCandidatePhysicsState *__restrict__ baselinePrefixStates,
        DeviceSample *__restrict__ baselinePrefixBestSamples,
        double *__restrict__
                baselinePrefixClosestTargetDistanceSquared,
        const std::uint32_t *__restrict__ simulationCandidateSlots,
        const std::uint64_t *__restrict__ simulationKeys,
        std::uint32_t eventCapacity,
        DeviceSample *__restrict__ candidateBestSamples,
        double *__restrict__ blockClosestTargetDistanceSquared,
        CudaCandidatePhysicsState *__restrict__ finishCheckpointStates,
        std::uint32_t *__restrict__ finishCheckpointTicks,
        const CudaSearchInputEvent *__restrict__ baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *__restrict__ candidateEvents,
        const std::int32_t *__restrict__ candidateInputValues,
        const std::uint32_t *__restrict__ compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *__restrict__ eventCounts,
        DeviceCandidateStatus *__restrict__ statuses,
        const bool *__restrict__ activeCandidates,
        cuda::collision::CudaCollisionSearchTile *__restrict__
                collisionScratch,
        cuda::collision::CudaCollisionSearchTile *__restrict__
                shapeCollisionScratch,
        GmIso4 *__restrict__ shapeWorldScratch,
        GmBoxAligned *__restrict__ movingBoundsScratch,
        cuda::collision::CudaCollisionSurfaceHit *
                __restrict__ surfaceHitScratch,
        cuda::collision::CudaCollisionMeshRange *
                __restrict__ meshRangeScratch,
        std::uint32_t *__restrict__ meshCellScratch,
        std::uint16_t *__restrict__ responseOrderScratch,
        std::uint32_t scratchStride,
        std::uint32_t shapeCapacity,
        const std::uint32_t *__restrict__ cancellation) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
    sceneData = reinterpret_cast<const CudaPackedSceneHeader *>(
            cuda::research::ForeverValidatorSessionSceneBytes());
    configurationData =
            reinterpret_cast<
                    const CudaPackedStaticConfigurationHeader *>(
                    cuda::research::
                            ForeverValidatorSessionConfigurationBytes());
#elif defined(FOREVERVALIDATOR_CUDA_RESEARCH_WATER_ONLY)
    sceneData = &cuda::research::StaticScene;
    configurationData =
            &cuda::research::StaticConfiguration;
#endif
    const std::uint32_t workSlot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (workSlot >= candidateCount) {
        return;
    }
    const std::uint32_t slot = simulationCandidateSlots == nullptr
            ? workSlot : simulationCandidateSlots[workSlot];
    const std::uint64_t simulationKey = simulationKeys == nullptr
            ? 0u : simulationKeys[workSlot];
    if (simulationKey == UINT64_MAX) {
        return;
    }
    const std::uint32_t firstSimulationTick =
            static_cast<std::uint32_t>(simulationKey >> 32u);
    if (!activeCandidates[slot]) {
        return;
    }
    DeviceHotPathStorage<CollectHotPathMetrics> hotPathStorage;
    cuda::collision::CudaHotPathCounters *hotPathCounters = nullptr;
    if constexpr (CollectHotPathMetrics) {
        hotPathStorage.record.physicallySimulatedCandidateCount = 1u;
        hotPathStorage.record.firstSimulationTick = firstSimulationTick;
        hotPathCounters = &hotPathStorage.record.physics;
    }
    const std::uint64_t candidateId = firstCandidateId + slot;
    const CudaSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    const std::uint32_t eventCount = eventCounts[slot];
    CandidateInputCursor inputCursor(
            baselineInputs, baselineInputCount, events,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount, slot, candidateCount);
    CudaSearchInputEvent nextEvent{};
    bool hasNextEvent =
            eventCount != 0u && inputCursor.Next(&nextEvent);
    const CudaSearchEvaluatorConfiguration configuredEvaluator =
            *evaluator;
    const DeviceSample incumbent = candidateBestSamples[0];
    const bool pruneFinishTime =
            configuredEvaluator.kind ==
                    CudaSearchEvaluatorKind::FinishTime &&
            incumbent.preciseFinish;
    if (!ValidPackedInputs(sceneData, configurationData) ||
        branchState->schemaVersion != CudaCandidateState::SchemaVersion) {
        statuses[slot] =
                DeviceCandidateStatus::UnsupportedPhysicsTransition;
        if constexpr (CollectHotPathMetrics) {
            auto *records = reinterpret_cast<DeviceHotPathRecord *>(
                    reinterpret_cast<std::byte *>(
                            blockClosestTargetDistanceSquared) +
                    HotPathRecordsOffset(scratchStride));
            records[workSlot] = hotPathStorage.record;
        }
        return;
    }

    State state =
            LoadSearchState<State, SimulateStunts>(branchState);
    if constexpr (!SimulateStunts) {
        if (simulationKeys != nullptr &&
            firstSimulationTick != 0u) {
            state = baselinePrefixStates[firstSimulationTick - 1u];
        }
    }
    state.candidateId = static_cast<std::uint32_t>(candidateId);
    cuda::collision::CudaCollisionSearchScratch candidateScratch{
            0u,
            0u,
            0u,
            false,
            true,
            collisionScratch,
            shapeCollisionScratch,
            shapeWorldScratch,
            movingBoundsScratch,
            surfaceHitScratch,
            meshRangeScratch,
            meshCellScratch,
            workSlot,
            scratchStride,
            shapeCapacity};
    candidateScratch.responseOrderStorage =
            responseOrderScratch;
    DeviceControlState controlState = *mutableBoundaryControls;
    if (firstSimulationTick != 0u) {
        const std::int64_t previousPublicTime =
                branchTimeMs +
                static_cast<std::int64_t>(firstSimulationTick) *
                        tickDurationMs;
        const std::int64_t previousSuffixTime =
                previousPublicTime - mutableFromTimeMs;
        while (hasNextEvent &&
               nextEvent.timeMs <= previousSuffixTime) {
            ApplyControlEvent(
                    controlState, nextEvent, mutableFromTimeMs);
            hasNextEvent = inputCursor.Next(&nextEvent);
        }
    }
    bool evaluatorReported = false;
    const bool maximize = MaximizesScore(configuredEvaluator.kind);
    DeviceSample localBest;
    double closestTargetDistanceSquared =
            cuda_search_progress_detail::
                    InvalidClosestTargetDistanceSquared;
    if (!baseline && simulationKeys != nullptr &&
        firstSimulationTick != 0u) {
        localBest = BindSampleToCandidate(
                baselinePrefixBestSamples[firstSimulationTick - 1u],
                firstCandidateId, slot, evaluationTickCount);
        closestTargetDistanceSquared =
                baselinePrefixClosestTargetDistanceSquared[
                        firstSimulationTick - 1u];
    }
    const std::int64_t firstSimulationPublicTimeMs =
            branchTimeMs +
            static_cast<std::int64_t>(firstSimulationTick + 1u) *
                    tickDurationMs;
    const std::uint32_t evaluationIndexOffset =
            firstSimulationPublicTimeMs >= evaluationStartTimeMs
            ? static_cast<std::uint32_t>(
                      (firstSimulationPublicTimeMs -
                       evaluationStartTimeMs) /
                      tickDurationMs)
            : 0u;
    std::uint32_t evaluationIndex = evaluationIndexOffset;
    if (finishCheckpointTicks != nullptr) {
        finishCheckpointTicks[slot] = FinishCheckpointInvalidTick;
    }
    for (std::uint32_t tickIndex = firstSimulationTick;
         tickIndex < timelineTickCount; ++tickIndex) {
        if ((tickIndex & 63u) == 0u &&
            *reinterpret_cast<volatile const std::uint32_t *>(
                    cancellation) != 0u) {
            statuses[slot] = DeviceCandidateStatus::Cancelled;
            candidateBestSamples[slot + 1u] = localBest;
            goto finalize;
        }
        const std::int64_t publicTime =
                branchTimeMs +
                static_cast<std::int64_t>(tickIndex + 1u) *
                        tickDurationMs;
        const std::int64_t suffixTime =
                publicTime - mutableFromTimeMs;
        while (hasNextEvent && nextEvent.timeMs <= suffixTime) {
            ApplyControlEvent(
                    controlState, nextEvent, mutableFromTimeMs);
            hasNextEvent = inputCursor.Next(&nextEvent);
        }
        CudaControlTick tick = baselineTicks[tickIndex];
        tick.controls = ControlsFromState(controlState);
        tick.stuntsInput =
                StuntsFromState(controlState, prestartDurationMs);
        if constexpr (!SimulateStunts) {
            if (finishCheckpointStates != nullptr &&
                !state.race.progress.raceCompleted) {
                finishCheckpointStates[slot] = state;
                finishCheckpointTicks[slot] = tickIndex;
            }
        }
        const GmVec3 previousPosition = state.body.current.position;
        ApplyControlPrefix(state, tick);
        if (!state.firstStep) {
            if constexpr (SteadyTimeline) {
                cuda::transition::PrepareSteadyStep(state, tick);
            } else {
                cuda::transition::PrepareStep(
                        state, tick,
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData));
            }
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        if constexpr (SteadyTimeline) {
            state.vehicle.mobil.physicsUpdatesEnabled = true;
        } else {
            state.vehicle.mobil.physicsUpdatesEnabled =
                    (tick.actionFlags &
                     CudaControlActionSuppressVehicleForceCallbacks) ==
                    0u;
            for (std::uint32_t respawn = 0u;
                 respawn < tick.respawnAtCheckpointCount; ++respawn) {
                if (cuda::transition::Respawn(
                            state,
                            static_cast<const
                                    CudaPackedStaticConfigurationHeader *>(
                                    configurationData))) {
                    ++state.incrementalRespawnCount;
                    if constexpr (SimulateStunts) {
                        cuda::stunts::ApplyRespawnPenalty(
                                state.stunts);
                    }
                }
            }
        }
        if constexpr (CollectHotPathMetrics) {
            ++hotPathStorage.record.executedTickCount;
        }
        const cuda::physics::Status physicsStatus =
                cuda::physics::Step<
                        false,
                        (MinimumBlocksPerSm >
                         1u),
                        true,
                        true,
                        true,
                        (MinimumBlocksPerSm !=
                         ThroughputKernelMinimumBlocksPerSm),
                        SimulateStunts,
                        Handling,
                        CollectHotPathMetrics>(
                        static_cast<const CudaPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, candidateScratch, hotPathCounters);
        if (physicsStatus != cuda::physics::Status::Success) {
            statuses[slot] =
                    DeviceCandidateStatus::UnsupportedPhysicsTransition;
            candidateBestSamples[slot + 1u] = localBest;
            goto finalize;
        }
        if constexpr (CollectHotPathMetrics) {
            ++hotPathStorage.record.completedTickCount;
        }
        if constexpr (SimulateStunts) {
            const cuda::stunts::Status stuntStatus =
                    cuda::stunts::Update(state, tick);
        if (stuntStatus != cuda::stunts::Status::Success) {
            statuses[slot] =
                    DeviceCandidateStatus::CapacityExceeded;
            candidateBestSamples[slot + 1u] = localBest;
            goto finalize;
        }
        }
        state.firstStep = false;
        ++state.controlCursor;
        if constexpr (!SimulateStunts) {
            if (baseline && baselinePrefixStates != nullptr) {
                baselinePrefixStates[tickIndex] = state;
            }
        }
        if (publicTime < evaluationStartTimeMs) {
            if (baseline && baselinePrefixBestSamples != nullptr) {
                baselinePrefixBestSamples[tickIndex] = localBest;
                baselinePrefixClosestTargetDistanceSquared[tickIndex] =
                        closestTargetDistanceSquared;
            }
            continue;
        }
        if (configuredEvaluator.kind ==
            CudaSearchEvaluatorKind::VolumeEntry) {
            const GmVec3 &position = state.body.current.position;
            const double distanceSquared =
                    cuda_search_progress_detail::
                            SquaredDistanceToTargetVolume(
                                    configuredEvaluator.values,
                                    position.x, position.y, position.z);
            closestTargetDistanceSquared =
                    cuda_search_progress_detail::
                            UpdateClosestTargetDistanceSquared(
                                    closestTargetDistanceSquared,
                                    distanceSquared, false);
        }
        if (conditionInstructionCount != 0u &&
            !EvaluateCondition(
                    condition, conditionInstructionCount, state,
                    CudaSearchConditionIterationValue(
                            baseline, candidateId),
                    lastImprovementTimeSeconds,
                    lastRestartTimeSeconds, currentTimeSeconds)) {
            ++evaluationIndex;
            if (state.race.progress.raceCompleted) break;
            continue;
        }
        std::uint32_t stuntsScore = 0u;
        if constexpr (SimulateStunts) {
            stuntsScore = state.stunts.stuntsScore;
        }
        DeviceSample sample = EvaluateState(
                configuredEvaluator, state, previousPosition,
                static_cast<double>(publicTime - tickDurationMs),
                static_cast<double>(publicTime),
                stuntsScore,
                &evaluatorReported);
        if (sample.valid &&
            configuredEvaluator.kind ==
                    CudaSearchEvaluatorKind::VolumeEntry) {
            closestTargetDistanceSquared =
                    cuda_search_progress_detail::
                            UpdateClosestTargetDistanceSquared(
                                    closestTargetDistanceSquared,
                                    closestTargetDistanceSquared, true);
        }
        sample.candidateId = candidateId;
        sample.candidateSlot = slot;
        sample.evaluationTick = evaluationIndex;
        sample.logicalOrder =
                1u +
                static_cast<std::uint64_t>(slot) *
                        evaluationTickCount +
                evaluationIndex;
        sample.mutation = !baseline;
        if (StrictlyBetter(sample, localBest, maximize)) {
            localBest = sample;
        }
        if (baseline && baselinePrefixBestSamples != nullptr) {
            baselinePrefixBestSamples[tickIndex] = localBest;
            baselinePrefixClosestTargetDistanceSquared[tickIndex] =
                    closestTargetDistanceSquared;
        }
        if (sample.valid &&
            configuredEvaluator.kind ==
                    CudaSearchEvaluatorKind::VolumeEntry) {
            candidateBestSamples[slot + 1u] = localBest;
            goto finalize;
        }
        ++evaluationIndex;
        if (configuredEvaluator.kind ==
            CudaSearchEvaluatorKind::FinishTime) {
            if (evaluatorReported) {
                candidateBestSamples[slot + 1u] = localBest;
                goto finalize;
            }
            if (pruneFinishTime &&
                static_cast<double>(publicTime) >= incumbent.timeMs) {
                candidateBestSamples[slot + 1u] = DeviceSample{};
                goto finalize;
            }
        }
    }
finalize:
    candidateBestSamples[slot + 1u] = localBest;
    if constexpr (CollectHotPathMetrics) {
        auto *records = reinterpret_cast<DeviceHotPathRecord *>(
                reinterpret_cast<std::byte *>(
                        blockClosestTargetDistanceSquared) +
                HotPathRecordsOffset(scratchStride));
        records[workSlot] = hotPathStorage.record;
    }
    if (closestTargetDistanceSquared !=
        cuda_search_progress_detail::
                InvalidClosestTargetDistanceSquared) {
        cuda_search_progress_detail::AtomicMinNonnegativeDouble(
                closestTargetDistanceSquared,
                &blockClosestTargetDistanceSquared[blockIdx.x]);
    }
}

template <
        typename State,
        bool SimulateStunts>
__global__ void RefineSearchFinishTimesKernel(
        const void *sceneData,
        const void *configurationData,
        const CudaCandidateState *branchState,
        const DeviceControlState *mutableBoundaryControls,
        const CudaControlTick *baselineTicks,
        std::uint32_t timelineTickCount,
        std::uint32_t tickDurationMs,
        std::uint32_t prestartDurationMs,
        std::int64_t branchTimeMs,
        std::int64_t mutableFromTimeMs,
        std::int64_t evaluationStartTimeMs,
        std::uint32_t eventCapacity,
        DeviceSample *candidateBestSamples,
        cuda::finish::Refinement *finishRefinements,
        const CudaCandidatePhysicsState *finishCheckpointStates,
        const std::uint32_t *finishCheckpointTicks,
        const CudaSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *candidateEvents,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *eventCounts,
        DeviceCandidateStatus *statuses,
        const bool *activeCandidates,
        cuda::collision::CudaCollisionSearchTile *collisionScratch,
        cuda::collision::CudaCollisionSearchTile *shapeCollisionScratch,
        GmIso4 *shapeWorldScratch,
        GmBoxAligned *movingBoundsScratch,
        cuda::collision::CudaCollisionSurfaceHit *
                surfaceHitScratch,
        cuda::collision::CudaCollisionMeshRange *
                meshRangeScratch,
        std::uint32_t *meshCellScratch,
        std::uint16_t *responseOrderScratch,
        std::uint32_t scratchStride,
        std::uint32_t shapeCapacity,
        const std::uint32_t *cancellation,
        std::uint32_t candidateCount) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount || !activeCandidates[slot]) {
        return;
    }
    DeviceSample &sample = candidateBestSamples[slot + 1u];
    if (!sample.valid ||
        statuses[slot] != DeviceCandidateStatus::Success) {
        return;
    }
    cuda::finish::Refinement &refinement =
            finishRefinements[slot];
    const std::uint64_t prestartNs =
            static_cast<std::uint64_t>(prestartDurationMs) *
            1000000u;
    refinement = {};
    const DeviceSample incumbent = candidateBestSamples[0];
    std::uint64_t incumbentUpperNs =
            ~std::uint64_t{0};
    if (incumbent.preciseFinish) {
        incumbentUpperNs = prestartNs +
                static_cast<std::uint64_t>(incumbent.score);
        const double coarseLowerMs =
                sample.timeMs -
                static_cast<double>(prestartDurationMs) -
                static_cast<double>(tickDurationMs);
        if (coarseLowerMs >= incumbent.timeMs) {
            sample = {};
            return;
        }
    }
    const CudaSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    const std::uint32_t eventCount = eventCounts[slot];
    CandidateInputCursor inputCursor(
            baselineInputs, baselineInputCount, events,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount, slot, candidateCount);
    CudaSearchInputEvent nextEvent{};
    bool hasNextEvent =
            eventCount != 0u && inputCursor.Next(&nextEvent);
    State state =
            LoadSearchState<State, SimulateStunts>(branchState);
    state.candidateId =
            static_cast<std::uint32_t>(sample.candidateId);
    cuda::collision::CudaCollisionSearchScratch candidateScratch{
            0u,
            0u,
            0u,
            false,
            false,
            collisionScratch,
            shapeCollisionScratch,
            shapeWorldScratch,
            movingBoundsScratch,
            surfaceHitScratch,
            meshRangeScratch,
            meshCellScratch,
            slot,
            scratchStride,
            shapeCapacity};
    candidateScratch.responseOrderStorage =
            responseOrderScratch;
    DeviceControlState controlState = *mutableBoundaryControls;
    std::uint32_t firstTickIndex = 0u;
    if constexpr (!SimulateStunts) {
        const std::uint32_t firstEvaluationTick =
                evaluationStartTimeMs <= branchTimeMs
                ? 0u
                : static_cast<std::uint32_t>(
                          (evaluationStartTimeMs - branchTimeMs) /
                                  tickDurationMs -
                          1u);
        const std::uint32_t targetTickIndex =
                firstEvaluationTick + sample.evaluationTick;
        const std::uint32_t checkpointTick =
                finishCheckpointTicks == nullptr
                ? FinishCheckpointInvalidTick
                : finishCheckpointTicks[slot];
        if (finishCheckpointStates != nullptr &&
            checkpointTick != FinishCheckpointInvalidTick &&
            targetTickIndex >= checkpointTick) {
            state = finishCheckpointStates[slot];
            firstTickIndex = checkpointTick;
            const std::int64_t checkpointPublicTime =
                    branchTimeMs +
                    static_cast<std::int64_t>(firstTickIndex) *
                            tickDurationMs;
            const std::int64_t checkpointSuffixTime =
                    checkpointPublicTime - mutableFromTimeMs;
            while (hasNextEvent &&
                   nextEvent.timeMs <= checkpointSuffixTime) {
                ApplyControlEvent(
                        controlState, nextEvent, mutableFromTimeMs);
                hasNextEvent = inputCursor.Next(&nextEvent);
            }
        }
    }
    for (std::uint32_t tickIndex = firstTickIndex;
         tickIndex < timelineTickCount; ++tickIndex) {
        if ((tickIndex & 63u) == 0u &&
            *reinterpret_cast<volatile const std::uint32_t *>(
                    cancellation) != 0u) {
            statuses[slot] = DeviceCandidateStatus::Cancelled;
            return;
        }
        const std::int64_t publicTime =
                branchTimeMs +
                static_cast<std::int64_t>(tickIndex + 1u) *
                        tickDurationMs;
        const std::int64_t suffixTime =
                publicTime - mutableFromTimeMs;
        while (hasNextEvent && nextEvent.timeMs <= suffixTime) {
            ApplyControlEvent(
                    controlState, nextEvent, mutableFromTimeMs);
            hasNextEvent = inputCursor.Next(&nextEvent);
        }
        CudaControlTick tick = baselineTicks[tickIndex];
        tick.controls = ControlsFromState(controlState);
        tick.stuntsInput =
                StuntsFromState(controlState, prestartDurationMs);
        ApplyControlPrefix(state, tick);
        if (!state.firstStep) {
            cuda::transition::PrepareStep(
                    state, tick,
                    static_cast<const
                            CudaPackedStaticConfigurationHeader *>(
                            configurationData));
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        state.vehicle.mobil.physicsUpdatesEnabled =
                (tick.actionFlags &
                 CudaControlActionSuppressVehicleForceCallbacks) == 0u;
        for (std::uint32_t respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn) {
            if (cuda::transition::Respawn(
                        state,
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData))) {
                ++state.incrementalRespawnCount;
                if constexpr (SimulateStunts) {
                    cuda::stunts::ApplyRespawnPenalty(state.stunts);
                }
            }
        }
        const cuda::physics::Status physicsStatus =
                cuda::finish::StepAndRefine<
                        false, true, true, true, true, true>(
                        static_cast<const CudaPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, tick, candidateScratch, refinement,
                        1u,
                        incumbentUpperNs);
        if (physicsStatus != cuda::physics::Status::Success ||
            refinement.failed) {
            statuses[slot] =
                    DeviceCandidateStatus::UnsupportedPhysicsTransition;
            return;
        }
        if (refinement.rejected) {
            sample = {};
            return;
        }
        if (refinement.present) {
            if (refinement.estimate.lowerBoundNs < prestartNs ||
                refinement.estimate.upperBoundNs < prestartNs) {
                statuses[slot] =
                        DeviceCandidateStatus::
                                UnsupportedPhysicsTransition;
                return;
            }
            sample.score = static_cast<double>(
                    refinement.estimate.upperBoundNs - prestartNs);
            sample.timeMs = sample.score / 1000000.0;
            sample.preciseFinish = true;
            return;
        }
        if constexpr (SimulateStunts) {
            const cuda::stunts::Status stuntStatus =
                    cuda::stunts::Update(state, tick);
            if (stuntStatus != cuda::stunts::Status::Success) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                return;
            }
        }
        state.firstStep = false;
        ++state.controlCursor;
    }
    statuses[slot] =
            DeviceCandidateStatus::UnsupportedPhysicsTransition;
}

__global__ void CaptureSearchWinnerStateKernel(
        const void *sceneData,
        const void *configurationData,
        const CudaCandidateState *branchState,
        const DeviceControlState *mutableBoundaryControls,
        const CudaControlTick *baselineTicks,
        const DeviceSample *reducedBest,
        const cuda::finish::Refinement *finishRefinements,
        std::uint32_t tickDurationMs,
        std::uint32_t prestartDurationMs,
        std::int64_t branchTimeMs,
        std::int64_t mutableFromTimeMs,
        std::int64_t evaluationStartTimeMs,
        std::uint32_t eventCapacity,
        const CudaSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *candidateEvents,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        std::uint32_t candidateCount,
        const std::uint32_t *eventCounts,
        DeviceCandidateStatus *statuses,
        cuda::collision::CudaCollisionSearchTile *collisionScratch,
        cuda::collision::CudaCollisionSearchTile *shapeCollisionScratch,
        GmIso4 *shapeWorldScratch,
        GmBoxAligned *movingBoundsScratch,
        cuda::collision::CudaCollisionSurfaceHit *
                surfaceHitScratch,
        cuda::collision::CudaCollisionMeshRange *
                meshRangeScratch,
        std::uint32_t *meshCellScratch,
        std::uint16_t *responseOrderScratch,
        std::uint32_t scratchStride,
        std::uint32_t shapeCapacity,
        CudaCandidateState *capturedWinnerState) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const DeviceSample winner = *reducedBest;
    if (!winner.valid ||
        winner.candidateSlot == InvalidCandidateSlot) {
        return;
    }
    const std::uint32_t slot = winner.candidateSlot;
    const CudaSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    const std::uint32_t eventCount = eventCounts[slot];
    CandidateInputCursor inputCursor(
            baselineInputs, baselineInputCount, events,
            candidateInputValues, compactInputOffsets,
            compactRandomSteeringPipeline, compactEditPipeline,
            sparseMutationPipeline, candidateEdits,
            sparseCandidateEvents, eventCount, slot, candidateCount);
    CudaSearchInputEvent nextEvent{};
    bool hasNextEvent =
            eventCount != 0u && inputCursor.Next(&nextEvent);
    const std::uint32_t evaluationStartTick =
            static_cast<std::uint32_t>(
                    (evaluationStartTimeMs -
                     (branchTimeMs + tickDurationMs)) /
                    tickDurationMs);
    const std::uint32_t targetTick =
            evaluationStartTick + winner.evaluationTick;

    cuda::collision::CudaCollisionSearchScratch candidateScratch{
            0u,
            0u,
            0u,
            false,
            true,
            collisionScratch,
            shapeCollisionScratch,
            shapeWorldScratch,
            movingBoundsScratch,
            surfaceHitScratch,
            meshRangeScratch,
            meshCellScratch,
            slot,
            scratchStride,
            shapeCapacity};
    candidateScratch.responseOrderStorage =
            responseOrderScratch;
    CudaCandidateState state = *branchState;
    state.candidateId =
            static_cast<std::uint32_t>(winner.candidateId);
    DeviceControlState controlState = *mutableBoundaryControls;
    for (std::uint32_t tickIndex = 0u;
         tickIndex <= targetTick; ++tickIndex) {
        const std::int64_t publicTime =
                branchTimeMs +
                static_cast<std::int64_t>(tickIndex + 1u) *
                        tickDurationMs;
        const std::int64_t suffixTime =
                publicTime - mutableFromTimeMs;
        while (hasNextEvent && nextEvent.timeMs <= suffixTime) {
            ApplyControlEvent(
                    controlState, nextEvent, mutableFromTimeMs);
            hasNextEvent = inputCursor.Next(&nextEvent);
        }
        CudaControlTick tick = baselineTicks[tickIndex];
        tick.controls = ControlsFromState(controlState);
        tick.stuntsInput =
                StuntsFromState(controlState, prestartDurationMs);
        ApplyControlPrefix(state, tick);
        if (!state.firstStep) {
            cuda::transition::PrepareStep(
                    state, tick,
                    static_cast<const
                            CudaPackedStaticConfigurationHeader *>(
                            configurationData));
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        state.vehicle.mobil.physicsUpdatesEnabled =
                (tick.actionFlags &
                 CudaControlActionSuppressVehicleForceCallbacks) == 0u;
        for (std::uint32_t respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn) {
            if (cuda::transition::Respawn(
                        state,
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData))) {
                ++state.incrementalRespawnCount;
                cuda::stunts::ApplyRespawnPenalty(
                        state.stunts);
            }
        }
        const cuda::physics::Status physicsStatus =
                cuda::physics::Step<false, false, true>(
                        static_cast<const CudaPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                CudaPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, candidateScratch);
        if (physicsStatus != cuda::physics::Status::Success) {
            statuses[slot] =
                    DeviceCandidateStatus::UnsupportedPhysicsTransition;
            return;
        }
        if (state.stuntsEnabled) {
            const cuda::stunts::Status stuntStatus =
                    cuda::stunts::Update(state, tick);
            if (stuntStatus != cuda::stunts::Status::Success) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                return;
            }
        }
        state.firstStep = false;
        ++state.controlCursor;
    }
    cuda::collision::detail::CaptureReplacementOverflow(
            candidateScratch,
            state.collisionReplacementOverflow);
    if (finishRefinements != nullptr &&
        finishRefinements[slot].present) {
        const forevervalidator::FinishTimeEstimate &finishTime =
                finishRefinements[slot].estimate;
        state.finishTime.present = true;
        state.finishTime.value = {
                finishTime.lowerBoundNs,
                finishTime.upperBoundNs,
                finishTime.estimatedNs};
    }
    *capturedWinnerState = state;
}

struct DeviceBatchCounters {
    std::uint32_t evaluatedCandidateCount = 0u;
    std::uint32_t simulatedCandidateCount = 0u;
    std::uint64_t evaluatorCalls = 0u;
    std::uint64_t qualifyingCandidateCount = 0u;
    std::uint64_t totalMutationCount = 0u;
    std::uint64_t mutationImprovementCount = 0u;
};

struct AddDeviceBatchCounters {
    __device__ DeviceBatchCounters operator()(
            const DeviceBatchCounters &left,
            const DeviceBatchCounters &right) const {
        return {
                left.evaluatedCandidateCount +
                        right.evaluatedCandidateCount,
                left.simulatedCandidateCount +
                        right.simulatedCandidateCount,
                left.evaluatorCalls + right.evaluatorCalls,
                left.qualifyingCandidateCount +
                        right.qualifyingCandidateCount,
                left.totalMutationCount + right.totalMutationCount,
                left.mutationImprovementCount +
                        right.mutationImprovementCount};
    }
};

__global__ __launch_bounds__(BatchSummaryBlockSize)
void AccumulateSearchBatchSummaryKernel(
        const DeviceSample *__restrict__ prefixBestSamples,
        const DeviceSample *__restrict__ candidateBestSamples,
        const std::uint32_t *__restrict__ mutationCounts,
        const DeviceCandidateStatus *__restrict__ statuses,
        const bool *__restrict__ activeCandidates,
        const std::uint32_t *__restrict__ representativeSlots,
        bool deduplicateCandidateInputs,
        std::uint32_t candidateCount,
        std::uint32_t evaluationTickCount,
        const double *__restrict__ blockClosestTargetDistanceSquared,
        std::uint32_t simulationBlockCount,
        DeviceBatchSummary *__restrict__ summary) {
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    const bool validSlot = slot < candidateCount;
    const DeviceCandidateStatus candidateStatus = validSlot
            ? statuses[slot]
            : DeviceCandidateStatus::Success;
    if (validSlot &&
        candidateStatus != DeviceCandidateStatus::Success) {
        CudaSearchStatus batchStatus = CudaSearchStatus::Success;
        if (candidateStatus == DeviceCandidateStatus::Cancelled) {
            batchStatus = CudaSearchStatus::Cancelled;
        } else if (candidateStatus ==
                   DeviceCandidateStatus::CapacityExceeded) {
            batchStatus = CudaSearchStatus::CapacityExceeded;
        } else if (candidateStatus ==
                   DeviceCandidateStatus::UnsupportedPhysicsTransition) {
            batchStatus =
                    CudaSearchStatus::UnsupportedPhysicsTransition;
        }
        const std::uint64_t encoded =
                (static_cast<std::uint64_t>(slot + 1u) << 32u) |
                static_cast<std::uint32_t>(batchStatus);
        atomicMax(
                reinterpret_cast<unsigned long long *>(
                        &summary->lastFailure),
                static_cast<unsigned long long>(encoded));
    }
    DeviceBatchCounters local{};
    if (validSlot) {
        const bool active = activeCandidates[slot];
        local.evaluatedCandidateCount = active ? 1u : 0u;
        local.simulatedCandidateCount =
                active &&
                        (!deduplicateCandidateInputs ||
                         representativeSlots[slot] == slot)
                ? 1u : 0u;
        local.evaluatorCalls = active
                ? static_cast<std::uint64_t>(evaluationTickCount)
                : 0u;
        local.qualifyingCandidateCount =
                cuda_search_progress_detail::
                        IsQualifyingSearchCandidate(
                                active,
                                candidateBestSamples[slot + 1u].valid)
                ? 1u : 0u;
        local.totalMutationCount = mutationCounts[slot];
        const DeviceSample previous = prefixBestSamples[slot];
        const DeviceSample current = prefixBestSamples[slot + 1u];
        local.mutationImprovementCount =
                current.valid &&
                        current.logicalOrder != previous.logicalOrder
                ? 1u : 0u;
    }
    using BlockReduce = cub::BlockReduce<
            DeviceBatchCounters, BatchSummaryBlockSize>;
    __shared__ typename BlockReduce::TempStorage reductionStorage;
    const DeviceBatchCounters block =
            BlockReduce(reductionStorage).Reduce(
                    local, AddDeviceBatchCounters{});
    if (threadIdx.x == 0u) {
        atomicAdd(
                &summary->evaluatedCandidateCount,
                block.evaluatedCandidateCount);
        atomicAdd(
                &summary->simulatedCandidateCount,
                block.simulatedCandidateCount);
        atomicAdd(
                reinterpret_cast<unsigned long long *>(
                        &summary->evaluatorCalls),
                static_cast<unsigned long long>(block.evaluatorCalls));
        atomicAdd(
                reinterpret_cast<unsigned long long *>(
                        &summary->qualifyingCandidateCount),
                static_cast<unsigned long long>(
                        block.qualifyingCandidateCount));
        atomicAdd(
                reinterpret_cast<unsigned long long *>(
                        &summary->totalMutationCount),
                static_cast<unsigned long long>(
                        block.totalMutationCount));
        atomicAdd(
                reinterpret_cast<unsigned long long *>(
                        &summary->mutationImprovementCount),
                static_cast<unsigned long long>(
                        block.mutationImprovementCount));
        double closestTargetDistanceSquared =
                cuda_search_progress_detail::
                        InvalidClosestTargetDistanceSquared;
        for (std::uint32_t summarySlot = blockIdx.x;
             summarySlot < simulationBlockCount;
             summarySlot += gridDim.x) {
            closestTargetDistanceSquared =
                    cuda_search_progress_detail::
                            UpdateClosestTargetDistanceSquared(
                                    closestTargetDistanceSquared,
                                    blockClosestTargetDistanceSquared[
                                            summarySlot],
                                    false);
        }
        if (closestTargetDistanceSquared !=
            cuda_search_progress_detail::
                    InvalidClosestTargetDistanceSquared) {
            cuda_search_progress_detail::AtomicMinNonnegativeDouble(
                    closestTargetDistanceSquared,
                    &summary->closestTargetDistanceSquared);
        }
    }
}

__global__ void FinalizeSearchBatchKernel(
        const DeviceSample *reducedBest,
        const CudaCandidateState *capturedWinnerState,
        const DeviceSample *candidateBestSamples,
        const CudaSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const CudaSearchInputEvent *candidateEvents,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        cuda::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *eventCounts,
        const std::uint32_t *mutationCounts,
        const DeviceCandidateStatus *statuses,
        const bool *activeCandidates,
        std::uint32_t candidateCount,
        std::uint32_t eventCapacity,
        std::uint32_t evaluationTickCount,
        bool maximize,
        bool baseline,
        bool captureBestState,
        DeviceSample *globalBestSample,
        CudaCandidateState *globalBestState,
        CudaSearchInputEvent *globalBestInputs,
        std::uint32_t *globalBestEventCount,
        std::uint32_t *globalBestMutationCount,
        DeviceBatchSummary *summary) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    DeviceBatchSummary result = *summary;
    if (result.lastFailure != 0u) {
        result.status = static_cast<CudaSearchStatus>(
                static_cast<std::uint32_t>(result.lastFailure));
    }

    const DeviceSample winner = *reducedBest;
    if (winner.valid &&
        winner.candidateSlot != InvalidCandidateSlot) {
        const std::uint32_t slot = winner.candidateSlot;
        const DeviceSample candidateBest =
                candidateBestSamples[slot + 1u];
        if (candidateBest.valid) {
            *globalBestSample = candidateBest;
            if (captureBestState) {
                *globalBestState = *capturedWinnerState;
            }
            *globalBestEventCount = eventCounts[slot];
            *globalBestMutationCount = mutationCounts[slot];
            const CudaSearchInputEvent *materializedInputs =
                    candidateEvents == nullptr
                    ? nullptr
                    : candidateEvents +
                              static_cast<std::uint64_t>(slot) *
                                      eventCapacity;
            CandidateInputCursor inputCursor(
                    baselineInputs, baselineInputCount,
                    materializedInputs, candidateInputValues,
                    compactInputOffsets,
                    compactRandomSteeringPipeline,
                    compactEditPipeline, sparseMutationPipeline,
                    candidateEdits, sparseCandidateEvents,
                    eventCounts[slot], slot, candidateCount);
            for (std::uint32_t index = 0u;
                 index < eventCounts[slot]; ++index) {
                if (!inputCursor.Next(globalBestInputs + index)) {
                    result.status =
                            CudaSearchStatus::CapacityExceeded;
                    break;
                }
            }
            result.bestChanged = true;
        }
    }
    result.bestValid = globalBestSample->valid;
    result.bestMutation = globalBestSample->mutation;
    result.bestCandidateId = globalBestSample->candidateId;
    result.bestMutationCount = *globalBestMutationCount;
    result.globalEventCount = *globalBestEventCount;
    *summary = result;
}

}  // namespace

struct CudaSearchExecutor::Impl {
    struct SimulationKernelMetrics {
        std::uint32_t registersPerThread = 0u;
        std::uint64_t localBytesPerThread = 0u;
        std::uint32_t activeBlocksPerMultiprocessor = 0u;
        double theoreticalOccupancy = 0.0;
    };

    struct SimulationKernelTuning {
        std::uint32_t candidateCount = 0u;
        std::array<float, 3u> milliseconds{};
        std::array<std::uint32_t, 3u> sampleCounts{};
        std::uint32_t successfulBatchesSinceRevalidation = 0u;
        std::size_t nextRevalidationIndex = 0u;
    };

    CudaSearchExecutorConfiguration configuration;
    std::vector<CudaSearchInputEvent> immutableInputPrefix;
    std::vector<CudaSearchInputEvent> immutableInputTail;
    std::int64_t mutableFromTimeMs = 0;
    std::uint32_t timelineTickCount = 0u;
    std::uint32_t evaluationTickCount = 0u;
    std::uint32_t collisionShapeCount = 0u;
    std::uint64_t residentBytes = 0u;
    std::uint64_t winnerSelectionBytes = 0u;
    std::uint64_t initialUploadBytes = 0u;
    bool baselineEvaluated = false;
    bool baselinePrefixesValid = false;
    bool baselinePrefixReuseEligible = false;
    bool baselineInputsCanonical = false;
    bool compactRandomSteeringPipeline = false;
    bool compactEditPipeline = false;
    bool sparseMutationPipeline = false;
    bool directDeletionPipeline = false;
    bool directExistingEventPipeline = false;
    bool materializesCandidateEvents = true;
    bool packedEditStorage = false;
    bool editStorageAliasesTemporary = false;
    bool needsTemporaryEvents = true;
    bool needsPassBaselineEvents = true;
    bool needsEligibleIndices = true;
    CudaSearchPrefixReusePlan prefixReusePlan;
    bool steadyTimeline = false;
    // Mirrors the resident winner so unchanged batches only transfer summary
    // telemetry from the device.
    bool hostBestCacheValid = false;
    std::uint32_t hostBestEventCount = 0u;
    std::uint32_t compactInputCount = 0u;
    std::uint32_t sharedEligibleCount = 0u;
    std::uint32_t editCapacity = 0u;
    std::uint32_t eraseCapacity = 0u;
    std::size_t editStorageBytes = 0u;
    std::uint32_t multiprocessorCount = 0u;
    CudaHandlingSpecialization handlingSpecialization =
            CudaHandlingSpecialization::Generic;
    SimulationKernelMetrics throughputKernelMetrics;
    SimulationKernelMetrics tailKernelMetrics;
    SimulationKernelMetrics denseTailKernelMetrics;
    std::vector<SimulationKernelTuning> simulationKernelTunings;
    std::shared_ptr<const cuda::specialization::SessionModule>
            specializedModule;

    DeviceAllocation<CudaCandidateState> branchState;
    DeviceAllocation<DeviceControlState> mutableBoundaryControls;
    DeviceAllocation<CudaControlTick> baselineTicks;
    DeviceAllocation<CudaSearchInputEvent> baselineInputs;
    DeviceAllocation<CudaCandidatePhysicsState> baselinePrefixStates;
    DeviceAllocation<DeviceSample> baselinePrefixBestSamples;
    DeviceAllocation<double>
            baselinePrefixClosestTargetDistanceSquared;
    DeviceAllocation<CudaSearchModifierConfiguration> modifiers;
    DeviceAllocation<double> smoothWeights;
    DeviceAllocation<CudaSearchEvaluatorConfiguration> evaluator;
    DeviceAllocation<CudaSearchConditionInstruction> condition;
    DeviceAllocation<CudaCandidateState> capturedWinnerState;
    DeviceAllocation<DeviceSample> candidateBestSamples;
    DeviceAllocation<cuda::finish::Refinement> finishRefinements;
    DeviceAllocation<CudaCandidatePhysicsState> finishCheckpointStates;
    DeviceAllocation<std::uint32_t> finishCheckpointTicks;
    DeviceAllocation<std::uint32_t> randomStateWords;
    DeviceAllocation<CudaSearchInputEvent> candidateEvents;
    DeviceAllocation<std::uint32_t> compactInputIndices;
    DeviceAllocation<std::uint32_t> compactInputOffsets;
    DeviceAllocation<std::int32_t> candidateInputValues;
    DeviceAllocation<CudaSearchInputEvent> temporaryEvents;
    DeviceAllocation<CudaSearchInputEvent> passBaselineEvents;
    DeviceAllocation<std::uint32_t> eligibleIndices;
    DeviceAllocation<std::uint32_t> sharedEligibleIndices;
    DeviceAllocation<std::uint32_t> sparseReferences;
    DeviceAllocation<std::uint32_t> sparseSnapshotReferences;
    DeviceAllocation<CudaSearchInputEvent> sparseEdits;
    DeviceAllocation<CudaSearchInputEvent> sparseScratchEdits;
    DeviceAllocation<std::byte> editBacking;
    DeviceAllocation<std::uint32_t> eventCounts;
    DeviceAllocation<std::uint32_t> mutationCounts;
    DeviceAllocation<DeviceCandidateStatus> statuses;
    DeviceAllocation<bool> activeCandidates;
    DeviceAllocation<std::uint64_t> candidateSimulationKeys;
    DeviceAllocation<std::uint64_t> sortedCandidateSimulationKeys;
    DeviceAllocation<std::uint32_t> candidateSlots;
    DeviceAllocation<std::uint32_t> sortedCandidateSlots;
    DeviceAllocation<std::uint64_t> candidateInputHashes;
    DeviceAllocation<std::uint64_t> sortedCandidateInputHashes;
    DeviceAllocation<std::uint32_t> candidateRepresentativeSlots;
    DeviceAllocation<std::byte> candidateSortTemporary;
    DeviceAllocation<DeviceSample> prefixBestSamples;
    DeviceAllocation<std::byte> winnerScanTemporary;
    DeviceAllocation<DeviceSample> reducedBest;
    DeviceAllocation<cuda::collision::CudaCollisionSearchTile>
            collisionScratch;
    DeviceAllocation<cuda::collision::CudaCollisionSearchTile>
            shapeCollisionScratch;
    DeviceAllocation<GmIso4> shapeWorldScratch;
    DeviceAllocation<GmBoxAligned> movingBoundsScratch;
    DeviceAllocation<cuda::collision::CudaCollisionSurfaceHit>
            surfaceHitScratch;
    DeviceAllocation<cuda::collision::CudaCollisionMeshRange>
            meshRangeScratch;
    DeviceAllocation<std::uint32_t> meshCellScratch;
    DeviceAllocation<std::uint16_t> responseOrderScratch;
    MappedCancellation cancellation;
    DeviceAllocation<DeviceSample> globalBestSample;
    DeviceAllocation<CudaCandidateState> globalBestState;
    DeviceAllocation<CudaSearchInputEvent> globalBestInputs;
    DeviceAllocation<std::uint32_t> globalBestEventCount;
    DeviceAllocation<std::uint32_t> globalBestMutationCount;
    DeviceAllocation<DeviceBatchSummary> summary;
    DeviceAllocation<double> closestTargetDistanceSquaredByBlock;
    DeviceAllocation<std::byte> hotPathScratch;
    CudaSearchBest hostBestCache;

    bool DeduplicationStorageEligible(
            std::uint32_t candidateCapacity) const noexcept {
        return prefixReusePlan.DeduplicationStorageEligible(
                candidateCapacity);
    }

    double *ClosestTargetDistanceScratch() const {
        return configuration.collectHotPathMetrics
                ? reinterpret_cast<double *>(hotPathScratch.Get())
                : closestTargetDistanceSquaredByBlock.Get();
    }

    DeviceHotPathRecord *HotPathRecords() const {
        if (!configuration.collectHotPathMetrics) {
            return nullptr;
        }
        return reinterpret_cast<DeviceHotPathRecord *>(
                hotPathScratch.Get() +
                HotPathRecordsOffset(configuration.maximumBatchSize));
    }

    void UpdateResidentBytes() {
        residentBytes = 0u;
#define ADD_BYTES(member) residentBytes += member.Bytes()
        ADD_BYTES(branchState);
        ADD_BYTES(mutableBoundaryControls);
        ADD_BYTES(baselineTicks);
        ADD_BYTES(baselineInputs);
        ADD_BYTES(baselinePrefixStates);
        ADD_BYTES(baselinePrefixBestSamples);
        ADD_BYTES(baselinePrefixClosestTargetDistanceSquared);
        ADD_BYTES(modifiers);
        ADD_BYTES(smoothWeights);
        ADD_BYTES(evaluator);
        ADD_BYTES(condition);
        ADD_BYTES(capturedWinnerState);
        ADD_BYTES(candidateBestSamples);
        ADD_BYTES(finishRefinements);
        ADD_BYTES(finishCheckpointStates);
        ADD_BYTES(finishCheckpointTicks);
        ADD_BYTES(randomStateWords);
        ADD_BYTES(candidateEvents);
        ADD_BYTES(compactInputIndices);
        ADD_BYTES(compactInputOffsets);
        ADD_BYTES(candidateInputValues);
        ADD_BYTES(temporaryEvents);
        ADD_BYTES(passBaselineEvents);
        ADD_BYTES(eligibleIndices);
        ADD_BYTES(sharedEligibleIndices);
        ADD_BYTES(sparseReferences);
        ADD_BYTES(sparseSnapshotReferences);
        ADD_BYTES(sparseEdits);
        ADD_BYTES(sparseScratchEdits);
        ADD_BYTES(editBacking);
        ADD_BYTES(eventCounts);
        ADD_BYTES(mutationCounts);
        ADD_BYTES(statuses);
        ADD_BYTES(activeCandidates);
        ADD_BYTES(candidateSimulationKeys);
        ADD_BYTES(sortedCandidateSimulationKeys);
        ADD_BYTES(candidateSlots);
        ADD_BYTES(sortedCandidateSlots);
        ADD_BYTES(candidateInputHashes);
        ADD_BYTES(sortedCandidateInputHashes);
        ADD_BYTES(candidateRepresentativeSlots);
        ADD_BYTES(candidateSortTemporary);
        ADD_BYTES(prefixBestSamples);
        ADD_BYTES(winnerScanTemporary);
        ADD_BYTES(reducedBest);
        ADD_BYTES(collisionScratch);
        ADD_BYTES(shapeCollisionScratch);
        ADD_BYTES(shapeWorldScratch);
        ADD_BYTES(movingBoundsScratch);
        ADD_BYTES(surfaceHitScratch);
        ADD_BYTES(meshRangeScratch);
        ADD_BYTES(meshCellScratch);
        ADD_BYTES(responseOrderScratch);
        ADD_BYTES(cancellation);
        ADD_BYTES(globalBestSample);
        ADD_BYTES(globalBestState);
        ADD_BYTES(globalBestInputs);
        ADD_BYTES(globalBestEventCount);
        ADD_BYTES(globalBestMutationCount);
        ADD_BYTES(summary);
        ADD_BYTES(closestTargetDistanceSquaredByBlock);
        ADD_BYTES(hotPathScratch);
#undef ADD_BYTES
        winnerSelectionBytes =
                candidateBestSamples.Bytes() +
                prefixBestSamples.Bytes() +
                winnerScanTemporary.Bytes() +
                reducedBest.Bytes();
    }

    static std::size_t AlignStorage(
            std::size_t offset,
            std::size_t alignment) {
        return (offset + alignment - 1u) &
                ~(alignment - 1u);
    }

    static std::size_t EditStorageSize(
            std::uint32_t candidateStride,
            std::uint32_t outputCapacity,
            std::uint32_t suppressedCapacity,
            bool packed) {
        const std::size_t outputs =
                static_cast<std::size_t>(candidateStride) *
                outputCapacity;
        const std::size_t suppressed =
                static_cast<std::size_t>(candidateStride) *
                suppressedCapacity;
        std::size_t offset = 0u;
        const auto add = [&](std::size_t count,
                             std::size_t size,
                             std::size_t alignment) {
            offset = AlignStorage(offset, alignment);
            offset += count * size;
        };
        add(candidateStride, sizeof(std::uint32_t),
            alignof(std::uint32_t));
        add(candidateStride, sizeof(std::uint32_t),
            alignof(std::uint32_t));
        add(outputs, sizeof(std::uint32_t),
            alignof(std::uint32_t));
        add(outputs, sizeof(std::int32_t),
            alignof(std::int32_t));
        if (packed) {
            add(outputs, sizeof(std::uint8_t),
                alignof(std::uint8_t));
        } else {
            add(outputs, sizeof(std::uint32_t),
                alignof(std::uint32_t));
            add(outputs, sizeof(std::uint32_t),
                alignof(std::uint32_t));
        }
        add(outputs, sizeof(std::int32_t),
            alignof(std::int32_t));
        add(suppressed,
            packed ? sizeof(std::uint16_t)
                   : sizeof(std::uint32_t),
            packed ? alignof(std::uint16_t)
                   : alignof(std::uint32_t));
        return offset;
    }

    cuda::candidate_events::CoalescedEditStorage CandidateEdits(
            std::uint32_t candidateStride) const {
        if (!compactEditPipeline) {
            return {};
        }
        std::byte *base = editStorageAliasesTemporary
                ? reinterpret_cast<std::byte *>(
                          temporaryEvents.Get())
                : editBacking.Get();
        std::size_t offset = 0u;
        const auto take = [&](std::size_t count,
                              std::size_t size,
                              std::size_t alignment) {
            offset = AlignStorage(offset, alignment);
            std::byte *result = base + offset;
            offset += count * size;
            return result;
        };
        const std::size_t outputs =
                static_cast<std::size_t>(candidateStride) *
                editCapacity;
        const std::size_t suppressed =
                static_cast<std::size_t>(candidateStride) *
                eraseCapacity;
        cuda::candidate_events::CoalescedEditStorage storage;
        storage.counts = reinterpret_cast<std::uint32_t *>(
                take(candidateStride, sizeof(std::uint32_t),
                     alignof(std::uint32_t)));
        storage.erasedCounts =
                reinterpret_cast<std::uint32_t *>(
                        take(candidateStride,
                             sizeof(std::uint32_t),
                             alignof(std::uint32_t)));
        if (packedEditStorage) {
            storage.packedOutputActions =
                    reinterpret_cast<std::uint32_t *>(
                            take(outputs, sizeof(std::uint32_t),
                                 alignof(std::uint32_t)));
        } else {
            storage.outputIndices =
                    reinterpret_cast<std::uint32_t *>(
                            take(outputs, sizeof(std::uint32_t),
                                 alignof(std::uint32_t)));
        }
        storage.times = reinterpret_cast<std::int32_t *>(
                take(outputs, sizeof(std::int32_t),
                     alignof(std::int32_t)));
        if (packedEditStorage) {
            storage.packedValueKinds =
                    reinterpret_cast<std::uint8_t *>(
                            take(outputs, sizeof(std::uint8_t),
                                 alignof(std::uint8_t)));
        } else {
            storage.actions =
                    reinterpret_cast<std::uint32_t *>(
                            take(outputs, sizeof(std::uint32_t),
                                 alignof(std::uint32_t)));
            storage.valueKinds =
                    reinterpret_cast<std::uint32_t *>(
                            take(outputs, sizeof(std::uint32_t),
                                 alignof(std::uint32_t)));
        }
        storage.values = reinterpret_cast<std::int32_t *>(
                take(outputs, sizeof(std::int32_t),
                     alignof(std::int32_t)));
        if (packedEditStorage) {
            storage.packedErasedSourceIndices =
                    reinterpret_cast<std::uint16_t *>(
                            take(suppressed,
                                 sizeof(std::uint16_t),
                                 alignof(std::uint16_t)));
        } else {
            storage.erasedSourceIndices =
                    reinterpret_cast<std::uint32_t *>(
                            take(suppressed,
                                 sizeof(std::uint32_t),
                                 alignof(std::uint32_t)));
        }
        storage.candidateStride = candidateStride;
        storage.editCapacity = editCapacity;
        storage.eraseCapacity = eraseCapacity;
        return storage;
    }

    sparse_events::Storage SparseCandidateEvents(
            std::uint32_t candidateStride) const {
        if (!sparseMutationPipeline) {
            return {};
        }
        return {
                sparseReferences.Get(),
                sparseSnapshotReferences.Get(),
                sparseEdits.Get(),
                sparseScratchEdits.Get(),
                candidateStride,
                static_cast<std::uint32_t>(
                        configuration.maximumEventCount)};
    }

    template <
            std::uint32_t MinimumBlocksPerSm,
            CudaHandlingSpecialization Handling,
            bool CollectHotPathMetrics = false>
    const void *SimulationKernel() const {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_WATER_ONLY)
        return reinterpret_cast<const void *>(
                SimulateSearchCandidatesKernel<
                        CudaCandidatePhysicsState,
                        false,
                        true,
                        Handling,
                        MinimumBlocksPerSm,
                        CollectHotPathMetrics>);
#else
        if (configuration.branchState.stuntsEnabled &&
            configuration.evaluator.kind !=
                    CudaSearchEvaluatorKind::FinishTime) {
            return reinterpret_cast<const void *>(
                    SimulateSearchCandidatesKernel<
                            CudaCandidateState,
                            true,
                            false,
                            Handling,
                            MinimumBlocksPerSm,
                            CollectHotPathMetrics>);
        }
        return steadyTimeline
                ? reinterpret_cast<const void *>(
                          SimulateSearchCandidatesKernel<
                                  CudaCandidatePhysicsState,
                                  false,
                                  true,
                                  Handling,
                                  MinimumBlocksPerSm,
                                  CollectHotPathMetrics>)
                : reinterpret_cast<const void *>(
                          SimulateSearchCandidatesKernel<
                                  CudaCandidatePhysicsState,
                                  false,
                                  false,
                                  Handling,
                                  MinimumBlocksPerSm,
                                  CollectHotPathMetrics>);
#endif
    }

    template <
            std::uint32_t MinimumBlocksPerSm,
            bool CollectHotPathMetrics = false>
    const void *SelectedSimulationKernel() const {
        if constexpr (CollectHotPathMetrics) {
            // Profiling is intentionally one unambiguous generic AOT kernel.
            // It never inherits session or handling specialization identity.
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    CudaHandlingSpecialization::Generic,
                    true>();
        }
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_WATER_ONLY)
        return SimulationKernel<
                MinimumBlocksPerSm,
                CudaHandlingSpecialization::GearedDriveWater>();
#else
        switch (handlingSpecialization) {
        case CudaHandlingSpecialization::Legacy:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    CudaHandlingSpecialization::Legacy>();
        case CudaHandlingSpecialization::GearedDriveDry:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    CudaHandlingSpecialization::GearedDriveDry>();
        case CudaHandlingSpecialization::GearedDriveWater:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    CudaHandlingSpecialization::GearedDriveWater>();
        case CudaHandlingSpecialization::Generic:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    CudaHandlingSpecialization::Generic>();
        }
        return SimulationKernel<
                MinimumBlocksPerSm,
                CudaHandlingSpecialization::Generic>();
#endif
    }

    bool LoadSimulationKernelMetrics(
            const void *kernel,
            const cudaDeviceProp &properties,
            SimulationKernelMetrics *metrics,
            std::string *diagnostic) {
        cudaFuncAttributes attributes{};
        cudaError_t error =
                cudaFuncGetAttributes(&attributes, kernel);
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "querying CUDA simulation kernel attributes",
                        error);
            }
            return false;
        }
        int activeBlocks = 0;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks, kernel, SimulationBlockSize, 0u);
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "querying CUDA simulation occupancy", error);
            }
            return false;
        }
        metrics->registersPerThread =
                static_cast<std::uint32_t>(attributes.numRegs);
        metrics->localBytesPerThread =
                static_cast<std::uint64_t>(
                        attributes.localSizeBytes);
        metrics->activeBlocksPerMultiprocessor =
                static_cast<std::uint32_t>(activeBlocks);
        metrics->theoreticalOccupancy =
                properties.maxThreadsPerMultiProcessor == 0
                ? 0.0
                : static_cast<double>(
                          activeBlocks * SimulationBlockSize) /
                          properties.maxThreadsPerMultiProcessor;
        if (diagnostic != nullptr) {
            diagnostic->clear();
        }
        return true;
    }

    bool LoadSimulationKernelMetrics(std::string *diagnostic) {
        int device = 0;
        cudaDeviceProp properties{};
        cudaError_t error = cudaGetDevice(&device);
        if (error == cudaSuccess) {
            error = cudaGetDeviceProperties(&properties, device);
        }
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "querying CUDA device properties", error);
            }
            return false;
        }
        multiprocessorCount =
                static_cast<std::uint32_t>(
                        properties.multiProcessorCount);
        if (!configuration.collectHotPathMetrics &&
            specializedModule && specializedModule->Ready()) {
            const auto load =
                    [&](std::uint32_t minimumBlocks,
                        SimulationKernelMetrics *metrics) {
                const cuda::specialization::KernelMetrics &source =
                        specializedModule->Metrics(minimumBlocks);
                metrics->registersPerThread =
                        source.registersPerThread;
                metrics->localBytesPerThread =
                        source.localBytesPerThread;
                metrics->activeBlocksPerMultiprocessor =
                        source.activeBlocksPerMultiprocessor;
                metrics->theoreticalOccupancy =
                        properties.maxThreadsPerMultiProcessor == 0
                        ? 0.0
                        : static_cast<double>(
                                  source.activeBlocksPerMultiprocessor *
                                  SimulationBlockSize) /
                                  properties.maxThreadsPerMultiProcessor;
            };
            load(ThroughputKernelMinimumBlocksPerSm,
                 &throughputKernelMetrics);
            load(TailKernelMinimumBlocksPerSm,
                 &tailKernelMetrics);
            load(DenseTailKernelMinimumBlocksPerSm,
                 &denseTailKernelMetrics);
            if (diagnostic != nullptr) {
                diagnostic->clear();
            }
            return true;
        }
#if !defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
        if (configuration.collectHotPathMetrics) {
            return LoadSimulationKernelMetrics(
                           SelectedSimulationKernel<
                                   ThroughputKernelMinimumBlocksPerSm,
                                   true>(),
                           properties,
                           &throughputKernelMetrics,
                           diagnostic) &&
                   LoadSimulationKernelMetrics(
                           SelectedSimulationKernel<
                                   TailKernelMinimumBlocksPerSm,
                                   true>(),
                           properties,
                           &tailKernelMetrics,
                           diagnostic) &&
                   LoadSimulationKernelMetrics(
                           SelectedSimulationKernel<
                                   DenseTailKernelMinimumBlocksPerSm,
                                   true>(),
                           properties,
                           &denseTailKernelMetrics,
                           diagnostic);
        }
#endif
        return LoadSimulationKernelMetrics(
                       SelectedSimulationKernel<
                               ThroughputKernelMinimumBlocksPerSm>(),
                       properties,
                       &throughputKernelMetrics,
                       diagnostic) &&
               LoadSimulationKernelMetrics(
                       SelectedSimulationKernel<
                               TailKernelMinimumBlocksPerSm>(),
                       properties,
                       &tailKernelMetrics,
                       diagnostic) &&
               LoadSimulationKernelMetrics(
                       SelectedSimulationKernel<
                               DenseTailKernelMinimumBlocksPerSm>(),
                       properties,
                       &denseTailKernelMetrics,
                       diagnostic);
    }

    bool ReserveBatchCapacity(
            std::uint32_t candidateCount,
            std::string *diagnostic) {
        if (candidateCount <= configuration.maximumBatchSize) {
            if (diagnostic != nullptr) {
                diagnostic->clear();
            }
            return true;
        }
        const std::uint64_t eventSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                configuration.maximumEventCount;
        const std::uint64_t compactValueSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                compactInputCount;
        const std::uint64_t winnerSlots64 =
                1u + static_cast<std::uint64_t>(candidateCount);
        const std::uint64_t collisionSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                cuda::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                cuda::collision::ShapeCollisionCapacity;
        const std::uint64_t collisionTileStride64 =
                (static_cast<std::uint64_t>(candidateCount) +
                 cuda::collision::CudaCollisionSearchTileWidth - 1u) /
                cuda::collision::CudaCollisionSearchTileWidth;
        const std::uint64_t collisionTileSlots64 =
                collisionTileStride64 *
                cuda::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionTileSlots64 =
                collisionTileStride64 *
                cuda::collision::ShapeCollisionCapacity;
        const std::uint64_t shapeQuerySlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                collisionShapeCount;
        const std::uint64_t surfaceHitSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                cuda::collision::SurfaceHitCapacity;
        const std::uint64_t meshRangeSlots64 =
                surfaceHitSlots64;
        const std::uint64_t meshCellSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                cuda::collision::MeshCellHitCapacity;
        if (eventSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            compactValueSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            winnerSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            collisionSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            shapeCollisionSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            collisionTileSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            shapeCollisionTileSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            shapeQuerySlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            surfaceHitSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            meshRangeSlots64 >
                    std::numeric_limits<std::size_t>::max() ||
            meshCellSlots64 >
                    std::numeric_limits<std::size_t>::max()) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "CUDA search calibration buffer dimensions overflow";
            }
            return false;
        }

        const std::size_t candidates = candidateCount;
        const std::size_t eventSlots =
                static_cast<std::size_t>(eventSlots64);
        const std::size_t compactValueSlots =
                static_cast<std::size_t>(compactValueSlots64);
        const std::size_t winnerSlots =
                static_cast<std::size_t>(winnerSlots64);
        const std::size_t collisionSlots =
                static_cast<std::size_t>(collisionSlots64);
        const std::size_t collisionTileSlots =
                static_cast<std::size_t>(collisionTileSlots64);
        const std::size_t shapeCollisionTileSlots =
                static_cast<std::size_t>(
                        shapeCollisionTileSlots64);
        const std::size_t shapeQuerySlots =
                static_cast<std::size_t>(shapeQuerySlots64);
        const std::size_t surfaceHitSlots =
                static_cast<std::size_t>(surfaceHitSlots64);
        const std::size_t meshRangeSlots =
                static_cast<std::size_t>(meshRangeSlots64);
        const std::size_t meshCellSlots =
                static_cast<std::size_t>(meshCellSlots64);
        const std::size_t summaryBlockCount =
                (candidates + SimulationBlockSize - 1u) /
                SimulationBlockSize;
        const std::size_t prefixCandidateSlots =
                baselinePrefixReuseEligible ? candidates : 0u;
        const std::size_t deduplicationCandidateSlots =
                DeduplicationStorageEligible(candidateCount)
                ? candidates : 0u;
        DeviceAllocation<DeviceSample> nextCandidateBestSamples;
        DeviceAllocation<cuda::finish::Refinement>
                nextFinishRefinements;
        DeviceAllocation<CudaCandidatePhysicsState>
                nextFinishCheckpointStates;
        DeviceAllocation<std::uint32_t> nextFinishCheckpointTicks;
        DeviceAllocation<std::uint32_t> nextRandomStateWords;
        DeviceAllocation<CudaSearchInputEvent> nextCandidateEvents;
        DeviceAllocation<std::int32_t> nextCandidateInputValues;
        DeviceAllocation<CudaSearchInputEvent> nextTemporaryEvents;
        DeviceAllocation<CudaSearchInputEvent> nextPassBaselineEvents;
        DeviceAllocation<std::uint32_t> nextEligibleIndices;
        DeviceAllocation<std::uint32_t> nextSparseReferences;
        DeviceAllocation<std::uint32_t> nextSparseSnapshotReferences;
        DeviceAllocation<CudaSearchInputEvent> nextSparseEdits;
        DeviceAllocation<CudaSearchInputEvent> nextSparseScratchEdits;
        DeviceAllocation<std::byte> nextEditBacking;
        const std::size_t nextEditStorageBytes =
                compactEditPipeline
                ? EditStorageSize(
                          candidateCount, editCapacity,
                          eraseCapacity, packedEditStorage)
                : 0u;
        DeviceAllocation<std::uint32_t> nextEventCounts;
        DeviceAllocation<std::uint32_t> nextMutationCounts;
        DeviceAllocation<DeviceCandidateStatus> nextStatuses;
        DeviceAllocation<bool> nextActiveCandidates;
        DeviceAllocation<std::uint64_t> nextCandidateSimulationKeys;
        DeviceAllocation<std::uint64_t> nextSortedCandidateSimulationKeys;
        DeviceAllocation<std::uint32_t> nextCandidateSlots;
        DeviceAllocation<std::uint32_t> nextSortedCandidateSlots;
        DeviceAllocation<std::uint64_t> nextCandidateInputHashes;
        DeviceAllocation<std::uint64_t> nextSortedCandidateInputHashes;
        DeviceAllocation<std::uint32_t> nextCandidateRepresentativeSlots;
        DeviceAllocation<std::byte> nextCandidateSortTemporary;
        DeviceAllocation<DeviceSample> nextPrefixBestSamples;
        DeviceAllocation<std::byte> nextWinnerScanTemporary;
        DeviceAllocation<cuda::collision::CudaCollisionSearchTile>
                nextCollisionScratch;
        DeviceAllocation<cuda::collision::CudaCollisionSearchTile>
                nextShapeCollisionScratch;
        DeviceAllocation<GmIso4> nextShapeWorldScratch;
        DeviceAllocation<GmBoxAligned> nextMovingBoundsScratch;
        DeviceAllocation<cuda::collision::CudaCollisionSurfaceHit>
                nextSurfaceHitScratch;
        DeviceAllocation<cuda::collision::CudaCollisionMeshRange>
                nextMeshRangeScratch;
        DeviceAllocation<std::uint32_t> nextMeshCellScratch;
        DeviceAllocation<std::uint16_t> nextResponseOrderScratch;
        DeviceAllocation<double> nextClosestTargetDistanceSquaredByBlock;
        DeviceAllocation<std::byte> nextHotPathScratch;
        if (!nextCandidateBestSamples.Allocate(winnerSlots) ||
            !nextFinishRefinements.Allocate(
                    configuration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !nextFinishCheckpointStates.Allocate(
                    configuration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !nextFinishCheckpointTicks.Allocate(
                    configuration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !nextRandomStateWords.Allocate(candidates * 624u) ||
            !nextCandidateEvents.Allocate(
                    materializesCandidateEvents ? eventSlots : 0u) ||
            !nextCandidateInputValues.Allocate(
                    compactRandomSteeringPipeline
                            ? compactValueSlots
                            : 0u) ||
            !nextTemporaryEvents.Allocate(
                    needsTemporaryEvents ? eventSlots : 0u) ||
            !nextPassBaselineEvents.Allocate(
                    needsPassBaselineEvents ? eventSlots : 0u) ||
            !nextEligibleIndices.Allocate(
                    needsEligibleIndices ? eventSlots : 0u) ||
            !nextSparseReferences.Allocate(
                    sparseMutationPipeline ? eventSlots : 0u) ||
            !nextSparseSnapshotReferences.Allocate(
                    sparseMutationPipeline ? eventSlots : 0u) ||
            !nextSparseEdits.Allocate(
                    sparseMutationPipeline ? eventSlots : 0u) ||
            !nextSparseScratchEdits.Allocate(
                    sparseMutationPipeline ? eventSlots : 0u) ||
            !nextEditBacking.Allocate(
                    editStorageAliasesTemporary
                            ? 0u : nextEditStorageBytes) ||
            !nextEventCounts.Allocate(candidates) ||
            !nextMutationCounts.Allocate(candidates) ||
            !nextStatuses.Allocate(candidates) ||
            !nextActiveCandidates.Allocate(candidates) ||
            !nextCandidateSimulationKeys.Allocate(prefixCandidateSlots) ||
            !nextSortedCandidateSimulationKeys.Allocate(
                    prefixCandidateSlots) ||
            !nextCandidateSlots.Allocate(prefixCandidateSlots) ||
            !nextSortedCandidateSlots.Allocate(prefixCandidateSlots) ||
            !nextCandidateInputHashes.Allocate(
                    deduplicationCandidateSlots) ||
            !nextSortedCandidateInputHashes.Allocate(
                    deduplicationCandidateSlots) ||
            !nextCandidateRepresentativeSlots.Allocate(
                    deduplicationCandidateSlots) ||
            !nextPrefixBestSamples.Allocate(winnerSlots) ||
            !nextCollisionScratch.Allocate(collisionTileSlots) ||
            !nextShapeCollisionScratch.Allocate(
                    shapeCollisionTileSlots) ||
            !nextShapeWorldScratch.Allocate(shapeQuerySlots) ||
            !nextMovingBoundsScratch.Allocate(shapeQuerySlots) ||
            !nextSurfaceHitScratch.Allocate(surfaceHitSlots) ||
            !nextMeshRangeScratch.Allocate(meshRangeSlots) ||
            !nextMeshCellScratch.Allocate(meshCellSlots) ||
            !nextResponseOrderScratch.Allocate(collisionSlots) ||
            !nextClosestTargetDistanceSquaredByBlock.Allocate(
                    summaryBlockCount) ||
            !nextHotPathScratch.Allocate(
                    configuration.collectHotPathMetrics
                            ? HotPathScratchBytes(candidateCount)
                            : 0u)) {
            static_cast<void>(cudaGetLastError());
            if (diagnostic != nullptr) {
                *diagnostic =
                        "CUDA calibration could not reserve a larger real batch";
            }
            return false;
        }
        std::size_t winnerScanBytes = 0u;
        const cudaError_t winnerScanError =
                cub::DeviceScan::InclusiveScan(
                        nullptr, winnerScanBytes,
                        nextCandidateBestSamples.Get(),
                        nextPrefixBestSamples.Get(),
                        BetterSample{
                                MaximizesScore(
                                        configuration.evaluator.kind)},
                        winnerSlots);
        if (winnerScanError != cudaSuccess ||
            !nextWinnerScanTemporary.Allocate(winnerScanBytes)) {
            static_cast<void>(cudaGetLastError());
            if (diagnostic != nullptr) {
                *diagnostic = winnerScanError != cudaSuccess
                        ? CudaFailure(
                                  "sizing calibrated CUDA winner scan",
                                  winnerScanError)
                        : "CUDA calibration winner scan allocation failed";
            }
            return false;
        }

        std::size_t candidateSortBytes = 0u;
        cudaError_t candidateSortError = cudaSuccess;
        std::size_t candidateHashSortBytes = 0u;
        if (baselinePrefixReuseEligible) {
            candidateSortError = cub::DeviceRadixSort::SortPairs(
                    nullptr, candidateSortBytes,
                    nextCandidateSimulationKeys.Get(),
                    nextSortedCandidateSimulationKeys.Get(),
                    nextCandidateSlots.Get(),
                    nextSortedCandidateSlots.Get(),
                    candidates);
        }
        if (candidateSortError == cudaSuccess &&
            DeduplicationStorageEligible(candidateCount)) {
            candidateSortError = cub::DeviceRadixSort::SortPairs(
                    nullptr, candidateHashSortBytes,
                    nextCandidateInputHashes.Get(),
                    nextSortedCandidateInputHashes.Get(),
                    nextCandidateSlots.Get(),
                    nextSortedCandidateSlots.Get(),
                    candidates);
            if (candidateHashSortBytes > candidateSortBytes) {
                candidateSortBytes = candidateHashSortBytes;
            }
        }
        if (candidateSortError != cudaSuccess ||
            !nextCandidateSortTemporary.Allocate(
                    candidateSortBytes)) {
            static_cast<void>(cudaGetLastError());
            if (diagnostic != nullptr) {
                *diagnostic = candidateSortError != cudaSuccess
                        ? CudaFailure(
                                  "sizing calibrated CUDA candidate "
                                  "prefix sort",
                                  candidateSortError)
                        : "CUDA calibration candidate prefix sort "
                          "allocation failed";
            }
            return false;
        }
        candidateBestSamples = std::move(nextCandidateBestSamples);
        finishRefinements = std::move(nextFinishRefinements);
        finishCheckpointStates =
                std::move(nextFinishCheckpointStates);
        finishCheckpointTicks =
                std::move(nextFinishCheckpointTicks);
        randomStateWords = std::move(nextRandomStateWords);
        candidateEvents = std::move(nextCandidateEvents);
        candidateInputValues =
                std::move(nextCandidateInputValues);
        temporaryEvents = std::move(nextTemporaryEvents);
        passBaselineEvents = std::move(nextPassBaselineEvents);
        eligibleIndices = std::move(nextEligibleIndices);
        sparseReferences = std::move(nextSparseReferences);
        sparseSnapshotReferences =
                std::move(nextSparseSnapshotReferences);
        sparseEdits = std::move(nextSparseEdits);
        sparseScratchEdits = std::move(nextSparseScratchEdits);
        editBacking = std::move(nextEditBacking);
        editStorageBytes = nextEditStorageBytes;
        eventCounts = std::move(nextEventCounts);
        mutationCounts = std::move(nextMutationCounts);
        statuses = std::move(nextStatuses);
        activeCandidates = std::move(nextActiveCandidates);
        candidateSimulationKeys =
                std::move(nextCandidateSimulationKeys);
        sortedCandidateSimulationKeys =
                std::move(nextSortedCandidateSimulationKeys);
        candidateSlots = std::move(nextCandidateSlots);
        sortedCandidateSlots =
                std::move(nextSortedCandidateSlots);
        candidateInputHashes =
                std::move(nextCandidateInputHashes);
        sortedCandidateInputHashes =
                std::move(nextSortedCandidateInputHashes);
        candidateRepresentativeSlots =
                std::move(nextCandidateRepresentativeSlots);
        candidateSortTemporary =
                std::move(nextCandidateSortTemporary);
        prefixBestSamples = std::move(nextPrefixBestSamples);
        winnerScanTemporary =
                std::move(nextWinnerScanTemporary);
        collisionScratch = std::move(nextCollisionScratch);
        shapeCollisionScratch =
                std::move(nextShapeCollisionScratch);
        shapeWorldScratch = std::move(nextShapeWorldScratch);
        movingBoundsScratch =
                std::move(nextMovingBoundsScratch);
        surfaceHitScratch = std::move(nextSurfaceHitScratch);
        meshRangeScratch = std::move(nextMeshRangeScratch);
        meshCellScratch = std::move(nextMeshCellScratch);
        responseOrderScratch =
                std::move(nextResponseOrderScratch);
        closestTargetDistanceSquaredByBlock =
                std::move(nextClosestTargetDistanceSquaredByBlock);
        hotPathScratch = std::move(nextHotPathScratch);
        configuration.maximumBatchSize = candidateCount;
        UpdateResidentBytes();
        if (diagnostic != nullptr) {
            diagnostic->clear();
        }
        return true;
    }

    CudaSearchBatchExecution ExecuteImpl(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool baseline,
            const std::function<bool()> &cancellationRequested) noexcept {
        CudaSearchBatchExecution result;
        result.firstCandidateId = firstCandidateId;
        result.candidateCount = candidateCount;
        result.residentDeviceBytes = residentBytes;
        const std::size_t aliasedEditBytes =
                editStorageAliasesTemporary
                ? std::min(
                          editStorageBytes,
                          temporaryEvents.Bytes())
                : 0u;
        result.candidateInputDeviceBytes =
                candidateInputValues.Bytes() +
                (sparseMutationPipeline
                         ? sparseReferences.Bytes() +
                                   sparseEdits.Bytes()
                         : compactEditPipeline
                         ? editStorageBytes
                         : candidateEvents.Bytes());
        result.mutationScratchDeviceBytes =
                randomStateWords.Bytes() +
                (compactEditPipeline
                         ? candidateEvents.Bytes() : 0u) +
                temporaryEvents.Bytes() -
                aliasedEditBytes +
                passBaselineEvents.Bytes() +
                eligibleIndices.Bytes() +
                sharedEligibleIndices.Bytes() +
                sparseSnapshotReferences.Bytes() +
                sparseScratchEdits.Bytes() +
                candidateSimulationKeys.Bytes() +
                sortedCandidateSimulationKeys.Bytes() +
                candidateSlots.Bytes() +
                sortedCandidateSlots.Bytes() +
                candidateInputHashes.Bytes() +
                sortedCandidateInputHashes.Bytes() +
                candidateRepresentativeSlots.Bytes() +
                candidateSortTemporary.Bytes();
        result.baselinePrefixDeviceBytes =
                baselinePrefixStates.Bytes() +
                baselinePrefixBestSamples.Bytes() +
                baselinePrefixClosestTargetDistanceSquared.Bytes();
        result.candidatePrefixDeviceBytes =
                candidateSimulationKeys.Bytes() +
                sortedCandidateSimulationKeys.Bytes() +
                candidateSlots.Bytes() +
                sortedCandidateSlots.Bytes() +
                candidateSortTemporary.Bytes();
        result.candidateDeduplicationDeviceBytes =
                candidateInputHashes.Bytes() +
                sortedCandidateInputHashes.Bytes() +
                candidateRepresentativeSlots.Bytes();
        result.mutationDeviceBytes =
                baselineInputs.Bytes() +
                modifiers.Bytes() +
                smoothWeights.Bytes() +
                compactInputIndices.Bytes() +
                compactInputOffsets.Bytes() +
                result.candidateInputDeviceBytes +
                result.mutationScratchDeviceBytes +
                eventCounts.Bytes() +
                mutationCounts.Bytes() +
                statuses.Bytes() +
                activeCandidates.Bytes();
        result.winnerSelectionDeviceBytes = winnerSelectionBytes;
        if ((!baseline && !baselineEvaluated) ||
            candidateCount == 0u ||
            candidateCount > configuration.maximumBatchSize) {
            result.status = CudaSearchStatus::InvalidArgument;
            result.diagnostic = !baselineEvaluated && !baseline
                    ? "CUDA baseline must be evaluated before mutation batches"
                    : "invalid CUDA search batch size";
            return result;
        }
        const std::size_t winnerCount =
                static_cast<std::size_t>(candidateCount) + 1u;
        bool cancelled = false;
        if (cancellationRequested) {
            try {
                cancelled = cancellationRequested();
            } catch (...) {
                cancelled = true;
            }
        }
        *cancellation.Host() = cancelled ? 1u : 0u;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        cudaError_t error = cudaSuccess;

        Event started;
        Event winnerInitialized;
        Event mutationsGenerated;
        Event simulationFinished;
        Event finishRefined;
        Event winnerReduced;
        Event winnerStateCaptured;
        Event finished;
        if (!started.Valid() || !winnerInitialized.Valid() ||
            !mutationsGenerated.Valid() ||
            !simulationFinished.Valid() ||
            !finishRefined.Valid() ||
            !winnerReduced.Valid() ||
            !winnerStateCaptured.Valid() || !finished.Valid()) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic = "CUDA search event creation failed";
            return result;
        }
        cudaEventRecord(started.Get());
        constexpr std::uint32_t blockSize = BatchSummaryBlockSize;
        const std::uint32_t candidateBlocks =
                (candidateCount - 1u) / blockSize + 1u;
        const std::uint32_t simulationBlocks =
                (candidateCount - 1u) / SimulationBlockSize + 1u;
        InitializeSearchBatchSummaryKernel<<<1u, BatchSummaryBlockSize>>>(
                summary.Get(),
                ClosestTargetDistanceScratch(),
                simulationBlocks);
        SeedCandidateBestSamplesKernel<<<1u, 1u>>>(
                candidateBestSamples.Get(), globalBestSample.Get());
        cudaEventRecord(winnerInitialized.Get());
        GenerateSearchCandidatesKernel<<<candidateBlocks, blockSize>>>(
                baselineInputs.Get(),
                static_cast<std::uint32_t>(
                        configuration.baselineInputs.size()),
                static_cast<std::uint32_t>(
                        immutableInputTail.size()),
                modifiers.Get(),
                static_cast<std::uint32_t>(
                        configuration.modifiers.size()),
                smoothWeights.Get(),
                mutableBoundaryControls.Get(),
                configuration.tickDurationMs,
                firstCandidateId,
                candidateCount,
                baseline,
                configuration.useLegacyMutationPipelineForTesting,
                baselineInputsCanonical,
                compactRandomSteeringPipeline,
                compactEditPipeline,
                sparseMutationPipeline,
                directDeletionPipeline,
                directExistingEventPipeline,
                static_cast<std::uint32_t>(
                        configuration.maximumEventCount),
                compactInputIndices.Get(),
                compactInputCount,
                candidateInputValues.Get(),
                candidateBestSamples.Get(),
                randomStateWords.Get(),
                candidateEvents.Get(),
                temporaryEvents.Get(),
                passBaselineEvents.Get(),
                eligibleIndices.Get(),
                sharedEligibleIndices.Get(),
                sharedEligibleCount,
                CandidateEdits(candidateCount),
                SparseCandidateEvents(candidateCount),
                eventCounts.Get(),
                mutationCounts.Get(),
                statuses.Get(),
                activeCandidates.Get(),
                cancellation.Get());
        if (compactEditPipeline && materializesCandidateEvents) {
            EncodeSearchCandidateEditsKernel
                    <<<candidateBlocks, blockSize>>>(
                    baselineInputs.Get(),
                    static_cast<std::uint32_t>(
                            configuration.baselineInputs.size()),
                    candidateEvents.Get(),
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    eventCounts.Get(),
                    eligibleIndices.Get(),
                    CandidateEdits(candidateCount),
                    statuses.Get(),
                    activeCandidates.Get(),
                    candidateCount);
        }
        const auto residentWaves =
                [&](const SimulationKernelMetrics &metrics) {
            const std::uint64_t residentBlocks =
                    static_cast<std::uint64_t>(
                            metrics.activeBlocksPerMultiprocessor) *
                    multiprocessorCount;
            return residentBlocks == 0u
                    ? UINT64_MAX
                    : (simulationBlocks + residentBlocks - 1u) /
                              residentBlocks;
        };
        std::uint32_t selectedMinimumBlocks =
                ThroughputKernelMinimumBlocksPerSm;
        const SimulationKernelMetrics *simulationMetrics =
                &throughputKernelMetrics;
        std::uint64_t selectedWaves =
                residentWaves(*simulationMetrics);
        const auto considerKernel =
                [&](std::uint32_t minimumBlocks,
                    const SimulationKernelMetrics &metrics) {
            const std::uint64_t waves = residentWaves(metrics);
            if (waves < selectedWaves) {
                selectedMinimumBlocks = minimumBlocks;
                simulationMetrics = &metrics;
                selectedWaves = waves;
            }
        };
        considerKernel(
                TailKernelMinimumBlocksPerSm,
                tailKernelMetrics);
        considerKernel(
                DenseTailKernelMinimumBlocksPerSm,
                denseTailKernelMetrics);
        const std::uint32_t heuristicMinimumBlocks =
                selectedMinimumBlocks;
        SimulationKernelTuning *simulationTuning = nullptr;
        bool simulationTuningExploration = false;
        bool simulationTuningRevalidation = false;
        constexpr std::uint32_t
                SimulationTuningRevalidationInterval = 64u;
        const std::uint64_t minimumTuningCandidateCount =
                static_cast<std::uint64_t>(multiprocessorCount) *
                SimulationBlockSize * 4u;
        if (!baseline &&
            !configuration.collectHotPathMetrics &&
            configuration.
                    simulationMinimumBlocksPerMultiprocessorForTesting ==
                    0u &&
            multiprocessorCount != 0u &&
            candidateCount >= minimumTuningCandidateCount) {
            const auto existing = std::find_if(
                    simulationKernelTunings.begin(),
                    simulationKernelTunings.end(),
                    [&](const SimulationKernelTuning &tuning) {
                        return tuning.candidateCount == candidateCount;
                    });
            if (existing == simulationKernelTunings.end()) {
                simulationKernelTunings.push_back(
                        SimulationKernelTuning{candidateCount});
                simulationTuning = &simulationKernelTunings.back();
            } else {
                simulationTuning = &*existing;
            }
            const auto kernelIndex = [](std::uint32_t minimumBlocks) {
                return minimumBlocks ==
                                       ThroughputKernelMinimumBlocksPerSm
                        ? 0u
                        : minimumBlocks ==
                                          TailKernelMinimumBlocksPerSm
                        ? 1u
                        : 2u;
            };
            const auto selectKernel =
                    [&](std::uint32_t minimumBlocks,
                        const SimulationKernelMetrics &metrics) {
                selectedMinimumBlocks = minimumBlocks;
                simulationMetrics = &metrics;
                selectedWaves = residentWaves(metrics);
            };
            const auto selectUnmeasured =
                    [&](std::uint32_t minimumBlocks,
                        const SimulationKernelMetrics &metrics) {
                if (simulationTuning->sampleCounts[
                            kernelIndex(minimumBlocks)] != 0u) {
                    return false;
                }
                selectKernel(minimumBlocks, metrics);
                return true;
            };
            bool exploring =
                    heuristicMinimumBlocks ==
                            ThroughputKernelMinimumBlocksPerSm
                    ? selectUnmeasured(
                              ThroughputKernelMinimumBlocksPerSm,
                              throughputKernelMetrics)
                    : heuristicMinimumBlocks ==
                                      TailKernelMinimumBlocksPerSm
                    ? selectUnmeasured(
                              TailKernelMinimumBlocksPerSm,
                              tailKernelMetrics)
                    : selectUnmeasured(
                              DenseTailKernelMinimumBlocksPerSm,
                              denseTailKernelMetrics);
            if (!exploring) {
                exploring = selectUnmeasured(
                        ThroughputKernelMinimumBlocksPerSm,
                        throughputKernelMetrics);
            }
            if (!exploring) {
                exploring = selectUnmeasured(
                        TailKernelMinimumBlocksPerSm,
                        tailKernelMetrics);
            }
            if (!exploring) {
                exploring = selectUnmeasured(
                        DenseTailKernelMinimumBlocksPerSm,
                        denseTailKernelMetrics);
            }
            simulationTuningExploration = exploring;
            if (!exploring) {
                const std::size_t heuristicIndex =
                        kernelIndex(heuristicMinimumBlocks);
                std::size_t bestIndex = heuristicIndex;
                for (std::size_t index = 0u;
                     index < simulationTuning->milliseconds.size();
                     ++index) {
                    if (simulationTuning->milliseconds[index] <
                        simulationTuning->milliseconds[bestIndex]) {
                        bestIndex = index;
                    }
                }
                // Require a material measured win before overriding the
                // occupancy-based fallback. This prevents one noisy launch
                // from changing dispatch for otherwise equivalent kernels.
                if (simulationTuning->milliseconds[bestIndex] <
                    simulationTuning->milliseconds[heuristicIndex] *
                            0.97f) {
                    if (bestIndex == 0u) {
                        selectKernel(
                                ThroughputKernelMinimumBlocksPerSm,
                                throughputKernelMetrics);
                    } else if (bestIndex == 1u) {
                        selectKernel(
                                TailKernelMinimumBlocksPerSm,
                                tailKernelMetrics);
                    } else {
                        selectKernel(
                                DenseTailKernelMinimumBlocksPerSm,
                                denseTailKernelMetrics);
                    }
                }
                const std::size_t selectedIndex =
                        kernelIndex(selectedMinimumBlocks);
                if (simulationTuning->
                            successfulBatchesSinceRevalidation >=
                    SimulationTuningRevalidationInterval) {
                    for (std::size_t offset = 0u;
                         offset <
                         simulationTuning->milliseconds.size();
                         ++offset) {
                        const std::size_t revalidationIndex =
                                (simulationTuning->
                                 nextRevalidationIndex +
                                 offset) %
                                simulationTuning->milliseconds.size();
                        if (revalidationIndex == selectedIndex) {
                            continue;
                        }
                        if (revalidationIndex == 0u) {
                            selectKernel(
                                    ThroughputKernelMinimumBlocksPerSm,
                                    throughputKernelMetrics);
                        } else if (revalidationIndex == 1u) {
                            selectKernel(
                                    TailKernelMinimumBlocksPerSm,
                                    tailKernelMetrics);
                        } else {
                            selectKernel(
                                    DenseTailKernelMinimumBlocksPerSm,
                                    denseTailKernelMetrics);
                        }
                        simulationTuningRevalidation = true;
                        break;
                    }
                }
            }
        }
        const auto forceKernel =
                [&](std::uint32_t minimumBlocks,
                    const SimulationKernelMetrics &metrics) {
            if (configuration.
                        simulationMinimumBlocksPerMultiprocessorForTesting ==
                minimumBlocks) {
                selectedMinimumBlocks = minimumBlocks;
                simulationMetrics = &metrics;
                selectedWaves = residentWaves(metrics);
            }
        };
        forceKernel(
                ThroughputKernelMinimumBlocksPerSm,
                throughputKernelMetrics);
        forceKernel(
                TailKernelMinimumBlocksPerSm,
                tailKernelMetrics);
        forceKernel(
                DenseTailKernelMinimumBlocksPerSm,
                denseTailKernelMetrics);
        const std::uint32_t conditionInstructionCount =
                configuration.condition
                ? static_cast<std::uint32_t>(
                          configuration.condition->instructions.size())
                : 0u;
        const bool useBaselinePrefixes =
                !baseline && baselinePrefixesValid;
        const bool deduplicateCandidateInputs =
                useBaselinePrefixes &&
                DeduplicationStorageEligible(candidateCount);
        result.baselinePrefixReuseActive = useBaselinePrefixes;
        result.candidateDeduplicationActive =
                deduplicateCandidateInputs;
        std::uint32_t deduplicationReplicaLimit = 1u;
        if (deduplicateCandidateInputs) {
            const std::uint64_t residentWorkers =
                    static_cast<std::uint64_t>(
                            (simulationMetrics->
                                             activeBlocksPerMultiprocessor *
                                     3u +
                             7u) /
                                    8u) *
                    multiprocessorCount *
                    SimulationBlockSize;
            const std::uint64_t replicas =
                    (residentWorkers +
                     prefixReusePlan.lowEntropyChoiceCount - 1u) /
                    prefixReusePlan.lowEntropyChoiceCount;
            deduplicationReplicaLimit =
                    static_cast<std::uint32_t>(
                            replicas == 0u
                            ? 1u
                            : replicas > UINT32_MAX
                            ? UINT32_MAX
                            : replicas);
            if (configuration.
                        deduplicationReplicaLimitForTesting != 0u) {
                deduplicationReplicaLimit = configuration.
                        deduplicationReplicaLimitForTesting;
            }
        }
        const std::uint32_t *simulationCandidateSlots = nullptr;
        const std::uint64_t *simulationKeys = nullptr;
        if (useBaselinePrefixes) {
            ComputeSearchCandidateSimulationKeysKernel
                    <<<candidateBlocks, blockSize>>>(
                    baselineInputs.Get(),
                    static_cast<std::uint32_t>(
                            configuration.baselineInputs.size()),
                    candidateEvents.Get(),
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    candidateInputValues.Get(),
                    compactInputOffsets.Get(),
                    compactRandomSteeringPipeline,
                    compactEditPipeline,
                    sparseMutationPipeline,
                    CandidateEdits(candidateCount),
                    SparseCandidateEvents(candidateCount),
                    eventCounts.Get(), activeCandidates.Get(),
                    mutableBoundaryControls.Get(),
                    mutableFromTimeMs, configuration.branchTimeMs,
                    configuration.tickDurationMs,
                    timelineTickCount,
                    configuration.sortCandidatesByLocality,
                    candidateSimulationKeys.Get(), candidateSlots.Get(),
                    candidateCount);
            std::size_t candidateSortBytes =
                    candidateSortTemporary.Bytes();
            if (deduplicateCandidateInputs) {
                HashSearchCandidateInputsKernel
                        <<<candidateBlocks, blockSize>>>(
                        baselineInputs.Get(),
                        static_cast<std::uint32_t>(
                                configuration.baselineInputs.size()),
                        candidateEvents.Get(),
                        static_cast<std::uint32_t>(
                                configuration.maximumEventCount),
                        candidateInputValues.Get(),
                        compactInputOffsets.Get(),
                        compactRandomSteeringPipeline,
                        compactEditPipeline,
                        sparseMutationPipeline,
                        CandidateEdits(candidateCount),
                        SparseCandidateEvents(candidateCount),
                        eventCounts.Get(), activeCandidates.Get(),
                        candidateInputHashes.Get(), candidateCount);
                error = cub::DeviceRadixSort::SortPairs(
                        candidateSortTemporary.Get(),
                        candidateSortBytes,
                        candidateInputHashes.Get(),
                        sortedCandidateInputHashes.Get(),
                        candidateSlots.Get(),
                        sortedCandidateSlots.Get(),
                        candidateCount);
                if (error == cudaSuccess) {
                    MarkUniqueSearchCandidateInputsKernel
                            <<<candidateBlocks, blockSize>>>(
                            sortedCandidateInputHashes.Get(),
                            sortedCandidateSlots.Get(),
                            candidateSimulationKeys.Get(),
                            baselineInputs.Get(),
                            static_cast<std::uint32_t>(
                                    configuration.baselineInputs.size()),
                            candidateEvents.Get(),
                            static_cast<std::uint32_t>(
                                    configuration.maximumEventCount),
                            candidateInputValues.Get(),
                            compactInputOffsets.Get(),
                            compactRandomSteeringPipeline,
                            compactEditPipeline,
                            sparseMutationPipeline,
                            CandidateEdits(candidateCount),
                            SparseCandidateEvents(candidateCount),
                            eventCounts.Get(), activeCandidates.Get(),
                            sortedCandidateSimulationKeys.Get(),
                            candidateSlots.Get(),
                            candidateRepresentativeSlots.Get(),
                            deduplicationReplicaLimit,
                            candidateCount);
                    candidateSortBytes =
                            candidateSortTemporary.Bytes();
                    error = cub::DeviceRadixSort::SortPairs(
                            candidateSortTemporary.Get(),
                            candidateSortBytes,
                            sortedCandidateSimulationKeys.Get(),
                            candidateSimulationKeys.Get(),
                            candidateSlots.Get(),
                            sortedCandidateSlots.Get(),
                            candidateCount);
                }
                simulationCandidateSlots =
                        sortedCandidateSlots.Get();
                simulationKeys = candidateSimulationKeys.Get();
            } else {
                error = cub::DeviceRadixSort::SortPairs(
                        candidateSortTemporary.Get(),
                        candidateSortBytes,
                        candidateSimulationKeys.Get(),
                        sortedCandidateSimulationKeys.Get(),
                        candidateSlots.Get(),
                        sortedCandidateSlots.Get(),
                        candidateCount);
                simulationCandidateSlots =
                        sortedCandidateSlots.Get();
                simulationKeys = sortedCandidateSimulationKeys.Get();
            }
            if (error != cudaSuccess) {
                result.status = CudaSearchStatus::DeviceFailure;
                result.diagnostic = CudaFailure(
                        "sorting CUDA candidate prefix work", error);
                return result;
            }
        }
        cudaEventRecord(mutationsGenerated.Get());
        const double lastImprovementTimeSeconds =
                configuration.condition
                ? configuration.condition->lastImprovementTimeSeconds
                : 0.0;
        const double lastRestartTimeSeconds =
                configuration.condition
                ? configuration.condition->lastRestartTimeSeconds
                : 0.0;
        const double currentTimeSeconds =
                std::chrono::duration<double>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
        if (specializedModule &&
            !configuration.collectHotPathMetrics) {
            const CUresult simulationLaunch = LaunchDriverKernel(
                    specializedModule->Kernel(
                            selectedMinimumBlocks),
                    simulationBlocks,
                    configuration.deviceScene,
                    configuration.deviceStaticConfiguration,
                    branchState.Get(),
                    mutableBoundaryControls.Get(),
                    baselineTicks.Get(),
                    timelineTickCount,
                    evaluator.Get(),
                    condition.Get(),
                    conditionInstructionCount,
                    lastImprovementTimeSeconds,
                    lastRestartTimeSeconds,
                    currentTimeSeconds,
                    configuration.tickDurationMs,
                    configuration.prestartDurationMs,
                    configuration.branchTimeMs,
                    mutableFromTimeMs,
                    configuration.evaluationStartTimeMs,
                    evaluationTickCount,
                    firstCandidateId,
                     candidateCount,
                     baseline,
                     baselinePrefixStates.Get(),
                     baselinePrefixBestSamples.Get(),
                     baselinePrefixClosestTargetDistanceSquared.Get(),
                     simulationCandidateSlots,
                    simulationKeys,
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    candidateBestSamples.Get(),
                    ClosestTargetDistanceScratch(),
                    finishCheckpointStates.Get(),
                    finishCheckpointTicks.Get(),
                    baselineInputs.Get(),
                    static_cast<std::uint32_t>(
                            configuration.baselineInputs.size()),
                    candidateEvents.Get(),
                    candidateInputValues.Get(),
                    compactInputOffsets.Get(),
                    compactInputCount,
                    compactRandomSteeringPipeline,
                    compactEditPipeline,
                    sparseMutationPipeline,
                    CandidateEdits(candidateCount),
                    SparseCandidateEvents(candidateCount),
                    eventCounts.Get(),
                    statuses.Get(),
                    activeCandidates.Get(),
                    collisionScratch.Get(),
                    shapeCollisionScratch.Get(),
                    shapeWorldScratch.Get(),
                    movingBoundsScratch.Get(),
                    surfaceHitScratch.Get(),
                    meshRangeScratch.Get(),
                    meshCellScratch.Get(),
                    responseOrderScratch.Get(),
                    static_cast<std::uint32_t>(
                            configuration.maximumBatchSize),
                    collisionShapeCount,
                    cancellation.Get());
            if (simulationLaunch != CUDA_SUCCESS) {
                const char *message = nullptr;
                cuGetErrorString(simulationLaunch, &message);
                result.status = CudaSearchStatus::DeviceFailure;
                result.diagnostic =
                        "launching specialized CUDA simulation kernel: " +
                        std::string(
                                message == nullptr ? "unknown" : message);
                return result;
            }
        } else {
            const void *simulationKernel = nullptr;
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
            if (selectedMinimumBlocks ==
                DenseTailKernelMinimumBlocksPerSm) {
                simulationKernel = SelectedSimulationKernel<
                        DenseTailKernelMinimumBlocksPerSm>();
            } else if (selectedMinimumBlocks ==
                       TailKernelMinimumBlocksPerSm) {
                simulationKernel = SelectedSimulationKernel<
                        TailKernelMinimumBlocksPerSm>();
            } else {
                simulationKernel = SelectedSimulationKernel<
                        ThroughputKernelMinimumBlocksPerSm>();
            }
#else
            if (selectedMinimumBlocks ==
                DenseTailKernelMinimumBlocksPerSm) {
                simulationKernel = configuration.collectHotPathMetrics
                        ? SelectedSimulationKernel<
                                  DenseTailKernelMinimumBlocksPerSm,
                                  true>()
                        : SelectedSimulationKernel<
                                  DenseTailKernelMinimumBlocksPerSm>();
            } else if (selectedMinimumBlocks ==
                       TailKernelMinimumBlocksPerSm) {
                simulationKernel = configuration.collectHotPathMetrics
                        ? SelectedSimulationKernel<
                                  TailKernelMinimumBlocksPerSm,
                                  true>()
                        : SelectedSimulationKernel<
                                  TailKernelMinimumBlocksPerSm>();
            } else {
                simulationKernel = configuration.collectHotPathMetrics
                        ? SelectedSimulationKernel<
                                  ThroughputKernelMinimumBlocksPerSm,
                                  true>()
                        : SelectedSimulationKernel<
                                  ThroughputKernelMinimumBlocksPerSm>();
            }
#endif
            const cudaError_t simulationLaunch = LaunchRuntimeKernel(
                    simulationKernel,
                    simulationBlocks,
                    configuration.deviceScene,
                    configuration.deviceStaticConfiguration,
                    branchState.Get(),
                    mutableBoundaryControls.Get(),
                    baselineTicks.Get(),
                    timelineTickCount,
                    evaluator.Get(),
                    condition.Get(),
                    conditionInstructionCount,
                    lastImprovementTimeSeconds,
                    lastRestartTimeSeconds,
                    currentTimeSeconds,
                    configuration.tickDurationMs,
                    configuration.prestartDurationMs,
                    configuration.branchTimeMs,
                    mutableFromTimeMs,
                    configuration.evaluationStartTimeMs,
                    evaluationTickCount,
                    firstCandidateId,
                     candidateCount,
                     baseline,
                     baselinePrefixStates.Get(),
                     baselinePrefixBestSamples.Get(),
                     baselinePrefixClosestTargetDistanceSquared.Get(),
                     simulationCandidateSlots,
                    simulationKeys,
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    candidateBestSamples.Get(),
                    ClosestTargetDistanceScratch(),
                    finishCheckpointStates.Get(),
                    finishCheckpointTicks.Get(),
                    baselineInputs.Get(),
                    static_cast<std::uint32_t>(
                            configuration.baselineInputs.size()),
                    candidateEvents.Get(),
                    candidateInputValues.Get(),
                    compactInputOffsets.Get(),
                    compactInputCount,
                    compactRandomSteeringPipeline,
                    compactEditPipeline,
                    sparseMutationPipeline,
                    CandidateEdits(candidateCount),
                    SparseCandidateEvents(candidateCount),
                    eventCounts.Get(),
                    statuses.Get(),
                    activeCandidates.Get(),
                    collisionScratch.Get(),
                    shapeCollisionScratch.Get(),
                    shapeWorldScratch.Get(),
                    movingBoundsScratch.Get(),
                    surfaceHitScratch.Get(),
                    meshRangeScratch.Get(),
                    meshCellScratch.Get(),
                    responseOrderScratch.Get(),
                    static_cast<std::uint32_t>(
                            configuration.maximumBatchSize),
                    collisionShapeCount,
                    cancellation.Get());
            if (simulationLaunch != cudaSuccess) {
                result.status = CudaSearchStatus::DeviceFailure;
                result.diagnostic = CudaFailure(
                        "launching CUDA simulation kernel",
                        simulationLaunch);
                return result;
            }
            if (configuration.collectHotPathMetrics) {
                result.hotPath.forcedRuntimeGenericKernel = true;
            }
        }
        if (deduplicateCandidateInputs) {
            ExpandDeduplicatedSearchSamplesKernel
                    <<<candidateBlocks, blockSize>>>(
                    candidateBestSamples.Get(), statuses.Get(),
                    candidateRepresentativeSlots.Get(),
                    firstCandidateId, evaluationTickCount,
                    candidateCount);
        }
        cudaEventRecord(simulationFinished.Get());
        if (configuration.evaluator.kind ==
            CudaSearchEvaluatorKind::FinishTime) {
            const auto launchFinishRefinement =
                    [&](auto stateType, auto simulateStunts) {
                using State = decltype(stateType);
                constexpr bool SimulateStunts =
                        decltype(simulateStunts)::value;
                RefineSearchFinishTimesKernel<
                        State, SimulateStunts>
                        <<<simulationBlocks, SimulationBlockSize>>>(
                        configuration.deviceScene,
                        configuration.deviceStaticConfiguration,
                        branchState.Get(),
                        mutableBoundaryControls.Get(),
                        baselineTicks.Get(),
                        timelineTickCount,
                        configuration.tickDurationMs,
                        configuration.prestartDurationMs,
                        configuration.branchTimeMs,
                        mutableFromTimeMs,
                        configuration.evaluationStartTimeMs,
                        static_cast<std::uint32_t>(
                                configuration.maximumEventCount),
                        candidateBestSamples.Get(),
                        finishRefinements.Get(),
                        finishCheckpointStates.Get(),
                        finishCheckpointTicks.Get(),
                        baselineInputs.Get(),
                        static_cast<std::uint32_t>(
                                configuration.baselineInputs.size()),
                        candidateEvents.Get(),
                        candidateInputValues.Get(),
                        compactInputOffsets.Get(),
                        compactInputCount,
                        compactRandomSteeringPipeline,
                        compactEditPipeline,
                        sparseMutationPipeline,
                        CandidateEdits(candidateCount),
                        SparseCandidateEvents(candidateCount),
                        eventCounts.Get(),
                        statuses.Get(),
                        activeCandidates.Get(),
                        collisionScratch.Get(),
                        shapeCollisionScratch.Get(),
                        shapeWorldScratch.Get(),
                        movingBoundsScratch.Get(),
                        surfaceHitScratch.Get(),
                        meshRangeScratch.Get(),
                        meshCellScratch.Get(),
                        responseOrderScratch.Get(),
                        configuration.maximumBatchSize,
                        collisionShapeCount,
                        cancellation.Get(),
                        candidateCount);
            };
            launchFinishRefinement(
                    CudaCandidatePhysicsState{}, std::false_type{});
        }
        cudaEventRecord(finishRefined.Get());
        std::size_t temporaryBytes = winnerScanTemporary.Bytes();
        error = cub::DeviceScan::InclusiveScan(
                winnerScanTemporary.Get(), temporaryBytes,
                candidateBestSamples.Get(), prefixBestSamples.Get(),
                BetterSample{
                        MaximizesScore(configuration.evaluator.kind)},
                winnerCount);
        if (error == cudaSuccess) {
            error = cudaMemcpyAsync(
                    reducedBest.Get(),
                    prefixBestSamples.Get() + winnerCount - 1u,
                    sizeof(DeviceSample), cudaMemcpyDeviceToDevice);
        }
        if (error != cudaSuccess) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic =
                    CudaFailure("launching CUDA winner scan", error);
            return result;
        }
        cudaEventRecord(winnerReduced.Get());
        if (configuration.captureBestState) {
            CaptureSearchWinnerStateKernel<<<1u, 1u>>>(
                    configuration.deviceScene,
                    configuration.deviceStaticConfiguration,
                    branchState.Get(),
                    mutableBoundaryControls.Get(),
                    baselineTicks.Get(),
                    reducedBest.Get(),
                    configuration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? finishRefinements.Get()
                            : nullptr,
                    configuration.tickDurationMs,
                    configuration.prestartDurationMs,
                    configuration.branchTimeMs,
                    mutableFromTimeMs,
                    configuration.evaluationStartTimeMs,
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    baselineInputs.Get(),
                    static_cast<std::uint32_t>(
                            configuration.baselineInputs.size()),
                    candidateEvents.Get(),
                    candidateInputValues.Get(),
                    compactInputOffsets.Get(),
                    compactInputCount,
                    compactRandomSteeringPipeline,
                    compactEditPipeline,
                    sparseMutationPipeline,
                    CandidateEdits(candidateCount),
                    SparseCandidateEvents(candidateCount),
                    candidateCount,
                    eventCounts.Get(),
                    statuses.Get(),
                    collisionScratch.Get(),
                    shapeCollisionScratch.Get(),
                    shapeWorldScratch.Get(),
                    movingBoundsScratch.Get(),
                    surfaceHitScratch.Get(),
                    meshRangeScratch.Get(),
                    meshCellScratch.Get(),
                    responseOrderScratch.Get(),
                    configuration.maximumBatchSize,
                    collisionShapeCount,
                    capturedWinnerState.Get());
        }
        cudaEventRecord(winnerStateCaptured.Get());
        AccumulateSearchBatchSummaryKernel
                <<<candidateBlocks, blockSize>>>(
                prefixBestSamples.Get(), candidateBestSamples.Get(),
                mutationCounts.Get(),
                statuses.Get(), activeCandidates.Get(),
                deduplicateCandidateInputs
                        ? candidateRepresentativeSlots.Get()
                        : nullptr,
                deduplicateCandidateInputs,
                candidateCount, evaluationTickCount,
                ClosestTargetDistanceScratch(), simulationBlocks,
                summary.Get());
        FinalizeSearchBatchKernel<<<1u, 1u>>>(
                reducedBest.Get(),
                capturedWinnerState.Get(),
                candidateBestSamples.Get(),
                baselineInputs.Get(),
                static_cast<std::uint32_t>(
                        configuration.baselineInputs.size()),
                candidateEvents.Get(),
                candidateInputValues.Get(),
                compactInputOffsets.Get(),
                compactInputCount,
                compactRandomSteeringPipeline,
                compactEditPipeline,
                sparseMutationPipeline,
                CandidateEdits(candidateCount),
                SparseCandidateEvents(candidateCount),
                eventCounts.Get(),
                mutationCounts.Get(),
                statuses.Get(),
                activeCandidates.Get(),
                candidateCount,
                static_cast<std::uint32_t>(
                        configuration.maximumEventCount),
                evaluationTickCount,
                MaximizesScore(configuration.evaluator.kind),
                baseline,
                configuration.captureBestState,
                globalBestSample.Get(),
                globalBestState.Get(),
                globalBestInputs.Get(),
                globalBestEventCount.Get(),
                globalBestMutationCount.Get(),
                summary.Get());
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic =
                    CudaFailure("launching CUDA search kernels", error);
            return result;
        }
        cudaEventRecord(finished.Get());
        if (!cancellationRequested) {
            error = cudaEventSynchronize(finished.Get());
        } else {
            while ((error = cudaEventQuery(finished.Get())) ==
                   cudaErrorNotReady) {
                if (!cancelled) {
                    try {
                        cancelled = cancellationRequested();
                    } catch (...) {
                        cancelled = true;
                    }
                    if (cancelled) {
                        *cancellation.Host() = 1u;
                        std::atomic_thread_fence(
                                std::memory_order_seq_cst);
                    }
                }
                std::this_thread::sleep_for(
                        std::chrono::milliseconds(1));
            }
        }
        if (error != cudaSuccess) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic =
                    CudaFailure("synchronizing CUDA search batch", error);
            return result;
        }
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, started.Get(), finished.Get());
        result.kernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds, started.Get(), winnerInitialized.Get());
        result.scoreInitializationKernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds,
                winnerInitialized.Get(),
                mutationsGenerated.Get());
        result.mutationKernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds,
                mutationsGenerated.Get(),
                simulationFinished.Get());
        result.simulationKernelMilliseconds = milliseconds;
        if (!configuration.collectHotPathMetrics &&
            simulationTuning != nullptr && !cancelled &&
            milliseconds > 0.0f) {
            const std::size_t tuningIndex =
                    selectedMinimumBlocks ==
                                    ThroughputKernelMinimumBlocksPerSm
                    ? 0u
                    : selectedMinimumBlocks ==
                                      TailKernelMinimumBlocksPerSm
                    ? 1u
                    : 2u;
            const std::uint32_t previousSamples =
                    simulationTuning->sampleCounts[tuningIndex];
            if (previousSamples == 0u) {
                simulationTuning->milliseconds[tuningIndex] =
                        milliseconds;
            } else {
                // Keep a recent, damped estimate so the selected kernel is
                // corrected quickly when the first sample was atypical or
                // the session's candidate mix changes over time.
                simulationTuning->milliseconds[tuningIndex] =
                        simulationTuning->milliseconds[tuningIndex] *
                                0.75f +
                        milliseconds * 0.25f;
            }
            if (previousSamples != UINT32_MAX) {
                simulationTuning->sampleCounts[tuningIndex] =
                        previousSamples + 1u;
            }
            if (simulationTuningRevalidation) {
                simulationTuning->successfulBatchesSinceRevalidation =
                        0u;
                simulationTuning->nextRevalidationIndex =
                        (tuningIndex + 1u) %
                        simulationTuning->milliseconds.size();
            } else if (!simulationTuningExploration &&
                       simulationTuning->
                                   successfulBatchesSinceRevalidation !=
                               UINT32_MAX) {
                ++simulationTuning->
                        successfulBatchesSinceRevalidation;
            }
        }
        cudaEventElapsedTime(
                &milliseconds,
                simulationFinished.Get(),
                finishRefined.Get());
        result.finishRefinementKernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds,
                finishRefined.Get(),
                finished.Get());
        result.winnerKernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds,
                finishRefined.Get(),
                winnerReduced.Get());
        result.winnerReductionKernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds,
                winnerReduced.Get(),
                winnerStateCaptured.Get());
        result.winnerStateCaptureKernelMilliseconds = milliseconds;
        cudaEventElapsedTime(
                &milliseconds,
                winnerStateCaptured.Get(),
                finished.Get());
        result.finalizationKernelMilliseconds = milliseconds;
        result.simulationSelectedMinimumBlocksPerMultiprocessor =
                selectedMinimumBlocks;
        result.simulationThreadsPerBlock = SimulationBlockSize;
        result.simulationRegistersPerThread =
                simulationMetrics->registersPerThread;
        result.simulationLocalBytesPerThread =
                simulationMetrics->localBytesPerThread;
        result.simulationActiveBlocksPerMultiprocessor =
                simulationMetrics->activeBlocksPerMultiprocessor;
        result.simulationTheoreticalOccupancy =
                simulationMetrics->theoreticalOccupancy;

        DeviceBatchSummary hostSummary;
        error = cudaMemcpy(
                &hostSummary, summary.Get(), sizeof(hostSummary),
                cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic =
                    CudaFailure("copying CUDA search summary", error);
            return result;
        }
        result.deviceToHostBytes += sizeof(hostSummary);
        result.status = hostSummary.status;
        result.evaluatedCandidateCount =
                hostSummary.evaluatedCandidateCount;
        result.simulatedCandidateCount =
                hostSummary.simulatedCandidateCount;
        result.deduplicatedCandidateCount =
                hostSummary.evaluatedCandidateCount >=
                                hostSummary.simulatedCandidateCount
                ? hostSummary.evaluatedCandidateCount -
                          hostSummary.simulatedCandidateCount
                : 0u;
        result.evaluatorCalls = hostSummary.evaluatorCalls;
        result.qualifyingCandidateCount =
                hostSummary.qualifyingCandidateCount;
        if (configuration.evaluator.kind ==
                    CudaSearchEvaluatorKind::VolumeEntry &&
            hostSummary.closestTargetDistanceSquared <
                    cuda_search_progress_detail::
                            InvalidClosestTargetDistanceSquared) {
            result.closestTargetDistance = std::sqrt(
                    hostSummary.closestTargetDistanceSquared);
        }
        result.totalMutationCount =
                hostSummary.totalMutationCount;
        result.mutationImprovementCount =
                baseline ? 0u
                         : hostSummary.mutationImprovementCount;
        result.bestChanged = hostSummary.bestChanged;
        if (hostSummary.bestValid) {
            const bool hostBestCacheMatches =
                    hostBestCacheValid && hostBestCache.valid &&
                    hostBestCache.mutation ==
                            hostSummary.bestMutation &&
                    hostBestCache.stateCaptured ==
                            configuration.captureBestState &&
                    hostBestCache.candidateId ==
                            hostSummary.bestCandidateId &&
                    hostBestCache.mutationCount ==
                            hostSummary.bestMutationCount &&
                    hostBestEventCount ==
                            hostSummary.globalEventCount &&
                    hostBestCache.inputs.size() ==
                            immutableInputPrefix.size() +
                                    hostSummary.globalEventCount +
                                    immutableInputTail.size();
            if (!hostSummary.bestChanged &&
                hostBestCacheMatches) {
                result.best = hostBestCache;
            } else {
                hostBestCacheValid = false;
                result.best.valid = true;
                result.best.mutation = hostSummary.bestMutation;
                result.best.stateCaptured =
                        configuration.captureBestState;
                result.best.candidateId =
                        hostSummary.bestCandidateId;
                result.best.mutationCount =
                        hostSummary.bestMutationCount;
                DeviceSample bestSample;
                error = cudaMemcpy(
                        &bestSample, globalBestSample.Get(),
                        sizeof(bestSample), cudaMemcpyDeviceToHost);
                if (error == cudaSuccess &&
                    configuration.captureBestState) {
                    error = cudaMemcpy(
                            &result.best.state, globalBestState.Get(),
                            sizeof(result.best.state),
                            cudaMemcpyDeviceToHost);
                }
                if (error != cudaSuccess) {
                    result.status = CudaSearchStatus::DeviceFailure;
                    result.diagnostic = CudaFailure(
                            "copying CUDA winning state", error);
                    return result;
                }
                result.best.score = bestSample.score;
                result.best.evaluationTick =
                        bestSample.evaluationTick;
                result.best.timeMs = bestSample.timeMs;
                result.best.detail0 = bestSample.detail0;
                result.best.detail1 = bestSample.detail1;
                result.best.inputs = immutableInputPrefix;
                const std::size_t suffixOffset =
                        result.best.inputs.size();
                result.best.inputs.resize(
                        suffixOffset + hostSummary.globalEventCount);
                if (hostSummary.globalEventCount != 0u) {
                    error = cudaMemcpy(
                            result.best.inputs.data() + suffixOffset,
                            globalBestInputs.Get(),
                            hostSummary.globalEventCount *
                                    sizeof(CudaSearchInputEvent),
                            cudaMemcpyDeviceToHost);
                    if (error != cudaSuccess) {
                        result.status =
                                CudaSearchStatus::DeviceFailure;
                        result.diagnostic = CudaFailure(
                                "copying CUDA winning inputs", error);
                        return result;
                    }
                    for (std::size_t index = suffixOffset;
                         index < result.best.inputs.size(); ++index) {
                        result.best.inputs[index] =
                                cuda_search_detail::
                                        AbsoluteSuffixEvent(
                                                result.best.inputs[index],
                                                mutableFromTimeMs);
                    }
                }
                result.best.inputs.reserve(
                        result.best.inputs.size() +
                        immutableInputTail.size());
                for (CudaSearchInputEvent input :
                     immutableInputTail) {
                    result.best.inputs.push_back(
                            cuda_search_detail::AbsoluteSuffixEvent(
                                    input, mutableFromTimeMs));
                }
                result.deviceToHostBytes += sizeof(bestSample) +
                        (configuration.captureBestState
                                 ? sizeof(result.best.state) : 0u) +
                        hostSummary.globalEventCount *
                                sizeof(CudaSearchInputEvent);
                hostBestCache = result.best;
                hostBestEventCount =
                        hostSummary.globalEventCount;
                hostBestCacheValid = true;
            }
        }
        if (baseline && result.status == CudaSearchStatus::Success) {
            baselineEvaluated = true;
            // A successful VolumeEntry baseline exits at its first hit, so
            // later prefix states were intentionally never produced. Fall
            // back to full candidate simulation in that already-qualified
            // case; misses and non-terminating evaluators populate every
            // prefix state and remain safe to reuse.
            baselinePrefixesValid =
                    baselinePrefixReuseEligible &&
                    !(configuration.evaluator.kind ==
                                      CudaSearchEvaluatorKind::VolumeEntry &&
                      hostSummary.bestValid);
        }
        if (result.status != CudaSearchStatus::Success &&
            result.diagnostic.empty()) {
            result.diagnostic =
                    std::string("CUDA search batch status: ") +
                    CudaSearchStatusName(result.status);
        }
        return result;
    }

    CudaSearchBatchExecution Execute(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool baseline,
            const std::function<bool()> &cancellationRequested) noexcept {
        if (!configuration.collectHotPathMetrics) {
            return ExecuteImpl(
                    firstCandidateId, candidateCount, baseline,
                    cancellationRequested);
        }

        const std::uint32_t recordCount = std::min(
                candidateCount, configuration.maximumBatchSize);
        const std::size_t hotPathBytes =
                static_cast<std::size_t>(recordCount) *
                sizeof(DeviceHotPathRecord);
        cudaError_t error = cudaSuccess;
        if (hotPathBytes != 0u) {
            error = cudaMemset(
                    HotPathRecords(), 0, hotPathBytes);
        }
        if (error != cudaSuccess) {
            CudaSearchBatchExecution result;
            result.firstCandidateId = firstCandidateId;
            result.candidateCount = candidateCount;
            result.residentDeviceBytes = residentBytes;
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic = CudaFailure(
                    "clearing CUDA hot-path records", error);
            return result;
        }

        CudaSearchBatchExecution result = ExecuteImpl(
                firstCandidateId, candidateCount, baseline,
                cancellationRequested);
        if (hotPathBytes == 0u) {
            result.hotPath.collected = true;
            result.hotPath.complete =
                    result.status == CudaSearchStatus::Success;
            return result;
        }

        std::unique_ptr<DeviceHotPathRecord[]> hostRecords(
                new (std::nothrow) DeviceHotPathRecord[recordCount]);
        if (!hostRecords) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic =
                    "allocating CUDA hot-path host records failed";
            return result;
        }
        error = cudaMemcpy(
                hostRecords.get(), HotPathRecords(), hotPathBytes,
                cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic = CudaFailure(
                    "copying CUDA hot-path records", error);
            return result;
        }
        result.deviceToHostBytes += hotPathBytes;
        result.hotPath.collected = true;
        for (std::uint32_t index = 0u; index < recordCount; ++index) {
            AccumulateHotPathRecord(
                    &result.hotPath, hostRecords[index]);
        }
        result.hotPath.complete =
                result.status == CudaSearchStatus::Success;
        return result;
    }
};

CudaSearchExecutor::CudaSearchExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaSearchExecutor::~CudaSearchExecutor() = default;
CudaSearchExecutor::CudaSearchExecutor(CudaSearchExecutor &&) noexcept =
        default;
CudaSearchExecutor &CudaSearchExecutor::operator=(
        CudaSearchExecutor &&) noexcept = default;

std::unique_ptr<CudaSearchExecutor> CudaSearchExecutor::Create(
        const CudaSearchExecutorConfiguration &configuration,
        std::string *diagnostic) noexcept {
    try {
        if (configuration.deviceScene == nullptr ||
            configuration.deviceStaticConfiguration == nullptr ||
            configuration.maximumBatchSize == 0u ||
            configuration.tickDurationMs == 0u ||
            configuration.maximumEventCount <
                    configuration.baselineInputs.size() ||
            configuration.maximumEventCount > UINT32_MAX ||
            configuration.baselineTicks.empty() ||
            configuration.modifiers.empty() ||
            configuration.branchTimeMs < 0 ||
            configuration.branchTimeMs >
                    INT32_MAX - configuration.tickDurationMs ||
            configuration.evaluationStartTimeMs <
                    configuration.branchTimeMs +
                            configuration.tickDurationMs ||
             configuration.evaluationEndTimeMs <
                     configuration.evaluationStartTimeMs ||
            !IsCudaSearchSimulationMinimumBlocksValid(
                    configuration.
                            simulationMinimumBlocksPerMultiprocessorForTesting) ||
            (configuration.incumbent &&
             configuration.incumbent->evaluationTick >=
                     static_cast<std::uint64_t>(
                             configuration.evaluationEndTimeMs -
                             configuration.evaluationStartTimeMs) /
                                     configuration.tickDurationMs +
                             1u)) {
            if (diagnostic != nullptr) {
                *diagnostic = "invalid CUDA search executor configuration";
            }
            return {};
        }
        const std::uint64_t evaluationTicks =
                static_cast<std::uint64_t>(
                        configuration.evaluationEndTimeMs -
                        configuration.evaluationStartTimeMs) /
                        configuration.tickDurationMs +
                1u;
        if (evaluationTicks == 0u ||
            evaluationTicks > UINT32_MAX) {
            if (diagnostic != nullptr) {
                *diagnostic = "CUDA evaluation timeline is too large";
            }
            return {};
        }
        CudaPackedStaticConfigurationHeader packedConfiguration{};
        const cudaError_t configurationCopyError = cudaMemcpy(
                &packedConfiguration,
                configuration.deviceStaticConfiguration,
                sizeof(packedConfiguration),
                cudaMemcpyDeviceToHost);
        if (configurationCopyError != cudaSuccess ||
            packedConfiguration.magic !=
                    CudaPackedStaticConfigurationHeader::Magic ||
            packedConfiguration.schemaVersion !=
                    CudaPackedStaticConfigurationHeader::
                            SchemaVersion) {
            if (diagnostic != nullptr) {
                *diagnostic = configurationCopyError != cudaSuccess
                        ? CudaFailure(
                                  "reading CUDA static configuration",
                                  configurationCopyError)
                        : "invalid CUDA static configuration header";
            }
            return {};
        }
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_FOUR_WHEELS)
        if (configuration.branchState.vehicle.wheels.count != 4u) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "research wheel-count fact does not match";
            }
            return {};
        }
        for (std::uint32_t index = 0u; index < 4u; ++index) {
            const CudaWheelState &stateWheel =
                    configuration.branchState.vehicle.wheels.
                            values[index];
            const VehicleWheelDefinition &definitionWheel =
                    reinterpret_cast<
                            const VehicleWheelDefinition *>(
                            &packedConfiguration.wheels.wheels)[index];
            const bool immutableFieldsMatch =
                    stateWheel.killsLateralSpeedOnContact ==
                            definitionWheel.
                                    killsLateralSpeedOnContact &&
                    stateWheel.axle ==
                            static_cast<std::uint32_t>(
                                    definitionWheel.axle) &&
                    std::memcmp(
                            &stateWheel.rollingRadius,
                            &definitionWheel.rollingRadius,
                            sizeof(float)) == 0 &&
                    std::memcmp(
                            &stateWheel.forceApplicationPoint,
                            &definitionWheel.forceApplicationPoint,
                            sizeof(GmVec3)) == 0 &&
                    std::memcmp(
                            &stateWheel.restPose,
                            &definitionWheel.restSurfacePose,
                            sizeof(GmIso4)) == 0;
            if (!immutableFieldsMatch) {
                if (diagnostic != nullptr) {
                    *diagnostic =
                            "research immutable wheel facts do not match";
                }
                return {};
            }
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_CANONICAL_WHEEL_FACTS)
            const GmMat3 identityRotation = {
                    {1.0f, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    {0.0f, 0.0f, 1.0f},
            };
            const VehicleWheelDefinition &firstWheel =
                    *reinterpret_cast<
                            const VehicleWheelDefinition *>(
                            &packedConfiguration.wheels.wheels);
            const VehicleWheelAxle expectedAxle =
                    index < 2u
                    ? VehicleWheelAxle::Front
                    : VehicleWheelAxle::Rear;
            const bool canonicalFactsMatch =
                    definitionWheel.axle == expectedAxle &&
                    definitionWheel.
                            killsLateralSpeedOnContact &&
                    std::memcmp(
                            &definitionWheel.rollingRadius,
                            &firstWheel.rollingRadius,
                            sizeof(float)) == 0 &&
                    std::memcmp(
                            &definitionWheel.restSurfacePose.rotation,
                            &identityRotation,
                            sizeof(GmMat3)) == 0;
            if (!canonicalFactsMatch) {
                if (diagnostic != nullptr) {
                    *diagnostic =
                            "research canonical wheel facts do not match";
                }
                return {};
            }
#endif
        }
#endif
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_WATER_ONLY)
        CudaPackedSceneHeader packedScene{};
        const cudaError_t sceneCopyError = cudaMemcpy(
                &packedScene,
                configuration.deviceScene,
                sizeof(packedScene),
                cudaMemcpyDeviceToHost);
        if (sceneCopyError != cudaSuccess ||
            packedScene.magic != CudaPackedSceneHeader::Magic ||
            packedScene.schemaVersion !=
                    CudaPackedSceneHeader::SchemaVersion) {
            if (diagnostic != nullptr) {
                *diagnostic = sceneCopyError != cudaSuccess
                        ? CudaFailure(
                                  "reading CUDA scene header",
                                  sceneCopyError)
                        : "invalid CUDA scene header";
            }
            return {};
        }
        const std::uint64_t sceneBase =
                reinterpret_cast<std::uintptr_t>(
                        configuration.deviceScene);
        cudaError_t researchConstantError =
                cudaMemcpyToSymbol(
                        cuda::research::StaticSceneBase,
                        &sceneBase,
                        sizeof(sceneBase),
                        0u,
                        cudaMemcpyHostToDevice);
        if (researchConstantError == cudaSuccess) {
            researchConstantError = cudaMemcpyToSymbol(
                    cuda::research::StaticScene,
                    &packedScene,
                    sizeof(packedScene),
                    0u,
                    cudaMemcpyHostToDevice);
        }
        if (researchConstantError != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "copying research CUDA scene header",
                        researchConstantError);
            }
            return {};
        }
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_CONSTANT_COLLISION_SHAPES)
        if (packedConfiguration.collisionShapes.count != 8u ||
            packedConfiguration.collisionShapes.stride !=
                    sizeof(CudaVehicleCollisionShape)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "research collision shape facts do not match";
            }
            return {};
        }
        std::array<CudaVehicleCollisionShape, 8u>
                constantCollisionShapes{};
        const auto *configurationBytes =
                static_cast<const std::byte *>(
                        configuration.deviceStaticConfiguration);
        researchConstantError = cudaMemcpy(
                constantCollisionShapes.data(),
                configurationBytes +
                        packedConfiguration.collisionShapes.offset,
                sizeof(constantCollisionShapes),
                cudaMemcpyDeviceToHost);
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES)
        constexpr std::uint32_t ExpectedWheelIndices[] = {
                UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                0u, 1u, 3u, 2u};
        if (researchConstantError == cudaSuccess) {
            for (std::uint32_t index = 0u;
                 index < constantCollisionShapes.size(); ++index) {
                if (constantCollisionShapes[index].
                                parentShapeIndex != UINT32_MAX ||
                    constantCollisionShapes[index].wheelIndex !=
                            ExpectedWheelIndices[index]) {
                    if (diagnostic != nullptr) {
                        *diagnostic =
                                "research collision shape topology "
                                "does not match";
                    }
                    return {};
                }
            }
        }
#endif
        if (researchConstantError == cudaSuccess) {
            researchConstantError = cudaMemcpyToSymbol(
                    cuda::research::StaticCollisionShapes,
                    constantCollisionShapes.data(),
                    sizeof(constantCollisionShapes),
                    0u,
                    cudaMemcpyHostToDevice);
        }
        if (researchConstantError != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "copying research CUDA collision shapes",
                        researchConstantError);
            }
            return {};
        }
#endif
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_CONSTANT_CURVE_KEYS)
        if (packedConfiguration.curveKeys.count > 4096u ||
            packedConfiguration.curveKeys.stride !=
                    sizeof(CudaTuningCurveKey)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "research tuning curve key facts do not match";
            }
            return {};
        }
        std::vector<CudaTuningCurveKey> constantCurveKeys(
                packedConfiguration.curveKeys.count);
        researchConstantError = cudaMemcpy(
                constantCurveKeys.data(),
                configurationBytes +
                        packedConfiguration.curveKeys.offset,
                constantCurveKeys.size() *
                        sizeof(CudaTuningCurveKey),
                cudaMemcpyDeviceToHost);
        if (researchConstantError == cudaSuccess &&
            !constantCurveKeys.empty()) {
            researchConstantError = cudaMemcpyToSymbol(
                    cuda::research::StaticCurveKeys,
                    constantCurveKeys.data(),
                    constantCurveKeys.size() *
                            sizeof(CudaTuningCurveKey),
                    0u,
                    cudaMemcpyHostToDevice);
        }
        if (researchConstantError != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "copying research CUDA tuning curve keys",
                        researchConstantError);
            }
            return {};
        }
#endif
        const std::uint64_t configurationBase =
                reinterpret_cast<std::uintptr_t>(
                        configuration.deviceStaticConfiguration);
        const cudaError_t constantBaseError =
                cudaMemcpyToSymbol(
                        cuda::research::StaticConfigurationBase,
                        &configurationBase,
                        sizeof(configurationBase),
                        0u,
                        cudaMemcpyHostToDevice);
        if (constantBaseError != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "copying research CUDA configuration base",
                        constantBaseError);
            }
            return {};
        }
        const cudaError_t constantConfigurationError =
                cudaMemcpyToSymbol(
                        cuda::research::StaticConfiguration,
                        &packedConfiguration,
                        sizeof(packedConfiguration),
                        0u,
                        cudaMemcpyHostToDevice);
        if (constantConfigurationError != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "copying research CUDA static configuration",
                        constantConfigurationError);
            }
            return {};
        }
#endif
        const std::int64_t mutableFromTimeMs =
                configuration.branchTimeMs +
                configuration.tickDurationMs;
        cuda_search_detail::SearchInputPartition inputPartition;
        if (!cuda_search_detail::PartitionSearchInputs(
                    configuration.baselineInputs,
                    configuration.branchTimeMs,
                    mutableFromTimeMs,
                    &inputPartition)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "CUDA search inputs cannot be partitioned at the mutable boundary";
            }
            return {};
        }
        CudaSearchExecutorConfiguration preparedConfiguration =
                configuration;
        preparedConfiguration.baselineInputs =
                inputPartition.mutableSuffix;
        preparedConfiguration.maximumEventCount -=
                inputPartition.immutablePrefix.size();
        for (CudaSearchModifierConfiguration &modifier :
             preparedConfiguration.modifiers) {
            if (modifier.window.maximumTimeMs < mutableFromTimeMs) {
                if (diagnostic != nullptr) {
                    *diagnostic =
                            "CUDA modifier window does not intersect the mutable suffix";
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
        const bool windowLocalMutation =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting;
        if (windowLocalMutation) {
            const std::int64_t evaluationEndRelativeMs =
                    preparedConfiguration.evaluationEndTimeMs -
                    mutableFromTimeMs;
            std::int64_t materializationEndTimeMs =
                    evaluationEndRelativeMs;
            for (const CudaSearchModifierConfiguration &modifier :
                 preparedConfiguration.modifiers) {
                materializationEndTimeMs = std::max(
                        materializationEndTimeMs,
                        modifier.window.maximumTimeMs);
            }
            cuda_search_detail::SearchInputWindow inputWindow;
            if (!cuda_search_detail::PartitionSearchInputWindow(
                        preparedConfiguration.baselineInputs,
                        materializationEndTimeMs,
                        &inputWindow) ||
                inputWindow.immutableTail.size() >
                        preparedConfiguration.maximumEventCount) {
                if (diagnostic != nullptr) {
                    *diagnostic =
                            "CUDA search inputs cannot be partitioned at the materialization horizon";
                }
                return {};
            }
            preparedConfiguration.maximumEventCount -=
                    inputWindow.immutableTail.size();
            preparedConfiguration.baselineInputs =
                    std::move(inputWindow.materialized);
            immutableInputTail =
                    std::move(inputWindow.immutableTail);
        }
        const std::uint64_t candidateEvents =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                preparedConfiguration.maximumEventCount;
        const std::uint64_t winnerSampleCount =
                1u + static_cast<std::uint64_t>(
                             preparedConfiguration.maximumBatchSize);
        const std::uint64_t collisionCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                cuda::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                cuda::collision::ShapeCollisionCapacity;
        const std::uint64_t collisionTileStride =
                (static_cast<std::uint64_t>(
                         preparedConfiguration.maximumBatchSize) +
                 cuda::collision::CudaCollisionSearchTileWidth - 1u) /
                cuda::collision::CudaCollisionSearchTileWidth;
        const std::uint64_t collisionTileCount =
                collisionTileStride *
                cuda::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionTileCount =
                collisionTileStride *
                cuda::collision::ShapeCollisionCapacity;
        const std::uint64_t shapeQueryCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                packedConfiguration.collisionShapes.count;
        const std::uint64_t surfaceHitCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                cuda::collision::SurfaceHitCapacity;
        const std::uint64_t meshRangeCount =
                surfaceHitCount;
        const std::uint64_t meshCellCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                cuda::collision::MeshCellHitCapacity;
        if (candidateEvents >
                    std::numeric_limits<std::size_t>::max() ||
            winnerSampleCount >
                    std::numeric_limits<std::size_t>::max() ||
            collisionCount >
                    std::numeric_limits<std::size_t>::max() ||
            shapeCollisionCount >
                    std::numeric_limits<std::size_t>::max() ||
            collisionTileCount >
                    std::numeric_limits<std::size_t>::max() ||
            shapeCollisionTileCount >
                    std::numeric_limits<std::size_t>::max() ||
            shapeQueryCount >
                    std::numeric_limits<std::size_t>::max() ||
            surfaceHitCount >
                    std::numeric_limits<std::size_t>::max() ||
            meshRangeCount >
                    std::numeric_limits<std::size_t>::max() ||
            meshCellCount >
                    std::numeric_limits<std::size_t>::max()) {
            if (diagnostic != nullptr) {
                *diagnostic = "CUDA search buffer dimensions overflow";
            }
            return {};
        }

        auto impl = std::make_unique<Impl>();
        impl->configuration = preparedConfiguration;
        if (!preparedConfiguration.collectHotPathMetrics) {
            impl->specializedModule =
                    preparedConfiguration.sessionSpecialization;
        }
        switch (packedConfiguration.tuning.handlingModel) {
        case static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_Standard):
        case static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_Lateral):
            impl->handlingSpecialization =
                    CudaHandlingSpecialization::Legacy;
            break;
        case static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_GearedDrive):
            impl->handlingSpecialization =
                    packedConfiguration.water.present
                    ? CudaHandlingSpecialization::
                              GearedDriveWater
                    : CudaHandlingSpecialization::
                              GearedDriveDry;
            break;
        default:
            impl->handlingSpecialization =
                    CudaHandlingSpecialization::Generic;
            break;
        }
        impl->immutableInputPrefix =
                std::move(inputPartition.immutablePrefix);
        impl->immutableInputTail =
                std::move(immutableInputTail);
        impl->mutableFromTimeMs = mutableFromTimeMs;
        impl->timelineTickCount = static_cast<std::uint32_t>(
                preparedConfiguration.baselineTicks.size());
        impl->steadyTimeline = std::all_of(
                preparedConfiguration.baselineTicks.begin(),
                preparedConfiguration.baselineTicks.end(),
                [](const CudaControlTick &tick) {
                    return tick.actionFlags == 0u &&
                            tick.respawnAtCheckpointCount == 0u;
                });
        impl->evaluationTickCount =
                static_cast<std::uint32_t>(evaluationTicks);
        impl->collisionShapeCount =
                packedConfiguration.collisionShapes.count;
        impl->baselineInputsCanonical = CanonicalBaselineInputs(
                preparedConfiguration.baselineInputs,
                0);
        impl->compactRandomSteeringPipeline =
                !preparedConfiguration.useLegacyMutationPipelineForTesting &&
                impl->baselineInputsCanonical &&
                std::all_of(
                        preparedConfiguration.modifiers.begin(),
                        preparedConfiguration.modifiers.end(),
                        [&](const CudaSearchModifierConfiguration &modifier) {
                            return modifier.kind ==
                                           CudaSearchModifierKind::
                                                   RandomSteering &&
                                    modifier.window.minimumTimeMs >=
                                            0;
                        });
        std::vector<std::uint32_t> compactInputIndices;
        std::vector<std::uint32_t> compactInputOffsets;
        if (impl->compactRandomSteeringPipeline) {
            compactInputIndices.reserve(
                    preparedConfiguration.baselineInputs.size());
            compactInputOffsets.assign(
                    preparedConfiguration.baselineInputs.size(), UINT32_MAX);
            for (std::size_t inputIndex = 0u;
                 inputIndex < preparedConfiguration.baselineInputs.size();
                 ++inputIndex) {
                const CudaSearchInputEvent &input =
                        preparedConfiguration.baselineInputs[inputIndex];
                const bool selected =
                        input.action == 4u && input.valueKind == 2u &&
                        std::any_of(
                                preparedConfiguration.modifiers.begin(),
                                preparedConfiguration.modifiers.end(),
                                [&](const CudaSearchModifierConfiguration
                                            &modifier) {
                                    return input.timeMs >=
                                                    modifier.window.
                                                            minimumTimeMs &&
                                            input.timeMs <=
                                                    modifier.window.
                                                            maximumTimeMs;
                                });
                if (!selected) {
                    continue;
                }
                compactInputOffsets[inputIndex] =
                        static_cast<std::uint32_t>(
                                compactInputIndices.size());
                compactInputIndices.push_back(
                        static_cast<std::uint32_t>(inputIndex));
            }
        }
        impl->compactInputCount =
                static_cast<std::uint32_t>(
                        compactInputIndices.size());
        const bool directDeletionPipeline =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting &&
                impl->baselineInputsCanonical &&
                preparedConfiguration.modifiers.size() == 1u &&
                preparedConfiguration.modifiers[0].kind ==
                        CudaSearchModifierKind::InputDeletion;
        const bool directExistingEventPipeline =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting &&
                impl->baselineInputsCanonical &&
                preparedConfiguration.modifiers.size() == 1u &&
                preparedConfiguration.modifiers[0].kind ==
                        CudaSearchModifierKind::ExistingEvent &&
                preparedConfiguration.modifiers[0].
                                timeParameterMs == 0;
        impl->sparseMutationPipeline =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting &&
                impl->baselineInputsCanonical &&
                !impl->compactRandomSteeringPipeline &&
                !directDeletionPipeline &&
                !directExistingEventPipeline;
        impl->compactEditPipeline =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting &&
                !impl->compactRandomSteeringPipeline &&
                !impl->sparseMutationPipeline;
        impl->directDeletionPipeline =
                impl->compactEditPipeline && directDeletionPipeline;
        impl->directExistingEventPipeline =
                impl->compactEditPipeline &&
                directExistingEventPipeline;
        impl->prefixReusePlan =
                PlanCudaSearchPrefixReuse(preparedConfiguration);
        impl->baselinePrefixReuseEligible =
                impl->prefixReusePlan.enabled;
        impl->materializesCandidateEvents =
                preparedConfiguration.
                        useLegacyMutationPipelineForTesting ||
                (impl->compactEditPipeline &&
                 !impl->directDeletionPipeline &&
                 !impl->directExistingEventPipeline);
        impl->editCapacity = impl->compactEditPipeline
                ? CompactOutputEditCapacity(
                          preparedConfiguration)
                : 0u;
        impl->eraseCapacity = impl->compactEditPipeline
                ? static_cast<std::uint32_t>(
                          preparedConfiguration.baselineInputs.size())
                : 0u;
        impl->packedEditStorage =
                impl->compactEditPipeline &&
                preparedConfiguration.maximumEventCount <=
                        UINT16_MAX &&
                std::all_of(
                        preparedConfiguration.baselineInputs.begin(),
                        preparedConfiguration.baselineInputs.end(),
                        [](const CudaSearchInputEvent &event) {
                            return event.action <= UINT16_MAX &&
                                    event.valueKind <= UINT8_MAX;
                        });
        impl->needsTemporaryEvents =
                impl->materializesCandidateEvents;
        impl->needsPassBaselineEvents =
                !impl->sparseMutationPipeline &&
                (preparedConfiguration.useLegacyMutationPipelineForTesting ||
                 !impl->baselineInputsCanonical ||
                 std::any_of(
                        preparedConfiguration.modifiers.begin(),
                        preparedConfiguration.modifiers.end(),
                        [&](const CudaSearchModifierConfiguration &modifier) {
                            const bool heldInsertion =
                                    modifier.kind ==
                                            CudaSearchModifierKind::
                                                    InputInsertion &&
                                    ((modifier.steering.enabled != 0u &&
                                      modifier.steering.maximumHoldMs > 0) ||
                                     (modifier.accelerate.enabled != 0u &&
                                      modifier.accelerate.maximumHoldMs > 0) ||
                                     (modifier.brake.enabled != 0u &&
                                      modifier.brake.maximumHoldMs > 0));
                            return heldInsertion ||
                                    modifier.window.minimumTimeMs <
                                            0;
                        }));
        impl->needsEligibleIndices =
                impl->compactEditPipeline ||
                preparedConfiguration.useLegacyMutationPipelineForTesting ||
                std::any_of(
                        preparedConfiguration.modifiers.begin(),
                        preparedConfiguration.modifiers.end(),
                        [](const CudaSearchModifierConfiguration &modifier) {
                            return modifier.kind ==
                                            CudaSearchModifierKind::
                                                    ExistingEvent ||
                                    modifier.kind ==
                                            CudaSearchModifierKind::
                                                    InputInsertion ||
                                    modifier.kind ==
                                            CudaSearchModifierKind::
                                                    InputDeletion;
                        });
        std::vector<std::uint32_t> sharedEligibleIndices;
        if (impl->sparseMutationPipeline &&
            std::any_of(
                    preparedConfiguration.modifiers.begin(),
                    preparedConfiguration.modifiers.end(),
                    [](const CudaSearchModifierConfiguration &modifier) {
                        return modifier.kind ==
                                       CudaSearchModifierKind::ExistingEvent ||
                                modifier.kind ==
                                       CudaSearchModifierKind::InputDeletion;
                    })) {
            sharedEligibleIndices.reserve(
                    preparedConfiguration.baselineInputs.size());
            for (std::size_t inputIndex = 0u;
                 inputIndex < preparedConfiguration.baselineInputs.size();
                 ++inputIndex) {
                const CudaSearchInputEvent &event =
                        preparedConfiguration.baselineInputs[inputIndex];
                const bool eligible = std::any_of(
                        preparedConfiguration.modifiers.begin(),
                        preparedConfiguration.modifiers.end(),
                        [&](const CudaSearchModifierConfiguration &modifier) {
                            if (event.timeMs <
                                        modifier.window.minimumTimeMs ||
                                event.timeMs >
                                        modifier.window.maximumTimeMs) {
                                return false;
                            }
                            if (modifier.kind ==
                                CudaSearchModifierKind::ExistingEvent) {
                                return (event.action == 4u &&
                                        event.valueKind == 2u) ||
                                        ((modifier.optionFlags & 2u) != 0u &&
                                         (event.action == 1u ||
                                          event.action == 2u)) ||
                                        ((modifier.optionFlags & 4u) != 0u &&
                                         event.action == 3u);
                            }
                            if (modifier.kind ==
                                CudaSearchModifierKind::InputDeletion) {
                                return (modifier.steering.enabled != 0u &&
                                        modifier_ops::ActionInGroup(
                                                event.action, 0u)) ||
                                        (modifier.accelerate.enabled != 0u &&
                                         modifier_ops::ActionInGroup(
                                                 event.action, 1u)) ||
                                        (modifier.brake.enabled != 0u &&
                                         modifier_ops::ActionInGroup(
                                                 event.action, 2u));
                            }
                            return false;
                        });
                if (eligible) {
                    sharedEligibleIndices.push_back(
                            static_cast<std::uint32_t>(inputIndex));
                }
            }
        }
        impl->sharedEligibleCount =
                static_cast<std::uint32_t>(
                        sharedEligibleIndices.size());
        const std::size_t candidates =
                preparedConfiguration.maximumBatchSize;
        const std::size_t eventSlots =
                static_cast<std::size_t>(candidateEvents);
        const std::size_t winnerSampleSlots =
                static_cast<std::size_t>(winnerSampleCount);
        const std::size_t summaryBlockCount =
                (candidates + SimulationBlockSize - 1u) /
                SimulationBlockSize;
        const std::size_t collisionSlots =
                static_cast<std::size_t>(collisionCount);
        const std::size_t collisionTileSlots =
                static_cast<std::size_t>(collisionTileCount);
        const std::size_t shapeCollisionTileSlots =
                static_cast<std::size_t>(
                        shapeCollisionTileCount);
        const std::size_t shapeQuerySlots =
                static_cast<std::size_t>(shapeQueryCount);
        const std::size_t surfaceHitSlots =
                static_cast<std::size_t>(surfaceHitCount);
        const std::size_t meshRangeSlots =
                static_cast<std::size_t>(meshRangeCount);
        const std::size_t meshCellSlots =
                static_cast<std::size_t>(meshCellCount);
        const std::size_t compactValueSlots =
                impl->compactRandomSteeringPipeline
                ? candidates * impl->compactInputCount
                : 0u;
        const std::size_t prefixCandidateSlots =
                impl->baselinePrefixReuseEligible ? candidates : 0u;
        const std::size_t deduplicationCandidateSlots =
                impl->DeduplicationStorageEligible(
                        preparedConfiguration.maximumBatchSize)
                ? candidates : 0u;
        const std::size_t baselinePrefixSlots =
                impl->baselinePrefixReuseEligible
                ? preparedConfiguration.baselineTicks.size() : 0u;
        impl->editStorageBytes =
                impl->compactEditPipeline
                ? Impl::EditStorageSize(
                          preparedConfiguration.maximumBatchSize,
                          impl->editCapacity,
                          impl->eraseCapacity,
                          impl->packedEditStorage)
                : 0u;
        impl->editStorageAliasesTemporary =
                impl->compactEditPipeline &&
                impl->materializesCandidateEvents &&
                eventSlots <=
                        std::numeric_limits<std::size_t>::max() /
                                sizeof(CudaSearchInputEvent) &&
                impl->editStorageBytes <=
                        eventSlots *
                                sizeof(CudaSearchInputEvent);
        if (!impl->shapeCollisionScratch.Allocate(
                    shapeCollisionTileSlots) ||
            !impl->branchState.Allocate(1u) ||
            !impl->mutableBoundaryControls.Allocate(1u) ||
            !impl->baselineTicks.Allocate(
                    preparedConfiguration.baselineTicks.size()) ||
            !impl->baselineInputs.Allocate(
                    preparedConfiguration.baselineInputs.size()) ||
            !impl->baselinePrefixStates.Allocate(
                    baselinePrefixSlots) ||
            !impl->baselinePrefixBestSamples.Allocate(
                    baselinePrefixSlots) ||
            !impl->baselinePrefixClosestTargetDistanceSquared.Allocate(
                    baselinePrefixSlots) ||
            !impl->modifiers.Allocate(
                    preparedConfiguration.modifiers.size()) ||
            !impl->smoothWeights.Allocate(
                    preparedConfiguration.smoothWeights.size()) ||
            !impl->evaluator.Allocate(1u) ||
            !impl->condition.Allocate(
                    preparedConfiguration.condition
                            ? preparedConfiguration.condition->instructions.size()
                            : 0u) ||
            !impl->capturedWinnerState.Allocate(1u) ||
            !impl->candidateBestSamples.Allocate(winnerSampleSlots) ||
            !impl->finishRefinements.Allocate(
                    preparedConfiguration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !impl->finishCheckpointStates.Allocate(
                    preparedConfiguration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !impl->finishCheckpointTicks.Allocate(
                    preparedConfiguration.evaluator.kind ==
                                    CudaSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !impl->randomStateWords.Allocate(candidates * 624u) ||
            !impl->candidateEvents.Allocate(
                    impl->materializesCandidateEvents
                            ? eventSlots : 0u) ||
            !impl->compactInputIndices.Allocate(
                    compactInputIndices.size()) ||
            !impl->compactInputOffsets.Allocate(
                    impl->compactRandomSteeringPipeline
                            ? compactInputOffsets.size()
                            : 0u) ||
            !impl->candidateInputValues.Allocate(
                    compactValueSlots) ||
            !impl->temporaryEvents.Allocate(
                    impl->needsTemporaryEvents ? eventSlots : 0u) ||
            !impl->passBaselineEvents.Allocate(
                    impl->needsPassBaselineEvents ? eventSlots : 0u) ||
            !impl->eligibleIndices.Allocate(
                    impl->needsEligibleIndices ? eventSlots : 0u) ||
            !impl->sharedEligibleIndices.Allocate(
                    sharedEligibleIndices.size()) ||
            !impl->sparseReferences.Allocate(
                    impl->sparseMutationPipeline ? eventSlots : 0u) ||
            !impl->sparseSnapshotReferences.Allocate(
                    impl->sparseMutationPipeline ? eventSlots : 0u) ||
            !impl->sparseEdits.Allocate(
                    impl->sparseMutationPipeline ? eventSlots : 0u) ||
            !impl->sparseScratchEdits.Allocate(
                    impl->sparseMutationPipeline ? eventSlots : 0u) ||
            !impl->editBacking.Allocate(
                    impl->editStorageAliasesTemporary
                            ? 0u
                            : impl->editStorageBytes) ||
            !impl->eventCounts.Allocate(candidates) ||
            !impl->mutationCounts.Allocate(candidates) ||
            !impl->statuses.Allocate(candidates) ||
            !impl->activeCandidates.Allocate(candidates) ||
            !impl->candidateSimulationKeys.Allocate(
                    prefixCandidateSlots) ||
            !impl->sortedCandidateSimulationKeys.Allocate(
                    prefixCandidateSlots) ||
            !impl->candidateSlots.Allocate(prefixCandidateSlots) ||
            !impl->sortedCandidateSlots.Allocate(
                    prefixCandidateSlots) ||
            !impl->candidateInputHashes.Allocate(
                    deduplicationCandidateSlots) ||
            !impl->sortedCandidateInputHashes.Allocate(
                    deduplicationCandidateSlots) ||
            !impl->candidateRepresentativeSlots.Allocate(
                    deduplicationCandidateSlots) ||
            !impl->prefixBestSamples.Allocate(winnerSampleSlots) ||
            !impl->reducedBest.Allocate(1u) ||
            !impl->collisionScratch.Allocate(collisionTileSlots) ||
            !impl->shapeWorldScratch.Allocate(shapeQuerySlots) ||
            !impl->movingBoundsScratch.Allocate(shapeQuerySlots) ||
            !impl->surfaceHitScratch.Allocate(surfaceHitSlots) ||
            !impl->meshRangeScratch.Allocate(meshRangeSlots) ||
            !impl->meshCellScratch.Allocate(meshCellSlots) ||
            !impl->responseOrderScratch.Allocate(collisionSlots) ||
            !impl->cancellation.Allocate() ||
            !impl->globalBestSample.Allocate(1u) ||
            !impl->globalBestState.Allocate(1u) ||
            !impl->globalBestInputs.Allocate(
                    preparedConfiguration.maximumEventCount) ||
            !impl->globalBestEventCount.Allocate(1u) ||
            !impl->globalBestMutationCount.Allocate(1u) ||
            !impl->closestTargetDistanceSquaredByBlock.Allocate(
                    summaryBlockCount) ||
            !impl->hotPathScratch.Allocate(
                    preparedConfiguration.collectHotPathMetrics
                            ? HotPathScratchBytes(
                                      preparedConfiguration.maximumBatchSize)
                            : 0u) ||
            !impl->summary.Allocate(1u)) {
            if (diagnostic != nullptr) {
                *diagnostic = "CUDA resident search allocation failed";
            }
            return {};
        }
        std::size_t winnerScanBytes = 0u;
        cudaError_t error = cub::DeviceScan::InclusiveScan(
                nullptr, winnerScanBytes,
                impl->candidateBestSamples.Get(),
                impl->prefixBestSamples.Get(),
                BetterSample{
                        MaximizesScore(
                                preparedConfiguration.evaluator.kind)},
                winnerSampleSlots);
        if (error != cudaSuccess ||
            !impl->winnerScanTemporary.Allocate(winnerScanBytes)) {
            if (diagnostic != nullptr) {
                *diagnostic = error != cudaSuccess
                        ? CudaFailure(
                                  "sizing CUDA winner scan", error)
                        : "CUDA winner scan allocation failed";
            }
            return {};
        }
        std::size_t candidateSortBytes = 0u;
        error = cudaSuccess;
        std::size_t candidateHashSortBytes = 0u;
        if (impl->baselinePrefixReuseEligible) {
            error = cub::DeviceRadixSort::SortPairs(
                    nullptr, candidateSortBytes,
                    impl->candidateSimulationKeys.Get(),
                    impl->sortedCandidateSimulationKeys.Get(),
                    impl->candidateSlots.Get(),
                    impl->sortedCandidateSlots.Get(),
                    candidates);
        }
        if (error == cudaSuccess &&
            impl->DeduplicationStorageEligible(
                    preparedConfiguration.maximumBatchSize)) {
            error = cub::DeviceRadixSort::SortPairs(
                    nullptr, candidateHashSortBytes,
                    impl->candidateInputHashes.Get(),
                    impl->sortedCandidateInputHashes.Get(),
                    impl->candidateSlots.Get(),
                    impl->sortedCandidateSlots.Get(),
                    candidates);
            if (candidateHashSortBytes > candidateSortBytes) {
                candidateSortBytes = candidateHashSortBytes;
            }
        }
        if (error != cudaSuccess ||
            !impl->candidateSortTemporary.Allocate(
                    candidateSortBytes)) {
            if (diagnostic != nullptr) {
                *diagnostic = error != cudaSuccess
                        ? CudaFailure(
                                  "sizing CUDA candidate prefix sort",
                                  error)
                        : "CUDA candidate prefix sort allocation failed";
            }
            return {};
        }
#define UPLOAD(allocation, source, label)                                    \
        do {                                                                  \
            if (!(source).empty()) {                                          \
                error = cudaMemcpy(                                           \
                        (allocation).Get(), (source).data(),                   \
                        (source).size() * sizeof((source)[0]),                 \
                        cudaMemcpyHostToDevice);                               \
                if (error != cudaSuccess) {                                   \
                    if (diagnostic != nullptr) {                              \
                        *diagnostic = CudaFailure(label, error);               \
                    }                                                         \
                    return {};                                                \
                }                                                             \
                impl->initialUploadBytes +=                                   \
                        (source).size() * sizeof((source)[0]);                 \
            }                                                                 \
        } while (false)
        error = cudaMemcpy(
                impl->branchState.Get(),
                &preparedConfiguration.branchState,
                sizeof(preparedConfiguration.branchState),
                cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        CudaFailure("uploading CUDA branch state", error);
            }
            return {};
        }
        impl->initialUploadBytes += sizeof(preparedConfiguration.branchState);
        error = cudaMemcpy(
                impl->mutableBoundaryControls.Get(),
                &inputPartition.mutableBoundaryControls,
                sizeof(inputPartition.mutableBoundaryControls),
                cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        CudaFailure("uploading CUDA branch controls", error);
            }
            return {};
        }
        impl->initialUploadBytes +=
                sizeof(inputPartition.mutableBoundaryControls);
        UPLOAD(impl->baselineTicks,
               preparedConfiguration.baselineTicks,
               "uploading CUDA baseline ticks");
        UPLOAD(impl->baselineInputs,
               preparedConfiguration.baselineInputs,
               "uploading CUDA baseline inputs");
        UPLOAD(impl->modifiers,
               preparedConfiguration.modifiers,
               "uploading CUDA modifier configuration");
        UPLOAD(impl->smoothWeights,
               preparedConfiguration.smoothWeights,
               "uploading CUDA smooth weights");
        if (preparedConfiguration.condition) {
            UPLOAD(impl->condition,
                   preparedConfiguration.condition->instructions,
                   "uploading CUDA condition program");
        }
        UPLOAD(impl->compactInputIndices,
               compactInputIndices,
               "uploading CUDA compact input indices");
        UPLOAD(impl->compactInputOffsets,
               compactInputOffsets,
               "uploading CUDA compact input offsets");
        UPLOAD(impl->sharedEligibleIndices,
               sharedEligibleIndices,
               "uploading CUDA shared eligible indices");
#undef UPLOAD
        error = cudaMemcpy(
                impl->evaluator.Get(),
                &preparedConfiguration.evaluator,
                sizeof(preparedConfiguration.evaluator),
                cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = CudaFailure(
                        "uploading CUDA evaluator configuration",
                        error);
            }
            return {};
        }
        impl->initialUploadBytes += sizeof(preparedConfiguration.evaluator);
        if (preparedConfiguration.incumbent) {
            const CudaSearchIncumbent &incumbent =
                    *preparedConfiguration.incumbent;
            DeviceSample sample;
            sample.score = incumbent.score;
            sample.timeMs = incumbent.timeMs;
            sample.detail0 = incumbent.detail0;
            sample.detail1 = incumbent.detail1;
            sample.candidateId = incumbent.candidateId;
            sample.logicalOrder = 0u;
            sample.candidateSlot = InvalidCandidateSlot;
            sample.evaluationTick = incumbent.evaluationTick;
            sample.valid = true;
            sample.mutation = incumbent.mutation;
            sample.preciseFinish = incumbent.preciseFinish;
            const std::uint32_t eventCount = static_cast<std::uint32_t>(
                    preparedConfiguration.baselineInputs.size());
            error = cudaMemcpy(
                    impl->globalBestSample.Get(), &sample, sizeof(sample),
                    cudaMemcpyHostToDevice);
            if (error == cudaSuccess) {
                error = cudaMemcpy(
                        impl->globalBestEventCount.Get(), &eventCount,
                        sizeof(eventCount), cudaMemcpyHostToDevice);
            }
            if (error == cudaSuccess) {
                error = cudaMemcpy(
                        impl->globalBestMutationCount.Get(),
                        &incumbent.mutationCount,
                        sizeof(incumbent.mutationCount),
                        cudaMemcpyHostToDevice);
            }
            if (error == cudaSuccess && eventCount != 0u) {
                error = cudaMemcpy(
                        impl->globalBestInputs.Get(),
                        impl->baselineInputs.Get(),
                        eventCount * sizeof(CudaSearchInputEvent),
                        cudaMemcpyDeviceToDevice);
            }
            impl->initialUploadBytes += sizeof(sample) +
                    sizeof(eventCount) + sizeof(incumbent.mutationCount) +
                    eventCount * sizeof(CudaSearchInputEvent);
            impl->baselineEvaluated = true;
        } else {
            error = cudaMemset(
                    impl->globalBestSample.Get(), 0,
                    impl->globalBestSample.Bytes());
            if (error == cudaSuccess) {
                error = cudaMemset(
                        impl->globalBestEventCount.Get(), 0,
                        impl->globalBestEventCount.Bytes());
            }
            if (error == cudaSuccess) {
                error = cudaMemset(
                        impl->globalBestMutationCount.Get(), 0,
                        impl->globalBestMutationCount.Bytes());
            }
        }
        if (error != cudaSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        CudaFailure("initializing CUDA search state", error);
            }
            return {};
        }

        if (!impl->LoadSimulationKernelMetrics(diagnostic)) {
            return {};
        }
        impl->UpdateResidentBytes();
        if (diagnostic != nullptr) {
            diagnostic->clear();
        }
        return std::unique_ptr<CudaSearchExecutor>(
                new CudaSearchExecutor(std::move(impl)));
    } catch (const std::bad_alloc &) {
        if (diagnostic != nullptr) {
            *diagnostic = "CUDA search host allocation failed";
        }
        return {};
    } catch (...) {
        if (diagnostic != nullptr) {
            *diagnostic = "unexpected CUDA search creation failure";
        }
        return {};
    }
}

CudaSearchBatchExecution CudaSearchExecutor::EvaluateBaseline() noexcept {
    return EvaluateBaseline(std::function<bool()>{});
}

CudaSearchBatchExecution CudaSearchExecutor::EvaluateBaseline(
        const std::function<bool()> &cancellationRequested) noexcept {
    if (!impl_ || impl_->baselineEvaluated) {
        CudaSearchBatchExecution result;
        result.status = CudaSearchStatus::InvalidArgument;
        result.diagnostic = !impl_
                ? "CUDA search executor is invalid"
                : "CUDA baseline was already evaluated";
        return result;
    }
    CudaSearchBatchExecution result =
            impl_->Execute(0u, 1u, true, cancellationRequested);
    result.hostToDeviceBytes += impl_->initialUploadBytes;
    return result;
}

CudaSearchBatchExecution CudaSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool cancellationRequested) noexcept {
    const std::function<bool()> probe = cancellationRequested
            ? std::function<bool()>([] { return true; })
            : std::function<bool()>{};
    return RunBatch(firstCandidateId, candidateCount, probe);
}

CudaSearchBatchExecution CudaSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        const std::function<bool()> &cancellationRequested) noexcept {
    if (!impl_) {
        CudaSearchBatchExecution result;
        result.status = CudaSearchStatus::InvalidArgument;
        result.diagnostic = "CUDA search executor is invalid";
        return result;
    }
    if (candidateCount != 0u &&
        firstCandidateId >
                std::numeric_limits<std::uint64_t>::max() -
                        (candidateCount - 1u)) {
        CudaSearchBatchExecution result;
        result.status = CudaSearchStatus::InvalidArgument;
        result.firstCandidateId = firstCandidateId;
        result.candidateCount = candidateCount;
        result.diagnostic = "CUDA candidate ID range overflow";
        return result;
    }
    return impl_->Execute(
            firstCandidateId, candidateCount, false,
            cancellationRequested);
}

bool CudaSearchExecutor::ReserveBatchCapacity(
        std::uint32_t candidateCount,
        std::string *diagnostic) noexcept {
    try {
        if (!impl_ || candidateCount == 0u) {
            if (diagnostic != nullptr) {
                *diagnostic = !impl_
                        ? "CUDA search executor is invalid"
                        : "CUDA batch capacity must be positive";
            }
            return false;
        }
        return impl_->ReserveBatchCapacity(candidateCount, diagnostic);
    } catch (const std::bad_alloc &) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "CUDA calibration host allocation failed";
        }
        return false;
    } catch (...) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "unexpected CUDA calibration allocation failure";
        }
        return false;
    }
}

bool CudaSearchExecutor::UpdateConditionTimes(
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

std::uint32_t CudaSearchExecutor::BatchCapacity() const noexcept {
    return impl_ ? impl_->configuration.maximumBatchSize : 0u;
}

}  // namespace forevervalidator::simulation
