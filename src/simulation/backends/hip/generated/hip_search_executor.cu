// Generated from src/simulation/backends/cuda/cuda_search_executor.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_SESSION_LTO)
#define FOREVERVALIDATOR_HIP_RESEARCH_WHEEL_FORCE_MODE 2u
#define FOREVERVALIDATOR_HIP_RESEARCH_STEADY_INTEGRATION
#define FOREVERVALIDATOR_HIP_RESEARCH_FULL_ANGULAR_DYNAMICS
#define FOREVERVALIDATOR_HIP_RESEARCH_ONE_UNIFORM_FORCE_FIELD
#define FOREVERVALIDATOR_HIP_RESEARCH_EIGHT_ROOT_SHAPES
#define FOREVERVALIDATOR_HIP_RESEARCH_FOUR_WHEELS
#define FOREVERVALIDATOR_HIP_RESEARCH_CANONICAL_WHEEL_FACTS
#define FOREVERVALIDATOR_HIP_RESEARCH_WATER_ONLY
#endif

#include "simulation/backends/hip/generated/hip_search_executor.h"
#include "simulation/backends/hip/hip_search_arch_adapter.h"
#include "simulation/backends/gpu_memory_budget.h"

#include <hip/hip_runtime.h>
#include "simulation/backends/hip/hip_reduce_adapter.h"

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

#include "simulation/backends/hip/generated/hip_candidate_events.cuh"
#include "simulation/backends/hip/generated/hip_sparse_candidate_events.cuh"
#include "simulation/backends/hip/generated/hip_exact_math.cuh"
#include "simulation/backends/hip/generated/hip_finish_time_refinement.cuh"
#include "simulation/backends/hip/generated/hip_modifier_event_ops.cuh"
#include "simulation/backends/hip/generated/hip_physics_step.cuh"
#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_scene_layout.h"
#include "simulation/backends/hip/generated/hip_session_specialization.h"
#include "simulation/backends/hip/generated/hip_search_branch_state.cuh"
#include "simulation/backends/hip/generated/hip_search_winner_selection.cuh"
#include "simulation/backends/hip/generated/hip_stunts.cuh"
#include "simulation/backends/hip/generated/hip_vehicle_transitions.cuh"

namespace forevervalidator::simulation {

#if defined(FOREVERVALIDATOR_HIP_RESEARCH_SESSION_LTO)
extern "C" __device__
HipPackedStaticConfigurationHeader
ForeverValidatorSessionConfiguration();
extern "C" __device__
HipPackedSceneHeader ForeverValidatorSessionScene();
#endif

namespace {

constexpr std::uint32_t SimulationBlockSize = 32u;
constexpr std::uint32_t ThroughputKernelMinimumBlocksPerSm = 16u;
constexpr std::uint32_t TailKernelMinimumBlocksPerSm = 17u;
constexpr std::uint32_t DenseTailKernelMinimumBlocksPerSm = 24u;
// Keep the state immediately before the most recent unfinished tick. Exact
// finish refinement then replays one tick instead of a long timeline suffix.
constexpr std::uint32_t FinishCheckpointInvalidTick = UINT32_MAX;

template<typename... Arguments, std::size_t... Indices>
hipError_t LaunchDriverKernelImpl(
        hipFunction_t function,
        std::uint32_t blocks,
        std::tuple<Arguments...> &arguments,
        std::index_sequence<Indices...>) {
    std::array<void *, sizeof...(Arguments)> pointers{
            static_cast<void *>(&std::get<Indices>(arguments))...};
    return hipModuleLaunchKernel(
            function,
            blocks, 1u, 1u,
            SimulationBlockSize, 1u, 1u,
            0u, nullptr, pointers.data(), nullptr);
}

template<typename... Arguments>
hipError_t LaunchDriverKernel(
        hipFunction_t function,
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
hipError_t LaunchRuntimeKernelImpl(
        const void *function,
        std::uint32_t blocks,
        std::tuple<Arguments...> &arguments,
        std::index_sequence<Indices...>) {
    std::array<void *, sizeof...(Arguments)> pointers{
            static_cast<void *>(&std::get<Indices>(arguments))...};
    return hipLaunchKernel(reinterpret_cast<const void*>(
            function),
            dim3(blocks, 1u, 1u),
            dim3(SimulationBlockSize, 1u, 1u),
            pointers.data(),
            0u,
            nullptr);
}

template<typename... Arguments>
hipError_t LaunchRuntimeKernel(
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

using hip_search_detail::BetterSample;
using hip_search_detail::ControlsFromState;
using hip_search_detail::DeviceControlState;
using hip_search_detail::DeviceSample;
using hip_search_detail::InvalidCandidateSlot;
using hip_search_detail::ApplyControlEvent;
using hip_search_detail::StuntsFromState;
using hip_search_detail::StrictlyBetter;
namespace modifier_ops = hip_search_modifier_detail;
namespace sparse_events = hip::sparse_candidate_events;

struct DeviceBatchSummary {
    HipSearchStatus status = HipSearchStatus::Success;
    std::uint32_t evaluatedCandidateCount = 0u;
    std::uint64_t evaluatorCalls = 0u;
    std::uint64_t totalMutationCount = 0u;
    std::uint64_t mutationImprovementCount = 0u;
    std::uint32_t globalEventCount = 0u;
    bool bestChanged = false;
    bool bestValid = false;
    bool bestMutation = false;
    std::uint64_t bestCandidateId = 0u;
    std::uint32_t bestMutationCount = 0u;
};

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
        std::size_t available = 0, total = 0;
        if (hipMemGetInfo(&available, &total) != hipSuccess ||
            !GpuAllocationFitsBudget(count * sizeof(T), available, total)) {
            return false;
        }
        if (hipMalloc(
                    reinterpret_cast<void **>(&data_),
                    count * sizeof(T)) != hipSuccess) {
            data_ = nullptr;
            return false;
        }
        count_ = count;
        return true;
    }

    void Reset() {
        if (data_ != nullptr) {
            hipFree(data_);
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
        if (hipHostAlloc(
                    reinterpret_cast<void **>(&host_),
                    sizeof(*host_),
                    hipHostMallocMapped | hipHostMallocPortable) !=
            hipSuccess) {
            return false;
        }
        if (hipHostGetDevicePointer(
                    reinterpret_cast<void **>(&device_),
                    host_, 0u) != hipSuccess) {
            Reset();
            return false;
        }
        *host_ = 0u;
        return true;
    }

    void Reset() {
        if (host_ != nullptr) {
            hipHostFree(host_);
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
        valid_ = hipEventCreate(&event_) == hipSuccess;
    }
    ~Event() {
        if (valid_) {
            hipEventDestroy(event_);
        }
    }
    bool Valid() const { return valid_; }
    hipEvent_t Get() const { return event_; }

private:
    hipEvent_t event_{};
    bool valid_ = false;
};

std::string HipFailure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
            hipGetErrorName(error) + " (" +
            hipGetErrorString(error) + ")";
}

bool CanonicalBaselineInputs(
        const std::vector<HipSearchInputEvent> &inputs,
        std::int64_t mutableFromTimeMs) {
    bool mutableInputSeen = false;
    std::int32_t previousMutableTime = INT32_MIN;
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const HipSearchInputEvent &input = inputs[index];
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
        const HipSearchExecutorConfiguration &configuration) {
    std::uint64_t baselineEdits = 0u;
    std::uint64_t insertedEdits = 0u;
    const auto baselineMatches =
            [&](const HipSearchModifierConfiguration &modifier,
                const auto &selected) {
                return static_cast<std::uint64_t>(std::count_if(
                        configuration.baselineInputs.begin(),
                        configuration.baselineInputs.end(),
                        [&](const HipSearchInputEvent &event) {
                            return event.timeMs >=
                                            modifier.window.minimumTimeMs &&
                                    event.timeMs <=
                                            modifier.window.maximumTimeMs &&
                                    selected(event);
                        }));
            };
    for (const HipSearchModifierConfiguration &modifier :
         configuration.modifiers) {
        switch (modifier.kind) {
        case HipSearchModifierKind::RandomSteering:
            baselineEdits += baselineMatches(
                    modifier,
                    [](const HipSearchInputEvent &event) {
                        return event.action == 4u &&
                                event.valueKind == 2u;
                    });
            break;
        case HipSearchModifierKind::ExistingEvent:
            baselineEdits += modifier.maximumCount;
            break;
        case HipSearchModifierKind::SmoothSteering: {
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
        case HipSearchModifierKind::InputInsertion:
            for (const HipSearchChannel *channel :
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
        case HipSearchModifierKind::InputDeletion:
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

__device__ bool IsAnalog(const HipSearchInputEvent &event) {
    return event.valueKind == 2u;
}

__device__ bool IsSwitch(const HipSearchInputEvent &event) {
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

__device__ bool SameEvent(const HipSearchInputEvent &left,
                          const HipSearchInputEvent &right) {
    return left.timeMs == right.timeMs &&
            left.action == right.action &&
            left.valueKind == right.valueKind &&
            left.value == right.value;
}

__device__ HipSearchInputEvent CandidateInputAt(
        const HipSearchInputEvent *baselineInputs,
        const HipSearchInputEvent *materializedInputs,
        const std::int32_t *compactValues,
        const std::uint32_t *compactOffsets,
        bool compact,
        std::uint32_t index,
        std::uint32_t candidateSlot,
        std::uint32_t candidateStride) {
    if (!compact) {
        return materializedInputs[index];
    }
    HipSearchInputEvent result = baselineInputs[index];
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
            const HipSearchInputEvent *baselineInputs,
            std::uint32_t baselineInputCount,
            const HipSearchInputEvent *materializedInputs,
            const std::int32_t *compactValues,
            const std::uint32_t *compactOffsets,
            bool compactRandom,
            bool compactEdits,
            bool sparseEvents,
            hip::candidate_events::CoalescedEditStorage edits,
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

    __device__ bool Next(HipSearchInputEvent *event) {
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
    const HipSearchInputEvent *baselineInputs_ = nullptr;
    const HipSearchInputEvent *materializedInputs_ = nullptr;
    const std::int32_t *compactValues_ = nullptr;
    const std::uint32_t *compactOffsets_ = nullptr;
    bool compactRandom_ = false;
    bool compactEdits_ = false;
    bool sparseEvents_ = false;
    std::uint32_t candidateSlot_ = 0u;
    std::uint32_t candidateStride_ = 0u;
    std::uint32_t finalCount_ = 0u;
    std::uint32_t index_ = 0u;
    hip::candidate_events::CandidateCursor editCursor_;
    sparse_events::Cursor sparseCursor_;
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
        HipSearchInputEvent *events,
        std::uint32_t count,
        HipSearchInputEvent *temporary,
        const HipSearchInputEvent *passBaseline,
        std::uint32_t passBaselineCount,
        std::int64_t mutableFromTimeMs,
        bool legacyMutationPipeline,
        std::uint32_t capacity) {
    (void)legacyMutationPipeline;
    return hip::candidate_events::NormalizeWithPrefix(
            events, count, temporary,
            passBaseline, passBaselineCount,
            mutableFromTimeMs, capacity);
}

__device__ std::uint32_t EffectiveChangeCount(
        const HipSearchInputEvent *baseline,
        std::uint32_t baselineCount,
        const HipSearchInputEvent *events,
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
        const HipSearchInputEvent *baseline,
        std::uint32_t baselineCount,
        const HipSearchInputEvent *events,
        std::uint32_t eventCount,
        std::uint32_t *sourceStates,
        hip::candidate_events::CoalescedEditStorage storage,
        std::uint32_t slot) {
    storage.counts[slot] = 0u;
    storage.erasedCounts[slot] = 0u;
    for (std::uint32_t source = 0u;
         source < baselineCount; ++source) {
        sourceStates[source] = 0u;
    }

    hip::candidate_events::EditWriter writer(storage, slot);
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
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::int64_t timeMs,
        std::int32_t initialState,
        bool sortedByTime = false) {
    return modifier_ops::ChannelStateAt(
            events, count, 4u, 2u, timeMs,
            sortedByTime, initialState);
}

__device__ bool SwitchStateAt(
        const HipSearchInputEvent *events,
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

__device__ bool PushEvent(HipSearchInputEvent *events,
                          std::uint32_t *count,
                          std::uint32_t capacity,
                          HipSearchInputEvent event) {
    if (*count >= capacity) {
        return false;
    }
    events[*count] = event;
    ++*count;
    return true;
}

__device__ HipSearchInputEvent AnalogEvent(
        std::int64_t timeMs,
        std::uint32_t action,
        std::int32_t value) {
    return {static_cast<std::int32_t>(timeMs), action, 2u, value};
}

__device__ HipSearchInputEvent SwitchEvent(
        std::int64_t timeMs,
        std::uint32_t action,
        bool value) {
    return {static_cast<std::int32_t>(timeMs),
            action,
            1u,
            value ? 1 : 0};
}

__device__ std::uint32_t CollectEligible(
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::uint32_t *eligible,
        const HipSearchModifierConfiguration &modifier,
        bool sortedByTime) {
    return modifier_ops::CollectExistingEventEligible(
            events, count, eligible,
            modifier.window.minimumTimeMs,
            modifier.window.maximumTimeMs,
            modifier.optionFlags, sortedByTime);
}

__device__ bool ApplyModifier(
        const HipSearchModifierConfiguration &modifier,
        std::uint32_t passIndex,
        std::uint64_t candidateId,
        DeviceMt19937 &random,
        std::uint32_t tickDurationMs,
        std::int64_t mutableFromTimeMs,
        const DeviceControlState &initialControls,
        const HipSearchInputEvent *globalBaseline,
        std::uint32_t globalBaselineCount,
        HipSearchInputEvent *events,
        std::uint32_t *eventCount,
        std::uint32_t eventCapacity,
        HipSearchInputEvent *temporary,
        HipSearchInputEvent *passBaseline,
        std::uint32_t *eligible,
        const double *smoothWeights,
        bool legacyMutationPipeline,
        bool *normalized) {
    const std::uint32_t passBaselineCount = *eventCount;
    const bool passBaselineCanonical = *normalized;
    const bool insertionRestoresHeldState =
            modifier.kind == HipSearchModifierKind::InputInsertion &&
            ((modifier.steering.enabled != 0u &&
              modifier.steering.maximumHoldMs > 0) ||
             (modifier.accelerate.enabled != 0u &&
              modifier.accelerate.maximumHoldMs > 0) ||
             (modifier.brake.enabled != 0u &&
              modifier.brake.maximumHoldMs > 0));
    const bool needsPassSnapshot =
            legacyMutationPipeline ||
            insertionRestoresHeldState ||
            (modifier.kind == HipSearchModifierKind::InputInsertion &&
             (modifier.optionFlags & 2u) != 0u) ||
            modifier.window.minimumTimeMs < mutableFromTimeMs;
    if (needsPassSnapshot) {
        for (std::uint32_t index = 0u;
             index < passBaselineCount; ++index) {
            passBaseline[index] = events[index];
        }
    }
    const HipSearchInputEvent *normalizationBaseline =
            needsPassSnapshot ? passBaseline : globalBaseline;
    const std::uint32_t normalizationBaselineCount =
            needsPassSnapshot ? passBaselineCount : globalBaselineCount;
    random.Seed(modifier.window.seed, candidateId, passIndex);

    switch (modifier.kind) {
    case HipSearchModifierKind::RandomSteering: {
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
            HipSearchInputEvent &event = events[index];
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
    case HipSearchModifierKind::ExistingEvent: {
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
            HipSearchInputEvent &event = events[eligible[index]];
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
    case HipSearchModifierKind::SmoothSteering:
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
    case HipSearchModifierKind::InputInsertion: {
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
                [&](const HipSearchChannel &channel,
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
        if ((modifier.optionFlags & 2u) != 0u && passBaselineCanonical) {
            std::int32_t held = initialControls.steerValue;
            std::uint32_t source = 0u;
            std::uint32_t retained = 0u;
            for (std::uint32_t index = 0u; index < *eventCount; ++index) {
                const HipSearchInputEvent event = events[index];
                while (source < passBaselineCount && passBaseline[source].timeMs < event.timeMs) ++source;
                bool originalEvent = false;
                for (std::uint32_t match = source; match < passBaselineCount &&
                     passBaseline[match].timeMs == event.timeMs; ++match) {
                    originalEvent = originalEvent || SameEvent(passBaseline[match], event);
                }
                if (event.action == 4u && IsAnalog(event)) {
                    if (!originalEvent && event.value == held) continue;
                    held = event.value;
                }
                events[retained++] = event;
            }
            *eventCount = retained;
        }
        *normalized = true;
        break;
    }
    case HipSearchModifierKind::InputDeletion: {
        const auto deleteChannel =
                [&](const HipSearchChannel &channel,
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
        const HipSearchInputEvent &event,
        const HipSearchModifierConfiguration &modifier) {
    return event.timeMs >= modifier.window.minimumTimeMs &&
            event.timeMs <= modifier.window.maximumTimeMs &&
            ((event.action == 4u && event.valueKind == 2u) ||
             ((modifier.optionFlags & 2u) != 0u &&
              (event.action == 1u || event.action == 2u)) ||
             ((modifier.optionFlags & 4u) != 0u &&
              event.action == 3u));
}

__device__ bool SparseDeletionEligible(
        const HipSearchInputEvent &event,
        const HipSearchModifierConfiguration &modifier,
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
        const HipSearchInputEvent *baseline,
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
        const HipSearchModifierConfiguration &modifier,
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
    case HipSearchModifierKind::RandomSteering:
        for (std::uint32_t index = 0u;
             index < candidate->Count(); ++index) {
            HipSearchInputEvent event = candidate->EventAt(index);
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
    case HipSearchModifierKind::ExistingEvent: {
        const std::uint32_t eligibleCount = CollectSparseEligible(
                *candidate, eligible, sharedEligible,
                sharedEligibleCount, identityReferences,
                [&](const HipSearchInputEvent &event) {
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
            HipSearchInputEvent event = candidate->EventAt(ordinal);
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
    case HipSearchModifierKind::SmoothSteering:
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
    case HipSearchModifierKind::InputInsertion: {
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
                const HipSearchInputEvent startEvent =
                        AnalogEvent(start, 4u, value);
                const HipSearchInputEvent endEvent =
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
                [&](const HipSearchChannel &channel,
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
                        const HipSearchInputEvent startEvent =
                                SwitchEvent(start, action, !previous);
                        const HipSearchInputEvent endEvent =
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
        if ((modifier.optionFlags & 2u) != 0u) {
            std::int32_t held = initialControls.steerValue;
            std::uint32_t source = 0u;
            std::uint32_t removed = 0u;
            for (std::uint32_t index = 0u; index < candidate->Count(); ++index) {
                const HipSearchInputEvent event = candidate->EventAt(index);
                while (source < snapshotCount && candidate->SnapshotEventAt(source).timeMs < event.timeMs) ++source;
                bool originalEvent = false;
                for (std::uint32_t match = source; match < snapshotCount &&
                     candidate->SnapshotEventAt(match).timeMs == event.timeMs; ++match) {
                    originalEvent = originalEvent || SameEvent(candidate->SnapshotEventAt(match), event);
                }
                if (event.action == 4u && IsAnalog(event)) {
                    if (!originalEvent && event.value == held) {
                        eligible[removed++] = index;
                        continue;
                    }
                    held = event.value;
                }
            }
            candidate->EraseSortedOrdinals(eligible, removed);
        }
        break;
    }
    case HipSearchModifierKind::InputDeletion: {
        bool sharedIdentityReferences = identityReferences;
        const auto deleteChannel =
                [&](const HipSearchChannel &channel,
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
                                    [&](const HipSearchInputEvent &event) {
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

__device__ void ApplyControlPrefix(HipCandidatePhysicsState &state,
                                   const HipControlTick &tick) {
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
            static_cast<const HipPackedSceneHeader *>(sceneData);
    const auto *configuration =
            static_cast<const HipPackedStaticConfigurationHeader *>(
                    configurationData);
    return scene->magic == HipPackedSceneHeader::Magic &&
            scene->schemaVersion == HipPackedSceneHeader::SchemaVersion &&
            configuration->magic ==
                    HipPackedStaticConfigurationHeader::Magic &&
            configuration->schemaVersion ==
                    HipPackedStaticConfigurationHeader::SchemaVersion;
}

__device__ bool ContainsVolume(
        const HipSearchEvaluatorConfiguration &evaluator,
        const GmVec3 &position) {
    return static_cast<double>(position.x) >= evaluator.values[0] &&
            static_cast<double>(position.y) >= evaluator.values[1] &&
            static_cast<double>(position.z) >= evaluator.values[2] &&
            static_cast<double>(position.x) <= evaluator.values[3] &&
            static_cast<double>(position.y) <= evaluator.values[4] &&
            static_cast<double>(position.z) <= evaluator.values[5];
}

__device__ bool SegmentEntry(
        const HipSearchEvaluatorConfiguration &evaluator,
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
    const double sinPitch = 2.0 *
            (static_cast<double>(q.w) * q.x -
             static_cast<double>(q.y) * q.z);
    const double sinYaw = 2.0 *
            (static_cast<double>(q.w) * q.y +
             static_cast<double>(q.x) * q.z);
    const double cosYaw = 1.0 - 2.0 *
            (static_cast<double>(q.x) * q.x +
             static_cast<double>(q.y) * q.y);
    const double sinRoll = 2.0 *
            (static_cast<double>(q.w) * q.z +
             static_cast<double>(q.x) * q.y);
    const double cosRoll = 1.0 - 2.0 *
            (static_cast<double>(q.x) * q.x +
             static_cast<double>(q.z) * q.z);
    const double pitch = fabs(sinPitch) >= 1.0
            ? copysign(1.57079632679489661923, sinPitch)
            : asin(sinPitch);
    return {atan2(sinYaw, cosYaw), pitch,
            atan2(sinRoll, cosRoll), true};
}

__device__ DeviceConditionValue ConditionSource(
        HipSearchConditionValue source,
        const HipCandidatePhysicsState &state,
        std::uint64_t iterationCount,
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
    case HipSearchConditionValue::Position: return vector(current.position);
    case HipSearchConditionValue::PreviousPosition: return vector(previous.position);
    case HipSearchConditionValue::Velocity: return vector(current.linearSpeed);
    case HipSearchConditionValue::PreviousVelocity: return vector(previous.linearSpeed);
    case HipSearchConditionValue::LocalVelocity: return vector(currentLocal);
    case HipSearchConditionValue::PreviousLocalVelocity: return vector(previousLocal);
    case HipSearchConditionValue::AngularVelocity: return vector(current.angularSpeed);
    case HipSearchConditionValue::PreviousAngularVelocity: return vector(previous.angularSpeed);
    case HipSearchConditionValue::Yaw: return {ConditionAngles(current.rotationQuat).x};
    case HipSearchConditionValue::Pitch: return {ConditionAngles(current.rotationQuat).y};
    case HipSearchConditionValue::Roll: return {ConditionAngles(current.rotationQuat).z};
    case HipSearchConditionValue::PreviousYaw: return {ConditionAngles(previous.rotationQuat).x};
    case HipSearchConditionValue::PreviousPitch: return {ConditionAngles(previous.rotationQuat).y};
    case HipSearchConditionValue::PreviousRoll: return {ConditionAngles(previous.rotationQuat).z};
    case HipSearchConditionValue::Speed: return {ConditionLength(current.linearSpeed)};
    case HipSearchConditionValue::PreviousSpeed: return {ConditionLength(previous.linearSpeed)};
    case HipSearchConditionValue::LocalSpeed: return {ConditionLength(currentLocal)};
    case HipSearchConditionValue::PreviousLocalSpeed: return {ConditionLength(previousLocal)};
    case HipSearchConditionValue::FreeWheeling: return {state.vehicle.controls.forcedLowSpeedFriction ? 1.0 : 0.0};
    case HipSearchConditionValue::LateralContact: return {state.vehicle.contacts.lateralSlowDownContactActive ? 1.0 : 0.0};
    case HipSearchConditionValue::Sliding: {
        bool sliding = false;
        for (std::uint32_t i = 0u; i < hip::facts::WheelCount(state.vehicle); ++i) {
            sliding = sliding || (state.vehicle.wheels.values[i].realTime.contactPresent &&
                                  state.vehicle.wheels.values[i].realTime.slipping);
        }
        return {sliding ? 1.0 : 0.0};
    }
    case HipSearchConditionValue::Gear:
        return {state.vehicle.engine.useLowSpeedGateB ? -1.0 :
                static_cast<double>(state.vehicle.engine.gearIndex)};
    case HipSearchConditionValue::Rpm: return {state.vehicle.engine.engineInputMemory};
    case HipSearchConditionValue::TurningRate: return {state.vehicle.radiusSteering.steerAngle};
    case HipSearchConditionValue::TurboType: return {static_cast<double>(state.vehicle.turbo.type)};
    case HipSearchConditionValue::TurboBoostFactor:
        return {state.vehicle.turbo.type == CSceneVehicleCar::ETurboType_Roulette
                ? static_cast<double>(state.vehicle.turbo.type2Phase) + 1.0
                : state.vehicle.turbo.impulseScale};
    case HipSearchConditionValue::Iterations: return {static_cast<double>(iterationCount)};
    case HipSearchConditionValue::LastImprovementTime: return {lastImprovementTimeSeconds};
    case HipSearchConditionValue::LastRestartTime: return {lastRestartTimeSeconds};
    case HipSearchConditionValue::CurrentTime: return {currentTimeSeconds};
    case HipSearchConditionValue::CompletedLaps:
        return {static_cast<double>(state.race.progress.completedLapCount)};
    case HipSearchConditionValue::CheckpointCount:
        return {static_cast<double>(state.race.progress.checkpointCount)};
    default: break;
    }
    const std::uint32_t raw = static_cast<std::uint32_t>(source);
    const std::uint32_t ground0 = static_cast<std::uint32_t>(HipSearchConditionValue::WheelGroundContact0);
    const std::uint32_t sliding0 = static_cast<std::uint32_t>(HipSearchConditionValue::WheelSliding0);
    const std::uint32_t surface0 = static_cast<std::uint32_t>(HipSearchConditionValue::WheelSurface0);
    std::uint32_t wheel = 0u;
    if (raw >= ground0 && raw < ground0 + 4u) {
        wheel = raw - ground0;
        return {wheel < hip::facts::WheelCount(state.vehicle) &&
                state.vehicle.wheels.values[wheel].realTime.contactPresent ? 1.0 : 0.0};
    }
    if (raw >= sliding0 && raw < sliding0 + 4u) {
        wheel = raw - sliding0;
        const auto &value = state.vehicle.wheels.values[wheel].realTime;
        return {wheel < hip::facts::WheelCount(state.vehicle) &&
                value.contactPresent && value.slipping ? 1.0 : 0.0};
    }
    wheel = raw - surface0;
    if (raw >= surface0 && raw < surface0 + 4u &&
        wheel < hip::facts::WheelCount(state.vehicle)) {
        const auto &value = state.vehicle.wheels.values[wheel].realTime;
        return {value.contactPresent
                ? static_cast<double>(value.contactMaterial)
                : 65535.0};
    }
    return {};
}

struct DeviceExpressionResult {
    double value = 0.0;
    bool valid = false;
};

__device__ __noinline__ DeviceExpressionResult EvaluateExpression(
        const HipSearchConditionInstruction *instructions,
        std::uint32_t instructionCount,
        const HipCandidatePhysicsState &state,
        std::uint64_t iterationCount,
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds,
        double currentTimeSeconds) {
    if (instructionCount == 0u) {
        return {1.0, true};
    }
    DeviceConditionValue stack[32];
    std::uint32_t size = 0u;
    for (std::uint32_t index = 0u; index < instructionCount; ++index) {
        const HipSearchConditionInstruction instruction = instructions[index];
        if (instruction.opcode == HipSearchConditionOpcode::Constant) {
            if (size >= 32u) return {};
            stack[size++] = {instruction.x};
            continue;
        }
        if (instruction.opcode == HipSearchConditionOpcode::ConstantVector) {
            if (size >= 32u) return {};
            stack[size++] = {instruction.x, instruction.y, instruction.z, true};
            continue;
        }
        if (instruction.opcode == HipSearchConditionOpcode::Scalar ||
            instruction.opcode == HipSearchConditionOpcode::Vector) {
            if (size >= 32u) return {};
            DeviceConditionValue value = ConditionSource(
                    instruction.value, state, iterationCount,
                    lastImprovementTimeSeconds,
                    lastRestartTimeSeconds, currentTimeSeconds);
            if (instruction.opcode == HipSearchConditionOpcode::Scalar &&
                value.vector) {
                const int component = static_cast<int>(instruction.x);
                value = {component == 1 ? value.x
                         : component == 2 ? value.y
                         : component == 3 ? value.z : 0.0};
            }
            if ((instruction.opcode == HipSearchConditionOpcode::Scalar &&
                 value.vector) ||
                (instruction.opcode == HipSearchConditionOpcode::Vector &&
                 !value.vector)) return {};
            stack[size++] = value;
            continue;
        }
        if (instruction.opcode == HipSearchConditionOpcode::KilometersPerHour ||
            instruction.opcode == HipSearchConditionOpcode::Degrees) {
            if (size == 0u || stack[size - 1u].vector) return {};
            stack[size - 1u].x *= instruction.opcode ==
                    HipSearchConditionOpcode::KilometersPerHour
                    ? 3.6 : 57.2957795130823208768;
            continue;
        }
        if (size < 2u) return {};
        const DeviceConditionValue right = stack[--size];
        DeviceConditionValue &left = stack[size - 1u];
        switch (instruction.opcode) {
        case HipSearchConditionOpcode::Distance:
            if (!left.vector || !right.vector) return {};
            left = {sqrt((left.x-right.x)*(left.x-right.x) +
                         (left.y-right.y)*(left.y-right.y) +
                         (left.z-right.z)*(left.z-right.z))};
            break;
        case HipSearchConditionOpcode::Add: left.x += right.x; break;
        case HipSearchConditionOpcode::Subtract: left.x -= right.x; break;
        case HipSearchConditionOpcode::Multiply: left.x *= right.x; break;
        case HipSearchConditionOpcode::Divide: left.x = right.x == 0.0 ? 0.0 : left.x / right.x; break;
        case HipSearchConditionOpcode::Greater: left = {left.x > right.x ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::Less: left = {left.x < right.x ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::GreaterOrEqual: left = {left.x >= right.x ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::LessOrEqual: left = {left.x <= right.x ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::Equal: left = {left.x == right.x ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::NotEqual: left = {left.x != right.x ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::LogicalAnd: left = {left.x != 0.0 && right.x != 0.0 ? 1.0 : 0.0}; break;
        case HipSearchConditionOpcode::LogicalOr: left = {left.x != 0.0 || right.x != 0.0 ? 1.0 : 0.0}; break;
        default: return {};
        }
    }
    if (size != 1u || stack[0].vector) return {};
    return {stack[0].x, true};
}

__device__ bool EvaluateCondition(
        const HipSearchConditionInstruction *instructions,
        std::uint32_t instructionCount,
        const HipCandidatePhysicsState &state,
        std::uint64_t iterationCount,
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds,
        double currentTimeSeconds) {
    const DeviceExpressionResult result = EvaluateExpression(
            instructions, instructionCount, state, iterationCount,
            lastImprovementTimeSeconds, lastRestartTimeSeconds,
            currentTimeSeconds);
    return result.valid && result.value != 0.0;
}

__device__ bool UpdateScriptedSample(
        const HipSearchEvaluatorConfiguration &evaluator,
        const HipCandidatePhysicsState &state,
        std::uint64_t iterationCount,
        double lastImprovementTimeSeconds,
        double lastRestartTimeSeconds,
        double currentTimeSeconds,
        double currentTimeMs,
        DeviceSample *sample) {
    double values[16];
    double scores[16];
    for (std::uint32_t i = 0u;
         i < evaluator.scriptedObjectiveCount; ++i) {
        const HipSearchScriptedObjective objective =
                evaluator.scriptedObjectives[i];
        const DeviceExpressionResult expression = EvaluateExpression(
                &evaluator.scriptedInstructions[
                        objective.firstInstruction],
                objective.instructionCount, state, iterationCount,
                lastImprovementTimeSeconds,
                lastRestartTimeSeconds, currentTimeSeconds);
        values[i] = expression.value;
        if (!expression.valid || !isfinite(values[i])) return false;
        scores[i] = objective.kind == 0u
                ? -values[i]
                : objective.kind == 1u
                  ? values[i]
                  : -fabs(values[i] - objective.target);
        if (!isfinite(scores[i])) return false;
    }
    for (std::uint32_t i = 0u;
         i < evaluator.scriptedObjectiveCount; ++i) {
        if (!sample->valid ||
            scores[i] > sample->objectiveScores[i]) {
            sample->objectiveScores[i] = scores[i];
            sample->metricValues[i] = values[i];
        }
    }
    sample->valid = true;
    sample->scriptedObjectiveCount =
            evaluator.scriptedObjectiveCount;
    sample->score = sample->objectiveScores[0];
    sample->timeMs = currentTimeMs;
    return true;
}

__device__ DeviceSample EvaluateState(
        const HipSearchEvaluatorConfiguration &evaluator,
        const HipCandidatePhysicsState &state,
        const GmVec3 &previousPosition,
        double previousTimeMs,
        double currentTimeMs,
        std::uint32_t stuntsScore,
        bool *reported) {
    DeviceSample result;
    result.timeMs = currentTimeMs;
    const GmVec3 &position = state.body.current.position;
    switch (evaluator.kind) {
    case HipSearchEvaluatorKind::Velocity: {
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
    case HipSearchEvaluatorKind::Point: {
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
    case HipSearchEvaluatorKind::Pose: {
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
    case HipSearchEvaluatorKind::VolumeEntry:
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
    case HipSearchEvaluatorKind::StuntPoints:
        result.score = static_cast<double>(stuntsScore);
        result.valid = true;
        break;
    case HipSearchEvaluatorKind::FinishTime:
        if (*reported || !state.race.progress.raceCompleted) {
            return result;
        }
        *reported = true;
        result.timeMs =
                state.race.progress.lastPrepareTimeMs;
        result.score = result.timeMs;
        result.valid = true;
        break;
    case HipSearchEvaluatorKind::ConditionTimeEarliest:
    case HipSearchEvaluatorKind::ConditionTimeLatest:
        // Only reached once the search conditions hold; the first such tick
        // is the candidate's score.
        if (*reported) {
            return result;
        }
        *reported = true;
        result.timeMs = currentTimeMs;
        result.score = currentTimeMs;
        result.valid = true;
        break;
    }
    return result;
}

// Race counters captured before a tick, enough to name every checkpoint and
// finish the tick accepts.
struct CheckpointProgressBefore {
    std::uint32_t lapCheckpoints = 0u;
    std::uint32_t events = 0u;
    std::uint32_t laps = 0u;
    std::uint32_t finishes = 0u;
};

__device__ std::uint32_t CheckpointSlotOfBlock(
        const HipPackedSceneHeader *scene,
        std::uint32_t raceBlockId) {
    const HipSceneActor *actors =
            hip::collision::detail::SceneSection<HipSceneActor>(
                    scene, scene->actors);
    for (std::uint32_t index = 0u; index < scene->actors.count; ++index) {
        if (actors[index].raceBlockId == raceBlockId &&
            actors[index].checkpointSlot != UINT32_MAX) {
            return actors[index].checkpointSlot;
        }
    }
    return UINT32_MAX;
}

// Whether this tick accepted the configured checkpoint or finish. Events in a
// tick are accepted in order: the rest of the current lap's checkpoints, at
// most one finish, then the next lap's checkpoints. Ordinals, laps and global
// event indices follow from the counters; a map slot is known for the finish
// and for the tick's last accepted checkpoint (its block).
__device__ bool CheckpointEventAccepted(
        const HipSearchEvaluatorConfiguration &evaluator,
        const HipCandidatePhysicsState &state,
        const CheckpointProgressBefore &before,
        const HipPackedSceneHeader *scene) {
    const ReplayRaceProgress &after = state.race.progress;
    if (after.totalCheckpointEventCount == before.events &&
        after.finishCount == before.finishes) {
        return false;
    }
    const bool finishTarget = evaluator.values[0] != 0.0;
    const std::uint32_t ordinal =
            static_cast<std::uint32_t>(evaluator.values[1]);
    const std::uint32_t lap = static_cast<std::uint32_t>(evaluator.values[2]);
    const bool anySlot = evaluator.values[3] < 0.0;
    const std::uint32_t slot =
            anySlot ? 0u : static_cast<std::uint32_t>(evaluator.values[3]);
    const std::uint64_t eventIndex =
            static_cast<std::uint64_t>(evaluator.values[4]) |
            (static_cast<std::uint64_t>(evaluator.values[5]) << 32u);
    const std::uint32_t required = after.requiredCheckpointCount;
    const std::uint32_t finishes = after.finishCount - before.finishes;
    if (finishes > 1u) {
        return false;
    }
    const bool shortcut =
            state.race.replayPlayMode ==
            static_cast<std::uint32_t>(EChallengePlayMode::Shortcut);
    const std::uint32_t lapA = before.laps + 1u;
    const std::uint32_t endA =
            finishes != 0u ? required : after.currentLapCheckpointCount;
    if (endA < before.lapCheckpoints) {
        return false;
    }
    const std::uint32_t countA = endA - before.lapCheckpoints;
    const std::uint32_t countB =
            finishes != 0u && !after.raceCompleted
            ? after.currentLapCheckpointCount : 0u;
    const std::uint32_t finishEvents = finishes != 0u && !shortcut ? 1u : 0u;
    if (after.totalCheckpointEventCount - before.events !=
        countA + finishEvents + countB) {
        return false;
    }
    std::uint64_t index = 0u;
    bool lastCheckpoint = false;
    if (finishTarget) {
        if (finishes == 0u || lap != lapA ||
            (!anySlot && slot != required)) {
            return false;
        }
        index = static_cast<std::uint64_t>(before.events) + countA + 1u;
        return eventIndex == 0u || eventIndex == index;
    }
    if (lap == lapA && ordinal >= before.lapCheckpoints && ordinal < endA) {
        index = static_cast<std::uint64_t>(before.events) + 1u +
                (ordinal - before.lapCheckpoints);
        lastCheckpoint = countB == 0u && ordinal + 1u == endA;
    } else if (countB != 0u && lap == lapA + 1u && ordinal < countB) {
        index = static_cast<std::uint64_t>(before.events) + countA + 2u +
                ordinal;
        lastCheckpoint = ordinal + 1u == countB;
    } else {
        return false;
    }
    if (eventIndex != 0u && eventIndex != index) {
        return false;
    }
    if (anySlot) {
        return true;
    }
    return lastCheckpoint &&
            CheckpointSlotOfBlock(scene, after.lastAcceptedBlockId) == slot;
}

__host__ __device__ bool MaximizesScore(
        HipSearchEvaluatorKind kind) {
    return kind == HipSearchEvaluatorKind::Velocity ||
            kind == HipSearchEvaluatorKind::StuntPoints ||
            kind == HipSearchEvaluatorKind::ConditionTimeLatest;
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

__global__ void SelectScriptedWinnerKernel(
        const DeviceSample *samples,
        std::uint32_t candidateCount,
        DeviceSample *winner) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    DeviceSample incumbent = samples[0];
    for (std::uint32_t slot = 0u; slot < candidateCount; ++slot) {
        if (StrictlyDominates(samples[slot + 1u], incumbent)) {
            incumbent = samples[slot + 1u];
        }
    }
    *winner = incumbent;
}

__global__ void GenerateSearchCandidatesKernel(
        const HipSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        std::uint32_t immutableTailInputCount,
        const HipSearchModifierConfiguration *modifiers,
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
        HipSearchInputEvent *candidateEvents,
        HipSearchInputEvent *temporaryEvents,
        HipSearchInputEvent *passBaselineEvents,
        std::uint32_t *eligibleIndices,
        const std::uint32_t *sharedEligibleIndices,
        std::uint32_t sharedEligibleCount,
        hip::candidate_events::CoalescedEditStorage candidateEdits,
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
    HipSearchInputEvent *events =
            candidateEvents == nullptr
            ? nullptr
            : candidateEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    HipSearchInputEvent *temporary =
            temporaryEvents == nullptr
            ? nullptr
            : temporaryEvents +
                      static_cast<std::uint64_t>(slot) * eventCapacity;
    HipSearchInputEvent *passBaseline =
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
                    HipSearchModifierKind::RandomSteering;
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
                const HipSearchModifierConfiguration modifier =
                        modifiers[pass];
                random.Seed(
                        modifier.window.seed, candidateId, pass);
                for (std::uint32_t compactIndex = 0u;
                     compactIndex < compactInputCount;
                     ++compactIndex) {
                    const HipSearchInputEvent &input =
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
                const HipSearchModifierConfiguration modifier =
                        modifiers[0];
                random.Seed(
                        modifier.window.seed, candidateId, 0u);
                sparseCandidate.BeginInitialize();
                for (std::uint32_t index = 0u;
                     index < baselineInputCount; ++index) {
                    HipSearchInputEvent event =
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
                const HipSearchModifierConfiguration modifier =
                        modifiers[pass];
                if ((modifier.kind ==
                             HipSearchModifierKind::ExistingEvent &&
                     modifier.timeParameterMs != 0) ||
                    modifier.kind ==
                            HipSearchModifierKind::SmoothSteering ||
                    modifier.kind ==
                            HipSearchModifierKind::InputInsertion ||
                    modifier.kind ==
                            HipSearchModifierKind::InputDeletion) {
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
            const HipSearchModifierConfiguration modifier =
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
            hip::candidate_events::EditWriter writer(
                    candidateEdits, slot);
            std::uint32_t mutationCount = 0u;
            for (std::uint32_t index = 0u;
                 index < count; ++index) {
                const std::uint32_t source = eligible[index];
                HipSearchInputEvent event =
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
            hip::candidate_events::SortOutputEdits(
                    candidateEdits, slot);
            hip::candidate_events::SortErasedSources(
                    candidateEdits, slot);
            eventCounts[slot] = baselineInputCount;
            mutationCounts[slot] = mutationCount;
            activeCandidates[slot] = mutationCount != 0u;
            return;
        }
        if (directDeletionPipeline) {
            const HipSearchModifierConfiguration modifier =
                    modifiers[0];
            random.Seed(
                    modifier.window.seed, candidateId, 0u);
            hip::candidate_events::EditWriter writer(
                    candidateEdits, slot);
            const auto deleteChannel =
                    [&](const HipSearchChannel &channel,
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
            hip::candidate_events::SortErasedSources(
                    candidateEdits, slot);
            eventCount = baselineInputCount -
                    candidateEdits.erasedCounts[slot];
            hip::candidate_events::CandidateCursor cursor({
                    {baselineInputs, baselineInputCount, 0},
                    candidateEdits,
                    slot,
                    eventCount});
            std::uint32_t mutationCount =
                    baselineInputCount - eventCount;
            for (std::uint32_t output = 0u;
                 output < eventCount; ++output) {
                HipSearchInputEvent event{};
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
        const HipSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const HipSearchInputEvent *candidateEvents,
        std::uint32_t eventCapacity,
        const std::uint32_t *eventCounts,
        std::uint32_t *sourceStates,
        hip::candidate_events::CoalescedEditStorage candidateEdits,
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
    const HipSearchInputEvent *events =
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

template <typename State, bool SimulateStunts>
__device__ State LoadSearchState(
        const HipCandidateState *branchState) {
    if constexpr (SimulateStunts) {
        return *branchState;
    } else {
        return static_cast<const HipCandidatePhysicsState &>(
                *branchState);
    }
}

template <
        typename State,
        bool SimulateStunts,
        bool SteadyTimeline,
        HipHandlingSpecialization Handling,
        std::uint32_t MinimumBlocksPerSm>
__global__ FOREVERVALIDATOR_HIP_SEARCH_LAUNCH_BOUNDS(
        SimulationBlockSize,
        MinimumBlocksPerSm) void SimulateSearchCandidatesKernel(
        const void *__restrict__ sceneData,
        const void *__restrict__ configurationData,
        const HipCandidateState *__restrict__ branchState,
        const DeviceControlState *__restrict__ mutableBoundaryControls,
        const HipControlTick *__restrict__ baselineTicks,
        std::uint32_t timelineTickCount,
        const HipSearchEvaluatorConfiguration *__restrict__ evaluator,
        const HipSearchConditionInstruction *__restrict__ condition,
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
        std::uint32_t eventCapacity,
        DeviceSample *__restrict__ candidateBestSamples,
        HipCandidatePhysicsState *__restrict__ finishCheckpointStates,
        std::uint32_t *__restrict__ finishCheckpointTicks,
        const HipSearchInputEvent *__restrict__ baselineInputs,
        std::uint32_t baselineInputCount,
        const HipSearchInputEvent *__restrict__ candidateEvents,
        const std::int32_t *__restrict__ candidateInputValues,
        const std::uint32_t *__restrict__ compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        hip::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *__restrict__ eventCounts,
        DeviceCandidateStatus *__restrict__ statuses,
        const bool *__restrict__ activeCandidates,
        hip::collision::HipCollisionSearchTile *__restrict__
                collisionScratch,
        hip::collision::HipCollisionSearchTile *__restrict__
                shapeCollisionScratch,
        GmIso4 *__restrict__ shapeWorldScratch,
        GmBoxAligned *__restrict__ movingBoundsScratch,
        hip::collision::HipCollisionSurfaceHit *
                __restrict__ surfaceHitScratch,
        hip::collision::HipCollisionMeshRange *
                __restrict__ meshRangeScratch,
        std::uint32_t *__restrict__ meshCellScratch,
        std::uint16_t *__restrict__ responseOrderScratch,
        std::uint32_t scratchStride,
        std::uint32_t shapeCapacity,
        const std::uint32_t *__restrict__ cancellation) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_SESSION_LTO)
    sceneData = reinterpret_cast<const HipPackedSceneHeader *>(
            hip::research::ForeverValidatorSessionSceneBytes());
    configurationData =
            reinterpret_cast<
                    const HipPackedStaticConfigurationHeader *>(
                    hip::research::
                            ForeverValidatorSessionConfigurationBytes());
#elif defined(FOREVERVALIDATOR_HIP_RESEARCH_WATER_ONLY)
    sceneData = &hip::research::StaticScene;
    configurationData =
            &hip::research::StaticConfiguration;
#endif
    const std::uint32_t slot =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= candidateCount || !activeCandidates[slot]) {
        return;
    }
    const std::uint64_t candidateId = firstCandidateId + slot;
    const HipSearchInputEvent *events =
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
    HipSearchInputEvent nextEvent{};
    bool hasNextEvent =
            eventCount != 0u && inputCursor.Next(&nextEvent);
    const HipSearchEvaluatorConfiguration &configuredEvaluator =
            *evaluator;
    const DeviceSample incumbent = candidateBestSamples[0];
    const bool pruneFinishTime =
            configuredEvaluator.kind ==
                    HipSearchEvaluatorKind::FinishTime &&
            incumbent.preciseFinish;
    if (!ValidPackedInputs(sceneData, configurationData) ||
        branchState->schemaVersion != HipCandidateState::SchemaVersion) {
        statuses[slot] =
                DeviceCandidateStatus::UnsupportedPhysicsTransition;
        return;
    }

    State state =
            LoadSearchState<State, SimulateStunts>(branchState);
    state.candidateId = static_cast<std::uint32_t>(candidateId);
    hip::collision::HipCollisionSearchScratch candidateScratch{
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
    DeviceControlState controlState = *mutableBoundaryControls;
    bool evaluatorReported = false;
    const bool maximize = MaximizesScore(configuredEvaluator.kind);
    DeviceSample localBest;
    std::uint32_t evaluationIndex = 0u;
    if (finishCheckpointTicks != nullptr) {
        finishCheckpointTicks[slot] = FinishCheckpointInvalidTick;
    }
    for (std::uint32_t tickIndex = 0u;
         tickIndex < timelineTickCount; ++tickIndex) {
        if ((tickIndex & 63u) == 0u &&
            *reinterpret_cast<volatile const std::uint32_t *>(
                    cancellation) != 0u) {
            statuses[slot] = DeviceCandidateStatus::Cancelled;
            candidateBestSamples[slot + 1u] = localBest;
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
        HipControlTick tick = baselineTicks[tickIndex];
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
        const CheckpointProgressBefore checkpointBefore{
                state.race.progress.currentLapCheckpointCount,
                state.race.progress.totalCheckpointEventCount,
                state.race.progress.completedLapCount,
                state.race.progress.finishCount};
        ApplyControlPrefix(state, tick);
        if (!state.firstStep) {
            if constexpr (SteadyTimeline) {
                hip::transition::PrepareSteadyStep(state, tick);
            } else {
                hip::transition::PrepareStep(
                        state, tick,
                        static_cast<const
                                HipPackedStaticConfigurationHeader *>(
                                configurationData));
            }
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        if constexpr (SteadyTimeline) {
            state.vehicle.mobil.physicsUpdatesEnabled = true;
        } else {
            state.vehicle.mobil.physicsUpdatesEnabled =
                    (tick.actionFlags &
                     HipControlActionSuppressVehicleForceCallbacks) ==
                    0u;
            for (std::uint32_t respawn = 0u;
                 respawn < tick.respawnAtCheckpointCount; ++respawn) {
                if (hip::transition::Respawn(
                            state,
                            static_cast<const
                                    HipPackedStaticConfigurationHeader *>(
                                    configurationData))) {
                    ++state.incrementalRespawnCount;
                    if constexpr (SimulateStunts) {
                        hip::stunts::ApplyRespawnPenalty(
                                state.stunts);
                    }
                }
            }
        }
        const hip::physics::Status physicsStatus =
                hip::physics::Step<
                        false,
                        (MinimumBlocksPerSm >
                         1u),
                        true,
                        true,
                        true,
                        (MinimumBlocksPerSm !=
                         ThroughputKernelMinimumBlocksPerSm),
                        SimulateStunts,
                        Handling>(
                        static_cast<const HipPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                HipPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, candidateScratch);
        if (physicsStatus != hip::physics::Status::Success) {
            statuses[slot] =
                    DeviceCandidateStatus::UnsupportedPhysicsTransition;
            candidateBestSamples[slot + 1u] = localBest;
            return;
        }
        if constexpr (SimulateStunts) {
            const hip::stunts::Status stuntStatus =
                    hip::stunts::Update(state, tick);
            if (stuntStatus != hip::stunts::Status::Success) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                candidateBestSamples[slot + 1u] = localBest;
                return;
            }
        }
        state.firstStep = false;
        ++state.controlCursor;
        if (publicTime < evaluationStartTimeMs) {
            continue;
        }
        if (conditionInstructionCount != 0u &&
            !EvaluateCondition(
                    condition, conditionInstructionCount, state,
                    baseline ? 0u : candidateId + 1u,
                    lastImprovementTimeSeconds,
                    lastRestartTimeSeconds, currentTimeSeconds)) {
            ++evaluationIndex;
            if (state.race.progress.raceCompleted) break;
            continue;
        }
        if (configuredEvaluator.kind ==
            HipSearchEvaluatorKind::Scripted) {
            if (UpdateScriptedSample(
                        configuredEvaluator, state,
                        baseline ? 0u : candidateId + 1u,
                        lastImprovementTimeSeconds,
                        lastRestartTimeSeconds, currentTimeSeconds,
                        static_cast<double>(publicTime), &localBest)) {
                localBest.mutation = !baseline;
                localBest.candidateId = candidateId;
                localBest.candidateSlot = slot;
                localBest.evaluationTick = evaluationIndex;
                localBest.eventCount = eventCount;
                localBest.logicalOrder = 1u +
                        static_cast<std::uint64_t>(slot) *
                                evaluationTickCount + evaluationIndex;
            }
            ++evaluationIndex;
            if (state.race.progress.raceCompleted) break;
            continue;
        }
        if (configuredEvaluator.kind ==
            HipSearchEvaluatorKind::CheckpointEvent) {
            if (CheckpointEventAccepted(
                        configuredEvaluator, state, checkpointBefore,
                        static_cast<const HipPackedSceneHeader *>(
                                sceneData))) {
                DeviceSample sample{};
                sample.score = static_cast<double>(publicTime);
                sample.timeMs = static_cast<double>(publicTime);
                sample.candidateId = candidateId;
                sample.candidateSlot = slot;
                sample.evaluationTick = evaluationIndex;
                sample.eventCount = eventCount;
                sample.logicalOrder =
                        1u +
                        static_cast<std::uint64_t>(slot) *
                                evaluationTickCount +
                        evaluationIndex;
                sample.mutation = !baseline;
                sample.valid = true;
                candidateBestSamples[slot + 1u] = sample;
                return;
            }
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
        sample.candidateId = candidateId;
        sample.candidateSlot = slot;
        sample.evaluationTick = evaluationIndex;
        sample.eventCount = eventCount;
        sample.logicalOrder =
                1u +
                static_cast<std::uint64_t>(slot) *
                        evaluationTickCount +
                evaluationIndex;
        sample.mutation = !baseline;
        if (StrictlyBetter(sample, localBest, maximize)) {
            localBest = sample;
        }
        ++evaluationIndex;
        if (evaluatorReported &&
            (configuredEvaluator.kind ==
                     HipSearchEvaluatorKind::VolumeEntry ||
             configuredEvaluator.kind ==
                     HipSearchEvaluatorKind::ConditionTimeEarliest ||
             configuredEvaluator.kind ==
                     HipSearchEvaluatorKind::ConditionTimeLatest)) {
            candidateBestSamples[slot + 1u] = localBest;
            return;
        }
        if (configuredEvaluator.kind ==
            HipSearchEvaluatorKind::FinishTime) {
            if (evaluatorReported) {
                candidateBestSamples[slot + 1u] = localBest;
                return;
            }
            if (pruneFinishTime &&
                static_cast<double>(publicTime) >= incumbent.timeMs) {
                candidateBestSamples[slot + 1u] = DeviceSample{};
                return;
            }
        }
    }
    candidateBestSamples[slot + 1u] = localBest;
}

template <
        typename State,
        bool SimulateStunts>
__global__ void RefineSearchFinishTimesKernel(
        const void *sceneData,
        const void *configurationData,
        const HipCandidateState *branchState,
        const DeviceControlState *mutableBoundaryControls,
        const HipControlTick *baselineTicks,
        std::uint32_t timelineTickCount,
        std::uint32_t tickDurationMs,
        std::uint32_t prestartDurationMs,
        std::int64_t branchTimeMs,
        std::int64_t mutableFromTimeMs,
        std::int64_t evaluationStartTimeMs,
        std::uint32_t eventCapacity,
        DeviceSample *candidateBestSamples,
        hip::finish::Refinement *finishRefinements,
        const HipCandidatePhysicsState *finishCheckpointStates,
        const std::uint32_t *finishCheckpointTicks,
        const HipSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const HipSearchInputEvent *candidateEvents,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        hip::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *eventCounts,
        DeviceCandidateStatus *statuses,
        const bool *activeCandidates,
        hip::collision::HipCollisionSearchTile *collisionScratch,
        hip::collision::HipCollisionSearchTile *shapeCollisionScratch,
        GmIso4 *shapeWorldScratch,
        GmBoxAligned *movingBoundsScratch,
        hip::collision::HipCollisionSurfaceHit *
                surfaceHitScratch,
        hip::collision::HipCollisionMeshRange *
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
    hip::finish::Refinement &refinement =
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
    const HipSearchInputEvent *events =
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
    HipSearchInputEvent nextEvent{};
    bool hasNextEvent =
            eventCount != 0u && inputCursor.Next(&nextEvent);
    State state =
            LoadSearchState<State, SimulateStunts>(branchState);
    state.candidateId =
            static_cast<std::uint32_t>(sample.candidateId);
    hip::collision::HipCollisionSearchScratch candidateScratch{
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
        HipControlTick tick = baselineTicks[tickIndex];
        tick.controls = ControlsFromState(controlState);
        tick.stuntsInput =
                StuntsFromState(controlState, prestartDurationMs);
        ApplyControlPrefix(state, tick);
        if (!state.firstStep) {
            hip::transition::PrepareStep(
                    state, tick,
                    static_cast<const
                            HipPackedStaticConfigurationHeader *>(
                            configurationData));
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        state.vehicle.mobil.physicsUpdatesEnabled =
                (tick.actionFlags &
                 HipControlActionSuppressVehicleForceCallbacks) == 0u;
        for (std::uint32_t respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn) {
            if (hip::transition::Respawn(
                        state,
                        static_cast<const
                                HipPackedStaticConfigurationHeader *>(
                                configurationData))) {
                ++state.incrementalRespawnCount;
                if constexpr (SimulateStunts) {
                    hip::stunts::ApplyRespawnPenalty(state.stunts);
                }
            }
        }
        const hip::physics::Status physicsStatus =
                hip::finish::StepAndRefine<
                        false, true, true, true, true, true>(
                        static_cast<const HipPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                HipPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, tick, candidateScratch, refinement,
                        1u,
                        incumbentUpperNs);
        if (physicsStatus != hip::physics::Status::Success ||
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
            const hip::stunts::Status stuntStatus =
                    hip::stunts::Update(state, tick);
            if (stuntStatus != hip::stunts::Status::Success) {
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
        const HipCandidateState *branchState,
        const DeviceControlState *mutableBoundaryControls,
        const HipControlTick *baselineTicks,
        const DeviceSample *reducedBest,
        const hip::finish::Refinement *finishRefinements,
        std::uint32_t tickDurationMs,
        std::uint32_t prestartDurationMs,
        std::int64_t branchTimeMs,
        std::int64_t mutableFromTimeMs,
        std::int64_t evaluationStartTimeMs,
        std::uint32_t eventCapacity,
        const HipSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const HipSearchInputEvent *candidateEvents,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        hip::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        std::uint32_t candidateCount,
        const std::uint32_t *eventCounts,
        DeviceCandidateStatus *statuses,
        hip::collision::HipCollisionSearchTile *collisionScratch,
        hip::collision::HipCollisionSearchTile *shapeCollisionScratch,
        GmIso4 *shapeWorldScratch,
        GmBoxAligned *movingBoundsScratch,
        hip::collision::HipCollisionSurfaceHit *
                surfaceHitScratch,
        hip::collision::HipCollisionMeshRange *
                meshRangeScratch,
        std::uint32_t *meshCellScratch,
        std::uint16_t *responseOrderScratch,
        std::uint32_t scratchStride,
        std::uint32_t shapeCapacity,
        HipCandidateState *capturedWinnerState) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const DeviceSample winner = *reducedBest;
    if (!winner.valid ||
        winner.candidateSlot == InvalidCandidateSlot) {
        return;
    }
    const std::uint32_t slot = winner.candidateSlot;
    const HipSearchInputEvent *events =
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
    HipSearchInputEvent nextEvent{};
    bool hasNextEvent =
            eventCount != 0u && inputCursor.Next(&nextEvent);
    const std::uint32_t evaluationStartTick =
            static_cast<std::uint32_t>(
                    (evaluationStartTimeMs -
                     (branchTimeMs + tickDurationMs)) /
                    tickDurationMs);
    const std::uint32_t targetTick =
            evaluationStartTick + winner.evaluationTick;

    hip::collision::HipCollisionSearchScratch candidateScratch{
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
    HipCandidateState state = *branchState;
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
        HipControlTick tick = baselineTicks[tickIndex];
        tick.controls = ControlsFromState(controlState);
        tick.stuntsInput =
                StuntsFromState(controlState, prestartDurationMs);
        ApplyControlPrefix(state, tick);
        if (!state.firstStep) {
            hip::transition::PrepareStep(
                    state, tick,
                    static_cast<const
                            HipPackedStaticConfigurationHeader *>(
                            configurationData));
        }
        state.vehicle.mobil.absorbContactEnabled = true;
        state.vehicle.mobil.physicsUpdatesEnabled =
                (tick.actionFlags &
                 HipControlActionSuppressVehicleForceCallbacks) == 0u;
        for (std::uint32_t respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn) {
            if (hip::transition::Respawn(
                        state,
                        static_cast<const
                                HipPackedStaticConfigurationHeader *>(
                                configurationData))) {
                ++state.incrementalRespawnCount;
                hip::stunts::ApplyRespawnPenalty(
                        state.stunts);
            }
        }
        const hip::physics::Status physicsStatus =
                hip::physics::Step<false, false, true>(
                        static_cast<const HipPackedSceneHeader *>(
                                sceneData),
                        static_cast<const
                                HipPackedStaticConfigurationHeader *>(
                                configurationData),
                        state, candidateScratch);
        if (physicsStatus != hip::physics::Status::Success) {
            statuses[slot] =
                    DeviceCandidateStatus::UnsupportedPhysicsTransition;
            return;
        }
        if (state.stuntsEnabled) {
            const hip::stunts::Status stuntStatus =
                    hip::stunts::Update(state, tick);
            if (stuntStatus != hip::stunts::Status::Success) {
                statuses[slot] =
                        DeviceCandidateStatus::CapacityExceeded;
                return;
            }
        }
        state.firstStep = false;
        ++state.controlCursor;
    }
    hip::collision::detail::CaptureReplacementOverflow(
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

__global__ void FinalizeSearchBatchKernel(
        const DeviceSample *reducedBest,
        const HipCandidateState *capturedWinnerState,
        const DeviceSample *candidateBestSamples,
        const HipSearchInputEvent *baselineInputs,
        std::uint32_t baselineInputCount,
        const HipSearchInputEvent *candidateEvents,
        const std::int32_t *candidateInputValues,
        const std::uint32_t *compactInputOffsets,
        std::uint32_t compactInputCount,
        bool compactRandomSteeringPipeline,
        bool compactEditPipeline,
        bool sparseMutationPipeline,
        hip::candidate_events::CoalescedEditStorage candidateEdits,
        sparse_events::Storage sparseCandidateEvents,
        const std::uint32_t *eventCounts,
        const std::uint32_t *mutationCounts,
        const DeviceCandidateStatus *statuses,
        const bool *activeCandidates,
        std::uint32_t candidateCount,
        std::uint32_t eventCapacity,
        std::uint32_t evaluationTickCount,
        bool maximize,
        bool scripted,
        bool baseline,
        bool captureBestState,
        DeviceSample *globalBestSample,
        HipCandidateState *globalBestState,
        HipSearchInputEvent *globalBestInputs,
        std::uint32_t *globalBestEventCount,
        std::uint32_t *globalBestMutationCount,
        DeviceBatchSummary *summary) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    DeviceBatchSummary result;
    for (std::uint32_t slot = 0u; slot < candidateCount; ++slot) {
        if (statuses[slot] == DeviceCandidateStatus::Cancelled) {
            result.status = HipSearchStatus::Cancelled;
        } else if (statuses[slot] ==
                   DeviceCandidateStatus::CapacityExceeded) {
            result.status = HipSearchStatus::CapacityExceeded;
        } else if (statuses[slot] ==
                   DeviceCandidateStatus::UnsupportedPhysicsTransition) {
            result.status =
                    HipSearchStatus::UnsupportedPhysicsTransition;
        }
        if (activeCandidates[slot]) {
            ++result.evaluatedCandidateCount;
            result.evaluatorCalls += evaluationTickCount;
        }
        result.totalMutationCount += mutationCounts[slot];
    }

    DeviceSample incumbent = candidateBestSamples[0];
    if (!baseline) {
        for (std::uint32_t slot = 0u;
             slot < candidateCount; ++slot) {
            const DeviceSample sample =
                    candidateBestSamples[slot + 1u];
            if (scripted
                        ? StrictlyDominates(sample, incumbent)
                        : StrictlyBetter(sample, incumbent, maximize)) {
                ++result.mutationImprovementCount;
                incumbent = sample;
            }
        }
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
            const HipSearchInputEvent *materializedInputs =
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
                            HipSearchStatus::CapacityExceeded;
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

struct HipSearchExecutor::Impl {
    struct SimulationKernelMetrics {
        std::uint32_t registersPerThread = 0u;
        std::uint64_t localBytesPerThread = 0u;
        std::uint32_t activeBlocksPerMultiprocessor = 0u;
        double theoreticalOccupancy = 0.0;
    };

    HipSearchExecutorConfiguration configuration;
    std::vector<HipSearchInputEvent> immutableInputPrefix;
    std::vector<HipSearchInputEvent> immutableInputTail;
    std::int64_t mutableFromTimeMs = 0;
    std::uint32_t timelineTickCount = 0u;
    std::uint32_t evaluationTickCount = 0u;
    std::uint32_t collisionShapeCount = 0u;
    std::uint64_t residentBytes = 0u;
    std::uint64_t winnerSelectionBytes = 0u;
    std::uint64_t initialUploadBytes = 0u;
    bool baselineEvaluated = false;
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
    bool steadyTimeline = false;
    std::uint32_t compactInputCount = 0u;
    std::uint32_t sharedEligibleCount = 0u;
    std::uint32_t editCapacity = 0u;
    std::uint32_t eraseCapacity = 0u;
    std::size_t editStorageBytes = 0u;
    std::uint32_t multiprocessorCount = 0u;
    HipHandlingSpecialization handlingSpecialization =
            HipHandlingSpecialization::Generic;
    SimulationKernelMetrics throughputKernelMetrics;
    SimulationKernelMetrics tailKernelMetrics;
    SimulationKernelMetrics denseTailKernelMetrics;
    std::shared_ptr<const hip::specialization::SessionModule>
            specializedModule;

    DeviceAllocation<HipCandidateState> branchState;
    DeviceAllocation<DeviceControlState> mutableBoundaryControls;
    DeviceAllocation<HipControlTick> baselineTicks;
    DeviceAllocation<HipSearchInputEvent> baselineInputs;
    DeviceAllocation<HipSearchModifierConfiguration> modifiers;
    DeviceAllocation<double> smoothWeights;
    DeviceAllocation<HipSearchEvaluatorConfiguration> evaluator;
    DeviceAllocation<HipSearchConditionInstruction> condition;
    DeviceAllocation<HipCandidateState> capturedWinnerState;
    DeviceAllocation<DeviceSample> candidateBestSamples;
    DeviceAllocation<hip::finish::Refinement> finishRefinements;
    DeviceAllocation<HipCandidatePhysicsState> finishCheckpointStates;
    DeviceAllocation<std::uint32_t> finishCheckpointTicks;
    DeviceAllocation<std::uint32_t> randomStateWords;
    DeviceAllocation<HipSearchInputEvent> candidateEvents;
    DeviceAllocation<std::uint32_t> compactInputIndices;
    DeviceAllocation<std::uint32_t> compactInputOffsets;
    DeviceAllocation<std::int32_t> candidateInputValues;
    DeviceAllocation<HipSearchInputEvent> temporaryEvents;
    DeviceAllocation<HipSearchInputEvent> passBaselineEvents;
    DeviceAllocation<std::uint32_t> eligibleIndices;
    DeviceAllocation<std::uint32_t> sharedEligibleIndices;
    DeviceAllocation<std::uint32_t> sparseReferences;
    DeviceAllocation<std::uint32_t> sparseSnapshotReferences;
    DeviceAllocation<HipSearchInputEvent> sparseEdits;
    DeviceAllocation<HipSearchInputEvent> sparseScratchEdits;
    DeviceAllocation<std::byte> editBacking;
    DeviceAllocation<std::uint32_t> eventCounts;
    DeviceAllocation<std::uint32_t> mutationCounts;
    DeviceAllocation<DeviceCandidateStatus> statuses;
    DeviceAllocation<bool> activeCandidates;
    DeviceAllocation<DeviceSample> reducedBest;
    DeviceAllocation<std::byte> reductionTemporary;
    DeviceAllocation<hip::collision::HipCollisionSearchTile>
            collisionScratch;
    DeviceAllocation<hip::collision::HipCollisionSearchTile>
            shapeCollisionScratch;
    DeviceAllocation<GmIso4> shapeWorldScratch;
    DeviceAllocation<GmBoxAligned> movingBoundsScratch;
    DeviceAllocation<hip::collision::HipCollisionSurfaceHit>
            surfaceHitScratch;
    DeviceAllocation<hip::collision::HipCollisionMeshRange>
            meshRangeScratch;
    DeviceAllocation<std::uint32_t> meshCellScratch;
    DeviceAllocation<std::uint16_t> responseOrderScratch;
    MappedCancellation cancellation;
    DeviceAllocation<DeviceSample> globalBestSample;
    DeviceAllocation<HipCandidateState> globalBestState;
    DeviceAllocation<HipSearchInputEvent> globalBestInputs;
    DeviceAllocation<std::uint32_t> globalBestEventCount;
    DeviceAllocation<std::uint32_t> globalBestMutationCount;
    DeviceAllocation<DeviceBatchSummary> summary;

    void UpdateResidentBytes() {
        residentBytes = 0u;
#define ADD_BYTES(member) residentBytes += member.Bytes()
        ADD_BYTES(branchState);
        ADD_BYTES(mutableBoundaryControls);
        ADD_BYTES(baselineTicks);
        ADD_BYTES(baselineInputs);
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
        ADD_BYTES(reducedBest);
        ADD_BYTES(reductionTemporary);
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
#undef ADD_BYTES
        winnerSelectionBytes =
                candidateBestSamples.Bytes() +
                reducedBest.Bytes() +
                reductionTemporary.Bytes();
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

    hip::candidate_events::CoalescedEditStorage CandidateEdits(
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
        hip::candidate_events::CoalescedEditStorage storage;
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
            HipHandlingSpecialization Handling>
    const void *SimulationKernel() const {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_WATER_ONLY)
        return reinterpret_cast<const void *>(
                SimulateSearchCandidatesKernel<
                        HipCandidatePhysicsState,
                        false,
                        true,
                        Handling,
                        MinimumBlocksPerSm>);
#else
        if (configuration.branchState.stuntsEnabled &&
            configuration.evaluator.kind !=
                    HipSearchEvaluatorKind::FinishTime) {
            return reinterpret_cast<const void *>(
                    SimulateSearchCandidatesKernel<
                            HipCandidateState,
                            true,
                            false,
                            Handling,
                            MinimumBlocksPerSm>);
        }
        return steadyTimeline
                ? reinterpret_cast<const void *>(
                          SimulateSearchCandidatesKernel<
                                  HipCandidatePhysicsState,
                                  false,
                                  true,
                                  Handling,
                                  MinimumBlocksPerSm>)
                : reinterpret_cast<const void *>(
                          SimulateSearchCandidatesKernel<
                                  HipCandidatePhysicsState,
                                  false,
                                  false,
                                  Handling,
                                  MinimumBlocksPerSm>);
#endif
    }

    template <std::uint32_t MinimumBlocksPerSm>
    const void *SelectedSimulationKernel() const {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_WATER_ONLY)
        return SimulationKernel<
                MinimumBlocksPerSm,
                HipHandlingSpecialization::GearedDriveWater>();
#else
        switch (handlingSpecialization) {
        case HipHandlingSpecialization::Legacy:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    HipHandlingSpecialization::Legacy>();
        case HipHandlingSpecialization::GearedDriveDry:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    HipHandlingSpecialization::GearedDriveDry>();
        case HipHandlingSpecialization::GearedDriveWater:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    HipHandlingSpecialization::GearedDriveWater>();
        case HipHandlingSpecialization::Generic:
            return SimulationKernel<
                    MinimumBlocksPerSm,
                    HipHandlingSpecialization::Generic>();
        }
        return SimulationKernel<
                MinimumBlocksPerSm,
                HipHandlingSpecialization::Generic>();
#endif
    }

    bool LoadSimulationKernelMetrics(
            const void *kernel,
            const hipDeviceProp_t &properties,
            SimulationKernelMetrics *metrics,
            std::string *diagnostic) {
        hipFuncAttributes attributes{};
        hipError_t error =
                hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(kernel));
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "querying HIP simulation kernel attributes",
                        error);
            }
            return false;
        }
        int activeBlocks = 0;
        error = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks, kernel, SimulationBlockSize, 0u);
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "querying HIP simulation occupancy", error);
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
        hipDeviceProp_t properties{};
        hipError_t error = hipGetDevice(&device);
        if (error == hipSuccess) {
            error = hipGetDeviceProperties(&properties, device);
        }
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "querying HIP device properties", error);
            }
            return false;
        }
        multiprocessorCount =
                static_cast<std::uint32_t>(
                        properties.multiProcessorCount);
        if (specializedModule && specializedModule->Ready()) {
            const auto load =
                    [&](std::uint32_t minimumBlocks,
                        SimulationKernelMetrics *metrics) {
                const hip::specialization::KernelMetrics &source =
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
                hip::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                hip::collision::ShapeCollisionCapacity;
        const std::uint64_t collisionTileStride64 =
                (static_cast<std::uint64_t>(candidateCount) +
                 hip::collision::HipCollisionSearchTileWidth - 1u) /
                hip::collision::HipCollisionSearchTileWidth;
        const std::uint64_t collisionTileSlots64 =
                collisionTileStride64 *
                hip::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionTileSlots64 =
                collisionTileStride64 *
                hip::collision::ShapeCollisionCapacity;
        const std::uint64_t shapeQuerySlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                collisionShapeCount;
        const std::uint64_t surfaceHitSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                hip::collision::SurfaceHitCapacity;
        const std::uint64_t meshRangeSlots64 =
                surfaceHitSlots64;
        const std::uint64_t meshCellSlots64 =
                static_cast<std::uint64_t>(candidateCount) *
                hip::collision::MeshCellHitCapacity;
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
                        "HIP search calibration buffer dimensions overflow";
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
        DeviceAllocation<DeviceSample> nextCandidateBestSamples;
        DeviceAllocation<hip::finish::Refinement>
                nextFinishRefinements;
        DeviceAllocation<HipCandidatePhysicsState>
                nextFinishCheckpointStates;
        DeviceAllocation<std::uint32_t> nextFinishCheckpointTicks;
        DeviceAllocation<std::uint32_t> nextRandomStateWords;
        DeviceAllocation<HipSearchInputEvent> nextCandidateEvents;
        DeviceAllocation<std::int32_t> nextCandidateInputValues;
        DeviceAllocation<HipSearchInputEvent> nextTemporaryEvents;
        DeviceAllocation<HipSearchInputEvent> nextPassBaselineEvents;
        DeviceAllocation<std::uint32_t> nextEligibleIndices;
        DeviceAllocation<std::uint32_t> nextSparseReferences;
        DeviceAllocation<std::uint32_t> nextSparseSnapshotReferences;
        DeviceAllocation<HipSearchInputEvent> nextSparseEdits;
        DeviceAllocation<HipSearchInputEvent> nextSparseScratchEdits;
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
        DeviceAllocation<std::byte> nextReductionTemporary;
        DeviceAllocation<hip::collision::HipCollisionSearchTile>
                nextCollisionScratch;
        DeviceAllocation<hip::collision::HipCollisionSearchTile>
                nextShapeCollisionScratch;
        DeviceAllocation<GmIso4> nextShapeWorldScratch;
        DeviceAllocation<GmBoxAligned> nextMovingBoundsScratch;
        DeviceAllocation<hip::collision::HipCollisionSurfaceHit>
                nextSurfaceHitScratch;
        DeviceAllocation<hip::collision::HipCollisionMeshRange>
                nextMeshRangeScratch;
        DeviceAllocation<std::uint32_t> nextMeshCellScratch;
        DeviceAllocation<std::uint16_t> nextResponseOrderScratch;
        if (!nextCandidateBestSamples.Allocate(winnerSlots) ||
            !nextFinishRefinements.Allocate(
                    configuration.evaluator.kind ==
                                    HipSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !nextFinishCheckpointStates.Allocate(
                    configuration.evaluator.kind ==
                                    HipSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !nextFinishCheckpointTicks.Allocate(
                    configuration.evaluator.kind ==
                                    HipSearchEvaluatorKind::FinishTime
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
            !nextCollisionScratch.Allocate(collisionTileSlots) ||
            !nextShapeCollisionScratch.Allocate(
                    shapeCollisionTileSlots) ||
            !nextShapeWorldScratch.Allocate(shapeQuerySlots) ||
            !nextMovingBoundsScratch.Allocate(shapeQuerySlots) ||
            !nextSurfaceHitScratch.Allocate(surfaceHitSlots) ||
            !nextMeshRangeScratch.Allocate(meshRangeSlots) ||
            !nextMeshCellScratch.Allocate(meshCellSlots) ||
            !nextResponseOrderScratch.Allocate(collisionSlots)) {
            static_cast<void>(hipGetLastError());
            if (diagnostic != nullptr) {
                *diagnostic =
                        "HIP could not reserve a larger batch within memory headroom or allocator limits; reduce parallel samples or use Optimized CPU";
            }
            return false;
        }

        std::size_t reductionBytes = 0u;
        const hipError_t error = hipcub::DeviceReduce::Reduce(
                nullptr, reductionBytes,
                nextCandidateBestSamples.Get(), reducedBest.Get(),
                winnerSlots,
                BetterSample{
                        MaximizesScore(configuration.evaluator.kind)},
                DeviceSample{});
        if (error != hipSuccess ||
            !nextReductionTemporary.Allocate(reductionBytes)) {
            static_cast<void>(hipGetLastError());
            if (diagnostic != nullptr) {
                *diagnostic = error != hipSuccess
                        ? HipFailure(
                                  "sizing calibrated HIP winner reduction",
                                  error)
                        : "HIP calibration winner reduction allocation failed";
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
        reductionTemporary = std::move(nextReductionTemporary);
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
        configuration.maximumBatchSize = candidateCount;
        UpdateResidentBytes();
        if (diagnostic != nullptr) {
            diagnostic->clear();
        }
        return true;
    }

    HipSearchBatchExecution Execute(
            std::uint64_t firstCandidateId,
            std::uint32_t candidateCount,
            bool baseline,
            const std::function<bool()> &cancellationRequested) noexcept {
        HipSearchBatchExecution result;
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
                sparseScratchEdits.Bytes();
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
            result.status = HipSearchStatus::InvalidArgument;
            result.diagnostic = !baselineEvaluated && !baseline
                    ? "HIP baseline must be evaluated before mutation batches"
                    : "invalid HIP search batch size";
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
        hipError_t error = hipSuccess;

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
            result.status = HipSearchStatus::DeviceFailure;
            result.diagnostic = "HIP search event creation failed";
            return result;
        }
        hipEventRecord(started.Get());
        constexpr std::uint32_t blockSize = 128u;
        SeedCandidateBestSamplesKernel<<<1u, 1u>>>(
                candidateBestSamples.Get(), globalBestSample.Get());
        hipEventRecord(winnerInitialized.Get());
        const std::uint32_t candidateBlocks =
                (candidateCount - 1u) / blockSize + 1u;
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
        hipEventRecord(mutationsGenerated.Get());
        const std::uint32_t simulationBlocks =
                (candidateCount - 1u) / SimulationBlockSize + 1u;
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
        const std::uint32_t conditionInstructionCount =
                configuration.condition
                ? static_cast<std::uint32_t>(
                          configuration.condition->instructions.size())
                : 0u;
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
        result.evaluationCurrentTimeSeconds = currentTimeSeconds;
        if (specializedModule) {
            const hipError_t simulationLaunch = LaunchDriverKernel(
                    specializedModule->Kernel(selectedMinimumBlocks),
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
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    candidateBestSamples.Get(),
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
            if (simulationLaunch != hipSuccess) {
                const char *message = nullptr;
                hipDrvGetErrorString(simulationLaunch, &message);
                result.status = HipSearchStatus::DeviceFailure;
                result.diagnostic =
                        "launching specialized HIP simulation kernel: " +
                        std::string(
                                message == nullptr ? "unknown" : message);
                return result;
            }
        } else {
            const void *simulationKernel = nullptr;
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
            const hipError_t simulationLaunch = LaunchRuntimeKernel(
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
                    static_cast<std::uint32_t>(
                            configuration.maximumEventCount),
                    candidateBestSamples.Get(),
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
            if (simulationLaunch != hipSuccess) {
                result.status = HipSearchStatus::DeviceFailure;
                result.diagnostic = HipFailure(
                        "launching HIP simulation kernel",
                        simulationLaunch);
                return result;
            }
        }
        hipEventRecord(simulationFinished.Get());
        if (configuration.evaluator.kind ==
            HipSearchEvaluatorKind::FinishTime) {
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
                    HipCandidatePhysicsState{}, std::false_type{});
        }
        hipEventRecord(finishRefined.Get());
        std::size_t temporaryBytes = reductionTemporary.Bytes();
        if (configuration.evaluator.kind ==
            HipSearchEvaluatorKind::Scripted) {
            SelectScriptedWinnerKernel<<<1u, 1u>>>(
                    candidateBestSamples.Get(), candidateCount,
                    reducedBest.Get());
            error = hipGetLastError();
        } else {
            error = hipcub::DeviceReduce::Reduce(
                    reductionTemporary.Get(), temporaryBytes,
                    candidateBestSamples.Get(), reducedBest.Get(),
                    winnerCount,
                    BetterSample{
                            MaximizesScore(configuration.evaluator.kind)},
                    DeviceSample{});
        }
        if (error != hipSuccess) {
            result.status = HipSearchStatus::DeviceFailure;
            result.diagnostic =
                    HipFailure("launching HIP winner reduction", error);
            return result;
        }
        hipEventRecord(winnerReduced.Get());
        if (configuration.captureBestState) {
            CaptureSearchWinnerStateKernel<<<1u, 1u>>>(
                    configuration.deviceScene,
                    configuration.deviceStaticConfiguration,
                    branchState.Get(),
                    mutableBoundaryControls.Get(),
                    baselineTicks.Get(),
                    reducedBest.Get(),
                    configuration.evaluator.kind ==
                                    HipSearchEvaluatorKind::FinishTime
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
        hipEventRecord(winnerStateCaptured.Get());
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
                configuration.evaluator.kind ==
                        HipSearchEvaluatorKind::Scripted,
                baseline,
                configuration.captureBestState,
                globalBestSample.Get(),
                globalBestState.Get(),
                globalBestInputs.Get(),
                globalBestEventCount.Get(),
                globalBestMutationCount.Get(),
                summary.Get());
        error = hipGetLastError();
        if (error != hipSuccess) {
            result.status = HipSearchStatus::DeviceFailure;
            result.diagnostic =
                    HipFailure("launching HIP search kernels", error);
            return result;
        }
        hipEventRecord(finished.Get());
        while ((error = hipEventQuery(finished.Get())) ==
               hipErrorNotReady) {
            if (!cancelled && cancellationRequested) {
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
        if (error != hipSuccess) {
            result.status = HipSearchStatus::DeviceFailure;
            result.diagnostic =
                    HipFailure("synchronizing HIP search batch", error);
            return result;
        }
        float milliseconds = 0.0f;
        hipEventElapsedTime(&milliseconds, started.Get(), finished.Get());
        result.kernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds, started.Get(), winnerInitialized.Get());
        result.scoreInitializationKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                winnerInitialized.Get(),
                mutationsGenerated.Get());
        result.mutationKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                mutationsGenerated.Get(),
                simulationFinished.Get());
        result.simulationKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                simulationFinished.Get(),
                finishRefined.Get());
        result.finishRefinementKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                finishRefined.Get(),
                finished.Get());
        result.winnerKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                finishRefined.Get(),
                winnerReduced.Get());
        result.winnerReductionKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                winnerReduced.Get(),
                winnerStateCaptured.Get());
        result.winnerStateCaptureKernelMilliseconds = milliseconds;
        hipEventElapsedTime(
                &milliseconds,
                winnerStateCaptured.Get(),
                finished.Get());
        result.finalizationKernelMilliseconds = milliseconds;
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
        error = hipMemcpy(
                &hostSummary, summary.Get(), sizeof(hostSummary),
                hipMemcpyDeviceToHost);
        if (error != hipSuccess) {
            result.status = HipSearchStatus::DeviceFailure;
            result.diagnostic =
                    HipFailure("copying HIP search summary", error);
            return result;
        }
        result.deviceToHostBytes += sizeof(hostSummary);
        result.status = hostSummary.status;
        result.evaluatedCandidateCount =
                hostSummary.evaluatedCandidateCount;
        result.evaluatorCalls = hostSummary.evaluatorCalls;
        result.totalMutationCount =
                hostSummary.totalMutationCount;
        result.mutationImprovementCount =
                baseline ? 0u
                         : hostSummary.mutationImprovementCount;
        result.bestChanged = hostSummary.bestChanged;
        if (hostSummary.bestValid) {
            result.best.valid = true;
            result.best.mutation = hostSummary.bestMutation;
            result.best.stateCaptured = configuration.captureBestState;
            result.best.candidateId = hostSummary.bestCandidateId;
            result.best.mutationCount =
                    hostSummary.bestMutationCount;
            DeviceSample bestSample;
            error = hipMemcpy(
                    &bestSample, globalBestSample.Get(),
                    sizeof(bestSample), hipMemcpyDeviceToHost);
            if (error == hipSuccess && configuration.captureBestState) {
                error = hipMemcpy(
                        &result.best.state, globalBestState.Get(),
                        sizeof(result.best.state),
                        hipMemcpyDeviceToHost);
            }
            if (error != hipSuccess) {
                result.status = HipSearchStatus::DeviceFailure;
                result.diagnostic =
                        HipFailure("copying HIP winning state", error);
                return result;
            }
            result.best.score = bestSample.score;
            result.best.evaluationTick = bestSample.evaluationTick;
            result.best.timeMs = bestSample.timeMs;
            result.best.detail0 = bestSample.detail0;
            result.best.detail1 = bestSample.detail1;
            result.best.scriptedObjectiveCount =
                    bestSample.scriptedObjectiveCount;
            for (std::uint32_t i = 0u;
                 i < bestSample.scriptedObjectiveCount; ++i) {
                result.best.objectiveScores[i] =
                        bestSample.objectiveScores[i];
                result.best.metricValues[i] =
                        bestSample.metricValues[i];
            }
            result.best.inputs = immutableInputPrefix;
            const std::size_t suffixOffset =
                    result.best.inputs.size();
            result.best.inputs.resize(
                    suffixOffset + hostSummary.globalEventCount);
            if (hostSummary.globalEventCount != 0u) {
                error = hipMemcpy(
                        result.best.inputs.data() + suffixOffset,
                        globalBestInputs.Get(),
                        hostSummary.globalEventCount *
                                sizeof(HipSearchInputEvent),
                        hipMemcpyDeviceToHost);
                if (error != hipSuccess) {
                    result.status = HipSearchStatus::DeviceFailure;
                    result.diagnostic =
                            HipFailure(
                                    "copying HIP winning inputs", error);
                    return result;
                }
                for (std::size_t index = suffixOffset;
                     index < result.best.inputs.size(); ++index) {
                    result.best.inputs[index] =
                            hip_search_detail::AbsoluteSuffixEvent(
                                    result.best.inputs[index],
                                    mutableFromTimeMs);
                }
            }
            result.best.inputs.reserve(
                    result.best.inputs.size() +
                    immutableInputTail.size());
            for (HipSearchInputEvent input : immutableInputTail) {
                result.best.inputs.push_back(
                        hip_search_detail::AbsoluteSuffixEvent(
                                input, mutableFromTimeMs));
            }
            result.deviceToHostBytes += sizeof(bestSample) +
                    (configuration.captureBestState
                             ? sizeof(result.best.state) : 0u) +
                    hostSummary.globalEventCount *
                            sizeof(HipSearchInputEvent);
        }
        if (baseline && result.status == HipSearchStatus::Success) {
            baselineEvaluated = true;
        }
        if (result.status != HipSearchStatus::Success &&
            result.diagnostic.empty()) {
            result.diagnostic =
                    std::string("HIP search batch status: ") +
                    HipSearchStatusName(result.status);
        }
        return result;
    }
};

HipSearchExecutor::HipSearchExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
HipSearchExecutor::~HipSearchExecutor() = default;
HipSearchExecutor::HipSearchExecutor(HipSearchExecutor &&) noexcept =
        default;
HipSearchExecutor &HipSearchExecutor::operator=(
        HipSearchExecutor &&) noexcept = default;

std::unique_ptr<HipSearchExecutor> HipSearchExecutor::Create(
        const HipSearchExecutorConfiguration &configuration,
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
            (configuration.incumbent &&
             configuration.incumbent->evaluationTick >=
                     static_cast<std::uint64_t>(
                             configuration.evaluationEndTimeMs -
                             configuration.evaluationStartTimeMs) /
                                     configuration.tickDurationMs +
                             1u)) {
            if (diagnostic != nullptr) {
                *diagnostic = "invalid HIP search executor configuration";
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
                *diagnostic = "HIP evaluation timeline is too large";
            }
            return {};
        }
        HipPackedStaticConfigurationHeader packedConfiguration{};
        const hipError_t configurationCopyError = hipMemcpy(
                &packedConfiguration,
                configuration.deviceStaticConfiguration,
                sizeof(packedConfiguration),
                hipMemcpyDeviceToHost);
        if (configurationCopyError != hipSuccess ||
            packedConfiguration.magic !=
                    HipPackedStaticConfigurationHeader::Magic ||
            packedConfiguration.schemaVersion !=
                    HipPackedStaticConfigurationHeader::
                            SchemaVersion) {
            if (diagnostic != nullptr) {
                *diagnostic = configurationCopyError != hipSuccess
                        ? HipFailure(
                                  "reading HIP static configuration",
                                  configurationCopyError)
                        : "invalid HIP static configuration header";
            }
            return {};
        }
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_FOUR_WHEELS)
        if (configuration.branchState.vehicle.wheels.count != 4u) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "research wheel-count fact does not match";
            }
            return {};
        }
        for (std::uint32_t index = 0u; index < 4u; ++index) {
            const HipWheelState &stateWheel =
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
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CANONICAL_WHEEL_FACTS)
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
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_WATER_ONLY)
        HipPackedSceneHeader packedScene{};
        const hipError_t sceneCopyError = hipMemcpy(
                &packedScene,
                configuration.deviceScene,
                sizeof(packedScene),
                hipMemcpyDeviceToHost);
        if (sceneCopyError != hipSuccess ||
            packedScene.magic != HipPackedSceneHeader::Magic ||
            packedScene.schemaVersion !=
                    HipPackedSceneHeader::SchemaVersion) {
            if (diagnostic != nullptr) {
                *diagnostic = sceneCopyError != hipSuccess
                        ? HipFailure(
                                  "reading HIP scene header",
                                  sceneCopyError)
                        : "invalid HIP scene header";
            }
            return {};
        }
        const std::uint64_t sceneBase =
                reinterpret_cast<std::uintptr_t>(
                        configuration.deviceScene);
        hipError_t researchConstantError =
                hipMemcpyToSymbol(HIP_SYMBOL(
                        hip::research::StaticSceneBase),
                        &sceneBase,
                        sizeof(sceneBase),
                        0u,
                        hipMemcpyHostToDevice);
        if (researchConstantError == hipSuccess) {
            researchConstantError = hipMemcpyToSymbol(HIP_SYMBOL(
                    hip::research::StaticScene),
                    &packedScene,
                    sizeof(packedScene),
                    0u,
                    hipMemcpyHostToDevice);
        }
        if (researchConstantError != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "copying research HIP scene header",
                        researchConstantError);
            }
            return {};
        }
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_COLLISION_SHAPES)
        if (packedConfiguration.collisionShapes.count != 8u ||
            packedConfiguration.collisionShapes.stride !=
                    sizeof(HipVehicleCollisionShape)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "research collision shape facts do not match";
            }
            return {};
        }
        std::array<HipVehicleCollisionShape, 8u>
                constantCollisionShapes{};
        const auto *configurationBytes =
                static_cast<const std::byte *>(
                        configuration.deviceStaticConfiguration);
        researchConstantError = hipMemcpy(
                constantCollisionShapes.data(),
                configurationBytes +
                        packedConfiguration.collisionShapes.offset,
                sizeof(constantCollisionShapes),
                hipMemcpyDeviceToHost);
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_EIGHT_ROOT_SHAPES)
        constexpr std::uint32_t ExpectedWheelIndices[] = {
                UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                0u, 1u, 3u, 2u};
        if (researchConstantError == hipSuccess) {
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
        if (researchConstantError == hipSuccess) {
            researchConstantError = hipMemcpyToSymbol(HIP_SYMBOL(
                    hip::research::StaticCollisionShapes),
                    constantCollisionShapes.data(),
                    sizeof(constantCollisionShapes),
                    0u,
                    hipMemcpyHostToDevice);
        }
        if (researchConstantError != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "copying research HIP collision shapes",
                        researchConstantError);
            }
            return {};
        }
#endif
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_CURVE_KEYS)
        if (packedConfiguration.curveKeys.count > 4096u ||
            packedConfiguration.curveKeys.stride !=
                    sizeof(HipTuningCurveKey)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "research tuning curve key facts do not match";
            }
            return {};
        }
        std::vector<HipTuningCurveKey> constantCurveKeys(
                packedConfiguration.curveKeys.count);
        researchConstantError = hipMemcpy(
                constantCurveKeys.data(),
                configurationBytes +
                        packedConfiguration.curveKeys.offset,
                constantCurveKeys.size() *
                        sizeof(HipTuningCurveKey),
                hipMemcpyDeviceToHost);
        if (researchConstantError == hipSuccess &&
            !constantCurveKeys.empty()) {
            researchConstantError = hipMemcpyToSymbol(HIP_SYMBOL(
                    hip::research::StaticCurveKeys),
                    constantCurveKeys.data(),
                    constantCurveKeys.size() *
                            sizeof(HipTuningCurveKey),
                    0u,
                    hipMemcpyHostToDevice);
        }
        if (researchConstantError != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "copying research HIP tuning curve keys",
                        researchConstantError);
            }
            return {};
        }
#endif
        const std::uint64_t configurationBase =
                reinterpret_cast<std::uintptr_t>(
                        configuration.deviceStaticConfiguration);
        const hipError_t constantBaseError =
                hipMemcpyToSymbol(HIP_SYMBOL(
                        hip::research::StaticConfigurationBase),
                        &configurationBase,
                        sizeof(configurationBase),
                        0u,
                        hipMemcpyHostToDevice);
        if (constantBaseError != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "copying research HIP configuration base",
                        constantBaseError);
            }
            return {};
        }
        const hipError_t constantConfigurationError =
                hipMemcpyToSymbol(HIP_SYMBOL(
                        hip::research::StaticConfiguration),
                        &packedConfiguration,
                        sizeof(packedConfiguration),
                        0u,
                        hipMemcpyHostToDevice);
        if (constantConfigurationError != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "copying research HIP static configuration",
                        constantConfigurationError);
            }
            return {};
        }
#endif
        const std::int64_t mutableFromTimeMs =
                configuration.branchTimeMs +
                configuration.tickDurationMs;
        hip_search_detail::SearchInputPartition inputPartition;
        if (!hip_search_detail::PartitionSearchInputs(
                    configuration.baselineInputs,
                    configuration.branchTimeMs,
                    mutableFromTimeMs,
                    &inputPartition)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        "HIP search inputs cannot be partitioned at the mutable boundary";
            }
            return {};
        }
        HipSearchExecutorConfiguration preparedConfiguration =
                configuration;
        preparedConfiguration.baselineInputs =
                inputPartition.mutableSuffix;
        preparedConfiguration.maximumEventCount -=
                inputPartition.immutablePrefix.size();
        for (HipSearchModifierConfiguration &modifier :
             preparedConfiguration.modifiers) {
            if (modifier.window.maximumTimeMs < mutableFromTimeMs) {
                if (diagnostic != nullptr) {
                    *diagnostic =
                            "HIP modifier window does not intersect the mutable suffix";
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
        std::vector<HipSearchInputEvent> immutableInputTail;
        const bool windowLocalMutation =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting;
        if (windowLocalMutation) {
            const std::int64_t evaluationEndRelativeMs =
                    preparedConfiguration.evaluationEndTimeMs -
                    mutableFromTimeMs;
            std::int64_t materializationEndTimeMs =
                    evaluationEndRelativeMs;
            for (const HipSearchModifierConfiguration &modifier :
                 preparedConfiguration.modifiers) {
                materializationEndTimeMs = std::max(
                        materializationEndTimeMs,
                        modifier.window.maximumTimeMs);
            }
            hip_search_detail::SearchInputWindow inputWindow;
            if (!hip_search_detail::PartitionSearchInputWindow(
                        preparedConfiguration.baselineInputs,
                        materializationEndTimeMs,
                        &inputWindow) ||
                inputWindow.immutableTail.size() >
                        preparedConfiguration.maximumEventCount) {
                if (diagnostic != nullptr) {
                    *diagnostic =
                            "HIP search inputs cannot be partitioned at the materialization horizon";
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
                hip::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                hip::collision::ShapeCollisionCapacity;
        const std::uint64_t collisionTileStride =
                (static_cast<std::uint64_t>(
                         preparedConfiguration.maximumBatchSize) +
                 hip::collision::HipCollisionSearchTileWidth - 1u) /
                hip::collision::HipCollisionSearchTileWidth;
        const std::uint64_t collisionTileCount =
                collisionTileStride *
                hip::collision::CollisionCapacity;
        const std::uint64_t shapeCollisionTileCount =
                collisionTileStride *
                hip::collision::ShapeCollisionCapacity;
        const std::uint64_t shapeQueryCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                packedConfiguration.collisionShapes.count;
        const std::uint64_t surfaceHitCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                hip::collision::SurfaceHitCapacity;
        const std::uint64_t meshRangeCount =
                surfaceHitCount;
        const std::uint64_t meshCellCount =
                static_cast<std::uint64_t>(
                        preparedConfiguration.maximumBatchSize) *
                hip::collision::MeshCellHitCapacity;
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
                *diagnostic = "HIP search buffer dimensions overflow";
            }
            return {};
        }

        auto impl = std::make_unique<Impl>();
        impl->configuration = preparedConfiguration;
        impl->specializedModule =
                preparedConfiguration.sessionSpecialization;
        switch (packedConfiguration.tuning.handlingModel) {
        case static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_Standard):
        case static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_Lateral):
            impl->handlingSpecialization =
                    HipHandlingSpecialization::Legacy;
            break;
        case static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_GearedDrive):
            impl->handlingSpecialization =
                    packedConfiguration.water.present
                    ? HipHandlingSpecialization::
                              GearedDriveWater
                    : HipHandlingSpecialization::
                              GearedDriveDry;
            break;
        default:
            impl->handlingSpecialization =
                    HipHandlingSpecialization::Generic;
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
                [](const HipControlTick &tick) {
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
                        [&](const HipSearchModifierConfiguration &modifier) {
                            return modifier.kind ==
                                           HipSearchModifierKind::
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
                const HipSearchInputEvent &input =
                        preparedConfiguration.baselineInputs[inputIndex];
                const bool selected =
                        input.action == 4u && input.valueKind == 2u &&
                        std::any_of(
                                preparedConfiguration.modifiers.begin(),
                                preparedConfiguration.modifiers.end(),
                                [&](const HipSearchModifierConfiguration
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
        impl->sparseMutationPipeline =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting &&
                impl->baselineInputsCanonical &&
                !impl->compactRandomSteeringPipeline;
        impl->compactEditPipeline =
                !preparedConfiguration.
                         useLegacyMutationPipelineForTesting &&
                !impl->compactRandomSteeringPipeline &&
                !impl->sparseMutationPipeline;
        impl->directDeletionPipeline =
                impl->compactEditPipeline &&
                impl->baselineInputsCanonical &&
                preparedConfiguration.modifiers.size() == 1u &&
                preparedConfiguration.modifiers[0].kind ==
                        HipSearchModifierKind::InputDeletion;
        impl->directExistingEventPipeline =
                impl->compactEditPipeline &&
                impl->baselineInputsCanonical &&
                preparedConfiguration.modifiers.size() == 1u &&
                preparedConfiguration.modifiers[0].kind ==
                        HipSearchModifierKind::ExistingEvent &&
                preparedConfiguration.modifiers[0].
                                timeParameterMs == 0;
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
                        [](const HipSearchInputEvent &event) {
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
                        [&](const HipSearchModifierConfiguration &modifier) {
                            const bool heldInsertion =
                                    modifier.kind ==
                                            HipSearchModifierKind::
                                                    InputInsertion &&
                                    ((modifier.steering.enabled != 0u &&
                                      modifier.steering.maximumHoldMs > 0) ||
                                     (modifier.accelerate.enabled != 0u &&
                                      modifier.accelerate.maximumHoldMs > 0) ||
                                     (modifier.brake.enabled != 0u &&
                                      modifier.brake.maximumHoldMs > 0));
                            return heldInsertion ||
                                    (modifier.kind == HipSearchModifierKind::InputInsertion &&
                                     (modifier.optionFlags & 2u) != 0u) ||
                                    modifier.window.minimumTimeMs <
                                            0;
                        }));
        impl->needsEligibleIndices =
                impl->compactEditPipeline ||
                preparedConfiguration.useLegacyMutationPipelineForTesting ||
                std::any_of(
                        preparedConfiguration.modifiers.begin(),
                        preparedConfiguration.modifiers.end(),
                        [](const HipSearchModifierConfiguration &modifier) {
                            return modifier.kind ==
                                            HipSearchModifierKind::
                                                    ExistingEvent ||
                                    modifier.kind ==
                                            HipSearchModifierKind::
                                                    InputInsertion ||
                                    modifier.kind ==
                                            HipSearchModifierKind::
                                                    InputDeletion;
                        });
        std::vector<std::uint32_t> sharedEligibleIndices;
        if (impl->sparseMutationPipeline &&
            std::any_of(
                    preparedConfiguration.modifiers.begin(),
                    preparedConfiguration.modifiers.end(),
                    [](const HipSearchModifierConfiguration &modifier) {
                        return modifier.kind ==
                                       HipSearchModifierKind::ExistingEvent ||
                                modifier.kind ==
                                       HipSearchModifierKind::InputDeletion;
                    })) {
            sharedEligibleIndices.reserve(
                    preparedConfiguration.baselineInputs.size());
            for (std::size_t inputIndex = 0u;
                 inputIndex < preparedConfiguration.baselineInputs.size();
                 ++inputIndex) {
                const HipSearchInputEvent &event =
                        preparedConfiguration.baselineInputs[inputIndex];
                const bool eligible = std::any_of(
                        preparedConfiguration.modifiers.begin(),
                        preparedConfiguration.modifiers.end(),
                        [&](const HipSearchModifierConfiguration &modifier) {
                            if (event.timeMs <
                                        modifier.window.minimumTimeMs ||
                                event.timeMs >
                                        modifier.window.maximumTimeMs) {
                                return false;
                            }
                            if (modifier.kind ==
                                HipSearchModifierKind::ExistingEvent) {
                                return (event.action == 4u &&
                                        event.valueKind == 2u) ||
                                        ((modifier.optionFlags & 2u) != 0u &&
                                         (event.action == 1u ||
                                          event.action == 2u)) ||
                                        ((modifier.optionFlags & 4u) != 0u &&
                                         event.action == 3u);
                            }
                            if (modifier.kind ==
                                HipSearchModifierKind::InputDeletion) {
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
                                sizeof(HipSearchInputEvent) &&
                impl->editStorageBytes <=
                        eventSlots *
                                sizeof(HipSearchInputEvent);
        if (!impl->shapeCollisionScratch.Allocate(
                    shapeCollisionTileSlots) ||
            !impl->branchState.Allocate(1u) ||
            !impl->mutableBoundaryControls.Allocate(1u) ||
            !impl->baselineTicks.Allocate(
                    preparedConfiguration.baselineTicks.size()) ||
            !impl->baselineInputs.Allocate(
                    preparedConfiguration.baselineInputs.size()) ||
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
                                    HipSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !impl->finishCheckpointStates.Allocate(
                    preparedConfiguration.evaluator.kind ==
                                    HipSearchEvaluatorKind::FinishTime
                            ? candidates
                            : 0u) ||
            !impl->finishCheckpointTicks.Allocate(
                    preparedConfiguration.evaluator.kind ==
                                    HipSearchEvaluatorKind::FinishTime
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
            !impl->summary.Allocate(1u)) {
            if (diagnostic != nullptr) {
                *diagnostic = "HIP resident search allocation rejected by memory headroom or allocator limits; reduce input density or use Optimized CPU";
            }
            return {};
        }
        std::size_t reductionBytes = 0u;
        hipError_t error = hipcub::DeviceReduce::Reduce(
                nullptr, reductionBytes,
                impl->candidateBestSamples.Get(),
                impl->reducedBest.Get(),
                winnerSampleSlots,
                BetterSample{
                        MaximizesScore(
                                preparedConfiguration.evaluator.kind)},
                DeviceSample{});
        if (error != hipSuccess ||
            !impl->reductionTemporary.Allocate(reductionBytes)) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        error != hipSuccess
                        ? HipFailure(
                                  "sizing HIP winner reduction", error)
                        : "HIP winner reduction allocation failed";
            }
            return {};
        }

#define UPLOAD(allocation, source, label)                                    \
        do {                                                                  \
            if (!(source).empty()) {                                          \
                error = hipMemcpy(                                           \
                        (allocation).Get(), (source).data(),                   \
                        (source).size() * sizeof((source)[0]),                 \
                        hipMemcpyHostToDevice);                               \
                if (error != hipSuccess) {                                   \
                    if (diagnostic != nullptr) {                              \
                        *diagnostic = HipFailure(label, error);               \
                    }                                                         \
                    return {};                                                \
                }                                                             \
                impl->initialUploadBytes +=                                   \
                        (source).size() * sizeof((source)[0]);                 \
            }                                                                 \
        } while (false)
        error = hipMemcpy(
                impl->branchState.Get(),
                &preparedConfiguration.branchState,
                sizeof(preparedConfiguration.branchState),
                hipMemcpyHostToDevice);
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        HipFailure("uploading HIP branch state", error);
            }
            return {};
        }
        impl->initialUploadBytes += sizeof(preparedConfiguration.branchState);
        error = hipMemcpy(
                impl->mutableBoundaryControls.Get(),
                &inputPartition.mutableBoundaryControls,
                sizeof(inputPartition.mutableBoundaryControls),
                hipMemcpyHostToDevice);
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        HipFailure("uploading HIP branch controls", error);
            }
            return {};
        }
        impl->initialUploadBytes +=
                sizeof(inputPartition.mutableBoundaryControls);
        UPLOAD(impl->baselineTicks,
               preparedConfiguration.baselineTicks,
               "uploading HIP baseline ticks");
        UPLOAD(impl->baselineInputs,
               preparedConfiguration.baselineInputs,
               "uploading HIP baseline inputs");
        UPLOAD(impl->modifiers,
               preparedConfiguration.modifiers,
               "uploading HIP modifier configuration");
        UPLOAD(impl->smoothWeights,
               preparedConfiguration.smoothWeights,
               "uploading HIP smooth weights");
        if (preparedConfiguration.condition) {
            UPLOAD(impl->condition,
                   preparedConfiguration.condition->instructions,
                   "uploading HIP condition program");
        }
        UPLOAD(impl->compactInputIndices,
               compactInputIndices,
               "uploading HIP compact input indices");
        UPLOAD(impl->compactInputOffsets,
               compactInputOffsets,
               "uploading HIP compact input offsets");
        UPLOAD(impl->sharedEligibleIndices,
               sharedEligibleIndices,
               "uploading HIP shared eligible indices");
#undef UPLOAD
        error = hipMemcpy(
                impl->evaluator.Get(),
                &preparedConfiguration.evaluator,
                sizeof(preparedConfiguration.evaluator),
                hipMemcpyHostToDevice);
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic = HipFailure(
                        "uploading HIP evaluator configuration",
                        error);
            }
            return {};
        }
        impl->initialUploadBytes += sizeof(preparedConfiguration.evaluator);
        if (preparedConfiguration.incumbent) {
            const HipSearchIncumbent &incumbent =
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
            sample.scriptedObjectiveCount =
                    incumbent.scriptedObjectiveCount;
            for (std::uint32_t i = 0u;
                 i < sample.scriptedObjectiveCount; ++i) {
                sample.objectiveScores[i] = incumbent.objectiveScores[i];
                sample.metricValues[i] = incumbent.metricValues[i];
            }
            const std::uint32_t eventCount = static_cast<std::uint32_t>(
                    preparedConfiguration.baselineInputs.size());
            sample.eventCount = eventCount;
            error = hipMemcpy(
                    impl->globalBestSample.Get(), &sample, sizeof(sample),
                    hipMemcpyHostToDevice);
            if (error == hipSuccess) {
                error = hipMemcpy(
                        impl->globalBestEventCount.Get(), &eventCount,
                        sizeof(eventCount), hipMemcpyHostToDevice);
            }
            if (error == hipSuccess) {
                error = hipMemcpy(
                        impl->globalBestMutationCount.Get(),
                        &incumbent.mutationCount,
                        sizeof(incumbent.mutationCount),
                        hipMemcpyHostToDevice);
            }
            if (error == hipSuccess && eventCount != 0u) {
                error = hipMemcpy(
                        impl->globalBestInputs.Get(),
                        impl->baselineInputs.Get(),
                        eventCount * sizeof(HipSearchInputEvent),
                        hipMemcpyDeviceToDevice);
            }
            impl->initialUploadBytes += sizeof(sample) +
                    sizeof(eventCount) + sizeof(incumbent.mutationCount) +
                    eventCount * sizeof(HipSearchInputEvent);
            impl->baselineEvaluated = true;
        } else {
            error = hipMemset(
                    impl->globalBestSample.Get(), 0,
                    impl->globalBestSample.Bytes());
            if (error == hipSuccess) {
                error = hipMemset(
                        impl->globalBestEventCount.Get(), 0,
                        impl->globalBestEventCount.Bytes());
            }
            if (error == hipSuccess) {
                error = hipMemset(
                        impl->globalBestMutationCount.Get(), 0,
                        impl->globalBestMutationCount.Bytes());
            }
        }
        if (error != hipSuccess) {
            if (diagnostic != nullptr) {
                *diagnostic =
                        HipFailure("initializing HIP search state", error);
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
        return std::unique_ptr<HipSearchExecutor>(
                new HipSearchExecutor(std::move(impl)));
    } catch (const std::bad_alloc &) {
        if (diagnostic != nullptr) {
            *diagnostic = "HIP search host allocation failed";
        }
        return {};
    } catch (...) {
        if (diagnostic != nullptr) {
            *diagnostic = "unexpected HIP search creation failure";
        }
        return {};
    }
}

HipSearchBatchExecution HipSearchExecutor::EvaluateBaseline() noexcept {
    return EvaluateBaseline(std::function<bool()>{});
}

HipSearchBatchExecution HipSearchExecutor::EvaluateBaseline(
        const std::function<bool()> &cancellationRequested) noexcept {
    if (!impl_ || impl_->baselineEvaluated) {
        HipSearchBatchExecution result;
        result.status = HipSearchStatus::InvalidArgument;
        result.diagnostic = !impl_
                ? "HIP search executor is invalid"
                : "HIP baseline was already evaluated";
        return result;
    }
    HipSearchBatchExecution result =
            impl_->Execute(0u, 1u, true, cancellationRequested);
    result.hostToDeviceBytes += impl_->initialUploadBytes;
    return result;
}

HipSearchBatchExecution HipSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool cancellationRequested) noexcept {
    const std::function<bool()> probe = cancellationRequested
            ? std::function<bool()>([] { return true; })
            : std::function<bool()>{};
    return RunBatch(firstCandidateId, candidateCount, probe);
}

HipSearchBatchExecution HipSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        const std::function<bool()> &cancellationRequested) noexcept {
    if (!impl_) {
        HipSearchBatchExecution result;
        result.status = HipSearchStatus::InvalidArgument;
        result.diagnostic = "HIP search executor is invalid";
        return result;
    }
    if (candidateCount != 0u &&
        firstCandidateId >
                std::numeric_limits<std::uint64_t>::max() -
                        (candidateCount - 1u)) {
        HipSearchBatchExecution result;
        result.status = HipSearchStatus::InvalidArgument;
        result.firstCandidateId = firstCandidateId;
        result.candidateCount = candidateCount;
        result.diagnostic = "HIP candidate ID range overflow";
        return result;
    }
    return impl_->Execute(
            firstCandidateId, candidateCount, false,
            cancellationRequested);
}

bool HipSearchExecutor::ReserveBatchCapacity(
        std::uint32_t candidateCount,
        std::string *diagnostic) noexcept {
    try {
        if (!impl_ || candidateCount == 0u) {
            if (diagnostic != nullptr) {
                *diagnostic = !impl_
                        ? "HIP search executor is invalid"
                        : "HIP batch capacity must be positive";
            }
            return false;
        }
        return impl_->ReserveBatchCapacity(candidateCount, diagnostic);
    } catch (const std::bad_alloc &) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "HIP calibration host allocation failed";
        }
        return false;
    } catch (...) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "unexpected HIP calibration allocation failure";
        }
        return false;
    }
}

bool HipSearchExecutor::UpdateConditionTimes(
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

std::uint32_t HipSearchExecutor::BatchCapacity() const noexcept {
    return impl_ ? impl_->configuration.maximumBatchSize : 0u;
}

}  // namespace forevervalidator::simulation
