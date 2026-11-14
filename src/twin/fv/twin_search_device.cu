// Twin search (GPU engine): a batch's per-candidate host work on the device (see twin_search_device.h).
#include "twin_search_device.h"
#include "../gpu/tw_gpu_arch.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>

#include "simulation/backends/cuda/cuda_candidate_events.cuh"
#include "simulation/backends/cuda/cuda_modifier_event_ops.cuh"
#include "simulation/backends/cuda/cuda_search_branch_state.cuh"
#include "simulation/backends/cuda/cuda_search_segments.cuh"

namespace forevervalidator::simulation::twin_search {

namespace device_extract {
// The names the extracted functions use, as cuda_search_executor.cu declares them.
using cuda_search_detail::ApplyControlEvent;
using cuda_search_detail::DeviceControlState;
namespace modifier_ops = cuda_search_modifier_detail;

// FV's DeviceMt19937 (cuda_search_executor.cu; extracted below as DeviceMt19937Fv) with the same draws (Next, Uniform*:
// FV's code) and the same words, its seeding and twist rewritten for the device (round 9, G: FV's version reads back
// every word it has just stored, so each step waits for a round trip to memory, and a pass's few draws pay a whole
// twist):
// - Seeding (std::seed_seq's generate over 624 words, steps k = 0 .. 1247): step k reads words k and k + 306 and the
//   word of step k - 1, and updates words k + 306, k + 317 and k (mod 624). Word k + 306 at step k is what step k - 11
//   left in it (its word k + 317), so those 11 words stay in registers (`window`): never stored there, never loaded.
//   Words k and k + 317 were last written 306 or more steps earlier, or never (the initial 0x8b8b8b8b, known from k: no
//   fill), so a group of 11 steps loads its 22 words together and no load waits for a store. 2 stores and at most 2
//   loads a step instead of 3 and 3, and no 624-word fill.
// - Twist: its first 227 words depend only on the seeding's words i, i + 1 and i + 397, so after a seeding each draw
//   twists its own word (in place, FV's arithmetic); the rest of the twist runs only if a 228th draw comes (FV's search
//   draws a few numbers a pass and seeds again).
// DeviceGeneration::Init checks it against DeviceMt19937Fv (draw counts around the lazy twist's edges, seeding again).
class DeviceMt19937 {
public:
    __device__ DeviceMt19937(std::uint32_t *stateWords, std::uint32_t slot, std::uint32_t stride)
        : stateWords_(stateWords), slot_(slot), stride_(stride) {}

    __device__ void Seed(std::uint32_t seed, std::uint64_t candidateId, std::uint32_t passIndex) {
        const std::uint32_t s0 = seed, s1 = static_cast<std::uint32_t>(candidateId), s2 = static_cast<std::uint32_t>(candidateId >> 32u),
                            s3 = passIndex;
        constexpr std::uint32_t n = 624u, p = 306u, q = 317u, G = q - p, init = 0x8b8b8b8bu;
        // step 0: every word is still `init`
        constexpr std::uint32_t r1 = 1371501266u, r2 = r1 + 4u;
        State(p) = init + r1;
        State(0u) = r2;
        std::uint32_t window[G];   // window[j]: word k + j + p before step k + j of the current group
#pragma unroll
        for (std::uint32_t j = 0; j < G; ++j) window[j] = init;
        window[G - 1u] = init + r2;   // step 11 reads word 317, step 0's
        std::uint32_t previous = r2;  // the word of step k - 1
        std::uint32_t a[G], c[G];
        std::uint32_t k = 1u;
        for (; k + G <= 2u * n; k += G) {
#pragma unroll
            for (std::uint32_t j = 0; j < G; ++j) Load(k + j, a[j], c[j]);
#pragma unroll
            for (std::uint32_t j = 0; j < G; ++j) Step(k + j, a[j], window[j], c[j], previous, s0, s1, s2, s3);
        }
        // the last steps k .. 2n - 1 (fewer than G), then the window's words no later step reads
        const std::uint32_t tail = 2u * n - k;
#pragma unroll
        for (std::uint32_t j = 0; j < G; ++j) {
            if (j < tail) Load(k + j, a[j], c[j]);
        }
#pragma unroll
        for (std::uint32_t j = 0; j < G; ++j) {
            if (j < tail) Step(k + j, a[j], window[j], c[j], previous, s0, s1, s2, s3);
        }
#pragma unroll
        for (std::uint32_t j = 0; j < G; ++j) {
            const std::uint32_t from = j < tail ? k + j : k + j - G;   // the step that left it (>= n)
            State(Wrap(from - n + q)) = window[j];
        }
        cursor_ = n;
        lazy_ = 0u;
    }

    __device__ std::uint32_t Next() {
        std::uint32_t value;
        if (lazy_ < 227u) {   // FV's twist, part 0's step i = lazy_, on the seeding's words
            constexpr std::uint32_t upperMask = 0x80000000u, lowerMask = 0x7fffffffu, coefficient = 0x9908b0dfu;
            const std::uint32_t i = lazy_;
            const std::uint32_t current = i == 0u ? State(0u) : held_;
            const std::uint32_t next = State(i + 1u), distant = State(i + 397u);
            const std::uint32_t y = (current & upperMask) | (next & lowerMask);
            value = distant ^ (y >> 1u) ^ ((y & 1u) ? coefficient : 0u);
            State(i) = value;
            held_ = next;
            lazy_ = i + 1u;
        } else {
            if (lazy_ == 227u) {   // the rest of the twist, then FV's draws from word 227
                TwistFrom(227u);
                cursor_ = 227u;
                lazy_ = kNotLazy;
            }
            if (cursor_ >= 624u) {
                TwistFrom(0u);
            }
            value = State(cursor_++);
        }
        value ^= value >> 11u;
        value ^= (value << 7u) & 0x9d2c5680u;
        value ^= (value << 15u) & 0xefc60000u;
        value ^= value >> 18u;
        return value;
    }

    __device__ std::uint64_t UniformUnsigned(std::uint64_t minimum, std::uint64_t maximum) {
        if (minimum > maximum) {
            const std::uint64_t swap = minimum;
            minimum = maximum;
            maximum = swap;
        }
        const std::uint64_t range = maximum - minimum;
        std::uint64_t result = 0u;
        if (range < UINT32_MAX) {
            const std::uint32_t extendedRange = static_cast<std::uint32_t>(range + 1u);
            std::uint64_t product = static_cast<std::uint64_t>(Next()) * extendedRange;
            std::uint32_t low = static_cast<std::uint32_t>(product);
            if (low < extendedRange) {
                const std::uint32_t threshold = static_cast<std::uint32_t>(-extendedRange) % extendedRange;
                while (low < threshold) {
                    product = static_cast<std::uint64_t>(Next()) * extendedRange;
                    low = static_cast<std::uint32_t>(product);
                }
            }
            result = product >> 32u;
        } else if (range == UINT32_MAX) {
            result = Next();
        } else {
            do {
                constexpr std::uint64_t generatorRange = UINT64_C(1) << 32u;
                const std::uint64_t high = UniformUnsigned(0u, range / generatorRange);
                const std::uint64_t temporary = generatorRange * high;
                result = temporary + Next();
                if (result <= range && result >= temporary) {
                    break;
                }
            } while (true);
        }
        return result + minimum;
    }

    __device__ std::uint32_t UniformU32(std::uint32_t minimum, std::uint32_t maximum) {
        return static_cast<std::uint32_t>(UniformUnsigned(minimum, maximum));
    }

    __device__ std::int32_t UniformS32(std::int32_t minimum, std::int32_t maximum) {
        if (minimum > maximum) {
            const std::int32_t swap = minimum;
            minimum = maximum;
            maximum = swap;
        }
        const std::uint32_t unsignedMinimum = static_cast<std::uint32_t>(minimum);
        const std::uint32_t range = static_cast<std::uint32_t>(maximum) - unsignedMinimum;
        return static_cast<std::int32_t>(static_cast<std::uint32_t>(UniformUnsigned(0u, range)) + unsignedMinimum);
    }

    __device__ std::int64_t UniformS64(std::int64_t minimum, std::int64_t maximum) {
        if (minimum > maximum) {
            const std::int64_t swap = minimum;
            minimum = maximum;
            maximum = swap;
        }
        const std::uint64_t unsignedMinimum = static_cast<std::uint64_t>(minimum);
        const std::uint64_t range = static_cast<std::uint64_t>(maximum) - unsignedMinimum;
        return static_cast<std::int64_t>(UniformUnsigned(0u, range) + unsignedMinimum);
    }

private:
    static constexpr std::uint32_t kNotLazy = 0xFFFFFFFFu;

    __device__ std::uint32_t &State(std::uint32_t index) { return stateWords_[static_cast<std::uint64_t>(index) * stride_ + slot_]; }
    __device__ static std::uint32_t Wrap(std::uint32_t i) { return i >= 624u ? i - 624u : i; }

    // the seeding's step k: its words k (a) and k + 317 (c) as they are before it (the initial value if never written)
    __device__ void Load(std::uint32_t k, std::uint32_t &a, std::uint32_t &c) {
        constexpr std::uint32_t n = 624u, p = 306u, q = 317u, init = 0x8b8b8b8bu;
        const std::uint32_t kn = k < n ? k : k - n;
        a = k >= p ? State(kn) : init;
        c = k >= p + 1u ? State(Wrap(kn + q)) : init;
    }
    // step k (FV's: State(kpn) op= r1; State(kqn) op= r2; State(kn) = r2): w is word k + 306 before it and becomes its word
    // k + 317 (step k + 11's w)
    __device__ void Step(std::uint32_t k, std::uint32_t a, std::uint32_t &w, std::uint32_t c, std::uint32_t &previous, std::uint32_t s0,
                         std::uint32_t s1, std::uint32_t s2, std::uint32_t s3) {
        constexpr std::uint32_t n = 624u, p = 306u;
        const std::uint32_t kn = k < n ? k : k - n;
        std::uint32_t r;
        if (k < n) {
            const std::uint32_t argument = a ^ w ^ previous;
            const std::uint32_t x = 1664525u * (argument ^ (argument >> 27u));
            r = x + kn + (k == 1u ? s0 : k == 2u ? s1 : k == 3u ? s2 : k == 4u ? s3 : 0u);
            State(Wrap(kn + p)) = w + x;
            w = c + r;
        } else {
            const std::uint32_t argument = a + w + previous;
            const std::uint32_t x = 1566083941u * (argument ^ (argument >> 27u));
            r = x - kn;
            State(Wrap(kn + p)) = w ^ x;
            w = c ^ r;
        }
        State(kn) = r;
        previous = r;
    }

    // FV's twist from word `begin` (0, or 227 after the lazy part): words 0..226 from words i + 1 and i + 397 (not yet
    // rewritten), 227..622 from i + 1 and i - 227 (rewritten 227 steps before), 623 from 0 and 396 (rewritten): groups of G
    // with their reads first.
    __device__ void TwistFrom(std::uint32_t begin) {
        constexpr std::uint32_t upperMask = 0x80000000u, lowerMask = 0x7fffffffu, coefficient = 0x9908b0dfu;
        constexpr std::uint32_t G = 8u;
        std::uint32_t current = State(begin);
        for (int part = begin == 0u ? 0 : 1; part < 2; ++part) {
            const std::uint32_t from = part == 0 ? 0u : 227u, end = part == 0 ? 227u : 623u;
            std::uint32_t i = from;
            for (; i + G <= end; i += G) {
                std::uint32_t next[G], distant[G];
#pragma unroll
                for (std::uint32_t j = 0; j < G; ++j) {
                    next[j] = State(i + j + 1u);
                    distant[j] = State(part == 0 ? i + j + 397u : i + j - 227u);
                }
#pragma unroll
                for (std::uint32_t j = 0; j < G; ++j) {
                    const std::uint32_t value = (current & upperMask) | (next[j] & lowerMask);
                    State(i + j) = distant[j] ^ (value >> 1u) ^ ((value & 1u) ? coefficient : 0u);
                    current = next[j];
                }
            }
            for (; i < end; ++i) {
                const std::uint32_t next = State(i + 1u);
                const std::uint32_t distant = State(part == 0 ? i + 397u : i - 227u);
                const std::uint32_t value = (current & upperMask) | (next & lowerMask);
                State(i) = distant ^ (value >> 1u) ^ ((value & 1u) ? coefficient : 0u);
                current = next;
            }
        }
        const std::uint32_t value = (State(623u) & upperMask) | (State(0u) & lowerMask);
        State(623u) = State(396u) ^ (value >> 1u) ^ ((value & 1u) ? coefficient : 0u);
        cursor_ = begin;
    }

    std::uint32_t *stateWords_ = nullptr;
    std::uint32_t slot_ = 0u;
    std::uint32_t stride_ = 0u;
    std::uint32_t cursor_ = 624u;
    std::uint32_t lazy_ = kNotLazy;   // after a seeding: the words twisted so distant (0..227); kNotLazy: FV's cursor
    std::uint32_t held_ = 0u;         // the lazy twist's word lazy_ as the seeding left it
};

#include "twin_cuda_search_extract_device.inc"
}  // namespace device_extract

namespace {

using cuda_search_detail::DeviceControlState;
using tw::gpu::JudgeState;
using tw::gpu::SearchJob;
using tw::gpu::SearchResult;

bool Check(cudaError_t e, const char *what, std::string *error) {
    if (e == cudaSuccess) return true;
    if (error != nullptr) *error = std::string("twin search device: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

struct Buffer {
    void *p = nullptr;
    std::size_t bytes = 0;
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
        if (p != nullptr) cudaFree(p);
    }
    bool Fit(std::size_t n, std::string *error) {
        if (n <= bytes) return true;
        if (p != nullptr) cudaFree(p);
        p = nullptr;
        bytes = 0;
        if (!Check(cudaMalloc(&p, n), "cudaMalloc", error)) return p = nullptr, false;
        bytes = n;
        return true;
    }
    template <class T> T *As() const { return static_cast<T *>(p); }
};

struct HostBuffer {
    void *p = nullptr;
    std::size_t bytes = 0;
    HostBuffer() = default;
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer &operator=(const HostBuffer &) = delete;
    ~HostBuffer() {
        if (p != nullptr) cudaFreeHost(p);
    }
    bool Fit(std::size_t n, std::string *error) {
        if (n <= bytes) return true;
        if (p != nullptr) cudaFreeHost(p);
        p = nullptr;
        bytes = 0;
        if (!Check(cudaMallocHost(&p, n), "cudaMallocHost", error)) return p = nullptr, false;
        bytes = n;
        return true;
    }
    template <class T> T *As() const { return static_cast<T *>(p); }
};

// The program and the session's tables (device pointers), one generator thread's scratch, a chunk's outputs.
struct GenParams {
    const CudaSearchInputEvent *base;
    const CudaSearchModifierConfiguration *modifiers;
    const CudaSearchMutationSegment *segments;
    const double *smoothWeights;
    const std::uint32_t *actionBits, *snap;
    const JudgeState *prefix;
    std::uint32_t baseCount, modifierCount, segmentCount, changedSegmentCount;
    std::uint32_t tickDurationMs, capacity, tailCount, ticks, snapCount;
    std::uint32_t prefixStop, n, gen;
    std::int64_t branchTimeMs, mutableFromTimeMs;
    std::uint64_t firstId;
    DeviceControlState boundary;
    JudgeState initial;
    std::uint8_t canonical, baseline, stopStatus;
    // scratch: MT19937 words (word-major, `gen` threads), each thread's events / temporary / pass baseline / eligible
    std::uint32_t *mt;
    CudaSearchInputEvent *events, *temporary, *passBaseline;
    std::uint32_t *eligible;
    // outputs
    std::uint32_t *controls;   // slot-major rows of `ticks` words
    std::uint32_t *first;      // the first changed tick of a physics slot, UINT32_MAX for the others (the sort key)
    DeviceSlot *slots;
    std::uint32_t *counters;   // [0] physics slots, [1] jobs
};

// CandidateGenerator::Generate (twin_search_candidates.cpp) on the device: GenerateSearchCandidatesKernel's
// materialized path over the window-local inputs, normalising after every pass. Returns the event count;
// *status 2 = capacity exceeded.
__device__ std::uint32_t GenerateOne(const GenParams &P, std::uint64_t candidateId, device_extract::DeviceMt19937 &random,
                                     CudaSearchInputEvent *events, CudaSearchInputEvent *temporary,
                                     CudaSearchInputEvent *passBaseline, std::uint32_t *eligible, std::uint8_t *status,
                                     std::uint32_t *mutationCount) {
    const CudaSearchInputEvent *base = P.base;
    const std::uint32_t baseCount = P.baseCount, capacity = P.capacity;
    for (std::uint32_t i = 0u; i < baseCount; ++i) events[i] = base[i];
    std::uint32_t eventCount = baseCount;
    *status = 0u;
    *mutationCount = 0u;
    if (!P.baseline) {
        const std::uint32_t modifierCount = P.modifierCount;
        const std::uint32_t segmentCount = P.segmentCount;
        bool normalized = P.canonical != 0u;
        const cuda_search_detail::SegmentSelection selection{
                candidateId ^ (static_cast<std::uint64_t>(P.modifiers[0].window.seed) << 32u), segmentCount,
                P.changedSegmentCount};
        for (std::uint32_t segmentIndex = 0u; segmentIndex < (segmentCount == 0u ? 1u : segmentCount); ++segmentIndex) {
            if (segmentCount != 0u && !selection.Selected(segmentIndex)) continue;
            for (std::uint32_t pass = 0u; pass < modifierCount; ++pass) {
                CudaSearchModifierConfiguration modifier = P.modifiers[pass];
                if (segmentCount != 0u) {
                    const CudaSearchMutationSegment segment = P.segments[segmentIndex];
                    modifier.window.minimumTimeMs = modifier.window.minimumTimeMs > segment.minimumTimeMs
                                                            ? modifier.window.minimumTimeMs
                                                            : segment.minimumTimeMs;
                    modifier.window.maximumTimeMs = modifier.window.maximumTimeMs < segment.maximumTimeMs
                                                            ? modifier.window.maximumTimeMs
                                                            : segment.maximumTimeMs;
                    if (modifier.window.minimumTimeMs > modifier.window.maximumTimeMs) continue;
                }
                if (!device_extract::ApplyModifier(modifier, pass + segmentIndex * modifierCount, candidateId, random,
                                                   P.tickDurationMs, 0, P.boundary, base, baseCount, events, &eventCount,
                                                   capacity, temporary, passBaseline, eligible, P.smoothWeights, true,
                                                   &normalized)) {
                    *status = 2u;
                    return eventCount;
                }
            }
            if (segmentCount != 0u) {
                if (!normalized) {
                    eventCount = device_extract::NormalizeEvents(events, eventCount, temporary, base, baseCount, 0, true,
                                                                 capacity);
                    if (eventCount == UINT32_MAX) {
                        *status = 2u;
                        return capacity;
                    }
                }
                const std::int64_t end = P.segments[segmentIndex].maximumTimeMs;
                if (!selection.Contains(P.segments, end + P.tickDurationMs) &&
                    !cuda_search_detail::RestoreSegmentControls(base, baseCount, events, &eventCount, capacity, P.boundary,
                                                                end)) {
                    *status = 2u;
                    return eventCount;
                }
                normalized = false;
            }
            for (std::uint32_t i = 0u; i < baseCount; ++i) passBaseline[i] = base[i];
            eventCount = device_extract::NormalizeEvents(events, eventCount, temporary, passBaseline, baseCount, 0, true,
                                                         capacity);
            if (eventCount == UINT32_MAX) {
                *status = 2u;
                return capacity;
            }
            normalized = true;
        }
    }
    std::uint32_t mutations = P.baseline ? 0u : device_extract::EffectiveChangeCount(base, baseCount, events, eventCount);
    if (!P.baseline && eventCount != baseCount) mutations += P.tailCount;
    *mutationCount = mutations;
    return eventCount;
}

__device__ __forceinline__ void JudgeInto(DeviceSlot &d, const JudgeState &j) {
    d.score = j.score;
    d.timeMs = j.timeMs;
    d.detail0 = j.detail0;
    d.detail1 = j.detail1;
    d.evaluationTick = j.evaluationTick;
    d.valid = j.valid;
    d.preciseFinish = j.preciseFinish;
    d.flags |= kSlotJudge;
}

// GenerateKernel's block, 3 a SM (at most 168 registers: with 2 a SM, 8,192 threads' 64 blocks took two waves on 30 SMs)
constexpr std::uint32_t kGenBlock = 128u;

// (A) of the batch path per slot: generation, controls, the first change, the shared prefix's judge. Grid-stride:
// `gen` threads, each with its own scratch.
__device__ __forceinline__ void GenerateSlots(const GenParams &P, std::uint32_t g, CudaSearchInputEvent *events,
                                              CudaSearchInputEvent *temporary, CudaSearchInputEvent *passBaseline,
                                              std::uint32_t *eligible) {
    device_extract::DeviceMt19937 random(P.mt, g, P.gen);
    for (std::uint32_t s = g; s < P.n; s += P.gen) {
        DeviceSlot d{};
        d.job = UINT32_MAX;
        std::uint8_t status = 0u;
        std::uint32_t mutations = 0u;
        const std::uint32_t eventCount =
                GenerateOne(P, P.firstId + s, random, events, temporary, passBaseline, eligible, &status, &mutations);
        std::uint32_t firstKey = UINT32_MAX;
        d.eventCount = eventCount;
        if (status != 0u) {   // capacity exceeded: inactive, no mutation
            d.status = 2u;
            d.eventCount = eventCount < P.capacity ? eventCount : P.capacity;
        } else {
            d.mutationCount = mutations;
            const bool active = P.baseline != 0u || mutations != 0u;
            if (active) {
                d.flags = kSlotActive;
                // BuildTwinControls, and the first tick whose control differs from the baseline's
                std::uint32_t *row = P.controls + static_cast<std::size_t>(s) * P.ticks;
                DeviceControlState control = P.boundary;
                std::uint32_t next = 0u, first = 0u;
                bool differ = false;
                for (std::uint32_t t = 0u; t < P.ticks; ++t) {
                    const std::int64_t publicTime = P.branchTimeMs + static_cast<std::int64_t>(t + 1u) * P.tickDurationMs;
                    const std::int64_t suffixTime = publicTime - P.mutableFromTimeMs;
                    while (next < eventCount && events[next].timeMs <= suffixTime) {
                        cuda_search_detail::ApplyControlEvent(control, events[next], P.mutableFromTimeMs);
                        ++next;
                    }
                    const auto c = cuda_search_detail::ControlsFromState(control);
                    std::uint32_t word = static_cast<std::uint32_t>(static_cast<std::int32_t>(c.steering * 65536.0f)) & 0x3FFFFu;
                    if (c.lowSpeedGateA > 0.0f) word |= 1u << 18;
                    if (c.lowSpeedGateB > 0.0f) word |= 1u << 19;
                    word |= P.actionBits[t];
                    row[t] = word;
                    if (!differ) {
                        if (t < P.snapCount && word == P.snap[t]) first = t + 1u;
                        else differ = true;
                    }
                }
                d.first = first;
                // the shared prefix's judge (PrepareSlot)
                if (first > 0u && P.prefixStop < first) {
                    JudgeInto(d, P.prefix[P.prefixStop]);
                    d.status = P.stopStatus;
                } else if (first == P.ticks) {
                    JudgeInto(d, P.prefix[first - 1u]);
                } else {
                    d.flags |= kSlotPhysics;
                    firstKey = first;
                    atomicAdd(&P.counters[0], 1u);
                }
            }
        }
        P.first[s] = firstKey;
        P.slots[s] = d;
    }
}

// The thread's event arrays (FV's events, temporary, pass baseline, eligible): kLocal > 0 (capacity <= kLocal), in local
// memory, which the hardware interleaves across a warp's threads word by word, so the warp's accesses to the same event index
// (the copies, the normalization's passes) coalesce; 0, in the set's global scratch, thread-major (each thread's array
// contiguous: a warp's access to one index touches 32 sectors, 4-5 times slower). Three sizes: a promoted baseline with a
// steering event every tick passes 128 within a second of window.
constexpr std::uint32_t kLocalEvents = 128u;
constexpr std::uint32_t kLocalEventsLarge = 256u;
constexpr std::uint32_t kLocalEventsHuge = 512u;
template <std::uint32_t kLocal>
__global__ void __launch_bounds__(kGenBlock, 3) GenerateKernel(const TW_GRID_CONSTANT GenParams P) {
    const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= P.gen) return;
    if constexpr (kLocal != 0u) {
        uint4 events[kLocal], temporary[kLocal], passBaseline[kLocal];   // no initializers (CudaSearchInputEvent zeroes its fields)
        std::uint32_t eligible[kLocal];
        GenerateSlots(P, g, reinterpret_cast<CudaSearchInputEvent *>(events), reinterpret_cast<CudaSearchInputEvent *>(temporary),
                      reinterpret_cast<CudaSearchInputEvent *>(passBaseline), eligible);
    } else {
        const std::size_t cap = P.capacity > 0u ? P.capacity : 1u;
        GenerateSlots(P, g, P.events + g * cap, P.temporary + g * cap, P.passBaseline + g * cap, P.eligible + g * cap);
    }
}

// The pool's order: physics slots first, by first changed tick, then their controls from there (numeric words),
// then slot; the rest after them.
struct SlotOrder {
    const std::uint32_t *first;
    const std::uint32_t *controls;
    std::uint32_t ticks;
    std::uint32_t rows;   // 0: first change, then slot (measurement: duplicates are not all neighbours, so fewer fold)
    __device__ bool operator()(std::uint32_t a, std::uint32_t b) const {
        const std::uint32_t fa = first[a], fb = first[b];
        if (fa != fb) return fa < fb;
        if (fa != UINT32_MAX && rows != 0u) {
            const std::uint32_t *ra = controls + static_cast<std::size_t>(a) * ticks;
            const std::uint32_t *rb = controls + static_cast<std::size_t>(b) * ticks;
            for (std::uint32_t t = fa; t < ticks; ++t) {
                if (ra[t] != rb[t]) return ra[t] < rb[t];
            }
        }
        return a < b;
    }
};

__global__ void IotaKernel(std::uint32_t *keys, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keys[i] = i;
}

// One pass of a merge sort without shared memory (cub's merge sort runs at a large shared memory configuration and its
// partition kernel at a small one, so beside the lane runner's blocks each waits for an SM of its own): the runs of `width` in
// `in` merged in pairs into `out`, each key placed by its rank in the other run (the order is total: no two keys are equal).
__global__ void MergePassKernel(const std::uint32_t *in, std::uint32_t *out, std::uint32_t n, std::uint32_t width, const SlotOrder order) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t key = in[i];
    const std::uint32_t base = i / (2u * width) * (2u * width);
    const bool left = i - base < width;
    const std::uint32_t lo = left ? min(n, base + width) : base;   // the other run [lo, hi)
    const std::uint32_t hi = left ? min(n, base + 2u * width) : base + width;
    std::uint32_t a = lo, b = hi;   // its first key not before `key`
    while (a < b) {
        const std::uint32_t m = (a + b) >> 1;
        if (order(in[m], key))
            a = m + 1u;
        else
            b = m;
    }
    out[base + (left ? i - base : i - base - width) + (a - lo)] = key;
}

// The inclusive sum of in[0, n) into out without shared memory (cub's scan takes 32 KB of shared memory a block, so beside the
// lane runner's blocks it waits for an SM of its own): each warp's 32 values (ScanWarps, the warps' sums), the sums' exclusive
// prefix by one warp (ScanSums), each warp's prefix added (ScanAdd).
__global__ void ScanWarpsKernel(const std::uint32_t *in, std::uint32_t *out, std::uint32_t *sums, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u;
    std::uint32_t v = i < n ? in[i] : 0u;
    for (std::uint32_t d = 1u; d < 32u; d <<= 1) {
        const std::uint32_t u = __shfl_up_sync(0xFFFFFFFFu, v, d);
        if (lane >= d) v += u;
    }
    if (i < n) out[i] = v;
    if (lane == 31u && (i >> 5) < (n + 31u) / 32u) sums[i >> 5] = v;
}
__global__ void ScanSumsKernel(std::uint32_t *sums, std::uint32_t m) {   // one warp: the m sums' exclusive prefix, in place
    const std::uint32_t lane = threadIdx.x, per = (m + 31u) / 32u, b = lane * per, e = min(m, b + per);
    std::uint32_t t = 0u;
    for (std::uint32_t k = b; k < e; ++k) t += sums[k];
    std::uint32_t x = t;
    for (std::uint32_t d = 1u; d < 32u; d <<= 1) {
        const std::uint32_t u = __shfl_up_sync(0xFFFFFFFFu, x, d);
        if (lane >= d) x += u;
    }
    std::uint32_t run = x - t;
    for (std::uint32_t k = b; k < e; ++k) {
        const std::uint32_t v = sums[k];
        sums[k] = run;
        run += v;
    }
}
__global__ void ScanAddKernel(std::uint32_t *out, const std::uint32_t *sums, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += sums[i >> 5];
}

// A physics slot equal to the one before it in the order (same first change, same controls) is a duplicate.
__global__ void HeadsKernel(const std::uint32_t *order, const std::uint32_t *first, const std::uint32_t *controls, std::uint32_t ticks,
                            const std::uint32_t *counters, std::uint32_t n, std::uint32_t *head) {
    const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const std::uint32_t physics = counters[0];
    std::uint32_t h = 0u;
    if (p < physics) {
        h = 1u;
        if (p > 0u) {
            const std::uint32_t a = order[p], b = order[p - 1u];
            if (first[a] == first[b]) {
                const std::uint32_t *ra = controls + static_cast<std::size_t>(a) * ticks;
                const std::uint32_t *rb = controls + static_cast<std::size_t>(b) * ticks;
                bool same = true;
                for (std::uint32_t t = first[a]; t < ticks && same; ++t) same = ra[t] == rb[t];
                h = same ? 0u : 1u;
            }
        }
    }
    head[p] = h;
}

// The jobs in pool order (a run's first slot), every physics slot's job.
__global__ void JobsKernel(const TW_GRID_CONSTANT GenParams P, const std::uint32_t *order, const std::uint32_t *head,
                           const std::uint32_t *rank, SearchJob *jobs) {
    const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t physics = P.counters[0];
    if (p >= physics) return;
    const std::uint32_t s = order[p];
    const std::uint32_t j = rank[p] - 1u;
    P.slots[s].job = j;
    if (head[p] != 0u) {
        const std::uint32_t f = P.first[s];
        SearchJob job;
        job.start = f;   // 0: the branch, k: the snapshot after tick k - 1
        job.controls = static_cast<std::uint32_t>(static_cast<std::size_t>(s) * P.ticks + f);
        job.ticks = P.ticks - f;
        job.firstTick = f;
        job.judge = f > 0u ? P.prefix[f - 1u] : P.initial;
        jobs[j] = job;
    }
    if (p == physics - 1u) P.counters[1] = rank[p];
}

// Every physics slot takes its job's result.
__global__ void ResultsKernel(DeviceSlot *slots, std::uint32_t n, const SearchResult *results, bool pose) {
    const std::uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    DeviceSlot d = slots[s];
    if ((d.flags & kSlotPhysics) == 0u) return;
    const SearchResult &r = results[d.job];
    if (r.lost != 0u || r.poseOverflow != 0u) {
        d.flags |= kSlotHost;
    } else {
        d.status = r.verdict == tw::gpu::kJudgeUnsupported ? 3u : 0u;
        if (pose) d.flags |= kSlotPose;
        else JudgeInto(d, r.judge);
    }
    slots[s] = d;
}

// DeviceMt19937 against FV's DeviceMt19937Fv: per thread its own (seed, id, pass), 2,000 draws (the lazy twist, its rest, full
// twists) and one draw of each Uniform kind, then seedings again over the used state with draw counts around the lazy twist's
// edges (each thread 4 of them); *bad counts the threads that differ.
__global__ void GeneratorCheckKernel(std::uint32_t *wordsA, std::uint32_t *wordsB, std::uint32_t threads, unsigned *bad) {
    const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;
    device_extract::DeviceMt19937 a(wordsA, t, threads);
    device_extract::DeviceMt19937Fv b(wordsB, t, threads);
    const std::uint32_t seed = 0x9e3779b9u * (t + 1u);
    const std::uint64_t id = (static_cast<std::uint64_t>(t) << 33u) ^ (0x123456789abcdefull * (t + 7u));
    a.Seed(seed, id, t & 7u);
    b.Seed(seed, id, t & 7u);
    bool same = true;
    for (int k = 0; k < 2000; ++k) same = same && a.Next() == b.Next();
    same = same && a.UniformS32(-65536, 65536) == b.UniformS32(-65536, 65536) && a.UniformU32(0u, t) == b.UniformU32(0u, t) &&
           a.UniformS64(-5, 1ll << 40) == b.UniformS64(-5, 1ll << 40);
    const std::uint32_t counts[13]{0u, 1u, 2u, 5u, 226u, 227u, 228u, 229u, 623u, 624u, 625u, 851u, 852u};
    for (std::uint32_t r = 0; r < 4u; ++r) {   // seeding again over a used state
        const std::uint32_t c = counts[(t + r * 5u) % 13u];
        a.Seed(seed ^ (r + 1u), id + r + 1u, 3u + r);
        b.Seed(seed ^ (r + 1u), id + r + 1u, 3u + r);
        for (std::uint32_t k = 0; k < c; ++k) same = same && a.Next() == b.Next();
        same = same && a.UniformS32(-65536, 65536) == b.UniformS32(-65536, 65536) && a.UniformU32(0u, t) == b.UniformU32(0u, t);
    }
    a.Seed(seed ^ 1u, id + 1u, 3u);
    b.Seed(seed ^ 1u, id + 1u, 3u);
    for (int k = 0; k < 700; ++k) same = same && a.Next() == b.Next();
    if (!same) atomicAdd(bad, 1u);
}

constexpr std::uint32_t kBlock = 128u;
// The local event arrays' size for a capacity (0: the global scratch). Local memory is reserved for every thread the device
// can hold: a size whose reservation passes an eighth of the device's memory is not used.
// FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN_EVENTS=global (measurement): the generator's event arrays in the global scratch.
std::uint32_t LocalEvents(std::uint32_t capacity) {
    static const bool global = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN_EVENTS");
        return v != nullptr && std::strcmp(v, "global") == 0;
    }();
    static const std::uint64_t budget = [] {
        int device = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&p, device) != cudaSuccess) return std::uint64_t{0};
        const std::uint64_t threads = static_cast<std::uint64_t>(p.maxThreadsPerMultiProcessor) * p.multiProcessorCount;
        return threads == 0u ? 0u : static_cast<std::uint64_t>(p.totalGlobalMem) / 8u / threads;   // bytes a thread
    }();
    if (global) return 0u;
    for (const std::uint32_t size : {kLocalEvents, kLocalEventsLarge, kLocalEventsHuge}) {
        const std::uint64_t bytes = std::uint64_t{size} * (3u * sizeof(CudaSearchInputEvent) + sizeof(std::uint32_t));
        if (capacity <= size) return size == kLocalEvents || bytes <= budget ? size : 0u;
    }
    return 0u;
}
bool UseLocalEvents(std::uint32_t capacity) { return LocalEvents(capacity) != 0u; }
inline std::uint32_t Blocks(std::size_t n) { return static_cast<std::uint32_t>((n + kBlock - 1u) / kBlock); }

}  // namespace

struct DeviceGeneration::Dev {
    // the session
    Buffer base, modifiers, segments, smoothWeights, actionBits, snap;
    GenParams params{};
    std::uint32_t gen = 0u;   // generator threads per set
    bool pose = false;        // Pose: the jobs' results to the host too
    struct Set {
        Buffer mt, events, temporary, passBaseline, eligible;
        Buffer controls, first, slots, counters, prefix;
        Buffer order, orderTmp, head, rank, jobs, sortTemp, scanTemp, sums;
        HostBuffer hSlots, hResults, hCounters;
        cudaEvent_t g0 = nullptr, g1 = nullptr, g2 = nullptr;   // the generation's kernels, the outcomes copied (trace)
        std::size_t slotsCap = 0u;
        std::uint32_t n = 0u;
        bool pose = false;
        ~Set() {
            if (g0 != nullptr) cudaEventDestroy(g0);
            if (g1 != nullptr) cudaEventDestroy(g1);
            if (g2 != nullptr) cudaEventDestroy(g2);
        }
    } sets[tw::gpu::SearchEngine::kMaxSets];
    int used = 2;   // SetSets
};

DeviceGeneration::DeviceGeneration() = default;
DeviceGeneration::~DeviceGeneration() = default;

bool DeviceGeneration::Init(const MutationProgram &program, std::int64_t branchTimeMs, const std::uint32_t *actionBits,
                            std::uint32_t ticks, const std::uint32_t *snap, std::uint32_t snapCount, const JudgeState &initial,
                            bool pose, std::string *error) {
    auto d = std::make_unique<Dev>();
    d->pose = pose;
    auto put = [&](Buffer &b, const void *src, std::size_t bytes) {
        if (!b.Fit(std::max<std::size_t>(bytes, 16u), error)) return false;
        return bytes == 0u || Check(cudaMemcpy(b.p, src, bytes, cudaMemcpyHostToDevice), "upload", error);
    };
    if (!put(d->base, program.baselineInputs.data(), sizeof(CudaSearchInputEvent) * program.baselineInputs.size()) ||
        !put(d->modifiers, program.modifiers.data(), sizeof(CudaSearchModifierConfiguration) * program.modifiers.size()) ||
        !put(d->segments, program.segments.data(), sizeof(CudaSearchMutationSegment) * program.segments.size()) ||
        !put(d->smoothWeights, program.smoothWeights.data(), sizeof(double) * program.smoothWeights.size()) ||
        !put(d->actionBits, actionBits, sizeof(std::uint32_t) * ticks) || !put(d->snap, snap, sizeof(std::uint32_t) * snapCount))
        return false;
    GenParams &P = d->params;
    P = GenParams{};
    P.base = d->base.As<CudaSearchInputEvent>();
    P.modifiers = d->modifiers.As<CudaSearchModifierConfiguration>();
    P.segments = d->segments.As<CudaSearchMutationSegment>();
    P.smoothWeights = program.smoothWeights.empty() ? nullptr : d->smoothWeights.As<double>();
    P.actionBits = d->actionBits.As<std::uint32_t>();
    P.snap = d->snap.As<std::uint32_t>();
    P.baseCount = static_cast<std::uint32_t>(program.baselineInputs.size());
    P.modifierCount = static_cast<std::uint32_t>(program.modifiers.size());
    P.segmentCount = static_cast<std::uint32_t>(program.segments.size());
    P.tickDurationMs = program.tickDurationMs;
    P.capacity = program.eventCapacity;
    P.tailCount = static_cast<std::uint32_t>(program.immutableTail.size());
    P.ticks = ticks;
    P.snapCount = std::min(snapCount, ticks);
    P.branchTimeMs = branchTimeMs;
    P.mutableFromTimeMs = program.mutableFromTimeMs;
    P.boundary = program.mutableBoundaryControls;
    P.initial = initial;
    P.canonical = program.baselineInputsCanonical ? 1u : 0u;
    // generator threads per set (FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN_THREADS): each owns its scratch
    d->gen = 8192u;
    if (const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN_THREADS"); v != nullptr && std::atoi(v) >= 32) {
        d->gen = static_cast<std::uint32_t>(std::atoi(v));
    }
    for (Dev::Set &s : d->sets) {
        if (!Check(cudaEventCreate(&s.g0), "cudaEventCreate", error) || !Check(cudaEventCreate(&s.g1), "cudaEventCreate", error) ||
            !Check(cudaEventCreate(&s.g2), "cudaEventCreate", error))
            return false;
    }
    {   // every kernel loaded now (lazy loading: a kernel's first launch makes the work queued after it, on any stream, wait
        // for the work queued before it, so a batch run ahead would wait for the first batch's kernels)
        cudaFuncAttributes a;
        if (!Check(cudaFuncGetAttributes(&a, GenerateKernel<kLocalEvents>), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, GenerateKernel<kLocalEventsLarge>), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, GenerateKernel<kLocalEventsHuge>), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, GenerateKernel<0u>), "load", error) || !Check(cudaFuncGetAttributes(&a, IotaKernel), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, MergePassKernel), "load", error) || !Check(cudaFuncGetAttributes(&a, HeadsKernel), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, ScanWarpsKernel), "load", error) || !Check(cudaFuncGetAttributes(&a, ScanSumsKernel), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, ScanAddKernel), "load", error) || !Check(cudaFuncGetAttributes(&a, JobsKernel), "load", error) ||
            !Check(cudaFuncGetAttributes(&a, ResultsKernel), "load", error))
            return false;
    }
    {   // the device generator draws FV's numbers
        constexpr std::uint32_t threads = 4096u;
        Buffer a, b, bad;
        unsigned count = 0u;
        if (!a.Fit(sizeof(std::uint32_t) * 624u * threads, error) || !b.Fit(sizeof(std::uint32_t) * 624u * threads, error) ||
            !bad.Fit(sizeof(unsigned), error) || !Check(cudaMemset(bad.p, 0, sizeof(unsigned)), "memset", error))
            return false;
        GeneratorCheckKernel<<<Blocks(threads), kBlock>>>(a.As<std::uint32_t>(), b.As<std::uint32_t>(), threads, bad.As<unsigned>());
        if (!Check(cudaGetLastError(), "generator check", error) ||
            !Check(cudaMemcpy(&count, bad.p, sizeof count, cudaMemcpyDeviceToHost), "generator check", error))
            return false;
        if (count != 0u) {
            if (error != nullptr) *error = "twin search device: the device MT19937 differs from FV's (" + std::to_string(count) + " of 4096)";
            return false;
        }
    }
    dev_ = std::move(d);
    return true;
}

void DeviceGeneration::SetSets(int sets) { dev_->used = std::max(2, std::min(sets, tw::gpu::SearchEngine::kMaxSets)); }

bool DeviceGeneration::Rebase(const MutationProgram &program, std::int64_t branchTimeMs, const std::uint32_t *actionBits,
                              std::uint32_t ticks, const std::uint32_t *snap, std::uint32_t snapCount, const JudgeState &initial,
                              std::string *error) {
    Dev &d = *dev_;
    auto put = [&](Buffer &b, const void *src, std::size_t bytes) {
        if (!b.Fit(std::max<std::size_t>(bytes, 16u), error)) return false;
        return bytes == 0u || Check(cudaMemcpy(b.p, src, bytes, cudaMemcpyHostToDevice), "upload", error);
    };
    if (!put(d.base, program.baselineInputs.data(), sizeof(CudaSearchInputEvent) * program.baselineInputs.size()) ||
        !put(d.modifiers, program.modifiers.data(), sizeof(CudaSearchModifierConfiguration) * program.modifiers.size()) ||
        !put(d.segments, program.segments.data(), sizeof(CudaSearchMutationSegment) * program.segments.size()) ||
        !put(d.smoothWeights, program.smoothWeights.data(), sizeof(double) * program.smoothWeights.size()) ||
        !put(d.actionBits, actionBits, sizeof(std::uint32_t) * ticks) || !put(d.snap, snap, sizeof(std::uint32_t) * snapCount))
        return false;
    GenParams &P = d.params;
    P.base = d.base.As<CudaSearchInputEvent>();
    P.modifiers = d.modifiers.As<CudaSearchModifierConfiguration>();
    P.segments = d.segments.As<CudaSearchMutationSegment>();
    P.smoothWeights = program.smoothWeights.empty() ? nullptr : d.smoothWeights.As<double>();
    P.actionBits = d.actionBits.As<std::uint32_t>();
    P.snap = d.snap.As<std::uint32_t>();
    P.baseCount = static_cast<std::uint32_t>(program.baselineInputs.size());
    P.modifierCount = static_cast<std::uint32_t>(program.modifiers.size());
    P.segmentCount = static_cast<std::uint32_t>(program.segments.size());
    P.tickDurationMs = program.tickDurationMs;
    P.capacity = program.eventCapacity;
    P.tailCount = static_cast<std::uint32_t>(program.immutableTail.size());
    P.ticks = ticks;
    P.snapCount = std::min(snapCount, ticks);
    P.branchTimeMs = branchTimeMs;
    P.mutableFromTimeMs = program.mutableFromTimeMs;
    P.boundary = program.mutableBoundaryControls;
    P.initial = initial;
    P.canonical = program.baselineInputsCanonical ? 1u : 0u;
    for (int k = 0; k < d.used; ++k) d.sets[k].slotsCap = 0u;   // capacity and ticks may differ: Reserve fits again
    return true;
}

bool DeviceGeneration::Reserve(std::size_t slots, std::string *error) {
    Dev &d = *dev_;
    slots = std::max<std::size_t>(slots, 1u);
    const std::size_t cap = std::max<std::uint32_t>(d.params.capacity, 1u);
    const std::size_t gen = std::min<std::size_t>(d.gen, slots);
    const std::size_t ticks = std::max<std::uint32_t>(d.params.ticks, 1u);
    for (int k = 0; k < d.used; ++k) {
        Dev::Set &s = d.sets[k];
        if (s.slotsCap >= slots) continue;
        std::size_t sortBytes = 0u, scanBytes = 0u;
        SlotOrder order{nullptr, nullptr, 0u, 1u};
        if (!Check(cub::DeviceMergeSort::SortKeys(nullptr, sortBytes, static_cast<std::uint32_t *>(nullptr),
                                                  static_cast<std::int64_t>(slots), order, nullptr),
                   "sort size", error) ||
            !Check(cub::DeviceScan::InclusiveSum(nullptr, scanBytes, static_cast<const std::uint32_t *>(nullptr),
                                                 static_cast<std::uint32_t *>(nullptr), static_cast<std::int64_t>(slots), nullptr),
                   "scan size", error))
            return false;
        const std::size_t scratch = UseLocalEvents(d.params.capacity) ? 1u : cap * gen;   // the event arrays in local memory: none here
        if (!s.mt.Fit(sizeof(std::uint32_t) * 624u * gen, error) || !s.events.Fit(sizeof(CudaSearchInputEvent) * scratch, error) ||
            !s.temporary.Fit(sizeof(CudaSearchInputEvent) * scratch, error) ||
            !s.passBaseline.Fit(sizeof(CudaSearchInputEvent) * scratch, error) ||
            !s.eligible.Fit(sizeof(std::uint32_t) * scratch, error) || !s.controls.Fit(sizeof(std::uint32_t) * ticks * slots, error) ||
            !s.first.Fit(sizeof(std::uint32_t) * slots, error) || !s.slots.Fit(sizeof(DeviceSlot) * slots, error) ||
            !s.counters.Fit(sizeof(std::uint32_t) * 4u, error) || !s.prefix.Fit(sizeof(JudgeState) * ticks, error) ||
            !s.order.Fit(sizeof(std::uint32_t) * slots, error) || !s.orderTmp.Fit(sizeof(std::uint32_t) * slots, error) ||
            !s.head.Fit(sizeof(std::uint32_t) * slots, error) ||
            !s.rank.Fit(sizeof(std::uint32_t) * slots, error) || !s.jobs.Fit(sizeof(SearchJob) * slots, error) ||
            !s.sortTemp.Fit(std::max<std::size_t>(sortBytes, 16u), error) || !s.scanTemp.Fit(std::max<std::size_t>(scanBytes, 16u), error) ||
            !s.sums.Fit(sizeof(std::uint32_t) * ((slots + 31u) / 32u), error) ||
            !s.hSlots.Fit(sizeof(DeviceSlot) * slots, error) || !s.hCounters.Fit(sizeof(std::uint32_t) * 4u, error) ||
            (d.pose && !s.hResults.Fit(sizeof(SearchResult) * slots, error)))
            return false;
        s.slotsCap = slots;
    }
    return true;
}

bool DeviceGeneration::Generate(int set, void *stream, std::uint64_t firstId, std::uint32_t n, bool baseline,
                                std::uint32_t changedSegmentCount, const JudgeState *prefix, std::uint32_t prefixCount,
                                std::uint32_t prefixStop, std::uint8_t stopStatus, std::string *error) {
    Dev &d = *dev_;
    if (!Reserve(n, error)) return false;
    Dev::Set &s = d.sets[set];
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    s.n = n;
    GenParams P = d.params;
    P.changedSegmentCount = changedSegmentCount;
    P.prefix = s.prefix.As<JudgeState>();
    P.prefixStop = prefixStop;
    P.stopStatus = stopStatus;
    P.n = n;
    P.gen = static_cast<std::uint32_t>(std::min<std::size_t>(d.gen, std::max<std::uint32_t>(n, 1u)));
    P.firstId = firstId;
    P.baseline = baseline ? 1u : 0u;
    P.mt = s.mt.As<std::uint32_t>();
    P.events = s.events.As<CudaSearchInputEvent>();
    P.temporary = s.temporary.As<CudaSearchInputEvent>();
    P.passBaseline = s.passBaseline.As<CudaSearchInputEvent>();
    P.eligible = s.eligible.As<std::uint32_t>();
    P.controls = s.controls.As<std::uint32_t>();
    P.first = s.first.As<std::uint32_t>();
    P.slots = s.slots.As<DeviceSlot>();
    P.counters = s.counters.As<std::uint32_t>();
    if (prefixCount > 0u && !Check(cudaMemcpyAsync(s.prefix.p, prefix, sizeof(JudgeState) * std::min(prefixCount, d.params.ticks),
                                                   cudaMemcpyHostToDevice, st),
                                   "prefix H2D", error))
        return false;
    if (!Check(cudaMemsetAsync(s.counters.p, 0, sizeof(std::uint32_t) * 4u, st), "counters", error)) return false;
    if (n == 0u) return true;
    cudaEventRecord(s.g0, st);
    static const std::uint32_t genBlock = [] {   // FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN_BLOCK (measurement; at most kGenBlock)
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN_BLOCK");
        const int b = v != nullptr ? std::atoi(v) : 0;
        return b >= 32 && b <= static_cast<int>(kGenBlock) ? static_cast<std::uint32_t>(b) : kGenBlock;
    }();
    const std::uint32_t genBlocks = (P.gen + genBlock - 1u) / genBlock;
    switch (LocalEvents(P.capacity)) {
    case kLocalEvents: GenerateKernel<kLocalEvents><<<genBlocks, genBlock, 0, st>>>(P); break;
    case kLocalEventsLarge: GenerateKernel<kLocalEventsLarge><<<genBlocks, genBlock, 0, st>>>(P); break;
    case kLocalEventsHuge: GenerateKernel<kLocalEventsHuge><<<genBlocks, genBlock, 0, st>>>(P); break;
    default: GenerateKernel<0u><<<genBlocks, genBlock, 0, st>>>(P); break;
    }
    IotaKernel<<<Blocks(n), kBlock, 0, st>>>(s.order.As<std::uint32_t>(), n);
    std::size_t sortBytes = s.sortTemp.bytes, scanBytes = s.scanTemp.bytes;
    // FOREVERVALIDATOR_TWIN_SEARCH_GPU_SORT=cub (measurement): cub's merge sort and scan (their kernels' shared memory
    // configurations differ from the lane runner's: beside its blocks each waits for an SM of its own)
    static const bool own = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_SORT");
        return v == nullptr || std::strcmp(v, "cub") != 0;
    }();
    static const std::uint32_t rows = [] {   // FOREVERVALIDATOR_TWIN_SEARCH_GPU_ORDER=first (measurement)
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_ORDER");
        return v != nullptr && std::strcmp(v, "first") == 0 ? 0u : 1u;
    }();
    const SlotOrder order{P.first, P.controls, P.ticks, rows};
    if (!Check(cudaGetLastError(), "generate launch", error)) return false;
    std::uint32_t *sorted = s.order.As<std::uint32_t>();
    if (own) {
        std::uint32_t *other = s.orderTmp.As<std::uint32_t>();
        for (std::uint32_t w = 1u; w < n; w <<= 1) {
            MergePassKernel<<<Blocks(n), kBlock, 0, st>>>(sorted, other, n, w, order);
            std::swap(sorted, other);
        }
    } else if (!Check(cub::DeviceMergeSort::SortKeys(s.sortTemp.p, sortBytes, sorted, static_cast<std::int64_t>(n), order, st), "sort",
                      error)) {
        return false;
    }
    HeadsKernel<<<Blocks(n), kBlock, 0, st>>>(sorted, P.first, P.controls, P.ticks, P.counters, n,
                                               s.head.As<std::uint32_t>());
    if (own) {
        ScanWarpsKernel<<<Blocks(n), kBlock, 0, st>>>(s.head.As<const std::uint32_t>(), s.rank.As<std::uint32_t>(),
                                                      s.sums.As<std::uint32_t>(), n);
        ScanSumsKernel<<<1, 32, 0, st>>>(s.sums.As<std::uint32_t>(), (n + 31u) / 32u);
        ScanAddKernel<<<Blocks(n), kBlock, 0, st>>>(s.rank.As<std::uint32_t>(), s.sums.As<const std::uint32_t>(), n);
    } else if (!Check(cub::DeviceScan::InclusiveSum(s.scanTemp.p, scanBytes, s.head.As<const std::uint32_t>(), s.rank.As<std::uint32_t>(),
                                                    static_cast<std::int64_t>(n), st),
                      "scan", error)) {
        return false;
    }
    JobsKernel<<<Blocks(n), kBlock, 0, st>>>(P, sorted, s.head.As<std::uint32_t>(), s.rank.As<std::uint32_t>(),
                                             s.jobs.As<SearchJob>());
    cudaEventRecord(s.g1, st);
    return Check(cudaGetLastError(), "jobs launch", error);
}

const SearchJob *DeviceGeneration::Jobs(int set) const { return dev_->sets[set].jobs.As<SearchJob>(); }
const std::uint32_t *DeviceGeneration::Controls(int set) const { return dev_->sets[set].controls.As<std::uint32_t>(); }
const std::uint32_t *DeviceGeneration::JobCount(int set) const { return dev_->sets[set].counters.As<std::uint32_t>() + 1; }

bool DeviceGeneration::Finish(int set, void *stream, const SearchResult *results, bool pose, std::string *error) {
    Dev::Set &s = dev_->sets[set];
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    s.pose = pose;
    if (s.n == 0u) return true;
    ResultsKernel<<<Blocks(s.n), kBlock, 0, st>>>(s.slots.As<DeviceSlot>(), s.n, results, pose);
    if (!Check(cudaGetLastError(), "results launch", error) ||
        !Check(cudaMemcpyAsync(s.hSlots.p, s.slots.p, sizeof(DeviceSlot) * s.n, cudaMemcpyDeviceToHost, st), "slots D2H", error) ||
        !Check(cudaMemcpyAsync(s.hCounters.p, s.counters.p, sizeof(std::uint32_t) * 4u, cudaMemcpyDeviceToHost, st), "counters D2H", error))
        return false;
    if (pose) {
        if (s.hResults.bytes < sizeof(SearchResult) * s.n) return *error = "twin search device: results not reserved", false;
        if (!Check(cudaMemcpyAsync(s.hResults.p, results, sizeof(SearchResult) * s.n, cudaMemcpyDeviceToHost, st), "results D2H", error))
            return false;
    }
    cudaEventRecord(s.g2, st);
    return true;
}

const DeviceSlot *DeviceGeneration::Slots(int set) const { return dev_->sets[set].hSlots.As<DeviceSlot>(); }
const SearchResult *DeviceGeneration::Results(int set) const { return dev_->sets[set].hResults.As<SearchResult>(); }
std::uint32_t DeviceGeneration::PhysicsSlots(int set) const { return dev_->sets[set].hCounters.As<std::uint32_t>()[0]; }
float DeviceGeneration::GenerationMs(int set) const {
    const Dev::Set &s = dev_->sets[set];
    float ms = 0.0f;
    if (s.n == 0u || cudaEventElapsedTime(&ms, s.g0, s.g1) != cudaSuccess) return 0.0f;
    return ms;
}
float DeviceGeneration::OutcomesMs(int set) const {
    const Dev::Set &s = dev_->sets[set];
    float ms = 0.0f;
    if (s.n == 0u || cudaEventElapsedTime(&ms, s.g1, s.g2) != cudaSuccess) return 0.0f;
    return ms;
}
std::size_t DeviceGeneration::SetBytes() const {
    const Dev::Set &s = dev_->sets[0];
    return s.mt.bytes + s.events.bytes + s.temporary.bytes + s.passBaseline.bytes + s.eligible.bytes + s.controls.bytes + s.first.bytes +
           s.slots.bytes + s.counters.bytes + s.prefix.bytes + s.order.bytes + s.orderTmp.bytes + s.head.bytes + s.rank.bytes + s.jobs.bytes +
           s.sortTemp.bytes + s.scanTemp.bytes + s.sums.bytes;
}
std::uint32_t DeviceGeneration::GeneratorThreads() const { return dev_->gen; }
std::uint32_t DeviceGeneration::JobsRun(int set) const { return dev_->sets[set].hCounters.As<std::uint32_t>()[1]; }

}  // namespace forevervalidator::simulation::twin_search
