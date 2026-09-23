// Generated from src/simulation/backends/cuda/cuda_candidate_events.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_CANDIDATE_EVENTS_CUH
#define FOREVERVALIDATOR_HIP_CANDIDATE_EVENTS_CUH

#include <cstddef>
#include <cstdint>

#include "simulation/backends/hip/generated/hip_search_executor.h"

#if defined(__HIPCC__)
#define FOREVERVALIDATOR_CANDIDATE_HD __host__ __device__
#else
#define FOREVERVALIDATOR_CANDIDATE_HD
#endif

namespace forevervalidator::simulation::hip::candidate_events {

using Event = HipSearchInputEvent;

struct CanonicalSuffix {
    const Event *events = nullptr;
    std::uint32_t count = 0u;
    std::int64_t mutableFromTimeMs = 0;
};

FOREVERVALIDATOR_CANDIDATE_HD inline CanonicalSuffix SuffixFrom(
        const Event *baseline,
        std::uint32_t count,
        std::int64_t mutableFromTimeMs) {
    std::uint32_t first = 0u;
    std::uint32_t remaining = count;
    while (remaining != 0u) {
        const std::uint32_t step = remaining / 2u;
        const std::uint32_t middle = first + step;
        if (baseline[middle].timeMs < mutableFromTimeMs) {
            first = middle + 1u;
            remaining -= step + 1u;
        } else {
            remaining = step;
        }
    }
    return {
            baseline + first,
            count - first,
            mutableFromTimeMs};
}

struct Edit {
    std::uint32_t outputIndex = 0u;
    Event event{};
};

// Edit ordinal is the major dimension. Adjacent candidate lanes therefore
// read adjacent words while consuming the same part of their sparse stream.
// Output edits carry final events. Suppressed baseline sources are separate
// because they need only one word.
struct CoalescedEditStorage {
    std::uint32_t *counts = nullptr;
    std::uint32_t *erasedCounts = nullptr;
    std::uint32_t *outputIndices = nullptr;
    std::int32_t *times = nullptr;
    std::uint32_t *actions = nullptr;
    std::uint32_t *valueKinds = nullptr;
    std::int32_t *values = nullptr;
    std::uint32_t *erasedSourceIndices = nullptr;
    std::uint32_t *packedOutputActions = nullptr;
    std::uint8_t *packedValueKinds = nullptr;
    std::uint16_t *packedErasedSourceIndices = nullptr;
    std::uint32_t candidateStride = 0u;
    std::uint32_t editCapacity = 0u;
    std::uint32_t eraseCapacity = 0u;
};

FOREVERVALIDATOR_CANDIDATE_HD inline std::uint64_t EditOffset(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal) {
    return static_cast<std::uint64_t>(ordinal) *
                    storage.candidateStride +
            candidate;
}

FOREVERVALIDATOR_CANDIDATE_HD inline Edit LoadEdit(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal) {
    const std::uint64_t offset = EditOffset(storage, candidate, ordinal);
    const bool packed = storage.packedOutputActions != nullptr;
    const std::uint32_t outputAction = packed
            ? storage.packedOutputActions[offset]
            : 0u;
    return {
            packed ? outputAction & UINT16_MAX
                   : storage.outputIndices[offset],
            {storage.times[offset],
             packed ? outputAction >> 16u
                    : storage.actions[offset],
             packed ? storage.packedValueKinds[offset]
                    : storage.valueKinds[offset],
             storage.values[offset]}};
}

FOREVERVALIDATOR_CANDIDATE_HD inline void StoreEdit(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal,
        const Edit &edit) {
    const std::uint64_t offset =
            EditOffset(storage, candidate, ordinal);
    if (storage.packedOutputActions != nullptr) {
        storage.packedOutputActions[offset] =
                edit.outputIndex |
                (edit.event.action << 16u);
        storage.packedValueKinds[offset] =
                static_cast<std::uint8_t>(
                        edit.event.valueKind);
    } else {
        storage.outputIndices[offset] = edit.outputIndex;
        storage.actions[offset] = edit.event.action;
        storage.valueKinds[offset] = edit.event.valueKind;
    }
    storage.times[offset] = edit.event.timeMs;
    storage.values[offset] = edit.event.value;
}

FOREVERVALIDATOR_CANDIDATE_HD inline void SortOutputEdits(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate) {
    const std::uint32_t count = storage.counts[candidate];
    for (std::uint32_t index = 1u; index < count; ++index) {
        const Edit value = LoadEdit(storage, candidate, index);
        std::uint32_t insertion = index;
        while (insertion != 0u &&
               LoadEdit(
                       storage, candidate,
                       insertion - 1u).outputIndex >
                       value.outputIndex) {
            StoreEdit(
                    storage, candidate, insertion,
                    LoadEdit(
                            storage, candidate,
                            insertion - 1u));
            --insertion;
        }
        StoreEdit(storage, candidate, insertion, value);
    }
}

FOREVERVALIDATOR_CANDIDATE_HD inline std::uint32_t LoadErasedSource(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal) {
    const std::uint64_t offset =
            EditOffset(storage, candidate, ordinal);
    return storage.packedErasedSourceIndices != nullptr
            ? storage.packedErasedSourceIndices[offset]
            : storage.erasedSourceIndices[offset];
}

FOREVERVALIDATOR_CANDIDATE_HD inline void StoreErasedSource(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal,
        std::uint32_t sourceIndex) {
    const std::uint64_t offset =
            EditOffset(storage, candidate, ordinal);
    if (storage.packedErasedSourceIndices != nullptr) {
        storage.packedErasedSourceIndices[offset] =
                static_cast<std::uint16_t>(sourceIndex);
    } else {
        storage.erasedSourceIndices[offset] = sourceIndex;
    }
}

FOREVERVALIDATOR_CANDIDATE_HD inline void SortErasedSources(
        const CoalescedEditStorage &storage,
        std::uint32_t candidate) {
    const std::uint32_t count = storage.erasedCounts[candidate];
    for (std::uint32_t index = 1u; index < count; ++index) {
        const std::uint32_t value =
                LoadErasedSource(storage, candidate, index);
        std::uint32_t insertion = index;
        while (insertion != 0u &&
               LoadErasedSource(
                       storage, candidate, insertion - 1u) > value) {
            StoreErasedSource(
                    storage, candidate, insertion,
                    LoadErasedSource(
                            storage, candidate, insertion - 1u));
            --insertion;
        }
        StoreErasedSource(storage, candidate, insertion, value);
    }
}

class EditWriter {
public:
    FOREVERVALIDATOR_CANDIDATE_HD EditWriter(
            CoalescedEditStorage storage,
            std::uint32_t candidate)
        : storage_(storage), candidate_(candidate) {}

    FOREVERVALIDATOR_CANDIDATE_HD bool Output(const Edit &edit) {
        if (storage_.counts == nullptr ||
            storage_.counts[candidate_] >= storage_.editCapacity) {
            return false;
        }
        const std::uint32_t ordinal = storage_.counts[candidate_]++;
        StoreEdit(storage_, candidate_, ordinal, edit);
        return true;
    }

    FOREVERVALIDATOR_CANDIDATE_HD bool Replace(
            std::uint32_t outputIndex,
            const Event &event) {
        return Output({outputIndex, event});
    }

    FOREVERVALIDATOR_CANDIDATE_HD bool Insert(
            std::uint32_t outputIndex,
            const Event &event) {
        return Output({outputIndex, event});
    }

    FOREVERVALIDATOR_CANDIDATE_HD bool Erase(
            std::uint32_t sourceIndex) {
        if (storage_.erasedCounts == nullptr ||
            storage_.erasedCounts[candidate_] >=
                    storage_.eraseCapacity) {
            return false;
        }
        const std::uint32_t ordinal =
                storage_.erasedCounts[candidate_]++;
        StoreErasedSource(
                storage_, candidate_, ordinal, sourceIndex);
        return true;
    }

private:
    CoalescedEditStorage storage_{};
    std::uint32_t candidate_ = 0u;
};

struct CandidateView {
    CanonicalSuffix baseline{};
    CoalescedEditStorage edits{};
    std::uint32_t candidate = 0u;
    std::uint32_t finalCount = 0u;

    FOREVERVALIDATOR_CANDIDATE_HD std::uint32_t EditCount() const {
        return edits.counts == nullptr ? 0u : edits.counts[candidate];
    }

    FOREVERVALIDATOR_CANDIDATE_HD std::uint32_t ErasedCount() const {
        return edits.erasedCounts == nullptr
                ? 0u : edits.erasedCounts[candidate];
    }

};

class CandidateCursor {
public:
    FOREVERVALIDATOR_CANDIDATE_HD explicit CandidateCursor(
            CandidateView view)
        : view_(view) {}

    FOREVERVALIDATOR_CANDIDATE_HD bool Next(Event *event) {
        if (event == nullptr || outputIndex_ >= view_.finalCount) {
            return false;
        }
        if (editOrdinal_ < view_.EditCount()) {
            const Edit edit = LoadEdit(
                    view_.edits, view_.candidate, editOrdinal_);
            if (edit.outputIndex == outputIndex_) {
                *event = edit.event;
                ++editOrdinal_;
                ++outputIndex_;
                return true;
            }
        }
        while (baselineIndex_ < view_.baseline.count) {
            while (erasedOrdinal_ < view_.ErasedCount() &&
                   LoadErasedSource(
                           view_.edits, view_.candidate,
                           erasedOrdinal_) < baselineIndex_) {
                ++erasedOrdinal_;
            }
            if (erasedOrdinal_ >= view_.ErasedCount() ||
                LoadErasedSource(
                        view_.edits, view_.candidate,
                        erasedOrdinal_) != baselineIndex_) {
                break;
            }
            ++baselineIndex_;
            ++erasedOrdinal_;
        }
        if (baselineIndex_ >= view_.baseline.count) {
            return false;
        }
        *event = view_.baseline.events[baselineIndex_++];
        ++outputIndex_;
        return true;
    }

private:
    CandidateView view_{};
    std::uint32_t outputIndex_ = 0u;
    std::uint32_t baselineIndex_ = 0u;
    std::uint32_t editOrdinal_ = 0u;
    std::uint32_t erasedOrdinal_ = 0u;
};

FOREVERVALIDATOR_CANDIDATE_HD inline std::int32_t SaturateValue(
        const Event &event) {
    if (event.valueKind == 2u) {
        return event.value < -65536
                ? -65536
                : event.value > 65536 ? 65536 : event.value;
    }
    if (event.valueKind == 1u) {
        return event.value == 0 ? 0 : 1;
    }
    return event.value;
}

FOREVERVALIDATOR_CANDIDATE_HD inline std::uint32_t HashAction(
        std::uint32_t action) {
    action ^= action >> 16u;
    action *= 0x7feb352du;
    action ^= action >> 15u;
    action *= 0x846ca68bu;
    return action ^ (action >> 16u);
}

// Stable mergesort by time followed by an expected-linear hash deduplication
// per equal-time group. The first occurrence fixes stable order and the last
// occurrence supplies the value.
FOREVERVALIDATOR_CANDIDATE_HD inline std::uint32_t NormalizeSuffix(
        Event *events,
        std::uint32_t count,
        Event *scratch) {
    for (std::uint32_t index = 0u; index < count; ++index) {
        if (events[index].timeMs < 0) {
            events[index].timeMs = 0;
        }
        events[index].value = SaturateValue(events[index]);
    }

    Event *source = events;
    Event *destination = scratch;
    for (std::uint32_t width = 1u; width < count;) {
        for (std::uint32_t begin = 0u; begin < count;
             begin += width * 2u) {
            const std::uint32_t middle =
                    begin + width < count ? begin + width : count;
            const std::uint32_t end =
                    middle + width < count ? middle + width : count;
            std::uint32_t left = begin;
            std::uint32_t right = middle;
            std::uint32_t output = begin;
            while (left < middle || right < end) {
                if (right >= end ||
                    (left < middle &&
                     source[left].timeMs <= source[right].timeMs)) {
                    destination[output++] = source[left++];
                } else {
                    destination[output++] = source[right++];
                }
            }
        }
        Event *swap = source;
        source = destination;
        destination = swap;
        if (width > count / 2u) {
            break;
        }
        width *= 2u;
    }

    std::uint32_t outputCount = 0u;
    for (std::uint32_t groupBegin = 0u; groupBegin < count;) {
        std::uint32_t groupEnd = groupBegin + 1u;
        while (groupEnd < count &&
               source[groupEnd].timeMs == source[groupBegin].timeMs) {
            ++groupEnd;
        }
        const std::uint32_t groupCount = groupEnd - groupBegin;
        for (std::uint32_t index = 0u; index < groupCount; ++index) {
            destination[outputCount + index] =
                    source[groupBegin + index];
            source[groupBegin + index].timeMs = 0;
        }

        for (std::uint32_t index = 0u; index < groupCount; ++index) {
            const Event value = destination[outputCount + index];
            std::uint32_t bucket =
                    HashAction(value.action) % groupCount;
            while (source[groupBegin + bucket].timeMs != 0 &&
                   source[groupBegin + bucket].action != value.action) {
                bucket = bucket + 1u == groupCount
                        ? 0u : bucket + 1u;
            }
            Event &entry = source[groupBegin + bucket];
            if (entry.timeMs == 0) {
                entry.timeMs = 1;
                entry.action = value.action;
                entry.valueKind = index;
            } else {
                destination[
                        outputCount + entry.valueKind] = value;
            }
        }

        std::uint32_t uniqueCount = 0u;
        for (std::uint32_t index = 0u; index < groupCount; ++index) {
            const Event value = destination[outputCount + index];
            std::uint32_t bucket =
                    HashAction(value.action) % groupCount;
            while (source[groupBegin + bucket].action != value.action) {
                bucket = bucket + 1u == groupCount
                        ? 0u : bucket + 1u;
            }
            if (source[groupBegin + bucket].valueKind == index) {
                destination[outputCount + uniqueCount++] = value;
            }
        }
        outputCount += uniqueCount;
        groupBegin = groupEnd;
    }

    if (destination != events) {
        for (std::uint32_t index = 0u; index < outputCount; ++index) {
            events[index] = destination[index];
        }
    }
    return outputCount;
}

FOREVERVALIDATOR_CANDIDATE_HD inline std::uint32_t NormalizeWithPrefix(
        Event *events,
        std::uint32_t count,
        Event *scratch,
        const Event *baseline,
        std::uint32_t baselineCount,
        std::int64_t mutableFromTimeMs,
        std::uint32_t capacity) {
    std::uint32_t suffixCount = 0u;
    for (std::uint32_t index = 0u; index < count; ++index) {
        if (events[index].timeMs >= mutableFromTimeMs) {
            events[suffixCount++] = events[index];
        }
    }
    suffixCount = NormalizeSuffix(events, suffixCount, scratch);

    std::uint32_t prefixCount = 0u;
    for (std::uint32_t index = 0u;
         index < baselineCount; ++index) {
        if (baseline[index].timeMs < mutableFromTimeMs) {
            ++prefixCount;
        }
    }
    if (prefixCount + suffixCount > capacity) {
        return UINT32_MAX;
    }
    for (std::uint32_t index = suffixCount; index != 0u; --index) {
        events[prefixCount + index - 1u] = events[index - 1u];
    }
    std::uint32_t prefixOutput = 0u;
    for (std::uint32_t index = 0u;
         index < baselineCount; ++index) {
        if (baseline[index].timeMs < mutableFromTimeMs) {
            events[prefixOutput++] = baseline[index];
        }
    }
    return prefixCount + suffixCount;
}

FOREVERVALIDATOR_CANDIDATE_HD inline bool Materialize(
        const CandidateView &candidate,
        Event *events,
        std::uint32_t capacity) {
    if (candidate.finalCount > capacity) {
        return false;
    }
    CandidateCursor cursor(candidate);
    for (std::uint32_t index = 0u;
         index < candidate.finalCount; ++index) {
        if (!cursor.Next(events + index)) {
            return false;
        }
    }
    return true;
}

}  // namespace forevervalidator::simulation::hip::candidate_events

#undef FOREVERVALIDATOR_CANDIDATE_HD

#endif
