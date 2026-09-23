// Generated from src/simulation/backends/cuda/cuda_modifier_event_ops.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_MODIFIER_EVENT_OPS_CUH
#define FOREVERVALIDATOR_HIP_MODIFIER_EVENT_OPS_CUH

#include <cstdint>

#include "simulation/backends/hip/generated/hip_search_executor.h"

namespace forevervalidator::simulation::hip_search_modifier_detail {

#if defined(__HIPCC__)
#define FOREVERVALIDATOR_HIP_HD __host__ __device__
#else
#define FOREVERVALIDATOR_HIP_HD
#endif

FOREVERVALIDATOR_HIP_HD inline std::uint32_t LowerBoundTime(
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::int64_t timeMs) {
    std::uint32_t first = 0u;
    while (first < count) {
        const std::uint32_t middle = first + (count - first) / 2u;
        if (events[middle].timeMs < timeMs) {
            first = middle + 1u;
        } else {
            count = middle;
        }
    }
    return first;
}

FOREVERVALIDATOR_HIP_HD inline std::uint32_t UpperBoundTime(
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::int64_t timeMs) {
    std::uint32_t first = 0u;
    while (first < count) {
        const std::uint32_t middle = first + (count - first) / 2u;
        if (events[middle].timeMs <= timeMs) {
            first = middle + 1u;
        } else {
            count = middle;
        }
    }
    return first;
}

FOREVERVALIDATOR_HIP_HD inline std::int32_t ChannelStateAt(
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::uint32_t action,
        std::uint32_t valueKind,
        std::int64_t timeMs,
        bool sortedByTime,
        std::int32_t initialState = 0) {
    if (sortedByTime) {
        std::uint32_t index = UpperBoundTime(events, count, timeMs);
        while (index != 0u) {
            const HipSearchInputEvent &event = events[--index];
            if (event.action == action &&
                event.valueKind == valueKind) {
                return event.value;
            }
        }
        return initialState;
    }

    std::int32_t state = initialState;
    std::int64_t bestTime = INT64_MIN;
    for (std::uint32_t index = 0u; index < count; ++index) {
        const HipSearchInputEvent &event = events[index];
        if (event.action != action ||
            event.valueKind != valueKind ||
            event.timeMs > timeMs ||
            event.timeMs < bestTime) {
            continue;
        }
        state = event.value;
        bestTime = event.timeMs;
    }
    return state;
}

// The smooth modifier appends one ascending run of generated events to an
// already canonical stream. Querying both sorted runs keeps later generated
// samples observable without falling back to a full scan for every tick.
FOREVERVALIDATOR_HIP_HD inline std::int32_t
ChannelStateAtWithAppendedRun(
        const HipSearchInputEvent *events,
        std::uint32_t canonicalCount,
        std::uint32_t count,
        std::uint32_t action,
        std::uint32_t valueKind,
        std::int64_t timeMs,
        std::int32_t initialState = 0) {
    std::int32_t state = initialState;
    std::int64_t bestTime = INT64_MIN;
    std::uint32_t index =
            UpperBoundTime(events, canonicalCount, timeMs);
    while (index != 0u) {
        const HipSearchInputEvent &event = events[--index];
        if (event.action == action &&
            event.valueKind == valueKind) {
            state = event.value;
            bestTime = event.timeMs;
            break;
        }
    }

    std::uint32_t first = canonicalCount;
    std::uint32_t end = count;
    while (first < end) {
        const std::uint32_t middle =
                first + (end - first) / 2u;
        if (events[middle].timeMs <= timeMs) {
            first = middle + 1u;
        } else {
            end = middle;
        }
    }
    index = first;
    while (index > canonicalCount) {
        const HipSearchInputEvent &event = events[--index];
        if (event.action == action &&
            event.valueKind == valueKind) {
            if (event.timeMs >= bestTime) {
                state = event.value;
            }
            break;
        }
    }
    return state;
}

// Returns the channel state at start while compacting matching events in the
// inclusive replacement range. This combines the two full scans previously
// needed by held-input insertion.
FOREVERVALIDATOR_HIP_HD inline std::int32_t
RemoveActionRangeAndReadState(
        HipSearchInputEvent *events,
        std::uint32_t *count,
        std::uint32_t action,
        std::uint32_t valueKind,
        std::int64_t start,
        std::int64_t end,
        std::int32_t initialState = 0) {
    std::int32_t state = initialState;
    std::int64_t bestTime = INT64_MIN;
    std::uint32_t destination = 0u;
    for (std::uint32_t index = 0u; index < *count; ++index) {
        const HipSearchInputEvent event = events[index];
        if (event.action == action &&
            event.valueKind == valueKind &&
            event.timeMs <= start &&
            event.timeMs >= bestTime) {
            state = event.value;
            bestTime = event.timeMs;
        }
        if (event.action == action &&
            event.timeMs >= start && event.timeMs <= end) {
            continue;
        }
        events[destination++] = event;
    }
    *count = destination;
    return state;
}

FOREVERVALIDATOR_HIP_HD inline bool ActionInGroup(
        std::uint32_t action,
        std::uint32_t group) {
    return group == 0u ? action == 4u
            : group == 1u ? action == 1u || action == 2u
                          : action == 3u;
}

FOREVERVALIDATOR_HIP_HD inline std::uint32_t
CollectExistingEventEligible(
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::uint32_t *eligible,
        std::int64_t minimumTimeMs,
        std::int64_t maximumTimeMs,
        std::uint32_t optionFlags,
        bool sortedByTime) {
    const std::uint32_t begin = sortedByTime
            ? LowerBoundTime(events, count, minimumTimeMs)
            : 0u;
    const std::uint32_t end = sortedByTime
            ? UpperBoundTime(events, count, maximumTimeMs)
            : count;
    std::uint32_t result = 0u;
    for (std::uint32_t index = begin; index < end; ++index) {
        const HipSearchInputEvent &event = events[index];
        if ((!sortedByTime &&
             (event.timeMs < minimumTimeMs ||
              event.timeMs > maximumTimeMs)) ||
            !((event.action == 4u && event.valueKind == 2u) ||
              ((optionFlags & 2u) != 0u &&
               (event.action == 1u || event.action == 2u)) ||
              ((optionFlags & 4u) != 0u &&
               event.action == 3u))) {
            continue;
        }
        eligible[result++] = index;
    }
    return result;
}

FOREVERVALIDATOR_HIP_HD inline std::uint32_t
CollectDeletionEligible(
        const HipSearchInputEvent *events,
        std::uint32_t count,
        std::uint32_t *eligible,
        std::int64_t minimumTimeMs,
        std::int64_t maximumTimeMs,
        std::uint32_t group,
        bool sortedByTime) {
    const std::uint32_t begin = sortedByTime
            ? LowerBoundTime(events, count, minimumTimeMs)
            : 0u;
    const std::uint32_t end = sortedByTime
            ? UpperBoundTime(events, count, maximumTimeMs)
            : count;
    std::uint32_t result = 0u;
    for (std::uint32_t index = begin; index < end; ++index) {
        const HipSearchInputEvent &event = events[index];
        if ((!sortedByTime &&
             (event.timeMs < minimumTimeMs ||
              event.timeMs > maximumTimeMs)) ||
            !ActionInGroup(event.action, group)) {
            continue;
        }
        eligible[result++] = index;
    }
    return result;
}

// Removes a rank while retaining the same ordered eligibility list that a
// fresh full scan would produce. Removed event indices accumulate at the tail.
FOREVERVALIDATOR_HIP_HD inline void SelectDeletionRank(
        std::uint32_t *eligible,
        std::uint32_t *remaining,
        std::uint32_t rank) {
    const std::uint32_t selected = eligible[rank];
    for (std::uint32_t index = rank + 1u;
         index < *remaining; ++index) {
        eligible[index - 1u] = eligible[index];
    }
    --*remaining;
    eligible[*remaining] = selected;
}

FOREVERVALIDATOR_HIP_HD inline void CompactSelectedDeletionTail(
        HipSearchInputEvent *events,
        std::uint32_t *eventCount,
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

    std::uint32_t selected = remaining;
    std::uint32_t destination = 0u;
    for (std::uint32_t index = 0u; index < *eventCount; ++index) {
        if (selected < initialEligibleCount &&
            eligible[selected] == index) {
            ++selected;
            continue;
        }
        events[destination++] = events[index];
    }
    *eventCount = destination;
}

#undef FOREVERVALIDATOR_HIP_HD

}  // namespace forevervalidator::simulation::hip_search_modifier_detail

#endif
