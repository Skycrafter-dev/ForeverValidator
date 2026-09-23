// Generated from tests/cuda_modifier_event_ops_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_modifier_event_ops.cuh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using forevervalidator::simulation::HipSearchInputEvent;
namespace modifier_ops =
        forevervalidator::simulation::hip_search_modifier_detail;

bool SameEvent(const HipSearchInputEvent &left,
               const HipSearchInputEvent &right) {
    return left.timeMs == right.timeMs &&
            left.action == right.action &&
            left.valueKind == right.valueKind &&
            left.value == right.value;
}

bool SameEvents(const std::vector<HipSearchInputEvent> &left,
                const std::vector<HipSearchInputEvent> &right) {
    return left.size() == right.size() &&
            std::equal(
                    left.begin(), left.end(), right.begin(),
                    SameEvent);
}

std::int32_t LegacyStateAt(
        const std::vector<HipSearchInputEvent> &events,
        std::uint32_t action,
        std::uint32_t valueKind,
        std::int64_t timeMs) {
    std::int32_t state = 0;
    std::int64_t bestTime = INT64_MIN;
    for (const HipSearchInputEvent &event : events) {
        if (event.action == action &&
            event.valueKind == valueKind &&
            event.timeMs <= timeMs &&
            event.timeMs >= bestTime) {
            state = event.value;
            bestTime = event.timeMs;
        }
    }
    return state;
}

std::int32_t LegacyRemoveAndRead(
        std::vector<HipSearchInputEvent> *events,
        std::uint32_t action,
        std::uint32_t valueKind,
        std::int64_t start,
        std::int64_t end) {
    const std::int32_t state =
            LegacyStateAt(*events, action, valueKind, start);
    events->erase(
            std::remove_if(
                    events->begin(), events->end(),
                    [&](const HipSearchInputEvent &event) {
                        return event.action == action &&
                                event.timeMs >= start &&
                                event.timeMs <= end;
                    }),
            events->end());
    return state;
}

bool StateAndInsertionRangeParity() {
    std::vector<HipSearchInputEvent> sorted{
            {0, 4u, 2u, -120},
            {10, 1u, 1u, 1},
            {10, 4u, 2u, 100},
            {20, 4u, 2u, 200},
            {30, 3u, 1u, 1},
            {30, 4u, 2u, 300},
            {40, 4u, 2u, 400},
            {50, 1u, 1u, 0},
    };
    for (std::int64_t time = -5; time <= 60; ++time) {
        if (LegacyStateAt(sorted, 4u, 2u, time) !=
            modifier_ops::ChannelStateAt(
                    sorted.data(),
                    static_cast<std::uint32_t>(sorted.size()),
                    4u, 2u, time, true)) {
            return false;
        }
    }
    std::vector<HipSearchInputEvent> appended = sorted;
    const std::uint32_t canonicalCount =
            static_cast<std::uint32_t>(appended.size());
    appended.push_back({10, 4u, 2u, 1010});
    appended.push_back({20, 4u, 2u, 2020});
    appended.push_back({30, 4u, 2u, 3030});
    for (std::int64_t time = -5; time <= 60; ++time) {
        if (LegacyStateAt(appended, 4u, 2u, time) !=
            modifier_ops::ChannelStateAtWithAppendedRun(
                    appended.data(), canonicalCount,
                    static_cast<std::uint32_t>(
                            appended.size()),
                    4u, 2u, time)) {
            return false;
        }
    }

    std::vector<HipSearchInputEvent> legacy = sorted;
    std::vector<HipSearchInputEvent> optimized = sorted;
    const std::int32_t legacyState =
            LegacyRemoveAndRead(&legacy, 4u, 2u, 10, 30);
    std::uint32_t optimizedCount =
            static_cast<std::uint32_t>(optimized.size());
    const std::int32_t optimizedState =
            modifier_ops::RemoveActionRangeAndReadState(
                    optimized.data(), &optimizedCount,
                    4u, 2u, 10, 30);
    optimized.resize(optimizedCount);
    if (legacyState != optimizedState ||
        !SameEvents(legacy, optimized)) {
        return false;
    }

    std::reverse(legacy.begin(), legacy.end());
    optimized = legacy;
    const std::int32_t unsortedLegacyState =
            LegacyRemoveAndRead(&legacy, 1u, 1u, 5, 45);
    optimizedCount =
            static_cast<std::uint32_t>(optimized.size());
    const std::int32_t unsortedOptimizedState =
            modifier_ops::RemoveActionRangeAndReadState(
                    optimized.data(), &optimizedCount,
                    1u, 1u, 5, 45);
    optimized.resize(optimizedCount);
    return unsortedLegacyState == unsortedOptimizedState &&
            SameEvents(legacy, optimized);
}

bool WindowAndExistingEligibilityParity() {
    std::vector<HipSearchInputEvent> events;
    for (std::int32_t index = 0; index < 200; ++index) {
        const std::uint32_t action =
                index % 6 == 0 ? 4u
                : index % 6 == 1 ? 1u
                : index % 6 == 2 ? 2u
                : index % 6 == 3 ? 3u
                                 : 5u;
        events.push_back(
                {index * 10, action,
                 index % 13 == 0 ? 0u
                 : action == 4u ? 2u : 1u,
                 index});
    }
    std::vector<std::uint32_t> legacy(events.size());
    std::vector<std::uint32_t> optimized(events.size());
    for (const auto &window :
         {std::pair<std::int64_t, std::int64_t>{0, 1990},
          {0, 0}, {105, 495}, {1990, 2500}}) {
        const std::uint32_t begin =
                modifier_ops::LowerBoundTime(
                        events.data(),
                        static_cast<std::uint32_t>(events.size()),
                        window.first);
        const std::uint32_t end =
                modifier_ops::UpperBoundTime(
                        events.data(),
                        static_cast<std::uint32_t>(events.size()),
                        window.second);
        std::uint32_t linearWindowCount = 0u;
        std::uint32_t legacyCount = 0u;
        for (std::uint32_t index = 0u;
             index < events.size(); ++index) {
            const HipSearchInputEvent &event = events[index];
            if (event.timeMs < window.first ||
                event.timeMs > window.second) {
                continue;
            }
            ++linearWindowCount;
            if ((event.action == 4u &&
                 event.valueKind == 2u) ||
                event.action == 1u || event.action == 2u ||
                event.action == 3u) {
                legacy[legacyCount++] = index;
            }
        }
        const std::uint32_t optimizedCount =
                modifier_ops::CollectExistingEventEligible(
                        events.data(),
                        static_cast<std::uint32_t>(events.size()),
                        optimized.data(), window.first,
                        window.second, 2u | 4u, true);
        if (end - begin != linearWindowCount ||
            optimizedCount != legacyCount ||
            !std::equal(
                    legacy.begin(), legacy.begin() + legacyCount,
                    optimized.begin())) {
            return false;
        }
    }
    return true;
}

std::vector<std::uint32_t> LegacyEligible(
        const std::vector<HipSearchInputEvent> &events,
        std::int64_t minimumTimeMs,
        std::int64_t maximumTimeMs,
        std::uint32_t group) {
    std::vector<std::uint32_t> result;
    for (std::uint32_t index = 0u;
         index < events.size(); ++index) {
        const HipSearchInputEvent &event = events[index];
        if (event.timeMs >= minimumTimeMs &&
            event.timeMs <= maximumTimeMs &&
            modifier_ops::ActionInGroup(event.action, group)) {
            result.push_back(index);
        }
    }
    return result;
}

void LegacyDelete(
        std::vector<HipSearchInputEvent> *events,
        std::int64_t minimumTimeMs,
        std::int64_t maximumTimeMs,
        std::uint32_t group,
        const std::vector<std::uint32_t> &draws) {
    for (std::uint32_t draw : draws) {
        const std::vector<std::uint32_t> eligible =
                LegacyEligible(
                        *events, minimumTimeMs,
                        maximumTimeMs, group);
        if (eligible.empty()) {
            return;
        }
        events->erase(
                events->begin() +
                eligible[draw % eligible.size()]);
    }
}

void OptimizedDelete(
        std::vector<HipSearchInputEvent> *events,
        std::int64_t minimumTimeMs,
        std::int64_t maximumTimeMs,
        std::uint32_t group,
        const std::vector<std::uint32_t> &draws) {
    std::vector<std::uint32_t> scratch(events->size());
    std::uint32_t eventCount =
            static_cast<std::uint32_t>(events->size());
    const std::uint32_t initialEligibleCount =
            modifier_ops::CollectDeletionEligible(
                    events->data(), eventCount, scratch.data(),
                    minimumTimeMs, maximumTimeMs, group, true);
    std::uint32_t remaining = initialEligibleCount;
    for (std::uint32_t draw : draws) {
        if (remaining == 0u) {
            break;
        }
        modifier_ops::SelectDeletionRank(
                scratch.data(), &remaining, draw % remaining);
    }
    modifier_ops::CompactSelectedDeletionTail(
            events->data(), &eventCount, scratch.data(),
            remaining, initialEligibleCount);
    events->resize(eventCount);
}

bool DeletionParity() {
    std::vector<HipSearchInputEvent> baseline;
    for (std::int32_t index = 0; index < 600; ++index) {
        const std::uint32_t action =
                index % 5 == 0 ? 4u
                : index % 5 == 1 ? 1u
                : index % 5 == 2 ? 2u
                : index % 5 == 3 ? 3u
                                 : 5u;
        baseline.push_back(
                {index * 10, action,
                 action == 4u ? 2u : 1u,
                 index * 31});
    }
    const std::vector<std::uint32_t> draws{
            17u, 0u, 999u, 41u, 7u, 7u, 12345u, 3u,
            91u, 12u, 64u, 1u, 77u, 5u, 888u, 2u};
    for (std::uint32_t group = 0u; group < 3u; ++group) {
        for (const auto &window :
             {std::pair<std::int64_t, std::int64_t>{0, 5990},
              {100, 500}, {2500, 2600}, {5900, 7000}}) {
            std::vector<HipSearchInputEvent> legacy = baseline;
            std::vector<HipSearchInputEvent> optimized = baseline;
            LegacyDelete(
                    &legacy, window.first, window.second,
                    group, draws);
            OptimizedDelete(
                    &optimized, window.first, window.second,
                    group, draws);
            if (!SameEvents(legacy, optimized)) {
                return false;
            }
        }
    }

    std::vector<HipSearchInputEvent> legacy = baseline;
    std::vector<HipSearchInputEvent> optimized = baseline;
    for (std::uint32_t group = 0u; group < 3u; ++group) {
        LegacyDelete(&legacy, 250, 5750, group, draws);
        OptimizedDelete(
                &optimized, 250, 5750, group, draws);
    }
    return SameEvents(legacy, optimized);
}

}  // namespace

int main() {
    if (!StateAndInsertionRangeParity()) {
        std::cerr << "state/insertion range parity failed\n";
        return 1;
    }
    if (!WindowAndExistingEligibilityParity()) {
        std::cerr << "window/existing-event parity failed\n";
        return 1;
    }
    if (!DeletionParity()) {
        std::cerr << "deletion parity failed\n";
        return 1;
    }
    return 0;
}
