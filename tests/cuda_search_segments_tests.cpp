#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "simulation/backends/cuda/cuda_candidate_events.cuh"
#include "simulation/backends/cuda/cuda_search_segments.cuh"
#include "format/replay/replay_input_timeline.h"
#include "validation/planning/search_input_boundary.h"

using namespace forevervalidator::simulation;
using namespace forevervalidator::simulation::cuda_search_detail;

static void Require(bool value) {
    if (!value) throw std::runtime_error("segment mutation invariant failed");
}

static void PromotedBoundaryInputs() {
    using namespace forevervalidator::experimental;
    const auto publicEvent = [](CudaSearchInputEvent event) {
        PhysicsSandboxInputEvent result;
        result.timeMs = event.timeMs;
        result.action = static_cast<PhysicsSandboxInputAction>(event.action);
        result.value.kind = static_cast<PhysicsSandboxInputValueKind>(event.valueKind);
        if (event.valueKind == 2u) result.value.analog = event.value;
        else result.value.switchState = static_cast<PhysicsSandboxSwitchState>(event.value);
        return result;
    };
    for (std::uint32_t action : {4u, 2u}) {
        for (int lateOffset : {0, 1, 2, 3, 10}) {
            std::vector<CudaSearchInputEvent> baseline{{0, action, 2, 0}};
            if (lateOffset != 0) baseline.push_back({100 + lateOffset, action, 2, 1234});
            for (int generation = 0; generation < 3; ++generation) {
                std::vector<PhysicsSandboxInputEvent> publicInputs;
                for (auto event : baseline) publicInputs.push_back(publicEvent(event));
                const auto late = search_limits::SearchInputBaselineEnd(
                        publicInputs, 100, {{10, 100}});
                const auto visibleCount = static_cast<std::uint32_t>(late - publicInputs.begin());
                Require(search_limits::SearchInputBaselineEnd(publicInputs, 100, {}) ==
                        std::upper_bound(publicInputs.begin(), publicInputs.end(), 100,
                            [](int time, const auto &event) { return time < event.timeMs; }));
                CudaSearchInputEvent events[64]{}, scratch[64]{};
                std::copy_n(baseline.begin(), visibleCount, events);
                std::uint32_t count = visibleCount;
                events[count++] = {100, action, 2, generation % 2 ? -65536 : 65536};
                count = cuda::candidate_events::NormalizeSuffix(events, count, scratch);
                Require(RestoreSegmentControls(baseline.data(), visibleCount, events,
                                               &count, 64, {}, 100));
                count = cuda::candidate_events::NormalizeSuffix(events, count, scratch);
                std::vector<CudaSearchInputEvent> winner(events, events + count);
                winner.insert(winner.end(), baseline.begin() + visibleCount, baseline.end());
                std::vector<ReplayInputEvent> replayEvents;
                for (auto event : winner) {
                    replayEvents.push_back({static_cast<std::uint32_t>(100000 + event.timeMs),
                        static_cast<ReplayInputActionKind>(event.action), event.valueKind == 2u
                            ? ReplayInputActionValue::Analog(event.value)
                            : ReplayInputActionValue::Switch(static_cast<ReplayInputSwitchState>(event.value))});
                }
                ReplayInputTimeline timeline;
                Require(ReplayInputTimeline::Create({}, {}, std::move(replayEvents), &timeline) ==
                        ReplayInputTimelineCreateResult::Success);
                for (int time : {103, 110, 120}) {
                    DeviceControlState expected{}, actual{};
                    for (auto event : baseline) if (event.timeMs <= time) ApplyControlEvent(expected, event);
                    for (auto event : winner) if (event.timeMs <= time) ApplyControlEvent(actual, event);
                    const auto a = ControlsFromState(expected), b = ControlsFromState(actual);
                    Require(a.steering == b.steering && a.lowSpeedGateA == b.lowSpeedGateA &&
                            a.lowSpeedGateB == b.lowSpeedGateB);
                }
                for (auto event : baseline) if (event.timeMs > 100) {
                    Require(std::any_of(winner.begin(), winner.end(), [&](auto kept) {
                        return kept.timeMs == event.timeMs && kept.action == event.action &&
                               kept.valueKind == event.valueKind && kept.value == event.value;
                    }));
                }
                baseline = std::move(winner);
            }
        }
    }
}

int main() {
    try {
        PromotedBoundaryInputs();
        bool varied = false;
        for (std::uint32_t count = 1u; count <= 32u; ++count) {
            for (std::uint32_t changed = 1u; changed <= count; ++changed) {
                for (std::uint64_t candidate = 1u; candidate <= 64u;
                     ++candidate) {
                    SegmentSelection selection{candidate, count, changed};
                    std::uint32_t selected = 0u;
                    for (std::uint32_t index = 0u; index < count; ++index) {
                        selected += selection.Selected(index);
                        varied |=
                                selection.Selected(index) !=
                                SegmentSelection{candidate + 1u, count, changed}
                                        .Selected(index);
                    }
                    Require(selected == changed);
                }
            }
        }
        Require(varied);
        const CudaSearchMutationSegment segments[]{{0, 20}, {30, 50}, {60, 80}};
        SegmentSelection all{123, 3, 3};
        Require(all.Contains(segments, 30) && !all.Contains(segments, 90));
        for (bool analogWins : {false, true}) {
            for (bool deletion : {false, true}) {
                std::vector<CudaSearchInputEvent> baseline{
                        {0, 5, 1, 1},
                        {0, 1, 1, 1},
                        {analogWins ? 10 : 0, 4, 2, 12000},
                        {analogWins ? 10 : 0, 2, 2, 30000},
                        {25, 4, 2, -17000},
                        {27, 1, 1, 0},
                        {40, 4, 2, -3000},
                        {50, 3, 1, 1}};
                std::stable_sort(
                        baseline.begin(), baseline.end(),
                        [](auto a, auto b) { return a.timeMs < b.timeMs; });
                std::vector<CudaSearchInputEvent> candidate(32), scratch(32);
                std::copy(baseline.begin(), baseline.end(), candidate.begin());
                std::uint32_t size =
                        static_cast<std::uint32_t>(baseline.size());
                Require(RestoreSegmentControls(baseline.data(), size,
                                               candidate.data(), &size, 32, {},
                                               20));
                Require(size == baseline.size());
                if (deletion) {
                    size = static_cast<std::uint32_t>(
                            std::remove_if(candidate.begin(),
                                           candidate.begin() + size,
                                           [](auto event) {
                                               return event.timeMs <= 20;
                                           }) -
                            candidate.begin());
                } else {
                    candidate[size++] = {20, 4, 2, -65536};
                    candidate[size++] = {20, 1, 1, 0};
                }
                size = cuda::candidate_events::NormalizeSuffix(
                        candidate.data(), size, scratch.data());
                Require(RestoreSegmentControls(
                        baseline.data(),
                        static_cast<std::uint32_t>(baseline.size()),
                        candidate.data(), &size, 32, {}, 20));
                size = cuda::candidate_events::NormalizeSuffix(
                        candidate.data(), size, scratch.data());
                for (std::int32_t time : {23, 30, 40, 50, 60}) {
                    DeviceControlState expected{}, actual{};
                    for (auto event : baseline)
                        if (event.timeMs <= time)
                            ApplyControlEvent(expected, event);
                    for (std::uint32_t index = 0; index < size; ++index)
                        if (candidate[index].timeMs <= time)
                            ApplyControlEvent(actual, candidate[index]);
                    const auto a = ControlsFromState(expected),
                               b = ControlsFromState(actual);
                    Require(a.lowSpeedGateA == b.lowSpeedGateA &&
                            a.lowSpeedGateB == b.lowSpeedGateB &&
                            a.steering == b.steering);
                }
                for (auto event : baseline)
                    if (event.timeMs > 20) {
                        Require(std::any_of(
                                candidate.begin(), candidate.begin() + size,
                                [&](auto other) {
                                    return event.timeMs == other.timeMs &&
                                           event.action == other.action &&
                                           event.valueKind == other.valueKind &&
                                           event.value == other.value;
                                }));
                    }
            }
        }
        std::cout << "segment selection and baseline control restoration "
                     "passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
