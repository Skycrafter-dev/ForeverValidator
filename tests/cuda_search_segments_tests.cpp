#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "simulation/backends/cuda/cuda_candidate_events.cuh"
#include "simulation/backends/cuda/cuda_search_segments.cuh"

using namespace forevervalidator::simulation;
using namespace forevervalidator::simulation::cuda_search_detail;

static void Require(bool value) {
    if (!value) throw std::runtime_error("segment mutation invariant failed");
}

int main() {
    try {
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
