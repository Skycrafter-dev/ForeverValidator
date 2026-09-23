// Generated from tests/cuda_search_winner_reduction_tests.cu by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_search_winner_selection.cuh"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using forevervalidator::simulation::hip_search_detail::BetterSample;
using forevervalidator::simulation::hip_search_detail::DeviceSample;
using forevervalidator::simulation::hip_search_detail::InvalidCandidateSlot;
using forevervalidator::simulation::hip_search_detail::StrictlyBetter;

static_assert(sizeof(DeviceSample) == 64u);

struct History {
    DeviceSample incumbent;
    std::vector<std::vector<DeviceSample>> candidates;
};

bool Equal(const DeviceSample &left, const DeviceSample &right) {
    return left.score == right.score &&
           left.timeMs == right.timeMs &&
           left.detail0 == right.detail0 &&
           left.detail1 == right.detail1 &&
           left.candidateId == right.candidateId &&
           left.logicalOrder == right.logicalOrder &&
           left.candidateSlot == right.candidateSlot &&
           left.evaluationTick == right.evaluationTick &&
           left.eventCount == right.eventCount &&
           left.valid == right.valid &&
           left.mutation == right.mutation;
}

DeviceSample LegacyWinner(const History &history, bool maximize) {
    BetterSample better{maximize};
    DeviceSample winner = history.incumbent;
    for (const auto &candidate : history.candidates) {
        for (const DeviceSample &sample : candidate) {
            winner = better(winner, sample);
        }
    }
    return winner;
}

DeviceSample CandidateBestWinner(
        const History &history,
        bool maximize) {
    BetterSample better{maximize};
    DeviceSample winner = history.incumbent;
    for (const auto &candidate : history.candidates) {
        DeviceSample candidateBest;
        for (const DeviceSample &sample : candidate) {
            candidateBest = better(candidateBest, sample);
        }
        winner = better(winner, candidateBest);
    }
    return winner;
}

std::uint64_t CandidateImprovementCount(
        const History &history,
        bool maximize) {
    BetterSample better{maximize};
    DeviceSample incumbent = history.incumbent;
    std::uint64_t count = 0u;
    for (const auto &candidate : history.candidates) {
        DeviceSample candidateBest;
        for (const DeviceSample &sample : candidate) {
            candidateBest = better(candidateBest, sample);
        }
        if (StrictlyBetter(candidateBest, incumbent, maximize)) {
            ++count;
            incumbent = candidateBest;
        }
    }
    return count;
}

DeviceSample Sample(
        std::uint32_t slot,
        std::uint32_t tick,
        std::uint32_t tickCount,
        double score,
        bool valid = true,
        std::uint32_t eventCount = 10u) {
    DeviceSample sample;
    sample.score = score;
    sample.timeMs = 1000.0 + tick * 10.0;
    sample.detail0 = slot * 1000.0 + tick;
    sample.detail1 = score * 3.0;
    sample.candidateId = 10000u + slot;
    sample.logicalOrder =
            1u + static_cast<std::uint64_t>(slot) * tickCount + tick;
    sample.candidateSlot = slot;
    sample.evaluationTick = tick;
    sample.eventCount = eventCount;
    sample.valid = valid;
    sample.mutation = true;
    return sample;
}

bool CheckEquivalent(
        const std::string &name,
        const History &history,
        bool maximize) {
    const DeviceSample legacy = LegacyWinner(history, maximize);
    const DeviceSample reduced =
            CandidateBestWinner(history, maximize);
    if (!Equal(legacy, reduced)) {
        std::cerr << name << ": winner mismatch\n";
        return false;
    }
    return true;
}

bool CheckEvaluatorKinds() {
    constexpr std::uint32_t CandidateCount = 37u;
    constexpr std::uint32_t TickCount = 257u;
    for (std::uint32_t evaluator = 0u; evaluator < 5u; ++evaluator) {
        History history;
        history.incumbent.valid = true;
        history.incumbent.logicalOrder = 0u;
        history.incumbent.score = evaluator == 0u ? -1000.0 : 1000.0;
        history.candidates.resize(CandidateCount);
        for (std::uint32_t slot = 0u;
             slot < CandidateCount; ++slot) {
            for (std::uint32_t tick = 0u;
                 tick < TickCount; ++tick) {
                const std::uint32_t value =
                        (slot * 811u + tick * 313u +
                         evaluator * 101u) %
                        997u;
                history.candidates[slot].push_back(
                        Sample(
                                slot, tick, TickCount,
                                static_cast<double>(value),
                                (slot + tick + evaluator) % 19u != 0u));
            }
        }
        if (!CheckEquivalent(
                    "evaluator " + std::to_string(evaluator),
                    history, evaluator == 0u)) {
            return false;
        }
    }
    return true;
}

bool CheckTiesAndInvalidCandidates() {
    constexpr std::uint32_t TickCount = 4u;
    History history;
    history.incumbent = Sample(
            InvalidCandidateSlot, 0u, TickCount, 4.0);
    history.incumbent.logicalOrder = 0u;
    history.incumbent.candidateSlot = InvalidCandidateSlot;
    history.incumbent.mutation = false;
    history.candidates = {
            {Sample(0u, 0u, TickCount, 7.0, false),
             Sample(0u, 1u, TickCount, 4.0),
             Sample(0u, 2u, TickCount, 4.0)},
            {Sample(1u, 0u, TickCount, 4.0),
             Sample(1u, 1u, TickCount, 3.0, false)},
            {Sample(2u, 0u, TickCount, 8.0, false)}};
    if (!CheckEquivalent("incumbent tie", history, false) ||
        LegacyWinner(history, false).candidateSlot !=
                InvalidCandidateSlot) {
        return false;
    }

    history.incumbent.valid = false;
    history.candidates[0][1].score = 2.0;
    history.candidates[0][2].score = 2.0;
    history.candidates[1][0].score = 2.0;
    const DeviceSample winner = LegacyWinner(history, false);
    return CheckEquivalent("candidate and tick ties", history, false) &&
           winner.candidateSlot == 0u &&
           winner.evaluationTick == 1u;
}

bool CheckCandidateImprovementSemantics() {
    constexpr std::uint32_t TickCount = 3u;
    History history;
    history.incumbent.valid = true;
    history.incumbent.score = 10.0;
    history.incumbent.logicalOrder = 0u;
    history.candidates = {
            {Sample(0u, 0u, TickCount, 12.0),
             Sample(0u, 1u, TickCount, 9.0),
             Sample(0u, 2u, TickCount, 8.0)},
            {Sample(1u, 0u, TickCount, 8.0),
             Sample(1u, 1u, TickCount, 11.0)},
            {Sample(2u, 0u, TickCount, 7.0)}};
    return CandidateImprovementCount(history, false) == 2u &&
           CheckEquivalent("candidate improvements", history, false);
}

bool CheckInputCountTieBreak() {
    constexpr std::uint32_t TickCount = 1u;
    DeviceSample incumbent = Sample(
            InvalidCandidateSlot, 0u, TickCount, 4.0,
            true, 10u);
    incumbent.logicalOrder = 0u;
    incumbent.candidateSlot = InvalidCandidateSlot;
    incumbent.mutation = false;

    const DeviceSample smaller = Sample(
            0u, 0u, TickCount, 4.0, true, 9u);
    const DeviceSample larger = Sample(
            1u, 0u, TickCount, 4.0, true, 11u);
    const DeviceSample sameSize = Sample(
            2u, 0u, TickCount, 4.0, true, 10u);

    for (const bool maximize : {false, true}) {
        BetterSample better{maximize};
        if (!Equal(better(incumbent, smaller), smaller) ||
            !StrictlyBetter(smaller, incumbent, maximize) ||
            !Equal(better(incumbent, larger), incumbent) ||
            StrictlyBetter(larger, incumbent, maximize) ||
            !Equal(better(incumbent, sameSize), incumbent) ||
            StrictlyBetter(sameSize, incumbent, maximize)) {
            return false;
        }
    }

    History history;
    history.incumbent = incumbent;
    history.candidates = {{larger}, {smaller}, {sameSize}};
    return CandidateImprovementCount(history, false) == 1u &&
           CandidateImprovementCount(history, true) == 1u;
}

bool CheckLargeDimensionsAndDeterminism() {
    constexpr std::uint32_t CandidateCount = 1024u;
    constexpr std::uint32_t TickCount = 4096u;
    const auto run = [] {
        BetterSample better{false};
        DeviceSample legacy;
        DeviceSample reduced;
        for (std::uint32_t slot = 0u;
             slot < CandidateCount; ++slot) {
            DeviceSample candidateBest;
            for (std::uint32_t tick = 0u;
                 tick < TickCount; ++tick) {
                const double score = static_cast<double>(
                        (slot * 65537u + tick * 8191u) % 104729u);
                const DeviceSample sample =
                        Sample(slot, tick, TickCount, score);
                legacy = better(legacy, sample);
                candidateBest = better(candidateBest, sample);
            }
            reduced = better(reduced, candidateBest);
        }
        return std::pair<DeviceSample, DeviceSample>{legacy, reduced};
    };
    const auto first = run();
    const auto second = run();
    return Equal(first.first, first.second) &&
           Equal(first.first, second.first) &&
           Equal(first.second, second.second);
}

}  // namespace

int main() {
    if (!CheckEvaluatorKinds() ||
        !CheckTiesAndInvalidCandidates() ||
        !CheckCandidateImprovementSemantics() ||
        !CheckInputCountTieBreak() ||
        !CheckLargeDimensionsAndDeterminism()) {
        return 1;
    }
    return 0;
}
