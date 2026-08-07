#include "simulation/backends/cuda/cuda_search_winner_selection.cuh"
#include "simulation/backends/cuda/cuda_search_progress.cuh"
#include "simulation/backends/cuda/cuda_search_executor.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using forevervalidator::simulation::cuda_search_detail::BetterSample;
using forevervalidator::simulation::cuda_search_detail::BindSampleToCandidate;
using forevervalidator::simulation::cuda_search_detail::DeviceSample;
using forevervalidator::simulation::cuda_search_detail::InvalidCandidateSlot;
using forevervalidator::simulation::cuda_search_detail::StrictlyBetter;
using forevervalidator::simulation::cuda_search_progress_detail::
        InvalidClosestTargetDistanceSquared;
using forevervalidator::simulation::cuda_search_progress_detail::
        IsQualifyingSearchCandidate;
using forevervalidator::simulation::cuda_search_progress_detail::
        SquaredDistanceToTargetVolume;
using forevervalidator::simulation::cuda_search_progress_detail::
        UpdateClosestTargetDistanceSquared;
using forevervalidator::simulation::CudaSearchEvaluatorKind;
using forevervalidator::simulation::CudaSearchConditionIterationValue;
using forevervalidator::simulation::CudaSearchMaximumBaselinePrefixTickCount;
using forevervalidator::simulation::CudaSearchModifierConfiguration;
using forevervalidator::simulation::CudaSearchModifierKind;
using forevervalidator::simulation::IsCudaSearchSimulationMinimumBlocksValid;
using forevervalidator::simulation::PlanCudaSearchPrefixReuse;

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
        bool valid = true) {
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

bool CheckLargeDimensionsAndDeterminism() {
    constexpr std::uint32_t CandidateCount = 1024u;
    constexpr std::uint32_t TickCount = 4096u;
    const auto run = [=] {
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

bool CheckTargetProgressSemantics() {
    const double bounds[6]{-1.0, -2.0, -3.0, 1.0, 2.0, 3.0};
    double closest = InvalidClosestTargetDistanceSquared;
    closest = UpdateClosestTargetDistanceSquared(
            closest,
            SquaredDistanceToTargetVolume(bounds, 4.0, 6.0, 3.0),
            false);
    if (closest != 25.0 ||
        SquaredDistanceToTargetVolume(bounds, 0.0, 0.0, 0.0) != 0.0) {
        return false;
    }
    closest = UpdateClosestTargetDistanceSquared(
            closest,
            SquaredDistanceToTargetVolume(bounds, 8.0, 8.0, 8.0),
            true);
    if (closest != 0.0) {
        return false;
    }

    // Slots 1 and 3 are deduplicated logical candidates whose representative
    // validity has already been expanded by the search executor.
    const bool active[]{true, true, true, true, false};
    const bool expandedValidity[]{true, true, false, false, true};
    std::uint64_t qualifying = 0u;
    for (std::uint32_t slot = 0u; slot < 5u; ++slot) {
        if (IsQualifyingSearchCandidate(
                    active[slot], expandedValidity[slot])) {
            ++qualifying;
        }
    }
    return qualifying == 2u;
}

bool CheckBaselinePrefixInheritance() {
    DeviceSample baseline = Sample(0u, 3u, 12u, 42.0);
    baseline.candidateId = 0u;
    baseline.candidateSlot = InvalidCandidateSlot;
    baseline.logicalOrder = 0u;
    baseline.mutation = false;
    const DeviceSample inherited = BindSampleToCandidate(
            baseline, 9000u, 5u, 12u);
    if (!inherited.valid || !inherited.mutation ||
        inherited.candidateId != 9005u ||
        inherited.candidateSlot != 5u ||
        inherited.evaluationTick != 3u ||
        inherited.logicalOrder != 64u ||
        inherited.score != baseline.score ||
        inherited.timeMs != baseline.timeMs ||
        !IsQualifyingSearchCandidate(true, inherited.valid)) {
        return false;
    }
    double closest = 9.0;
    closest = UpdateClosestTargetDistanceSquared(
            closest, 16.0, false);
    closest = UpdateClosestTargetDistanceSquared(
            closest, 4.0, false);
    return closest == 4.0;
}

CudaSearchModifierConfiguration LowEntropyInsertion() {
    CudaSearchModifierConfiguration modifier;
    modifier.kind = CudaSearchModifierKind::InputInsertion;
    modifier.window.minimumTimeMs = 500;
    modifier.window.maximumTimeMs = 520;
    modifier.steering.enabled = 1u;
    modifier.steering.minimumCount = 1u;
    modifier.steering.maximumCount = 1u;
    modifier.optionFlags = 1u;
    modifier.secondaryAnalogMinimum = 1;
    modifier.secondaryAnalogMaximum = 1;
    return modifier;
}

bool CheckPrefixReusePlanning() {
    const CudaSearchModifierConfiguration modifier =
            LowEntropyInsertion();
    const auto planFor = [&](bool reuse,
                             bool stunts,
                             bool condition,
                             CudaSearchEvaluatorKind evaluator,
                             bool deduplicate) {
        return PlanCudaSearchPrefixReuse(
                reuse, stunts, condition, evaluator,
                deduplicate, 100u, 10u, &modifier, 1u);
    };
    auto plan = planFor(
            true, false, false,
            CudaSearchEvaluatorKind::Velocity, true);
    if (!plan.enabled || plan.lowEntropyChoiceCount != 3u ||
        !plan.DeduplicationStorageEligible(12u) ||
        plan.DeduplicationStorageEligible(8u)) {
        return false;
    }

    plan = planFor(
            true, false, false,
            CudaSearchEvaluatorKind::Velocity, false);
    if (!plan.enabled || plan.lowEntropyChoiceCount != 0u) {
        return false;
    }
    if (planFor(
                false, false, false,
                CudaSearchEvaluatorKind::Velocity, true).enabled) {
        return false;
    }
    if (planFor(
                true, true, false,
                CudaSearchEvaluatorKind::Velocity, true).enabled) {
        return false;
    }
    if (planFor(
                true, false, false,
                CudaSearchEvaluatorKind::FinishTime, true).enabled) {
        return false;
    }
    if (planFor(
                true, false, true,
                CudaSearchEvaluatorKind::Velocity, true).enabled) {
        return false;
    }
    const auto longHorizonPlan = PlanCudaSearchPrefixReuse(
            true, false, false,
            CudaSearchEvaluatorKind::Velocity, true,
            CudaSearchMaximumBaselinePrefixTickCount + 1u,
            10u, &modifier, 1u);
    if (longHorizonPlan.enabled ||
        longHorizonPlan.lowEntropyChoiceCount != 0u ||
        longHorizonPlan.DeduplicationStorageEligible(UINT32_MAX)) {
        return false;
    }
    return IsCudaSearchSimulationMinimumBlocksValid(0u) &&
            IsCudaSearchSimulationMinimumBlocksValid(16u) &&
            IsCudaSearchSimulationMinimumBlocksValid(12u) &&
            IsCudaSearchSimulationMinimumBlocksValid(8u) &&
            !IsCudaSearchSimulationMinimumBlocksValid(1u) &&
            !IsCudaSearchSimulationMinimumBlocksValid(24u);
}

bool CheckIterationConditionBoundary() {
    const double maximumMutationIteration =
            CudaSearchConditionIterationValue(false, UINT64_MAX);
    return CudaSearchConditionIterationValue(true, UINT64_MAX) == 0.0 &&
            CudaSearchConditionIterationValue(false, 0u) == 1.0 &&
            maximumMutationIteration == 18446744073709551616.0 &&
            maximumMutationIteration > 0.0;
}

}  // namespace

int main() {
    if (!CheckEvaluatorKinds() ||
        !CheckTiesAndInvalidCandidates() ||
        !CheckCandidateImprovementSemantics() ||
        !CheckLargeDimensionsAndDeterminism() ||
        !CheckTargetProgressSemantics() ||
        !CheckBaselinePrefixInheritance() ||
        !CheckPrefixReusePlanning() ||
        !CheckIterationConditionBoundary()) {
        return 1;
    }
    return 0;
}
