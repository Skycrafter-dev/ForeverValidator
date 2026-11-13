// Twin search (M13): host candidate generation (see twin_search_candidates.h).
#include "twin_search_candidates.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <forevervalidator/experimental/around_walls_table.h>

#include "simulation/backends/cuda/cuda_candidate_events.cuh"
#include "simulation/backends/cuda/cuda_custom_volume.cuh"
#include "simulation/backends/cuda/cuda_modifier_event_ops.cuh"
#include "simulation/backends/cuda/cuda_search_segments.cuh"

namespace forevervalidator::simulation::twin_search {

namespace extracted {
// The names the extracted functions use, as cuda_search_executor.cu declares them.
using cuda_search_detail::ApplyControlEvent;
using cuda_search_detail::DeviceControlState;
using cuda_search_detail::DeviceSample;
namespace modifier_ops = cuda_search_modifier_detail;

// Verbatim CUDA source: its switches cover only the kinds each function
// handles (nvcc does not warn about the others).
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#endif
#include "twin_cuda_search_extract.inc"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}  // namespace extracted

struct CandidateGenerator::Scratch {
    std::vector<CudaSearchInputEvent> events, temporary, passBaseline;
    std::vector<std::uint32_t> eligible;
    std::uint32_t random[624];
};

bool MaximizesSearchScore(CudaSearchEvaluatorKind kind, std::uint32_t optionFlags) {
    return extracted::MaximizesScore(kind, optionFlags);
}

cuda_search_detail::DeviceSample EvaluateSearchState(const CudaSearchEvaluatorConfiguration &evaluator,
                                                     const CudaCandidatePhysicsState &state,
                                                     const GmVec3 &previousPosition, const GmVec3 &previousVelocity,
                                                     double previousTimeMs, double currentTimeMs, std::uint32_t stuntsScore,
                                                     bool *reported) {
    return extracted::EvaluateState(evaluator, state, previousPosition, previousVelocity, previousTimeMs, currentTimeMs,
                                    stuntsScore, reported);
}

bool PrepareMutationProgram(const CudaSearchExecutorConfiguration &configuration, MutationProgram *program,
                            std::string *diagnostic) {
    auto fail = [&](const char *message) {
        if (diagnostic != nullptr) *diagnostic = message;
        return false;
    };
    if (program == nullptr || configuration.tickDurationMs == 0u || configuration.modifiers.empty() ||
        configuration.maximumEventCount < configuration.baselineInputs.size() ||
        configuration.maximumEventCount > UINT32_MAX || configuration.branchTimeMs < 0 ||
        configuration.branchTimeMs > INT32_MAX - static_cast<std::int64_t>(configuration.tickDurationMs)) {
        return fail("invalid twin search mutation configuration");
    }
    MutationProgram p;
    p.tickDurationMs = configuration.tickDurationMs;
    p.branchTimeMs = configuration.branchTimeMs;
    p.mutableFromTimeMs = configuration.branchTimeMs + configuration.tickDurationMs;
    // As CudaSearchExecutor::Create: split at the mutable boundary.
    cuda_search_detail::SearchInputPartition partition;
    if (!cuda_search_detail::PartitionSearchInputs(configuration.baselineInputs, configuration.branchTimeMs,
                                                   p.mutableFromTimeMs, &partition)) {
        return fail("search inputs cannot be partitioned at the mutable boundary");
    }
    p.changedSegmentCount = configuration.changedSegmentCount;
    p.segments = configuration.mutationSegments;
    if (!p.segments.empty()) {
        if (configuration.tickDurationMs < 3u || configuration.changedSegmentCount == 0u ||
            configuration.changedSegmentCount > p.segments.size()) {
            return fail("mutation segment count or tick duration is invalid");
        }
        std::int64_t previousEnd = p.mutableFromTimeMs - 1;
        for (CudaSearchMutationSegment &segment : p.segments) {
            if (segment.minimumTimeMs <= previousEnd || segment.maximumTimeMs < segment.minimumTimeMs ||
                segment.maximumTimeMs > INT32_MAX - static_cast<std::int64_t>(configuration.tickDurationMs) ||
                segment.minimumTimeMs % configuration.tickDurationMs != 0 ||
                segment.maximumTimeMs % configuration.tickDurationMs != 0) {
                return fail("mutation segments must be ordered, disjoint, mutable tick ranges");
            }
            previousEnd = segment.maximumTimeMs;
            segment.minimumTimeMs -= p.mutableFromTimeMs;
            segment.maximumTimeMs -= p.mutableFromTimeMs;
        }
    }
    p.baselineInputs = std::move(partition.mutableSuffix);
    p.immutablePrefix = std::move(partition.immutablePrefix);
    p.mutableBoundaryControls = partition.mutableBoundaryControls;
    p.eventCapacity = static_cast<std::uint32_t>(configuration.maximumEventCount - p.immutablePrefix.size());
    p.modifiers = configuration.modifiers;
    for (CudaSearchModifierConfiguration &modifier : p.modifiers) {
        if (modifier.window.maximumTimeMs < p.mutableFromTimeMs) {
            return fail("modifier window does not intersect the mutable suffix");
        }
        modifier.window.minimumTimeMs =
                std::max(modifier.window.minimumTimeMs, p.mutableFromTimeMs) - p.mutableFromTimeMs;
        modifier.window.maximumTimeMs -= p.mutableFromTimeMs;
    }
    p.smoothWeights = configuration.smoothWeights;
    // Window-local mutation (CudaSearchExecutor::Create when the reference
    // pipeline is not requested): events past every window and the
    // evaluation end are an immutable tail, untouched by any modifier.
    std::int64_t materializationEndTimeMs = configuration.evaluationEndTimeMs - p.mutableFromTimeMs;
    for (const CudaSearchModifierConfiguration &modifier : p.modifiers) {
        materializationEndTimeMs = std::max(materializationEndTimeMs, modifier.window.maximumTimeMs);
    }
    for (const CudaSearchMutationSegment &segment : p.segments) {
        materializationEndTimeMs =
                std::max(materializationEndTimeMs, segment.maximumTimeMs + configuration.tickDurationMs);
    }
    cuda_search_detail::SearchInputWindow window;
    if (!cuda_search_detail::PartitionSearchInputWindow(p.baselineInputs, materializationEndTimeMs, &window) ||
        window.immutableTail.size() > p.eventCapacity) {
        return fail("search inputs cannot be partitioned at the materialization horizon");
    }
    p.eventCapacity -= static_cast<std::uint32_t>(window.immutableTail.size());
    p.baselineInputs = std::move(window.materialized);
    p.immutableTail = std::move(window.immutableTail);
    p.baselineInputsCanonical = extracted::CanonicalBaselineInputs(p.baselineInputs, 0);
    if (!p.segments.empty() && !p.baselineInputsCanonical) {
        return fail("segmented mutation requires canonical baseline inputs");
    }
    if (p.baselineInputs.size() > p.eventCapacity) {
        return fail("search baseline exceeds the event capacity");
    }
    *program = std::move(p);
    return true;
}

CandidateGenerator::CandidateGenerator(const MutationProgram &program)
    : program_(program), scratch_(new Scratch) {
    const std::size_t capacity = std::max<std::size_t>(program.eventCapacity, 1u);
    scratch_->events.resize(capacity);
    scratch_->temporary.resize(capacity);
    scratch_->passBaseline.resize(capacity);
    scratch_->eligible.resize(capacity);
}

CandidateGenerator::~CandidateGenerator() { delete scratch_; }

// GenerateSearchCandidatesKernel's materialized path over the window-local
// inputs (normalising after every pass, as legacyMutationPipeline does).
void CandidateGenerator::Generate(std::uint64_t candidateId, bool baseline, GeneratedCandidate *out) {
    const MutationProgram &p = program_;
    const CudaSearchInputEvent *base = p.baselineInputs.data();
    const std::uint32_t baseCount = static_cast<std::uint32_t>(p.baselineInputs.size());
    const std::uint32_t capacity = p.eventCapacity;
    CudaSearchInputEvent *events = scratch_->events.data();
    CudaSearchInputEvent *temporary = scratch_->temporary.data();
    CudaSearchInputEvent *passBaseline = scratch_->passBaseline.data();
    std::uint32_t *eligible = scratch_->eligible.data();
    extracted::DeviceMt19937 random(scratch_->random, 0u, 1u);

    std::copy(base, base + baseCount, events);
    std::uint32_t eventCount = baseCount;
    out->status = CandidateStatus::Success;
    auto capacityExceeded = [&](std::uint32_t count) {
        out->status = CandidateStatus::CapacityExceeded;
        out->active = false;
        out->mutationCount = 0u;
        out->events.assign(events, events + std::min(count, capacity));
    };

    if (!baseline) {
        const std::uint32_t modifierCount = static_cast<std::uint32_t>(p.modifiers.size());
        const std::uint32_t segmentCount = static_cast<std::uint32_t>(p.segments.size());
        bool normalized = p.baselineInputsCanonical;
        const cuda_search_detail::SegmentSelection selection{
                candidateId ^ (static_cast<std::uint64_t>(p.modifiers[0].window.seed) << 32u), segmentCount,
                p.changedSegmentCount};
        for (std::uint32_t segmentIndex = 0u; segmentIndex < (segmentCount == 0u ? 1u : segmentCount);
             ++segmentIndex) {
            if (segmentCount != 0u && !selection.Selected(segmentIndex)) continue;
            for (std::uint32_t pass = 0u; pass < modifierCount; ++pass) {
                CudaSearchModifierConfiguration modifier = p.modifiers[pass];
                if (segmentCount != 0u) {
                    const CudaSearchMutationSegment segment = p.segments[segmentIndex];
                    modifier.window.minimumTimeMs = std::max(modifier.window.minimumTimeMs, segment.minimumTimeMs);
                    modifier.window.maximumTimeMs = std::min(modifier.window.maximumTimeMs, segment.maximumTimeMs);
                    if (modifier.window.minimumTimeMs > modifier.window.maximumTimeMs) continue;
                }
                if (!extracted::ApplyModifier(modifier, pass + segmentIndex * modifierCount, candidateId, random,
                                              p.tickDurationMs, 0, p.mutableBoundaryControls, base, baseCount,
                                              events, &eventCount, capacity, temporary, passBaseline, eligible,
                                              p.smoothWeights.data(), true, &normalized)) {
                    capacityExceeded(eventCount);
                    return;
                }
            }
            if (segmentCount != 0u) {
                if (!normalized) {
                    eventCount = extracted::NormalizeEvents(events, eventCount, temporary, base, baseCount, 0, true,
                                                            capacity);
                    if (eventCount == UINT32_MAX) {
                        capacityExceeded(capacity);
                        return;
                    }
                }
                const std::int64_t end = p.segments[segmentIndex].maximumTimeMs;
                if (!selection.Contains(p.segments.data(), end + p.tickDurationMs) &&
                    !cuda_search_detail::RestoreSegmentControls(base, baseCount, events, &eventCount, capacity,
                                                                p.mutableBoundaryControls, end)) {
                    capacityExceeded(eventCount);
                    return;
                }
                normalized = false;
            }
            // legacyMutationPipeline: always renormalise against the baseline.
            std::copy(base, base + baseCount, passBaseline);
            eventCount = extracted::NormalizeEvents(events, eventCount, temporary, passBaseline, baseCount, 0, true,
                                                    capacity);
            if (eventCount == UINT32_MAX) {
                capacityExceeded(capacity);
                return;
            }
            normalized = true;
        }
    }
    out->mutationCount = baseline ? 0u : extracted::EffectiveChangeCount(base, baseCount, events, eventCount);
    if (!baseline && eventCount != baseCount) {
        out->mutationCount += static_cast<std::uint32_t>(p.immutableTail.size());
    }
    out->active = baseline || out->mutationCount != 0u;
    out->events.assign(events, events + eventCount);
}

}  // namespace forevervalidator::simulation::twin_search
