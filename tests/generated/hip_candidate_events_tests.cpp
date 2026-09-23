// Generated from tests/cuda_candidate_events_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "simulation/backends/hip/generated/hip_candidate_events.cuh"
#include "simulation/backends/hip/generated/hip_sparse_candidate_events.cuh"

namespace {

using forevervalidator::simulation::HipSearchInputEvent;
namespace candidate_events =
        forevervalidator::simulation::hip::candidate_events;
namespace sparse_candidate_events =
        forevervalidator::simulation::hip::sparse_candidate_events;

bool Same(const HipSearchInputEvent &left,
          const HipSearchInputEvent &right) {
    return left.timeMs == right.timeMs &&
            left.action == right.action &&
            left.valueKind == right.valueKind &&
            left.value == right.value;
}

std::vector<HipSearchInputEvent> LegacyNormalize(
        std::vector<HipSearchInputEvent> events,
        const std::vector<HipSearchInputEvent> &baseline,
        std::int64_t mutableFromTimeMs) {
    std::vector<HipSearchInputEvent> sorted;
    for (HipSearchInputEvent value : events) {
        if (value.timeMs < mutableFromTimeMs) {
            continue;
        }
        value.timeMs = std::max(value.timeMs, 0);
        if (value.valueKind == 2u) {
            value.value = std::clamp(value.value, -65536, 65536);
        } else if (value.valueKind == 1u) {
            value.value = value.value == 0 ? 0 : 1;
        }
        const auto insertion = std::upper_bound(
                sorted.begin(), sorted.end(), value.timeMs,
                [](std::int32_t time,
                   const HipSearchInputEvent &event) {
                    return time < event.timeMs;
                });
        sorted.insert(insertion, value);
    }

    std::vector<HipSearchInputEvent> suffix;
    for (const HipSearchInputEvent &value : sorted) {
        const auto duplicate = std::find_if(
                suffix.rbegin(), suffix.rend(),
                [&](const HipSearchInputEvent &candidate) {
                    return candidate.timeMs == value.timeMs &&
                            candidate.action == value.action;
                });
        if (duplicate == suffix.rend()) {
            suffix.push_back(value);
        } else {
            *duplicate = value;
        }
    }

    std::vector<HipSearchInputEvent> result;
    for (const HipSearchInputEvent &value : baseline) {
        if (value.timeMs < mutableFromTimeMs) {
            result.push_back(value);
        }
    }
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
}

bool DifferentialNormalization() {
    std::mt19937 random(0x4f87a221u);
    for (std::uint32_t trial = 0u; trial < 20000u; ++trial) {
        const std::uint32_t count = random() % 96u;
        const std::uint32_t baselineCount = random() % 24u;
        const std::int64_t boundary =
                static_cast<std::int32_t>(random() % 60u) - 10;
        std::vector<HipSearchInputEvent> events;
        std::vector<HipSearchInputEvent> baseline;
        events.reserve(128u);
        baseline.reserve(32u);
        for (std::uint32_t index = 0u;
             index < baselineCount; ++index) {
            baseline.push_back({
                    static_cast<std::int32_t>(random() % 90u) - 20,
                    static_cast<std::uint32_t>(random() % 9u),
                    static_cast<std::uint32_t>(
                            1u + random() % 2u),
                    static_cast<std::int32_t>(random() % 180000u) -
                            90000});
        }
        for (std::uint32_t index = 0u; index < count; ++index) {
            events.push_back({
                    static_cast<std::int32_t>(random() % 90u) - 20,
                    static_cast<std::uint32_t>(random() % 9u),
                    static_cast<std::uint32_t>(
                            1u + random() % 2u),
                    static_cast<std::int32_t>(random() % 180000u) -
                            90000});
        }

        const std::vector<HipSearchInputEvent> expected =
                LegacyNormalize(events, baseline, boundary);
        events.resize(128u);
        std::vector<HipSearchInputEvent> scratch(128u);
        const std::uint32_t actualCount =
                candidate_events::NormalizeWithPrefix(
                        events.data(), count, scratch.data(),
                        baseline.data(),
                        static_cast<std::uint32_t>(baseline.size()),
                        boundary,
                        static_cast<std::uint32_t>(events.size()));
        if (actualCount != expected.size()) {
            std::cerr << "normalization count mismatch at trial "
                      << trial << "\n";
            return false;
        }
        for (std::uint32_t index = 0u;
             index < actualCount; ++index) {
            if (!Same(events[index], expected[index])) {
                std::cerr << "normalization value mismatch at trial "
                          << trial << ", index " << index << "\n";
                return false;
            }
        }
    }
    return true;
}

bool CompactEdits() {
    constexpr std::uint32_t candidates = 32u;
    constexpr std::uint32_t editCapacity = 8u;
    std::vector<std::uint32_t> counts(candidates);
    std::vector<std::uint32_t> erasedCounts(candidates);
    std::vector<std::uint32_t> outputIndices(
            candidates * editCapacity);
    std::vector<std::int32_t> times(candidates * editCapacity);
    std::vector<std::uint32_t> actions(candidates * editCapacity);
    std::vector<std::uint32_t> valueKinds(candidates * editCapacity);
    std::vector<std::int32_t> values(candidates * editCapacity);
    std::vector<std::uint32_t> erasedSourceIndices(
            candidates * editCapacity);
    candidate_events::CoalescedEditStorage storage{
            counts.data(), erasedCounts.data(),
            outputIndices.data(), times.data(),
            actions.data(), valueKinds.data(), values.data(),
            erasedSourceIndices.data(),
            nullptr, nullptr, nullptr,
            candidates, editCapacity, editCapacity};

    const HipSearchInputEvent baseline[]{
            {10, 4u, 2u, 100},
            {20, 1u, 1u, 1},
            {30, 4u, 2u, 300}};
    const auto suffix = candidate_events::SuffixFrom(
            baseline, 3u, 10);
    candidate_events::EditWriter writer(storage, 7u);
    if (!writer.Replace(
                0u, {10, 4u, 2u, -50}) ||
        !writer.Insert(1u, {15, 3u, 1u, 1}) ||
        !writer.Erase(0u) ||
        !writer.Erase(1u)) {
        return false;
    }
    candidate_events::CandidateView view{
            suffix, storage, 7u, 3u};
    std::vector<HipSearchInputEvent> materialized(8u);
    if (!candidate_events::Materialize(
                view, materialized.data(),
                static_cast<std::uint32_t>(materialized.size())) ||
        !Same(materialized[0], {10, 4u, 2u, -50}) ||
        !Same(materialized[1], {15, 3u, 1u, 1}) ||
        !Same(materialized[2], {30, 4u, 2u, 300})) {
        return false;
    }

    // Ordinal-major indexing is contiguous across a warp.
    return candidate_events::EditOffset(storage, 0u, 3u) + 31u ==
            candidate_events::EditOffset(storage, 31u, 3u);
}

bool ReorderedReplacement() {
    constexpr std::uint32_t candidates = 4u;
    std::uint32_t counts[candidates]{};
    std::uint32_t erasedCounts[candidates]{};
    std::uint32_t outputIndices[candidates * 2u]{};
    std::int32_t times[candidates * 2u]{};
    std::uint32_t actions[candidates * 2u]{};
    std::uint32_t valueKinds[candidates * 2u]{};
    std::int32_t values[candidates * 2u]{};
    std::uint32_t erasedSourceIndices[candidates * 2u]{};
    candidate_events::CoalescedEditStorage storage{
            counts, erasedCounts, outputIndices,
            times, actions, valueKinds, values, erasedSourceIndices,
            nullptr, nullptr, nullptr,
            candidates, 2u, 2u};
    const HipSearchInputEvent baseline[]{
            {10, 4u, 2u, 1},
            {20, 1u, 1u, 1},
            {30, 4u, 2u, 3}};
    candidate_events::EditWriter writer(storage, 2u);
    writer.Replace(0u, {5, 4u, 2u, 75});
    writer.Erase(2u);
    candidate_events::CandidateView view{
            {baseline, 3u, 10}, storage, 2u, 3u};
    candidate_events::CandidateCursor cursor(view);
    HipSearchInputEvent events[3]{};
    return cursor.Next(events) &&
            cursor.Next(events + 1u) &&
            cursor.Next(events + 2u) &&
            Same(events[0], {5, 4u, 2u, 75}) &&
            Same(events[1], baseline[0]) &&
            Same(events[2], baseline[1]);
}

bool PackedEdits() {
    constexpr std::uint32_t candidates = 3u;
    constexpr std::uint32_t capacity = 2u;
    std::uint32_t counts[candidates]{};
    std::uint32_t erasedCounts[candidates]{};
    std::uint32_t packedOutputActions[candidates * capacity]{};
    std::int32_t times[candidates * capacity]{};
    std::uint8_t packedValueKinds[candidates * capacity]{};
    std::int32_t values[candidates * capacity]{};
    std::uint16_t packedErasedSources[candidates * capacity]{};
    candidate_events::CoalescedEditStorage storage;
    storage.counts = counts;
    storage.erasedCounts = erasedCounts;
    storage.times = times;
    storage.values = values;
    storage.packedOutputActions = packedOutputActions;
    storage.packedValueKinds = packedValueKinds;
    storage.packedErasedSourceIndices = packedErasedSources;
    storage.candidateStride = candidates;
    storage.editCapacity = capacity;
    storage.eraseCapacity = capacity;

    const HipSearchInputEvent baseline[]{
            {10, 4u, 2u, 10},
            {20, 1u, 1u, 1}};
    candidate_events::EditWriter writer(storage, 1u);
    if (!writer.Insert(0u, {5, 3u, 1u, 1}) ||
        !writer.Erase(1u)) {
        return false;
    }
    HipSearchInputEvent result[2]{};
    return candidate_events::Materialize(
                   {{baseline, 2u, 0}, storage, 1u, 2u},
                   result, 2u) &&
            Same(result[0], {5, 3u, 1u, 1}) &&
            Same(result[1], baseline[0]);
}

bool SortedDirectEdits() {
    constexpr std::uint32_t candidates = 2u;
    constexpr std::uint32_t capacity = 3u;
    std::uint32_t counts[candidates]{};
    std::uint32_t erasedCounts[candidates]{};
    std::uint32_t outputIndices[candidates * capacity]{};
    std::int32_t times[candidates * capacity]{};
    std::uint32_t actions[candidates * capacity]{};
    std::uint32_t valueKinds[candidates * capacity]{};
    std::int32_t values[candidates * capacity]{};
    std::uint32_t erasedSources[candidates * capacity]{};
    candidate_events::CoalescedEditStorage storage{
            counts, erasedCounts, outputIndices, times,
            actions, valueKinds, values, erasedSources,
            nullptr, nullptr, nullptr,
            candidates, capacity, capacity};
    const HipSearchInputEvent baseline[]{
            {10, 4u, 2u, 10},
            {20, 1u, 1u, 1},
            {30, 4u, 2u, 30}};
    candidate_events::EditWriter writer(storage, 1u);
    if (!writer.Insert(2u, {30, 4u, 2u, 31}) ||
        !writer.Erase(2u) ||
        !writer.Insert(0u, {10, 4u, 2u, 11}) ||
        !writer.Erase(0u)) {
        return false;
    }
    candidate_events::SortOutputEdits(storage, 1u);
    candidate_events::SortErasedSources(storage, 1u);
    HipSearchInputEvent result[3]{};
    return candidate_events::Materialize(
                   {{baseline, 3u, 0}, storage, 1u, 3u},
                   result, 3u) &&
            Same(result[0], {10, 4u, 2u, 11}) &&
            Same(result[1], baseline[1]) &&
            Same(result[2], {30, 4u, 2u, 31});
}

bool EqualTimeStableLastWrite() {
    std::vector<HipSearchInputEvent> events{
            {20, UINT32_MAX, 2u, 1},
            {20, 7u, 1u, 1},
            {20, UINT32_MAX, 2u, 2},
            {20, 3u, 1u, 0},
            {20, 7u, 1u, 0}};
    std::vector<HipSearchInputEvent> scratch(events.size());
    const std::uint32_t count = candidate_events::NormalizeSuffix(
            events.data(),
            static_cast<std::uint32_t>(events.size()),
            scratch.data());
    return count == 3u &&
            Same(events[0], {20, UINT32_MAX, 2u, 2}) &&
            Same(events[1], {20, 7u, 1u, 0}) &&
            Same(events[2], {20, 3u, 1u, 0});
}

bool SparseShiftedCursor() {
    constexpr std::uint32_t capacity = 8u;
    const HipSearchInputEvent baseline[]{
            {10, 4u, 2u, 10},
            {20, 1u, 1u, 1},
            {30, 4u, 2u, 30},
            {40, 3u, 1u, 0}};
    std::uint32_t references[capacity]{};
    std::uint32_t snapshotReferences[capacity]{};
    HipSearchInputEvent edits[capacity]{};
    HipSearchInputEvent scratch[capacity]{};
    sparse_candidate_events::Storage storage{
            references, snapshotReferences, edits, scratch,
            1u, capacity};
    sparse_candidate_events::Candidate candidate(
            baseline, 4u, storage, 0u);
    candidate.Initialize();
    if (!candidate.SetAt(2u, {35, 4u, 2u, 99}) ||
        !candidate.SetAt(0u, {35, 4u, 2u, 11})) {
        return false;
    }
    candidate.Canonicalize();
    if (candidate.Count() != 3u) {
        return false;
    }
    sparse_candidate_events::Cursor cursor(
            baseline, storage, 0u, candidate.Count());
    HipSearchInputEvent result[3]{};
    if (!cursor.Next(result) ||
        !cursor.Next(result + 1u) ||
        !cursor.Next(result + 2u) ||
        !Same(result[0], baseline[1]) ||
        !Same(result[1], {35, 4u, 2u, 99}) ||
        !Same(result[2], baseline[3])) {
        return false;
    }

    const std::uint32_t snapshotCount = candidate.Snapshot();
    if (!candidate.UpsertCanonical({35, 4u, 2u, -7}) ||
        candidate.SnapshotChannelStateAt(
                snapshotCount, 4u, 2u, 35, 0) != 99 ||
        candidate.ChannelStateAt(4u, 2u, 35, 0) != -7) {
        return false;
    }
    candidate.CompactEdits();
    return candidate.EditCount() == 1u;
}

bool SparseSmoothRun() {
    constexpr std::uint32_t capacity = 16u;
    const HipSearchInputEvent baseline[]{
            {0, 4u, 2u, 100},
            {10, 1u, 1u, 1},
            {15, 4u, 2u, 200},
            {20, 3u, 1u, 1},
            {30, 4u, 2u, -100}};
    std::uint32_t references[capacity]{};
    std::uint32_t snapshotReferences[capacity]{};
    HipSearchInputEvent edits[capacity]{};
    HipSearchInputEvent scratch[capacity]{};
    sparse_candidate_events::Storage storage{
            references, snapshotReferences, edits, scratch,
            1u, capacity};
    sparse_candidate_events::Candidate candidate(
            baseline, 5u, storage, 0u);
    candidate.Initialize();
    const double weights[]{1.0, 0.5};
    if (!candidate.ApplySmoothSteeringRun(
                10, 30, 10u, 20, 1000, weights, 0u, 0) ||
        candidate.Count() != 7u) {
        return false;
    }
    const HipSearchInputEvent expected[]{
            {0, 4u, 2u, 100},
            {10, 1u, 1u, 1},
            {10, 4u, 2u, 600},
            {15, 4u, 2u, 200},
            {20, 3u, 1u, 1},
            {20, 4u, 2u, 1200},
            {30, 4u, 2u, 400}};
    sparse_candidate_events::Cursor cursor(
            baseline, storage, 0u, candidate.Count());
    for (const HipSearchInputEvent &event : expected) {
        HipSearchInputEvent actual{};
        if (!cursor.Next(&actual) || !Same(actual, event)) {
            return false;
        }
    }
    return true;
}

bool SparseFusedInitialize() {
    constexpr std::uint32_t capacity = 8u;
    const HipSearchInputEvent baseline[]{
            {0, 4u, 2u, 100},
            {10, 1u, 1u, 1},
            {20, 4u, 2u, 200},
            {30, 3u, 1u, 0}};
    std::uint32_t references[capacity]{};
    std::uint32_t snapshotReferences[capacity]{};
    HipSearchInputEvent edits[capacity]{};
    HipSearchInputEvent scratch[capacity]{};
    sparse_candidate_events::Storage storage{
            references, snapshotReferences, edits, scratch,
            1u, capacity};
    sparse_candidate_events::Candidate candidate(
            baseline, 4u, storage, 0u);
    candidate.BeginInitialize();
    candidate.InitializeBaselineAt(0u);
    candidate.InitializeBaselineAt(1u);
    if (!candidate.InitializeEditAt(
                2u, {20, 4u, 2u, -200})) {
        return false;
    }
    candidate.InitializeBaselineAt(3u);
    sparse_candidate_events::Cursor cursor(
            baseline, storage, 0u, candidate.Count());
    const HipSearchInputEvent expected[]{
            baseline[0], baseline[1],
            {20, 4u, 2u, -200}, baseline[3]};
    for (const HipSearchInputEvent &event : expected) {
        HipSearchInputEvent actual{};
        if (!cursor.Next(&actual) || !Same(actual, event)) {
            return false;
        }
    }
    return candidate.EditCount() == 1u;
}

bool SparseInsertionBatch() {
    constexpr std::uint32_t capacity = 64u;
    const HipSearchInputEvent baseline[]{
            {0, 4u, 2u, 100},
            {0, 1u, 1u, 0},
            {0, 3u, 1u, 0},
            {10, 4u, 2u, 200},
            {10, 1u, 1u, 1},
            {15, 3u, 1u, 1},
            {20, 4u, 2u, 300},
            {20, 1u, 1u, 0},
            {25, 3u, 1u, 0},
            {30, 4u, 2u, 400},
            {40, 4u, 2u, 500}};
    std::uint32_t sequentialReferences[capacity]{};
    std::uint32_t sequentialSnapshotReferences[capacity]{};
    HipSearchInputEvent sequentialEdits[capacity]{};
    HipSearchInputEvent sequentialScratch[capacity]{};
    sparse_candidate_events::Storage sequentialStorage{
            sequentialReferences, sequentialSnapshotReferences,
            sequentialEdits, sequentialScratch, 1u, capacity};
    sparse_candidate_events::Candidate sequential(
            baseline, 11u, sequentialStorage, 0u);
    sequential.Initialize();

    std::uint32_t batchedReferences[capacity]{};
    std::uint32_t batchedSnapshotReferences[capacity]{};
    HipSearchInputEvent batchedEdits[capacity]{};
    HipSearchInputEvent batchedScratch[capacity]{};
    sparse_candidate_events::Storage batchedStorage{
            batchedReferences, batchedSnapshotReferences,
            batchedEdits, batchedScratch, 1u, capacity};
    sparse_candidate_events::Candidate batched(
            baseline, 11u, batchedStorage, 0u);
    batched.Initialize();

    // Exercise copying a sparse event from the pass-start snapshot.
    if (!sequential.SetAt(9u, {30, 4u, 2u, 450}) ||
        !batched.SetAt(9u, {30, 4u, 2u, 450})) {
        return false;
    }
    const std::uint32_t sequentialSnapshot = sequential.Snapshot();
    const std::uint32_t batchedSnapshot = batched.BeginInsertionBatch();
    std::int32_t operationTimes[capacity]{};
    std::uint32_t operation = 0u;

    const auto insertAnalog = [&](std::uint32_t firstOperation,
                                  std::uint32_t relativeOperation,
                                  std::int32_t start,
                                  std::int32_t end,
                                  std::int32_t delta) {
        const std::int32_t sequentialPrevious =
                sequential.RemoveActionRangeAndReadState(
                        4u, 2u, start, end, 0);
        const std::int32_t batchedPrevious =
                batched.InsertionBatchChannelStateAt(
                        batchedSnapshot, operationTimes,
                        firstOperation, relativeOperation,
                        4u, 2u, start, 0);
        const std::int32_t sequentialRestore =
                sequential.SnapshotChannelStateAt(
                        sequentialSnapshot, 4u, 2u, end, 0);
        const std::int32_t batchedRestore =
                batched.SnapshotChannelStateAt(
                        batchedSnapshot, 4u, 2u, end, 0);
        const HipSearchInputEvent startEvent{
                start, 4u, 2u, sequentialPrevious + delta};
        const HipSearchInputEvent endEvent{
                end, 4u, 2u, sequentialRestore};
        return sequentialPrevious == batchedPrevious &&
                sequentialRestore == batchedRestore &&
                sequential.UpsertCanonical(startEvent) &&
                (end == start || sequential.UpsertCanonical(endEvent)) &&
                batched.AppendInsertionBatchOperation(
                        operationTimes, operation++,
                        startEvent, endEvent, end > start);
    };
    if (!insertAnalog(0u, 0u, 10, 30, 1000) ||
        !insertAnalog(0u, 1u, 20, 40, -250)) {
        return false;
    }
    constexpr std::uint32_t steeringOperations = 2u;

    const auto insertSwitch = [&](std::uint32_t action,
                                  std::uint32_t firstOperation,
                                  std::uint32_t relativeOperation,
                                  std::int32_t start,
                                  std::int32_t end) {
        const std::int32_t sequentialPrevious =
                sequential.RemoveActionRangeAndReadState(
                        action, 1u, start, end, 0);
        const std::int32_t batchedPrevious =
                batched.InsertionBatchChannelStateAt(
                        batchedSnapshot, operationTimes,
                        firstOperation, relativeOperation,
                        action, 1u, start, 0);
        const std::int32_t sequentialRestore =
                sequential.SnapshotChannelStateAt(
                        sequentialSnapshot, action, 1u, end, 0);
        const std::int32_t batchedRestore =
                batched.SnapshotChannelStateAt(
                        batchedSnapshot, action, 1u, end, 0);
        const HipSearchInputEvent startEvent{
                start, action, 1u,
                sequentialPrevious == 0 ? 1 : 0};
        const HipSearchInputEvent endEvent{
                end, action, 1u, sequentialRestore};
        return sequentialPrevious == batchedPrevious &&
                sequentialRestore == batchedRestore &&
                sequential.UpsertCanonical(startEvent) &&
                (end == start || sequential.UpsertCanonical(endEvent)) &&
                batched.AppendInsertionBatchOperation(
                        operationTimes, operation++,
                        startEvent, endEvent, end > start);
    };
    if (!insertSwitch(1u, steeringOperations, 0u, 10, 20) ||
        !insertSwitch(1u, steeringOperations, 1u, 20, 30)) {
        return false;
    }
    constexpr std::uint32_t accelerateOperations = 2u;
    if (!insertSwitch(
                3u, steeringOperations + accelerateOperations,
                0u, 15, 35) ||
        !batched.FinishInsertionBatch(
                batchedSnapshot, operationTimes,
                steeringOperations, accelerateOperations, 1u) ||
        sequential.Count() != batched.Count()) {
        return false;
    }

    sparse_candidate_events::Cursor sequentialCursor(
            baseline, sequentialStorage, 0u, sequential.Count());
    sparse_candidate_events::Cursor batchedCursor(
            baseline, batchedStorage, 0u, batched.Count());
    for (std::uint32_t index = 0u;
         index < sequential.Count(); ++index) {
        HipSearchInputEvent sequentialEvent{};
        HipSearchInputEvent batchedEvent{};
        if (!sequentialCursor.Next(&sequentialEvent) ||
            !batchedCursor.Next(&batchedEvent) ||
            !Same(sequentialEvent, batchedEvent)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    if (!DifferentialNormalization() ||
        !CompactEdits() ||
        !ReorderedReplacement() ||
        !PackedEdits() ||
        !SortedDirectEdits() ||
        !EqualTimeStableLastWrite() ||
        !SparseShiftedCursor() ||
        !SparseSmoothRun() ||
        !SparseFusedInitialize() ||
        !SparseInsertionBatch()) {
        return 1;
    }
    std::cout << "HIP candidate event tests passed\n";
    return 0;
}
