// Generated from src/simulation/backends/cuda/cuda_sparse_candidate_events.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_SPARSE_CANDIDATE_EVENTS_CUH
#define FOREVERVALIDATOR_HIP_SPARSE_CANDIDATE_EVENTS_CUH

#include <cmath>
#include <cstdint>

#include "simulation/backends/hip/generated/hip_search_executor.h"

#if defined(__HIPCC__)
#define FOREVERVALIDATOR_SPARSE_HD __host__ __device__
#else
#define FOREVERVALIDATOR_SPARSE_HD
#endif

namespace forevervalidator::simulation::hip::sparse_candidate_events {

using Event = HipSearchInputEvent;

constexpr std::uint32_t SparseReferenceBit = UINT32_C(0x80000000);
constexpr std::uint32_t ReferenceIndexMask = UINT32_C(0x7fffffff);

struct Storage {
    // Ordinal is the major dimension so adjacent candidate lanes access
    // adjacent words while walking the same part of their streams.
    std::uint32_t *references = nullptr;
    std::uint32_t *snapshotReferences = nullptr;
    Event *edits = nullptr;
    Event *scratchEdits = nullptr;
    std::uint32_t candidateStride = 0u;
    std::uint32_t eventCapacity = 0u;
};

FOREVERVALIDATOR_SPARSE_HD inline std::uint64_t Offset(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal) {
    return static_cast<std::uint64_t>(ordinal) *
                    storage.candidateStride +
            candidate;
}

FOREVERVALIDATOR_SPARSE_HD inline bool IsSparseReference(
        std::uint32_t reference) {
    return (reference & SparseReferenceBit) != 0u;
}

FOREVERVALIDATOR_SPARSE_HD inline std::uint32_t ReferenceIndex(
        std::uint32_t reference) {
    return reference & ReferenceIndexMask;
}

FOREVERVALIDATOR_SPARSE_HD inline std::uint32_t SparseReference(
        std::uint32_t editIndex) {
    return SparseReferenceBit | editIndex;
}

FOREVERVALIDATOR_SPARSE_HD inline std::uint32_t LoadReference(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal) {
    return storage.references[Offset(storage, candidate, ordinal)];
}

FOREVERVALIDATOR_SPARSE_HD inline void StoreReference(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t ordinal,
        std::uint32_t reference) {
    storage.references[Offset(storage, candidate, ordinal)] = reference;
}

FOREVERVALIDATOR_SPARSE_HD inline Event LoadEdit(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t editIndex) {
    return storage.edits[Offset(storage, candidate, editIndex)];
}

FOREVERVALIDATOR_SPARSE_HD inline void StoreEdit(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t editIndex,
        const Event &event) {
    storage.edits[Offset(storage, candidate, editIndex)] = event;
}

FOREVERVALIDATOR_SPARSE_HD inline Event LoadScratchEdit(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t editIndex) {
    return storage.scratchEdits[Offset(storage, candidate, editIndex)];
}

FOREVERVALIDATOR_SPARSE_HD inline void StoreScratchEdit(
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t editIndex,
        const Event &event) {
    storage.scratchEdits[Offset(storage, candidate, editIndex)] = event;
}

FOREVERVALIDATOR_SPARSE_HD inline Event EventForReference(
        const Event *baseline,
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t reference) {
    return IsSparseReference(reference)
            ? LoadEdit(storage, candidate, ReferenceIndex(reference))
            : baseline[reference];
}

FOREVERVALIDATOR_SPARSE_HD inline Event SnapshotEventForReference(
        const Event *baseline,
        const Storage &storage,
        std::uint32_t candidate,
        std::uint32_t reference) {
    return IsSparseReference(reference)
            ? LoadScratchEdit(
                      storage, candidate, ReferenceIndex(reference))
            : baseline[reference];
}

class Candidate {
public:
    FOREVERVALIDATOR_SPARSE_HD Candidate(
            const Event *baseline,
            std::uint32_t baselineCount,
            Storage storage,
            std::uint32_t candidate,
            std::uint32_t count = 0u,
            std::uint32_t editCount = 0u)
        : baseline_(baseline),
          baselineCount_(baselineCount),
          storage_(storage),
          candidate_(candidate),
          count_(count),
          editCount_(editCount) {}

    FOREVERVALIDATOR_SPARSE_HD void Initialize() {
        count_ = baselineCount_;
        editCount_ = 0u;
        for (std::uint32_t index = 0u; index < count_; ++index) {
            StoreReference(storage_, candidate_, index, index);
        }
    }

    FOREVERVALIDATOR_SPARSE_HD void BeginInitialize() {
        count_ = baselineCount_;
        editCount_ = 0u;
    }

    FOREVERVALIDATOR_SPARSE_HD void InitializeBaselineAt(
            std::uint32_t ordinal) {
        StoreReference(storage_, candidate_, ordinal, ordinal);
    }

    FOREVERVALIDATOR_SPARSE_HD bool InitializeEditAt(
            std::uint32_t ordinal,
            const Event &event) {
        const std::uint32_t editIndex = AllocateEdit(event);
        if (editIndex == UINT32_MAX) {
            return false;
        }
        StoreReference(
                storage_, candidate_, ordinal,
                SparseReference(editIndex));
        return true;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t Count() const {
        return count_;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t EditCount() const {
        return editCount_;
    }

    FOREVERVALIDATOR_SPARSE_HD Event EventAt(
            std::uint32_t ordinal) const {
        return EventForReference(
                baseline_, storage_, candidate_,
                LoadReference(storage_, candidate_, ordinal));
    }

    FOREVERVALIDATOR_SPARSE_HD Event SnapshotEventAt(
            std::uint32_t ordinal) const {
        return SnapshotEventForReference(
                baseline_, storage_, candidate_,
                storage_.snapshotReferences[
                        Offset(storage_, candidate_, ordinal)]);
    }

    FOREVERVALIDATOR_SPARSE_HD bool SetAt(
            std::uint32_t ordinal,
            const Event &event) {
        const std::uint32_t reference =
                LoadReference(storage_, candidate_, ordinal);
        if (IsSparseReference(reference)) {
            StoreEdit(
                    storage_, candidate_, ReferenceIndex(reference), event);
            return true;
        }
        const std::uint32_t editIndex = AllocateEdit(event);
        if (editIndex == UINT32_MAX) {
            return false;
        }
        StoreReference(
                storage_, candidate_, ordinal,
                SparseReference(editIndex));
        return true;
    }

    FOREVERVALIDATOR_SPARSE_HD bool Append(const Event &event) {
        if (count_ >= storage_.eventCapacity) {
            return false;
        }
        const std::uint32_t editIndex = AllocateEdit(event);
        if (editIndex == UINT32_MAX) {
            return false;
        }
        StoreReference(
                storage_, candidate_, count_,
                SparseReference(editIndex));
        ++count_;
        return true;
    }

    FOREVERVALIDATOR_SPARSE_HD void EraseAt(
            std::uint32_t ordinal) {
        for (std::uint32_t index = ordinal + 1u;
             index < count_; ++index) {
            StoreReference(
                    storage_, candidate_, index - 1u,
                    LoadReference(storage_, candidate_, index));
        }
        --count_;
    }

    FOREVERVALIDATOR_SPARSE_HD void EraseSortedOrdinals(
            const std::uint32_t *ordinals,
            std::uint32_t ordinalCount) {
        std::uint32_t destination = 0u;
        std::uint32_t erased = 0u;
        for (std::uint32_t source = 0u;
             source < count_; ++source) {
            if (erased < ordinalCount && source == ordinals[erased]) {
                ++erased;
                continue;
            }
            StoreReference(
                    storage_, candidate_, destination++,
                    LoadReference(storage_, candidate_, source));
        }
        count_ = destination;
    }

    FOREVERVALIDATOR_SPARSE_HD std::int32_t ChannelStateAt(
            std::uint32_t action,
            std::uint32_t valueKind,
            std::int64_t timeMs,
            std::int32_t initialState) const {
        std::uint32_t index = UpperBoundTime(timeMs);
        while (index != 0u) {
            const Event event = EventAt(--index);
            if (event.action == action && event.valueKind == valueKind) {
                return event.value;
            }
        }
        return initialState;
    }

    FOREVERVALIDATOR_SPARSE_HD std::int32_t SnapshotChannelStateAt(
            std::uint32_t snapshotCount,
            std::uint32_t action,
            std::uint32_t valueKind,
            std::int64_t timeMs,
            std::int32_t initialState) const {
        std::uint32_t index = SnapshotUpperBoundTime(
                snapshotCount, timeMs);
        while (index != 0u) {
            const Event event = SnapshotEventAt(--index);
            if (event.action == action && event.valueKind == valueKind) {
                return event.value;
            }
        }
        return initialState;
    }

    FOREVERVALIDATOR_SPARSE_HD std::int32_t
    RemoveActionRangeAndReadState(
            std::uint32_t action,
            std::uint32_t valueKind,
            std::int64_t start,
            std::int64_t end,
            std::int32_t initialState) {
        const std::int32_t state = ChannelStateAt(
                action, valueKind, start, initialState);
        const std::uint32_t rangeBegin = LowerBoundTime(start);
        const std::uint32_t rangeEnd = UpperBoundTime(end);
        std::uint32_t destination = rangeBegin;
        for (std::uint32_t index = rangeBegin;
             index < rangeEnd; ++index) {
            const std::uint32_t reference =
                    LoadReference(storage_, candidate_, index);
            const Event event = EventForReference(
                    baseline_, storage_, candidate_, reference);
            if (event.action == action) {
                continue;
            }
            StoreReference(
                    storage_, candidate_, destination++, reference);
        }
        const std::uint32_t removed = rangeEnd - destination;
        for (std::uint32_t index = rangeEnd;
             index < count_; ++index) {
            StoreReference(
                    storage_, candidate_, index - removed,
                    LoadReference(storage_, candidate_, index));
        }
        count_ -= removed;
        return state;
    }

    FOREVERVALIDATOR_SPARSE_HD bool UpsertCanonical(
            const Event &event) {
        const std::uint32_t insertion = UpperBoundTime(event.timeMs);
        const std::uint32_t groupBegin = LowerBoundTime(event.timeMs);
        for (std::uint32_t duplicate = groupBegin;
             duplicate < insertion; ++duplicate) {
            const Event candidate = EventAt(duplicate);
            if (candidate.action == event.action) {
                return SetAt(duplicate, event);
            }
        }
        if (count_ >= storage_.eventCapacity) {
            return false;
        }
        const std::uint32_t editIndex = AllocateEdit(event);
        if (editIndex == UINT32_MAX) {
            return false;
        }
        for (std::uint32_t index = count_;
             index > insertion; --index) {
            StoreReference(
                    storage_, candidate_, index,
                    LoadReference(storage_, candidate_, index - 1u));
        }
        StoreReference(
                storage_, candidate_, insertion,
                SparseReference(editIndex));
        ++count_;
        return true;
    }

    // Smooth steering writes a complete ordered run at once. This avoids an
    // ordered insertion and state lookup for every tick in the deformation.
    FOREVERVALIDATOR_SPARSE_HD bool ApplySmoothSteeringRun(
            std::int64_t start,
            std::int64_t end,
            std::uint32_t tickDurationMs,
            std::int64_t center,
            std::int32_t amplitude,
            const double *weights,
            std::uint32_t weightOffset,
            std::int32_t initialState) {
        if (start > end || tickDurationMs == 0u) {
            return true;
        }
        const std::uint64_t tickCount64 =
                static_cast<std::uint64_t>(end - start) /
                        tickDurationMs +
                1u;
        if (tickCount64 > UINT32_MAX ||
            !EnsureEditCapacity(
                    static_cast<std::uint32_t>(tickCount64))) {
            return false;
        }

        const std::uint32_t rangeBegin = LowerBoundTime(start);
        std::int32_t state = initialState;
        for (std::uint32_t index = rangeBegin; index != 0u;) {
            const Event event = EventAt(--index);
            if (event.action == 4u && event.valueKind == 2u) {
                state = event.value;
                break;
            }
        }

        for (std::uint32_t index = 0u; index < rangeBegin; ++index) {
            storage_.snapshotReferences[
                    Offset(storage_, candidate_, index)] =
                    LoadReference(storage_, candidate_, index);
        }
        std::uint32_t input = rangeBegin;
        std::uint32_t output = rangeBegin;
        for (std::int64_t time = start;
             time <= end; time += tickDurationMs) {
            while (input < count_ && EventAt(input).timeMs < time) {
                const std::uint32_t reference =
                        LoadReference(storage_, candidate_, input++);
                const Event event = EventForReference(
                        baseline_, storage_, candidate_, reference);
                if (event.action == 4u && event.valueKind == 2u) {
                    state = event.value;
                }
                storage_.snapshotReferences[
                        Offset(storage_, candidate_, output++)] = reference;
            }

            const std::uint32_t groupBegin = input;
            while (input < count_ && EventAt(input).timeMs == time) {
                ++input;
            }
            const std::uint32_t groupEnd = input;
            std::uint32_t steeringOrdinal = UINT32_MAX;
            for (std::uint32_t index = groupBegin;
                 index < groupEnd; ++index) {
                const Event event = EventAt(index);
                if (event.action == 4u && event.valueKind == 2u) {
                    state = event.value;
                    steeringOrdinal = index;
                    break;
                }
            }

            const std::uint64_t distance =
                    static_cast<std::uint64_t>(
                            time > center ? time - center : center - time);
            const std::uint32_t weightIndex =
                    weightOffset +
                    static_cast<std::uint32_t>(
                            distance / tickDurationMs);
            const std::int64_t delta = static_cast<std::int64_t>(
                    llround(static_cast<double>(amplitude) *
                            weights[weightIndex]));
            std::int64_t next = static_cast<std::int64_t>(state) + delta;
            if (next < -65536) {
                next = -65536;
            }
            if (next > 65536) {
                next = 65536;
            }
            state = static_cast<std::int32_t>(next);
            const Event replacement{
                    static_cast<std::int32_t>(time), 4u, 2u, state};
            const std::uint32_t editIndex = AllocateEdit(replacement);
            if (editIndex == UINT32_MAX) {
                return false;
            }
            const std::uint32_t replacementReference =
                    SparseReference(editIndex);

            for (std::uint32_t index = groupBegin;
                 index < groupEnd; ++index) {
                if (output >= storage_.eventCapacity) {
                    return false;
                }
                storage_.snapshotReferences[
                        Offset(storage_, candidate_, output++)] =
                        index == steeringOrdinal
                        ? replacementReference
                        : LoadReference(storage_, candidate_, index);
            }
            if (steeringOrdinal == UINT32_MAX) {
                if (output >= storage_.eventCapacity) {
                    return false;
                }
                storage_.snapshotReferences[
                        Offset(storage_, candidate_, output++)] =
                        replacementReference;
            }
        }
        while (input < count_) {
            if (output >= storage_.eventCapacity) {
                return false;
            }
            storage_.snapshotReferences[
                    Offset(storage_, candidate_, output++)] =
                    LoadReference(storage_, candidate_, input++);
        }
        for (std::uint32_t index = 0u; index < output; ++index) {
            StoreReference(
                    storage_, candidate_, index,
                    storage_.snapshotReferences[
                            Offset(storage_, candidate_, index)]);
        }
        count_ = output;
        return true;
    }

    // Stable ordering preserves source order for shifted events with equal
    // timestamps. Deduplication keeps the first output position and the last
    // value, matching NormalizeSuffix without copying Event arrays.
    FOREVERVALIDATOR_SPARSE_HD void Canonicalize() {
        for (std::uint32_t index = 1u; index < count_; ++index) {
            const std::uint32_t reference =
                    LoadReference(storage_, candidate_, index);
            const Event event = EventForReference(
                    baseline_, storage_, candidate_, reference);
            std::uint32_t insertion = index;
            while (insertion != 0u &&
                   EventAt(insertion - 1u).timeMs > event.timeMs) {
                StoreReference(
                        storage_, candidate_, insertion,
                        LoadReference(
                                storage_, candidate_, insertion - 1u));
                --insertion;
            }
            StoreReference(
                    storage_, candidate_, insertion, reference);
        }

        std::uint32_t output = 0u;
        for (std::uint32_t index = 0u; index < count_; ++index) {
            const std::uint32_t reference =
                    LoadReference(storage_, candidate_, index);
            const Event event = EventForReference(
                    baseline_, storage_, candidate_, reference);
            std::uint32_t duplicate = UINT32_MAX;
            std::uint32_t previous = output;
            while (previous != 0u) {
                const Event candidate = EventForReference(
                        baseline_, storage_, candidate_,
                        LoadReference(
                                storage_, candidate_, previous - 1u));
                if (candidate.timeMs != event.timeMs) {
                    break;
                }
                --previous;
                if (candidate.action == event.action) {
                    duplicate = previous;
                    break;
                }
            }
            if (duplicate == UINT32_MAX) {
                StoreReference(
                        storage_, candidate_, output++, reference);
            } else {
                StoreReference(
                        storage_, candidate_, duplicate, reference);
            }
        }
        count_ = output;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t Snapshot() {
        std::uint32_t snapshotEditCount = 0u;
        for (std::uint32_t index = 0u; index < count_; ++index) {
            const std::uint32_t reference =
                    LoadReference(storage_, candidate_, index);
            std::uint32_t snapshotReference = reference;
            if (IsSparseReference(reference)) {
                StoreScratchEdit(
                        storage_, candidate_, snapshotEditCount,
                        LoadEdit(
                                storage_, candidate_,
                                ReferenceIndex(reference)));
                snapshotReference = SparseReference(snapshotEditCount++);
            }
            storage_.snapshotReferences[
                    Offset(storage_, candidate_, index)] =
                    snapshotReference;
        }
        return count_;
    }

    // Input insertion is sequential within each control channel, but applying
    // every interval with immediate ordered erases/inserts repeatedly moves the
    // whole global event stream. Keep the pass-start stream as a snapshot,
    // record the compact interval boundaries, and merge the final result once.
    FOREVERVALIDATOR_SPARSE_HD std::uint32_t BeginInsertionBatch() {
        const std::uint32_t snapshotCount = Snapshot();
        editCount_ = 0u;
        return snapshotCount;
    }

    FOREVERVALIDATOR_SPARSE_HD std::int32_t
    InsertionBatchChannelStateAt(
            std::uint32_t snapshotCount,
            const std::int32_t *operationTimes,
            std::uint32_t firstOperation,
            std::uint32_t operationCount,
            std::uint32_t action,
            std::uint32_t valueKind,
            std::int64_t timeMs,
            std::int32_t initialState) const {
        bool present = false;
        Event best{};
        std::uint32_t snapshotIndex = SnapshotUpperBoundTime(
                snapshotCount, timeMs);
        while (snapshotIndex != 0u) {
            const Event event = SnapshotEventAt(--snapshotIndex);
            if (event.action != action || event.valueKind != valueKind ||
                OperationRangeCovers(
                        operationTimes, firstOperation,
                        operationCount, event.timeMs)) {
                continue;
            }
            best = event;
            present = true;
            break;
        }

        for (std::uint32_t relative = 0u;
             relative < operationCount; ++relative) {
            const std::uint32_t operation = firstOperation + relative;
            const Event start = LoadEdit(
                    storage_, candidate_, operation * 2u);
            if (start.timeMs <= timeMs &&
                !OperationRangeCovers(
                        operationTimes, operation + 1u,
                        operationCount - relative - 1u,
                        start.timeMs) &&
                (!present || start.timeMs >= best.timeMs)) {
                best = start;
                present = true;
            }
            const Event end = LoadEdit(
                    storage_, candidate_, operation * 2u + 1u);
            if (end.action != UINT32_MAX && end.timeMs <= timeMs &&
                !OperationRangeCovers(
                        operationTimes, operation + 1u,
                        operationCount - relative - 1u,
                        end.timeMs) &&
                (!present || end.timeMs >= best.timeMs)) {
                best = end;
                present = true;
            }
        }
        return present ? best.value : initialState;
    }

    FOREVERVALIDATOR_SPARSE_HD bool AppendInsertionBatchOperation(
            std::int32_t *operationTimes,
            std::uint32_t operation,
            const Event &start,
            const Event &end,
            bool hasEnd) {
        const std::uint64_t required =
                static_cast<std::uint64_t>(operation + 1u) * 2u;
        if (required > storage_.eventCapacity) {
            return false;
        }
        operationTimes[operation * 2u] = start.timeMs;
        operationTimes[operation * 2u + 1u] =
                hasEnd ? end.timeMs : start.timeMs;
        StoreEdit(storage_, candidate_, operation * 2u, start);
        StoreEdit(
                storage_, candidate_, operation * 2u + 1u,
                hasEnd ? end
                       : Event{start.timeMs, UINT32_MAX, 0u, 0});
        editCount_ = static_cast<std::uint32_t>(required);
        return true;
    }

    FOREVERVALIDATOR_SPARSE_HD bool FinishInsertionBatch(
            std::uint32_t snapshotCount,
            const std::int32_t *operationTimes,
            std::uint32_t steeringOperationCount,
            std::uint32_t accelerateOperationCount,
            std::uint32_t brakeOperationCount) {
        const std::uint32_t accelerateFirst = steeringOperationCount;
        const std::uint32_t brakeFirst =
                accelerateFirst + accelerateOperationCount;
        const std::uint32_t totalOperations =
                brakeFirst + brakeOperationCount;

        std::uint32_t liveBoundaryCount = 0u;
        for (std::uint32_t operation = 0u;
             operation < totalOperations; ++operation) {
            const std::uint32_t rangeEnd =
                    operation < accelerateFirst
                    ? accelerateFirst
                    : (operation < brakeFirst
                               ? brakeFirst
                               : totalOperations);
            for (std::uint32_t boundary = 0u;
                 boundary < 2u; ++boundary) {
                const Event event = LoadEdit(
                        storage_, candidate_, operation * 2u + boundary);
                if (event.action == UINT32_MAX ||
                    OperationRangeCovers(
                            operationTimes, operation + 1u,
                            rangeEnd - operation - 1u,
                            event.timeMs)) {
                    continue;
                }
                StoreEdit(
                        storage_, candidate_, liveBoundaryCount++, event);
            }
        }

        // Stable time ordering preserves operation order for equal-time
        // boundaries, exactly matching repeated UpperBound insertions.
        for (std::uint32_t index = 1u;
             index < liveBoundaryCount; ++index) {
            const Event event = LoadEdit(storage_, candidate_, index);
            std::uint32_t insertion = index;
            while (insertion != 0u &&
                   LoadEdit(
                           storage_, candidate_,
                           insertion - 1u).timeMs > event.timeMs) {
                StoreEdit(
                        storage_, candidate_, insertion,
                        LoadEdit(
                                storage_, candidate_,
                                insertion - 1u));
                --insertion;
            }
            StoreEdit(storage_, candidate_, insertion, event);
        }

        const std::uint32_t mergedSteeringOperationCount =
                MergeOperationRanges(
                        const_cast<std::int32_t *>(operationTimes),
                        0u, steeringOperationCount);
        const std::uint32_t mergedAccelerateOperationCount =
                MergeOperationRanges(
                        const_cast<std::int32_t *>(operationTimes),
                        accelerateFirst, accelerateOperationCount);
        const std::uint32_t mergedBrakeOperationCount =
                MergeOperationRanges(
                        const_cast<std::int32_t *>(operationTimes),
                        brakeFirst, brakeOperationCount);

        editCount_ = liveBoundaryCount;
        count_ = 0u;
        std::uint32_t snapshotIndex = 0u;
        std::uint32_t boundaryIndex = 0u;
        std::uint32_t steeringRange = 0u;
        std::uint32_t accelerateRange = 0u;
        std::uint32_t brakeRange = 0u;
        const auto coveredByMergedRanges =
                [&](const Event &event) {
                    std::uint32_t firstOperation = 0u;
                    std::uint32_t operationCount = 0u;
                    std::uint32_t *range = nullptr;
                    if (event.action == 4u) {
                        operationCount = mergedSteeringOperationCount;
                        range = &steeringRange;
                    } else if (event.action == 1u) {
                        firstOperation = accelerateFirst;
                        operationCount = mergedAccelerateOperationCount;
                        range = &accelerateRange;
                    } else if (event.action == 3u) {
                        firstOperation = brakeFirst;
                        operationCount = mergedBrakeOperationCount;
                        range = &brakeRange;
                    } else {
                        return false;
                    }
                    while (*range < operationCount &&
                           operationTimes[
                                   (firstOperation + *range) * 2u + 1u] <
                                   event.timeMs) {
                        ++*range;
                    }
                    return *range < operationCount &&
                            operationTimes[
                                    (firstOperation + *range) * 2u] <=
                                    event.timeMs;
                };
        while (true) {
            while (snapshotIndex < snapshotCount) {
                const Event event = SnapshotEventAt(snapshotIndex);
                if (!coveredByMergedRanges(event)) {
                    break;
                }
                ++snapshotIndex;
            }
            if (snapshotIndex >= snapshotCount &&
                boundaryIndex >= liveBoundaryCount) {
                break;
            }
            const bool takeSnapshot =
                    snapshotIndex < snapshotCount &&
                    (boundaryIndex >= liveBoundaryCount ||
                     SnapshotEventAt(snapshotIndex).timeMs <=
                             LoadEdit(
                                     storage_, candidate_,
                                     boundaryIndex).timeMs);
            if (count_ >= storage_.eventCapacity) {
                return false;
            }
            if (takeSnapshot) {
                const std::uint32_t snapshotReference =
                        storage_.snapshotReferences[
                                Offset(
                                        storage_, candidate_,
                                        snapshotIndex++)];
                if (!IsSparseReference(snapshotReference)) {
                    StoreReference(
                            storage_, candidate_, count_++,
                            snapshotReference);
                    continue;
                }
                if (editCount_ >= storage_.eventCapacity) {
                    return false;
                }
                StoreEdit(
                        storage_, candidate_, editCount_,
                        LoadScratchEdit(
                                storage_, candidate_,
                                ReferenceIndex(snapshotReference)));
                StoreReference(
                        storage_, candidate_, count_++,
                        SparseReference(editCount_++));
            } else {
                StoreReference(
                        storage_, candidate_, count_++,
                        SparseReference(boundaryIndex++));
            }
        }
        return true;
    }

    FOREVERVALIDATOR_SPARSE_HD void CompactEdits() {
        std::uint32_t live = 0u;
        for (std::uint32_t index = 0u; index < count_; ++index) {
            const std::uint32_t reference =
                    LoadReference(storage_, candidate_, index);
            if (!IsSparseReference(reference)) {
                continue;
            }
            StoreScratchEdit(
                    storage_, candidate_, live,
                    LoadEdit(
                            storage_, candidate_,
                            ReferenceIndex(reference)));
            StoreReference(
                    storage_, candidate_, index,
                    SparseReference(live++));
        }
        for (std::uint32_t index = 0u; index < live; ++index) {
            StoreEdit(
                    storage_, candidate_, index,
                    LoadScratchEdit(storage_, candidate_, index));
        }
        editCount_ = live;
    }

private:
    FOREVERVALIDATOR_SPARSE_HD static bool OperationRangeCovers(
            const std::int32_t *operationTimes,
            std::uint32_t firstOperation,
            std::uint32_t operationCount,
            std::int64_t timeMs) {
        for (std::uint32_t relative = 0u;
             relative < operationCount; ++relative) {
            const std::uint32_t operation = firstOperation + relative;
            if (timeMs >= operationTimes[operation * 2u] &&
                timeMs <= operationTimes[operation * 2u + 1u]) {
                return true;
            }
        }
        return false;
    }

    FOREVERVALIDATOR_SPARSE_HD static std::uint32_t
    MergeOperationRanges(
            std::int32_t *operationTimes,
            std::uint32_t firstOperation,
            std::uint32_t operationCount) {
        for (std::uint32_t index = 1u;
             index < operationCount; ++index) {
            const std::int32_t start = operationTimes[
                    (firstOperation + index) * 2u];
            const std::int32_t end = operationTimes[
                    (firstOperation + index) * 2u + 1u];
            std::uint32_t insertion = index;
            while (insertion != 0u &&
                   operationTimes[
                           (firstOperation + insertion - 1u) * 2u] >
                           start) {
                operationTimes[
                        (firstOperation + insertion) * 2u] =
                        operationTimes[
                                (firstOperation + insertion - 1u) * 2u];
                operationTimes[
                        (firstOperation + insertion) * 2u + 1u] =
                        operationTimes[
                                (firstOperation + insertion - 1u) * 2u + 1u];
                --insertion;
            }
            operationTimes[
                    (firstOperation + insertion) * 2u] = start;
            operationTimes[
                    (firstOperation + insertion) * 2u + 1u] = end;
        }
        std::uint32_t merged = 0u;
        for (std::uint32_t index = 0u;
             index < operationCount; ++index) {
            const std::int32_t start = operationTimes[
                    (firstOperation + index) * 2u];
            const std::int32_t end = operationTimes[
                    (firstOperation + index) * 2u + 1u];
            if (merged != 0u &&
                start <= operationTimes[
                                 (firstOperation + merged - 1u) * 2u + 1u]) {
                std::int32_t &mergedEnd = operationTimes[
                        (firstOperation + merged - 1u) * 2u + 1u];
                if (end > mergedEnd) {
                    mergedEnd = end;
                }
                continue;
            }
            operationTimes[(firstOperation + merged) * 2u] = start;
            operationTimes[(firstOperation + merged) * 2u + 1u] = end;
            ++merged;
        }
        return merged;
    }

    FOREVERVALIDATOR_SPARSE_HD bool EnsureEditCapacity(
            std::uint32_t additional) {
        if (additional <= storage_.eventCapacity - editCount_) {
            return true;
        }
        CompactEdits();
        return additional <= storage_.eventCapacity - editCount_;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t LowerBoundTime(
            std::int64_t timeMs) const {
        std::uint32_t first = 0u;
        std::uint32_t last = count_;
        while (first < last) {
            const std::uint32_t middle = first + (last - first) / 2u;
            if (EventAt(middle).timeMs < timeMs) {
                first = middle + 1u;
            } else {
                last = middle;
            }
        }
        return first;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t UpperBoundTime(
            std::int64_t timeMs) const {
        std::uint32_t first = 0u;
        std::uint32_t last = count_;
        while (first < last) {
            const std::uint32_t middle = first + (last - first) / 2u;
            if (EventAt(middle).timeMs <= timeMs) {
                first = middle + 1u;
            } else {
                last = middle;
            }
        }
        return first;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t SnapshotUpperBoundTime(
            std::uint32_t snapshotCount,
            std::int64_t timeMs) const {
        std::uint32_t first = 0u;
        std::uint32_t last = snapshotCount;
        while (first < last) {
            const std::uint32_t middle = first + (last - first) / 2u;
            if (SnapshotEventAt(middle).timeMs <= timeMs) {
                first = middle + 1u;
            } else {
                last = middle;
            }
        }
        return first;
    }

    FOREVERVALIDATOR_SPARSE_HD bool EditReferenced(
            std::uint32_t editIndex) const {
        for (std::uint32_t index = 0u; index < count_; ++index) {
            const std::uint32_t reference =
                    LoadReference(storage_, candidate_, index);
            if (IsSparseReference(reference) &&
                ReferenceIndex(reference) == editIndex) {
                return true;
            }
        }
        return false;
    }

    FOREVERVALIDATOR_SPARSE_HD std::uint32_t AllocateEdit(
            const Event &event) {
        std::uint32_t editIndex = UINT32_MAX;
        if (editCount_ < storage_.eventCapacity) {
            editIndex = editCount_++;
        } else {
            for (std::uint32_t index = 0u;
                 index < storage_.eventCapacity; ++index) {
                if (!EditReferenced(index)) {
                    editIndex = index;
                    break;
                }
            }
        }
        if (editIndex != UINT32_MAX) {
            StoreEdit(storage_, candidate_, editIndex, event);
        }
        return editIndex;
    }

    const Event *baseline_ = nullptr;
    std::uint32_t baselineCount_ = 0u;
    Storage storage_{};
    std::uint32_t candidate_ = 0u;
    std::uint32_t count_ = 0u;
    std::uint32_t editCount_ = 0u;
};

class Cursor {
public:
    FOREVERVALIDATOR_SPARSE_HD Cursor(
            const Event *baseline,
            Storage storage,
            std::uint32_t candidate,
            std::uint32_t count)
        : baseline_(baseline),
          storage_(storage),
          candidate_(candidate),
          count_(count) {}

    FOREVERVALIDATOR_SPARSE_HD bool Next(Event *event) {
        if (index_ >= count_) {
            return false;
        }
        *event = EventForReference(
                baseline_, storage_, candidate_,
                LoadReference(storage_, candidate_, index_++));
        return true;
    }

private:
    const Event *baseline_ = nullptr;
    Storage storage_{};
    std::uint32_t candidate_ = 0u;
    std::uint32_t count_ = 0u;
    std::uint32_t index_ = 0u;
};

#undef FOREVERVALIDATOR_SPARSE_HD

}  // namespace forevervalidator::simulation::hip::sparse_candidate_events

#endif
