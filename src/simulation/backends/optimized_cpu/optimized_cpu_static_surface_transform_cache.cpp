#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <typeinfo>
#include <utility>

#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_bounds_overlap.h"

namespace {

const OptimizedCpuStaticMeshTriangleSidecar *FindTriangleSidecar(
        const std::vector<std::unique_ptr<
                OptimizedCpuStaticMeshTriangleSidecar>> &sidecars,
        const GmSurfMesh &mesh) {
    for (const auto &sidecar : sidecars) {
        if (sidecar->IsFor(mesh)) {
            return sidecar.get();
        }
    }
    return nullptr;
}

bool ContainsMesh(const std::vector<const GmSurfMesh *> &meshes,
                  const GmSurfMesh &mesh) {
    for (const GmSurfMesh *candidate : meshes) {
        if (candidate == &mesh) {
            return true;
        }
    }
    return false;
}

bool IsBoundedBroadPhaseFloat(float value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0u) {
        return true;
    }
    const std::uint32_t exponent = magnitude >> 23u;
    return exponent >= 67u && exponent <= 187u;
}

bool IsBoundedBroadPhaseVector(const GmVec3 &value) noexcept {
    return IsBoundedBroadPhaseFloat(value.x) &&
           IsBoundedBroadPhaseFloat(value.y) &&
           IsBoundedBroadPhaseFloat(value.z);
}

bool IsBoundedBroadPhaseIso(const GmIso4 &value) noexcept {
    return IsBoundedBroadPhaseVector(value.rotation.basisX) &&
           IsBoundedBroadPhaseVector(value.rotation.basisY) &&
           IsBoundedBroadPhaseVector(value.rotation.basisZ) &&
           IsBoundedBroadPhaseVector(value.translation);
}

bool IsBoundedMovingTree(const CPlugTree &tree) noexcept {
    if (!IsBoundedBroadPhaseVector(tree.Box().center) ||
        !IsBoundedBroadPhaseVector(tree.Box().halfExtents) ||
        (tree.HasLocalTransform() &&
         !IsBoundedBroadPhaseIso(tree.LocalIso()))) {
        return false;
    }
    for (u32 childIndex = 0u; childIndex < tree.GetChildCount(); ++childIndex) {
        if (!IsBoundedMovingTree(*tree.GetChild(childIndex))) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool OptimizedCpuStaticSurfaceTransformGroup::
WholeTreeBoundsOverlapAnySurface(
        const GmBoxAligned &movingBounds,
        std::uint64_t *recordTests) const noexcept {
    if (sourceRecords_ == nullptr || sourceRecordCount_ == 0u) {
        return true;
    }

    u32 sourceIndex = 0u;
    while (sourceIndex < sourceRecordCount_) {
        const CHmsCollisionManagerSColOctreeCell &record =
                sourceRecords_[sourceIndex];
        if (recordTests != nullptr) {
            ++*recordTests;
        }
        if (!forevervalidator::simulation::OptimizedCpuStaticBoundsOverlap(
                    movingBounds, record.Bounds())) {
            sourceIndex += record.SubtreeEntryCount();
            continue;
        }
        if (record.ContainsSurface()) {
            return true;
        }
        ++sourceIndex;
    }
    return false;
}

bool OptimizedCpuStaticSurfaceTransformGroup::
BroadPhaseArithmeticIsBoundedFor(
        const CPlugTree &movingTree,
        const GmIso4 &movingIso) const noexcept {
    if (!staticBroadPhaseArithmeticIsBounded_ ||
        !IsBoundedBroadPhaseIso(movingIso)) {
        return false;
    }
    for (std::size_t index = 0u;
         index < boundedMovingTreeCount_;
         ++index) {
        if (boundedMovingTrees_[index] == &movingTree) {
            return true;
        }
    }
    if (!IsBoundedMovingTree(movingTree) ||
        boundedMovingTreeCount_ == boundedMovingTrees_.size()) {
        return false;
    }
    boundedMovingTrees_[boundedMovingTreeCount_++] = &movingTree;
    return true;
}

bool OptimizedCpuStaticSurfaceTransformGroup::ShouldProbeWholePass(
        const CPlugTree &movingTree) const noexcept {
    WholePassPredictorEntry *freeEntry = nullptr;
    for (WholePassPredictorEntry &entry : wholePassPredictors_) {
        if (entry.movingTree == &movingTree) {
            if (entry.predictedEmpty || entry.probesUntilReacquire == 0u) {
                return true;
            }
            --entry.probesUntilReacquire;
            return false;
        }
        if (entry.movingTree == nullptr && freeEntry == nullptr) {
            freeEntry = &entry;
        }
    }
    if (freeEntry == nullptr) {
        return false;
    }
    freeEntry->movingTree = &movingTree;
    return true;
}

void OptimizedCpuStaticSurfaceTransformGroup::ObserveWholePassProbe(
        const CPlugTree &movingTree,
        bool empty) const noexcept {
    for (WholePassPredictorEntry &entry : wholePassPredictors_) {
        if (entry.movingTree != &movingTree) {
            continue;
        }
        entry.predictedEmpty = empty;
        entry.probesUntilReacquire = empty ? 0u : 7u;
        return;
    }
}

bool OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpanFor(
        const CPlugTree &movingTree,
        u32 temporalSlotOrdinal,
        const GmBoxAligned &movingBounds,
        TemporalCandidateSpan *result) const noexcept {
    if (result == nullptr || sourceRecords_ == nullptr ||
        sourceRecordCount_ == 0u) {
        return false;
    }

    auto contains = [](const GmBoxAligned &outer,
                       const GmBoxAligned &inner) {
        return inner.center.x - inner.halfExtents.x >=
                       outer.center.x - outer.halfExtents.x &&
               inner.center.x + inner.halfExtents.x <=
                       outer.center.x + outer.halfExtents.x &&
               inner.center.y - inner.halfExtents.y >=
                       outer.center.y - outer.halfExtents.y &&
               inner.center.y + inner.halfExtents.y <=
                       outer.center.y + outer.halfExtents.y &&
               inner.center.z - inner.halfExtents.z >=
                       outer.center.z - outer.halfExtents.z &&
               inner.center.z + inner.halfExtents.z <=
                       outer.center.z + outer.halfExtents.z;
    };

    static const bool Instrument =
            std::getenv("FOREVERVALIDATOR_OPTIMIZED_CPU_TEMPORAL_STATS") !=
            nullptr;
    if (Instrument) {
        ++temporalQueryCount_;
        u32 sourceIndex = 0u;
        while (sourceIndex < sourceRecordCount_) {
            const CHmsCollisionManagerSColOctreeCell &record =
                    sourceRecords_[sourceIndex];
            ++temporalAuthoritativeRecordTests_;
            if (!forevervalidator::simulation::
                        OptimizedCpuStaticBoundsOverlap(
                                movingBounds, record.Bounds())) {
                sourceIndex += record.SubtreeEntryCount();
                continue;
            }
            ++sourceIndex;
        }
    }
    TemporalCandidateEntry *entry = nullptr;
    bool entryTagMatches = false;
    if (temporalSlotOrdinal < ordinalTemporalCandidates_.size()) {
        entry = &ordinalTemporalCandidates_[temporalSlotOrdinal];
        if (entry->movingTree == &movingTree) {
            entryTagMatches = true;
        } else if (entry->movingTree == nullptr) {
            entry->movingTree = &movingTree;
        } else {
            entry = nullptr;
        }
    }
    if (entry == nullptr) {
        entry = TemporalCandidateFallbackFor(
                movingTree, &entryTagMatches);
        if (entry == nullptr) {
            return false;
        }
    }
    if (entryTagMatches &&
        contains(entry->validityBounds, movingBounds)) {
        if (Instrument) {
            ++temporalHitCount_;
            temporalCandidateRecordTests_ +=
                    entry->candidateRecordIndices.size();
        }
        result->size = entry->candidateRecordIndices.size();
        result->data = result->size == 0u
                ? nullptr
                : entry->candidateRecordIndices.data();
        return true;
    }

    try {
        constexpr float FatMargin = 8.0f;
        entry->validityBounds = movingBounds;
        entry->validityBounds.halfExtents.x += FatMargin;
        entry->validityBounds.halfExtents.y += FatMargin;
        entry->validityBounds.halfExtents.z += FatMargin;
        entry->candidateRecordIndices.clear();

        u32 sourceIndex = 0u;
        while (sourceIndex < sourceRecordCount_) {
            const CHmsCollisionManagerSColOctreeCell &record =
                    sourceRecords_[sourceIndex];
            if (Instrument) {
                ++temporalBaselineRecordTests_;
            }
            if (!forevervalidator::simulation::
                        OptimizedCpuStaticBoundsOverlap(
                                entry->validityBounds,
                                record.Bounds())) {
                sourceIndex += record.SubtreeEntryCount();
                continue;
            }
            if (record.ContainsSurface()) {
                entry->candidateRecordIndices.push_back(sourceIndex);
            }
            ++sourceIndex;
        }

        if (Instrument) {
            ++temporalRebuildCount_;
            temporalCandidateRecordTests_ +=
                    entry->candidateRecordIndices.size();
        }
        result->size = entry->candidateRecordIndices.size();
        result->data = result->size == 0u
                ? nullptr
                : entry->candidateRecordIndices.data();
        return true;
    } catch (const std::bad_alloc &) {
        if (entry != nullptr) {
            entry->movingTree = nullptr;
            entry->candidateRecordIndices.clear();
        }
        return false;
    }
}

#if defined(_MSC_VER)
#define FOREVERVALIDATOR_COLD_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define FOREVERVALIDATOR_COLD_NOINLINE __attribute__((cold, noinline))
#else
#define FOREVERVALIDATOR_COLD_NOINLINE
#endif

FOREVERVALIDATOR_COLD_NOINLINE
OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateEntry *
OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateFallbackFor(
        const CPlugTree &movingTree,
        bool *entryTagMatches) const noexcept {
    for (TemporalCandidateEntry &candidate : temporalCandidates_) {
        if (candidate.movingTree == &movingTree) {
            *entryTagMatches = true;
            return &candidate;
        }
    }
    try {
        temporalCandidates_.push_back({});
        TemporalCandidateEntry *entry = &temporalCandidates_.back();
        entry->movingTree = &movingTree;
        *entryTagMatches = false;
        return entry;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

#undef FOREVERVALIDATOR_COLD_NOINLINE

void OptimizedCpuStaticSurfaceTransformGroup::ClearTemporalCandidates(
        void) const noexcept {
    if (std::getenv("FOREVERVALIDATOR_OPTIMIZED_CPU_TEMPORAL_STATS") !=
        nullptr) {
        std::fprintf(
                stderr,
                "optimized_cpu_temporal queries=%llu hits=%llu rebuilds=%llu "
                "authoritative_record_tests=%llu rebuild_record_tests=%llu "
                "candidate_record_tests=%llu\n",
                static_cast<unsigned long long>(temporalQueryCount_),
                static_cast<unsigned long long>(temporalHitCount_),
                static_cast<unsigned long long>(temporalRebuildCount_),
                static_cast<unsigned long long>(
                        temporalAuthoritativeRecordTests_),
                static_cast<unsigned long long>(
                        temporalBaselineRecordTests_),
                static_cast<unsigned long long>(
                        temporalCandidateRecordTests_));
    }
    for (TemporalCandidateEntry &entry : ordinalTemporalCandidates_) {
        entry = TemporalCandidateEntry{};
    }
    temporalCandidates_.clear();
    temporalQueryCount_ = 0u;
    temporalHitCount_ = 0u;
    temporalRebuildCount_ = 0u;
    temporalAuthoritativeRecordTests_ = 0u;
    temporalBaselineRecordTests_ = 0u;
    temporalCandidateRecordTests_ = 0u;
}

OptimizedCpuStaticSurfaceTransformCache::
~OptimizedCpuStaticSurfaceTransformCache(void) {
    ClearTemporalCandidates();
}

bool OptimizedCpuStaticSurfaceTransformCache::TryRebuild(
        const CHmsCollisionManagerSZone &zone) noexcept {
    try {
        OptimizedCpuStaticSurfaceTransformCache rebuilt;
        rebuilt.sourceZone_ = &zone;
        for (u32 groupIndex = 0u;
             groupIndex < CHmsCollisionManager_GroupCount;
             ++groupIndex) {
            const CHmsCollisionManagerSGroup *group =
                    zone.GroupAtOrNull(groupIndex);
            OptimizedCpuStaticSurfaceTransformGroup &cachedGroup =
                    rebuilt.groups_[groupIndex];
            cachedGroup.sourceGroup_ = group;
            if (group == nullptr) {
                continue;
            }
            const auto &records = group->staticTrees.Entries();
            cachedGroup.sourceRecords_ = records.data();
            cachedGroup.sourceRecordCount_ = records.size();
            cachedGroup.staticBroadPhaseArithmeticIsBounded_ = true;
            for (const CHmsCollisionManagerSColOctreeCell &record : records) {
                const GmBoxAligned &bounds = record.Bounds();
                if (!IsBoundedBroadPhaseVector(bounds.center) ||
                    !IsBoundedBroadPhaseVector(bounds.halfExtents)) {
                    cachedGroup.staticBroadPhaseArithmeticIsBounded_ = false;
                    break;
                }
            }
            cachedGroup.inverses_.resize(records.size());
            cachedGroup.triangleSidecars_.resize(records.size(), nullptr);
            for (std::size_t recordIndex = 0u;
                 recordIndex < records.size();
                 ++recordIndex) {
                if (!records[recordIndex].ContainsSurface()) {
                    continue;
                }
                const CHmsCollisionManagerSColOctreeCell::StaticSurface
                        &surface = records[recordIndex].SurfaceData();
                cachedGroup.inverses_[recordIndex].SetInverse(
                        surface.location);

                const GmSurf *geometry = surface.surface == nullptr
                        ? nullptr
                        : surface.surface->Geometry();
                if (geometry == nullptr ||
                    typeid(*geometry) != typeid(GmSurfMesh)) {
                    continue;
                }
                const GmSurfMesh &mesh =
                        static_cast<const GmSurfMesh &>(*geometry);
                if (ContainsMesh(
                            rebuilt.unavailableTriangleSidecarMeshes_,
                            mesh)) {
                    continue;
                }
                const OptimizedCpuStaticMeshTriangleSidecar *sidecar =
                        FindTriangleSidecar(
                                rebuilt.triangleSidecars_, mesh);
                if (sidecar == nullptr) {
                    auto candidate = std::make_unique<
                            OptimizedCpuStaticMeshTriangleSidecar>();
                    if (!candidate->TryBuild(mesh)) {
                        rebuilt.unavailableTriangleSidecarMeshes_.push_back(
                                &mesh);
                        continue;
                    }
                    sidecar = candidate.get();
                    rebuilt.triangleSidecars_.push_back(
                            std::move(candidate));
                }
                cachedGroup.triangleSidecars_[recordIndex] = sidecar;
            }
        }
        *this = std::move(rebuilt);
        return true;
    } catch (const std::bad_alloc &) {
        Clear();
        return false;
    }
}

void OptimizedCpuStaticSurfaceTransformCache::Clear(void) noexcept {
    sourceZone_ = nullptr;
    certifiedZone_ = nullptr;
    for (OptimizedCpuStaticSurfaceTransformGroup &group : groups_) {
        group.ClearTemporalCandidates();
        group.sourceGroup_ = nullptr;
        group.sourceRecords_ = nullptr;
        group.sourceRecordCount_ = 0u;
        group.staticBroadPhaseArithmeticIsBounded_ = false;
        group.boundedMovingTrees_.fill(nullptr);
        group.boundedMovingTreeCount_ = 0u;
        group.wholePassPredictors_.fill({});
        group.inverses_.clear();
        group.triangleSidecars_.clear();
    }
    triangleSidecars_.clear();
    unavailableTriangleSidecarMeshes_.clear();
}

bool OptimizedCpuStaticSurfaceTransformCache::CertifyForAdvance(
        const CHmsCollisionManagerSZone &zone) noexcept {
    certifiedZone_ = nullptr;
    if (sourceZone_ != &zone) {
        return false;
    }

    for (u32 groupIndex = 0u;
         groupIndex < CHmsCollisionManager_GroupCount;
         ++groupIndex) {
        const CHmsCollisionManagerSGroup *group =
                zone.GroupAtOrNull(groupIndex);
        const OptimizedCpuStaticSurfaceTransformGroup &cachedGroup =
                groups_[groupIndex];
        if (cachedGroup.sourceGroup_ != group) {
            return false;
        }
        if (group == nullptr) {
            continue;
        }

        const auto &records = group->staticTrees.Entries();
        if (cachedGroup.sourceRecords_ != records.data() ||
            cachedGroup.sourceRecordCount_ != records.size() ||
            cachedGroup.inverses_.size() != records.size() ||
            cachedGroup.triangleSidecars_.size() != records.size()) {
            return false;
        }
        for (std::size_t recordIndex = 0u;
             recordIndex < records.size();
             ++recordIndex) {
            const CHmsCollisionManagerSColOctreeCell &record =
                    records[recordIndex];
            // Static tree flags remain immutable during the synchronous
            // advance, so cached record loops can omit this dependent load.
            if (record.ContainsSurface() &&
                (record.SurfaceData().tree == nullptr ||
                 !record.SurfaceData().tree->AllowsSurfaceCollision())) {
                return false;
            }
            const OptimizedCpuStaticMeshTriangleSidecar *sidecar =
                    cachedGroup.triangleSidecars_[recordIndex];
            if (sidecar == nullptr) {
                continue;
            }
            if (!record.ContainsSurface()) {
                return false;
            }
            const CHmsCollisionManagerSColOctreeCell::StaticSurface
                    &surface = record.SurfaceData();
            const GmSurf *geometry = surface.surface == nullptr
                    ? nullptr
                    : surface.surface->Geometry();
            if (geometry == nullptr ||
                typeid(*geometry) != typeid(GmSurfMesh) ||
                !sidecar->IsFor(
                        static_cast<const GmSurfMesh &>(*geometry))) {
                return false;
            }
        }
    }

    certifiedZone_ = &zone;
    return true;
}

void OptimizedCpuStaticSurfaceTransformCache::ClearTemporalCandidates(
        void) noexcept {
    certifiedZone_ = nullptr;
    for (OptimizedCpuStaticSurfaceTransformGroup &group : groups_) {
        group.ClearTemporalCandidates();
    }
}

bool OptimizedCpuStaticSurfaceTransformCache::IsFor(
        const CHmsCollisionManagerSZone &zone) const noexcept {
    return sourceZone_ == &zone;
}

bool OptimizedCpuStaticSurfaceTransformCache::IsCertifiedFor(
        const CHmsCollisionManagerSZone &zone) const noexcept {
    return certifiedZone_ == &zone;
}

const OptimizedCpuStaticSurfaceTransformGroup *
OptimizedCpuStaticSurfaceTransformCache::GroupFor(
        const CHmsCollisionManagerSGroup &group) const noexcept {
    for (const OptimizedCpuStaticSurfaceTransformGroup &candidate : groups_) {
        if (candidate.sourceGroup_ != &group) {
            continue;
        }
        // Static-tree records are immutable after scene construction. Check
        // the vector backing and count once per group pass; any rebuild uses
        // the established uncached OptimizedCpu path for the whole pass.
        const auto &records = group.staticTrees.Entries();
        if (candidate.sourceRecords_ != records.data() ||
            candidate.sourceRecordCount_ != records.size() ||
            candidate.inverses_.size() != records.size() ||
            candidate.triangleSidecars_.size() != records.size()) {
            return nullptr;
        }
        return &candidate;
    }
    return nullptr;
}
