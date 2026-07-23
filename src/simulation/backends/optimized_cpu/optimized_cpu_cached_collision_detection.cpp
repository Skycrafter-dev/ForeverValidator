// OptimizedCpu static collision traversal with immutable transform sidecars.

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_triangle_mesh_query.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_bounds_overlap.h"

#include <cstdio>
#include <cstdlib>

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

using forevervalidator::simulation::OptimizedCpuStaticBoundsOverlap;

namespace {

struct WholePassProbeStats {
    std::uint64_t passes = 0u;
    std::uint64_t probes = 0u;
    std::uint64_t empty = 0u;
    std::uint64_t recordTests = 0u;

    ~WholePassProbeStats() {
        if (std::getenv("FOREVERVALIDATOR_OPTIMIZED_CPU_WHOLE_PASS_STATS") !=
            nullptr) {
            std::fprintf(stderr,
                         "optimized_cpu_whole_pass passes=%llu probes=%llu empty=%llu "
                         "record_tests=%llu\n",
                         static_cast<unsigned long long>(passes),
                         static_cast<unsigned long long>(probes),
                         static_cast<unsigned long long>(empty),
                         static_cast<unsigned long long>(recordTests));
        }
    }
};

WholePassProbeStats &WholePassStats() {
    static WholePassProbeStats stats;
    return stats;
}

bool TrySkipWholeTreeBoundsEmpty(
        const GmIso4 &movingIso,
        const CPlugTree &movingTree,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms,
        std::uint64_t *recordTests) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    const unsigned int savedMxcsr = _mm_getcsr();
    constexpr unsigned int MxcsrControlMask = 0xffc0u;
    constexpr unsigned int DeterministicMxcsrControl = 0x1f80u;
    constexpr unsigned int InexactStatus = 0x20u;
    if ((savedMxcsr & MxcsrControlMask) != DeterministicMxcsrControl ||
        (savedMxcsr & InexactStatus) == 0u ||
        !transforms.BroadPhaseArithmeticIsBoundedFor(
                movingTree, movingIso)) {
        return false;
    }
    GmBoxAligned movingBounds;
    movingTree.GetTransformedCollisionBox(movingIso, movingBounds);
    const bool empty = !transforms.WholeTreeBoundsOverlapAnySurface(
            movingBounds, recordTests);
    return empty;
#else
    (void)movingIso;
    (void)movingTree;
    (void)transforms;
    (void)recordTests;
    return false;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((hot, noinline))
#endif
void DetectNativeBinary32CachedTemporalSpan(
        CHmsCollisionManagerSZone &zone,
        const GmBoxAligned &movingBox,
        CPlugTree &movingTree,
        CPlugSurface &movingSurface,
        const GmIso4 &movingLocation,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms,
        const OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
                &temporalCandidates) {
    const u32 *candidate = temporalCandidates.data;
    const u32 *const end = candidate + temporalCandidates.size;
    const CHmsCollisionManagerSColOctreeCell *const records =
            transforms.RecordData();
    const GmIso4 *const inverses = transforms.InverseData();
    const OptimizedCpuStaticMeshTriangleSidecar *const *const
            triangleSidecars = transforms.TriangleSidecarData();
    for (; candidate != end; ++candidate) {
        const u32 staticTreeIndex = *candidate;
        const CHmsCollisionManagerSColOctreeCell *record =
                &records[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(
                    movingBox, record->Bounds())) {
            continue;
        }

        // The span contains only surface records, and the advance certificate
        // proves every static surface tree is collision-enabled.
        const CHmsCollisionManagerSColOctreeCell::StaticSurface
                &staticSurface = record->SurfaceData();
        SHmsSphereBufferContact *sphereContact = nullptr;
        CHmsCollisionBuffer *buffer = zone.ChooseCollisionOutputBuffer(
                &movingTree, &movingSurface, &sphereContact);
        const u32 firstNew = buffer->PhysicalCollisionCount();
        const SPlugSurfaceLocatedPair surfacePair = {
            movingSurface,
            movingLocation,
            *staticSurface.surface,
            staticSurface.location,
        };
        const int collided =
                ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32WithStaticCache(
                        surfacePair,
                        inverses[staticTreeIndex],
                        triangleSidecars[staticTreeIndex],
                        *buffer);
        if (collided == 0) {
            continue;
        }
        if (sphereContact != nullptr) {
            zone.AddSphereContactOnce(sphereContact);
        }
        zone.TagNewStaticCollisions(
                buffer, firstNew, &movingTree, record);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((cold, noinline))
#endif
void DetectNativeBinary32CachedColdFallback(
        CHmsCollisionManagerSZone &zone,
        const GmBoxAligned &movingBox,
        CPlugTree &movingTree,
        CPlugSurface &movingSurface,
        const GmIso4 &movingLocation,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms,
        GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees) {
    const auto processSurfaceRecord = [&](u32 staticTreeIndex) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(
                    movingBox, record->Bounds()) ||
            !record->ContainsSurface()) {
            return;
        }

        const CHmsCollisionManagerSColOctreeCell::StaticSurface
                &staticSurface = record->SurfaceData();
        SHmsSphereBufferContact *sphereContact = nullptr;
        CHmsCollisionBuffer *buffer = zone.ChooseCollisionOutputBuffer(
                &movingTree, &movingSurface, &sphereContact);
        const u32 firstNew = buffer->PhysicalCollisionCount();
        const SPlugSurfaceLocatedPair surfacePair = {
            movingSurface,
            movingLocation,
            *staticSurface.surface,
            staticSurface.location,
        };
        const int collided =
                ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32WithStaticCache(
                        surfacePair,
                        transforms.InverseAt(staticTreeIndex),
                        transforms.TriangleSidecarAt(staticTreeIndex),
                        *buffer);
        if (collided == 0) {
            return;
        }
        if (sphereContact != nullptr) {
            zone.AddSphereContactOnce(sphereContact);
        }
        zone.TagNewStaticCollisions(
                buffer, firstNew, &movingTree, record);
    };

    const u32 staticTreeCount = staticTrees.GetCount();
    for (u32 staticTreeIndex = 0u;
         staticTreeIndex < staticTreeCount;) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(
                    movingBox, record->Bounds())) {
            staticTreeIndex += record->SubtreeEntryCount();
            continue;
        }
        processSurfaceRecord(staticTreeIndex);
        ++staticTreeIndex;
    }
}

}  // namespace

void CHmsCollisionManager::SZone::DetectCollisionsCorpusOptimizedCpuCached(
        CHmsCollisionBuffer &collisionBuffer,
        CHmsCorpus *corpus,
        const OptimizedCpuStaticSurfaceTransformCache &transforms) {
    activeCollisionBuffer = &collisionBuffer;

    const u32 groupIndex = corpus->Item()->CollisionGroup();
    CHmsCollisionManagerSGroup *group = &groups[groupIndex - 1u];

    for (CHmsCollisionManagerSAgainstGroup &againstEntry :
         group->againstGroups) {
        CHmsCollisionManagerSAgainstGroup *against = &againstEntry;
        activeCollisionGroupPair = against->collisionGroupPair;

        for (const SGroup::MovingCorpusState &target :
             against->targetGroup->movingCorpuses) {
            CHmsCorpus *other = target.corpus;
            if (against->collisionSchedule.IsEnabled(*corpus, *other)) {
                DetectCollisionBetween(corpus, other);
            }
        }

        activeStaticTargetGroup = against->targetGroup;
        if (against->targetGroup->StaticTreeCount() > 1u) {
            activeCorpusA = corpus;
            const OptimizedCpuStaticSurfaceTransformGroup *groupTransforms =
                    transforms.GroupFor(*against->targetGroup);
            if (groupTransforms == nullptr) {
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpu(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree());
            } else {
                u32 nextTemporalSlotOrdinal = 0u;
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree(),
                        nextTemporalSlotOrdinal,
                        *groupTransforms);
            }
        }
    }

    MergeQueuedSphereContacts(collisionBuffer);
}

void CHmsCollisionManager::SZone::
DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
        const GmIso4 &movingIsoRef,
        const CPlugTree &movingTree,
        u32 &nextTemporalSlotOrdinal,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms) {
    CHmsCollisionManagerSZone *zone = this;
    const GmIso4 *movingIso = &movingIsoRef;
    const u32 temporalSlotOrdinal = nextTemporalSlotOrdinal++;
    if (!movingTree.HasWorldBox()) {
        return;
    }

    GmIso4 localIso;
    movingTree.ComposeCollisionIso(*movingIso, localIso);

    const u32 childCount = movingTree.GetChildCount();
    for (u32 childIndex = 0u; childIndex < childCount; ++childIndex) {
        zone->DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
                localIso,
                *movingTree.GetChild(childIndex),
                nextTemporalSlotOrdinal,
                transforms);
    }

    CPlugSurface *movingSurface = movingTree.Surface();
    if (movingSurface == nullptr) {
        return;
    }
    CPlugTree *movingTreeNode = const_cast<CPlugTree *>(&movingTree);

    GmBoxAligned movingBox;
    movingTree.GetTransformedCollisionBox(*movingIso, movingBox);

    GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees =
            zone->activeStaticTargetGroup->staticTrees;
    const u32 staticTreeCount = staticTrees.GetCount();
    const auto processSurfaceRecord = [&](u32 staticTreeIndex,
                                          bool surfaceIsKnown) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(movingBox, record->Bounds()) ||
            (!surfaceIsKnown && !record->ContainsSurface())) {
            return;
        }

        const SColOctreeCell::StaticSurface &staticSurface =
                record->SurfaceData();
        SHmsSphereBufferContact *sphereContact = nullptr;
        CHmsCollisionBuffer *buffer = zone->ChooseCollisionOutputBuffer(
                movingTreeNode, movingSurface, &sphereContact);
        const u32 firstNew = buffer->PhysicalCollisionCount();
        const SPlugSurfaceLocatedPair surfacePair = {
            *movingSurface,
            localIso,
            *staticSurface.surface,
            staticSurface.location,
        };

        const int surfaceCollisionResult =
                ComputeCollisionOptimizedCpuWithStaticMeshTriangleSidecar(
                        surfacePair,
                        transforms.InverseAt(staticTreeIndex),
                        transforms.TriangleSidecarAt(staticTreeIndex),
                        *buffer);
        if (surfaceCollisionResult) {
            if (sphereContact != nullptr) {
                zone->AddSphereContactOnce(sphereContact);
            }
            zone->TagNewStaticCollisions(
                    buffer, firstNew, movingTreeNode, record);
        }
    };

    OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
            temporalCandidates;
    if (transforms.TemporalCandidateSpanFor(
                movingTree,
                temporalSlotOrdinal,
                movingBox,
                &temporalCandidates)) {
        for (std::size_t candidateIndex = 0u;
             candidateIndex < temporalCandidates.size;
             ++candidateIndex) {
            processSurfaceRecord(
                    temporalCandidates.data[candidateIndex], true);
        }
        return;
    }

    for (u32 staticTreeIndex = 0u;
         staticTreeIndex < staticTreeCount;) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(movingBox, record->Bounds())) {
            staticTreeIndex += record->SubtreeEntryCount();
            continue;
        }

        processSurfaceRecord(staticTreeIndex, false);

        ++staticTreeIndex;
    }
}

void CHmsCollisionManager::SZone::
DetectCollisionsCorpusOptimizedCpuNativeBinary32Cached(
        CHmsCollisionBuffer &collisionBuffer,
        CHmsCorpus *corpus,
        const OptimizedCpuStaticSurfaceTransformCache &transforms) {
    activeCollisionBuffer = &collisionBuffer;

    const u32 groupIndex = corpus->Item()->CollisionGroup();
    CHmsCollisionManagerSGroup *group = &groups[groupIndex - 1u];

    for (CHmsCollisionManagerSAgainstGroup &againstEntry :
         group->againstGroups) {
        CHmsCollisionManagerSAgainstGroup *against = &againstEntry;
        activeCollisionGroupPair = against->collisionGroupPair;

        for (const SGroup::MovingCorpusState &target :
             against->targetGroup->movingCorpuses) {
            CHmsCorpus *other = target.corpus;
            if (against->collisionSchedule.IsEnabled(*corpus, *other)) {
                DetectCollisionBetween(corpus, other);
            }
        }

        activeStaticTargetGroup = against->targetGroup;
        if (against->targetGroup->StaticTreeCount() > 1u) {
            activeCorpusA = corpus;
            const OptimizedCpuStaticSurfaceTransformGroup *groupTransforms =
                    transforms.GroupFor(*against->targetGroup);
            if (groupTransforms == nullptr) {
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree());
            } else {
                static const bool InstrumentWholePass =
                        std::getenv(
                                "FOREVERVALIDATOR_OPTIMIZED_CPU_WHOLE_PASS_STATS") !=
                        nullptr;
                WholePassProbeStats *stats = InstrumentWholePass
                        ? &WholePassStats()
                        : nullptr;
                if (stats != nullptr) {
                    ++stats->passes;
                }
                bool empty = false;
                if (groupTransforms->ShouldProbeWholePass(
                            *corpus->CollisionTree())) {
                    if (stats != nullptr) {
                        ++stats->probes;
                    }
                    empty = TrySkipWholeTreeBoundsEmpty(
                            *corpus->LocationIso(),
                            *corpus->CollisionTree(),
                            *groupTransforms,
                            stats == nullptr ? nullptr : &stats->recordTests);
                    groupTransforms->ObserveWholePassProbe(
                            *corpus->CollisionTree(), empty);
                }
                if (stats != nullptr) {
                    stats->empty += empty;
                }
                if (empty) {
                    continue;
                }
                u32 nextTemporalSlotOrdinal = 0u;
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree(),
                        nextTemporalSlotOrdinal,
                        *groupTransforms);
            }
        }
    }

    MergeQueuedSphereContacts(collisionBuffer);
}

void CHmsCollisionManager::SZone::
DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
        const GmIso4 &movingIsoRef,
        const CPlugTree &movingTree,
        u32 &nextTemporalSlotOrdinal,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms) {
    CHmsCollisionManagerSZone *zone = this;
    const GmIso4 *movingIso = &movingIsoRef;
    const u32 temporalSlotOrdinal = nextTemporalSlotOrdinal++;
    if (!movingTree.HasWorldBox()) {
        return;
    }

    GmIso4 localIso;
    movingTree.ComposeCollisionIso(*movingIso, localIso);

    const u32 childCount = movingTree.GetChildCount();
    for (u32 childIndex = 0u; childIndex < childCount; ++childIndex) {
        zone->DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
                localIso,
                *movingTree.GetChild(childIndex),
                nextTemporalSlotOrdinal,
                transforms);
    }

    CPlugSurface *movingSurface = movingTree.Surface();
    if (movingSurface == nullptr) {
        return;
    }
    CPlugTree *movingTreeNode = const_cast<CPlugTree *>(&movingTree);

    GmBoxAligned movingBox;
    movingTree.GetTransformedCollisionBox(*movingIso, movingBox);

    GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees =
            zone->activeStaticTargetGroup->staticTrees;

    OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
            temporalCandidates;
    if (transforms.TemporalCandidateSpanFor(
                movingTree,
                temporalSlotOrdinal,
                movingBox,
                &temporalCandidates)) {
        if (temporalCandidates.size != 0u) {
            DetectNativeBinary32CachedTemporalSpan(
                    *zone,
                    movingBox,
                    *movingTreeNode,
                    *movingSurface,
                    localIso,
                    transforms,
                    temporalCandidates);
        }
        return;
    }

    DetectNativeBinary32CachedColdFallback(
            *zone,
            movingBox,
            *movingTreeNode,
            *movingSurface,
            localIso,
            transforms,
            staticTrees);
}
