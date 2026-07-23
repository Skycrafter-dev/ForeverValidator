#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "engine/physics/collision/hms_collision_manager.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_mesh_triangle_sidecar.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_scene_fingerprint.h"

class OptimizedCpuStaticSurfaceTransformGroup {
public:
    struct TemporalCandidateSpan {
        const u32 *data = nullptr;
        std::size_t size = 0u;
    };

    const CHmsCollisionManagerSColOctreeCell *RecordData(void) const noexcept {
        return sourceRecords_;
    }

    const GmIso4 *InverseData(void) const noexcept {
        return inverses_.data();
    }

    const GmIso4 &InverseAt(u32 staticTreeIndex) const noexcept {
        return inverses_[staticTreeIndex];
    }

    const OptimizedCpuStaticMeshTriangleSidecar *const
            *TriangleSidecarData(void) const noexcept {
        return triangleSidecars_.data();
    }

    const OptimizedCpuStaticMeshTriangleSidecar *TriangleSidecarAt(
            u32 staticTreeIndex) const noexcept {
        return triangleSidecars_[staticTreeIndex];
    }

    bool TemporalCandidateSpanFor(
            const CPlugTree &movingTree,
            u32 temporalSlotOrdinal,
            const GmBoxAligned &movingBounds,
            TemporalCandidateSpan *result) const noexcept;
    bool WholeTreeBoundsOverlapAnySurface(
            const GmBoxAligned &movingBounds,
            std::uint64_t *recordTests = nullptr) const noexcept;
    bool BroadPhaseArithmeticIsBoundedFor(
            const CPlugTree &movingTree,
            const GmIso4 &movingIso) const noexcept;
    bool ShouldProbeWholePass(const CPlugTree &movingTree) const noexcept;
    void ObserveWholePassProbe(
            const CPlugTree &movingTree,
            bool empty) const noexcept;
    void ClearTemporalCandidates(void) const noexcept;

private:
    friend class OptimizedCpuStaticSurfaceTransformCache;

    struct TemporalCandidateEntry {
        const CPlugTree *movingTree = nullptr;
        GmBoxAligned validityBounds{};
        std::vector<u32> candidateRecordIndices;
    };

    TemporalCandidateEntry *TemporalCandidateFallbackFor(
            const CPlugTree &movingTree,
            bool *entryTagMatches) const noexcept;

    const CHmsCollisionManagerSGroup *sourceGroup_ = nullptr;
    const CHmsCollisionManagerSColOctreeCell *sourceRecords_ = nullptr;
    std::size_t sourceRecordCount_ = 0u;
    std::vector<GmIso4> inverses_;
    std::vector<const OptimizedCpuStaticMeshTriangleSidecar *>
            triangleSidecars_;
    bool staticBroadPhaseArithmeticIsBounded_ = false;
    mutable std::array<const CPlugTree *, 8u> boundedMovingTrees_{};
    mutable std::size_t boundedMovingTreeCount_ = 0u;
    struct WholePassPredictorEntry {
        const CPlugTree *movingTree = nullptr;
        std::uint8_t probesUntilReacquire = 0u;
        bool predictedEmpty = true;
    };
    mutable std::array<WholePassPredictorEntry, 8u> wholePassPredictors_{};
    mutable std::array<TemporalCandidateEntry, 64u>
            ordinalTemporalCandidates_{};
    mutable std::vector<TemporalCandidateEntry> temporalCandidates_;
    mutable std::uint64_t temporalQueryCount_ = 0u;
    mutable std::uint64_t temporalHitCount_ = 0u;
    mutable std::uint64_t temporalRebuildCount_ = 0u;
    mutable std::uint64_t temporalAuthoritativeRecordTests_ = 0u;
    mutable std::uint64_t temporalBaselineRecordTests_ = 0u;
    mutable std::uint64_t temporalCandidateRecordTests_ = 0u;
};

class OptimizedCpuStaticSurfaceTransformCache {
public:
    OptimizedCpuStaticSurfaceTransformCache(void) = default;
    OptimizedCpuStaticSurfaceTransformCache(
            OptimizedCpuStaticSurfaceTransformCache &&) = default;
    OptimizedCpuStaticSurfaceTransformCache &operator=(
            OptimizedCpuStaticSurfaceTransformCache &&) = default;
    ~OptimizedCpuStaticSurfaceTransformCache(void);

    OptimizedCpuStaticSurfaceTransformCache(
            const OptimizedCpuStaticSurfaceTransformCache &) = delete;
    OptimizedCpuStaticSurfaceTransformCache &operator=(
            const OptimizedCpuStaticSurfaceTransformCache &) = delete;

    bool TryRebuild(
            const CHmsCollisionManagerSZone &zone) noexcept;
    void Clear(void) noexcept;
    void ClearTemporalCandidates(void) noexcept;
    bool CertifyForAdvance(
            const CHmsCollisionManagerSZone &zone) noexcept;
    bool IsFor(const CHmsCollisionManagerSZone &zone) const noexcept;
    bool IsCertifiedFor(
            const CHmsCollisionManagerSZone &zone) const noexcept;
    const OptimizedCpuStaticSurfaceTransformGroup *GroupFor(
            const CHmsCollisionManagerSGroup &group) const noexcept;
    std::optional<OptimizedCpuStaticSceneFingerprint>
            CaptureSourceFingerprintForTesting(void) const noexcept;

private:
    const CHmsCollisionManagerSZone *sourceZone_ = nullptr;
    const CHmsCollisionManagerSZone *certifiedZone_ = nullptr;
    std::array<OptimizedCpuStaticSurfaceTransformGroup,
               CHmsCollisionManager_GroupCount> groups_{};
    std::vector<std::unique_ptr<OptimizedCpuStaticMeshTriangleSidecar>>
            triangleSidecars_;
    std::vector<const GmSurfMesh *> unavailableTriangleSidecarMeshes_;
};
