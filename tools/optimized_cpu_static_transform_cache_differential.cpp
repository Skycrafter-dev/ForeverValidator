#include <array>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <memory>

#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "engine/scene/plug_solid.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"
#include "simulation/runtime/replay_deterministic_execution.h"

struct OptimizedCpuStaticMeshTriangleSidecarTestAccess {
    static bool RejectsSourceMeshMutation(
            OptimizedCpuStaticMeshTriangleSidecar &sidecar,
            OptimizedCpuStaticSurfaceTransformCache &cache,
            const CHmsCollisionManagerSZone &zone) {
        const GmSurfMesh *saved = sidecar.sourceMesh_;
        sidecar.sourceMesh_ = nullptr;
        const bool rejected = !cache.CertifyForAdvance(zone) &&
                !cache.IsCertifiedFor(zone);
        sidecar.sourceMesh_ = saved;
        return rejected && cache.CertifyForAdvance(zone) &&
                cache.IsCertifiedFor(zone);
    }

    static bool RejectsVertexBackingMutation(
            OptimizedCpuStaticMeshTriangleSidecar &sidecar,
            OptimizedCpuStaticSurfaceTransformCache &cache,
            const CHmsCollisionManagerSZone &zone) {
        const GmVec3 *saved = sidecar.sourceVertices_;
        sidecar.sourceVertices_ = nullptr;
        const bool rejected = !cache.CertifyForAdvance(zone) &&
                !cache.IsCertifiedFor(zone);
        sidecar.sourceVertices_ = saved;
        return rejected && cache.CertifyForAdvance(zone) &&
                cache.IsCertifiedFor(zone);
    }

    static bool RejectsTriangleBackingMutation(
            OptimizedCpuStaticMeshTriangleSidecar &sidecar,
            OptimizedCpuStaticSurfaceTransformCache &cache,
            const CHmsCollisionManagerSZone &zone) {
        const GmSurfMeshTriangle *saved = sidecar.sourceTriangles_;
        sidecar.sourceTriangles_ = nullptr;
        const bool rejected = !cache.CertifyForAdvance(zone) &&
                !cache.IsCertifiedFor(zone);
        sidecar.sourceTriangles_ = saved;
        return rejected && cache.CertifyForAdvance(zone) &&
                cache.IsCertifiedFor(zone);
    }

    static bool RejectsCellBackingMutation(
            OptimizedCpuStaticMeshTriangleSidecar &sidecar,
            OptimizedCpuStaticSurfaceTransformCache &cache,
            const CHmsCollisionManagerSZone &zone) {
        const GmMeshOctreeCell *saved = sidecar.sourceCells_;
        sidecar.sourceCells_ = nullptr;
        const bool rejected = !cache.CertifyForAdvance(zone) &&
                !cache.IsCertifiedFor(zone);
        sidecar.sourceCells_ = saved;
        return rejected && cache.CertifyForAdvance(zone) &&
                cache.IsCertifiedFor(zone);
    }
};

namespace {

bool SameBits(const GmIso4 &lhs, const GmIso4 &rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

bool CheckTemporalSpan(
        const OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
                &span,
        const OptimizedCpuStaticSurfaceTransformGroup &group,
        u32 sourceRecordCount) {
    if ((span.size == 0u) != (span.data == nullptr) ||
        group.RecordData() == nullptr || group.InverseData() == nullptr ||
        group.TriangleSidecarData() == nullptr) {
        return false;
    }
    for (std::size_t index = 0u; index < span.size; ++index) {
        const u32 recordIndex = span.data[index];
        if (recordIndex >= sourceRecordCount ||
            (index != 0u && span.data[index - 1u] >= recordIndex) ||
            !group.RecordData()[recordIndex].ContainsSurface() ||
            !SameBits(group.InverseData()[recordIndex],
                      group.InverseAt(recordIndex)) ||
            group.TriangleSidecarData()[recordIndex] !=
                    group.TriangleSidecarAt(recordIndex)) {
            return false;
        }
    }
    return true;
}

bool CheckInverse(const char *caseName,
                  const GmIso4 &cached,
                  const GmIso4 &source) {
    GmIso4 expected;
    expected.SetInverse(source);
    if (SameBits(cached, expected)) {
        return true;
    }
    std::fprintf(stderr, "%s cached inverse differs\n", caseName);
    return false;
}

GmIso4 FirstTransform(void) {
    GmIso4 transform;
    transform.SetIdentity();
    transform.rotation.RotateX(0.37f);
    transform.rotation.RotateY(-0.21f);
    transform.rotation.RotateZ(0.13f);
    transform.translation = {-0.0f, 17.25f, -91.5f};
    return transform;
}

GmIso4 SecondTransform(void) {
    GmIso4 transform;
    transform.SetIdentity();
    transform.rotation.RotateZ(-0.83f);
    transform.rotation.RotateX(0.19f);
    transform.translation = {31.0f, -0.0f, 0.125f};
    return transform;
}

}  // namespace

int main(void) {
    tmnf::simulation::DeterministicExecutionScope deterministicScope;
    if (!deterministicScope.Established()) {
        std::fprintf(stderr, "could not establish deterministic execution\n");
        return 1;
    }

    const GmIso4 firstTransform = FirstTransform();
    const GmIso4 secondTransform = SecondTransform();

    CHmsCollisionManager manager;
    CHmsItem staticItem;
    CHmsCorpus staticCorpus;
    CMwNodRef<CPlugSolid> staticSolid = MakeMwNod<CPlugSolid>();
    CMwNodRef<CPlugSurface> staticSurface = MakeMwNod<CPlugSurface>();
    CMwNodRef<CPlugSurfaceGeom> staticGeometry =
            MakeMwNod<CPlugSurfaceGeom>();
    auto mesh = std::make_unique<GmSurfMesh>();
    mesh->material = GmLocalMaterialIndex::FromIndex(0u);
    GmSurfMeshTriangle triangle{};
    triangle.normal = {0.0f, 0.0f, 1.0f};
    triangle.vertexIndex = {0u, 1u, 2u};
    triangle.material = GmLocalMaterialIndex::FromIndex(0u);
    if (!mesh->SetGeometry(
                {{0.0f, 0.0f, 0.0f},
                 {1.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f}},
                {triangle},
                {},
                GmSurfMesh::PlaneSource::Archived)) {
        std::fprintf(stderr, "could not construct static mesh fixture\n");
        return 1;
    }
    const GmSurfMesh *sourceMesh = mesh.get();
    staticGeometry->SetGmSurf(std::move(mesh));
    staticSurface->SetGeometry(staticGeometry.Get());
    auto staticTree = std::make_unique<CPlugTree>();
    CPlugTree::SFlags treeState;
    treeState.collisionEnabled = true;
    treeState.rooted = false;
    treeState.locationDirty = false;
    staticTree->ApplyLoadedState(treeState);
    staticTree->SetSurface(staticSurface.Get());
    staticTree->SetTreeBounds(staticSurface->GeomBox());
    staticSolid->SetOwnedTree(std::move(staticTree), 0);
    CHmsItem::Properties itemProperties;
    itemProperties.collisionGroup = CHmsItem::ECollisionGroup_Static;
    itemProperties.collisionStatic = true;
    staticItem.SetProperties(itemProperties);
    staticItem.SetSolid(staticSolid.Get());
    staticCorpus.SetItem(&staticItem);
    staticCorpus.SetLocation(firstTransform);

    CHmsCollisionManagerSZone firstZone(17u, &manager);
    CHmsCollisionManagerSZone secondZone(23u, &manager);
    CHmsCollisionManagerSGroup *staticGroup = firstZone.GroupAtOrNull(
            static_cast<u32>(CHmsItem::ECollisionGroup_Static) - 1u);
    staticGroup->AddCorpus(&staticCorpus);
    staticGroup->UpdateStaticCollisionTrees(1);
    if (staticGroup->StaticTreeCount() != 2u) {
        std::fprintf(stderr, "could not construct static cache fixture\n");
        return 1;
    }

    OptimizedCpuStaticSurfaceTransformCache cache;
    if (cache.IsFor(firstZone) || !cache.TryRebuild(firstZone) ||
        !cache.IsFor(firstZone) || cache.IsFor(secondZone) ||
        !cache.CertifyForAdvance(firstZone) ||
        !cache.IsCertifiedFor(firstZone) ||
        cache.IsCertifiedFor(secondZone)) {
        std::fprintf(stderr, "whole-cache zone identity lifecycle differs\n");
        return 1;
    }
    for (u32 groupIndex = 0u;
         groupIndex < CHmsCollisionManager_GroupCount;
         ++groupIndex) {
        const CHmsCollisionManagerSGroup *group =
                firstZone.GroupAtOrNull(groupIndex);
        if (group == nullptr || cache.GroupFor(*group) == nullptr) {
            std::fprintf(stderr, "whole-cache group identity differs\n");
            return 1;
        }
    }
    const OptimizedCpuStaticSurfaceTransformGroup *cachedStaticGroup =
            cache.GroupFor(*staticGroup);
    if (cachedStaticGroup == nullptr ||
        !CheckInverse("initial-build",
                      cachedStaticGroup->InverseAt(1u),
                      firstTransform) ||
        cachedStaticGroup->TriangleSidecarAt(0u) != nullptr ||
        cachedStaticGroup->TriangleSidecarAt(1u) == nullptr ||
        !cachedStaticGroup->TriangleSidecarAt(1u)->IsFor(*sourceMesh)) {
        std::fprintf(stderr, "initial static triangle sidecar differs\n");
        return 1;
    }
    staticGroup->ClearAllStatic();
    if (cache.GroupFor(*staticGroup) != nullptr ||
        cache.CertifyForAdvance(firstZone) ||
        cache.IsCertifiedFor(firstZone)) {
        std::fprintf(stderr, "changed backing count retained cached group\n");
        return 1;
    }

    staticCorpus.SetLocation(secondTransform);
    staticGroup->UpdateStaticCollisionTrees(1);
    if (!cache.TryRebuild(firstZone)) {
        std::fprintf(stderr, "whole-cache rebuild failed\n");
        return 1;
    }
    cachedStaticGroup = cache.GroupFor(*staticGroup);
    if (cachedStaticGroup == nullptr ||
        !CheckInverse("rebuilt",
                      cachedStaticGroup->InverseAt(1u),
                      secondTransform) ||
        cachedStaticGroup->TriangleSidecarAt(1u) == nullptr ||
        !cachedStaticGroup->TriangleSidecarAt(1u)->IsFor(*sourceMesh)) {
        std::fprintf(stderr, "rebuilt static triangle sidecar differs\n");
        return 1;
    }
    auto *mutableSidecar =
            const_cast<OptimizedCpuStaticMeshTriangleSidecar *>(
                    cachedStaticGroup->TriangleSidecarAt(1u));
    if (mutableSidecar == nullptr ||
        !cache.CertifyForAdvance(firstZone) ||
        !OptimizedCpuStaticMeshTriangleSidecarTestAccess::
                RejectsSourceMeshMutation(
                        *mutableSidecar, cache, firstZone) ||
        !OptimizedCpuStaticMeshTriangleSidecarTestAccess::
                RejectsVertexBackingMutation(
                        *mutableSidecar, cache, firstZone) ||
        !OptimizedCpuStaticMeshTriangleSidecarTestAccess::
                RejectsTriangleBackingMutation(
                        *mutableSidecar, cache, firstZone) ||
        !OptimizedCpuStaticMeshTriangleSidecarTestAccess::
                RejectsCellBackingMutation(
                        *mutableSidecar, cache, firstZone)) {
        std::fprintf(stderr, "advance sidecar certificate differs\n");
        return 1;
    }

    GmBoxAligned nearBounds;
    nearBounds.SetMult(staticSurface->GeomBox(), secondTransform);
    OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
            temporalSpan;
    CPlugTree *movingTree = staticSolid->CollisionTree();
    constexpr u32 MovingTreeTemporalSlotOrdinal = 4u;
    if (movingTree == nullptr) {
        std::fprintf(stderr, "missing static tree certificate fixture\n");
        return 1;
    }
    movingTree->SetCollisionEnabled(false);
    if (cache.CertifyForAdvance(firstZone) ||
        cache.IsCertifiedFor(firstZone)) {
        std::fprintf(stderr,
                     "disabled static tree retained advance certificate\n");
        return 1;
    }
    movingTree->SetCollisionEnabled(true);
    if (!cache.CertifyForAdvance(firstZone) ||
        !cache.IsCertifiedFor(firstZone)) {
        std::fprintf(stderr,
                     "restored static tree lost advance certificate\n");
        return 1;
    }

    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_DIVBYZERO);
    const int certificateExceptions =
            std::fetestexcept(FE_ALL_EXCEPT);
    const int certificateRounding = std::fegetround();
    if (!cache.CertifyForAdvance(firstZone) ||
        std::fetestexcept(FE_ALL_EXCEPT) != certificateExceptions ||
        std::fegetround() != certificateRounding) {
        std::fprintf(stderr, "advance certificate changed fenv state\n");
        return 1;
    }
    std::feclearexcept(FE_ALL_EXCEPT);

    if (!cachedStaticGroup->TemporalCandidateSpanFor(
                *movingTree,
                MovingTreeTemporalSlotOrdinal,
                nearBounds,
                &temporalSpan) ||
        temporalSpan.size != 1u || temporalSpan.data == nullptr ||
        temporalSpan.data[0] != 1u ||
        !CheckTemporalSpan(
                temporalSpan,
                *cachedStaticGroup,
                staticGroup->StaticTreeCount())) {
        std::fprintf(stderr, "initial temporal candidate span differs\n");
        return 1;
    }

    GmBoxAligned containedBounds = nearBounds;
    containedBounds.center.x += 0.25f;
    if (!cachedStaticGroup->TemporalCandidateSpanFor(
                *movingTree,
                MovingTreeTemporalSlotOrdinal,
                containedBounds,
                &temporalSpan) ||
        temporalSpan.size != 1u || temporalSpan.data == nullptr ||
        temporalSpan.data[0] != 1u) {
        std::fprintf(stderr, "contained temporal candidate span differs\n");
        return 1;
    }

    CPlugTree preorderRoot;
    auto preorderFirstChild = std::make_unique<CPlugTree>();
    CPlugTree *preorderFirstChildPtr = preorderFirstChild.get();
    auto preorderGrandchild = std::make_unique<CPlugTree>();
    CPlugTree *preorderGrandchildPtr = preorderGrandchild.get();
    preorderFirstChild->AddOwnedChild(std::move(preorderGrandchild));
    auto preorderSecondChild = std::make_unique<CPlugTree>();
    CPlugTree *preorderSecondChildPtr = preorderSecondChild.get();
    preorderRoot.AddOwnedChild(std::move(preorderFirstChild));
    preorderRoot.AddOwnedChild(std::move(preorderSecondChild));

    std::array<const CPlugTree *, 4u> preorderTrees{};
    u32 preorderTreeCount = 0u;
    const auto collectPreorder =
            [&](const auto &self, const CPlugTree &tree) -> void {
        preorderTrees[preorderTreeCount++] = &tree;
        for (u32 childIndex = 0u;
             childIndex < tree.GetChildCount();
             ++childIndex) {
            self(self, *tree.GetChild(childIndex));
        }
    };
    collectPreorder(collectPreorder, preorderRoot);
    const std::array<const CPlugTree *, 4u> expectedPreorder = {
        &preorderRoot,
        preorderFirstChildPtr,
        preorderGrandchildPtr,
        preorderSecondChildPtr,
    };
    if (preorderTreeCount != expectedPreorder.size() ||
        preorderTrees != expectedPreorder) {
        std::fprintf(stderr, "preorder temporal slot schedule differs\n");
        return 1;
    }
    for (u32 ordinal = 0u; ordinal < preorderTreeCount; ++ordinal) {
        if (!cachedStaticGroup->TemporalCandidateSpanFor(
                    *preorderTrees[ordinal],
                    ordinal,
                    nearBounds,
                    &temporalSpan) ||
            temporalSpan.size != 1u || temporalSpan.data == nullptr ||
            temporalSpan.data[0] != 1u) {
            std::fprintf(stderr, "preorder temporal candidate span differs\n");
            return 1;
        }
    }

    CPlugTree ordinalTagCollisionTree;
    if (!cachedStaticGroup->TemporalCandidateSpanFor(
                ordinalTagCollisionTree,
                MovingTreeTemporalSlotOrdinal,
                nearBounds,
                &temporalSpan) ||
        temporalSpan.size != 1u || temporalSpan.data == nullptr ||
        temporalSpan.data[0] != 1u) {
        std::fprintf(stderr, "ordinal tag collision fallback differs\n");
        return 1;
    }

    GmBoxAligned farBounds = nearBounds;
    farBounds.center.x += 10000.0f;
    if (!cachedStaticGroup->TemporalCandidateSpanFor(
                *movingTree,
                MovingTreeTemporalSlotOrdinal,
                farBounds,
                &temporalSpan) ||
        temporalSpan.size != 0u || temporalSpan.data != nullptr ||
        !CheckTemporalSpan(
                temporalSpan,
                *cachedStaticGroup,
                staticGroup->StaticTreeCount())) {
        std::fprintf(stderr, "rebuilt empty temporal span differs\n");
        return 1;
    }

    cache.ClearTemporalCandidates();
    if (cache.IsCertifiedFor(firstZone) ||
        !cache.CertifyForAdvance(firstZone) ||
        !cachedStaticGroup->TemporalCandidateSpanFor(
                *movingTree,
                MovingTreeTemporalSlotOrdinal,
                nearBounds,
                &temporalSpan) ||
        temporalSpan.size != 1u || temporalSpan.data == nullptr ||
        temporalSpan.data[0] != 1u) {
        std::fprintf(stderr, "invalidated temporal candidate span differs\n");
        return 1;
    }

    auto replacementMesh = std::make_unique<GmSurfMesh>();
    replacementMesh->material = GmLocalMaterialIndex::FromIndex(0u);
    if (!replacementMesh->SetGeometry(
                {{0.0f, 0.0f, 0.0f},
                 {1.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f}},
                {triangle},
                {},
                GmSurfMesh::PlaneSource::Archived)) {
        std::fprintf(stderr, "could not construct replacement mesh\n");
        return 1;
    }
    staticGeometry->SetGmSurf(std::move(replacementMesh));
    if (cache.CertifyForAdvance(firstZone) ||
        cache.IsCertifiedFor(firstZone)) {
        std::fprintf(stderr,
                     "changed record geometry retained advance certificate\n");
        return 1;
    }

    cache.Clear();
    if (cache.IsFor(firstZone) || cache.IsCertifiedFor(firstZone) ||
        cache.GroupFor(*firstZone.GroupAtOrNull(0u)) != nullptr) {
        std::fprintf(stderr, "cleared whole cache retained source identity\n");
        return 1;
    }

    std::printf("static_transform_cache_cases=31 result=identical\n");
    return 0;
}
