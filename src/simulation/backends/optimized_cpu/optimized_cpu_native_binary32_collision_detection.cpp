// OptimizedCpu-only static collision traversal for native binary32 kernels.

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"

void CHmsCollisionManager::SZone::
DetectCollisionsCorpusOptimizedCpuNativeBinary32(
        CHmsCollisionBuffer &collisionBuffer,
        CHmsCorpus *corpus) {
    activeCollisionBuffer = &collisionBuffer;

    u32 groupIndex = corpus->Item()->CollisionGroup();
    CHmsCollisionManagerSGroup *group = &groups[groupIndex - 1];

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
            DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32(
                    *corpus->LocationIso(),
                    *corpus->CollisionTree());
        }
    }

    MergeQueuedSphereContacts(collisionBuffer);
}

void CHmsCollisionManager::SZone::
DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32(
        const GmIso4 &movingIsoRef,
        const CPlugTree &movingTree) {
    CHmsCollisionManagerSZone *zone = this;
    const GmIso4 *movingIso = &movingIsoRef;
    if (!movingTree.HasWorldBox()) {
        return;
    }

    GmIso4 localIso;
    movingTree.ComposeCollisionIso(*movingIso, localIso);

    u32 childCount = movingTree.GetChildCount();
    for (u32 childIndex = 0; childIndex < childCount; childIndex++) {
        zone->DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32(
                localIso,
                *movingTree.GetChild(childIndex));
    }

    CPlugSurface *movingSurface = movingTree.Surface();
    if (movingSurface == 0) {
        return;
    }
    CPlugTree *movingTreeNode = const_cast<CPlugTree *>(&movingTree);

    GmBoxAligned movingBox;
    movingTree.GetTransformedCollisionBox(*movingIso, movingBox);

    GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees =
            zone->activeStaticTargetGroup->staticTrees;
    u32 staticTreeCount = staticTrees.GetCount();
    for (u32 staticTreeIndex = 0;
         staticTreeIndex < staticTreeCount;) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!movingBox.TestInter(record->Bounds())) {
            staticTreeIndex += record->SubtreeEntryCount();
            continue;
        }

        if (record->ContainsSurface() &&
            record->SurfaceData().tree->AllowsSurfaceCollision()) {
            const SColOctreeCell::StaticSurface &staticSurface =
                    record->SurfaceData();
            SHmsSphereBufferContact *sphereContact = 0;
            CHmsCollisionBuffer *buffer =
                    zone->ChooseCollisionOutputBuffer(
                            movingTreeNode,
                            movingSurface,
                            &sphereContact);
            u32 firstNew = buffer->PhysicalCollisionCount();
            SPlugSurfaceLocatedPair surfacePair = {
                *movingSurface,
                localIso,
                *staticSurface.surface,
                staticSurface.location,
            };

            const int surfaceCollisionResult =
                    ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32(
                            surfacePair, *buffer);
            if (surfaceCollisionResult) {
                if (sphereContact != 0) {
                    zone->AddSphereContactOnce(sphereContact);
                }
                zone->TagNewStaticCollisions(
                        buffer, firstNew, movingTreeNode, record);
            }
        }

        staticTreeIndex++;
    }
}
