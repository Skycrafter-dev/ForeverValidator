#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "engine/physics/collision/collision_pair_schedule.h"
#include "engine/core/engine_types.h"
#include "engine/physics/geometry/gm_octree.h"
#include "engine/core/gm_types.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/collision/hms_collision_types.h"
#include "engine/scene/scene_vehicle_water_zone.h"
struct CHmsCorpus;
struct CHmsZoneDynamic;
struct CPlugSurface;
struct CPlugTree;
class OptimizedCpuStaticSurfaceTransformCache;
class OptimizedCpuStaticSurfaceTransformGroup;

struct CHmsCollisionManager {
    struct SZone;

    struct SColOctreeCell {
        struct StaticSurface {
            GmIso4 location{};
            CPlugSurface *surface = nullptr;
            CPlugTree *tree = nullptr;
            CHmsCorpus *corpus = nullptr;
        };

        static SColOctreeCell Branch(const GmBoxAligned &bounds);
        static SColOctreeCell Surface(const GmBoxAligned &bounds,
                                      const GmIso4 &location,
                                      CPlugSurface &surface,
                                      CPlugTree &tree,
                                      CHmsCorpus &corpus);

        const GmBoxAligned &Bounds(void) const { return bounds_; }
        u32 SubtreeEntryCount(void) const { return subtreeEntryCount_; }
        void SetBounds(const GmBoxAligned &bounds) { bounds_ = bounds; }
        void SetSubtreeEntryCount(u32 count) { subtreeEntryCount_ = count; }
        bool ContainsSurface(void) const { return staticSurface_.has_value(); }
        const StaticSurface &SurfaceData(void) const { return *staticSurface_; }

    private:
        u32 subtreeEntryCount_ = 1u;
        GmBoxAligned bounds_{};
        std::optional<StaticSurface> staticSurface_;
    };

    struct SGroup {
        struct SAgainstGroup {
            SGroup *targetGroup = nullptr;
            const CHmsItem::SHmsCollisionGroupPair *collisionGroupPair = nullptr;
            CollisionPairSchedule collisionSchedule;

            SAgainstGroup(void);
            ~SAgainstGroup(void);
            void Reset(void);
            SGroup *TargetGroup(void) const;
            const CHmsItem::SHmsCollisionGroupPair *CollisionGroupPair(void) const;
            int TargetsGroup(const SGroup *group) const;
            u32 TargetGroupNonStaticCorpusCount(void) const;
            int HasStaticTargetTrees(void) const;
        };

        SGroup(void);
        ~SGroup(void);
        void Reset(void);
        SAgainstGroup &AgainstGroupAt(u32 index);
        const SAgainstGroup &AgainstGroupAt(u32 index) const;
        SAgainstGroup *FindAgainstGroupTargeting(const SGroup *target);
        const SAgainstGroup *FindAgainstGroupTargeting(const SGroup *target) const;
        void RegisterCollisionPairsFor(CHmsCorpus &corpus);
        void UnregisterCollisionPairsFor(CHmsCorpus &corpus);
        u32 AllCorpusCount(void) const;
        u32 NonStaticCorpusCount(void) const;
        u32 AgainstGroupCount(void) const;
        u32 StaticTreeCount(void) const;
        int HasAgainstGroups(void) const;
        bool IsDisabled(void) const { return disabled; }
        CHmsCorpus *NonStaticCorpusAt(u32 index) const;
        void ComputeNonStaticCorpusInfos(void);
        void AddNonStaticCorpus(CHmsCorpus *corpus);
        void AddCorpus(CHmsCorpus *corpus);
        void ClearAllStatic(void);
        void AddStaticSurfacesFromTree(
                CHmsCorpus *corpus,
                CPlugTree *tree,
                const GmIso4 &parentIso,
                std::vector<SColOctreeCell> &staticSurfaceRecords);
        void RemoveNonStaticCorpus(CHmsCorpus *corpus);
        void RemoveCorpus(CHmsCorpus *corpus);
        void UpdateStaticCollisionTrees(int mode);
        void ComputeIsToPerformCollisions(void);

    private:
        friend struct SZone;
        friend class OptimizedCpuStaticSurfaceTransformCache;

        std::vector<CHmsCorpus *> allCorpuses;
        struct MovingCorpusState {
            CHmsCorpus *corpus = nullptr;
            float linearSpeedSquared = 0.0f;
        };

        std::vector<MovingCorpusState> movingCorpuses;
        std::vector<SAgainstGroup> againstGroups;
        GmOctree<SColOctreeCell> staticTrees;
        u32 groupIndex = 0u;
        bool disabled = false;
    };

    struct SZone {
        struct SPlugTreeLocatedPair {
            CPlugTree *treeA = nullptr;
            const GmIso4 *isoA = nullptr;
            CPlugTree *treeB = nullptr;
            const GmIso4 *isoB = nullptr;
        };
        struct TreeSphereContact {
            CPlugTree *tree = nullptr;
            std::unique_ptr<SHmsSphereBufferContact> contact;
        };

        CSceneVehicleWaterZone waterZone;
        SZone(unsigned long zoneId, CHmsCollisionManager *manager);
        ~SZone(void);
        void Reset(void);
        void ConfigureWater(const WaterOccupancyGrid &occupancy,
                            float surfaceHeight,
                            float secondaryCullHeight);
        void InitGroupHeaders(void);
        u32 AppendDefaultAgainstGroups(void);
        SGroup *GroupAtOrNull(u32 groupIndex);
        const SGroup *GroupAtOrNull(u32 groupIndex) const;
        SGroup *GroupForReplayStaticRole(
                CHmsReplayStaticCollisionGroupRole role);
        const SGroup *GroupForReplayStaticRole(
                CHmsReplayStaticCollisionGroupRole role) const;
        void AddCorpus(CHmsCorpus *corpus);
        void RemoveCorpus(CHmsCorpus *corpus);
        void UpdateStaticCollisionTrees(void);
        void PrepareCollisions(void);
        int ComputeCollisionTree1RootOnly(
                const SPlugTreeLocatedPair &pair,
                const GmBoxAligned &boxA);
        int ComputeCollisionTree2RootOnly(
                const SPlugTreeLocatedPair &pair,
                const GmBoxAligned &boxB);
        int ComputeCollision(
                const SPlugTreeLocatedPair &pair);
        SHmsSphereBufferContact *EnsureTreeSphereContact(CPlugTree *tree);
        void AddSphereContactOnce(SHmsSphereBufferContact *sphereContact);
        void MergeQueuedSphereContacts(CHmsCollisionBuffer &collisionBuffer);
        void BeginReplayStaticCollisionPass(
                CHmsCollisionBuffer *collisionBuffer,
                CHmsCorpus *sourceCorpus);
        void SelectReplayStaticCollisionTarget(
                const SGroup::SAgainstGroup &against);
        CHmsCollisionBuffer *ChooseCollisionOutputBuffer(CPlugTree *treeA,
                                                         CPlugSurface *surfaceA,
                                                         SHmsSphereBufferContact **sphereContactOut);
        void TagNewCollisions(CHmsCollisionBuffer *buffer,
                              u32 firstNew,
                              CPlugTree *treeA,
                              CPlugTree *treeB);
        void TagNewStaticCollisions(CHmsCollisionBuffer *buffer,
                                    u32 firstNew,
                                    CPlugTree *movingTree,
                                    const SColOctreeCell *record);
        int ComputeSurfaceCollisionAndTag(CPlugTree *treeA,
                                          const GmIso4 *isoA,
                                          CPlugSurface *surfaceA,
                                          CPlugTree *treeB,
                                          const GmIso4 *isoB,
                                          CPlugSurface *surfaceB);
        void DetectCollisionBetweenTreeAndStaticCollisionTree(
                const GmIso4 &movingIso,
                const CPlugTree &movingTree);
        void DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpu(
                const GmIso4 &movingIso,
                const CPlugTree &movingTree);
        void DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32(
                const GmIso4 &movingIso,
                const CPlugTree &movingTree);
        void DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
                const GmIso4 &movingIso,
                const CPlugTree &movingTree,
                u32 &nextTemporalSlotOrdinal,
                const OptimizedCpuStaticSurfaceTransformGroup &transforms);
        void DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
                const GmIso4 &movingIso,
                const CPlugTree &movingTree,
                u32 &nextTemporalSlotOrdinal,
                const OptimizedCpuStaticSurfaceTransformGroup &transforms);
        void DetectCollisionBetween(CHmsCorpus *a, CHmsCorpus *b);
        void DetectCollisionsCorpus(
                CHmsCollisionBuffer &collisionBuffer,
                CHmsCorpus *corpus);
        void DetectCollisionsCorpusOptimizedCpu(
                CHmsCollisionBuffer &collisionBuffer,
                CHmsCorpus *corpus);
        void DetectCollisionsCorpusOptimizedCpuNativeBinary32(
                CHmsCollisionBuffer &collisionBuffer,
                CHmsCorpus *corpus);
        void DetectCollisionsCorpusOptimizedCpuCached(
                CHmsCollisionBuffer &collisionBuffer,
                CHmsCorpus *corpus,
                const OptimizedCpuStaticSurfaceTransformCache &transforms);
        void DetectCollisionsCorpusOptimizedCpuNativeBinary32Cached(
                CHmsCollisionBuffer &collisionBuffer,
                CHmsCorpus *corpus,
                const OptimizedCpuStaticSurfaceTransformCache &transforms);

    private:
        friend struct CHmsZoneDynamic;

        std::array<SGroup, CHmsCollisionManager_GroupCount> groups;
        const CHmsItem::SHmsCollisionGroupPair *activeCollisionGroupPair = nullptr;
        CHmsCorpus *activeCorpusA = nullptr;
        CHmsCorpus *activeCorpusB = nullptr;
        CHmsCollisionBuffer *activeCollisionBuffer = nullptr;
        SGroup *activeStaticTargetGroup = nullptr;
        u32 zoneId = 0u;
        CHmsCollisionManager *manager = nullptr;
        std::vector<CHmsCorpus *> registeredCorpuses;
        u32 nextCorpusRegistrationOrder = 0u;
        std::vector<TreeSphereContact> ownedSphereContacts;
        std::vector<SHmsSphereBufferContact *> sphereBufferContacts;
        std::vector<CHmsZoneDynamic *> boundDynamicZones;

        void AttachDynamicZone(CHmsZoneDynamic &zone);
        void DetachDynamicZone(CHmsZoneDynamic &zone);
        void DetachBoundDynamicZones(void);
        void DetachRegisteredCorpuses(void);
    };
};


using CHmsCollisionManagerSColOctreeCell = CHmsCollisionManager::SColOctreeCell;
using CHmsCollisionManagerSGroup = CHmsCollisionManager::SGroup;
using CHmsCollisionManagerSZone = CHmsCollisionManager::SZone;
using CHmsCollisionManagerSAgainstGroup = CHmsCollisionManager::SGroup::SAgainstGroup;
