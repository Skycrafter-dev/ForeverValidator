// Collision-zone ownership, registration, and scheduling lifecycle.

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <memory>
#include <limits>
#include <utility>

#include "engine/core/cmw_nod.h"
#include "engine/physics/geometry/geometry_helpers.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/world/hms_zone.h"
#include "engine/physics/geometry/physics_tolerances.h"
#include "engine/scene/plug_solid.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"

namespace {

bool IntersectSegmentTree3(
        CPlugTree *tree, const GmIso4 &parent,
        const GmVec3 &start, const GmVec3 &delta,
        float &ratio, EPlugSurfaceMaterialId &material) {
    if (tree == nullptr || !tree->HasWorldBox()) return false;
    GmBoxAligned bounds;
    bounds.SetMult(tree->Box(), parent);
    if (!bounds.TestInterSegment(start, start.AddForCollision(delta))) return false;
    GmIso4 location;
    tree->ComposeCollisionIso(parent, location);
    bool hit = false;
    if (const CPlugSurface *surface = tree->Surface()) {
        GmLocalMaterialIndex localMaterial;
        if (GmSurf::ClipSegment3({surface->Geometry(), &location},
                                start, delta, ratio, localMaterial)) {
            material = surface->SurfaceMaterialIdFromLocalIndex(localMaterial);
            hit = true;
        }
    }
    for (u32 index = 0u; index < tree->GetChildCount(); ++index) {
        float childRatio;
        EPlugSurfaceMaterialId childMaterial;
        if (IntersectSegmentTree3(tree->GetChild(index), location, start, delta,
                                  childRatio, childMaterial) &&
            (!hit || ratio > childRatio)) {
            ratio = childRatio;
            material = childMaterial;
            hit = true;
        }
    }
    return hit;
}

} // namespace

bool CHmsCollisionManager::SZone::IntersectSegment3(
        CHmsItem::ECollisionGroup groupId,
        const GmVec3 &start, const GmVec3 &delta,
        float &ratio, EPlugSurfaceMaterialId &material) const {
    const u32 index = static_cast<u32>(groupId);
    ratio = std::numeric_limits<float>::max();
    if (index == 0u || index > CHmsCollisionManager_GroupCount) return false;
    const SGroup &group = groups[index - 1u];
    const GmVec3 end = start.AddForCollision(delta);
    bool hit = false;
    for (u32 cellIndex = 0u; cellIndex < group.staticTrees.GetCount();) {
        const auto &cell = group.staticTrees[cellIndex];
        if (!cell.Bounds().TestInterSegment(start, end)) {
            cellIndex += cell.SubtreeEntryCount();
            continue;
        }
        ++cellIndex;
        if (!cell.ContainsSurface()) continue;
        const auto &record = cell.SurfaceData();
        float candidate;
        GmLocalMaterialIndex localMaterial;
        if (GmSurf::ClipSegment3({record.surface->Geometry(), &record.location},
                                start, delta, candidate, localMaterial) &&
            ratio > candidate) {
            ratio = candidate;
            material = record.surface->SurfaceMaterialIdFromLocalIndex(localMaterial);
            hit = true;
        }
    }
    for (u32 corpusIndex = 0u; corpusIndex < group.NonStaticCorpusCount(); ++corpusIndex) {
        const CHmsCorpus *corpus = group.NonStaticCorpusAt(corpusIndex);
        const GmIso4 *location = corpus->LocationIso();
        float candidate;
        EPlugSurfaceMaterialId candidateMaterial;
        if (location != nullptr &&
            IntersectSegmentTree3(corpus->CollisionTree(), *location,
                                  start, delta, candidate, candidateMaterial) &&
            ratio > candidate) {
            ratio = candidate;
            material = candidateMaterial;
            hit = true;
        }
    }
    return hit;
}

CSceneVehicleWaterZone::CSceneVehicleWaterZone(void)
        : enabled(false),
          waterMap(),
          surfaceHeight(-1000000.0f),
          secondaryCullHeight(1000000.0f) {}

void CSceneVehicleWaterZone::Disable(void) {
    enabled = false;
    waterMap = GmMap2<unsigned char>();
    surfaceHeight = -1000000.0f;
    secondaryCullHeight = 1000000.0f;
}

void CSceneVehicleWaterZone::Configure(
        const WaterOccupancyGrid &occupancy,
        float surfaceHeight,
        float secondaryCullHeight) {
    const bool declaresPlaneMap =
            !occupancy.planeIndices.empty() ||
            !occupancy.planes.empty() ||
            occupancy.outsidePlaneIndex != 0u;
    const bool hasPlaneMap =
            declaresPlaneMap &&
            occupancy.planeIndices.size() == occupancy.cells.size() &&
            occupancy.outsidePlaneIndex <= occupancy.planes.size() &&
            std::all_of(
                    occupancy.planeIndices.begin(),
                    occupancy.planeIndices.end(),
                    [&occupancy](std::uint8_t index) {
                        return index <= occupancy.planes.size();
                    });
    if (declaresPlaneMap && !hasPlaneMap) {
        Disable();
        return;
    }
    const unsigned char outside = hasPlaneMap
            ? occupancy.outsidePlaneIndex
            : static_cast<unsigned char>(occupancy.outside);
    waterMap.Init(occupancy.cellSize,
                  occupancy.origin,
                  occupancy.dimensions,
                  outside);
    for (u32 y = 0u; y < occupancy.dimensions.y; ++y) {
        for (u32 x = 0u; x < occupancy.dimensions.x; ++x) {
            const std::size_t index =
                    static_cast<std::size_t>(y) * occupancy.dimensions.x + x;
            if (index >= occupancy.cells.size()) {
                break;
            }
            const unsigned char value = hasPlaneMap
                    ? occupancy.planeIndices[index]
                    : static_cast<unsigned char>(occupancy.cells[index]);
            waterMap.SetValue({x, y}, value);
        }
    }
    this->surfaceHeight = surfaceHeight;
    this->secondaryCullHeight = secondaryCullHeight;
    enabled = true;
}

bool CSceneVehicleWaterZone::AcceptsRegion(
        const GmVec2 &sample,
        float lowerY,
        float upperY) const {
    if (!enabled) {
        return false;
    }

    const bool isInside = waterMap.IsInside(sample);
    const unsigned char water =
            static_cast<unsigned char>(WaterOccupancy::Water);
    if (!isInside && waterMap.OutsideValue() == water &&
        surfaceHeight > lowerY) {
        return true;
    }
    if (!(upperY > secondaryCullHeight) || !(surfaceHeight > lowerY)) {
        return false;
    }
    return waterMap.GetValue(sample) == water;
}

void CHmsCollisionManager::SZone::InitGroupHeaders(void) {
    for (u32 groupIndex = 0;
         groupIndex < CHmsCollisionManager_GroupCount;
         groupIndex++) {
        this->groups[groupIndex].groupIndex = groupIndex;
        this->groups[groupIndex].disabled = groupIndex == 3u;
    }
}


void CHmsCollisionManager::SZone::AttachDynamicZone(
        CHmsZoneDynamic &zone) {
    if (std::find(boundDynamicZones.begin(), boundDynamicZones.end(), &zone) ==
        boundDynamicZones.end()) {
        boundDynamicZones.push_back(&zone);
    }
    zone.BindCollisionManager(*this);
}

void CHmsCollisionManager::SZone::DetachDynamicZone(
        CHmsZoneDynamic &zone) {
    const auto boundZone = std::find(
            boundDynamicZones.begin(), boundDynamicZones.end(), &zone);
    if (boundZone != boundDynamicZones.end()) {
        *boundZone = boundDynamicZones.back();
        boundDynamicZones.pop_back();
    }
    zone.UnbindCollisionManager(*this);
}

void CHmsCollisionManager::SZone::DetachBoundDynamicZones(void) {
    for (CHmsZoneDynamic *zone : boundDynamicZones) {
        if (zone != nullptr) {
            zone->UnbindCollisionManager(*this);
        }
    }
    boundDynamicZones.clear();
}

void CHmsCollisionManager::SZone::DetachRegisteredCorpuses(void) {
    for (CHmsCorpus *corpus : registeredCorpuses) {
        if (corpus != nullptr) {
            corpus->ClearCollisionManagerRegistration(*this);
        }
    }
    registeredCorpuses.clear();
}

void CHmsCollisionManager::SZone::Reset(void) {
    DetachRegisteredCorpuses();
    DetachBoundDynamicZones();
    for (SGroup &group : groups) {
        group.Reset();
    }
    waterZone.Disable();
    activeCollisionGroupPair = nullptr;
    activeCorpusA = nullptr;
    activeCorpusB = nullptr;
    activeCollisionBuffer = nullptr;
    activeStaticTargetGroup = nullptr;
    zoneId = 0u;
    manager = nullptr;
    nextCorpusRegistrationOrder = 0u;
    ownedSphereContacts.clear();
    sphereContactCache = {};
    sphereBufferContacts.clear();
    InitGroupHeaders();
    AppendDefaultAgainstGroups();
}

CHmsCollisionManager::SGroup *CHmsCollisionManager::SZone::GroupAtOrNull(
        u32 groupIndex) {
    return groupIndex < CHmsCollisionManager_GroupCount ? &groups[groupIndex]
                                                        : nullptr;
}

const CHmsCollisionManager::SGroup *
CHmsCollisionManager::SZone::GroupAtOrNull(u32 groupIndex) const {
    return groupIndex < CHmsCollisionManager_GroupCount ? &groups[groupIndex]
                                                        : nullptr;
}

CHmsCollisionManager::SGroup *
CHmsCollisionManager::SZone::GroupForReplayStaticRole(
        CHmsReplayStaticCollisionGroupRole role) {
    return GroupAtOrNull(static_cast<u32>(role));
}

const CHmsCollisionManager::SGroup *
CHmsCollisionManager::SZone::GroupForReplayStaticRole(
        CHmsReplayStaticCollisionGroupRole role) const {
    return GroupAtOrNull(static_cast<u32>(role));
}

void CHmsCollisionManager::SZone::ConfigureWater(
        const WaterOccupancyGrid &occupancy,
        float surfaceHeight,
        float secondaryCullHeight) {
    waterZone.Configure(occupancy,
                        surfaceHeight,
                        secondaryCullHeight);
}

u32 CHmsCollisionManager::SZone::AppendDefaultAgainstGroups(void) {
    u32 pairCount = 0u;
    const CHmsItem::SHmsCollisionGroupPair *pairs =
            CHmsItem::DefaultCollisionGroupPairs(&pairCount);
    u32 appended = 0u;
    for (u32 pairIndex = 0; pairIndex < pairCount; pairIndex++) {
        const CHmsItem::SHmsCollisionGroupPair *pair = &pairs[pairIndex];
        u32 groupA = pair->groupA;
        u32 groupB = pair->groupB;
        if (groupA == 0u || groupA > CHmsCollisionManager_GroupCount ||
            groupB == 0u || groupB > CHmsCollisionManager_GroupCount) {
            continue;
        }

        u32 sourceIndex = groupA - 1u;
        this->groups[sourceIndex].againstGroups.emplace_back();
        CHmsCollisionManagerSAgainstGroup *against =
                &this->groups[sourceIndex].againstGroups.back();
        against->targetGroup = &this->groups[groupB - 1u];
        against->collisionGroupPair = pair;
        appended++;

        if (groupA != groupB) {
            sourceIndex = groupB - 1u;
            this->groups[sourceIndex].againstGroups.emplace_back();
            against = &this->groups[sourceIndex].againstGroups.back();
            against->targetGroup = &this->groups[groupA - 1u];
            against->collisionGroupPair = pair;
            appended++;
        }
    }
    return appended;
}

CHmsCollisionManager::SZone::SZone(
        unsigned long zoneId,
        CHmsCollisionManager *manager)
        : zoneId(static_cast<u32>(zoneId)), manager(manager) {
    InitGroupHeaders();
    AppendDefaultAgainstGroups();
}

CHmsCollisionManager::SZone::~SZone(void) {
    DetachRegisteredCorpuses();
    DetachBoundDynamicZones();
}


void CHmsCollisionManager::SZone::AddCorpus(CHmsCorpus *corpus) {
    if (corpus == nullptr || corpus->Item() == nullptr) {
        return;
    }
    const bool alreadyRegistered =
            corpus->RegisteredCollisionManagerZone() == this;
    if (alreadyRegistered) {
        return;
    }
    const u32 groupIndex = corpus->Item()->CollisionGroup();
    if (CHmsCollisionManagerSZone *previous =
                corpus->RegisteredCollisionManagerZone()) {
        previous->RemoveCorpus(corpus);
    }

    SGroup *group = groupIndex != 0u &&
                            groupIndex <= CHmsCollisionManager_GroupCount
                    ? &groups[groupIndex - 1u]
                    : nullptr;
    const CHmsCorpusId id = CHmsCorpusId::FromRegistrationOrder(
            nextCorpusRegistrationOrder++);
    registeredCorpuses.push_back(corpus);
    corpus->BindCollisionManagerRegistration(*this, group, id);
    if (group != nullptr) {
        group->AddCorpus(corpus);
    }
}


void CHmsCollisionManager::SZone::RemoveCorpus(CHmsCorpus *corpus) {
    if (corpus == nullptr ||
        corpus->RegisteredCollisionManagerZone() != this) {
        return;
    }
    if (SGroup *group = corpus->RegisteredCollisionManagerGroup()) {
        group->RemoveCorpus(corpus);
    }

    if (activeCorpusA == corpus) {
        activeCorpusA = nullptr;
    }
    if (activeCorpusB == corpus) {
        activeCorpusB = nullptr;
    }

    const auto registered = std::find(
            registeredCorpuses.begin(), registeredCorpuses.end(), corpus);
    if (registered != registeredCorpuses.end()) {
        *registered = registeredCorpuses.back();
        registeredCorpuses.pop_back();
    }
    corpus->ClearCollisionManagerRegistration(*this);
}


void CHmsCollisionManager::SZone::UpdateStaticCollisionTrees() {
    for (u32 groupIndex = 0; groupIndex < 5; groupIndex++) {
        groups[groupIndex].UpdateStaticCollisionTrees(1);
    }
}


void CHmsCollisionManager::SZone::PrepareCollisions() {
    for (u32 groupIndex = 0; groupIndex < 5; groupIndex++) {
        groups[groupIndex].ComputeNonStaticCorpusInfos();
    }
    for (u32 groupIndex = 0; groupIndex < 5; groupIndex++) {
        groups[groupIndex].ComputeIsToPerformCollisions();
    }
}
