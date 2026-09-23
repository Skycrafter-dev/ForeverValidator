// Generated from src/simulation/backends/cuda/cuda_collision_layout.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_COLLISION_LAYOUT_H
#define FOREVERVALIDATOR_HIP_COLLISION_LAYOUT_H

#include <cstdint>

#include "engine/core/gm_types.h"

namespace forevervalidator::simulation::hip::collision {

constexpr std::uint32_t CollisionCapacity = 512u;
constexpr std::uint32_t ShapeCollisionCapacity = 256u;
constexpr std::uint32_t SurfaceHitCapacity = 128u;
constexpr std::uint32_t MeshCellHitCapacity = 1024u;
static_assert(MeshCellHitCapacity <= 65535u);
static_assert(CollisionCapacity <= MeshCellHitCapacity);

enum class Status : std::uint32_t {
    Success,
    Overflow,
    UnsupportedGeometry,
    InvalidScene,
};

enum class OverflowReason : std::uint32_t {
    None,
    ShapeCollisionCapacity,
    CollisionCapacity,
    OrderingStackCapacity,
    MeshCellCapacity,
    CollisionReplacementCapacity,
};

struct HipCollision {
    GmVec3 separation{};
    GmVec3 impulseNormal{};
    GmVec3 contactPoint{};
    std::uint32_t materialA = 0u;
    std::uint32_t materialB = 0u;
    bool sphereMergePrimary = false;
    GmVec3 extraNegated{};
    std::uint32_t movingShapeIndex = UINT32_MAX;
    std::uint32_t staticSurfaceIndex = UINT32_MAX;
    std::uint32_t staticActorIndex = UINT32_MAX;
};

struct HipCollisionSurfaceHit {
    std::uint32_t surfaceIndex = UINT32_MAX;
    std::uint32_t shapeMask = 0u;
};

struct HipCollisionMeshRange {
    std::uint16_t first = 0u;
    std::uint16_t count = 0u;
};

constexpr std::uint32_t HipCollisionSearchTileWidth = 32u;

struct HipCollisionSearchTile {
    float separationX[HipCollisionSearchTileWidth];
    float separationY[HipCollisionSearchTileWidth];
    float separationZ[HipCollisionSearchTileWidth];
    float impulseNormalX[HipCollisionSearchTileWidth];
    float impulseNormalY[HipCollisionSearchTileWidth];
    float impulseNormalZ[HipCollisionSearchTileWidth];
    float contactPointX[HipCollisionSearchTileWidth];
    float contactPointY[HipCollisionSearchTileWidth];
    float contactPointZ[HipCollisionSearchTileWidth];
    float extraNegatedX[HipCollisionSearchTileWidth];
    float extraNegatedY[HipCollisionSearchTileWidth];
    float extraNegatedZ[HipCollisionSearchTileWidth];
    std::uint32_t materialA[HipCollisionSearchTileWidth];
    std::uint32_t materialB[HipCollisionSearchTileWidth];
    std::uint32_t movingShapeIndex[HipCollisionSearchTileWidth];
    std::uint32_t staticSurfaceIndex[HipCollisionSearchTileWidth];
    std::uint32_t staticActorIndex[HipCollisionSearchTileWidth];
    bool sphereMergePrimary[HipCollisionSearchTileWidth];
};

struct HipCollisionSearchVectorReference {
    float &x;
    float &y;
    float &z;

#if defined(__HIPCC__)
    __device__ operator GmVec3() const {
        return {x, y, z};
    }

    __device__ HipCollisionSearchVectorReference &operator=(
            const GmVec3 &value) {
        x = value.x;
        y = value.y;
        z = value.z;
        return *this;
    }

    __device__ HipCollisionSearchVectorReference &operator=(
            const HipCollisionSearchVectorReference &value) {
        x = value.x;
        y = value.y;
        z = value.z;
        return *this;
    }
#endif
};

struct HipCollisionSearchConstVectorReference {
    const float &x;
    const float &y;
    const float &z;

#if defined(__HIPCC__)
    __device__ operator GmVec3() const {
        return {x, y, z};
    }
#endif
};

struct HipCollisionSearchReference {
    HipCollisionSearchVectorReference separation;
    HipCollisionSearchVectorReference impulseNormal;
    HipCollisionSearchVectorReference contactPoint;
    std::uint32_t &materialA;
    std::uint32_t &materialB;
    bool &sphereMergePrimary;
    HipCollisionSearchVectorReference extraNegated;
    std::uint32_t &movingShapeIndex;
    std::uint32_t &staticSurfaceIndex;
    std::uint32_t &staticActorIndex;
};

struct HipCollisionSearchConstReference {
    HipCollisionSearchConstVectorReference separation;
    HipCollisionSearchConstVectorReference impulseNormal;
    HipCollisionSearchConstVectorReference contactPoint;
    const std::uint32_t &materialA;
    const std::uint32_t &materialB;
    const bool &sphereMergePrimary;
    HipCollisionSearchConstVectorReference extraNegated;
    const std::uint32_t &movingShapeIndex;
    const std::uint32_t &staticSurfaceIndex;
    const std::uint32_t &staticActorIndex;
};

struct HipCollisionScratch {
    std::uint32_t collisionCount = 0u;
    std::uint32_t shapeCollisionCount = 0u;
    std::uint32_t accelerationCellVisits = 0u;
    std::uint32_t accelerationSurfaceVisits = 0u;
    std::uint32_t meshCellVisits = 0u;
    std::uint32_t meshCellIntersections = 0u;
    std::uint32_t meshTriangleCells = 0u;
    std::uint32_t triangleTests = 0u;
    std::uint32_t triangleHits = 0u;
    std::uint32_t firstVisitedShape = UINT32_MAX;
    std::uint32_t firstVisitedSurface = UINT32_MAX;
    GmIso4 firstShapeWorld{};
    GmBoxAligned firstMovingBounds{};
    GmBoxAligned firstEllipsoidBox{};
    GmBoxAligned firstSurfaceWorldBounds{};
    GmBoxAligned firstMeshRootBounds{};
    std::uint32_t firstResponseWheelIndex = UINT32_MAX;
    GmVec3 firstResponseReplacementBefore{};
    GmVec3 firstResponseReplacementAfter{};
    bool overflow = false;
    OverflowReason overflowReason = OverflowReason::None;
    std::uint32_t replacementOverflowCount = 0u;
    HipCollision collisions[CollisionCapacity]{};
    HipCollision shapeCollisions[ShapeCollisionCapacity]{};
};

struct HipCollisionSearchScratch {
    std::uint32_t collisionCount;
    std::uint32_t shapeCollisionCount;
    std::uint32_t surfaceHitCount;
    bool overflow;
    bool surfaceCacheEnabled;
    HipCollisionSearchTile *collisionStorage;
    HipCollisionSearchTile *shapeCollisionStorage;
    GmIso4 *shapeWorldStorage;
    GmBoxAligned *movingBoundsStorage;
    HipCollisionSurfaceHit *surfaceHitStorage;
    HipCollisionMeshRange *meshRangeStorage;
    std::uint32_t *meshCellStorage;
    std::uint32_t slot;
    std::uint32_t stride;
    std::uint32_t shapeCapacity;
    OverflowReason overflowReason = OverflowReason::None;
    bool surfaceCacheValid = false;
    std::uint32_t meshCellCount = 0u;
    bool meshCacheValid = false;
    std::uint32_t replacementOverflowCount = 0u;
    float replacementSumX = 0.0f;
    float replacementSumY = 0.0f;
    float replacementSumZ = 0.0f;
    std::uint16_t *responseOrderStorage = nullptr;
};

}  // namespace forevervalidator::simulation::hip::collision

#endif
