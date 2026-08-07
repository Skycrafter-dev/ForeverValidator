#ifndef FOREVERVALIDATOR_CUDA_COLLISION_LAYOUT_H
#define FOREVERVALIDATOR_CUDA_COLLISION_LAYOUT_H

#include <cstdint>

#include "engine/core/gm_types.h"

namespace forevervalidator::simulation::cuda::collision {

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

struct CudaCollision {
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

struct CudaCollisionSurfaceHit {
    std::uint32_t surfaceIndex = UINT32_MAX;
    std::uint32_t shapeMask = 0u;
};

struct CudaCollisionMeshRange {
    std::uint16_t first = 0u;
    std::uint16_t count = 0u;
};

constexpr std::uint32_t CudaCollisionSearchTileWidth = 32u;

struct CudaCollisionSearchTile {
    float separationX[CudaCollisionSearchTileWidth];
    float separationY[CudaCollisionSearchTileWidth];
    float separationZ[CudaCollisionSearchTileWidth];
    float impulseNormalX[CudaCollisionSearchTileWidth];
    float impulseNormalY[CudaCollisionSearchTileWidth];
    float impulseNormalZ[CudaCollisionSearchTileWidth];
    float contactPointX[CudaCollisionSearchTileWidth];
    float contactPointY[CudaCollisionSearchTileWidth];
    float contactPointZ[CudaCollisionSearchTileWidth];
    float extraNegatedX[CudaCollisionSearchTileWidth];
    float extraNegatedY[CudaCollisionSearchTileWidth];
    float extraNegatedZ[CudaCollisionSearchTileWidth];
    std::uint32_t materialA[CudaCollisionSearchTileWidth];
    std::uint32_t materialB[CudaCollisionSearchTileWidth];
    std::uint32_t movingShapeIndex[CudaCollisionSearchTileWidth];
    std::uint32_t staticSurfaceIndex[CudaCollisionSearchTileWidth];
    std::uint32_t staticActorIndex[CudaCollisionSearchTileWidth];
    bool sphereMergePrimary[CudaCollisionSearchTileWidth];
};

struct CudaCollisionSearchVectorReference {
    float &x;
    float &y;
    float &z;

#if defined(__CUDACC__)
    __device__ operator GmVec3() const {
        return {x, y, z};
    }

    __device__ CudaCollisionSearchVectorReference &operator=(
            const GmVec3 &value) {
        x = value.x;
        y = value.y;
        z = value.z;
        return *this;
    }

    __device__ CudaCollisionSearchVectorReference &operator=(
            const CudaCollisionSearchVectorReference &value) {
        x = value.x;
        y = value.y;
        z = value.z;
        return *this;
    }
#endif
};

struct CudaCollisionSearchConstVectorReference {
    const float &x;
    const float &y;
    const float &z;

#if defined(__CUDACC__)
    __device__ operator GmVec3() const {
        return {x, y, z};
    }
#endif
};

struct CudaCollisionSearchReference {
    CudaCollisionSearchVectorReference separation;
    CudaCollisionSearchVectorReference impulseNormal;
    CudaCollisionSearchVectorReference contactPoint;
    std::uint32_t &materialA;
    std::uint32_t &materialB;
    bool &sphereMergePrimary;
    CudaCollisionSearchVectorReference extraNegated;
    std::uint32_t &movingShapeIndex;
    std::uint32_t &staticSurfaceIndex;
    std::uint32_t &staticActorIndex;
};

struct CudaCollisionSearchConstReference {
    CudaCollisionSearchConstVectorReference separation;
    CudaCollisionSearchConstVectorReference impulseNormal;
    CudaCollisionSearchConstVectorReference contactPoint;
    const std::uint32_t &materialA;
    const std::uint32_t &materialB;
    const bool &sphereMergePrimary;
    CudaCollisionSearchConstVectorReference extraNegated;
    const std::uint32_t &movingShapeIndex;
    const std::uint32_t &staticSurfaceIndex;
    const std::uint32_t &staticActorIndex;
};

struct CudaCollisionScratch {
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
    CudaCollision collisions[CollisionCapacity]{};
    CudaCollision shapeCollisions[ShapeCollisionCapacity]{};
};

// Per-candidate counters owned only by the compile-time-instrumented search
// kernel. The production specialization neither stores nor updates them.
struct CudaHotPathCounters {
    std::uint64_t physicsSubstepCount = 0u;
    std::uint64_t maximumSubstepsPerTick = 0u;
    std::uint64_t collisionDetectCount = 0u;
    std::uint64_t surfaceCacheEligibleCount = 0u;
    std::uint64_t surfaceCacheReuseCount = 0u;
    std::uint64_t surfaceCacheRefreshCount = 0u;
    // Surface-hit cache refresh failure; mesh-leaf cache construction is a
    // distinct later stage.
    std::uint64_t surfaceCacheRefreshFailureCount = 0u;
    std::uint64_t meshCacheReuseCount = 0u;
    std::uint64_t accelerationCellVisitCount = 0u;
    std::uint64_t accelerationSurfaceVisitCount = 0u;
    std::uint64_t octreeCellVisitCount = 0u;
    std::uint64_t cachedTriangleLeafVisitCount = 0u;
    std::uint64_t triangleTestCount = 0u;
    std::uint64_t triangleHitCount = 0u;
    std::uint64_t rawContactCount = 0u;
    std::uint64_t responseSortCallCount = 0u;
    std::uint64_t responseSortItemCount = 0u;
    std::uint64_t maximumResponseSortItemCount = 0u;
    std::uint64_t groundForcePassCount = 0u;
    std::uint64_t airForcePassCount = 0u;
    std::uint64_t waterForcePassCount = 0u;
    std::uint64_t physicsCallbackDisabledForcePassCount = 0u;
    std::uint64_t zeroDynamicsForcePassCount = 0u;
};

struct CudaCollisionSearchScratch {
    std::uint32_t collisionCount;
    std::uint32_t shapeCollisionCount;
    std::uint32_t surfaceHitCount;
    bool overflow;
    bool surfaceCacheEnabled;
    CudaCollisionSearchTile *collisionStorage;
    CudaCollisionSearchTile *shapeCollisionStorage;
    GmIso4 *shapeWorldStorage;
    GmBoxAligned *movingBoundsStorage;
    CudaCollisionSurfaceHit *surfaceHitStorage;
    CudaCollisionMeshRange *meshRangeStorage;
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

}  // namespace forevervalidator::simulation::cuda::collision

#endif
