#ifndef FOREVERVALIDATOR_CUDA_COLLISION_CUH
#define FOREVERVALIDATOR_CUDA_COLLISION_CUH

#include <cstdint>

#include "simulation/backends/cuda/cuda_exact_math.cuh"
#include "simulation/backends/cuda/cuda_memory.cuh"
#include "simulation/backends/cuda/cuda_collision_layout.h"
#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_tuning.cuh"

namespace forevervalidator::simulation::cuda::collision {

namespace detail {

constexpr float DirectionEpsilonSquared = 1.0e-10f;
constexpr float CollisionDistance = 1.0e-5f;
constexpr float SphereNormalAlignment = 0.8660254f;
static_assert(
        CudaCollisionReplacementOverflowCapacity <=
        ShapeCollisionCapacity * 2u);

__device__ inline CudaCollision &CollisionAt(
        CudaCollisionScratch &scratch,
        std::uint32_t index) {
    return scratch.collisions[index];
}

__device__ inline const CudaCollision &CollisionAt(
        const CudaCollisionScratch &scratch,
        std::uint32_t index) {
    return scratch.collisions[index];
}

__device__ inline CudaCollisionSearchReference CollisionAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    const std::uint32_t tileStride =
            (scratch.stride + CudaCollisionSearchTileWidth - 1u) /
            CudaCollisionSearchTileWidth;
    CudaCollisionSearchTile &tile =
            scratch.collisionStorage[
                    static_cast<std::uint64_t>(index) * tileStride +
                    scratch.slot / CudaCollisionSearchTileWidth];
    const std::uint32_t lane =
            scratch.slot % CudaCollisionSearchTileWidth;
    return {
            {tile.separationX[lane], tile.separationY[lane],
             tile.separationZ[lane]},
            {tile.impulseNormalX[lane], tile.impulseNormalY[lane],
             tile.impulseNormalZ[lane]},
            {tile.contactPointX[lane], tile.contactPointY[lane],
             tile.contactPointZ[lane]},
            tile.materialA[lane],
            tile.materialB[lane],
            tile.sphereMergePrimary[lane],
            {tile.extraNegatedX[lane], tile.extraNegatedY[lane],
             tile.extraNegatedZ[lane]},
            tile.movingShapeIndex[lane],
            tile.staticSurfaceIndex[lane],
            tile.staticActorIndex[lane],
    };
}

__device__ inline CudaCollisionSearchConstReference CollisionAt(
        const CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    const std::uint32_t tileStride =
            (scratch.stride + CudaCollisionSearchTileWidth - 1u) /
            CudaCollisionSearchTileWidth;
    const CudaCollisionSearchTile &tile =
            scratch.collisionStorage[
                    static_cast<std::uint64_t>(index) * tileStride +
                    scratch.slot / CudaCollisionSearchTileWidth];
    const std::uint32_t lane =
            scratch.slot % CudaCollisionSearchTileWidth;
    return {
            {tile.separationX[lane], tile.separationY[lane],
             tile.separationZ[lane]},
            {tile.impulseNormalX[lane], tile.impulseNormalY[lane],
             tile.impulseNormalZ[lane]},
            {tile.contactPointX[lane], tile.contactPointY[lane],
             tile.contactPointZ[lane]},
            tile.materialA[lane],
            tile.materialB[lane],
            tile.sphereMergePrimary[lane],
            {tile.extraNegatedX[lane], tile.extraNegatedY[lane],
             tile.extraNegatedZ[lane]},
            tile.movingShapeIndex[lane],
            tile.staticSurfaceIndex[lane],
            tile.staticActorIndex[lane],
    };
}

__device__ inline CudaCollision &ShapeCollisionAt(
        CudaCollisionScratch &scratch,
        std::uint32_t index) {
    return scratch.shapeCollisions[index];
}

__device__ inline const CudaCollision &ShapeCollisionAt(
        const CudaCollisionScratch &scratch,
        std::uint32_t index) {
    return scratch.shapeCollisions[index];
}

__device__ inline CudaCollisionSearchReference ShapeCollisionAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    const std::uint32_t tileStride =
            (scratch.stride +
             CudaCollisionSearchTileWidth - 1u) /
            CudaCollisionSearchTileWidth;
    CudaCollisionSearchTile &tile =
            scratch.shapeCollisionStorage[
                    static_cast<std::uint64_t>(index) * tileStride +
                    scratch.slot / CudaCollisionSearchTileWidth];
    const std::uint32_t lane =
            scratch.slot % CudaCollisionSearchTileWidth;
    return {
            {tile.separationX[lane], tile.separationY[lane],
             tile.separationZ[lane]},
            {tile.impulseNormalX[lane], tile.impulseNormalY[lane],
             tile.impulseNormalZ[lane]},
            {tile.contactPointX[lane], tile.contactPointY[lane],
             tile.contactPointZ[lane]},
            tile.materialA[lane],
            tile.materialB[lane],
            tile.sphereMergePrimary[lane],
            {tile.extraNegatedX[lane], tile.extraNegatedY[lane],
             tile.extraNegatedZ[lane]},
            tile.movingShapeIndex[lane],
            tile.staticSurfaceIndex[lane],
            tile.staticActorIndex[lane],
    };
}

__device__ inline CudaCollisionSearchConstReference ShapeCollisionAt(
    const CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    const std::uint32_t tileStride =
            (scratch.stride +
             CudaCollisionSearchTileWidth - 1u) /
            CudaCollisionSearchTileWidth;
    const CudaCollisionSearchTile &tile =
            scratch.shapeCollisionStorage[
                    static_cast<std::uint64_t>(index) * tileStride +
                    scratch.slot / CudaCollisionSearchTileWidth];
    const std::uint32_t lane =
            scratch.slot % CudaCollisionSearchTileWidth;
    return {
            {tile.separationX[lane], tile.separationY[lane],
             tile.separationZ[lane]},
            {tile.impulseNormalX[lane], tile.impulseNormalY[lane],
             tile.impulseNormalZ[lane]},
            {tile.contactPointX[lane], tile.contactPointY[lane],
             tile.contactPointZ[lane]},
            tile.materialA[lane],
            tile.materialB[lane],
            tile.sphereMergePrimary[lane],
            {tile.extraNegatedX[lane], tile.extraNegatedY[lane],
             tile.extraNegatedZ[lane]},
            tile.movingShapeIndex[lane],
            tile.staticSurfaceIndex[lane],
            tile.staticActorIndex[lane],
    };
}

__device__ inline GmVec3 &ReplacementOverflowAt(
        CudaCollisionScratch &scratch,
        std::uint32_t index) {
    CudaCollision &storage =
            ShapeCollisionAt(scratch, index >> 1u);
    return (index & 1u) == 0u
            ? storage.extraNegated
            : storage.contactPoint;
}

__device__ inline const GmVec3 &ReplacementOverflowAt(
        const CudaCollisionScratch &scratch,
        std::uint32_t index) {
    const CudaCollision &storage =
            ShapeCollisionAt(scratch, index >> 1u);
    return (index & 1u) == 0u
            ? storage.extraNegated
            : storage.contactPoint;
}

__device__ inline CudaCollisionSearchVectorReference
ReplacementOverflowAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    const CudaCollisionSearchReference storage =
            ShapeCollisionAt(scratch, index >> 1u);
    return (index & 1u) == 0u
            ? storage.extraNegated
            : storage.contactPoint;
}

__device__ inline CudaCollisionSearchConstVectorReference
ReplacementOverflowAt(
        const CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    const CudaCollisionSearchConstReference storage =
            ShapeCollisionAt(scratch, index >> 1u);
    return (index & 1u) == 0u
            ? storage.extraNegated
            : storage.contactPoint;
}

template<typename Scratch>
__device__ inline void CaptureReplacementOverflow(
        const Scratch &scratch,
        CudaFixedArray<
                GmVec3,
                CudaCollisionReplacementOverflowCapacity>
                &destination) {
    const std::uint32_t previousCount = destination.count;
    destination.count = scratch.replacementOverflowCount;
    for (std::uint32_t index = 0u;
         index < destination.count; ++index) {
        destination.values[index] =
                ReplacementOverflowAt(scratch, index);
    }
    for (std::uint32_t index = destination.count;
         index < previousCount; ++index) {
        destination.values[index] = {};
    }
}

__device__ inline GmIso4 &ShapeWorldAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t traversal) {
    return scratch.shapeWorldStorage[
            static_cast<std::uint64_t>(traversal) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline const GmIso4 &ShapeWorldAt(
        const CudaCollisionSearchScratch &scratch,
        std::uint32_t traversal) {
    return scratch.shapeWorldStorage[
            static_cast<std::uint64_t>(traversal) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline GmBoxAligned UnifiedMovingBoundsAt(
        const CudaCollisionSearchScratch &scratch) {
    GmBoxAligned result;
    memory::CopyBytes<sizeof(result)>(
            &result,
            &scratch.shapeWorldStorage[scratch.slot]);
    return result;
}

__device__ inline void StoreUnifiedMovingBounds(
        CudaCollisionSearchScratch &scratch,
        const GmBoxAligned &value) {
    memory::CopyBytes<sizeof(value)>(
            &scratch.shapeWorldStorage[scratch.slot],
            &value);
}

__device__ inline GmBoxAligned &MovingBoundsAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t traversal) {
    return scratch.movingBoundsStorage[
            static_cast<std::uint64_t>(traversal) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline const GmBoxAligned &MovingBoundsAt(
        const CudaCollisionSearchScratch &scratch,
        std::uint32_t traversal) {
    return scratch.movingBoundsStorage[
            static_cast<std::uint64_t>(traversal) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline CudaCollisionSurfaceHit &SurfaceHitAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    return scratch.surfaceHitStorage[
            static_cast<std::uint64_t>(index) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline CudaCollisionMeshRange &MeshRangeAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    return scratch.meshRangeStorage[
            static_cast<std::uint64_t>(index) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline const CudaCollisionMeshRange &MeshRangeAt(
        const CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    return scratch.meshRangeStorage[
            static_cast<std::uint64_t>(index) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline std::uint32_t &MeshCellAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    return scratch.meshCellStorage[
            static_cast<std::uint64_t>(index) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline const std::uint32_t &MeshCellAt(
        const CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    return scratch.meshCellStorage[
            static_cast<std::uint64_t>(index) *
                    scratch.stride +
            scratch.slot];
}

__device__ inline CudaCollision &OrderedCollisionAt(
        CudaCollisionScratch &scratch,
        std::uint32_t index) {
    return CollisionAt(scratch, index);
}

__device__ inline CudaCollisionSearchReference OrderedCollisionAt(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t index) {
    return CollisionAt(
            scratch,
            scratch.responseOrderStorage[
                static_cast<std::uint64_t>(index) *
                            scratch.stride +
                    scratch.slot]);
}

__device__ inline void InitializeResponseOrder(
        CudaCollisionScratch &) {}

__device__ inline void InitializeResponseOrder(
        CudaCollisionSearchScratch &scratch) {
    for (std::uint32_t index = 0u;
         index < scratch.collisionCount; ++index) {
        scratch.responseOrderStorage[
                static_cast<std::uint64_t>(index) *
                        scratch.stride +
                scratch.slot] =
                static_cast<std::uint16_t>(index);
    }
}

__device__ inline void SwapOrdered(
        CudaCollisionScratch &scratch,
        std::uint32_t left,
        std::uint32_t right) {
    const CudaCollision temporary =
            CollisionAt(scratch, left);
    CollisionAt(scratch, left) =
            CollisionAt(scratch, right);
    CollisionAt(scratch, right) = temporary;
}

__device__ inline void SwapOrdered(
        CudaCollisionSearchScratch &scratch,
        std::uint32_t left,
        std::uint32_t right) {
    std::uint16_t &leftIndex =
            scratch.responseOrderStorage[
                    static_cast<std::uint64_t>(left) *
                            scratch.stride +
                    scratch.slot];
    std::uint16_t &rightIndex =
            scratch.responseOrderStorage[
                    static_cast<std::uint64_t>(right) *
                            scratch.stride +
                    scratch.slot];
    const std::uint16_t temporary = leftIndex;
    leftIndex = rightIndex;
    rightIndex = temporary;
}

template <bool TrackDiagnostics, typename Scratch>
__device__ inline void Clear(Scratch &scratch) {
    scratch.collisionCount = 0u;
    scratch.shapeCollisionCount = 0u;
    if constexpr (TrackDiagnostics) {
        scratch.accelerationCellVisits = 0u;
        scratch.accelerationSurfaceVisits = 0u;
        scratch.meshCellVisits = 0u;
        scratch.meshCellIntersections = 0u;
        scratch.meshTriangleCells = 0u;
        scratch.triangleTests = 0u;
        scratch.triangleHits = 0u;
        scratch.firstVisitedShape = UINT32_MAX;
        scratch.firstVisitedSurface = UINT32_MAX;
    }
    scratch.overflow = false;
    scratch.overflowReason = OverflowReason::None;
}

template <typename Scratch>
__device__ inline std::uint32_t AddShape(
        Scratch &scratch) {
    if (scratch.shapeCollisionCount >=
        ShapeCollisionCapacity) {
        scratch.overflow = true;
        scratch.overflowReason =
                OverflowReason::ShapeCollisionCapacity;
        return UINT32_MAX;
    }
    const std::uint32_t result = scratch.shapeCollisionCount++;
    ShapeCollisionAt(scratch, result).sphereMergePrimary = false;
    return result;
}

template <typename Scratch, typename Collision>
__device__ inline void AddMain(
        Scratch &scratch,
        const Collision &value) {
    if (scratch.collisionCount >= CollisionCapacity) {
        scratch.overflow = true;
        scratch.overflowReason = OverflowReason::CollisionCapacity;
        return;
    }
    decltype(auto) destination =
            CollisionAt(scratch, scratch.collisionCount++);
    destination.separation = value.separation;
    destination.impulseNormal = value.impulseNormal;
    destination.contactPoint = value.contactPoint;
    destination.materialA = value.materialA;
    destination.materialB = value.materialB;
    destination.sphereMergePrimary = value.sphereMergePrimary;
    destination.extraNegated = value.extraNegated;
    destination.movingShapeIndex = value.movingShapeIndex;
    destination.staticActorIndex = value.staticActorIndex;
}

template<typename T>
__device__ inline const T *SceneSection(
        const CudaPackedSceneHeader *scene,
        const CudaSceneSection &section) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
    const auto *base =
            reinterpret_cast<const std::byte *>(
                    ::forevervalidator::simulation::cuda::
                            research::SessionSceneBase());
#elif defined(FOREVERVALIDATOR_CUDA_RESEARCH_CONSTANT_SCENE)
    const auto *base =
            reinterpret_cast<const std::byte *>(
                    ::forevervalidator::simulation::cuda::
                            research::StaticSceneBase);
#else
    const auto *base =
            reinterpret_cast<const std::byte *>(scene);
#endif
    return reinterpret_cast<const T *>(
            base + section.offset);
}

__device__ __forceinline__ float Dot(
        GmVec3 left, GmVec3 right) {
    const float xy = left.x * right.x + left.y * right.y;
    return xy + left.z * right.z;
}

__device__ inline GmVec3 Add(
        const GmVec3 &left, const GmVec3 &right) {
    return {
            left.x + right.x,
            left.y + right.y,
            left.z + right.z,
    };
}

__device__ inline GmVec3 Subtract(
        const GmVec3 &left, const GmVec3 &right) {
    return {
            left.x - right.x,
            left.y - right.y,
            left.z - right.z,
    };
}

__device__ inline GmVec3 Scale(
        const GmVec3 &value, float scale) {
    return {
            value.x * scale,
            value.y * scale,
            value.z * scale,
    };
}

__device__ inline GmVec3 Negate(const GmVec3 &value) {
    return {-value.x, -value.y, -value.z};
}

__device__ inline GmVec3 Cross(
        const GmVec3 &left, const GmVec3 &right) {
    return {
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x,
    };
}

__device__ inline GmVec3 Normalize(
        const GmVec3 &value, float epsilonSquared) {
    GmVec3 result = value;
    const float lengthSquared = Dot(result, result);
    if (epsilonSquared < lengthSquared) {
        result = Scale(
                result, 1.0f / exact::Sqrt(lengthSquared));
    }
    return result;
}

__device__ inline GmVec3 TransformDirection(
        const GmMat3 &matrix, const GmVec3 &direction) {
    return {
            (matrix.basisX.x * direction.x +
             matrix.basisY.x * direction.y) +
                    matrix.basisZ.x * direction.z,
            (matrix.basisX.y * direction.x +
             matrix.basisY.y * direction.y) +
                    matrix.basisZ.y * direction.z,
            (matrix.basisX.z * direction.x +
             matrix.basisY.z * direction.y) +
                    matrix.basisZ.z * direction.z,
    };
}

__device__ inline GmVec3 TransformPoint(
        const GmIso4 &transform, const GmVec3 &point) {
    return Add(
            TransformDirection(transform.rotation, point),
            transform.translation);
}

__device__ inline GmMat3 Compose(
        const GmMat3 &first, const GmMat3 &second) {
    return {
            TransformDirection(second, first.basisX),
            TransformDirection(second, first.basisY),
            TransformDirection(second, first.basisZ),
    };
}

__device__ inline GmMat3 Transpose(const GmMat3 &matrix) {
    return {
            {matrix.basisX.x, matrix.basisY.x,
             matrix.basisZ.x},
            {matrix.basisX.y, matrix.basisY.y,
             matrix.basisZ.y},
            {matrix.basisX.z, matrix.basisY.z,
             matrix.basisZ.z},
    };
}

__device__ inline GmIso4 Inverse(const GmIso4 &transform) {
    const GmMat3 inverseRotation =
            Transpose(transform.rotation);
    return {
            inverseRotation,
            TransformDirection(
                    inverseRotation,
                    Negate(transform.translation)),
    };
}

__device__ inline GmIso4 Compose(
        const GmIso4 &first, const GmIso4 &second) {
    return {
            Compose(first.rotation, second.rotation),
            TransformPoint(second, first.translation),
    };
}

__device__ inline GmIso4 MultInverse(
        const GmIso4 &transform, const GmIso4 &right) {
    return Compose(transform, Inverse(right));
}

__device__ inline GmIso4 DiagonalTransform(
        const GmVec3 &scale, const GmVec3 &translation) {
    return {
            {{scale.x, 0.0f, 0.0f},
             {0.0f, scale.y, 0.0f},
             {0.0f, 0.0f, scale.z}},
            translation,
    };
}

__device__ inline void ScaleRows(
        GmIso4 &transform, const GmVec3 &scale) {
    transform.rotation.basisX.x =
            scale.x * transform.rotation.basisX.x;
    transform.rotation.basisY.x =
            transform.rotation.basisY.x * scale.x;
    transform.rotation.basisZ.x =
            scale.x * transform.rotation.basisZ.x;
    transform.translation.x *= scale.x;
    transform.rotation.basisX.y =
            scale.y * transform.rotation.basisX.y;
    transform.rotation.basisY.y =
            transform.rotation.basisY.y * scale.y;
    transform.rotation.basisZ.y =
            scale.y * transform.rotation.basisZ.y;
    transform.translation.y *= scale.y;
    transform.rotation.basisX.z =
            scale.z * transform.rotation.basisX.z;
    transform.rotation.basisY.z =
            transform.rotation.basisY.z * scale.z;
    transform.rotation.basisZ.z =
            scale.z * transform.rotation.basisZ.z;
    transform.translation.z *= scale.z;
}

__device__ inline GmBoxAligned TransformBox(
        const GmBoxAligned &box, const GmIso4 &transform) {
    return {
            TransformPoint(transform, box.center),
            {
                    (fabsf(transform.rotation.basisX.x) *
                                     box.halfExtents.x +
                     fabsf(transform.rotation.basisY.x) *
                                     box.halfExtents.y) +
                            fabsf(transform.rotation.basisZ.x) *
                                    box.halfExtents.z,
                    (fabsf(transform.rotation.basisX.y) *
                                     box.halfExtents.x +
                     fabsf(transform.rotation.basisY.y) *
                                     box.halfExtents.y) +
                            fabsf(transform.rotation.basisZ.y) *
                                    box.halfExtents.z,
                    (fabsf(transform.rotation.basisX.z) *
                                     box.halfExtents.x +
                     fabsf(transform.rotation.basisY.z) *
                                     box.halfExtents.y) +
                            fabsf(transform.rotation.basisZ.z) *
                                    box.halfExtents.z,
            },
    };
}

__device__ inline bool BoundsIntersect(
        const GmBoxAligned &query,
        const GmBoxAligned &candidate) {
    if (candidate.halfExtents.z + query.halfExtents.z <
        fabsf(candidate.center.z - query.center.z)) {
        return false;
    }
    if (candidate.halfExtents.y + query.halfExtents.y <
        fabsf(candidate.center.y - query.center.y)) {
        return false;
    }
    return !(candidate.halfExtents.x + query.halfExtents.x <
             fabsf(candidate.center.x - query.center.x));
}

__device__ inline bool BoundsContain(
        const GmBoxAligned &outer,
        const GmBoxAligned &inner) {
    constexpr float FloatEpsilon =
            1.1920928955078125e-7f;
    const float slack =
            8.0f * FloatEpsilon *
            (1.0f +
             fabsf(outer.center.x) +
             fabsf(outer.center.y) +
             fabsf(outer.center.z) +
             fabsf(inner.center.x) +
             fabsf(inner.center.y) +
             fabsf(inner.center.z) +
             outer.halfExtents.x +
             outer.halfExtents.y +
             outer.halfExtents.z +
             inner.halfExtents.x +
             inner.halfExtents.y +
             inner.halfExtents.z);
    return fabsf(inner.center.x - outer.center.x) +
                           inner.halfExtents.x + slack <=
                    outer.halfExtents.x &&
            fabsf(inner.center.y - outer.center.y) +
                            inner.halfExtents.y + slack <=
                    outer.halfExtents.y &&
            fabsf(inner.center.z - outer.center.z) +
                            inner.halfExtents.z + slack <=
                    outer.halfExtents.z;
}

__device__ inline GmBoxAligned ExpandBoundsAlong(
        const GmBoxAligned &bounds,
        const GmVec3 &travel,
        float margin) {
    return {
            {
                    bounds.center.x + travel.x * 0.5f,
                    bounds.center.y + travel.y * 0.5f,
                    bounds.center.z + travel.z * 0.5f,
            },
            {
                    bounds.halfExtents.x +
                            fabsf(travel.x) * 0.5f + margin,
                    bounds.halfExtents.y +
                            fabsf(travel.y) * 0.5f + margin,
                    bounds.halfExtents.z +
                            fabsf(travel.z) * 0.5f + margin,
            },
    };
}

__device__ inline GmBoxAligned IncludeBounds(
        const GmBoxAligned &left,
        const GmBoxAligned &right) {
    const GmVec3 lower = {
            fminf(left.center.x - left.halfExtents.x,
                  right.center.x - right.halfExtents.x),
            fminf(left.center.y - left.halfExtents.y,
                  right.center.y - right.halfExtents.y),
            fminf(left.center.z - left.halfExtents.z,
                  right.center.z - right.halfExtents.z),
    };
    const GmVec3 upper = {
            fmaxf(left.center.x + left.halfExtents.x,
                  right.center.x + right.halfExtents.x),
            fmaxf(left.center.y + left.halfExtents.y,
                  right.center.y + right.halfExtents.y),
            fmaxf(left.center.z + left.halfExtents.z,
                  right.center.z + right.halfExtents.z),
    };
    return {
            Scale(Add(lower, upper), 0.5f),
            Scale(Subtract(upper, lower), 0.5f),
    };
}

__device__ inline void ExpandBoundsForRounding(
        GmBoxAligned &bounds) {
    constexpr float FloatEpsilon =
            1.1920928955078125e-7f;
    const float margin =
            16.0f * FloatEpsilon *
            (1.0f +
             fabsf(bounds.center.x) +
             fabsf(bounds.center.y) +
             fabsf(bounds.center.z) +
             bounds.halfExtents.x +
             bounds.halfExtents.y +
             bounds.halfExtents.z);
    bounds.halfExtents.x += margin;
    bounds.halfExtents.y += margin;
    bounds.halfExtents.z += margin;
}

constexpr float EmptyAirCertificateDistance = 5.0f;
constexpr float EmptyAirCertificateMargin = 0.0625f;
constexpr std::uint8_t EmptyAirProbeCooldownOpportunities = 8u;

enum class EmptyAirProbeResult : std::uint32_t {
    Ineligible,
    Clear,
    Blocked,
};

__device__ inline bool FiniteBounds(
        const GmBoxAligned &bounds) {
    return isfinite(bounds.center.x) &&
            isfinite(bounds.center.y) &&
            isfinite(bounds.center.z) &&
            isfinite(bounds.halfExtents.x) &&
            isfinite(bounds.halfExtents.y) &&
            isfinite(bounds.halfExtents.z) &&
            bounds.halfExtents.x >= 0.0f &&
            bounds.halfExtents.y >= 0.0f &&
            bounds.halfExtents.z >= 0.0f;
}

template<bool CollectHotPathMetrics>
__device__ inline void InvalidateEmptyAirCertificate(
        CudaEmptyAirCertificateState<true> &state,
        CudaHotPathCounters *hotPathCounters,
        bool resetProbeCooldown = false) {
    if (state.active) {
        if constexpr (CollectHotPathMetrics) {
            ++hotPathCounters->emptyAirCertificateInvalidationCount;
        }
        state.active = false;
    }
    if (resetProbeCooldown) {
        state.probeCooldown = 0u;
    }
}

// Extends an already-proven empty one-tick broad-phase cache along the
// current velocity. Every live collision-shape bound is included in the
// query; callers may reuse the result only while every live bound remains
// contained. Malformed packed ranges fail closed before any cache mutation.
template<bool CollectHotPathMetrics>
__device__ __noinline__ EmptyAirProbeResult
TryExtendEmptyAirCertificate(
        const CudaPackedSceneHeader *scene,
        const CudaSceneAccelerationCell *acceleration,
        const CudaSceneSurface *surfaces,
        const CudaSceneOctreeCell *octreeCells,
        const GmVec3 &linearSpeed,
        const GmVec3 &shortTravel,
        std::uint32_t collisionShapeCount,
        CudaCollisionSearchScratch &scratch,
        CudaHotPathCounters *hotPathCounters) {
    if (scene == nullptr || collisionShapeCount == 0u ||
        collisionShapeCount > scratch.shapeCapacity ||
        scratch.movingBoundsStorage == nullptr ||
        scratch.stride == 0u || scratch.slot >= scratch.stride ||
        (scene->accelerationCells.count != 0u &&
         acceleration == nullptr) ||
        (scene->surfaces.count != 0u && surfaces == nullptr) ||
        (scene->octreeCells.count != 0u && octreeCells == nullptr)) {
        return EmptyAirProbeResult::Blocked;
    }

    const float speedSquared =
            (linearSpeed.y * linearSpeed.y +
             linearSpeed.x * linearSpeed.x) +
            linearSpeed.z * linearSpeed.z;
    if (!isfinite(speedSquared) ||
        speedSquared <= DirectionEpsilonSquared) {
        return EmptyAirProbeResult::Ineligible;
    }
    const float speed = exact::Sqrt(speedSquared);
    const float shortTravelSquared =
            (shortTravel.y * shortTravel.y +
             shortTravel.x * shortTravel.x) +
            shortTravel.z * shortTravel.z;
    if (!isfinite(speed) || speed <= 0.0f ||
        !isfinite(shortTravelSquared) ||
        shortTravelSquared < 0.0f) {
        return EmptyAirProbeResult::Ineligible;
    }
    const float shortDistance = exact::Sqrt(shortTravelSquared);
    if (!isfinite(shortDistance)) {
        return EmptyAirProbeResult::Ineligible;
    }
    const float additionalDistance =
            EmptyAirCertificateDistance - shortDistance;
    GmVec3 additionalTravel{};
    if (additionalDistance > 0.0f) {
        additionalTravel = Scale(
                linearSpeed, additionalDistance / speed);
        if (!isfinite(additionalTravel.x) ||
            !isfinite(additionalTravel.y) ||
            !isfinite(additionalTravel.z)) {
            return EmptyAirProbeResult::Ineligible;
        }
    }

    GmBoxAligned query{};
    for (std::uint32_t traversal = 0u;
         traversal < collisionShapeCount;
         ++traversal) {
        const GmBoxAligned shortBounds =
                MovingBoundsAt(scratch, traversal);
        if (!FiniteBounds(shortBounds)) {
            return EmptyAirProbeResult::Blocked;
        }
        GmBoxAligned extended = shortBounds;
        if (additionalDistance > 0.0f) {
            extended = ExpandBoundsAlong(
                    shortBounds,
                    additionalTravel,
                    EmptyAirCertificateMargin);
            ExpandBoundsForRounding(extended);
        }
        query = traversal == 0u
                ? extended
                : IncludeBounds(query, extended);
    }
    ExpandBoundsForRounding(query);
    if (!FiniteBounds(query)) {
        return EmptyAirProbeResult::Blocked;
    }

    constexpr std::uint32_t TargetGroups[] = {1u, 3u, 4u};
    for (std::uint32_t groupIndex = 0u;
         groupIndex < 3u;
         ++groupIndex) {
        const CudaSceneAccelerationRange range =
                scene->accelerationGroups[
                        TargetGroups[groupIndex] - 1u];
        // Validate the complete range before recognizing the canonical
        // one-cell empty root. Otherwise a malformed short range could evade
        // the certificate proof entirely.
        if (range.cellCount == 0u ||
            range.firstCell > scene->accelerationCells.count ||
            range.cellCount >
                    scene->accelerationCells.count - range.firstCell) {
            return EmptyAirProbeResult::Blocked;
        }
        const CudaSceneAccelerationCell &root =
                acceleration[range.firstCell];
        if (!FiniteBounds(root.bounds) ||
            root.surfaceIndex != UINT32_MAX ||
            root.subtreeEntryCount != range.cellCount) {
            return EmptyAirProbeResult::Blocked;
        }
        if (range.cellCount == 1u) {
            continue;
        }

        std::uint32_t cursor = 0u;
        while (cursor < range.cellCount) {
            if constexpr (CollectHotPathMetrics) {
                ++hotPathCounters->
                        emptyAirProbeAccelerationCellVisitCount;
            }
            const CudaSceneAccelerationCell &cell =
                    acceleration[range.firstCell + cursor];
            if (!FiniteBounds(cell.bounds) ||
                cell.subtreeEntryCount == 0u ||
                cell.subtreeEntryCount >
                        range.cellCount - cursor ||
                (cell.surfaceIndex != UINT32_MAX &&
                 cell.subtreeEntryCount != 1u) ||
                (cursor != 0u &&
                 cell.surfaceIndex == UINT32_MAX &&
                 cell.subtreeEntryCount == 1u)) {
                return EmptyAirProbeResult::Blocked;
            }
            if (!BoundsIntersect(query, cell.bounds)) {
                cursor += cell.subtreeEntryCount;
                continue;
            }
            ++cursor;
            if (cell.surfaceIndex == UINT32_MAX) {
                continue;
            }
            if (cell.surfaceIndex >= scene->surfaces.count) {
                return EmptyAirProbeResult::Blocked;
            }
            const CudaSceneSurface &surface =
                    surfaces[cell.surfaceIndex];
            if (surface.type != static_cast<std::uint32_t>(
                        GmSurf::EGmSurfType::Mesh) ||
                surface.firstOctreeCell > scene->octreeCells.count ||
                surface.octreeCellCount >
                        scene->octreeCells.count -
                                surface.firstOctreeCell ||
                surface.firstTriangle > scene->triangles.count ||
                surface.triangleCount >
                        scene->triangles.count -
                                surface.firstTriangle ||
                (surface.triangleCount != 0u &&
                 surface.octreeCellCount == 0u)) {
                return EmptyAirProbeResult::Blocked;
            }
            if (surface.triangleCount == 0u) {
                continue;
            }

            GmBoxAligned localQuery =
                    TransformBox(query, surface.worldToLocal);
            ExpandBoundsForRounding(localQuery);
            if (!FiniteBounds(localQuery)) {
                return EmptyAirProbeResult::Blocked;
            }
            const CudaSceneOctreeCell &octreeRoot =
                    octreeCells[surface.firstOctreeCell];
            if (!FiniteBounds(octreeRoot.bounds) ||
                octreeRoot.subtreeEntryCount !=
                        surface.octreeCellCount) {
                return EmptyAirProbeResult::Blocked;
            }
            std::uint32_t octreeCursor = 0u;
            while (octreeCursor < surface.octreeCellCount) {
                if constexpr (CollectHotPathMetrics) {
                    ++hotPathCounters->
                            emptyAirProbeOctreeCellVisitCount;
                }
                const CudaSceneOctreeCell &entry =
                        octreeCells[
                                surface.firstOctreeCell +
                                octreeCursor];
                if (!FiniteBounds(entry.bounds) ||
                    entry.subtreeEntryCount == 0u ||
                    entry.subtreeEntryCount >
                            surface.octreeCellCount -
                                    octreeCursor ||
                    (entry.containsTriangle != 0u &&
                     entry.subtreeEntryCount != 1u)) {
                    return EmptyAirProbeResult::Blocked;
                }
                if (!BoundsIntersect(localQuery, entry.bounds)) {
                    octreeCursor += entry.subtreeEntryCount;
                    continue;
                }
                ++octreeCursor;
                if (entry.containsTriangle == 0u) {
                    continue;
                }
                if (entry.triangleIndex >= surface.triangleCount) {
                    return EmptyAirProbeResult::Blocked;
                }
                // A triangle leaf intersects the conservative swept query.
                return EmptyAirProbeResult::Blocked;
            }
        }
    }

    if (additionalDistance <= 0.0f) {
        // The existing empty short cache already spans five metres. Preserve
        // its exact bounds so normal containment remains the sole reuse gate.
        return EmptyAirProbeResult::Clear;
    }
    for (std::uint32_t traversal = 0u;
         traversal < collisionShapeCount;
         ++traversal) {
        GmBoxAligned extended = ExpandBoundsAlong(
                MovingBoundsAt(scratch, traversal),
                additionalTravel,
                EmptyAirCertificateMargin);
        ExpandBoundsForRounding(extended);
        MovingBoundsAt(scratch, traversal) = extended;
    }
    return EmptyAirProbeResult::Clear;
}

__device__ inline std::uint16_t LocalMaterialIndex(
        GmLocalMaterialIndex value) {
    std::uint16_t result = 0u;
    memory::CopyBytes<sizeof(result)>(&result, &value);
    return result;
}

__device__ inline std::uint32_t SurfaceMaterial(
        const CudaPackedSceneHeader *scene,
        const CudaSceneSurface &surface,
        GmLocalMaterialIndex local) {
    const std::uint32_t *materials =
            SceneSection<std::uint32_t>(
                    scene, scene->materials);
    const std::uint32_t index = LocalMaterialIndex(local);
    return index < surface.materialCount
            ? materials[surface.firstMaterial + index]
            : static_cast<std::uint32_t>(
                      EPlugSurfaceMaterialId_Concrete);
}

__device__ inline GmVec3 TransformEllipsoidNormal(
        const GmVec3 &normal, const GmMat3 &rotation) {
    return {
            (rotation.basisX.x * normal.x +
             rotation.basisY.x * normal.y) +
                    rotation.basisZ.x * normal.z,
            (rotation.basisX.y * normal.x +
             rotation.basisY.y * normal.y) +
                    rotation.basisZ.y * normal.z,
            (rotation.basisX.z * normal.x +
             rotation.basisY.z * normal.y) +
                    rotation.basisZ.z * normal.z,
    };
}

template <typename Scratch>
struct UnitSphereTriangleQuery {
    Scratch &scratch;
    GmVec3 center{};
    float radius = 1.0f;
    GmVec3 triangleNormal{};

    __device__ std::uint32_t AddCollision(void) {
        const std::uint32_t index = AddShape(scratch);
        if (index == UINT32_MAX) return UINT32_MAX;
        return index;
    }

    __device__ int EmitFeature(
            const GmVec3 &point,
            float minimumDistanceSquared,
            bool requireContainment) {
        const GmVec3 toCenter = Subtract(center, point);
        const float distanceSquared = Dot(toCenter, toCenter);
        if ((requireContainment &&
             radius * radius < distanceSquared) ||
            !(minimumDistanceSquared < distanceSquared)) {
            return 0;
        }
        const float distance = exact::Sqrt(distanceSquared);
        const float inverse = 1.0f / distance;
        const GmVec3 normal = Scale(toCenter, inverse);
        const GmVec3 penetration = Scale(
                toCenter, (distance - radius) * inverse);
        const std::uint32_t collisionIndex = AddCollision();
        if (collisionIndex == UINT32_MAX) return 0;
        decltype(auto) collision =
                ShapeCollisionAt(scratch, collisionIndex);
        collision.impulseNormal = normal;
        collision.separation = Scale(
                triangleNormal,
                Dot(penetration, triangleNormal));
        collision.contactPoint = point;
        collision.extraNegated = triangleNormal;
        return 1;
    }

    __device__ int EmitEndpointB(
            const GmVec3 &point, float minimumDistance) {
        const GmVec3 toCenter = Subtract(center, point);
        const float distanceSquared = Dot(toCenter, toCenter);
        const float distance = exact::Sqrt(distanceSquared);
        if (radius * radius < distance ||
            !(minimumDistance < distance)) {
            return 0;
        }
        const float endpointDistance = exact::Sqrt(distance);
        const float inverse = 1.0f / endpointDistance;
        const GmVec3 normal = Scale(toCenter, inverse);
        const GmVec3 penetration = Scale(
                toCenter,
                (endpointDistance - radius) * inverse);
        const std::uint32_t collisionIndex = AddCollision();
        if (collisionIndex == UINT32_MAX) return 0;
        decltype(auto) collision =
                ShapeCollisionAt(scratch, collisionIndex);
        collision.impulseNormal = normal;
        collision.separation = Scale(
                triangleNormal,
                Dot(penetration, triangleNormal));
        collision.contactPoint = point;
        collision.extraNegated = triangleNormal;
        return 1;
    }

    __device__ int Collide(const GmVec3 vertices[3]) {
        const float planeDistance = Dot(
                Subtract(center, vertices[0]), triangleNormal);
        if (radius < planeDistance || planeDistance < 0.0f) {
            return 0;
        }
        const float edgeReach = exact::Sqrt(
                radius * radius -
                planeDistance * planeDistance);
        const GmVec3 projected = Add(
                center,
                Scale(triangleNormal, -planeDistance));
        for (std::uint32_t edge = 0u; edge < 3u; ++edge) {
            const std::uint32_t next =
                    edge == 2u ? 0u : edge + 1u;
            const GmVec3 start = vertices[edge];
            const GmVec3 end = vertices[next];
            const GmVec3 direction = Normalize(
                    Subtract(end, start),
                    DirectionEpsilonSquared);
            const GmVec3 edgeNormal =
                    Cross(direction, triangleNormal);
            const float edgeDistance = Dot(
                    Subtract(projected, start), edgeNormal);
            if (edgeReach < edgeDistance) return 0;
            if (edgeDistance > 0.0f) {
                const float fromStart = Dot(
                        Subtract(projected, start), direction);
                if (fromStart < 0.0f) {
                    return EmitFeature(
                            start, DirectionEpsilonSquared, true);
                }
                const float fromEnd = Dot(
                        Subtract(projected, end), direction);
                if (!(0.0f < fromEnd)) {
                    return EmitFeature(
                            Add(projected,
                                Scale(edgeNormal, -edgeDistance)),
                            CollisionDistance, false);
                }
                return EmitEndpointB(
                        end, DirectionEpsilonSquared);
            }
        }
        if (planeDistance > 0.0f) {
            const std::uint32_t collisionIndex = AddCollision();
            if (collisionIndex == UINT32_MAX) return 0;
            decltype(auto) collision =
                    ShapeCollisionAt(scratch, collisionIndex);
            collision.impulseNormal = triangleNormal;
            collision.separation = Scale(
                    triangleNormal, planeDistance - radius);
            collision.contactPoint = projected;
            collision.sphereMergePrimary = true;
            collision.extraNegated = triangleNormal;
            return 1;
        }
        return 0;
    }
};

template <typename Scratch>
__device__ inline void TransformNewCollisions(
        Scratch &scratch,
        std::uint32_t firstNew,
        const GmIso4 &contactToWorld,
        const GmIso4 &normalToWorld,
        std::uint32_t materialA,
        std::uint32_t materialB,
        std::uint32_t movingShapeIndex,
        std::uint32_t staticSurfaceIndex,
        std::uint32_t staticActorIndex) {
    for (std::uint32_t index = firstNew;
         index < scratch.shapeCollisionCount; ++index) {
        decltype(auto) collision =
                ShapeCollisionAt(scratch, index);
        collision.materialA = materialA;
        collision.materialB = materialB;
        collision.movingShapeIndex = movingShapeIndex;
        collision.staticSurfaceIndex = staticSurfaceIndex;
        collision.staticActorIndex = staticActorIndex;
        collision.contactPoint = TransformPoint(
                contactToWorld, collision.contactPoint);
        collision.impulseNormal = Normalize(
                TransformEllipsoidNormal(
                        collision.impulseNormal,
                        normalToWorld.rotation),
                DirectionEpsilonSquared);
        collision.separation = TransformDirection(
                contactToWorld.rotation, collision.separation);
    }
}

template <
        bool TrackDiagnostics,
        bool CollectHotPathMetrics = false,
        typename Scratch>
__device__ inline int SphereMesh(
        const CudaPackedSceneHeader *scene,
        const CudaPackedStaticConfigurationHeader *configuration,
        const CudaSceneSurface &surface,
        std::uint32_t surfaceIndex,
        std::uint32_t actorIndex,
        const CudaVehicleCollisionShape &shape,
        std::uint32_t shapeIndex,
        const GmIso4 &shapeWorld,
        Scratch &scratch,
        CudaHotPathCounters *hotPathCounters = nullptr) {
    const float radius = shape.localBounds.halfExtents.y;
    const GmIso4 sphereToMesh =
            Compose(shapeWorld, surface.worldToLocal);
    const GmBoxAligned sphereBox = TransformBox(
            {{0.0f, 0.0f, 0.0f}, {radius, radius, radius}},
            sphereToMesh);
    const GmVec3 sphereCenterMesh = sphereToMesh.translation;
    if constexpr (TrackDiagnostics) {
        if (scratch.firstVisitedSurface == UINT32_MAX) {
            scratch.firstVisitedShape = shape.archiveOrder;
            scratch.firstVisitedSurface = surfaceIndex;
            scratch.firstShapeWorld = shapeWorld;
            scratch.firstEllipsoidBox = sphereBox;
            scratch.firstSurfaceWorldBounds = surface.worldBounds;
        }
    }
    const CudaSceneTriangle *triangles =
            SceneSection<CudaSceneTriangle>(
                    scene, scene->triangles);
    const CudaSceneOctreeCell *cells =
            SceneSection<CudaSceneOctreeCell>(
                    scene, scene->octreeCells);
    int hit = 0;
    std::uint32_t cell = 0u;
    while (cell < surface.octreeCellCount) {
        if constexpr (CollectHotPathMetrics) {
            ++hotPathCounters->octreeCellVisitCount;
        }
        if constexpr (TrackDiagnostics) {
            ++scratch.meshCellVisits;
        }
        const CudaSceneOctreeCell &entry =
                cells[surface.firstOctreeCell + cell];
        if (!BoundsIntersect(sphereBox, entry.bounds)) {
            cell += entry.subtreeEntryCount;
            continue;
        }
        if constexpr (TrackDiagnostics) {
            ++scratch.meshCellIntersections;
        }
        ++cell;
        if (!entry.containsTriangle ||
            entry.triangleIndex >= surface.triangleCount) {
            continue;
        }
        if constexpr (TrackDiagnostics) {
            ++scratch.meshTriangleCells;
            ++scratch.triangleTests;
        }
        if constexpr (CollectHotPathMetrics) {
            ++hotPathCounters->triangleTestCount;
        }
        const CudaSceneTriangle &triangle =
                triangles[surface.firstTriangle +
                          entry.triangleIndex];
        const GmVec3 triangleVertices[3] = {
                triangle.vertices[0],
                triangle.vertices[1],
                triangle.vertices[2],
        };
        const std::uint32_t firstNew =
                scratch.shapeCollisionCount;
        UnitSphereTriangleQuery<Scratch> query{
                scratch,
                sphereCenterMesh,
                radius,
                triangle.normal,
        };
        if (query.Collide(triangleVertices)) {
            if constexpr (CollectHotPathMetrics) {
                ++hotPathCounters->triangleHitCount;
            }
            if constexpr (TrackDiagnostics) {
                ++scratch.triangleHits;
            }
            const std::uint32_t materialA =
                    shape.wheelIndex != UINT32_MAX &&
                                    configuration->tuning.
                                                    contactResponse.
                                            singleMaterial <
                                            EPlugSurfaceMaterialId_Count
                    ? static_cast<std::uint32_t>(
                              configuration->tuning.
                                      contactResponse.
                                      singleMaterial)
                    : shape.surfaceMaterial;
            const std::uint32_t materialB =
                    SurfaceMaterial(
                            scene, surface,
                            triangle.material);
            for (std::uint32_t index = firstNew;
                 index < scratch.shapeCollisionCount; ++index) {
                decltype(auto) collision =
                        ShapeCollisionAt(scratch, index);
                collision.materialA = materialA;
                collision.materialB = materialB;
                collision.movingShapeIndex = shapeIndex;
                collision.staticSurfaceIndex = surfaceIndex;
                collision.staticActorIndex = actorIndex;
                collision.impulseNormal = TransformDirection(
                        surface.localToWorld.rotation,
                        collision.impulseNormal);
                collision.separation = TransformDirection(
                        surface.localToWorld.rotation,
                        collision.separation);
                collision.contactPoint = TransformPoint(
                        surface.localToWorld,
                        collision.contactPoint);
            }
            hit = 1;
        }
        if (scratch.overflow) return hit;
    }
    return hit;
}

template <
        bool TrackDiagnostics,
        bool UseMeshCellCache = false,
        bool CollectHotPathMetrics = false,
        typename Scratch>
__device__ inline int EllipsoidMesh(
        const CudaPackedSceneHeader *scene,
        const CudaPackedStaticConfigurationHeader *configuration,
        const CudaSceneSurface &surface,
        std::uint32_t surfaceIndex,
        std::uint32_t actorIndex,
        const CudaVehicleCollisionShape &shape,
        std::uint32_t shapeIndex,
        const GmIso4 &shapeWorld,
        Scratch &scratch,
        std::uint32_t cachedCellFirst = 0u,
        std::uint32_t cachedCellCount = 0u,
        CudaHotPathCounters *hotPathCounters = nullptr) {
    const GmVec3 radii = shape.localBounds.halfExtents;
    const GmVec3 inverseRadii = {
            1.0f / radii.x,
            1.0f / radii.y,
            1.0f / radii.z,
    };
    const GmIso4 ellipsoidToMesh =
            Compose(shapeWorld, surface.worldToLocal);
    const GmBoxAligned ellipsoidBox = TransformBox(
            {{0.0f, 0.0f, 0.0f}, radii},
            ellipsoidToMesh);
    if constexpr (TrackDiagnostics) {
        if (scratch.firstVisitedSurface == UINT32_MAX) {
            scratch.firstVisitedShape = shape.archiveOrder;
            scratch.firstVisitedSurface = surfaceIndex;
            scratch.firstShapeWorld = shapeWorld;
            scratch.firstEllipsoidBox = ellipsoidBox;
            scratch.firstSurfaceWorldBounds = surface.worldBounds;
        }
    }
    const GmIso4 meshToEllipsoid = Inverse(ellipsoidToMesh);
    GmIso4 meshToUnit = meshToEllipsoid;
    ScaleRows(meshToUnit, inverseRadii);
    GmIso4 contactToWorld = DiagonalTransform(
            radii, {0.0f, 0.0f, 0.0f});
    contactToWorld =
            MultInverse(contactToWorld, meshToEllipsoid);
    contactToWorld =
            Compose(contactToWorld, surface.localToWorld);
    GmIso4 normalToWorld = DiagonalTransform(
            inverseRadii, {0.0f, 0.0f, 0.0f});
    normalToWorld =
            MultInverse(normalToWorld, meshToEllipsoid);
    normalToWorld =
            Compose(normalToWorld, surface.localToWorld);
    const CudaSceneTriangle *triangles =
            SceneSection<CudaSceneTriangle>(
                    scene, scene->triangles);
    const CudaSceneOctreeCell *cells =
            SceneSection<CudaSceneOctreeCell>(
                    scene, scene->octreeCells);
    if constexpr (TrackDiagnostics) {
        if (scratch.firstVisitedSurface == surfaceIndex &&
            surface.octreeCellCount != 0u) {
            scratch.firstMeshRootBounds =
                    cells[surface.firstOctreeCell].bounds;
        }
    }
    std::uint32_t cell = 0u;
    std::uint32_t cachedCell = 0u;
    int hit = 0;
    while (UseMeshCellCache
                   ? cachedCell < cachedCellCount
                   : cell < surface.octreeCellCount) {
        if constexpr (CollectHotPathMetrics) {
            if constexpr (UseMeshCellCache) {
                ++hotPathCounters->cachedTriangleLeafVisitCount;
            } else {
                ++hotPathCounters->octreeCellVisitCount;
            }
        }
        if constexpr (TrackDiagnostics) {
            ++scratch.meshCellVisits;
        }
        std::uint32_t cellIndex = cell;
        if constexpr (UseMeshCellCache) {
            cellIndex = MeshCellAt(
                    scratch, cachedCellFirst + cachedCell);
            ++cachedCell;
        }
        const CudaSceneOctreeCell &entry =
                cells[surface.firstOctreeCell + cellIndex];
        if (!BoundsIntersect(ellipsoidBox, entry.bounds)) {
            if constexpr (!UseMeshCellCache) {
                cell += entry.subtreeEntryCount;
            }
            continue;
        }
        if constexpr (TrackDiagnostics) {
            ++scratch.meshCellIntersections;
        }
        if constexpr (!UseMeshCellCache) {
            ++cell;
        }
        if constexpr (!UseMeshCellCache) {
            if (!entry.containsTriangle ||
                entry.triangleIndex >= surface.triangleCount) {
                continue;
            }
        }
        if constexpr (TrackDiagnostics) {
            ++scratch.meshTriangleCells;
        }
        const CudaSceneTriangle &triangle =
                triangles[surface.firstTriangle +
                          entry.triangleIndex];
        if constexpr (TrackDiagnostics) {
            ++scratch.triangleTests;
        }
        if constexpr (CollectHotPathMetrics) {
            ++hotPathCounters->triangleTestCount;
        }
        const GmVec3 unitVertices[3] = {
                TransformPoint(
                        meshToUnit,
                        triangle.vertices[0]),
                TransformPoint(
                        meshToUnit,
                        triangle.vertices[1]),
                TransformPoint(
                        meshToUnit,
                        triangle.vertices[2]),
        };
        const GmVec3 edge01 =
                Subtract(unitVertices[1], unitVertices[0]);
        const GmVec3 edge02 =
                Subtract(unitVertices[2], unitVertices[0]);
        const float normalX =
                edge02.z * edge01.y - edge02.y * edge01.z;
        const float normalY =
                edge01.z * edge02.x - edge02.z * edge01.x;
        const float normalZ =
                edge01.x * edge02.y - edge02.x * edge01.y;
        const float normalLengthSquared =
                (normalY * normalY + normalX * normalX) +
                normalZ * normalZ;
        if (!(normalLengthSquared >
              DirectionEpsilonSquared)) {
            continue;
        }
        const float normalLength =
                exact::Sqrt(normalLengthSquared);
        const float inverseNormalLength = 1.0f / normalLength;
        const GmVec3 triangleNormal = {
                normalX * inverseNormalLength,
                normalY * inverseNormalLength,
                inverseNormalLength * normalZ,
        };
        const std::uint32_t firstNew =
                scratch.shapeCollisionCount;
        UnitSphereTriangleQuery<Scratch> query{
                scratch,
                {},
                1.0f,
                triangleNormal,
        };
        if (query.Collide(unitVertices)) {
            if constexpr (CollectHotPathMetrics) {
                ++hotPathCounters->triangleHitCount;
            }
            if constexpr (TrackDiagnostics) {
                ++scratch.triangleHits;
            }
            TransformNewCollisions(
                    scratch, firstNew,
                    contactToWorld, normalToWorld,
                    shape.wheelIndex != UINT32_MAX &&
                                    configuration->tuning.
                                                    contactResponse.
                                            singleMaterial <
                                            EPlugSurfaceMaterialId_Count
                    ? static_cast<std::uint32_t>(
                              configuration->tuning.
                                      contactResponse.
                                      singleMaterial)
                    : shape.surfaceMaterial,
                    SurfaceMaterial(
                            scene, surface,
                            triangle.material),
                    shapeIndex,
                    surfaceIndex,
                    actorIndex);
            hit = 1;
        }
        if (scratch.overflow) return hit;
    }
    return hit;
}

#if 0
struct EllipsoidMeshContext {
    GmBoxAligned bounds;
    GmIso4 meshToUnit;
    GmIso4 contactToWorld;
    GmIso4 normalToWorld;
};

__device__ inline EllipsoidMeshContext PrepareEllipsoidMeshContext(
        const CudaSceneSurface &surface,
        const CudaVehicleCollisionShape &shape,
        const GmIso4 &shapeWorld) {
    const GmVec3 radii = shape.localBounds.halfExtents;
    const GmVec3 inverseRadii = {
            1.0f / radii.x,
            1.0f / radii.y,
            1.0f / radii.z,
    };
    const GmIso4 ellipsoidToMesh =
            Compose(shapeWorld, surface.worldToLocal);
    const GmBoxAligned bounds = TransformBox(
            {{0.0f, 0.0f, 0.0f}, radii},
            ellipsoidToMesh);
    const GmIso4 meshToEllipsoid = Inverse(ellipsoidToMesh);
    GmIso4 meshToUnit = meshToEllipsoid;
    ScaleRows(meshToUnit, inverseRadii);
    GmIso4 contactToWorld = DiagonalTransform(
            radii, {0.0f, 0.0f, 0.0f});
    contactToWorld =
            MultInverse(contactToWorld, meshToEllipsoid);
    contactToWorld =
            Compose(contactToWorld, surface.localToWorld);
    GmIso4 normalToWorld = DiagonalTransform(
            inverseRadii, {0.0f, 0.0f, 0.0f});
    normalToWorld =
            MultInverse(normalToWorld, meshToEllipsoid);
    normalToWorld =
            Compose(normalToWorld, surface.localToWorld);
    return {
            bounds,
            meshToUnit,
            contactToWorld,
            normalToWorld,
    };
}

template<typename Scratch>
__device__ inline bool TestEllipsoidMeshTriangle(
        const EllipsoidMeshContext &context,
        const GmVec3 (&meshVertices)[3],
        std::uint32_t materialA,
        std::uint32_t materialB,
        std::uint32_t shapeIndex,
        std::uint32_t surfaceIndex,
        std::uint32_t actorIndex,
        Scratch &scratch) {
    const GmVec3 unitVertices[3] = {
            TransformPoint(context.meshToUnit, meshVertices[0]),
            TransformPoint(context.meshToUnit, meshVertices[1]),
            TransformPoint(context.meshToUnit, meshVertices[2]),
    };
    const GmVec3 edge01 =
            Subtract(unitVertices[1], unitVertices[0]);
    const GmVec3 edge02 =
            Subtract(unitVertices[2], unitVertices[0]);
    const float normalX =
            edge02.z * edge01.y - edge02.y * edge01.z;
    const float normalY =
            edge01.z * edge02.x - edge02.z * edge01.x;
    const float normalZ =
            edge01.x * edge02.y - edge02.x * edge01.y;
    const float normalLengthSquared =
            (normalY * normalY + normalX * normalX) +
            normalZ * normalZ;
    if (!(normalLengthSquared > DirectionEpsilonSquared)) {
        return false;
    }
    const float normalLength =
            exact::Sqrt(normalLengthSquared);
    const float inverseNormalLength = 1.0f / normalLength;
    const GmVec3 triangleNormal = {
            normalX * inverseNormalLength,
            normalY * inverseNormalLength,
            inverseNormalLength * normalZ,
    };
    const std::uint32_t firstNew =
            scratch.shapeCollisionCount;
    UnitSphereTriangleQuery<Scratch> query{
            scratch,
            {},
            1.0f,
            triangleNormal,
    };
    if (!query.Collide(unitVertices)) {
        return false;
    }
    TransformNewCollisions(
            scratch,
            firstNew,
            context.contactToWorld,
            context.normalToWorld,
            materialA,
            materialB,
            shapeIndex,
            surfaceIndex,
            actorIndex);
    return true;
}

__device__ inline void EllipsoidMeshPairCached(
        const CudaPackedSceneHeader *scene,
        const CudaPackedStaticConfigurationHeader *configuration,
        const CudaSceneSurface &surface,
        std::uint32_t surfaceIndex,
        std::uint32_t actorIndex,
        const CudaVehicleCollisionShape &firstShape,
        std::uint32_t firstShapeIndex,
        const GmIso4 &firstShapeWorld,
        const CudaVehicleCollisionShape &secondShape,
        std::uint32_t secondShapeIndex,
        const GmIso4 &secondShapeWorld,
        CudaCollisionSearchTile *primaryStorage,
        CudaCollisionSearchTile *secondaryStorage,
        std::uint32_t *secondCollisionCount,
        CudaCollisionSearchScratch &scratch,
        std::uint32_t cachedCellFirst,
        std::uint32_t cachedCellCount) {
    const EllipsoidMeshContext first =
            PrepareEllipsoidMeshContext(
                    surface, firstShape, firstShapeWorld);
    const EllipsoidMeshContext second =
            PrepareEllipsoidMeshContext(
                    surface, secondShape, secondShapeWorld);
    const CudaSceneTriangle *triangles =
            SceneSection<CudaSceneTriangle>(
                    scene, scene->triangles);
    const CudaSceneOctreeCell *cells =
            SceneSection<CudaSceneOctreeCell>(
                    scene, scene->octreeCells);
    const std::uint32_t firstMaterial =
            firstShape.wheelIndex != UINT32_MAX &&
                            configuration->tuning.
                                            contactResponse.
                                    singleMaterial <
                                    EPlugSurfaceMaterialId_Count
                    ? static_cast<std::uint32_t>(
                              configuration->tuning.
                                      contactResponse.
                                      singleMaterial)
                    : firstShape.surfaceMaterial;
    const std::uint32_t secondMaterial =
            secondShape.wheelIndex != UINT32_MAX &&
                            configuration->tuning.
                                            contactResponse.
                                    singleMaterial <
                                    EPlugSurfaceMaterialId_Count
                    ? static_cast<std::uint32_t>(
                              configuration->tuning.
                                      contactResponse.
                                      singleMaterial)
                    : secondShape.surfaceMaterial;
    for (std::uint32_t cachedCell = 0u;
         cachedCell < cachedCellCount;
         ++cachedCell) {
        const std::uint32_t cellIndex = MeshCellAt(
                scratch, cachedCellFirst + cachedCell);
        const CudaSceneOctreeCell &entry =
                cells[surface.firstOctreeCell + cellIndex];
        const bool firstIntersects =
                BoundsIntersect(first.bounds, entry.bounds);
        const bool secondIntersects =
                BoundsIntersect(second.bounds, entry.bounds);
        if (!firstIntersects && !secondIntersects) {
            continue;
        }
        const CudaSceneTriangle &triangle =
                triangles[surface.firstTriangle +
                          entry.triangleIndex];
        const GmVec3 meshVertices[3] = {
                triangle.vertices[0],
                triangle.vertices[1],
                triangle.vertices[2],
        };
        const std::uint32_t materialB =
                SurfaceMaterial(
                        scene, surface, triangle.material);
        if (firstIntersects) {
            TestEllipsoidMeshTriangle(
                    first,
                    meshVertices,
                    firstMaterial,
                    materialB,
                    firstShapeIndex,
                    surfaceIndex,
                    actorIndex,
                    scratch);
        }
        if (secondIntersects) {
            const std::uint32_t firstCollisionCount =
                    scratch.shapeCollisionCount;
            scratch.shapeCollisionStorage = secondaryStorage;
            scratch.shapeCollisionCount =
                    *secondCollisionCount;
            TestEllipsoidMeshTriangle(
                    second,
                    meshVertices,
                    secondMaterial,
                    materialB,
                    secondShapeIndex,
                    surfaceIndex,
                    actorIndex,
                    scratch);
            *secondCollisionCount =
                    scratch.shapeCollisionCount;
            scratch.shapeCollisionStorage = primaryStorage;
            scratch.shapeCollisionCount =
                    firstCollisionCount;
        }
        if (scratch.overflow) {
            return;
        }
    }
}
#endif

// Preserve octree preorder while caching only triangle leaves. Each live
// shape still applies the authoritative bounds and triangle tests, so its
// contacts remain an ordered subset of this conservative union query.
template<
        bool UnifiedBounds = false,
        bool CollectHotPathMetrics = false>
__device__ inline void BuildMeshCellCache(
        const CudaPackedSceneHeader *scene,
        const CudaSceneSurface *surfaces,
        std::uint32_t collisionShapeCount,
        CudaCollisionSearchScratch &scratch,
        CudaHotPathCounters *hotPathCounters = nullptr) {
    const CudaSceneOctreeCell *cells =
            SceneSection<CudaSceneOctreeCell>(
                    scene, scene->octreeCells);
    scratch.meshCellCount = 0u;
    scratch.meshCacheValid = true;
    for (std::uint32_t hitIndex = 0u;
         hitIndex < scratch.surfaceHitCount;
         ++hitIndex) {
        CudaCollisionMeshRange &range =
                MeshRangeAt(scratch, hitIndex);
        range = {
                static_cast<std::uint16_t>(
                        scratch.meshCellCount),
                0u};
        const CudaCollisionSurfaceHit hit =
                SurfaceHitAt(scratch, hitIndex);
        const CudaSceneSurface &surface =
                surfaces[hit.surfaceIndex];
        if (surface.type != static_cast<std::uint32_t>(
                    GmSurf::EGmSurfType::Mesh)) {
            scratch.meshCacheValid = false;
            return;
        }
        GmBoxAligned worldBounds;
        bool hasBounds = false;
        if constexpr (UnifiedBounds) {
            worldBounds = UnifiedMovingBoundsAt(scratch);
            hasBounds = true;
        } else {
            for (std::uint32_t traversal = 0u;
                 traversal < collisionShapeCount;
                 ++traversal) {
                if ((hit.shapeMask & (1u << traversal)) == 0u) {
                    continue;
                }
                const GmBoxAligned shapeBounds =
                        MovingBoundsAt(scratch, traversal);
                worldBounds = hasBounds
                        ? IncludeBounds(worldBounds, shapeBounds)
                        : shapeBounds;
                hasBounds = true;
            }
        }
        if (!hasBounds) {
            continue;
        }
        GmBoxAligned localBounds =
                TransformBox(worldBounds, surface.worldToLocal);
        ExpandBoundsForRounding(localBounds);
        std::uint32_t cell = 0u;
        while (cell < surface.octreeCellCount) {
            if constexpr (CollectHotPathMetrics) {
                ++hotPathCounters->octreeCellVisitCount;
            }
            const std::uint32_t cellIndex = cell;
            const CudaSceneOctreeCell &entry =
                    cells[surface.firstOctreeCell + cellIndex];
            if (!BoundsIntersect(localBounds, entry.bounds)) {
                cell += entry.subtreeEntryCount;
                continue;
            }
            ++cell;
            if (!entry.containsTriangle ||
                entry.triangleIndex >= surface.triangleCount) {
                continue;
            }
            if (scratch.meshCellCount >=
                MeshCellHitCapacity) {
                scratch.meshCellCount = 0u;
                scratch.meshCacheValid = false;
                return;
            }
            MeshCellAt(scratch, scratch.meshCellCount++) =
                    cellIndex;
            ++range.count;
        }
    }
}

__device__ inline bool NearlyEqual(
        float value, float reference) {
    const float tolerance = fabsf(reference) * 1.0e-5f;
    return reference - tolerance <= value &&
           value <= reference + tolerance;
}

__device__ inline bool NearlyEqual(
        const GmVec3 &left, const GmVec3 &right) {
    return NearlyEqual(left.x, right.x) &&
           NearlyEqual(left.y, right.y) &&
           NearlyEqual(left.z, right.z);
}

template <bool CollectHotPathMetrics = false, typename Scratch>
__device__ inline void MergeShapeContacts(
        Scratch &scratch,
        CudaHotPathCounters *hotPathCounters = nullptr) {
    if constexpr (CollectHotPathMetrics) {
        hotPathCounters->rawContactCount +=
                scratch.shapeCollisionCount;
    }
    const std::uint32_t firstTarget =
            scratch.collisionCount;
    for (std::uint32_t index = 0u;
         index < scratch.shapeCollisionCount; ++index) {
        if (ShapeCollisionAt(
                    scratch, index).sphereMergePrimary) {
            AddMain(scratch, ShapeCollisionAt(scratch, index));
        }
    }
    const std::uint32_t targetAfterPrimaries =
            scratch.collisionCount;
    for (std::uint32_t index = 0u;
         index < scratch.shapeCollisionCount; ++index) {
        decltype(auto) collision =
                ShapeCollisionAt(scratch, index);
        if (collision.sphereMergePrimary) continue;
        std::uint32_t target = firstTarget;
        for (; target < targetAfterPrimaries; ++target) {
            decltype(auto) primary =
                    CollisionAt(scratch, target);
            if (NearlyEqual(
                        collision.extraNegated,
                        primary.extraNegated) ||
                SphereNormalAlignment <
                        Dot(collision.impulseNormal,
                            primary.impulseNormal)) {
                break;
            }
        }
        if (target == targetAfterPrimaries) {
            AddMain(scratch, collision);
        }
    }
    scratch.shapeCollisionCount = 0u;
}

__device__ inline GmIso4 BodyPose(
        const CudaDynamicBodyState &body) {
    return {body.current.rotation, body.current.position};
}

#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES)
__device__ inline GmIso4 RootShapeWorldPose(
        std::uint32_t shapeIndex,
        const CudaVehicleCollisionShape *shapes,
        const CudaCandidatePhysicsState &candidate,
        const GmIso4 &bodyPose) {
    GmIso4 shapeBodyPose;
    if (shapeIndex < 4u) {
        shapeBodyPose = shapes[shapeIndex].bodyPose;
    } else {
        const std::uint32_t wheelIndex =
                (shapeIndex - 4u) ^
                (shapeIndex >= 6u ? 1u : 0u);
        shapeBodyPose =
                candidate.vehicle.wheels.values[
                        wheelIndex].currentPose;
    }
    return Compose(shapeBodyPose, bodyPose);
}
#endif

__device__ inline GmIso4 ShapeBodyPose(
        const CudaVehicleCollisionShape &shape,
        const CudaCandidatePhysicsState &candidate) {
    if (shape.wheelIndex == UINT32_MAX) {
        return shape.bodyPose;
    }
    if (shape.wheelIndex < facts::WheelCount(candidate.vehicle)) {
        return candidate.vehicle.wheels.values[
                shape.wheelIndex].currentPose;
    }
    return shape.bodyPose;
}

__device__ inline GmIso4 ShapeWorldPose(
        std::uint32_t shapeIndex,
        const CudaVehicleCollisionShape *shapes,
        const CudaCandidatePhysicsState &candidate,
        const GmIso4 &bodyPose) {
    const CudaVehicleCollisionShape &shape = shapes[shapeIndex];
    if (shape.parentShapeIndex == UINT32_MAX ||
        shape.wheelIndex != UINT32_MAX) {
        return Compose(
                ShapeBodyPose(shape, candidate), bodyPose);
    }

    std::uint32_t depth = 1u;
    for (std::uint32_t parent = shape.parentShapeIndex;
         parent != UINT32_MAX;
         parent = shapes[parent].parentShapeIndex) {
        ++depth;
    }
    GmIso4 world = bodyPose;
    while (depth != 0u) {
        std::uint32_t node = shapeIndex;
        for (std::uint32_t climb = 1u;
             climb < depth; ++climb) {
            node = shapes[node].parentShapeIndex;
        }
        const CudaVehicleCollisionShape &entry = shapes[node];
        if (entry.wheelIndex != UINT32_MAX) {
            world = Compose(
                    ShapeBodyPose(entry, candidate), bodyPose);
        } else {
            world = Compose(entry.localPose, world);
        }
        --depth;
    }
    return world;
}

template<typename Scratch>
__device__ inline GmIso4 CachedShapeWorldPose(
        std::uint32_t shapeIndex,
        const CudaVehicleCollisionShape *shapes,
        const CudaCandidatePhysicsState &candidate,
        const GmIso4 &bodyPose,
        const Scratch &scratch) {
    const CudaVehicleCollisionShape &shape =
            shapes[shapeIndex];
    if (shape.wheelIndex == UINT32_MAX &&
        shape.parentShapeIndex < shapeIndex &&
        shapes[shape.parentShapeIndex].traversalOrder ==
                shape.parentShapeIndex) {
        return Compose(
                shape.localPose,
                ShapeWorldAt(
                        scratch, shape.parentShapeIndex));
    }
    return ShapeWorldPose(
            shapeIndex, shapes, candidate, bodyPose);
}

template <typename Left, typename Right>
__device__ inline int CompareForResponse(
        const Left &left,
        const Right &right) {
    const float leftValues[] = {
            left.contactPoint.x,
            left.contactPoint.y,
            left.contactPoint.z,
            left.impulseNormal.x,
            left.impulseNormal.y,
            left.impulseNormal.z,
            left.separation.x,
            left.separation.y,
            left.separation.z,
    };
    const float rightValues[] = {
            right.contactPoint.x,
            right.contactPoint.y,
            right.contactPoint.z,
            right.impulseNormal.x,
            right.impulseNormal.y,
            right.impulseNormal.z,
            right.separation.x,
            right.separation.y,
            right.separation.z,
    };
    for (std::uint32_t index = 0u; index < 9u; ++index) {
        const float leftValue = leftValues[index];
        const float rightValue = rightValues[index];
        if (!(rightValue <= leftValue)) return 1;
        if (rightValue < leftValue) return -1;
    }
    if (!left.sphereMergePrimary &&
        right.sphereMergePrimary) {
        return -1;
    }
    return 1;
}

template <bool CollectHotPathMetrics = false, typename Scratch>
__device__ inline void SortForResponse(
        Scratch &scratch,
        CudaHotPathCounters *hotPathCounters = nullptr) {
    if constexpr (CollectHotPathMetrics) {
        ++hotPathCounters->responseSortCallCount;
        hotPathCounters->responseSortItemCount +=
                scratch.collisionCount;
        if (scratch.collisionCount >
            hotPathCounters->maximumResponseSortItemCount) {
            hotPathCounters->maximumResponseSortItemCount =
                    scratch.collisionCount;
        }
    }
    constexpr std::uint32_t Cutoff = 8u;
    constexpr std::uint32_t StackSize = 30u;
    InitializeResponseOrder(scratch);
    if (scratch.collisionCount < 2u) return;
    std::uint32_t lowStack[StackSize]{};
    std::uint32_t highStack[StackSize]{};
    std::uint32_t stackDepth = 0u;
    std::uint32_t low = 0u;
    std::uint32_t high = scratch.collisionCount - 1u;
    for (;;) {
        const std::uint32_t count = high - low + 1u;
        if (count <= Cutoff) {
            while (high > low) {
                std::uint32_t selected = low;
                for (std::uint32_t cursor = low + 1u;
                     cursor <= high; ++cursor) {
                    if (CompareForResponse(
                                OrderedCollisionAt(scratch, cursor),
                                OrderedCollisionAt(scratch, selected)) > 0) {
                        selected = cursor;
                    }
                }
                if (selected != high) {
                    SwapOrdered(scratch, selected, high);
                }
                --high;
            }
        } else {
            std::uint32_t middle = low + count / 2u;
            if (CompareForResponse(
                        OrderedCollisionAt(scratch, low),
                        OrderedCollisionAt(scratch, middle)) > 0) {
                SwapOrdered(scratch, low, middle);
            }
            if (CompareForResponse(
                        OrderedCollisionAt(scratch, low),
                        OrderedCollisionAt(scratch, high)) > 0) {
                SwapOrdered(scratch, low, high);
            }
            if (CompareForResponse(
                        OrderedCollisionAt(scratch, middle),
                        OrderedCollisionAt(scratch, high)) > 0) {
                SwapOrdered(scratch, middle, high);
            }
            std::uint32_t lowCursor = low;
            std::uint32_t highCursor = high;
            for (;;) {
                if (middle > lowCursor) {
                    do {
                        ++lowCursor;
                    } while (
                            lowCursor < middle &&
                            CompareForResponse(
                                    OrderedCollisionAt(
                                            scratch, lowCursor),
                                    OrderedCollisionAt(
                                            scratch, middle)) <= 0);
                }
                if (middle <= lowCursor) {
                    do {
                        ++lowCursor;
                    } while (
                            lowCursor <= high &&
                            CompareForResponse(
                                    OrderedCollisionAt(
                                            scratch, lowCursor),
                                    OrderedCollisionAt(
                                            scratch, middle)) <= 0);
                }
                do {
                    --highCursor;
                } while (
                        highCursor > middle &&
                        CompareForResponse(
                                OrderedCollisionAt(
                                        scratch, highCursor),
                                OrderedCollisionAt(
                                        scratch, middle)) > 0);
                if (highCursor < lowCursor) break;
                SwapOrdered(scratch, lowCursor, highCursor);
                if (middle == highCursor) {
                    middle = lowCursor;
                } else if (middle == lowCursor) {
                    middle = highCursor;
                }
            }
            ++highCursor;
            if (middle < highCursor) {
                do {
                    --highCursor;
                } while (
                        highCursor > middle &&
                        CompareForResponse(
                                OrderedCollisionAt(
                                        scratch, highCursor),
                                OrderedCollisionAt(
                                        scratch, middle)) == 0);
            }
            if (middle >= highCursor) {
                do {
                    --highCursor;
                } while (
                        highCursor > low &&
                        CompareForResponse(
                                OrderedCollisionAt(
                                        scratch, highCursor),
                                OrderedCollisionAt(
                                        scratch, middle)) == 0);
            }
            const std::uint32_t leftSpan =
                    highCursor - low;
            const std::uint32_t rightSpan =
                    high - lowCursor;
            if (leftSpan >= rightSpan) {
                if (low < highCursor) {
                    if (stackDepth >= StackSize) {
                        scratch.overflow = true;
                        scratch.overflowReason =
                                OverflowReason::OrderingStackCapacity;
                        return;
                    }
                    lowStack[stackDepth] = low;
                    highStack[stackDepth] = highCursor;
                    ++stackDepth;
                }
                if (lowCursor < high) {
                    low = lowCursor;
                    continue;
                }
            } else {
                if (lowCursor < high) {
                    if (stackDepth >= StackSize) {
                        scratch.overflow = true;
                        scratch.overflowReason =
                                OverflowReason::OrderingStackCapacity;
                        return;
                    }
                    lowStack[stackDepth] = lowCursor;
                    highStack[stackDepth] = high;
                    ++stackDepth;
                }
                if (low < highCursor) {
                    high = highCursor;
                    continue;
                }
            }
        }
        if (stackDepth == 0u) return;
        --stackDepth;
        low = lowStack[stackDepth];
        high = highStack[stackDepth];
    }
}

}  // namespace detail

template <
        bool TrackDiagnostics = true,
        bool TrustedInputs = false,
        bool EightOrderedEllipsoids = false,
        bool WarpCoherentAcceleration = false,
        bool TriggerOnly = false,
        bool UseEmptyAirCertificate = false,
        bool CollectHotPathMetrics = false,
        typename Scratch = CudaCollisionScratch>
__device__ inline Status Detect(
        const CudaPackedSceneHeader *scene,
        const CudaPackedStaticConfigurationHeader *configuration,
        const CudaCandidatePhysicsState &candidate,
        Scratch &scratch,
        CudaHotPathCounters *hotPathCounters = nullptr,
        CudaEmptyAirCertificateState<UseEmptyAirCertificate>
                *emptyAirState = nullptr) {
    detail::Clear<TrackDiagnostics>(scratch);
    if constexpr (UseEmptyAirCertificate) {
        if (emptyAirState == nullptr) {
            return Status::InvalidScene;
        }
    }
    if constexpr (CollectHotPathMetrics) {
        ++hotPathCounters->collisionDetectCount;
    }
    if constexpr (!TrustedInputs) {
        if (scene == nullptr || configuration == nullptr ||
            !ValidCudaPackedSceneHeader(*scene) ||
            configuration->magic !=
                    CudaPackedStaticConfigurationHeader::Magic) {
            return Status::InvalidScene;
        }
    }
    const CudaSceneSurface *surfaces =
            detail::SceneSection<CudaSceneSurface>(
                    scene, scene->surfaces);
    const CudaSceneAccelerationCell *acceleration =
            detail::SceneSection<CudaSceneAccelerationCell>(
                    scene, scene->accelerationCells);
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
    const CudaVehicleCollisionShape *shapes =
            reinterpret_cast<const CudaVehicleCollisionShape *>(
                    research::
                            ForeverValidatorSessionCollisionShapeBytes());
#elif defined(FOREVERVALIDATOR_CUDA_RESEARCH_CONSTANT_COLLISION_SHAPES)
    const CudaVehicleCollisionShape *shapes =
            research::StaticCollisionShapes;
#else
    const CudaVehicleCollisionShape *shapes =
            tuning::Section<CudaVehicleCollisionShape>(
                    configuration,
                    configuration->collisionShapes);
#endif
    const GmIso4 bodyPose = detail::BodyPose(candidate.body);

    if constexpr (!TrackDiagnostics) {
        // A contained live bound can only visit an ordered subset of the
        // cached broad-phase query, so reusing its hits preserves exact
        // contact discovery and response ordering.
        constexpr std::uint32_t CachedShapeCount = 8u;
        constexpr float SurfaceCacheMargin = 0.0625f;
        constexpr float SurfaceCacheHorizonTicks = 1.0f;
        const std::uint32_t collisionShapeCount =
                EightOrderedEllipsoids
                ? 8u
                : configuration->collisionShapes.count;
        const float surfaceCacheHorizon =
                __int2float_rn(static_cast<std::int32_t>(
                        candidate.world.schemePeriodMs)) *
                (0.001f * SurfaceCacheHorizonTicks);
        const GmVec3 surfaceCacheTravel = detail::Scale(
                candidate.body.current.linearSpeed,
                surfaceCacheHorizon);
        bool useSurfaceCache =
                !TriggerOnly &&
                scratch.surfaceCacheEnabled &&
                (EightOrderedEllipsoids ||
                 collisionShapeCount == 5u ||
                 collisionShapeCount == CachedShapeCount) &&
                scratch.shapeCapacity >= collisionShapeCount;
        bool refreshSurfaceCache = !scratch.surfaceCacheValid;
        if (useSurfaceCache) {
            for (std::uint32_t traversal = 0u;
                 traversal < collisionShapeCount;
                 ++traversal) {
                const std::uint32_t shapeIndex = traversal;
                if constexpr (!EightOrderedEllipsoids) {
                    if (shapes[shapeIndex].traversalOrder != traversal ||
                        shapes[shapeIndex].surfaceType !=
                                static_cast<std::uint32_t>(
                                        GmSurf::EGmSurfType::
                                                Ellipsoid)) {
                        useSurfaceCache = false;
                        break;
                    }
                }
                GmIso4 shapeWorld;
                if constexpr (EightOrderedEllipsoids) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES)
                    shapeWorld = detail::RootShapeWorldPose(
                            shapeIndex, shapes,
                            candidate, bodyPose);
#else
                    const CudaVehicleCollisionShape &shape =
                            shapes[shapeIndex];
                    shapeWorld =
                            shape.wheelIndex == UINT32_MAX &&
                                    shape.parentShapeIndex < shapeIndex &&
                                    shapes[shape.parentShapeIndex].
                                                    traversalOrder ==
                                            shape.parentShapeIndex
                            ? detail::Compose(
                                      shape.localPose,
                                      detail::ShapeWorldAt(
                                              scratch,
                                              shape.parentShapeIndex))
                            : detail::ShapeWorldPose(
                                      shapeIndex, shapes,
                                      candidate, bodyPose);
#endif
                } else {
                    shapeWorld =
                            detail::CachedShapeWorldPose(
                                    shapeIndex, shapes,
                                    candidate, bodyPose, scratch);
                    detail::ShapeWorldAt(scratch, traversal) =
                            shapeWorld;
                }
                const GmBoxAligned movingBounds =
                        detail::TransformBox(
                                shapes[shapeIndex].localBounds,
                                shapeWorld);
                if constexpr (EightOrderedEllipsoids) {
                    if (!refreshSurfaceCache &&
                        !detail::BoundsContain(
                                detail::MovingBoundsAt(
                                        scratch, traversal),
                        movingBounds)) {
                        refreshSurfaceCache = true;
                    }
                } else {
                    if (!refreshSurfaceCache &&
                        !detail::BoundsContain(
                                detail::MovingBoundsAt(
                                        scratch, traversal),
                                movingBounds)) {
                        refreshSurfaceCache = true;
                        for (std::uint32_t previous = 0u;
                             previous < traversal;
                             ++previous) {
                            detail::MovingBoundsAt(
                                    scratch, previous) =
                                    detail::ExpandBoundsAlong(
                                            detail::TransformBox(
                                                    shapes[previous].
                                                            localBounds,
                                                    detail::ShapeWorldAt(
                                                            scratch,
                                                            previous)),
                                            surfaceCacheTravel,
                                            SurfaceCacheMargin);
                        }
                    }
                    if (refreshSurfaceCache) {
                        detail::MovingBoundsAt(
                                scratch, traversal) =
                                detail::ExpandBoundsAlong(
                                        movingBounds,
                                        surfaceCacheTravel,
                                        SurfaceCacheMargin);
                    }
                }
            }
            if constexpr (EightOrderedEllipsoids) {
                if (refreshSurfaceCache) {
                    bool hasUnifiedMovingBounds = false;
                    GmVec3 unifiedLower;
                    GmVec3 unifiedUpper;
                    for (std::uint32_t traversal = 0u;
                         traversal < collisionShapeCount;
                         ++traversal) {
                        GmIso4 shapeWorld;
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES)
                        shapeWorld =
                                detail::RootShapeWorldPose(
                                        traversal, shapes,
                                        candidate, bodyPose);
#else
                        shapeWorld =
                                detail::ShapeWorldPose(
                                        traversal, shapes,
                                        candidate, bodyPose);
#endif
                        const GmBoxAligned movingBounds =
                                detail::TransformBox(
                                        shapes[traversal].localBounds,
                                        shapeWorld);
                        GmBoxAligned cachedBounds =
                                detail::ExpandBoundsAlong(
                                        movingBounds,
                                        surfaceCacheTravel,
                                        SurfaceCacheMargin);
                        detail::ExpandBoundsForRounding(
                                cachedBounds);
                        detail::MovingBoundsAt(
                                scratch, traversal) = cachedBounds;
                        const GmVec3 lower = {
                                cachedBounds.center.x -
                                        cachedBounds.halfExtents.x,
                                cachedBounds.center.y -
                                        cachedBounds.halfExtents.y,
                                cachedBounds.center.z -
                                        cachedBounds.halfExtents.z,
                        };
                        const GmVec3 upper = {
                                cachedBounds.center.x +
                                        cachedBounds.halfExtents.x,
                                cachedBounds.center.y +
                                        cachedBounds.halfExtents.y,
                                cachedBounds.center.z +
                                        cachedBounds.halfExtents.z,
                        };
                        if (hasUnifiedMovingBounds) {
                            unifiedLower = {
                                    fminf(unifiedLower.x, lower.x),
                                    fminf(unifiedLower.y, lower.y),
                                    fminf(unifiedLower.z, lower.z),
                            };
                            unifiedUpper = {
                                    fmaxf(unifiedUpper.x, upper.x),
                                    fmaxf(unifiedUpper.y, upper.y),
                                    fmaxf(unifiedUpper.z, upper.z),
                            };
                        } else {
                            unifiedLower = lower;
                            unifiedUpper = upper;
                            hasUnifiedMovingBounds = true;
                        }
                    }
                    GmBoxAligned unifiedMovingBounds = {
                            detail::Scale(
                                    detail::Add(
                                            unifiedLower,
                                            unifiedUpper),
                                    0.5f),
                            detail::Scale(
                                    detail::Subtract(
                                            unifiedUpper,
                                            unifiedLower),
                                    0.5f),
                    };
                    detail::ExpandBoundsForRounding(
                            unifiedMovingBounds);
                    detail::StoreUnifiedMovingBounds(
                            scratch, unifiedMovingBounds);
                }
            }
        }
        if constexpr (UseEmptyAirCertificate) {
            if (!useSurfaceCache || refreshSurfaceCache) {
                detail::InvalidateEmptyAirCertificate<
                        CollectHotPathMetrics>(
                        *emptyAirState, hotPathCounters,
                        !useSurfaceCache);
            }
        }
        if constexpr (CollectHotPathMetrics) {
            if (useSurfaceCache) {
                ++hotPathCounters->surfaceCacheEligibleCount;
                if (refreshSurfaceCache) {
                    ++hotPathCounters->surfaceCacheRefreshCount;
                } else {
                    ++hotPathCounters->surfaceCacheReuseCount;
                }
            }
        }
        if (useSurfaceCache && refreshSurfaceCache) {
            scratch.meshCacheValid = false;
            scratch.surfaceHitCount = 0u;
            constexpr std::uint32_t TargetGroups[] = {1u, 3u, 4u};
            const std::uint32_t targetGroupCount =
                    TriggerOnly ? 1u : 3u;
            for (std::uint32_t groupIndex = 0u;
                 groupIndex < targetGroupCount; ++groupIndex) {
                const std::uint32_t group = TargetGroups[groupIndex];
                const CudaSceneAccelerationRange range =
                        scene->accelerationGroups[group - 1u];
                if (range.cellCount <= 1u) continue;
                if constexpr (EightOrderedEllipsoids) {
                    std::uint32_t cursor = 0u;
                    for (;;) {
                        std::uint32_t index = cursor;
#if __CUDA_ARCH__ >= 800
                        if constexpr (WarpCoherentAcceleration) {
                            index = __reduce_min_sync(
                                    __activemask(), index);
                        }
#endif
                        if (index >= range.cellCount) break;
                        std::uint32_t shapeMask = 0u;
                        if (cursor == index) {
                            if constexpr (CollectHotPathMetrics) {
                                ++hotPathCounters->
                                        accelerationCellVisitCount;
                            }
                            const CudaSceneAccelerationCell &cell =
                                    acceleration[
                                            range.firstCell + index];
                            const bool intersects =
                                    detail::BoundsIntersect(
                                            detail::
                                                    UnifiedMovingBoundsAt(
                                                            scratch),
                                            cell.bounds);
                            cursor = index +
                                    (intersects
                                             ? 1u
                                             : cell.subtreeEntryCount);
                            if (intersects &&
                                cell.surfaceIndex != UINT32_MAX &&
                                cell.surfaceIndex <
                                        scene->surfaces.count) {
                                if constexpr (CollectHotPathMetrics) {
                                    ++hotPathCounters->
                                            accelerationSurfaceVisitCount;
                                }
                                for (std::uint32_t traversal = 0u;
                                     traversal <
                                             collisionShapeCount;
                                     ++traversal) {
                                    if (detail::BoundsIntersect(
                                                detail::
                                                        MovingBoundsAt(
                                                                scratch,
                                                                traversal),
                                                cell.bounds)) {
                                        shapeMask |=
                                                1u << traversal;
                                    }
                                }
                            }
                            if (shapeMask != 0u) {
                                if (scratch.surfaceHitCount >=
                                    SurfaceHitCapacity) {
                                    useSurfaceCache = false;
                                } else {
                                    detail::SurfaceHitAt(
                                            scratch,
                                            scratch.
                                                    surfaceHitCount++) = {
                                            cell.surfaceIndex,
                                            shapeMask};
                                }
                            }
                        }
                        if (!useSurfaceCache) break;
                    }
                    if (!useSurfaceCache) break;
                } else {
                    std::uint32_t cursors[CachedShapeCount];
#pragma unroll
                    for (std::uint32_t traversal = 0u;
                         traversal < CachedShapeCount;
                         ++traversal) {
                        cursors[traversal] =
                                traversal < collisionShapeCount
                                ? 0u
                                : range.cellCount;
                    }
                    for (;;) {
                        std::uint32_t index = range.cellCount;
#pragma unroll
                        for (std::uint32_t traversal = 0u;
                             traversal < CachedShapeCount;
                             ++traversal) {
                            if (cursors[traversal] < index) {
                                index = cursors[traversal];
                            }
                        }
#if __CUDA_ARCH__ >= 800
                        if constexpr (WarpCoherentAcceleration) {
                            index = __reduce_min_sync(
                                    __activemask(), index);
                        }
#endif
                        if (index >= range.cellCount) break;
                        const CudaSceneAccelerationCell &cell =
                                acceleration[
                                        range.firstCell + index];
                        if constexpr (CollectHotPathMetrics) {
                            ++hotPathCounters->
                                    accelerationCellVisitCount;
                        }
                        std::uint32_t shapeMask = 0u;
#pragma unroll
                        for (std::uint32_t traversal = 0u;
                             traversal < CachedShapeCount;
                             ++traversal) {
                            if (cursors[traversal] != index) {
                                continue;
                            }
                            const bool intersects =
                                    detail::BoundsIntersect(
                                            detail::MovingBoundsAt(
                                                    scratch,
                                                    traversal),
                                            cell.bounds);
                            cursors[traversal] = index +
                                    (intersects
                                             ? 1u
                                             : cell.subtreeEntryCount);
                            if (intersects) {
                                shapeMask |= 1u << traversal;
                            }
                        }
                        if (shapeMask == 0u ||
                            cell.surfaceIndex == UINT32_MAX ||
                            cell.surfaceIndex >=
                                    scene->surfaces.count) {
                            continue;
                        }
                        if constexpr (CollectHotPathMetrics) {
                            ++hotPathCounters->
                                    accelerationSurfaceVisitCount;
                        }
                        if (scratch.surfaceHitCount >=
                            SurfaceHitCapacity) {
                            useSurfaceCache = false;
                            break;
                        }
                        detail::SurfaceHitAt(
                                scratch,
                                scratch.surfaceHitCount++) = {
                                cell.surfaceIndex,
                                shapeMask};
                    }
                    if (!useSurfaceCache) break;
                }
            }
            scratch.surfaceCacheValid = useSurfaceCache;
            if constexpr (CollectHotPathMetrics) {
                if (!scratch.surfaceCacheValid) {
                    ++hotPathCounters->
                            surfaceCacheRefreshFailureCount;
                }
            }
            if (scratch.surfaceCacheValid) {
                detail::BuildMeshCellCache<
                        EightOrderedEllipsoids,
                        CollectHotPathMetrics>(
                        scene, surfaces,
                        collisionShapeCount, scratch,
                        hotPathCounters);
            }
            if constexpr (UseEmptyAirCertificate) {
                const bool emptyShortCache =
                        scratch.surfaceCacheValid &&
                        (scratch.surfaceHitCount == 0u ||
                         (scratch.meshCacheValid &&
                          scratch.meshCellCount == 0u));
                if (emptyShortCache) {
                    if constexpr (CollectHotPathMetrics) {
                        ++hotPathCounters->emptyAirOpportunityCount;
                    }
                    if (emptyAirState->probeCooldown != 0u) {
                        --emptyAirState->probeCooldown;
                    } else {
                        const detail::EmptyAirProbeResult probe =
                                detail::TryExtendEmptyAirCertificate<
                                        CollectHotPathMetrics>(
                                        scene,
                                        acceleration,
                                        surfaces,
                                        detail::SceneSection<
                                                CudaSceneOctreeCell>(
                                                scene,
                                                scene->octreeCells),
                                        candidate.body.current.linearSpeed,
                                        surfaceCacheTravel,
                                        collisionShapeCount,
                                        scratch,
                                        hotPathCounters);
                        if (probe !=
                            detail::EmptyAirProbeResult::Ineligible) {
                            if constexpr (CollectHotPathMetrics) {
                                ++hotPathCounters->
                                        emptyAirProbeAttemptCount;
                            }
                            if (probe ==
                                detail::EmptyAirProbeResult::Clear) {
                                emptyAirState->active = true;
                                emptyAirState->probeCooldown = 0u;
                                scratch.surfaceHitCount = 0u;
                                scratch.meshCellCount = 0u;
                                scratch.meshCacheValid = true;
                                if constexpr (CollectHotPathMetrics) {
                                    ++hotPathCounters->
                                            emptyAirProbeSuccessCount;
                                }
                            } else {
                                emptyAirState->probeCooldown =
                                        detail::
                                                EmptyAirProbeCooldownOpportunities;
                                if constexpr (CollectHotPathMetrics) {
                                    ++hotPathCounters->
                                            emptyAirProbeBlockedCount;
                                }
                            }
                        }
                    }
                }
            }
        }
        if constexpr (CollectHotPathMetrics) {
            if (useSurfaceCache && !refreshSurfaceCache &&
                scratch.meshCacheValid) {
                ++hotPathCounters->meshCacheReuseCount;
            }
        }
        if constexpr (UseEmptyAirCertificate) {
            const bool emptyCollisionCache =
                    scratch.surfaceHitCount == 0u;
            if (emptyAirState->active &&
                useSurfaceCache && scratch.surfaceCacheValid &&
                emptyCollisionCache) {
                if constexpr (CollectHotPathMetrics) {
                    ++hotPathCounters->
                            emptyAirZeroHitFastReturnCount;
                    if (!refreshSurfaceCache) {
                        ++hotPathCounters->
                                emptyAirCertificateReuseCount;
                    }
                }
                detail::SortForResponse<CollectHotPathMetrics>(
                        scratch, hotPathCounters);
                return Status::Success;
            }
        }
        if (useSurfaceCache) {
#if 0
            if constexpr (EightOrderedEllipsoids) {
                const std::uint32_t tileStride =
                        (scratch.stride +
                         CudaCollisionSearchTileWidth - 1u) /
                        CudaCollisionSearchTileWidth;
                CudaCollisionSearchTile *const
                        primaryShapeCollisionStorage =
                                scratch.shapeCollisionStorage;
                CudaCollisionSearchTile *const
                        secondaryShapeCollisionStorage =
                                primaryShapeCollisionStorage +
                        static_cast<std::uint64_t>(
                                ShapeCollisionCapacity) *
                        tileStride;
                for (std::uint32_t firstTraversal = 0u;
                     firstTraversal < collisionShapeCount;
                     firstTraversal += 2u) {
                    std::uint32_t secondShapeCollisionCount = 0u;
                    const std::uint32_t secondTraversal =
                            firstTraversal + 1u;
                    const CudaVehicleCollisionShape &firstShape =
                            shapes[firstTraversal];
                    const CudaVehicleCollisionShape &secondShape =
                            shapes[secondTraversal];
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES)
                    const GmIso4 firstShapeWorld =
                            detail::RootShapeWorldPose(
                                    firstTraversal,
                                    shapes,
                                    candidate,
                                    bodyPose);
                    const GmIso4 secondShapeWorld =
                            detail::RootShapeWorldPose(
                                    secondTraversal,
                                    shapes,
                                    candidate,
                                    bodyPose);
#else
                    const GmIso4 firstShapeWorld =
                            detail::ShapeWorldPose(
                                    firstTraversal,
                                    shapes,
                                    candidate,
                                    bodyPose);
                    const GmIso4 secondShapeWorld =
                            detail::ShapeWorldPose(
                                    secondTraversal,
                                    shapes,
                                    candidate,
                                    bodyPose);
#endif
                    for (std::uint32_t hitIndex = 0u;
                         hitIndex < scratch.surfaceHitCount;
                         ++hitIndex) {
                        const CudaCollisionSurfaceHit hit =
                                detail::SurfaceHitAt(
                                        scratch, hitIndex);
                        if ((hit.shapeMask &
                             ((1u << firstTraversal) |
                              (1u << secondTraversal))) == 0u) {
                            continue;
                        }
                        const CudaSceneSurface &surface =
                                surfaces[hit.surfaceIndex];
                        const CudaCollisionMeshRange range =
                                detail::MeshRangeAt(
                                        scratch, hitIndex);
                        const bool firstEnabled =
                                (hit.shapeMask &
                                 (1u << firstTraversal)) != 0u;
                        const bool secondEnabled =
                                (hit.shapeMask &
                                 (1u << secondTraversal)) != 0u;
                        if (scratch.meshCacheValid &&
                            firstEnabled &&
                            secondEnabled) {
                            detail::EllipsoidMeshPairCached(
                                    scene,
                                    configuration,
                                    surface,
                                    hit.surfaceIndex,
                                    surface.actorIndex,
                                    firstShape,
                                    firstTraversal,
                                    firstShapeWorld,
                                    secondShape,
                                    secondTraversal,
                                    secondShapeWorld,
                                    primaryShapeCollisionStorage,
                                    secondaryShapeCollisionStorage,
                                    &secondShapeCollisionCount,
                                    scratch,
                                    range.first,
                                    range.count);
                            if (scratch.overflow) {
                                return Status::Overflow;
                            }
                            continue;
                        }
                        if (firstEnabled) {
                            if (scratch.meshCacheValid) {
                                detail::EllipsoidMesh<false, true>(
                                        scene,
                                        configuration,
                                        surface,
                                        hit.surfaceIndex,
                                        surface.actorIndex,
                                        firstShape,
                                        firstTraversal,
                                        firstShapeWorld,
                                        scratch,
                                        range.first,
                                        range.count);
                            } else {
                                detail::EllipsoidMesh<false>(
                                        scene,
                                        configuration,
                                        surface,
                                        hit.surfaceIndex,
                                        surface.actorIndex,
                                        firstShape,
                                        firstTraversal,
                                        firstShapeWorld,
                                        scratch);
                            }
                        }
                        if (secondEnabled) {
                            const std::uint32_t
                                    firstShapeCollisionCount =
                                            scratch.
                                                    shapeCollisionCount;
                            scratch.shapeCollisionStorage =
                                    secondaryShapeCollisionStorage;
                            scratch.shapeCollisionCount =
                                    secondShapeCollisionCount;
                            if (scratch.meshCacheValid) {
                                detail::EllipsoidMesh<false, true>(
                                        scene,
                                        configuration,
                                        surface,
                                        hit.surfaceIndex,
                                        surface.actorIndex,
                                        secondShape,
                                        secondTraversal,
                                        secondShapeWorld,
                                        scratch,
                                        range.first,
                                        range.count);
                            } else {
                                detail::EllipsoidMesh<false>(
                                        scene,
                                        configuration,
                                        surface,
                                        hit.surfaceIndex,
                                        surface.actorIndex,
                                        secondShape,
                                        secondTraversal,
                                        secondShapeWorld,
                                        scratch);
                            }
                            secondShapeCollisionCount =
                                    scratch.shapeCollisionCount;
                            scratch.shapeCollisionStorage =
                                    primaryShapeCollisionStorage;
                            scratch.shapeCollisionCount =
                                    firstShapeCollisionCount;
                        }
                        if (scratch.overflow) {
                            return Status::Overflow;
                        }
                    }
                    detail::MergeShapeContacts(scratch);
                    scratch.shapeCollisionStorage =
                            secondaryShapeCollisionStorage;
                    scratch.shapeCollisionCount =
                            secondShapeCollisionCount;
                    detail::MergeShapeContacts(scratch);
                    scratch.shapeCollisionStorage =
                            primaryShapeCollisionStorage;
                    if (scratch.overflow) {
                        return Status::Overflow;
                    }
                }
                detail::SortForResponse(scratch);
                return Status::Success;
            }
#endif
            for (std::uint32_t traversal = 0u;
                 traversal < collisionShapeCount;
                 ++traversal) {
                const std::uint32_t shapeIndex = traversal;
                const CudaVehicleCollisionShape &shape =
                        shapes[shapeIndex];
                GmIso4 shapeWorld;
                if constexpr (EightOrderedEllipsoids) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_EIGHT_ROOT_SHAPES)
                    shapeWorld = detail::RootShapeWorldPose(
                            shapeIndex, shapes,
                            candidate, bodyPose);
#else
                    shapeWorld = detail::ShapeWorldPose(
                            shapeIndex, shapes,
                            candidate, bodyPose);
#endif
                } else {
                    shapeWorld =
                            detail::ShapeWorldAt(
                                    scratch, traversal);
                }
                for (std::uint32_t hitIndex = 0u;
                     hitIndex < scratch.surfaceHitCount;
                     ++hitIndex) {
                    const CudaCollisionSurfaceHit hit =
                            detail::SurfaceHitAt(
                                    scratch, hitIndex);
                    if ((hit.shapeMask &
                         (1u << traversal)) == 0u) {
                        continue;
                    }
                    const CudaSceneSurface &surface =
                            surfaces[hit.surfaceIndex];
                    if (surface.type !=
                        static_cast<std::uint32_t>(
                                GmSurf::EGmSurfType::Mesh)) {
                        return Status::UnsupportedGeometry;
                    }
                    if (scratch.meshCacheValid) {
                        const CudaCollisionMeshRange range =
                                detail::MeshRangeAt(
                                        scratch, hitIndex);
                        detail::EllipsoidMesh<
                                false, true,
                                CollectHotPathMetrics>(
                                scene, configuration, surface,
                                hit.surfaceIndex,
                                surface.actorIndex,
                                shape, shapeIndex,
                                shapeWorld,
                                scratch, range.first,
                                range.count,
                                hotPathCounters);
                    } else {
                        detail::EllipsoidMesh<
                                false, false,
                                CollectHotPathMetrics>(
                                scene, configuration, surface,
                                hit.surfaceIndex,
                                surface.actorIndex,
                                shape, shapeIndex,
                                shapeWorld,
                                scratch, 0u, 0u,
                                hotPathCounters);
                    }
                    if (scratch.overflow) {
                        return Status::Overflow;
                    }
                }
                detail::MergeShapeContacts<CollectHotPathMetrics>(
                        scratch, hotPathCounters);
                if (scratch.overflow) {
                    return Status::Overflow;
                }
            }
            detail::SortForResponse<CollectHotPathMetrics>(
                    scratch, hotPathCounters);
            return scratch.overflow
                    ? Status::Overflow
                    : Status::Success;
        }
    }

    const std::uint32_t collisionShapeCount =
            EightOrderedEllipsoids
            ? 8u
            : configuration->collisionShapes.count;
    for (std::uint32_t traversal = 0u;
         traversal < collisionShapeCount;
         ++traversal) {
        const std::uint32_t shapeIndex = traversal;
        if constexpr (!EightOrderedEllipsoids) {
            if (shapes[shapeIndex].traversalOrder != traversal) {
                return Status::InvalidScene;
            }
        }
        const CudaVehicleCollisionShape &shape =
                shapes[shapeIndex];
        if constexpr (!EightOrderedEllipsoids) {
            if (shape.surfaceType != static_cast<std::uint32_t>(
                        GmSurf::EGmSurfType::Ellipsoid) &&
                shape.surfaceType != static_cast<std::uint32_t>(
                        GmSurf::EGmSurfType::Sphere)) {
                return Status::UnsupportedGeometry;
            }
        }
        const GmIso4 shapeWorld =
                detail::ShapeWorldPose(
                        shapeIndex, shapes,
                        candidate, bodyPose);
        const GmBoxAligned movingBounds =
                detail::TransformBox(
                        shape.localBounds, shapeWorld);
        constexpr std::uint32_t TargetGroups[] = {1u, 3u, 4u};
        const std::uint32_t targetGroupCount =
                TriggerOnly ? 1u : 3u;
        for (std::uint32_t groupIndex = 0u;
             groupIndex < targetGroupCount; ++groupIndex) {
            const std::uint32_t group = TargetGroups[groupIndex];
            const CudaSceneAccelerationRange range =
                    scene->accelerationGroups[group - 1u];
            if (range.cellCount <= 1u) continue;
            std::uint32_t index = 0u;
            while (index < range.cellCount) {
                if constexpr (CollectHotPathMetrics) {
                    ++hotPathCounters->accelerationCellVisitCount;
                }
                if constexpr (TrackDiagnostics) {
                    ++scratch.accelerationCellVisits;
                }
                const CudaSceneAccelerationCell &cell =
                        acceleration[range.firstCell + index];
                if (!detail::BoundsIntersect(
                            movingBounds, cell.bounds)) {
                    index += cell.subtreeEntryCount;
                    continue;
                }
                ++index;
                if (cell.surfaceIndex == UINT32_MAX ||
                    cell.surfaceIndex >= scene->surfaces.count) {
                    continue;
                }
                const CudaSceneSurface &surface =
                        surfaces[cell.surfaceIndex];
                if constexpr (CollectHotPathMetrics) {
                    ++hotPathCounters->accelerationSurfaceVisitCount;
                }
                if constexpr (TrackDiagnostics) {
                    ++scratch.accelerationSurfaceVisits;
                }
                if (surface.type != static_cast<std::uint32_t>(
                            GmSurf::EGmSurfType::Mesh)) {
                    return Status::UnsupportedGeometry;
                }
                if constexpr (EightOrderedEllipsoids) {
                    detail::EllipsoidMesh<
                            TrackDiagnostics, false,
                            CollectHotPathMetrics>(
                            scene, configuration, surface,
                            cell.surfaceIndex,
                            surface.actorIndex, shape, shapeIndex,
                            shapeWorld, scratch, 0u, 0u,
                            hotPathCounters);
                } else if (shape.surfaceType ==
                           static_cast<std::uint32_t>(
                                   GmSurf::EGmSurfType::Sphere)) {
                    detail::SphereMesh<
                            TrackDiagnostics,
                            CollectHotPathMetrics>(
                            scene, configuration, surface,
                            cell.surfaceIndex,
                            surface.actorIndex, shape, shapeIndex,
                            shapeWorld, scratch,
                            hotPathCounters);
                } else {
                    detail::EllipsoidMesh<
                            TrackDiagnostics, false,
                            CollectHotPathMetrics>(
                            scene, configuration, surface,
                            cell.surfaceIndex,
                            surface.actorIndex, shape, shapeIndex,
                            shapeWorld, scratch, 0u, 0u,
                            hotPathCounters);
                }
                if (scratch.overflow) return Status::Overflow;
            }
        }
        detail::MergeShapeContacts<CollectHotPathMetrics>(
                scratch, hotPathCounters);
        if (scratch.overflow) return Status::Overflow;
    }
    detail::SortForResponse<CollectHotPathMetrics>(
            scratch, hotPathCounters);
    if (scratch.overflow) return Status::Overflow;
    return Status::Success;
}

}  // namespace forevervalidator::simulation::cuda::collision

#endif
