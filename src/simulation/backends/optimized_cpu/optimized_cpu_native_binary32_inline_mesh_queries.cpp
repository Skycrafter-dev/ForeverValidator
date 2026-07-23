// OptimizedCpu native-binary32 mesh queries with exact scalar geometry kept in
// this translation unit so the compiler can inline it without changing
// authoritative helpers or the selected square-root policy.

#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"

#include <cmath>
#include <limits>
#include <typeinfo>
#include <vector>

#if (defined(__i386__) || defined(__x86_64__)) && \
        (defined(__GNUC__) || defined(__clang__)) && defined(__SSE2__)
#define FV_E024_HAS_INLINE_SSE2 1
#include <emmintrin.h>
#include <xmmintrin.h>
#else
#define FV_E024_HAS_INLINE_SSE2 0
#endif

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"
#include "engine/physics/geometry/plug_surface.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_mesh_triangle_sidecar.h"

struct GmSurfMeshOptimizedCpuAccess {
    static const std::vector<GmVec3> &Vertices(const GmSurfMesh &mesh) {
        return mesh.vertices;
    }

    static const std::vector<GmSurfMeshTriangle> &Triangles(
            const GmSurfMesh &mesh) {
        return mesh.triangles;
    }

    static const std::vector<GmMeshOctreeCell> &OctreeCells(
            const GmSurfMesh &mesh) {
        return mesh.octreeCells;
    }
};

#if defined(_MSC_VER)
#define FV_E012_ALWAYS_INLINE __forceinline
#define FV_E012_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define FV_E012_ALWAYS_INLINE inline __attribute__((always_inline))
#define FV_E012_NOINLINE __attribute__((noinline))
#else
#define FV_E012_ALWAYS_INLINE inline
#define FV_E012_NOINLINE
#endif

namespace {

FV_E012_ALWAYS_INLINE float NativeBinary32Sqrt(float value) noexcept {
    return forevervalidator::simulation::
            OptimizedCpuBinary32SqrtX86Sse2(value);
}

FV_E012_ALWAYS_INLINE float NativeBinary32PositiveSqrt(float value) noexcept {
#if FV_E024_HAS_INLINE_SSE2
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(value)));
#else
    return NativeBinary32Sqrt(value);
#endif
}

// These expressions mirror the authoritative helpers' operand order. Do not
// simplify their grouping or commute operands, including apparently symmetric
// products where NaN payload and signed-zero behavior can be observable.
FV_E012_ALWAYS_INLINE float SumProducts(const GmVec3 &left,
                                        const GmVec3 &right) {
    const float xy = left.x * right.x + left.y * right.y;
    return xy + left.z * right.z;
}

FV_E012_ALWAYS_INLINE GmVec3 Subtract(const GmVec3 &left,
                                      const GmVec3 &right) {
    return {
        (left.x - right.x),
        (left.y - right.y),
        (left.z - right.z),
    };
}

FV_E012_ALWAYS_INLINE GmVec3 Add(const GmVec3 &left,
                                 const GmVec3 &right) {
    return {
        (left.x + right.x),
        (left.y + right.y),
        (left.z + right.z),
    };
}

FV_E012_ALWAYS_INLINE GmVec3 Scale(const GmVec3 &value, float scale) {
    return {
        (value.x * scale),
        (value.y * scale),
        (value.z * scale),
    };
}

FV_E012_ALWAYS_INLINE GmVec3 Negate(const GmVec3 &value) {
    return {-value.x, -value.y, -value.z};
}

FV_E012_ALWAYS_INLINE GmVec3 Cross(const GmVec3 &left,
                                   const GmVec3 &right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

FV_E012_ALWAYS_INLINE GmVec3 Normalize(const GmVec3 &value,
                                       float epsilonSq) {
    GmVec3 out = value;
    const float lengthSq = SumProducts(out, out);
    if (epsilonSq < lengthSq) {
        out = Scale(out, 1.0f / NativeBinary32PositiveSqrt(lengthSq));
    }
    return out;
}

FV_E012_ALWAYS_INLINE GmVec3 TransformDirection(
        const GmMat3 &matrix,
        const GmVec3 &direction) {
#if FV_E024_HAS_INLINE_SSE2
    // Lane 3 duplicates x so the packed operations cannot raise a sticky
    // exception which the three authoritative component expressions do not.
    const __m128 basisX = _mm_set_ps(
            matrix.basisX.x,
            matrix.basisX.z,
            matrix.basisX.y,
            matrix.basisX.x);
    const __m128 basisY = _mm_set_ps(
            matrix.basisY.x,
            matrix.basisY.z,
            matrix.basisY.y,
            matrix.basisY.x);
    const __m128 basisZ = _mm_set_ps(
            matrix.basisZ.x,
            matrix.basisZ.z,
            matrix.basisZ.y,
            matrix.basisZ.x);
    const __m128 xy = _mm_add_ps(
            _mm_mul_ps(basisX, _mm_set1_ps(direction.x)),
            _mm_mul_ps(basisY, _mm_set1_ps(direction.y)));
    const __m128 transformed = _mm_add_ps(
            xy,
            _mm_mul_ps(basisZ, _mm_set1_ps(direction.z)));
    alignas(16) float components[4];
    _mm_store_ps(components, transformed);
    return {components[0], components[1], components[2]};
#else
    const GmVec3 rowX = {
        matrix.basisX.x,
        matrix.basisY.x,
        matrix.basisZ.x,
    };
    const GmVec3 rowY = {
        matrix.basisX.y,
        matrix.basisY.y,
        matrix.basisZ.y,
    };
    const GmVec3 rowZ = {
        matrix.basisX.z,
        matrix.basisY.z,
        matrix.basisZ.z,
    };
    return {
        SumProducts(rowX, direction),
        SumProducts(rowY, direction),
        SumProducts(rowZ, direction),
    };
#endif
}

FV_E012_ALWAYS_INLINE GmVec3 TransformPoint(const GmIso4 &transform,
                                            const GmVec3 &point) {
    return Add(TransformDirection(transform.rotation, point),
               transform.translation);
}

FV_E012_ALWAYS_INLINE GmMat3 Compose(const GmMat3 &first,
                                     const GmMat3 &second) {
    GmMat3 result;
    result.basisX = TransformDirection(second, first.basisX);
    result.basisY = TransformDirection(second, first.basisY);
    result.basisZ = TransformDirection(second, first.basisZ);
    return result;
}

FV_E012_ALWAYS_INLINE GmMat3 Transpose(const GmMat3 &matrix) {
    return {
        {matrix.basisX.x, matrix.basisY.x, matrix.basisZ.x},
        {matrix.basisX.y, matrix.basisY.y, matrix.basisZ.y},
        {matrix.basisX.z, matrix.basisY.z, matrix.basisZ.z},
    };
}

FV_E012_ALWAYS_INLINE GmIso4 Inverse(const GmIso4 &transform) {
    const GmMat3 inverseRotation = Transpose(transform.rotation);
    const GmVec3 inverseTranslation =
            TransformDirection(inverseRotation, Negate(transform.translation));
    return {inverseRotation, inverseTranslation};
}

FV_E012_ALWAYS_INLINE GmIso4 Compose(const GmIso4 &first,
                                     const GmIso4 &second) {
    const GmIso4 left = first;
    const GmIso4 right = second;
    return {
        Compose(left.rotation, right.rotation),
        TransformPoint(right, left.translation),
    };
}

FV_E012_ALWAYS_INLINE GmIso4 MultInverse(const GmIso4 &transform,
                                         const GmIso4 &right) {
    const GmIso4 inverse = Inverse(right);
    return Compose(transform, inverse);
}

FV_E012_ALWAYS_INLINE GmIso4 Identity(void) {
    return {
        {{1.0f, 0.0f, 0.0f},
         {0.0f, 1.0f, 0.0f},
         {0.0f, 0.0f, 1.0f}},
        {},
    };
}

FV_E012_ALWAYS_INLINE GmIso4 DiagonalTransform(
        const GmVec3 &scale,
        const GmVec3 &translation) {
    return {
        {{scale.x, 0.0f, 0.0f},
         {0.0f, scale.y, 0.0f},
         {0.0f, 0.0f, scale.z}},
        translation,
    };
}

FV_E012_ALWAYS_INLINE void ScaleRows(GmIso4 *transform,
                                     const GmVec3 &rowScale) {
    transform->rotation.basisX.x =
            rowScale.x * transform->rotation.basisX.x;
    transform->rotation.basisY.x =
            transform->rotation.basisY.x * rowScale.x;
    transform->rotation.basisZ.x =
            rowScale.x * transform->rotation.basisZ.x;
    transform->translation.x = transform->translation.x * rowScale.x;

    transform->rotation.basisX.y =
            rowScale.y * transform->rotation.basisX.y;
    transform->rotation.basisY.y =
            transform->rotation.basisY.y * rowScale.y;
    transform->rotation.basisZ.y =
            rowScale.y * transform->rotation.basisZ.y;
    transform->translation.y = transform->translation.y * rowScale.y;

    transform->rotation.basisX.z =
            rowScale.z * transform->rotation.basisX.z;
    transform->rotation.basisY.z =
            transform->rotation.basisY.z * rowScale.z;
    transform->rotation.basisZ.z =
            rowScale.z * transform->rotation.basisZ.z;
    transform->translation.z = transform->translation.z * rowScale.z;
}

FV_E012_ALWAYS_INLINE GmBoxAligned TransformBox(
        const GmBoxAligned &box,
        const GmIso4 &transform) {
    const GmBoxAligned source = box;
    GmMat3 absoluteRotation;
    absoluteRotation.basisX = {
        std::fabs(transform.rotation.basisX.x),
        std::fabs(transform.rotation.basisX.y),
        std::fabs(transform.rotation.basisX.z),
    };
    absoluteRotation.basisY = {
        std::fabs(transform.rotation.basisY.x),
        std::fabs(transform.rotation.basisY.y),
        std::fabs(transform.rotation.basisY.z),
    };
    absoluteRotation.basisZ = {
        std::fabs(transform.rotation.basisZ.x),
        std::fabs(transform.rotation.basisZ.y),
        std::fabs(transform.rotation.basisZ.z),
    };
    return {
        TransformPoint(transform, source.center),
        TransformDirection(absoluteRotation, source.halfExtents),
    };
}

FV_E012_ALWAYS_INLINE GmVec3 TransformEllipsoidWorldNormal(
        const GmVec3 &normal,
        const GmMat3 &rotation) {
    const GmVec3 source = normal;
    const float resultX =
            (rotation.basisX.x * source.x +
             rotation.basisY.x * source.y) +
            rotation.basisZ.x * source.z;
    const float resultY =
            (rotation.basisX.y * source.x +
             rotation.basisY.y * source.y) +
            rotation.basisZ.y * source.z;
    const float resultZ =
            (rotation.basisX.z * source.x +
             rotation.basisY.z * source.y) +
            rotation.basisZ.z * source.z;
    return {resultX, resultY, resultZ};
}

FV_E012_ALWAYS_INLINE bool BoundsIntersect(
        const GmBoxAligned &query,
        const GmBoxAligned &candidate) {
    if (candidate.halfExtents.z + query.halfExtents.z <
        std::fabs(candidate.center.z - query.center.z)) {
        return false;
    }
    if (candidate.halfExtents.y + query.halfExtents.y <
        std::fabs(candidate.center.y - query.center.y)) {
        return false;
    }
    return !(candidate.halfExtents.x + query.halfExtents.x <
             std::fabs(candidate.center.x - query.center.x));
}

FV_E012_ALWAYS_INLINE bool CanUseEllipsoidPlaneCertificate(
        const GmIso4 &ellipsoidToMesh,
        const GmVec3 &radii) {
    const double determinant =
            static_cast<double>(ellipsoidToMesh.rotation.basisX.x) *
                    (static_cast<double>(ellipsoidToMesh.rotation.basisY.y) *
                             ellipsoidToMesh.rotation.basisZ.z -
                     static_cast<double>(ellipsoidToMesh.rotation.basisY.z) *
                             ellipsoidToMesh.rotation.basisZ.y) -
            static_cast<double>(ellipsoidToMesh.rotation.basisY.x) *
                    (static_cast<double>(ellipsoidToMesh.rotation.basisX.y) *
                             ellipsoidToMesh.rotation.basisZ.z -
                     static_cast<double>(ellipsoidToMesh.rotation.basisX.z) *
                             ellipsoidToMesh.rotation.basisZ.y) +
            static_cast<double>(ellipsoidToMesh.rotation.basisZ.x) *
                    (static_cast<double>(ellipsoidToMesh.rotation.basisX.y) *
                             ellipsoidToMesh.rotation.basisY.z -
                     static_cast<double>(ellipsoidToMesh.rotation.basisX.z) *
                             ellipsoidToMesh.rotation.basisY.y);
    return 0.0 < determinant &&
           0.0f < radii.x && 0.0f < radii.y && 0.0f < radii.z;
}

FV_E012_ALWAYS_INLINE int EllipsoidTrianglePlaneRejection(
        const GmIso4 &ellipsoidToMesh,
        const GmVec3 &radii,
        const OptimizedCpuStaticMeshTriangleData &triangle) {
    const GmVec3 &normal = triangle.geometricNormal;
    const GmVec3 &vertex = triangle.vertices[0];
    const double relativeX =
            static_cast<double>(ellipsoidToMesh.translation.x) - vertex.x;
    const double relativeY =
            static_cast<double>(ellipsoidToMesh.translation.y) - vertex.y;
    const double relativeZ =
            static_cast<double>(ellipsoidToMesh.translation.z) - vertex.z;
    const double distance =
            (relativeX * normal.x + relativeY * normal.y) +
            relativeZ * normal.z;
    const double localNormalX =
            (static_cast<double>(ellipsoidToMesh.rotation.basisX.x) * normal.x +
             static_cast<double>(ellipsoidToMesh.rotation.basisX.y) * normal.y) +
            static_cast<double>(ellipsoidToMesh.rotation.basisX.z) * normal.z;
    const double localNormalY =
            (static_cast<double>(ellipsoidToMesh.rotation.basisY.x) * normal.x +
             static_cast<double>(ellipsoidToMesh.rotation.basisY.y) * normal.y) +
            static_cast<double>(ellipsoidToMesh.rotation.basisY.z) * normal.z;
    const double localNormalZ =
            (static_cast<double>(ellipsoidToMesh.rotation.basisZ.x) * normal.x +
             static_cast<double>(ellipsoidToMesh.rotation.basisZ.y) * normal.y) +
            static_cast<double>(ellipsoidToMesh.rotation.basisZ.z) * normal.z;
    const double scaledX = static_cast<double>(radii.x) * localNormalX;
    const double scaledY = static_cast<double>(radii.y) * localNormalY;
    const double scaledZ = static_cast<double>(radii.z) * localNormalZ;
    const double supportSquared =
            (scaledX * scaledX + scaledY * scaledY) + scaledZ * scaledZ;
    if (!std::isfinite(distance) || !std::isfinite(supportSquared) ||
        supportSquared < 0.0) {
        return 0;
    }

    constexpr double guardScale =
            256.0 * std::numeric_limits<float>::epsilon();
    if (0.0 < distance) {
        const double guardedDistance =
                distance * (1.0 - guardScale) - guardScale;
        const double guardedSupportSquared =
                supportSquared *
                ((1.0 + guardScale) * (1.0 + guardScale));
        if (0.0 < guardedDistance &&
            guardedSupportSquared < guardedDistance * guardedDistance) {
            return 1;
        }
        return 0;
    }
    const double magnitude = -distance;
    const double guardedMagnitude =
            magnitude * (1.0 - guardScale) - guardScale;
    return 0.0 < guardedMagnitude &&
                   guardScale * guardScale * supportSquared <
                           guardedMagnitude * guardedMagnitude
            ? -1
            : 0;
}

struct SInlineSphereMeshCollide {
    CGmCollisionBuffer *collisionBuffer;
    GmVec3 sphereCenterMesh;
    float radius;
    GmLocalMaterialIndex materialA;
    GmVec3 triangleNormal;
    GmLocalMaterialIndex triangleMaterial;

    FV_E012_NOINLINE int EmitFeatureCollision(
            GmVec3 featurePointLocal,
            float minDistanceSq,
            bool requireRadiusContainment) {
        const GmVec3 featureToCenter =
                Subtract(sphereCenterMesh, featurePointLocal);
        const float distanceSq = SumProducts(featureToCenter, featureToCenter);
        if ((requireRadiusContainment && radius * radius < distanceSq) ||
            !(minDistanceSq < distanceSq)) {
            return 0;
        }

        const float distance = NativeBinary32PositiveSqrt(distanceSq);
        const float invDistance = 1.0f / distance;
        const GmVec3 normal = {
            featureToCenter.x * invDistance,
            featureToCenter.y * invDistance,
            featureToCenter.z * invDistance,
        };
        const float distanceMinusRadius = distance - radius;
        const float penetrationScale = distanceMinusRadius * invDistance;
        const GmVec3 penetration = {
            featureToCenter.x * penetrationScale,
            featureToCenter.y * penetrationScale,
            featureToCenter.z * penetrationScale,
        };
        const float separationAlongTriangleNormal =
                SumProducts(penetration, triangleNormal);

        GmCollision *collision = &collisionBuffer->AddCollision();
        collision->impulseNormal = normal;
        collision->separation =
                Scale(triangleNormal, separationAlongTriangleNormal);
        collision->contactPoint = featurePointLocal;
        collision->localMaterialA = materialA;
        collision->localMaterialB = triangleMaterial;
        collision->sphereMergePrimary = 0;
        collision->extraNegated = triangleNormal;
        return 1;
    }

    FV_E012_NOINLINE int EmitEndpointBCollision(
            GmVec3 featurePointLocal,
            float minDistance) {
        const GmVec3 featureToCenter =
                Subtract(sphereCenterMesh, featurePointLocal);
        const float distanceSq = SumProducts(featureToCenter, featureToCenter);
        const float distance = NativeBinary32Sqrt(distanceSq);
        if (radius * radius < distance || !(minDistance < distance)) {
            return 0;
        }

        const float endpointDistance = NativeBinary32PositiveSqrt(distance);
        const float invEndpointDistance = 1.0f / endpointDistance;
        const GmVec3 normal = {
            featureToCenter.x * invEndpointDistance,
            featureToCenter.y * invEndpointDistance,
            featureToCenter.z * invEndpointDistance,
        };
        const float endpointDistanceMinusRadius = endpointDistance - radius;
        const float endpointPenetrationScale =
                endpointDistanceMinusRadius * invEndpointDistance;
        const GmVec3 penetration = {
            featureToCenter.x * endpointPenetrationScale,
            featureToCenter.y * endpointPenetrationScale,
            featureToCenter.z * endpointPenetrationScale,
        };
        const float separationAlongTriangleNormal =
                SumProducts(penetration, triangleNormal);

        GmCollision *collision = &collisionBuffer->AddCollision();
        collision->impulseNormal = normal;
        collision->separation =
                Scale(triangleNormal, separationAlongTriangleNormal);
        collision->contactPoint = featurePointLocal;
        collision->localMaterialA = materialA;
        collision->localMaterialB = triangleMaterial;
        collision->sphereMergePrimary = 0;
        collision->extraNegated = triangleNormal;
        return 1;
    }

    FV_E012_NOINLINE int CollideTriangle(const GmVec3 vertices[3]) {
        const float planeDistance = SumProducts(
                Subtract(sphereCenterMesh, vertices[0]), triangleNormal);
        if (radius < planeDistance || planeDistance < 0.0f) {
            return 0;
        }

        const float edgeReach = NativeBinary32Sqrt(
                radius * radius - planeDistance * planeDistance);
        const GmVec3 projectedPoint = Add(
                sphereCenterMesh, Scale(triangleNormal, -planeDistance));

        for (u32 edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
            const u32 nextIndex = edgeIndex == 2u ? 0u : edgeIndex + 1u;
            const GmVec3 edgeStart = vertices[edgeIndex];
            const GmVec3 edgeEnd = vertices[nextIndex];
            const GmVec3 edgeDir = Normalize(
                    Subtract(edgeEnd, edgeStart),
                    PhysicsTolerance::SurfaceDirectionLengthSquared);
            const GmVec3 edgeNormal = Cross(edgeDir, triangleNormal);
            const float edgeDistance = SumProducts(
                    Subtract(projectedPoint, edgeStart), edgeNormal);

            if (edgeReach < edgeDistance) {
                return 0;
            }

            if (edgeDistance > 0.0f) {
                const float alongFromStart = SumProducts(
                        Subtract(projectedPoint, edgeStart), edgeDir);
                if (alongFromStart < 0.0f) {
                    return EmitFeatureCollision(
                            edgeStart,
                            PhysicsTolerance::SurfaceDirectionLengthSquared,
                            true);
                }

                const float alongFromEnd = SumProducts(
                        Subtract(projectedPoint, edgeEnd), edgeDir);
                if (!(0.0f < alongFromEnd)) {
                    const GmVec3 featurePoint = Add(
                            projectedPoint,
                            Scale(edgeNormal, -edgeDistance));
                    return EmitFeatureCollision(
                            featurePoint,
                            PhysicsTolerance::CollisionDistance,
                            false);
                }

                return EmitEndpointBCollision(
                        edgeEnd,
                        PhysicsTolerance::SurfaceDirectionLengthSquared);
            }
        }

        if (planeDistance > 0.0f) {
            GmCollision *collision = &collisionBuffer->AddCollision();
            collision->impulseNormal = triangleNormal;
            collision->separation =
                    Scale(triangleNormal, planeDistance - radius);
            collision->contactPoint = projectedPoint;
            collision->localMaterialA = materialA;
            collision->localMaterialB = triangleMaterial;
            collision->sphereMergePrimary = 1;
            collision->extraNegated = triangleNormal;
            return 1;
        }

        return 0;
    }

    FV_E012_NOINLINE int CollideTriangle(
            const OptimizedCpuStaticMeshTriangleData &triangle) {
        const float planeDistance = SumProducts(
                Subtract(sphereCenterMesh, triangle.vertices[0]),
                triangleNormal);
        if (radius < planeDistance || planeDistance < 0.0f) {
            return 0;
        }

        const float edgeReach = NativeBinary32Sqrt(
                radius * radius - planeDistance * planeDistance);
        const GmVec3 projectedPoint = Add(
                sphereCenterMesh, Scale(triangleNormal, -planeDistance));

        for (u32 edgeIndex = 0u; edgeIndex < 3u; ++edgeIndex) {
            const u32 nextIndex = edgeIndex == 2u ? 0u : edgeIndex + 1u;
            const GmVec3 edgeStart = triangle.vertices[edgeIndex];
            const GmVec3 edgeEnd = triangle.vertices[nextIndex];
            const GmVec3 edgeDir = triangle.edgeDirections[edgeIndex];
            const GmVec3 edgeNormal = triangle.edgeNormals[edgeIndex];
            const float edgeDistance = SumProducts(
                    Subtract(projectedPoint, edgeStart), edgeNormal);

            if (edgeReach < edgeDistance) {
                return 0;
            }

            if (edgeDistance > 0.0f) {
                const float alongFromStart = SumProducts(
                        Subtract(projectedPoint, edgeStart), edgeDir);
                if (alongFromStart < 0.0f) {
                    return EmitFeatureCollision(
                            edgeStart,
                            PhysicsTolerance::SurfaceDirectionLengthSquared,
                            true);
                }

                const float alongFromEnd = SumProducts(
                        Subtract(projectedPoint, edgeEnd), edgeDir);
                if (!(0.0f < alongFromEnd)) {
                    const GmVec3 featurePoint = Add(
                            projectedPoint,
                            Scale(edgeNormal, -edgeDistance));
                    return EmitFeatureCollision(
                            featurePoint,
                            PhysicsTolerance::CollisionDistance,
                            false);
                }

                return EmitEndpointBCollision(
                        edgeEnd,
                        PhysicsTolerance::SurfaceDirectionLengthSquared);
            }
        }

        if (planeDistance > 0.0f) {
            GmCollision *collision = &collisionBuffer->AddCollision();
            collision->impulseNormal = triangleNormal;
            collision->separation =
                    Scale(triangleNormal, planeDistance - radius);
            collision->contactPoint = projectedPoint;
            collision->localMaterialA = materialA;
            collision->localMaterialB = triangleMaterial;
            collision->sphereMergePrimary = 1;
            collision->extraNegated = triangleNormal;
            return 1;
        }

        return 0;
    }
};

FV_E012_ALWAYS_INLINE void TransformSphereCollisionsToWorld(
        CGmCollisionBuffer &collisionBuffer,
        u32 firstNew,
        const GmIso4 &meshIso) {
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         collisionIndex++) {
        GmCollision *collision =
                &collisionBuffer.GetCollision(collisionIndex);
        const GmMat3 meshRotation = meshIso.rotation;
        collision->impulseNormal =
                TransformDirection(meshRotation, collision->impulseNormal);
        collision->separation =
                TransformDirection(meshRotation, collision->separation);
        collision->contactPoint =
                TransformPoint(meshIso, collision->contactPoint);
    }
}

FV_E012_ALWAYS_INLINE void TransformEllipsoidCollisionsToWorld(
        CGmCollisionBuffer &collisionBuffer,
        u32 firstNew,
        const GmIso4 &unitContactToWorld,
        const GmIso4 &unitNormalToWorld) {
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         collisionIndex++) {
        GmCollision *collision =
                &collisionBuffer.GetCollision(collisionIndex);
        collision->contactPoint = TransformPoint(
                unitContactToWorld, collision->contactPoint);
        collision->impulseNormal = TransformEllipsoidWorldNormal(
                collision->impulseNormal, unitNormalToWorld.rotation);
        collision->impulseNormal = Normalize(
                collision->impulseNormal,
                PhysicsTolerance::SurfaceDirectionLengthSquared);
        collision->separation = TransformDirection(
                unitContactToWorld.rotation, collision->separation);
    }
}

int RunInlineSphereMeshQuery(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &meshLocated,
        const GmIso4 *meshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar *staticTriangles,
        CGmCollisionBuffer &collisionBuffer) {
    const GmSurfSphere &sphereSurface =
            static_cast<const GmSurfSphere &>(*sphere.surf);
    const GmSurfMesh &mesh =
            static_cast<const GmSurfMesh &>(*meshLocated.surf);
    const float radius = sphereSurface.radius;

    GmIso4 sphereToMesh =
            sphere.enabled == 0 ? Identity() : *sphere.iso;
    if (meshLocated.enabled != 0) {
        sphereToMesh = meshInverse == nullptr
                ? MultInverse(sphereToMesh, *meshLocated.iso)
                : Compose(sphereToMesh, *meshInverse);
    }

    const u32 firstNew = collisionBuffer.GetCurrentCount();
    const GmVec3 zero = {0.0f, 0.0f, 0.0f};
    const GmVec3 sphereHalf = {radius, radius, radius};
    const GmBoxAligned sphereBox = TransformBox(
            {zero, sphereHalf}, sphereToMesh);
    const GmVec3 sphereCenterMesh = sphereToMesh.translation;
    int hit = 0;
    const std::vector<GmVec3> &vertices =
            GmSurfMeshOptimizedCpuAccess::Vertices(mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshOptimizedCpuAccess::Triangles(mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshOptimizedCpuAccess::OctreeCells(mesh);
    const u32 cellCount = static_cast<u32>(cells.size());
    OptimizedCpuStaticUniformGrid::CandidateSpan candidateSpan;
    const bool hasDirectCandidates = staticTriangles != nullptr &&
            staticTriangles->DirectCandidateTriangleSpan(
                    sphereBox, &candidateSpan);
    u32 sourceCellIndex = 0u;
    std::size_t candidateIndex = 0u;
    while (hasDirectCandidates
                   ? candidateIndex < candidateSpan.size
                   : sourceCellIndex < cellCount) {
        const GmMeshOctreeCell *cell = nullptr;
        const GmBoxAligned *candidateBounds = nullptr;
        u32 triangleIndex = 0u;
        if (hasDirectCandidates) {
            const OptimizedCpuStaticMeshDirectTrianglePosting &posting =
                    staticTriangles->DirectTriangleAt(
                            candidateSpan.data[candidateIndex++]);
            candidateBounds = &posting.bounds;
            triangleIndex = posting.triangleIndex;
        } else {
            cell = &cells[sourceCellIndex];
            candidateBounds = &cell->Bounds();
        }
        if (!BoundsIntersect(sphereBox, *candidateBounds)) {
            if (!hasDirectCandidates) {
                sourceCellIndex += cell->SubtreeEntryCount();
            }
            continue;
        }
        if (!hasDirectCandidates) {
            ++sourceCellIndex;
        }

        if (hasDirectCandidates || cell->ContainsTriangle()) {
            if (!hasDirectCandidates) {
                triangleIndex = cell->TriangleIndex();
            }
            if (staticTriangles != nullptr) {
                const OptimizedCpuStaticMeshTriangleData &triangle =
                        staticTriangles->TriangleAt(triangleIndex);
                SInlineSphereMeshCollide triangleCollide = {
                    &collisionBuffer,
                    sphereCenterMesh,
                    radius,
                    sphereSurface.material,
                    triangle.normal,
                    triangle.material,
                };
                if (triangleCollide.CollideTriangle(triangle)) {
                    hit = 1;
                }
            } else {
                const GmSurfMeshTriangle *triangle =
                        &triangles[triangleIndex];
                const GmVec3 triangleVertices[3] = {
                    vertices[triangle->vertexIndex[0]],
                    vertices[triangle->vertexIndex[1]],
                    vertices[triangle->vertexIndex[2]],
                };
                SInlineSphereMeshCollide triangleCollide = {
                    &collisionBuffer,
                    sphereCenterMesh,
                    radius,
                    sphereSurface.material,
                    triangle->normal,
                    triangle->material,
                };
                if (triangleCollide.CollideTriangle(triangleVertices)) {
                    hit = 1;
                }
            }
        }
    }

    if (hit) {
        TransformSphereCollisionsToWorld(
                collisionBuffer, firstNew, *meshLocated.iso);
    }
    return hit;
}

int RunInlineEllipsoidMeshQuery(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &meshLocated,
        const GmIso4 *meshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar *staticTriangles,
        CGmCollisionBuffer &collisionBuffer) {
    const GmSurfEllipsoid &ellipsoidSurface =
            static_cast<const GmSurfEllipsoid &>(*ellipsoid.surf);
    const GmSurfMesh &mesh =
            static_cast<const GmSurfMesh &>(*meshLocated.surf);
    const GmVec3 radii = ellipsoidSurface.radii;
    const GmVec3 invRadii = {
        1.0f / radii.x,
        1.0f / radii.y,
        1.0f / radii.z,
    };

    GmIso4 ellipsoidToMesh =
            ellipsoid.enabled == 0 ? Identity() : *ellipsoid.iso;
    if (meshLocated.enabled != 0) {
        ellipsoidToMesh = meshInverse == nullptr
                ? MultInverse(ellipsoidToMesh, *meshLocated.iso)
                : Compose(ellipsoidToMesh, *meshInverse);
    }
    const GmVec3 zero = {0.0f, 0.0f, 0.0f};
    const GmBoxAligned ellipsoidBox = TransformBox(
            {zero, radii}, ellipsoidToMesh);
    const bool canUsePlaneCertificate =
            CanUseEllipsoidPlaneCertificate(ellipsoidToMesh, radii);

    const GmIso4 meshToEllipsoid = Inverse(ellipsoidToMesh);
    GmIso4 meshToUnitSphereTransform = meshToEllipsoid;
    ScaleRows(&meshToUnitSphereTransform, invRadii);
    GmIso4 unitSphereContactToWorldTransform =
            DiagonalTransform(radii, zero);
    unitSphereContactToWorldTransform = MultInverse(
            unitSphereContactToWorldTransform, meshToEllipsoid);
    unitSphereContactToWorldTransform = Compose(
            unitSphereContactToWorldTransform, *meshLocated.iso);
    GmIso4 unitSphereNormalToWorldTransform =
            DiagonalTransform(invRadii, zero);
    unitSphereNormalToWorldTransform = MultInverse(
            unitSphereNormalToWorldTransform, meshToEllipsoid);
    unitSphereNormalToWorldTransform = Compose(
            unitSphereNormalToWorldTransform, *meshLocated.iso);

    int hit = 0;
    const std::vector<GmVec3> &vertices =
            GmSurfMeshOptimizedCpuAccess::Vertices(mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshOptimizedCpuAccess::Triangles(mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshOptimizedCpuAccess::OctreeCells(mesh);
    const u32 cellCount = static_cast<u32>(cells.size());
    OptimizedCpuStaticUniformGrid::CandidateSpan candidateSpan;
    const bool hasDirectCandidates = staticTriangles != nullptr &&
            staticTriangles->DirectCandidateTriangleSpan(
                    ellipsoidBox, &candidateSpan);
    u32 sourceCellIndex = 0u;
    std::size_t candidateIndex = 0u;
    while (hasDirectCandidates
                   ? candidateIndex < candidateSpan.size
                   : sourceCellIndex < cellCount) {
        const GmMeshOctreeCell *cell = nullptr;
        const GmBoxAligned *candidateBounds = nullptr;
        u32 triangleIndex = 0u;
        if (hasDirectCandidates) {
            const OptimizedCpuStaticMeshDirectTrianglePosting &posting =
                    staticTriangles->DirectTriangleAt(
                            candidateSpan.data[candidateIndex++]);
            candidateBounds = &posting.bounds;
            triangleIndex = posting.triangleIndex;
        } else {
            cell = &cells[sourceCellIndex];
            candidateBounds = &cell->Bounds();
        }
        if (!BoundsIntersect(ellipsoidBox, *candidateBounds)) {
            if (!hasDirectCandidates) {
                sourceCellIndex += cell->SubtreeEntryCount();
            }
            continue;
        }
        if (!hasDirectCandidates) {
            ++sourceCellIndex;
        }

        if (hasDirectCandidates || cell->ContainsTriangle()) {
            if (!hasDirectCandidates) {
                triangleIndex = cell->TriangleIndex();
            }
            const GmSurfMeshTriangle *triangle = &triangles[triangleIndex];
            const OptimizedCpuStaticMeshTriangleData *staticTriangle =
                    staticTriangles == nullptr
                    ? nullptr
                    : &staticTriangles->TriangleAt(triangleIndex);
            if (staticTriangle != nullptr && canUsePlaneCertificate) {
                const int planeRejection =
                        EllipsoidTrianglePlaneRejection(
                                ellipsoidToMesh, radii, *staticTriangle);
                if (planeRejection != 0) {
                    continue;
                }
            }
            const GmVec3 verticesUnit[3] = {
                TransformPoint(meshToUnitSphereTransform,
                               staticTriangle == nullptr
                                       ? vertices[triangle->vertexIndex[0]]
                                       : staticTriangle->vertices[0]),
                TransformPoint(meshToUnitSphereTransform,
                               staticTriangle == nullptr
                                       ? vertices[triangle->vertexIndex[1]]
                                       : staticTriangle->vertices[1]),
                TransformPoint(meshToUnitSphereTransform,
                               staticTriangle == nullptr
                                       ? vertices[triangle->vertexIndex[2]]
                                       : staticTriangle->vertices[2]),
            };

            const GmVec3 edge01 =
                    Subtract(verticesUnit[1], verticesUnit[0]);
            const GmVec3 edge02 =
                    Subtract(verticesUnit[2], verticesUnit[0]);
            const float normalX =
                    edge02.z * edge01.y - edge02.y * edge01.z;
            const float normalY =
                    edge01.z * edge02.x - edge02.z * edge01.x;
            const float normalZ =
                    edge01.x * edge02.y - edge02.x * edge01.y;
            GmVec3 unitTriangleNormal = {normalX, normalY, normalZ};
            const float normalLenSq =
                    (normalY * normalY + normalX * normalX) +
                    normalZ * normalZ;
            if (normalLenSq >
                PhysicsTolerance::SurfaceDirectionLengthSquared) {
                const float normalLen =
                        NativeBinary32PositiveSqrt(normalLenSq);
                const float invNormalLen = 1.0f / normalLen;
                unitTriangleNormal = {
                    normalX * invNormalLen,
                    normalY * invNormalLen,
                    invNormalLen * normalZ,
                };

                const u32 firstNew = collisionBuffer.GetCurrentCount();
                SInlineSphereMeshCollide triangleCollide = {
                    &collisionBuffer,
                    zero,
                    1.0f,
                    ellipsoidSurface.material,
                    unitTriangleNormal,
                    staticTriangle == nullptr
                            ? triangle->material
                            : staticTriangle->material,
                };
                if (triangleCollide.CollideTriangle(verticesUnit)) {
                    TransformEllipsoidCollisionsToWorld(
                            collisionBuffer,
                            firstNew,
                            unitSphereContactToWorldTransform,
                            unitSphereNormalToWorldTransform);
                    hit = 1;
                }
            }
        }
    }

    return hit;
}

}  // namespace

int GmCollision_Sphere_Mesh_InlineMathOptimizedCpuNativeBinary32(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &meshLocated,
        CGmCollisionBuffer &collisionBuffer) {
    return RunInlineSphereMeshQuery(
            sphere, meshLocated, nullptr, nullptr, collisionBuffer);
}

int GmCollision_Sphere_Mesh_InlineMathOptimizedCpuNativeBinary32WithStaticCache(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &meshLocated,
        const GmIso4 &meshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar &triangles,
        CGmCollisionBuffer &collisionBuffer) {
    const GmSurfMesh &mesh =
            static_cast<const GmSurfMesh &>(*meshLocated.surf);
    const OptimizedCpuStaticMeshTriangleSidecar *usableTriangles =
            triangles.IsFor(mesh) ? &triangles : nullptr;
    return RunInlineSphereMeshQuery(
            sphere,
            meshLocated,
            &meshInverse,
            usableTriangles,
            collisionBuffer);
}

int GmCollision_Ellipsoid_Mesh_InlineMathOptimizedCpuNativeBinary32(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &meshLocated,
        CGmCollisionBuffer &collisionBuffer) {
    return RunInlineEllipsoidMeshQuery(
            ellipsoid, meshLocated, nullptr, nullptr, collisionBuffer);
}

int GmCollision_Ellipsoid_Mesh_InlineMathOptimizedCpuNativeBinary32WithStaticInverse(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &meshLocated,
        const GmIso4 &meshInverse,
        CGmCollisionBuffer &collisionBuffer) {
    return RunInlineEllipsoidMeshQuery(
            ellipsoid,
            meshLocated,
            &meshInverse,
            nullptr,
            collisionBuffer);
}

int ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32(
        const SPlugSurfaceLocatedPair &pairRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const u32 firstNew = collisionBufferRef.GetCurrentCount();
    LocatedGmSurf surfaceB = {
        pairRef.SecondSurface().Geometry(),
        &pairRef.SecondLocation(),
        1,
    };
    LocatedGmSurf surfaceA = {
        pairRef.FirstSurface().Geometry(),
        &pairRef.FirstLocation(),
        1,
    };

    int collided = 0;
    if (surfaceA.surf == nullptr || surfaceB.surf == nullptr ||
        typeid(*surfaceB.surf) != typeid(GmSurfMesh)) {
        collided = GmSurf::ComputeCollision(
                surfaceA, surfaceB, collisionBufferRef);
    } else {
        const std::type_info &typeA = typeid(*surfaceA.surf);
        if (typeA == typeid(GmSurfSphere)) {
            collided =
                    GmCollision_Sphere_Mesh_InlineMathOptimizedCpuNativeBinary32(
                            surfaceA, surfaceB, collisionBufferRef);
        } else if (typeA == typeid(GmSurfEllipsoid)) {
            collided =
                    GmCollision_Ellipsoid_Mesh_InlineMathOptimizedCpuNativeBinary32(
                            surfaceA, surfaceB, collisionBufferRef);
        } else {
            collided = GmSurf::ComputeCollision(
                    surfaceA, surfaceB, collisionBufferRef);
        }
    }
    if (!collided) {
        return 0;
    }

    const u32 count = collisionBufferRef.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         collisionIndex++) {
        GmCollision &collision =
                collisionBufferRef.GetCollision(collisionIndex);
        collision.materialA =
                pairRef.FirstSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialA);
        collision.materialB =
                pairRef.SecondSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialB);
    }

    return 1;
}

#if defined(FOREVERVALIDATOR_RELEASE_IPO_BUILD) && \
        defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
int ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32WithStaticCache(
        const SPlugSurfaceLocatedPair &pairRef,
        const GmIso4 &staticMeshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar *triangles,
        CGmCollisionBuffer &collisionBufferRef) {
    const u32 firstNew = collisionBufferRef.GetCurrentCount();
    LocatedGmSurf surfaceB = {
        pairRef.SecondSurface().Geometry(),
        &pairRef.SecondLocation(),
        1,
    };
    LocatedGmSurf surfaceA = {
        pairRef.FirstSurface().Geometry(),
        &pairRef.FirstLocation(),
        1,
    };

    int collided = 0;
    if (surfaceA.surf == nullptr || surfaceB.surf == nullptr ||
        typeid(*surfaceB.surf) != typeid(GmSurfMesh)) {
        collided = GmSurf::ComputeCollision(
                surfaceA, surfaceB, collisionBufferRef);
    } else {
        const OptimizedCpuStaticMeshTriangleSidecar *usableTriangles =
                triangles;
        const std::type_info &typeA = typeid(*surfaceA.surf);
        if (typeA == typeid(GmSurfSphere)) {
            collided = RunInlineSphereMeshQuery(
                    surfaceA,
                    surfaceB,
                    &staticMeshInverse,
                    usableTriangles,
                    collisionBufferRef);
        } else if (typeA == typeid(GmSurfEllipsoid)) {
            collided = RunInlineEllipsoidMeshQuery(
                    surfaceA,
                    surfaceB,
                    &staticMeshInverse,
                    usableTriangles,
                    collisionBufferRef);
        } else {
            collided = GmSurf::ComputeCollision(
                    surfaceA, surfaceB, collisionBufferRef);
        }
    }
    if (!collided) {
        return 0;
    }

    const u32 count = collisionBufferRef.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         ++collisionIndex) {
        GmCollision &collision =
                collisionBufferRef.GetCollision(collisionIndex);
        collision.materialA =
                pairRef.FirstSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialA);
        collision.materialB =
                pairRef.SecondSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialB);
    }

    return 1;
}
#if defined(FOREVERVALIDATOR_RELEASE_IPO_BUILD) && \
        defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#undef FV_E012_ALWAYS_INLINE
#undef FV_E012_NOINLINE
#undef FV_E024_HAS_INLINE_SSE2
