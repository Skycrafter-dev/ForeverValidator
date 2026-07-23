// OptimizedCpu-only native-binary32 sphere and ellipsoid mesh queries.

#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"

#include <cmath>
#include <typeinfo>
#include <vector>

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"
#include "engine/physics/geometry/plug_surface.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"

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

namespace {

float NativeBinary32Sqrt(float value) noexcept {
    return forevervalidator::simulation::
            OptimizedCpuBinary32SqrtX86Sse2(value);
}

GmVec3 NormalizeForCollisionNativeBinary32(
        const GmVec3 &value,
        float epsilonSq) {
    GmVec3 out = value;
    const float lengthSq = out.Dot(out);
    if (epsilonSq < lengthSq) {
        out = out.ScaleForCollision(
                1.0f / NativeBinary32Sqrt(lengthSq));
    }
    return out;
}

bool NativeBinary32BoundsIntersect(const GmBoxAligned &query,
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

struct SSphereMeshCollideOptimizedCpuNativeBinary32 {
    CGmCollisionBuffer *collisionBuffer;
    GmVec3 sphereCenterMesh;
    float radius;
    GmLocalMaterialIndex materialA;
    GmVec3 triangleNormal;
    GmLocalMaterialIndex triangleMaterial;

    int EmitFeatureCollision(GmVec3 featurePointLocal,
                             float minDistanceSq,
                             bool requireRadiusContainment);
    int EmitEndpointBCollision(GmVec3 featurePointLocal, float minDistance);
    int CollideTriangle(const GmVec3 vertices[3]);
};

int SSphereMeshCollideOptimizedCpuNativeBinary32::EmitFeatureCollision(
        GmVec3 featurePointLocal,
        float minDistanceSq,
        bool requireRadiusContainment) {
    const GmVec3 featureToCenter =
        sphereCenterMesh.SubtractForCollision(featurePointLocal);
    const float distanceSq = (featureToCenter.Dot(featureToCenter));
    if ((requireRadiusContainment && radius * radius < distanceSq) ||
        !(minDistanceSq < distanceSq)) {
        return 0;
    }

    const float distance = (NativeBinary32Sqrt(distanceSq));
    const float invDistance = (1.0f / distance);
    const GmVec3 normal = ((GmVec3){
        (featureToCenter.x * invDistance),
        (featureToCenter.y * invDistance),
        (featureToCenter.z * invDistance),
    });
    const float distanceMinusRadius = (distance - radius);
    const float penetrationScale = (distanceMinusRadius * invDistance);
    const GmVec3 penetration = ((GmVec3){
        (featureToCenter.x * penetrationScale),
        (featureToCenter.y * penetrationScale),
        (featureToCenter.z * penetrationScale),
    });
    const float separationAlongTriangleNormal =
        (penetration.Dot(triangleNormal));

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->impulseNormal = normal;
    collision->separation =
        triangleNormal.ScaleForCollision(separationAlongTriangleNormal);
    collision->contactPoint = featurePointLocal;
    collision->localMaterialA = materialA;
    collision->localMaterialB = triangleMaterial;
    collision->sphereMergePrimary = 0;
    collision->extraNegated = triangleNormal;
    return 1;
}

int SSphereMeshCollideOptimizedCpuNativeBinary32::EmitEndpointBCollision(
        GmVec3 featurePointLocal,
        float minDistance) {
    const GmVec3 featureToCenter =
        sphereCenterMesh.SubtractForCollision(featurePointLocal);
    const float distanceSq = (featureToCenter.Dot(featureToCenter));
    const float distance = (NativeBinary32Sqrt(distanceSq));
    if (radius * radius < distance || !(minDistance < distance)) {
        return 0;
    }

    const float endpointDistance = (NativeBinary32Sqrt(distance));
    const float invEndpointDistance = (1.0f / endpointDistance);
    const GmVec3 normal = ((GmVec3){
        (featureToCenter.x * invEndpointDistance),
        (featureToCenter.y * invEndpointDistance),
        (featureToCenter.z * invEndpointDistance),
    });
    const float endpointDistanceMinusRadius = (endpointDistance - radius);
    const float endpointPenetrationScale =
        (endpointDistanceMinusRadius * invEndpointDistance);
    const GmVec3 penetration = ((GmVec3){
        (featureToCenter.x * endpointPenetrationScale),
        (featureToCenter.y * endpointPenetrationScale),
        (featureToCenter.z * endpointPenetrationScale),
    });
    const float separationAlongTriangleNormal =
        (penetration.Dot(triangleNormal));

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->impulseNormal = normal;
    collision->separation =
        triangleNormal.ScaleForCollision(separationAlongTriangleNormal);
    collision->contactPoint = featurePointLocal;
    collision->localMaterialA = materialA;
    collision->localMaterialB = triangleMaterial;
    collision->sphereMergePrimary = 0;
    collision->extraNegated = triangleNormal;
    return 1;
}

int SSphereMeshCollideOptimizedCpuNativeBinary32::CollideTriangle(
        const GmVec3 vertices[3]) {
    const float planeDistance =
        sphereCenterMesh.SubtractForCollision(vertices[0]).Dot(triangleNormal);
    if (radius < planeDistance || planeDistance < 0.0f) {
        return 0;
    }

    const float edgeReach = NativeBinary32Sqrt(
            radius * radius - planeDistance * planeDistance);
    const GmVec3 projectedPoint =
        sphereCenterMesh.AddForCollision(
                triangleNormal.ScaleForCollision(-planeDistance));

    for (u32 edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
        const u32 nextIndex = edgeIndex == 2u ? 0u : edgeIndex + 1u;
        const GmVec3 edgeStart = vertices[edgeIndex];
        const GmVec3 edgeEnd = vertices[nextIndex];
        const GmVec3 edgeDir = NormalizeForCollisionNativeBinary32(
                edgeEnd.SubtractForCollision(edgeStart),
                PhysicsTolerance::SurfaceDirectionLengthSquared);
        const GmVec3 edgeNormal = edgeDir.Cross(triangleNormal);
        const float edgeDistance =
            projectedPoint.SubtractForCollision(edgeStart).Dot(edgeNormal);

        if (edgeReach < edgeDistance) {
            return 0;
        }

        if (edgeDistance > 0.0f) {
            const float alongFromStart =
                projectedPoint.SubtractForCollision(edgeStart).Dot(edgeDir);
            if (alongFromStart < 0.0f) {
                return EmitFeatureCollision(
                        edgeStart,
                        PhysicsTolerance::SurfaceDirectionLengthSquared,
                        true);
            }

            const float alongFromEnd =
                projectedPoint.SubtractForCollision(edgeEnd).Dot(edgeDir);
            if (!(0.0f < alongFromEnd)) {
                const GmVec3 featurePoint =
                    projectedPoint.AddForCollision(
                            edgeNormal.ScaleForCollision(-edgeDistance));
                return EmitFeatureCollision(featurePoint,
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
                triangleNormal.ScaleForCollision(planeDistance - radius);
        collision->contactPoint = projectedPoint;
        collision->localMaterialA = materialA;
        collision->localMaterialB = triangleMaterial;
        collision->sphereMergePrimary = 1;
        collision->extraNegated = triangleNormal;
        return 1;
    }

    return 0;
}

void TransformSphereMeshCollisionsToWorldNativeBinary32(
        CGmCollisionBuffer &collisionBuffer,
        u32 firstNew,
        const GmIso4 &meshIso) {
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         collisionIndex++) {
        GmCollision *collision =
                &collisionBuffer.GetCollision(collisionIndex);
        const GmMat3 meshRotation = meshIso.RotationMatrix();
        collision->impulseNormal.Mult(meshRotation);
        collision->separation.Mult(meshRotation);
        collision->contactPoint.Mult(meshIso);
    }
}

struct SEllipsoidMeshCollideOptimizedCpuNativeBinary32 {
    CGmCollisionBuffer *collisionBuffer;
    GmVec3 sphereCenterMesh;
    float radius;
    GmLocalMaterialIndex materialA;
    GmVec3 triangleNormal;
    GmLocalMaterialIndex triangleMaterial;

    int EmitFeatureCollision(GmVec3 featurePointLocal,
                             float minDistanceSq,
                             bool requireRadiusContainment);
    int EmitEndpointBCollision(GmVec3 featurePointLocal, float minDistance);
    int CollideTriangle(const GmVec3 vertices[3]);
};

int SEllipsoidMeshCollideOptimizedCpuNativeBinary32::EmitFeatureCollision(
        GmVec3 featurePointLocal,
        float minDistanceSq,
        bool requireRadiusContainment) {
    const GmVec3 featureToCenter =
        sphereCenterMesh.SubtractForCollision(featurePointLocal);
    const float distanceSq = (featureToCenter.Dot(featureToCenter));
    if ((requireRadiusContainment && radius * radius < distanceSq) ||
        !(minDistanceSq < distanceSq)) {
        return 0;
    }

    const float distance = (NativeBinary32Sqrt(distanceSq));
    const float invDistance = (1.0f / distance);
    const GmVec3 normal = ((GmVec3){
        (featureToCenter.x * invDistance),
        (featureToCenter.y * invDistance),
        (featureToCenter.z * invDistance),
    });
    const float distanceMinusRadius = (distance - radius);
    const float penetrationScale = (distanceMinusRadius * invDistance);
    const GmVec3 penetration = ((GmVec3){
        (featureToCenter.x * penetrationScale),
        (featureToCenter.y * penetrationScale),
        (featureToCenter.z * penetrationScale),
    });
    const float separationAlongTriangleNormal =
        (penetration.Dot(triangleNormal));

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->impulseNormal = normal;
    collision->separation =
        triangleNormal.ScaleForCollision(separationAlongTriangleNormal);
    collision->contactPoint = featurePointLocal;
    collision->localMaterialA = materialA;
    collision->localMaterialB = triangleMaterial;
    collision->sphereMergePrimary = 0;
    collision->extraNegated = triangleNormal;
    return 1;
}

int SEllipsoidMeshCollideOptimizedCpuNativeBinary32::EmitEndpointBCollision(
        GmVec3 featurePointLocal,
        float minDistance) {
    const GmVec3 featureToCenter =
        sphereCenterMesh.SubtractForCollision(featurePointLocal);
    const float distanceSq = (featureToCenter.Dot(featureToCenter));
    const float distance = (NativeBinary32Sqrt(distanceSq));
    if (radius * radius < distance || !(minDistance < distance)) {
        return 0;
    }

    const float endpointDistance = (NativeBinary32Sqrt(distance));
    const float invEndpointDistance = (1.0f / endpointDistance);
    const GmVec3 normal = ((GmVec3){
        (featureToCenter.x * invEndpointDistance),
        (featureToCenter.y * invEndpointDistance),
        (featureToCenter.z * invEndpointDistance),
    });
    const float endpointDistanceMinusRadius = (endpointDistance - radius);
    const float endpointPenetrationScale =
        (endpointDistanceMinusRadius * invEndpointDistance);
    const GmVec3 penetration = ((GmVec3){
        (featureToCenter.x * endpointPenetrationScale),
        (featureToCenter.y * endpointPenetrationScale),
        (featureToCenter.z * endpointPenetrationScale),
    });
    const float separationAlongTriangleNormal =
        (penetration.Dot(triangleNormal));

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->impulseNormal = normal;
    collision->separation =
        triangleNormal.ScaleForCollision(separationAlongTriangleNormal);
    collision->contactPoint = featurePointLocal;
    collision->localMaterialA = materialA;
    collision->localMaterialB = triangleMaterial;
    collision->sphereMergePrimary = 0;
    collision->extraNegated = triangleNormal;
    return 1;
}

int SEllipsoidMeshCollideOptimizedCpuNativeBinary32::CollideTriangle(
        const GmVec3 vertices[3]) {
    const float planeDistance =
        sphereCenterMesh.SubtractForCollision(vertices[0]).Dot(triangleNormal);
    if (radius < planeDistance || planeDistance < 0.0f) {
        return 0;
    }

    const float edgeReach = NativeBinary32Sqrt(
            radius * radius - planeDistance * planeDistance);
    const GmVec3 projectedPoint =
        sphereCenterMesh.AddForCollision(
                triangleNormal.ScaleForCollision(-planeDistance));

    for (u32 edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
        const u32 nextIndex = edgeIndex == 2u ? 0u : edgeIndex + 1u;
        const GmVec3 edgeStart = vertices[edgeIndex];
        const GmVec3 edgeEnd = vertices[nextIndex];
        const GmVec3 edgeDir = NormalizeForCollisionNativeBinary32(
                edgeEnd.SubtractForCollision(edgeStart),
                PhysicsTolerance::SurfaceDirectionLengthSquared);
        const GmVec3 edgeNormal = edgeDir.Cross(triangleNormal);
        const float edgeDistance =
            projectedPoint.SubtractForCollision(edgeStart).Dot(edgeNormal);

        if (edgeReach < edgeDistance) {
            return 0;
        }

        if (edgeDistance > 0.0f) {
            const float alongFromStart =
                projectedPoint.SubtractForCollision(edgeStart).Dot(edgeDir);
            if (alongFromStart < 0.0f) {
                return EmitFeatureCollision(
                        edgeStart,
                        PhysicsTolerance::SurfaceDirectionLengthSquared,
                        true);
            }

            const float alongFromEnd =
                projectedPoint.SubtractForCollision(edgeEnd).Dot(edgeDir);
            if (!(0.0f < alongFromEnd)) {
                const GmVec3 featurePoint =
                    projectedPoint.AddForCollision(
                            edgeNormal.ScaleForCollision(-edgeDistance));
                return EmitFeatureCollision(featurePoint,
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
                triangleNormal.ScaleForCollision(planeDistance - radius);
        collision->contactPoint = projectedPoint;
        collision->localMaterialA = materialA;
        collision->localMaterialB = triangleMaterial;
        collision->sphereMergePrimary = 1;
        collision->extraNegated = triangleNormal;
        return 1;
    }

    return 0;
}

void TransformEllipsoidMeshCollisionsToWorldNativeBinary32(
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
        collision->contactPoint.Mult(unitContactToWorld);
        collision->impulseNormal.MultEllipsoidMeshWorldNormalForGmSurf(
                unitNormalToWorld.rotation);
        collision->impulseNormal = NormalizeForCollisionNativeBinary32(
                collision->impulseNormal,
                PhysicsTolerance::SurfaceDirectionLengthSquared);
        collision->separation.Mult(unitContactToWorld.RotationMatrix());
    }
}

}  // namespace

int GmCollision_Sphere_Mesh_OptimizedCpuNativeBinary32(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &meshLocatedRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *sphere = &sphereRef;
    const LocatedGmSurf *meshLocated = &meshLocatedRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmSurfMesh *mesh = meshLocated->Mesh();
    const float radius = sphere->SphereRadius();

    GmIso4 sphereToMesh;
    if (sphere->enabled == 0) {
        sphereToMesh.SetIdentity();
    } else {
        sphereToMesh = *sphere->iso;
    }
    if (meshLocated->enabled != 0) {
        sphereToMesh.MultInverse(*meshLocated->iso);
    }

    const u32 firstNew = collisionBuffer->GetCurrentCount();
    const GmVec3 zero = {0.0f, 0.0f, 0.0f};
    const GmVec3 sphereHalf = {radius, radius, radius};
    GmBoxAligned sphereBox =
            GmBoxAligned::FromCenterHalfExtents(zero, sphereHalf);
    sphereBox.Mult(sphereToMesh);

    const GmVec3 sphereCenterMesh = sphereToMesh.TranslationForGmSurf();
    int hit = 0;
    const std::vector<GmVec3> &vertices =
            GmSurfMeshOptimizedCpuAccess::Vertices(*mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshOptimizedCpuAccess::Triangles(*mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshOptimizedCpuAccess::OctreeCells(*mesh);
    const u32 cellCount = static_cast<u32>(cells.size());

    for (u32 cellIndex = 0; cellIndex < cellCount;) {
        const GmMeshOctreeCell *cell = &cells[cellIndex];
        if (!NativeBinary32BoundsIntersect(sphereBox, cell->Bounds())) {
            cellIndex += cell->SubtreeEntryCount();
            continue;
        }

        if (cell->ContainsTriangle()) {
            const GmSurfMeshTriangle *triangle =
                    &triangles[cell->TriangleIndex()];
            const GmVec3 triangleVertices[3] = {
                vertices[triangle->vertexIndex[0]],
                vertices[triangle->vertexIndex[1]],
                vertices[triangle->vertexIndex[2]],
            };
            SSphereMeshCollideOptimizedCpuNativeBinary32 triangleCollide = {
                collisionBuffer,
                sphereCenterMesh,
                radius,
                sphere->LocalMaterial(),
                triangle->normal,
                triangle->material,
            };
            if (triangleCollide.CollideTriangle(triangleVertices)) {
                hit = 1;
            }
        }

        cellIndex++;
    }

    if (hit) {
        TransformSphereMeshCollisionsToWorldNativeBinary32(
                *collisionBuffer, firstNew, *meshLocated->iso);
    }
    return hit;
}

int GmCollision_Ellipsoid_Mesh_OptimizedCpuNativeBinary32(
        const LocatedGmSurf &ellipsoidRef,
        const LocatedGmSurf &meshLocatedRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *ellipsoid = &ellipsoidRef;
    const LocatedGmSurf *meshLocated = &meshLocatedRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmSurfMesh *mesh = meshLocated->Mesh();
    const GmVec3 radii = ellipsoid->EllipsoidRadii();
    const GmVec3 invRadii = {
        (1.0f / radii.x),
        (1.0f / radii.y),
        (1.0f / radii.z),
    };

    GmIso4 ellipsoidToMesh;
    if (ellipsoid->enabled == 0) {
        ellipsoidToMesh.SetIdentity();
    } else {
        ellipsoidToMesh = *ellipsoid->iso;
    }
    if (meshLocated->enabled != 0) {
        ellipsoidToMesh.MultInverse(*meshLocated->iso);
    }
    const GmVec3 zero = {0.0f, 0.0f, 0.0f};
    GmBoxAligned ellipsoidBox =
            GmBoxAligned::FromCenterHalfExtents(zero, radii);
    ellipsoidBox.Mult(ellipsoidToMesh);

    GmIso4 meshToEllipsoid;
    meshToEllipsoid.SetInverse(ellipsoidToMesh);
    GmIso4 meshToUnitSphereTransform;
    meshToUnitSphereTransform = meshToEllipsoid;
    meshToUnitSphereTransform.ScaleRowsForGmSurf(invRadii);
    GmIso4 unitSphereContactToWorldTransform;
    unitSphereContactToWorldTransform.SetNUScaleTrans(radii, zero);
    unitSphereContactToWorldTransform.MultInverse(meshToEllipsoid);
    unitSphereContactToWorldTransform.Mult(*meshLocated->iso);
    GmIso4 unitSphereNormalToWorldTransform;
    unitSphereNormalToWorldTransform.SetNUScaleTrans(invRadii, zero);
    unitSphereNormalToWorldTransform.MultInverse(meshToEllipsoid);
    unitSphereNormalToWorldTransform.Mult(*meshLocated->iso);

    int hit = 0;
    const std::vector<GmVec3> &vertices =
            GmSurfMeshOptimizedCpuAccess::Vertices(*mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshOptimizedCpuAccess::Triangles(*mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshOptimizedCpuAccess::OctreeCells(*mesh);
    const u32 cellCount = static_cast<u32>(cells.size());
    for (u32 cellIndex = 0; cellIndex < cellCount;) {
        const GmMeshOctreeCell *cell = &cells[cellIndex];
        if (!NativeBinary32BoundsIntersect(ellipsoidBox, cell->Bounds())) {
            cellIndex += cell->SubtreeEntryCount();
            continue;
        }

        if (cell->ContainsTriangle()) {
            const GmSurfMeshTriangle *triangle =
                    &triangles[cell->TriangleIndex()];
            GmVec3 verticesUnit[3] = {
                meshToUnitSphereTransform.SetMultPointForGmSurf(
                        vertices[triangle->vertexIndex[0]]),
                meshToUnitSphereTransform.SetMultPointForGmSurf(
                        vertices[triangle->vertexIndex[1]]),
                meshToUnitSphereTransform.SetMultPointForGmSurf(
                        vertices[triangle->vertexIndex[2]]),
            };

            const GmVec3 edge01 =
                    verticesUnit[1].SubtractForCollision(verticesUnit[0]);
            const GmVec3 edge02 =
                    verticesUnit[2].SubtractForCollision(verticesUnit[0]);
            const float normalX = (edge02.z * edge01.y -
                                              edge02.y * edge01.z);
            const float normalY = (edge01.z * edge02.x -
                                              edge02.z * edge01.x);
            const float normalZ = (edge01.x * edge02.y -
                                              edge02.x * edge01.y);
            GmVec3 unitTriangleNormal = {normalX, normalY, normalZ};
            const float normalLenSq = (
                (normalY * normalY + normalX * normalX) +
                normalZ * normalZ);
            if (normalLenSq >
                    PhysicsTolerance::SurfaceDirectionLengthSquared) {
                const float normalLen =
                        (NativeBinary32Sqrt(normalLenSq));
                const float invNormalLen = (1.0f / normalLen);
                unitTriangleNormal = (GmVec3){
                    (normalX * invNormalLen),
                    (normalY * invNormalLen),
                    (invNormalLen * normalZ),
                };

                const u32 firstNew = collisionBuffer->GetCurrentCount();
                SEllipsoidMeshCollideOptimizedCpuNativeBinary32
                        triangleCollide = {
                    collisionBuffer,
                    zero,
                    1.0f,
                    ellipsoid->LocalMaterial(),
                    unitTriangleNormal,
                    triangle->material,
                };
                if (triangleCollide.CollideTriangle(verticesUnit)) {
                    TransformEllipsoidMeshCollisionsToWorldNativeBinary32(
                            *collisionBuffer,
                            firstNew,
                            unitSphereContactToWorldTransform,
                            unitSphereNormalToWorldTransform);
                    hit = 1;
                }
            }
        }

        cellIndex++;
    }

    return hit;
}

int ComputePlugSurfaceCollisionOptimizedCpuNativeBinary32(
        const SPlugSurfaceLocatedPair &pairRef,
        CGmCollisionBuffer &collisionBufferRef) {
    u32 firstNew = collisionBufferRef.GetCurrentCount();
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
            collided = GmCollision_Sphere_Mesh_OptimizedCpuNativeBinary32(
                    surfaceA, surfaceB, collisionBufferRef);
        } else if (typeA == typeid(GmSurfEllipsoid)) {
            collided = GmCollision_Ellipsoid_Mesh_OptimizedCpuNativeBinary32(
                    surfaceA, surfaceB, collisionBufferRef);
        } else {
            collided = GmSurf::ComputeCollision(
                    surfaceA, surfaceB, collisionBufferRef);
        }
    }
    if (!collided) {
        return 0;
    }

    u32 count = collisionBufferRef.GetCurrentCount();
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
