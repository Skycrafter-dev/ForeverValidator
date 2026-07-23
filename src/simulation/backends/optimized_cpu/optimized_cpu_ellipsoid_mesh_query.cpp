// OptimizedCpu-only ellipsoid query against triangle meshes.

#include <cmath>
#include <vector>

#include "engine/core/binary32_math.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"

struct GmSurfMeshEllipsoidOptimizedCpuAccess {
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

struct SEllipsoidMeshCollideOptimizedCpu {
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

int SEllipsoidMeshCollideOptimizedCpu::EmitFeatureCollision(
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

    const float distance = (CIsqrt(distanceSq));
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

int SEllipsoidMeshCollideOptimizedCpu::EmitEndpointBCollision(
        GmVec3 featurePointLocal,
        float minDistance) {
    const GmVec3 featureToCenter =
        sphereCenterMesh.SubtractForCollision(featurePointLocal);
    const float distanceSq = (featureToCenter.Dot(featureToCenter));
    const float distance = (CIsqrt(distanceSq));
    if (radius * radius < distance || !(minDistance < distance)) {
        return 0;
    }

    const float endpointDistance = (CIsqrt(distance));
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

int SEllipsoidMeshCollideOptimizedCpu::CollideTriangle(
        const GmVec3 vertices[3]) {
    const float planeDistance =
        sphereCenterMesh.SubtractForCollision(vertices[0]).Dot(triangleNormal);
    if (radius < planeDistance || planeDistance < 0.0f) {
        return 0;
    }

    const float edgeReach = CIsqrt(
            radius * radius - planeDistance * planeDistance);
    const GmVec3 projectedPoint =
        sphereCenterMesh.AddForCollision(
                triangleNormal.ScaleForCollision(-planeDistance));

    for (u32 edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
        const u32 nextIndex = edgeIndex == 2u ? 0u : edgeIndex + 1u;
        const GmVec3 edgeStart = vertices[edgeIndex];
        const GmVec3 edgeEnd = vertices[nextIndex];
        const GmVec3 edgeDir =
            edgeEnd.SubtractForCollision(edgeStart).NormalizeForCollision(
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

void TransformEllipsoidMeshCollisionsToWorldOptimizedCpu(
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
        collision->impulseNormal =
            collision->impulseNormal.NormalizeForCollision(
                    PhysicsTolerance::SurfaceDirectionLengthSquared);
        collision->separation.Mult(unitContactToWorld.RotationMatrix());
    }
}

bool OptimizedCpuEllipsoidBoundsIntersect(const GmBoxAligned &query,
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

}  // namespace

int GmCollision_Ellipsoid_Mesh_OptimizedCpu(
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
            GmSurfMeshEllipsoidOptimizedCpuAccess::Vertices(*mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshEllipsoidOptimizedCpuAccess::Triangles(*mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshEllipsoidOptimizedCpuAccess::OctreeCells(*mesh);
    const u32 cellCount = static_cast<u32>(cells.size());
    for (u32 cellIndex = 0; cellIndex < cellCount;) {
        const GmMeshOctreeCell *cell = &cells[cellIndex];
        if (!OptimizedCpuEllipsoidBoundsIntersect(
                    ellipsoidBox, cell->Bounds())) {
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
                const float normalLen = (CIsqrt(normalLenSq));
                const float invNormalLen = (1.0f / normalLen);
                unitTriangleNormal = (GmVec3){
                    (normalX * invNormalLen),
                    (normalY * invNormalLen),
                    (invNormalLen * normalZ),
                };

                const u32 firstNew = collisionBuffer->GetCurrentCount();
                SEllipsoidMeshCollideOptimizedCpu triangleCollide = {
                    collisionBuffer,
                    zero,
                    1.0f,
                    ellipsoid->LocalMaterial(),
                    unitTriangleNormal,
                    triangle->material,
                };
                if (triangleCollide.CollideTriangle(verticesUnit)) {
                    TransformEllipsoidMeshCollisionsToWorldOptimizedCpu(
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
