// OptimizedCpu mesh queries using an immutable static-surface inverse.

#include "simulation/backends/optimized_cpu/optimized_cpu_static_inverse_mesh_queries.h"

#include <cmath>
#include <typeinfo>
#include <vector>

#include "engine/core/binary32_math.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"

struct GmSurfMeshStaticInverseOptimizedCpuAccess {
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

struct SStaticInverseSphereMeshCollideOptimizedCpu {
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

int SStaticInverseSphereMeshCollideOptimizedCpu::EmitFeatureCollision(
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

int SStaticInverseSphereMeshCollideOptimizedCpu::EmitEndpointBCollision(
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

int SStaticInverseSphereMeshCollideOptimizedCpu::CollideTriangle(
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

void TransformStaticInverseMeshCollisionsToWorldOptimizedCpu(
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

void TransformStaticInverseEllipsoidMeshCollisionsToWorldOptimizedCpu(
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

bool StaticInverseOptimizedCpuBoundsIntersect(
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

int FinishStaticInverseSurfaceMaterials(
        const SPlugSurfaceLocatedPair &pair,
        u32 firstNew,
        int collided,
        CGmCollisionBuffer &collisionBuffer) {
    if (!collided) {
        return 0;
    }
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         ++collisionIndex) {
        GmCollision &collision =
                collisionBuffer.GetCollision(collisionIndex);
        collision.materialA =
                pair.FirstSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialA);
        collision.materialB =
                pair.SecondSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialB);
    }
    return 1;
}

}  // namespace

int GmCollision_Sphere_Mesh_OptimizedCpuWithMeshInverse(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &meshLocatedRef,
        const GmIso4 &meshInverse,
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
        sphereToMesh.Mult(meshInverse);
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
            GmSurfMeshStaticInverseOptimizedCpuAccess::Vertices(*mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshStaticInverseOptimizedCpuAccess::Triangles(*mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshStaticInverseOptimizedCpuAccess::OctreeCells(*mesh);
    const u32 cellCount = static_cast<u32>(cells.size());

    for (u32 cellIndex = 0; cellIndex < cellCount;) {
        const GmMeshOctreeCell *cell = &cells[cellIndex];
        if (!StaticInverseOptimizedCpuBoundsIntersect(
                    sphereBox, cell->Bounds())) {
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
            SStaticInverseSphereMeshCollideOptimizedCpu triangleCollide = {
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
        TransformStaticInverseMeshCollisionsToWorldOptimizedCpu(
                *collisionBuffer, firstNew, *meshLocated->iso);
    }
    return hit;
}

int GmCollision_Ellipsoid_Mesh_OptimizedCpuWithMeshInverse(
        const LocatedGmSurf &ellipsoidRef,
        const LocatedGmSurf &meshLocatedRef,
        const GmIso4 &meshInverse,
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
        ellipsoidToMesh.Mult(meshInverse);
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
            GmSurfMeshStaticInverseOptimizedCpuAccess::Vertices(*mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshStaticInverseOptimizedCpuAccess::Triangles(*mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshStaticInverseOptimizedCpuAccess::OctreeCells(*mesh);
    const u32 cellCount = static_cast<u32>(cells.size());
    for (u32 cellIndex = 0; cellIndex < cellCount;) {
        const GmMeshOctreeCell *cell = &cells[cellIndex];
        if (!StaticInverseOptimizedCpuBoundsIntersect(
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
                SStaticInverseSphereMeshCollideOptimizedCpu triangleCollide = {
                    collisionBuffer,
                    zero,
                    1.0f,
                    ellipsoid->LocalMaterial(),
                    unitTriangleNormal,
                    triangle->material,
                };
                if (triangleCollide.CollideTriangle(verticesUnit)) {
                    TransformStaticInverseEllipsoidMeshCollisionsToWorldOptimizedCpu(
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

int ComputeCollisionOptimizedCpuWithStaticMeshInverse(
        const SPlugSurfaceLocatedPair &pair,
        const GmIso4 &staticMeshInverse,
        CGmCollisionBuffer &collisionBuffer) {
    const GmSurf *surfaceAGeometry = pair.FirstSurface().Geometry();
    const GmSurf *surfaceBGeometry = pair.SecondSurface().Geometry();
    if (surfaceAGeometry == nullptr || surfaceBGeometry == nullptr ||
        typeid(*surfaceBGeometry) != typeid(GmSurfMesh)) {
        return CPlugSurface::ComputeCollisionOptimizedCpu(
                pair, collisionBuffer);
    }

    const std::type_info &typeA = typeid(*surfaceAGeometry);
    if (typeA != typeid(GmSurfSphere) &&
        typeA != typeid(GmSurfEllipsoid)) {
        return CPlugSurface::ComputeCollisionOptimizedCpu(
                pair, collisionBuffer);
    }

    const u32 firstNew = collisionBuffer.GetCurrentCount();
    const LocatedGmSurf surfaceA = {
        surfaceAGeometry,
        &pair.FirstLocation(),
        true,
    };
    const LocatedGmSurf surfaceB = {
        surfaceBGeometry,
        &pair.SecondLocation(),
        true,
    };
    const int collided = typeA == typeid(GmSurfSphere)
            ? GmCollision_Sphere_Mesh_OptimizedCpuWithMeshInverse(
                      surfaceA,
                      surfaceB,
                      staticMeshInverse,
                      collisionBuffer)
            : GmCollision_Ellipsoid_Mesh_OptimizedCpuWithMeshInverse(
                      surfaceA,
                      surfaceB,
                      staticMeshInverse,
                      collisionBuffer);
    return FinishStaticInverseSurfaceMaterials(
            pair, firstNew, collided, collisionBuffer);
}
