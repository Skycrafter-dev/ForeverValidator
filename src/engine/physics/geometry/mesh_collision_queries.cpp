// Sphere, ellipsoid, and box queries against triangle meshes.

#include <cmath>

#include "engine/core/binary32_math.h"
#include "engine/physics/geometry/geometry_helpers.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"

struct SSphereMeshCollide {
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

int SSphereMeshCollide::EmitFeatureCollision(GmVec3 featurePointLocal,
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
    const GmVec3 normal = (GmVec3{
        (featureToCenter.x * invDistance),
        (featureToCenter.y * invDistance),
        (featureToCenter.z * invDistance),
    });
    const float distanceMinusRadius = (distance - radius);
    const float penetrationScale = (distanceMinusRadius * invDistance);
    const GmVec3 penetration = (GmVec3{
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

int SSphereMeshCollide::EmitEndpointBCollision(GmVec3 featurePointLocal,
                                               float minDistance) {
    const GmVec3 featureToCenter =
        sphereCenterMesh.SubtractForCollision(featurePointLocal);
    const float distanceSq = (featureToCenter.Dot(featureToCenter));
    const float distance = (CIsqrt(distanceSq));
    if (radius * radius < distance || !(minDistance < distance)) {
        return 0;
    }

    // Endpoint B intentionally uses a second root for its asymmetric response.
    const float endpointDistance = (CIsqrt(distance));
    const float invEndpointDistance = (1.0f / endpointDistance);
    const GmVec3 normal = (GmVec3{
        (featureToCenter.x * invEndpointDistance),
        (featureToCenter.y * invEndpointDistance),
        (featureToCenter.z * invEndpointDistance),
    });
    const float endpointDistanceMinusRadius = (endpointDistance - radius);
    const float endpointPenetrationScale =
        (endpointDistanceMinusRadius * invEndpointDistance);
    const GmVec3 penetration = (GmVec3{
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

int SSphereMeshCollide::CollideTriangle(const GmVec3 vertices[3]) {
    const float planeDistance =
        sphereCenterMesh.SubtractForCollision(vertices[0]).Dot(triangleNormal);
    if (radius < planeDistance || planeDistance < 0.0f) {
        return 0;
    }

    const float edgeReach = CIsqrt(radius * radius - planeDistance * planeDistance);
    const GmVec3 projectedPoint =
        sphereCenterMesh.AddForCollision(triangleNormal.ScaleForCollision(-planeDistance));

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
                    projectedPoint.AddForCollision(edgeNormal.ScaleForCollision(-edgeDistance));
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
        collision->separation = triangleNormal.ScaleForCollision(planeDistance - radius);
        collision->contactPoint = projectedPoint;
        collision->localMaterialA = materialA;
        collision->localMaterialB = triangleMaterial;
        collision->sphereMergePrimary = 1;
        collision->extraNegated = triangleNormal;
        return 1;
    }

    return 0;
}

static void TransformMeshCollisionsToWorldForGmSurf(
        CGmCollisionBuffer &collisionBuffer,
        u32 firstNew,
        const GmIso4 &meshIso) {
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew; collisionIndex < count; collisionIndex++) {
        GmCollision *collision = &collisionBuffer.GetCollision(collisionIndex);
        const GmMat3 meshRotation = meshIso.RotationMatrix();
        collision->impulseNormal.Mult(meshRotation);
        collision->separation.Mult(meshRotation);
        collision->contactPoint.Mult(meshIso);
    }
}

static void TransformEllipsoidMeshCollisionsToWorldForGmSurf(
        CGmCollisionBuffer &collisionBuffer,
        u32 firstNew,
        const GmIso4 &unitContactToWorld,
        const GmIso4 &unitNormalToWorld) {
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew; collisionIndex < count; collisionIndex++) {
        GmCollision *collision = &collisionBuffer.GetCollision(collisionIndex);
        collision->contactPoint.Mult(unitContactToWorld);
        collision->impulseNormal.MultEllipsoidMeshWorldNormalForGmSurf(
                unitNormalToWorld.rotation);
        collision->impulseNormal =
            collision->impulseNormal.NormalizeForCollision(
                    PhysicsTolerance::SurfaceDirectionLengthSquared);
        collision->separation.Mult(unitContactToWorld.RotationMatrix());
    }
}

int GmCollision_Sphere_Mesh(
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
    GmBoxAligned sphereBox = GmBoxAligned::FromCenterHalfExtents(zero, sphereHalf);
    sphereBox.Mult(sphereToMesh);

    const GmVec3 sphereCenterMesh = sphereToMesh.TranslationForGmSurf();
    int hit = 0;

    for (u32 cellIndex = 0; cellIndex < mesh->OctreeCellCount();) {
        const GmMeshOctreeCell *cell = &mesh->OctreeCell(cellIndex);
        if (!sphereBox.TestInter(cell->Bounds())) {
            cellIndex += cell->SubtreeEntryCount();
            continue;
        }

        if (cell->ContainsTriangle()) {
            const GmSurfMeshTriangle *triangle =
                &mesh->Triangle(cell->TriangleIndex());
            const GmVec3 vertices[3] = {
                mesh->Vertex(triangle->vertexIndex[0]),
                mesh->Vertex(triangle->vertexIndex[1]),
                mesh->Vertex(triangle->vertexIndex[2]),
            };
            SSphereMeshCollide triangleCollide = {
                collisionBuffer,
                sphereCenterMesh,
                radius,
                sphere->LocalMaterial(),
                triangle->normal,
                triangle->material,
            };
            if (triangleCollide.CollideTriangle(vertices)) {
                hit = 1;
            }
        }

        cellIndex++;
    }

    if (hit) {
        TransformMeshCollisionsToWorldForGmSurf(
                *collisionBuffer, firstNew, *meshLocated->iso);
    }
    return hit;
}

int GmCollision_Ellipsoid_Mesh(
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
    GmBoxAligned ellipsoidBox = GmBoxAligned::FromCenterHalfExtents(zero, radii);
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
    for (u32 cellIndex = 0; cellIndex < mesh->OctreeCellCount();) {
        const GmMeshOctreeCell *cell = &mesh->OctreeCell(cellIndex);
        if (!ellipsoidBox.TestInter(cell->Bounds())) {
            cellIndex += cell->SubtreeEntryCount();
            continue;
        }

        if (cell->ContainsTriangle()) {
            const GmSurfMeshTriangle *triangle =
                &mesh->Triangle(cell->TriangleIndex());
            GmVec3 verticesUnit[3] = {
                meshToUnitSphereTransform.SetMultPointForGmSurf(mesh->Vertex(triangle->vertexIndex[0])),
                meshToUnitSphereTransform.SetMultPointForGmSurf(mesh->Vertex(triangle->vertexIndex[1])),
                meshToUnitSphereTransform.SetMultPointForGmSurf(mesh->Vertex(triangle->vertexIndex[2])),
            };

            const GmVec3 edge01 = verticesUnit[1].SubtractForCollision(verticesUnit[0]);
            const GmVec3 edge02 = verticesUnit[2].SubtractForCollision(verticesUnit[0]);
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
            if (normalLenSq > PhysicsTolerance::SurfaceDirectionLengthSquared) {
                const float normalLen = (CIsqrt(normalLenSq));
                const float invNormalLen = (1.0f / normalLen);
                unitTriangleNormal = GmVec3{
                    (normalX * invNormalLen),
                    (normalY * invNormalLen),
                    (invNormalLen * normalZ),
                };

                const u32 firstNew = collisionBuffer->GetCurrentCount();
                SSphereMeshCollide triangleCollide = {
                    collisionBuffer,
                    zero,
                    1.0f,
                    ellipsoid->LocalMaterial(),
                    unitTriangleNormal,
                    triangle->material,
                };
                if (triangleCollide.CollideTriangle(verticesUnit)) {
                    TransformEllipsoidMeshCollisionsToWorldForGmSurf(
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

int GmCollision_Box_Mesh(
        const LocatedGmSurf &boxRef,
        const LocatedGmSurf &meshLocatedRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *box = &boxRef;
    const LocatedGmSurf *meshLocated = &meshLocatedRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmSurfMesh *mesh = meshLocated->Mesh();
    const GmVec3 boxCenter = box->BoxCenter();
    const GmVec3 boxHalfExtents = box->BoxHalfExtents();

    GmIso4 boxToMesh;
    if (box->enabled == 0) {
        boxToMesh.SetIdentity();
    } else {
        boxToMesh = *box->iso;
    }
    if (meshLocated->enabled != 0) {
        boxToMesh.MultInverse(*meshLocated->iso);
    }

    GmBoxAligned boxQuery = GmBoxAligned::FromCenterHalfExtents(boxCenter, boxHalfExtents);
    boxQuery.Mult(boxToMesh);

    for (u32 cellIndex = 0; cellIndex < mesh->OctreeCellCount();) {
        const GmMeshOctreeCell *cell = &mesh->OctreeCell(cellIndex);
        if (!boxQuery.TestInter(cell->Bounds())) {
            cellIndex += cell->SubtreeEntryCount();
            continue;
        }

        if (cell->ContainsTriangle()) {
            const GmSurfMeshTriangle *triangle =
                &mesh->Triangle(cell->TriangleIndex());
            const GmVec3 meshVertices[3] = {
                mesh->Vertex(triangle->vertexIndex[0]),
                mesh->Vertex(triangle->vertexIndex[1]),
                mesh->Vertex(triangle->vertexIndex[2]),
            };
            const GmVec3 boxLocalVertices[3] = {
                boxToMesh.InverseRotateTranslatedPointForGmSurf(meshVertices[0]).SubtractForCollision(boxCenter),
                boxToMesh.InverseRotateTranslatedPointForGmSurf(meshVertices[1]).SubtractForCollision(boxCenter),
                boxToMesh.InverseRotateTranslatedPointForGmSurf(meshVertices[2]).SubtractForCollision(boxCenter),
            };

            const GmBoxAligned localBox =
                GmBoxAligned::FromCenterHalfExtents(GmVec3{0.0f, 0.0f, 0.0f},
                                                    boxHalfExtents);
            if (localBox.OverlapsTriangleInLocalSpaceForGmSurf(boxLocalVertices[0],
                                                               boxLocalVertices[1],
                                                               boxLocalVertices[2])) {
                GmCollision *collision = &collisionBuffer->AddCollision();
                collision->separation = GmVec3{0.0f, 0.0f, 0.0f};
                collision->contactPoint = meshVertices[0];
                collision->contactPoint.Mult(*meshLocated->iso);

                collision->impulseNormal = triangle->normal;
                collision->impulseNormal.Mult(
                        meshLocated->iso->RotationMatrix());

                collision->localMaterialA = box->LocalMaterial();
                collision->localMaterialB = triangle->material;
                return 1;
            }
        }

        cellIndex++;
    }

    return 0;
}

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

bool OptimizedCpuBoundsIntersect(const GmBoxAligned &query,
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

int GmCollision_Sphere_Mesh_OptimizedCpu(
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
        if (!OptimizedCpuBoundsIntersect(sphereBox, cell->Bounds())) {
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
            SSphereMeshCollide triangleCollide = {
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
        TransformMeshCollisionsToWorldForGmSurf(
                *collisionBuffer, firstNew, *meshLocated->iso);
    }
    return hit;
}
