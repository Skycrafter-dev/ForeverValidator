// OptimizedCpu sphere-mesh query using immutable static-triangle data.

#include "simulation/backends/optimized_cpu/optimized_cpu_static_triangle_mesh_query.h"

#include <cmath>
#include <typeinfo>
#include <vector>

#include "engine/core/binary32_math.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/physics_tolerances.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_inverse_mesh_queries.h"

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

struct SStaticTriangleSphereMeshCollideOptimizedCpu {
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
    int CollideTriangle(
            const OptimizedCpuStaticMeshTriangleData &triangle);
};

int SStaticTriangleSphereMeshCollideOptimizedCpu::EmitFeatureCollision(
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

int SStaticTriangleSphereMeshCollideOptimizedCpu::EmitEndpointBCollision(
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

int SStaticTriangleSphereMeshCollideOptimizedCpu::CollideTriangle(
        const OptimizedCpuStaticMeshTriangleData &triangle) {
    const float planeDistance =
        sphereCenterMesh.SubtractForCollision(triangle.vertices[0])
                .Dot(triangleNormal);
    if (radius < planeDistance || planeDistance < 0.0f) {
        return 0;
    }

    const float edgeReach = CIsqrt(
            radius * radius - planeDistance * planeDistance);
    const GmVec3 projectedPoint =
        sphereCenterMesh.AddForCollision(
                triangleNormal.ScaleForCollision(-planeDistance));

    for (u32 edgeIndex = 0u; edgeIndex < 3u; ++edgeIndex) {
        const u32 nextIndex = edgeIndex == 2u ? 0u : edgeIndex + 1u;
        const GmVec3 edgeStart = triangle.vertices[edgeIndex];
        const GmVec3 edgeEnd = triangle.vertices[nextIndex];
        const GmVec3 edgeDir = triangle.edgeDirections[edgeIndex];
        const GmVec3 edgeNormal = triangle.edgeNormals[edgeIndex];
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

bool StaticTriangleOptimizedCpuBoundsIntersect(
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

void TransformStaticTriangleMeshCollisionsToWorldOptimizedCpu(
        CGmCollisionBuffer &collisionBuffer,
        u32 firstNew,
        const GmIso4 &meshIso) {
    const u32 count = collisionBuffer.GetCurrentCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         ++collisionIndex) {
        GmCollision *collision =
                &collisionBuffer.GetCollision(collisionIndex);
        const GmMat3 meshRotation = meshIso.RotationMatrix();
        collision->impulseNormal.Mult(meshRotation);
        collision->separation.Mult(meshRotation);
        collision->contactPoint.Mult(meshIso);
    }
}

int FinishStaticTriangleSurfaceMaterials(
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

int RunStaticTriangleSphereMeshQuery(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &meshLocatedRef,
        const GmIso4 &meshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar &triangles,
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
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshStaticInverseOptimizedCpuAccess::OctreeCells(*mesh);
    const u32 cellCount = static_cast<u32>(cells.size());

    for (u32 cellIndex = 0u; cellIndex < cellCount;) {
        const GmMeshOctreeCell *cell = &cells[cellIndex];
        if (!StaticTriangleOptimizedCpuBoundsIntersect(
                    sphereBox, cell->Bounds())) {
            cellIndex += cell->SubtreeEntryCount();
            continue;
        }

        if (cell->ContainsTriangle()) {
            const OptimizedCpuStaticMeshTriangleData &triangle =
                    triangles.TriangleAt(cell->TriangleIndex());
            SStaticTriangleSphereMeshCollideOptimizedCpu triangleCollide = {
                collisionBuffer,
                sphereCenterMesh,
                radius,
                sphere->LocalMaterial(),
                triangle.normal,
                triangle.material,
            };
            if (triangleCollide.CollideTriangle(triangle)) {
                hit = 1;
            }
        }

        ++cellIndex;
    }

    if (hit) {
        TransformStaticTriangleMeshCollisionsToWorldOptimizedCpu(
                *collisionBuffer, firstNew, *meshLocated->iso);
    }
    return hit;
}

}  // namespace

int GmCollision_Sphere_Mesh_OptimizedCpuWithStaticTriangleSidecar(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &meshLocatedRef,
        const GmIso4 &meshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar &triangles,
        CGmCollisionBuffer &collisionBufferRef) {
    const GmSurfMesh *mesh = meshLocatedRef.Mesh();
    if (!triangles.IsFor(*mesh)) {
        return GmCollision_Sphere_Mesh_OptimizedCpuWithMeshInverse(
                sphereRef, meshLocatedRef, meshInverse, collisionBufferRef);
    }
    return RunStaticTriangleSphereMeshQuery(
            sphereRef,
            meshLocatedRef,
            meshInverse,
            triangles,
            collisionBufferRef);
}

int ComputeCollisionOptimizedCpuWithStaticMeshTriangleSidecar(
        const SPlugSurfaceLocatedPair &pair,
        const GmIso4 &staticMeshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar *triangles,
        CGmCollisionBuffer &collisionBuffer) {
    if (triangles == nullptr) {
        return ComputeCollisionOptimizedCpuWithStaticMeshInverse(
                pair, staticMeshInverse, collisionBuffer);
    }

    const GmSurf *surfaceAGeometry = pair.FirstSurface().Geometry();
    const GmSurf *surfaceBGeometry = pair.SecondSurface().Geometry();
    if (surfaceAGeometry == nullptr || surfaceBGeometry == nullptr ||
        typeid(*surfaceAGeometry) != typeid(GmSurfSphere) ||
        typeid(*surfaceBGeometry) != typeid(GmSurfMesh)) {
        return ComputeCollisionOptimizedCpuWithStaticMeshInverse(
                pair, staticMeshInverse, collisionBuffer);
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
    const int collided =
            RunStaticTriangleSphereMeshQuery(
                    surfaceA,
                    surfaceB,
                    staticMeshInverse,
                    *triangles,
                    collisionBuffer);
    return FinishStaticTriangleSurfaceMaterials(
            pair, firstNew, collided, collisionBuffer);
}
