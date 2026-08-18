// Primitive shape intersections and contact construction.

#include <cmath>

#include "engine/core/binary32_math.h"
#include "engine/physics/geometry/geometry_helpers.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"
struct SSpherePolygonCollide {
    const LocatedGmSurf *sphere;
    const LocatedGmSurf *polygon;
    const GmSurfPolygon *poly;
    CGmCollisionBuffer *collisionBuffer;
    GmVec3 sphereCenterLocal;

    int EmitFeatureCollision(GmVec3 featurePointLocal,
                             int rejectIfOutsideRadius);
};

int SSpherePolygonCollide::EmitFeatureCollision(GmVec3 featurePointLocal,
                                                int rejectIfOutsideRadius) {
    const float radius = sphere->SphereRadius();
    const GmVec3 centerToFeature = sphereCenterLocal.SubtractForCollision(featurePointLocal);
    const float distance = CIsqrt(centerToFeature.Dot(centerToFeature));
    if (rejectIfOutsideRadius && radius < distance) {
        return 0;
    }

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->impulseNormal = poly->normal;
    collision->contactPoint = featurePointLocal;

    const float scale = (distance - radius) * (1.0f / distance);
    const GmVec3 scaledDelta = centerToFeature.ScaleForCollision(scale);
    const float separationAlongNormal = poly->normal.Dot(scaledDelta);
    collision->separation = poly->normal.ScaleForCollision(separationAlongNormal);

    collision->FinalizePolygonCollisionForGmSurf(*polygon, 1);
    collision->FinishPolygonMaterialsForGmSurf(*sphere, *polygon);
    return 1;
}

int GmCollision_NotHandled(const LocatedGmSurf &a,
                           const LocatedGmSurf &b,
                           CGmCollisionBuffer &collisionBuffer) {
    (void)a;
    (void)b;
    (void)collisionBuffer;
    return 0;
}

int GmCollision_Sphere_Sphere(
        const LocatedGmSurf &aRef,
        const LocatedGmSurf &bRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *a = &aRef;
    const LocatedGmSurf *b = &bRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmVec3 centerA = a->iso->TranslationForGmSurf();
    const GmVec3 centerB = b->iso->TranslationForGmSurf();
    const GmVec3 deltaAB = centerB.SubtractForCollision(centerA);

    const float radiusA = a->SphereRadius();
    const float radiusB = b->SphereRadius();
    const float radiusSum = radiusA + radiusB;

    const float distanceSq = deltaAB.Dot(deltaAB);
    if (!(distanceSq < radiusSum * radiusSum)) {
        return 0;
    }

    GmCollision *collision = &collisionBuffer->AddCollision();
    const float distance = CIsqrt(distanceSq);

    if (!(PhysicsTolerance::CollisionDistance < distance)) {
        collision->impulseNormal = GmVec3{0.0f, -1.0f, 0.0f};
        collision->separation = GmVec3{0.0f, radiusB, 0.0f};
        collision->contactPoint = centerA;
    } else {
        const GmVec3 unitAB = deltaAB.ScaleForCollision(1.0f / distance);
        collision->impulseNormal = unitAB.ScaleForCollision(-1.0f);
        collision->separation = unitAB.ScaleForCollision(radiusSum - distance);
        collision->contactPoint = GmVec3{
            centerA.x + radiusA * unitAB.x,
            centerA.y + radiusA * unitAB.y,
            centerA.z + radiusA * unitAB.z,
        };
    }

    collision->localMaterialA = a->LocalMaterial();
    collision->localMaterialB = b->LocalMaterial();
    return 1;
}

int GmCollision_Sphere_Ellipsoid(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &ellipsoidRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *sphere = &sphereRef;
    const LocatedGmSurf *ellipsoid = &ellipsoidRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmVec3 sphereCenter = sphere->iso->TranslationForGmSurf();
    const GmVec3 ellipsoidCenter = ellipsoid->iso->TranslationForGmSurf();
    const GmVec3 deltaSphereToEllipsoid = ellipsoidCenter.SubtractForCollision(sphereCenter);

    const float sphereRadius = sphere->SphereRadius();
    const float ellipsoidBoundRadius = ellipsoid->EllipsoidBoundRadius();
    const float radiusSum = sphereRadius + ellipsoidBoundRadius;

    const float distanceSq = deltaSphereToEllipsoid.Dot(deltaSphereToEllipsoid);
    if (!(distanceSq < radiusSum * radiusSum)) {
        return 0;
    }

    GmCollision *collision = &collisionBuffer->AddCollision();
    const float distance = CIsqrt(distanceSq);

    if (!(PhysicsTolerance::CollisionDistance < distance)) {
        collision->impulseNormal = GmVec3{0.0f, -1.0f, 0.0f};
        collision->separation = GmVec3{0.0f, radiusSum, 0.0f};
        collision->contactPoint = sphereCenter;
    } else {
        const GmVec3 unitSphereToEllipsoid =
            deltaSphereToEllipsoid.ScaleForCollision(1.0f / distance);
        collision->impulseNormal = unitSphereToEllipsoid.ScaleForCollision(-1.0f);
        collision->separation = unitSphereToEllipsoid.ScaleForCollision(radiusSum - distance);
        collision->contactPoint = GmVec3{
            sphereCenter.x + sphereRadius * unitSphereToEllipsoid.x,
            sphereCenter.y + sphereRadius * unitSphereToEllipsoid.y,
            sphereCenter.z + sphereRadius * unitSphereToEllipsoid.z,
        };
    }

    collision->localMaterialA = sphere->LocalMaterial();
    collision->localMaterialB = ellipsoid->LocalMaterial();
    return 1;
}

int GmCollision_Sphere_Box(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &boxRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *sphere = &sphereRef;
    const LocatedGmSurf *box = &boxRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmVec3 sphereCenter = sphere->iso->TranslationForGmSurf();
    const GmVec3 boxWorldCenter = box->iso->TranslationForGmSurf();

    GmVec3 sphereLocal = sphereCenter.SubtractForCollision(boxWorldCenter);
    sphereLocal.MultTranspose(box->iso->RotationMatrix());

    const GmVec3 boxCenter = box->BoxCenter();
    const GmVec3 boxHalfExtents = box->BoxHalfExtents();
    const GmVec3 boxMin = boxCenter.SubtractForCollision(boxHalfExtents);
    const GmVec3 boxMax = {
        boxCenter.x + boxHalfExtents.x,
        boxCenter.y + boxHalfExtents.y,
        boxCenter.z + boxHalfExtents.z,
    };

    if (boxMin.x <= sphereLocal.x && sphereLocal.x <= boxMax.x &&
        boxMin.y <= sphereLocal.y && sphereLocal.y <= boxMax.y &&
        boxMin.z <= sphereLocal.z && sphereLocal.z <= boxMax.z) {
        return 0;
    }

    int axis;
    float faceCoord;
    if (sphereLocal.x < boxMin.x) {
        axis = 0;
        faceCoord = boxMin.x;
    } else if (sphereLocal.x > boxMax.x) {
        axis = 0;
        faceCoord = boxMax.x;
    } else if (sphereLocal.y < boxMin.y) {
        axis = 1;
        faceCoord = boxMin.y;
    } else if (sphereLocal.y > boxMax.y) {
        axis = 1;
        faceCoord = boxMax.y;
    } else if (sphereLocal.z < boxMin.z) {
        axis = 2;
        faceCoord = boxMin.z;
    } else if (sphereLocal.z > boxMax.z) {
        axis = 2;
        faceCoord = boxMax.z;
    } else {
        return 0;
    }

    const float sphereRadius = sphere->SphereRadius();
    const float sphereRadiusSq = sphereRadius * sphereRadius;
    const float axisDelta =
        axis == 0 ? sphereLocal.x - faceCoord :
        axis == 1 ? sphereLocal.y - faceCoord :
                    sphereLocal.z - faceCoord;
    if (sphereRadiusSq < axisDelta * axisDelta) {
        return 0;
    }

    GmVec3 closestLocal = {
        axis == 0 ? faceCoord : GmBoxAligned::ClampScalarForGmSurf(sphereLocal.x, boxMin.x, boxMax.x),
        axis == 1 ? faceCoord : GmBoxAligned::ClampScalarForGmSurf(sphereLocal.y, boxMin.y, boxMax.y),
        axis == 2 ? faceCoord : GmBoxAligned::ClampScalarForGmSurf(sphereLocal.z, boxMin.z, boxMax.z),
    };
    const GmVec3 localDelta = sphereLocal.SubtractForCollision(closestLocal);
    if (localDelta.Dot(localDelta) > sphereRadiusSq) {
        return 0;
    }

    GmVec3 contactPoint = closestLocal;
    contactPoint.Mult(*box->iso);

    const GmVec3 contactToSphere = sphereCenter.SubtractForCollision(contactPoint);
    const float distance = CIsqrt(contactToSphere.Dot(contactToSphere));
    const GmVec3 normal = contactToSphere.ScaleForCollision(1.0f / distance);
    const float penetrationScale = distance - CIsqrt(sphereRadiusSq);

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->contactPoint = contactPoint;
    collision->impulseNormal = normal;
    collision->separation = normal.ScaleForCollision(penetrationScale);
    collision->localMaterialA = sphere->LocalMaterial();
    collision->localMaterialB = box->LocalMaterial();
    return 1;
}

int GmCollision_Box_Box(
        const LocatedGmSurf &aRef,
        const LocatedGmSurf &bRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *a = &aRef;
    const LocatedGmSurf *b = &bRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    GmIso4 bInA = *b->iso;
    bInA.MultInverse(*a->iso);

    GmMat3 absBInA{};
    absBInA.basisX = {
        std::fabs(bInA.rotation.basisX.x),
        std::fabs(bInA.rotation.basisX.y),
        std::fabs(bInA.rotation.basisX.z),
    };
    absBInA.basisY = {
        std::fabs(bInA.rotation.basisY.x),
        std::fabs(bInA.rotation.basisY.y),
        std::fabs(bInA.rotation.basisY.z),
    };
    absBInA.basisZ = {
        std::fabs(bInA.rotation.basisZ.x),
        std::fabs(bInA.rotation.basisZ.y),
        std::fabs(bInA.rotation.basisZ.z),
    };

    const GmBoxAligned boxA = GmBoxAligned::FromCenterHalfExtents(
            a->BoxCenter(), a->BoxHalfExtents());
    const GmBoxAligned boxB = GmBoxAligned::FromCenterHalfExtents(
            b->BoxCenter(), b->BoxHalfExtents());
    if (!boxA.OverlapsOrientedRelativeTo(boxB, bInA, absBInA)) {
        return 0;
    }

    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->contactPoint = a->iso->TranslationForGmSurf();
    collision->separation = GmVec3{0.0f, 0.0f, 0.0f};
    collision->impulseNormal = GmVec3{1.0f, 0.0f, 0.0f};
    collision->localMaterialA = a->LocalMaterial();
    collision->localMaterialB = b->LocalMaterial();
    return 1;
}

int GmCollision_Sphere_Polygon(
        const LocatedGmSurf &sphereRef,
        const LocatedGmSurf &polygonRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *sphere = &sphereRef;
    const LocatedGmSurf *polygon = &polygonRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmSurfPolygon *poly = polygon->Polygon();
    const float radius = sphere->SphereRadius();

    GmVec3 sphereCenter = {0.0f, 0.0f, 0.0f};
    if (sphere->enabled != 0) {
        sphereCenter = sphere->iso->TranslationForGmSurf();
    }
    if (polygon->enabled != 0) {
        sphereCenter.MultInverseIso4LocalForGmSurf(*polygon->iso);
    }
    SSpherePolygonCollide spherePolygon = {
        sphere,
        polygon,
        poly,
        collisionBuffer,
        sphereCenter,
    };

    const GmVec3 sphereFromVertex0 = sphereCenter.SubtractForCollision(poly->vertices[0]);
    const float planeDistance = sphereFromVertex0.Dot(poly->normal);
    if (radius < planeDistance) {
        return 0;
    }
    if (planeDistance < 0.0f && !poly->backSide) {
        return 0;
    }

    const float edgeReach = CIsqrt(radius * radius - planeDistance * planeDistance);
    const GmVec3 projectedPoint =
        sphereCenter.AddForCollision(poly->normal.ScaleForCollision(-planeDistance));

    for (u32 vertexIndex = 0; vertexIndex < poly->vertexCount; vertexIndex++) {
        const u32 nextVertexIndex =
            vertexIndex == (u32)poly->vertexCount - 1u ? 0u : vertexIndex + 1u;
        const GmVec3 vi = poly->vertices[vertexIndex];
        const GmVec3 vj = poly->vertices[nextVertexIndex];
        const GmVec3 edgeDir =
            vj.SubtractForCollision(vi).NormalizeForCollision(
                    PhysicsTolerance::SurfaceDirectionLengthSquared);
        const GmVec3 edgeNormal = edgeDir.Cross(poly->normal);
        const float edgeDistance = projectedPoint.SubtractForCollision(vi).Dot(edgeNormal);

        if (edgeReach < edgeDistance) {
            return 0;
        }
        if (!(planeDistance < 0.0f)) {
            if (edgeDistance > 0.0f) {
                const float alongFromStart = projectedPoint.SubtractForCollision(vi).Dot(edgeDir);
                if (alongFromStart < 0.0f) {
                    return spherePolygon.EmitFeatureCollision(vi, 1);
                }

                const float alongFromEnd = projectedPoint.SubtractForCollision(vj).Dot(edgeDir);
                if (!(0.0f < alongFromEnd)) {
                    const GmVec3 featurePoint =
                        projectedPoint.AddForCollision(edgeNormal.ScaleForCollision(-edgeDistance));
                    return spherePolygon.EmitFeatureCollision(featurePoint, 0);
                }

                return spherePolygon.EmitFeatureCollision(vj, 1);
            }
        } else if (edgeDistance > 0.0f) {
            return 0;
        }
    }

    GmCollision *collision = &collisionBuffer->AddCollision();
    if (!(0.0f < planeDistance)) {
        collision->impulseNormal = poly->normal.ScaleForCollision(-1.0f);
        collision->separation = poly->normal.ScaleForCollision(planeDistance - radius);
        collision->FinalizePolygonCollisionForGmSurf(*polygon, 0);
        collision->contactPoint = projectedPoint;
    } else {
        collision->impulseNormal = poly->normal;
        collision->separation = poly->normal.ScaleForCollision(planeDistance - radius);
        collision->contactPoint = projectedPoint;
        collision->FinalizePolygonCollisionForGmSurf(*polygon, 1);
    }

    collision->FinishPolygonMaterialsForGmSurf(*sphere, *polygon);
    return 1;
}

int GmCollision_Ellipsoid_Polygon(
        const LocatedGmSurf &ellipsoidRef,
        const LocatedGmSurf &polygonRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *ellipsoid = &ellipsoidRef;
    const LocatedGmSurf *polygon = &polygonRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmSurfPolygon *poly = polygon->Polygon();
    const GmVec3 radii = ellipsoid->EllipsoidRadii();

    GmVec3 ellipsoidCenter = {0.0f, 0.0f, 0.0f};
    if (ellipsoid->enabled != 0) {
        ellipsoidCenter = ellipsoid->iso->TranslationForGmSurf();
    }
    if (polygon->enabled != 0) {
        ellipsoidCenter.MultInverseIso4LocalForGmSurf(*polygon->iso);
    }

    const float planeDistance = ellipsoidCenter.SubtractForCollision(poly->vertices[0]).Dot(poly->normal);
    const float boundRadius = ellipsoid->EllipsoidBoundRadius();
    if (boundRadius < planeDistance) {
        return 0;
    }
    if (planeDistance < 0.0f && !poly->backSide) {
        return 0;
    }

    GmIso4 baseIso;
    if (polygon->enabled != 0) {
        baseIso = *polygon->iso;
    } else {
        baseIso.SetIdentity();
    }
    if (ellipsoid->enabled != 0) {
        baseIso.MultInverse(*ellipsoid->iso);
    }

    const GmVec3 invRadii = {1.0f / radii.x, 1.0f / radii.y, 1.0f / radii.z};
    GmIso4 unitFromPolygon = baseIso;
    unitFromPolygon.ScaleRowsForGmSurf(invRadii);

    GmSurfPolygon tempPolygon(poly->vertexCount);
    tempPolygon.material = poly->material;
    tempPolygon.backSide = poly->backSide;
    for (u32 vertexIndex = 0; vertexIndex < 4u; vertexIndex++) {
        tempPolygon.vertices[vertexIndex] = {0.0f, 0.0f, 0.0f};
    }
    tempPolygon.normal = {0.0f, 0.0f, 0.0f};
    for (u32 vertexIndex = 0; vertexIndex < tempPolygon.vertexCount; vertexIndex++) {
        tempPolygon.vertices[vertexIndex] = poly->vertices[vertexIndex];
        tempPolygon.vertices[vertexIndex].Mult(unitFromPolygon);
    }
    tempPolygon.ComputeNormalFromVertices();

    GmSurfSphere tempSphere;
    tempSphere.material = ellipsoid->LocalMaterial();
    tempSphere.radius = 1.0f;

    GmIso4 identityIso;
    identityIso.SetIdentity();
    LocatedGmSurf tempSphereLocated = {&tempSphere, &identityIso, 0};
    LocatedGmSurf tempPolygonLocated = {&tempPolygon, &identityIso, 0};

    const u32 firstNew = collisionBuffer->GetCurrentCount();
    if (!GmCollision_Sphere_Polygon(
                tempSphereLocated,
                tempPolygonLocated,
                *collisionBuffer)) {
        return 0;
    }

    GmVec3 zero = {0.0f, 0.0f, 0.0f};
    GmIso4 polygonFromUnit;
    polygonFromUnit.SetNUScaleTrans(radii, zero);
    polygonFromUnit.MultInverse(baseIso);

    GmIso4 normalFromUnit;
    normalFromUnit.SetNUScaleTrans(invRadii, zero);
    normalFromUnit.MultInverse(baseIso);

    const u32 count = collisionBuffer->GetCurrentCount();
    for (u32 collisionIndex = firstNew; collisionIndex < count; collisionIndex++) {
        GmCollision *collision = &collisionBuffer->GetCollision(collisionIndex);
        collision->contactPoint.Mult(polygonFromUnit);
        collision->separation.Mult(polygonFromUnit.RotationMatrix());
        collision->impulseNormal.TransformAndNormalizeDirectionForGmSurf(
                normalFromUnit,
                PhysicsTolerance::SurfaceDirectionLengthSquared);
        collision->extraNegated.TransformAndNormalizeDirectionForGmSurf(
                normalFromUnit,
                PhysicsTolerance::SurfaceDirectionLengthSquared);
    }

    return 1;
}
