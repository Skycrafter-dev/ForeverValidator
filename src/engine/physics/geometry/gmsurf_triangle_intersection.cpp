// Triangle/triangle intersection and paired mesh-tree traversal.

#include <cmath>

#include "engine/core/binary32_math.h"
#include "engine/physics/geometry/geometry_helpers.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/gmsurf_mesh_pair_query.h"
#include "engine/physics/geometry/physics_tolerances.h"
namespace {
float surfCollisionAbsFloat(float value) { return std::fabs(value); }
} // namespace

struct TriTriInterval {
    float base;
    float delta0;
    float delta1;
    float scale0;
    float scale1;
    int coplanar;
    static TriTriInterval BuildNoDiv(float axisCoord0,
                                     float axisCoord1,
                                     float axisCoord2,
                                     float signedDist0,
                                     float signedDist1,
                                     float signedDist2);
    int OverlapsScaled(const TriTriInterval &rhs) const;
};

TriTriInterval TriTriInterval::BuildNoDiv(float axisCoord0,
                                          float axisCoord1,
                                          float axisCoord2,
                                          float signedDist0,
                                          float signedDist1,
                                          float signedDist2) {
    TriTriInterval out = {};

    if (!(0.0f < signedDist0 * signedDist1)) {
        if (!(0.0f < signedDist0 * signedDist2)) {
            if (0.0f < signedDist1 * signedDist2 || signedDist0 != 0.0f) {
                out.delta0 = (axisCoord1 - axisCoord0) * signedDist0;
                out.delta1 = (axisCoord2 - axisCoord0) * signedDist0;
                out.scale0 = signedDist0 - signedDist1;
                out.scale1 = signedDist0 - signedDist2;
                out.base = axisCoord0;
                return out;
            }

            if (signedDist1 != 0.0f) {
                out.delta0 = (axisCoord0 - axisCoord1) * signedDist1;
                out.delta1 = (axisCoord2 - axisCoord1) * signedDist1;
                out.scale0 = signedDist1 - signedDist0;
                out.scale1 = signedDist1 - signedDist2;
                out.base = axisCoord1;
                return out;
            }

            if (signedDist2 != 0.0f) {
                out.delta0 = (axisCoord0 - axisCoord2) * signedDist2;
                out.delta1 = (axisCoord1 - axisCoord2) * signedDist2;
                out.scale0 = signedDist2 - signedDist0;
                out.scale1 = signedDist2 - signedDist1;
                out.base = axisCoord2;
                return out;
            }

            out.coplanar = 1;
            return out;
        }

        out.delta0 = signedDist1 * (axisCoord0 - axisCoord1);
        out.delta1 = signedDist1 * (axisCoord2 - axisCoord1);
        out.scale0 = signedDist1 - signedDist0;
        out.scale1 = signedDist1 - signedDist2;
        out.base = axisCoord1;
        return out;
    }

    out.delta0 = signedDist2 * (axisCoord0 - axisCoord2);
    out.delta1 = signedDist2 * (axisCoord1 - axisCoord2);
    out.scale0 = signedDist2 - signedDist0;
    out.scale1 = signedDist2 - signedDist1;
    out.base = axisCoord2;
    return out;
}

int TriTriInterval::OverlapsScaled(const TriTriInterval &rhs) const {
    const float aScale = scale0 * scale1;
    const float bScale = rhs.scale0 * rhs.scale1;
    const float commonScale = aScale * bScale;

    float intervalAStart = commonScale * base + bScale * delta0 * scale1;
    float intervalAEnd = commonScale * base + bScale * delta1 * scale0;
    float intervalBStart = commonScale * rhs.base + aScale * rhs.delta0 * rhs.scale1;
    float intervalBEnd = commonScale * rhs.base + aScale * rhs.delta1 * rhs.scale0;

    if (intervalAEnd < intervalAStart) {
        const float swapValue = intervalAStart;
        intervalAStart = intervalAEnd;
        intervalAEnd = swapValue;
    }
    if (intervalBEnd < intervalBStart) {
        const float swapValue = intervalBStart;
        intervalBStart = intervalBEnd;
        intervalBEnd = swapValue;
    }

    return !(intervalAEnd < intervalBStart) && !(intervalBEnd < intervalAStart);
}

struct CoplanarTriTriProjection {
    GmAxis axisU;
    GmAxis axisV;

    static CoplanarTriTriProjection FromNormal(const GmVec3 &normal);
    float ProjectU(const GmVec3 &v) const;
    float ProjectV(const GmVec3 &v) const;
    int EdgeEdgeIntersectsInclusive(const GmVec3 &edgeAStart,
                                    const GmVec3 &edgeAEnd,
                                    const GmVec3 &edgeBStart,
                                    const GmVec3 &edgeBEnd) const;
    float EdgeSignedArea(const GmVec3 &edgeStart,
                         const GmVec3 &edgeEnd,
                         const GmVec3 &point) const;
    int PointStrictlyInsideTri(const GmVec3 &point,
                               const GmVec3 &tri0,
                               const GmVec3 &tri1,
                               const GmVec3 &tri2) const;
};

CoplanarTriTriProjection CoplanarTriTriProjection::FromNormal(const GmVec3 &normal) {
    const float absX = surfCollisionAbsFloat(normal.x);
    const float absY = surfCollisionAbsFloat(normal.y);
    const float absZ = surfCollisionAbsFloat(normal.z);

    if (!(absY < absX)) {
        if (!(absY < absZ)) {
            return {GmAxis::X, GmAxis::Z};
        }
    } else if (absZ < absX) {
        return {GmAxis::Y, GmAxis::Z};
    }

    return {GmAxis::X, GmAxis::Y};
}

float CoplanarTriTriProjection::ProjectU(const GmVec3 &v) const {
    return v.Component(axisU);
}

float CoplanarTriTriProjection::ProjectV(const GmVec3 &v) const {
    return v.Component(axisV);
}

int CoplanarTriTriProjection::EdgeEdgeIntersectsInclusive(
        const GmVec3 &edgeAStart,
        const GmVec3 &edgeAEnd,
        const GmVec3 &edgeBStart,
        const GmVec3 &edgeBEnd) const {
    const float ax = ProjectU(edgeAEnd) - ProjectU(edgeAStart);
    const float ay = ProjectV(edgeAEnd) - ProjectV(edgeAStart);
    const float bx = ProjectU(edgeBStart) - ProjectU(edgeBEnd);
    const float by = ProjectV(edgeBStart) - ProjectV(edgeBEnd);
    const float cx = ProjectU(edgeAStart) - ProjectU(edgeBStart);
    const float cy = ProjectV(edgeAStart) - ProjectV(edgeBStart);

    const float determinant = ay * bx - ax * by;
    const float firstParamNumerator = by * cx - bx * cy;
    if (determinant > 0.0f) {
        if (firstParamNumerator >= 0.0f &&
            firstParamNumerator <= determinant) {
            const float secondParamNumerator = ax * cy - ay * cx;
            return secondParamNumerator >= 0.0f &&
                   secondParamNumerator <= determinant;
        }
        return 0;
    } else if (determinant < 0.0f) {
        if (firstParamNumerator <= 0.0f &&
            determinant <= firstParamNumerator) {
            const float secondParamNumerator = ax * cy - ay * cx;
            return secondParamNumerator <= 0.0f &&
                   determinant <= secondParamNumerator;
        }
        return 0;
    } else {
        return 0;
    }
}

float CoplanarTriTriProjection::EdgeSignedArea(const GmVec3 &edgeStart,
                                               const GmVec3 &edgeEnd,
                                               const GmVec3 &point) const {
    const float edgeU = ProjectU(edgeEnd) - ProjectU(edgeStart);
    const float edgeV = ProjectV(edgeEnd) - ProjectV(edgeStart);
    const float pointU = ProjectU(point) - ProjectU(edgeStart);
    const float pointV = ProjectV(point) - ProjectV(edgeStart);
    return edgeV * pointU - edgeU * pointV;
}

int CoplanarTriTriProjection::PointStrictlyInsideTri(const GmVec3 &point,
                                                     const GmVec3 &tri0,
                                                     const GmVec3 &tri1,
                                                     const GmVec3 &tri2) const {
    const float side0 = EdgeSignedArea(tri0, tri1, point);
    const float side1 = EdgeSignedArea(tri1, tri2, point);
    const float side2 = EdgeSignedArea(tri2, tri0, point);
    return side0 * side1 > 0.0f && side0 * side2 > 0.0f;
}

int GmCoplanarTriTri(const GmVec3 &normal,
                     const GmVec3 &triA0,
                     const GmVec3 &triA1,
                     const GmVec3 &triA2,
                     const GmVec3 &triB0,
                     const GmVec3 &triB1,
                     const GmVec3 &triB2) {
    const CoplanarTriTriProjection projection =
        CoplanarTriTriProjection::FromNormal(normal);

    const GmVec3 triA[3] = {triA0, triA1, triA2};
    const GmVec3 triB[3] = {triB0, triB1, triB2};
    for (u32 edgeA = 0; edgeA < 3; edgeA++) {
        const u32 nextA = edgeA == 2u ? 0u : edgeA + 1u;
        for (u32 edgeB = 0; edgeB < 3; edgeB++) {
            const u32 nextB = edgeB == 2u ? 0u : edgeB + 1u;
            if (projection.EdgeEdgeIntersectsInclusive(triA[edgeA],
                                                       triA[nextA],
                                                       triB[edgeB],
                                                       triB[nextB])) {
                return 1;
            }
        }
    }

    if (projection.PointStrictlyInsideTri(triA[0], triB[0], triB[1], triB[2])) {
        return 1;
    }
    if (projection.PointStrictlyInsideTri(triB[0], triA[0], triA[1], triA[2])) {
        return 1;
    }
    return 0;
}

void SMeshMeshCollide::EmitCollision(const GmSurfMeshTriangle &triangleA,
                                     const GmSurfMeshTriangle &triangleB) {
    GmCollision *collision = &collisionBuffer->AddCollision();
    collision->separation = GmVec3{0.0f, 0.0f, 0.0f};

    collision->contactPoint = meshA->Vertex(triangleA.vertexIndex[0]);
    collision->contactPoint.Mult(*isoA);

    collision->impulseNormal = triangleB.normal;
    collision->impulseNormal.Mult(bToA.RotationMatrix());
    collision->impulseNormal.Mult(isoA->RotationMatrix());

    collision->localMaterialA = triangleA.material;
    collision->localMaterialB = triangleB.material;
}

int SMeshMeshCollide::TriTri(const GmMeshOctreeCell &cellA,
                             const GmMeshOctreeCell &cellB) {
    const GmSurfMeshTriangle *triangleA =
        &meshA->Triangle(cellA.TriangleIndex());
    const GmSurfMeshTriangle *triangleB =
        &meshB->Triangle(cellB.TriangleIndex());

    const GmVec3 triAVerticesInA[3] = {
        meshA->Vertex(triangleA->vertexIndex[0]),
        meshA->Vertex(triangleA->vertexIndex[1]),
        meshA->Vertex(triangleA->vertexIndex[2]),
    };
    const GmVec3 triBVerticesInA[3] = {
        bToA.TransformPointForGmSurf(meshB->Vertex(triangleB->vertexIndex[0])),
        bToA.TransformPointForGmSurf(meshB->Vertex(triangleB->vertexIndex[1])),
        bToA.TransformPointForGmSurf(meshB->Vertex(triangleB->vertexIndex[2])),
    };

    const GmVec3 normalA = triAVerticesInA[1]
            .SubtractForCollision(triAVerticesInA[0])
            .Cross(triAVerticesInA[2].SubtractForCollision(triAVerticesInA[0]));
    const float planeA = -normalA.Dot(triAVerticesInA[0]);
    const float triBDistancesToPlaneA[3] = {
        normalA.Dot(triBVerticesInA[0]) + planeA,
        normalA.Dot(triBVerticesInA[1]) + planeA,
        normalA.Dot(triBVerticesInA[2]) + planeA,
    };
    if (0.0f < triBDistancesToPlaneA[0] * triBDistancesToPlaneA[1] &&
        0.0f < triBDistancesToPlaneA[0] * triBDistancesToPlaneA[2]) {
        return 0;
    }

    const GmVec3 normalB = triBVerticesInA[1]
            .SubtractForCollision(triBVerticesInA[0])
            .Cross(triBVerticesInA[2].SubtractForCollision(triBVerticesInA[0]));
    const float planeB = -normalB.Dot(triBVerticesInA[0]);
    const float triADistancesToPlaneB[3] = {
        normalB.Dot(triAVerticesInA[0]) + planeB,
        normalB.Dot(triAVerticesInA[1]) + planeB,
        normalB.Dot(triAVerticesInA[2]) + planeB,
    };
    if (0.0f < triADistancesToPlaneB[0] * triADistancesToPlaneB[1] &&
        0.0f < triADistancesToPlaneB[0] * triADistancesToPlaneB[2]) {
        return 0;
    }

    const GmVec3 lineDir = normalA.Cross(normalB);
    GmAxis axis = GmAxis::X;
    float maxAbs = surfCollisionAbsFloat(lineDir.x);
    const float absY = surfCollisionAbsFloat(lineDir.y);
    if (maxAbs < absY) {
        axis = GmAxis::Y;
        maxAbs = absY;
    }
    if (maxAbs < surfCollisionAbsFloat(lineDir.z)) {
        axis = GmAxis::Z;
    }

    const TriTriInterval intervalA =
        TriTriInterval::BuildNoDiv(triAVerticesInA[0].Component(axis),
                                     triAVerticesInA[1].Component(axis),
                                     triAVerticesInA[2].Component(axis),
                                     triADistancesToPlaneB[0],
                                     triADistancesToPlaneB[1],
                                     triADistancesToPlaneB[2]);
    if (intervalA.coplanar) {
        if (!GmCoplanarTriTri(normalA,
                              triAVerticesInA[0],
                              triAVerticesInA[1],
                              triAVerticesInA[2],
                              triBVerticesInA[0],
                              triBVerticesInA[1],
                              triBVerticesInA[2])) {
            return 0;
        }
        EmitCollision(*triangleA, *triangleB);
        return 1;
    }

    const TriTriInterval intervalB =
        TriTriInterval::BuildNoDiv(triBVerticesInA[0].Component(axis),
                                     triBVerticesInA[1].Component(axis),
                                     triBVerticesInA[2].Component(axis),
                                     triBDistancesToPlaneA[0],
                                     triBDistancesToPlaneA[1],
                                     triBDistancesToPlaneA[2]);
    if (intervalB.coplanar) {
        if (!GmCoplanarTriTri(normalA,
                              triAVerticesInA[0],
                              triAVerticesInA[1],
                              triAVerticesInA[2],
                              triBVerticesInA[0],
                              triBVerticesInA[1],
                              triBVerticesInA[2])) {
            return 0;
        }
        EmitCollision(*triangleA, *triangleB);
        return 1;
    }

    if (!intervalA.OverlapsScaled(intervalB)) {
        return 0;
    }

    EmitCollision(*triangleA, *triangleB);
    return 1;
}

void SMeshMeshCollide::TreeTree(
        unsigned long cellIndexA,
        unsigned long cellIndexB) {
    const GmMeshOctreeCell *cellA = &meshA->OctreeCell(cellIndexA);
    const GmMeshOctreeCell *cellB = &meshB->OctreeCell(cellIndexB);

    if (!cellA->Bounds().OverlapsOrientedRelativeTo(
                cellB->Bounds(), bToA, absBToARotation)) {
        return;
    }

    const int leafA = cellA->ContainsTriangle();
    const int leafB = cellB->ContainsTriangle();
    if (leafA && leafB) {
        if (TriTri(*cellA, *cellB)) {
            hit = 1;
            return;
        }
    }

    if (!leafA &&
        (leafB || cellA->SubtreeEntryCount() > cellB->SubtreeEntryCount())) {
        const u32 end = cellIndexA + cellA->SubtreeEntryCount();
        for (u32 child = cellIndexA + 1u; child < end;) {
            TreeTree(child, cellIndexB);
            if (hit) {
                return;
            }
            child += meshA->OctreeCell(child).SubtreeEntryCount();
        }
        return;
    }

    if (!leafB) {
        const u32 end = cellIndexB + cellB->SubtreeEntryCount();
        for (u32 child = cellIndexB + 1u; child < end;) {
            TreeTree(cellIndexA, child);
            if (hit) {
                return;
            }
            child += meshB->OctreeCell(child).SubtreeEntryCount();
        }
    }
}

int GmCollision_Mesh_Mesh(
        const LocatedGmSurf &aRef,
        const LocatedGmSurf &bRef,
        CGmCollisionBuffer &collisionBufferRef) {
    const LocatedGmSurf *a = &aRef;
    const LocatedGmSurf *b = &bRef;
    CGmCollisionBuffer *collisionBuffer = &collisionBufferRef;
    const GmSurfMesh *meshA = a->Mesh();
    const GmSurfMesh *meshB = b->Mesh();

    SMeshMeshCollide ctx = {};
    ctx.meshA = meshA;
    ctx.meshB = meshB;
    ctx.collisionBuffer = collisionBuffer;
    ctx.isoA = a->iso;

    ctx.bToA = *b->iso;
    ctx.bToA.MultInverse(*a->iso);
    ctx.absBToARotation.basisX = {
        surfCollisionAbsFloat(ctx.bToA.rotation.basisX.x),
        surfCollisionAbsFloat(ctx.bToA.rotation.basisX.y),
        surfCollisionAbsFloat(ctx.bToA.rotation.basisX.z),
    };
    ctx.absBToARotation.basisY = {
        surfCollisionAbsFloat(ctx.bToA.rotation.basisY.x),
        surfCollisionAbsFloat(ctx.bToA.rotation.basisY.y),
        surfCollisionAbsFloat(ctx.bToA.rotation.basisY.z),
    };
    ctx.absBToARotation.basisZ = {
        surfCollisionAbsFloat(ctx.bToA.rotation.basisZ.x),
        surfCollisionAbsFloat(ctx.bToA.rotation.basisZ.y),
        surfCollisionAbsFloat(ctx.bToA.rotation.basisZ.z),
    };

    ctx.TreeTree(0, 0);
    return (int)ctx.hit;
}
