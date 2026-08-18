// Deterministic geometry primitives shared by shape intersections.

#include <cmath>

#include "engine/core/binary32_math.h"
#include "engine/physics/geometry/geometry_helpers.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/physics_tolerances.h"
namespace {
constexpr std::array<GmAxis, 3> SpatialAxes = {
    GmAxis::X,
    GmAxis::Y,
    GmAxis::Z,
};

struct AxisCycle {
    GmAxis axis;
    GmAxis next;
    GmAxis other;
};

constexpr std::array<AxisCycle, 3> SpatialAxisCycles = {{
    {GmAxis::X, GmAxis::Y, GmAxis::Z},
    {GmAxis::Y, GmAxis::Z, GmAxis::X},
    {GmAxis::Z, GmAxis::X, GmAxis::Y},
}};
} // namespace

void GmSurfPolygon::ComputeNormalFromVertices() {
    GmSurfPolygon *polygon = this;
    const float edge10x = (polygon->vertices[1].x - polygon->vertices[0].x);
    const float edge10y = (polygon->vertices[1].y - polygon->vertices[0].y);
    const float edge10z = (polygon->vertices[1].z - polygon->vertices[0].z);
    const float edge20x = (polygon->vertices[2].x - polygon->vertices[0].x);
    const float edge20y = (polygon->vertices[2].y - polygon->vertices[0].y);
    const float edge20z = (polygon->vertices[2].z - polygon->vertices[0].z);

    polygon->normal.x = (edge20z * edge10y - edge20y * edge10z);
    polygon->normal.y = (edge10z * edge20x - edge20z * edge10x);
    polygon->normal.z = (edge10x * edge20y - edge20x * edge10y);

    const float lengthSq = (
            (
                    polygon->normal.y * polygon->normal.y +
                    polygon->normal.x * polygon->normal.x) +
            polygon->normal.z * polygon->normal.z);
    if (!(PhysicsTolerance::SurfaceDirectionLengthSquared <= lengthSq)) {
        polygon->normal = GmVec3{1.0f, 0.0f, 0.0f};
        return;
    }

    const float length = (CIsqrt(lengthSq));
    const float invLength = (1.0f / length);
    polygon->normal.x = (invLength * polygon->normal.x);
    polygon->normal.y = (polygon->normal.y * invLength);
    polygon->normal.z = (invLength * polygon->normal.z);
}

float GmBoxAligned::ClampScalarForGmSurf(float value, float minValue, float maxValue) {
    if (value > maxValue) {
        return maxValue;
    }
    if (value < minValue) {
        return minValue;
    }
    return value;
}

static float surfCollisionAbsFloat(float value) {
    return std::fabs(value);
}

int GmBoxAligned::SatAxisOverlapsForGmSurf(float distance, float radius) {
    return !(radius < surfCollisionAbsFloat(distance));
}

GmVec3 GmIso4::TranslationForGmSurf(void) const {
    return translation;
}

GmVec3 GmIso4::TransformPointForGmSurf(const GmVec3 &pointRef) const {
    GmVec3 point = pointRef;
    point.Mult(*this);
    return point;
}

GmVec3 GmIso4::SetMultPointForGmSurf(const GmVec3 &point) const {
    GmVec3 out;
    out.SetMult(point, *this);
    return out;
}

GmVec3 GmIso4::InverseRotateTranslatedPointForGmSurf(const GmVec3 &point) const {
    GmVec3 out = point.SubtractForCollision(TranslationForGmSurf());
    out.MultTranspose(RotationMatrix());
    return out;
}

float LocatedGmSurf::SphereRadius(void) const {
    return static_cast<const GmSurfSphere &>(Surface()).radius;
}

GmLocalMaterialIndex LocatedGmSurf::LocalMaterial(void) const {
    return Surface().material;
}

GmVec3 LocatedGmSurf::BoxCenter(void) const {
    return static_cast<const GmSurfBox &>(Surface()).center;
}

GmVec3 LocatedGmSurf::BoxHalfExtents(void) const {
    return static_cast<const GmSurfBox &>(Surface()).halfExtents;
}

float LocatedGmSurf::EllipsoidBoundRadius(void) const {
    const GmSurfEllipsoid &ellipsoid =
            static_cast<const GmSurfEllipsoid &>(Surface());
    float radius = ellipsoid.radii.x;
    const float radiusY = ellipsoid.radii.y;
    const float radiusZ = ellipsoid.radii.z;
    if (!(radiusY <= radius)) {
        radius = radiusY;
    }
    if (!(radiusZ <= radius)) {
        radius = radiusZ;
    }
    return radius;
}

GmVec3 LocatedGmSurf::EllipsoidRadii(void) const {
    return static_cast<const GmSurfEllipsoid &>(Surface()).radii;
}

const GmSurfPolygon *LocatedGmSurf::Polygon(void) const {
    return &static_cast<const GmSurfPolygon &>(Surface());
}

const GmSurfMesh *LocatedGmSurf::Mesh(void) const {
    return &static_cast<const GmSurfMesh &>(Surface());
}

u32 GmSurfMesh::VertexCount(void) const {
    return static_cast<u32>(vertices.size());
}

u32 GmSurfMesh::TriangleCount(void) const {
    return static_cast<u32>(triangles.size());
}

u32 GmSurfMesh::OctreeCellCount(void) const {
    return static_cast<u32>(octreeCells.size());
}

GmVec3 &GmSurfMesh::MutableVertex(u32 index) {
    return vertices[index];
}

const GmVec3 &GmSurfMesh::Vertex(u32 index) const {
    return vertices[index];
}

GmSurfMeshTriangle &GmSurfMesh::MutableTriangle(u32 index) {
    return triangles[index];
}

const GmSurfMeshTriangle &GmSurfMesh::Triangle(u32 index) const {
    return triangles[index];
}

const GmMeshOctreeCell &GmSurfMesh::OctreeCell(u32 index) const {
    return octreeCells[index];
}

void GmSurfMesh::RecomputeAllPlanes(void) {
    for (u32 triangleIndex = 0u; triangleIndex < TriangleCount(); triangleIndex++) {
        ComputePlane(triangleIndex);
    }
}

GmBoxAligned GmBoxAligned::FromCenterHalfExtents(const GmVec3 &center,
                                                 const GmVec3 &halfExtents) {
    return GmBoxAligned{center, halfExtents};
}

GmVec3 GmBoxAligned::Center(void) const {
    return center;
}

GmVec3 GmBoxAligned::HalfExtents(void) const {
    return halfExtents;
}

int GmBoxAligned::OverlapsOrientedRelativeTo(const GmBoxAligned &rhs,
                                             const GmIso4 &rhsToSelf,
                                             const GmMat3 &absRhsToSelfRotation) const {
    const GmVec3 centerBInA =
        rhsToSelf.TransformPointForGmSurf(rhs.center);
    const GmVec3 delta = centerBInA.SubtractForCollision(center);

    for (const GmAxis axis : SpatialAxes) {
        const float radius =
            halfExtents.Component(axis) +
            absRhsToSelfRotation.Element(axis, GmAxis::X) * rhs.halfExtents.x +
            absRhsToSelfRotation.Element(axis, GmAxis::Y) * rhs.halfExtents.y +
            absRhsToSelfRotation.Element(axis, GmAxis::Z) * rhs.halfExtents.z;
        if (!SatAxisOverlapsForGmSurf(delta.Component(axis), radius)) {
            return 0;
        }
    }

    for (const GmAxis axis : SpatialAxes) {
        const float distance =
            rhsToSelf.Element(GmAxis::X, axis) * delta.x +
            rhsToSelf.Element(GmAxis::Y, axis) * delta.y +
            rhsToSelf.Element(GmAxis::Z, axis) * delta.z;
        const float radius =
            rhs.halfExtents.Component(axis) +
            absRhsToSelfRotation.Element(GmAxis::X, axis) * halfExtents.x +
            absRhsToSelfRotation.Element(GmAxis::Y, axis) * halfExtents.y +
            absRhsToSelfRotation.Element(GmAxis::Z, axis) * halfExtents.z;
        if (!SatAxisOverlapsForGmSurf(distance, radius)) {
            return 0;
        }
    }

    for (const AxisCycle &a : SpatialAxisCycles) {
        for (const AxisCycle &b : SpatialAxisCycles) {
            const float distance =
                delta.Component(a.other) * rhsToSelf.Element(a.next, b.axis) -
                delta.Component(a.next) * rhsToSelf.Element(a.other, b.axis);
            const float radius =
                halfExtents.Component(a.next) *
                        absRhsToSelfRotation.Element(a.other, b.axis) +
                halfExtents.Component(a.other) *
                        absRhsToSelfRotation.Element(a.next, b.axis) +
                rhs.halfExtents.Component(b.next) *
                        absRhsToSelfRotation.Element(a.axis, b.other) +
                rhs.halfExtents.Component(b.other) *
                        absRhsToSelfRotation.Element(a.axis, b.next);
            if (!SatAxisOverlapsForGmSurf(distance, radius)) {
                return 0;
            }
        }
    }

    return 1;
}

void GmCollision::FinalizePolygonCollisionForGmSurf(const LocatedGmSurf &polygon,
                                                    int transformContactPoint) {
    if (polygon.enabled == 0) {
        return;
    }

    if (transformContactPoint) {
        contactPoint.Mult(*polygon.iso);
    }
    const GmMat3 polygonRotation = polygon.iso->RotationMatrix();
    impulseNormal.Mult(polygonRotation);
    separation.Mult(polygonRotation);
}

void GmCollision::FinishPolygonMaterialsForGmSurf(const LocatedGmSurf &sphere,
                                                  const LocatedGmSurf &polygon) {
    localMaterialA = sphere.LocalMaterial();
    localMaterialB = polygon.LocalMaterial();
}

void GmIso4::ScaleRowsForGmSurf(const GmVec3 &rowScale) {
    Element(GmAxis::X, GmAxis::X) = (rowScale.x * Element(GmAxis::X, GmAxis::X));
    Element(GmAxis::X, GmAxis::Y) = (Element(GmAxis::X, GmAxis::Y) * rowScale.x);
    Element(GmAxis::X, GmAxis::Z) = (rowScale.x * Element(GmAxis::X, GmAxis::Z));
    translation.x = (translation.x * rowScale.x);

    Element(GmAxis::Y, GmAxis::X) = (rowScale.y * Element(GmAxis::Y, GmAxis::X));
    Element(GmAxis::Y, GmAxis::Y) = (Element(GmAxis::Y, GmAxis::Y) * rowScale.y);
    Element(GmAxis::Y, GmAxis::Z) = (rowScale.y * Element(GmAxis::Y, GmAxis::Z));
    translation.y = (translation.y * rowScale.y);

    Element(GmAxis::Z, GmAxis::X) = (rowScale.z * Element(GmAxis::Z, GmAxis::X));
    Element(GmAxis::Z, GmAxis::Y) = (Element(GmAxis::Z, GmAxis::Y) * rowScale.z);
    Element(GmAxis::Z, GmAxis::Z) = (rowScale.z * Element(GmAxis::Z, GmAxis::Z));
    translation.z = (translation.z * rowScale.z);
}

int GmBoxAligned::TriangleAxisOverlapsForGmSurf(const GmVec3 &triangleVertex0,
                                                const GmVec3 &triangleVertex1,
                                                const GmVec3 &triangleVertex2,
                                                const GmVec3 &axis) const {
    const float p0 = triangleVertex0.Dot(axis);
    const float p1 = triangleVertex1.Dot(axis);
    const float p2 = triangleVertex2.Dot(axis);
    const GmVec3 projections = {p0, p1, p2};
    const float minP = projections.MinComponentForGmSurf();
    const float maxP = projections.MaxComponentForGmSurf();
    const float radius =
        halfExtents.x * surfCollisionAbsFloat(axis.x) +
        halfExtents.y * surfCollisionAbsFloat(axis.y) +
        halfExtents.z * surfCollisionAbsFloat(axis.z);
    return !(radius < minP) && !(maxP < -radius);
}

int GmBoxAligned::TrianglePlaneOverlapsForGmSurf(const GmVec3 &triangleVertex0,
                                                 const GmVec3 &normal) const {
    const float planeOffset = -normal.Dot(triangleVertex0);
    GmVec3 vmin;
    GmVec3 vmax;

    if (!(0.0f < normal.x)) {
        vmin.x = -halfExtents.x;
        vmax.x = halfExtents.x;
    } else {
        vmin.x = halfExtents.x;
        vmax.x = -halfExtents.x;
    }
    if (!(0.0f < normal.y)) {
        vmin.y = -halfExtents.y;
        vmax.y = halfExtents.y;
    } else {
        vmin.y = halfExtents.y;
        vmax.y = -halfExtents.y;
    }
    if (!(0.0f < normal.z)) {
        vmin.z = -halfExtents.z;
        vmax.z = halfExtents.z;
    } else {
        vmin.z = halfExtents.z;
        vmax.z = -halfExtents.z;
    }

    return !(0.0f < vmax.Dot(normal) + planeOffset) &&
           0.0f <= vmin.Dot(normal) + planeOffset;
}

int GmBoxAligned::OverlapsTriangleInLocalSpaceForGmSurf(
        const GmVec3 &triangleVertex0,
        const GmVec3 &triangleVertex1,
        const GmVec3 &triangleVertex2) const {
    const GmVec3 triangleX = {triangleVertex0.x, triangleVertex1.x, triangleVertex2.x};
    const float minX = triangleX.MinComponentForGmSurf();
    const float maxX = triangleX.MaxComponentForGmSurf();
    if (halfExtents.x < minX || maxX < -halfExtents.x) {
        return 0;
    }

    const GmVec3 triangleY = {triangleVertex0.y, triangleVertex1.y, triangleVertex2.y};
    const float minY = triangleY.MinComponentForGmSurf();
    const float maxY = triangleY.MaxComponentForGmSurf();
    if (halfExtents.y < minY || maxY < -halfExtents.y) {
        return 0;
    }

    const GmVec3 triangleZ = {triangleVertex0.z, triangleVertex1.z, triangleVertex2.z};
    const float minZ = triangleZ.MinComponentForGmSurf();
    const float maxZ = triangleZ.MaxComponentForGmSurf();
    if (halfExtents.z < minZ || maxZ < -halfExtents.z) {
        return 0;
    }

    const GmVec3 edge01 = triangleVertex1.SubtractForCollision(triangleVertex0);
    const GmVec3 edge12 = triangleVertex2.SubtractForCollision(triangleVertex1);
    const GmVec3 edge20 = triangleVertex0.SubtractForCollision(triangleVertex2);
    const GmVec3 normal = edge01.Cross(edge12);
    if (!TrianglePlaneOverlapsForGmSurf(triangleVertex0, normal)) {
        return 0;
    }

    const std::array<GmVec3, 3> edges = {edge01, edge12, edge20};
    const std::array<GmVec3, 3> boxAxes = {{
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    }};
    for (const GmVec3 &edge : edges) {
        for (const GmVec3 &boxAxis : boxAxes) {
            if (!TriangleAxisOverlapsForGmSurf(triangleVertex0,
                                               triangleVertex1,
                                               triangleVertex2,
                                               edge.Cross(boxAxis))) {
                return 0;
            }
        }
    }

    return 1;
}
