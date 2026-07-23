#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "engine/core/engine_types.h"
#include "engine/core/gm_types.h"
#include "engine/game/surface_material.h"
struct CGmCollisionBuffer;

struct GmSurf;
struct GmSurfPolygon;
struct GmSurfMesh;
struct GmSurfMeshOptimizedCpuAccess;
struct GmSurfMeshEllipsoidOptimizedCpuAccess;
struct GmSurfMeshStaticInverseOptimizedCpuAccess;

struct LocatedGmSurf {
    const GmSurf *surf = nullptr;
    const GmIso4 *iso = nullptr;
    bool enabled = true;

    const GmSurf &Surface(void) const;
    float SphereRadius(void) const;
    GmLocalMaterialIndex LocalMaterial(void) const;
    GmVec3 BoxCenter(void) const;
    GmVec3 BoxHalfExtents(void) const;
    float EllipsoidBoundRadius(void) const;
    GmVec3 EllipsoidRadii(void) const;
    const GmSurfPolygon *Polygon(void) const;
    const GmSurfMesh *Mesh(void) const;
};

struct GmCollision {
    GmVec3 separation{};
    GmVec3 impulseNormal{};
    GmVec3 contactPoint{};
    GmLocalMaterialIndex localMaterialA;
    GmLocalMaterialIndex localMaterialB;
    EPlugSurfaceMaterialId materialA = EPlugSurfaceMaterialId_Concrete;
    EPlugSurfaceMaterialId materialB = EPlugSurfaceMaterialId_Concrete;
    bool sphereMergePrimary = false;
    GmVec3 extraNegated{};

    void Neg(void);
    void FinalizePolygonCollisionForGmSurf(const LocatedGmSurf &polygon,
                                           int transformContactPoint);
    void FinishPolygonMaterialsForGmSurf(const LocatedGmSurf &sphere,
                                         const LocatedGmSurf &polygon);
};

struct GmSurf {
    enum class EGmSurfType : unsigned long {
        Sphere = 0u,
        Ellipsoid = 1u,
        Polygon = 5u,
        Box = 6u,
        Mesh = 7u,
    };

    GmSurf(void);
    virtual ~GmSurf(void);

    GmLocalMaterialIndex material;

    static int ComputeCollision(const LocatedGmSurf &a,
                                const LocatedGmSurf &b,
                                CGmCollisionBuffer &collisionBuffer);
    void CreateDefaultData(void);
    void GetBoundingBox(GmBoxAligned &out) const;
    virtual int UsesSphereContactBuffer(void) const;
    virtual int GetRollingRadius(float &outRadius) const;

protected:
    virtual void ComputeBoundingBox(GmBoxAligned &out) const = 0;
    virtual int Collide(const LocatedGmSurf &self,
                        const LocatedGmSurf &other,
                        CGmCollisionBuffer &collisionBuffer) const = 0;
};

struct GmSurfSphere : GmSurf {
    float radius = 0.0f;

    GmSurfSphere(void);
    ~GmSurfSphere(void) override = default;
    void GetSphereBoundingBox(GmBoxAligned &out) const;
    int UsesSphereContactBuffer(void) const override;
    int GetRollingRadius(float &outRadius) const override;

protected:
    void ComputeBoundingBox(GmBoxAligned &out) const override;
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override;
};

struct GmSurfEllipsoid : GmSurf {
    GmVec3 radii{};

    GmSurfEllipsoid(void);
    ~GmSurfEllipsoid(void) override;
    void GetEllipsoidBoundingBox(GmBoxAligned &out) const;
    int UsesSphereContactBuffer(void) const override;
    int GetRollingRadius(float &outRadius) const override;

protected:
    void ComputeBoundingBox(GmBoxAligned &out) const override;
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override;
};

struct GmSurfBox : GmSurf {
    GmVec3 center{};
    GmVec3 halfExtents{};

    GmSurfBox(void);

protected:
    void ComputeBoundingBox(GmBoxAligned &out) const override;
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override;
};

struct GmSurfPolygon : GmSurf {
    std::array<GmVec3, 4> vertices{};
    std::uint8_t vertexCount = 0u;
    GmVec3 normal{};
    bool backSide = false;

    explicit GmSurfPolygon(std::uint8_t vertexCount);
    ~GmSurfPolygon(void) override = default;
    void ComputeNormalFromVertices(void);

protected:
    void ComputeBoundingBox(GmBoxAligned &out) const override;
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override;
};

struct GmSurfMeshTriangle {
    GmVec3 normal{};
    float planeDistance = 0.0f;
    std::array<u32, 3> vertexIndex{};
    GmLocalMaterialIndex material;
};

struct SMeshOctreeCell {
    static SMeshOctreeCell Branch(const GmBoxAligned &bounds);
    static SMeshOctreeCell Triangle(const GmBoxAligned &bounds,
                                    u32 triangleIndex);

    const GmBoxAligned &Bounds(void) const { return bounds_; }
    bool ContainsTriangle(void) const { return triangleIndex_.has_value(); }
    u32 TriangleIndex(void) const { return *triangleIndex_; }
    u32 SubtreeEntryCount(void) const { return subtreeEntryCount_; }
    void SetBounds(const GmBoxAligned &bounds) { bounds_ = bounds; }
    void SetSubtreeEntryCount(u32 count) { subtreeEntryCount_ = count; }

private:
    u32 subtreeEntryCount_ = 1u;
    GmBoxAligned bounds_{};
    std::optional<u32> triangleIndex_;
};

template <typename Cell>
class GmOctree;

template <>
class GmOctree<SMeshOctreeCell> {
public:
    u32 BuildOctreeRecurse(u32 sourceCount, SMeshOctreeCell *sourceCells);
    u32 BuildBintreeRecurse(u32 sourceCount,
                            SMeshOctreeCell *sourceCells,
                            u32 depth,
                            u32 maxDepth,
                            u32 minLeafCount,
                            float volumeThreshold);
    void Build(u32 sourceCount,
               SMeshOctreeCell *sourceCells,
               int useBintree,
               u32 maxDepth,
               u32 minLeafCount,
               float volumeThreshold);

    const std::vector<SMeshOctreeCell> &Cells(void) const;
    std::vector<SMeshOctreeCell> TakeCells(void);

private:
    u32 BuildOctreeNodes(const std::vector<SMeshOctreeCell> &sourceCells);
    u32 BuildBintreeNodes(const std::vector<SMeshOctreeCell> &sourceCells,
                          u32 depth,
                          u32 maxDepth,
                          u32 minLeafCount,
                          float volumeThreshold);
    u32 AppendInternal(void);
    void AppendLeaf(const SMeshOctreeCell &source);

    std::vector<SMeshOctreeCell> cells;
};

using GmMeshOctree = GmOctree<SMeshOctreeCell>;

struct GmSurfMesh : GmSurf {
    enum class PlaneSource {
        Generated,
        Archived,
    };

    GmSurfMesh(void);
    ~GmSurfMesh(void) override;
    void GetMeshBoundingBox(GmBoxAligned &out) const;
    void ComputePlane(unsigned long triangleIndex);
    void BuildOctree(void);

    bool SetGeometry(std::vector<GmVec3> meshVertices,
                     std::vector<GmSurfMeshTriangle> meshTriangles,
                     std::vector<GmMeshOctreeCell> meshOctreeCells = {},
                     PlaneSource planeSource = PlaneSource::Generated);
    void ClearGeometry(void);
    void TranslateVerticesAbove(float yThreshold, float yDelta);
    void TransformByNOMat(const GmIso4 &transform);
    u32 VertexCount(void) const;
    u32 TriangleCount(void) const;
    u32 OctreeCellCount(void) const;
    const GmVec3 &Vertex(u32 index) const;
    const GmSurfMeshTriangle &Triangle(u32 index) const;
    const GmMeshOctreeCell &OctreeCell(u32 index) const;

protected:
    void ComputeBoundingBox(GmBoxAligned &out) const override;
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override;

private:
    friend struct GmSurfMeshOptimizedCpuAccess;
    friend struct GmSurfMeshEllipsoidOptimizedCpuAccess;
    friend struct GmSurfMeshStaticInverseOptimizedCpuAccess;

    GmVec3 &MutableVertex(u32 index);
    GmSurfMeshTriangle &MutableTriangle(u32 index);
    void RecomputeAllPlanes(void);
    void RefreshDerivedGeometry(void);
    void SetOctree(std::vector<GmMeshOctreeCell> cells) {
        octreeCells = std::move(cells);
    }

    std::vector<GmVec3> vertices;
    std::vector<GmSurfMeshTriangle> triangles;
    std::vector<GmMeshOctreeCell> octreeCells;
};
