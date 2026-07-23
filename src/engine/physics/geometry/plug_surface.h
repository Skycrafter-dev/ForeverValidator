#pragma once

#include "engine/core/engine_types.h"
#include <memory>
#include <vector>

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gm_surface.h"
#include "engine/core/mw_id.h"
#include "engine/rendering/plug.h"
#include "engine/rendering/plug_material.h"
struct CPlugSurface;

struct CPlugSurfaceGeom : CPlug {
    CPlugSurfaceGeom(void);
    ~CPlugSurfaceGeom(void) override;
    virtual void CreateDefaultData(void);
    void SetGmSurf(std::unique_ptr<GmSurf> surf);
    const GmSurf *Surface(void) const;
    const CMwId &Id(void) const;
    void SetId(const CMwId &id);
    bool HasDefaultData(void) const;
    void SetHasDefaultData(bool hasDefaultData);
    const GmBoxAligned &Bounds(void) const;
    void ComputeBoundingBox(void);
    void TranslateMeshVerticesAbove(float yThreshold, float yDelta);
    void TransformByNOMat(const GmIso4 &transform);

private:
    CMwId id_;
    bool hasDefaultData_ = true;
    GmBoxAligned bounds_{};
    std::unique_ptr<GmSurf> gmSurf;
};

struct CPlugSurfaceMaterialData {
    float frictionCoef = 0.0f;
    float restitutionCoef = 0.0f;
    static CPlugSurfaceMaterialData ForMaterial(
            EPlugSurfaceMaterialId materialId);
    float GetRestitutionCoefWith(
            const CPlugSurfaceMaterialData &other);
};

struct SPlugSurfaceLocatedPair {
    SPlugSurfaceLocatedPair(CPlugSurface &surfaceA,
                            const GmIso4 &isoA,
                            CPlugSurface &surfaceB,
                            const GmIso4 &isoB);
    CPlugSurface &FirstSurface(void) const;
    const GmIso4 &FirstLocation(void) const;
    CPlugSurface &SecondSurface(void) const;
    const GmIso4 &SecondLocation(void) const;

private:
    CPlugSurface &surfaceA_;
    const GmIso4 &isoA_;
    CPlugSurface &surfaceB_;
    const GmIso4 &isoB_;
};

struct CPlugSurface : CPlug {
    CPlugSurface(void);
    ~CPlugSurface(void) override;
    void SetGeometry(CPlugSurfaceGeom *geom);
    CPlugSurfaceGeom *GeometryNode(void) const;
    const GmSurf *Geometry(void) const;
    std::unique_ptr<CPlugSurface> CloneMeshSurface(void) const;
    int SetMaterialCount(u32 materialCount);
    void ClearMaterials(void);
    void AttachSingleMaterial(CPlugMaterial &material);
    int SetMaterialAt(u32 localIndex, CPlugMaterial *material);
    const GmBoxAligned &GeomBox(void) const;
    u32 MaterialCount(void) const;
    CPlugMaterial *MaterialAt(u32 localIndex) const;
    CPlugMaterial *SingleMaterial(void) const;
    int AllowsStaticCollisionRecordAppend(void) const;
    int UsesSphereContactBuffer(void) const;
    void TranslateMeshVerticesAbove(float yThreshold, float yDelta);
    EPlugSurfaceMaterialId SurfaceMaterialIdFromLocalIndex(
            GmLocalMaterialIndex localIndex) const;
    static int ComputeCollision(const SPlugSurfaceLocatedPair &pair,
                                CGmCollisionBuffer &collisionBuffer);
    static int ComputeCollisionOptimizedCpu(
            const SPlugSurfaceLocatedPair &pair,
            CGmCollisionBuffer &collisionBuffer);

private:
    CMwNodRef<CPlugSurfaceGeom> geom;
    std::vector<CMwNodRef<CPlugMaterial>> materials;
};
