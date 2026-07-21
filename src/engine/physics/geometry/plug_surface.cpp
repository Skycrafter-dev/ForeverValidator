#include "engine/physics/geometry/plug_surface.h"
#include <new>

CPlugSurface::CPlugSurface(void) = default;

CPlugSurface::~CPlugSurface(void) = default;

SPlugSurfaceLocatedPair::SPlugSurfaceLocatedPair(
        CPlugSurface &surfaceA,
        const GmIso4 &isoA,
        CPlugSurface &surfaceB,
        const GmIso4 &isoB)
        : surfaceA_(surfaceA),
          isoA_(isoA),
          surfaceB_(surfaceB),
          isoB_(isoB) {}

CPlugSurface &SPlugSurfaceLocatedPair::FirstSurface(void) const {
    return surfaceA_;
}

const GmIso4 &SPlugSurfaceLocatedPair::FirstLocation(void) const {
    return isoA_;
}

CPlugSurface &SPlugSurfaceLocatedPair::SecondSurface(void) const {
    return surfaceB_;
}

const GmIso4 &SPlugSurfaceLocatedPair::SecondLocation(void) const {
    return isoB_;
}
void CPlugSurface::SetGeometry(CPlugSurfaceGeom *newGeom) {
    geom.MwSetNod(newGeom);
}

CPlugSurfaceGeom *CPlugSurface::GeometryNode(void) const {
    return geom;
}

int CPlugSurface::SetMaterialCount(u32 materialCount) {
    try {
        materials.assign(materialCount, CMwNodRef<CPlugMaterial>());
    } catch (const std::bad_alloc &) {
        materials.clear();
        return 0;
    }
    return 1;
}

void CPlugSurface::ClearMaterials(void) {
    materials.clear();
}

void CPlugSurface::AttachSingleMaterial(CPlugMaterial &material) {
    materials.assign(1u, CMwNodRef<CPlugMaterial>(&material));
}

int CPlugSurface::SetMaterialAt(u32 localIndex,
                                CPlugMaterial *material) {
    if (localIndex >= materials.size()) {
        return 0;
    }
    materials[localIndex].MwSetNod(material);
    return 1;
}

std::unique_ptr<CPlugSurface> CPlugSurface::CloneMeshSurface(void) const {
    if (geom == nullptr) {
        return nullptr;
    }
    const auto *sourceMesh =
            dynamic_cast<const GmSurfMesh *>(geom->Surface());
    if (sourceMesh == nullptr) {
        return nullptr;
    }

    try {
        auto clone = std::make_unique<CPlugSurface>();
        auto clonedGeom = std::make_unique<CPlugSurfaceGeom>();
        clonedGeom->SetId(geom->Id());
        clonedGeom->SetHasDefaultData(geom->HasDefaultData());
        clonedGeom->SetGmSurf(
                std::make_unique<GmSurfMesh>(*sourceMesh));
        clone->SetGeometry(clonedGeom.release());
        clone->materials = materials;
        return clone;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

void CPlugSurface::TranslateMeshVerticesAbove(float yThreshold,
                                              float yDelta) {
    if (geom != nullptr) {
        geom->TranslateMeshVerticesAbove(yThreshold, yDelta);
    }
}

const GmSurf *CPlugSurface::Geometry(void) const {
    return geom != nullptr ? geom->Surface() : nullptr;
}

const GmBoxAligned &CPlugSurface::GeomBox(void) const {
    return geom->Bounds();
}

u32 CPlugSurface::MaterialCount(void) const {
    return (u32)materials.size();
}

CPlugMaterial *CPlugSurface::MaterialAt(u32 localIndex) const {
    return materials[localIndex];
}

CPlugMaterial *CPlugSurface::SingleMaterial(void) const {
    return materials.size() == 1u ? materials.front().Get() : nullptr;
}

int CPlugSurface::AllowsStaticCollisionRecordAppend(void) const {
    // An invalid negative extent suppresses the record; NaN remains appendable.
    return !(0.0f > GeomBox().halfExtents.x);
}
