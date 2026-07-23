#include "engine/physics/geometry/plug_surface.h"

#include <typeinfo>

#include "engine/physics/geometry/gmsurf_collision.h"

const GmSurf &LocatedGmSurf::Surface(void) const {
    return *surf;
}

void GmCollision::Neg(void) {
    impulseNormal.x = -impulseNormal.x;
    impulseNormal.y = -impulseNormal.y;
    impulseNormal.z = -impulseNormal.z;

    const GmLocalMaterialIndex oldLocalMaterialA = localMaterialA;
    const EPlugSurfaceMaterialId oldMaterialA = materialA;
    separation.x = -separation.x;
    localMaterialA = localMaterialB;
    localMaterialB = oldLocalMaterialA;
    materialA = materialB;
    materialB = oldMaterialA;
    separation.y = -separation.y;
    separation.z = -separation.z;

    extraNegated.x = -extraNegated.x;
    extraNegated.y = -extraNegated.y;
    extraNegated.z = -extraNegated.z;
}

int CPlugSurface::UsesSphereContactBuffer(void) const {
    const GmSurf *surface = Geometry();
    return surface != nullptr && surface->UsesSphereContactBuffer();
}

EPlugSurfaceMaterialId CPlugSurface::SurfaceMaterialIdFromLocalIndex(
        GmLocalMaterialIndex localIndex) const {
    return MaterialAt(localIndex.Index())->SurfaceMaterialId();
}

int GmSurf::ComputeCollision(
        const LocatedGmSurf &aRef,
        const LocatedGmSurf &bRef,
        CGmCollisionBuffer &collisionBufferRef) {
    if (aRef.surf == nullptr || bRef.surf == nullptr) {
        return 0;
    }
    return aRef.surf->Collide(aRef, bRef, collisionBufferRef);
}

int CPlugSurface::ComputeCollision(
        const SPlugSurfaceLocatedPair &pairRef,
        CGmCollisionBuffer &collisionBufferRef) {
    u32 firstNew = collisionBufferRef.GetCurrentCount();
    LocatedGmSurf surfaceB = {
        pairRef.SecondSurface().Geometry(),
        &pairRef.SecondLocation(),
        1,
    };
    LocatedGmSurf surfaceA = {
        pairRef.FirstSurface().Geometry(),
        &pairRef.FirstLocation(),
        1,
    };

    if (!GmSurf::ComputeCollision(surfaceA, surfaceB, collisionBufferRef)) {
        return 0;
    }

    u32 count = collisionBufferRef.GetCurrentCount();
    for (u32 collisionIndex = firstNew; collisionIndex < count;
         collisionIndex++) {
        GmCollision &collision =
                collisionBufferRef.GetCollision(collisionIndex);
        collision.materialA =
                pairRef.FirstSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialA);
        collision.materialB =
                pairRef.SecondSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialB);
    }

    return 1;
}

int CPlugSurface::ComputeCollisionOptimizedCpu(
        const SPlugSurfaceLocatedPair &pairRef,
        CGmCollisionBuffer &collisionBufferRef) {
    u32 firstNew = collisionBufferRef.GetCurrentCount();
    LocatedGmSurf surfaceB = {
        pairRef.SecondSurface().Geometry(),
        &pairRef.SecondLocation(),
        1,
    };
    LocatedGmSurf surfaceA = {
        pairRef.FirstSurface().Geometry(),
        &pairRef.FirstLocation(),
        1,
    };

    int collided = 0;
    if (surfaceA.surf == nullptr || surfaceB.surf == nullptr ||
        typeid(*surfaceB.surf) != typeid(GmSurfMesh)) {
        collided = GmSurf::ComputeCollision(
                surfaceA, surfaceB, collisionBufferRef);
    } else {
        const std::type_info &typeA = typeid(*surfaceA.surf);
        if (typeA == typeid(GmSurfSphere)) {
            collided = GmCollision_Sphere_Mesh_OptimizedCpu(
                    surfaceA, surfaceB, collisionBufferRef);
        } else if (typeA == typeid(GmSurfEllipsoid)) {
            collided = GmCollision_Ellipsoid_Mesh_OptimizedCpu(
                    surfaceA, surfaceB, collisionBufferRef);
        } else {
            collided = GmSurf::ComputeCollision(
                    surfaceA, surfaceB, collisionBufferRef);
        }
    }
    if (!collided) {
        return 0;
    }

    u32 count = collisionBufferRef.GetCurrentCount();
    for (u32 collisionIndex = firstNew; collisionIndex < count;
         collisionIndex++) {
        GmCollision &collision =
                collisionBufferRef.GetCollision(collisionIndex);
        collision.materialA =
                pairRef.FirstSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialA);
        collision.materialB =
                pairRef.SecondSurface().SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialB);
    }

    return 1;
}
