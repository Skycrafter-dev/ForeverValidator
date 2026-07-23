#pragma once

#include "engine/physics/geometry/plug_surface.h"

int GmCollision_Sphere_Mesh_OptimizedCpuWithMeshInverse(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &mesh,
        const GmIso4 &meshInverse,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Ellipsoid_Mesh_OptimizedCpuWithMeshInverse(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        const GmIso4 &meshInverse,
        CGmCollisionBuffer &collisionBuffer);
int ComputeCollisionOptimizedCpuWithStaticMeshInverse(
        const SPlugSurfaceLocatedPair &pair,
        const GmIso4 &staticMeshInverse,
        CGmCollisionBuffer &collisionBuffer);
