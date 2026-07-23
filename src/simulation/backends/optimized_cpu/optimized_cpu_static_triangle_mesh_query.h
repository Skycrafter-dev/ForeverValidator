#pragma once

#include "engine/physics/geometry/plug_surface.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_mesh_triangle_sidecar.h"

int GmCollision_Sphere_Mesh_OptimizedCpuWithStaticTriangleSidecar(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &mesh,
        const GmIso4 &meshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar &triangles,
        CGmCollisionBuffer &collisionBuffer);

int ComputeCollisionOptimizedCpuWithStaticMeshTriangleSidecar(
        const SPlugSurfaceLocatedPair &pair,
        const GmIso4 &staticMeshInverse,
        const OptimizedCpuStaticMeshTriangleSidecar *triangles,
        CGmCollisionBuffer &collisionBuffer);
