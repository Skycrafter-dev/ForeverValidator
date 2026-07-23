#include "simulation/runtime/replay_physics_world.h"

#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"

void ReplayPhysicsWorld::StepOptimizedCpuCached(
        const OptimizedCpuStaticSurfaceTransformCache &transforms) {
    zone_.PhysicsStep2OptimizedCpuCached(transforms);
}

void ReplayPhysicsWorld::StepOptimizedCpuNativeBinary32Cached(
        const OptimizedCpuStaticSurfaceTransformCache &transforms,
        forevervalidator::simulation::
                OptimizedCpuModel3VehicleForceContext &model3Context) {
    zone_.PhysicsStep2OptimizedCpuNativeBinary32Cached(
            transforms, model3Context);
}
