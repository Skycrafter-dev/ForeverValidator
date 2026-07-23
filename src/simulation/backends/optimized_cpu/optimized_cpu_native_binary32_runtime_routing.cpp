#include "simulation/runtime/replay_physics_world.h"

void ReplayPhysicsWorld::StepOptimizedCpuNativeBinary32(
        forevervalidator::simulation::
                OptimizedCpuModel3VehicleForceContext &model3Context) {
    zone_.PhysicsStep2OptimizedCpuNativeBinary32(model3Context);
}
