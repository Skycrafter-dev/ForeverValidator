// Twin FV adapter: the car's collision ellipsoids for the M3 collision stage
// (src/twin/collide/tw_collide.h), from FV's flat static configuration
// (BuildCudaHostStaticConfiguration). Only src/twin/fv/ includes FV headers.
#ifndef TW_FV_COMPILE_CAR_SHAPES_H
#define TW_FV_COMPILE_CAR_SHAPES_H

#include <string>

#include "tw_collide.h"

namespace forevervalidator::simulation {
struct CudaHostStaticConfiguration;
}

namespace tw::fv {

// The 8 ellipsoids in tree (traversal) order: shape-to-car body pose
// (CudaVehicleCollisionShape::bodyPose = SetMult(localPose, parent bodyPose)), radii
// (localBounds.halfExtents), wheel index; and the four wheel rest surface poses.
// Refuses anything but 8 ellipsoids, or a wheel shape with a parent.
bool CompileCarShapes(const forevervalidator::simulation::CudaHostStaticConfiguration &cfg,
                      tw::collide::CarShapes *out, std::string *diag);

}  // namespace tw::fv

#endif
