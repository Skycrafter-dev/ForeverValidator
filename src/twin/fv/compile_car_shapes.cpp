// See compile_car_shapes.h.
#include "compile_car_shapes.h"

#include <cstdint>

#include "engine/physics/geometry/gm_surface.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"

namespace tw::fv {

using forevervalidator::simulation::CudaHostStaticConfiguration;
using forevervalidator::simulation::CudaVehicleCollisionShape;

namespace {
void CopyIso(const GmIso4 &a, tw::collide::Iso *o) {
    const GmVec3 *cols[3] = {&a.rotation.basisX, &a.rotation.basisY, &a.rotation.basisZ};
    for (int c = 0; c < 3; ++c) { o->r[3 * c] = cols[c]->x; o->r[3 * c + 1] = cols[c]->y; o->r[3 * c + 2] = cols[c]->z; }
    o->t[0] = a.translation.x; o->t[1] = a.translation.y; o->t[2] = a.translation.z;
}
}  // namespace

bool CompileCarShapes(const CudaHostStaticConfiguration &cfg, tw::collide::CarShapes *out, std::string *diag) {
    tw::collide::CarShapes k;
    if (cfg.collisionShapes.size() != 8u) { *diag = "car collision model is not 8 shapes"; return false; }
    bool seen[8] = {};
    for (std::size_t i = 0; i < 8u; ++i) {
        const CudaVehicleCollisionShape &sh = cfg.collisionShapes[i];
        if (sh.traversalOrder >= 8u || seen[sh.traversalOrder]) { *diag = "collision shapes: bad traversal order"; return false; }
        if (sh.surfaceType != static_cast<std::uint32_t>(GmSurf::EGmSurfType::Ellipsoid)) { *diag = "collision shape is not an ellipsoid"; return false; }
        seen[sh.traversalOrder] = true;
        tw::collide::CarShape &o = k.shape[sh.traversalOrder];
        CopyIso(sh.bodyPose, &o.bodyPose);
        o.radii[0] = sh.localBounds.halfExtents.x; o.radii[1] = sh.localBounds.halfExtents.y; o.radii[2] = sh.localBounds.halfExtents.z;
        if (sh.wheelIndex == UINT32_MAX) {
            o.wheel = -1;
        } else {
            if (sh.wheelIndex > 3u || sh.parentShapeIndex != UINT32_MAX) { *diag = "wheel shape with a parent or bad wheel index"; return false; }
            o.wheel = static_cast<std::int32_t>(sh.wheelIndex);
        }
    }
    for (int w = 0; w < 4; ++w) CopyIso(cfg.wheels.wheels[static_cast<std::size_t>(w)].restSurfacePose, &k.wheelRest[w]);
    *out = k;
    return true;
}

}  // namespace tw::fv
