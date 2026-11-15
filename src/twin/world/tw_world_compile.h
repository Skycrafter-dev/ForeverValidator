// FV-dependent: compile FV's flat static-scene export into the twin world.
#ifndef TW_WORLD_COMPILE_H
#define TW_WORLD_COMPILE_H

#include <string>

#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "tw_world.h"

namespace tw {

// Keeps mesh surfaces that FV appends as static collision records and whose
// actor is a static block (group 4) or a race trigger. Triangles keep FV's
// stored vertex order; vertices are transformed with FV's own GmVec3::SetMult.
bool CompileTwinWorld(const forevervalidator::simulation::CudaHostScene &scene, World *out, std::string *diag,
                      SourceScene *source = nullptr);

}  // namespace tw

#endif
