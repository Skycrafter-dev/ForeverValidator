// Twin GPU engine: the substep's large stages as separate device functions, each in its own translation unit
// (separable compilation): forces + integration (tw_gpu_k_forces.cu), C's collision (tw_gpu_k_detect.cu) and
// the response (tw_gpu_k_respond.cu). Together in one module the kernel does not compile in 10 GB of host
// memory (ptxas); alone each takes one to a few GB.
#ifndef TW_GPU_STAGES_CUH
#define TW_GPU_STAGES_CUH

#include "forces_io.h"
#include "response.h"
#include "tw_gpu.h"
#include "tw_state.h"
#include "vec3.h"

namespace tw::model {
struct CarForceConsts;
struct WaterZone;
}  // namespace tw::model

namespace tw::gpu::dev {

struct DevWorld;

// CarForces, then CarIntegrateR with the force stage's pose rotation (TwinStages::Forces + Integrate, fv policy).
__device__ void ForcesIntegrate(const model::CarForceConsts &k, const model::WaterZone *water, model::TwinState &s, const model::CarTickIn &in,
                                float h, model::CarForceOut &f);
// RespondR (TwinStages::Respond).
__device__ void Respond(const model::CarForceConsts &k, model::TwinState &s, const model::M3 &R, const model::M3 &Rw, const model::RespContact *c,
                        int n, model::RespTick &tick, model::RespOut &out);
// C's collision at a car pose (tw_gpu_collide.cuh Detect).
__device__ void DetectAt(const DevWorld &w, const float pos[3], const float quat[4], const float absorb[4], GpuContacts &out);

}  // namespace tw::gpu::dev

#endif
