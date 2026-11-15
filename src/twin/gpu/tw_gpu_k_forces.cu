// Twin GPU engine: forces + integration (tw_gpu_stages.cuh).
#include "car_force_consts.h"
#include "forces.h"
#include "tw_gpu_stages.cuh"

namespace tw::gpu::dev {

__device__ void ForcesIntegrate(const model::CarForceConsts &k, const model::WaterZone *water, model::TwinState &s, const model::CarTickIn &in,
                                float h, model::CarForceOut &f) {
    model::CarForces(k, water, s, in, h, f);
    model::M3 R;
    std::memcpy(&R, f.poseRot, sizeof R);
    model::CarIntegrateR(k, s, f, h, R);
}

}  // namespace tw::gpu::dev
