// Twin GPU engine: the response (tw_gpu_stages.cuh).
#include "car_force_consts.h"
#include "response.h"
#include "tw_gpu_stages.cuh"

namespace tw::gpu::dev {

__device__ void Respond(const model::CarForceConsts &k, model::TwinState &s, const model::M3 &R, const model::M3 &Rw, const model::RespContact *c,
                        int n, model::RespTick &tick, model::RespOut &out) {
    model::RespondR(k, s, R, Rw, c, n, tick, out);
}

}  // namespace tw::gpu::dev
