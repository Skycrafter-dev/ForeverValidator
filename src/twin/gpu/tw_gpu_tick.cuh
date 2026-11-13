// Twin GPU engine: the tick as device functions, one candidate per thread (step/tick.h RunTick with
// step/twin_stages.h TwinStages on the fv policy, the product's).
//
//   TickBegin  race::TwRaceTickBegin, the tick's inputs
//   Plan       FV's substep count (policy fv: no predictor)
//   per substep: CarForces, CarIntegrateR with the force stage's pose, C's collision (tw_gpu_collide.cuh),
//              RespondR with the post-integration rotation, the trigger hits to the race layer
//   TickEnd    the completing substep rerun with shortened steps (race::TwRefineFinish) and TwStoreFinish
#ifndef TW_GPU_TICK_CUH
#define TW_GPU_TICK_CUH

#include <cstring>

#include "car_force_consts.h"
#include "forces.h"
#include "policy.h"
#include "response.h"
#include "tw_gpu_collide.cuh"
#include "tw_gpu_stages.cuh"
#include "tw_race.h"
#include "tw_state.h"

namespace tw::gpu::dev {

// One world's packs on the device.
struct DevPacks {
    const model::CarForceConsts *k;
    const model::WaterZone *water;   // null: no water
    const race::TwRaceMap *race;
    DevWorld w;
};

// Measurement (QueueStats::stageCycles, clk non-null): SM clock cycles summed per thread: [0] forces + integration,
// [1] collision, [2] response (inside RunSubstep), [3] the tick start (race, inputs, substep count), [4] the finish
// snapshot copies, [5] whole RunSubstep calls.
// RunSubstep with TwinStages: Forces, Integrate, Detect, Respond, RaceHits (inline: the lane runner has one call site;
// out of line it cost ~1%). kInline: forces + integration and the response inlined here (the lane runner's pass: a
// call to a stage's module saves and restores ~50 callee-saved registers in local memory and reaches the state
// through generic pointers); else the stages' modules (tw_gpu_stages.cuh: v1 and the checks, whose extra copies of
// the substep would multiply ptxas's work).
#ifndef TW_GPU_INLINE_DETECT
#define TW_GPU_INLINE_DETECT 1
#endif
constexpr bool kInlineDetect = TW_GPU_INLINE_DETECT != 0;   // kInline: the collision inlined too (Detect)
// K: the car's constants (*P.k, or the same bytes in constant memory: tw_gpu_session.cu).
template <bool kInline = false>
__device__ __forceinline__ void RunSubstepK(const DevPacks &P, const model::CarForceConsts &K, model::TwinState &s, const model::CarTickIn &in, float h,
                                         model::RespTick &respTick, race::TwRaceEvents &raceEv, GpuContacts &contacts, std::uint32_t *lost,
                                         unsigned long long *clk) {
    const long long t0 = clock64();
    model::CarForceOut f;
    if constexpr (kInline) {
        model::CarForces(K, P.water, s, in, h, f, P.k->fakeTextureRed);   // the texture from global memory (P.k: K's bytes)
        model::M3 R;
        std::memcpy(&R, f.poseRot, sizeof R);
        model::CarIntegrateR(K, s, f, h, R);
    } else {
        ForcesIntegrate(K, P.water, s, in, h, f);
    }
    const long long t1 = clock64();
    if constexpr (kInline && kInlineDetect) {   // B2's DetectAt module for its measurements (stats, repeat)
        if (P.w.stats == nullptr && P.w.repeat == 1)
            Detect(P.w, s.pos, s.quat, s.absorb, contacts);
        else
            DetectAt(P.w, s.pos, s.quat, s.absorb, contacts);
    } else {
        DetectAt(P.w, s.pos, s.quat, s.absorb, contacts);
    }
    const long long t2 = clock64();
    *lost += contacts.rawLost;
    model::M3 Rw;   // the substep-start pose (CarForceOut::poseRot)
    std::memcpy(&Rw, f.poseRot, sizeof Rw);
    const model::M3 Rpost = model::rot_from_quat(s.quat);
    model::RespOut resp;
    if constexpr (kInline)
        model::RespondR(K, s, Rpost, Rw, contacts.c, contacts.n, respTick, resp);
    else
        Respond(K, s, Rpost, Rw, contacts.c, contacts.n, respTick, resp);
    const long long t3 = clock64();
    if (clk != nullptr) {
        clk[0] += static_cast<unsigned long long>(t1 - t0);
        clk[1] += static_cast<unsigned long long>(t2 - t1);
        clk[2] += static_cast<unsigned long long>(t3 - t2);
    }
    if (resp.triggerCount != 0u) {
        race::TwTriggerHit hits[model::kRespMaxContacts];
        for (std::uint32_t i = 0; i < resp.triggerCount; ++i) {
            const model::RespContact &ct = contacts.c[resp.trigger[i]];
            hits[i].trigger = static_cast<std::uint16_t>(ct.peerId);
            hits[i].shape = ct.shape;
            hits[i].flags = 0u;
        }
        race::TwRaceOnTriggerHits(*P.race, s, hits, resp.triggerCount, raceEv);
    }
}
template <bool kInline = false>
__device__ __forceinline__ void RunSubstep(const DevPacks &P, model::TwinState &s, const model::CarTickIn &in, float h, model::RespTick &respTick,
                                        race::TwRaceEvents &raceEv, GpuContacts &contacts, std::uint32_t *lost, unsigned long long *clk) {
    RunSubstepK<kInline>(P, *P.k, s, in, h, respTick, raceEv, contacts, lost, clk);
}

// RunTick: `contacts` and `finishStart` are the thread's scratch; *substeps counts the substeps run.
__device__ inline void RunTick(const DevPacks &P, model::TwinState &s, std::uint32_t control, GpuContacts &contacts,
                               model::TwinState &finishStart, std::uint32_t *substeps, std::uint32_t *lost, unsigned long long *clk) {
    const long long b0 = clock64();
    race::TwRaceEvents raceEv;
    race::TwRaceEventsClear(raceEv);
    model::RespTick respTick;
    std::memset(&respTick, 0, sizeof respTick);
    race::TwRaceTickBegin(*P.race, s, control, race::TwCtlRespawnField(control), s.tickMs + 10u, P.k->restAbsorb, raceEv);
    const race::TwTickInputs ti = race::TwCtlPhysicsInputs(control);
    model::CarTickIn in;
    in.gas = ti.gas;
    in.brake = ti.brake;
    in.steer = ti.steer;
    in.tickMs = s.tickMs;
    std::uint32_t n = step::FvSubstepCount(s, step::kTickDt);
    if (n == 0u) n = 1u;
    if (clk != nullptr) clk[3] += static_cast<unsigned long long>(clock64() - b0);
    bool found = false;
    double startSeconds = 0.0, elapsed = 0.0;
    float duration = 0.0f;
    step::ForEachSubstepDt(n, step::kTickDt, [&](std::uint32_t, float h) {
        const bool track = !found && !s.raceCompleted;
        const long long c0 = clock64();
        if (track) finishStart = s;
        const long long c1 = clock64();
        if (clk != nullptr) clk[4] += static_cast<unsigned long long>(c1 - c0);
        RunSubstep(P, s, in, h, respTick, raceEv, contacts, lost, clk);
        if (clk != nullptr) clk[5] += static_cast<unsigned long long>(clock64() - c1);
        if (track && s.raceCompleted) {
            found = true;
            startSeconds = elapsed;
            duration = h;
        }
        elapsed += static_cast<double>(h);
        ++*substeps;
    });
    if (raceEv.raceCompleted == 0u || !found) return;
    // TickEnd (R4.2): the completing substep again from its start, fresh stage state, shortened steps
    auto probe = [&](float partial) {
        model::TwinState q = finishStart;
        race::TwRaceEvents ev{};
        model::RespTick rt{};
        RunSubstep(P, q, in, partial, rt, ev, contacts, lost, nullptr);
        return q.raceCompleted != 0u;
    };
    const race::TwFinishSubstep sub{s.tickMs, startSeconds, duration};
    std::uint64_t lower = 0u, upper = 0u;
    if (race::TwRefineFinish(sub, probe, &lower, &upper)) race::TwStoreFinish(s, lower);
}

}  // namespace tw::gpu::dev

#endif
