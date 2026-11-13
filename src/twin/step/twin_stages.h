// The twin's own tick stages (all-twin path; static dispatch for tick.h). FV-free.
//
//   TickBegin  race::TwRaceTickBegin (docs/twin/m6a-race.md "The tick contract")
//   Plan       PlanSubsteps (policy.h) with the pack's ContactPredictor
//   Forces     model::CarForcesR       Integrate  model::CarIntegrateR
//   Detect     the pack's CollisionStage (collide/, M3)
//   Respond    model::RespondOrdered (sort + response + position correction)
//   RaceHits   race::TwRaceOnTriggerHits, hits = {peerId (trigger id), shape} of
//              RespOut::trigger in processing order
//   TickEnd    race::TwRefineFinish (rerun of the completing substep) + TwStoreFinish
#ifndef TW_STEP_TWIN_STAGES_H
#define TW_STEP_TWIN_STAGES_H

#include <cstdint>
#include <cstring>

#include "forces.h"
#include "policy.h"
#include "response.h"
#include "stages.h"
#include "tick.h"
#include "tw_race.h"
#include "tw_state.h"

namespace tw {
namespace step {

constexpr std::uint32_t kTickMs = 10u;

// The speculative policy's tick-start test (L7, n0) from the twin's own
// collision: a detection at the tick-start pose gives K(0), and an event if a
// contact point moves along its normal faster than q.normalSpeed (as FV's probe:
// v + w x (P - pos)) or a contact is deeper than q.depth. Used when no external
// predictor is set; the detection is kept for SpeculateBegin.
class StartPassProbe final : public ContactPredictor {
public:
    StartPassProbe(const CollisionStage *collision, CollisionCache *cache) : collision_(collision), cache_(cache) {}
    bool ContactEvent(const model::TwinState &, const PredictQuery &) const override { return true; }   // not a probe
    bool StartEvent(const model::TwinState &s, const PredictQuery &q) const override {
        collision_->DetectCached(s, cache_, contacts);
        valid = true;
        q.startTouching = contacts.n > 0;
        float maxNormalSpeed = 0.0f, maxDepth = 0.0f;
        for (int i = 0; i < contacts.n; ++i) {
            const model::RespContact &c = contacts.c[i];
            const float rx = c.P[0] - s.pos[0], ry = c.P[1] - s.pos[1], rz = c.P[2] - s.pos[2];
            const float vx = s.vel[0] + s.angVel[1] * rz - s.angVel[2] * ry;
            const float vy = s.vel[1] + s.angVel[2] * rx - s.angVel[0] * rz;
            const float vz = s.vel[2] + s.angVel[0] * ry - s.angVel[1] * rx;
            const float nn = tw::math::sqrt((c.N[0] * c.N[0] + c.N[1] * c.N[1]) + c.N[2] * c.N[2]);
            if (nn > 1e-9f) {
                const float vn = tw::math::abs(((vx * c.N[0] + vy * c.N[1]) + vz * c.N[2]) / nn);
                if (vn > maxNormalSpeed) maxNormalSpeed = vn;
            }
            const float d = tw::math::sqrt((c.Sep[0] * c.Sep[0] + c.Sep[1] * c.Sep[1]) + c.Sep[2] * c.Sep[2]);
            if (d > maxDepth) maxDepth = d;
        }
        return maxNormalSpeed > q.normalSpeed || maxDepth > q.depth;
    }
    mutable SubstepContacts contacts;   // the tick-start pass
    mutable bool valid = false;

private:
    const CollisionStage *collision_;
    CollisionCache *cache_;   // per candidate, may be null (stages.h)
};

// Everything the all-twin tick reads (immutable, shared).
struct TwinPacks {
    const model::CarForceConsts *car = nullptr;
    const model::WaterZone *water = nullptr;       // null: no water on the map
    const CollisionStage *collision = nullptr;
    const race::TwRaceMap *race = nullptr;
    const ContactPredictor *predictor = nullptr;   // needed by ContactAware only
    PolicyConfig policy{};
};

inline bool TwinPacksComplete(const TwinPacks &p) {
    return p.car != nullptr && p.collision != nullptr && p.race != nullptr &&
           (p.policy.kind != PolicyKind::ContactAware || p.predictor != nullptr);   // Speculative: own start pass
}

struct TwinStages {
    const TwinPacks &p;
    CollisionCache *cache;     // the caller's per-candidate collision cache, may be null (stages.h)
    race::TwRaceEvents raceEv{};
    model::RespTick respTick{};
    SubstepContacts scratch;   // finish probe reruns, tick-start detection
    SubstepPlan plan{};        // this tick's plan (ballistic correction)
    bool subcycled = false;    // L8 ran this substep's forces + integration
    StartPassProbe startProbe{p.collision, cache};
    float tickDt = kTickDt;
    model::M3 Rpost{};         // post-integration rotation (Integrate -> Respond)
    // Speculative policy: contact keys (shape << 16 | material << 8 | trigger) of
    // the last collision pass and of the tick start; saved side state.
    std::uint32_t lastKeys[model::kRespMaxContacts], startKeys[model::kRespMaxContacts];
    int lastKeyCount = 0, startKeyCount = 0;
    race::TwRaceEvents savedRaceEv{};
    model::RespTick savedRespTick{};
    std::uint32_t savedTriggerHits = 0u;

    static int Keys(const SubstepContacts &c, std::uint32_t *k) {   // sorted, unique
        int n = 0;
        for (int i = 0; i < c.n; ++i) {
            const std::uint32_t key = static_cast<std::uint32_t>(c.c[i].shape) << 16 |
                                      static_cast<std::uint32_t>(c.c[i].material) << 8 |
                                      ((c.c[i].flags & model::kRespTrigger) ? 1u : 0u);
            int j = n;
            while (j > 0 && k[j - 1] > key) --j;
            if (j > 0 && k[j - 1] == key) continue;
            for (int m = n; m > j; --m) k[m] = k[m - 1];
            k[j] = key;
            ++n;
        }
        return n;
    }

    explicit TwinStages(const TwinPacks &packs, CollisionCache *collisionCache = nullptr)
        : p(packs), cache(collisionCache) {}

    TW_INLINE float GravityY(const model::TwinState &s) const {
        return p.car->gravity[1] * (s.groundedPrev ? p.car->fsGrounded : p.car->fsAirborne);
    }

    TW_INLINE void TickBegin(model::TwinState &s, TickContext &t) {
        race::TwRaceEventsClear(raceEv);
        std::memset(&respTick, 0, sizeof respTick);
        race::TwRaceTickBegin(*p.race, s, t.control, t.respawnCount, s.tickMs + kTickMs, p.car->restAbsorb, raceEv);
        const race::TwTickInputs in = race::TwCtlPhysicsInputs(t.control);
        t.in.gas = in.gas;
        t.in.brake = in.brake;
        t.in.steer = in.steer;
        t.in.tickMs = s.tickMs;
        t.respawned = raceEv.respawned;
    }
    TW_INLINE SubstepPlan Plan(const model::TwinState &s, TickContext &t) {
        startProbe.valid = false;
        const ContactPredictor *predictor =
                p.predictor != nullptr || p.policy.kind != PolicyKind::Speculative ? p.predictor : &startProbe;
        plan = PlanSubsteps(p.policy, s, GravityY(s), predictor, t.dt);
        tickDt = t.dt;
        return plan;
    }
    TW_INLINE void SubstepBegin(model::TwinState &, float) {}
    TW_INLINE void Forces(model::TwinState &s, const TickContext &t, float h, model::CarForceOut &out) {
        if (p.policy.subcycle && h == tickDt && plan.nFv > 1u && !plan.event && !plan.startTouching) {
            // L8: an airborne 1-substep tick runs the force model and the integration FV's n times
            // at FV's split of dt, with one collision pass at the tick end (no contact assumed in
            // between, as the 1-substep decision already does); FV's arithmetic, so no b1 here
            bool first = true;
            ForEachSubstepDt(plan.nFv, h, [&](std::uint32_t, float hi) {
                model::CarForceOut o;
                model::CarForces(*p.car, p.water, s, t.in, hi, o);
                model::M3 R;
                std::memcpy(&R, o.poseRot, sizeof R);
                model::CarIntegrateR(*p.car, s, o, hi, R);
                if (first) out = o;
                first = false;
            });
            subcycled = true;
            return;
        }
        model::CarForces(*p.car, p.water, s, t.in, h, out);
    }
    TW_INLINE void Integrate(model::TwinState &s, const model::CarForceOut &f, float h) {
        if (subcycled) {   // L8 already integrated the tick
            subcycled = false;
            Rpost = model::rot_from_quat(s.quat);
            return;
        }
        // the old rotation is the force stage's pose (CarForceOut::poseRot = R(q) at substep start)
        model::M3 R;
        std::memcpy(&R, f.poseRot, sizeof R);
        model::CarIntegrateR(*p.car, s, f, h, R);
        if (p.policy.ballistic && plan.nFv > 1u && h == tickDt) {
            // one substep for the whole tick: FV's n-substep forward-Euler term for this force,
            // F/m dt^2 (n-1)/(2n) (substep-policy.md "ballistic correction")
            const float c = (h * h) * (static_cast<float>(plan.nFv - 1u) / (2.0f * static_cast<float>(plan.nFv)));
            for (int i = 0; i < 3; ++i) s.pos[i] = s.pos[i] + (f.force[i] * p.car->invMass) * c;
        }
        Rpost = model::rot_from_quat(s.quat);
    }
    TW_INLINE void Detect(const model::TwinState &s, SubstepContacts &out) {
        p.collision->DetectCached(s, cache, out);
        lastKeyCount = Keys(out, lastKeys);
    }
    // K(0): a detection at the tick-start pose (TwinState does not carry the
    // previous tick's last pass; the product can keep it per candidate instead).
    TW_INLINE void SpeculateBegin(const model::TwinState &s, TickContext &t) {
        if (startProbe.valid) {
            startKeyCount = Keys(startProbe.contacts, startKeys);
        } else {
            p.collision->DetectCached(s, cache, scratch);
            startKeyCount = Keys(scratch, startKeys);
        }
        savedRaceEv = raceEv;
        savedRespTick = respTick;
        savedTriggerHits = t.triggerHits;
    }
    TW_INLINE bool SpeculateChanged(const model::TwinState &, TickContext &) {
        if (lastKeyCount != startKeyCount) return true;
        for (int i = 0; i < lastKeyCount; ++i)
            if (lastKeys[i] != startKeys[i]) return true;
        return false;
    }
    TW_INLINE void SpeculateEnd(model::TwinState &, TickContext &t, bool keep) {
        if (keep) return;
        raceEv = savedRaceEv;
        respTick = savedRespTick;
        t.triggerHits = savedTriggerHits;
    }
    TW_INLINE void Respond(model::TwinState &s, const SubstepPose &pose, const SubstepContacts &c,
                           model::RespOut &out) {
        model::M3 Rw;
        std::memcpy(&Rw, pose.rot, sizeof Rw);
        model::RespondR(*p.car, s, Rpost, Rw, c.c, c.n, respTick, out);
    }
    TW_INLINE void RaceHits(model::TwinState &s, const SubstepContacts &c, const model::RespOut &r, TickContext &t) {
        race::TwTriggerHit hits[model::kRespMaxContacts];
        for (std::uint32_t i = 0; i < r.triggerCount; ++i) {
            const model::RespContact &ct = c.c[r.trigger[i]];
            hits[i].trigger = static_cast<std::uint16_t>(ct.peerId);
            hits[i].shape = ct.shape;
            hits[i].flags = 0u;
        }
        race::TwRaceOnTriggerHits(*p.race, s, hits, r.triggerCount, raceEv);
        t.triggerHits += r.triggerCount;
    }
    TW_INLINE void SubstepEnd(model::TwinState &, float) {}
    TW_INLINE void TickEnd(model::TwinState &s, TickContext &t, const FinishSubstep &finish) {
        t.checkpoints = raceEv.cpAccepted;
        t.cpLastSlot = raceEv.cpLastSlot;
        t.wheelContacts = respTick.wheelContacts;
        t.bodyContacts = respTick.bodyContacts;
        t.finishPassage = raceEv.finish != 0u;
        t.finished = raceEv.raceCompleted != 0u;
        if (!t.finished || !finish.found) return;
        t.finishTickMs = s.tickMs;
        // R4.2: rerun the completing substep from its start state with shortened steps
        TickContext probeTick = t;
        auto probe = [&](float partial) {
            model::TwinState q;
            std::memcpy(&q, &finish.start, sizeof q);
            TwinStages again(p, cache);
            RunSubstep(again, q, probeTick, partial, scratch);
            return q.raceCompleted != 0u;
        };
        race::TwFinishSubstep sub{s.tickMs, finish.startSeconds, finish.duration};
        std::uint64_t lower = 0u, upper = 0u;
        if (race::TwRefineFinish(sub, probe, &lower, &upper)) race::TwStoreFinish(s, lower);
    }
};

}  // namespace step
}  // namespace tw

#endif
