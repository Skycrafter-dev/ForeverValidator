// Twin batch runner (M10): persistent candidate lanes over TwinLanes<L>, with
// the tick contract of src/twin/step (stages.h, tick.h) per lane.
//
// Every lane runs one candidate: a start state at a tick start, a tick budget
// and a stage object of type LS bound to it (the per-candidate side of the
// tick contract: race, substep plan, collision, finish bracket). The runner
// owns forces, integration and the contact response, and runs them wide
// across lanes when enough lanes take part.
//
// One pass advances every busy lane by one substep:
//   refill        free lanes pull the next candidate (`next`, e.g. a shared atomic index)
//   tick begin    lanes not in a tick, on the lane's AoS copy: a fresh TickContext,
//                 LS::TickBegin (race transitions, respawns, controls -> t.in),
//                 the lane's RespTick reset, LS::Plan, LS::TrackFinish (finish snapshots
//                 this tick). Every tick is wide: n, split and remainder; a speculative
//                 one (plan.speculate, L7) also takes
//                 a per-lane snapshot (in SoA, one blend per row for all such lanes) and
//                 LS::SpeculateBegin, and is judged at its end: kept, or restored and rerun
//                 with nEvent from the tick start, wide, as tick.h's rerun.
//                 Hooks and state: the per-tick hooks see the AoS copy with LS::TickFields
//                 current (all fields if TickFields is null or LS::TickNeedsFullState());
//                 the speculation hooks see the tick fields plus pos, quat and absorb.
//   finish        tracked lanes (race not completed, finish not found yet): the substep-start
//                 state into an SoA snapshot (one blend per row); if the race completes in
//                 the substep it becomes the FinishSubstep (start, offset, duration), as tick.h
//   forces        S1 + C1..C20 \  wide: CarForcesL / CarIntegrateL on the wide lanes;
//   integrate     S2           /  narrow: CarForces / CarIntegrateR per lane (AoS);
//                                 then, if LS::Ballistic() (PolicyConfig::ballistic), a lane
//                                 whose tick runs one substep while FV's count is n > 1 adds
//                                 F/m dt^2 (n-1)/(2n) to pos (TwinStages::Integrate);
//                                 L8 (LS::Subcycle(), PolicyConfig::subcycle): a lane whose
//                                 airborne event-free tick runs one substep with FV's count n > 1
//                                 runs forces + integration n times at FV's split instead (wide:
//                                 the first micro-step with the other lanes, the rest masked),
//                                 its first CarForceOut is the response pose, no b1
//                                 (TwinStages::Forces / Integrate)
//   detect        S3a  Policy::detect on the pose rows (16 lanes) or pose views (pos, quat, absorb:
//                      what the collision reads), else LS::Detect per lane on a pose view
//   respond       S3b + S4  wide: RespondLanes on the lanes with contacts; narrow: RespondR per lane
//   race hits     LS::RaceHits for lanes whose response saw trigger contacts (AoS)
//   advance       next substep; at the tick's end LS::TickEnd and the evaluator (AoS)
// Bit identity: each candidate gives exactly RunAlone (step::RunTick with the
// same LS and the L = 1 kernels), whatever the width, the policy, the other
// lanes and the refill timing: the kernels are lane-pure, the conversions are
// lossless and per lane the stage sequence is tick.h's. A race that completes
// in a tick for which LS::TrackFinish said no has no finish snapshot: the lane's
// result status is UntrackedFinish (an LS whose TrackFinish is a heuristic must
// make it conservative; TwinLaneStages tracks every tick of an uncompleted race).
#ifndef TW_BATCH_BATCH_RUNNER_H
#define TW_BATCH_BATCH_RUNNER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include "forces.h"
#include "policy.h"
#include "response.h"
#include "response_lanes.h"
#include "stages.h"
#include "tick.h"
#include "tw_state.h"

namespace tw {
namespace batch {

// One candidate. `payload`: whatever the stages need (controls, recorded data).
struct Candidate {
    std::uint64_t id = 0u;
    model::TwinState start{};
    std::uint32_t ticks = 0u;        // tick budget
    std::uint32_t firstTick = 0u;    // absolute tick index of the candidate's first tick (evaluator alignment)
    const void *payload = nullptr;
};

enum class LaneStatus : std::uint8_t {
    Ok = 0,               // tick budget done
    Stopped = 1,          // the evaluator returned false
    UntrackedFinish = 2,  // the race completed in a wide tick (no finish snapshot): LS::TrackFinish too optimistic
};

// Read-only view of one lane's state, whichever form is current (AoS copy or
// TwinLanes column). Fields by TwinState member index (tw_state.h TwinFields()).
template <int L>
struct LaneRead {
    const model::TwinState *aos = nullptr;     // current AoS copy, or null
    const model::TwinLanes<L> *lanes = nullptr;
    int lane = 0;

    static int Field(const char *member) {     // index into TwinFields(), -1 if unknown
        std::size_t n = 0;
        const model::TwFieldInfo *f = model::TwinFields(&n);
        for (std::size_t i = 0; i < n; ++i)
            if (std::strcmp(f[i].name, member) == 0) return static_cast<int>(i);
        return -1;
    }
    std::uint32_t Word(int field, int k = 0) const;   // element k, zero / sign extended to 32 bits
    float Float(int field, int k = 0) const {
        const std::uint32_t w = Word(field, k);
        float x;
        std::memcpy(&x, &w, 4);
        return x;
    }
};

// Byte offset of each field in TwinLanes<L> (same X-macro order as TwinFields()).
template <int L>
inline const std::uint32_t *LaneFieldOffsets() {
    static const std::uint32_t off[] = {
#define TW_S(name, ty, golden, codec, life) static_cast<std::uint32_t>(offsetof(model::TwinLanes<L>, name)),
#define TW_A(name, ty, count, golden, codec, life) static_cast<std::uint32_t>(offsetof(model::TwinLanes<L>, name)),
#include "state_fields.inc"
#undef TW_S
#undef TW_A
    };
    return off;
}

template <int L>
inline std::uint32_t LaneRead<L>::Word(int field, int k) const {
    std::size_t n = 0;
    const model::TwFieldInfo &f = model::TwinFields(&n)[field];
    if (aos != nullptr) {
        std::uint32_t v = 0;
        std::memcpy(&v, reinterpret_cast<const unsigned char *>(aos) + f.offset + k * f.elementBytes, f.elementBytes);
        if (f.type == model::TwType::I8) v = static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int8_t>(v)));
        return v;
    }
    std::uint32_t v;
    std::memcpy(&v, reinterpret_cast<const unsigned char *>(lanes) + LaneFieldOffsets<L>()[field] + (k * L + lane) * 4u, 4);
    return v;
}

// Wide (one call for all lanes) when at least this many lanes take part, else narrow (per lane, L = 1).
// Optional batch collision (one call per pass for the wide lanes, instead of LS::Detect per lane).
// Contract, per lane l of `mask`: out[l] equals what the lane's CollisionStage::DetectCached
// gives for (*poses[l], caches[l]), bit for bit, with the same cache effects the stage relies on;
// the runner then calls LS::Detected(out[l]) for the LS's own bookkeeping (TwinStages: the
// speculation keys). poses[l] reads pos, quat and absorb only. One detector per runner (one world).
// DetectRows16 (16 lanes): the same contract with the poses as lane rows, TwinLanes<16>'s pos[3][16],
// quat[4][16] and absorb[4][16] (lane l of element k at [k][l]); false: not supported, the runner
// then hands DetectLanes per-lane pose views.
// Contact-free proofs (Policy::freeTicks): DetectRows16 sets bit l of *empty when lane l's detection had no
// candidate; then ProveRows16 (the same thread, before the next DetectRows16) returns the lanes of `ask` (a
// subset of those) whose shapes, swept along w from this pose and grown by d (sweep rows w x, w y, w z, d),
// meet no candidate either: every pose whose shape points all lie within d of the segment the same points
// sweep from this pose (less the runner's float margin, FreeHolds) has an empty detection. FreeLever: the
// largest |local shape centre| + ||local rotation||_2 x max radius, the runner's lever for the rotation;
// < 0: no proofs.
struct LaneDetect {
    virtual ~LaneDetect() = default;
    virtual void DetectLanes(int lanes, const model::TwinState *const *poses, std::uint32_t mask,
                             step::CollisionCache *const *caches, step::SubstepContacts *out) const = 0;
    virtual bool DetectRows16(const float (*pos)[16], const float (*quat)[16], const float (*absorb)[16], std::uint32_t mask,
                              step::CollisionCache *const *caches, step::SubstepContacts *out, std::uint32_t *empty = nullptr) const {
        (void)pos; (void)quat; (void)absorb; (void)mask; (void)caches; (void)out; (void)empty;
        return false;
    }
    virtual std::uint32_t ProveRows16(const float (*sweep)[16], std::uint32_t ask, step::CollisionCache *const *caches) const {
        (void)sweep; (void)ask; (void)caches;
        return 0u;
    }
    virtual float FreeLever() const { return -1.0f; }
};

struct Policy {
    int forcesMinLanes = 3;    // forces + integrate: lanes in a wide tick
    int respondMinLanes = 4;   // response: those of them with contacts this substep
    const LaneDetect *detect = nullptr;   // batch collision; null: LS::Detect per lane
    // Idea 5 (docs/twin/collision-ideas-x2.md), 16 lanes with a detector that proves (FreeLever() >= 0): a lane whose
    // detection has no candidate asks for a proof that the region its shapes sweep along v * (the rest of the tick +
    // freeTicksAhead ticks), grown by an allowance, holds no candidate either (DetectRows16); while its pose provably
    // stays inside that region (FreeHolds), across ticks, it skips the detection: that detection is empty, so the
    // result is the same. A failed proof is not asked again before the lane's next tick. freeCheck: detect anyway and
    // count the skippable substeps whose detection is not empty (must be 0). Measured 2026-11-11 (search queue, stage
    // c): 11% of lane substeps skipped, 0 violations, but +2-3% cycles per tick (the skipped detections are the cheap,
    // airborne ones, ~700 cycles; the proofs cost as much): off by default.
    bool freeTicks = false;
    bool freeCheck = false;
    float freeTicksAhead = 1.0f;
    float freeSlack = 1.25f;   // the allowance asked for: this times the expected deviation from the sweep
};

struct Stats {
    std::uint64_t passes = 0, laneSubsteps = 0, candidates = 0, ticks = 0;
    std::uint64_t trackedTicks = 0, speculativeTicks = 0, speculationRejects = 0, untrackedFinishes = 0;
    std::uint64_t forcesWide = 0, forcesNarrowLanes = 0, respondWide = 0, respondNarrowLanes = 0, raceHitLanes = 0;
    std::uint64_t toAos = 0, toLanes = 0;
    std::uint64_t ballisticLanes = 0;   // wide-tick substeps with the ballistic correction
    std::uint64_t speculationReruns = 0;   // rejected wide speculations rerun wide
    std::uint64_t subcycledLanes = 0;      // L8 substeps (lanes)
    std::uint64_t freeProofs = 0, freeProven = 0, freeSkipped = 0, freeEscapes = 0, freeViolations = 0;   // Policy::freeTicks
};

// The ballistic correction (substep-policy.md; twin_stages.h TwinStages::Integrate,
// same float ops): a tick run as one substep (h == dt) whose FV count is nFv > 1
// gets FV's n-substep forward-Euler position term for the force F.
TW_INLINE bool BallisticApplies(bool ballistic, std::uint32_t nFv, float h, float dt) {
    return ballistic && nFv > 1u && h == dt;
}
TW_INLINE float BallisticScale(std::uint32_t nFv, float h) {
    return (h * h) * (static_cast<float>(nFv - 1u) / (2.0f * static_cast<float>(nFv)));
}
// L8 subcycling (TwinStages::Forces): an airborne (nothing touched at the tick start), event-free
// tick run as one substep whose FV count is nFv > 1: forces + integration nFv times at FV's split.
TW_INLINE bool SubcycleTick(bool subcycle, const step::SubstepPlan &p) {
    return subcycle && p.nFv > 1u && !p.event && !p.startTouching;
}
TW_INLINE bool SubcycleApplies(bool subcycle, const step::SubstepPlan &p, float h, float dt) {
    return SubcycleTick(subcycle, p) && h == dt;
}

TW_INLINE model::M3 RotOfColumns(const float *c) {
    model::M3 m;
    m.c0 = model::V3{c[0], c[1], c[2]};
    m.c1 = model::V3{c[3], c[4], c[5]};
    m.c2 = model::V3{c[6], c[7], c[8]};
    return m;
}

// LS with the L = 1 kernels (b1 and L8 as TwinStages): the full TickStages contract
// for tick.h. Used by RunAlone (the reference).
template <class LS>
struct AloneStages {
    LS &ls;
    const model::CarForceConsts &k;
    const model::WaterZone *water;
    model::RespTick &respTick;
    model::M3 Rpost{};
    model::RespTick savedRespTick{};
    const step::SubstepPlan *plan = nullptr;   // set: Plan returns it (the runner planned already)
    step::SubstepPlan planned{};               // this tick's plan as Plan returned it (b1, L8; TwinStages::plan)
    std::uint32_t nFv = 1u;                    // this tick's FV count and duration (ballistic correction)
    float tickDt = step::kTickDt;
    bool subcycled = false;                    // L8 ran this substep's forces + integration

    AloneStages(LS &l, const model::CarForceConsts &kk, const model::WaterZone *w, model::RespTick &rt)
        : ls(l), k(kk), water(w), respTick(rt) {}

    TW_INLINE void TickBegin(model::TwinState &s, step::TickContext &t) {
        ls.TickBegin(s, t);
        std::memset(&respTick, 0, sizeof respTick);
    }
    TW_INLINE step::SubstepPlan Plan(const model::TwinState &s, step::TickContext &t) {
        const step::SubstepPlan p = plan != nullptr ? *plan : ls.Plan(s, t);
        planned = p;
        nFv = p.nFv;
        tickDt = t.dt;
        return p;
    }
    TW_INLINE void SubstepBegin(model::TwinState &, float) {}
    TW_INLINE void Forces(model::TwinState &s, const step::TickContext &t, float h, model::CarForceOut &out) {
        if (SubcycleApplies(ls.Subcycle(), planned, h, tickDt)) {   // L8 (TwinStages::Forces)
            bool first = true;
            step::ForEachSubstepDt(planned.nFv, h, [&](std::uint32_t, float hi) {
                model::CarForceOut o;
                model::CarForces(k, water, s, t.in, hi, o);
                model::CarIntegrateR(k, s, o, hi, RotOfColumns(o.poseRot), &Rpost);
                if (first) out = o;
                first = false;
            });
            subcycled = true;
            return;
        }
        model::CarForces(k, water, s, t.in, h, out);
    }
    TW_INLINE void Integrate(model::TwinState &s, const model::CarForceOut &f, float h) {
        if (subcycled) {   // L8 already integrated the tick
            subcycled = false;
            return;
        }
        model::CarIntegrateR(k, s, f, h, RotOfColumns(f.poseRot), &Rpost);
        if (BallisticApplies(ls.Ballistic(), nFv, h, tickDt)) {
            const float c = BallisticScale(nFv, h);
            for (int i = 0; i < 3; ++i) s.pos[i] = s.pos[i] + (f.force[i] * k.invMass) * c;
        }
    }
    TW_INLINE void Detect(const model::TwinState &s, step::SubstepContacts &out) { ls.Detect(s, out); }
    TW_INLINE void Respond(model::TwinState &s, const step::SubstepPose &pose, const step::SubstepContacts &c,
                           model::RespOut &out) {
        model::RespondR(k, s, Rpost, RotOfColumns(pose.rot), c.c, c.n, respTick, out);
    }
    TW_INLINE void RaceHits(model::TwinState &s, const step::SubstepContacts &c, const model::RespOut &r, step::TickContext &t) {
        ls.RaceHits(s, c, r, t);
    }
    TW_INLINE void SubstepEnd(model::TwinState &, float) {}
    TW_INLINE void SpeculateBegin(const model::TwinState &s, step::TickContext &t) {
        ls.SpeculateBegin(s, t);
        savedRespTick = respTick;
    }
    TW_INLINE bool SpeculateChanged(const model::TwinState &s, step::TickContext &t) { return ls.SpeculateChanged(s, t); }
    TW_INLINE void SpeculateEnd(model::TwinState &s, step::TickContext &t, bool keep) {
        ls.SpeculateEnd(s, t, keep);
        if (!keep) respTick = savedRespTick;
    }
    TW_INLINE void TickEnd(model::TwinState &s, step::TickContext &t, const step::FinishSubstep &finish) {
        ls.TickEnd(s, t, finish);
    }
};

// The reference: one candidate alone at L = 1 through tick.h. evaluate(tick index,
// state, TickContext) after every tick; false stops. Returns the substeps run.
template <class LS, class Evaluate>
inline std::uint64_t RunAlone(const model::CarForceConsts &k, const model::WaterZone *water, LS &ls, const Candidate &c,
                              model::TwinState &s, LaneStatus *status, Evaluate &&evaluate) {
    std::memcpy(&s, &c.start, sizeof s);
    ls.Bind(c);
    model::RespTick respTick{};
    AloneStages<LS> st(ls, k, water, respTick);
    std::uint64_t substeps = 0u;
    *status = LaneStatus::Ok;
    for (std::uint32_t tick = 0; tick < c.ticks; ++tick) {
        step::TickContext t;
        step::RunTick(st, s, 0u, t);
        substeps += t.substepsRun;
        if (!evaluate(tick, s, t)) {
            *status = LaneStatus::Stopped;
            break;
        }
    }
    return substeps;
}

template <int L, class LS>
class Runner {
public:
    Runner(const model::CarForceConsts &k, const model::WaterZone *water, const LS &prototype, Policy policy = Policy{})
        : k_(k), water_(water), policy_(policy) {
        // inactive lanes still go through the masked kernels: keep them a valid (zero) state
        std::memset(static_cast<void *>(&s_), 0, sizeof s_);
        if (policy_.freeTicks && policy_.detect != nullptr && L == 16) freeLever_ = policy_.detect->FreeLever();
        for (int l = 0; l < L; ++l) ls_[l] = prototype;
        for (int l = 0; l < L; ++l) cont_[l] = contacts_[l].c;
        if constexpr (HasWantsDetected<LS>::value) detected_ = prototype.WantsDetected();
        int n = 0;
        const char *const *names = TickFieldsOf(prototype, &n);
        if (names != nullptr) {   // per-tick hooks touch only these fields: partial AoS sync, one entry per element
            ntf_ = TickElems(names, n, te_, tfEnd_);
            // the hooks write back only LS::TickWriteFields (a subset of the tick fields), when declared
            int nw = 0;
            const char *const *wnames = TickWriteFieldsOf(&nw);
            bool subset = wnames != nullptr;
            for (int i = 0; i < nw && subset; ++i) {
                bool found = false;
                for (int j = 0; j < n && !found; ++j) found = std::strcmp(wnames[i], names[j]) == 0;
                subset = found;
            }
            ntfOut_ = subset ? TickElems(wnames, nw, teOut_, tfOutEnd_) : -1;
            if (ntf_ < 0 || ntfOut_ < 0) {   // write back every tick field
                ntfOut_ = ntf_;
                std::memcpy(teOut_, te_, sizeof te_);
                std::memcpy(tfOutEnd_, tfEnd_, sizeof tfEnd_);
            }
        }
    }

    const Stats &stats() const { return stats_; }
    // Refill only lanes [0, n) (1 <= n <= L): fewer candidates in flight, e.g. to spread a small
    // batch over several runners. No effect on any candidate's result.
    void SetLaneLimit(int n) { laneLimit_ = n < 1 ? 1 : n > L ? L : n; }

    // next(const Candidate *&) -> bool: the next candidate, false when none is left.
    // evaluate(const Candidate &, std::uint32_t tickIndex, const LaneRead<L> &, const step::TickContext &) -> bool:
    //   after every tick of every lane; false stops the candidate.
    // done(const Candidate &, const model::TwinState &end, std::uint32_t ticks, LaneStatus).
    template <class Next, class Evaluate, class Done>
    void Run(Next &&next, Evaluate &&evaluate, Done &&done) {
        for (int l = 0; l < L; ++l) lane_[l] = Lane{};
        busy_ = 0u;
        bool more = true;
        for (;;) {
            // ---- refill (the free lanes in order) ----
            const std::uint32_t limit = laneLimit_ >= 32 ? ~0u : (1u << laneLimit_) - 1u;
            for (std::uint32_t m = ~busy_ & limit; m != 0u && more; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                for (;;) {
                    const Candidate *c = nullptr;
                    if (!next(c)) {
                        more = false;
                        break;
                    }
                    if (c->ticks == 0u) {
                        done(*c, c->start, 0u, LaneStatus::Ok);
                        continue;
                    }
                    lane_[l] = Lane{};
                    lane_[l].c = c;
                    busy_ |= 1u << l;
                    free_ &= ~(1u << l);
                    std::memcpy(&aos_[l], &c->start, sizeof aos_[l]);
                    aosMask_ |= 1u << l;
                    tfSync_ &= ~(1u << l);
                    ls_[l].Bind(*c);
                    caches_[l] = ls_[l].DetectCache();
                    ++stats_.candidates;
                    break;
                }
            }
            const std::uint32_t busy = busy_;
            if (busy == 0u) break;
            ++stats_.passes;

            // ---- tick begin ----
            std::uint32_t wide = 0u, snapLanes = 0u;
            for (std::uint32_t bm = busy; bm != 0u; bm &= bm - 1u) {
                const int l = __builtin_ctz(bm);
                Lane &ln = lane_[l];
                if (!ln.inTick) {
                    ln.inTick = true;
                    freeBack_ &= ~(1u << l);
                    if (ntf_ < 0 || ls_[l].TickNeedsFullState()) ToAos(l);
                    else if (!InAos(l)) SyncTickIn(l);   // tick fields only; the lanes stay current
                    ctx_[l] = step::TickContext{};
                    ls_[l].TickBegin(aos_[l], ctx_[l]);
                    model::RespTick &rt = respTick_[l];
                    std::memset(&rt, 0, sizeof rt);
                    ctx_[l].plan = ls_[l].Plan(aos_[l], ctx_[l]);
                    const step::SubstepPlan &plan = ctx_[l].plan;
                    ln.track = ls_[l].TrackFinish(aos_[l], ctx_[l]);
                    if (ln.track) track_ |= 1u << l;
                    else track_ &= ~(1u << l);
                    if (!InAos(l)) SyncTickOut(l);   // the hooks' writes back into the lanes
                    if (ln.track) {   // finish snapshots at the substep starts, as tick.h (below)
                        finish_[l].found = false;
                        found_ &= ~(1u << l);
                        elapsed_[l] = 0.0;
                        ++stats_.trackedTicks;
                    }
                    spec_[l] = plan.speculate;
                    if (spec_[l]) {   // L7 speculation, wide: snapshot now, judge at the tick end
                        ++stats_.speculativeTicks;
                        snapAos_[l] = InAos(l);
                        if (snapAos_[l]) {
                            std::memcpy(&snapshot_[l], &aos_[l], sizeof snapshot_[l]);
                        } else {   // lanes current: the lane's column, copied below for all such lanes at once
                            snapLanes |= 1u << l;
                            PoseToAos(l);   // SpeculateBegin sees the tick fields and the pose
                        }
                        ls_[l].SpeculateBegin(aos_[l], ctx_[l]);
                        savedRespTick_[l] = rt;
                    }
                    // ballistic lanes of this tick (h == dt is checked per substep)
                    if (ls_[l].Ballistic() && plan.nFv > 1u) ballistic_ |= 1u << l;
                    else ballistic_ &= ~(1u << l);
                    // L8 lanes of this tick, from the plan as Plan returned it (as TwinStages::plan: a
                    // rejected speculation's rerun keeps it); h == dt is checked per substep
                    if (SubcycleTick(ls_[l].Subcycle(), plan)) subcycle_ |= 1u << l;
                    else subcycle_ &= ~(1u << l);
                    ln.n = plan.n;
                    ln.remaining = ctx_[l].dt;
                    ln.split = ln.n > 1u ? ctx_[l].dt / tw::math::from_u32(ln.n) : 0.0f;
                    in_.gas[l] = ctx_[l].in.gas;
                    in_.brake[l] = ctx_[l].in.brake;
                    in_.steer[l] = ctx_[l].in.steer;
                    in_.tickMs[l] = ctx_[l].in.tickMs;
                    wasCompleted_[l] = RaceCompleted(l);
                    tick_.wheelContacts[l] = 0u;
                    tick_.bodyContacts[l] = 0u;
                    tick_.bodyNormalSum[0][l] = tick_.bodyNormalSum[1][l] = tick_.bodyNormalSum[2][l] = 0.0f;
                }
                h_[l] = ln.sub + 1u < ln.n ? ln.split : ln.remaining;
                wide |= 1u << l;
            }
            if (snapLanes != 0u) Blend(&snapLanes_, snapLanes);
            if (wide == 0u) continue;
            const int nWide = __builtin_popcount(wide);

            // ---- finish tracking (tick.h RunSubsteps): the substep-start state of every tracked lane
            // whose race is not completed and whose finish substep is not found yet ----
            std::uint32_t trackNow = wide & track_ & ~found_;
            if (trackNow != 0u) trackNow &= ~CompletedMask(trackNow);
            if (trackNow != 0u) {
                std::uint32_t soa = 0u;
                for (std::uint32_t m = trackNow; m != 0u; m &= m - 1u) {
                    const int l = __builtin_ctz(m);
                    if (InAos(l)) std::memcpy(&finish_[l].start, &aos_[l], sizeof aos_[l]);
                    else soa |= 1u << l;
                }
                finSoa_ = (finSoa_ & ~trackNow) | soa;
                if (soa != 0u) Blend(&finLanes_, soa);
            }

            // ---- forces + integrate ----
            if (nWide >= policy_.forcesMinLanes) {
                ToLanes(wide);
                std::uint32_t sc = 0u;   // L8 lanes this substep: the first micro-step runs with the others
                for (std::uint32_t m = wide & subcycle_; m != 0u; m &= m - 1u) {
                    const int l = __builtin_ctz(m);
                    if (h_[l] == ctx_[l].dt) sc |= 1u << l;
                }
                const float(&h0)[L] = sc != 0u ? MicroFirst(sc) : h_;
                model::CarForcesL<L>(k_, water_, s_, in_, h0, fout_, wide);
                model::CarIntegrateL<L>(k_, s_, fout_, h0, wide, postRot_);
                if (sc != 0u) MicroRest(sc);
                for (std::uint32_t m = wide & ballistic_ & ~sc; m != 0u; m &= m - 1u) {
                    const int l = __builtin_ctz(m);
                    if (!(h_[l] == ctx_[l].dt)) continue;
                    const float c = BallisticScale(ctx_[l].plan.nFv, h_[l]);
                    for (int i = 0; i < 3; ++i) s_.pos[i][l] = s_.pos[i][l] + (fout_.force[i][l] * k_.invMass) * c;
                    ++stats_.ballisticLanes;
                }
                ++stats_.forcesWide;
            } else {
                ForcesNarrow(wide);
            }

            tfSync_ &= ~wide;   // the substep changes the state
            // ---- detect (pose rows, or pose views: pos, quat, absorb) ----
            std::uint32_t touching = 0u;
            if (!(policy_.detect != nullptr && DetectRows(wide))) {
                for (int l = 0; l < L; ++l) {
                    if (!((wide >> l) & 1u)) continue;
                    const model::TwinState *pose = &aos_[l];
                    if (!InAos(l)) {
                        model::TwinState &v = view_[l];
                        for (int a = 0; a < 3; ++a) v.pos[a] = s_.pos[a][l];
                        for (int a = 0; a < 4; ++a) v.quat[a] = s_.quat[a][l];
                        for (int w = 0; w < 4; ++w) v.absorb[w] = s_.absorb[w][l];
                        pose = &v;
                    }
                    poses_[l] = pose;
                    if (policy_.detect == nullptr) ls_[l].Detect(*pose, contacts_[l]);
                }
                if (policy_.detect != nullptr) policy_.detect->DetectLanes(L, poses_, wide, caches_, contacts_);
            }
            if (policy_.detect != nullptr && detected_)
                for (std::uint32_t m = wide; m != 0u; m &= m - 1u) {
                    const int l = __builtin_ctz(m);
                    ls_[l].Detected(contacts_[l]);
                }
            for (std::uint32_t m = wide; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                ncont_[l] = contacts_[l].n;   // (cont_[l] is contacts_[l].c)
                if (ncont_[l] > 0) touching |= 1u << l;
            }

            // ---- respond ----
            std::uint32_t hits = 0u;
            if (touching != 0u) {
                if (__builtin_popcount(touching) >= policy_.respondMinLanes) {
                    ToLanes(touching);
                    model::RespondLanes<L>(k_, s_, postRot_, fout_.poseRot, touching, cont_, ncont_, tick_, out_);
                    ++stats_.respondWide;
                } else {
                    RespondNarrow(touching);
                }
                for (std::uint32_t m = touching; m != 0u; m &= m - 1u)
                    if (out_[__builtin_ctz(m)].triggerCount != 0u) hits |= 1u << __builtin_ctz(m);
            }

            // ---- race hits ----
            for (std::uint32_t m = hits; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                ToAos(l);
                ls_[l].RaceHits(aos_[l], contacts_[l], out_[l], ctx_[l]);
                ++stats_.raceHitLanes;
            }

            for (std::uint32_t m = trackNow != 0u ? CompletedMask(trackNow) : 0u; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                step::FinishSubstep &f = finish_[l];
                f.found = true;
                found_ |= 1u << l;
                f.startSeconds = elapsed_[l];
                f.duration = h_[l];
                if ((finSoa_ >> l) & 1u) {   // padding bytes as the lane's own copy (the candidate's start)
                    std::memcpy(&f.start, &lane_[l].c->start, sizeof f.start);
                    model::TwinStoreLane(finLanes_, l, &f.start);
                }
            }
            for (std::uint32_t m = wide & track_; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                elapsed_[l] += static_cast<double>(h_[l]);
            }

            // ---- advance ----
            stats_.laneSubsteps += static_cast<std::uint64_t>(nWide);
            for (std::uint32_t m = wide; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                Lane &ln = lane_[l];
                ++ctx_[l].substepsRun;
                if (ln.sub + 1u < ln.n) {
                    ln.remaining = ln.remaining - ln.split;
                    ++ln.sub;
                    continue;
                }
                ln.sub = 0u;
                if (ntf_ < 0) ToAos(l);
                if (!ln.track && !wasCompleted_[l] && RaceCompleted(l)) {   // completed without a snapshot
                    ln.untracked = true;
                    ++stats_.untrackedFinishes;
                }
                respTickAos_[l] = model::RespTick{tick_.wheelContacts[l], tick_.bodyContacts[l],
                                                  {tick_.bodyNormalSum[0][l], tick_.bodyNormalSum[1][l],
                                                   tick_.bodyNormalSum[2][l]}};
                if (spec_[l]) {
                    spec_[l] = false;
                    // the speculation hooks see the tick fields and the pose (full state: ntf_ < 0)
                    if (!InAos(l)) {
                        SyncTickIn(l);
                        PoseToAos(l);
                    }
                    if (!ls_[l].SpeculateChanged(aos_[l], ctx_[l])) {
                        // kept; EndTick follows on the same AoS copy (its SyncTickIn is skipped: tfSync_) and
                        // writes the tick fields back once after TickEnd
                        ls_[l].SpeculateEnd(aos_[l], ctx_[l], true);
                    } else {   // tick.h: restore, undo the stages' side state, rerun with nEvent (wide)
                        ++stats_.speculationRejects;
                        if (snapAos_[l]) {
                            std::memcpy(&aos_[l], &snapshot_[l], sizeof aos_[l]);
                            aosMask_ |= 1u << l;
                        } else {
                            RestoreLane(l);
                            aosMask_ &= ~(1u << l);
                            SyncTickIn(l);
                            PoseToAos(l);
                        }
                        ls_[l].SpeculateEnd(aos_[l], ctx_[l], false);
                        if (!InAos(l)) SyncTickOut(l);
                        respTick_[l] = savedRespTick_[l];
                        ln.untracked = false;
                        finish_[l].found = false;   // tick.h: the rerun tracks the finish again
                        found_ &= ~(1u << l);
                        elapsed_[l] = 0.0;
                        ctx_[l].substepsWasted += ctx_[l].plan.n;
                        ctx_[l].plan.event = true;
                        ctx_[l].plan.n = ctx_[l].plan.nEvent;   // plan.speculate stays set, as in tick.h
                        // the rerun: same tick, nEvent substeps from the tick start, in the next passes
                        ln.n = ctx_[l].plan.n;
                        ln.remaining = ctx_[l].dt;
                        ln.split = ln.n > 1u ? ctx_[l].dt / tw::math::from_u32(ln.n) : 0.0f;
                        tick_.wheelContacts[l] = 0u;
                        tick_.bodyContacts[l] = 0u;
                        tick_.bodyNormalSum[0][l] = tick_.bodyNormalSum[1][l] = tick_.bodyNormalSum[2][l] = 0.0f;
                        wasCompleted_[l] = RaceCompleted(l);
                        ++stats_.speculationReruns;
                        continue;
                    }
                }
                if (ln.track && finish_[l].found) {   // finish tick: TickEnd (R4.2, TwStoreFinish) on the full state
                    if (!InAos(l) && ((tfSync_ >> l) & 1u)) SyncTickOut(l);   // the speculation hooks' writes
                    ToAos(l);
                }
                EndTick(l, evaluate, done, ln.track ? &finish_[l] : nullptr);
            }
        }
    }

    // The per-tick response counters of lane l (rules snapshot input), valid in the evaluator.
    const model::RespTick &LaneRespTick(int l) const { return respTickAos_[l]; }

private:
    struct Lane {
        const Candidate *c = nullptr;
        std::uint32_t tick = 0u, sub = 0u, n = 1u;
        float split = 0.0f, remaining = 0.0f;
        bool untracked = false;
        bool inTick = false;   // tick begun (a rejected speculation reruns with sub = 0, inTick set)
        bool track = false;    // LS::TrackFinish: finish snapshots at the substep starts this tick
    };

    // Tick end of lane l (AoS current): LS::TickEnd, the evaluator, budget. True if the lane was freed.
    template <class Evaluate, class Done>
    bool EndTick(int l, Evaluate &evaluate, Done &done, const step::FinishSubstep *finish) {
        Lane &ln = lane_[l];
        static const step::FinishSubstep kNone{};
        const bool partial = !InAos(l);   // lanes current: TickEnd sees the tick fields only
        // (an LS whose partial TickEnd touches no state: no sync, unless hooks wrote the AoS copy since the last
        // SyncTickIn (a kept speculation): their writes go back)
        constexpr bool kSync = !TickEndPartialPure<LS>::value;
        if (partial && kSync) SyncTickIn(l);
        ls_[l].TickEnd(aos_[l], ctx_[l], finish != nullptr ? *finish : kNone);
        if (partial && (kSync || ((tfSync_ >> l) & 1u))) SyncTickOut(l);
        ++stats_.ticks;
        ln.inTick = false;
        const std::uint32_t tickIndex = ln.tick++;
        LaneRead<L> view;
        view.aos = partial ? nullptr : &aos_[l];
        view.lanes = &s_;
        view.lane = l;
        const bool go = evaluate(*ln.c, tickIndex, view, ctx_[l]);
        if (go && !ln.untracked && ln.tick < ln.c->ticks) return false;
        const LaneStatus status = ln.untracked ? LaneStatus::UntrackedFinish : go ? LaneStatus::Ok : LaneStatus::Stopped;
        ToAos(l);
        done(*ln.c, aos_[l], ln.tick, status);
        lane_[l] = Lane{};
        aosMask_ &= ~(1u << l);
        busy_ &= ~(1u << l);
        return true;
    }

    struct TickElem {
        std::uint32_t lane;    // byte offset of element row (lane 0) in TwinLanes<L>
        std::uint32_t aos;     // byte offset in TwinState
    };
    static constexpr int kMaxTickElems = 96;

    bool InAos(int l) const { return ((aosMask_ >> l) & 1u) != 0u; }
    bool RaceCompleted(int l) const { return InAos(l) ? aos_[l].raceCompleted != 0u : s_.raceCompleted[0][l] != 0u; }
    // the lanes of m whose race is completed (RaceCompleted, all lanes' rows at once)
    std::uint32_t CompletedMask(std::uint32_t m) const {
        using F = tw::math::vf<L>;
        using I = model::LaneI<F>;
        std::uint32_t c = model::MaskBits<F>(model::IGt(I::load(reinterpret_cast<const std::int32_t *>(s_.raceCompleted[0])), I(0))) &
                          m & ~aosMask_;
        for (std::uint32_t a = m & aosMask_; a != 0u; a &= a - 1u) {
            const int l = __builtin_ctz(a);
            if (aos_[l].raceCompleted != 0u) c |= 1u << l;
        }
        return c;
    }
    // Optional LS declaration: bool WantsDetected() const: false when Detected does nothing (default true)
    template <class T, class = void> struct HasWantsDetected : std::false_type {};
    template <class T>
    struct HasWantsDetected<T, std::void_t<decltype(std::declval<const T &>().WantsDetected())>> : std::true_type {};

    // Optional LS declarations (absent: the defaults below).
    //   const char *const *TickFieldsFor(int *n) const: this LS instance's tick fields (default: static TickFields)
    //   static const char *const *TickWriteFields(int *n): the tick fields the per-tick hooks may write on a
    //       partial-state tick (default: all of them); the rest are written back unchanged, so not at all
    //   static constexpr bool kTickEndPartialPure: TickEnd on a partial-state tick (no full-state finish) neither
    //       reads nor writes the state (default false: the tick fields are synced around it)
    template <class T, class = void> struct HasTickFieldsFor : std::false_type {};
    template <class T>
    struct HasTickFieldsFor<T, std::void_t<decltype(std::declval<const T &>().TickFieldsFor(std::declval<int *>()))>>
            : std::true_type {};
    template <class T, class = void> struct HasTickWriteFields : std::false_type {};
    template <class T>
    struct HasTickWriteFields<T, std::void_t<decltype(T::TickWriteFields(std::declval<int *>()))>> : std::true_type {};
    template <class T, class = void> struct TickEndPartialPure : std::false_type {};
    template <class T>
    struct TickEndPartialPure<T, std::void_t<decltype(T::kTickEndPartialPure)>> : std::integral_constant<bool, T::kTickEndPartialPure> {};
    static const char *const *TickFieldsOf(const LS &ls, int *n) {
        if constexpr (HasTickFieldsFor<LS>::value) return ls.TickFieldsFor(n);
        else return LS::TickFields(n);
    }
    static const char *const *TickWriteFieldsOf(int *n) {
        if constexpr (HasTickWriteFields<LS>::value) return LS::TickWriteFields(n);
        else {
            *n = 0;
            return nullptr;
        }
    }
    // the elements of the named fields, grouped by element kind (4-byte, 2-byte, 1-byte unsigned, 1-byte signed):
    // branch-free copies; an unknown name or too many elements: -1 (full conversions instead)
    static int TickElems(const char *const *names, int n, TickElem *te, int (&end)[4]) {
        std::size_t nf = 0;
        const model::TwFieldInfo *fi = model::TwinFields(&nf);
        const std::uint32_t *lo = LaneFieldOffsets<L>();
        int count = 0;
        for (int kind = 0; kind < 4; ++kind) {
            for (int i = 0; i < n; ++i) {
                const int f = LaneRead<L>::Field(names[i]);
                if (f < 0) return -1;
                const bool sign = fi[f].type == model::TwType::I8;
                const int k = fi[f].elementBytes == 4u ? 0 : fi[f].elementBytes == 2u ? 1 : sign ? 3 : 2;
                if (k != kind) continue;
                for (int e = 0; e < fi[f].count; ++e) {
                    if (count == kMaxTickElems) return -1;
                    TickElem &t = te[count++];
                    t.aos = static_cast<std::uint16_t>(fi[f].offset + e * fi[f].elementBytes);
                    t.lane = lo[f] + static_cast<std::uint32_t>(e * L) * 4u;
                }
            }
            end[kind] = count;
        }
        return count;
    }

    // the declared tick fields between the lanes and the AoS copy (TwinLoadLane / TwinStoreLane casts)
    // (lane words hold the AoS value zero / sign extended, little-endian: the low bytes are the value)
    void SyncTickIn(int l) {
        if ((tfSync_ >> l) & 1u) return;   // nothing changed either copy since the last sync
        unsigned char *a = reinterpret_cast<unsigned char *>(&aos_[l]);
        const unsigned char *g = reinterpret_cast<const unsigned char *>(&s_) + static_cast<std::size_t>(l) * 4u;
        int i = 0;
        for (; i < tfEnd_[0]; ++i) std::memcpy(a + te_[i].aos, g + te_[i].lane, 4);
        for (; i < tfEnd_[1]; ++i) std::memcpy(a + te_[i].aos, g + te_[i].lane, 2);
        for (; i < ntf_; ++i) a[te_[i].aos] = g[te_[i].lane];
        tfSync_ |= 1u << l;
    }
    void SyncTickOut(int l) {   // after hooks that may have written the tick fields
        const unsigned char *a = reinterpret_cast<const unsigned char *>(&aos_[l]);
        unsigned char *g = reinterpret_cast<unsigned char *>(&s_) + static_cast<std::size_t>(l) * 4u;
        int i = 0;   // the fields the hooks may write (the others are as SyncTickIn read them)
        const TickElem *te = teOut_;
        for (; i < tfOutEnd_[0]; ++i) std::memcpy(g + te[i].lane, a + te[i].aos, 4);
        for (; i < tfOutEnd_[1]; ++i) {
            std::uint16_t v;
            std::memcpy(&v, a + te[i].aos, 2);
            const std::uint32_t w = v;
            std::memcpy(g + te[i].lane, &w, 4);
        }
        for (; i < tfOutEnd_[2]; ++i) {
            const std::uint32_t w = a[te[i].aos];
            std::memcpy(g + te[i].lane, &w, 4);
        }
        for (; i < ntfOut_; ++i) {
            const std::uint32_t w = static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int8_t>(a[te[i].aos])));
            std::memcpy(g + te[i].lane, &w, 4);
        }
        tfSync_ |= 1u << l;
    }

    // The narrow paths, out of Run's hot code (few lanes): forces + integration per lane (AoS), rare; the
    // response per touching lane.
    TW_NOINLINE_COLD void ForcesNarrow(std::uint32_t wide) {
        for (int l = 0; l < L; ++l) {
            if (!((wide >> l) & 1u)) continue;
            ToAos(l);
            model::CarForceOut f;
            model::M3 post;
            if (((subcycle_ >> l) & 1u) && h_[l] == ctx_[l].dt) {   // L8 (TwinStages::Forces)
                bool first = true;
                step::ForEachSubstepDt(ctx_[l].plan.nFv, h_[l], [&](std::uint32_t, float hi) {
                    model::CarForceOut o;
                    model::CarForces(k_, water_, aos_[l], ctx_[l].in, hi, o);
                    model::CarIntegrateR(k_, aos_[l], o, hi, RotOfColumns(o.poseRot), &post);
                    if (first) f = o;
                    first = false;
                });
                ++stats_.subcycledLanes;
            } else {
                model::CarForces(k_, water_, aos_[l], ctx_[l].in, h_[l], f);
                model::CarIntegrateR(k_, aos_[l], f, h_[l], RotOfColumns(f.poseRot), &post);
            }
            if (((ballistic_ >> l) & 1u) && !((subcycle_ >> l) & 1u) && h_[l] == ctx_[l].dt) {   // no b1 on L8
                const float c = BallisticScale(ctx_[l].plan.nFv, h_[l]);
                for (int i = 0; i < 3; ++i) aos_[l].pos[i] = aos_[l].pos[i] + (f.force[i] * k_.invMass) * c;
                ++stats_.ballisticLanes;
            }
            for (int i = 0; i < 9; ++i) fout_.poseRot[i][l] = f.poseRot[i];
            for (int i = 0; i < 3; ++i) fout_.posePos[i][l] = f.posePos[i];
            const float pc[9] = {post.c0.x, post.c0.y, post.c0.z, post.c1.x, post.c1.y, post.c1.z,
                                 post.c2.x, post.c2.y, post.c2.z};
            for (int i = 0; i < 9; ++i) postRot_[i][l] = pc[i];
            ++stats_.forcesNarrowLanes;
        }
    }
    TW_NOINLINE void RespondNarrow(std::uint32_t touching) {
        for (std::uint32_t m = touching; m != 0u; m &= m - 1u) {
            const int l = __builtin_ctz(m);
            ToAos(l);
            model::RespTick t{tick_.wheelContacts[l], tick_.bodyContacts[l],
                              {tick_.bodyNormalSum[0][l], tick_.bodyNormalSum[1][l], tick_.bodyNormalSum[2][l]}};
            float rc[9], wc[9];
            for (int i = 0; i < 9; ++i) {
                rc[i] = postRot_[i][l];
                wc[i] = fout_.poseRot[i][l];
            }
            model::RespondR(k_, aos_[l], RotOfColumns(rc), RotOfColumns(wc), cont_[l], ncont_[l], t, out_[l]);
            tick_.wheelContacts[l] = t.wheelContacts;
            tick_.bodyContacts[l] = t.bodyContacts;
            for (int a = 0; a < 3; ++a) tick_.bodyNormalSum[a][l] = t.bodyNormalSum[a];
            ++stats_.respondNarrowLanes;
        }
    }

    // L8 micro-steps (ForEachSubstepDt(nFv, h) per lane of `sc`): the first one's durations (the other
    // lanes keep h_), then the rest, masked, into separate outputs (CarForcesL / CarIntegrateL store
    // every lane): the lanes' final rotation goes to postRot_, fout_ keeps the first CarForceOut.
    const float (&MicroFirst(std::uint32_t sc))[L] {
        for (int l = 0; l < L; ++l) hMicro_[l] = h_[l];
        for (std::uint32_t m = sc; m != 0u; m &= m - 1u) {
            const int l = __builtin_ctz(m);
            const std::uint32_t n = ctx_[l].plan.nFv;
            microSplit_[l] = h_[l] / tw::math::from_u32(n);
            microRem_[l] = h_[l] - microSplit_[l];
            hMicro_[l] = microSplit_[l];
        }
        return hMicro_;
    }
    void MicroRest(std::uint32_t sc) {
        std::uint32_t nmax = 0u;
        for (std::uint32_t m = sc; m != 0u; m &= m - 1u) nmax = std::max(nmax, ctx_[__builtin_ctz(m)].plan.nFv);
        for (std::uint32_t j = 1u; j < nmax; ++j) {
            std::uint32_t act = 0u;
            for (std::uint32_t m = sc; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                const std::uint32_t n = ctx_[l].plan.nFv;
                if (j >= n) continue;
                act |= 1u << l;
                if (j + 1u < n) {
                    hMicro_[l] = microSplit_[l];
                    microRem_[l] = microRem_[l] - microSplit_[l];
                } else {
                    hMicro_[l] = microRem_[l];
                }
            }
            model::CarForcesL<L>(k_, water_, s_, in_, hMicro_, foutMicro_, act);
            model::CarIntegrateL<L>(k_, s_, foutMicro_, hMicro_, act, postMicro_);
            using F = tw::math::vf<L>;
            const auto mk = model::MaskOf<F>(act);
            for (int i = 0; i < 9; ++i) select(mk, F::load(postMicro_[i]), F::load(postRot_[i])).store(postRot_[i]);
        }
        stats_.subcycledLanes += static_cast<std::uint64_t>(__builtin_popcount(sc));
    }

    // Policy::detect with the poses as rows (L = 16): the lanes' own rows, or, when a wide lane is in its AoS copy
    // (narrow forces), a copy of them with those lanes' poses. False: the detector takes pose views only.
    bool DetectRows(std::uint32_t wide) {
        if constexpr (L == 16) {
            const float(*pos)[16] = s_.pos;
            const float(*quat)[16] = s_.quat;
            const float(*absorb)[16] = s_.absorb;
            const std::uint32_t aos = wide & aosMask_;   // (caches_: the lanes' DetectCache, set at Bind)
            if (aos != 0u) {
                std::memcpy(rowPos_, s_.pos, sizeof rowPos_);
                std::memcpy(rowQuat_, s_.quat, sizeof rowQuat_);
                std::memcpy(rowAbsorb_, s_.absorb, sizeof rowAbsorb_);
                for (std::uint32_t m = aos; m != 0u; m &= m - 1u) {
                    const int l = __builtin_ctz(m);
                    for (int a = 0; a < 3; ++a) rowPos_[a][l] = aos_[l].pos[a];
                    for (int a = 0; a < 4; ++a) rowQuat_[a][l] = aos_[l].quat[a];
                    for (int w = 0; w < 4; ++w) rowAbsorb_[w][l] = aos_[l].absorb[w];
                }
                pos = rowPos_;
                quat = rowQuat_;
                absorb = rowAbsorb_;
            }
            if (!policy_.freeTicks || freeLever_ < 0.0f) return policy_.detect->DetectRows16(pos, quat, absorb, wide, caches_, contacts_);
            // Idea 5: lanes holding a proof whose pose is still inside its region skip the detection (freeCheck: detect
            // them anyway and count non-empty results); detected lanes without a candidate ask for one, unless their
            // last try this tick failed
            std::uint32_t det = wide, skip = 0u;
            for (std::uint32_t m = wide & free_; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                if (FreeHolds(l, pos, quat, absorb)) {
                    skip |= 1u << l;
                } else {
                    free_ &= ~(1u << l);
                    ++stats_.freeEscapes;
                }
            }
            if (!policy_.freeCheck) det &= ~skip;
            std::uint32_t empty = 0u;
            if (det != 0u && !policy_.detect->DetectRows16(pos, quat, absorb, det, caches_, contacts_, &empty)) {
                free_ = 0u;
                return false;
            }
            for (std::uint32_t m = skip; m != 0u; m &= m - 1u) {
                const int l = __builtin_ctz(m);
                ++stats_.freeSkipped;
                if (policy_.freeCheck) {
                    if (contacts_[l].n != 0 || contacts_[l].overflow) ++stats_.freeViolations;
                    continue;
                }
                contacts_[l].n = 0;
                contacts_[l].overflow = false;
            }
            const std::uint32_t ask = empty & det & ~free_ & ~freeBack_;
            if (ask == 0u) return true;
            for (std::uint32_t m = ask; m != 0u; m &= m - 1u) FreeSweep(__builtin_ctz(m), pos);
            const std::uint32_t proven = policy_.detect->ProveRows16(sweep_, ask, caches_) & ask;
            for (std::uint32_t m = proven; m != 0u; m &= m - 1u) FreeAnchor(__builtin_ctz(m), pos, quat, absorb);
            free_ |= proven;
            freeBack_ |= ask & ~proven;
            stats_.freeProofs += static_cast<std::uint64_t>(__builtin_popcount(ask));
            stats_.freeProven += static_cast<std::uint64_t>(__builtin_popcount(proven));
            return true;
        } else {
            (void)wide;
            return false;
        }
    }

    // Idea 5. The region lane l asks for: the sweep w = v t (t: the rest of the tick + freeTicksAhead ticks) and the
    // allowance d around it: a guess from the gravity and the spin (only how often a proof holds depends on it, never
    // the result), plus the float margin FreeHolds keeps.
    void FreeSweep(int l, const float (*pos)[16]) {
        float pm = 0.0f, am = 0.0f, w2 = 0.0f, ww = 0.0f, w1 = 0.0f;
        const float t = std::max(0.0f, lane_[l].remaining - h_[l]) + policy_.freeTicksAhead * ctx_[l].dt;
        for (int a = 0; a < 3; ++a) {
            pm = std::max(pm, std::fabs(pos[a][l]));
            const float v = InAos(l) ? aos_[l].vel[a] : s_.vel[a][l], w = InAos(l) ? aos_[l].angVel[a] : s_.angVel[a][l];
            freeW_[l][a] = v * t;
            sweep_[a][l] = freeW_[l][a];
            ww += freeW_[l][a] * freeW_[l][a];
            w1 += std::fabs(freeW_[l][a]);
            w2 += w * w;
        }
        for (int w = 0; w < 4; ++w) am = std::max(am, std::fabs(InAos(l) ? aos_[l].absorb[w] : s_.absorb[w][l]));
        freeWW_[l] = ww;
        freeLeverL_[l] = freeLever_ + am;
        // (FreeHolds measures the rotation by ||dR||_F, about sqrt(2) |omega| t)
        const float guess = policy_.freeSlack * (20.0f * t * t + 1.4143f * std::sqrt(w2) * t * freeLeverL_[l]) + 0.005f;
        // float rounding of the shape boxes, the planes and this bound; tol's change with the pose (~1e-5 / m)
        freeMargin_[l] = 2e-3f + 8e-6f * (pm + w1 + guess) + 1e-4f * (w1 + guess);
        freeD_[l] = guess + freeMargin_[l];
        sweep_[3][l] = freeD_[l];
    }
    // The anchor of lane l's proof: this pass's pose (its sweep and allowance from FreeSweep).
    void FreeAnchor(int l, const float (*pos)[16], const float (*quat)[16], const float (*absorb)[16]) {
        for (int a = 0; a < 3; ++a) freePos_[l][a] = pos[a][l];
        const float q[4] = {quat[0][l], quat[1][l], quat[2][l], quat[3][l]};
        const model::M3 R = model::rot_from_quat(q);
        const float r[9] = {R.c0.x, R.c0.y, R.c0.z, R.c1.x, R.c1.y, R.c1.z, R.c2.x, R.c2.y, R.c2.z};
        for (int i = 0; i < 9; ++i) freeR_[l][i] = r[i];
        for (int w = 0; w < 4; ++w) freeAbs_[l][w] = absorb[w][l];
    }
    // Every point of every shape of lane l's pose is within freeD_ - freeMargin_ of the segment its anchor point sweeps
    // along freeW_: dev + ||dR||_F (|local centre| + |absorb| + ||local R||_F rmax) + ||R||_F |dabsorb|, dev the distance
    // of dpos from the segment (shape point x = pos + R (c_k - absorb e_y) + R R_k diag(r) s, |s| <= 1). NaN: false.
    bool FreeHolds(int l, const float (*pos)[16], const float (*quat)[16], const float (*absorb)[16]) const {
        float dp[3], dw = 0.0f, dr2 = 0.0f, rf2 = 0.0f, da = 0.0f;
        for (int a = 0; a < 3; ++a) {
            dp[a] = pos[a][l] - freePos_[l][a];
            dw += dp[a] * freeW_[l][a];
        }
        float sg = freeWW_[l] > 0.0f ? dw / freeWW_[l] : 0.0f;
        sg = sg < 0.0f ? 0.0f : sg > 1.0f ? 1.0f : sg;
        float dev2 = 0.0f;
        for (int a = 0; a < 3; ++a) {
            const float e = dp[a] - sg * freeW_[l][a];
            dev2 += e * e;
        }
        const float q[4] = {quat[0][l], quat[1][l], quat[2][l], quat[3][l]};
        const model::M3 R = model::rot_from_quat(q);
        const float r[9] = {R.c0.x, R.c0.y, R.c0.z, R.c1.x, R.c1.y, R.c1.z, R.c2.x, R.c2.y, R.c2.z};
        for (int i = 0; i < 9; ++i) {
            const float d = r[i] - freeR_[l][i];
            dr2 += d * d;
            rf2 += r[i] * r[i];
        }
        for (int w = 0; w < 4; ++w) da = std::max(da, std::fabs(absorb[w][l] - freeAbs_[l][w]));
        const float d = (std::sqrt(dev2) + std::sqrt(dr2) * freeLeverL_[l] + std::sqrt(rf2) * da) * 1.0001f;
        return d + freeMargin_[l] <= freeD_[l];
    }

    // pos, quat, absorb of lane l into its AoS copy (the collision's inputs)
    void PoseToAos(int l) {
        for (int a = 0; a < 3; ++a) aos_[l].pos[a] = s_.pos[a][l];
        for (int a = 0; a < 4; ++a) aos_[l].quat[a] = s_.quat[a][l];
        for (int w = 0; w < 4; ++w) aos_[l].absorb[w] = s_.absorb[w][l];
    }
    // Snapshots in SoA (speculation: snapLanes_, finish tracking: finLanes_): the columns of `mask`
    // copied in (one blend per row for all of them), and one lane's column back.
    static_assert(sizeof(model::TwinLanes<L>) % (4u * L) == 0u, "TwinLanes<L>: whole rows of L words");
    static constexpr std::size_t kRows = sizeof(model::TwinLanes<L>) / (4u * L);
    void Blend(model::TwinLanes<L> *into, std::uint32_t mask) {
        using F = tw::math::vf<L>;
        using I = tw::math::vi<L>;
        const std::int32_t *src = reinterpret_cast<const std::int32_t *>(&s_);
        std::int32_t *dst = reinterpret_cast<std::int32_t *>(into);
#if TW_HAVE_AVX512F
        if constexpr (L == 16) {   // masked stores: the destination is not read
            const __mmask16 k = static_cast<__mmask16>(mask);
            for (std::size_t r = 0; r < kRows; ++r, src += L, dst += L) _mm512_mask_store_epi32(dst, k, _mm512_load_si512(src));
            return;
        }
#endif
        const auto m = model::MaskOf<F>(mask);
        for (std::size_t r = 0; r < kRows; ++r, src += L, dst += L) select(m, I::load(src), I::load(dst)).store(dst);
    }
    void RestoreLane(int l) {
        tfSync_ &= ~(1u << l);
        const std::uint32_t *src = reinterpret_cast<const std::uint32_t *>(&snapLanes_) + l;
        std::uint32_t *dst = reinterpret_cast<std::uint32_t *>(&s_) + l;
        for (std::size_t r = 0; r < kRows; ++r) dst[r * L] = src[r * L];
    }

    void ToLanes(std::uint32_t mask) {
        for (std::uint32_t m = mask & aosMask_; m != 0u; m &= m - 1u) {
            const int l = __builtin_ctz(m);
            model::TwinLoadLane(&s_, l, aos_[l]);
            aosMask_ &= ~(1u << l);
            ++stats_.toLanes;
        }
    }
    void ToAos(int l) {
        if (InAos(l)) return;
        model::TwinStoreLane(s_, l, &aos_[l]);
        aosMask_ |= 1u << l;
        ++stats_.toAos;
    }

    const model::CarForceConsts &k_;
    const model::WaterZone *water_;
    Policy policy_;
    int laneLimit_ = L;
    Stats stats_;
    Lane lane_[L];
    LS ls_[L];
    step::TickContext ctx_[L];
    alignas(64) model::TwinLanes<L> s_;
    alignas(64) model::CarTickInL<L> in_;
    alignas(64) float h_[L];
    std::uint32_t ballistic_ = 0u;   // lanes whose wide tick gets the ballistic correction on an h == dt substep
    std::uint32_t subcycle_ = 0u;    // lanes whose tick is L8-subcycled on an h == dt substep
    alignas(64) float hMicro_[L];
    alignas(64) float microSplit_[L];
    alignas(64) float microRem_[L];
    alignas(64) model::CarForceOutL<L> foutMicro_;
    alignas(64) float postMicro_[9][L];
    alignas(64) model::CarForceOutL<L> fout_;
    alignas(64) float postRot_[9][L];
    alignas(64) model::RespTickLanes<L> tick_;
    model::RespTick respTick_[L];       // per-lane counters (speculation save / restore)
    model::RespTick respTickAos_[L];    // last completed tick (evaluator)
    model::RespOut out_[L];
    const model::RespContact *cont_[L];
    int ncont_[L];
    model::TwinState aos_[L];
    model::TwinState view_[L];
    const model::TwinState *poses_[L] = {};      // this pass's pose views (Policy::detect)
    alignas(64) float rowPos_[3][16];            // DetectRows: pose rows when a wide lane is in its AoS copy
    alignas(64) float rowQuat_[4][16];
    alignas(64) float rowAbsorb_[4][16];
    std::uint32_t free_ = 0u;                    // Policy::freeTicks: lanes holding a contact-free proof
    std::uint32_t freeBack_ = 0u;                // ... lanes whose proof failed this tick (no new try before the next)
    float freeLever_ = -1.0f;                    // the detector's FreeLever (< 0: no proofs)
    alignas(64) float sweep_[4][16];             // each lane's sweep w and allowance d asked for this pass (d = 0: none)
    float freeD_[L], freeMargin_[L], freeLeverL_[L], freeWW_[L], freePos_[L][3], freeW_[L][3], freeR_[L][9], freeAbs_[L][4];   // each proof's anchor
    step::CollisionCache *caches_[L] = {};
    std::uint32_t aosMask_ = 0u;            // lanes whose AoS copy is current (else their TwinLanes column)
    std::uint32_t busy_ = 0u;               // lanes holding a candidate
    std::uint32_t track_ = 0u, found_ = 0u;  // Lane::track and finish_[l].found of this tick, as masks
    bool detected_ = true;                  // LS::Detected does something (LS::WantsDetected)
    bool wasCompleted_[L] = {};
    TickElem te_[kMaxTickElems];           // the tick fields' elements; ntf_ < 0: full conversions
    int tfEnd_[4] = {0, 0, 0, 0};          // ends of the 4-byte, 2-byte, u8, i8 groups in te_
    int ntf_ = -1;
    TickElem teOut_[kMaxTickElems];        // ... those the hooks may write (SyncTickOut)
    int tfOutEnd_[4] = {0, 0, 0, 0};
    int ntfOut_ = -1;
    std::uint32_t tfSync_ = 0u;            // lanes whose AoS tick fields are current: equal to the lanes', or ahead
                                           // (hooks wrote them since the last SyncTickIn); cleared when the lanes change
    bool spec_[L] = {};                    // wide tick under speculation
    model::TwinState snapshot_[L];         // its tick-start state (lane AoS-current at the tick start)
    bool snapAos_[L] = {};                 // ... in snapshot_, else in snapLanes_
    alignas(64) model::TwinLanes<L> snapLanes_;
    model::RespTick savedRespTick_[L];
    step::FinishSubstep finish_[L];        // finish tracking (tracked ticks)
    double elapsed_[L] = {};               // offset of the current substep inside the tick (tick.h)
    std::uint32_t finSoa_ = 0u;            // finish_[l].start is in finLanes_ (else in finish_[l].start)
    alignas(64) model::TwinLanes<L> finLanes_;
    step::SubstepContacts contacts_[L];
};

}  // namespace batch
}  // namespace tw

#endif
