// Twin substep policies (design.md §5.1, docs/twin/substep-policy.md). FV-free.
//
// A policy decides, once per tick from the tick-start state, how many
// substeps the tick runs. FV's own rule is
//     n = min(1000, trunc((|v| + |w|) * dt / 0.3) + 1)
// (physics_step.cpp ComputeSubstepCount; n = 0 after a wrap runs one substep).
// The other policies run fewer substeps where FV's extra ones change nothing
// perceptible (M-H2 yardstick, docs/twin/behavior.md):
//
//   fv                    FV's n                                      Tier 1
//   capK[vV]              min(n, K); FV's n at |v| >= V km/h          Tier 2 (cap3: fails checkpoint timing)
//   contactKdD[eE][wW][vV]  no contact event: min(n, K) substeps;
//                         a contact event: min(n, max(K, E)), or FV's n at |v| >= W km/h;
//                         FV's n at |v| >= V km/h                    Tier 2 (contact1d0e3w900: 2.21 substeps/tick, M-H2 PASS)
//
// A contact event is a change of the set of touching contacts between the
// tick-start pose and the poses extrapolated (constant v and w, gravity) to
// dt/2, dt and dt + D/max(|v|, 1 m/s). Who answers that question is the
// ContactPredictor: the FV exact probe now (tools/twin/hybrid/), the twin's
// own collision or B's voxel predictor later. A predictor may answer "event"
// more often than the exact probe (costs substeps, never behaviour); it must
// not miss events the exact probe sees on the M-H2 scenarios.
#ifndef TW_STEP_POLICY_H
#define TW_STEP_POLICY_H

#include <cstdint>
#include <string>

#include "tw_math.h"
#include "tw_state.h"

namespace tw {
namespace step {

// FV's tick duration: float(int(10 ms)) * 0.001f = 0x3c23d70b (0.0100000007),
// one ulp above 0.01f (physics_step.cpp secondsFromMwTimeForPhysicsStep2).
constexpr float kTickDt = 10.0f * 0.001f;
constexpr float kMaxStepDistance = 0.3f;    // solidPhysicalResponseCoefB, 0.3 for every car (board 10 §1.2)
constexpr std::uint32_t kMaxSubsteps = 1000u;

enum class PolicyKind : std::uint8_t {
    Fv = 0,
    Cap = 1,
    ContactAware = 2,
    Speculative = 3,   // L7 (design.md §5.1): run K substeps, keep them if the contact key did not change
};

struct PolicyConfig {
    PolicyKind kind = PolicyKind::Fv;
    std::uint32_t k = 1u;              // Cap: at most K; ContactAware: substeps without an event
    float lookahead = 0.0f;            // ContactAware: D (m)
    std::uint32_t eventCap = kMaxSubsteps;  // ContactAware: substeps on an event = min(n, max(K, E))
    float eventFullKmh = 1e30f;        // ContactAware: an event at |v| >= W km/h gets FV's n
    float fvAboveKmh = 1e30f;          // Cap / ContactAware: FV's n at |v| >= V km/h
    float normalSpeed = 1e30f;         // ContactAware n: also an event if a tick-start contact point moves
                                       // along its normal faster than N m/s (n0: any normal motion)
    float depth = 1e30f;               // ContactAware p: also an event if a tick-start contact is deeper than P m
    bool probeHalf = true;             // ContactAware h0: no probe at dt/2 (only dt and dt + D/|v|)
    bool probeMidOnly = false;         // ContactAware m1: one probe, at dt/2 only (the one-detection fallback)
    float lookStep = 0.0f;             // ContactAware sS: dense lookahead, probes at dt + j S / |v|, j = 1..ceil(D/S)
    float twoAboveKmh = 1e30f;         // ContactAware / Speculative uU: event-free ticks at |v| >= U km/h run min(n, 2)
    bool twoAirborneOnly = false;      // a1: ... only when nothing touches at the tick start
    float twoRotation = 1e30f;         // rR: event-free ticks with |w| dt >= R rad run min(n, 2), at any speed
    float fullRotation = 1e30f;        // fF: event-free ticks with |w| dt >= F rad run FV's n
    bool subcycle = false;             // x1 (L8, car-force agent's form): an airborne 1-substep tick (nothing touching
                                       // at the tick start) with FV count n runs forces + integration n times at FV's
                                       // split of dt, collision once at the tick end; b1 then does not apply
    bool ballistic = false;            // b1: a 1-substep tick whose FV count is n adds F/m dt^2 (n-1)/(2n) to the
                                       // position: FV's n-substep forward-Euler result for a constant force

    static PolicyConfig Fv() { return PolicyConfig{}; }
    static PolicyConfig Cap(std::uint32_t k, float fvAboveKmh = 1e30f) {
        PolicyConfig p;
        p.kind = PolicyKind::Cap;
        p.k = k;
        p.fvAboveKmh = fvAboveKmh;
        return p;
    }
    static PolicyConfig Contact(std::uint32_t k, float lookahead, std::uint32_t eventCap, float eventFullKmh,
                                float fvAboveKmh = 1e30f) {
        PolicyConfig p;
        p.kind = PolicyKind::ContactAware;
        p.k = k;
        p.lookahead = lookahead;
        p.eventCap = eventCap;
        p.eventFullKmh = eventFullKmh;
        p.fvAboveKmh = fvAboveKmh;
        return p;
    }
    // The M-H2 winner (docs/twin/substep-policy.md): contact1d0e3w900n0. The
    // testbed's parser read 'n' from "contact" itself, so every contact policy
    // of the sweep ran with N = 0 (any normal motion of a tick-start contact
    // is an event); without it (n absent = off) contact1d0e3w900 fails M-H2.
    static PolicyConfig Recommended() {
        PolicyConfig p = Contact(1u, 0.0f, 3u, 900.0f);
        p.normalSpeed = 0.0f;
        return p;
    }
};

// TWIN_POLICY grammar: fv | capK[vV] | contactKdD[eE][wW][vV][nN][pP][h0][m1][sS][uU[a1]][rR][b1] |
// specK[eE][wW][vV][nN][pP][uU[a1]][rR][b1] (L7: K speculative substeps), keys
// read after K (tools/twin/swap/policy_tu.py read them from position 1, so
// its 'n' always matched "contact": N = 0). Returns false on anything else.
bool ParsePolicy(const char *text, PolicyConfig *out, std::string *error = nullptr);
// Canonical name, e.g. "fv", "cap3", "cap3v900", "contact1d0e3w900n0".
std::string PolicyName(const PolicyConfig &policy);

// What the predictor is asked. Times are from the tick-start state.
struct PredictQuery {
    float dt = kTickDt;
    float lookahead = 0.0f;    // D (m): last probe at dt + D / max(|v|, 1)
    float gravityY = -10.0f;   // world y acceleration of the extrapolation (gravity * fs(groundedPrev))
    float normalSpeed = 1e30f; // event if a tick-start contact point moves along its normal faster (m/s)
    float depth = 1e30f;       // event if a tick-start contact is deeper (m)
    bool probeHalf = true;     // also probe at dt/2
    bool probeMidOnly = false; // only probe at dt/2
    float lookStep = 0.0f;     // > 0: probes every lookStep metres of travel up to lookahead past dt
    mutable bool startTouching = true;   // set by the predictor: anything touches at the tick start
};

class ContactPredictor {
public:
    virtual ~ContactPredictor() = default;
    // true = the contact set changes within the probe times (run more substeps).
    virtual bool ContactEvent(const model::TwinState &tickStart, const PredictQuery &query) const = 0;
    // Speculative policy: an event decided from the tick-start contacts alone
    // (normal-speed / depth tests on the last collision pass); the key change
    // itself is judged after the speculative substeps (stages, SpeculateChanged).
    virtual bool StartEvent(const model::TwinState &, const PredictQuery &) const { return false; }
};

// FV's |v| and |w| (gm_vector.cpp PhysicsStep2{Linear,Angular}SpeedLength).
TW_INLINE float FvLinearSpeed(const model::TwinState &s) {
    return tw::math::sqrt((s.vel[1] * s.vel[1] + s.vel[0] * s.vel[0]) + s.vel[2] * s.vel[2]);
}
TW_INLINE float FvAngularSpeed(const model::TwinState &s) {
    return tw::math::sqrt((s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1]) + s.angVel[2] * s.angVel[2]);
}

// FV's count (before the 0 -> 1 run rule): trunc_u32_wrap((|v|+|w|) dt / 0.3) + 1, at most 1000.
TW_INLINE std::uint32_t FvSubstepCount(const model::TwinState &s, float dt = kTickDt) {
    const float scaled = ((FvLinearSpeed(s) + FvAngularSpeed(s)) * dt) / kMaxStepDistance;
    // to_u32_trunc_wrap is plain truncation on [0, 2^31) (and -0): its general form only off that range
    const std::uint32_t t = scaled >= 0.0f && scaled < 0x1p31f ? static_cast<std::uint32_t>(static_cast<std::int32_t>(scaled))
                                                                 : tw::math::to_u32_trunc_wrap(scaled);
    std::uint32_t n = t + 1u;
    if (n > kMaxSubsteps) n = kMaxSubsteps;
    return n;
}

struct SubstepPlan {
    std::uint32_t n = 1u;      // substeps this tick (>= 1)
    std::uint32_t nFv = 1u;    // FV's count (>= 1)
    bool probed = false;       // the predictor was asked
    bool event = false;        // ... and answered "contact event"
    bool speculate = false;    // Speculative: run n substeps, then keep them or rerun with nEvent
    bool startTouching = true; // the predictor saw a contact at the tick start (false: airborne)
    std::uint32_t nEvent = 1u; // substeps on an event
};

// Event-free ticks: K substeps, or min(n, 2) under the u / r floors.
TW_INLINE std::uint32_t FreeSubsteps(const PolicyConfig &p, const model::TwinState &s, std::uint32_t nFv, float kmh,
                                     float dt, bool touching) {
    const std::uint32_t k = p.k < 1u ? 1u : p.k;
    bool two = !(kmh < p.twoAboveKmh) && !(p.twoAirborneOnly && touching);
    if (p.twoRotation < 1e29f) {
        const float w = tw::math::sqrt((s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1]) + s.angVel[2] * s.angVel[2]);
        two = two || !(w * dt < p.twoRotation);
    }
    if (p.fullRotation < 1e29f) {
        const float w = tw::math::sqrt((s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1]) + s.angVel[2] * s.angVel[2]);
        if (!(w * dt < p.fullRotation)) return nFv;
    }
    if (two && k < 2u) return nFv < 2u ? nFv : 2u;
    return k;
}

// Substeps on an event tick: min(n, max(K, E)), or FV's n at |v| >= W km/h.
TW_INLINE std::uint32_t EventSubsteps(const PolicyConfig &p, std::uint32_t nFv, float kmh) {
    if (!(kmh < p.eventFullKmh)) return nFv;
    const std::uint32_t k = p.k < 1u ? 1u : p.k;
    const std::uint32_t e = p.eventCap > k ? p.eventCap : k;
    return nFv < e ? nFv : e;
}

// The plan for one tick. `predictor` may be null only for Fv and Cap.
TW_INLINE SubstepPlan PlanSubsteps(const PolicyConfig &p, const model::TwinState &s, float gravityY,
                                   const ContactPredictor *predictor, float dt = kTickDt) {
    SubstepPlan plan;
    std::uint32_t nFv = FvSubstepCount(s, dt);
    if (nFv == 0u) nFv = 1u;   // FV: substeps = 0 skips the split loop and runs the last substep only
    plan.nFv = nFv;
    plan.n = nFv;
    if (p.kind == PolicyKind::Fv) return plan;
    const float kmh = FvLinearSpeed(s) * 3.6f;
    if (!(kmh < p.fvAboveKmh)) return plan;
    const std::uint32_t k = p.k < 1u ? 1u : p.k;
    if (p.kind == PolicyKind::Cap) {
        plan.n = nFv < k ? nFv : k;
        return plan;
    }
    if (nFv <= k) return plan;
    PredictQuery q;
    q.dt = dt;
    q.lookahead = p.lookahead;
    q.gravityY = gravityY;
    q.normalSpeed = p.normalSpeed;
    q.depth = p.depth;
    q.probeHalf = p.probeHalf;
    q.probeMidOnly = p.probeMidOnly;
    q.lookStep = p.lookStep;
    plan.probed = true;
    plan.nEvent = EventSubsteps(p, nFv, kmh);
    if (p.kind == PolicyKind::Speculative) {
        plan.event = predictor != nullptr ? predictor->StartEvent(s, q) : true;
        plan.n = plan.event ? plan.nEvent : FreeSubsteps(p, s, nFv, kmh, dt, q.startTouching);
        plan.speculate = !plan.event && plan.n < plan.nEvent;   // nothing to gain when the free count is already n
        plan.startTouching = q.startTouching;
        return plan;
    }
    plan.event = predictor != nullptr ? predictor->ContactEvent(s, q) : true;
    plan.n = plan.event ? plan.nEvent : FreeSubsteps(p, s, nFv, kmh, dt, q.startTouching);
    plan.startTouching = q.startTouching;
    return plan;
}

// FV's substep durations: n - 1 equal splits dt / float(n), the last one the
// remainder by repeated subtraction (physics_step.cpp:89-106). f(i, h).
template <class F>
TW_INLINE void ForEachSubstepDt(std::uint32_t n, float dt, F &&f) {
    float remaining = dt;
    if (n > 1u) {
        const float split = dt / tw::math::from_u32(n);
        for (std::uint32_t i = 0; i + 1u < n; ++i) {
            f(i, split);
            remaining = remaining - split;
        }
    }
    f(n > 0u ? n - 1u : 0u, remaining);
}

}  // namespace step
}  // namespace tw

#endif
