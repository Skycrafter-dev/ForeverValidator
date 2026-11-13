// Twin tick loop (design.md §5.2, board 10 §1.1-1.3). FV-free, header-only.
//
// RunTick(stages, s, control, t): TickBegin, Plan, the substeps, TickEnd.
// RunSubsteps(stages, s, t): Plan and the substeps only; the FV-hosted
// hybrids call it from inside FV's PhysicsStep2, where FV has already done
// T1-T3 and does T5 itself.
//
// `S` is any type with the TickStages member functions (stages.h): the
// production TwinStages (static dispatch, twin_stages.h) or a TickStages
// subclass (virtual dispatch). Per substep:
//
//   SubstepBegin(s, h)                  (FV: NotifyBeforeCollisionSubstep; twin: nothing)
//   Forces(s, t, h, f)                  S1 + C1..C20
//   Integrate(s, f, h)                  S2
//   Detect(s, contacts)                 S3a
//   Respond(s, pose, contacts, resp)    S3b + S4
//   RaceHits(s, contacts, resp, t)      only if the response saw trigger contacts
//   SubstepEnd(s, h)
//
// The substep in which the race completes is remembered (start state, offset
// inside the tick, duration) for the sub-tick finish bracket in TickEnd.
#ifndef TW_STEP_TICK_H
#define TW_STEP_TICK_H

#include <cstdint>
#include <cstring>

#include "policy.h"
#include "stages.h"

namespace tw {
namespace step {

struct FinishSubstep {
    bool found = false;
    double startSeconds = 0.0;   // offset of the completing substep inside the tick
    float duration = 0.0f;
    model::TwinState start{};    // state at that substep's start
};

// One substep of duration h. `contacts` is caller scratch.
template <class S>
TW_INLINE void RunSubstep(S &st, model::TwinState &s, TickContext &t, float h, SubstepContacts &contacts) {
    st.SubstepBegin(s, h);
    model::CarForceOut f;
    st.Forces(s, t, h, f);
    st.Integrate(s, f, h);
    st.Detect(s, contacts);
    SubstepPose pose;
    std::memcpy(pose.pos, f.posePos, sizeof pose.pos);
    std::memcpy(pose.rot, f.poseRot, sizeof pose.rot);
    model::RespOut resp;
    st.Respond(s, pose, contacts, resp);
    if (resp.triggerCount != 0u) st.RaceHits(s, contacts, resp, t);
    st.SubstepEnd(s, h);
    ++t.substepsRun;
}

template <class S>
TW_INLINE void RunSubsteps(S &st, model::TwinState &s, TickContext &t, FinishSubstep *finish = nullptr) {
    t.plan = st.Plan(s, t);
    SubstepContacts contacts;
    auto run = [&](std::uint32_t n) {
        double elapsed = 0.0;
        ForEachSubstepDt(n, t.dt, [&](std::uint32_t, float h) {
            const bool track = finish != nullptr && !finish->found && !s.raceCompleted;
            if (track) std::memcpy(&finish->start, &s, sizeof s);
            RunSubstep(st, s, t, h, contacts);
            if (track && s.raceCompleted) {
                finish->found = true;
                finish->startSeconds = elapsed;
                finish->duration = h;
            }
            elapsed += static_cast<double>(h);
        });
    };
    if (t.plan.speculate) {   // L7: try plan.n substeps; keep them if the contact key held
        model::TwinState snapshot;
        std::memcpy(&snapshot, &s, sizeof s);
        st.SpeculateBegin(s, t);
        run(t.plan.n);
        if (!st.SpeculateChanged(s, t)) {
            st.SpeculateEnd(s, t, true);
            return;
        }
        std::memcpy(&s, &snapshot, sizeof s);
        st.SpeculateEnd(s, t, false);
        if (finish != nullptr) finish->found = false;
        t.substepsWasted += t.plan.n;
        t.plan.event = true;
        t.plan.n = t.plan.nEvent;
    }
    run(t.plan.n);
}

template <class S>
TW_INLINE void RunTick(S &st, model::TwinState &s, std::uint32_t control, TickContext &t) {
    t.control = control;
    st.TickBegin(s, t);
    FinishSubstep finish;
    RunSubsteps(st, s, t, &finish);
    st.TickEnd(s, t, finish);
}

}  // namespace step
}  // namespace tw

#endif
