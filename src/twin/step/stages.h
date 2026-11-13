// Twin tick stages (design.md §5.2, board 10 §1). FV-free.
//
// The tick (tick.h) is a fixed sequence of stages; every stage is one call on
// a Stages object, so each can be the twin's kernel, an FV stand-in (the
// hybrids in tools/twin/hybrid/) or a test double:
//
//   TickBegin   T1-T3   controls, race transitions, respawns
//   Plan        T4      substep count (policy.h + the ContactPredictor)
//   per substep h (SubstepBegin / SubstepEnd around it):
//     Forces    S1 + C1..C20   gravity, drag, car forces (writes the next-substep fs/lds, clears contact scratch)
//     Integrate S2             pos with the old v, quaternion about the COM, w clamp
//     Detect    S3a            merged per-shape contacts at the post-integration pose (CollisionStage)
//     Respond   S3b + S4       impulses, wheel / body contact state, position correction
//     RaceHits  S3 callbacks   trigger contacts in processing order -> checkpoints, finish
//   TickEnd     T5            finish bracket, per-tick snapshot
//
// The production kernel instantiates tick.h with TwinStages (twin_stages.h,
// static dispatch); TickStages below is the same contract with virtual calls
// for the hybrids.
#ifndef TW_STEP_STAGES_H
#define TW_STEP_STAGES_H

#include <cstdint>
#include <memory>

#include "forces_io.h"
#include "policy.h"
#include "response.h"
#include "tw_state.h"

namespace tw {
namespace step {

// One substep's contacts as the collision stage hands them to the response
// (world frame, collision order, unsorted; triggers flagged kRespTrigger).
struct SubstepContacts {
    int n = 0;
    bool overflow = false;               // more than kRespMaxContacts (kept the first ones)
    model::RespContact c[model::kRespMaxContacts];
};

// The substep-start pose the force stage saw (FV write state): the response
// uses its rotation for the booster peer axis; collision ignores it.
struct SubstepPose {
    float pos[3];
    float rot[9];                        // columns, as CarForceOut::poseRot
};

// Per-candidate collision cache (lead decision 2026-11-08). The caller owns it
// (a TwinStep caller per candidate, a batch lane), outside the 448 B physics
// state; the stage creates it (CollisionStage::NewCache) and defines what it
// holds. Result-invariant: every entry is checked against the current pose on
// use, and a cache of another stage or world counts as cold, so the stage's
// output is bit-identical with any cache content (cold, stale, after a
// snapshot restore). Flush() is therefore always valid and never required.
class CollisionCache {
public:
    virtual ~CollisionCache() = default;
    virtual void Flush() = 0;
    bool flushEveryCall = false;   // test mode (CI): Flush() before every detection
    const void *const kind;        // the implementation's tag (a stage only uses its own caches)

protected:
    explicit CollisionCache(const void *k) : kind(k) {}
};

// Collision detection at the state's (post-integration) pose. Implemented by
// the twin's collide/ (M3) or by FV in the hybrids.
class CollisionStage {
public:
    virtual ~CollisionStage() = default;
    virtual void Detect(const model::TwinState &s, SubstepContacts &out) const = 0;
    // A cache for this stage, or null (the stage keeps none).
    virtual std::unique_ptr<CollisionCache> NewCache() const { return nullptr; }
    // Detect with a per-candidate cache (null: uncached); the same bits as Detect.
    virtual void DetectCached(const model::TwinState &s, CollisionCache *cache, SubstepContacts &out) const {
        (void)cache;
        Detect(s, out);
    }
};

// Per-tick context the stages share.
struct TickContext {
    std::uint32_t control = 0u;          // TwinControl (race/tw_race.h)
    std::uint32_t respawnCount = 0u;     // decoded respawns before physics
    float dt = kTickDt;                  // tick duration (s)
    model::CarTickIn in{};               // inputs the force model sees all tick
    SubstepPlan plan{};
    std::uint32_t substepsRun = 0u;      // including speculative substeps that were thrown away
    std::uint32_t substepsWasted = 0u;   // Speculative: substeps of a rejected speculation
    std::uint32_t triggerHits = 0u;      // trigger contacts handed to the race stage
    std::uint32_t checkpoints = 0u;      // accepted passages (checkpoints and finish)
    std::uint32_t cpLastSlot = 0xFFFFFFFFu;   // slot of the last accepted passage this tick (none: 0xFFFFFFFF)
    bool finishPassage = false;          // a finish passage was accepted this tick (lap end or race end)
    std::uint32_t wheelContacts = 0u;    // this tick's response counters (FV AfterContacts: fr.has_*_contact)
    std::uint32_t bodyContacts = 0u;
    std::uint32_t respawned = 0u;
    bool finished = false;               // race completed this tick
    std::uint32_t finishTickMs = 0u;
};

struct FinishSubstep;  // tick.h

// Virtual stage set (hybrids, tests). Same contract as TwinStages.
class TickStages {
public:
    virtual ~TickStages() = default;
    virtual void TickBegin(model::TwinState &s, TickContext &t) = 0;
    virtual SubstepPlan Plan(const model::TwinState &s, TickContext &t) = 0;
    virtual void SubstepBegin(model::TwinState &s, float h) = 0;
    virtual void Forces(model::TwinState &s, const TickContext &t, float h, model::CarForceOut &out) = 0;
    virtual void Integrate(model::TwinState &s, const model::CarForceOut &f, float h) = 0;
    virtual void Detect(const model::TwinState &s, SubstepContacts &out) = 0;
    // `pose`: the substep-start pose (Forces' CarForceOut posePos / poseRot).
    virtual void Respond(model::TwinState &s, const SubstepPose &pose, const SubstepContacts &c,
                         model::RespOut &out) = 0;
    virtual void RaceHits(model::TwinState &s, const SubstepContacts &c, const model::RespOut &resp,
                          TickContext &t) = 0;
    virtual void SubstepEnd(model::TwinState &s, float h) = 0;
    // Speculative policy (L7): around the speculative substeps. SpeculateChanged
    // compares the contact key of the last collision pass with the tick-start
    // key; SpeculateEnd(keep = false) undoes the stages' own side state (the
    // TwinState itself is restored by the loop).
    virtual void SpeculateBegin(const model::TwinState &s, TickContext &t) = 0;
    virtual bool SpeculateChanged(const model::TwinState &s, TickContext &t) = 0;
    virtual void SpeculateEnd(model::TwinState &s, TickContext &t, bool keep) = 0;
    // `finish`: the substep in which the race completed this tick, if any.
    virtual void TickEnd(model::TwinState &s, TickContext &t, const FinishSubstep &finish) = 0;
};

}  // namespace step
}  // namespace tw

#endif
