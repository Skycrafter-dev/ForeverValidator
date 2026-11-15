// The all-twin stages per batch lane: step::TwinStages (src/twin/step/twin_stages.h,
// the tick agent's tick contract) bound to one candidate's control stream.
//
// The batch runner (batch_runner.h) calls the per-lane hooks; Forces,
// Integrate and Respond stay the runner's (wide or narrow), so TwinStages'
// own versions are not used here. Per tick, as TwinStep (tw_step.cpp
// StepKernel): a fresh TickContext with respawnCount decoded from the
// control (or the exact count of an escape tick), RunTick's stage sequence.
//
// TrackFinish: every tick of a race that is not completed. The runner keeps
// the finish snapshot of tracked lanes at every substep start (tick.h), in
// SoA for all of them at once, so no checkpoint-count assumption is needed
// and LaneStatus::UntrackedFinish cannot occur.
//
// Partial sync: on an ordinary tick the per-tick hooks touch only TickFields()
// (TwinStages: TickBegin -> tickMs, and speedBlocked / speedBlocked2 on the
// begin-race bit that every race tick carries; Plan -> vel, angVel, groundedPrev,
// and under the speculative policy its StartPassProbe -> pos, quat, absorb (a
// detection) and vel, angVel; TrackFinish -> raceCompleted; TickEnd of a tick
// without a finish -> nothing, and of a finish tick the finish fields
// (TwStoreFinish), which the runner hands over as a full AoS state; the
// speculation hooks -> the pose). TickFieldsFor drops the pose outside the
// speculative policy; of these, the hooks write only TickWriteFields() (TickBegin's),
// and TickEnd of a partial tick nothing (kTickEndPartialPure: no sync around it). A tick with a spawn, race-start reset or
// respawns (VehicleReset and the respawn placement touch any field), and any
// tick under an external ContactPredictor (its reads are not declared), gets
// the full state (TickNeedsFullState). The list mirrors race::TwRaceTickBegin
// and TwinStages: the batch == TwinStep identity test on real replays
// (tools/twin/twin_batch_real.cpp) is its check.
//
// Collision cache (stages.h CollisionCache, agreed with B): one per lane,
// created at the first Bind (CollisionStage::NewCache) and flushed at every
// Bind; TwinStages detects through it. Result-invariant by contract.
#ifndef TW_BATCH_TWIN_LANE_STAGES_H
#define TW_BATCH_TWIN_LANE_STAGES_H

#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "batch_runner.h"
#include "twin_stages.h"

namespace tw {
namespace batch {

// Candidate::payload of a twin candidate.
struct TwinCandidateControls {
    const std::uint32_t *controls = nullptr;          // TwinControl per tick (ticks entries)
    const std::pair<std::uint32_t, std::uint32_t> *respawnExact = nullptr;   // (tick, exact count) of escape ticks
    std::uint32_t respawnExactCount = 0u;
};

class TwinLaneStages {
public:
    explicit TwinLaneStages(const step::TwinPacks *packs = nullptr, bool flushEveryCall = false)
        : packs_(packs), flushEveryCall_(flushEveryCall) {}
    TwinLaneStages(const TwinLaneStages &o) : packs_(o.packs_), flushEveryCall_(o.flushEveryCall_) {}
    TwinLaneStages &operator=(const TwinLaneStages &o) {
        if (this != &o) {
            Reset();
            packs_ = o.packs_;
            flushEveryCall_ = o.flushEveryCall_;
            cache_.reset();
        }
        return *this;
    }
    ~TwinLaneStages() { Reset(); }

    static const char *const *TickFields(int *n) {
        static const char *const f[] = {"tickMs", "pos",          "quat",         "vel",         "angVel",
                                        "absorb", "groundedPrev", "raceCompleted", "speedBlocked", "speedBlocked2"};
        *n = static_cast<int>(sizeof f / sizeof f[0]);
        return f;
    }
    // This instance's: the pose only under the speculative policy (StartPassProbe's detection).
    const char *const *TickFieldsFor(int *n) const {
        if (packs_ == nullptr || packs_->policy.kind == step::PolicyKind::Speculative) return TickFields(n);
        static const char *const f[] = {"tickMs", "vel", "angVel", "groundedPrev", "raceCompleted", "speedBlocked",
                                        "speedBlocked2"};
        *n = static_cast<int>(sizeof f / sizeof f[0]);
        return f;
    }
    // What the hooks write on a partial-state tick: TickBegin's tick time and speed blocks (TickEnd and the
    // speculation hooks write no state there).
    static const char *const *TickWriteFields(int *n) {
        static const char *const f[] = {"tickMs", "speedBlocked", "speedBlocked2"};
        *n = static_cast<int>(sizeof f / sizeof f[0]);
        return f;
    }
    static constexpr bool kTickEndPartialPure = true;   // TwinStages::TickEnd reads the state on a finish only
    // called before TickBegin of tick tick_
    bool TickNeedsFullState() const {
        if (packs_->predictor != nullptr) return true;
        constexpr std::uint32_t kFull = race::kCtlRespawnMask | race::kCtlRespawnEscape | race::kCtlRaceStartReset |
                                        race::kCtlEstablishSpawn;
        if (in_->controls[tick_] & kFull) return true;
        for (std::uint32_t i = 0; i < in_->respawnExactCount; ++i)
            if (in_->respawnExact[i].first == tick_ && in_->respawnExact[i].second != 0u) return true;
        return false;
    }
    bool Ballistic() const { return packs_->policy.ballistic; }   // runner's Integrate: TwinStages::Integrate's correction
    bool Subcycle() const { return packs_->policy.subcycle; }     // runner's Forces: TwinStages::Forces' L8

    void Bind(const Candidate &c) {
        Reset();
        if (!cache_ && packs_->collision != nullptr) {
            cache_ = packs_->collision->NewCache();
            if (cache_) cache_->flushEveryCall = flushEveryCall_;
        }
        if (cache_) cache_->Flush();
        new (stagesMem_) step::TwinStages(*packs_, cache_.get());
        bound_ = true;
        in_ = static_cast<const TwinCandidateControls *>(c.payload);
        tick_ = 0u;
    }

    void TickBegin(model::TwinState &s, step::TickContext &t) {
        const std::uint32_t control = in_->controls[tick_];
        t.control = control;
        t.respawnCount = race::TwCtlRespawnField(control);
        for (std::uint32_t i = 0; i < in_->respawnExactCount; ++i)
            if (in_->respawnExact[i].first == tick_) t.respawnCount = in_->respawnExact[i].second;
        St().TickBegin(s, t);
    }
    step::SubstepPlan Plan(const model::TwinState &s, step::TickContext &t) { return St().Plan(s, t); }
    bool TrackFinish(const model::TwinState &s, const step::TickContext &) const { return !s.raceCompleted; }
    void Detect(const model::TwinState &s, step::SubstepContacts &out) { St().Detect(s, out); }
    // Batch collision (Policy::detect): the lane's cache, and TwinStages::Detect's bookkeeping after
    // the stage's DetectCached (the speculation keys). Detect == DetectCached + Detected; keep the
    // two in step with TwinStages::Detect (the real identity test checks it). The keys are read by
    // SpeculateChanged only, which runs for speculative ticks only (PolicyKind::Speculative): other
    // policies skip them.
    step::CollisionCache *DetectCache() { return cache_.get(); }
    bool WantsDetected() const { return packs_ == nullptr || packs_->policy.kind == step::PolicyKind::Speculative; }
    void Detected(const step::SubstepContacts &out) {
        if (packs_->policy.kind != step::PolicyKind::Speculative) return;
        St().lastKeyCount = step::TwinStages::Keys(out, St().lastKeys);
    }
    void RaceHits(model::TwinState &s, const step::SubstepContacts &c, const model::RespOut &r, step::TickContext &t) {
        St().RaceHits(s, c, r, t);
    }
    void SpeculateBegin(const model::TwinState &s, step::TickContext &t) { St().SpeculateBegin(s, t); }
    bool SpeculateChanged(const model::TwinState &s, step::TickContext &t) { return St().SpeculateChanged(s, t); }
    void SpeculateEnd(model::TwinState &s, step::TickContext &t, bool keep) { St().SpeculateEnd(s, t, keep); }
    void TickEnd(model::TwinState &s, step::TickContext &t, const step::FinishSubstep &f) {
        St().TickEnd(s, t, f);
        ++tick_;
    }

private:
    step::TwinStages &St() { return *std::launder(reinterpret_cast<step::TwinStages *>(stagesMem_)); }
    void Reset() {
        if (bound_) St().~TwinStages();
        bound_ = false;
    }

    const step::TwinPacks *packs_ = nullptr;
    const TwinCandidateControls *in_ = nullptr;
    std::unique_ptr<step::CollisionCache> cache_;   // this lane's, kept across candidates
    std::uint32_t tick_ = 0u;
    bool bound_ = false;
    bool flushEveryCall_ = false;
    alignas(step::TwinStages) unsigned char stagesMem_[sizeof(step::TwinStages)];
};

}  // namespace batch
}  // namespace tw

#endif
