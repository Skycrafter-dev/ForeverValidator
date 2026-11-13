// Twin ISA dispatch: the tick entry of one variant (tw_isa.h).
//
// Compiled once per variant with that variant's ISA flags and
//   TW_ISA_KIND   0 baseline | 1 avx2 | 2 avx512
//   TW_ISA_ENTRY  the variant's extern "C" table function
// together with the tick-path sources (src/twin/CMakeLists.txt
// TW_ISA_TICK_SOURCES); isolate_variant.sh then renames every other symbol.
#include "tw_isa.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "cc_stage.h"
#include <memory>

#include "atlas_stage.h"
#include "batch_runner.h"
#include "policy.h"
#include "twin_lane_stages.h"
#include "tw_collision_choice.h"
#include "tw_exact_fast.h"
#include "tw_fpenv.h"
#include "tw_math.h"
#include "stages.h"

#ifndef TW_ISA_KIND
#error "TW_ISA_KIND must be defined (0 baseline, 1 avx2, 2 avx512)"
#endif
#ifndef TW_ISA_ENTRY
#error "TW_ISA_ENTRY must name the variant's table function"
#endif

namespace {

using namespace tw;

struct Instance {
    collide::ExactCollision collision;
    collide_c::CcCollision collisionC;   // C's stage: "c" (the product default) or a C experiment (tw_collision_choice.h)
    std::unique_ptr<atlas::AtlasCollision> collisionAtlas;   // E's atlas stage ("atlas")
    const step::CollisionStage *stage = nullptr;
    step::Model model;
};

isa::Layout LayoutHere() {
    isa::Layout l;
    l.abi = 2u;
    std::uint32_t *w = l.words;
    *w++ = sizeof(isa::TickInputs);         *w++ = alignof(isa::TickInputs);
    *w++ = sizeof(world::WorldPackView);    *w++ = alignof(world::WorldPackView);
    *w++ = sizeof(collide::CarShapes);      *w++ = alignof(collide::CarShapes);
    *w++ = sizeof(collide::CollideOptions); *w++ = alignof(collide::CollideOptions);
    *w++ = sizeof(model::CarForceConsts);   *w++ = alignof(model::CarForceConsts);
    *w++ = sizeof(model::WaterZone);        *w++ = alignof(model::WaterZone);
    *w++ = sizeof(race::TwRaceMap);         *w++ = alignof(race::TwRaceMap);
    *w++ = sizeof(model::TwinState);        *w++ = alignof(model::TwinState);
    *w++ = sizeof(step::Events);            *w++ = alignof(step::Events);
    *w++ = sizeof(step::StopRule);          *w++ = alignof(step::StopRule);
    *w++ = sizeof(step::Model);             *w++ = alignof(step::Model);
    *w++ = sizeof(step::PolicyConfig);      *w++ = alignof(step::PolicyConfig);
    *w++ = sizeof(isa::BatchJob);           *w++ = alignof(isa::BatchJob);
    *w++ = sizeof(isa::BatchTick);          *w++ = alignof(isa::BatchTick);
    *w++ = sizeof(isa::BatchCallbacks);     *w++ = alignof(isa::BatchCallbacks);
    *w++ = sizeof(isa::BatchPolicy);        *w++ = alignof(isa::BatchPolicy);
    *w++ = sizeof(isa::BatchStats);         *w++ = alignof(isa::BatchStats);
    return l;
}

void *Create(const isa::TickInputs &in, const char **diagnostic) {
    if (in.view == nullptr || in.shapes == nullptr || in.car == nullptr || in.race == nullptr) {
        if (diagnostic) *diagnostic = "twin isa: incomplete tick inputs";
        return nullptr;
    }
    Instance *instance = new (std::nothrow) Instance;
    if (instance == nullptr) {
        if (diagnostic) *diagnostic = "twin isa: allocation failed";
        return nullptr;
    }
    std::string error;
    // The world's stage, resolved here so hooks (b-noise) point at this variant's own copy.
    collision_choice::Choice choice;
    if (!collision_choice::Resolve(in.collisionStage != nullptr ? in.collisionStage : collision_choice::kNoise, in.collide,
                                   &choice)) {
        delete instance;
        if (diagnostic) *diagnostic = "twin isa: unknown collision stage";
        return nullptr;
    }
    if (!instance->collision.Init(*in.view, *in.shapes, &error, choice.b)) {
        delete instance;
        if (diagnostic) *diagnostic = "twin isa: collision stage init failed";
        return nullptr;
    }
    instance->model.policy = in.policy;
    instance->model.car = in.car;
    instance->model.water = in.water;
    instance->model.race = in.race;
    instance->stage = &instance->collision;
    if (choice.useC) {
        if (!instance->collisionC.Init(*in.view, *in.shapes, &error, choice.c)) {
            delete instance;
            if (diagnostic) *diagnostic = "twin isa: collision stage (C) init failed";
            return nullptr;
        }
        instance->stage = &instance->collisionC;
    }
    if (choice.useAtlas) {
        instance->collisionAtlas.reset(new (std::nothrow) atlas::AtlasCollision());
        if (!instance->collisionAtlas || !instance->collisionAtlas->Init(*in.view, *in.shapes, &error)) {
            delete instance;
            if (diagnostic) *diagnostic = "twin isa: collision stage (atlas) init failed";
            return nullptr;
        }
        instance->stage = instance->collisionAtlas.get();
    }
    instance->model.collision = instance->stage;
    return instance;
}

void Destroy(void *instance) { delete static_cast<Instance *>(instance); }

step::Status Step(const void *instance, model::TwinState &state, step::Control control, step::Events *events,
                  step::CollisionCache *cache) {
    return step::Step(static_cast<const Instance *>(instance)->model, state, control, events, cache);
}

std::uint32_t StepTicks(const void *instance, model::TwinState &state, const step::Control *controls,
                        std::size_t count, step::StopRule stop, step::Events *events,
                        const std::uint32_t *respawnCounts, step::CollisionCache *cache) {
    return step::StepTicks(static_cast<const Instance *>(instance)->model, state, controls, count, stop, events,
                           respawnCounts, cache);
}

step::CollisionCache *NewCache(const void *instance) {
    return static_cast<const Instance *>(instance)->stage->NewCache().release();
}

void DestroyCache(step::CollisionCache *cache) { delete cache; }

// ---- batch (search) ----------------------------------------------------------------------
// Runner<L, TwinLaneStages> over the instance's packs. Each in-flight job holds one Slot (the
// Candidate the runner points at, its controls and its sticky finish), at most L + 1 at a time.
// The runner works in the canonical FP mode; the callbacks run in the caller's.
struct BatchBase {
    virtual ~BatchBase() = default;
    virtual void Run(const isa::BatchCallbacks &cb, isa::BatchStats *stats) = 0;
};

// Batch collision for C's stage (CcCollision::DetectLanes / DetectLanes16 on the runner's pose rows: the front with
// lane = candidate, the lanes' survivors in shared 16-lane packets).
// FOREVERVALIDATOR_TWIN_LANE_DETECT=0 turns it off (the runner then detects lane by lane).
struct CcLaneDetect final : batch::LaneDetect {
    const collide_c::CcCollision *c = nullptr;
    void DetectLanes(int lanes, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                     step::SubstepContacts *out) const override {
        c->DetectLanes(lanes, poses, mask, caches, out);
    }
    bool DetectRows16(const float (*pos)[16], const float (*quat)[16], const float (*absorb)[16], std::uint32_t mask,
                      step::CollisionCache *const *caches, step::SubstepContacts *out, std::uint32_t *empty) const override {
        return c->DetectLanes16(pos, quat, absorb, mask, caches, out, empty);
    }
    std::uint32_t ProveRows16(const float (*sweep)[16], std::uint32_t ask, step::CollisionCache *const *caches) const override {
        return c->ProveLanes16(sweep, ask, caches);
    }
    float FreeLever() const override { return c->FreeLever(); }
    static const CcLaneDetect *For(const Instance &in, CcLaneDetect *self) {
        if (in.stage != &in.collisionC || !in.collisionC.LanesBatch()) return nullptr;
        const char *env = std::getenv("FOREVERVALIDATOR_TWIN_LANE_DETECT");
        if (env != nullptr && std::strcmp(env, "0") == 0) return nullptr;
        self->c = &in.collisionC;
        return self;
    }
};

template <int L>
class Batch final : public BatchBase {
public:
    Batch(const Instance &in, const isa::BatchPolicy &policy)
        : fullState_(policy.fullState), packs_(MakePacks(in)), proto_(&packs_),
          runner_(*in.model.car, in.model.water, proto_, RunnerPolicy(policy, CcLaneDetect::For(in, &laneDetect_))) {
        fPos_ = batch::LaneRead<L>::Field("pos");
        fVel_ = batch::LaneRead<L>::Field("vel");
        fQuat_ = batch::LaneRead<L>::Field("quat");
        fTickMs_ = batch::LaneRead<L>::Field("tickMs");
        fCompleted_ = batch::LaneRead<L>::Field("raceCompleted");
        fFinishPresent_ = batch::LaneRead<L>::Field("finishPresent");
        fFinishLowerNs_ = batch::LaneRead<L>::Field("finishLowerNs");
    }

    void Run(const isa::BatchCallbacks &cb, isa::BatchStats *stats) override {
        const unsigned caller = math::fp_mode_get();
        {
            math::FpGuard guard;
            RunKernel(cb, caller);
        }
        if (stats != nullptr) {
            const batch::Stats &s = runner_.stats();
            stats->jobs = s.candidates;
            stats->ticks = s.ticks;
            stats->laneSubsteps = s.laneSubsteps;
            stats->passes = s.passes;
            stats->forcesWide = s.forcesWide;
            stats->forcesNarrowLanes = s.forcesNarrowLanes;
            stats->respondWide = s.respondWide;
            stats->respondNarrowLanes = s.respondNarrowLanes;
            stats->freeProofs = s.freeProofs;
            stats->freeProven = s.freeProven;
            stats->freeSkipped = s.freeSkipped;
            stats->freeEscapes = s.freeEscapes;
            stats->freeViolations = s.freeViolations;
        }
    }

private:
    static constexpr int kSlots = 2 * L + 2;
    struct Slot {   // slot k: the runner's candidate is cands_[k]
        batch::TwinCandidateControls controls;
        bool busy = false, finished = false;
        std::uint64_t finishLowerNs = 0u;
    };

    static step::TwinPacks MakePacks(const Instance &in) {
        step::TwinPacks p;
        p.car = in.model.car;
        p.water = in.model.water;
        p.collision = in.model.collision;
        p.race = in.model.race;
        p.predictor = in.model.predictor;
        p.policy = step::ModelPolicyConfig(in.model);
        return p;
    }
    static batch::Policy RunnerPolicy(const isa::BatchPolicy &p, const batch::LaneDetect *detect) {
        batch::Policy r;
        r.detect = detect;
        // Idea 5 (batch::Policy::freeTicks): FOREVERVALIDATOR_TWIN_FREE_TICKS=1 skips proven contact-free substeps,
        // =check detects them anyway and counts the non-empty ones (BatchStats::freeViolations, must be 0)
        if (const char *env = std::getenv("FOREVERVALIDATOR_TWIN_FREE_TICKS")) {
            r.freeTicks = std::strcmp(env, "1") == 0 || std::strcmp(env, "check") == 0;
            r.freeCheck = std::strcmp(env, "check") == 0;
        }
        if (const char *env = std::getenv("FOREVERVALIDATOR_TWIN_FREE_AHEAD")) r.freeTicksAhead = static_cast<float>(std::atof(env));
        if (const char *env = std::getenv("FOREVERVALIDATOR_TWIN_FREE_SLACK")) r.freeSlack = static_cast<float>(std::atof(env));
        r.forcesMinLanes = p.forcesMinLanes;
        r.respondMinLanes = p.respondMinLanes;
        return r;
    }

    // The caller's FP mode around a callback (no switch when it is the canonical one already).
    struct CallerMode {
        unsigned caller;
        bool sw;
        explicit CallerMode(unsigned c) : caller(c), sw((c & math::kFpModeControlMask) != (math::kCanonicalFpMode & math::kFpModeControlMask)) {
            if (sw) math::fp_mode_set(caller);
        }
        ~CallerMode() {
            if (sw) math::fp_mode_set(math::kCanonicalFpMode);
        }
    };

    __attribute__((noinline)) void RunKernel(const isa::BatchCallbacks &cb, unsigned caller) {
        for (Slot &s : slots_) s.busy = false;
        auto next = [&](const batch::Candidate *&out) -> bool {
            isa::BatchJob job;
            bool ok;
            {
                CallerMode m(caller);
                ok = cb.next(cb.user, &job);
            }
            if (!ok) return false;
            int k = 0;
            while (slots_[k].busy) ++k;   // at most L + 1 jobs are in flight
            Slot *slot = &slots_[k];
            batch::Candidate &cand = cands_[k];
            slot->busy = true;
            slot->finished = false;
            slot->finishLowerNs = 0u;
            slot->controls = batch::TwinCandidateControls{};
            slot->controls.controls = job.controls;
            cand.id = job.id;
            std::memcpy(&cand.start, job.start, sizeof cand.start);
            cand.ticks = job.ticks;
            cand.firstTick = 0u;
            cand.payload = &slot->controls;
            out = &cand;
            return true;
        };
        auto evaluate = [&](const batch::Candidate &c, std::uint32_t tickIndex, const batch::LaneRead<L> &v,
                            const step::TickContext &t) -> bool {
            Slot &slot = SlotOf(c);
            isa::BatchTick tick;
            tick.id = c.id;
            tick.tick = tickIndex;
            // the fields straight from the current copy (LaneRead's words without the field table)
            if (v.aos != nullptr) {
                const model::TwinState &s = *v.aos;
                for (int a = 0; a < 3; ++a) tick.pos[a] = s.pos[a];
                for (int a = 0; a < 3; ++a) tick.vel[a] = s.vel[a];
                for (int a = 0; a < 4; ++a) tick.quat[a] = s.quat[a];
                tick.tickMs = s.tickMs;
                tick.raceCompleted = s.raceCompleted != 0u;
                if (t.finished && !slot.finished) {   // step::StepKernel's Events::finished / finishLowerNs
                    slot.finished = true;
                    slot.finishLowerNs = s.finishPresent != 0u
                            ? (static_cast<std::uint64_t>(s.finishLowerNs[1]) << 32) | s.finishLowerNs[0]
                            : 0u;
                }
            } else {
                const model::TwinLanes<L> &s = *v.lanes;
                const int l = v.lane;
                for (int a = 0; a < 3; ++a) tick.pos[a] = s.pos[a][l];
                for (int a = 0; a < 3; ++a) tick.vel[a] = s.vel[a][l];
                for (int a = 0; a < 4; ++a) tick.quat[a] = s.quat[a][l];
                tick.tickMs = s.tickMs[0][l];
                tick.raceCompleted = s.raceCompleted[0][l] != 0u;
                if (t.finished && !slot.finished) {
                    slot.finished = true;
                    slot.finishLowerNs = s.finishPresent[0][l] != 0u
                            ? (static_cast<std::uint64_t>(s.finishLowerNs[1][l]) << 32) | s.finishLowerNs[0][l]
                            : 0u;
                }
            }
            tick.finished = slot.finished;
            tick.finishLowerNs = slot.finishLowerNs;
            if (fullState_) {
                if (v.aos != nullptr) {
                    std::memcpy(&full_, v.aos, sizeof full_);
                } else {
                    std::memcpy(&full_, &c.start, sizeof full_);   // padding bytes as the lane's own copy
                    model::TwinStoreLane(*v.lanes, v.lane, &full_);
                }
                tick.state = &full_;
            }
            CallerMode m(caller);
            return cb.evaluate(cb.user, tick);
        };
        auto done = [&](const batch::Candidate &c, const model::TwinState &, std::uint32_t ticks, batch::LaneStatus) {
            Slot &slot = SlotOf(c);
            {
                CallerMode m(caller);
                cb.done(cb.user, c.id, ticks);
            }
            slot.busy = false;
        };
        runner_.SetLaneLimit(cb.laneLimit > 0 ? cb.laneLimit : L);
        runner_.Run(next, evaluate, done);
    }

    Slot &SlotOf(const batch::Candidate &c) { return slots_[static_cast<std::size_t>(&c - cands_)]; }

    bool fullState_;
    step::TwinPacks packs_;
    batch::TwinLaneStages proto_;
    CcLaneDetect laneDetect_;
    batch::Runner<L, batch::TwinLaneStages> runner_;
    Slot slots_[kSlots];
    batch::Candidate cands_[kSlots];
    model::TwinState full_{};
    int fPos_ = -1, fVel_ = -1, fQuat_ = -1, fTickMs_ = -1, fCompleted_ = -1, fFinishPresent_ = -1, fFinishLowerNs_ = -1;
};

void *NewBatch(const void *instance, const isa::BatchPolicy &policy) {
    const Instance *in = static_cast<const Instance *>(instance);
    if (in == nullptr || in->model.car == nullptr || in->model.collision == nullptr || in->model.race == nullptr) return nullptr;
    BatchBase *b = nullptr;
    if (policy.lanes == 16) b = new (std::nothrow) Batch<16>(*in, policy);
    else if (policy.lanes == 8) b = new (std::nothrow) Batch<8>(*in, policy);
    return b;
}
void DestroyBatch(void *batch) { delete static_cast<BatchBase *>(batch); }
void RunBatch(void *batch, const isa::BatchCallbacks &cb, isa::BatchStats *stats) {
    static_cast<BatchBase *>(batch)->Run(cb, stats);
}

// ---- selftest ------------------------------------------------------------------------
struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    void U32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            h ^= (v >> (8 * i)) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    void F32(float f) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) b = 0x7fc00000u;
        U32(b);
    }
};

struct Lcg {
    std::uint64_t s;
    explicit Lcg(std::uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    std::uint32_t Next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(s >> 32);
    }
    float Uniform(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(Next() >> 8) * (1.0f / 16777216.0f));
    }
};

template <int W>
void MathAtWidth(Fnv &h) {
    using F = math::vf<W>;
    alignas(64) float x[W], y[W], out[W];
    static float all[6][1024];
    Lcg rng(3);
    for (int block = 0; block < 1024 / W; ++block) {
        for (int i = 0; i < W; ++i) {
            x[i] = rng.Uniform(-12.0f, 12.0f);
            y[i] = rng.Uniform(-1.0f, 1.0f);
        }
        const F vx = F::load(x), vy = F::load(y);
        const F r[6] = {math::sin(vx), math::atan2(vy, vx), math::exp(vy * F(4.0f)), math::sqrt(math::abs(vx)),
                        math::asin(vy), vx * vy + vy / (vx + F(13.0f))};
        for (int f = 0; f < 6; ++f) {
            r[f].store(out);
            for (int i = 0; i < W; ++i) all[f][block * W + i] = out[i];
        }
    }
    for (auto &row : all)
        for (float v : row) h.F32(v);
}

void HashRaw(Fnv &h, const std::vector<exact::Raw> &raw) {
    h.U32(static_cast<std::uint32_t>(raw.size()));
    for (const exact::Raw &r : raw) {
        for (const exact::V *v : {&r.P, &r.N, &r.Sep, &r.K}) {
            h.F32(v->x);
            h.F32(v->y);
            h.F32(v->z);
        }
        h.U32(r.primary ? 1u : 0u);
        h.U32(r.tri);
        h.U32(r.shape);
    }
}

std::uint64_t Selftest() {
    math::FpGuard guard;
    Fnv h;
    // Lane math at every width: native registers or pairs, per this ISA.
    MathAtWidth<1>(h);
    MathAtWidth<4>(h);
    MathAtWidth<8>(h);
    MathAtWidth<16>(h);
    // Exact-fast test: per-pair Collide and the packet path (AVX-512 lanes
    // in the AVX-512 variant, the per-pair fallback elsewhere).
    Lcg rng(9);
    std::vector<exact::Frame> frames;
    std::vector<exact::FastTri> tris;
    std::vector<exact::TriVerts> verts;
    std::vector<exact::Pair> pairs;
    for (std::uint32_t t = 0; t < 256u; ++t) {
        exact::TriVerts tv;
        for (int k = 0; k < 3; ++k) tv.v[k] = {rng.Uniform(-2.0f, 2.0f), rng.Uniform(-0.2f, 0.2f), rng.Uniform(-2.0f, 2.0f)};
        exact::FastTri ft;
        exact::MakeFastTri(tv.v, &ft);
        verts.push_back(tv);
        tris.push_back(ft);
    }
    const float meshIso[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    for (std::uint32_t f = 0; f < 64u; ++f) {
        const float a = rng.Uniform(-3.14159f, 3.14159f);
        const float c = math::cos(a), s = math::sin(a);
        const float ellR[9] = {c, 0, -s, 0, 1, 0, s, 0, c};
        const float ellT[3] = {rng.Uniform(-1.5f, 1.5f), rng.Uniform(-0.3f, 0.5f), rng.Uniform(-1.5f, 1.5f)};
        const float radii[3] = {rng.Uniform(0.2f, 0.6f), rng.Uniform(0.2f, 0.6f), rng.Uniform(0.2f, 0.6f)};
        frames.push_back(exact::MakeFrame(ellR, ellT, radii, meshIso, 2.2f));
        for (std::uint32_t t = 0; t < 256u; t += 7u) pairs.push_back({f, t, t, f & 7u});
    }
    std::vector<exact::Raw> single, packets;
    for (const exact::Pair &p : pairs) {
        const std::size_t before = single.size();
        exact::Collide(frames[p.frame], tris[p.modelTri], verts[p.modelTri], p.tri, &single);
        for (std::size_t i = before; i < single.size(); ++i) single[i].shape = static_cast<std::uint8_t>(p.shape);
    }
    exact::CollidePackets(frames.data(), tris.data(), verts.data(), pairs.data(), pairs.size(), &packets);
    HashRaw(h, single);
    HashRaw(h, packets);
    return h.h;
}

const isa::TickTable kTable = {
        static_cast<isa::Kind>(TW_ISA_KIND),
        TW_ISA_KIND == 0 ? "baseline" : TW_ISA_KIND == 1 ? "avx2" : "avx512",
        LayoutHere,
        Create,
        Destroy,
        Step,
        StepTicks,
        NewCache,
        DestroyCache,
        NewBatch,
        DestroyBatch,
        RunBatch,
        Selftest,
};

}  // namespace

extern "C" const tw::isa::TickTable *TW_ISA_ENTRY() { return &kTable; }
