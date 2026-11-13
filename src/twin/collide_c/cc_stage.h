// Architect C: rival exact collision stage (drop-in for tw::collide::ExactCollision).
//
// Same contract as src/twin/collide/tw_collide.h: the merged per-shape contacts of the 8 car
// ellipsoids, FV's bits, shape-major RespContact; Detect is const and pure; the per-candidate
// state lives in CcCache (step::CollisionCache), owned by the caller and result-invariant (the
// output is bit-identical with any cache content, cold, stale or flushed every call).
//
// The pipeline differs from B's stage only in how the same FV-form work is organised:
//   - candidates: the force agent's world-space memo + prefilter (tw_car_query.h) reorganised
//     (cc_memo.h: refill without sort or plane rotation, planes made lazily, union pre-test,
//     the 8 shape setups in SSE lanes);
//   - FV frames: all (shape, instance) frames of the call in SSE lanes (4 per instruction,
//     each lane FV's scalar operations, so the same bits);
//   - narrow phase: unit-space vertices, FV's normal, the squared-form decision and the FACE
//     payload in SSE lanes over all survivors of the call (shape-major, so each shape's raw
//     contacts keep FV order); FV's form (features) scalar per pair;
//
// The noise path (the product's stage "c"), AVX-512 builds, as of the 2026-11-09 collision round
// (docs/twin/search-throughput.md "Collision round"; the same output bits as before, c-legacy
// keeps the previous pipeline for A/B):
//   - the front: in the batch runner's lanes (DetectLanes16, on its pose rows) with lane = candidate,
//     16 candidates per pass (shape poses, the narrow phase's rows, boxes and union, memo checks,
//     prefilter fields, transposed into each lane's ShapeTable); single calls: PrepareLane, the same
//     operations with lane = shape (8 lanes);
//   - memo refill from FvBvh, a tree over the triangles in FV order (no sort), every entry's
//     prefilter planes made at the refill in 16 lanes;
//   - Filter: the near entries' planes transposed once, the body shapes' triangle-plane test
//     first, then (entry, shape) pairs in 16-lane packets; survivors handed over compact; in the
//     batch lanes entered once per pass (FilterPass16: pass 0 of lane i + 1 before lane i's near
//     blocks, survivors of all lanes in one list);
//   - narrow phase (Narrow16): each survivor its own lane of a 16-lane packet (its own FV
//     meshToUnit), and DetectLanes16 packs the survivors of all the pass's candidates together;
//     the merge from the packets' results with per-triangle output records (OutRec): lanes whose
//     hits are all FACEs in shape order straight from the packets' masks, the others MergeFlat;
//     each contact one masked store (FillOut16);
//   - optional (FOREVERVALIDATOR_TWIN_FREE_TICKS, batch::Policy::freeTicks, not on by default):
//     contact-free proofs for the region a lane's shapes sweep (DetectLanes16 empty / ProveLanes16).
#ifndef CC_STAGE_H
#define CC_STAGE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stages.h"
#include "tw_accel.h"
#include "tw_car_query.h"
#include "cc_memo.h"
#include "tw_collide.h"
#include "tw_exact_fast.h"
#include "tw_world_pack.h"

namespace tw::collide_c {

// A triangle's RespContact output fields (per world triangle; MergeOut reads one record per contact).
struct OutRec {
    float peerZ[3];
    std::uint32_t peerId;      // the trigger id for a trigger triangle, else the actor's peerId
    std::uint8_t zero, material, flags, pad;   // flags: kRespTrigger; zero, pad: 0 (bytes 12-19 are RespContact's peerId and
                                               // shape | material << 8 | flags << 16 without the shape and the primary bit)
};

struct StageOptions {
    float unionMargin = 0.25f;   // memo box: union box inflated by this (m) ...
    float leadPoses = 30.0f;     // ... and extended along the last call-to-call motion (0.25 / 18: +2-5% on
                                 // search candidates against 0.5 / 12, docs/twin/search-throughput.md) ...
    float memoMaxLead = 7.5f;    // ... by at most this per axis (m). 30 / 7.5 m against 18 / 4 m (2026-11-11, recorded
                                 // queue, 12-map halves): -0.9% / -0.5% instructions, refill share 8.9 -> 6.8% of cycles
    bool simdFrames = true;      // FV frames in SSE lanes
    bool simdNarrow = true;      // unit-space work + FACE payload in SSE lanes
    bool lastCache = true;       // the previous call's answer for the same pose bits
    // Noise-level narrow phase (lead decision 2026-11-08): FV's decisions on unit-space vertices mapped
    // straight from world, no FV frames; contacts within float noise of FV's, not its bits.
    bool noise = false;
    // Behaviour relaxations for the relaxation menu (architect A; noise path only, not FV's behaviour):
    int mergeMode = 0;           // 0 FV's per-shape merge; 1 no merge (every raw contact); 2 the deepest contact per shape
    bool skipBody = false;       // the body shapes are not tested at all (the ceiling of any body simplification)
    bool skipAbsorbed = true;    // skip FV's form for a feature-only pair the merge certainly drops (cc_stage.cpp)
    bool bodyGate = false;       // one test of each entry against the box holding the 4 body shapes (cc_memo.h)
    int stopAfter = 0;           // measurement only (wrong output): 1 after the memo, 2 after the filter, 3 after the frames
    // Noise path, AVX-512 builds: DetectLanes runs the narrow phase over all lanes' survivors in 16-lane packets,
    // each lane its own FV meshToUnit (the jobs' operations, so the same bits). False: DetectCached per lane.
    // (On one call, one partly filled packet is slower than the per-(shape, instance) jobs: its latency chain.)
    bool narrow16 = true;
    // The pipeline as of 2026-11-09 (A/B reference, the same output bits): shape poses and filters per call in SSE,
    // the BVH refill with a sort and lazy planes, the pair filter with transposes, the per-call narrow phase,
    // MergeOut through the world tables, no batch lanes. Stage name c-legacy.
    bool legacy = false;
};

struct StageStats {
    std::uint64_t calls = 0, lastHits = 0, refills = 0, survivors = 0, frames = 0, face = 0, fvForm = 0, reject = 0, raw = 0, merged = 0;
    std::uint64_t bodySurvivors = 0, bodyRaw = 0, bodyMerged = 0, skipped = 0, fvRuns = 0;
    std::uint64_t tBoxes = 0, tMemo = 0, tFilter = 0, tFrames = 0, tNarrow = 0, tMerge = 0;
};

class CcCache final : public step::CollisionCache {
public:
    CcCache();
    void Flush() override;
    std::size_t Bytes() const;

    std::uint64_t owner = 0;            // CcCollision id (0: cold)
    Memo m;
    cq::Lists lists;
    float lastCenter[3] = {0, 0, 0};
    bool haveLast = false;
    struct Last {                       // the last call's answer, keyed by the car pose's bits (pos, quat, absorb)
        std::uint64_t owner = 0;
        float key[11];
        step::SubstepContacts out;
    };
    Last last;
    std::uint64_t refills = 0, reuses = 0, lastHits = 0;

    static const char kKind;
};

class CcCollision final : public step::CollisionStage {
public:
    bool Init(const world::WorldPackView &view, const collide::CarShapes &car, std::string *error, const StageOptions &opt = {});

    void Detect(const model::TwinState &s, step::SubstepContacts &out) const override;
    std::unique_ptr<step::CollisionCache> NewCache() const override;
    void DetectCached(const model::TwinState &s, step::CollisionCache *cache, step::SubstepContacts &out) const override;
    void DetectPose(const float pos[3], const float quat[4], const float absorb[4], step::SubstepContacts &out,
                    step::CollisionCache *cache = nullptr) const;
    // Core (also on recorded / dumped shape poses). cache: a CcCache or null (per-thread memo). No last-answer
    // cache here (DetectPose and DetectLanes key it on the car pose).
    void DetectShapes(const collide::ShapePose shapes[8], step::SubstepContacts &out, CcCache *cache = nullptr,
                      StageStats *stats = nullptr) const;

    // Batch lanes (batch_runner.h LaneDetect's contract): per lane l of `mask`, out[l] = DetectCached(*poses[l],
    // caches[l]) bit for bit. The noise path's narrow phase runs over the survivors of all lanes in 16-lane packets.
    void DetectLanes(int lanes, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                     step::SubstepContacts *out) const;
    // The same with the poses as lane rows (row f of a field: lane l at [f][l]; 16 lanes, the batch runner's
    // TwinLanes<16> layout). The front (8 shape poses, boxes, their union, the memo checks, the prefilter fields) runs
    // with lane = candidate, 16 candidates per pass: PrepareLane's operations per (candidate, shape), so the same
    // bits. No last-answer cache (result-invariant). False: not batchable (no AVX-512, relaxations, or a lane
    // without a CcCache): the caller detects lane by lane.
    // empty: batch::LaneDetect's contact-free proofs (Idea 5): the lanes without a candidate and without a memo entry near
    // their shapes (proofs elsewhere fail more often than they hold). ProveLanes16, right after on the same thread: of
    // those in `ask`, the lanes whose shapes swept along w and grown by d (sweep rows w x, w y, w z, d) leave Filter
    // without a survivor too, their memo box containing the swept union.
    bool DetectLanes16(const float (*pos)[16], const float (*quat)[16], const float (*absorb)[16], std::uint32_t mask,
                       step::CollisionCache *const *caches, step::SubstepContacts *out, std::uint32_t *empty = nullptr) const;
    std::uint32_t ProveLanes16(const float (*sweep)[16], std::uint32_t ask, step::CollisionCache *const *caches) const;
    // The largest |local shape centre| + ||local rotation||_F x max radius over the 8 shapes (LaneDetect::FreeLever).
    float FreeLever() const;
    // Whether DetectLanes batches (else it is DetectCached per lane).
    bool LanesBatch() const;
    bool LanesBatchable() const;   // the noise path without relaxations (PrepareLane, the lean merge)

    const collide::CarShapes &Car() const { return car_; }
    const StageOptions &Options() const { return opt_; }

private:
    void Shapes(const collide::ShapePose shapes[8], step::SubstepContacts &out, CcCache &cc, StageStats *stats) const;
    // boxes, memo, shape filters and Filter: the per-shape survivors in cc.lists. False: stopped (stopAfter).
    bool Prepare(const collide::ShapePose shapes[8], CcCache &cc, StageStats *stats, std::uint64_t *t0, Survivors *sink = nullptr) const;
    // The front of one call with lane = shape (8 lanes; DetectPose): ShapePoses, ShapeBoxes and ShapeFilters' operations
    // per lane (DetectLanes16's per (candidate, shape), the same bits), the shape poses also as rows (AoS) for the narrow
    // phase; then memo and Filter.
    void PrepareLane(const float pos[3], const float quat[4], const float absorb[4], collide::ShapePose rows[8], CcCache &cc,
                     Survivors *sink) const;
    // Measurement (CC_STEADYSTATS in cc_stage.cpp; defined only in such a build): the steady-contact counters of one
    // DetectLanes16 pass, after its merge (t: the pass's phase TSC laps, rc: the car rotations, lane = candidate).
    void SteadyCount(std::uint32_t mask, step::CollisionCache *const *caches, const step::SubstepContacts *out, const std::uint64_t *t,
                     const float (*rc)[16]) const;
    // Measurement (CC_GRIDSTATS in cc_stage.cpp, cc_gridstats.inc; defined only in such a build): the static world grid
    // counters of one DetectLanes16 pass, after its merge; the grid forms' memory for this world (at Init); the filter's
    // TSC by shape group (before the pass's FilterPass16).
    void GridCount(std::uint32_t mask, step::CollisionCache *const *caches) const;
    void GridMem() const;
    void FilterTime(Memo *const *memos, std::uint32_t ready) const;
    // Measurement (CC_OSLABSTATS in cc_stage.cpp, cc_oslabstats.inc; defined only in such a build): the oriented slab
    // counters of one DetectLanes16 pass, after its merge; the filter's TSC with the slab-skippable entries removed.
    void OslabCount(std::uint32_t mask, step::CollisionCache *const *caches) const;
    void OslabTime(Memo *const *memos, std::uint32_t ready) const;
    struct LaneConst {   // the shapes' constants, lane k = shape k
        alignas(32) float lr[9][8], lt[3][8], rad[3][8], r2[3][8], rmax[8], rmin[8], ratio[8];
        std::int32_t wheel[8];
    };
    LaneConst lc_{};
    alignas(64) float irad_[3][16] = {};   // Narrow16: 1 / radius a of shape k & 7
    unsigned bodyShapes_ = 0;   // bit k: shape k is a body shape (Filter's body test)

    const world::WorldPackView *view_ = nullptr;
    collide::CarShapes car_{};
    StageOptions opt_{};
    std::uint64_t id_ = 0;
    Bvh4 bvh_;
    FvBvh fvb_;                             // the memo's refill query (FV order, no sort)
    std::vector<OutRec> outRec_;            // per world triangle
    std::vector<exact::FastTri> fast_;      // per pack model triangle
    std::vector<exact::TriVerts> verts_;    // per pack model triangle: mesh-local vertices, FV order
    std::vector<float> modelAbs_;           // per model: max |vertex coordinate|
};

// FOREVERVALIDATOR_TWIN_COLLIDE values that select C's stage, and their options: c (noise), c-exact,
// c-nomerge, c-deepest, c-nobody, c-legacy (the noise path's pipeline before 2026-11-09, same bits). False
// for any other value (B's stage).
bool ParseStageSwitch(const char *value, StageOptions *opt);

// The noise-level narrow phase as B's stage hook (tw_collide.h CollideOptions::narrow): the same
// kernel over the PreNarrow memo (per-entry vertices, instance isos, output fields).
namespace noise {
void Narrow(const collide::PreNarrow &pre, step::SubstepContacts &out, collide::CollideStats *stats);
}

}  // namespace tw::collide_c

#endif
