// M3: the twin's exact collision stage (FV-free).
//
// Merged per-shape contacts of the 8 car ellipsoids against the world package
// (tw_world_pack.h), as FV's Reference detection produces them:
//   - shape world pose = SetMult(shape-to-car pose, car iso); car iso = (R(quat), pos);
//     wheel shapes use the rest surface pose with translation.y += -absorb[w]
//     (FV OffsetCurrentSurfaceY);
//   - broad phase: a BVH4 over the world triangle AABBs. One query per call covers the union
//     of the 8 ellipsoids' world AABBs, inflated; it is memoised in the caller's per-candidate
//     cache (ExactCache; per thread without one) while later unions stay inside it (exact: the
//     per-shape AABB filter then gives the same candidates as a query per shape, so the output
//     never depends on the memo);
//   - candidates in FV detection order (TriInfo::fvOrder), so each shape's raw
//     contacts come in FV's order;
//   - narrow phase: FV's GmCollision_Ellipsoid_Mesh in its own form, in the
//     surface's mesh-local frame (mesh_collision_queries.cpp: the mesh -> unit
//     sphere transform, SSphereMeshCollide::CollideTriangle, the transforms back
//     to world), operation for operation;
//   - FV's per-shape merge (SHmsSphereBufferContact::MergeAndAddToCollisions):
//     primaries in order, then the non-primaries no primary absorbs;
//   - output shape-major (shape 0..7), the response's input (RespContact):
//     peerZ / peerId from the pack's actor table, a trigger contact flagged
//     kRespTrigger with peerId = its trigger id; at most kRespMaxContacts.
//
// Detect reads exactly TwinState::pos, quat and absorb. The stage holds no
// per-candidate state: Detect is const, pure and thread-safe. Per-candidate
// state lives in ExactCache (step::CollisionCache), owned by the caller and
// result-invariant: the output is bit-identical with any cache content.
#ifndef TW_COLLIDE_H
#define TW_COLLIDE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stages.h"
#include "tw_accel.h"
#include "tw_car_query.h"
#include "tw_exact_fast.h"
#include "tw_patches.h"
#include "tw_world_pack.h"

namespace tw {
namespace collide {

// FV GmIso4: rotation columns basisX, basisY, basisZ, then translation.
struct Iso {
    float r[9];
    float t[3];
};

struct CarShape {
    Iso bodyPose;            // shape -> car (FV GetThisToRootTransfo of a body shape; CUDA bodyPose)
    float radii[3];          // ellipsoid radii (FV EllipsoidRadii = CUDA localBounds.halfExtents)
    std::int32_t wheel;      // canonical wheel index (FL, FR, RR, RL) of a wheel shape, -1 for a body shape
};

// The car's collision shapes in FV tree order (0-3 body, 4 FL, 5 FR, 6 RL, 7 RR).
struct CarShapes {
    static constexpr std::uint32_t kMagic = 0x53435754u;  // "TWCS"
    std::uint32_t magic = kMagic;
    std::uint32_t count = 8u;
    CarShape shape[8];
    Iso wheelRest[4];        // FV wheel restSurfacePose (car frame), canonical wheel order
};

// One ellipsoid at its world pose.
struct ShapePose {
    Iso world;
    float radii[3];
};

// Shape world poses from the car pose (exactly FV's composition).
void ShapePoses(const CarShapes &car, const float pos[3], const float quat[4], const float absorb[4], ShapePose out[8]);

// DetectShapes in steps, for callers that form some steps themselves (the batch lanes stage):
//   ShapeBoxes -> ExactCollision::UpdateMemo -> ShapeFilters -> ExactCollision::DetectFiltered
// The shapes' closed, inflated world AABBs (the BVH leaf test) and their union.
void ShapeBoxes(const ShapePose shapes[8], Box3 q[8], Box3 *u);
// Per shape the prefilter's frame against the (updated) memo: tw_car_query MakeShapeFilter.
void ShapeFilters(const cq::Memo &memo, const ShapePose shapes[8], const Box3 q[8], cq::ShapeFilter sf[8]);

// FV's per-shape merge of shape k's raw contacts (in FV order) and the RespContact fill appended to `out`
// (kRespMaxContacts cap: sets out.overflow). Returns the merged count (before the cap). scratch: reused.
std::size_t MergeShapeTo(const world::WorldPackView &view, const std::vector<exact::Raw> &raw, int k, std::vector<int> *scratch,
                         step::SubstepContacts &out);
std::size_t MergeShapeTo(const world::WorldPackView &view, const exact::Raw *raw, std::size_t n, int k, std::vector<int> *scratch,
                         step::SubstepContacts &out);
// The same with the output fields from the memo: Raw::tri is the memo entry.
std::size_t MergeShapeTo(const cq::Memo &memo, const exact::Raw *raw, std::size_t n, int k, std::vector<int> *scratch, step::SubstepContacts &out);

struct CollideStats {
    std::uint64_t calls = 0, candidates = 0, triangleTests = 0, rawContacts = 0, merged = 0, overflow = 0;
    std::uint64_t movingPairCandidates = 0, queriesTouchingMovingPair = 0;   // FV's moving-pair corpus ("Flags")
    std::uint64_t memoRefills = 0, memoReuses = 0, memoEntries = 0;           // union query memo (cache totals; entries: sum per call)
    std::uint64_t tBoxes = 0, tMemo = 0, tFilter = 0, tFrames = 0, tTests = 0, tMerge = 0, frames = 0;   // TSC split (stats pass)
    std::uint64_t fvFrames = 0, decReject = 0, decFace = 0, decFv = 0;      // FV frames built; Decide outcomes of the survivors
    std::uint64_t bodyCandidates = 0, bodySurvivors = 0, bodyRaw = 0, wheelSurvivors = 0, wheelRaw = 0;
    std::uint64_t absorbedSkips = 0;                                       // feature-only pairs FV's merge would drop, not run
    std::uint64_t patchTests = 0;                                          // merged patches: loop tests
    std::uint64_t wheelCalls = 0, closedWheelFaces = 0;                    // R7: wheel shapes with survivors / in closed form
};

// One call's pre-narrow-phase output, read-only and valid during the narrow phase: the 8 shapes' poses and
// prefilter frames, the candidate's memo (per entry: tri, Ref{instance, modelTri, model, local}, AABB, world
// planes, mesh-local vertices (verts), output fields (outf); per memo instance: iso (inst)) and per shape the
// survivors (memo indices, FV order).
struct PreNarrow {
    const ShapePose *shapes;          // [8]
    const cq::ShapeFilter *filters;   // [8]
    const cq::Memo *memo;
    const cq::Lists *lists;
    const world::WorldPackView *view;
};
// A narrow phase: appends the call's merged contacts to out (n = 0 on entry), shape-major, in FV's per-shape
// merge and order (MergeShapeTo). The default (null) is the stage's bit-exact one.
using NarrowFn = void (*)(const PreNarrow &pre, step::SubstepContacts &out, CollideStats *stats);

// Coplanar patches (tw_patches.h): off, or "merged": a patch is one element (its loop's FACE or feature),
// internal edges and per-member duplicates make no contacts (m2-broadphase.md §6; not FV's bits).
constexpr std::uint8_t kPatchOff = 0, kPatchMerged = 1;

struct CollideOptions {
    bool excludeMovingPair = false;   // measurement only: leave the moving-pair corpus (kInstMovingPair) out of the BVH
    float unionMargin = 0.25f;        // the per-thread union query covers the shapes' union box inflated by this (m)
    float leadPoses = 12.0f;          // ... and extended along the last call-to-call motion by this many calls
    int profileStop = 0;              // measurement only, wrong output: stop after 4 the memo, 3 the shape filters, 1 the filter,
                                      // 2 the frames and pairs
    bool checkFrames = false;         // test: compare every FV frame with exact::MakeFrameFv (FrameMismatches)
    bool compactMemo = true;          // the narrow phase and the output read the memo's per-entry copies
    bool laneSetup = true;            // DetectPose: shape poses, boxes and filters with the 8 shapes in lanes
    NarrowFn narrow = nullptr;        // another narrow phase after the filter (null: the bit-exact one)
    std::uint8_t patches = kPatchOff; // kPatchMerged: the bit-exact narrow phase with merged coplanar patches
    bool closedWheelFace = false;     // R7: a wheel on one coplanar patch gets FV's FACE in closed form (float noise)
    bool checkSetup = false;          // test: compare the lane setup with the scalar forms (SetupMismatches)
};

// ExactCollision's per-candidate cache (stages.h CollisionCache): the union query memo.
// Every use checks it against the current pose (and the owning stage), so any content
// gives the same output; Flush() only drops the working set.
class ExactCache final : public step::CollisionCache {
public:
    ExactCache();
    void Flush() override;
    std::size_t Bytes() const;   // heap held (grows to the largest working set seen)
    std::uint64_t refills = 0, reuses = 0;

    // the triangles whose AABB overlaps a box around the car, in FV order (tw_car_query.h), and the per-shape lists
    struct Memo {
        std::uint64_t owner = 0;            // ExactCollision id (0: cold)
        cq::Memo q;
        cq::Lists lists;                    // per call (scratch)
        float lastCenter[3] = {0, 0, 0};    // the previous call's union centre: the memo box extends along the motion
        bool haveLast = false;
    };
    Memo memo;
    // the last call's answer, keyed by the bits of the pose Detect reads (pos, quat, absorb) and the stage:
    // the same bits give the same contacts (the speculative policy's tick-start pass often repeats the
    // previous tick's last pose). Contract: whoever writes it (DetectPose, or another stage form over the
    // same ExactCollision) stores exactly DetectShapes' output for that pose, with owner = Id().
    struct Last {
        std::uint64_t owner = 0;            // ExactCollision id (0: none)
        std::uint32_t key[11] = {};
        step::SubstepContacts out;
    };
    Last last;
    std::uint64_t lastHits = 0;

    static const char kKind;            // CollisionCache::kind of this type
};

class ExactCollision final : public step::CollisionStage {
public:
    // The view (and the pack behind it) must outlive the stage.
    bool Init(const world::WorldPackView &view, const CarShapes &car, std::string *error, const CollideOptions &opt = {});

    void Detect(const model::TwinState &s, step::SubstepContacts &out) const override;
    std::unique_ptr<step::CollisionCache> NewCache() const override;
    // cache: an ExactCache (another kind counts as none), or null: a per-thread memo.
    void DetectCached(const model::TwinState &s, step::CollisionCache *cache, step::SubstepContacts &out) const override;
    // Same, from the three state fields Detect reads (batch runners: no TwinState needed).
    void DetectPose(const float pos[3], const float quat[4], const float absorb[4], step::SubstepContacts &out,
                    step::CollisionCache *cache = nullptr) const;
    // Core: contacts of 8 ellipsoids at given world poses (also used against recorded FV poses).
    void DetectShapes(const ShapePose shapes[8], step::SubstepContacts &out, CollideStats *stats = nullptr,
                      ExactCache *cache = nullptr) const;
    // Steps of DetectShapes (see ShapeBoxes): the memo check / refill for the union box u, and the rest after
    // the shape filters (Filter, frames, narrow phase, merge). cache: the candidate's (or a per-thread) cache.
    const cq::Memo &UpdateMemo(ExactCache &cache, const Box3 &u, CollideStats *stats = nullptr) const;
    void DetectFiltered(const ShapePose shapes[8], const Box3 &u, const cq::ShapeFilter sf[8], ExactCache &cache, step::SubstepContacts &out,
                        CollideStats *stats = nullptr) const;
    // The first form (a BVH4 query per shape, FV's form on every candidate): the reference for A/B checks.
    void DetectShapesRef(const ShapePose shapes[8], step::SubstepContacts &out, CollideStats *stats = nullptr) const;

    const CarShapes &Car() const { return car_; }
    // read-only parts for other stage forms over the same world (the batch lanes stage)
    const world::WorldPackView &View() const { return *view_; }
    const Bvh4 &Bvh() const { return bvh_; }
    const exact::FastTri *Fast() const { return fast_.data(); }       // per pack model triangle
    const float *ModelAbs() const { return modelAbs_.data(); }          // per model
    std::uint64_t Id() const { return id_; }                            // ExactCache owner key
    const CollideOptions &Options() const { return opt_; }
    std::uint64_t FrameMismatches() const { return frameMismatches_; }   // CollideOptions::checkFrames
    std::uint64_t SetupMismatches() const { return setupMismatches_; }   // CollideOptions::checkSetup
    std::size_t Bytes() const { return bvh_.Bytes(); }

private:
    const world::WorldPackView *view_ = nullptr;
    CarShapes car_{};
    Bvh4 bvh_;
    CollideOptions opt_{};
    std::uint64_t id_ = 0;                       // memo key (unique per Init)
    std::vector<exact::FastTri> fast_;           // per pack model triangle (ModelTris index)
    std::vector<exact::TriVerts> triVerts_;      // per pack model triangle: its mesh-local vertices, FV order
    patch::Set patches_;                         // CollideOptions::patches
    std::vector<float> modelAbs_;                // per model: max |vertex coordinate| (prefilter tolerance)
    mutable std::atomic<std::uint64_t> frameMismatches_{0};   // checkFrames test mode
    mutable std::atomic<std::uint64_t> setupMismatches_{0};   // checkSetup test mode
    struct LaneConsts {   // the 8 shapes' constants, SoA (lane = shape)
        alignas(32) float lr[9][8], lt[3][8], rad[3][8], r2[3][8], rmax[8], rmin[8], ratio[8];
        int wheel[8];
    } lane_{};
    void LaneShapes(const float pos[3], const float quat[4], const float absorb[4], ShapePose shapes[8], Box3 q[8], Box3 *u, float wr[9][8],
                    float wt[3][8]) const;
    void LaneFilters(const cq::Memo &memo, const float wr[9][8], const float wt[3][8], const Box3 q[8], cq::ShapeFilter sf[8]) const;
    bool ClosedWheelFace(const ShapePose &sp, const cq::ShapeFilter &sf, const cq::Memo &m, const cq::Lists &lists, int k,
                         model::RespContact *out) const;
};

}  // namespace collide
}  // namespace tw

#endif
