// M3 collision for batch lanes (src/twin/batch, Policy::detect): the detections of up to 16 lanes
// of one batch pass together. Each lane's contacts are ExactCollision's (tw_collide.h) bit for bit,
// cache effects included, so the lane's run is the single-run twin's.
//
// DetectLanes (default path): lanes at the same pose bits (search candidates from one snapshot,
// before their first change) are detected once; each distinct pose goes through the single-run
// stage with its lane's own cache (DetectCached); a copy lane gets that answer, and its cache the
// flush and last answer DetectPose would leave.
//
// The group path (SharedMemo(true), DetectLanesShared / DetectShapesLanes), measured slower than the
// default on real runs (docs/twin/m3-exact-fast.md §7):
//   1. per lane: the shape poses (ShapePoses), the stage's query boxes and their union;
//   2. candidate memos shared by the lanes (tw_car_query.h BuildMemo): a lane takes a memo whose box
//      contains its union, else one is refilled for it and the pending lanes that fit (union extent
//      at most kGroupExtent), or its own memo when no group forms;
//   3. per lane: cq::Filter (the stage's box test + the world-space support prefilter);
//   4. the survivors of all lanes and shapes in FV order with FV's frames (exact::FvRot shared per
//      call, exact::FvFrameFrom), decided in 16-pair packets across lanes and shapes
//      (exact::CollidePackets steps 2-3);
//   5. per (lane, shape): the stage's merge and contact fill.
// Every memo box contains the lane's query boxes, so every triangle the stage tests in a lane is a
// candidate here, in the same FV order; the prefilter and decisions only drop pairs FV's form rejects.
//
// One instance per runner (per thread and world): the memos are mutable state.
#ifndef TW_COLLIDE_LANES_H
#define TW_COLLIDE_LANES_H

#include <cstdint>
#include <string>
#include <vector>

#include "tw_accel.h"
#include "tw_car_query.h"
#include "tw_collide.h"
#include "tw_exact_fast.h"
#include "tw_world_pack.h"

namespace tw {
namespace collide {

class ExactCollisionLanes {
public:
    static constexpr int kMaxLanes = 16;
    static constexpr int kPool = 4;              // shared memos
    static constexpr float kMargin = 0.75f;      // memo box: the group union inflated by this (m)
    static constexpr float kGroupExtent = 12.0f; // max extent of a group's union per axis (m)
    static constexpr float kLead = 6.0f;         // memo box extended along the motion since its last refill, x calls
    static constexpr int kRot = 64;              // FV rotation parts shared per call (lanes x shapes x instances)

    // single: the stage whose results each lane reproduces (its caches are the lanes' caches).
    bool Init(const world::WorldPackView &view, const ExactCollision &single, std::string *error);

    // poses[l]: pos, quat, absorb of lane l (other fields unused); caches[l]: the lane's cache (may be null).
    void DetectLanes(int L, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                     step::SubstepContacts *out) const;

    // The group path (SharedMemo(true)): shared memos and cross-lane packets.
    void DetectLanesShared(int L, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                           step::SubstepContacts *out) const;
    // Core: lanes in `todo` at given shape poses (shapes[l]: lane l's 8; the stage's DetectShapes per lane); no cache.
    void DetectShapesLanes(int L, const ShapePose *const *shapes, std::uint32_t todo, step::SubstepContacts *out) const;

    struct Stats {
        std::uint64_t calls = 0, lanes = 0, lastHits = 0, memoRefills = 0, memoEntries = 0, survivors = 0, frames = 0, raw = 0, sharedRot = 0, dupLanes = 0;
        std::uint64_t tBoxes = 0, tMemo = 0, tShape = 0, tFilter = 0, tFrames = 0, tPackets = 0, tMerge = 0;   // TSC (profile mode)
    };
    const Stats &stats() const { return stats_; }
    void ResetStats() const { stats_ = Stats{}; }
    void Profile(bool on) const { profile_ = on; }   // TSC split in stats() (costs the timer reads)
    // DetectLanes' path for distinct poses: the group path (shared memos, cross-lane packets) or the single-run
    // stage per lane through its own cache (default; faster on the real runs measured, docs/twin/m10-batch.md).
    void SharedMemo(bool on) { sharedMemo_ = on; }

private:
    struct PoolMemo {
        cq::Memo q;
        bool have = false;
        std::uint64_t used = 0;   // last call that used it (LRU)
        float center[3] = {0, 0, 0};   // the union centre at the last refill
    };
    const world::WorldPackView *view_ = nullptr;
    const ExactCollision *single_ = nullptr;
    std::vector<exact::TriVerts> verts_;   // per pack model triangle (CollidePackets' vertex rows)
    mutable cq::ShapeConst sc_[8];
    mutable float scRadii_[8][3] = {};
    // mutable per-thread state
    mutable PoolMemo pool_[kPool];       // shared by grouped lanes
    mutable PoolMemo own_[kMaxLanes];    // a lane's own, when no group forms
    mutable exact::FvRot rot_[kRot];
    mutable int nRot_ = 0, nextRot_ = 0;
    mutable std::uint64_t tick_ = 0;
    mutable cq::Lists lists_[kMaxLanes];
    mutable std::vector<exact::Frame> frames_;
    mutable std::vector<exact::Pair> pairs_;
    mutable std::vector<exact::Raw> raw_;
    mutable std::vector<int> merged_;
    mutable Stats stats_;
    mutable bool profile_ = false;
    bool sharedMemo_ = false;
};

}  // namespace collide
}  // namespace tw


#if __has_include("batch_runner.h")
#include "batch_runner.h"
namespace tw {
namespace collide {
// The batch runner's hook (batch::Policy::detect): one per runner (per thread and world).
class LanesDetect final : public batch::LaneDetect {
public:
    explicit LanesDetect(const ExactCollisionLanes &s) : s_(s) {}
    void DetectLanes(int lanes, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                     step::SubstepContacts *out) const override {
        s_.DetectLanes(lanes, poses, mask, caches, out);
    }

private:
    const ExactCollisionLanes &s_;
};
}  // namespace collide
}  // namespace tw
#endif

#endif
