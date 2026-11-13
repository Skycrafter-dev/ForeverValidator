// Architect C: candidate memo for the rival stage (cc_stage.h), a variant of the force agent's
// tw_car_query.h with the same per-shape survivors (same box test, same world-space prefilter and
// tolerance), organised so that a refill and a call touch less:
//   - refill = one BVH query and a sort into FV order, no plane rotation;
//   - the four world-space prefilter planes of an entry are made the first time the entry
//     reaches a shape box (most memo entries never do);
//   - per call, the memo is box-tested against the union of the 8 shape boxes first (16 lanes),
//     then only those entries against the 8 shapes;
#ifndef CC_MEMO_H
#define CC_MEMO_H

#include <cstdint>
#include <vector>

#include "tw_accel.h"
#include "tw_collide.h"
#include "tw_car_query.h"
#include "tw_exact_fast.h"
#include "tw_world.h"
#include "tw_world_pack.h"

namespace tw::collide_c {

struct Memo {
    Box3 box = EmptyBox();
    float origin[3] = {0.0f, 0.0f, 0.0f};
    std::uint32_t n = 0;                  // entries (FV order); arrays are capacities, padded to 16
    std::vector<std::uint64_t> keys;      // refill scratch: (fvOrder, tri)
    std::vector<float> lo[3], hi[3];      // triangle AABBs (padding: empty boxes)
    std::vector<std::uint32_t> fv;        // TriInfo::fvOrder
    std::vector<std::uint32_t> tri;       // refill scratch (BVH path): world triangle ids
    std::vector<cq::Ref> ref;             // instance, global model triangle, model, world triangle id (local)
    std::vector<exact::FastTri> planes;   // world-space prefilter planes (offsets relative to origin), lazy
    std::vector<std::uint8_t> ready;      // planes made
    bool allReady = false;                // every entry's planes made at the refill
    // an entry's vertices rotated to world, R_m v (FV stored order), and its instance translation t_m,
    // made with its planes when `verts` (per pack model triangle) is set: the noise-level narrow
    // phase's input. Independent of the memo (origin, box), so the output is cache-invariant.
    struct WVerts { float v[9]; float t[3]; };
    std::vector<WVerts> wv;
    const exact::TriVerts *verts = nullptr;
    float maxD1 = 0.0f, maxT1 = 0.0f, maxModelAbs = 0.0f;   // tolerance terms over the memo's instances (cq::Memo)
    // per-call scratch
    std::vector<std::uint32_t> near;
    std::vector<float> nlo[3], nhi[3];
    std::vector<std::uint32_t> pairEntry, pairShape;
    std::uint64_t planesMade = 0, nearEntries = 0, pairs = 0;   // counters
    std::uint64_t pairsOf[8] = {}, survOf[8] = {}, gated = 0;
    std::uint64_t fine[8] = {};   // CC_FINE: TSC per filter sub-phase
};

// The world's triangles in FV detection order (rank = TriInfo::fvOrder) with an implicit 16-ary tree over
// runs of 16 consecutive ranks: FV's order follows its record bintree and mesh octrees, so runs are
// spatially compact. A query visits children in order and emits the ranks whose AABB overlaps the box in
// increasing order: the memo needs no sort. Level 1 runs are cut where the instance changes. Each node keeps its
// children's boxes in one aligned block (6 x 64 bytes, SoA within the block, empty boxes past the last child), so
// a visited node is one 16-lane test on 6 adjacent cache lines.
class FvBvh {
public:
    struct alignas(64) Block { float lo[3][16], hi[3][16]; };
    void Build(const world::WorldPackView &view);
    std::uint32_t Size() const { return n_; }
    // f(first, block, mask) per leaf block in rank order: the triangles of ranks first + lane (lane a set bit of mask,
    // increasing) overlap q (closed test), their AABBs in that lane of block; every overlapping triangle once
    template <class F> void Query(const Box3 &q, F &&f) const {
        if (n_ == 0u) return;
        unsigned mk = Test16(root_, q) & ((rootCount_ >= 16u ? 0x10000u : (1u << rootCount_)) - 1u);
        if (top_ == 0) { if (mk) f(0u, root_, mk); return; }
        for (; mk; mk &= mk - 1u) Visit(top_, static_cast<std::uint32_t>(__builtin_ctz(mk)), q, f);
    }
    std::vector<cq::Ref> ref;             // per rank: instance, global model triangle, model, world triangle id (local)
    struct Level {                        // level k >= 1: node j's children are [first[j], first[j] + count[j]) of level k - 1,
        std::vector<std::uint32_t> first; // their boxes block[j]
        std::vector<std::uint8_t> count;
        std::vector<Block> block;
        std::uint32_t n = 0;
    };
private:
    std::vector<Level> levels_;           // levels_[0]: the triangles (n only); above: runs of up to 16 consecutive nodes
    Block root_;                          // the top level's nodes (or the triangles of a world with at most 16)
    std::uint32_t rootCount_ = 0;
    int top_ = 0;
    std::uint32_t n_ = 0;
    static unsigned Test16(const Block &B, const Box3 &q);
    template <class F> void Visit(int level, std::uint32_t j, const Box3 &q, F &f) const {
        const Level &L = levels_[static_cast<std::size_t>(level)];
        const Block &B = L.block[j];
        const std::uint32_t first = L.first[j], count = L.count[j];
        unsigned mk = Test16(B, q) & ((count >= 16u ? 0x10000u : (1u << count)) - 1u);
        if (level == 1) { if (mk) f(first, B, mk); return; }
        for (; mk; mk &= mk - 1u) Visit(level - 1, first + static_cast<std::uint32_t>(__builtin_ctz(mk)), q, f);
    }
};

// fv: refill through the FV-ordered tree (else the BVH and a sort); fast (with fv, AVX-512): every entry's
// planes at the refill (16 per pass) instead of lazily.
void Refill(const world::WorldPackView &view, const Bvh4 &bvh, const float *modelAbs, const Box3 &box, const float origin[3], Memo *memo,
            const FvBvh *fv = nullptr, const exact::FastTri *fast = nullptr);

// The 8 shapes' query boxes and prefilter fields, lane k = shape k (SoA). lo and hi are contiguous (48 floats) and
// t holds 13 rows of 8 and one spare (DetectLanes16 writes them as 64-byte rows of two fields; 10 cache lines).
struct ShapeTable {
    alignas(64) float lo[3][8];
    float hi[3][8];
    alignas(64) float t[14][8];    // c0 c1 c2 (centre - memo origin), q0..q5, reach, low, tol, lazy; [13]: spare
};
enum { kTc0 = 0, kTq0 = 3, kTreach = 9, kTlow = 10, kTtol = 11, kTlazy = 12 };

// The stage's query boxes (each ellipsoid's exact AABB: h = sqrt(Q_aa), B's eps) and their union.
void ShapeBoxes(const collide::ShapePose shapes[8], ShapeTable *st, Box3 *u);
// cq::MakeShapeFilter's fields for the 8 shapes against this memo (same formula).
void ShapeFilters(const Memo &memo, const collide::ShapePose shapes[8], ShapeTable *st);

// Per shape: the memo entries whose AABB meets the shape box and that pass the world-space
// prefilter, in FV order. `u`: the union of the 8 shape boxes. `fast`: per pack model triangle.
// Body gate: a box holding the 4 body ellipsoids, oriented with the body shapes (the car's axes).
// An entry meeting a body shape box is first tested once against it: a triangle plane that puts
// the whole box beyond every body shape's prefilter bound (in front of it, outside an edge side
// plane, or behind the triangle plane) rejects the entry for all body shapes, as the per-shape
// prefilter would (the margin G covers its tolerance and slack and the gate's own rounding).
struct BodyGate {
    bool on = false;
    unsigned shapes = 0;   // bit k: body shape k
    float c[3];            // box centre - memo origin
    float u[3][3];         // box axes (world unit vectors)
    float e[3];            // half extents along them
    float G;
};
// The gate for the body shapes `bodyShapes` (bits) at these poses (call after ShapeFilters).
void MakeBodyGate(const Memo &memo, const collide::ShapePose shapes[8], const ShapeTable &st, unsigned bodyShapes, BodyGate *g);

// Per shape: the memo entries whose AABB meets the shape box and that pass the world-space
// prefilter, in FV order. `u`: the union of the 8 shape boxes. `fast`: per pack model triangle.
// Compact survivors (AVX-512 refills with all planes made): (memo entry, shape) in pair order, FV order within
// each shape. With a sink, Filter fills it instead of `out`.
struct Survivors {
    std::vector<std::uint32_t> e, k;
    std::uint32_t n = 0;
};
// The memo entries meeting the union box (Filter's pass 0), compressed with their boxes.
struct NearSet {
    std::vector<std::uint32_t> near;
    std::vector<float> lo[3], hi[3];
    std::uint32_t nn = 0;
};
// split (with a sink, AVX-512, all planes made): 1 = pass 0 only, into *split; 2 = the rest, from *split (made by
// a phase 1 call on the same memo and boxes). DetectLanes interleaves lanes: pass 0 of the next lane, then the
// rest of this one. 0 = both.
void Filter(const world::WorldPackView &view, const exact::FastTri *fast, Memo &memo, const ShapeTable &st, const Box3 &u, cq::Lists *out,
            const BodyGate &gate, Survivors *sink = nullptr, NearSet *split = nullptr, int phase = 0);

// Filter for the lanes of a batch pass at once (AVX-512, DetectLanes16): per lane l of `mask` (its memo all ready, no
// body gate), Filter's sink-path survivors, the same tests on the same pairs in the same order, appended lane after lane
// to out (out->s0[l], s1[l]: lane l's range; out->nn[l]: its near entries). Pass 0 runs for all lanes first, into one
// near buffer, then each lane's near blocks with the pass's constants in registers. False (nothing done): no AVX-512.
struct PassSurvivors {
    std::vector<std::uint32_t> e, k;
    std::uint32_t n = 0;
    std::uint32_t s0[16], s1[16], nn[16];
};
bool FilterPass16(Memo *const *memos, const ShapeTable *st, const Box3 *u, std::uint32_t mask, unsigned bodyShapes, PassSurvivors *out);

}  // namespace tw::collide_c

#endif
