// Broad-phase / collision candidates measured by tw_bench (M2 bake-off).
//
// Every candidate produces, per substep, the (shape, triangle) pairs that go
// to the exact test (ExactTest). Pairs are encoded as (shape << 28) | triId.
// All filters are conservative: closed-box tests on boxes inflated by kEps,
// and cache containment is tested on the same inflated boxes (the inflated-box
// containment fix, archA-on-designB.md item 4).
#ifndef TW_CANDIDATES_H
#define TW_CANDIDATES_H

#include <cmath>
#include <cstdint>
#include <vector>

#include "tw_accel.h"
#include "tw_exact.h"
#include "tw_world.h"

namespace tw {

constexpr float kEps = 1e-3f;  // conservative inflation of shape boxes (m)
inline std::uint32_t PairCode(int s, std::uint32_t tri) { return (static_cast<std::uint32_t>(s) << 28) | tri; }
inline int PairShape(std::uint32_t p) { return static_cast<int>(p >> 28); }
inline std::uint32_t PairTri(std::uint32_t p) { return p & 0x0fffffffu; }

struct Frame {  // per-substep derived shape data
    ShapeMap sm[kShapes];
    Box3 inflated[kShapes];
    Box3 unionBox;
    float speed;
};

inline void MakeFrame(const Substep &s, Frame *f) {
    f->unionBox = EmptyBox();
    for (int k = 0; k < kShapes; ++k) {
        MakeShapeMap(s.shapes[k], &f->sm[k]);
        f->inflated[k] = Inflate(f->sm[k].box, kEps);
        f->unionBox = Union(f->unionBox, f->inflated[k]);
    }
    f->speed = std::sqrt(s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] + s.vel[2] * s.vel[2]);
}

struct Stats {
    double pairs = 0, refreshes = 0, cacheTris = 0, exactCalls = 0, overflow256 = 0, wakes = 0;
};

// (1) BVH4 query of the union box every substep, then per-shape box filter.
struct BvhDirect {
    const World *w; const Bvh4 *bvh;
    void Reset() {}
    void Run(const Substep &, const Frame &f, std::vector<std::uint32_t> &out, Stats &st) {
        bvh->Query(f.unionBox, [&](std::uint32_t id) {
            const Tri &t = w->tris[id];
            const Box3 tb = {{t.lo[0], t.lo[1], t.lo[2]}, {t.hi[0], t.hi[1], t.hi[2]}};
            for (int s = 0; s < kShapes; ++s)
                if (Overlap(tb, f.inflated[s])) out.push_back(PairCode(s, id));
        });
        st.pairs += static_cast<double>(out.size());
    }
};

// (2) Loose grid query of the union box every substep, same filter.
struct GridDirect {
    const World *w; const LooseGrid *grid;
    void Reset() {}
    void Run(const Substep &, const Frame &f, std::vector<std::uint32_t> &out, Stats &st) {
        grid->Query(f.unionBox, [&](std::uint32_t id) {
            const Tri &t = w->tris[id];
            const Box3 tb = {{t.lo[0], t.lo[1], t.lo[2]}, {t.hi[0], t.hi[1], t.hi[2]}};
            for (int s = 0; s < kShapes; ++s)
                if (Overlap(tb, f.inflated[s])) out.push_back(PairCode(s, id));
        });
        st.pairs += static_cast<double>(out.size());
    }
};

// (3) Design B: per-candidate cache of the union box + speed margin, refreshed
// by a BVH4 query, stored as SoA tiles of 16; per substep tile x union, then
// tile x 8 shapes.
struct TileCache {
    const World *w; const Bvh4 *bvh;
    Box3 box = EmptyBox();
    bool valid = false;
    std::vector<std::uint32_t> ids;
    std::vector<float> lo[3], hi[3];
    void Reset() { valid = false; }
    void Run(const Substep &, const Frame &f, std::vector<std::uint32_t> &out, Stats &st) {
        if (!valid || !Contains(box, f.unionBox)) {
            const float m = 0.25f + f.speed * 0.01f;
            box = Inflate(f.unionBox, m);
            ids.clear();
            bvh->Query(box, [&](std::uint32_t id) { ids.push_back(id); });
            const std::size_t n = (ids.size() + 15) / 16 * 16;
            for (int a = 0; a < 3; ++a) { lo[a].assign(n, 1e30f); hi[a].assign(n, -1e30f); }
            for (std::size_t k = 0; k < ids.size(); ++k)
                for (int a = 0; a < 3; ++a) { lo[a][k] = w->tris[ids[k]].lo[a]; hi[a][k] = w->tris[ids[k]].hi[a]; }
            valid = true;
            st.refreshes += 1;
            if (ids.size() > 256) st.overflow256 += 1;
        }
        st.cacheTris += static_cast<double>(ids.size());
        const std::size_t n = lo[0].size();
        for (std::size_t base = 0; base < n; base += 16) {
            std::uint32_t um = 0;
            for (int l = 0; l < 16; ++l) {
                const std::size_t k = base + l;
                const bool ov = lo[0][k] <= f.unionBox.hi[0] && f.unionBox.lo[0] <= hi[0][k] && lo[1][k] <= f.unionBox.hi[1] &&
                                f.unionBox.lo[1] <= hi[1][k] && lo[2][k] <= f.unionBox.hi[2] && f.unionBox.lo[2] <= hi[2][k];
                um |= static_cast<std::uint32_t>(ov) << l;
            }
            if (!um) continue;
            for (int s = 0; s < kShapes; ++s) {
                const Box3 &q = f.inflated[s];
                std::uint32_t sm = 0;
                for (int l = 0; l < 16; ++l) {
                    const std::size_t k = base + l;
                    const bool ov = lo[0][k] <= q.hi[0] && q.lo[0] <= hi[0][k] && lo[1][k] <= q.hi[1] && q.lo[1] <= hi[1][k] &&
                                    lo[2][k] <= q.hi[2] && q.lo[2] <= hi[2][k];
                    sm |= static_cast<std::uint32_t>(ov) << l;
                }
                sm &= um;
                while (sm) { const int l = __builtin_ctz(sm); sm &= sm - 1; out.push_back(PairCode(s, ids[base + l])); }
            }
        }
        st.pairs += static_cast<double>(out.size());
    }
};

// (4) Coherent per-shape pair cache with conservative sleep.
// Each shape keeps the triangles overlapping its box + speed margin. Each pair
// carries a wake odometer value: the pair cannot produce a contact before the
// shape's odometer (sum of |Δcentre| + ||ΔM||_F per substep, an upper bound on
// how much any reach margin can shrink) passes it. Awake pairs get the
// world-space reach margin (plane + 3 edge prisms, subtract-first); only pairs
// with margin <= 0 go to the exact test. Exact-preserving by construction.
struct Coherent {
    const World *w; const Bvh4 *bvh; const std::vector<TriEdges> *edges = nullptr;
    struct Pair { std::uint32_t tri; double wake; };
    struct ShapeCache { Box3 box = EmptyBox(); bool valid = false; std::vector<Pair> pairs; double odo = 0; ShapeMap prev; bool havePrev = false; };
    ShapeCache sc[kShapes];
    void Reset() { for (auto &c : sc) { c.valid = false; c.havePrev = false; c.odo = 0; c.pairs.clear(); } }
    static float FrobDiff(const ShapeMap &a, const ShapeMap &b) {
        float s = 0;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { const float d = a.M[i][j] - b.M[i][j]; s += d * d; }
        return std::sqrt(s);
    }
    void Run(const Substep &, const Frame &f, std::vector<std::uint32_t> &out, Stats &st) {
        for (int s = 0; s < kShapes; ++s) {
            ShapeCache &c = sc[s];
            const ShapeMap &m = f.sm[s];
            if (c.havePrev) {
                const float dx = m.t[0] - c.prev.t[0], dy = m.t[1] - c.prev.t[1], dz = m.t[2] - c.prev.t[2];
                c.odo += static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz) + FrobDiff(m, c.prev)) * 1.0001 + 1e-7;
            }
            c.prev = m; c.havePrev = true;
            if (!c.valid || !Contains(c.box, f.inflated[s])) {
                const float margin = 0.25f + f.speed * 0.01f;
                c.box = Inflate(f.inflated[s], margin);
                c.pairs.clear();
                bvh->Query(c.box, [&](std::uint32_t id) { c.pairs.push_back({id, -1e300}); });
                c.valid = true;
                st.refreshes += 1.0 / kShapes;
            }
            st.cacheTris += static_cast<double>(c.pairs.size());
            for (Pair &p : c.pairs) {
                if (p.wake > c.odo) continue;
                st.wakes += 1;
                const float mu = edges ? ReachMarginPre(m, w->tris[p.tri], w->extra[p.tri], (*edges)[p.tri], kEps)
                                       : ReachMargin(m, w->tris[p.tri], w->extra[p.tri], kEps);
                if (mu > 0.0f) { p.wake = c.odo + static_cast<double>(mu); continue; }
                p.wake = c.odo;  // re-evaluate next substep
                out.push_back(PairCode(s, p.tri));
            }
        }
        st.pairs += static_cast<double>(out.size());
    }
};


// (5) Anchored coherent cache. Two levels: a car-level cache (union box +
// 1 m + 3 ticks of travel, one BVH4 query) and per-shape lists filtered from
// it (shape box + 0.25 m + 1 tick of travel). Per shape, a full evaluation
// computes every cached pair's reach margin; pairs with margin <= 0 are
// "active" (exact-tested every substep), the smallest positive margin is the
// shape's slack. Until the shape has moved |t - t_a| + ||M - M_a||_F >= slack
// away from the anchor pose (both terms bound how much any margin can change,
// path-independently), only the active pairs are tested. Exact-preserving.
struct Anchored {
    const World *w; const Bvh4 *bvh;
    Box3 carBox = EmptyBox(); bool carValid = false;
    std::vector<std::uint32_t> carTris;
    struct S { Box3 box = EmptyBox(); bool valid = false; std::vector<std::uint32_t> tris, active; float slack = -1.0f; float ta[3]; float Ma[3][3]; };
    S sc[kShapes];
    void Reset() { carValid = false; for (auto &c : sc) { c.valid = false; c.slack = -1.0f; } }
    void Run(const Substep &, const Frame &f, std::vector<std::uint32_t> &out, Stats &st) {
        if (!carValid || !Contains(carBox, f.unionBox)) {
            carBox = Inflate(f.unionBox, 1.0f + f.speed * 0.03f);
            carTris.clear();
            bvh->Query(carBox, [&](std::uint32_t id) { carTris.push_back(id); });
            carValid = true;
            st.refreshes += 1;
            for (auto &c : sc) c.valid = false;
        }
        st.cacheTris += static_cast<double>(carTris.size());
        for (int s = 0; s < kShapes; ++s) {
            S &c = sc[s];
            const ShapeMap &m = f.sm[s];
            bool eval = false;
            if (!c.valid || !Contains(c.box, f.inflated[s])) {
                c.box = Inflate(f.inflated[s], 0.25f + f.speed * 0.01f);
                if (!Contains(carBox, c.box)) {  // clip: keep shape box inside the car cache
                    for (int a = 0; a < 3; ++a) { c.box.lo[a] = std::fmax(c.box.lo[a], carBox.lo[a]); c.box.hi[a] = std::fmin(c.box.hi[a], carBox.hi[a]); }
                }
                c.tris.clear();
                for (std::uint32_t id : carTris) {
                    const Tri &t = w->tris[id];
                    const Box3 tb = {{t.lo[0], t.lo[1], t.lo[2]}, {t.hi[0], t.hi[1], t.hi[2]}};
                    if (Overlap(tb, c.box)) c.tris.push_back(id);
                }
                c.valid = true; eval = true;
                st.overflow256 += 1.0 / kShapes;  // reused: per-shape list rebuilds
            } else {
                const float dx = m.t[0] - c.ta[0], dy = m.t[1] - c.ta[1], dz = m.t[2] - c.ta[2];
                float fr = 0;
                for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { const float d = m.M[i][j] - c.Ma[i][j]; fr += d * d; }
                const float moved = (std::sqrt(dx * dx + dy * dy + dz * dz) + std::sqrt(fr)) * 1.0001f + 1e-6f;
                eval = !(moved < c.slack);
            }
            if (eval) {
                st.wakes += 1;
                c.active.clear();
                float slack = 1e30f;
                for (std::uint32_t id : c.tris) {
                    const float mu = ReachMargin(m, w->tris[id], w->extra[id], kEps);
                    if (mu <= 0.0f) c.active.push_back(id);
                    else if (mu < slack) slack = mu;
                }
                c.slack = slack;
                for (int a = 0; a < 3; ++a) c.ta[a] = m.t[a];
                for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) c.Ma[i][j] = m.M[i][j];
            }
            for (std::uint32_t id : c.active) out.push_back(PairCode(s, id));
        }
        st.pairs += static_cast<double>(out.size());
    }
};

}  // namespace tw

#endif
