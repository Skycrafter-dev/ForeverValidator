// Twin GPU engine: the collision world's host build (tw_gpu_world.h), from the same world pack and car
// shapes as C's stage (cc_stage.cpp CcCollision::Init, cc_memo.cpp FvBvh::Build).
#include "tw_gpu_world.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "response.h"
#include "tw_collide.h"
#include "tw_exact_fast.h"
#include "tw_world_pack.h"

namespace tw::gpu {

namespace {

struct SBox {
    float lo[3], hi[3];
};
void Grow(SBox &b, const SBox &c) {
    for (int a = 0; a < 3; ++a) { b.lo[a] = std::min(b.lo[a], c.lo[a]); b.hi[a] = std::max(b.hi[a], c.hi[a]); }
}
SBox Empty() { return SBox{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}}; }
double Area(const SBox &b) {
    const double d0 = std::max(0.0, double(b.hi[0]) - b.lo[0]), d1 = std::max(0.0, double(b.hi[1]) - b.lo[1]),
                 d2 = std::max(0.0, double(b.hi[2]) - b.lo[2]);
    return d0 * d1 + d1 * d2 + d2 * d0;
}

// A binary BVH over the leaves (binned SAH on the boxes' centres), the build's intermediate.
struct BNode {
    SBox box;
    std::int32_t l = -1, r = -1;   // children (internal)
    std::int32_t prim = -1;        // a leaf's index (a single leaf)
    std::uint32_t minPrim = 0;     // the smallest leaf index below (children ordered by it: records come out nearly sorted)
};

std::int32_t BuildBinary(std::vector<BNode> &out, const std::vector<SBox> &boxes, std::uint32_t *idx, std::uint32_t n) {
    const std::int32_t me = static_cast<std::int32_t>(out.size());
    out.push_back(BNode{});
    SBox b = Empty(), cb = Empty();
    std::uint32_t mp = 0xFFFFFFFFu;
    for (std::uint32_t i = 0; i < n; ++i) {
        const SBox &x = boxes[idx[i]];
        Grow(b, x);
        for (int a = 0; a < 3; ++a) {
            const float c = 0.5f * (x.lo[a] + x.hi[a]);
            cb.lo[a] = std::min(cb.lo[a], c);
            cb.hi[a] = std::max(cb.hi[a], c);
        }
        mp = std::min(mp, idx[i]);
    }
    out[me].box = b;
    out[me].minPrim = mp;
    if (n == 1u) {
        out[me].prim = static_cast<std::int32_t>(idx[0]);
        return me;
    }
    // binned SAH on each axis of the centres' box (32 bins), the cheapest split; the median of the longest axis when
    // no axis has a usable split
    constexpr int kBins = 32;
    int bestAxis = -1, bestK = -1;
    double best = 1e300;
    auto centre = [&](std::uint32_t j, int a) { return 0.5f * (boxes[j].lo[a] + boxes[j].hi[a]); };
    auto bin = [&](std::uint32_t j, int a) {
        const float ext = cb.hi[a] - cb.lo[a];
        const int k = static_cast<int>(float(kBins) * ((centre(j, a) - cb.lo[a]) / ext));
        return std::min(kBins - 1, std::max(0, k));
    };
    for (int a = 0; a < 3; ++a) {
        if (!(cb.hi[a] - cb.lo[a] > 0.0f)) continue;
        SBox bb[kBins];
        std::uint32_t bc[kBins] = {};
        for (int k = 0; k < kBins; ++k) bb[k] = Empty();
        for (std::uint32_t i = 0; i < n; ++i) {
            const int k = bin(idx[i], a);
            ++bc[k];
            Grow(bb[k], boxes[idx[i]]);
        }
        double rightA[kBins];
        std::uint32_t rightN[kBins];
        SBox acc = Empty();
        std::uint32_t cnt = 0u;
        for (int k = kBins - 1; k > 0; --k) {
            Grow(acc, bb[k]);
            cnt += bc[k];
            rightA[k] = Area(acc);
            rightN[k] = cnt;
        }
        acc = Empty();
        cnt = 0u;
        for (int k = 0; k < kBins - 1; ++k) {   // split after bin k
            Grow(acc, bb[k]);
            cnt += bc[k];
            if (cnt == 0u || rightN[k + 1] == 0u) continue;
            const double cost = Area(acc) * cnt + rightA[k + 1] * rightN[k + 1];
            if (cost < best) { best = cost; bestAxis = a; bestK = k; }
        }
    }
    std::uint32_t mid = n / 2u;
    bool split = false;
    if (bestAxis >= 0) {
        std::uint32_t *p = std::partition(idx, idx + n, [&](std::uint32_t j) { return bin(j, bestAxis) <= bestK; });
        mid = static_cast<std::uint32_t>(p - idx);
        split = mid > 0u && mid < n;
    }
    if (!split) {   // no usable bin split: the median along the longest axis
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (cb.hi[a] - cb.lo[a] > cb.hi[axis] - cb.lo[axis]) axis = a;
        mid = n / 2u;
        std::nth_element(idx, idx + mid, idx + n, [&](std::uint32_t x, std::uint32_t y) {
            return centre(x, axis) < centre(y, axis) || (centre(x, axis) == centre(y, axis) && x < y);
        });
    }
    const std::int32_t l = BuildBinary(out, boxes, idx, mid);
    const std::int32_t r = BuildBinary(out, boxes, idx + mid, n - mid);
    out[me].l = l;
    out[me].r = r;
    return me;
}

// The spatial tree over the walk's leaves (runs of consecutive ranks, their NodeQs given; the FV tree stays the fallback
// at node 0): a binary BVH (binned SAH), collapsed into 16-wide nodes (a node's children: its binary children, the
// largest-area internal one opened until 16), laid out breadth first after the FV tree, a node's children side by side
// (a leaf child: its leaf NodeQ), children ordered by their smallest leaf. w->root: its root; w->stackNeed: the walk's
// pending entries at most (sum of children - 1 down a path).
template <class Quant>
void BuildSpatial(const std::vector<SBox> &leafBox, const std::vector<NodeQ> &leafQ, Quant quant, HostWorld *w) {
    const std::uint32_t nl = static_cast<std::uint32_t>(leafBox.size());
    std::vector<std::uint32_t> idx(nl);
    for (std::uint32_t j = 0; j < nl; ++j) idx[j] = j;
    std::vector<BNode> bn;
    bn.reserve(std::size_t(2) * nl);
    const std::int32_t broot = BuildBinary(bn, leafBox, idx.data(), nl);
    auto kidsOf = [&](std::int32_t b, std::int32_t kids[16]) {
        int k = 0;
        kids[k++] = bn[b].l;
        kids[k++] = bn[b].r;
        while (k < 16) {
            int best = -1;
            double ba = -1.0;
            for (int i = 0; i < k; ++i)
                if (bn[kids[i]].prim < 0) {
                    const double a = Area(bn[kids[i]].box);
                    if (a > ba) { ba = a; best = i; }
                }
            if (best < 0) break;
            const std::int32_t x = kids[best];
            kids[best] = bn[x].l;
            kids[k++] = bn[x].r;
        }
        std::sort(kids, kids + k, [&](std::int32_t x, std::int32_t y) { return bn[x].minPrim < bn[y].minPrim; });
        return k;
    };
    struct Item {
        std::int32_t b;
        std::uint32_t slot, need;   // the pending entries above it
    };
    const std::uint32_t base = static_cast<std::uint32_t>(w->qnodes.size());
    w->qnodes.push_back(NodeQ{});
    std::vector<Item> queue{Item{broot, base, 0u}};
    std::uint32_t need = 0u;
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        const Item it = queue[qi];
        std::int32_t kids[16];
        const int k = kidsOf(it.b, kids);
        const std::uint32_t first = static_cast<std::uint32_t>(w->qnodes.size());
        w->qnodes.resize(first + static_cast<std::uint32_t>(k));
        const std::uint32_t myNeed = it.need + static_cast<std::uint32_t>(k - 1);
        need = std::max(need, myNeed);
        for (int i = 0; i < k; ++i) {
            if (bn[kids[i]].prim >= 0) w->qnodes[first + static_cast<std::uint32_t>(i)] = leafQ[static_cast<std::size_t>(bn[kids[i]].prim)];
            else queue.push_back(Item{kids[i], first + static_cast<std::uint32_t>(i), myNeed});
        }
        NodeQ &Q = w->qnodes[it.slot];   // as the FV tree's (fillQ): the node's own box as the frame, the children's boxes in it
        Q = NodeQ{};
        const SBox &own = bn[it.b].box;
        for (int a = 0; a < 3; ++a) {
            Q.org[a] = own.lo[a];
            const float ext = own.hi[a] - own.lo[a];
            const float sc = ext > 0.0f ? 127.0f / ext : 0.0f;
            Q.scale[a] = std::isfinite(sc) ? sc : 0.0f;
        }
        Q.first = first;
        Q.meta = static_cast<std::uint32_t>(k);
        for (int a = 0; a < 3; ++a)
            for (int j = 0; j < 4; ++j) Q.hi[a][j] = 0x80808080u;   // every hi byte has bit 7 set (the walk's guard bit)
        for (int c = 0; c < k; ++c) {
            const SBox &cbx = bn[kids[c]].box;
            for (int a = 0; a < 3; ++a) {
                Q.lo[a][c / 4] |= quant(cbx.lo[a], Q.org[a], Q.scale[a]) << (8 * (c % 4));
                Q.hi[a][c / 4] |= quant(cbx.hi[a], Q.org[a], Q.scale[a]) << (8 * (c % 4));
            }
        }
    }
    w->root = base;
    w->stackNeed = need;
}

}  // namespace

bool BuildHostWorld(const world::WorldPackView &view, const collide::CarShapes &car, HostWorld *w, std::string *error) {
    if (car.magic != collide::CarShapes::kMagic || car.count != 8u) {
        if (error) *error = "gpu world: car shapes missing or not 8";
        return false;
    }
    *w = HostWorld{};
    for (int k = 0; k < 8; ++k) {   // CcCollision::Init's lane constants
        const collide::CarShape &sh = car.shape[k];
        const collide::Iso &local = sh.wheel >= 0 ? car.wheelRest[sh.wheel] : sh.bodyPose;
        for (int j = 0; j < 9; ++j) w->car.lr[k][j] = local.r[j];
        for (int a = 0; a < 3; ++a) {
            w->car.lt[k][a] = local.t[a];
            w->car.rad[k][a] = sh.radii[a];
            w->car.r2[k][a] = sh.radii[a] * sh.radii[a];
            w->car.irad[k][a] = 1.0f / sh.radii[a];
        }
        for (int a = 0; a < 3; ++a) {   // the half extent along car axis a: sqrt(sum_i rad_i^2 lr_{i,a}^2) (lr's columns i)
            double q = 0.0;
            for (int i = 0; i < 3; ++i) q += double(sh.radii[i]) * sh.radii[i] * (double(local.r[3 * i + a]) * local.r[3 * i + a]);
            w->car.hc[k][a] = std::nextafter(static_cast<float>(2.002 * std::sqrt(q)), 1e30f);
        }
        w->car.wheel[k] = sh.wheel;
    }
    w->k.dirEpsSq = exact::kDirEpsSq;
    w->k.collisionDistance = exact::kCollisionDistance;
    w->k.normalAlignment = 0.8660254f;   // cc_stage.cpp kNormalAlignment

    const std::uint32_t n = static_cast<std::uint32_t>(view.tris.size());
    w->ranks = n;
    w->rank.assign(n, RankRef{0u, 0u, 0u});
    std::vector<float> lo[3], hi[3];   // the current level's node boxes (level 0: the triangles, by rank)
    for (int a = 0; a < 3; ++a) { lo[a].assign(n, 1e30f); hi[a].assign(n, -1e30f); }
    for (std::uint32_t t = 0; t < n; ++t) {
        const world::TriInfo &ti = view.triInfo[t];
        const std::uint32_t r = ti.fvOrder;
        const world::Instance &in = view.instances[ti.instance];
        w->rank[r] = RankRef{ti.instance, view.models[in.model].firstTri + ti.modelTri, t};
        for (int a = 0; a < 3; ++a) { lo[a][r] = view.tris[t].lo[a]; hi[a][r] = view.tris[t].hi[a]; }
    }
    // levels bottom up (FvBvh::Build): level 1 = runs of up to 16 ranks cut where the instance changes, above =
    // runs of 16 nodes, until at most 16 nodes are left; then one root over them
    struct Level { std::vector<std::uint32_t> first, count; std::vector<NodeQ> q; };
    std::vector<Level> levels;
    // NodeQ: the node's own box (lo, hi) as the frame, the current level's boxes [first, first + count) as children
    auto quant = [](float x, float org, float scale) {
        const float f = std::floor((x - org) * scale);
        return static_cast<std::uint32_t>(std::min(std::max(f, 0.0f), 127.0f));
    };
    auto fillQ = [&](NodeQ &Q, std::uint32_t first, std::uint32_t count, const float own[2][3], bool leaf, std::uint32_t firstOut) {
        Q = NodeQ{};
        for (int a = 0; a < 3; ++a) {
            Q.org[a] = own[0][a];
            const float ext = own[1][a] - own[0][a];
            const float sc = ext > 0.0f ? 127.0f / ext : 0.0f;
            Q.scale[a] = std::isfinite(sc) ? sc : 0.0f;
        }
        Q.first = firstOut;
        Q.meta = count | (leaf ? 1u : 0u) << 8;
        for (int a = 0; a < 3; ++a)
            for (int j = 0; j < 4; ++j) Q.hi[a][j] = 0x80808080u;   // every hi byte has bit 7 set (the walk's guard bit)
        for (std::uint32_t c = 0; c < count; ++c)
            for (int a = 0; a < 3; ++a) {
                Q.lo[a][c / 4] |= quant(lo[a][first + c], Q.org[a], Q.scale[a]) << (8 * (c % 4));
                Q.hi[a][c / 4] |= quant(hi[a][first + c], Q.org[a], Q.scale[a]) << (8 * (c % 4));
            }
    };
    w->rankBox.resize(n);
    for (std::uint32_t r = 0; r < n; ++r)
        for (int a = 0; a < 3; ++a) { w->rankBox[r].lo[a] = lo[a][r]; w->rankBox[r].hi[a] = hi[a][r]; }
    std::uint32_t cn = n;
    while (cn > 16u) {
        std::vector<std::uint32_t> cuts{0u};
        if (levels.empty()) {
            for (std::uint32_t r = 1; r <= n; ++r)
                if (r == n || r - cuts.back() == 16u || w->rank[r].instance != w->rank[r - 1].instance) cuts.push_back(r);
        } else {
            for (std::uint32_t j = 16u; j < cn; j += 16u) cuts.push_back(j);
            cuts.push_back(cn);
        }
        Level P;
        const std::size_t pn = cuts.size() - 1u;
        P.first.resize(pn);
        P.count.resize(pn);
        P.q.resize(pn);
        std::vector<float> plo[3], phi[3];
        for (int a = 0; a < 3; ++a) { plo[a].assign(pn, 1e30f); phi[a].assign(pn, -1e30f); }
        for (std::size_t j = 0; j < pn; ++j) {
            P.first[j] = cuts[j];
            P.count[j] = cuts[j + 1] - cuts[j];
            for (std::uint32_t i = cuts[j]; i < cuts[j + 1]; ++i)
                for (int a = 0; a < 3; ++a) {
                    plo[a][j] = std::min(plo[a][j], lo[a][i]);
                    phi[a][j] = std::max(phi[a][j], hi[a][i]);
                }
            const float own[2][3] = {{plo[0][j], plo[1][j], plo[2][j]}, {phi[0][j], phi[1][j], phi[2][j]}};
            fillQ(P.q[j], cuts[j], cuts[j + 1] - cuts[j], own, levels.empty(), 0u);   // first: patched at the flattening
        }
        levels.push_back(std::move(P));
        for (int a = 0; a < 3; ++a) { lo[a].swap(plo[a]); hi[a].swap(phi[a]); }
        cn = static_cast<std::uint32_t>(levels.back().first.size());
    }
    // the FV tree, flattened: its root at node 0 (the walk's fallback), then levels[0] (the leaves), then up: node index =
    // offset[level] + j
    std::vector<std::uint32_t> offset(levels.size() + 1u, 1u);
    for (std::size_t L = 0; L < levels.size(); ++L) offset[L + 1] = offset[L] + static_cast<std::uint32_t>(levels[L].first.size());
    const std::uint32_t total = offset[levels.size()];
    w->nodes.resize(total);
    w->qnodes.resize(total);
    for (std::size_t L = 0; L < levels.size(); ++L)
        for (std::size_t j = 0; j < levels[L].first.size(); ++j) {
            const std::uint32_t g = offset[L] + static_cast<std::uint32_t>(j);
            w->nodes[g].first = L == 0 ? levels[L].first[j] : offset[L - 1] + levels[L].first[j];
            w->nodes[g].count = static_cast<std::uint8_t>(levels[L].count[j]);
            w->nodes[g].leaf = L == 0 ? 1u : 0u;
            w->nodes[g].pad = 0u;
            w->qnodes[g] = levels[L].q[j];
            w->qnodes[g].first = w->nodes[g].first;
        }
    w->nodes[0].first = levels.empty() ? 0u : offset[levels.size() - 1u];
    w->nodes[0].count = static_cast<std::uint8_t>(cn);
    w->nodes[0].leaf = levels.empty() ? 1u : 0u;
    {
        float own[2][3];
        for (int a = 0; a < 3; ++a) {
            own[0][a] = 1e30f;
            own[1][a] = -1e30f;
            for (std::uint32_t c = 0; c < cn; ++c) { own[0][a] = std::min(own[0][a], lo[a][c]); own[1][a] = std::max(own[1][a], hi[a][c]); }
        }
        fillQ(w->qnodes[0], 0u, cn, own, levels.empty(), w->nodes[0].first);
    }
    w->depth = static_cast<std::uint32_t>(levels.size());
    w->root = 0u;
    w->stackNeed = 0u;
    if (levels.size() >= 2u) {   // more than 16 FV leaves: the spatial tree, over the walk's own leaves
        // The walk's leaves: runs of up to 16 consecutive ranks (any such run gives a (first rank, mask) record), cut to
        // minimize the sum over leaves of the volume of the leaf's box grown by a query box q (10 x 6.7 x 10 m: the
        // chance a query meets it) plus that of q alone (a leaf's own cost), by dynamic programming over the ranks.
        // FvBvh's runs (16 ranks or an instance's end) were looser: rpg nodes a call 13.5 -> 9.3, leaf hits 9.6 -> 7.6.
        const double q[3] = {10.0, 6.7, 10.0}, lambda = q[0] * q[1] * q[2];
        auto rankBox = [&](std::uint32_t r) {
            SBox x;
            for (int a = 0; a < 3; ++a) { x.lo[a] = w->rankBox[r].lo[a]; x.hi[a] = w->rankBox[r].hi[a]; }
            return x;
        };
        std::vector<double> f(std::size_t(n) + 1u, 1e300);
        std::vector<std::uint32_t> from(std::size_t(n) + 1u, 0u);
        f[0] = 0.0;
        for (std::uint32_t i = 1; i <= n; ++i) {   // f[i]: the best cut of ranks [0, i); the last leaf [j, i)
            SBox b = Empty();
            for (std::uint32_t j = i; j-- > (i > 16u ? i - 16u : 0u);) {
                Grow(b, rankBox(j));
                const double c = (double(b.hi[0]) - b.lo[0] + q[0]) * (double(b.hi[1]) - b.lo[1] + q[1]) * (double(b.hi[2]) - b.lo[2] + q[2]) + lambda;
                if (f[j] + c < f[i]) { f[i] = f[j] + c; from[i] = j; }
            }
        }
        std::vector<std::uint32_t> cuts;
        for (std::uint32_t i = n; i > 0; i = from[i]) cuts.push_back(i);
        cuts.push_back(0u);
        std::reverse(cuts.begin(), cuts.end());
        const std::size_t nl = cuts.size() - 1u;
        std::vector<SBox> leafBox(nl);
        std::vector<NodeQ> leafQ(nl);
        for (std::size_t j = 0; j < nl; ++j) {
            SBox b = Empty();
            for (std::uint32_t r = cuts[j]; r < cuts[j + 1]; ++r) Grow(b, rankBox(r));
            leafBox[j] = b;
            NodeQ &Q = leafQ[j];   // as fillQ: the leaf's own box as the frame, its ranks' boxes in it
            Q = NodeQ{};
            for (int a = 0; a < 3; ++a) {
                Q.org[a] = b.lo[a];
                const float ext = b.hi[a] - b.lo[a];
                const float sc = ext > 0.0f ? 127.0f / ext : 0.0f;
                Q.scale[a] = std::isfinite(sc) ? sc : 0.0f;
                for (int k = 0; k < 4; ++k) Q.hi[a][k] = 0x80808080u;   // the guard bit (NodeQ)
            }
            const std::uint32_t cnt = cuts[j + 1] - cuts[j];
            Q.first = cuts[j];
            Q.meta = cnt | 1u << 8;
            for (std::uint32_t c = 0; c < cnt; ++c)
                for (int a = 0; a < 3; ++a) {
                    Q.lo[a][c / 4] |= quant(w->rankBox[cuts[j] + c].lo[a], Q.org[a], Q.scale[a]) << (8 * (c % 4));
                    Q.hi[a][c / 4] |= quant(w->rankBox[cuts[j] + c].hi[a], Q.org[a], Q.scale[a]) << (8 * (c % 4));
                }
        }
        BuildSpatial(leafBox, leafQ, quant, w);
    }

    w->plane.assign(std::size_t(4) * n, 0.0f);
    w->edge.assign(n, RankE{});
    for (std::uint32_t t = 0; t < n; ++t) {   // the world triangle's plane, in double
        const Tri &tr = view.tris[t];
        const double e1[3] = {double(tr.v[1][0]) - tr.v[0][0], double(tr.v[1][1]) - tr.v[0][1], double(tr.v[1][2]) - tr.v[0][2]};
        const double e2[3] = {double(tr.v[2][0]) - tr.v[0][0], double(tr.v[2][1]) - tr.v[0][1], double(tr.v[2][2]) - tr.v[0][2]};
        const double c[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        const double len = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        if (!(len > 1e-12)) continue;
        float *pl = &w->plane[std::size_t(4) * view.triInfo[t].fvOrder];
        for (int a = 0; a < 3; ++a) pl[a] = static_cast<float>(c[a] / len);
        pl[3] = static_cast<float>((c[0] * tr.v[0][0] + c[1] * tr.v[0][1] + c[2] * tr.v[0][2]) / len);
        // the edge planes: m = E x n (n the unit normal), turned away from the opposite vertex, unit; q = trunc(32767 m)
        // (m read as q 2^-15, exact); off = the largest m.v over the three vertices with that m, in double, rounded up:
        // the triangle lies in m.x <= off
        RankE &re = w->edge[view.triInfo[t].fvOrder];
        const double n[3] = {c[0] / len, c[1] / len, c[2] / len};
        for (int e = 0; e < 3; ++e) {
            const float *a = tr.v[e], *b = tr.v[e == 2 ? 0 : e + 1], *o = tr.v[e == 0 ? 2 : e - 1];
            const double E[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
            double m[3] = {E[1] * n[2] - E[2] * n[1], E[2] * n[0] - E[0] * n[2], E[0] * n[1] - E[1] * n[0]};
            if ((double(o[0]) - a[0]) * m[0] + (double(o[1]) - a[1]) * m[1] + (double(o[2]) - a[2]) * m[2] > 0.0)
                for (int k = 0; k < 3; ++k) m[k] = -m[k];
            const double ml = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
            if (!(ml > 1e-12)) continue;   // stays 0: never rejects
            std::int16_t q[3];
            double md[3];
            for (int k = 0; k < 3; ++k) {
                q[k] = static_cast<std::int16_t>(std::trunc(32767.0 * m[k] / ml));
                md[k] = std::ldexp(double(q[k]), -15);
            }
            double off = -1e300;
            for (int v = 0; v < 3; ++v) off = std::max(off, md[0] * tr.v[v][0] + md[1] * tr.v[v][1] + md[2] * tr.v[v][2]);
            float of = static_cast<float>(off);
            if (double(of) < off) of = std::nextafter(of, 1e30f);
            if (!std::isfinite(of)) continue;
            for (int k = 0; k < 3; ++k) re.m[e][k] = q[k];
            re.off[e] = of;
        }
    }
    w->iso.resize(std::size_t(12) * view.instances.size());
    for (std::size_t i = 0; i < view.instances.size(); ++i)
        std::copy(view.instances[i].iso, view.instances[i].iso + 12, w->iso.begin() + static_cast<std::ptrdiff_t>(12 * i));
    w->verts.assign(std::size_t(9) * view.modelTris.size(), 0.0f);
    for (std::size_t mi = 0; mi < view.models.size(); ++mi) {   // CcCollision::Init's verts_
        const world::Model &m = view.models[mi];
        for (std::uint32_t t = 0; t < m.triCount; ++t) {
            const world::ModelTri &mt = view.modelTris[m.firstTri + t];
            float *v = &w->verts[std::size_t(9) * (m.firstTri + t)];
            for (int c = 0; c < 3; ++c) {
                const world::Vec3f &p = view.modelVerts[m.firstVert + mt.v[c]];
                v[3 * c] = p.x;
                v[3 * c + 1] = p.y;
                v[3 * c + 2] = p.z;
            }
        }
    }
    w->outRec.resize(n);
    for (std::uint32_t t = 0; t < n; ++t) {   // CcCollision::Init's outRec_
        const Tri &tr = view.tris[t];
        const world::TriInfo &ti = view.triInfo[t];
        const world::Actor &a = view.ActorOf(tr);
        OutRecG &r = w->outRec[t];
        for (int i = 0; i < 3; ++i) r.peerZ[i] = a.peerZ[i];
        const bool trig = ti.trigger != world::kNoTrigger;
        r.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
        r.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
        r.flags = static_cast<std::uint8_t>(trig ? unsigned(model::kRespTrigger) : 0u);
        r.pad[0] = r.pad[1] = 0u;
    }
    return true;
}

}  // namespace tw::gpu
