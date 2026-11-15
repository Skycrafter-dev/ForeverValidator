// Oriented slab counters (CC_OSLABSTATS in cc_stage.cpp; docs/twin/collision-x2-night.md "Oriented slabs").
// Measurement only: included by cc_stage.cpp inside namespace tw::collide_c when CC_OSLABSTATS != 0; the output is
// untouched. The idea under test: per leaf block of the FV triangle tree (runs of up to 16 consecutive FV ranks, cut where
// the instance changes: the unit the refill copies into the memo), a static oriented slab {unit n, [dmin, dmax]} holding its
// triangles. A body ellipsoid whose support interval along n, n.c -+ sqrt(n^T Q n), lies above dmax + eps or below
// dmin - eps cannot touch any triangle of the leaf, so its pairs with them can go; a near entry cleared for all 4 bodies and
// outside every wheel box can skip the filter entirely (no plane rows, no box masks).
//
// Bit 1 (per lane-call, DetectLanes16 after the merge, OslabCount): over the lane's near entries (memo entries meeting the
// union box), today's body box pairs, survivors and raw contacts, and per proof variant
//   A    the body's exact ellipsoid AABB (sqrt(Q_aa) + today's eps) against the triangle's AABB (memo data; a slab per world
//        axis and triangle, no new memory),
//   M    the leaf's area-weighted mean normal slab,
//   T    the leaf's thinnest slab among {mean normal, world up, each triangle's normal},
//   AM, AT  A or M, A or T (one leaf slab plus the memo's boxes),
//   AN   A or any of T's candidate slabs (a multi-slab oracle: up to 18 slabs per leaf),
//   B    the 16-entry memo block's thinnest slab (same candidates; made at the refill: per-candidate upkeep), AB: A or B,
//   PA   A or the triangle's own plane, two-sided (thickness 0; needs the plane rows: reference only)
// the body box pairs cleared, today's body survivors cleared, violations (a raw body contact on a cleared pair: must be 0),
// near entries skippable entirely (outside every wheel box, today's or (A variants) the exact wheel AABB, and every body box
// pair cleared), lane-calls with every body pair cleared; slab thickness per body box pair (M, T, B); leaves and memory.
// Bit 2 (DetectLanes16, before FilterPass16, OslabTime): on every fifth pass each, timed cold before the pass's own filter:
// (0) today's FilterPass16; (1) with every shape box (and the union) the exact ellipsoid AABB; (2) the same with the near
// entries skippable under AT (exact wheel boxes) removed from the memo (boxes emptied, restored after); (3) a prototype of the
// pass-0 skip test (per memo block with a union hit: 4 exact wheel boxes, 4 bodies' exact AABBs and one slab per entry,
// float, sqrt-free) with its memo scan; (4) the scan alone. Projected saving per lane-call: (0) - (2) - ((3) - (4)).
namespace oslab {

using std::uint32_t;
using std::uint64_t;
constexpr double kEps = 2e-3;   // m: the slab test's float margin (support and n.c in float at Stadium coordinates)
enum { vA, vM, vT, vAM, vAT, vAN, vB, vAB, vPA, kV };
static const char *kVName[kV] = {"A  exact AABB", "M  leaf mean", "T  leaf thin", "AM", "AT", "AN any slab", "B  memo block", "AB", "PA own plane"};
constexpr int kTh = 11;
constexpr double kThEdge[kTh - 1] = {1e-4, 1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 0.1, 0.25, 0.5, 1.0};

struct Slab {
    double n[3], lo, hi;
};
struct Row {
    uint64_t calls = 0, near = 0, near0 = 0, nearW = 0, nearWA = 0, bodyPairs = 0, wheelPairs = 0, wheelPairsA = 0, surv = 0, raw = 0;
    uint64_t o1 = 0, o2 = 0, mismatch = 0, wraw = 0, wviol = 0, survW = 0, refills = 0;
    uint64_t clr[kV] = {}, clrS[kV] = {}, viol[kV] = {}, skip[kV] = {}, skipWA[kV] = {}, all4[kV] = {};
    uint64_t th[3][kTh] = {};
    uint64_t leaves = 0, tris = 0;
    // bit 2
    uint64_t tp[5] = {}, tl[5] = {}, tt[5] = {}, blocks = 0, hitBlocks = 0;
    void Add(const Row &r) {
        calls += r.calls; near += r.near; near0 += r.near0; nearW += r.nearW; nearWA += r.nearWA; bodyPairs += r.bodyPairs;
        wheelPairs += r.wheelPairs; wheelPairsA += r.wheelPairsA; surv += r.surv; raw += r.raw; o1 += r.o1; o2 += r.o2; mismatch += r.mismatch; wraw += r.wraw; wviol += r.wviol; survW += r.survW; refills += r.refills;
        for (int v = 0; v < kV; ++v) {
            clr[v] += r.clr[v]; clrS[v] += r.clrS[v]; viol[v] += r.viol[v]; skip[v] += r.skip[v]; skipWA[v] += r.skipWA[v]; all4[v] += r.all4[v];
        }
        for (int i = 0; i < 3; ++i)
            for (int h = 0; h < kTh; ++h) th[i][h] += r.th[i][h];
        leaves += r.leaves; tris += r.tris;
        for (int i = 0; i < 5; ++i) { tp[i] += r.tp[i]; tl[i] += r.tl[i]; tt[i] += r.tt[i]; }
        blocks += r.blocks; hitBlocks += r.hitBlocks;
    }
};
struct Table {
    std::mutex mu;
    std::vector<std::pair<uint32_t, Row>> rows;
    void Add(uint32_t key, const Row &r) {
        std::lock_guard<std::mutex> g(mu);
        for (auto &p : rows)
            if (p.first == key) {
                Row keep = p.second;
                p.second.Add(r);
                if (keep.leaves) { p.second.leaves = keep.leaves; p.second.tris = keep.tris; }   // per world, not summed over threads
                return;
            }
        rows.emplace_back(key, r);
    }
    static void Print(const char *name, const Row &w) {
        if (w.calls == 0u && w.tl[0] + w.tl[1] + w.tl[2] + w.tl[3] + w.tl[4] == 0u) return;
        const double n = double(std::max<uint64_t>(1, w.calls)), bp = double(std::max<uint64_t>(1, w.bodyPairs)), sv = double(std::max<uint64_t>(1, w.surv));
        const double nr = double(std::max<uint64_t>(1, w.near));
        std::fprintf(stderr, "oslab %-9s lane calls %llu | near %.2f (in no shape box %.2f, in a wheel box %.2f, exact wheel AABB %.2f) | body box pairs %.2f, "
                             "survivors %.3f, raw %.4f | wheel box pairs %.2f, exact AABB %.2f (-%.1f%%) | oracle: no body survivor %.3f, no body raw %.3f | "
                             "leaves %llu (%.1f tris), slab memory %.2f MB | raw wheel contacts %llu, outside the exact AABB %llu%s\n",
                     name, static_cast<unsigned long long>(w.calls), double(w.near) / n, double(w.near0) / n, double(w.nearW) / n, double(w.nearWA) / n,
                     double(w.bodyPairs) / n, double(w.surv) / n, double(w.raw) / n, double(w.wheelPairs) / n, double(w.wheelPairsA) / n,
                     100.0 * (1.0 - double(w.wheelPairsA) / double(std::max<uint64_t>(1, w.wheelPairs))), double(w.o1) / n, double(w.o2) / n,
                     static_cast<unsigned long long>(w.leaves), double(w.tris) / double(std::max<uint64_t>(1, w.leaves)), 20.0 * double(w.leaves) / 1e6,
                     static_cast<unsigned long long>(w.wraw), static_cast<unsigned long long>(w.wviol), w.mismatch ? " SHAPE MISMATCH" : "");
        std::fprintf(stderr, "oslab %-9s   per lane-call: survivors %.3f (bodies %.3f, wheels %.3f), refills %.4f\n", name, double(w.surv + w.survW) / n,
                     double(w.surv) / n, double(w.survW) / n, double(w.refills) / n);
        for (int v = 0; v < kV; ++v)
            std::fprintf(stderr, "oslab %-9s   %-14s body pairs cleared %.3f (left %.2f) | survivors cleared %.3f | violations %llu | near skippable %.3f "
                                 "(%.2f left), exact wheel boxes %.3f (%.2f left) | all 4 bodies cleared %.3f\n",
                         name, kVName[v], double(w.clr[v]) / bp, double(w.bodyPairs - w.clr[v]) / n, double(w.clrS[v]) / sv,
                         static_cast<unsigned long long>(w.viol[v]), double(w.skip[v]) / nr, double(w.near - w.skip[v]) / n, double(w.skipWA[v]) / nr,
                         double(w.near - w.skipWA[v]) / n, double(w.all4[v]) / n);
        static const char *tn[3] = {"M", "T", "B"};
        for (int i = 0; i < 3; ++i) {
            std::fprintf(stderr, "oslab %-9s   thickness %s per body pair:", name, tn[i]);
            uint64_t acc = 0;
            for (int h = 0; h < kTh; ++h) {
                acc += w.th[i][h];
                std::fprintf(stderr, " %s%g %.3f", h < kTh - 1 ? "<=" : ">", h < kTh - 1 ? kThEdge[h] : kThEdge[kTh - 2], double(acc) / bp);
            }
            std::fprintf(stderr, "\n");
        }
        if (w.tl[0] + w.tl[1] + w.tl[2] + w.tl[3] + w.tl[4]) {
            double t[5];
            for (int i = 0; i < 5; ++i) t[i] = double(w.tt[i]) / double(std::max<uint64_t>(1, w.tl[i]));
            const double hb = double(w.hitBlocks) / double(std::max<uint64_t>(1, w.tl[3]));
            std::fprintf(stderr, "oslab %-9s   TSC per lane-call (passes %llu %llu %llu %llu %llu): FilterPass16 today %.0f | exact AABBs %.0f (saved %.0f) | exact AABBs + "
                                 "AT skip %.0f (saved %.0f) | skip test: scan + test %.0f, scan alone %.0f: test %.0f (%.2f memo blocks with a union hit, %.1f TSC per "
                                 "block of 16 entries x 4 bodies) | AT net %.0f\n",
                         name, static_cast<unsigned long long>(w.tp[0]), static_cast<unsigned long long>(w.tp[1]), static_cast<unsigned long long>(w.tp[2]),
                         static_cast<unsigned long long>(w.tp[3]), static_cast<unsigned long long>(w.tp[4]), t[0], t[1], t[0] - t[1], t[2], t[0] - t[2], t[3], t[4],
                         t[3] - t[4], hb, (t[3] - t[4]) / std::max(1e-9, hb), t[0] - t[2] - (t[3] - t[4]));
        }
    }
    ~Table() {
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        Row all;
        for (auto &p : rows) {
            char name[32];
            std::snprintf(name, sizeof name, "%u", p.first);
            Print(name, p.second);
            all.Add(p.second);
        }
        Print("ALL", all);
    }
};
inline Table &Tab() { static Table t; return t; }

struct WorldLocal {
    uint32_t key = 0;
    bool built = false;
    std::vector<uint32_t> leafOf, leafFirst, leafCount;
    std::vector<int32_t> slabOff;   // per leaf: its slabs in pool ([0] mean, [1] thinnest, [2..] the candidates), -1 not made
    std::vector<uint8_t> slabN;
    std::vector<Slab> pool;
    Row row;
    // bit 2: the prototype's per-entry slab rows (SoA, aligned with a lane's memo entries)
    std::vector<float> sl[5];
};
struct Local {
    std::unordered_map<const void *, WorldLocal> worlds;
    uint64_t lane = 0, pass = 0;
    int every = 1;
    std::vector<uint64_t> clr;    // per memo entry: bit 4 v + k, body k cleared under variant v
    std::vector<uint8_t> boxk;    // per memo entry: today's 8 box bits
    std::vector<uint8_t> wa;      // per memo entry: exact wheel AABB bits (4..7)
    std::vector<uint32_t> nearE;
    std::unordered_map<const void *, uint64_t> lastRefills;   // per lane cache: its refill count at the last counted call
    Local() {
        if (const char *e = std::getenv("FV_OSLAB_EVERY")) every = std::max(1, std::atoi(e));
    }
    ~Local() {
        for (auto &p : worlds)
            if (p.second.row.calls || p.second.row.tl[0] + p.second.row.tl[1] + p.second.row.tl[2] + p.second.row.tl[3] + p.second.row.tl[4]) Tab().Add(p.second.key, p.second.row);
    }
};
inline Local &Loc() {
    (void)Tab();
    static thread_local Local loc;
    return loc;
}

inline void Interval(const tw::Tri &t, const double n[3], double *lo, double *hi) {
    for (int c = 0; c < 3; ++c) {
        const double d = double(t.v[c][0]) * n[0] + double(t.v[c][1]) * n[1] + double(t.v[c][2]) * n[2];
        *lo = std::min(*lo, d);
        *hi = std::max(*hi, d);
    }
}
inline bool TriNormal(const tw::Tri &t, double n[3]) {
    const double e1[3] = {double(t.v[1][0]) - t.v[0][0], double(t.v[1][1]) - t.v[0][1], double(t.v[1][2]) - t.v[0][2]};
    const double e2[3] = {double(t.v[2][0]) - t.v[0][0], double(t.v[2][1]) - t.v[0][1], double(t.v[2][2]) - t.v[0][2]};
    n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(l > 0.0)) return false;
    for (int a = 0; a < 3; ++a) n[a] /= l;
    return true;
}
// the slabs of a triangle set: [0] the area-weighted mean normal, [1] the thinnest of the candidates, [2..] the candidates
// (mean, world up, each triangle's normal)
template <class TriAt>
void MakeSlabs(int cnt, TriAt &&at, std::vector<Slab> *out) {
    double s[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < cnt; ++i) {
        const tw::Tri &t = at(i);
        const double e1[3] = {double(t.v[1][0]) - t.v[0][0], double(t.v[1][1]) - t.v[0][1], double(t.v[1][2]) - t.v[0][2]};
        const double e2[3] = {double(t.v[2][0]) - t.v[0][0], double(t.v[2][1]) - t.v[0][1], double(t.v[2][2]) - t.v[0][2]};
        s[0] += e1[1] * e2[2] - e1[2] * e2[1];
        s[1] += e1[2] * e2[0] - e1[0] * e2[2];
        s[2] += e1[0] * e2[1] - e1[1] * e2[0];
    }
    std::vector<std::array<double, 3>> cand;
    const double l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    if (l > 0.0) cand.push_back({s[0] / l, s[1] / l, s[2] / l});
    else cand.push_back({0.0, 1.0, 0.0});
    cand.push_back({0.0, 1.0, 0.0});
    for (int i = 0; i < cnt; ++i) {
        double n[3];
        if (TriNormal(at(i), n)) cand.push_back({n[0], n[1], n[2]});
    }
    const std::size_t base = out->size();
    out->resize(base + 2 + cand.size());
    double best = 1e300;
    for (std::size_t c = 0; c < cand.size(); ++c) {
        Slab &sl = (*out)[base + 2 + c];
        for (int a = 0; a < 3; ++a) sl.n[a] = cand[c][a];
        sl.lo = 1e300;
        sl.hi = -1e300;
        for (int i = 0; i < cnt; ++i) Interval(at(i), sl.n, &sl.lo, &sl.hi);
        if (sl.hi - sl.lo < best) { best = sl.hi - sl.lo; (*out)[base + 1] = sl; }
    }
    (*out)[base] = (*out)[base + 2];
}
// the ellipsoid (pose row: axes at 3 i, centre 9..11, radii 12..14) clear of the slab
inline bool Clear(const float *row, const Slab &s) {
    double sup = 0.0, nc = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double d = double(row[12 + i]) * (double(row[3 * i]) * s.n[0] + double(row[3 * i + 1]) * s.n[1] + double(row[3 * i + 2]) * s.n[2]);
        sup += d * d;
        nc += double(row[9 + i]) * s.n[i];
    }
    sup = std::sqrt(sup);
    return nc - sup > s.hi + kEps || nc + sup < s.lo - kEps;
}
inline __mmask16 BoxMask16(__m512 lx, __m512 ly, __m512 lz, __m512 hx, __m512 hy, __m512 hz, const float lo[3], const float hi[3], __mmask16 m) {
    m &= _mm512_cmp_ps_mask(lx, _mm512_set1_ps(hi[0]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_set1_ps(lo[0]), hx, _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(ly, _mm512_set1_ps(hi[1]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_set1_ps(lo[1]), hy, _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(lz, _mm512_set1_ps(hi[2]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_set1_ps(lo[2]), hz, _CMP_LE_OQ);
    return m;
}
inline int ThBin(double t) {
    int h = 0;
    while (h < kTh - 1 && t > kThEdge[h]) ++h;
    return h;
}

}  // namespace oslab

// the leaves of this world (FvBvh's level 1: runs of up to 16 consecutive ranks, cut where the instance changes)
static oslab::WorldLocal &OslabWorld(const void *self, const FvBvh &fvb) {
    using namespace oslab;
    Local &loc = Loc();
    WorldLocal &wl = loc.worlds[self];
    if (wl.built) return wl;
    wl.built = true;
    wl.key = fvb.Size();
    const uint32_t n = fvb.Size();
    wl.leafOf.assign(n, 0u);
    uint32_t first = 0;
    for (uint32_t r = 1; r <= n; ++r)
        if (r == n || r - first == 16u || fvb.ref[r].instance != fvb.ref[r - 1].instance) {
            const uint32_t j = static_cast<uint32_t>(wl.leafFirst.size());
            wl.leafFirst.push_back(first);
            wl.leafCount.push_back(r - first);
            for (uint32_t q = first; q < r; ++q) wl.leafOf[q] = j;
            first = r;
        }
    wl.slabOff.assign(wl.leafFirst.size(), -1);
    wl.slabN.assign(wl.leafFirst.size(), 0u);
    wl.row.leaves = wl.leafFirst.size();
    wl.row.tris = n;
    return wl;
}
static const oslab::Slab *OslabLeaf(oslab::WorldLocal &wl, const world::WorldPackView &v, const FvBvh &fvb, uint32_t leaf, int *count) {
    using namespace oslab;
    if (wl.slabOff[leaf] < 0) {
        wl.slabOff[leaf] = static_cast<int32_t>(wl.pool.size());
        const uint32_t f = wl.leafFirst[leaf], c = wl.leafCount[leaf];
        MakeSlabs(static_cast<int>(c), [&](int i) -> const tw::Tri & { return v.tris[fvb.ref[f + static_cast<uint32_t>(i)].local]; }, &wl.pool);
        wl.slabN[leaf] = static_cast<uint8_t>(wl.pool.size() - static_cast<std::size_t>(wl.slabOff[leaf]));
    }
    *count = wl.slabN[leaf];
    return wl.pool.data() + wl.slabOff[leaf];
}

// Bit 1: the counters of one DetectLanes16 pass (after its merge).
void CcCollision::OslabCount(std::uint32_t mask, step::CollisionCache *const *caches) const {
    using namespace oslab;
    Local &loc = Loc();
    WorldLocal &wl = OslabWorld(this, fvb_);
    Row &row = wl.row;
    Front16Scratch &fs = Front16Scratch_();
    const PassSurvivors &ps = fs.pass;
    const world::WorldPackView &v = *view_;
    std::vector<Slab> blk;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        if (loc.lane++ % uint64_t(loc.every) != 0u) continue;
        const CcCache &cc = *static_cast<const CcCache *>(caches[l]);
        const Memo &m = cc.m;
        const ShapeTable &st = fs.st[l];
        const Box3 &u = fs.u[l];
        ++row.calls;
        {
            uint64_t &last = loc.lastRefills[&cc];
            row.refills += cc.refills >= last ? cc.refills - last : cc.refills;
            last = cc.refills;
        }
        if (loc.clr.size() < m.n + 16u) { loc.clr.resize(2 * m.n + 16u); loc.boxk.resize(2 * m.n + 16u); loc.wa.resize(2 * m.n + 16u); }
        // the exact ellipsoid AABBs (today's eps: 1e-3 + 1e-6 |c|)
        double elo[8][3], ehi[8][3];
        for (int k = 0; k < 8; ++k) {
            const float *pr = fs.rows[l][k];
            for (int a = 0; a < 3; ++a) {
                double h = 0.0;
                for (int i = 0; i < 3; ++i) { const double d = double(pr[12 + i]) * double(pr[3 * i + a]); h += d * d; }
                h = std::sqrt(h);
                const double c = pr[9 + a], eps = 1e-3 + 1e-6 * std::fabs(c);
                elo[k][a] = c - h - eps;
                ehi[k][a] = c + h + eps;
            }
        }
        int blkCached = -1;
        uint32_t nNear = 0;
        uint64_t bodyPairsCall = 0, clrCall[kV] = {};
        for (std::uint32_t e = 0; e < m.n; ++e) {
            const float lo[3] = {m.lo[0][e], m.lo[1][e], m.lo[2][e]}, hi[3] = {m.hi[0][e], m.hi[1][e], m.hi[2][e]};
            loc.clr[e] = 0u;
            loc.boxk[e] = 0u;
            loc.wa[e] = 0u;
            if (!(lo[0] <= u.hi[0] && u.lo[0] <= hi[0] && lo[1] <= u.hi[1] && u.lo[1] <= hi[1] && lo[2] <= u.hi[2] && u.lo[2] <= hi[2])) continue;
            ++nNear;
            uint8_t bk = 0, wa = 0;
            for (int k = 0; k < 8; ++k) {
                if (lo[0] <= st.hi[0][k] && st.lo[0][k] <= hi[0] && lo[1] <= st.hi[1][k] && st.lo[1][k] <= hi[1] && lo[2] <= st.hi[2][k] && st.lo[2][k] <= hi[2])
                    bk |= uint8_t(1u << k);
                if (k >= 4 && lo[0] <= ehi[k][0] && elo[k][0] <= hi[0] && lo[1] <= ehi[k][1] && elo[k][1] <= hi[1] && lo[2] <= ehi[k][2] && elo[k][2] <= hi[2])
                    wa |= uint8_t(1u << k);
            }
            loc.boxk[e] = bk;
            loc.wa[e] = wa;
            row.near0 += bk == 0u;
            row.nearW += (bk & 0xF0u) != 0u;
            row.nearWA += wa != 0u;
            row.wheelPairs += static_cast<uint64_t>(__builtin_popcount(bk & 0xF0u));
            row.wheelPairsA += static_cast<uint64_t>(__builtin_popcount(wa));
            if (!(bk & 0x0Fu)) continue;
            const uint32_t rank = m.fv[e];
            const tw::Tri &tri = v.tris[m.ref[e].local];
            int nsl = 0;
            const Slab *ls = OslabLeaf(wl, v, fvb_, wl.leafOf[rank], &nsl);
            const int b = static_cast<int>(e / 16u);
            if (b != blkCached) {   // the memo block's slabs (its valid entries)
                blkCached = b;
                blk.clear();
                const uint32_t b0 = 16u * uint32_t(b), cnt = std::min<uint32_t>(16u, m.n - b0);
                MakeSlabs(static_cast<int>(cnt), [&](int i) -> const tw::Tri & { return v.tris[m.ref[b0 + uint32_t(i)].local]; }, &blk);
            }
            double tn[3];
            const bool hasN = TriNormal(tri, tn);
            Slab own{};
            if (hasN) {
                for (int a = 0; a < 3; ++a) own.n[a] = tn[a];
                own.lo = 1e300;
                own.hi = -1e300;
                Interval(tri, own.n, &own.lo, &own.hi);
            }
            for (int k = 0; k < 4; ++k) {
                if (!(bk >> k & 1u)) continue;
                const float *pr = fs.rows[l][k];
                ++row.bodyPairs;
                ++bodyPairsCall;
                bool c[kV];
                c[vA] = !(lo[0] <= ehi[k][0] && elo[k][0] <= hi[0] && lo[1] <= ehi[k][1] && elo[k][1] <= hi[1] && lo[2] <= ehi[k][2] && elo[k][2] <= hi[2]);
                c[vM] = Clear(pr, ls[0]);
                c[vT] = Clear(pr, ls[1]);
                c[vAM] = c[vA] || c[vM];
                c[vAT] = c[vA] || c[vT];
                bool any = c[vA] || c[vM] || c[vT];
                for (int i = 2; i < nsl && !any; ++i) any = Clear(pr, ls[i]);
                c[vAN] = any;
                c[vB] = Clear(pr, blk[1]);
                c[vAB] = c[vA] || c[vB];
                c[vPA] = c[vA] || (hasN && Clear(pr, own));
                for (int vv = 0; vv < kV; ++vv)
                    if (c[vv]) { loc.clr[e] |= uint64_t(1u) << (4 * vv + k); ++row.clr[vv]; ++clrCall[vv]; }
                ++row.th[0][ThBin(ls[0].hi - ls[0].lo)];
                ++row.th[1][ThBin(ls[1].hi - ls[1].lo)];
                ++row.th[2][ThBin(blk[1].hi - blk[1].lo)];
            }
        }
        row.near += nNear;
        // near entries skippable: outside every wheel box (today's, or the exact AABBs) and every body box pair cleared
        for (std::uint32_t e = 0; e < m.n; ++e) {
            const uint8_t bk = loc.boxk[e];
            const float lo[3] = {m.lo[0][e], m.lo[1][e], m.lo[2][e]}, hi[3] = {m.hi[0][e], m.hi[1][e], m.hi[2][e]};
            if (!(lo[0] <= u.hi[0] && u.lo[0] <= hi[0] && lo[1] <= u.hi[1] && u.lo[1] <= hi[1] && lo[2] <= u.hi[2] && u.lo[2] <= hi[2])) continue;
            for (int vv = 0; vv < kV; ++vv) {
                const unsigned cl = static_cast<unsigned>(loc.clr[e] >> (4 * vv)) & 0xFu;
                const bool bodiesOk = ((bk & 0x0Fu) & ~cl) == 0u;
                row.skip[vv] += bodiesOk && !(bk & 0xF0u);
                row.skipWA[vv] += bodiesOk && !loc.wa[e];
            }
        }
        for (int vv = 0; vv < kV; ++vv) row.all4[vv] += clrCall[vv] == bodyPairsCall;
        // today's survivors and raw contacts on the body pairs
        uint32_t bs = 0, br = 0;
        for (std::uint32_t j = ps.s0[l]; j < ps.s1[l]; ++j) {
            const uint32_t e = ps.e[j], k = ps.k[j];
            if (k >= 4u) {   // a raw wheel contact outside the exact wheel AABB (must be 0)
                const uint32_t s = fs.s0[l] + (j - ps.s0[l]);
                if (fs.hits[s >> 4] >> (s & 15u) & 1u) { ++row.wraw; row.wviol += !(loc.wa[e] >> k & 1u); }
                ++row.survW;
                continue;
            }
            ++bs;
            for (int vv = 0; vv < kV; ++vv) row.clrS[vv] += loc.clr[e] >> (4 * vv + int(k)) & 1u;
            const uint32_t s = fs.s0[l] + (j - ps.s0[l]);
            if (!(fs.hits[s >> 4] >> (s & 15u) & 1u)) continue;
            ++br;
            row.mismatch += fs.res[s].shape != k;
            for (int vv = 0; vv < kV; ++vv) row.viol[vv] += loc.clr[e] >> (4 * vv + int(k)) & 1u;
        }
        row.surv += bs;
        row.raw += br;
        row.o1 += bs == 0u;
        row.o2 += br == 0u;
    }
}

// Bit 2: before the pass's FilterPass16 (same thread, same memos and tables), in turn: (0) today's FilterPass16; (1) with every
// shape box (and the union) the exact ellipsoid AABB; (2) the same with the near entries skippable under AT removed from the
// memo (boxes emptied, restored after); (3) a prototype of the pass-0 skip test (per memo block with a union hit: 4 exact
// wheel boxes, 4 bodies' exact AABBs and one slab per entry, float, sqrt-free); each timed cold on every fourth pass.
void CcCollision::OslabTime(Memo *const *memos, std::uint32_t ready) const {
    using namespace oslab;
    if (ready == 0u) return;
    Local &loc = Loc();
    WorldLocal &wl = OslabWorld(this, fvb_);
    Row &row = wl.row;
    Front16Scratch &fs = Front16Scratch_();
    const world::WorldPackView &v = *view_;
    const int which = static_cast<int>(loc.pass++ % 5u);
    struct Saved { Memo *m; uint32_t e; float lo[3], hi[3]; };
    static thread_local std::vector<Saved> saved;
    static thread_local std::vector<float> rowsL[16][5];   // the prototype's per-entry slab rows per lane (n xyz, lo - eps, hi + eps)
    static thread_local ShapeTable stE[16];
    static thread_local Box3 uE[16];
    saved.clear();
    // per lane (every pass, so that each variant sees the same cache state): the exact boxes, the slab rows, the AT skips
    for (std::uint32_t mm = ready; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        Memo &m = *memos[l];
        stE[l] = fs.st[l];
        Box3 ub = EmptyBox();
        for (int k = 0; k < 8; ++k) {   // float, as a front would make them: sqrt(Q_aa), today's eps
            const float *pr = fs.rows[l][k];
            for (int a = 0; a < 3; ++a) {
                float h = 0.0f;
                for (int i = 0; i < 3; ++i) { const float d = pr[12 + i] * pr[3 * i + a]; h += d * d; }
                h = std::sqrt(h);
                const float c = pr[9 + a], eps = 1e-3f + 1e-6f * std::fabs(c);
                stE[l].lo[a][k] = (c - h) - eps;
                stE[l].hi[a][k] = (c + h) + eps;
                ub.lo[a] = std::min(ub.lo[a], stE[l].lo[a][k]);
                ub.hi[a] = std::max(ub.hi[a], stE[l].hi[a][k]);
            }
        }
        uE[l] = ub;
        const ShapeTable &st = stE[l];
        const std::size_t cap = (m.n + 15u) / 16u * 16u + 16u;
        for (int i = 0; i < 5; ++i) if (rowsL[l][i].size() < cap) rowsL[l][i].resize(2 * cap);
        for (std::uint32_t e = 0; e < m.n; ++e) {
            int nsl = 0;
            const Slab *ls = OslabLeaf(wl, v, fvb_, wl.leafOf[m.fv[e]], &nsl);
            for (int a = 0; a < 3; ++a) rowsL[l][a][e] = static_cast<float>(ls[1].n[a]);
            rowsL[l][3][e] = static_cast<float>(ls[1].lo - kEps);
            rowsL[l][4][e] = static_cast<float>(ls[1].hi + kEps);
            if (which != 2) continue;
            const float lo[3] = {m.lo[0][e], m.lo[1][e], m.lo[2][e]}, hi[3] = {m.hi[0][e], m.hi[1][e], m.hi[2][e]};
            if (!(lo[0] <= ub.hi[0] && ub.lo[0] <= hi[0] && lo[1] <= ub.hi[1] && ub.lo[1] <= hi[1] && lo[2] <= ub.hi[2] && ub.lo[2] <= hi[2])) continue;
            bool skip = true;
            for (int k = 0; k < 8 && skip; ++k) {
                const bool in = lo[0] <= st.hi[0][k] && st.lo[0][k] <= hi[0] && lo[1] <= st.hi[1][k] && st.lo[1][k] <= hi[1] && lo[2] <= st.hi[2][k] &&
                                st.lo[2][k] <= hi[2];
                if (!in) continue;
                skip = k < 4 && Clear(fs.rows[l][k], ls[1]);
            }
            if (!skip) continue;
            Saved sv{&m, e, {lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}};
            saved.push_back(sv);
        }
    }
    static thread_local PassSurvivors out;
    out.n = 0;
    std::uint64_t t0 = 0, t1 = 0;
    if (which == 0) {
        t0 = __rdtsc();
        FilterPass16(memos, fs.st, fs.u, ready, bodyShapes_, &out);
        t1 = __rdtsc();
    } else if (which == 1) {
        t0 = __rdtsc();
        FilterPass16(memos, stE, uE, ready, bodyShapes_, &out);
        t1 = __rdtsc();
    } else if (which == 2) {
        for (const Saved &s : saved)   // the skipped entries' boxes emptied (pass 0 drops them), restored after
            for (int a = 0; a < 3; ++a) { s.m->lo[a][s.e] = 1e30f; s.m->hi[a][s.e] = -1e30f; }
        t0 = __rdtsc();
        FilterPass16(memos, stE, uE, ready, bodyShapes_, &out);
        t1 = __rdtsc();
        for (const Saved &s : saved)
            for (int a = 0; a < 3; ++a) { s.m->lo[a][s.e] = s.lo[a]; s.m->hi[a][s.e] = s.hi[a]; }
    } else {
        using F = __m512;
        uint32_t sink = 0;
        const bool test = which == 3;
        t0 = __rdtsc();
        for (std::uint32_t mm = ready; mm != 0u; mm &= mm - 1u) {
            const int l = __builtin_ctz(mm);
            const Memo &m = *memos[l];
            const Box3 &ul = uE[l];
            const ShapeTable &st = stE[l];
            float cA[4][3], Qb[4][6];
            for (int k = 0; k < 4; ++k) {   // Q = sum_i r_i^2 a_i a_i^T: xx yy zz, the off-diagonals doubled
                const float *pr = fs.rows[l][k];
                auto q = [&](int x, int y) {
                    float s2 = 0.0f;
                    for (int i = 0; i < 3; ++i) s2 += pr[12 + i] * pr[12 + i] * pr[3 * i + x] * pr[3 * i + y];
                    return s2;
                };
                Qb[k][0] = q(0, 0); Qb[k][1] = q(1, 1); Qb[k][2] = q(2, 2);
                Qb[k][3] = 2.0f * q(0, 1); Qb[k][4] = 2.0f * q(0, 2); Qb[k][5] = 2.0f * q(1, 2);
                for (int a = 0; a < 3; ++a) cA[k][a] = pr[9 + a];
            }
            const float *nx = rowsL[l][0].data(), *ny = rowsL[l][1].data(), *nz = rowsL[l][2].data(), *dl = rowsL[l][3].data(), *dh = rowsL[l][4].data();
            for (std::uint32_t b = 0; b < m.n; b += 16) {
                const __mmask16 valid = static_cast<__mmask16>(m.n - b >= 16 ? 0xFFFFu : (1u << (m.n - b)) - 1u);
                const F lx = _mm512_loadu_ps(&m.lo[0][b]), ly = _mm512_loadu_ps(&m.lo[1][b]), lz = _mm512_loadu_ps(&m.lo[2][b]);
                const F hx = _mm512_loadu_ps(&m.hi[0][b]), hy = _mm512_loadu_ps(&m.hi[1][b]), hz = _mm512_loadu_ps(&m.hi[2][b]);
                const __mmask16 mk = oslab::BoxMask16(lx, ly, lz, hx, hy, hz, ul.lo, ul.hi, valid);
                if (!mk) continue;
                if (!test) { sink += static_cast<uint32_t>(__builtin_popcount(mk)); continue; }
                ++row.hitBlocks;
                __mmask16 keep = 0;
                const F n0 = _mm512_loadu_ps(nx + b), n1 = _mm512_loadu_ps(ny + b), n2 = _mm512_loadu_ps(nz + b);
                const F p00 = _mm512_mul_ps(n0, n0), p11 = _mm512_mul_ps(n1, n1), p22 = _mm512_mul_ps(n2, n2);
                const F p01 = _mm512_mul_ps(n0, n1), p02 = _mm512_mul_ps(n0, n2), p12 = _mm512_mul_ps(n1, n2);
                const F vlo = _mm512_loadu_ps(dl + b), vhi = _mm512_loadu_ps(dh + b);
                for (int k = 0; k < 8; ++k) {
                    const float blo[3] = {st.lo[0][k], st.lo[1][k], st.lo[2][k]}, bhi[3] = {st.hi[0][k], st.hi[1][k], st.hi[2][k]};
                    const __mmask16 in = oslab::BoxMask16(lx, ly, lz, hx, hy, hz, blo, bhi, mk);
                    if (k >= 4) { keep |= in; continue; }
                    const F nc = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(n0, _mm512_set1_ps(cA[k][0])), _mm512_mul_ps(n1, _mm512_set1_ps(cA[k][1]))),
                                               _mm512_mul_ps(n2, _mm512_set1_ps(cA[k][2])));
                    const F s2 = _mm512_add_ps(
                        _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(p00, _mm512_set1_ps(Qb[k][0])), _mm512_mul_ps(p11, _mm512_set1_ps(Qb[k][1]))),
                                      _mm512_add_ps(_mm512_mul_ps(p22, _mm512_set1_ps(Qb[k][2])), _mm512_mul_ps(p01, _mm512_set1_ps(Qb[k][3])))),
                        _mm512_add_ps(_mm512_mul_ps(p02, _mm512_set1_ps(Qb[k][4])), _mm512_mul_ps(p12, _mm512_set1_ps(Qb[k][5]))));
                    const F up = _mm512_sub_ps(nc, vhi), dn = _mm512_sub_ps(vlo, nc);
                    const __mmask16 clrUp = _mm512_mask_cmp_ps_mask(_mm512_cmp_ps_mask(up, _mm512_setzero_ps(), _CMP_GT_OQ), _mm512_mul_ps(up, up), s2, _CMP_GT_OQ);
                    const __mmask16 clrDn = _mm512_mask_cmp_ps_mask(_mm512_cmp_ps_mask(dn, _mm512_setzero_ps(), _CMP_GT_OQ), _mm512_mul_ps(dn, dn), s2, _CMP_GT_OQ);
                    keep |= in & ~(clrUp | clrDn);
                }
                sink += static_cast<uint32_t>(__builtin_popcount(keep));
            }
        }
        t1 = __rdtsc();
        static volatile uint32_t gSink;
        gSink = gSink + sink;
    }
    row.tp[which] += 1;
    row.tl[which] += static_cast<uint64_t>(__builtin_popcount(ready));
    row.tt[which] += t1 - t0;
}
