// Static world grid counters (CC_GRIDSTATS in cc_stage.cpp; docs/twin/collision-x2-night.md "Static world grid").
// Measurement only: included by cc_stage.cpp inside namespace tw::collide_c when CC_GRIDSTATS != 0; the output is
// untouched. The idea under test: a precomputed static grid of the world whose cell lists hold every triangle within a
// wheel's bounding radius (+ eps) of the cell, so that a wheel finds its candidates with one lookup at its centre's
// cell; the body shapes need a proof of their own.
//
// Bit 1 (per lane-call, DetectLanes16 after the merge, GridCount):
//   wheels: at cell sizes 0.25 / 0.5 / 0.75 / 1.0 m, the list of the wheel centre's cell (built lazily, exact
//   triangle-box distance <= 0.364 + 0.01 m, FV rank order) against today's exact wheel box pairs, survivors and raw
//   contacts; the list after today's world-space prefilter; the instances it spans (a per-model grid's lookups); a
//   completeness check (every raw wheel contact's triangle is in the list: must be 0 misses);
//   bodies (every FV_GRIDSTATS_BODY-th lane-call, default 8): oracles (no body survivor, no body raw contact) and
//   proofs: PL0 (every triangle within the body's bounding radius of its centre fails the triangle-plane test: the
//   cell-free limit of a plane-list grid), PL1 (the same on the list of the body centre's 1 m cell), FL1 (one floor
//   plane per 1 m cell: every triangle of the cell's list, clipped to its reach box, lies below it), SC (every
//   FV_GRIDSTATS_SPHERE-th lane-call, default 64: a sphere cover of each body, each sphere clear of all geometry by
//   its exact point distance: the cell-free limit of a distance-field proof), PL1 or FL1 per body.
// Bit 2 (DetectLanes16, before FilterPass16, FilterTime): TSC of FilterPass16 for the body shapes alone (the wheels'
//   boxes emptied, pass 0 on the union of the body boxes), for the wheels alone, and in full, each timed cold on every
//   third pass (the pass's own filter runs after it, untimed): what the filter would still cost per lane-call with the
//   wheels on a grid, or with the bodies proven clear.
// Bit 4 (at Init, per world): memory of the whole-map and per-model grids at each cell size, over all triangles and over
//   the block region (instances of models within 128 m, + 8 m), with the largest contributors: entries from the exact
//   expected cell count of each triangle (mixed volumes of the triangle and cube + ball), distinct cells by Monte
//   Carlo (entries x mean of 1 / list length over random entries), naive build time from measured test costs.
namespace gridstats {

using std::uint32_t;
using std::uint64_t;
constexpr int kSz = 4;
constexpr double kCell[kSz] = {0.25, 0.5, 0.75, 1.0};
constexpr double kEps = 0.01;       // m on the bounding radius (FV's contact needs a triangle point inside the ellipsoid)
constexpr double kBodyCell = 1.0;   // PL1 / FL1 cells
constexpr double kPlaneTol = 1e-3;  // m: the triangle-plane test's float margin (world coordinates up to ~1 km)
constexpr int kHist = 64;
constexpr int kBodyV = 5;           // PL0, PL1, FL1, SC, PL1|FL1
constexpr double kPi = 3.14159265358979323846;

struct D3 {
    double x, y, z;
};
inline D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double Dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline D3 Cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double Len2(D3 a) { return Dot(a, a); }
inline double Comp(D3 a, int i) { return i == 0 ? a.x : i == 1 ? a.y : a.z; }

// Ericson, Real-Time Collision Detection 5.1.5
inline D3 ClosestPtTri(D3 p, D3 a, D3 b, D3 c) {
    const D3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = Dot(ab, ap), d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    const D3 bp = p - b;
    const double d3 = Dot(ab, bp), d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));
    const D3 cp = p - c;
    const double d5 = Dot(ab, cp), d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    const double den = va + vb + vc;
    if (!(std::fabs(den) > 0.0)) return a;   // degenerate (unreachable for a proper triangle)
    return a + ab * (vb / den) + ac * (vc / den);
}
inline double SegClamp(double x) { return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x; }
// Ericson 5.1.9
inline double SegSegDist2(D3 p1, D3 q1, D3 p2, D3 q2) {
    const D3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const double a = Dot(d1, d1), e = Dot(d2, d2), f = Dot(d2, r);
    double s, t;
    if (a <= 1e-30 && e <= 1e-30) return Len2(p1 - p2);
    if (a <= 1e-30) { s = 0.0; t = SegClamp(f / e); }
    else {
        const double c = Dot(d1, r);
        if (e <= 1e-30) { t = 0.0; s = SegClamp(-c / a); }
        else {
            const double b = Dot(d1, d2), den = a * e - b * b;
            s = den > 0.0 ? SegClamp((b * f - c * e) / den) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = SegClamp(-c / a); }
            else if (t > 1.0) { t = 1.0; s = SegClamp((b - c) / a); }
        }
    }
    return Len2((p1 + d1 * s) - (p2 + d2 * t));
}
inline double PointBoxDist2(D3 p, const double lo[3], const double hi[3]) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double v = Comp(p, i);
        const double d = v < lo[i] ? lo[i] - v : v > hi[i] ? v - hi[i] : 0.0;
        s += d * d;
    }
    return s;
}
// Akenine-Moller's triangle / box overlap (separating axes), double
inline bool TriBoxOverlap(const D3 v[3], const double lo[3], const double hi[3]) {
    const D3 c{0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
    const double h[3] = {0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), 0.5 * (hi[2] - lo[2])};
    const D3 w[3] = {v[0] - c, v[1] - c, v[2] - c};
    const D3 e[3] = {w[1] - w[0], w[2] - w[1], w[0] - w[2]};
    for (int i = 0; i < 3; ++i) {   // box axes
        double mn = Comp(w[0], i), mx = mn;
        for (int j = 1; j < 3; ++j) { mn = std::min(mn, Comp(w[j], i)); mx = std::max(mx, Comp(w[j], i)); }
        if (mn > h[i] || mx < -h[i]) return false;
    }
    const D3 n = Cross(e[0], e[1]);   // triangle plane
    {
        const double r = h[0] * std::fabs(n.x) + h[1] * std::fabs(n.y) + h[2] * std::fabs(n.z);
        if (std::fabs(Dot(n, w[0])) > r) return false;
    }
    const D3 ax[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const D3 a = Cross(ax[i], e[j]);
            const double p0 = Dot(a, w[0]), p1 = Dot(a, w[1]), p2 = Dot(a, w[2]);
            const double mn = std::min(p0, std::min(p1, p2)), mx = std::max(p0, std::max(p1, p2));
            const double r = h[0] * std::fabs(a.x) + h[1] * std::fabs(a.y) + h[2] * std::fabs(a.z);
            if (mn > r || mx < -r) return false;
        }
    return true;
}
// exact squared distance between a triangle and a box (0 when they meet): closest features are vertex-box,
// box corner-triangle or edge-edge
inline double TriBoxDist2(const D3 v[3], const double lo[3], const double hi[3]) {
    if (TriBoxOverlap(v, lo, hi)) return 0.0;
    double best = 1e300;
    for (int i = 0; i < 3; ++i) best = std::min(best, PointBoxDist2(v[i], lo, hi));
    D3 cr[8];
    for (int i = 0; i < 8; ++i) cr[i] = D3{(i & 1) ? hi[0] : lo[0], (i & 2) ? hi[1] : lo[1], (i & 4) ? hi[2] : lo[2]};
    for (int i = 0; i < 8; ++i) best = std::min(best, Len2(cr[i] - ClosestPtTri(cr[i], v[0], v[1], v[2])));
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 3; ++b) {
            if (i & (1 << b)) continue;   // edge from corner i along axis b
            const D3 q = cr[i | (1 << b)];
            for (int j = 0; j < 3; ++j) best = std::min(best, SegSegDist2(v[j], v[(j + 1) % 3], cr[i], q));
        }
    return best;
}
// triangle within R of the box: cheap accepts / rejects first
inline bool TriNearBox(const D3 v[3], const double lo[3], const double hi[3], double R) {
    for (int i = 0; i < 3; ++i) {
        double mn = Comp(v[0], i), mx = mn;
        for (int j = 1; j < 3; ++j) { mn = std::min(mn, Comp(v[j], i)); mx = std::max(mx, Comp(v[j], i)); }
        if (mn > hi[i] + R || mx < lo[i] - R) return false;
    }
    const D3 c{0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
    const double hd = 0.5 * std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
    const double dc2 = Len2(c - ClosestPtTri(c, v[0], v[1], v[2]));
    if (dc2 <= R * R) return true;
    if (dc2 > (R + hd) * (R + hd)) return false;
    return TriBoxDist2(v, lo, hi) <= R * R;
}

inline uint64_t CellKey(long ix, long iy, long iz) {
    constexpr long kOff = 1L << 20;
    return (uint64_t(ix + kOff) << 42) | (uint64_t(iy + kOff) << 21) | uint64_t(iz + kOff);
}
inline void CellOf(const double p[3], double c, long ix[3]) {
    for (int a = 0; a < 3; ++a) ix[a] = static_cast<long>(std::floor(p[a] / c));
}

// lazily built lists of one world at one cell size (ranks in FV order)
struct Lists {
    std::unordered_map<uint64_t, uint32_t> idx;
    std::vector<uint32_t> off, cnt, aux;   // aux: distinct planes (body lists)
    std::vector<uint32_t> pool;
    std::uint64_t builds = 0, tests = 0;
};
struct Floor {
    double n[3] = {0, 1, 0};
    double H = 0.0;
    bool ok = false;
};

struct Row {
    uint64_t calls = 0, wheels = 0, box[8] = {}, surv[8] = {}, raw[8] = {};
    uint64_t list[kSz] = {}, pf[kSz] = {}, viol[kSz] = {}, inst[kSz] = {}, gen[kSz] = {}, multi[kSz] = {}, hist[kSz][kHist + 1] = {};
    uint64_t bodyCalls = 0, o1 = 0, o2 = 0, sphereCalls = 0;
    uint64_t clrAll[kBodyV] = {}, clr[kBodyV][4] = {}, len[kBodyV][4] = {}, left[kBodyV][4] = {}, planes[4] = {};
    uint64_t spheres = 0;
    void Add(const Row &r) {
        calls += r.calls; wheels += r.wheels;
        for (int k = 0; k < 8; ++k) { box[k] += r.box[k]; surv[k] += r.surv[k]; raw[k] += r.raw[k]; }
        for (int s = 0; s < kSz; ++s) {
            list[s] += r.list[s]; pf[s] += r.pf[s]; viol[s] += r.viol[s]; inst[s] += r.inst[s]; gen[s] += r.gen[s]; multi[s] += r.multi[s];
            for (int h = 0; h <= kHist; ++h) hist[s][h] += r.hist[s][h];
        }
        bodyCalls += r.bodyCalls; o1 += r.o1; o2 += r.o2; sphereCalls += r.sphereCalls; spheres += r.spheres;
        for (int v = 0; v < kBodyV; ++v) {
            clrAll[v] += r.clrAll[v];
            for (int k = 0; k < 4; ++k) { clr[v][k] += r.clr[v][k]; len[v][k] += r.len[v][k]; left[v][k] += r.left[v][k]; }
        }
        for (int k = 0; k < 4; ++k) planes[k] += r.planes[k];
    }
};
inline double Pct(const uint64_t h[kHist + 1], uint64_t n, double q) {
    const uint64_t want = static_cast<uint64_t>(std::ceil(q * double(n)));
    uint64_t acc = 0;
    for (int i = 0; i <= kHist; ++i) { acc += h[i]; if (acc >= want) return i; }
    return kHist;
}
struct Table {
    std::mutex mu;
    std::vector<std::pair<uint32_t, Row>> rows;
    std::vector<std::string> mem;   // bit 4 lines, per world
    void Add(uint32_t key, const Row &r) {
        std::lock_guard<std::mutex> g(mu);
        for (auto &p : rows) if (p.first == key) { p.second.Add(r); return; }
        rows.emplace_back(key, Row{});
        rows.back().second.Add(r);
    }
    static void Print(const char *name, const Row &w) {
        if (w.calls == 0u) return;
        const double n = double(w.calls), nw = double(std::max<uint64_t>(1, w.wheels));
        uint64_t bw = 0, sw = 0, rw = 0, bb = 0, sb = 0, rb = 0;
        for (int k = 0; k < 4; ++k) { bb += w.box[k]; sb += w.surv[k]; rb += w.raw[k]; }
        for (int k = 4; k < 8; ++k) { bw += w.box[k]; sw += w.surv[k]; rw += w.raw[k]; }
        std::fprintf(stderr, "grid %-9s lane calls %llu | per wheel: box pairs %.3f, survivors %.3f, raw %.3f | per call bodies: box pairs %.2f, "
                             "survivors %.3f, raw %.3f\n", name, static_cast<unsigned long long>(w.calls), double(bw) / nw, double(sw) / nw,
                     double(rw) / nw, double(bb) / n, double(sb) / n, double(rb) / n);
        for (int s = 0; s < kSz; ++s)
            std::fprintf(stderr, "grid %-9s wheel cell %.2f: list %.3f (x%.2f box pairs, x%.2f survivors), p50 %.0f p90 %.0f p99 %.0f, after prefilter %.3f, "
                                 "instances %.3f (>1: %.3f, general rotation: %.4f), missed contacts %llu\n", name, kCell[s], double(w.list[s]) / nw,
                         double(w.list[s]) / double(std::max<uint64_t>(1, bw)), double(w.list[s]) / double(std::max<uint64_t>(1, sw)),
                         Pct(w.hist[s], w.wheels, 0.5), Pct(w.hist[s], w.wheels, 0.9), Pct(w.hist[s], w.wheels, 0.99), double(w.pf[s]) / nw,
                         double(w.inst[s]) / nw, double(w.multi[s]) / nw, double(w.gen[s]) / nw, static_cast<unsigned long long>(w.viol[s]));
        if (w.bodyCalls) {
            const double b = double(w.bodyCalls);
            std::fprintf(stderr, "grid %-9s bodies: %llu sampled calls; oracle: no body survivor %.3f, no body raw contact %.3f\n", name,
                         static_cast<unsigned long long>(w.bodyCalls), double(w.o1) / b, double(w.o2) / b);
            static const char *vn[kBodyV] = {"PL0 plane list, cell-free", "PL1 plane list, 1 m cell", "FL1 floor plane, 1 m cell", "SC sphere cover, exact",
                                              "PL1 or FL1 per body"};
            for (int v = 0; v < kBodyV; ++v) {
                const double m = v == 3 ? double(std::max<uint64_t>(1, w.sphereCalls)) : b;
                std::fprintf(stderr, "grid %-9s   %-26s all 4 cleared %.3f | per body cleared %.3f %.3f %.3f %.3f | list %.1f %.1f %.1f %.1f | left %.2f %.2f %.2f %.2f",
                             name, vn[v], double(w.clrAll[v]) / m, double(w.clr[v][0]) / m, double(w.clr[v][1]) / m, double(w.clr[v][2]) / m,
                             double(w.clr[v][3]) / m, double(w.len[v][0]) / m, double(w.len[v][1]) / m, double(w.len[v][2]) / m,
                             double(w.len[v][3]) / m, double(w.left[v][0]) / m, double(w.left[v][1]) / m, double(w.left[v][2]) / m,
                             double(w.left[v][3]) / m);
                if (v == 1)
                    std::fprintf(stderr, " | distinct planes %.1f %.1f %.1f %.1f", double(w.planes[0]) / b, double(w.planes[1]) / b, double(w.planes[2]) / b,
                                 double(w.planes[3]) / b);
                if (v == 3) std::fprintf(stderr, " | %llu calls, spheres/call %.0f", static_cast<unsigned long long>(w.sphereCalls), double(w.spheres) / m);
                std::fprintf(stderr, "\n");
            }
        }
    }
    ~Table() {
        for (auto &l : mem) std::fputs(l.c_str(), stderr);
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

struct FTime {
    uint64_t passes[3] = {}, lanes[3] = {}, tsc[3] = {};   // [0]: body-only filter, [1]: the full filter, [2]: wheel-only filter
};
struct FTable {
    std::mutex mu;
    std::vector<std::pair<uint32_t, FTime>> rows;
    ~FTable() {
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        FTime all;
        auto line = [](const char *name, const FTime &f) {
            const double b = double(f.tsc[0]) / double(std::max<uint64_t>(1, f.lanes[0])), a = double(f.tsc[1]) / double(std::max<uint64_t>(1, f.lanes[1])),
                         w = double(f.tsc[2]) / double(std::max<uint64_t>(1, f.lanes[2]));
            std::fprintf(stderr, "gridfilter %-9s FilterPass16 TSC per lane-call: full %.0f (%llu passes), bodies only %.0f (%llu passes): wheels' share %.0f (%.1f%%); "
                                 "wheels only %.0f (%llu passes): bodies' share %.0f (%.1f%%)\n",
                         name, a, static_cast<unsigned long long>(f.passes[1]), b, static_cast<unsigned long long>(f.passes[0]), a - b, 100.0 * (a - b) / std::max(1.0, a),
                         w, static_cast<unsigned long long>(f.passes[2]), a - w, 100.0 * (a - w) / std::max(1.0, a));
        };
        for (auto &p : rows) {
            char name[32];
            std::snprintf(name, sizeof name, "%u", p.first);
            line(name, p.second);
            for (int i = 0; i < 3; ++i) { all.passes[i] += p.second.passes[i]; all.lanes[i] += p.second.lanes[i]; all.tsc[i] += p.second.tsc[i]; }
        }
        line("ALL", all);
    }
    void Add(uint32_t key, const FTime &f) {
        std::lock_guard<std::mutex> g(mu);
        for (auto &p : rows)
            if (p.first == key) {
                for (int i = 0; i < 3; ++i) { p.second.passes[i] += f.passes[i]; p.second.lanes[i] += f.lanes[i]; p.second.tsc[i] += f.tsc[i]; }
                return;
            }
        rows.emplace_back(key, f);
    }
};
inline FTable &FTab() { static FTable t; return t; }

struct WorldLocal {
    uint32_t key = 0;
    FTime ft;
    Lists wheel[kSz];
    Lists body[4];
    std::unordered_map<uint64_t, Floor> floor[4];
    Row row;
};
struct Local {
    std::unordered_map<const void *, WorldLocal> worlds;
    uint64_t lane = 0;
    int bodyEvery = 8, sphereEvery = 64;
    std::vector<uint32_t> tmp;
    Local() {
        if (const char *e = std::getenv("FV_GRIDSTATS_BODY")) bodyEvery = std::max(1, std::atoi(e));
        if (const char *e = std::getenv("FV_GRIDSTATS_SPHERE")) sphereEvery = std::max(1, std::atoi(e));
    }
    uint64_t pass = 0;
    ~Local() {
        for (auto &p : worlds) {
            if (p.second.row.calls) Tab().Add(p.second.key, p.second.row);
            if (p.second.ft.passes[0] + p.second.ft.passes[1] + p.second.ft.passes[2]) FTab().Add(p.second.key, p.second.ft);
        }
    }
};
inline Local &Loc() {
    (void)Tab();
    (void)FTab();
    static thread_local Local loc;
    return loc;
}

inline void TriVerts(const world::WorldPackView &v, uint32_t t, D3 out[3]) {
    const Tri &tr = v.tris[t];
    for (int c = 0; c < 3; ++c) out[c] = D3{tr.v[c][0], tr.v[c][1], tr.v[c][2]};
}
// the ranks of the triangles within R of the box (FV order; the FvBvh query emits ranks in increasing order)
template <class Keep>
void BoxList(const world::WorldPackView &v, const FvBvh &fvb, const double lo[3], const double hi[3], double R, std::vector<uint32_t> *out,
             uint64_t *tests, Keep &&keep) {
    Box3 q;
    for (int a = 0; a < 3; ++a) {
        q.lo[a] = static_cast<float>(lo[a] - R) - 1e-3f;
        q.hi[a] = static_cast<float>(hi[a] + R) + 1e-3f;
    }
    fvb.Query(q, [&](std::uint32_t first, const FvBvh::Block &, unsigned mk) {
        for (; mk; mk &= mk - 1u) {
            const uint32_t r = first + static_cast<uint32_t>(__builtin_ctz(mk));
            const cq::Ref &rf = fvb.ref[r];
            if (!keep(rf)) continue;
            D3 tv[3];
            TriVerts(v, rf.local, tv);
            ++*tests;
            if (TriNearBox(tv, lo, hi, R)) out->push_back(r);
        }
    });
}
inline const uint32_t *GetList(Lists &L, const world::WorldPackView &v, const FvBvh &fvb, const long ix[3], double c, double R, uint32_t *n,
                               uint32_t *slotOut = nullptr) {
    const uint64_t key = CellKey(ix[0], ix[1], ix[2]);
    auto it = L.idx.find(key);
    uint32_t slot;
    if (it == L.idx.end()) {
        double lo[3], hi[3];
        for (int a = 0; a < 3; ++a) { lo[a] = double(ix[a]) * c; hi[a] = double(ix[a] + 1) * c; }
        static thread_local std::vector<uint32_t> tmp;
        tmp.clear();
        BoxList(v, fvb, lo, hi, R, &tmp, &L.tests, [](const cq::Ref &) { return true; });
        slot = static_cast<uint32_t>(L.off.size());
        L.off.push_back(static_cast<uint32_t>(L.pool.size()));
        L.cnt.push_back(static_cast<uint32_t>(tmp.size()));
        L.aux.push_back(0u);
        L.pool.insert(L.pool.end(), tmp.begin(), tmp.end());
        L.idx.emplace(key, slot);
        ++L.builds;
    } else {
        slot = it->second;
    }
    *n = L.cnt[slot];
    if (slotOut) *slotOut = slot;
    return L.pool.data() + L.off[slot];
}

// the ellipsoid's support (half width) along unit n: sqrt(sum_i (r_i axis_i . n)^2); pose row: r[9] (axis i at 3 i), t, radii
inline double Support(const float *row, D3 n) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double d = double(row[12 + i]) * (double(row[3 * i]) * n.x + double(row[3 * i + 1]) * n.y + double(row[3 * i + 2]) * n.z);
        s += d * d;
    }
    return std::sqrt(s);
}
// FV's triangle-plane condition fails (no contact possible): the centre behind the front side, or the ellipsoid short of it
inline bool PlaneRejects(const D3 tv[3], const float *row) {
    D3 n = Cross(tv[1] - tv[0], tv[2] - tv[0]);
    const double l = std::sqrt(Len2(n));
    if (!(l > 0.0)) return true;   // FV skips a triangle whose cross product is 0
    n = n * (1.0 / l);
    const D3 t{row[9], row[10], row[11]};
    const double d = Dot(n, t - tv[0]);
    return d < -kPlaneTol || d > Support(row, n) + kPlaneTol;
}

// Sutherland-Hodgman clip of a convex polygon to an axis box
inline int ClipBox(const D3 *in, int n, const double lo[3], const double hi[3], D3 *out /* >= 9 */) {
    D3 a[12], b[12];
    for (int i = 0; i < n; ++i) a[i] = in[i];
    for (int ax = 0; ax < 3 && n > 0; ++ax)
        for (int side = 0; side < 2 && n > 0; ++side) {
            const double lim = side ? hi[ax] : lo[ax];
            auto inside = [&](D3 p) { return side ? Comp(p, ax) <= lim : Comp(p, ax) >= lim; };
            int m = 0;
            for (int i = 0; i < n; ++i) {
                const D3 p = a[i], q = a[(i + 1) % n];
                const bool ip = inside(p), iq = inside(q);
                if (ip) b[m++] = p;
                if (ip != iq) {
                    const double tt = (lim - Comp(p, ax)) / (Comp(q, ax) - Comp(p, ax));
                    b[m++] = p + (q - p) * tt;
                }
            }
            n = m;
            for (int i = 0; i < n; ++i) a[i] = b[i];
        }
    for (int i = 0; i < n; ++i) out[i] = a[i];
    return n;
}

// mixed-volume count of the cells (side c) whose box is within R of the triangle (expected over lattice offsets):
// vol(T + C + B_R) / c^3 = [A w(n) + 1/2 sum_e |e| area(proj_e(C + B_R)) + vol(C + B_R)] / c^3
inline double CellsNear(const D3 *v, int nv, double c, double R) {
    double vol = c * c * c + 6.0 * c * c * R + 3.0 * kPi * c * R * R + 4.0 / 3.0 * kPi * R * R * R;
    if (nv <= 0) return 0.0;
    D3 cr{0, 0, 0};
    for (int j = 1; j + 1 < nv; ++j) cr = cr + Cross(v[j] - v[0], v[j + 1] - v[0]);
    const double l2 = std::sqrt(Len2(cr));
    const double A = 0.5 * l2;
    if (l2 > 0.0) {
        const D3 n = cr * (1.0 / l2);
        vol += A * (c * (std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z)) + 2.0 * R);
    }
    for (int j = 0; j < nv; ++j) {
        const D3 e = v[(j + 1) % nv] - v[j];
        const double le = std::sqrt(Len2(e));
        if (!(le > 0.0)) continue;
        const D3 u = e * (1.0 / le);
        const double pa = c * c * (std::fabs(u.x) + std::fabs(u.y) + std::fabs(u.z));
        const double pp = 2.0 * c * (std::sqrt(std::max(0.0, 1.0 - u.x * u.x)) + std::sqrt(std::max(0.0, 1.0 - u.y * u.y)) + std::sqrt(std::max(0.0, 1.0 - u.z * u.z)));
        vol += 0.5 * le * (pa + pp * R + kPi * R * R);
    }
    return vol / (c * c * c);
}
inline double CellsNear(const D3 v[3], double c, double R) { return CellsNear(v, 3, c, R); }

struct Rng {
    uint64_t s;
    uint64_t Next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double U() { return double(Next() >> 11) * (1.0 / 9007199254740992.0); }
};

}  // namespace gridstats

// Bit 4: memory and build time of the two grid forms for this world (called at the end of Init).
void CcCollision::GridMem() const {
    using namespace gridstats;
    const world::WorldPackView &v = *view_;
    const std::size_t nt = v.tris.size();
    const double R = double(lc_.rmax[4]) + kEps;
    std::vector<double> w(nt), wm(nt);
    std::vector<uint32_t> instCount(v.models.size(), 0u);
    for (const world::Instance &in : v.instances) ++instCount[in.model];
    std::string out;
    char buf[512];
    uint32_t general = 0;
    for (const world::Instance &in : v.instances) general += in.rotation == world::kRotGeneral;
    std::snprintf(buf, sizeof buf, "gridmem %u: %zu instances (%u general rotation), %zu models, %zu model triangles\n",
                  static_cast<unsigned>(nt), v.instances.size(), general, v.models.size(), v.modelTris.size());
    out += buf;
    {   // where the entries are: per model, expected entries at 1 m (one instance), instances, area, extent
        std::vector<double> me(v.models.size(), 0.0), ma(v.models.size(), 0.0);
        double area = 0.0, big = 0.0, eAll = 0.0;
        for (std::size_t t = 0; t < nt; ++t) {
            D3 tv[3];
            TriVerts(v, static_cast<uint32_t>(t), tv);
            const double a = 0.5 * std::sqrt(Len2(Cross(tv[1] - tv[0], tv[2] - tv[0])));
            const uint32_t mdl = v.instances[v.triInfo[t].instance].model;
            const double e = CellsNear(tv, 1.0, R);
            area += a;
            big += a > 1000.0 ? a : 0.0;
            eAll += e;
            me[mdl] += e / double(instCount[mdl]);
            ma[mdl] += a / double(instCount[mdl]);
        }
        std::vector<uint32_t> ord(v.models.size());
        for (uint32_t i = 0; i < ord.size(); ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return me[a] * instCount[a] > me[b] * instCount[b]; });
        std::snprintf(buf, sizeof buf, "gridmem %u: world area %.2f km2 (triangles over 1000 m2: %.2f km2), bounds %.0f..%.0f x %.0f..%.0f x %.0f..%.0f m; "
                      "top models by whole-map entries at 1 m:\n", static_cast<unsigned>(nt), area / 1e6, big / 1e6, double(v.header->boundsLo[0]),
                      double(v.header->boundsHi[0]), double(v.header->boundsLo[1]), double(v.header->boundsHi[1]), double(v.header->boundsLo[2]),
                      double(v.header->boundsHi[2]));
        out += buf;
        for (std::size_t i = 0; i < std::min<std::size_t>(6, ord.size()); ++i) {
            const world::Model &md = v.models[ord[i]];
            std::snprintf(buf, sizeof buf, "gridmem %u:   model %u: %u triangles x %u instances, area %.0f m2 each, extent %.0f x %.0f x %.0f m, "
                          "%.1f%% of the entries\n", static_cast<unsigned>(nt), ord[i], md.triCount, instCount[ord[i]], ma[ord[i]],
                          double(md.hi[0] - md.lo[0]), double(md.hi[1] - md.lo[1]), double(md.hi[2] - md.lo[2]),
                          100.0 * me[ord[i]] * instCount[ord[i]] / std::max(1.0, eAll));
            out += buf;
        }
    }
    // the region a car can reach: the union of the boxes of instances whose model fits in 128 m (blocks; not the
    // stadium surroundings), inflated by 8 m
    double reg[2][3] = {{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
    for (const world::Instance &in : v.instances) {
        const world::Model &md = v.models[in.model];
        if (md.hi[0] - md.lo[0] > 128.0f || md.hi[1] - md.lo[1] > 128.0f || md.hi[2] - md.lo[2] > 128.0f) continue;
        for (int a = 0; a < 3; ++a) { reg[0][a] = std::min(reg[0][a], double(in.lo[a]) - 8.0); reg[1][a] = std::max(reg[1][a], double(in.hi[a]) + 8.0); }
    }
    std::snprintf(buf, sizeof buf, "gridmem %u: block region %.0f..%.0f x %.0f..%.0f x %.0f..%.0f m\n", static_cast<unsigned>(nt), reg[0][0], reg[1][0], reg[0][1],
                  reg[1][1], reg[0][2], reg[1][2]);
    out += buf;
    for (int region = 0; region < 2; ++region)
    for (int s = 0; s < kSz; ++s) {
        const double c = kCell[s];
        double E = 0.0, Em = 0.0, aabb = 0.0, aabbM = 0.0;
        for (std::size_t t = 0; t < nt; ++t) {
            D3 tv[3], cp[12];
            TriVerts(v, static_cast<uint32_t>(t), tv);
            int nc = 3;
            const D3 *poly = tv;
            if (region) {
                nc = ClipBox(tv, 3, reg[0], reg[1], cp);
                poly = cp;
            }
            const double e = nc > 0 ? CellsNear(poly, nc, c, R) : 0.0;
            const double inv = 1.0 / double(instCount[v.instances[v.triInfo[t].instance].model]);
            w[t] = E += e;
            wm[t] = Em += e * inv;
            if (nc == 0) continue;
            double cells = 1.0;
            for (int a = 0; a < 3; ++a) {
                double lo = 1e300, hi = -1e300;
                for (int j = 0; j < nc; ++j) { lo = std::min(lo, Comp(poly[j], a)); hi = std::max(hi, Comp(poly[j], a)); }
                cells *= std::floor((hi + R) / c) - std::floor((lo - R) / c) + 1.0;
            }
            aabb += cells;
            aabbM += cells * inv;
        }
        // distinct cells: entries x E[1 / list length] over uniform random entries (whole map: all triangles; per model: the
        // entry's own instance only)
        Rng rng{0x9E3779B97F4A7C15ull ^ (uint64_t(nt) * 1315423911ull) ^ uint64_t(s) ^ (uint64_t(region) << 40)};
        const int kSamples = 1500;
        double inv[2] = {0.0, 0.0};
        uint64_t tests = 0;
        std::vector<uint32_t> tmp;
        const auto t0 = std::chrono::steady_clock::now();
        for (int form = 0; form < 2; ++form) {
            const std::vector<double> &cum = form == 0 ? w : wm;
            int got = 0;
            for (int smp = 0; smp < kSamples * 4 && got < kSamples; ++smp) {
                const double x = rng.U() * cum.back();
                const uint32_t t = static_cast<uint32_t>(std::upper_bound(cum.begin(), cum.end(), x) - cum.begin());
                if (t >= nt) continue;
                D3 tv[3], cp[12];
                TriVerts(v, t, tv);
                int nc = 3;
                const D3 *poly = tv;
                if (region) {
                    nc = ClipBox(tv, 3, reg[0], reg[1], cp);
                    poly = cp;
                }
                if (nc == 0) continue;
                long lo[3], n3[3];
                for (int a = 0; a < 3; ++a) {
                    double l = 1e300, h = -1e300;
                    for (int j = 0; j < nc; ++j) { l = std::min(l, Comp(poly[j], a)); h = std::max(h, Comp(poly[j], a)); }
                    lo[a] = static_cast<long>(std::floor((l - R) / c));
                    n3[a] = static_cast<long>(std::floor((h + R) / c)) - lo[a] + 1;
                }
                bool found = false;
                double blo[3], bhi[3];
                for (int tr = 0; tr < 4000 && !found; ++tr, ++tests) {
                    for (int a = 0; a < 3; ++a) {
                        const long ix = lo[a] + static_cast<long>(rng.Next() % uint64_t(n3[a]));
                        blo[a] = double(ix) * c;
                        bhi[a] = double(ix + 1) * c;
                    }
                    found = poly == tv ? TriNearBox(tv, blo, bhi, R) : false;
                    if (poly != tv) {   // a cell near the clipped part: near the triangle, centre in the region
                        bool in = true;
                        for (int a = 0; a < 3; ++a) in &= 0.5 * (blo[a] + bhi[a]) >= reg[0][a] - R && 0.5 * (blo[a] + bhi[a]) <= reg[1][a] + R;
                        found = in && TriNearBox(tv, blo, bhi, R);
                    }
                }
                if (!found) continue;
                const uint32_t inst = v.triInfo[t].instance;
                tmp.clear();
                BoxList(v, fvb_, blo, bhi, R, &tmp, &tests, [&](const cq::Ref &rf) { return form == 0 || rf.instance == inst; });
                inv[form] += 1.0 / double(std::max<std::size_t>(1, tmp.size()));
                ++got;
            }
            inv[form] /= double(std::max(1, got));
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double perTest = sec / double(std::max<uint64_t>(1, tests));
        const double N = E * inv[0], Nm = Em * inv[1];
        // hash index: 16 B per cell (8-byte slot at load 0.5); lists: 4 B per entry (whole map: ranks; per model: 2-4 B
        // model triangle ids, counted at 4)
        std::snprintf(buf, sizeof buf,
                      "gridmem %u %s cell %.2f: whole map %.1fM entries, %.1fM cells (mean list %.2f), %.0f MB, naive build %.1f s | per model %.2fM entries, "
                      "%.2fM cells (mean list %.2f), %.1f MB, naive build %.2f s | test %.0f ns\n",
                      static_cast<unsigned>(nt), region ? "region" : "all   ", c, E / 1e6, N / 1e6, E / std::max(1.0, N), (16.0 * N + 4.0 * E) / 1e6, aabb * perTest,
                      Em / 1e6, Nm / 1e6, Em / std::max(1.0, Nm), (16.0 * Nm + 4.0 * Em) / 1e6, aabbM * perTest, perTest * 1e9);
        out += buf;
    }
    std::lock_guard<std::mutex> g(Tab().mu);
    Tab().mem.push_back(out);
}

// Bit 1: the counters of one DetectLanes16 pass (after its merge).
void CcCollision::GridCount(std::uint32_t mask, step::CollisionCache *const *caches) const {
    using namespace gridstats;
    Local &loc = Loc();
    WorldLocal &wl = loc.worlds[this];
    wl.key = static_cast<uint32_t>(fvb_.Size());
    Row &row = wl.row;
    Front16Scratch &fs = Front16Scratch_();
    const PassSurvivors &ps = fs.pass;
    const world::WorldPackView &v = *view_;
    const double Rw = double(lc_.rmax[4]) + kEps;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        const CcCache &cc = *static_cast<const CcCache *>(caches[l]);
        const Memo &m = cc.m;
        const ShapeTable &st = fs.st[l];
        ++row.calls;
        row.wheels += 4;
        // today's box pairs per shape (every memo entry against the shape box), survivors, raw contacts
        for (std::uint32_t e = 0; e < m.n; ++e)
            for (int k = 0; k < 8; ++k)
                row.box[k] += m.lo[0][e] <= st.hi[0][k] && st.lo[0][k] <= m.hi[0][e] && m.lo[1][e] <= st.hi[1][k] && st.lo[1][k] <= m.hi[1][e] &&
                              m.lo[2][e] <= st.hi[2][k] && st.lo[2][k] <= m.hi[2][e];
        uint32_t survK[8] = {}, rawK[8] = {};
        for (std::uint32_t j = ps.s0[l]; j < ps.s1[l]; ++j) ++survK[ps.k[j]];
        std::vector<uint32_t> &rawRank = loc.tmp;
        rawRank.clear();
        for (std::uint32_t s = fs.s0[l]; s < fs.s1[l]; ++s) {
            if (!(fs.hits[s >> 4] >> (s & 15u) & 1u)) continue;
            const Raw &r = fs.res[s];
            ++rawK[r.shape];
            rawRank.push_back(v.triInfo[r.tri].fvOrder << 3 | r.shape);
        }
        for (int k = 0; k < 8; ++k) { row.surv[k] += survK[k]; row.raw[k] += rawK[k]; }
        // wheels
        for (int k = 4; k < 8; ++k) {
            const double p[3] = {fs.wt[k][0][l], fs.wt[k][1][l], fs.wt[k][2][l]};
            for (int s = 0; s < kSz; ++s) {
                long ix[3];
                CellOf(p, kCell[s], ix);
                uint32_t n;
                const uint32_t *lst = GetList(wl.wheel[s], v, fvb_, ix, kCell[s], Rw, &n);
                row.list[s] += n;
                ++row.hist[s][std::min<uint32_t>(n, kHist)];
                for (uint32_t key : rawRank) {
                    if ((key & 7u) != uint32_t(k)) continue;
                    row.viol[s] += !std::binary_search(lst, lst + n, key >> 3);
                }
                // the list after today's prefilter (the lane's memo origin and shape filter: MakePlanes' and Pass' operations)
                uint32_t pf = 0, ni = 0, prevInst = ~0u;
                bool gen = false;
                uint32_t insts[64];
                for (uint32_t i = 0; i < n; ++i) {
                    const cq::Ref &rf = fvb_.ref[lst[i]];
                    const world::Instance &in = v.instances[rf.instance];
                    if (rf.instance != prevInst) {
                        bool seen = false;
                        for (uint32_t q = 0; q < ni && !seen; ++q) seen = insts[q] == rf.instance;
                        if (!seen && ni < 64) insts[ni++] = rf.instance;
                        prevInst = rf.instance;
                        gen |= in.rotation == world::kRotGeneral;
                    }
                    const float *R = in.iso;
                    float tmo[3];
                    for (int a = 0; a < 3; ++a) tmo[a] = static_cast<float>(double(in.iso[9 + a]) - double(m.origin[a]));
                    const float sg = (in.flags & world::kInstMirrored) ? -1.0f : 1.0f;
                    const exact::FastTri &f = fast_[rf.modelTri];
                    bool pass = true;
                    for (int j = 0; j < 4 && pass; ++j) {
                        const float sj = j == 0 ? sg : 1.0f;
                        const float nx = f.x[j], ny = f.y[j], nz = f.z[j];
                        const float wx = sj * ((R[0] * nx + R[3] * ny) + R[6] * nz);
                        const float wy = sj * ((R[1] * nx + R[4] * ny) + R[7] * nz);
                        const float wz = sj * ((R[2] * nx + R[5] * ny) + R[8] * nz);
                        const float wd = sj * f.d[j] + ((wx * tmo[0] + wy * tmo[1]) + wz * tmo[2]);
                        const float dist = ((wx * st.t[kTc0][k] + wy * st.t[kTc0 + 1][k]) + wz * st.t[kTc0 + 2][k]) - wd;
                        if ((j == 0 && dist < st.t[kTlow][k]) || dist > st.t[kTreach][k]) { pass = false; break; }
                        if (dist > st.t[kTlazy][k]) {
                            const float q0 = st.t[kTq0][k], q1 = st.t[kTq0 + 1][k], q2 = st.t[kTq0 + 2][k], q3 = st.t[kTq0 + 3][k], q4 = st.t[kTq0 + 4][k],
                                        q5 = st.t[kTq0 + 5][k];
                            const float a = (q0 * wx + q3 * wy) + q4 * wz;
                            const float b = (q3 * wx + q1 * wy) + q5 * wz;
                            const float cc2 = (q4 * wx + q5 * wy) + q2 * wz;
                            const float h2 = (a * wx + b * wy) + cc2 * wz;
                            const float x = dist - st.t[kTtol][k];
                            if (x > 0.0f && x * x > h2 * exact::kSupportSlack) pass = false;
                        }
                    }
                    pf += pass;
                }
                row.pf[s] += pf;
                row.inst[s] += ni;
                row.multi[s] += ni > 1;
                row.gen[s] += gen;
            }
        }
        // bodies, sampled
        const uint64_t lane = loc.lane++;
        if (lane % uint64_t(loc.bodyEvery) != 0u) continue;
        ++row.bodyCalls;
        uint32_t bs = 0, br = 0;
        for (int k = 0; k < 4; ++k) { bs += survK[k]; br += rawK[k]; }
        row.o1 += bs == 0u;
        row.o2 += br == 0u;
        const bool sph = lane % uint64_t(loc.sphereEvery) == 0u;
        row.sphereCalls += sph;
        bool all[kBodyV] = {true, true, true, true, true};
        bool pl1[4] = {false, false, false, false};
        for (int k = 0; k < 4; ++k) {
            const float *pr = fs.rows[l][k];
            const double Rb = double(lc_.rmax[k]) + kEps;
            const D3 t{pr[9], pr[10], pr[11]};
            // PL0: the triangles within Rb of the centre
            {
                std::vector<uint32_t> &tmp = loc.tmp;
                tmp.clear();
                const double lo[3] = {t.x, t.y, t.z}, hi[3] = {t.x, t.y, t.z};
                uint64_t tests = 0;
                BoxList(v, fvb_, lo, hi, Rb, &tmp, &tests, [](const cq::Ref &) { return true; });
                uint32_t left = 0;
                for (uint32_t r : tmp) {
                    D3 tv[3];
                    TriVerts(v, fvb_.ref[r].local, tv);
                    left += !PlaneRejects(tv, pr);
                }
                row.len[0][k] += tmp.size();
                row.left[0][k] += left;
                row.clr[0][k] += left == 0u;
                all[0] &= left == 0u;
            }
            // PL1 and FL1: the list of the centre's 1 m cell
            {
                const double p[3] = {t.x, t.y, t.z};
                long ix[3];
                CellOf(p, kBodyCell, ix);
                uint32_t n, slot;
                Lists &L = wl.body[k];
                const std::size_t before = L.off.size();
                const uint32_t *lst = GetList(L, v, fvb_, ix, kBodyCell, Rb, &n, &slot);
                if (L.off.size() != before) {   // new list: its distinct planes (FV's stored planes, world, bitwise)
                    std::vector<std::array<std::uint32_t, 4>> pl;
                    for (uint32_t i = 0; i < n; ++i) {
                        const world::TriPlane &tp = v.triPlanes[fvb_.ref[lst[i]].local];
                        std::array<std::uint32_t, 4> a;
                        std::memcpy(a.data(), &tp, 16);
                        pl.push_back(a);
                    }
                    std::sort(pl.begin(), pl.end());
                    L.aux[slot] = static_cast<uint32_t>(std::unique(pl.begin(), pl.end()) - pl.begin());
                }
                uint32_t left = 0;
                for (uint32_t i = 0; i < n; ++i) {
                    D3 tv[3];
                    TriVerts(v, fvb_.ref[lst[i]].local, tv);
                    left += !PlaneRejects(tv, pr);
                }
                row.len[1][k] += n;
                row.left[1][k] += left;
                row.clr[1][k] += left == 0u;
                pl1[k] = left == 0u;
                row.planes[k] += L.aux[slot];
                all[1] &= left == 0u;
                // FL1: the cell's floor plane (built once per cell)
                const uint64_t key = CellKey(ix[0], ix[1], ix[2]);
                auto it = wl.floor[k].find(key);
                if (it == wl.floor[k].end()) {
                    Floor F;
                    double lo[3], hi[3];
                    for (int a = 0; a < 3; ++a) { lo[a] = double(ix[a]) * kBodyCell - Rb; hi[a] = double(ix[a] + 1) * kBodyCell + Rb; }
                    std::vector<D3> pts;   // the list's triangles clipped to the reach box
                    struct Cand { double dist, area, base; D3 n; };
                    std::vector<Cand> cand;   // up-facing triangles: distance from the cell centre, area, offset, unit normal
                    const D3 cc{(double(ix[0]) + 0.5) * kBodyCell, (double(ix[1]) + 0.5) * kBodyCell, (double(ix[2]) + 0.5) * kBodyCell};
                    for (uint32_t i = 0; i < n; ++i) {
                        D3 tv[3], cp[12];
                        TriVerts(v, fvb_.ref[lst[i]].local, tv);
                        const int nc = ClipBox(tv, 3, lo, hi, cp);
                        for (int j = 0; j < nc; ++j) pts.push_back(cp[j]);
                        const D3 cr = Cross(tv[1] - tv[0], tv[2] - tv[0]);
                        const double l2 = std::sqrt(Len2(cr));
                        if (l2 > 0.0 && cr.y / l2 >= 0.5) {
                            const D3 nn = cr * (1.0 / l2);
                            cand.push_back({Len2(cc - ClosestPtTri(cc, tv[0], tv[1], tv[2])), 0.5 * l2, Dot(nn, tv[0]), nn});
                        }
                    }
                    // candidates: the 8 up-facing triangles nearest the cell centre; the floor: least excess of the clipped
                    // geometry above the candidate's own plane (ties: larger area)
                    std::sort(cand.begin(), cand.end(), [](const Cand &a, const Cand &b) { return a.dist < b.dist; });
                    if (cand.size() > 8u) cand.resize(8u);
                    double bestEx = 1e300, bestA = -1.0;
                    for (const Cand &cd : cand) {
                        double H = -1e300;
                        for (const D3 &q : pts) H = std::max(H, Dot(cd.n, q));
                        const double ex = H - cd.base;
                        if (ex < bestEx - 1e-9 || (std::fabs(ex - bestEx) <= 1e-9 && cd.area > bestA)) {
                            bestEx = ex;
                            bestA = cd.area;
                            F.ok = true;
                            F.n[0] = cd.n.x; F.n[1] = cd.n.y; F.n[2] = cd.n.z;
                            F.H = H;
                        }
                    }
                    if (n == 0u) { F.ok = true; F.H = -1e300; }
                    it = wl.floor[k].emplace(key, F).first;
                }
                const Floor &F = it->second;
                bool fl = false;
                if (F.ok) {
                    const D3 nn{F.n[0], F.n[1], F.n[2]};
                    fl = Dot(nn, t) - Support(pr, nn) > F.H + kPlaneTol;
                }
                row.len[2][k] += n;
                row.clr[2][k] += fl;
                all[2] &= fl;
                row.clr[4][k] += fl || pl1[k];
                all[4] &= fl || pl1[k];
            }
            // SC: sphere cover (2D lattice of spacing 0.3 m over the two largest axes, radius sqrt(2 (s/2)^2 + c^2))
            if (sph) {
                int ax[3] = {0, 1, 2};
                std::sort(ax, ax + 3, [&](int a, int b) { return pr[12 + a] > pr[12 + b]; });
                const double ra = pr[12 + ax[0]], rb = pr[12 + ax[1]], rc = pr[12 + ax[2]];
                const double sp = 0.3, h = 0.5 * sp, rho = std::sqrt(2.0 * h * h + rc * rc);
                const D3 ua{pr[3 * ax[0]], pr[3 * ax[0] + 1], pr[3 * ax[0] + 2]}, ub{pr[3 * ax[1]], pr[3 * ax[1] + 1], pr[3 * ax[1] + 2]};
                const int na = static_cast<int>(std::ceil(ra / sp)) + 1, nb = static_cast<int>(std::ceil(rb / sp)) + 1;
                bool clear = true;
                uint32_t touched = 0;
                for (int i = -na; i < na; ++i)
                    for (int j = -nb; j < nb; ++j) {
                        const double u = (i + 0.5) * sp, w2 = (j + 0.5) * sp;
                        const double du = std::max(0.0, std::fabs(u) - h) / ra, dv = std::max(0.0, std::fabs(w2) - h) / rb;
                        if (du * du + dv * dv > 1.0) continue;
                        ++row.spheres;
                        const D3 c = t + ua * u + ub * w2;
                        bool hit = false;
                        Box3 q;
                        const double cc[3] = {c.x, c.y, c.z};
                        for (int a = 0; a < 3; ++a) {
                            q.lo[a] = static_cast<float>(cc[a] - rho - kPlaneTol) - 1e-3f;
                            q.hi[a] = static_cast<float>(cc[a] + rho + kPlaneTol) + 1e-3f;
                        }
                        fvb_.Query(q, [&](std::uint32_t first, const FvBvh::Block &, unsigned mk) {
                            for (; mk && !hit; mk &= mk - 1u) {
                                D3 tv[3];
                                TriVerts(v, fvb_.ref[first + static_cast<uint32_t>(__builtin_ctz(mk))].local, tv);
                                hit = Len2(c - ClosestPtTri(c, tv[0], tv[1], tv[2])) <= (rho + kPlaneTol) * (rho + kPlaneTol);
                            }
                        });
                        touched += hit;
                        clear &= !hit;
                    }
                row.len[3][k] += 0;
                row.left[3][k] += touched;
                row.clr[3][k] += clear;
                all[3] &= clear;
            }
        }
        for (int vv = 0; vv < kBodyV; ++vv) row.clrAll[vv] += all[vv] && (vv != 3 || sph);
    }
}

// Bit 2: before the pass's FilterPass16 (same thread, same memos and tables): in turn the body-only, the full and the
// wheel-only filter timed (the pass's own FilterPass16 then runs warm after it, untimed here).
void CcCollision::FilterTime(Memo *const *memos, std::uint32_t ready) const {
    using namespace gridstats;
    if (ready == 0u) return;
    Local &loc = Loc();
    WorldLocal &wl = loc.worlds[this];
    wl.key = static_cast<uint32_t>(fvb_.Size());
    Front16Scratch &fs = Front16Scratch_();
    static thread_local ShapeTable st[16];
    static thread_local Box3 u[16];
    static thread_local PassSurvivors out;
    const int which = static_cast<int>(loc.pass++ % 3u);
    for (std::uint32_t mm = ready; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        st[l] = fs.st[l];
        u[l] = fs.u[l];
        if (which != 1) {   // 0: the wheels' boxes emptied, 2: the bodies'; pass 0 on the union of the others
            Box3 b = EmptyBox();
            for (int k = 0; k < 8; ++k) {
                if ((lc_.wheel[k] >= 0) == (which == 0)) {
                    for (int a = 0; a < 3; ++a) { st[l].lo[a][k] = 1e30f; st[l].hi[a][k] = -1e30f; }
                    continue;
                }
                for (int a = 0; a < 3; ++a) { b.lo[a] = std::min(b.lo[a], st[l].lo[a][k]); b.hi[a] = std::max(b.hi[a], st[l].hi[a][k]); }
            }
            u[l] = b;
        }
    }
    out.n = 0;
    const std::uint64_t t0 = __rdtsc();
    FilterPass16(memos, st, u, ready, bodyShapes_, &out);
    const std::uint64_t t1 = __rdtsc();
    wl.ft.passes[which] += 1;
    wl.ft.lanes[which] += static_cast<uint64_t>(__builtin_popcount(ready));
    wl.ft.tsc[which] += t1 - t0;
}
