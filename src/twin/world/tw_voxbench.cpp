// tw_voxbench: M2 experiment 4. Per-block-model voxel candidate lists (EMPTY / SINGLE_PLANE /
// GENERAL), one instance query per tick, FV's per-shape merge; compared with FV's merged
// per-shape contacts. Usage: tw_voxbench REC [REC...]   env TW_VOX=0.5 (voxel edge, m)
#include <x86intrin.h>

#include <algorithm>
#include <array>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "tw_blocks.h"
#include "tw_candidates.h"
#include "tw_voxels.h"

using namespace tw;
using Clock = std::chrono::steady_clock;

#ifdef TW_NOSPLIT
#define TW_TSC() std::uint64_t(0)  // no per-phase rdtsc: totals from wall time only
#else
#define TW_TSC() __rdtsc()
#endif
#ifdef TW_PROF
#define TW_NI __attribute__((noinline))
#else
#define TW_NI
#endif

namespace {

int g_classMap[5] = {0, 1, 2, 3, 4};
unsigned g_shapeMask = 0xff;  // timing experiments only
float g_delta = 0.0f;         // tick-level lists: travel margin (m); 0 = per-pose lookups
bool g_tile = false, g_lookupOnly = false, g_ahead = false;          // tick tiles with clearance skipping (TW_TILE)
// Core clock during the run: a chain of dependent adds runs at one add per cycle.
double CoreGHz() {
    std::uint64_t x = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < 25000000; ++i) {
        asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    }
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    return 2e8 / ns;
}
double g_ghz = 5.3;

std::size_t g_dbgSub = ~std::size_t(0), curSub = 0; int g_dbgShape = -1;
int ShapeClass(int s) { return g_classMap[s >= 4 ? 0 : 4 - s]; }  // wheels, shape 3, 2, 1, 0

// Deep SINGLE_PLANE contact: every coplanar internal-edge feature FV could emit has its unit-space
// point f on the plane disk {-d n_u + q : q perp n_u, |q| <= sqrt(1 - d^2)}; its world normal is
// N(d n_u - q) = (d/h) n - N q. Bound the angle to n from above (|a| and the largest singular value
// of the in-plane part, each maximised separately) and accept when it is below 30 deg with margin,
// so FV's merge drops every such feature (N.N' > cos 30) and the merged result is the FACE alone.
// Does the unit-space contact disk (centre -d n_u, radius sqrt(1 - d^2)) reach any internal edge of
// the SINGLE_PLANE triangles? A coplanar feature needs a triangle boundary point inside the disk, and the
// plane's boundary edges are out of reach by construction, so "no" means FV emits the FACE alone.
TW_NI inline bool DiskReachesEdge(const ShapeMap &m, const VoxSeg *S, const VoxEdge *E, std::uint32_t count, std::int32_t skipOwner, float d, const float Mn[3], float h) {
    const float ih = 1.0f / h;
    const float p[3] = {-d * Mn[0] * ih, -d * Mn[1] * ih, -d * Mn[2] * ih};
    const float r2 = (1.0f - d * d) * 1.0001f + 1e-6f;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (E[i].owner >= 0 && E[i].owner == skipOwner) continue;  // boundary edge of the FACE triangle itself
        float a[3], b[3];
        UnitPoint(m, S[E[i].seg].a, a); UnitPoint(m, S[E[i].seg].b, b);
        const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
        const float l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        float t = l2 > 0.0f ? (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / l2 : 0.0f;
        t = std::fmin(1.0f, std::fmax(0.0f, t));
        const float q[3] = {ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]};
        if (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] <= r2) return true;
    }
    return false;
}

TW_NI inline bool FeaturesDropped(const ShapeMap &m, const float n[3], const float Mn[3], float h, float d) {
    const float ih = 1.0f / h;
    const float nu[3] = {Mn[0] * ih, Mn[1] * ih, Mn[2] * ih};
    int ax = 0;
    for (int a = 1; a < 3; ++a) if (std::fabs(nu[a]) < std::fabs(nu[ax])) ax = a;
    float e1[3] = {ax == 0 ? 0.0f : (ax == 1 ? nu[2] : -nu[1]), ax == 1 ? 0.0f : (ax == 0 ? -nu[2] : nu[0]), ax == 2 ? 0.0f : (ax == 0 ? nu[1] : -nu[0])};
    const float il = 1.0f / std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    for (float &x : e1) x *= il;
    const float e2[3] = {nu[1] * e1[2] - nu[2] * e1[1], nu[2] * e1[0] - nu[0] * e1[2], nu[0] * e1[1] - nu[1] * e1[0]};
    float g1[3], g2[3];
    for (int i = 0; i < 3; ++i) {
        g1[i] = m.N[i][0] * e1[0] + m.N[i][1] * e1[1] + m.N[i][2] * e1[2];
        g2[i] = m.N[i][0] * e2[0] + m.N[i][1] * e2[1] + m.N[i][2] * e2[2];
    }
    const float a1 = n[0] * g1[0] + n[1] * g1[1] + n[2] * g1[2], a2 = n[0] * g2[0] + n[1] * g2[1] + n[2] * g2[2];
    float b1[3], b2[3];
    for (int i = 0; i < 3; ++i) { b1[i] = g1[i] - a1 * n[i]; b2[i] = g2[i] - a2 * n[i]; }
    const float P = b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2], Q = b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2];
    const float R = b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2];
    const float sig2 = 0.5f * (P + Q + std::sqrt((P - Q) * (P - Q) + 4.0f * R * R));
    const float hh2 = std::fmax(0.0f, 1.0f - d * d), hh = std::sqrt(hh2);
    const float den = d * ih - hh * std::sqrt(a1 * a1 + a2 * a2);
    return den > 0.0f && hh2 * sig2 <= (0.95f / 3.0f) * den * den;
}

struct Perm { int p[3]; float sgn[3]; };  // column j of R has its +-1 at row p[j]

// shape map in a signed-permutation instance frame: copies and sign flips only (+ one subtraction)
inline void ToModelPerm(const ShapeMap &w, const Instance &in, const Perm &P, ShapeMap *m) {
    for (int j = 0; j < 3; ++j) {
        const int pj = P.p[j]; const float sj = P.sgn[j];
        m->t[j] = sj * (w.t[pj] - in.T[pj]);
        for (int k = 0; k < 3; ++k) {
            m->M[j][k] = sj * w.M[pj][k];
            m->N[j][k] = sj * w.N[pj][k];
            m->A[k][j] = w.A[k][pj] * sj;
        }
    }
}

bool usePre = true, split = false;

// Cheap conservative reject before the exact test (model frame, squared forms, no sqrt or divide):
// the centre behind the plane, the plane out of reach, or the ellipsoid entirely outside one edge's side plane.
TW_NI inline bool Reject(const ShapeMap &m, const Model &model, const Element &e) {
    const float *n = e.n, *v = &model.loop[3 * e.first], *u = &model.edgeU[3 * e.first];
    const float sd = (n[0] * (m.t[0] - v[0]) + n[1] * (m.t[1] - v[1])) + n[2] * (m.t[2] - v[2]);
    if (sd < -1e-4f) return true;
    float Mn[3];
    for (int j = 0; j < 3; ++j) Mn[j] = (m.M[0][j] * n[0] + m.M[1][j] * n[1]) + m.M[2][j] * n[2];
    const float h2 = (Mn[0] * Mn[0] + Mn[1] * Mn[1]) + Mn[2] * Mn[2];
    if (sd > 0.0f && sd * sd > h2 * 1.0001f + 1e-8f) return true;
    for (std::uint32_t i = 0; i < e.count; ++i) {
        const float *a = v + 3 * i, *w = u + 3 * i;
        const float se = (w[0] * (m.t[0] - a[0]) + w[1] * (m.t[1] - a[1])) + w[2] * (m.t[2] - a[2]);
        if (!(se > 0.0f)) continue;
        float Mu[3];
        for (int j = 0; j < 3; ++j) Mu[j] = (m.M[0][j] * w[0] + m.M[1][j] * w[1]) + m.M[2][j] * w[2];
        const float hu2 = (Mu[0] * Mu[0] + Mu[1] * Mu[1]) + Mu[2] * Mu[2];
        if (se * se > hu2 * 1.0001f + 1e-8f) return true;
    }
    return false;
}

void Extrapolate(const tw::Substep &s, float tau, tw::Substep *o);

struct Pipe {
    const BlockWorld *bw; const VoxWorld *vw;
    std::vector<Perm> perm;
    std::vector<Box3> instBoxR;  // instance box inflated by the largest class radius
    std::vector<std::uint32_t> near, stamp;
    std::uint32_t cur = 0, tick = 0xffffffffu;
    Box3 tickBox = EmptyBox();
    float ir[kShapes][3]; bool haveIr = false;
    float dsafe[kShapes];
    ShapeMap sm[kShapes];
    Box3 shapeBox[kShapes];
    std::vector<MergedContact> raw, merged[kShapes];
    struct PairQ { int sh; std::uint32_t id; const VoxGrid *g; float tm[3]; std::uint32_t bi, local; const std::uint16_t *cell; const VoxList *L; bool fallback; };
    std::vector<PairQ> pq;
    // stats
    std::size_t clsSeen[8] = {}, groupExact[kShapes] = {}, planeHits = 0, groupsVisited = 0, deepByShape[kShapes] = {}, deepTried[kShapes] = {}, deepEdge[kShapes] = {}, itemsByShape[kShapes] = {}, exactByShape[kShapes] = {}, clsRest = 0, closedForms = 0, boxPass = 0, exactHits = 0, listItems = 0, nearSum = 0, clsEmpty = 0, clsSingle = 0, clsGeneral = 0, clsSingleDeep = 0, exactTests = 0, lookups = 0, queries = 0, requeries = 0, fallbackModels = 0;
    std::uint64_t tSetup = 0, tQuery = 0, tShapes = 0, tMerge = 0;

    void Init() {
        delta = g_delta; tickMode = g_delta > 0.0f && !std::getenv("TW_NOTICK");
        perm.resize(bw->instances.size()); instBoxR.resize(bw->instances.size()); stamp.assign(bw->instances.size(), 0);
        for (std::size_t i = 0; i < bw->instances.size(); ++i) {
            const Instance &in = bw->instances[i];
            for (int j = 0; j < 3; ++j) { perm[i].p[j] = 0; perm[i].sgn[j] = 1; for (int r = 0; r < 3; ++r) if (in.R[r][j] != 0.0f) { perm[i].p[j] = r; perm[i].sgn[j] = in.R[r][j]; } }
            instBoxR[i] = Inflate(in.box, vw->classRadius[VoxModel::kClasses - 1] + vw->margin + 0.01f);
        }
    }
    TW_NI void QueryInstances(const Box3 &q) {
        near.clear(); ++cur;
        auto visit = [&](const std::vector<std::uint32_t> &ids) {
            for (std::uint32_t id : ids) { if (stamp[id] == cur) continue; stamp[id] = cur; if (Overlap(instBoxR[id], q)) near.push_back(id); }
        };
        const int x0 = static_cast<int>(std::floor((q.lo[0] - 2.0f) / 32.0f)), x1 = static_cast<int>(std::floor((q.hi[0] + 2.0f) / 32.0f));
        const int y0 = static_cast<int>(std::floor((q.lo[1] - 2.0f) / 8.0f)), y1 = static_cast<int>(std::floor((q.hi[1] + 2.0f) / 8.0f));
        const int z0 = static_cast<int>(std::floor((q.lo[2] - 2.0f) / 32.0f)), z1 = static_cast<int>(std::floor((q.hi[2] + 2.0f) / 32.0f));
        for (int x = x0; x <= x1; ++x) for (int y = y0; y <= y1; ++y) for (int z = z0; z <= z1; ++z) {
            auto it = bw->cells.find(BlockWorld::CellKey(x, y, z));
            if (it != bw->cells.end()) visit(it->second);
        }
        auto huge = bw->cells.find(BlockWorld::CellKey(0x7ffff, 0x7ffff, 0x7ffff));
        if (huge != bw->cells.end()) visit(huge->second);
    }
    TW_NI void SetupShapes(const tw::Substep &s) {
        if (!haveIr) {
            for (int k = 0; k < kShapes; ++k) {
                float mx = 0, mn = 1e30f;
                for (int j = 0; j < 3; ++j) { ir[k][j] = 1.0f / s.shapes[k].r[j]; mx = std::fmax(mx, s.shapes[k].r[j]); mn = std::fmin(mn, s.shapes[k].r[j]); }
                const float q = 0.5f * mn / mx;  // tan(<30 deg) / anisotropy
                dsafe[k] = 1.0f / std::sqrt(1.0f + q * q);
                if (std::getenv("TW_VOXSTAT")) std::printf("  shape %d r (%.4f %.4f %.4f) dsafe %.4f\n", k, s.shapes[k].r[0], s.shapes[k].r[1], s.shapes[k].r[2], dsafe[k]);
                if (mx > vw->classRadius[ShapeClass(k)]) std::printf("WARNING shape %d radius %.4f > class radius %.4f\n", k, mx, vw->classRadius[ShapeClass(k)]);
            }
            haveIr = true;
        }
        for (int k = 0; k < kShapes; ++k) {
            const Shape &sh = s.shapes[k];
            ShapeMap &m = sm[k];
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) { m.A[j][i] = sh.axis[j][i] * ir[k][j]; m.M[i][j] = sh.axis[j][i] * sh.r[j]; m.N[i][j] = m.A[j][i]; }
            for (int i = 0; i < 3; ++i) {
                m.t[i] = sh.t[i];
                const float h = std::fabs(m.M[i][0]) + std::fabs(m.M[i][1]) + std::fabs(m.M[i][2]);
                shapeBox[k].lo[i] = sh.t[i] - h - kEps; shapeBox[k].hi[i] = sh.t[i] + h + kEps;
            }
        }
    }
    TW_NI void Emit(int s, const Instance &in, const Contact &c, std::uint8_t material) {
        MergedContact mc;
        ToWorldPoint(in, c.P, mc.c.P); ToWorldDir(in, c.N, mc.c.N); ToWorldDir(in, c.Sep, mc.c.Sep);
        for (int k = 0; k < 3; ++k) mc.c.K[k] = c.K[k];
        mc.c.kind = c.kind; mc.primary = c.kind == kFace; mc.material = material;
        raw.push_back(mc);
    }
    std::size_t np = 0, pb[kShapes] = {}, pe[kShapes] = {};
    bool tickMode = false, queried = false;
    float delta = 0.0f, anchor[kShapes][3] = {};
    std::uint32_t pairTick = 0xffffffffu;
    std::size_t relookups = 0, pairReuse = 0, lookupPasses = 0;
    std::uint64_t a0 = 0, a1 = 0, a2 = 0;
    virtual ~Pipe() = default;
    virtual void Prepare(const tw::Substep &s) {
        a0 = TW_TSC();
        SetupShapes(s);
        Box3 u = EmptyBox();
        for (int k = 0; k < kShapes; ++k) u = Union(u, shapeBox[k]);
        a1 = TW_TSC();
        if (s.tick != tick || !Contains(tickBox, u)) {
            const bool re = s.tick == tick;
            tick = s.tick;
            const float w = std::sqrt(s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1] + s.angVel[2] * s.angVel[2]);
            const float sp = std::sqrt(s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] + s.vel[2] * s.vel[2]);
            tickBox = Inflate(u, (sp + 2.6f * w) * 0.01f * 1.25f + 0.05f);
            QueryInstances(tickBox);
            queried = true;
            nearSum += near.size();
            if (re) ++requeries; else ++queries;
        }
        a2 = TW_TSC();
        // tick mode: the pairs found at the tick's anchor (the extrapolated mid-tick centres) stay valid while
        // every centre is within delta of its anchor (lists built for radius + delta)
        bool need = !tickMode || pairTick != s.tick || queried;
        queried = false;
        if (tickMode && !need)
            for (int sh = 0; sh < kShapes && !need; ++sh) {
                const float dx = sm[sh].t[0] - anchor[sh][0], dy = sm[sh].t[1] - anchor[sh][1], dz = sm[sh].t[2] - anchor[sh][2];
                if (dx * dx + dy * dy + dz * dz > delta * delta) { need = true; ++relookups; }
            }
        if (!need) { ++pairReuse; return; }
        if (tickMode && pairTick != s.tick) {
            tw::Substep mid;
            Extrapolate(s, 0.005f, &mid);
            for (int sh = 0; sh < kShapes; ++sh) {
                const float *c = mid.shapes[sh].t;
                const float dx = sm[sh].t[0] - c[0], dy = sm[sh].t[1] - c[1], dz = sm[sh].t[2] - c[2];
                const bool ok = dx * dx + dy * dy + dz * dz <= delta * delta;
                for (int a = 0; a < 3; ++a) anchor[sh][a] = ok ? c[a] : sm[sh].t[a];
            }
        } else {
            for (int sh = 0; sh < kShapes; ++sh) for (int a = 0; a < 3; ++a) anchor[sh][a] = sm[sh].t[a];
        }
        pairTick = s.tick;
        ++lookupPasses;
        // stage A: (shape, instance) pairs whose centre is in the instance's reach box and voxel grid
        if (pq.size() < kShapes * near.size()) pq.resize(kShapes * near.size());
        np = 0;
        for (int sh = 0; sh < kShapes; ++sh) {
            const int cls = ShapeClass(sh);
            const float *t = anchor[sh];
            pb[sh] = np;
            if (!((g_shapeMask >> sh) & 1)) { pe[sh] = np; continue; }
            for (std::uint32_t id : near) {
                const Box3 &ib = instBoxR[id];
                if (t[0] < ib.lo[0] || t[0] > ib.hi[0] || t[1] < ib.lo[1] || t[1] > ib.hi[1] || t[2] < ib.lo[2] || t[2] > ib.hi[2]) continue;
                const Instance &in = bw->instances[id];
                const VoxGrid &g = vw->models[in.model].grid[cls];
                ++lookups;
                PairQ &q = pq[np];
                q.sh = sh; q.id = id; q.g = &g; q.cell = nullptr; q.L = nullptr;
                if (in.permutation) { const Perm &P = perm[id]; for (int j = 0; j < 3; ++j) q.tm[j] = P.sgn[j] * (t[P.p[j]] - in.T[P.p[j]]); }
                else { ShapeMap tmp = sm[sh]; for (int a = 0; a < 3; ++a) tmp.t[a] = t[a]; ShapeMap o; ToModel(tmp, in, &o); q.tm[0] = o.t[0]; q.tm[1] = o.t[1]; q.tm[2] = o.t[2]; }
                if (g.dim[0] == 0) { q.fallback = true; ++np; continue; }
                q.fallback = false;
                int v[3]; bool inside = true;
                for (int a = 0; a < 3; ++a) {
                    const float f = (q.tm[a] - g.org[a]) * g.inv;
                    v[a] = static_cast<int>(f);
                    inside = inside && f >= 0.0f && v[a] < g.dim[a];
                }
                if (!inside) continue;
                q.bi = static_cast<std::uint32_t>((static_cast<std::size_t>(v[0] >> 3) * g.bdim[1] + (v[1] >> 3)) * g.bdim[2] + (v[2] >> 3));
                q.local = static_cast<std::uint32_t>(((v[0] & 7) << 6) | ((v[1] & 7) << 3) | (v[2] & 7));
                __builtin_prefetch(&g.brick[q.bi]);
                ++np;
            }
            pe[sh] = np;
        }
        // stages B-D: brick -> cell -> list -> items, each stage issued for all pairs before the next (memory-level parallelism)
        for (std::size_t k = 0; k < np; ++k) if (!pq[k].fallback) { PairQ &q = pq[k]; q.cell = &q.g->cells[(static_cast<std::size_t>(q.g->brick[q.bi]) << 9) | q.local]; __builtin_prefetch(q.cell); }
        for (std::size_t k = 0; k < np; ++k) if (pq[k].cell) { PairQ &q = pq[k]; const std::uint16_t lid = *q.cell; if (lid) { q.L = &q.g->lists[lid]; __builtin_prefetch(q.L); } }
        for (std::size_t k = 0; k < np; ++k) if (pq[k].L) { const PairQ &q = pq[k]; __builtin_prefetch(&q.g->items[q.L->first]); }
    }
    // the list's coplanar group: closed form, disk/angle checks, exact tests on its members when needed
    TW_NI void GroupContacts(int sh, std::uint32_t id, const Instance &in, const Model &model, const VoxGrid &g, const VoxList &L, const ShapeMap &mm, bool *exactAllOut) {
        bool exactAll = false;
        const std::uint16_t *it = &g.items[L.first];
                    // plane reach (closed form, squared compare first)
                    const float sd = (L.n[0] * (mm.t[0] - L.p[0]) + L.n[1] * (mm.t[1] - L.p[1])) + L.n[2] * (mm.t[2] - L.p[2]);
                    float Mn[3], h2 = 0.0f;
                    bool hit = sd > 0.0f;
                    if (hit) {
                        for (int j = 0; j < 3; ++j) Mn[j] = (mm.M[0][j] * L.n[0] + mm.M[1][j] * L.n[1]) + mm.M[2][j] * L.n[2];
                        h2 = (Mn[0] * Mn[0] + Mn[1] * Mn[1]) + Mn[2] * Mn[2];
                        hit = sd * sd <= h2;
                    }
                    if (hit) {
                        ++planeHits;
                        const float h = std::sqrt(h2), d = sd / h;
                        float MMn[3];
                        for (int i = 0; i < 3; ++i) MMn[i] = (mm.M[i][0] * Mn[0] + mm.M[i][1] * Mn[1]) + mm.M[i][2] * Mn[2];
                        const float kk = sd / h2;
                        const float P[3] = {mm.t[0] - kk * MMn[0], mm.t[1] - kk * MMn[1], mm.t[2] - kk * MMn[2]};
                        // the member holding P (FV's FACE triangle); covered voxels hold P by construction
                        std::int32_t T = L.covered ? 0x7fffffff : -1;
                        int faces = L.covered ? 1 : 0;
                        if (!L.covered) for (std::uint32_t j = 0; j < L.memberCount; ++j) {  // every member holding P gives FV a FACE
                            const Element &e = model.elements[it[j]];
                            const float *v = &model.loop[3 * e.first], *u = &model.edgeU[3 * e.first];
                            bool in3 = true;
                            for (int i = 0; i < 3 && in3; ++i)
                                in3 = (u[3 * i] * (P[0] - v[3 * i]) + u[3 * i + 1] * (P[1] - v[3 * i + 1])) + u[3 * i + 2] * (P[2] - v[3 * i + 2]) <= 0.0f;
                            if (in3) { if (T < 0) T = static_cast<std::int32_t>(j); ++faces; }
                        }
                        if (faces > 1) T = 0x7fffffff;  // several FACE triangles: test the disk against every edge
                        // any other member's boundary point inside the unit-space contact disk? (FV feature candidates)
                        const bool reach = DiskReachesEdge(mm, vw->models[in.model].segs.data(), &g.edges[L.edgeFirst], L.edgeCount, T, d, Mn, h);
                        bool closed = !reach;
                        if (reach && T >= 0) { ++deepEdge[sh]; closed = FeaturesDropped(mm, L.n, Mn, h, d); }
                        if (closed) {
                            if (T >= 0) {
                                if (g_dbgSub == curSub && g_dbgShape == sh) {
                                    std::printf("  DBG closed FACE inst %u model %u cls %u covered %u T %d members %u edges %u mat %u sd %.6f d %.6f reach %d tm (%.4f %.4f %.4f) P (%.4f %.4f %.4f) voxel %.3f\n", id, in.model, L.cls, L.covered, T, L.memberCount, L.edgeCount, L.material, sd, d, reach ? 1 : 0, mm.t[0], mm.t[1], mm.t[2], P[0], P[1], P[2], g.size);
                                    for (std::uint32_t j = 0; j < L.memberCount; ++j) {
                                        const Element &e = model.elements[it[j]];
                                        const float *v = &model.loop[3 * e.first];
                                        std::printf("     member %u mat %u (%.4f %.4f %.4f) (%.4f %.4f %.4f) (%.4f %.4f %.4f)\n", it[j], e.material, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
                                    }
                                }
                                const float ks = (sd - h) / h2, ih = 1.0f / h;
                                Contact c; c.kind = kFace;
                                for (int i = 0; i < 3; ++i) { c.P[i] = P[i]; c.N[i] = L.n[i]; c.Sep[i] = ks * MMn[i]; c.K[i] = Mn[i] * ih; }
                                for (int fcount = 0; fcount < faces; ++fcount) Emit(sh, in, c, L.material);
                                ++closedForms;
                            }
                        } else {
                            ++groupExact[sh]; exactAll = true;
                            for (std::uint32_t j = 0; j < L.memberCount; ++j) {
                                const Element &e = model.elements[it[j]];
                                if (usePre && Reject(mm, model, e)) continue;
                                ++exactTests; ++exactByShape[sh];
                                Contact c; bool fast;
                                if (ElementTest(mm, model, e, &c, &fast) != kNone) { Emit(sh, in, c, e.material); ++exactHits; }
                            }
                        }
                    }
        if (exactAll) *exactAllOut = true;
    }
    // one rest triangle: model-frame AABB, squared-form reject, exact test
    TW_NI void RestTri(int sh, const Instance &in, const Model &model, const Element &e, const ShapeMap &mm, const float hm[3]) {
        ++listItems; ++itemsByShape[sh];
        if (e.lo[0] > mm.t[0] + hm[0] || e.hi[0] < mm.t[0] - hm[0] || e.lo[1] > mm.t[1] + hm[1] || e.hi[1] < mm.t[1] - hm[1] ||
            e.lo[2] > mm.t[2] + hm[2] || e.hi[2] < mm.t[2] - hm[2]) return;
        ++boxPass;
        if (usePre && Reject(mm, model, e)) return;
        ++exactTests; ++exactByShape[sh];
        Contact c; bool fast;
        if (ElementTest(mm, model, e, &c, &fast) != kNone) { Emit(sh, in, c, e.material); ++exactHits; }
    }
    virtual void ProcessShape(int sh) {
        {
            raw.clear();
            bool anySingle = false, anyGeneral = false, anyRest = false, deep = false;
            for (std::size_t k = pb[sh]; k < pe[sh]; ++k) {
                const PairQ &q = pq[k];
                const std::uint32_t id = q.id;
                const Instance &in = bw->instances[id];
                const VoxGrid &g = *q.g;
                const Model &model = bw->models[in.model];
                ShapeMap mm;
                if (q.fallback) {  // model without voxels (overflow): test elements overlapping the shape box
                    ++fallbackModels;
                    if (in.permutation) ToModelPerm(sm[sh], in, perm[id], &mm); else ToModel(sm[sh], in, &mm);
                    for (std::uint32_t ei = 0; ei < model.elements.size(); ++ei) {
                        const Element &e = model.elements[ei];
                        bool ov = true;
                        for (int a = 0; a < 3; ++a) ov = ov && e.lo[a] <= mm.t[a] + 2.0f && e.hi[a] >= mm.t[a] - 2.0f;
                        if (!ov) continue;
                        Contact c; bool fast;
                        ++exactTests;
                        if (ElementTest(mm, model, e, &c, &fast) != kNone) Emit(sh, in, c, e.material);
                    }
                    anyGeneral = true;
                    continue;
                }
                if (!q.L) continue;
                const VoxList &L = *q.L;
                if (g_lookupOnly) { anySingle |= L.cls != 0; continue; }
                if (in.permutation) ToModelPerm(sm[sh], in, perm[id], &mm); else ToModel(sm[sh], in, &mm);
                ++clsSeen[L.cls];
                const std::uint16_t *it = &g.items[L.first];
                bool exactAll = false;
                if (L.memberCount) GroupContacts(sh, id, in, model, g, L, mm, &exactAll);
                if (L.memberCount < L.count) {
                    anyRest = true;
                    for (std::uint32_t j = L.memberCount; j < L.count; ++j) __builtin_prefetch(&model.elements[it[j]]);
                    float hm[3];
                    for (int i = 0; i < 3; ++i) hm[i] = std::fabs(mm.M[i][0]) + std::fabs(mm.M[i][1]) + std::fabs(mm.M[i][2]) + 1e-3f;
                    for (std::uint32_t j = L.memberCount; j < L.count; ++j) RestTri(sh, in, model, model.elements[it[j]], mm, hm);
                }
                if (exactAll) anyGeneral = true; else anySingle = true;
            }
            if (anyGeneral) ++clsGeneral; else if (anyRest) ++clsRest; else if (anySingle) ++clsSingle; else ++clsEmpty;
            clsSingleDeep += deep;
            const std::uint64_t m0 = split ? __rdtsc() : 0;
            bool prim = false, feat = false;
            for (const MergedContact &c : raw) { prim |= c.primary; feat |= !c.primary; }
            if (prim && feat) MergeShape(raw, &merged[sh]); else merged[sh].swap(raw);
            if (split) tMerge += __rdtsc() - m0;
        }
    }
    void Step(const tw::Substep &s) {
        Prepare(s);
        for (int sh = 0; sh < kShapes; ++sh) ProcessShape(sh);
        const std::uint64_t a3 = TW_TSC();
        tSetup += a1 - a0; tQuery += a2 - a1; tShapes += a3 - a2;
    }
    // ---- contact-set predictor (substep policy): K = set of (shape, material) in FV's merged contacts
    using KSet = std::array<std::uint64_t, 4>;
    static void KAdd(KSet &k, int m) { k[m >> 6] |= 1ull << (m & 63); }
    KSet KOf(int sh) const { KSet k{}; for (const MergedContact &c : merged[sh]) KAdd(k, c.material); return k; }
    std::size_t fastShapes = 0, slowShapes = 0;
    // Exact material set of one shape from the voxel data alone, or false (unknown: run the narrow phase).
    // Exact because only FACE contacts (always kept by FV's merge) and features of the same group (same
    // material) can come from a reached plane holding P, and the rest triangles all fail the cheap reject.
    virtual bool FastK(int sh, KSet *out) {
        KSet k{};
        for (std::size_t q = pb[sh]; q < pe[sh]; ++q) {
            const PairQ &p = pq[q];
            if (p.fallback) return false;
            if (!p.L) continue;
            const VoxList &L = *p.L;
            const Instance &in = bw->instances[p.id];
            ShapeMap mm;
            if (in.permutation) ToModelPerm(sm[sh], in, perm[p.id], &mm); else ToModel(sm[sh], in, &mm);
            const Model &model = bw->models[in.model];
            const std::uint16_t *it = &p.g->items[L.first];
            if (L.memberCount) {
                const float sd = (L.n[0] * (mm.t[0] - L.p[0]) + L.n[1] * (mm.t[1] - L.p[1])) + L.n[2] * (mm.t[2] - L.p[2]);
                if (sd > 0.0f) {
                    float Mn[3];
                    for (int j = 0; j < 3; ++j) Mn[j] = (mm.M[0][j] * L.n[0] + mm.M[1][j] * L.n[1]) + mm.M[2][j] * L.n[2];
                    const float h2 = (Mn[0] * Mn[0] + Mn[1] * Mn[1]) + Mn[2] * Mn[2];
                    if (sd * sd <= h2) {
                        bool face = L.covered != 0;
                        if (!face) {
                            float MMn[3];
                            for (int i = 0; i < 3; ++i) MMn[i] = (mm.M[i][0] * Mn[0] + mm.M[i][1] * Mn[1]) + mm.M[i][2] * Mn[2];
                            const float kk = sd / h2;
                            const float P[3] = {mm.t[0] - kk * MMn[0], mm.t[1] - kk * MMn[1], mm.t[2] - kk * MMn[2]};
                            for (std::uint32_t j = 0; j < L.memberCount && !face; ++j) {
                                const Element &e = model.elements[it[j]];
                                const float *v = &model.loop[3 * e.first], *u = &model.edgeU[3 * e.first];
                                bool in3 = true;
                                for (int i = 0; i < 3 && in3; ++i)
                                    in3 = (u[3 * i] * (P[0] - v[3 * i]) + u[3 * i + 1] * (P[1] - v[3 * i + 1])) + u[3 * i + 2] * (P[2] - v[3 * i + 2]) <= 0.0f;
                                face = in3;
                            }
                            if (!face) {
                                const float h = std::sqrt(h2);
                                if (DiskReachesEdge(mm, vw->models[in.model].segs.data(), &p.g->edges[L.edgeFirst], L.edgeCount, -1, sd / h, Mn, h)) return false;  // features possible
                            }
                        }
                        if (face) KAdd(k, L.material);
                    }
                }
            }
            if (L.memberCount < L.count) {
                float hm[3];
                for (int i = 0; i < 3; ++i) hm[i] = std::fabs(mm.M[i][0]) + std::fabs(mm.M[i][1]) + std::fabs(mm.M[i][2]) + 1e-3f;
                for (std::uint32_t j = L.memberCount; j < L.count; ++j) {
                    const Element &e = model.elements[it[j]];
                    if (e.lo[0] > mm.t[0] + hm[0] || e.hi[0] < mm.t[0] - hm[0] || e.lo[1] > mm.t[1] + hm[1] || e.hi[1] < mm.t[1] - hm[1] ||
                        e.lo[2] > mm.t[2] + hm[2] || e.hi[2] < mm.t[2] - hm[2]) continue;
                    if (!Reject(mm, model, e)) return false;
                }
            }
        }
        *out = k;
        return true;
    }
    // true = the contact set at this pose differs from K0 (early exit); exact: every shape through the narrow phase
    std::uint64_t tpPrep = 0, tpFast = 0, tpSlow = 0;
    bool ProbeDiffers(const tw::Substep &pose, const KSet *K0, bool exact, KSet *Kout) {
        const std::uint64_t q0 = TW_TSC();
        Prepare(pose);
        tpPrep += TW_TSC() - q0;
        bool differs = false;
        for (int sh = 0; sh < kShapes; ++sh) {
            KSet k;
            const std::uint64_t q1 = TW_TSC();
            const bool fast = !exact && FastK(sh, &k);
            const std::uint64_t q2 = TW_TSC();
            if (!exact) tpFast += q2 - q1;
            if (!fast) { ProcessShape(sh); k = KOf(sh); if (!exact) { ++slowShapes; tpSlow += TW_TSC() - q2; } }
            else ++fastShapes;
            if (Kout) Kout[sh] = k;
            if (k != K0[sh]) { differs = true; if (!exact) return true; }
        }
        return differs;
    }
};

// Tick tiles: the lists are fetched once per tick at an anchor pose (the extrapolated mid-tick pose) from
// voxel lists built for radius + delta. Each candidate (a list's coplanar group, or a rest triangle) gets a
// clearance g at the anchor: the shape must move by more than g before it can touch the candidate,
//   group:    g = max(sd - h(n), -sd)
//   triangle: g = max(sd - h(n), -sd, max_j(se_j - h(u_j)))   (h(u) = |M^T u|, se_j edge side-plane distances)
// A later pose moves a shape by D = |c - c_anchor| + ||M - M_anchor||_F, which bounds every change of sd, se_j
// and h(u); candidates with g > D are skipped, exactly. D > delta re-anchors (the lists only cover delta).
struct TilePipe : Pipe {
    struct Cand { float g; std::uint8_t isGroup, slot; std::uint16_t elem; const VoxList *L; const VoxGrid *grid; };
    static constexpr int kSlots = 16;
    std::vector<Cand> tile[kShapes];
    std::uint32_t slotInst[kShapes][kSlots]; int nslot[kShapes] = {};
    ShapeMap slotMM[kShapes][kSlots]; std::uint32_t slotPose[kShapes][kSlots] = {}; std::uint32_t poseId = 1;
    float ac[kShapes][3] = {}, aM[kShapes][9] = {}, D[kShapes] = {};
    std::uint32_t tileTick = 0xffffffffu;
    std::size_t anchorsN = 0, reanchorsN = 0, activeN = 0, candN = 0, posesN = 0, skippedShapes = 0;

    static float H(const ShapeMap &m, const float *u) {
        float a[3];
        for (int j = 0; j < 3; ++j) a[j] = (m.M[0][j] * u[0] + m.M[1][j] * u[1]) + m.M[2][j] * u[2];
        return std::sqrt((a[0] * a[0] + a[1] * a[1]) + a[2] * a[2]);
    }
    static float TriGap(const ShapeMap &m, const Model &model, const Element &e) {
        const float *n = e.n, *v = &model.loop[3 * e.first], *u = &model.edgeU[3 * e.first];
        if (!(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.5f)) return -1e30f;
        const float sd = (n[0] * (m.t[0] - v[0]) + n[1] * (m.t[1] - v[1])) + n[2] * (m.t[2] - v[2]);
        float g = std::fmax(sd - H(m, n), -sd);
        for (std::uint32_t i = 0; i < e.count; ++i) {
            const float *a = v + 3 * i, *w = u + 3 * i;
            const float se = (w[0] * (m.t[0] - a[0]) + w[1] * (m.t[1] - a[1])) + w[2] * (m.t[2] - a[2]);
            g = std::fmax(g, se - H(m, w));
        }
        return g - 1e-4f;
    }
    const ShapeMap &MM(int sh, int slot) {
        if (slotPose[sh][slot] != poseId) {
            const std::uint32_t id = slotInst[sh][slot];
            const Instance &in = bw->instances[id];
            if (in.permutation) ToModelPerm(sm[sh], in, perm[id], &slotMM[sh][slot]); else ToModel(sm[sh], in, &slotMM[sh][slot]);
            slotPose[sh][slot] = poseId;
        }
        return slotMM[sh][slot];
    }
    void Anchor(const tw::Substep &pose) {
        ++anchorsN;
        const bool tm = tickMode;
        tickMode = false;
        Pipe::Prepare(pose);  // shape maps at the anchor, instance query, lookups at the anchor centres
        tickMode = tm;
        ++poseId;
        for (int sh = 0; sh < kShapes; ++sh) {
            for (int a = 0; a < 3; ++a) ac[sh][a] = sm[sh].t[a];
            for (int i = 0; i < 9; ++i) aM[sh][i] = sm[sh].M[i / 3][i % 3];
            tile[sh].clear(); nslot[sh] = 0;
            for (std::size_t k = pb[sh]; k < pe[sh]; ++k) {
                const PairQ &q = pq[k];
                if (!q.fallback && !q.L) continue;
                if (nslot[sh] == kSlots) { std::fprintf(stderr, "tile slots exhausted\n"); std::abort(); }
                const int slot = nslot[sh]++;
                slotInst[sh][slot] = q.id;
                const ShapeMap &mm = MM(sh, slot);
                const Model &model = bw->models[bw->instances[q.id].model];
                if (q.fallback) {
                    for (std::uint32_t ei = 0; ei < model.elements.size(); ++ei)
                        tile[sh].push_back(Cand{TriGap(mm, model, model.elements[ei]), 0, static_cast<std::uint8_t>(slot), static_cast<std::uint16_t>(ei), nullptr, q.g});
                    continue;
                }
                const VoxList &L = *q.L;
                const std::uint16_t *it = &q.g->items[L.first];
                if (L.memberCount) {
                    const float sd = (L.n[0] * (mm.t[0] - L.p[0]) + L.n[1] * (mm.t[1] - L.p[1])) + L.n[2] * (mm.t[2] - L.p[2]);
                    tile[sh].push_back(Cand{std::fmax(sd - H(mm, L.n), -sd) - 1e-4f, 1, static_cast<std::uint8_t>(slot), 0, &L, q.g});
                }
                for (std::uint32_t j = L.memberCount; j < L.count; ++j)
                    tile[sh].push_back(Cand{TriGap(mm, model, model.elements[it[j]]), 0, static_cast<std::uint8_t>(slot), it[j], &L, q.g});
            }
            candN += tile[sh].size();
        }
    }
    void Prepare(const tw::Substep &s) override {
        a0 = TW_TSC();
        ++posesN;
        if (s.tick != tileTick) {
            tileTick = s.tick;
            tw::Substep mid;
            Extrapolate(s, 0.005f, &mid);
            Anchor(mid);
        }
        SetupShapes(s);
        ++poseId;
        bool re = false;
        for (int sh = 0; sh < kShapes; ++sh) {
            const float dx = sm[sh].t[0] - ac[sh][0], dy = sm[sh].t[1] - ac[sh][1], dz = sm[sh].t[2] - ac[sh][2];
            float f2 = 0.0f;
            for (int i = 0; i < 9; ++i) { const float q = sm[sh].M[i / 3][i % 3] - aM[sh][i]; f2 += q * q; }
            D[sh] = std::sqrt(dx * dx + dy * dy + dz * dz) + std::sqrt(f2) + 1e-5f;
            re = re || D[sh] > delta;
        }
        if (re) {  // re-anchor at this pose
            ++reanchorsN;
            Anchor(s);
            SetupShapes(s);
            ++poseId;
            for (int sh = 0; sh < kShapes; ++sh) D[sh] = 1e-5f;
        }
        a1 = a2 = TW_TSC();
    }
    void ProcessShape(int sh) override {
        raw.clear();
        bool any = false;
        for (const Cand &c : tile[sh]) {
            if (c.g > D[sh]) continue;
            any = true; ++activeN;
            const ShapeMap &mm = MM(sh, c.slot);
            const std::uint32_t id = slotInst[sh][c.slot];
            const Instance &in = bw->instances[id];
            const Model &model = bw->models[in.model];
            if (c.isGroup) { bool ex = false; GroupContacts(sh, id, in, model, *c.grid, *c.L, mm, &ex); }
            else {
                float hm[3];
                for (int i = 0; i < 3; ++i) hm[i] = std::fabs(mm.M[i][0]) + std::fabs(mm.M[i][1]) + std::fabs(mm.M[i][2]) + 1e-3f;
                RestTri(sh, in, model, model.elements[c.elem], mm, hm);
            }
        }
        if (!any) ++skippedShapes;
        bool prim = false, feat = false;
        for (const MergedContact &c : raw) { prim |= c.primary; feat |= !c.primary; }
        if (prim && feat) MergeShape(raw, &merged[sh]); else merged[sh].swap(raw);
    }
    bool FastK(int sh, KSet *out) override {
        KSet k{};
        for (const Cand &c : tile[sh]) {
            if (c.g > D[sh]) continue;
            const ShapeMap &mm = MM(sh, c.slot);
            const Instance &in = bw->instances[slotInst[sh][c.slot]];
            const Model &model = bw->models[in.model];
            if (!c.isGroup) {
                const Element &e = model.elements[c.elem];
                float hm[3];
                for (int i = 0; i < 3; ++i) hm[i] = std::fabs(mm.M[i][0]) + std::fabs(mm.M[i][1]) + std::fabs(mm.M[i][2]) + 1e-3f;
                if (e.lo[0] > mm.t[0] + hm[0] || e.hi[0] < mm.t[0] - hm[0] || e.lo[1] > mm.t[1] + hm[1] || e.hi[1] < mm.t[1] - hm[1] ||
                    e.lo[2] > mm.t[2] + hm[2] || e.hi[2] < mm.t[2] - hm[2]) continue;
                if (!Reject(mm, model, e)) return false;
                continue;
            }
            const VoxList &L = *c.L;
            const std::uint16_t *it = &c.grid->items[L.first];
            const float sd = (L.n[0] * (mm.t[0] - L.p[0]) + L.n[1] * (mm.t[1] - L.p[1])) + L.n[2] * (mm.t[2] - L.p[2]);
            if (!(sd > 0.0f)) continue;
            float Mn[3];
            for (int j = 0; j < 3; ++j) Mn[j] = (mm.M[0][j] * L.n[0] + mm.M[1][j] * L.n[1]) + mm.M[2][j] * L.n[2];
            const float h2 = (Mn[0] * Mn[0] + Mn[1] * Mn[1]) + Mn[2] * Mn[2];
            if (sd * sd > h2) continue;
            bool face = L.covered != 0;
            if (!face) {
                float MMn[3];
                for (int i = 0; i < 3; ++i) MMn[i] = (mm.M[i][0] * Mn[0] + mm.M[i][1] * Mn[1]) + mm.M[i][2] * Mn[2];
                const float kk = sd / h2;
                const float P[3] = {mm.t[0] - kk * MMn[0], mm.t[1] - kk * MMn[1], mm.t[2] - kk * MMn[2]};
                for (std::uint32_t j = 0; j < L.memberCount && !face; ++j) {
                    const Element &e = model.elements[it[j]];
                    const float *v = &model.loop[3 * e.first], *u = &model.edgeU[3 * e.first];
                    bool in3 = true;
                    for (int i = 0; i < 3 && in3; ++i)
                        in3 = (u[3 * i] * (P[0] - v[3 * i]) + u[3 * i + 1] * (P[1] - v[3 * i + 1])) + u[3 * i + 2] * (P[2] - v[3 * i + 2]) <= 0.0f;
                    face = in3;
                }
                if (!face) {
                    const float h = std::sqrt(h2);
                    if (DiskReachesEdge(mm, vw->models[in.model].segs.data(), &c.grid->edges[L.edgeFirst], L.edgeCount, -1, sd / h, Mn, h)) return false;
                }
            }
            if (face) KAdd(k, L.material);
        }
        *out = k;
        return true;
    }
};

// ---- lean pipeline: SIMD lookups across the 8 shapes (AVX-512VL, 8 lanes = shapes), lazy shape maps ----
// All grids' brick tables, bricks and lists are packed into world-global arrays so one gather serves lanes
// of different radius classes; per model, a lane table holds each lane's grid origin, scale, dims and offsets.
struct LaneGrid {
    alignas(32) float org[3][8];
    alignas(32) float inv[8];
    alignas(32) int dim[3][8];
    alignas(32) int bdim1[8], bdim2[8], brickOff[8], listOff[8];
    __mmask8 valid = 0;
};
struct PackedVox {
    std::vector<std::uint32_t> brick;   // global brick ids
    std::vector<std::uint16_t> cells;   // all bricks, + padding for 32-bit gathers
    std::vector<VoxList> lists;
    std::vector<const VoxGrid *> listGrid;
    std::vector<LaneGrid> lane;          // per model
    std::size_t Bytes() const { return brick.size() * 4 + cells.size() * 2 + lists.size() * (sizeof(VoxList) + sizeof(void *)) + lane.size() * sizeof(LaneGrid); }
    void Build(const VoxWorld &vw) {
        lane.assign(vw.models.size(), LaneGrid{});
        lists.assign(1, VoxList{}); listGrid.assign(1, nullptr);
        for (std::size_t mi = 0; mi < vw.models.size(); ++mi) {
            int bOff[VoxModel::kClasses] = {}, lOff[VoxModel::kClasses] = {};
            for (int c = 0; c < VoxModel::kClasses; ++c) {
                const VoxGrid &g = vw.models[mi].grid[c];
                if (g.dim[0] == 0) continue;
                bOff[c] = static_cast<int>(brick.size());
                const std::uint32_t cellBase = static_cast<std::uint32_t>(cells.size() >> 9);
                for (std::uint32_t b : g.brick) brick.push_back(b + cellBase);
                cells.insert(cells.end(), g.cells.begin(), g.cells.end());
                lOff[c] = static_cast<int>(lists.size());
                lists.insert(lists.end(), g.lists.begin(), g.lists.end());
                listGrid.insert(listGrid.end(), g.lists.size(), &g);
            }
            LaneGrid &L = lane[mi];
            for (int sh = 0; sh < kShapes; ++sh) {
                const int c = ShapeClass(sh);
                const VoxGrid &g = vw.models[mi].grid[c];
                if (g.dim[0] == 0) continue;
                L.valid |= static_cast<__mmask8>(1u << sh);
                for (int a = 0; a < 3; ++a) { L.org[a][sh] = g.org[a]; L.dim[a][sh] = g.dim[a]; }
                L.inv[sh] = g.inv; L.bdim1[sh] = g.bdim[1]; L.bdim2[sh] = g.bdim[2]; L.brickOff[sh] = bOff[c]; L.listOff[sh] = lOff[c];
            }
        }
        cells.push_back(0); cells.push_back(0);
    }
};

// Hot working set: cache lines of map data touched per tick (TW_WSET); kinds: 0 voxel index (brick table, bricks),
// 1 list headers, 2 list items, 3 triangles (element, vertices, edge normals), 4 plane edges and segments
struct WorkingSet {
    bool on = false;
    std::vector<std::pair<std::uintptr_t, int>> cur;
    std::unordered_map<std::uintptr_t, std::uint32_t> lastSeen;
    std::vector<std::uintptr_t> prevTick;
    std::uint32_t ticks = 0;
    double lines = 0, newLines = 0, footprint64 = 0, byKind[5] = {};
    void Touch(const void *p, std::size_t bytes, int kind) {
        if (!on) return;
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p) >> 6, b = (reinterpret_cast<std::uintptr_t>(p) + bytes - 1) >> 6;
        for (std::uintptr_t l = a; l <= b; ++l) cur.push_back({l, kind});
    }
    void EndTick() {
        if (!on || cur.empty()) return;
        std::sort(cur.begin(), cur.end());
        std::vector<std::uintptr_t> u;
        for (std::size_t i = 0; i < cur.size(); ++i)
            if (i == 0 || cur[i].first != cur[i - 1].first) { u.push_back(cur[i].first); byKind[cur[i].second] += 1; }
        std::size_t fresh = 0;
        for (std::uintptr_t l : u) {
            auto it = lastSeen.find(l);
            if (it == lastSeen.end() || ticks - it->second > 64) ++fresh;
            lastSeen[l] = ticks;
        }
        std::size_t live = 0;
        for (const auto &kv : lastSeen) live += ticks - kv.second <= 64;
        lines += u.size(); newLines += fresh; footprint64 += live;
        ++ticks; cur.clear();
    }
};
WorkingSet g_ws;

struct LeanPipe : Pipe {
    const PackedVox *pv = nullptr;
    const tw::Substep *cur = nullptr;
    struct Ent { std::uint8_t sh; std::uint8_t fallback; std::uint32_t id, list; };
    std::vector<Ent> ents, sorted;
    std::uint32_t eb[kShapes + 1] = {};
    bool ready[kShapes] = {};
    float classR[kShapes] = {};
    bool init = false;
    std::size_t entsN = 0, posesN = 0;
    std::uint32_t wsTick = 0xffffffffu, aheadTick = 0xffffffffu;
    void TouchList(const VoxList &L, const VoxGrid &g, const Model &model, int vm) {
        if (!g_ws.on) return;
        g_ws.Touch(&L, sizeof(VoxList), 1);
        g_ws.Touch(&g.items[L.first], 2 * L.count, 2);
        for (std::uint32_t j = 0; j < L.count; ++j) {
            const Element &e = model.elements[g.items[L.first + j]];
            g_ws.Touch(&e, sizeof(Element), 3); g_ws.Touch(&model.loop[3 * e.first], 36, 3); g_ws.Touch(&model.edgeU[3 * e.first], 36, 3);
        }
        if (L.edgeCount) {
            g_ws.Touch(&g.edges[L.edgeFirst], sizeof(VoxEdge) * L.edgeCount, 4);
            for (std::uint32_t j = 0; j < L.edgeCount; ++j) g_ws.Touch(&vw->models[vm].segs[g.edges[L.edgeFirst + j].seg], sizeof(VoxSeg), 4);
        }
    }

    TW_NI void SetupOne(int k) {
        if (ready[k]) return;
        const Shape &sh = cur->shapes[k];
        ShapeMap &m = sm[k];
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) { m.A[j][i] = sh.axis[j][i] * ir[k][j]; m.M[i][j] = sh.axis[j][i] * sh.r[j]; m.N[i][j] = m.A[j][i]; }
        for (int i = 0; i < 3; ++i) m.t[i] = sh.t[i];
        ready[k] = true;
    }
    void LaneLookups(const float (*c)[8], int level) {
        alignas(32) int gl[8];
        const __m256 cx = _mm256_load_ps(c[0]), cy = _mm256_load_ps(c[1]), cz = _mm256_load_ps(c[2]);
        for (std::uint32_t id : near) {
            const Box3 &ib = instBoxR[id];
            __mmask8 m = _mm256_cmp_ps_mask(cx, _mm256_set1_ps(ib.lo[0]), _CMP_GE_OQ) & _mm256_cmp_ps_mask(cx, _mm256_set1_ps(ib.hi[0]), _CMP_LE_OQ) &
                         _mm256_cmp_ps_mask(cy, _mm256_set1_ps(ib.lo[1]), _CMP_GE_OQ) & _mm256_cmp_ps_mask(cy, _mm256_set1_ps(ib.hi[1]), _CMP_LE_OQ) &
                         _mm256_cmp_ps_mask(cz, _mm256_set1_ps(ib.lo[2]), _CMP_GE_OQ) & _mm256_cmp_ps_mask(cz, _mm256_set1_ps(ib.hi[2]), _CMP_LE_OQ);
            m &= static_cast<__mmask8>(g_shapeMask);
            if (!m) continue;
            const Instance &in = bw->instances[id];
            const LaneGrid &lg = pv->lane[in.model];
            __mmask8 fb = level ? 0 : m & static_cast<__mmask8>(~lg.valid);
            while (fb) { const int k = __builtin_ctz(fb); fb &= fb - 1; ents.push_back(Ent{static_cast<std::uint8_t>(k), 1, id, 0}); }
            m &= lg.valid;
            if (!m) continue;
            if (!level) ++lookups;
            __m256 t0, t1, t2;
            if (in.permutation) {
                const Perm &P = perm[id];
                const __m256 src[3] = {cx, cy, cz};
                t0 = _mm256_mul_ps(_mm256_set1_ps(P.sgn[0]), _mm256_sub_ps(src[P.p[0]], _mm256_set1_ps(in.T[P.p[0]])));
                t1 = _mm256_mul_ps(_mm256_set1_ps(P.sgn[1]), _mm256_sub_ps(src[P.p[1]], _mm256_set1_ps(in.T[P.p[1]])));
                t2 = _mm256_mul_ps(_mm256_set1_ps(P.sgn[2]), _mm256_sub_ps(src[P.p[2]], _mm256_set1_ps(in.T[P.p[2]])));
            } else {
                const __m256 d0 = _mm256_sub_ps(cx, _mm256_set1_ps(in.T[0])), d1 = _mm256_sub_ps(cy, _mm256_set1_ps(in.T[1])), d2 = _mm256_sub_ps(cz, _mm256_set1_ps(in.T[2]));
                __m256 *o[3] = {&t0, &t1, &t2};
                for (int i = 0; i < 3; ++i)
                    *o[i] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(in.R[0][i]), d0), _mm256_mul_ps(_mm256_set1_ps(in.R[1][i]), d1)),
                                          _mm256_mul_ps(_mm256_set1_ps(in.R[2][i]), d2));
            }
            const __m256 f0 = _mm256_mul_ps(_mm256_sub_ps(t0, _mm256_load_ps(lg.org[0])), _mm256_load_ps(lg.inv));
            const __m256 f1 = _mm256_mul_ps(_mm256_sub_ps(t1, _mm256_load_ps(lg.org[1])), _mm256_load_ps(lg.inv));
            const __m256 f2 = _mm256_mul_ps(_mm256_sub_ps(t2, _mm256_load_ps(lg.org[2])), _mm256_load_ps(lg.inv));
            const __m256 zero = _mm256_setzero_ps();
            m &= _mm256_cmp_ps_mask(f0, zero, _CMP_GE_OQ) & _mm256_cmp_ps_mask(f1, zero, _CMP_GE_OQ) & _mm256_cmp_ps_mask(f2, zero, _CMP_GE_OQ);
            const __m256i v0 = _mm256_cvttps_epi32(f0), v1 = _mm256_cvttps_epi32(f1), v2 = _mm256_cvttps_epi32(f2);
            m &= _mm256_cmp_epi32_mask(v0, _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.dim[0])), _MM_CMPINT_LT) &
                 _mm256_cmp_epi32_mask(v1, _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.dim[1])), _MM_CMPINT_LT) &
                 _mm256_cmp_epi32_mask(v2, _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.dim[2])), _MM_CMPINT_LT);
            if (!m) continue;
            const __m256i bi = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(v0, 3), _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.bdim1))),
                                                                     _mm256_srli_epi32(v1, 3)),
                                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.bdim2))),
                                 _mm256_srli_epi32(v2, 3)),
                _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.brickOff)));
            const __m256i seven = _mm256_set1_epi32(7);
            const __m256i local = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(v0, seven), 6), _mm256_slli_epi32(_mm256_and_si256(v1, seven), 3)),
                                                  _mm256_and_si256(v2, seven));
            if (level == 4) {  // prefetch the brick table entries
                alignas(32) int bia[8];
                _mm256_store_si256(reinterpret_cast<__m256i *>(bia), bi);
                for (__mmask8 q = m; q; q &= q - 1) __builtin_prefetch(&pv->brick[bia[__builtin_ctz(q)]]);
                continue;
            }
            const __m256i bid = _mm256_mmask_i32gather_epi32(_mm256_setzero_si256(), m, bi, pv->brick.data(), 4);
            const __m256i cidx = _mm256_or_si256(_mm256_slli_epi32(bid, 9), local);
            if (level == 3) {  // prefetch the bricks' cells
                alignas(32) int cia[8];
                _mm256_store_si256(reinterpret_cast<__m256i *>(cia), cidx);
                for (__mmask8 q = m; q; q &= q - 1) __builtin_prefetch(&pv->cells[cia[__builtin_ctz(q)]]);
                continue;
            }
            const __m256i c32 = _mm256_mmask_i32gather_epi32(_mm256_setzero_si256(), m, cidx, pv->cells.data(), 2);
            const __m256i lid = _mm256_and_si256(c32, _mm256_set1_epi32(0xffff));
            if (g_ws.on) {
                alignas(32) int bia[8], cia[8];
                _mm256_store_si256(reinterpret_cast<__m256i *>(bia), bi); _mm256_store_si256(reinterpret_cast<__m256i *>(cia), cidx);
                for (__mmask8 mm2 = m; mm2; mm2 &= mm2 - 1) { const int k = __builtin_ctz(mm2); g_ws.Touch(&pv->brick[bia[k]], 4, 0); g_ws.Touch(&pv->cells[cia[k]], 2, 0); }
            }
            __mmask8 nz = m & _mm256_cmp_epi32_mask(lid, _mm256_setzero_si256(), _MM_CMPINT_NE);
            if (!nz) continue;
            _mm256_store_si256(reinterpret_cast<__m256i *>(gl), _mm256_add_epi32(lid, _mm256_load_si256(reinterpret_cast<const __m256i *>(lg.listOff))));
            while (nz) {
                const int k = __builtin_ctz(nz); nz &= nz - 1;
                if (level == 2) { __builtin_prefetch(&pv->lists[gl[k]]); continue; }
                if (level == 1) {  // list header is cached: prefetch its items and triangles
                    const VoxList &L = pv->lists[gl[k]];
                    const VoxGrid &g = *pv->listGrid[gl[k]];
                    const Model &model = bw->models[in.model];
                    const std::uint16_t *it = &g.items[L.first];
                    for (std::uint32_t j = 0; j < L.count; ++j) { const Element &e = model.elements[it[j]]; __builtin_prefetch(&e); __builtin_prefetch(&model.loop[3 * e.first]); }
                    continue;
                }
                __builtin_prefetch(&pv->lists[gl[k]]);
                ents.push_back(Ent{static_cast<std::uint8_t>(k), 0, id, static_cast<std::uint32_t>(gl[k])});
            }
        }
    }
    // centres extrapolated tau ahead (constant v, w, gravity), from the tick-start pose
    void CentresAhead(const tw::Substep &s, float tau, float (*c)[8]) {
        const float *R = s.carRot;
        double Rn[9];
        const double wx = s.angVel[0], wy = s.angVel[1], wz = s.angVel[2], w = std::sqrt(wx * wx + wy * wy + wz * wz);
        for (int col = 0; col < 3; ++col) {
            double v[3] = {R[3 * col], R[3 * col + 1], R[3 * col + 2]};
            if (w * tau > 1e-9) {
                const double ux = wx / w, uy = wy / w, uz = wz / w, a = w * tau, cs = std::cos(a), sn = std::sin(a);
                const double dot = ux * v[0] + uy * v[1] + uz * v[2];
                const double cx2 = uy * v[2] - uz * v[1], cy2 = uz * v[0] - ux * v[2], cz2 = ux * v[1] - uy * v[0];
                const double n0 = v[0] * cs + cx2 * sn + ux * dot * (1 - cs), n1 = v[1] * cs + cy2 * sn + uy * dot * (1 - cs), n2 = v[2] * cs + cz2 * sn + uz * dot * (1 - cs);
                v[0] = n0; v[1] = n1; v[2] = n2;
            }
            for (int i = 0; i < 3; ++i) Rn[3 * col + i] = v[i];
        }
        for (int k = 0; k < kShapes; ++k) {
            const float *t = s.shapes[k].t;
            double lt[3];
            for (int j = 0; j < 3; ++j) lt[j] = R[3 * j] * (t[0] - s.carPos[0]) + R[3 * j + 1] * (t[1] - s.carPos[1]) + R[3 * j + 2] * (t[2] - s.carPos[2]);
            for (int i = 0; i < 3; ++i)
                c[i][k] = static_cast<float>(s.carPos[i] + s.vel[i] * tau + (i == 1 ? -15.0 * tau * tau : 0.0) + Rn[i] * lt[0] + Rn[3 + i] * lt[1] + Rn[6 + i] * lt[2]);
        }
    }
    TW_NI void Prepare(const tw::Substep &s) override {
        if (!init) {
            SetupShapes(s);  // radii, warnings
            for (int k = 0; k < kShapes; ++k) classR[k] = vw->classRadius[ShapeClass(k)] + vw->margin + 1e-3f;
            init = true;
        }
        ++posesN;
        if (g_ws.on && s.tick != wsTick) { g_ws.EndTick(); wsTick = s.tick; }
        cur = &s;
        for (int k = 0; k < kShapes; ++k) ready[k] = false;
        alignas(32) float c[3][8];
        Box3 u = EmptyBox();
        for (int k = 0; k < kShapes; ++k)
            for (int a = 0; a < 3; ++a) {
                c[a][k] = s.shapes[k].t[a];
                u.lo[a] = std::fmin(u.lo[a], c[a][k] - classR[k]); u.hi[a] = std::fmax(u.hi[a], c[a][k] + classR[k]);
            }
        if (s.tick != tick || !Contains(tickBox, u)) {
            tick = s.tick;
            const float w = std::sqrt(s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1] + s.angVel[2] * s.angVel[2]);
            const float sp = std::sqrt(s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] + s.vel[2] * s.vel[2]);
            tickBox = Inflate(u, (sp + 2.6f * w) * 0.01f * 1.25f + 0.05f);
            QueryInstances(tickBox);
            ++queries;
        }
        ents.clear();
        LaneLookups(c, 0);
        if (g_ahead && s.tick != aheadTick) {  // software pipeline over the next ticks: level k for the pose k ticks ahead
            aheadTick = s.tick;
            for (int lv = 1; lv <= 4; ++lv) {
                alignas(32) float ca[3][8];
                CentresAhead(s, 0.01f * lv, ca);
                LaneLookups(ca, lv);
            }
        }
        // group by shape
        std::uint32_t cnt[kShapes] = {};
        for (const Ent &e : ents) ++cnt[e.sh];
        eb[0] = 0;
        for (int k = 0; k < kShapes; ++k) eb[k + 1] = eb[k] + cnt[k];
        sorted.resize(ents.size());
        std::uint32_t pos[kShapes];
        for (int k = 0; k < kShapes; ++k) pos[k] = eb[k];
        for (const Ent &e : ents) sorted[pos[e.sh]++] = e;
        entsN += ents.size();
    }
    TW_NI void ProcessShape(int sh) override {
        if (eb[sh] == eb[sh + 1]) { merged[sh].clear(); ++clsEmpty; return; }
        SetupOne(sh);
        raw.clear();
        for (std::uint32_t q = eb[sh]; q < eb[sh + 1]; ++q) {
            const Ent &e = sorted[q];
            const Instance &in = bw->instances[e.id];
            const Model &model = bw->models[in.model];
            ShapeMap mm;
            if (in.permutation) ToModelPerm(sm[sh], in, perm[e.id], &mm); else ToModel(sm[sh], in, &mm);
            if (e.fallback) {
                for (std::uint32_t ei = 0; ei < model.elements.size(); ++ei) {
                    const Element &el = model.elements[ei];
                    bool ov = true;
                    for (int a = 0; a < 3; ++a) ov = ov && el.lo[a] <= mm.t[a] + 2.0f && el.hi[a] >= mm.t[a] - 2.0f;
                    if (!ov) continue;
                    Contact c; bool fast;
                    if (ElementTest(mm, model, el, &c, &fast) != kNone) Emit(sh, in, c, el.material);
                }
                continue;
            }
            const VoxList &L = pv->lists[e.list];
            const VoxGrid &g = *pv->listGrid[e.list];
            const std::uint16_t *it = &g.items[L.first];
            TouchList(L, g, model, static_cast<int>(in.model));
            bool ex = false;
            if (L.memberCount) GroupContacts(sh, e.id, in, model, g, L, mm, &ex);
            if (L.memberCount < L.count) {
                float hm[3];
                for (int i = 0; i < 3; ++i) hm[i] = std::fabs(mm.M[i][0]) + std::fabs(mm.M[i][1]) + std::fabs(mm.M[i][2]) + 1e-3f;
                for (std::uint32_t j = L.memberCount; j < L.count; ++j) RestTri(sh, in, model, model.elements[it[j]], mm, hm);
            }
        }
        bool prim = false, feat = false;
        for (const MergedContact &c : raw) { prim |= c.primary; feat |= !c.primary; }
        if (prim && feat) MergeShape(raw, &merged[sh]); else merged[sh].swap(raw);
    }
    bool FastK(int sh, KSet *out) override {
        KSet k{};
        if (eb[sh] == eb[sh + 1]) { *out = k; return true; }
        SetupOne(sh);
        for (std::uint32_t q = eb[sh]; q < eb[sh + 1]; ++q) {
            const Ent &e = sorted[q];
            if (e.fallback) return false;
            const Instance &in = bw->instances[e.id];
            const Model &model = bw->models[in.model];
            ShapeMap mm;
            if (in.permutation) ToModelPerm(sm[sh], in, perm[e.id], &mm); else ToModel(sm[sh], in, &mm);
            const VoxList &L = pv->lists[e.list];
            const VoxGrid &g = *pv->listGrid[e.list];
            const std::uint16_t *it = &g.items[L.first];
            TouchList(L, g, model, static_cast<int>(in.model));
            if (L.memberCount) {
                const float sd = (L.n[0] * (mm.t[0] - L.p[0]) + L.n[1] * (mm.t[1] - L.p[1])) + L.n[2] * (mm.t[2] - L.p[2]);
                if (sd > 0.0f) {
                    float Mn[3];
                    for (int j = 0; j < 3; ++j) Mn[j] = (mm.M[0][j] * L.n[0] + mm.M[1][j] * L.n[1]) + mm.M[2][j] * L.n[2];
                    const float h2 = (Mn[0] * Mn[0] + Mn[1] * Mn[1]) + Mn[2] * Mn[2];
                    if (sd * sd <= h2) {
                        bool face = L.covered != 0;
                        if (!face) {
                            float MMn[3];
                            for (int i = 0; i < 3; ++i) MMn[i] = (mm.M[i][0] * Mn[0] + mm.M[i][1] * Mn[1]) + mm.M[i][2] * Mn[2];
                            const float kk = sd / h2;
                            const float P[3] = {mm.t[0] - kk * MMn[0], mm.t[1] - kk * MMn[1], mm.t[2] - kk * MMn[2]};
                            for (std::uint32_t j = 0; j < L.memberCount && !face; ++j) {
                                const Element &el = model.elements[it[j]];
                                const float *v = &model.loop[3 * el.first], *uu = &model.edgeU[3 * el.first];
                                bool in3 = true;
                                for (int i = 0; i < 3 && in3; ++i)
                                    in3 = (uu[3 * i] * (P[0] - v[3 * i]) + uu[3 * i + 1] * (P[1] - v[3 * i + 1])) + uu[3 * i + 2] * (P[2] - v[3 * i + 2]) <= 0.0f;
                                face = in3;
                            }
                            if (!face) {
                                const float h = std::sqrt(h2);
                                if (DiskReachesEdge(mm, vw->models[in.model].segs.data(), &g.edges[L.edgeFirst], L.edgeCount, -1, sd / h, Mn, h)) return false;
                            }
                        }
                        if (face) KAdd(k, L.material);
                    }
                }
            }
            if (L.memberCount < L.count) {
                float hm[3];
                for (int i = 0; i < 3; ++i) hm[i] = std::fabs(mm.M[i][0]) + std::fabs(mm.M[i][1]) + std::fabs(mm.M[i][2]) + 1e-3f;
                for (std::uint32_t j = L.memberCount; j < L.count; ++j) {
                    const Element &el = model.elements[it[j]];
                    if (el.lo[0] > mm.t[0] + hm[0] || el.hi[0] < mm.t[0] - hm[0] || el.lo[1] > mm.t[1] + hm[1] || el.hi[1] < mm.t[1] - hm[1] ||
                        el.lo[2] > mm.t[2] + hm[2] || el.hi[2] < mm.t[2] - hm[2]) continue;
                    if (!Reject(mm, model, el)) return false;
                }
            }
        }
        *out = k;
        return true;
    }
};
PackedVox *g_pv = nullptr;
bool g_lean = false;

std::unique_ptr<Pipe> MakePipe(const BlockWorld *bw, const VoxWorld *vw, bool allowTile = true) {
    std::unique_ptr<Pipe> p;
    if (allowTile && g_tile) p.reset(new TilePipe());
    else if (allowTile && g_lean) { LeanPipe *l = new LeanPipe(); l->pv = g_pv; p.reset(l); }
    else p.reset(new Pipe());
    p->bw = bw; p->vw = vw; p->Init();
    return p;
}

// Pose extrapolated like the harness probe (docs/twin/substep-policy.md): constant v and w, gravity
// -10 * forceScale (3) on y; car basis rotated about w by |w| t; shapes keep their car-local transforms.
void Extrapolate(const tw::Substep &s, float tau, tw::Substep *o) {
    *o = s;
    const double g = -30.0;
    const float *R = s.carRot;  // columns basisX, basisY, basisZ
    float Rn[9];
    const double wx = s.angVel[0], wy = s.angVel[1], wz = s.angVel[2], w = std::sqrt(wx * wx + wy * wy + wz * wz);
    for (int c = 0; c < 3; ++c) {
        double v[3] = {R[3 * c], R[3 * c + 1], R[3 * c + 2]};
        if (w * tau > 1e-9) {
            const double ux = wx / w, uy = wy / w, uz = wz / w, a = w * tau, cs = std::cos(a), sn = std::sin(a);
            const double dot = ux * v[0] + uy * v[1] + uz * v[2];
            const double cx = uy * v[2] - uz * v[1], cy = uz * v[0] - ux * v[2], cz = ux * v[1] - uy * v[0];
            const double n0 = v[0] * cs + cx * sn + ux * dot * (1 - cs), n1 = v[1] * cs + cy * sn + uy * dot * (1 - cs), n2 = v[2] * cs + cz * sn + uz * dot * (1 - cs);
            v[0] = n0; v[1] = n1; v[2] = n2;
        }
        for (int i = 0; i < 3; ++i) Rn[3 * c + i] = static_cast<float>(v[i]);
    }
    float p[3];
    for (int i = 0; i < 3; ++i) p[i] = static_cast<float>(s.carPos[i] + s.vel[i] * tau + (i == 1 ? 0.5 * g * tau * tau : 0.0));
    for (int i = 0; i < 3; ++i) o->carPos[i] = p[i];
    for (int i = 0; i < 9; ++i) o->carRot[i] = Rn[i];
    for (int k = 0; k < kShapes; ++k) {
        const Shape &sh = s.shapes[k];
        float lt[3], la[3][3];
        for (int j = 0; j < 3; ++j) {
            lt[j] = R[3 * j] * (sh.t[0] - s.carPos[0]) + R[3 * j + 1] * (sh.t[1] - s.carPos[1]) + R[3 * j + 2] * (sh.t[2] - s.carPos[2]);
            for (int a = 0; a < 3; ++a) la[a][j] = R[3 * j] * sh.axis[a][0] + R[3 * j + 1] * sh.axis[a][1] + R[3 * j + 2] * sh.axis[a][2];
        }
        Shape &d = o->shapes[k];
        for (int i = 0; i < 3; ++i) {
            d.t[i] = p[i] + Rn[i] * lt[0] + Rn[3 + i] * lt[1] + Rn[6 + i] * lt[2];
            for (int a = 0; a < 3; ++a) d.axis[a][i] = Rn[i] * la[a][0] + Rn[3 + i] * la[a][1] + Rn[6 + i] * la[a][2];
        }
    }
}

void RunPolicy(const Recording &rec, const BlockWorld &bw, const VoxWorld &vw, double k) {
    using KSet = Pipe::KSet;
    const std::size_t N = rec.substeps.size();
    // 1. K agreement with FV at every recorded substep (predictor path: voxel classes, narrow phase only when unknown)
    {
        auto qp = MakePipe(&bw, &vw); Pipe &q = *qp;
        std::size_t shapeMis = 0, subMis = 0, fvOnly = 0, twOnly = 0;
        KSet dummy[kShapes] = {};
        for (std::size_t i = 0; i < N; ++i) {
            const tw::Substep &s = rec.substeps[i];
            KSet K[kShapes];
            q.ProbeDiffers(s, dummy, false, nullptr);  // warm path counters only
            q.Prepare(s);
            for (int sh = 0; sh < kShapes; ++sh) if (!q.FastK(sh, &K[sh])) { q.ProcessShape(sh); K[sh] = q.KOf(sh); }
            bool any = false;
            for (int sh = 0; sh < kShapes; ++sh) {
                KSet F{};
                for (std::uint32_t c = s.firstMerged; c < s.firstMerged + s.mergedCount; ++c) if (rec.merged[c].shape == sh) Pipe::KAdd(F, rec.merged[c].material);
                if (F != K[sh]) {
                    ++shapeMis; any = true;
                    bool fvExtra = false, twExtra = false;
                    for (int w = 0; w < 4; ++w) { fvExtra |= (F[w] & ~K[sh][w]) != 0; twExtra |= (K[sh][w] & ~F[w]) != 0; }
                    fvOnly += fvExtra; twOnly += twExtra;
                }
            }
            subMis += any;
        }
        std::printf("predictor K vs FV merged K at all %zu substeps: substeps differing %zu (%.3f%%), (shape) sets differing %zu: FV has a material the twin lacks %zu, twin extra %zu\n",
                    N, subMis, 100.0 * subMis / N, shapeMis, fvOnly, twOnly);
    }
    // 2. policy decisions on the harness probe poses (t, t + dt/2, t + dt), fast vs exact (every shape through the narrow phase)
    std::vector<std::size_t> first; std::vector<int> nsub;
    for (std::size_t i = 0; i < N; ++i) {
        if (i == 0 || rec.substeps[i].tick != rec.substeps[i - 1].tick) { first.push_back(i); nsub.push_back(0); }
        ++nsub.back();
    }
    auto fp = MakePipe(&bw, &vw); Pipe &f = *fp;
    auto ep = MakePipe(&bw, &vw, false); Pipe &e = *ep; e.tickMode = false;  // exact reference: own lookups at every pose
    std::size_t probed = 0, evFast = 0, evExact = 0, falseNoChange = 0, falseChange = 0, evConservative = 0, consFalseNo = 0;
    double subsFast = 0, subsExact = 0, subsFV = 0, subsCons = 0;
    std::uint64_t tProbe = 0, tCons = 0;
    for (std::size_t t = 0; t < first.size(); ++t) {
        const tw::Substep &s0 = rec.substeps[first[t]];
        const int n = nsub[t];
        subsFV += n;
        const float speed = std::sqrt(s0.vel[0] * s0.vel[0] + s0.vel[1] * s0.vel[1] + s0.vel[2] * s0.vel[2]);
        auto policy = [&](bool event) { return !event ? 1 : (speed * 3.6f >= 900.0f ? n : std::min(n, 3)); };
        f.Step(s0);  // the tick's first collision pass gives K(t)
        e.Step(s0);
        if (n <= 1) { subsFast += 1; subsExact += 1; subsCons += 1; continue; }
        ++probed;
        KSet K0[kShapes], K0e[kShapes];
        for (int sh = 0; sh < kShapes; ++sh) { K0[sh] = f.KOf(sh); K0e[sh] = e.KOf(sh); }
        tw::Substep pose[2];
        const std::uint64_t r0 = __rdtsc();
        Extrapolate(s0, 0.005f, &pose[0]); Extrapolate(s0, 0.01f, &pose[1]);
        const bool evf = f.ProbeDiffers(pose[0], K0, false, nullptr) || f.ProbeDiffers(pose[1], K0, false, nullptr);
        const std::uint64_t r1 = __rdtsc();
        tProbe += r1 - r0;
        // voxel-only variant: unknown shapes count as a change (no narrow phase at all)
        bool evc = false;
        {
            const std::uint64_t c0 = __rdtsc();
            for (int pz = 0; pz < 2 && !evc; ++pz) {
                Pipe &g = f;
                g.Prepare(pose[pz]);
                for (int sh = 0; sh < kShapes && !evc; ++sh) { KSet kk; if (!g.FastK(sh, &kk) || kk != K0[sh]) evc = true; }
            }
            tCons += __rdtsc() - c0;
        }
        const bool eve = e.ProbeDiffers(pose[0], K0e, true, nullptr) | e.ProbeDiffers(pose[1], K0e, true, nullptr);
        evFast += evf; evExact += eve; evConservative += evc;
        falseNoChange += !evf && eve; falseChange += evf && !eve; consFalseNo += !evc && eve;
        subsFast += policy(evf); subsExact += policy(eve); subsCons += policy(evc);
    }
    const double T = static_cast<double>(first.size());
    std::printf("policy (contact1d0e3w900) on %zu ticks, %zu probed (FV n > 1): events fast %.1f%% / exact %.1f%% / voxel-only %.1f%% of probed ticks; "
                "false no-change: fast %zu, voxel-only %zu; false change (fast) %zu\n",
                first.size(), probed, 100.0 * evFast / std::max<std::size_t>(1, probed), 100.0 * evExact / std::max<std::size_t>(1, probed),
                100.0 * evConservative / std::max<std::size_t>(1, probed), falseNoChange, consFalseNo, falseChange);
    std::printf("substeps per tick: FV %.2f, policy with fast predictor %.2f, exact %.2f, voxel-only %.2f; predictor cost per probed tick: fast %.0f cyc (%.0f per tick overall), voxel-only %.0f cyc; "
                "shapes resolved from voxel data %.1f%% (rest through the narrow phase)\n",
                subsFV / T, subsFast / T, subsExact / T, subsCons / T, double(tProbe) * k / std::max<std::size_t>(1, probed), double(tProbe) * k / T,
                double(tCons) * k / std::max<std::size_t>(1, probed), 100.0 * f.fastShapes / std::max<std::size_t>(1, f.fastShapes + f.slowShapes));
    std::printf("fast predictor split per probed tick: lookups (setup + stages A-D) %.0f, voxel-class resolution %.0f, narrow phase on unknown shapes %.0f cyc\n",
                double(f.tpPrep) * k / std::max<std::size_t>(1, probed), double(f.tpFast) * k / std::max<std::size_t>(1, probed), double(f.tpSlow) * k / std::max<std::size_t>(1, probed));
}

// End-to-end policy run: per tick the collision pass at the tick start, the two probes (unless FV's n is 1),
// then the policy's remaining substeps at FV poses spread over the tick. Wall time = all collision + predictor.
void PolicyRun(const Recording &rec, const BlockWorld &bw, const VoxWorld &vw, bool fastPred, bool n0 = false) {
    using KSet = Pipe::KSet;
    const std::size_t N = rec.substeps.size();
    std::vector<std::size_t> first; std::vector<int> nsub;
    for (std::size_t i = 0; i < N; ++i) {
        if (i == 0 || rec.substeps[i].tick != rec.substeps[i - 1].tick) { first.push_back(i); nsub.push_back(0); }
        ++nsub.back();
    }
    auto fp = MakePipe(&bw, &vw); Pipe &f = *fp;
    std::size_t subs = 0, events = 0, touchEvents = 0;
    auto w0 = Clock::now();
    for (std::size_t t = 0; t < first.size(); ++t) {
        const tw::Substep &s0 = rec.substeps[first[t]];
        const int n = nsub[t];
        f.Step(s0); ++subs;
        int kk = 1;
        if (n > 1) {
            KSet K0[kShapes];
            for (int sh = 0; sh < kShapes; ++sh) K0[sh] = f.KOf(sh);
            bool ev = false;
            if (n0)  // contact1d0e3w900n0: any tick-start contact whose point moves along its normal is an event
                for (int sh = 0; sh < kShapes && !ev; ++sh)
                    for (const MergedContact &c : f.merged[sh]) {
                        const float r[3] = {c.c.P[0] - s0.carPos[0], c.c.P[1] - s0.carPos[1], c.c.P[2] - s0.carPos[2]};
                        const float vp[3] = {s0.vel[0] + s0.angVel[1] * r[2] - s0.angVel[2] * r[1], s0.vel[1] + s0.angVel[2] * r[0] - s0.angVel[0] * r[2],
                                             s0.vel[2] + s0.angVel[0] * r[1] - s0.angVel[1] * r[0]};
                        if (std::fabs(vp[0] * c.c.N[0] + vp[1] * c.c.N[1] + vp[2] * c.c.N[2]) > 0.0f) { ev = true; ++touchEvents; break; }
                    }
            tw::Substep pose[2];
            Extrapolate(s0, 0.005f, &pose[0]); Extrapolate(s0, 0.01f, &pose[1]);
            if (ev) {}
            else if (fastPred) ev = f.ProbeDiffers(pose[0], K0, false, nullptr) || f.ProbeDiffers(pose[1], K0, false, nullptr);
            else
                for (int pz = 0; pz < 2 && !ev; ++pz) {
                    f.Prepare(pose[pz]);
                    for (int sh = 0; sh < kShapes && !ev; ++sh) { KSet kq; if (!f.FastK(sh, &kq) || kq != K0[sh]) ev = true; }
                }
            const float speed = std::sqrt(s0.vel[0] * s0.vel[0] + s0.vel[1] * s0.vel[1] + s0.vel[2] * s0.vel[2]);
            if (ev) { ++events; kk = speed * 3.6f >= 900.0f ? n : std::min(n, 3); }
        }
        for (int j = 1; j < kk; ++j) { f.Step(rec.substeps[first[t] + static_cast<std::size_t>(j * n / kk)]); ++subs; }
    }
    const double cyc = std::chrono::duration<double, std::nano>(Clock::now() - w0).count() * g_ghz;
    const double T = static_cast<double>(first.size());
    std::printf("POLICY RUN%s (%s predictor): %.0f cyc/tick all collision + predictor, %.2f substeps/tick, events %.1f%% of ticks (touching with normal motion %.1f%%), lookup passes %.2f/tick "
                "(re-lookups %.3f/tick), pair reuse %.2f/tick\n",
                n0 ? " n0" : "", fastPred ? "fast" : "voxel-only", cyc / T, subs / T, 100.0 * events / T, 100.0 * touchEvents / T, double(f.lookupPasses) / T, double(f.relookups) / T, double(f.pairReuse) / T);
    if (g_ws.on && g_ws.ticks) {
        const double T2 = g_ws.ticks;
        std::printf("  hot working set (map data only): %.1f KB touched per tick (voxel index %.2f, list headers %.2f, items %.2f, triangles %.2f, plane edges %.2f KB), "
                    "%.1f KB new per tick (not touched in the previous 64 ticks), %.0f KB live over 64 ticks\n",
                    g_ws.lines * 64 / 1024 / T2, g_ws.byKind[0] * 64 / 1024 / T2, g_ws.byKind[1] * 64 / 1024 / T2, g_ws.byKind[2] * 64 / 1024 / T2, g_ws.byKind[3] * 64 / 1024 / T2,
                    g_ws.byKind[4] * 64 / 1024 / T2, g_ws.newLines * 64 / 1024 / T2, g_ws.footprint64 * 64 / 1024 / T2);
        g_ws = WorkingSet{}; g_ws.on = true;
    }
    if (const TilePipe *tp = dynamic_cast<const TilePipe *>(&f))
        std::printf("  tiles: anchors %.2f/tick (re-anchors %.3f/tick), candidates %.1f per anchor, active %.2f per pose, shapes skipped %.1f%% of shape-poses\n",
                    double(tp->anchorsN) / T, double(tp->reanchorsN) / T, double(tp->candN) / std::max<std::size_t>(1, tp->anchorsN),
                    double(tp->activeN) / std::max<std::size_t>(1, tp->posesN), 100.0 * tp->skippedShapes / std::max<std::size_t>(1, tp->posesN * kShapes));
}


double Pct(std::vector<float> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))];
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: tw_voxbench REC...\n"); return 2; }
    const float vox = std::getenv("TW_VOX") ? static_cast<float>(std::atof(std::getenv("TW_VOX"))) : 0.5f;
    for (int a = 1; a < argc; ++a) {
        Recording rec;
        if (!LoadRecording(argv[a], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a]); return 1; }
        const std::size_t N = rec.substeps.size();
        const double spt = static_cast<double>(N) / rec.ticks;
        BlockWorld bw; bw.Build(rec.source, false, 1e-4f, 1e-4f);
        g_tile = std::getenv("TW_TILE") != nullptr;
        g_lookupOnly = std::getenv("TW_LOOKUPONLY") != nullptr;
        g_ws.on = std::getenv("TW_WSET") != nullptr;
        g_ahead = std::getenv("TW_AHEAD") != nullptr;
        if (std::getenv("TW_DELTA")) g_delta = static_cast<float>(std::atof(std::getenv("TW_DELTA")));
        const double maxVox = std::getenv("TW_MAXVOX") ? std::atof(std::getenv("TW_MAXVOX")) : 16e6;
        if (std::getenv("TW_SHAPEMASK")) g_shapeMask = static_cast<unsigned>(std::strtoul(std::getenv("TW_SHAPEMASK"), nullptr, 0));
        usePre = !std::getenv("TW_NOPRE"); split = std::getenv("TW_SPLIT") != nullptr;
        VoxWorld vw;
        vw.useGroups = !std::getenv("TW_NOGROUPS");
        vw.margin = g_delta;
        if (std::getenv("TW_HUGEVOX")) vw.hugeMaxVoxels = std::atof(std::getenv("TW_HUGEVOX"));
        if (std::getenv("TW_CLASSES") && std::atoi(std::getenv("TW_CLASSES")) == 3) {  // shapes 2,3 share 1.039 m; shapes 0,1 share 1.902 m
            vw.buildClass[1] = vw.buildClass[3] = false; g_classMap[1] = 2; g_classMap[3] = 4;
        } vw.frontCull = !std::getenv("TW_NOFRONT"); vw.distCull = !std::getenv("TW_NODIST");
        vw.Build(bw, vox, maxVox);
        g_lean = std::getenv("TW_LEAN") != nullptr;
        PackedVox pvox;
        if (g_lean) { pvox.Build(vw); g_pv = &pvox; std::printf("packed voxel arrays: %.1f MB\n", pvox.Bytes() / 1e6); }
        std::size_t maxElems = 0; for (const Model &m : bw.models) maxElems = std::max(maxElems, m.elements.size());
        if (std::getenv("TW_RAWDUMP")) {
            const std::size_t i = std::strtoul(std::getenv("TW_RAWDUMP"), nullptr, 10);
            const tw::Substep &s = rec.substeps[i];
            std::printf("raw FV contacts at substep %zu (tick %u):\n", i, s.tick);
            for (std::uint32_t c = s.firstContact; c < s.firstContact + s.contactCount; ++c) {
                const FvContact &f = rec.contacts[c];
                std::printf("  shape %u %s m%u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f)\n", f.shape, f.primary ? "FACE" : "feat", f.material, f.P[0], f.P[1], f.P[2], f.N[0], f.N[1], f.N[2]);
            }
            for (std::size_t si = 0; si < rec.source.surfaces.size(); ++si) {
                const SourceSurface &ss = rec.source.surfaces[si];
                bool m9 = false;
                for (std::uint32_t t = ss.firstTri; t < ss.firstTri + ss.triCount; ++t) m9 |= rec.source.tris[t].material == 9;
                if (m9 && std::fabs(ss.iso[9] - 496.0f) < 40.0f && std::fabs(ss.iso[11] - 496.0f) < 40.0f)
                    std::printf("  surface %zu with material 9 near: iso T (%.2f %.2f %.2f), tris %u, actor %u, trigger %u\n", si, ss.iso[9], ss.iso[10], ss.iso[11], ss.triCount, ss.actor, ss.trigger);
            }
        }
        if (std::getenv("TW_VOXSTAT")) {
            std::vector<std::pair<std::size_t, std::size_t>> big;
            for (std::size_t mi = 0; mi < bw.models.size(); ++mi) {
                std::size_t v = 0; for (const VoxGrid &g : vw.models[mi].grid) v += std::size_t(g.dim[0]) * g.dim[1] * g.dim[2];
                big.push_back({v, mi});
            }
            std::sort(big.rbegin(), big.rend());
            std::vector<int> uses(bw.models.size(), 0); for (const Instance &in : bw.instances) ++uses[in.model];
            for (int q = 0; q < 12 && q < (int)big.size(); ++q) {
                const Model &m = bw.models[big[q].second]; const VoxGrid &g = vw.models[big[q].second].grid[VoxModel::kClasses - 1];
                std::printf("  model %zu: %zu voxels, %zu tris, %d instances, box (%.1f %.1f %.1f)-(%.1f %.1f %.1f), class2 dim %dx%dx%d size %.2f, lists %zu, bricks %zu/%zu\n", big[q].second, big[q].first,
                            m.elements.size(), uses[big[q].second], m.box.lo[0], m.box.lo[1], m.box.lo[2], m.box.hi[0], m.box.hi[1], m.box.hi[2], g.dim[0], g.dim[1], g.dim[2], g.size, g.lists.size(), g.nonEmptyBricks, g.brick.size());
            }
        }
        std::printf("## %s (%zu substeps, %.2f/tick), voxel %.2f m, tick margin %.2f m\n", rec.name.c_str(), N, spt, vox, g_delta);
        std::printf("voxel world: %zu models (max %zu tris/model, %zu grids without voxels, %zu coarsened for u16 list ids), %zu instances; %.1fM dense voxels, %zu non-empty bricks, %zu unique, %zu distinct lists, %zu list items, %zu edges; "
                    "voxel data %.1f MB (+ model triangles %.1f MB); build %.0f ms\n",
                    bw.models.size(), maxElems, vw.overflowGrids, vw.coarsenedGrids, bw.instances.size(), vw.voxels / 1e6, vw.bricks, vw.uniqueBricks, vw.lists, vw.items, vw.edgesTotal, vw.Bytes() / 1e6,
                    [&] { std::size_t b = 0; for (const Model &m : bw.models) b += m.loop.size() * 4 + m.elements.size() * sizeof(Element); return b / 1e6; }(),
                    vw.buildMs);
        // timed pass
        auto pp0 = MakePipe(&bw, &vw); Pipe &p = *pp0;
        if (std::getenv("TW_PASSES")) { auto ppp = MakePipe(&bw, &vw); for (int r = 0; r < std::atoi(std::getenv("TW_PASSES")); ++r) for (std::size_t i = 0; i < N; ++i) ppp->Step(rec.substeps[i]); }
        const double ghz0 = CoreGHz();
        auto w0 = Clock::now(); const std::uint64_t r0 = __rdtsc();
        for (std::size_t i = 0; i < N; ++i) p.Step(rec.substeps[i]);
        const std::uint64_t r1 = __rdtsc(); auto w1 = Clock::now();
        g_ghz = 0.5 * (ghz0 + CoreGHz());
        std::printf("core clock %.2f GHz (measured; cycles below = wall time x this clock)\n", g_ghz);
        const double wall = std::chrono::duration<double, std::nano>(w1 - w0).count() * g_ghz, k = wall / double(r1 - r0);
        const double shapeSubs = double(N) * kShapes;
        std::printf("cost: %.0f cyc/substep = %.0f cyc/tick (setup %.0f, instance query %.0f, shapes %.0f of which merge %.0f per substep) | "
                    "queries %.3f/tick, re-queries %.4f/substep, near instances %.1f, voxel lookups %.2f/substep, list items %.2f/substep, box pass %.2f, exact tests %.2f/substep (%.2f with a contact)\n",
                    wall / N, wall / N * spt, p.tSetup * k / N, p.tQuery * k / N, p.tShapes * k / N, p.tMerge * k / N, double(p.queries) / rec.ticks,
                    double(p.requeries) / N, double(p.nearSum) / double(std::max<std::size_t>(1, p.queries + p.requeries)), double(p.lookups) / N, double(p.listItems) / N, double(p.boxPass) / N, double(p.exactTests) / N, double(p.exactHits) / N);
        std::printf("lookup passes %.2f/tick (re-lookups %.3f/tick, pair reuse %.2f/tick)\n", double(p.lookupPasses) / rec.ticks, double(p.relookups) / rec.ticks, double(p.pairReuse) / rec.ticks);
        if (const TilePipe *tp = dynamic_cast<const TilePipe *>(&p))
            std::printf("tiles: anchors %.2f/tick (re-anchors %.3f/tick), candidates %.1f per anchor, active %.2f per pose, shapes skipped %.1f%% of shape-poses\n",
                        double(tp->anchorsN) / rec.ticks, double(tp->reanchorsN) / rec.ticks, double(tp->candN) / std::max<std::size_t>(1, tp->anchorsN),
                        double(tp->activeN) / std::max<std::size_t>(1, tp->posesN), 100.0 * tp->skippedShapes / std::max<std::size_t>(1, tp->posesN * kShapes));
        std::printf("per shape 0..7: plane hit reaching an edge / group needing exact tests, %% of the shape's substeps:");
        for (int k = 0; k < kShapes; ++k) std::printf(" %.2f/%.2f", 100.0 * p.deepEdge[k] / N, 100.0 * p.groupExact[k] / N);
        std::printf("\nper shape 0..7: rest items / exact tests per substep:");
        for (int k = 0; k < kShapes; ++k) std::printf(" %.2f/%.2f", double(p.itemsByShape[k]) / N, double(p.exactByShape[k]) / N);
        std::printf("\n");
        const double lk = double(std::max<std::size_t>(1, p.lookups));
        std::printf("voxel class per lookup: outside/EMPTY %.1f%%, SINGLE_PLANE (covered) %.1f%%, PLANE (not covered) %.1f%%, PLANE+rest %.1f%%, GENERAL %.1f%%; plane hits %.2f/substep\n",
                    100.0 * (p.lookups - p.clsSeen[1] - p.clsSeen[2] - p.clsSeen[3] - p.clsSeen[4]) / lk, 100.0 * p.clsSeen[1] / lk, 100.0 * p.clsSeen[2] / lk, 100.0 * p.clsSeen[3] / lk,
                    100.0 * p.clsSeen[4] / lk, double(p.planeHits) / N);
        std::printf("shape-substeps: EMPTY %.1f%%, closed form only %.1f%%, closed form + rest scan %.1f%%, needs exact tests on a group %.1f%%; closed-form FACEs %.2f/substep; fallback-model lookups %zu\n",
                    100.0 * p.clsEmpty / shapeSubs, 100.0 * p.clsSingle / shapeSubs, 100.0 * p.clsRest / shapeSubs, 100.0 * p.clsGeneral / shapeSubs, double(p.closedForms) / N, p.fallbackModels);
        {
            std::size_t byCls[VoxModel::kClasses] = {}, huge = 0;
            for (std::size_t mi = 0; mi < bw.models.size(); ++mi) {
                const Model &m = bw.models[mi];
                const float ext = std::fmax(m.box.hi[0] - m.box.lo[0], std::fmax(m.box.hi[1] - m.box.lo[1], m.box.hi[2] - m.box.lo[2]));
                for (int c = 0; c < VoxModel::kClasses; ++c) {
                    const VoxGrid &g = vw.models[mi].grid[c];
                    const std::size_t b = g.brick.size() * 4 + g.cells.size() * 2 + g.lists.size() * sizeof(VoxList) + g.items.size() * 2 + g.edges.size() * sizeof(VoxEdge);
                    byCls[c] += b; if (ext > 128.0f) huge += b;
                }
            }
            std::printf("memory by class (MB):");
            for (int c = 0; c < VoxModel::kClasses; ++c) std::printf(" %.1f", byCls[c] / 1e6);
            std::printf("; models larger than 128 m: %.1f MB\n", huge / 1e6);
        }
        if (std::getenv("TW_WARM")) {  // same substep twice, time the second call: cost without cache misses
            auto qp = MakePipe(&bw, &vw); Pipe &q = *qp;
            std::uint64_t warm = 0, warmNoQ = 0;
            for (std::size_t i = 0; i < N; ++i) {
                q.Step(rec.substeps[i]);
                const std::uint64_t w0t = __rdtsc(); q.Step(rec.substeps[i]); warmNoQ += __rdtsc() - w0t;
                q.tick = 0xffffffffu;
                const std::uint64_t w1t = __rdtsc(); q.Step(rec.substeps[i]); warm += __rdtsc() - w1t;
            }
            std::printf("warm (same substep again, all data in cache): %.0f cyc/substep without the instance query, %.0f with it\n", double(warmNoQ) * k / N, double(warm) * k / N);
        }
        // comparison pass vs FV merged
        auto cp = MakePipe(&bw, &vw); Pipe &c = *cp;
        if (std::getenv("TW_DBG")) { g_dbgSub = std::strtoul(std::getenv("TW_DBG"), nullptr, 10); g_dbgShape = std::atoi(std::getenv("TW_DBGSHAPE")); }
        std::size_t subsWith = 0, countEq = 0, fvTot = 0, twTot = 0, matched = 0, matEq = 0, primEq = 0, exc = 0, fvExtraFeat = 0, twExtra = 0, other = 0, valueOnly = 0;
        std::vector<float> dP, dA, dS;
        int printed = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const tw::Substep &s = rec.substeps[i];
            curSub = i;
            c.Step(s);
            for (int sh = 0; sh < kShapes; ++sh) {
                std::vector<const FvContact *> F;
                for (std::uint32_t q = s.firstMerged; q < s.firstMerged + s.mergedCount; ++q) if (rec.merged[q].shape == sh) F.push_back(&rec.merged[q]);
                const auto &W = c.merged[sh];
                if (F.empty() && W.empty()) continue;
                ++subsWith; fvTot += F.size(); twTot += W.size(); countEq += F.size() == W.size();
                std::vector<bool> used(W.size(), false);
                bool valueExc = false;
                for (const FvContact *f : F) {
                    int best = -1; float bc = 1e30f;
                    for (std::size_t q = 0; q < W.size(); ++q) {
                        if (used[q]) continue;
                        const float dx = W[q].c.P[0] - f->P[0], dy = W[q].c.P[1] - f->P[1], dz = W[q].c.P[2] - f->P[2];
                        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        const float dn = 1.0f - (W[q].c.N[0] * f->N[0] + W[q].c.N[1] * f->N[1] + W[q].c.N[2] * f->N[2]);
                        const float ds0 = std::fabs(std::sqrt(W[q].c.Sep[0] * W[q].c.Sep[0] + W[q].c.Sep[1] * W[q].c.Sep[1] + W[q].c.Sep[2] * W[q].c.Sep[2]) -
                                                    std::sqrt(f->Sep[0] * f->Sep[0] + f->Sep[1] * f->Sep[1] + f->Sep[2] * f->Sep[2]));
                        const float cost = d + dn + ds0 + (W[q].material != f->material ? 0.01f : 0.0f) + ((W[q].primary ? 1 : 0) != f->primary ? 0.01f : 0.0f);
                        if (d < 0.05f && cost < bc) { bc = cost; best = static_cast<int>(q); }
                    }
                    if (best < 0) continue;
                    used[best] = true; ++matched;
                    const MergedContact &w = W[best];
                    const float dx = w.c.P[0] - f->P[0], dy = w.c.P[1] - f->P[1], dz = w.c.P[2] - f->P[2];
                    const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    const float dot = std::fmax(-1.0f, std::fmin(1.0f, w.c.N[0] * f->N[0] + w.c.N[1] * f->N[1] + w.c.N[2] * f->N[2]));
                    const float ang = std::acos(dot) * 57.29578f;
                    const float ds = std::sqrt((w.c.Sep[0] - f->Sep[0]) * (w.c.Sep[0] - f->Sep[0]) + (w.c.Sep[1] - f->Sep[1]) * (w.c.Sep[1] - f->Sep[1]) +
                                               (w.c.Sep[2] - f->Sep[2]) * (w.c.Sep[2] - f->Sep[2]));
                    dP.push_back(d); dA.push_back(ang); dS.push_back(ds);
                    matEq += w.material == f->material; primEq += (w.primary ? 1 : 0) == f->primary;
                    if (d > 1e-3f || ang > 0.5f || ds > 1e-4f || w.material != f->material || (w.primary ? 1 : 0) != f->primary) valueExc = true;
                }
                if (F.size() != W.size() || valueExc) {
                    ++exc;
                    if (F.size() > W.size()) ++fvExtraFeat; else if (W.size() > F.size()) ++twExtra; else if (valueExc) ++valueOnly; else ++other;
                    if (std::getenv("TW_EXC") && printed < 12 && (!std::getenv("TW_EXCTWIN") || W.size() > F.size())) {
                        ++printed;
                        std::printf("  exception tick %u substep %zu shape %d: FV %zu twin %zu\n", s.tick, i, sh, F.size(), W.size());
                        for (const FvContact *f : F) std::printf("    FV   %s m%u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f) |Sep| %.5f\n", f->primary ? "FACE" : "feat", f->material, f->P[0], f->P[1], f->P[2], f->N[0], f->N[1], f->N[2],
                                                                std::sqrt(f->Sep[0] * f->Sep[0] + f->Sep[1] * f->Sep[1] + f->Sep[2] * f->Sep[2]));
                        for (const auto &w : W) std::printf("    twin %s m%u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f) |Sep| %.5f\n", w.primary ? "FACE" : "feat", w.material, w.c.P[0], w.c.P[1], w.c.P[2], w.c.N[0], w.c.N[1], w.c.N[2],
                                                            std::sqrt(w.c.Sep[0] * w.c.Sep[0] + w.c.Sep[1] * w.c.Sep[1] + w.c.Sep[2] * w.c.Sep[2]));
                    }
                }
            }
        }
        std::printf("vs FV merged: (substep,shape) with contacts %zu, count equal %.3f%%; contacts FV %zu twin %zu matched %zu; material eq %.3f%%, FACE/feature eq %.3f%%\n",
                    subsWith, 100.0 * countEq / std::max<std::size_t>(1, subsWith), fvTot, twTot, matched, 100.0 * matEq / std::max<std::size_t>(1, matched),
                    100.0 * primEq / std::max<std::size_t>(1, matched));
        std::printf("errors |dP| p50 %.2e p99 %.2e max %.2e m; normal p50 %.2e p99 %.2e max %.2e deg; |dSep| p50 %.2e p99 %.2e max %.2e m\n", Pct(dP, 0.5), Pct(dP, 0.99),
                    Pct(dP, 1.0), Pct(dA, 0.5), Pct(dA, 0.99), Pct(dA, 1.0), Pct(dS, 0.5), Pct(dS, 0.99), Pct(dS, 1.0));
        std::printf("exceptions %zu (%.3f%%): FV extra %zu, twin extra %zu, value-only %zu\n", exc, 100.0 * exc / std::max<std::size_t>(1, subsWith), fvExtraFeat, twExtra, valueOnly);
        std::fflush(stdout);
        if (!std::getenv("TW_NOPOLICY")) {
            RunPolicy(rec, bw, vw, k);
            PolicyRun(rec, bw, vw, false);
            PolicyRun(rec, bw, vw, true);
            PolicyRun(rec, bw, vw, false, true);
            PolicyRun(rec, bw, vw, true, true);
        }
    }
    return 0;
}
