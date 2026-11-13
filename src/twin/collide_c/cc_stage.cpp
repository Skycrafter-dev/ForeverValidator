// Architect C: rival exact collision stage. See cc_stage.h.
#include "cc_stage.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <x86intrin.h>

#include "cc_lanes.h"
#include "tw_math.h"

#ifndef CC_STEADYSTATS
#define CC_STEADYSTATS 0   // bit 1: steady-contact counters, bit 2: DetectLanes16 TSC phases; printed at exit, no output change
#endif
#if CC_STEADYSTATS
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
#endif
#ifndef CC_GRIDSTATS
#define CC_GRIDSTATS 0   // bit 1: static world grid counters per lane-call, bit 2: filter TSC by shape group, bit 4: grid memory per world (cc_gridstats.inc);
                         // printed at exit, no output change
#endif
#ifndef CC_OSLABSTATS
#define CC_OSLABSTATS 0   // bit 1: oriented slab counters per lane-call, bit 2: filter TSC with the slab-skippable entries removed (cc_oslabstats.inc)
#endif
#if CC_OSLABSTATS
#include <array>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
#endif
#if CC_GRIDSTATS
#include <array>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#endif

namespace tw::collide_c {

namespace {

using exact::Raw;
using lanes::B4;
using lanes::F4;
using FrameF = lanes::FrameFv<float>;
static_assert(sizeof(FrameF) == 36 * sizeof(float), "FrameFv<float>: 36 floats");

constexpr float kNormalAlignment = 0.8660254f;   // PhysicsTolerance::SphereContactNormalAlignment

// geometry_helpers.cpp NearlyEqual(value, reference)
inline bool NearlyEqual(float value, float reference) {
    const float tol = std::fabs(reference) * 1.0e-5f;
    return reference - tol <= value && value <= reference + tol;
}
inline bool NearlyEqual(const exact::V &a, const exact::V &b) { return NearlyEqual(a.x, b.x) && NearlyEqual(a.y, b.y) && NearlyEqual(a.z, b.z); }

struct Job {   // one (shape, instance) FV frame
    std::uint32_t shape, instance;
    std::uint32_t entry = 0;   // a memo entry of that instance (noise path: its iso)
};

enum : std::uint8_t { kRej = 0, kFaceC = 1, kFvViol = 2, kFvAny = 3 };
struct SurvOut {        // pass A's result for one survivor
    float x[21];        // unit-space vertices (9), FV's n (3), FACE payload P N Sep (9)
    std::uint8_t cls;
};

// The noise path's raw contacts of one shape: a growable buffer with an inline append (the
// out-of-line std::vector::push_back was ~5% of search time).
struct RawList {
    std::vector<Raw> buf;
    std::size_t n = 0;
    void clear() { n = 0; }
    bool empty() const { return n == 0u; }
    std::size_t size() const { return n; }
    const Raw *data() const { return buf.data(); }
    const Raw &operator[](std::size_t i) const { return buf[i]; }
    Raw *Next() {   // the slot of the next contact (filled by the caller)
        if (n == buf.size()) Grow();
        return &buf[n++];
    }
    __attribute__((noinline)) void Grow() { buf.resize(2u * buf.size() + 16u); }
};

struct Scratch {
    std::vector<SurvOut> so;
    std::vector<lanes::I3<float>> m2u;   // noise path: meshToUnit per (shape, instance) job
    std::vector<lanes::I3<float>> back, miso;   // ... the way back for the point (FV's c2w factors) and the instance iso
    std::vector<Job> jobs;
    std::vector<FrameF> frames;
    std::vector<std::uint32_t> pe, pf;   // the call's survivors, shape-major: memo entry, frame job
    std::vector<std::uint8_t> pk;        // ... shape
    std::vector<Raw> raw[8];             // per shape raw contacts (FV order)
    RawList rawn[8];                     // ... on the noise path
    std::vector<int> merged;
};

CC_LI F4 G4(const float *a, const float *b, const float *c, const float *d, int f) { return F4{_mm_setr_ps(a[f], b[f], c[f], d[f])}; }

// 4 consecutive fields [f, f+4) of 4 lane records -> 4 field vectors (transpose in registers)
CC_LI void G4x4(const float *a, const float *b, const float *c, const float *d, int f, F4 out[4]) {
    __m128 r0 = _mm_loadu_ps(a + f), r1 = _mm_loadu_ps(b + f), r2 = _mm_loadu_ps(c + f), r3 = _mm_loadu_ps(d + f);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    out[0].v = r0; out[1].v = r1; out[2].v = r2; out[3].v = r3;
}
// 12 floats (an I3) of 4 lane records at float offset f
CC_LI lanes::I3<F4> GatherIso(const float *a, const float *b, const float *c, const float *d, int f) {
    F4 x[12];
    G4x4(a, b, c, d, f, x);
    G4x4(a, b, c, d, f + 4, x + 4);
    G4x4(a, b, c, d, f + 8, x + 8);
    lanes::I3<F4> r;
    r.r.c[0] = {x[0], x[1], x[2]};
    r.r.c[1] = {x[3], x[4], x[5]};
    r.r.c[2] = {x[6], x[7], x[8]};
    r.t = {x[9], x[10], x[11]};
    return r;
}
CC_LI void ScatterIso(const lanes::I3<F4> &v, float *dst[4], int f) {
    const F4 x[12] = {v.r.c[0].x, v.r.c[0].y, v.r.c[0].z, v.r.c[1].x, v.r.c[1].y, v.r.c[1].z, v.r.c[2].x, v.r.c[2].y, v.r.c[2].z, v.t.x, v.t.y, v.t.z};
    for (int g = 0; g < 3; ++g) {
        __m128 r0 = x[4 * g].v, r1 = x[4 * g + 1].v, r2 = x[4 * g + 2].v, r3 = x[4 * g + 3].v;
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
        _mm_storeu_ps(dst[0] + f + 4 * g, r0);
        _mm_storeu_ps(dst[1] + f + 4 * g, r1);
        _mm_storeu_ps(dst[2] + f + 4 * g, r2);
        _mm_storeu_ps(dst[3] + f + 4 * g, r3);
    }
}

inline lanes::I3<float> IsoOf(const float *r9, const float *t3) {
    return lanes::I3<float>{lanes::M3<float>{{{r9[0], r9[1], r9[2]}, {r9[3], r9[4], r9[5]}, {r9[6], r9[7], r9[8]}}}, {t3[0], t3[1], t3[2]}};
}

inline exact::I ToExact(const lanes::I3<float> &a) {
    exact::I r;
    std::memcpy(&r, &a, sizeof r);
    return r;
}

// ---- noise-level narrow phase (StageOptions::noise) -------------------------------------------
// FV's unit space exactly: meshToUnit per (shape, instance) (FV's operations, 4 per SSE call) and
// FV's unit-space vertices, so every narrow-phase decision and the unit-space payload are FV's bits
// (noise::UnitContact4, all survivors in lanes, no scalar FV form). Only the way back to world
// differs: x = c + R D u, Sep = R D s, n_w = normalize(R D^-1 n_u) per shape, instead of FV's
// composed c2w / n2w per (shape, instance) (float noise: the merge's 30-degree test and the payload
// are within noise of FV's). Nothing depends on the cache.
struct alignas(16) ShapeMapF { float R[9]; float c[3]; float r[3]; float ir[3]; float pad[2]; };   // 20 floats

// what the noise-level narrow phase reads per survivor (memo entry e): its instance (job key), the
// instance's iso, its mesh-local vertices (FV order; base + offset, 9 floats) and the Raw::tri to store
struct NoiseCounts { std::uint64_t frames = 0, survivors = 0, face = 0, fvForm = 0, reject = 0; };
struct StageSrc {   // C's stage: C's memo, the pack's tables
    const world::WorldPackView &v;
    const Memo &m;
    const exact::TriVerts *verts;
    std::uint32_t Inst(std::uint32_t e) const { return m.ref[e].instance; }
    const float *Iso(std::uint32_t e) const { return v.instances[m.ref[e].instance].iso; }
    const float *VertBase() const { return reinterpret_cast<const float *>(verts); }
    std::uint32_t VertOffset(std::uint32_t e) const { return m.ref[e].modelTri * 9u; }
    std::uint32_t Tri(std::uint32_t e) const { return m.ref[e].local; }
};
struct PreSrc {     // B's stage (tw_collide.h PreNarrow): the memo's per-entry copies; Raw::tri = memo entry
    const cq::Memo &m;
    std::uint32_t Inst(std::uint32_t e) const { return m.ref[e].instance; }
    const float *Iso(std::uint32_t e) const { return m.inst[m.ref[e].local].iso; }
    const float *VertBase() const { return reinterpret_cast<const float *>(m.verts.data()); }
    std::uint32_t VertOffset(std::uint32_t e) const { return e * 9u; }
    std::uint32_t Tri(std::uint32_t e) const { return e; }
};

// survivors from which the 16-lane narrow phase runs instead of the 4-lane one (the same bits per lane;
// 5 or 3 cost 4-8% more on search candidates, docs/twin/search-throughput.md)
constexpr std::uint32_t kNoise16MinSurvivors = 10u;

template <class Src>
void NoiseNarrowT(const Src &src, const collide::ShapePose shapes[8], const cq::Lists &lists, Scratch &sc, NoiseCounts *stats) {
    ShapeMapF sm[8];
    for (int k = 0; k < 8; ++k) {
        const collide::ShapePose &sp = shapes[k];
        std::memcpy(sm[k].R, sp.world.r, sizeof sm[k].R);
        for (int a = 0; a < 3; ++a) {
            sm[k].c[a] = sp.world.t[a];
            sm[k].r[a] = sp.radii[a];
            sm[k].ir[a] = 1.0f / sp.radii[a];
        }
        sm[k].pad[0] = sm[k].pad[1] = 0.0f;
    }
    // survivors (shape-major) and their (shape, instance) jobs
    std::uint32_t np = 0;
    for (int k = 0; k < 8; ++k) np += lists.n[k];
    if (sc.pe.size() < np + 4) { sc.pe.resize(2 * np + 4); sc.pf.resize(2 * np + 4); sc.pk.resize(2 * np + 4); }
    sc.jobs.clear();
    {
        std::uint32_t p = 0;
        for (int k = 0; k < 8; ++k) {
            const std::uint32_t first = static_cast<std::uint32_t>(sc.jobs.size());
            for (std::uint32_t j = 0; j < lists.n[k]; ++j, ++p) {
                const std::uint32_t e = lists.idx[k][j];
                const std::uint32_t inst = src.Inst(e);
                std::uint32_t f = first;
                for (; f < sc.jobs.size(); ++f) if (sc.jobs[f].instance == inst) break;
                if (f == sc.jobs.size()) sc.jobs.push_back(Job{static_cast<std::uint32_t>(k), inst, e});
                sc.pe[p] = e;
                sc.pf[p] = f;
                sc.pk[p] = static_cast<std::uint8_t>(k);
            }
        }
    }
    const std::size_t nj = sc.jobs.size();
    if (sc.m2u.size() < nj + 4) sc.m2u.resize(nj + 4);
    if (sc.back.size() < nj + 4) { sc.back.resize(nj + 4); sc.miso.resize(nj + 4); }
    for (std::size_t j0 = 0; j0 < nj; j0 += 4) {
        const Job *jb[4];
        for (int l = 0; l < 4; ++l) jb[l] = &sc.jobs[j0 + l < nj ? j0 + l : j0];
        const collide::ShapePose *sp[4] = {&shapes[jb[0]->shape], &shapes[jb[1]->shape], &shapes[jb[2]->shape], &shapes[jb[3]->shape]};
        const float *mi[4] = {src.Iso(jb[0]->entry), src.Iso(jb[1]->entry), src.Iso(jb[2]->entry), src.Iso(jb[3]->entry)};
        const lanes::I3<F4> ell = GatherIso(sp[0]->world.r, sp[1]->world.r, sp[2]->world.r, sp[3]->world.r, 0);
        const lanes::I3<F4> mesh = GatherIso(mi[0], mi[1], mi[2], mi[3], 0);
        const lanes::V3<F4> rad = {G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 0), G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 1),
                                   G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 2)};
        float *dst[4];
        lanes::I3<F4> bk;
        const lanes::I3<F4> f = noise::MakeMeshToUnit(ell, rad, mesh, &bk);
        for (int l = 0; l < 4; ++l) dst[l] = reinterpret_cast<float *>(&sc.back[j0 + l]);
        ScatterIso(bk, dst, 0);
        for (int l = 0; l < 4; ++l) dst[l] = reinterpret_cast<float *>(&sc.miso[j0 + l]);
        ScatterIso(mesh, dst, 0);
        for (int l = 0; l < 4; ++l) dst[l] = reinterpret_cast<float *>(&sc.m2u[j0 + l]);
        ScatterIso(f, dst, 0);
    }
    if (stats) { stats->frames += nj; stats->survivors += np; }
    for (int k = 0; k < 8; ++k) sc.rawn[k].clear();
#if defined(__AVX512F__)
    if (nj <= 16 && np >= kNoise16MinSurvivors) {   // 16 lanes: per-lane frames and shape maps by permutes from 16-entry tables, vertices by gathers
        using noise::F16;
        using noise::B16;
        // the tables' columns that the permutes read: jobs j < nj (survivor lanes index those only;
        // padding lanes repeat survivor 0) and shapes k < 8
        alignas(64) float T[12][16] = {}, BK[12][16] = {}, MI[12][16] = {}, SM[18][16] = {};
        for (std::size_t j = 0; j < nj; ++j) {
            const float *a = reinterpret_cast<const float *>(&sc.m2u[j]), *b = reinterpret_cast<const float *>(&sc.back[j]),
                        *c = reinterpret_cast<const float *>(&sc.miso[j]);
            for (int f = 0; f < 12; ++f) { T[f][j] = a[f]; BK[f][j] = b[f]; MI[f][j] = c[f]; }
        }
        for (int k = 0; k < 8; ++k) {
            const float *src = reinterpret_cast<const float *>(&sm[k]);
            for (int f = 0; f < 18; ++f) SM[f][k] = src[f];
        }
        alignas(64) std::int32_t jx[16], kx[16], vx[16];
        for (std::uint32_t j0 = 0; j0 < np; j0 += 16) {
            const int nl = static_cast<int>(std::min<std::uint32_t>(16u, np - j0));
            for (int l = 0; l < 16; ++l) {
                const std::uint32_t pi = j0 + (l < nl ? static_cast<std::uint32_t>(l) : 0u);
                jx[l] = static_cast<std::int32_t>(sc.pf[pi]);
                kx[l] = sc.pk[pi];
                vx[l] = static_cast<std::int32_t>(src.VertOffset(sc.pe[pi]));
            }
            const __m512i J = _mm512_load_si512(jx), Ks = _mm512_load_si512(kx), Vb = _mm512_load_si512(vx);
            auto perm = [](const __m512i &idx, const float *row) { return F16{_mm512_permutexvar_ps(idx, _mm512_load_ps(row))}; };
            lanes::I3<F16> m2u;
            m2u.r.c[0] = {perm(J, T[0]), perm(J, T[1]), perm(J, T[2])};
            m2u.r.c[1] = {perm(J, T[3]), perm(J, T[4]), perm(J, T[5])};
            m2u.r.c[2] = {perm(J, T[6]), perm(J, T[7]), perm(J, T[8])};
            m2u.t = {perm(J, T[9]), perm(J, T[10]), perm(J, T[11])};
            const float *vbase = src.VertBase();
            F16 g[9];
            for (int c = 0; c < 9; ++c) g[c] = F16{_mm512_i32gather_ps(_mm512_add_epi32(Vb, _mm512_set1_epi32(c)), vbase, 4)};
            const lanes::V3<F16> mv[3] = {{g[0], g[1], g[2]}, {g[3], g[4], g[5]}, {g[6], g[7], g[8]}};
            lanes::V3<F16> vu[3];
            for (int c = 0; c < 3; ++c) vu[c] = lanes::TPoint(m2u, mv[c]);
            B16 face, feat;
            lanes::V3<F16> P, N, S, Kn;
            noise::UnitContact<F16, B16>(vu, exact::kDirEpsSq, exact::kCollisionDistance, &face, &feat, &P, &N, &S, &Kn);
            const int lm = nl == 16 ? 0xFFFF : (1 << nl) - 1;
            const int fb = face.m & lm, eb = feat.m & lm & ~fb;
            if (stats) {
                stats->face += __builtin_popcount(static_cast<unsigned>(fb));
                stats->fvForm += __builtin_popcount(static_cast<unsigned>(eb));
                stats->reject += nl - __builtin_popcount(static_cast<unsigned>(fb | eb));
            }
            if (!(fb | eb)) continue;
            F16 q[18];
            for (int f = 0; f < 18; ++f) q[f] = perm(Ks, SM[f]);
            const lanes::V3<F16> ax[3] = {{q[0], q[1], q[2]}, {q[3], q[4], q[5]}, {q[6], q[7], q[8]}};
            const lanes::V3<F16> c = {q[9], q[10], q[11]};
            const F16 r[3] = {q[12], q[13], q[14]}, ir[3] = {q[15], q[16], q[17]};
            auto R = [&](F16 u0, F16 u1, F16 u2) {
                return lanes::V3<F16>{(ax[0].x * u0 + ax[1].x * u1) + ax[2].x * u2, (ax[0].y * u0 + ax[1].y * u1) + ax[2].y * u2,
                                      (ax[0].z * u0 + ax[1].z * u1) + ax[2].z * u2};
            };
            auto isoOf = [&](const float (*tab)[16]) {
                lanes::I3<F16> t;
                t.r.c[0] = {perm(J, tab[0]), perm(J, tab[1]), perm(J, tab[2])};
                t.r.c[1] = {perm(J, tab[3]), perm(J, tab[4]), perm(J, tab[5])};
                t.r.c[2] = {perm(J, tab[6]), perm(J, tab[7]), perm(J, tab[8])};
                t.t = {perm(J, tab[9]), perm(J, tab[10]), perm(J, tab[11])};
                return t;
            };
            // the point back through FV's c2w factors, uncomposed (see the 4-lane path)
            const lanes::V3<F16> Pw = lanes::TPoint(isoOf(MI), lanes::TPoint(isoOf(BK), lanes::V3<F16>{r[0] * P.x, r[1] * P.y, r[2] * P.z}));
            (void)c;
            const lanes::V3<F16> Sw = R(r[0] * S.x, r[1] * S.y, r[2] * S.z);
            const lanes::V3<F16> Nw = lanes::NormalizeSel<F16, B16>(R(ir[0] * N.x, ir[1] * N.y, ir[2] * N.z), exact::kDirEpsSq);
            alignas(64) float X[12][16];
            const F16 *fields[12] = {&Pw.x, &Pw.y, &Pw.z, &Nw.x, &Nw.y, &Nw.z, &Sw.x, &Sw.y, &Sw.z, &Kn.x, &Kn.y, &Kn.z};
            for (int f = 0; f < 12; ++f) _mm512_store_ps(X[f], fields[f]->v);
            unsigned any = static_cast<unsigned>(fb | eb);
            while (any) {
                const int l = __builtin_ctz(any);
                any &= any - 1u;
                const std::uint32_t pi = j0 + static_cast<std::uint32_t>(l);
                *sc.rawn[sc.pk[pi]].Next() = Raw{exact::V{X[0][l], X[1][l], X[2][l]}, exact::V{X[3][l], X[4][l], X[5][l]}, exact::V{X[6][l], X[7][l], X[8][l]},
                                                 exact::V{X[9][l], X[10][l], X[11][l]}, (fb >> l & 1) != 0, src.Tri(sc.pe[pi])};
            }
        }
        return;
    }
#endif
    for (std::uint32_t j0 = 0; j0 < np; j0 += 4) {
        const float *fr[4], *tv[4], *g[4];
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t pi = j0 + l < np ? j0 + l : j0;
            fr[l] = reinterpret_cast<const float *>(&sc.m2u[sc.pf[pi]]);
            tv[l] = src.VertBase() + src.VertOffset(sc.pe[pi]);
            g[l] = reinterpret_cast<const float *>(&sm[sc.pk[pi]]);
        }
        const lanes::I3<F4> m2u = GatherIso(fr[0], fr[1], fr[2], fr[3], 0);
        F4 p[8];
        G4x4(tv[0], tv[1], tv[2], tv[3], 0, p);
        G4x4(tv[0], tv[1], tv[2], tv[3], 4, p + 4);
        const F4 p8 = G4(tv[0], tv[1], tv[2], tv[3], 8);
        const lanes::V3<F4> mv[3] = {{p[0], p[1], p[2]}, {p[3], p[4], p[5]}, {p[6], p[7], p8}};
        lanes::V3<F4> vu[3];
        for (int c = 0; c < 3; ++c) vu[c] = lanes::TPoint(m2u, mv[c]);   // FV's unit-space vertices
        B4 face, feat;
        lanes::V3<F4> P, N, S, Kn;
        noise::UnitContact4(vu, exact::kDirEpsSq, exact::kCollisionDistance, &face, &feat, &P, &N, &S, &Kn);
        const int nl = static_cast<int>(std::min<std::uint32_t>(4u, np - j0));
        const int lm = (1 << nl) - 1;
        const int fb = lanes::Bits(face) & lm, eb = lanes::Bits(feat) & lm & ~fb;
        if (stats) {
            stats->face += __builtin_popcount(static_cast<unsigned>(fb));
            stats->fvForm += __builtin_popcount(static_cast<unsigned>(eb));
            stats->reject += nl - __builtin_popcount(static_cast<unsigned>(fb | eb));
        }
        if (!(fb | eb)) continue;
        F4 q[20];
        for (int f = 0; f < 20; f += 4) G4x4(g[0], g[1], g[2], g[3], f, q + f);
        const lanes::V3<F4> ax[3] = {{q[0], q[1], q[2]}, {q[3], q[4], q[5]}, {q[6], q[7], q[8]}};
        const lanes::V3<F4> c = {q[9], q[10], q[11]};
        const F4 r[3] = {q[12], q[13], q[14]}, ir[3] = {q[15], q[16], q[17]};
        auto R = [&](F4 u0, F4 u1, F4 u2) {   // R u = axis0 u0 + axis1 u1 + axis2 u2
            return lanes::V3<F4>{(ax[0].x * u0 + ax[1].x * u1) + ax[2].x * u2, (ax[0].y * u0 + ax[1].y * u1) + ax[2].y * u2,
                                 (ax[0].z * u0 + ax[1].z * u1) + ax[2].z * u2};
        };
        // the point back through FV's c2w factors, uncomposed: the inverse of FV's rounded meshToEllipsoid,
        // then the instance (the unit-space point returns through the rounding it was found with; against a
        // double reference this is at or below FV's own float error, where the shape's map was up to 3x it)
        const float *fb4[4], *fm4[4];
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t pi = j0 + l < np ? j0 + l : j0;
            fb4[l] = reinterpret_cast<const float *>(&sc.back[sc.pf[pi]]);
            fm4[l] = reinterpret_cast<const float *>(&sc.miso[sc.pf[pi]]);
        }
        const lanes::V3<F4> Pw = lanes::TPoint(GatherIso(fm4[0], fm4[1], fm4[2], fm4[3], 0),
                                               lanes::TPoint(GatherIso(fb4[0], fb4[1], fb4[2], fb4[3], 0), lanes::V3<F4>{r[0] * P.x, r[1] * P.y, r[2] * P.z}));
        (void)c;
        const lanes::V3<F4> Sw = R(r[0] * S.x, r[1] * S.y, r[2] * S.z);
        const lanes::V3<F4> Nw = lanes::NormalizeSel<F4, B4>(R(ir[0] * N.x, ir[1] * N.y, ir[2] * N.z), exact::kDirEpsSq);
        alignas(16) float X[12][4];
        const F4 *fields[12] = {&Pw.x, &Pw.y, &Pw.z, &Nw.x, &Nw.y, &Nw.z, &Sw.x, &Sw.y, &Sw.z, &Kn.x, &Kn.y, &Kn.z};
        for (int f = 0; f < 12; ++f) _mm_store_ps(X[f], fields[f]->v);
        for (int l = 0; l < nl; ++l) {
            if (!((fb | eb) >> l & 1)) continue;
            const std::uint32_t pi = j0 + l;
            *sc.rawn[sc.pk[pi]].Next() = Raw{exact::V{X[0][l], X[1][l], X[2][l]}, exact::V{X[3][l], X[4][l], X[5][l]}, exact::V{X[6][l], X[7][l], X[8][l]},
                                             exact::V{X[9][l], X[10][l], X[11][l]}, (fb >> l & 1) != 0, src.Tri(sc.pe[pi])};
        }
    }
}

#if defined(__AVX512F__)
// ---- per-survivor 16-lane narrow phase (StageOptions::narrow16) -----------------------------------
// Each lane holds one survivor: its shape pose (ShapePose: world iso + radii, 15 floats), its instance's iso
// (12 floats) and its mesh-local vertices (9 floats), each loaded as a 16-float row and transposed. Each lane
// makes its own FV meshToUnit (noise::MakeMeshToUnitInv, the jobs' operations) and the way back, so a lane's
// contact has the bits of the per-(shape, instance) job path; lanes need not share a shape, an instance or
// even a candidate (DetectLanes packs the survivors of up to 16 candidates).
struct Stage16 {                            // one survivor of a packet: the memo's cq::Ref with `model` replaced by `row`
    std::uint32_t instance;                 // its instance (its iso: 12 floats)
    std::uint32_t modelTri;                 // its vertices (9 floats)
    std::uint32_t row;                      // its shape pose (15 floats) at rows + row x stride; its shape: row & 7
    std::uint32_t tri;                      // its Raw::tri
};
static_assert(sizeof(collide::ShapePose) == 15 * sizeof(float), "ShapePose: 15 floats");
static_assert(sizeof(Raw) == 15 * sizeof(float) && offsetof(Raw, K) == 36 && offsetof(Raw, primary) == 48 && offsetof(Raw, tri) == 52 &&
                  offsetof(Raw, shape) == 56 && sizeof(bool) == 1,
              "Narrow16: a Raw is 15 words (P N Sep K, primary, tri, shape)");
static_assert(sizeof(Stage16) == sizeof(cq::Ref) && offsetof(cq::Ref, instance) == offsetof(Stage16, instance) &&
              offsetof(cq::Ref, modelTri) == offsetof(Stage16, modelTri) && offsetof(cq::Ref, local) == offsetof(Stage16, tri),
              "Stage16: a cq::Ref with its third word replaced");
// a survivor's Stage16: its memo Ref (local = world triangle id) with the row in place of the model
CC_LI void StageOne(Stage16 *g, const cq::Ref *rf, std::uint32_t row) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(g), _mm_insert_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(rf)), static_cast<int>(row), 2));
}

// res[l]: lane l's contact when bit l of *hit is set (res: 16 slots); *faceOut (if given): the hits that are FACEs (primaries).
// Each survivor's rows: its shape pose at rows + row x kStride floats, its instance's iso, its vertices; each read as a 16-float row.
// irad[a][k]: 1 / radius a of shape k (the same division as per lane, so the same bits).
// Scheduled before register allocation with register pressure in view (GCC's sched1 + sched-pressure; x86 runs only the
// post-allocation pass by default): fewer spills of the 48 transposed rows, whose reloads were half of the blocked loads.
// Same operations, so the same bits.
template <int kStride>
__attribute__((noinline, optimize("schedule-insns", "sched-pressure"))) void Narrow16(const Stage16 *pk, std::uint32_t nl, const float *rows, const world::Instance *inst,
                                        const exact::TriVerts *verts, const float (*irad)[16], Raw *res, std::uint16_t *hit,
                                        NoiseCounts *stats, std::uint16_t *faceOut = nullptr) {
    using noise::F16;
    using noise::B16;
    // the packet's (row, tri) words as lane columns (16 Stage16 = 4 rows; lanes past nl: whatever the buffer holds, never output)
    __m512i rowv, triv;
    {
        const __m512i q01 = _mm512_permutex2var_epi64(_mm512_loadu_si512(pk), _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15),
                                                      _mm512_loadu_si512(pk + 4));
        const __m512i q23 = _mm512_permutex2var_epi64(_mm512_loadu_si512(pk + 8), _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15),
                                                      _mm512_loadu_si512(pk + 12));
        rowv = _mm512_permutex2var_epi32(q01, _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), q23);
        triv = _mm512_permutex2var_epi32(q01, _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), q23);
    }
    const __m512i kv = _mm512_and_si512(rowv, _mm512_set1_epi32(7));
    // rows -> lanes (16-float row loads and transposes: gathers cost ~3x more here); padding lanes repeat lane 0
    __m512 a[16], b[16], c[16];
#pragma GCC unroll 16
    for (std::uint32_t l = 0; l < 16; ++l) {
        const Stage16 &g = pk[l < nl ? l : 0u];
        a[l] = _mm512_loadu_ps(rows + std::size_t(g.row) * kStride);
        b[l] = _mm512_loadu_ps(inst[g.instance].iso);
        c[l] = _mm512_loadu_ps(reinterpret_cast<const float *>(&verts[g.modelTri]));
    }
    noise::Transpose16(a);
    noise::Transpose16(b);
    noise::Transpose16(c);
    lanes::I3<F16> ell, mesh;
    ell.r.c[0] = {F16{a[0]}, F16{a[1]}, F16{a[2]}};
    ell.r.c[1] = {F16{a[3]}, F16{a[4]}, F16{a[5]}};
    ell.r.c[2] = {F16{a[6]}, F16{a[7]}, F16{a[8]}};
    ell.t = {F16{a[9]}, F16{a[10]}, F16{a[11]}};
    const lanes::V3<F16> rad = {F16{a[12]}, F16{a[13]}, F16{a[14]}};
    mesh.r.c[0] = {F16{b[0]}, F16{b[1]}, F16{b[2]}};
    mesh.r.c[1] = {F16{b[3]}, F16{b[4]}, F16{b[5]}};
    mesh.r.c[2] = {F16{b[6]}, F16{b[7]}, F16{b[8]}};
    mesh.t = {F16{b[9]}, F16{b[10]}, F16{b[11]}};
    const lanes::V3<F16> invRad = {F16{_mm512_permutexvar_ps(kv, _mm512_loadu_ps(irad[0]))}, F16{_mm512_permutexvar_ps(kv, _mm512_loadu_ps(irad[1]))},
                                   F16{_mm512_permutexvar_ps(kv, _mm512_loadu_ps(irad[2]))}};
    lanes::I3<F16> bk;
    const lanes::I3<F16> m2u = noise::MakeMeshToUnitInv(ell, invRad, mesh, &bk);
    const lanes::V3<F16> mv[3] = {{F16{c[0]}, F16{c[1]}, F16{c[2]}}, {F16{c[3]}, F16{c[4]}, F16{c[5]}}, {F16{c[6]}, F16{c[7]}, F16{c[8]}}};
    lanes::V3<F16> vu[3];
    for (int k = 0; k < 3; ++k) vu[k] = lanes::TPoint(m2u, mv[k]);   // FV's unit-space vertices
    B16 face, feat;
    lanes::V3<F16> P, N, S, Kn;
    noise::UnitContact<F16, B16>(vu, exact::kDirEpsSq, exact::kCollisionDistance, &face, &feat, &P, &N, &S, &Kn);
    const int lm = nl == 16 ? 0xFFFF : (1 << nl) - 1;
    const int fb = face.m & lm, eb = feat.m & lm & ~fb;
    if (stats) {
        stats->face += __builtin_popcount(static_cast<unsigned>(fb));
        stats->fvForm += __builtin_popcount(static_cast<unsigned>(eb));
        stats->reject += static_cast<int>(nl) - __builtin_popcount(static_cast<unsigned>(fb | eb));
    }
    *hit = static_cast<std::uint16_t>(fb | eb);
    if (faceOut != nullptr) *faceOut = static_cast<std::uint16_t>(fb);
    if (!(fb | eb)) return;
    const F16 r[3] = {rad.x, rad.y, rad.z}, ir[3] = {invRad.x, invRad.y, invRad.z};
    const lanes::V3<F16> *ax = ell.r.c;
    auto R = [&](F16 u0, F16 u1, F16 u2) {
        return lanes::V3<F16>{(ax[0].x * u0 + ax[1].x * u1) + ax[2].x * u2, (ax[0].y * u0 + ax[1].y * u1) + ax[2].y * u2,
                              (ax[0].z * u0 + ax[1].z * u1) + ax[2].z * u2};
    };
    // the point back through FV's c2w factors, uncomposed (as the job path)
    const lanes::V3<F16> Pw = lanes::TPoint(mesh, lanes::TPoint(bk, lanes::V3<F16>{r[0] * P.x, r[1] * P.y, r[2] * P.z}));
    const lanes::V3<F16> Sw = R(r[0] * S.x, r[1] * S.y, r[2] * S.z);
    const lanes::V3<F16> Nw = lanes::NormalizeSel<F16, B16>(R(ir[0] * N.x, ir[1] * N.y, ir[2] * N.z), exact::kDirEpsSq);
    // the contacts as Raw rows (P N Sep K, primary, tri, shape: 15 words, padding bytes 0) by one transpose, every slot written
    // (only the hits' slots are read)
    __m512 o[16] = {Pw.x.v, Pw.y.v, Pw.z.v, Nw.x.v, Nw.y.v, Nw.z.v, Sw.x.v, Sw.y.v, Sw.z.v, Kn.x.v, Kn.y.v, Kn.z.v,
                    _mm512_castsi512_ps(_mm512_maskz_mov_epi32(static_cast<__mmask16>(fb), _mm512_set1_epi32(1))), _mm512_castsi512_ps(triv),
                    _mm512_castsi512_ps(kv), _mm512_setzero_ps()};
    noise::Transpose16(o);
#pragma GCC unroll 16
    for (int l = 0; l < 16; ++l) _mm512_mask_storeu_ps(&res[l], 0x7FFF, o[l]);
}
#endif

// per-shape merge (collision_buffer.cpp MergeAndAddToCollisions) and the RespContact output
template <class RawK>
void MergeOut(const world::WorldPackView &v, const collide::CarShapes &car, const RawK rawk[8], std::vector<int> &merged,
              step::SubstepContacts &out, StageStats *stats, int mergeMode = 0, const OutRec *outRec = nullptr) {
    for (int k = 0; k < 8; ++k) {
        const RawK &raw = rawk[k];
        if (raw.empty()) continue;
        const bool body = car.shape[k].wheel < 0;
        if (stats) { stats->raw += raw.size(); if (body) stats->bodyRaw += raw.size(); }
        merged.clear();
        if (mergeMode == 1) {   // relaxation: every raw contact, in FV order
            for (std::size_t r = 0; r < raw.size(); ++r) merged.push_back(static_cast<int>(r));
        } else if (mergeMode == 2) {   // relaxation: the deepest contact (largest |Sep|) only
            std::size_t best = 0;
            float bd = -1.0f;
            for (std::size_t r = 0; r < raw.size(); ++r) {
                const float d = exact::Dot(raw[r].Sep, raw[r].Sep);
                if (d > bd) { bd = d; best = r; }
            }
            merged.push_back(static_cast<int>(best));
        }
        if (mergeMode == 0)
        for (std::size_t r = 0; r < raw.size(); ++r) if (raw[r].primary) merged.push_back(static_cast<int>(r));
        const std::size_t nPrim = merged.size();
        for (std::size_t r = 0; r < raw.size() && mergeMode == 0; ++r) {
            const Raw &c = raw[r];
            if (c.primary) continue;
            std::size_t j = 0;
            for (; j < nPrim; ++j) {
                const Raw &p = raw[static_cast<std::size_t>(merged[j])];
                if (NearlyEqual(c.K, p.K) || kNormalAlignment < exact::Dot(c.N, p.N)) break;
            }
            if (j == nPrim) merged.push_back(static_cast<int>(r));
        }
        if (outRec != nullptr) {   // the triangle's output fields in one record
            for (int idx : merged) {
                if (out.n >= model::kRespMaxContacts) { out.overflow = true; break; }
                const Raw &c = raw[static_cast<std::size_t>(idx)];
                const OutRec &r = outRec[c.tri];
                model::RespContact &o = out.c[out.n++];
                o.P[0] = c.P.x; o.P[1] = c.P.y; o.P[2] = c.P.z;
                o.N[0] = c.N.x; o.N[1] = c.N.y; o.N[2] = c.N.z;
                o.Sep[0] = c.Sep.x; o.Sep[1] = c.Sep.y; o.Sep[2] = c.Sep.z;
                o.peerZ[0] = r.peerZ[0]; o.peerZ[1] = r.peerZ[1]; o.peerZ[2] = r.peerZ[2];
                o.peerId = r.peerId;
                o.shape = static_cast<std::uint8_t>(k);
                o.material = r.material;
                o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | r.flags);
                o.pad = 0u;
            }
            if (stats) { stats->merged += merged.size(); if (body) stats->bodyMerged += merged.size(); }
            continue;
        }
        for (int idx : merged) {
            if (out.n >= model::kRespMaxContacts) { out.overflow = true; break; }
            const Raw &c = raw[static_cast<std::size_t>(idx)];
            const Tri &tr = v.tris[c.tri];
            const world::TriInfo &ti = v.triInfo[c.tri];
            const world::Actor &a = v.ActorOf(tr);
            model::RespContact &o = out.c[out.n++];
            o.P[0] = c.P.x; o.P[1] = c.P.y; o.P[2] = c.P.z;
            o.N[0] = c.N.x; o.N[1] = c.N.y; o.N[2] = c.N.z;
            o.Sep[0] = c.Sep.x; o.Sep[1] = c.Sep.y; o.Sep[2] = c.Sep.z;
            o.peerZ[0] = a.peerZ[0]; o.peerZ[1] = a.peerZ[1]; o.peerZ[2] = a.peerZ[2];
            const bool trig = ti.trigger != world::kNoTrigger;
            o.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
            o.shape = static_cast<std::uint8_t>(k);
            o.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
            o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (trig ? unsigned(model::kRespTrigger) : 0u));
            o.pad = 0u;
        }
        if (stats) { stats->merged += merged.size(); if (body) stats->bodyMerged += merged.size(); }
    }
}

// MergeOut for the noise path with per-triangle output records, fewer data-dependent branches: only the
// non-empty shapes, and a shape whose raw contacts are all FACEs (primaries) is copied without merge tests.
// The same contacts in the same order as MergeOut.
inline void FillOut(model::RespContact &o, const Raw &c, const OutRec &r, int k) {
    o.P[0] = c.P.x; o.P[1] = c.P.y; o.P[2] = c.P.z;
    o.N[0] = c.N.x; o.N[1] = c.N.y; o.N[2] = c.N.z;
    o.Sep[0] = c.Sep.x; o.Sep[1] = c.Sep.y; o.Sep[2] = c.Sep.z;
    o.peerZ[0] = r.peerZ[0]; o.peerZ[1] = r.peerZ[1]; o.peerZ[2] = r.peerZ[2];
    o.peerId = r.peerId;
    o.shape = static_cast<std::uint8_t>(k);
    o.material = r.material;
    o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | r.flags);
    o.pad = 0u;
}
#if defined(__AVX2__)
void MergeOutLean(const RawList raw[8], const OutRec *outRec, step::SubstepContacts &out) {
    unsigned mask = 0;
    for (int k = 0; k < 8; ++k) mask |= static_cast<unsigned>(raw[k].n != 0u) << k;
    int n = 0;
    for (; mask; mask &= mask - 1u) {
        const int k = __builtin_ctz(mask);
        const RawList &rl = raw[k];
        const std::size_t nr = rl.n;
        const Raw *r = rl.data();
        bool allPrimary = true;
        for (std::size_t i = 0; i < nr; ++i) allPrimary &= r[i].primary;
        if (allPrimary && n + static_cast<int>(nr) <= model::kRespMaxContacts) {
            for (std::size_t i = 0; i < nr; ++i) FillOut(out.c[n++], r[i], outRec[r[i].tri], k);
            continue;
        }
        // FV's merge: every primary, then each non-primary not coplanar with (NearlyEqual K) nor within 30 degrees
        // of a primary of the shape
        int prim[model::kRespMaxContacts];
        int np = 0;
        for (std::size_t i = 0; i < nr && np < model::kRespMaxContacts; ++i) if (r[i].primary) prim[np++] = static_cast<int>(i);
        bool full = false;
        for (int j = 0; j < np; ++j) {
            if (n >= model::kRespMaxContacts) { full = true; break; }
            FillOut(out.c[n++], r[prim[j]], outRec[r[prim[j]].tri], k);
        }
        for (std::size_t i = 0; i < nr && !full; ++i) {
            const Raw &c = r[i];
            if (c.primary) continue;
            int j = 0;
            for (; j < np; ++j) {
                const Raw &p = r[prim[j]];
                if (NearlyEqual(c.K, p.K) || kNormalAlignment < exact::Dot(c.N, p.N)) break;
            }
            if (j < np) continue;
            if (n >= model::kRespMaxContacts) { full = true; break; }
            FillOut(out.c[n++], c, outRec[c.tri], k);
        }
        if (full) { out.overflow = true; break; }
    }
    out.n = n;
}

#endif

#if defined(__AVX512F__)
// FillOut as one 56-byte masked store: P N Sep from the Raw (its first 9 floats), peerZ, peerId and material << 8 | flags << 16
// from the OutRec (its 5 words, one masked load into words 9-13), then shape | primary << 16 or-ed in: the same bytes.
static_assert(offsetof(Raw, P) == 0 && offsetof(Raw, N) == 12 && offsetof(Raw, Sep) == 24 && offsetof(OutRec, peerId) == 12 &&
                  offsetof(OutRec, zero) == 16 && offsetof(OutRec, material) == 17 && offsetof(OutRec, flags) == 18 && offsetof(OutRec, pad) == 19 &&
                  offsetof(model::RespContact, peerZ) == 36 && offsetof(model::RespContact, shape) == 52 && offsetof(model::RespContact, material) == 53 &&
                  offsetof(model::RespContact, flags) == 54 && offsetof(model::RespContact, pad) == 55,
              "FillOut16 layout");
inline void FillOut16(model::RespContact &o, const Raw &c, const OutRec &r, int k) {
    const __m512 a = _mm512_maskz_loadu_ps(0x1FF, &c.P.x);
    // words 9-13 from the OutRec: a load based 9 words before it, masked to those words (the others are neither read nor faulted)
    const __m512 v = _mm512_mask_loadu_ps(a, 0x3E00, reinterpret_cast<const float *>(reinterpret_cast<std::uintptr_t>(&r) - 9u * sizeof(float)));
    const int w = k | static_cast<int>((c.primary ? unsigned(model::kRespPrimary) : 0u) << 16);
    _mm512_mask_storeu_ps(&o, 0x3FFF, _mm512_castsi512_ps(_mm512_mask_or_epi32(_mm512_castps_si512(v), 0x2000, _mm512_castps_si512(v), _mm512_set1_epi32(w))));
}

// MergeOutLean on the packets' flat results: survivors [s0, s1) of one call in pair order (FV order within each
// shape; shapes ascending when the call had one near block), contacts res[s] where hits[s / 16] bit s % 16.
// The same contacts in the same order as MergeOut. Up to 64 contacts (kRespMaxContacts), the shapes' groups as
// bit sets over the contacts' positions (one byte compare per shape present): no sort and no group scan, each
// group's members in position order as the stable sort by shape would give them.
static_assert(model::kRespMaxContacts == 64, "MergeFlat: one 64-bit set per shape");
void MergeFlat(const Raw *res, const std::uint16_t *hits, std::uint32_t s0, std::uint32_t s1, const OutRec *outRec,
               step::SubstepContacts &out) {
    if (s1 <= s0) { out.n = 0; return; }
    thread_local std::vector<int> cv(64);
    if (cv.size() < s1 - s0) cv.resize(2 * (s1 - s0));
    int *c = cv.data();
    int nc = 0;
    bool sorted = true, allPrimary = true;
    int prevShape = 0;
    unsigned present = 0u;   // bit k: shape k has a contact
    alignas(64) std::uint8_t shp[64], pri[64];   // the first 64 contacts' shapes and primary flags
    _mm512_store_si512(shp, _mm512_setzero_si512());
    _mm512_store_si512(pri, _mm512_setzero_si512());
    for (std::uint32_t w = s0 >> 4; w <= (s1 - 1u) >> 4; ++w) {   // the set bits of the range, in order
        std::uint32_t m = hits[w];
        if (w == s0 >> 4) m &= 0xFFFFu << (s0 & 15u);
        if (w == (s1 - 1u) >> 4) m &= 0xFFFFu >> (15u - ((s1 - 1u) & 15u));
        for (; m; m &= m - 1u) {
            const int si = static_cast<int>(16u * w + static_cast<std::uint32_t>(__builtin_ctz(m)));
            const Raw &r = res[si];
            sorted &= r.shape >= prevShape;
            allPrimary &= r.primary;
            prevShape = r.shape;
            present |= 1u << r.shape;
            shp[nc & 63] = r.shape;
            pri[nc & 63] = static_cast<std::uint8_t>(r.primary);
            c[nc++] = si;
        }
    }
    if (sorted && allPrimary && nc <= model::kRespMaxContacts) {   // the common case: every contact, in order
        for (int i = 0; i < nc; ++i) FillOut16(out.c[i], res[c[i]], outRec[res[c[i]].tri], res[c[i]].shape);
        out.n = nc;
        return;
    }
    int n = 0;
    if (nc <= model::kRespMaxContacts) {
        const __mmask64 valid = nc == 64 ? ~0ull : (1ull << nc) - 1ull;
        const __m512i vs = _mm512_load_si512(shp), vp = _mm512_load_si512(pri);
        const std::uint64_t pm = _mm512_mask_test_epi8_mask(valid, vp, vp);
        for (; present != 0u; present &= present - 1u) {   // shapes ascending
            const int k = __builtin_ctz(present);
            const std::uint64_t g = _mm512_mask_cmpeq_epi8_mask(valid, vs, _mm512_set1_epi8(static_cast<char>(k))), gp = g & pm;
            if (gp == g && n + __builtin_popcountll(g) <= model::kRespMaxContacts) {
                for (std::uint64_t b = g; b; b &= b - 1u) {
                    const Raw &r = res[c[__builtin_ctzll(b)]];
                    FillOut16(out.c[n++], r, outRec[r.tri], k);
                }
                continue;
            }
            bool full = false;
            for (std::uint64_t b = gp; b; b &= b - 1u) {   // every primary, in order
                if (n >= model::kRespMaxContacts) { full = true; break; }
                const Raw &r = res[c[__builtin_ctzll(b)]];
                FillOut16(out.c[n++], r, outRec[r.tri], k);
            }
            for (std::uint64_t b = g & ~pm; b && !full; b &= b - 1u) {   // a non-primary unless coplanar with or within 30 degrees of a primary
                const Raw &f = res[c[__builtin_ctzll(b)]];
                std::uint64_t q = gp;
                for (; q; q &= q - 1u) {
                    const Raw &p = res[c[__builtin_ctzll(q)]];
                    if (NearlyEqual(f.K, p.K) || kNormalAlignment < exact::Dot(f.N, p.N)) break;
                }
                if (q) continue;
                if (n >= model::kRespMaxContacts) { full = true; break; }
                FillOut16(out.c[n++], f, outRec[f.tri], k);
            }
            if (full) { out.overflow = true; break; }
        }
        out.n = n;
        return;
    }
    if (!sorted)   // stable by shape (insertion sort: few contacts)
        for (int i = 1; i < nc; ++i) {
            const int v = c[i];
            int j = i - 1;
            for (; j >= 0 && res[c[j]].shape > res[v].shape; --j) c[j + 1] = c[j];
            c[j + 1] = v;
        }
    for (int g0 = 0; g0 < nc;) {
        const int k = res[c[g0]].shape;
        int g1 = g0;
        bool allPrimary = true;
        for (; g1 < nc && res[c[g1]].shape == k; ++g1) allPrimary &= res[c[g1]].primary;
        if (allPrimary && n + (g1 - g0) <= model::kRespMaxContacts) {
            for (int i = g0; i < g1; ++i) FillOut16(out.c[n++], res[c[i]], outRec[res[c[i]].tri], k);
            g0 = g1;
            continue;
        }
        bool full = false;
        for (int i = g0; i < g1; ++i) {   // every primary, in order
            if (!res[c[i]].primary) continue;
            if (n >= model::kRespMaxContacts) { full = true; break; }
            FillOut16(out.c[n++], res[c[i]], outRec[res[c[i]].tri], k);
        }
        for (int i = g0; i < g1 && !full; ++i) {   // a non-primary unless coplanar with or within 30 degrees of a primary
            const Raw &f = res[c[i]];
            if (f.primary) continue;
            int j = g0;
            for (; j < g1; ++j) {
                const Raw &p = res[c[j]];
                if (p.primary && (NearlyEqual(f.K, p.K) || kNormalAlignment < exact::Dot(f.N, p.N))) break;
            }
            if (j < g1) continue;
            if (n >= model::kRespMaxContacts) { full = true; break; }
            FillOut16(out.c[n++], f, outRec[f.tri], k);
        }
        if (full) { out.overflow = true; break; }
        g0 = g1;
    }
    out.n = n;
}
#endif

#if CC_STEADYSTATS && defined(__AVX512F__)
// Steady-contact counters (docs/twin/collision-x2-night.md "Steady-contact fast path"): measurement only, the output
// is untouched. Per world (key: triangle count) and per lane-substep class (no contact / a wheel contact / body
// contacts only):
//   (A) per margin m, a shadow "cached survivor list" per candidate cache: at a full call, Filter's survivors with
//       every shape grown by m (ProveLanes16's grown table, w = 0: boxes +- m, reach + m, low - m, tol + m, lazy - m) on a
//       memo holding the union box + m (the lane's memo, else a temporary refill); the next calls are servable while
//       every shape's motion bound since the full call (|centre change| + ||car rotation change||_F x lever, plus the
//       prefilter tolerance's growth) stays under m (less a float margin). Servable share, list size against the exact
//       survivors, served calls per full call, and two checks on servable calls: exact survivors outside the list and
//       raw contacts outside it (both must be 0).
//   (B) the oracle bound: the raw contact set (shape, triangle, FACE or feature) equal to the previous call's and every
//       contact a clear FACE (its point >= 1 mm inside the triangle's edges, |Sep| >= 1e-4 m).
// Bit 2 of CC_STEADYSTATS: TSC per phase of DetectLanes16 (per pass, charged to the pass's lanes), per world.
namespace steady {
constexpr int kM = 5;
constexpr float kMargin[kM] = {0.05f, 0.1f, 0.2f, 0.3f, 0.5f};
// swept variants: each shape's centre swept along K x its last call-to-call motion, grown by D (rotation-free boxes:
// half extent |radii|_2 per axis); servable while every centre stays within D (less rotation and tolerance terms) of
// its segment
constexpr int kS = 6;
constexpr int kSweepK[kS] = {4, 8, 8, 16, 16, 32};
constexpr float kSweepD[kS] = {0.03f, 0.03f, 0.06f, 0.06f, 0.1f, 0.1f};
constexpr int kV = kM + kS;
constexpr int kCls = 3;   // 0: no contact, 1: a wheel contact, 2: body contacts only
constexpr int kPh = 7;    // TSC phases: front 1, memo, front 2, filter, staging, narrow, merge
struct Cls {
    std::uint64_t calls = 0, exact = 0, raw = 0, merged = 0, same = 0, sameFace = 0, oracle = 0;
    std::uint64_t serv[kV] = {}, servList[kV] = {}, servExact[kV] = {}, fulls[kV] = {}, fullList[kV] = {}, fullExact[kV] = {}, memoMiss[kV] = {},
                  violS[kV] = {}, violC[kV] = {}, violFull[kV] = {};
};
struct Row {
    Cls c[kCls];
    std::uint64_t passes = 0, lanes = 0, surv = 0, near = 0, pairs = 0, tsc[kPh] = {};
};
thread_local std::uint64_t gNear = 0, gPairs = 0;   // the pass's near entries and box pairs (FilterPass16's memo counters)
void AddRow(Row &d, const Row &r) {
    for (int c = 0; c < kCls; ++c) {
        Cls &a = d.c[c];
        const Cls &b = r.c[c];
        a.calls += b.calls; a.exact += b.exact; a.raw += b.raw; a.merged += b.merged; a.same += b.same; a.sameFace += b.sameFace; a.oracle += b.oracle;
        for (int i = 0; i < kV; ++i) {
            a.serv[i] += b.serv[i]; a.servList[i] += b.servList[i]; a.servExact[i] += b.servExact[i]; a.fulls[i] += b.fulls[i];
            a.fullList[i] += b.fullList[i]; a.fullExact[i] += b.fullExact[i]; a.memoMiss[i] += b.memoMiss[i]; a.violS[i] += b.violS[i];
            a.violC[i] += b.violC[i]; a.violFull[i] += b.violFull[i];
        }
    }
    d.passes += r.passes; d.lanes += r.lanes; d.surv += r.surv; d.near += r.near; d.pairs += r.pairs;
    for (int p = 0; p < kPh; ++p) d.tsc[p] += r.tsc[p];
}
struct Table {
    std::mutex mu;
    std::vector<std::pair<std::uint32_t, Row>> rows;
    void Add(std::uint32_t key, const Row &r) {
        std::lock_guard<std::mutex> g(mu);
        for (auto &p : rows) if (p.first == key) { AddRow(p.second, r); return; }
        rows.emplace_back(key, Row{});
        AddRow(rows.back().second, r);
    }
    static void Print(const char *name, const Row &w) {
        Cls t;
        Row tmp;
        for (int c = 0; c < kCls; ++c) { Row one; one.c[0] = w.c[c]; AddRow(tmp, one); }
        t = tmp.c[0];
        auto line = [&](const char *cn, const Cls &c) {
            if (c.calls == 0u) return;
            const double n = double(c.calls);
            std::fprintf(stderr, "steady %-9s %-6s calls %9llu  exact %.2f raw %.2f merged %.2f | oracle: same set %.3f, same+all FACE %.3f, "
                                 "same+clear FACE %.3f\n", name, cn, static_cast<unsigned long long>(c.calls), double(c.exact) / n, double(c.raw) / n,
                         double(c.merged) / n, double(c.same) / n, double(c.sameFace) / n, double(c.oracle) / n);
            for (int i = 0; i < kV; ++i) {
                const double s = double(std::max<std::uint64_t>(1, c.serv[i])), f = double(std::max<std::uint64_t>(1, c.fulls[i]));
                char vn[32];
                if (i < kM) std::snprintf(vn, sizeof vn, "m %.2f", double(kMargin[i]));
                else std::snprintf(vn, sizeof vn, "K%02d D%.2f", kSweepK[i - kM], double(kSweepD[i - kM]));
                std::fprintf(stderr, "steady %-9s %-6s  %s: servable %.3f, served list %.2f (exact %.2f), full list %.2f (exact %.2f), "
                                     "all-call list %.2f, served/full %.2f, memo miss %.3f, viol S %llu C %llu full %llu\n",
                             name, cn, vn, double(c.serv[i]) / n, double(c.servList[i]) / s, double(c.servExact[i]) / s,
                             double(c.fullList[i]) / f, double(c.fullExact[i]) / f, double(c.servList[i] + c.fullList[i]) / n,
                             double(c.serv[i]) / f, double(c.memoMiss[i]) / f, static_cast<unsigned long long>(c.violS[i]),
                             static_cast<unsigned long long>(c.violC[i]), static_cast<unsigned long long>(c.violFull[i]));
            }
        };
        if (CC_STEADYSTATS & 1) {
            line("all", t);
            line("air", w.c[0]);
            line("wheel", w.c[1]);
            line("body", w.c[2]);
        }
        if ((CC_STEADYSTATS & 2) && w.lanes != 0u) {
            const double l = double(w.lanes);
            static const char *ph[kPh] = {"front1", "memo", "front2", "filter", "staging", "narrow", "merge"};
            std::uint64_t tot = 0;
            for (int p = 0; p < kPh; ++p) tot += w.tsc[p];
            std::fprintf(stderr, "steadytsc %-9s lane calls %llu (%.2f lanes/pass), near %.2f, box pairs %.2f, survivors %.2f/lane call, TSC/lane call %.0f:",
                         name, static_cast<unsigned long long>(w.lanes), l / double(std::max<std::uint64_t>(1, w.passes)), double(w.near) / l,
                         double(w.pairs) / l, double(w.surv) / l, double(tot) / l);
            for (int p = 0; p < kPh; ++p) std::fprintf(stderr, " %s %.0f", ph[p], double(w.tsc[p]) / l);
            std::fprintf(stderr, " | narrow per survivor %.1f\n", double(w.tsc[5]) / double(std::max<std::uint64_t>(1, w.surv)));
        }
    }
    ~Table() {
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        Row all;
        for (auto &p : rows) {
            char name[32];
            std::snprintf(name, sizeof name, "%u", p.first);
            Print(name, p.second);
            AddRow(all, p.second);
        }
        Print("ALL", all);
    }
};
Table &Tab() { static Table t; return t; }
struct alignas(64) LPkt {   // the refilter prototype's list packet: 16 (entry, shape) pairs, planes and boxes SoA
    float pl[16][16];
    float lo[3][16], hi[3][16];
    std::int32_t k[16];
    std::uint32_t e[16];
    std::uint32_t n;
};
struct Shadow {
    bool valid = false;
    float wt[8][3], rc[9], tol[8], w[8][3];
    std::vector<std::uint64_t> keys;   // (world triangle << 3 | shape), sorted
    std::vector<LPkt> pk;              // the refilter prototype's packets (variant kRF)
};
constexpr int kRF = 2;   // the swept variant with the refilter prototype timed (K08 D0.06)
struct Proto { std::uint64_t lanes = 0, pkts = 0, tsc = 0, out = 0, nearS = 0, pairsS = 0, fulls = 0; };
// the refilter prototype: Filter's box test and Pass16 on a cached packet list against the lane's shape table, survivors
// (entry, shape) compressed out; returns the count
__attribute__((noinline)) std::uint32_t Refilter(const LPkt *pk, std::uint32_t np, const ShapeTable &t, std::uint32_t *oe, std::uint32_t *ok) {
    using F = __m512;
    auto T = [&](int i) { return _mm512_loadu_ps(t.t[i]); };
    auto dot = [](F ax, F ay, F az, F bx, F by, F bz) { return _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(ax, bx), _mm512_mul_ps(ay, by)), _mm512_mul_ps(az, bz)); };
    const F slack = _mm512_set1_ps(exact::kSupportSlack), zero = _mm512_setzero_ps();
    std::uint32_t on = 0;
    for (std::uint32_t p = 0; p < np; ++p) {
        const LPkt &P = pk[p];
        const __m512i K = _mm512_load_si512(P.k);
        __mmask16 m = static_cast<__mmask16>(P.n >= 16 ? 0xFFFFu : (1u << P.n) - 1u);
        for (int a = 0; a < 3; ++a) {
            const F slo = _mm512_permutexvar_ps(K, _mm512_loadu_ps(t.lo[a])), shi = _mm512_permutexvar_ps(K, _mm512_loadu_ps(t.hi[a]));
            m &= _mm512_cmp_ps_mask(_mm512_load_ps(P.lo[a]), shi, _CMP_LE_OQ) & _mm512_cmp_ps_mask(slo, _mm512_load_ps(P.hi[a]), _CMP_LE_OQ);
        }
        const F c0 = _mm512_permutexvar_ps(K, T(0)), c1 = _mm512_permutexvar_ps(K, T(1)), c2 = _mm512_permutexvar_ps(K, T(2));
        F q[6];
        for (int i = 0; i < 6; ++i) q[i] = _mm512_permutexvar_ps(K, T(3 + i));
        const F reach = _mm512_permutexvar_ps(K, T(9)), low = _mm512_permutexvar_ps(K, T(10)), tol = _mm512_permutexvar_ps(K, T(11)),
                lazy = _mm512_permutexvar_ps(K, T(12));
        F pl[16];
        for (int f = 0; f < 16; ++f) pl[f] = _mm512_load_ps(P.pl[f]);
        F dist[4];
        __mmask16 lz[4], any = 0;
        for (int j = 0; j < 4; ++j) {
            dist[j] = _mm512_sub_ps(dot(pl[j], pl[4 + j], pl[8 + j], c0, c1, c2), pl[12 + j]);
            if (j == 0) m &= ~_mm512_cmp_ps_mask(dist[0], low, _CMP_LT_OQ);
            m &= ~_mm512_cmp_ps_mask(dist[j], reach, _CMP_GT_OQ);
            lz[j] = _mm512_cmp_ps_mask(dist[j], lazy, _CMP_GT_OQ);
            any |= lz[j];
        }
        if (any & m)
            for (int j = 0; j < 4; ++j) {
                const F w0 = pl[j], w1 = pl[4 + j], w2 = pl[8 + j];
                const F a = dot(q[0], q[3], q[4], w0, w1, w2), b = dot(q[3], q[1], q[5], w0, w1, w2), c = dot(q[4], q[5], q[2], w0, w1, w2);
                const F h2 = dot(a, b, c, w0, w1, w2);
                const F x = _mm512_sub_ps(dist[j], tol);
                m &= ~(lz[j] & _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) & _mm512_cmp_ps_mask(_mm512_mul_ps(x, x), _mm512_mul_ps(h2, slack), _CMP_GT_OQ));
            }
        _mm512_storeu_si512(oe + on, _mm512_maskz_compress_epi32(m, _mm512_load_si512(P.e)));
        _mm512_storeu_si512(ok + on, _mm512_maskz_compress_epi32(m, K));
        on += static_cast<std::uint32_t>(__builtin_popcount(m));
    }
    return on;
}
struct Side {
    std::uint64_t stage = 0;
    Shadow s[kV];
    std::vector<std::uint64_t> prevRaw;
    bool havePrev = false;
    float prevWt[8][3];
    bool havePrevWt = false;
};
struct Local {
    Proto proto;
    std::vector<std::uint32_t> oe, ok, pe, pk2;
    std::vector<std::pair<std::uint32_t, Row>> rows;
    std::unordered_map<const void *, Side> side;
    std::vector<std::uint64_t> exactKeys, rawKeys, keys;
    Memo tmp;
    Survivors sink;
    cq::Lists lists;
    Row &Get(std::uint32_t key) {
        for (auto &p : rows) if (p.first == key) return p.second;
        rows.emplace_back(key, Row{});
        return rows.back().second;
    }
    ~Local() {
        for (auto &p : rows) Tab().Add(p.first, p.second);
        if (proto.lanes) {
            const double l = double(proto.lanes), f = double(std::max<std::uint64_t>(1, proto.fulls));
            std::fprintf(stderr, "steadyproto refilter (K08 D0.06): %llu served lanes, %.2f packets/lane, %.1f TSC/lane (%.1f per packet), %.2f survivors/lane;"
                                 " full calls: swept near %.2f, swept box pairs %.2f\n",
                         static_cast<unsigned long long>(proto.lanes), double(proto.pkts) / l, double(proto.tsc) / l,
                         double(proto.tsc) / double(std::max<std::uint64_t>(1, proto.pkts)), double(proto.out) / l, double(proto.nearS) / f,
                         double(proto.pairsS) / f);
        }
    }
};
Local &Loc() {
    (void)Tab();
    static thread_local Local loc;
    return loc;
}
// a FACE whose point is >= 1 mm inside every edge of its world triangle and whose |Sep| >= 1e-4 m
bool ClearFace(const Raw &r, const world::Instance &in, const exact::TriVerts &tv) {
    double p[3][3];
    for (int c = 0; c < 3; ++c) {
        const double v[3] = {tv.v[c].x, tv.v[c].y, tv.v[c].z};
        for (int a = 0; a < 3; ++a) p[c][a] = double(in.iso[a]) * v[0] + double(in.iso[3 + a]) * v[1] + double(in.iso[6 + a]) * v[2] + double(in.iso[9 + a]);
    }
    const double e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]}, e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(nl > 0.0)) return false;
    for (double &x : n) x /= nl;
    const double P[3] = {r.P.x, r.P.y, r.P.z};
    for (int i = 0; i < 3; ++i) {
        const double *a = p[i], *b = p[(i + 1) % 3];
        const double e[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        double u[3] = {e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]};
        const double ul = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if (!(ul > 0.0)) return false;
        const double d = (u[0] * (P[0] - a[0]) + u[1] * (P[1] - a[1]) + u[2] * (P[2] - a[2])) / ul;
        if (std::fabs(d) < 1e-3) return false;   // either side: the winding is FV's, not checked here
    }
    const double s = std::sqrt(double(r.Sep.x) * r.Sep.x + double(r.Sep.y) * r.Sep.y + double(r.Sep.z) * r.Sep.z);
    return s >= 1e-4;
}
}  // namespace steady
#endif

}  // namespace

const char CcCache::kKind = 0;

CcCache::CcCache() : step::CollisionCache(&kKind) {}

void CcCache::Flush() {
    owner = 0;
    haveLast = false;
    last.owner = 0;
}

std::size_t CcCache::Bytes() const {
    std::size_t b = m.tri.capacity() * sizeof(std::uint32_t) + m.ref.capacity() * sizeof(cq::Ref) + m.planes.capacity() * sizeof(exact::FastTri) +
                    m.fv.capacity() * sizeof(std::uint32_t) + m.ready.capacity() + m.near.capacity() * sizeof(std::uint32_t) +
                    (m.pairEntry.capacity() + m.pairShape.capacity()) * sizeof(std::uint32_t);
    for (int a = 0; a < 3; ++a) b += (m.lo[a].capacity() + m.hi[a].capacity() + m.nlo[a].capacity() + m.nhi[a].capacity()) * sizeof(float);
    for (int k = 0; k < 8; ++k) b += lists.idx[k].capacity() * sizeof(std::uint32_t);
    return b + sizeof(*this);
}

bool CcCollision::Init(const world::WorldPackView &view, const collide::CarShapes &car, std::string *error, const StageOptions &opt) {
    if (car.magic != collide::CarShapes::kMagic || car.count != 8u) {
        if (error) *error = "collide_c: car shapes missing or not 8";
        return false;
    }
    view_ = &view;
    car_ = car;
    opt_ = opt;
    bodyShapes_ = 0;
    for (int k = 0; k < 8; ++k) {   // the shapes' constants in lanes (PrepareLane)
        const collide::CarShape &sh = car.shape[k];
        const collide::Iso &local = sh.wheel >= 0 ? car.wheelRest[sh.wheel] : sh.bodyPose;
        for (int j = 0; j < 9; ++j) lc_.lr[j][k] = local.r[j];
        for (int a = 0; a < 3; ++a) {
            lc_.lt[a][k] = local.t[a];
            lc_.rad[a][k] = sh.radii[a];
            lc_.r2[a][k] = sh.radii[a] * sh.radii[a];
        }
        lc_.rmax[k] = std::max(sh.radii[0], std::max(sh.radii[1], sh.radii[2]));
        lc_.rmin[k] = std::min(sh.radii[0], std::min(sh.radii[1], sh.radii[2]));
        lc_.ratio[k] = lc_.rmax[k] / lc_.rmin[k];
        lc_.wheel[k] = sh.wheel;
        if (sh.wheel < 0) bodyShapes_ |= 1u << k;
    }
    for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 16; ++k) irad_[a][k] = 1.0f / car.shape[k & 7].radii[a];   // Narrow16: 1 / radius per shape
    static std::atomic<std::uint64_t> nextId{1};
    id_ = nextId.fetch_add(1) | (std::uint64_t(1) << 63);
    fast_.resize(view.modelTris.size());
    verts_.assign(view.modelTris.size() + 2u, exact::TriVerts{});   // + 2: Narrow16 reads a triangle's 9 floats as a 16-float row
    modelAbs_.assign(view.models.size(), 0.0f);
    for (std::size_t mi = 0; mi < view.models.size(); ++mi) {
        const world::Model &m = view.models[mi];
        float a = 0.0f;
        for (int k = 0; k < 3; ++k) a = math::max(a, math::max(std::fabs(m.lo[k]), std::fabs(m.hi[k])));
        modelAbs_[mi] = a;
        for (std::uint32_t t = 0; t < m.triCount; ++t) {
            const world::ModelTri &mt = view.modelTris[m.firstTri + t];
            exact::V vv[3];
            for (int c = 0; c < 3; ++c) { const world::Vec3f &p = view.modelVerts[m.firstVert + mt.v[c]]; vv[c] = exact::V{p.x, p.y, p.z}; }
            exact::MakeFastTri(vv, &fast_[m.firstTri + t]);
            for (int c = 0; c < 3; ++c) verts_[m.firstTri + t].v[c] = vv[c];
        }
    }
    World w;
    w.tris.assign(view.tris.begin(), view.tris.end());
    bvh_.Build(w);
    fvb_.Build(view);
    outRec_.resize(view.tris.size());
    for (std::size_t t = 0; t < view.tris.size(); ++t) {
        const Tri &tr = view.tris[t];
        const world::TriInfo &ti = view.triInfo[t];
        const world::Actor &a = view.ActorOf(tr);
        OutRec &r = outRec_[t];
        for (int i = 0; i < 3; ++i) r.peerZ[i] = a.peerZ[i];
        const bool trig = ti.trigger != world::kNoTrigger;
        r.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
        r.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
        r.flags = static_cast<std::uint8_t>(trig ? unsigned(model::kRespTrigger) : 0u);
        r.zero = r.pad = 0u;
    }
#if CC_GRIDSTATS && defined(__AVX512F__)
    if (CC_GRIDSTATS & 4) GridMem();
#endif
    return true;
}

void CcCollision::Detect(const model::TwinState &s, step::SubstepContacts &out) const { DetectPose(s.pos, s.quat, s.absorb, out); }

std::unique_ptr<step::CollisionCache> CcCollision::NewCache() const { return std::make_unique<CcCache>(); }

void CcCollision::DetectCached(const model::TwinState &s, step::CollisionCache *cache, step::SubstepContacts &out) const {
    DetectPose(s.pos, s.quat, s.absorb, out, cache);
}

void CcCollision::DetectPose(const float pos[3], const float quat[4], const float absorb[4], step::SubstepContacts &out,
                             step::CollisionCache *cache) const {
    thread_local CcCache threadCache;
    CcCache &cc = cache != nullptr && cache->kind == &CcCache::kKind ? *static_cast<CcCache *>(cache) : threadCache;
    if (cc.flushEveryCall) cc.Flush();
    float key[11];
    std::memcpy(key, pos, 3 * sizeof(float));
    std::memcpy(key + 3, quat, 4 * sizeof(float));
    std::memcpy(key + 7, absorb, 4 * sizeof(float));
    CcCache::Last &l = cc.last;
    if (opt_.lastCache && l.owner == id_ && std::memcmp(l.key, key, sizeof key) == 0) {   // same pose bits: same answer
        out.n = l.out.n;
        out.overflow = l.out.overflow;
        std::memcpy(out.c, l.out.c, sizeof(out.c[0]) * static_cast<std::size_t>(l.out.n));
        ++cc.lastHits;
        return;
    }
    collide::ShapePose shapes[8];
#if defined(__AVX512F__)
    if (LanesBatch()) {   // DetectLanes' pipeline for one call: lane = shape prepare, 16-lane packets
        struct One {
            collide::ShapePose shapes[8];
            float pad[4];   // Narrow16 reads each shape pose as a 16-float row
            Survivors sv;
            std::vector<Stage16> stage;
            std::vector<Raw> res;
            std::vector<std::uint16_t> hits;
        };
        thread_local One o1;
        PrepareLane(pos, quat, absorb, o1.shapes, cc, &o1.sv);
        const std::uint32_t ns = o1.sv.n;
        if (o1.stage.size() < ns + 16u) {
            o1.stage.resize(2 * ns + 32u);
            o1.res.resize(2 * ns + 32u);
            o1.hits.resize((2 * ns + 32u) / 16u + 1u);
        }
        const cq::Ref *ref = cc.m.ref.data();
        for (std::uint32_t i = 0; i < ns; ++i) StageOne(&o1.stage[i], &ref[o1.sv.e[i]], o1.sv.k[i]);
        const float *rows = reinterpret_cast<const float *>(o1.shapes);
        for (std::uint32_t p0 = 0; p0 < ns; p0 += 16u)
            Narrow16<15>(&o1.stage[p0], std::min<std::uint32_t>(16u, ns - p0), rows, view_->instances.begin(), verts_.data(), irad_, &o1.res[p0],
                         &o1.hits[p0 / 16u], nullptr);
        out.n = 0;
        out.overflow = false;
        MergeFlat(o1.res.data(), o1.hits.data(), 0u, ns, outRec_.data(), out);
    } else
#endif
#if defined(__AVX2__)
    if (LanesBatchable()) {   // lane = shape prepare (the same bits), the per-call narrow phase
        thread_local Scratch sc;
        PrepareLane(pos, quat, absorb, shapes, cc, nullptr);
        NoiseNarrowT(StageSrc{*view_, cc.m, verts_.data()}, shapes, cc.lists, sc, nullptr);
        out.n = 0;
        out.overflow = false;
        MergeOutLean(sc.rawn, outRec_.data(), out);
    } else
#endif
    {
        collide::ShapePoses(car_, pos, quat, absorb, shapes);
        DetectShapes(shapes, out, &cc);
    }
    if (opt_.lastCache) {
        l.owner = id_;
        std::memcpy(l.key, key, sizeof key);
        l.out.n = out.n;
        l.out.overflow = out.overflow;
        std::memcpy(l.out.c, out.c, sizeof(out.c[0]) * static_cast<std::size_t>(out.n));
    }
}

void CcCollision::DetectShapes(const collide::ShapePose shapes[8], step::SubstepContacts &out, CcCache *cache, StageStats *stats) const {
    thread_local CcCache threadCache;
    CcCache &cc = cache != nullptr ? *cache : threadCache;
    if (cc.flushEveryCall) cc.Flush();
    if (stats) ++stats->calls;
    Shapes(shapes, out, cc, stats);
}

bool CcCollision::LanesBatchable() const {
    return opt_.noise && !opt_.legacy && opt_.mergeMode == 0 && !opt_.skipBody && opt_.stopAfter == 0 && !opt_.bodyGate;
}

bool CcCollision::LanesBatch() const {
#if defined(__AVX512F__)
    return LanesBatchable() && opt_.narrow16;
#else
    return false;
#endif
}

void CcCollision::DetectLanes(int L, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                              step::SubstepContacts *out) const {
#if defined(__AVX512F__)
    if (L <= 16) {   // the poses as rows, then the lane = candidate path
        alignas(64) float pos[3][16] = {}, quat[4][16] = {}, absorb[4][16] = {};
        for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
            const int l = __builtin_ctz(mm);
            const model::TwinState &s = *poses[l];
            for (int a = 0; a < 3; ++a) pos[a][l] = s.pos[a];
            for (int a = 0; a < 4; ++a) quat[a][l] = s.quat[a];
            for (int w = 0; w < 4; ++w) absorb[w][l] = s.absorb[w];
        }
        if (DetectLanes16(pos, quat, absorb, mask, caches, out)) return;
    }
#endif
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        DetectCached(*poses[l], caches[l], out[l]);
    }
    (void)L;
}

#if defined(__AVX512F__)
namespace {
// DetectLanes16's per-thread scratch (one pass of 16 candidates).
struct alignas(64) Front16Scratch {
    float box[8][6][16];        // per shape: box lo xyz, hi xyz (lane = candidate)
    float wt[8][3][16];         // per shape: world centre
    float fq[8][6][16];         // per shape: q0..q5
    float t1[8][16];            // per shape: |t|_1
    float ulo[3][16], uhi[3][16];   // union of the 8 shape boxes
    float org[3][16], d1[16], tm[16], ma[16];   // per lane: its memo's origin, maxD1, maxT1, maxModelAbs
    float rows[16][8][16];      // per lane and shape: ShapePose (15 floats) + 0, Narrow16's 16-float rows
    ShapeTable st[16];          // per lane: Filter's shape table
    Box3 u[16];
    Survivors sv[2], proof;
    PassSurvivors pass;         // FilterPass16's survivors, lane after lane
    ShapeTable grown;           // Idea 5: a lane's shape table grown by its allowance
    std::vector<Stage16> stage;
    std::vector<Raw> res;
    std::vector<std::uint16_t> hits, face;   // per packet: its contacts, the FACEs among them
    std::uint32_t s0[16], s1[16];
    std::uint32_t nn[16];       // per lane: Filter's near entries (pass 0)
    bool sorted[16];            // per lane: its survivors in shape order (so are its contacts)
    std::uint32_t lastMask = 0u;   // the last DetectLanes16's lanes (ProveLanes16 reads its shape tables, unions and memos)
};
Front16Scratch &Front16Scratch_() {
    thread_local Front16Scratch fs;
    return fs;
}
static_assert(kTc0 == 0 && kTq0 == 3 && kTreach == 9 && kTlow == 10 && kTtol == 11 && kTlazy == 12, "DetectLanes16: ShapeTable::t field pairs");
static_assert(offsetof(ShapeTable, hi) == 24 * sizeof(float) && offsetof(ShapeTable, t) % 64 == 0, "DetectLanes16: ShapeTable rows");
}  // namespace
#if CC_GRIDSTATS
#include "cc_gridstats.inc"
#endif
#if CC_OSLABSTATS
#include "cc_oslabstats.inc"
#endif
#endif

float CcCollision::FreeLever() const {
    // per shape |local centre| + ||local rotation||_2 rmax, the spectral norm bounded by Gershgorin on A^T A (double)
    double lever = 0.0;
    for (int k = 0; k < 8; ++k) {
        double t2 = 0.0, lmax = 0.0;
        for (int a = 0; a < 3; ++a) t2 += double(lc_.lt[a][k]) * double(lc_.lt[a][k]);
        for (int i = 0; i < 3; ++i) {
            double row = 0.0;
            for (int j = 0; j < 3; ++j) {
                double g = 0.0;   // (A^T A)_ij, A's columns lr[3 i .. 3 i + 2]
                for (int a = 0; a < 3; ++a) g += double(lc_.lr[3 * i + a][k]) * double(lc_.lr[3 * j + a][k]);
                row += std::fabs(g);
            }
            lmax = std::max(lmax, row);
        }
        lever = std::max(lever, std::sqrt(t2) + std::sqrt(lmax) * double(lc_.rmax[k]));
    }
    return static_cast<float>(lever * 1.0001);
}

bool CcCollision::DetectLanes16(const float (*pos)[16], const float (*quat)[16], const float (*absorb)[16], std::uint32_t mask,
                                step::CollisionCache *const *caches, step::SubstepContacts *out, std::uint32_t *empty) const {
#if defined(__AVX512F__)
    if (!LanesBatch()) return false;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        if (caches[l] == nullptr || caches[l]->kind != &CcCache::kKind) return false;   // no lane cache: per lane
    }
    if (mask == 0u) {
        if (empty != nullptr) *empty = 0u;
        Front16Scratch_().lastMask = 0u;
        return true;
    }
    using Z = __m512;
    Front16Scratch &fs = Front16Scratch_();
    const LaneConst &c = lc_;
#if CC_STEADYSTATS
    std::uint64_t stT[8];
    int stP = 0;
    auto stLap = [&] { stT[stP++] = __rdtsc(); };
    stLap();
    alignas(64) float rcS[9][16];
#endif
    auto ld = [](const float *p) { return _mm512_load_ps(p); };
    auto bc = [](float v) { return _mm512_set1_ps(v); };
    auto add = [](Z a, Z b) { return _mm512_add_ps(a, b); };
    auto sub = [](Z a, Z b) { return _mm512_sub_ps(a, b); };
    auto mul = [](Z a, Z b) { return _mm512_mul_ps(a, b); };
    const Z sign = _mm512_set1_ps(-0.0f);
    auto absv = [&](Z a) { return _mm512_andnot_ps(sign, a); };

    // ---- front, part 1 (PrepareLane's ShapePoses, rows, ShapeBoxes and the memo-free filter terms; lane = candidate) ----
    // rot_from_quat per lane (vec3.h, the same operations)
    Z Rc[3][3];
    {
        const Z x = ld(quat[0]), y = ld(quat[1]), z = ld(quat[2]), w = ld(quat[3]);
        const Z two = bc(2.0f), one = bc(1.0f);
        const Z x2 = mul(x, two), y2 = mul(y, two), z2 = mul(z, two);
        const Z xw = mul(w, x2), yw = mul(w, y2), zw = mul(w, z2);
        const Z xx = mul(x, x2), xy = mul(x, y2), xz = mul(x, z2);
        const Z yy = mul(y, y2), yz = mul(y, z2), zz = mul(z, z2);
        Rc[0][0] = sub(sub(one, yy), zz); Rc[0][1] = add(xy, zw); Rc[0][2] = sub(xz, yw);
        Rc[1][0] = sub(xy, zw); Rc[1][1] = sub(sub(one, xx), zz); Rc[1][2] = add(yz, xw);
        Rc[2][0] = add(xz, yw); Rc[2][1] = sub(yz, xw); Rc[2][2] = sub(sub(one, xx), yy);
    }
#if CC_STEADYSTATS
    for (int i = 0; i < 9; ++i) _mm512_store_ps(rcS[i], Rc[i / 3][i % 3]);
#endif
    const Z px = ld(pos[0]), py = ld(pos[1]), pz = ld(pos[2]);
    for (int k = 0; k < 8; ++k) {
        // ShapePoses: SetMult(local, carIso) (TDir row a . v = (R.c0[a] v.x + R.c1[a] v.y) + R.c2[a] v.z)
        Z wr[9], wt[3];
        for (int i = 0; i < 3; ++i) {
            const Z vx = bc(c.lr[3 * i][k]), vy = bc(c.lr[3 * i + 1][k]), vz = bc(c.lr[3 * i + 2][k]);
            for (int a = 0; a < 3; ++a) wr[3 * i + a] = add(add(mul(Rc[0][a], vx), mul(Rc[1][a], vy)), mul(Rc[2][a], vz));
        }
        {
            const Z vx = bc(c.lt[0][k]), vz = bc(c.lt[2][k]);
            const Z vy = c.wheel[k] >= 0 ? add(bc(c.lt[1][k]), _mm512_xor_ps(ld(absorb[c.wheel[k]]), sign)) : bc(c.lt[1][k]);   // OffsetCurrentSurfaceY(-absorb)
            const Z p[3] = {px, py, pz};
            for (int a = 0; a < 3; ++a) wt[a] = add(add(add(mul(Rc[0][a], vx), mul(Rc[1][a], vy)), mul(Rc[2][a], vz)), p[a]);
        }
        const Z r0 = bc(c.rad[0][k]), r1 = bc(c.rad[1][k]), r2 = bc(c.rad[2][k]);
        {   // the rows (ShapePose: world r[9], t[3], radii[3]) for the narrow phase: lane l's row of shape k
            Z f[16] = {wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7], wr[8], wt[0], wt[1], wt[2], r0, r1, r2, _mm512_setzero_ps()};
            noise::Transpose16(f);
            for (int l = 0; l < 16; ++l) _mm512_store_ps(fs.rows[l][k], f[l]);
        }
        // ShapeFilters' terms that do not depend on the memo: Q = R diag(r^2) R^T
        const Z s0 = bc(c.r2[0][k]), s1 = bc(c.r2[1][k]), s2 = bc(c.r2[2][k]);
        auto q = [&](int x, int y) { return add(add(mul(s0, mul(wr[x], wr[y])), mul(s1, mul(wr[3 + x], wr[3 + y]))), mul(s2, mul(wr[6 + x], wr[6 + y]))); };
        const Z qd[3] = {q(0, 0), q(1, 1), q(2, 2)};
        // ShapeBoxes: the ellipsoid's exact AABB, h = sqrt(Q_aa), eps = 1e-3 + 1e-6 |c|, (c - h) - eps .. (c + h) + eps
        for (int a = 0; a < 3; ++a) {
            const Z h = _mm512_sqrt_ps(qd[a]);
            const Z eps = add(bc(1e-3f), mul(bc(1e-6f), absv(wt[a])));
            _mm512_store_ps(fs.box[k][a], sub(sub(wt[a], h), eps));
            _mm512_store_ps(fs.box[k][3 + a], add(add(wt[a], h), eps));
            _mm512_store_ps(fs.wt[k][a], wt[a]);
        }
        _mm512_store_ps(fs.fq[k][0], qd[0]);
        _mm512_store_ps(fs.fq[k][1], qd[1]);
        _mm512_store_ps(fs.fq[k][2], qd[2]);
        _mm512_store_ps(fs.fq[k][3], q(0, 1));
        _mm512_store_ps(fs.fq[k][4], q(0, 2));
        _mm512_store_ps(fs.fq[k][5], q(1, 2));
        _mm512_store_ps(fs.t1[k], add(add(absv(wt[0]), absv(wt[1])), absv(wt[2])));
    }
    // the union box: PrepareLane's horizontal min / max order, (k, k + 4), then (k, k + 2), then (0, 1)
    for (int a = 0; a < 6; ++a) {
        Z b[8];
        for (int k = 0; k < 8; ++k) b[k] = ld(fs.box[k][a]);
        auto mn = [&](Z x, Z y) { return a < 3 ? _mm512_min_ps(x, y) : _mm512_max_ps(x, y); };
        const Z u = mn(mn(mn(b[0], b[4]), mn(b[2], b[6])), mn(mn(b[1], b[5]), mn(b[3], b[7])));
        _mm512_store_ps(a < 3 ? fs.ulo[a] : fs.uhi[a - 3], u);
    }
    // the shape boxes into each lane's ShapeTable (lo[3][8], hi[3][8]): fields (lo x, lo y), (lo z, hi x), (hi y, hi z)
    for (int pr = 0; pr < 3; ++pr) {
        Z f[16];
        for (int k = 0; k < 8; ++k) {
            f[k] = ld(fs.box[k][2 * pr]);
            f[8 + k] = ld(fs.box[k][2 * pr + 1]);
        }
        noise::Transpose16(f);   // f[l]: lane l, field 2 pr in 0..7, field 2 pr + 1 in 8..15 (lo, hi: 48 contiguous floats)
        for (int l = 0; l < 16; ++l) _mm512_store_ps(&fs.st[l].lo[0][0] + 16 * pr, f[l]);
    }

#if CC_STEADYSTATS
    stLap();
#endif
    // ---- per lane: the memo (as PrepareLane) ----
    const world::WorldPackView &v = *view_;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        CcCache &cc = *static_cast<CcCache *>(caches[l]);
        if (cc.flushEveryCall) cc.Flush();
        Box3 &u = fs.u[l];
        for (int a = 0; a < 3; ++a) {
            u.lo[a] = fs.ulo[a][l];
            u.hi[a] = fs.uhi[a][l];
        }
        float center[3];
        for (int a = 0; a < 3; ++a) center[a] = 0.5f * (u.lo[a] + u.hi[a]);
        Memo &m = cc.m;
        if (cc.owner != id_ || !Contains(m.box, u)) {
            ++cc.refills;
            const bool same = cc.owner == id_ && cc.haveLast;
            cc.owner = id_;
            Box3 big = u;
            if (same)   // extend along the recent motion (only affects reuse, never the output)
                for (int a = 0; a < 3; ++a) {
                    const float d = math::max(-opt_.memoMaxLead, math::min(opt_.memoMaxLead, (center[a] - cc.lastCenter[a]) * opt_.leadPoses));
                    if (d > 0.0f) big.hi[a] += d; else big.lo[a] += d;
                }
            const Box3 box = Inflate(big, opt_.unionMargin);
            float origin[3];
            for (int a = 0; a < 3; ++a) origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
            Refill(v, bvh_, modelAbs_.data(), box, origin, &m, &fvb_, fast_.data());
        } else {
            ++cc.reuses;
        }
        for (int a = 0; a < 3; ++a) cc.lastCenter[a] = center[a];
        cc.haveLast = true;
        m.verts = nullptr;
        for (int a = 0; a < 3; ++a) fs.org[a][l] = m.origin[a];
        fs.d1[l] = m.maxD1;
        fs.tm[l] = m.maxT1;
        fs.ma[l] = m.maxModelAbs;
    }
#if CC_STEADYSTATS
    stLap();
#endif

    // ---- front, part 2: ShapeFilters' memo terms (as PrepareLane, lane = candidate), into the lanes' ShapeTables as
    // 64-byte rows of two fields (t[2 pr], t[2 pr + 1]) ----
    {
        const Z o0 = ld(fs.org[0]), o1 = ld(fs.org[1]), o2 = ld(fs.org[2]);
        const Z d1 = ld(fs.d1), tm = ld(fs.tm), ma3 = mul(bc(3.0f), ld(fs.ma));
        Z tol[8];
        for (int k = 0; k < 8; ++k) {
            const Z cv0 = sub(ld(fs.wt[k][0]), o0), cv1 = sub(ld(fs.wt[k][1]), o1), cv2 = sub(ld(fs.wt[k][2]), o2);
            const Z c1 = add(add(absv(cv0), absv(cv1)), absv(cv2));
            const Z inner = add(mul(bc(2.0f), add(c1, d1)), ma3);
            const Z rmax = bc(c.rmax[k]), ratio = bc(c.ratio[k]);
            tol[k] = add(add(mul(bc(1.0e-4f), rmax), mul(mul(bc(4.0e-6f), ratio), inner)), mul(bc(1.0e-6f), add(ld(fs.t1[k]), tm)));
        }
        auto rowsOut = [&](int pr, Z *f) {
            noise::Transpose16(f);
            for (int l = 0; l < 16; ++l) _mm512_store_ps(fs.st[l].t[2 * pr], f[l]);
        };
        Z f[16];
        for (int k = 0; k < 8; ++k) { f[k] = sub(ld(fs.wt[k][0]), o0); f[8 + k] = sub(ld(fs.wt[k][1]), o1); }
        rowsOut(0, f);   // c0, c1
        for (int k = 0; k < 8; ++k) { f[k] = sub(ld(fs.wt[k][2]), o2); f[8 + k] = ld(fs.fq[k][0]); }
        rowsOut(1, f);   // c2, q0
        for (int k = 0; k < 8; ++k) { f[k] = ld(fs.fq[k][1]); f[8 + k] = ld(fs.fq[k][2]); }
        rowsOut(2, f);   // q1, q2
        for (int k = 0; k < 8; ++k) { f[k] = ld(fs.fq[k][3]); f[8 + k] = ld(fs.fq[k][4]); }
        rowsOut(3, f);   // q3, q4
        for (int k = 0; k < 8; ++k) { f[k] = ld(fs.fq[k][5]); f[8 + k] = add(bc(c.rmax[k]), tol[k]); }
        rowsOut(4, f);   // q5, reach
        for (int k = 0; k < 8; ++k) { f[k] = _mm512_xor_ps(tol[k], sign); f[8 + k] = tol[k]; }
        rowsOut(5, f);   // low, tol
        for (int k = 0; k < 8; ++k) { f[k] = sub(bc(c.rmin[k]), tol[k]); f[8 + k] = _mm512_setzero_ps(); }
        rowsOut(6, f);   // lazy, spare
    }
#if CC_STEADYSTATS
    stLap();
#endif

    // ---- Filter for the pass's lanes at once (FilterPass16: pass 0 of every lane, then each lane's near blocks); a lane
    // whose memo is not all ready goes through Filter alone. Survivors staged for the shared packets (FV order within
    // each (lane, shape)) ----
    BodyGate gate;
    gate.shapes = bodyShapes_;
    std::vector<Stage16> &stg = fs.stage;
    std::uint32_t ns = 0;
    const exact::TriVerts *vt = verts_.data();
    PassSurvivors &ps = fs.pass;
    ps.n = 0;
    Memo *memos[16];
    std::uint32_t ready = 0u;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        memos[l] = &static_cast<CcCache *>(caches[l])->m;
        if (memos[l]->allReady) ready |= 1u << l;
    }
#if CC_STEADYSTATS
    std::uint64_t stNear0 = 0, stPairs0 = 0;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) { stNear0 += memos[__builtin_ctz(mm)]->nearEntries; stPairs0 += memos[__builtin_ctz(mm)]->pairs; }
#endif
#if CC_GRIDSTATS
    if (CC_GRIDSTATS & 2) FilterTime(memos, ready);
#endif
#if CC_OSLABSTATS
    if (CC_OSLABSTATS & 2) OslabTime(memos, ready);
#endif
    FilterPass16(memos, fs.st, fs.u, ready, bodyShapes_, &ps);
#if CC_STEADYSTATS
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) { steady::gNear += memos[__builtin_ctz(mm)]->nearEntries; steady::gPairs += memos[__builtin_ctz(mm)]->pairs; }
    steady::gNear -= stNear0;
    steady::gPairs -= stPairs0;
#endif
    for (std::uint32_t mm = mask & ~ready; mm != 0u; mm &= mm - 1u) {   // (not in the search: refills make every plane)
        const int l = __builtin_ctz(mm);
        CcCache &cc = *static_cast<CcCache *>(caches[l]);
        Survivors &sv = fs.sv[0];
        Filter(v, fast_.data(), cc.m, fs.st[l], fs.u[l], &cc.lists, gate, &sv);
        if (ps.e.size() < ps.n + sv.n + 16u) { ps.e.resize(2 * (ps.n + sv.n) + 32u); ps.k.resize(2 * (ps.n + sv.n) + 32u); }
        ps.s0[l] = ps.n;
        for (std::uint32_t j = 0; j < sv.n; ++j) { ps.e[ps.n] = sv.e[j]; ps.k[ps.n++] = sv.k[j]; }
        ps.s1[l] = ps.n;
        ps.nn[l] = 1u;   // unknown: no contact-free proof
    }
#if CC_STEADYSTATS
    stLap();
#endif
    if (stg.size() < ps.n + 16u) stg.resize(2 * ps.n + 32u);
    {
        Stage16 *sg = stg.data();
        const std::uint32_t *pe = ps.e.data(), *pk = ps.k.data();
        const OutRec *orec = outRec_.data();
        for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
            const int l = __builtin_ctz(mm);
            const cq::Ref *ref = memos[l]->ref.data();
            fs.s0[l] = ns;
            fs.nn[l] = ps.nn[l];
            std::uint32_t prevK = 0u;
            bool sorted = true;
            const std::uint32_t j1 = ps.s1[l], row0 = 8u * static_cast<std::uint32_t>(l);
            for (std::uint32_t j = ps.s0[l]; j < j1; ++j) {
                const std::uint32_t e = pe[j], k = pk[j];
                sorted &= k >= prevK;
                prevK = k;
                StageOne(&sg[ns++], &ref[e], row0 + k);
                _mm_prefetch(reinterpret_cast<const char *>(&orec[ref[e].local]), _MM_HINT_T0);   // MergeFlat reads it after the packets
            }
            fs.s1[l] = ns;
            fs.sorted[l] = sorted;
        }
    }
#if CC_STEADYSTATS
    stLap();
#endif
    // the narrow phase over all lanes' survivors, 16 per packet
    if (fs.res.size() < stg.size()) {
        fs.res.resize(stg.size());
        fs.hits.resize(stg.size() / 16u + 1u);
        fs.face.resize(stg.size() / 16u + 1u);
    }
    for (std::uint32_t p0 = 0; p0 < ns; p0 += 16u)
        Narrow16<16>(&stg[p0], std::min<std::uint32_t>(16u, ns - p0), &fs.rows[0][0][0], v.instances.begin(), vt, irad_, &fs.res[p0],
                     &fs.hits[p0 / 16u], nullptr, &fs.face[p0 / 16u]);
#if CC_STEADYSTATS
    stLap();
#endif
    // the merge: a lane whose survivors are in shape order and whose hits are all FACEs (primaries) keeps every contact in
    // order (MergeFlat's fast path, decided from the packets' masks); the others through MergeFlat
    const std::uint16_t *hits = fs.hits.data(), *face = fs.face.data();
    const Raw *res = fs.res.data();
    const OutRec *outRec = outRec_.data();
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        step::SubstepContacts &o = out[l];
        o.n = 0;
        o.overflow = false;
        const std::uint32_t s0 = fs.s0[l], s1 = fs.s1[l];
        if (s1 <= s0) continue;
        if (fs.sorted[l]) {
            const std::uint32_t w0 = s0 >> 4, w1 = (s1 - 1u) >> 4;
            std::uint32_t feat = 0u, cnt = 0u;
            for (std::uint32_t w = w0; w <= w1; ++w) {
                std::uint32_t m = hits[w];
                if (w == w0) m &= 0xFFFFu << (s0 & 15u);
                if (w == w1) m &= 0xFFFFu >> (15u - ((s1 - 1u) & 15u));
                feat |= m & ~static_cast<std::uint32_t>(face[w]);
                cnt += static_cast<std::uint32_t>(__builtin_popcount(m));
            }
            if (feat == 0u && cnt <= static_cast<std::uint32_t>(model::kRespMaxContacts)) {
                int n = 0;
                for (std::uint32_t w = w0; w <= w1; ++w) {
                    std::uint32_t m = hits[w];
                    if (w == w0) m &= 0xFFFFu << (s0 & 15u);
                    if (w == w1) m &= 0xFFFFu >> (15u - ((s1 - 1u) & 15u));
                    for (; m; m &= m - 1u) {
                        const Raw &r = res[16u * w + static_cast<std::uint32_t>(__builtin_ctz(m))];
                        FillOut16(o.c[n++], r, outRec[r.tri], r.shape);
                    }
                }
                o.n = n;
                continue;
            }
        }
        MergeFlat(res, hits, s0, s1, outRec, o);
    }
#if CC_STEADYSTATS
    stLap();
    SteadyCount(mask, caches, out, stT, rcS);
#endif
#if CC_GRIDSTATS
    if (CC_GRIDSTATS & 1) GridCount(mask, caches);
#endif
#if CC_OSLABSTATS
    if (CC_OSLABSTATS & 1) OslabCount(mask, caches);
#endif
    if (empty != nullptr) {   // Idea 5: the lanes without a candidate and without a near memo entry (ProveLanes16)
        *empty = 0u;
        for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
            const int l = __builtin_ctz(mm);
            if (fs.s1[l] == fs.s0[l] && fs.nn[l] == 0u) *empty |= 1u << l;
        }
    }
    fs.lastMask = mask;
    return true;
#else
    (void)pos; (void)quat; (void)absorb; (void)mask; (void)caches; (void)out; (void)empty;
    return false;
#endif
}

#if CC_STEADYSTATS && defined(__AVX512F__)
void CcCollision::SteadyCount(std::uint32_t mask, step::CollisionCache *const *caches, const step::SubstepContacts *out, const std::uint64_t *t,
                              const float (*rc)[16]) const {
    steady::Local &loc = steady::Loc();
    steady::Row &row = loc.Get(fvb_.Size());
    Front16Scratch &fs = Front16Scratch_();
    const PassSurvivors &ps = fs.pass;
    ++row.passes;
    row.lanes += static_cast<std::uint64_t>(__builtin_popcount(mask));
    for (int p = 0; p < steady::kPh; ++p) row.tsc[p] += t[p + 1] - t[p];
    row.near += steady::gNear;
    row.pairs += steady::gPairs;
    steady::gNear = steady::gPairs = 0;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        row.surv += fs.s1[l] - fs.s0[l];
    }
    if (!(CC_STEADYSTATS & 1)) return;
    // per shape: ||local rotation||_2 (Gershgorin on A^T A, as FreeLever) x |radii|_2: a bound on how far a point of the
    // shape's box corners or surface moves per unit of ||car rotation change||_2 (<= its Frobenius norm)
    float lever[8], planeLever[8];
    for (int k = 0; k < 8; ++k) {
        double lmax = 0.0;
        for (int i = 0; i < 3; ++i) {
            double rowSum = 0.0;
            for (int j = 0; j < 3; ++j) {
                double g = 0.0;
                for (int a = 0; a < 3; ++a) g += double(lc_.lr[3 * i + a][k]) * double(lc_.lr[3 * j + a][k]);
                rowSum += std::fabs(g);
            }
            lmax = std::max(lmax, rowSum);
        }
        double r2 = 0.0;
        for (int a = 0; a < 3; ++a) r2 += double(lc_.rad[a][k]) * double(lc_.rad[a][k]);
        lever[k] = static_cast<float>(std::sqrt(lmax) * std::sqrt(r2) * 1.0001);
        planeLever[k] = static_cast<float>(std::sqrt(lmax) * double(lc_.rmax[k]) * 1.0001);
    }
    static thread_local bool printed = false;
    if (!printed) {
        printed = true;
        for (int k = 0; k < 8; ++k)
            std::fprintf(stderr, "steady shape %d: wheel %d radii %.4f %.4f %.4f lever %.4f plane lever %.4f\n", k, lc_.wheel[k], double(lc_.rad[0][k]),
                         double(lc_.rad[1][k]), double(lc_.rad[2][k]), double(lever[k]), double(planeLever[k]));
    }
    const world::WorldPackView &v = *view_;
    const Stage16 *stg = fs.stage.data();
    BodyGate gate;
    gate.shapes = bodyShapes_;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        CcCache &cc = *static_cast<CcCache *>(caches[l]);
        steady::Side &sd = loc.side[&cc];
        if (sd.stage != id_) { sd = steady::Side{}; sd.stage = id_; }
        bool wheelC = false;
        for (int i = 0; i < out[l].n; ++i) wheelC |= lc_.wheel[out[l].c[i].shape] >= 0;
        steady::Cls &C = row.c[out[l].n == 0 ? 0 : wheelC ? 1 : 2];
        ++C.calls;
        std::vector<std::uint64_t> &ek = loc.exactKeys, &rk = loc.rawKeys, &keys = loc.keys;
        ek.clear();
        const cq::Ref *ref = cc.m.ref.data();
        for (std::uint32_t j = ps.s0[l]; j < ps.s1[l]; ++j) ek.push_back(std::uint64_t(ref[ps.e[j]].local) << 3 | ps.k[j]);
        C.exact += ek.size();
        rk.clear();
        bool allFace = true, allClear = true;
        for (std::uint32_t s = fs.s0[l]; s < fs.s1[l]; ++s) {
            if (!(fs.hits[s >> 4] >> (s & 15u) & 1u)) continue;
            const Raw &r = fs.res[s];
            const bool face = (fs.face[s >> 4] >> (s & 15u)) & 1u;
            rk.push_back(std::uint64_t(r.tri) << 4 | std::uint64_t(r.shape) << 1 | (face ? 1u : 0u));
            if (!face) allFace = allClear = false;
            else if (allClear && !steady::ClearFace(r, v.instances[stg[s].instance], verts_[stg[s].modelTri])) allClear = false;
        }
        std::sort(rk.begin(), rk.end());
        C.raw += rk.size();
        C.merged += static_cast<std::uint64_t>(out[l].n);
        if (sd.havePrev && rk == sd.prevRaw) {
            ++C.same;
            if (allFace) ++C.sameFace;
            if (allClear) ++C.oracle;
        }
        sd.prevRaw = rk;
        sd.havePrev = true;
        float wt[8][3], rcl[9], tol[8];
        for (int k = 0; k < 8; ++k) {
            for (int a = 0; a < 3; ++a) wt[k][a] = fs.wt[k][a][l];
            tol[k] = fs.st[l].t[kTtol][k];
        }
        for (int i = 0; i < 9; ++i) rcl[i] = rc[i][l];
        for (int i = 0; i < steady::kM; ++i) {
            steady::Shadow &sh = sd.s[i];
            const float mg = steady::kMargin[i];
            bool serv = sh.valid;
            if (serv) {
                float fr = 0.0f;
                for (int q = 0; q < 9; ++q) { const float d = rcl[q] - sh.rc[q]; fr += d * d; }
                fr = std::sqrt(fr);
                const float lim = 0.995f * mg - 2e-4f;
                for (int k = 0; k < 8 && serv; ++k) {
                    const float dx = wt[k][0] - sh.wt[k][0], dy = wt[k][1] - sh.wt[k][1], dz = wt[k][2] - sh.wt[k][2];
                    const float d = std::sqrt(dx * dx + dy * dy + dz * dz) + fr * lever[k] + std::max(0.0f, tol[k] - sh.tol[k]);
                    if (!(d <= lim)) serv = false;
                }
            }
            if (serv) {
                ++C.serv[i];
                C.servList[i] += sh.keys.size();
                C.servExact[i] += ek.size();
                for (std::uint64_t key : ek) C.violS[i] += !std::binary_search(sh.keys.begin(), sh.keys.end(), key);
                for (std::uint64_t key : rk) C.violC[i] += !std::binary_search(sh.keys.begin(), sh.keys.end(), (key >> 4) << 3 | ((key >> 1) & 7u));
                continue;
            }
            // a full call: the list for this pose, every shape grown by the margin
            ++C.fulls[i];
            const Box3 U = Inflate(fs.u[l], mg);
            Memo *mp = &cc.m;
            ShapeTable g = fs.st[l];
            if (!Contains(cc.m.box, U)) {   // a design would refill here: a temporary memo around the grown union
                ++C.memoMiss[i];
                const Box3 box = Inflate(U, opt_.unionMargin);
                float origin[3];
                for (int a = 0; a < 3; ++a) origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
                Refill(v, bvh_, modelAbs_.data(), box, origin, &loc.tmp, &fvb_, fast_.data());
                mp = &loc.tmp;
                collide::ShapePose sp[8];
                for (int k = 0; k < 8; ++k) std::memcpy(&sp[k], fs.rows[l][k], sizeof(collide::ShapePose));
                Box3 uu;
                ShapeBoxes(sp, &g, &uu);
                ShapeFilters(loc.tmp, sp, &g);
            }
            for (int k = 0; k < 8; ++k) {
                sh.tol[k] = g.t[kTtol][k];
                for (int a = 0; a < 3; ++a) {
                    g.lo[a][k] -= mg;
                    g.hi[a][k] += mg;
                }
                g.t[kTreach][k] += mg;
                g.t[kTlow][k] -= mg;
                g.t[kTtol][k] += mg;
                g.t[kTlazy][k] -= mg;
            }
            Filter(v, fast_.data(), *mp, g, U, &loc.lists, gate, &loc.sink);
            keys.clear();
            const cq::Ref *mref = mp->ref.data();
            for (std::uint32_t j = 0; j < loc.sink.n; ++j) keys.push_back(std::uint64_t(mref[loc.sink.e[j]].local) << 3 | loc.sink.k[j]);
            std::sort(keys.begin(), keys.end());
            sh.keys = keys;
            sh.valid = true;
            std::memcpy(sh.wt, wt, sizeof wt);
            std::memcpy(sh.rc, rcl, sizeof rcl);
            C.fullList[i] += keys.size();
            C.fullExact[i] += ek.size();
            for (std::uint64_t key : ek) C.violFull[i] += !std::binary_search(keys.begin(), keys.end(), key);
        }
        // swept variants
        float dlt[8][3];
        bool haveD = sd.havePrevWt;
        for (int k = 0; k < 8 && haveD; ++k) {
            for (int a = 0; a < 3; ++a) dlt[k][a] = wt[k][a] - sd.prevWt[k][a];
            haveD = dlt[k][0] * dlt[k][0] + dlt[k][1] * dlt[k][1] + dlt[k][2] * dlt[k][2] <= 0.35f * 0.35f;   // else another candidate
        }
        if (!haveD) std::memset(dlt, 0, sizeof dlt);
        std::memcpy(sd.prevWt, wt, sizeof wt);
        sd.havePrevWt = true;
        for (int iv = 0; iv < steady::kS; ++iv) {
            const int i = steady::kM + iv;
            steady::Shadow &sh = sd.s[i];
            const float D = steady::kSweepD[iv];
            bool serv = sh.valid;
            if (serv) {
                float fr = 0.0f;
                for (int q = 0; q < 9; ++q) { const float d = rcl[q] - sh.rc[q]; fr += d * d; }
                fr = std::sqrt(fr);
                const float lim = 0.995f * D - 2e-4f;
                for (int k = 0; k < 8 && serv; ++k) {
                    const float p[3] = {wt[k][0] - sh.wt[k][0], wt[k][1] - sh.wt[k][1], wt[k][2] - sh.wt[k][2]};
                    const float *w = sh.w[k];
                    const float L2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                    float tt = L2 > 0.0f ? (p[0] * w[0] + p[1] * w[1] + p[2] * w[2]) / L2 : 0.0f;
                    tt = std::min(1.0f, std::max(0.0f, tt));
                    const float e[3] = {p[0] - tt * w[0], p[1] - tt * w[1], p[2] - tt * w[2]};
                    const float d = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) + fr * lever[k] + std::max(0.0f, tol[k] - sh.tol[k]);
                    if (!(d <= lim)) serv = false;
                }
            }
            if (serv) {
                ++C.serv[i];
                C.servList[i] += sh.keys.size();
                C.servExact[i] += ek.size();
                for (std::uint64_t key : ek) C.violS[i] += !std::binary_search(sh.keys.begin(), sh.keys.end(), key);
                for (std::uint64_t key : rk) C.violC[i] += !std::binary_search(sh.keys.begin(), sh.keys.end(), (key >> 4) << 3 | ((key >> 1) & 7u));
                if (iv == steady::kRF) {
                    const std::uint32_t np = static_cast<std::uint32_t>(sh.pk.size());
                    if (loc.oe.size() < 16u * np + 16u) { loc.oe.resize(16u * np + 16u); loc.ok.resize(16u * np + 16u); }
                    const std::uint64_t r0 = __rdtsc();
                    const std::uint32_t no = steady::Refilter(sh.pk.data(), np, fs.st[l], loc.oe.data(), loc.ok.data());
                    const std::uint64_t r1 = __rdtsc();
                    const std::uint64_t r2 = __rdtsc();
                    loc.proto.tsc += (r1 - r0) - std::min(r1 - r0, r2 - r1);
                    ++loc.proto.lanes;
                    loc.proto.pkts += np;
                    loc.proto.out += no;
                }
                continue;
            }
            ++C.fulls[i];
            const float K = float(steady::kSweepK[iv]);
            float w[8][3];
            for (int k = 0; k < 8; ++k)
                for (int a = 0; a < 3; ++a) w[k][a] = K * dlt[k][a];
            // the shape boxes swept and grown, and their union (the box test's eps grows by ~1e-6 of the sweep: in D's margin)
            float blo[3][8], bhi[3][8];
            Box3 U = EmptyBox();
            for (int k = 0; k < 8; ++k)
                for (int a = 0; a < 3; ++a) {
                    const float lo0 = fs.st[l].lo[a][k], hi0 = fs.st[l].hi[a][k];
                    blo[a][k] = std::min(lo0, lo0 + w[k][a]) - D;
                    bhi[a][k] = std::max(hi0, hi0 + w[k][a]) + D;
                    U.lo[a] = std::min(U.lo[a], blo[a][k]);
                    U.hi[a] = std::max(U.hi[a], bhi[a][k]);
                }
            Memo *mp = &cc.m;
            ShapeTable g = fs.st[l];
            if (!Contains(cc.m.box, U)) {
                ++C.memoMiss[i];
                const Box3 box = Inflate(U, opt_.unionMargin);
                float origin[3];
                for (int a = 0; a < 3; ++a) origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
                Refill(v, bvh_, modelAbs_.data(), box, origin, &loc.tmp, &fvb_, fast_.data());
                mp = &loc.tmp;
                collide::ShapePose sp[8];
                for (int k = 0; k < 8; ++k) std::memcpy(&sp[k], fs.rows[l][k], sizeof(collide::ShapePose));
                Box3 uu;
                ShapeBoxes(sp, &g, &uu);
                ShapeFilters(loc.tmp, sp, &g);
            }
            // every memo entry against every swept shape (scalar; Pass's tests at both segment ends, the plane distance
            // being linear along the segment)
            keys.clear();
            loc.pe.clear();
            loc.pk2.clear();
            const Memo &M = *mp;
            for (std::uint32_t e = 0; e < M.n; ++e) {
                const exact::FastTri &pl = M.planes[e];
                if (iv == steady::kRF) {
                    bool inU = true;
                    for (int a = 0; a < 3 && inU; ++a) inU = M.lo[a][e] <= U.hi[a] && U.lo[a] <= M.hi[a][e];
                    loc.proto.nearS += inU;
                }
                for (int k = 0; k < 8; ++k) {
                    bool in = true;
                    for (int a = 0; a < 3 && in; ++a) in = M.lo[a][e] <= bhi[a][k] && blo[a][k] <= M.hi[a][e];
                    if (!in) continue;
                    if (iv == steady::kRF) ++loc.proto.pairsS;
                    const float c0 = g.t[kTc0][k], c1 = g.t[kTc0 + 1][k], c2 = g.t[kTc0 + 2][k];
                    const float *q = &g.t[kTq0][0];
                    const float q0 = q[k], q1 = q[8 + k], q2 = q[16 + k], q3 = q[24 + k], q4 = q[32 + k], q5 = q[40 + k];
                    const float reach = g.t[kTreach][k] + D, low = g.t[kTlow][k] - D, tl = g.t[kTtol][k] + D, lazy = g.t[kTlazy][k] - D;
                    for (int j = 0; j < 4 && in; ++j) {
                        const float wx = pl.x[j], wy = pl.y[j], wz = pl.z[j];
                        const float d0 = ((wx * c0 + wy * c1) + wz * c2) - pl.d[j];
                        const float d1 = d0 + ((wx * w[k][0] + wy * w[k][1]) + wz * w[k][2]);
                        const float lo = std::min(d0, d1), hi = std::max(d0, d1);
                        if ((j == 0 && hi < low) || lo > reach) { in = false; break; }
                        if (lo > lazy) {
                            const float aa = (q0 * wx + q3 * wy) + q4 * wz;
                            const float bb = (q3 * wx + q1 * wy) + q5 * wz;
                            const float cc2 = (q4 * wx + q5 * wy) + q2 * wz;
                            const float h2 = (aa * wx + bb * wy) + cc2 * wz;
                            const float x = lo - tl;
                            if (x > 0.0f && x * x > h2 * exact::kSupportSlack * 1.0001f) in = false;
                        }
                    }
                    if (in) {
                        keys.push_back(std::uint64_t(M.ref[e].local) << 3 | static_cast<std::uint32_t>(k));
                        if (iv == steady::kRF) { loc.pe.push_back(e); loc.pk2.push_back(static_cast<std::uint32_t>(k)); }
                    }
                }
            }
            if (iv == steady::kRF) {   // the prototype's packets (pair order: entry-major, FV order within each shape)
                ++loc.proto.fulls;
                const std::size_t n = loc.pe.size();
                sh.pk.assign((n + 15u) / 16u, steady::LPkt{});
                for (std::size_t j = 0; j < n; ++j) {
                    steady::LPkt &P = sh.pk[j / 16u];
                    const std::size_t ln = j % 16u;
                    const std::uint32_t e = loc.pe[j];
                    const float *row = reinterpret_cast<const float *>(&M.planes[e]);
                    for (int f = 0; f < 16; ++f) P.pl[f][ln] = row[f];
                    for (int a = 0; a < 3; ++a) { P.lo[a][ln] = M.lo[a][e]; P.hi[a][ln] = M.hi[a][e]; }
                    P.k[ln] = static_cast<std::int32_t>(loc.pk2[j]);
                    P.e[ln] = e;
                    P.n = static_cast<std::uint32_t>(ln + 1u);
                }
            }
            std::sort(keys.begin(), keys.end());
            sh.keys = keys;
            sh.valid = true;
            std::memcpy(sh.wt, wt, sizeof wt);
            std::memcpy(sh.rc, rcl, sizeof rcl);
            std::memcpy(sh.w, w, sizeof w);
            for (int k = 0; k < 8; ++k) sh.tol[k] = g.t[kTtol][k];
            C.fullList[i] += keys.size();
            C.fullExact[i] += ek.size();
            for (std::uint64_t key : ek) C.violFull[i] += !std::binary_search(keys.begin(), keys.end(), key);
        }
    }
}
#endif

void CcCollision::PrepareLane(const float pos[3], const float quat[4], const float absorb[4], collide::ShapePose rows[8], CcCache &cc,
                              Survivors *sink) const {
#if defined(__AVX2__)
    using Y = __m256;
    const LaneConst &c = lc_;
    auto ld = [](const float *p) { return _mm256_load_ps(p); };
    auto bc = [](float v) { return _mm256_set1_ps(v); };
    auto add = [](Y a, Y b) { return _mm256_add_ps(a, b); };
    auto sub = [](Y a, Y b) { return _mm256_sub_ps(a, b); };
    auto mul = [](Y a, Y b) { return _mm256_mul_ps(a, b); };
    const Y sign = _mm256_set1_ps(-0.0f);
    auto absv = [&](Y a) { return _mm256_andnot_ps(sign, a); };
    // ShapePoses: SetMult(local, carIso) per lane (tw_collide.cpp; TDir row a . v = (R.c0[a] v.x + R.c1[a] v.y) + R.c2[a] v.z)
    const model::M3 R = model::rot_from_quat(quat);
    const float Rc[3][3] = {{R.c0.x, R.c0.y, R.c0.z}, {R.c1.x, R.c1.y, R.c1.z}, {R.c2.x, R.c2.y, R.c2.z}};
    alignas(32) float lty[8];
    for (int k = 0; k < 8; ++k) lty[k] = c.wheel[k] >= 0 ? c.lt[1][k] + -absorb[c.wheel[k]] : c.lt[1][k];   // OffsetCurrentSurfaceY(-absorb)
    Y wr[9], wt[3];
    for (int i = 0; i < 3; ++i) {
        const Y vx = ld(c.lr[3 * i]), vy = ld(c.lr[3 * i + 1]), vz = ld(c.lr[3 * i + 2]);
        for (int a = 0; a < 3; ++a) wr[3 * i + a] = add(add(mul(bc(Rc[0][a]), vx), mul(bc(Rc[1][a]), vy)), mul(bc(Rc[2][a]), vz));
    }
    {
        const Y vx = ld(c.lt[0]), vy = ld(lty), vz = ld(c.lt[2]);
        for (int a = 0; a < 3; ++a) wt[a] = add(add(add(mul(bc(Rc[0][a]), vx), mul(bc(Rc[1][a]), vy)), mul(bc(Rc[2][a]), vz)), bc(pos[a]));
    }
    const Y r0 = ld(c.rad[0]), r1 = ld(c.rad[1]), r2 = ld(c.rad[2]);
    // the rows (ShapePose: world r[9], t[3], radii[3]) for the narrow phase: an 8 x 16 transpose, row k = shape k
    {
        const Y fl[16] = {wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7], wr[8], wt[0], wt[1], wt[2], r0, r1, r2, _mm256_setzero_ps()};
#if defined(__AVX512F__)
        __m512 f[16];
        for (int i = 0; i < 8; ++i) f[i] = _mm512_insertf32x8(_mm512_castps256_ps512(fl[i]), fl[i + 8], 1);   // lanes 0-7: field i, 8-15: field i+8
        for (int i = 8; i < 16; ++i) f[i] = _mm512_setzero_ps();
        noise::Transpose16(f);   // now f[k] (k < 8): fields 0..7 of shape k in lanes 0..7; f[k + 8]: its fields 8..15
        for (int k = 0; k < 8; ++k)   // fields 0-7 from f[k], 8-14 from f[k + 8]: one masked store of 15 floats
            _mm512_mask_storeu_ps(&rows[k], static_cast<__mmask16>(0x7FFF), _mm512_shuffle_f32x4(f[k], f[k + 8], _MM_SHUFFLE(1, 0, 1, 0)));
#else
        alignas(32) float tmp[16][8];
        for (int i = 0; i < 16; ++i) _mm256_store_ps(tmp[i], fl[i]);
        for (int k = 0; k < 8; ++k) {
            float *o = reinterpret_cast<float *>(&rows[k]);
            for (int i = 0; i < 15; ++i) o[i] = tmp[i][k];
        }
#endif
    }
    // ShapeBoxes: the ellipsoid's exact AABB, h = sqrt(Q_aa) (ShapeFilters' Q), eps = 1e-3 + 1e-6 |c|, (c - h) - eps .. (c + h) + eps
    ShapeTable st;
    Y ulo[3], uhi[3];
    for (int a = 0; a < 3; ++a) {
        const Y qs0 = ld(c.r2[0]), qs1 = ld(c.r2[1]), qs2 = ld(c.r2[2]);
        const Y h = _mm256_sqrt_ps(add(add(mul(qs0, mul(wr[a], wr[a])), mul(qs1, mul(wr[3 + a], wr[3 + a]))), mul(qs2, mul(wr[6 + a], wr[6 + a]))));
        const Y eps = add(bc(1e-3f), mul(bc(1e-6f), absv(wt[a])));
        const Y lo = sub(sub(wt[a], h), eps), hi = add(add(wt[a], h), eps);
        _mm256_store_ps(st.lo[a], lo);
        _mm256_store_ps(st.hi[a], hi);
        ulo[a] = lo;
        uhi[a] = hi;
    }
    Box3 u;
    for (int a = 0; a < 3; ++a) {   // horizontal min / max (exact)
        __m128 l = _mm_min_ps(_mm256_castps256_ps128(ulo[a]), _mm256_extractf128_ps(ulo[a], 1));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(uhi[a]), _mm256_extractf128_ps(uhi[a], 1));
        l = _mm_min_ps(l, _mm_movehl_ps(l, l));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        l = _mm_min_ss(l, _mm_shuffle_ps(l, l, 1));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        u.lo[a] = _mm_cvtss_f32(l);
        u.hi[a] = _mm_cvtss_f32(h);
    }
    // the memo (as Prepare)
    const world::WorldPackView &v = *view_;
    float center[3];
    for (int a = 0; a < 3; ++a) center[a] = 0.5f * (u.lo[a] + u.hi[a]);
    Memo &m = cc.m;
    if (cc.owner != id_ || !Contains(m.box, u)) {
        ++cc.refills;
        const bool same = cc.owner == id_ && cc.haveLast;
        cc.owner = id_;
        Box3 big = u;
        if (same)   // extend along the recent motion (only affects reuse, never the output)
            for (int a = 0; a < 3; ++a) {
                const float d = math::max(-opt_.memoMaxLead, math::min(opt_.memoMaxLead, (center[a] - cc.lastCenter[a]) * opt_.leadPoses));
                if (d > 0.0f) big.hi[a] += d; else big.lo[a] += d;
            }
        const Box3 box = Inflate(big, opt_.unionMargin);
        float origin[3];
        for (int a = 0; a < 3; ++a) origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
        Refill(v, bvh_, modelAbs_.data(), box, origin, &m, &fvb_, opt_.bodyGate ? nullptr : fast_.data());
    } else {
        ++cc.reuses;
    }
    for (int a = 0; a < 3; ++a) cc.lastCenter[a] = center[a];
    cc.haveLast = true;
    // ShapeFilters (cc_memo.cpp, the same formula per lane); rows stored whole (Filter loads them as 8 lanes:
    // a load spanning two smaller stores would stall on store forwarding)
    auto StT = [](float *row, Y v) { _mm256_store_ps(row, v); };
    {
        Y cv[3];
        for (int a = 0; a < 3; ++a) {
            cv[a] = sub(wt[a], bc(m.origin[a]));
            StT(st.t[kTc0 + a], cv[a]);
        }
        const Y s0 = ld(c.r2[0]), s1 = ld(c.r2[1]), s2 = ld(c.r2[2]);
        auto q = [&](int x, int y) { return add(add(mul(s0, mul(wr[x], wr[y])), mul(s1, mul(wr[3 + x], wr[3 + y]))), mul(s2, mul(wr[6 + x], wr[6 + y]))); };
        StT(st.t[kTq0 + 0], q(0, 0));
        StT(st.t[kTq0 + 1], q(1, 1));
        StT(st.t[kTq0 + 2], q(2, 2));
        StT(st.t[kTq0 + 3], q(0, 1));
        StT(st.t[kTq0 + 4], q(0, 2));
        StT(st.t[kTq0 + 5], q(1, 2));
        const Y c1 = add(add(absv(cv[0]), absv(cv[1])), absv(cv[2]));
        const Y t1 = add(add(absv(wt[0]), absv(wt[1])), absv(wt[2]));
        const Y inner = add(mul(bc(2.0f), add(c1, bc(m.maxD1))), mul(bc(3.0f), bc(m.maxModelAbs)));
        const Y rmax = ld(c.rmax), rmin = ld(c.rmin), ratio = ld(c.ratio);
        const Y tol = add(add(mul(bc(1.0e-4f), rmax), mul(mul(bc(4.0e-6f), ratio), inner)), mul(bc(1.0e-6f), add(t1, bc(m.maxT1))));
        StT(st.t[kTreach], add(rmax, tol));
        StT(st.t[kTlow], _mm256_xor_ps(tol, sign));
        StT(st.t[kTtol], tol);
        StT(st.t[kTlazy], sub(rmin, tol));
    }
    BodyGate gate;
    for (int k = 0; k < 8; ++k) if (c.wheel[k] < 0) gate.shapes |= 1u << k;
    m.verts = nullptr;
    Filter(v, fast_.data(), m, st, u, &cc.lists, gate, sink);
#else
    collide::ShapePoses(car_, pos, quat, absorb, rows);
    std::uint64_t t0 = 0;
    Prepare(rows, cc, nullptr, &t0, sink);
#endif
}

bool CcCollision::Prepare(const collide::ShapePose shapes[8], CcCache &cc, StageStats *stats, std::uint64_t *tp, Survivors *sink) const {
    const world::WorldPackView &v = *view_;
    std::uint64_t t0 = stats ? *tp : 0;
    std::uint64_t dummy = 0;
    auto lap = [&](std::uint64_t *acc) { if (stats) { const std::uint64_t t1 = __rdtsc(); *acc += t1 - t0; t0 = t1; *tp = t0; } };
    // the shapes' query boxes (B's: closed, inflated for rounding) and their union
    ShapeTable st;
    Box3 u;
    ShapeBoxes(shapes, &st, &u);
    if (opt_.skipBody) {   // relaxation (measurement ceiling): the body shapes take no part (empty boxes, union of the wheels)
        u = EmptyBox();
        for (int k = 0; k < 8; ++k) {
            const bool body = car_.shape[k].wheel < 0;
            for (int a = 0; a < 3; ++a) {
                if (body) { st.lo[a][k] = 1e30f; st.hi[a][k] = -1e30f; continue; }
                u.lo[a] = std::min(u.lo[a], st.lo[a][k]);
                u.hi[a] = std::max(u.hi[a], st.hi[a][k]);
            }
        }
    }
    lap(stats ? &stats->tBoxes : &dummy);
    float center[3];
    for (int a = 0; a < 3; ++a) center[a] = 0.5f * (u.lo[a] + u.hi[a]);
    Memo &m = cc.m;
    if (cc.owner != id_ || !Contains(m.box, u)) {
        ++cc.refills;
        if (stats) ++stats->refills;
        const bool same = cc.owner == id_ && cc.haveLast;
        cc.owner = id_;
        Box3 big = u;
        if (same)   // extend along the recent motion (only affects reuse, never the output)
            for (int a = 0; a < 3; ++a) {
                const float d = math::max(-opt_.memoMaxLead, math::min(opt_.memoMaxLead, (center[a] - cc.lastCenter[a]) * opt_.leadPoses));
                if (d > 0.0f) big.hi[a] += d; else big.lo[a] += d;
            }
        const Box3 box = Inflate(big, opt_.unionMargin);
        float origin[3];
        for (int a = 0; a < 3; ++a) origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
        Refill(v, bvh_, modelAbs_.data(), box, origin, &m, opt_.legacy ? nullptr : &fvb_, opt_.legacy || opt_.bodyGate ? nullptr : fast_.data());
    } else {
        ++cc.reuses;
    }
    for (int a = 0; a < 3; ++a) cc.lastCenter[a] = center[a];
    cc.haveLast = true;
    lap(stats ? &stats->tMemo : &dummy);
    if (opt_.stopAfter == 1) return false;
    // per shape: the memo's candidates in its box that pass the support prefilter, in FV order
    ShapeFilters(m, shapes, &st);
    BodyGate gate;
    for (int k = 0; k < 8; ++k) if (car_.shape[k].wheel < 0) gate.shapes |= 1u << k;   // the body shapes (Filter tests them plane first)
    if (opt_.bodyGate) {
        unsigned bodyShapes = 0;
        for (int k = 0; k < 8; ++k) if (car_.shape[k].wheel < 0) bodyShapes |= 1u << k;
        MakeBodyGate(m, shapes, st, bodyShapes, &gate);
    }
    m.verts = nullptr;
    Filter(v, fast_.data(), m, st, u, &cc.lists, gate, sink);
    lap(stats ? &stats->tFilter : &dummy);
    return opt_.stopAfter != 2;
}

void CcCollision::Shapes(const collide::ShapePose shapes[8], step::SubstepContacts &out, CcCache &cc, StageStats *stats) const {
    thread_local Scratch sc;
    const world::WorldPackView &v = *view_;
    out.n = 0;
    out.overflow = false;
    std::uint64_t t0 = stats ? __rdtsc() : 0;
    std::uint64_t dummy = 0;
    auto lap = [&](std::uint64_t *acc) { if (stats) { const std::uint64_t t1 = __rdtsc(); *acc += t1 - t0; t0 = t1; } };
    if (!Prepare(shapes, cc, stats, &t0)) return;
    Memo &m = cc.m;
    if (opt_.noise) {
        NoiseCounts nc;
        NoiseNarrowT(StageSrc{v, m, verts_.data()}, shapes, cc.lists, sc, stats ? &nc : nullptr);
        if (stats) {
            stats->frames += nc.frames; stats->survivors += nc.survivors; stats->face += nc.face; stats->fvForm += nc.fvForm; stats->reject += nc.reject;
            for (int k = 0; k < 8; ++k) if (car_.shape[k].wheel < 0) stats->bodySurvivors += cc.lists.n[k];
        }
        lap(stats ? &stats->tNarrow : &dummy);
        MergeOut(v, car_, sc.rawn, sc.merged, out, stats, opt_.mergeMode, opt_.legacy ? nullptr : outRec_.data());
        lap(stats ? &stats->tMerge : &dummy);
        return;
    }

    // ---- the call's survivors (shape-major) and their FV frames, one per (shape, instance) -----
    std::uint32_t np = 0;
    for (int k = 0; k < 8; ++k) np += cc.lists.n[k];
    if (sc.pe.size() < np + 4) { sc.pe.resize(2 * np + 4); sc.pf.resize(2 * np + 4); sc.pk.resize(2 * np + 4); }
    sc.jobs.clear();
    {
        std::uint32_t p = 0;
        for (int k = 0; k < 8; ++k) {
            const std::uint32_t ns = cc.lists.n[k];
            const std::uint32_t first = static_cast<std::uint32_t>(sc.jobs.size());
            const std::uint32_t *list = cc.lists.idx[k].data();
            for (std::uint32_t j = 0; j < ns; ++j, ++p) {
                const std::uint32_t e = list[j];
                const std::uint32_t inst = m.ref[e].instance;
                std::uint32_t f = first;
                for (; f < sc.jobs.size(); ++f) if (sc.jobs[f].instance == inst) break;
                if (f == sc.jobs.size()) sc.jobs.push_back(Job{static_cast<std::uint32_t>(k), inst});
                sc.pe[p] = e;
                sc.pf[p] = f;
                sc.pk[p] = static_cast<std::uint8_t>(k);
            }
        }
    }
    const std::size_t nj = sc.jobs.size();
    if (sc.frames.size() < nj + 4) sc.frames.resize(nj + 4);
    if (opt_.simdFrames) {
        for (std::size_t j0 = 0; j0 < nj; j0 += 4) {
            const Job *jb[4];
            for (int l = 0; l < 4; ++l) jb[l] = &sc.jobs[j0 + l < nj ? j0 + l : j0];
            const collide::ShapePose *sp[4] = {&shapes[jb[0]->shape], &shapes[jb[1]->shape], &shapes[jb[2]->shape], &shapes[jb[3]->shape]};
            const float *mi[4] = {v.instances[jb[0]->instance].iso, v.instances[jb[1]->instance].iso, v.instances[jb[2]->instance].iso,
                                  v.instances[jb[3]->instance].iso};
            // ShapePose = Iso {r[9], t[3]} + radii[3]: the ellipsoid iso is its first 12 floats
            const lanes::I3<F4> ell = GatherIso(sp[0]->world.r, sp[1]->world.r, sp[2]->world.r, sp[3]->world.r, 0);
            const lanes::I3<F4> mesh = GatherIso(mi[0], mi[1], mi[2], mi[3], 0);
            const lanes::V3<F4> rad = {G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 0), G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 1),
                                       G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 2)};
            const lanes::FrameFv<F4> f = lanes::MakeFrameFv(ell, rad, mesh);
            float *dst[4];
            for (int l = 0; l < 4; ++l) dst[l] = reinterpret_cast<float *>(&sc.frames[j0 + l]);
            ScatterIso(f.meshToUnit, dst, 0);
            ScatterIso(f.c2w, dst, 12);
            ScatterIso(f.n2w, dst, 24);
        }
    } else {
        for (std::size_t j = 0; j < nj; ++j) {
            const collide::ShapePose &sp = shapes[sc.jobs[j].shape];
            const float *mi = v.instances[sc.jobs[j].instance].iso;
            const lanes::I3<float> ell = IsoOf(sp.world.r, sp.world.t), mesh = IsoOf(mi, mi + 9);
            sc.frames[j] = lanes::MakeFrameFv(ell, lanes::V3<float>{sp.radii[0], sp.radii[1], sp.radii[2]}, mesh);
        }
    }
    if (stats) stats->frames += nj;
    lap(stats ? &stats->tFrames : &dummy);
    if (opt_.stopAfter == 3) return;

    // ---- narrow phase over all survivors: per shape, raw contacts in FV order --------------------
    for (int k = 0; k < 8; ++k) sc.raw[k].clear();
    if (stats) {
        stats->survivors += np;
        for (int k = 0; k < 8; ++k) if (car_.shape[k].wheel < 0) stats->bodySurvivors += cc.lists.n[k];
    }
    if (opt_.simdNarrow) {
        // pass A (SSE lanes over the call's survivors): unit-space vertices, FV's normal, the decision,
        // the FACE payload
        if (sc.so.size() < np + 4) sc.so.resize(2 * np + 4);
        for (std::uint32_t j0 = 0; j0 < np; j0 += 4) {
            std::uint32_t ix[4];
            for (int l = 0; l < 4; ++l) ix[l] = j0 + l < np ? j0 + l : j0;
            const float *fr[4], *tv[4];
            for (int l = 0; l < 4; ++l) {
                fr[l] = reinterpret_cast<const float *>(&sc.frames[sc.pf[ix[l]]]);
                tv[l] = reinterpret_cast<const float *>(&verts_[m.ref[sc.pe[ix[l]]].modelTri]);
            }
            const lanes::I3<F4> m2u = GatherIso(fr[0], fr[1], fr[2], fr[3], 0);
            F4 p[8];
            G4x4(tv[0], tv[1], tv[2], tv[3], 0, p);
            G4x4(tv[0], tv[1], tv[2], tv[3], 4, p + 4);
            const F4 p8 = G4(tv[0], tv[1], tv[2], tv[3], 8);
            const lanes::V3<F4> mv[3] = {{p[0], p[1], p[2]}, {p[3], p[4], p[5]}, {p[6], p[7], p8}};
            lanes::V3<F4> vu[3];
            for (int c = 0; c < 3; ++c) vu[c] = lanes::TPoint(m2u, mv[c]);
            F4 lenSq;
            const lanes::V3<F4> nr = lanes::UnitNormal(vu, &lenSq);
            B4 rej, face, fvb;
            lanes::Decide4(vu, nr, lenSq, exact::kDirEpsSq, exact::kMarginEps, &rej, &face, &fvb);
            const int nl = static_cast<int>(std::min<std::uint32_t>(4u, np - j0));
            const int lm = (1 << nl) - 1;
            const int rb = lanes::Bits(rej) & lm, fb = lanes::Bits(face) & lm, vb = lanes::Bits(fvb) & lm & ~rb;
            if (stats) {
                stats->reject += __builtin_popcount(rb);
                stats->face += __builtin_popcount(fb);
                stats->fvForm += nl - __builtin_popcount(rb | fb);
            }
            for (int l = 0; l < nl; ++l) sc.so[j0 + l].cls = (rb >> l & 1) ? kRej : (fb >> l & 1) ? kFaceC : (vb >> l & 1) ? kFvViol : kFvAny;
            if (rb == lm) continue;
            alignas(16) float X[21][4];   // vu (9), n (3), P N Sep (9)
            {
                const F4 len = lanes::Sqrt(lenSq);
                const F4 invLen = lanes::K(1.0f, len) / len;
                const lanes::V3<F4> n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};   // FV's n (FvCollide / EmitFace)
                for (int c = 0; c < 3; ++c) {
                    _mm_store_ps(X[3 * c], vu[c].x.v); _mm_store_ps(X[3 * c + 1], vu[c].y.v); _mm_store_ps(X[3 * c + 2], vu[c].z.v);
                }
                _mm_store_ps(X[9], n.x.v); _mm_store_ps(X[10], n.y.v); _mm_store_ps(X[11], n.z.v);
            }
            if (fb) {
                lanes::FrameFv<F4> f;
                f.meshToUnit = m2u;
                f.c2w = GatherIso(fr[0], fr[1], fr[2], fr[3], 12);
                f.n2w = GatherIso(fr[0], fr[1], fr[2], fr[3], 24);
                lanes::V3<F4> oP, oN, oS, oK;
                lanes::FacePayload<F4, B4>(f, vu, nr, lenSq, exact::kDirEpsSq, &oP, &oN, &oS, &oK);
                _mm_store_ps(X[12], oP.x.v); _mm_store_ps(X[13], oP.y.v); _mm_store_ps(X[14], oP.z.v);
                _mm_store_ps(X[15], oN.x.v); _mm_store_ps(X[16], oN.y.v); _mm_store_ps(X[17], oN.z.v);
                _mm_store_ps(X[18], oS.x.v); _mm_store_ps(X[19], oS.y.v); _mm_store_ps(X[20], oS.z.v);
            }
            for (int l = 0; l < nl; ++l) {
                if (rb >> l & 1) continue;
                SurvOut &o = sc.so[j0 + l];
                const int nf = (fb >> l & 1) ? 21 : 12;
                for (int f = 0; f < nf; ++f) o.x[f] = X[f][l];
            }
        }
        // pass B per shape, FV order: FACE payloads, FV's form for the rest. A feature-only pair (fvViol)
        // whose K (= its n) is NearlyEqual to the K of a FACE of the same shape is skipped: FV's merge
        // drops its contact, if it makes one (MergeAndAddToCollisions' first test, any primary).
        std::uint32_t p0 = 0;
        for (int k = 0; k < 8; ++k) {
            const std::uint32_t ns = cc.lists.n[k];
            std::vector<Raw> &raw = sc.raw[k];
            int nfK = 0;
            const float *fK[16];
            for (std::uint32_t pi = p0; pi < p0 + ns && nfK < 16; ++pi)
                if (sc.so[pi].cls == kFaceC) fK[nfK++] = sc.so[pi].x + 9;
            for (std::uint32_t pi = p0; pi < p0 + ns; ++pi) {
                const SurvOut &o = sc.so[pi];
                if (o.cls == kRej) continue;
                const std::uint32_t t = m.ref[sc.pe[pi]].local;
                if (o.cls == kFaceC) {
                    raw.push_back(Raw{exact::V{o.x[12], o.x[13], o.x[14]}, exact::V{o.x[15], o.x[16], o.x[17]}, exact::V{o.x[18], o.x[19], o.x[20]},
                                      exact::V{o.x[9], o.x[10], o.x[11]}, true, t});
                    continue;
                }
                if (o.cls == kFvViol && opt_.skipAbsorbed) {
                    const exact::V n = {o.x[9], o.x[10], o.x[11]};
                    int f = 0;
                    for (; f < nfK; ++f)
                        if (NearlyEqual(n, exact::V{fK[f][0], fK[f][1], fK[f][2]})) break;
                    if (f < nfK) { if (stats) ++stats->skipped; continue; }
                }
                const exact::V vul[3] = {{o.x[0], o.x[1], o.x[2]}, {o.x[3], o.x[4], o.x[5]}, {o.x[6], o.x[7], o.x[8]}};
                const FrameF &ff = sc.frames[sc.pf[pi]];
                exact::Frame frame;
                frame.c2w = ToExact(ff.c2w);
                frame.n2w = ToExact(ff.n2w);
                if (stats) ++stats->fvRuns;
                exact::FvCollide(frame, vul, t, &raw);
            }
            p0 += ns;
        }
    } else {
        for (std::uint32_t pi = 0; pi < np; ++pi) {
            const std::uint32_t e = sc.pe[pi];
            const FrameF &ff = sc.frames[sc.pf[pi]];
            exact::Frame frame;
            frame.meshToUnit = ToExact(ff.meshToUnit);
            frame.c2w = ToExact(ff.c2w);
            frame.n2w = ToExact(ff.n2w);
            const std::uint32_t t = m.ref[e].local;
            const exact::TriVerts &tv = verts_[m.ref[e].modelTri];
            exact::V vu[3];
            exact::UnitVerts(frame, tv, vu);
            float lenSq;
            const exact::V nr = exact::UnitNormal(vu, &lenSq);
            const int dd = exact::Decide(vu, nr, lenSq);
            if (stats) ++(dd == exact::kFace ? stats->face : dd == exact::kFvForm ? stats->fvForm : stats->reject);
            if (dd == exact::kFace) exact::EmitFace(frame, vu, nr, lenSq, t, &sc.raw[sc.pk[pi]]);
            else if (dd == exact::kFvForm) exact::FvCollide(frame, vu, t, &sc.raw[sc.pk[pi]]);
        }
    }
    lap(stats ? &stats->tNarrow : &dummy);

    MergeOut(v, car_, sc.raw, sc.merged, out, stats);
    lap(stats ? &stats->tMerge : &dummy);
}

std::uint32_t CcCollision::ProveLanes16(const float (*sweep)[16], std::uint32_t ask, step::CollisionCache *const *caches) const {
#if defined(__AVX512F__)
    // Idea 5, after DetectLanes16 on this thread: for each lane of `ask` (it had no candidate there), the region its shapes
    // sweep along w, grown by d: E + [0, w] + B(d) per shape. Filter's own tests on it: the boxes swept exactly (and
    // their union), the plane tests on the ball around the segment's middle that holds it, (E + w / 2) + B(|w| / 2 + d):
    // centre + w / 2 and tol' = tol + |w| / 2 + d, so reach' = rmax + tol', low' = -tol', lazy' = rmin - tol' (unit plane
    // normals: the support of E + B(r) is E's plus r). No survivor, and the memo box containing the swept union (it holds
    // every world triangle that can meet a swept box, and the memo of a pose inside them needs no refill, so the same
    // tolerance terms): no pose whose shape points stay within d - margin of the segment has a survivor (the margin is
    // the runner's: float rounding, tol's change with the pose, sqrt(kSupportSlack) - 1 times rmax).
    Front16Scratch &fs = Front16Scratch_();
    std::uint32_t proven = 0u;
    BodyGate gate;
    gate.shapes = bodyShapes_;
    for (std::uint32_t mm = ask & fs.lastMask; mm != 0u; mm &= mm - 1u) {
        const int l = __builtin_ctz(mm);
        const float d = sweep[3][l];
        if (!(d > 0.0f)) continue;
        const float w[3] = {sweep[0][l], sweep[1][l], sweep[2][l]};
        const float half = 0.5f * std::sqrt((w[0] * w[0] + w[1] * w[1]) + w[2] * w[2]);
        if (!(half >= 0.0f)) continue;
        const float r = (half + d) * 1.0001f;
        CcCache &cc = *static_cast<CcCache *>(caches[l]);
        ShapeTable &g = fs.grown;
        const ShapeTable &st = fs.st[l];
        Box3 U = EmptyBox();
        for (int a = 0; a < 3; ++a)
            for (int k = 0; k < 8; ++k) {
                g.lo[a][k] = std::min(st.lo[a][k], st.lo[a][k] + w[a]) - d;
                g.hi[a][k] = std::max(st.hi[a][k], st.hi[a][k] + w[a]) + d;
                U.lo[a] = std::min(U.lo[a], g.lo[a][k]);
                U.hi[a] = std::max(U.hi[a], g.hi[a][k]);
            }
        if (!Contains(cc.m.box, U)) continue;
        for (int f = 0; f < 14; ++f)
            for (int k = 0; k < 8; ++k) g.t[f][k] = st.t[f][k];
        for (int k = 0; k < 8; ++k) {
            for (int a = 0; a < 3; ++a) g.t[kTc0 + a][k] = st.t[kTc0 + a][k] + 0.5f * w[a];
            g.t[kTreach][k] = st.t[kTreach][k] + r;
            g.t[kTlow][k] = st.t[kTlow][k] - r;
            g.t[kTtol][k] = st.t[kTtol][k] + r;
            g.t[kTlazy][k] = st.t[kTlazy][k] - r;
        }
        Filter(*view_, fast_.data(), cc.m, g, U, &cc.lists, gate, &fs.proof, nullptr, 0);
        if (fs.proof.n == 0u) proven |= 1u << l;
    }
    return proven;
#else
    (void)sweep; (void)ask; (void)caches;
    return 0u;
#endif
}

bool ParseStageSwitch(const char *value, StageOptions *opt) {
    if (value == nullptr) return false;
    const std::string v(value);
    StageOptions o;
    if (v == "c") o.noise = true;
    else if (v == "c-legacy") { o.noise = true; o.legacy = true; }
    else if (v == "c-exact") o.noise = false;
    else if (v == "c-nomerge") { o.noise = true; o.mergeMode = 1; }
    else if (v == "c-deepest") { o.noise = true; o.mergeMode = 2; }
    else if (v == "c-nobody") { o.noise = true; o.skipBody = true; }
    else return false;
    *opt = o;
    return true;
}

namespace noise {

void Narrow(const collide::PreNarrow &pre, step::SubstepContacts &out, collide::CollideStats *stats) {
    thread_local Scratch sc;
    NoiseCounts nc;
    NoiseNarrowT(PreSrc{*pre.memo}, pre.shapes, *pre.lists, sc, stats ? &nc : nullptr);
    for (int k = 0; k < 8; ++k)
        if (!sc.rawn[k].empty()) collide::MergeShapeTo(*pre.memo, sc.rawn[k].data(), sc.rawn[k].size(), k, &sc.merged, out);
    if (stats) {
        stats->fvFrames += nc.frames;
        stats->triangleTests += nc.survivors;
        stats->decFace += nc.face;
        stats->decFv += nc.fvForm;
        stats->decReject += nc.reject;
    }
}

}  // namespace noise

}  // namespace tw::collide_c
