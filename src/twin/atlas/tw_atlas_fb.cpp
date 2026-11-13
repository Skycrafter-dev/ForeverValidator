// S2: the atlas fallback (architect D). See tw_atlas_fb.h.
#include "tw_atlas_fb.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "tw_exact_fast.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace tw::atlas {

namespace {

constexpr float kAlign = 0.8660254f;          // PhysicsTolerance::SphereContactNormalAlignment
// cos(30 degrees - 2e-3 rad): the proof's limit, a margin far above float error. No libm angle functions in the tick
// path (tests/twin/ci/audit_twin_objects.sh): the angle sum is compared through its cosine, sqrt only.
constexpr double kProofCos = 0.867023671067542;
constexpr float kSideMargin = 1e-4f;          // m: side-plane prefilter margin (float error of the local forms ~1e-6)

// geometry_helpers.cpp NearlyEqual(value, reference)
inline bool NearlyEq(float value, float reference) {
    const float tol = std::fabs(reference) * 1.0e-5f;
    return reference - tol <= value && value <= reference + tol;
}

// Conservative reject filter, float, world coordinates: true when the shape (centre t, M = axes diag(r)) can
// certainly not touch the triangle: centre behind its plane, plane out of reach, or the ellipsoid entirely beyond
// one edge's side plane. Margins (2e-3 m + 2e-6 of the world-scale offsets) are far above the float error of
// these forms at map scale (~2e-4 m), so a rejection is a no-contact outcome of the exact path. Survivors take the
// exact path, so the output bits do not depend on this filter (nor on the ISA that runs it).
inline bool RejectScalar(const Fallback::RejRec &r, const float t[3], const float M[3][3]) {
    const float sd = ((r.n[0] * t[0] + r.n[1] * t[1]) + r.n[2] * t[2]) - r.rho;
    const float mrg = 2e-3f + 2e-6f * std::fabs(r.rho);
    if (sd < -mrg) return true;
    float m[3];
    for (int j = 0; j < 3; ++j) m[j] = (M[0][j] * r.n[0] + M[1][j] * r.n[1]) + M[2][j] * r.n[2];
    const float s2 = (m[0] * m[0] + m[1] * m[1]) + m[2] * m[2];
    const float far = sd - mrg;
    if (far > 0.0f && far * far > s2 * 1.0001f) return true;
    // the plane section (an ellipse around the foot point, unit-space radius h = sqrt(1 - d^2)) entirely beyond one
    // edge line: every FV contact is a triangle point inside the ellipsoid, so inside the section
    const float is2 = 1.0f / s2;
    float g[3];
    for (int i = 0; i < 3; ++i) g[i] = (M[i][0] * m[0] + M[i][1] * m[1]) + M[i][2] * m[2];
    const float k1 = sd * is2;
    const float h2r = 1.0f - sd * sd * is2;
    const float h2 = (h2r > 0.0f ? h2r : 0.0f) + 1e-5f;
    for (int e = 0; e < 3; ++e) {
        const float uo = (r.u[e][0] * g[0] + r.u[e][1] * g[1]) + r.u[e][2] * g[2];
        const float se = ((((r.u[e][0] * t[0] + r.u[e][1] * t[1]) + r.u[e][2] * t[2]) - r.c[e]) - k1 * uo) - (2e-3f + 2e-6f * std::fabs(r.c[e]));
        if (!(se > 0.0f)) continue;
        float mu[3];
        for (int j = 0; j < 3; ++j) mu[j] = (M[0][j] * r.u[e][0] + M[1][j] * r.u[e][1]) + M[2][j] * r.u[e][2];
        const float su2 = (mu[0] * mu[0] + mu[1] * mu[1]) + mu[2] * mu[2];
        const float un = (mu[0] * m[0] + mu[1] * m[1]) + mu[2] * m[2];
        const float wr = su2 - un * un * is2;
        const float w = wr > 0.0f ? wr : 0.0f;
        if (se * se > w * h2 * 1.0001f) return true;
    }
    return false;
}

#if defined(__AVX512F__)
// 16 x 16 float transpose in registers (4 shuffle stages). The field order of the output registers is computed once
// (TransposePerm): out[TransposePerm()[f]] holds field f of the 16 input rows, lane l = row l.
inline void Transpose16(__m512 r[16]) {
    __m512 t[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; ++i) {
        r[4 * i] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[4 * i]), _mm512_castps_pd(t[4 * i + 2])));
        r[4 * i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[4 * i]), _mm512_castps_pd(t[4 * i + 2])));
        r[4 * i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[4 * i + 1]), _mm512_castps_pd(t[4 * i + 3])));
        r[4 * i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[4 * i + 1]), _mm512_castps_pd(t[4 * i + 3])));
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) {
            t[8 * i + j] = _mm512_shuffle_f32x4(r[8 * i + j], r[8 * i + j + 4], 0x88);
            t[8 * i + j + 4] = _mm512_shuffle_f32x4(r[8 * i + j], r[8 * i + j + 4], 0xdd);
        }
    for (int j = 0; j < 8; ++j) {
        r[j] = _mm512_shuffle_f32x4(t[j], t[j + 8], 0x88);
        r[j + 8] = _mm512_shuffle_f32x4(t[j], t[j + 8], 0xdd);
    }
}
struct TPerm { int reg[16]; bool ok; };
inline const TPerm &TransposePerm() {
    static const TPerm p = [] {
        TPerm q{};
        alignas(64) float in[16][16], out[16][16];
        for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) in[i][j] = float(i * 16 + j);
        __m512 r[16];
        for (int i = 0; i < 16; ++i) r[i] = _mm512_load_ps(in[i]);
        Transpose16(r);
        for (int k = 0; k < 16; ++k) _mm512_store_ps(out[k], r[k]);
        q.ok = true;
        for (int k = 0; k < 16; ++k) {
            const int f = int(out[k][0]) % 16;
            q.reg[f] = k;
            for (int l = 0; l < 16; ++l) q.ok = q.ok && int(out[k][l]) == l * 16 + f;   // lane l = row l, field f
        }
        return q;
    }();
    return p;
}
#endif

struct Cand {
    std::uint32_t order, tri;
    bool operator<(const Cand &o) const { return order < o.order || (order == o.order && tri < o.tri); }
    bool operator==(const Cand &o) const { return tri == o.tri; }
};

struct RawC {                 // one raw contact of the shape, in FV detection order
    std::uint32_t order, tri;
    bool primary;
    float P[3], N[3], Sep[3]; // payload (world)
    float K[3], Nk[3];        // merge keys (K; N for the 30-degree test)
    bool closed;              // FACE from the closed form (its keys may need FV's route)
};

struct Feat {                 // a triangle in reach with the foot point outside: FV may emit a feature
    std::uint32_t order, tri;
    float n[3];               // its own normal (world)
    float d, sg;              // unit-space plane distance, support along n
    float m[3];               // M^T n
};

struct Scratch {
    std::vector<Cand> cand;
    std::vector<std::uint32_t> surv;
    std::vector<RawC> raw;
    std::vector<Feat> feat;
    std::vector<exact::Raw> fv;
    std::vector<int> merged;
    struct FrameSlot { std::uint32_t inst; exact::Frame f; };
    std::vector<FrameSlot> frames;
};

// tan of the largest angle between plane n and any feature normal FV can emit from that plane at unit
// distance d (the feature lies in the unit-space disk of radius h = sqrt(1 - d^2) around the foot point).
// G(x) = axes diag(1/r) x maps unit space to world normals; G(n') = n / sigma.
double DiskTan(const collide::ShapePose &sp, const float nf[3], double d, double sg, const float mf[3]) {
    if (!(d > 0.0) || d > 1.0) return 1e30;
    const double h = std::sqrt(std::max(0.0, 1.0 - d * d));
    const double n[3] = {nf[0], nf[1], nf[2]};
    const double np[3] = {mf[0] / sg, mf[1] / sg, mf[2] / sg};   // n' in the shape's unit frame
    double a0[3] = {1, 0, 0};
    if (std::fabs(np[0]) > 0.6) { a0[0] = 0; a0[1] = 1; }
    double e1[3] = {a0[1] * np[2] - a0[2] * np[1], a0[2] * np[0] - a0[0] * np[2], a0[0] * np[1] - a0[1] * np[0]};
    const double l1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    for (double &x : e1) x /= l1;
    const double e2[3] = {np[1] * e1[2] - np[2] * e1[1], np[2] * e1[0] - np[0] * e1[2], np[0] * e1[1] - np[1] * e1[0]};
    double g1[3], g2[3];
    for (int i = 0; i < 3; ++i) {   // world image: sum_j axis_j[i] x_j / r_j (axis_j = world.r column j)
        g1[i] = 0; g2[i] = 0;
        for (int j = 0; j < 3; ++j) {
            const double a = double(sp.world.r[3 * j + i]) / sp.radii[j];
            g1[i] += a * e1[j];
            g2[i] += a * e2[j];
        }
    }
    const double t1 = n[0] * g1[0] + n[1] * g1[1] + n[2] * g1[2], t2 = n[0] * g2[0] + n[1] * g2[1] + n[2] * g2[2];
    double p1[3], p2[3];
    for (int i = 0; i < 3; ++i) { p1[i] = g1[i] - t1 * n[i]; p2[i] = g2[i] - t2 * n[i]; }
    const double A = p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2], B = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2],
                 C = p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2];
    const double lam = 0.5 * (A + C) + std::sqrt(0.25 * (A - C) * (A - C) + B * B);
    const double Gp = h * std::sqrt(lam) * 1.0001, Tm = h * std::sqrt(t1 * t1 + t2 * t2) * 1.0001;
    const double a = d / sg;
    return a > Tm ? Gp / (a - Tm) : 1e30;
}

}  // namespace

bool Fallback::Init(const collide::ExactCollision &stage, std::string *error) {
    stage_ = &stage;
    view_ = &stage.View();
    const world::WorldPackView &v = *view_;
    face_.resize(v.tris.size());
    for (std::size_t t = 0; t < v.tris.size(); ++t) {   // the board's triangle: mesh-local vertices through the instance, in double
        const world::TriInfo &ti = v.triInfo[t];
        const world::Instance &in = v.instances[ti.instance];
        const world::Model &m = v.models[in.model];
        const world::ModelTri &mt = v.modelTris[m.firstTri + ti.modelTri];
        double w[3][3];
        for (int c = 0; c < 3; ++c) {
            const world::Vec3f &p = v.modelVerts[m.firstVert + mt.v[c]];
            const float vm[3] = {p.x, p.y, p.z};
            WorldVertD(in.iso, vm, w[c]);
        }
        MakeFaceTriD(w, &face_[t]);
        SetFaceInstance(in.iso, in.rotation != world::kRotGeneral, &face_[t]);
    }
    rej_.resize(face_.size());
    for (std::size_t t = 0; t < face_.size(); ++t) {
        const FaceTri &f = face_[t];
        RejRec &r = rej_[t];
        if (!f.valid) { std::memset(&r, 0, sizeof r); continue; }   // never rejected (FV's form decides)
        for (int k = 0; k < 3; ++k) r.n[k] = f.n[k];
        r.rho = float(f.rho);
        for (int e = 0; e < 3; ++e) {
            for (int k = 0; k < 3; ++k) r.u[e][k] = f.u[e][k];
            r.c[e] = float(f.c[e]);
        }
    }
    (void)error;
    return true;
}

void Fallback::BvhCandidates(const collide::ShapePose &sp, std::vector<std::uint32_t> *cand) const {
    Box3 q;
    for (int a = 0; a < 3; ++a) {
        const float h = std::fabs(sp.world.r[a]) * sp.radii[0] + std::fabs(sp.world.r[3 + a]) * sp.radii[1] + std::fabs(sp.world.r[6 + a]) * sp.radii[2];
        q.lo[a] = sp.world.t[a] - h - 0.05f;
        q.hi[a] = sp.world.t[a] + h + 0.05f;
    }
    cand->clear();
    stage_->Bvh().Query(q, [&](std::uint32_t t) { cand->push_back(t); });
}

void Fallback::DetectShape(const collide::ShapePose shapes[8], int k, const std::uint32_t *cand, std::size_t n, const double origin[3],
                           step::SubstepContacts &out, FallbackStats *st, bool sortedUnique) const {
    (void)origin;
    thread_local Scratch sc;
    const world::WorldPackView &v = *view_;
    const collide::ShapePose &sp = shapes[k];
    if (st) { ++st->shapes; st->candidates += n; }
    // candidates in FV detection order, deduplicated
    sc.cand.resize(n);
    if (sortedUnique) {   // in FV order already: the position is the order key
        for (std::size_t i = 0; i < n; ++i) sc.cand[i] = Cand{static_cast<std::uint32_t>(i), cand[i]};
    } else {
        bool sorted = true;
        for (std::size_t i = 0; i < n; ++i) {
            sc.cand[i] = Cand{v.triInfo[cand[i]].fvOrder, cand[i]};
            if (i && !(sc.cand[i - 1] < sc.cand[i])) sorted = false;
        }
        if (!sorted) {
            std::sort(sc.cand.begin(), sc.cand.end());
            sc.cand.erase(std::unique(sc.cand.begin(), sc.cand.end()), sc.cand.end());
        }
    }
    sc.raw.clear();
    sc.feat.clear();
    sc.frames.clear();
    // FV's triangle test on FV's frame for (shape k, instance) (FV's bits); appends to sc.fv
    auto fvTest = [&](std::uint32_t t) -> int {
        if (st) ++st->fvForms;
#ifdef TW_ATLAS_FB_PROFILE
        static const bool noFv = std::getenv("FB_NOFV") != nullptr;
        if (noFv) return 0;   // profiling only: wrong output
#endif
        const world::TriInfo &ti = v.triInfo[t];
        const exact::Frame *fr = nullptr;
        for (const auto &s : sc.frames) if (s.inst == ti.instance) { fr = &s.f; break; }
        if (!fr) {
            sc.frames.emplace_back();
            sc.frames.back().inst = ti.instance;
            exact::MakeFrameFv(sp.world.r, sp.world.t, sp.radii, v.instances[ti.instance].iso, &sc.frames.back().f);
            fr = &sc.frames.back().f;
        }
        const world::Instance &in = v.instances[ti.instance];
        const world::Model &m = v.models[in.model];
        const world::ModelTri &mt = v.modelTris[m.firstTri + ti.modelTri];
        exact::TriVerts tv;
        for (int c = 0; c < 3; ++c) {
            const world::Vec3f &p = v.modelVerts[m.firstVert + mt.v[c]];
            tv.v[c] = exact::V{p.x, p.y, p.z};
        }
        sc.fv.clear();
        return exact::CollideFv(*fr, tv, t, &sc.fv);
    };
    auto pushFv = [&](const Cand &c) {
        if (!fvTest(c.tri)) return;
        const exact::Raw &r = sc.fv[0];
        RawC o;
        o.order = c.order; o.tri = c.tri; o.primary = r.primary; o.closed = false;
        o.P[0] = r.P.x; o.P[1] = r.P.y; o.P[2] = r.P.z;
        o.N[0] = r.N.x; o.N[1] = r.N.y; o.N[2] = r.N.z;
        o.Sep[0] = r.Sep.x; o.Sep[1] = r.Sep.y; o.Sep[2] = r.Sep.z;
        o.K[0] = r.K.x; o.K[1] = r.K.y; o.K[2] = r.K.z;
        for (int i = 0; i < 3; ++i) o.Nk[i] = o.N[i];
        if (st && !o.primary) ++st->fvFeatures;
        sc.raw.push_back(o);
    };
    // pass 1: closed-form FACEs, ties, and the triangles where a feature is possible
    ShapeLocal sl;
    MakeShapeLocal(sp.world.r, sp.world.t, sp.radii, &sl);
    // the conservative reject filter over every candidate (16 lanes with AVX-512), survivors kept in FV order
    sc.surv.clear();
    {
        const float tf[3] = {sp.world.t[0], sp.world.t[1], sp.world.t[2]};
        std::size_t i = 0;
#if defined(__AVX512F__)
        const float *base = reinterpret_cast<const float *>(rej_.data());
        const __m512 t0 = _mm512_set1_ps(tf[0]), t1 = _mm512_set1_ps(tf[1]), t2 = _mm512_set1_ps(tf[2]);
        __m512 Mv[3][3];
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) Mv[a][b] = _mm512_set1_ps(sl.M[a][b]);
        const __m512 k2m = _mm512_set1_ps(2e-3f), k2r = _mm512_set1_ps(2e-6f), slack = _mm512_set1_ps(1.0001f), zero = _mm512_setzero_ps();
        const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#ifdef TW_ATLAS_FB_PROFILE
        static const bool scalarFilter = std::getenv("FB_SCALARFILTER") != nullptr;
        const std::size_t simdEnd = scalarFilter ? 0 : sc.cand.size();
#else
        const std::size_t simdEnd = sc.cand.size();
#endif
        for (; i < simdEnd; i += 16) {   // the last chunk partial: unused lanes repeat its last candidate, masked out
            const std::size_t nl = std::min<std::size_t>(16, sc.cand.size() - i);
            __m512 rows[16];
            for (std::size_t l = 0; l < 16; ++l) rows[l] = _mm512_load_ps(base + std::size_t(sc.cand[i + (l < nl ? l : nl - 1)].tri) * 16);
            Transpose16(rows);
            const TPerm &tp = TransposePerm();
            auto G = [&](int f) { return rows[tp.reg[f]]; };
            const __m512 nx = G(0), ny = G(1), nz = G(2), rho = G(3);
            const __m512 sd = _mm512_sub_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(nx, t0), _mm512_mul_ps(ny, t1)), _mm512_mul_ps(nz, t2)), rho);
            const __m512 absRho = _mm512_abs_ps(rho);
            const __m512 mrg = _mm512_add_ps(k2m, _mm512_mul_ps(k2r, absRho));
            __mmask16 rej = _mm512_cmp_ps_mask(sd, _mm512_sub_ps(zero, mrg), _CMP_LT_OQ);
            __m512 m[3];
            for (int j = 0; j < 3; ++j)
                m[j] = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mv[0][j], nx), _mm512_mul_ps(Mv[1][j], ny)), _mm512_mul_ps(Mv[2][j], nz));
            const __m512 s2 = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(m[0], m[0]), _mm512_mul_ps(m[1], m[1])), _mm512_mul_ps(m[2], m[2]));
            const __m512 far = _mm512_sub_ps(sd, mrg);
            rej |= _mm512_cmp_ps_mask(far, zero, _CMP_GT_OQ) & _mm512_cmp_ps_mask(_mm512_mul_ps(far, far), _mm512_mul_ps(s2, slack), _CMP_GT_OQ);
            const __m512 is2 = _mm512_div_ps(_mm512_set1_ps(1.0f), s2);
            __m512 g[3];
            for (int a = 0; a < 3; ++a)
                g[a] = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mv[a][0], m[0]), _mm512_mul_ps(Mv[a][1], m[1])), _mm512_mul_ps(Mv[a][2], m[2]));
            const __m512 k1 = _mm512_mul_ps(sd, is2);
            const __m512 h2 = _mm512_add_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_mul_ps(_mm512_mul_ps(sd, sd), is2)), zero), _mm512_set1_ps(1e-5f));
            for (int e = 0; e < 3; ++e) {
                const __m512 ux = G(4 + 3 * e), uy = G(5 + 3 * e), uz = G(6 + 3 * e), c = G(13 + e);
                const __m512 uo = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(ux, g[0]), _mm512_mul_ps(uy, g[1])), _mm512_mul_ps(uz, g[2]));
                const __m512 se = _mm512_sub_ps(
                    _mm512_sub_ps(_mm512_sub_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(ux, t0), _mm512_mul_ps(uy, t1)), _mm512_mul_ps(uz, t2)), c),
                                  _mm512_mul_ps(k1, uo)),
                    _mm512_add_ps(k2m, _mm512_mul_ps(k2r, _mm512_abs_ps(c))));
                __m512 mu[3];
                for (int j = 0; j < 3; ++j)
                    mu[j] = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mv[0][j], ux), _mm512_mul_ps(Mv[1][j], uy)), _mm512_mul_ps(Mv[2][j], uz));
                const __m512 su2 = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(mu[0], mu[0]), _mm512_mul_ps(mu[1], mu[1])), _mm512_mul_ps(mu[2], mu[2]));
                const __m512 un = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(mu[0], m[0]), _mm512_mul_ps(mu[1], m[1])), _mm512_mul_ps(mu[2], m[2]));
                const __m512 w = _mm512_max_ps(_mm512_sub_ps(su2, _mm512_mul_ps(_mm512_mul_ps(un, un), is2)), zero);
                rej |= _mm512_cmp_ps_mask(se, zero, _CMP_GT_OQ) &
                       _mm512_cmp_ps_mask(_mm512_mul_ps(se, se), _mm512_mul_ps(_mm512_mul_ps(w, h2), slack), _CMP_GT_OQ);
            }
            const __mmask16 keep = static_cast<__mmask16>(~rej & ((1u << nl) - 1u));
            alignas(64) std::int32_t pos[16];
            _mm512_mask_compressstoreu_epi32(pos, keep, _mm512_add_epi32(iota, _mm512_set1_epi32(static_cast<int>(i))));
            const int nk = __builtin_popcount(static_cast<unsigned>(keep));
            for (int q = 0; q < nk; ++q) sc.surv.push_back(static_cast<std::uint32_t>(pos[q]));
        }
#endif
        for (; i < sc.cand.size(); ++i)
            if (!RejectScalar(rej_[sc.cand[i].tri], tf, sl.M)) sc.surv.push_back(static_cast<std::uint32_t>(i));
    }
    if (st) st->survivors += sc.surv.size();
#ifdef TW_ATLAS_FB_PROFILE
    static const bool filterOnly = std::getenv("FB_FILTERONLY") != nullptr;
    if (filterOnly) { if (!sc.surv.empty()) out.overflow = out.overflow || false; return; }   // profiling only: wrong output
#endif
    for (std::uint32_t si : sc.surv) {
        const Cand &c = sc.cand[si];
        const FaceTri &f = face_[c.tri];
        FaceOut fo;
        float tie;
        int why;
        if (ClosedFace(f, sl, &fo, &tie, &why)) {
            RawC o;
            o.order = c.order; o.tri = c.tri; o.primary = true; o.closed = true;
            model::RespContact tmp;
            FaceToWorld(fo, &tmp);
            for (int i = 0; i < 3; ++i) { o.P[i] = tmp.P[i]; o.N[i] = tmp.N[i]; o.Sep[i] = tmp.Sep[i]; o.K[i] = fo.K[i]; o.Nk[i] = fo.N[i]; }
            sc.raw.push_back(o);
            if (st) ++st->faces;
            continue;
        }
        if (why == kFaceBehind || why == kFaceFar) continue;
        if (why == kFaceTie || why == kFaceInvalid) {
            if (st) ++st->ties;
            pushFv(c);
            continue;
        }
        // kFaceOutside: in reach (eps < d < 1 - eps), foot point outside an edge. Section prefilter: every FV
        // contact on this triangle is a triangle point inside the ellipsoid, so inside the plane section, an
        // ellipse centred on the foot point. Its support along the in-plane edge normal u is
        // h·sqrt(|M^T u|^2 - ((M^T u).(M^T n))^2 / sigma^2), h = sqrt(1 - d^2); the section entirely beyond one
        // edge line means no contact.
        if (st) ++st->outside;
        bool rej = false;
        const float h2 = 1.0f - fo.d * fo.d;
        for (int e = 0; e < 3 && !rej; ++e) {
            const float se = fo.se[e] - kSideMargin;
            if (!(se > 0.0f)) continue;
            float mu[3];
            for (int j = 0; j < 3; ++j) mu[j] = (sl.M[0][j] * f.u[e][0] + sl.M[1][j] * f.u[e][1]) + sl.M[2][j] * f.u[e][2];
            const float uu = (mu[0] * mu[0] + mu[1] * mu[1]) + mu[2] * mu[2];
            const float un = (mu[0] * fo.m[0] + mu[1] * fo.m[1]) + mu[2] * fo.m[2];
            const float w = uu - un * un / fo.s2;
            const float sup2 = (w > 0.0f ? w : 0.0f) * h2 * 1.0001f;
            if (se * se > sup2) rej = true;
        }
        if (rej) { if (st) ++st->sideRejects; continue; }
        Feat ft;
        ft.order = c.order; ft.tri = c.tri;
        for (int i = 0; i < 3; ++i) { ft.n[i] = f.n[i]; ft.m[i] = fo.m[i]; }
        ft.sg = std::sqrt(fo.s2);
        ft.d = fo.d;
        sc.feat.push_back(ft);
    }
    // pass 2: features. Proof against the shape's FACEs (any primary so far: closed form or FV form), else FV form.
    bool anyFeature = false;
    if (!sc.feat.empty()) {
        std::size_t nPrim = 0;
        for (const RawC &r : sc.raw) nPrim += r.primary;
        for (const Feat &ft : sc.feat) {
            bool proven = false;
            if (nPrim) {
                const double tanT = DiskTan(sp, ft.n, ft.d, ft.sg, ft.m);
                if (tanT < 1e29) {
                    // theta = atan(tanT) in [0, 90 deg); alpha = angle(n_f, n_g) = acos(c). With c > 0 both are below 90
                    // degrees, so theta + alpha < 30 deg - margin  <=>  cos(theta + alpha) = cos(theta) c - sin(theta) sin(alpha)
                    // > kProofCos (cos is decreasing on [0, 180 deg]).
                    const double ct = 1.0 / std::sqrt(1.0 + tanT * tanT), st = tanT * ct;
                    for (const RawC &g : sc.raw) {
                        if (!g.primary) continue;
                        const double c = std::min(1.0, double(ft.n[0]) * g.Nk[0] + double(ft.n[1]) * g.Nk[1] + double(ft.n[2]) * g.Nk[2]);
                        if (!(c > 0.0)) continue;
                        const double sa = std::sqrt(std::max(0.0, 1.0 - c * c));
                        if (ct * c - st * sa > kProofCos) { proven = true; break; }
                    }
                }
            }
            if (proven) { if (st) ++st->proofs; continue; }
            const std::size_t before = sc.raw.size();
            pushFv(Cand{ft.order, ft.tri});
            if (sc.raw.size() > before && !sc.raw.back().primary) anyFeature = true;
        }
        if (anyFeature) {
            // The merge's NearlyEqual depends on FV's rounding residues where it can decide: a closed-form FACE
            // takes FV's keys when some feature lies on a nearly coplanar triangle (world normals within 1e-3,
            // far above what NearlyEqual's relative 1e-5 on unit-space normals can bridge with r_max / r_min <= 11).
            // Otherwise the keys cannot be NearlyEqual and only the 30-degree test decides.
            for (RawC &g : sc.raw) {
                if (!g.primary || !g.closed) continue;
                bool nearCoplanar = false;
                const FaceTri &gf = face_[g.tri];
                for (const RawC &c : sc.raw) {
                    if (c.primary) continue;
                    const FaceTri &cf = face_[c.tri];
                    const float dx = cf.n[0] - gf.n[0], dy = cf.n[1] - gf.n[1], dz = cf.n[2] - gf.n[2];
                    if (dx * dx + dy * dy + dz * dz <= 1e-6f) { nearCoplanar = true; break; }
                }
                if (!nearCoplanar) continue;
                if (st) ++st->fvFaceKeys;
                if (!fvTest(g.tri)) continue;   // FV disagrees outside the tie band (not expected): keep the closed keys
                const exact::Raw &r = sc.fv[0];
                if (!r.primary) continue;
                g.K[0] = r.K.x; g.K[1] = r.K.y; g.K[2] = r.K.z;
                g.Nk[0] = r.N.x; g.Nk[1] = r.N.y; g.Nk[2] = r.N.z;
            }
        }
        std::sort(sc.raw.begin(), sc.raw.end(), [](const RawC &a, const RawC &b) { return a.order < b.order; });
    }
    // FV's per-shape merge: primaries in order, then the non-primaries no primary absorbs
    std::vector<int> &merged = sc.merged;
    merged.clear();
    for (std::size_t r = 0; r < sc.raw.size(); ++r) if (sc.raw[r].primary) merged.push_back(static_cast<int>(r));
    const std::size_t nPrim = merged.size();
    for (std::size_t r = 0; r < sc.raw.size(); ++r) {
        const RawC &c = sc.raw[r];
        if (c.primary) continue;
        std::size_t j = 0;
        for (; j < nPrim; ++j) {
            const RawC &p = sc.raw[static_cast<std::size_t>(merged[j])];
            const bool ne = NearlyEq(c.K[0], p.K[0]) && NearlyEq(c.K[1], p.K[1]) && NearlyEq(c.K[2], p.K[2]);
            const float dot = (c.Nk[0] * p.Nk[0] + c.Nk[1] * p.Nk[1]) + c.Nk[2] * p.Nk[2];
            if (ne || kAlign < dot) break;
        }
        if (j == nPrim) merged.push_back(static_cast<int>(r));
    }
    for (int idx : merged) {
        if (out.n >= model::kRespMaxContacts) { out.overflow = true; break; }
        const RawC &c = sc.raw[static_cast<std::size_t>(idx)];
        const Tri &tr = v.tris[c.tri];
        const world::TriInfo &ti = v.triInfo[c.tri];
        const world::Actor &a = v.ActorOf(tr);
        model::RespContact &o = out.c[out.n++];
        for (int i = 0; i < 3; ++i) { o.P[i] = c.P[i]; o.N[i] = c.N[i]; o.Sep[i] = c.Sep[i]; o.peerZ[i] = a.peerZ[i]; }
        const bool trig = ti.trigger != world::kNoTrigger;
        o.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
        o.shape = static_cast<std::uint8_t>(k);
        o.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
        o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (trig ? unsigned(model::kRespTrigger) : 0u));
        o.pad = 0u;
        if (st) ++st->mergedOut;
    }
}

}  // namespace tw::atlas
