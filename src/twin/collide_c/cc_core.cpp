// Architect C car cache: tracked patches in closed form, odometer-bounded sleepers,
// exact fallback (B's tw_exact.h), FV's per-shape merge.
#include "cc_core.h"
#include "cc_face.h"
#include "cc_proxy.h"

#include <x86intrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../math/tw_lanes.h"
#include "../math/tw_math.h"

namespace cc {

namespace m = tw::math;

TouchCounter g_touch;
std::uint64_t g_split[8] = {};
std::uint64_t g_fine[16] = {};
#if CC_SPLIT
#define CC_TSC() __rdtsc()
#else
#define CC_TSC() std::uint64_t(0)
#endif
#if CC_FINE
#define CC_FT(i) do { const std::uint64_t ft_ = __rdtsc(); g_fine[i] += ft_ - fineLast_; fineLast_ = ft_; } while (0)
#define CC_FT0() (fineLast_ = __rdtsc())
#else
#define CC_FT(i) ((void)0)
#define CC_FT0() ((void)0)
#endif

void TouchCounter::Clear() {
    ++gen;
    lines[0] = lines[1] = lines[2] = 0;
}

#if CC_STATS
void Touch(const void *p, std::size_t bytes, int cls) {
    if (bytes == 0 || p == nullptr) return;
    TouchCounter &tc = g_touch;
    if (tc.slots.empty()) tc.slots.assign(1u << 18, 0);
    if (tc.lines[0] + tc.lines[1] + tc.lines[2] > (3u << 16)) return;  // table 3/4 full (caller did not clear): stop counting
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p) >> 6, b = (reinterpret_cast<std::uintptr_t>(p) + bytes - 1) >> 6;
    for (std::uintptr_t l = a; l <= b; ++l) {
        // slot value: gen in the top 20 bits, (line, class) hash below; collisions only undercount slightly
        const std::uint64_t key = (std::uint64_t(l) << 2) | std::uint64_t(cls);
        std::uint64_t h = key * 0x9e3779b97f4a7c15ull;
        std::size_t i = static_cast<std::size_t>(h >> 46);
        const std::uint64_t g20 = tc.gen & ((1ull << 20) - 1);
        const std::uint64_t tag = (g20 << 44) | (key & ((1ull << 44) - 1));
        for (;;) {
            std::uint64_t &s = tc.slots[i];
            if (s == tag) break;
            if ((s >> 44) != g20) { s = tag; ++tc.lines[cls]; break; }
            i = (i + 1) & ((1u << 18) - 1);
        }
    }
}
#endif

namespace {

inline float Dot3(const float a[3], const float b[3]) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }
inline float Gap(const Box3 &a, const Box3 &b) {
    float g = -1e30f;
    for (int k = 0; k < 3; ++k) {
        g = std::max(g, b.lo[k] - a.hi[k]);
        g = std::max(g, a.lo[k] - b.hi[k]);
    }
    return g;
}
inline void MtV(const float M[3][3], const float v[3], float o[3]) {  // o = M^T v
    for (int j = 0; j < 3; ++j) o[j] = (M[0][j] * v[0] + M[1][j] * v[1]) + M[2][j] * v[2];
}
inline void MV(const float M[3][3], const float v[3], float o[3]) {  // o = M v
    for (int i = 0; i < 3; ++i) o[i] = (M[i][0] * v[0] + M[i][1] * v[1]) + M[i][2] * v[2];
}
inline float Motion(const Pose &a, const Pose &b) {  // |dt| + ||dM||_F, rounded up
    const float dx = a.t[0] - b.t[0], dy = a.t[1] - b.t[1], dz = a.t[2] - b.t[2];
    float f = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) { const float d = a.M[i][j] - b.M[i][j]; f += d * d; }
    return (m::sqrt((dx * dx + dy * dy) + dz * dz) + m::sqrt(f)) * 1.0001f + 1e-6f;
}
inline float Sup(const Pose &p, const float u[3]) {  // support of the shape along unit u
    float v[3];
    MtV(p.M, u, v);
    return p.isBox ? (m::abs(v[0]) + m::abs(v[1])) + m::abs(v[2]) : m::sqrt(Dot3(v, v));
}
// largest signed distance of a shape centre along u, relative to p.t (0 for an ellipsoid; for the body proxy
// the support of the box of the 4 body centres)
inline float CentreSup(const Pose &p, const float u[3]) {
    if (!p.isBox) return 0.0f;
    float v[3];
    MtV(p.cM, u, v);
    return (m::abs(v[0]) + m::abs(v[1])) + m::abs(v[2]);
}
inline void EnsureSm(Pose &p) {
    if (!p.smValid) { tw::MakeShapeMap(*p.src, &p.sm); p.smValid = true; }
}

}  // namespace

void SetupPose(const tw::Shape &sh, Pose *p) {
    for (int i = 0; i < 3; ++i) p->t[i] = sh.t[i];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) p->M[i][j] = sh.axis[j][i] * sh.r[j];
    p->boxValid = false;
    p->rmax = std::max(sh.r[0], std::max(sh.r[1], sh.r[2]));
    p->rmin = std::min(sh.r[0], std::min(sh.r[1], sh.r[2]));
    p->smValid = false;
    p->isBox = false;
    p->src = &sh;
}

void MergeShape(const std::vector<Emitted> &in, std::vector<Emitted> *out) {
    out->clear();
    for (const Emitted &c : in) if (c.primary) out->push_back(c);
    const std::size_t np = out->size();
    for (const Emitted &f : in) {
        if (f.primary) continue;
        bool drop = false;
        for (std::size_t g = 0; g < np && !drop; ++g) {
            const tw::Contact &p = (*out)[g].c;
            bool eq = true;
            for (int k = 0; k < 3; ++k) eq = eq && std::fabs(f.c.K[k] - p.K[k]) <= 1e-5f * std::fabs(p.K[k]);
            const float dn = f.c.N[0] * p.N[0] + f.c.N[1] * p.N[1] + f.c.N[2] * p.N[2];
            drop = eq || dn > 0.8660254f;
        }
        if (!drop) out->push_back(f);
    }
}

// Every coplanar feature FV could emit lies on the unit-space contact disk (centre -d n_u, radius
// sqrt(1 - d^2)); its world normal is (d/h) n + N q with q in the disk plane. Bound the angle to n
// (parallel part by |a|, perpendicular part by the top singular value of the in-plane part of N) and
// accept below 30 deg with a 5 % margin on tan^2: FV's merge then drops every such feature.
// (Same bound as B's tw_voxbench FeaturesDropped.)
bool FeaturesDropped(const tw::ShapeMap &sm, const float n[3], const float Mn[3], float h, float d) {
    const float ih = 1.0f / h;
    const float nu[3] = {Mn[0] * ih, Mn[1] * ih, Mn[2] * ih};
    int ax = 0;
    for (int a = 1; a < 3; ++a) if (std::fabs(nu[a]) < std::fabs(nu[ax])) ax = a;
    float e1[3] = {ax == 0 ? 0.0f : (ax == 1 ? nu[2] : -nu[1]), ax == 1 ? 0.0f : (ax == 0 ? -nu[2] : nu[0]),
                   ax == 2 ? 0.0f : (ax == 0 ? nu[1] : -nu[0])};
    const float il = 1.0f / m::sqrt(Dot3(e1, e1));
    for (float &x : e1) x *= il;
    const float e2[3] = {nu[1] * e1[2] - nu[2] * e1[1], nu[2] * e1[0] - nu[0] * e1[2], nu[0] * e1[1] - nu[1] * e1[0]};
    float g1[3], g2[3];
    MV(sm.N, e1, g1);
    MV(sm.N, e2, g2);
    const float a1 = Dot3(n, g1), a2 = Dot3(n, g2);
    float b1[3], b2[3];
    for (int i = 0; i < 3; ++i) { b1[i] = g1[i] - a1 * n[i]; b2[i] = g2[i] - a2 * n[i]; }
    const float P = Dot3(b1, b1), Q = Dot3(b2, b2), R = Dot3(b1, b2);
    const float sig2 = 0.5f * (P + Q + m::sqrt((P - Q) * (P - Q) + 4.0f * R * R));
    const float hh2 = std::max(0.0f, 1.0f - d * d), hh = m::sqrt(hh2);
    const float den = d * ih - hh * m::sqrt(a1 * a1 + a2 * a2);
    return den > 0.0f && hh2 * sig2 <= (0.95f / 3.0f) * den * den;
}

void SleepList::Clear() {
    n = 0;
    wx.clear(); wy.clear(); wz.clear(); base.clear(); pid.clear();
    for (int a = 0; a < 3; ++a) { lo[a].clear(); hi[a].clear(); }
}

void SleepList::Push(std::uint32_t p, const float w[3], float b, const Box3 &box) {
    if (n % 16 == 0) {  // open a new padded block of 16 (padding never wakes)
        wx.resize(n + 16, 0.0f); wy.resize(n + 16, 0.0f); wz.resize(n + 16, 0.0f);
        base.resize(n + 16, 1e30f); pid.resize(n + 16, 0u);
        for (int a = 0; a < 3; ++a) { lo[a].resize(n + 16, 1e30f); hi[a].resize(n + 16, 1e30f); }
    }
    wx[n] = w[0]; wy[n] = w[1]; wz[n] = w[2]; base[n] = b; pid[n] = p;
    for (int a = 0; a < 3; ++a) { lo[a][n] = box.lo[a]; hi[a][n] = box.hi[a]; }
    ++n;
}

void SleepList::Remove(std::size_t i) {
    --n;
    wx[i] = wx[n]; wy[i] = wy[n]; wz[i] = wz[n]; base[i] = base[n]; pid[i] = pid[n];
    for (int a = 0; a < 3; ++a) { lo[a][i] = lo[a][n]; hi[a][i] = hi[a][n]; lo[a][n] = hi[a][n] = 1e30f; }
    wx[n] = wy[n] = wz[n] = 0.0f; base[n] = 1e30f;
    if (n % 16 == 0) {
        wx.resize(n); wy.resize(n); wz.resize(n); base.resize(n); pid.resize(n);
        for (int a = 0; a < 3; ++a) { lo[a].resize(n); hi[a].resize(n); }
    }
}

namespace {

inline float Lhs(const float w[3], const float O[4]) { return ((w[0] * O[0] + w[1] * O[1]) + w[2] * O[2]) + O[3]; }
inline float Rate(const float w[3], const float rate[4]) { return ((w[0] * rate[0] + w[1] * rate[1]) + w[2] * rate[2]) + rate[3] + 1e-4f; }

}  // namespace

// certificate along direction n (unit) with margin mg
void MakeCert(const float n[3], float mg, const float rate[4], Cert *c) {
    c->kind = kCertPlane;
    c->w[0] = m::abs(n[0]); c->w[1] = m::abs(n[1]); c->w[2] = m::abs(n[2]);
    c->m = mg;
    c->score = mg / Rate(c->w, rate);
}

Cert CarCache::Certify(const Pose &p, const Patch &pa, const float rate[4]) {
    CC_TOUCH(&pa, sizeof(Patch), kTouchHot);
    Cert best;
    best.w[0] = best.w[1] = best.w[2] = 0.0f;
    best.m = -1e30f;
    best.score = -1e30f;
    best.kind = kCertBox;
    auto offer = [&](const float w[3], float mg, int kind) {
        if (!(mg > 0.0f)) {
            if (best.score < 0.0f && mg > best.m) { best.m = mg; best.w[0] = w[0]; best.w[1] = w[1]; best.w[2] = w[2]; best.kind = kind; }
            return;
        }
        const float sc = mg / Rate(w, rate);
        if (sc > best.score) { best.score = sc; best.m = mg; best.w[0] = w[0]; best.w[1] = w[1]; best.w[2] = w[2]; best.kind = kind; }
    };
    // AABB separation per axis
    for (int a = 0; a < 3; ++a) {
        const float g = std::max(pa.box.lo[a] - p.B().hi[a], p.B().lo[a] - pa.box.hi[a]);
        const float w[3] = {a == 0 ? 1.0f : 0.0f, a == 1 ? 1.0f : 0.0f, a == 2 ? 1.0f : 0.0f};
        offer(w, g - 1e-5f, kCertBox);
    }
    if (pa.flags & kPatchExactOnly) return best;
    if (best.score >= earlyOut_) return best;  // the AABB gap alone lasts long enough: skip the slab and outline
    const float an[3] = {m::abs(pa.n[0]), m::abs(pa.n[1]), m::abs(pa.n[2])};
    const float dt[3] = {p.t[0] - pa.ref[0], p.t[1] - pa.ref[1], p.t[2] - pa.ref[2]};
    const float sd = Dot3(pa.n, dt);
    // centre behind every member plane (the alpha term grows by <= alpha*sqrt(3) per m of motion: x0.99)
    const float dc = (m::abs(p.t[0] - pa.cen[0]) + m::abs(p.t[1] - pa.cen[1])) + m::abs(p.t[2] - pa.cen[2]);
    offer(an, (-sd - CentreSup(p, pa.n) - (pa.tau + pa.alpha * (dc + pa.rad))) * 0.99f - 1e-4f, kCertPlane);
    offer(an, sd - pa.tau - Sup(p, pa.n) - 1e-4f, kCertPlane);  // slab out of reach
    const PEdge *E = &pw->edges[pa.firstEdge];
    CC_TOUCH(E, pa.edgeCount * sizeof(PEdge), kTouchHot);
    for (std::uint32_t j = 0; j < pa.edgeCount; ++j) {
        const float d[3] = {p.t[0] - E[j].a[0], p.t[1] - E[j].a[1], p.t[2] - E[j].a[2]};
        const float se = Dot3(E[j].u, d) - 3e-4f;
        const float au[3] = {m::abs(E[j].u[0]), m::abs(E[j].u[1]), m::abs(E[j].u[2])};
        const float ub = se - p.rmin;  // support >= rmin
        if (!(ub > 0.0f)) continue;
        if (best.score > 0.0f && ub / Rate(au, rate) <= best.score) continue;  // cannot beat the best
        offer(au, se - Sup(p, E[j].u), kCertEdge);
    }
    if (best.score < 0.0f) best.score = 0.0f;
    return best;
}

void CarCache::ExactMembers(Pose &p, const Patch &pa, std::vector<Emitted> &out) {
    EnsureSm(p);
    for (std::uint32_t i = pa.firstMember; i < pa.firstMember + pa.memberCount; ++i) {
        const Member &mb = pw->members[i];
        const tw::Tri &tr = world->tris[mb.tri];
        CC_TOUCH(&mb.tri, 4, kTouchExact);
        if (!tw::Overlap(p.B(), Box3{{tr.lo[0], tr.lo[1], tr.lo[2]}, {tr.hi[0], tr.hi[1], tr.hi[2]}})) continue;
        CC_TOUCH(&tr, sizeof(tw::Tri), kTouchExact);
        CC_CNT(cur_->exactTests);
        tw::Contact c;
        if (tw::ExactTest(p.sm, tr, &c) != tw::kNone) {
            CC_CNT(cur_->exactHits);
            out.push_back(Emitted{c, mb.tri, static_cast<std::uint8_t>(tr.meta & tw::kMetaMaterialMask), static_cast<std::uint8_t>(c.kind == tw::kFace)});
        }
    }
}

CarCache::Eval CarCache::EvalPatch(Pose &p, std::uint32_t pid, std::int32_t &hint, std::vector<Emitted> &out, int shape, const float rate[4],
                                   Cert *cert) {
    busy_ = true; busyBits_ |= (shape < 4 ? 32u : 8u);
    cert->m = 0.0f;
    const Patch &pa = pw->patches[pid];
    CC_TOUCH(&pa, sizeof(Patch), kTouchHot);
    CC_CNT(cur_->trackedEvals);
    (void)shape;
    if (pa.flags & kPatchExactOnly) {
        CC_CNT(cur_->fbExactOnly); CC_CNT(cur_->perShapeFallback[shape]);
        ExactMembers(p, pa, out);
        return kContact;
    }
    const float dt[3] = {p.t[0] - pa.ref[0], p.t[1] - pa.ref[1], p.t[2] - pa.ref[2]};
    const float sd = Dot3(pa.n, dt);
    float Mn[3];
    MtV(p.M, pa.n, Mn);
    const float H2 = Dot3(Mn, Mn);
    {
        const float dc = (m::abs(p.t[0] - pa.cen[0]) + m::abs(p.t[1] - pa.cen[1])) + m::abs(p.t[2] - pa.cen[2]);
        const float behind = -sd - (pa.tau + pa.alpha * (dc + pa.rad)) - 1e-5f;
        if (behind > 0.0f) { CC_CNT(cur_->emptyBehind); MakeCert(pa.n, behind * 0.99f, rate, cert); return kEmptyBehind; }
    }
    const float far = sd - pa.tau - 1e-5f;
    if (far > 0.0f && far * far > H2 * 1.00001f) {
        CC_CNT(cur_->emptyFar);
        MakeCert(pa.n, (far * far - H2) / (far + m::sqrt(H2)) * 0.9999f, rate, cert);  // = far - H without cancellation
        return kEmptyFar;
    }
    if (!(sd > pa.tau + 1e-5f)) {  // centre inside the slab: borderline for FV's back-face cull
        CC_CNT(cur_->fbBorder); CC_CNT(cur_->perShapeFallback[shape]);
        ExactMembers(p, pa, out);
        return kContact;
    }
    // reached: contact point on the patch plane, contact-disk radius bound
    const float H = m::sqrt(H2);
    float MMn[3];
    MV(p.M, Mn, MMn);
    const float kk = sd / H2;
    const float P[3] = {p.t[0] - kk * MMn[0], p.t[1] - kk * MMn[1], p.t[2] - kk * MMn[2]};
    const float dlo = std::max(0.0f, (sd - pa.tau) / H);
    const float rho = p.rmax * m::sqrt(std::max(0.0f, 1.0f - dlo * dlo)) * 1.0001f + pa.tau * p.rmax / H + 2e-4f;
    // walk the member triangles towards P
    std::int32_t mi = hint >= 0 ? hint : static_cast<std::int32_t>(pa.firstMember);
    bool found = false;
    for (int step = 0; step < 48; ++step) {
        const Member &mb = pw->members[static_cast<std::size_t>(mi)];
        CC_TOUCH(&mb, sizeof(Member), kTouchHot);
        CC_CNT(cur_->walkSteps);
        float smax = -1e30f;
        int imax = 0;
        for (int i = 0; i < 3; ++i) {
            const float d[3] = {P[0] - mb.a[i][0], P[1] - mb.a[i][1], P[2] - mb.a[i][2]};
            const float si = Dot3(mb.u[i], d);
            if (si > smax) { smax = si; imax = i; }
        }
        if (smax <= 0.0f) { found = true; break; }
        if (mb.nb[imax] < 0) {  // P beyond the convex outline
            *cert = Certify(p, pa, rate);
            if (cert->m > 0.0f || smax > rho) { CC_CNT(cur_->emptyOutside); return kEmptyOutside; }
            CC_CNT(cur_->fbOutside); CC_CNT(cur_->perShapeFallback[shape]);
            ExactMembers(p, pa, out);
            return kContact;
        }
        mi = mb.nb[imax];
    }
    if (!found) {
        CC_CNT(cur_->fbWalk); CC_CNT(cur_->perShapeFallback[shape]);
        ExactMembers(p, pa, out);
        return kContact;
    }
    hint = mi;
    // FACE of member T in closed form on T's own plane (FV: P = M p' + t, N = n, Sep = M n'(d - 1), K = n')
    const Member &T = pw->members[static_cast<std::size_t>(mi)];
    const float dT[3] = {p.t[0] - T.v0[0], p.t[1] - T.v0[1], p.t[2] - T.v0[2]};
    const float sdT = Dot3(T.n, dT);
    float MnT[3];
    MtV(p.M, T.n, MnT);
    const float H2T = Dot3(MnT, MnT);
    if (!(sdT > 1e-5f) || sdT * sdT > H2T * 0.9997f) {  // borderline reach (tangent, d > 0.99985): FV's own d is noisy there
        CC_CNT(cur_->fbMember); CC_CNT(cur_->perShapeFallback[shape]);
        ExactMembers(p, pa, out);
        return kContact;
    }
    float MMnT[3];
    MV(p.M, MnT, MMnT);
    const float kT = sdT / H2T;
    const float PT[3] = {p.t[0] - kT * MMnT[0], p.t[1] - kT * MMnT[1], p.t[2] - kT * MMnT[2]};
    float smaxT = -1e30f;
    for (int i = 0; i < 3; ++i) {
        const float d[3] = {PT[0] - T.a[i][0], PT[1] - T.a[i][1], PT[2] - T.a[i][2]};
        smaxT = std::max(smaxT, Dot3(T.u[i], d));
    }
    if (!(smaxT < -2e-5f)) {  // P on T's boundary within float noise
        CC_CNT(cur_->fbMember); CC_CNT(cur_->perShapeFallback[shape]);
        ExactMembers(p, pa, out);
        return kContact;
    }
    const float HT = m::sqrt(H2T), ihT = 1.0f / HT, dd = sdT * ihT;
    if (-smaxT >= rho) {
        CC_CNT(cur_->faceClosed);
    } else {
        // contact disk reaches T's edges: coplanar features of the neighbours are possible
        EnsureSm(p);
        if (!FeaturesDropped(p.sm, T.n, MnT, HT, dd)) {
            CC_CNT(cur_->fbDeep); CC_CNT(cur_->perShapeFallback[shape]);
            ExactMembers(p, pa, out);
            return kContact;
        }
        CC_CNT(cur_->faceAngle);
    }
    Emitted e;
    e.c.kind = tw::kFace;
    const float ks = (sdT - HT) / H2T;
    for (int i = 0; i < 3; ++i) {
        e.c.P[i] = PT[i];
        e.c.N[i] = T.n[i];
        e.c.Sep[i] = ks * MMnT[i];
        e.c.K[i] = MnT[i] * ihT;
    }
    e.tri = T.tri;
    e.material = static_cast<std::uint8_t>(T.meta & tw::kMetaMaterialMask);
    e.primary = 1;
    out.push_back(e);
    return kContact;
}

void CarCache::ProxyPose(const tw::Substep &sub, Pose *p) const {
    proxy::BodyBox b;
    for (int i = 0; i < 3; ++i) { b.c[i] = proxyC[i]; b.h[i] = proxyH[i]; b.hc[i] = proxyHC[i]; }
    proxy::BoxPose bp;
    proxy::WorldPose(b, sub.carPos, sub.carRot, &bp);
    for (int i = 0; i < 3; ++i) {
        p->t[i] = bp.t[i];
        for (int j = 0; j < 3; ++j) { p->M[i][j] = bp.M[i][j]; p->cM[i][j] = bp.cM[i][j]; }
    }
    p->boxValid = false;
    p->rmax = std::max(proxyH[0], std::max(proxyH[1], proxyH[2]));
    p->rmin = std::min(proxyH[0], std::min(proxyH[1], proxyH[2]));
    p->smValid = false;
    p->isBox = true;
    p->src = nullptr;
}

CarCache::Eval CarCache::EvalProxyInto(const Pose &bx, std::uint32_t pid, Pose *const *bodies, std::vector<Emitted> *const *raw, const float rate[4], Cert *cert,
                                       const tw::Substep *lazy, bool *valid) {
    const Patch &pa = pw->patches[pid];
    CC_TOUCH(&pa, sizeof(Patch), kTouchHot);
    CC_CNT(cur_->proxyEvals);
    busy_ = true; busyBits_ |= 16u;
    cert->m = 0.0f;
    if (!(pa.flags & kPatchExactOnly)) {
        const float dt[3] = {bx.t[0] - pa.ref[0], bx.t[1] - pa.ref[1], bx.t[2] - pa.ref[2]};
        const float sd = Dot3(pa.n, dt);
        const float dc = (m::abs(bx.t[0] - pa.cen[0]) + m::abs(bx.t[1] - pa.cen[1])) + m::abs(bx.t[2] - pa.cen[2]);
        const float behind = -sd - CentreSup(bx, pa.n) - (pa.tau + pa.alpha * (dc + pa.rad)) - 1e-5f;  // every body centre behind
        if (behind > 0.0f) { CC_CNT(cur_->proxyCertified); MakeCert(pa.n, behind * 0.99f, rate, cert); return kEmptyBehind; }
        const float far = sd - pa.tau - Sup(bx, pa.n) - 1e-5f;
        if (far > 0.0f) { CC_CNT(cur_->proxyCertified); MakeCert(pa.n, far, rate, cert); return kEmptyFar; }
        if (!*valid) {
            for (int k = 0; k < 4; ++k) SetupPose(lazy->shapes[k], bodies[k]);
            *valid = true;
        }
        if (BodiesCertified(bodies, pid)) { CC_CNT(cur_->proxyBodySlab); return kEmptyFar; }
        *cert = Certify(bx, pa, rate);
        if (cert->m > 0.0f) { CC_CNT(cur_->proxyCertified); return kEmptyOutside; }
    }
    if (!*valid) {  // body shape poses only when the proxy cannot certify
        for (int k = 0; k < 4; ++k) SetupPose(lazy->shapes[k], bodies[k]);
        *valid = true;
    }
    if (BodiesCertified(bodies, pid)) { CC_CNT(cur_->proxyBodySlab); return kEmptyFar; }
    CC_CNT(cur_->proxyExpand);
    for (int k = 0; k < 4; ++k) {
        if (!tw::Overlap(bodies[k]->B(), pa.box)) continue;
        std::int32_t h = -1;
        Cert c;
        EvalPatch(*bodies[k], pid, h, *raw[k], k, rate, &c);
    }
    return kContact;
}

void CarCache::Reset() {
    init_ = false;
    for (bool &v : primValid_) v = false;
    car_.Clear();
    for (int a = 0; a < 4; ++a) carO_[a] = 0.0f;
    for (float &r : carRate_) r = 0.0f;
    carBudget_ = 0.0f;
    holders_.Clear();
    queriesSinceSweep_ = 0;
    heldAtSweep_ = 0;
    havePrevQ_ = false;
    for (ShapeState &S : s) {  // all slots
        for (int a = 0; a < 4; ++a) S.O[a] = S.Oq[a] = S.rate[a] = 0.0f;
        S.sleep.Clear(); S.tracked.clear(); S.raw.clear(); S.merged.clear();
        for (auto &v : S.sl) v.clear();
    }
}

void CarCache::KOf(KSet out[kShapes]) const {
    for (int k = 0; k < kShapes; ++k) {  // per car shape (bodies 0-3 too)
        out[k] = KSet{};
        for (const Emitted &e : s[k].merged) KAdd(out[k], e.material);
    }
}

void CarCache::Dump(int k) {
    if (const char *ft = std::getenv("CC_FINDTRI")) {
        const std::uint32_t pid = pw->triPatch[std::strtoul(ft, nullptr, 10)];
        int where = 0;
        for (std::size_t i = 0; i < car_.n; ++i) where |= car_.pid[i] == pid ? 1 : 0;
        std::printf("  FIND patch %u (slot %d): car %d", pid, k, where);
        for (int q = 4; q < kSlots; ++q) {
            int sl = 0, tr = 0;
            for (std::size_t i = 0; i < s[q].sleep.n; ++i) sl |= s[q].sleep.pid[i] == pid ? 1 : 0;
            for (const Track &t : s[q].tracked) tr |= t.patch == pid ? 1 : 0;
            std::printf(" | slot %d sleep %d tracked %d", q, sl, tr);
        }
        std::printf(" | holders new? %d\n", holders_.Insert(pid) ? 1 : 0);
        return;
    }
    ShapeState &S = s[k];
    std::printf("  shape %d t (%.3f %.3f %.3f) r %.3f..%.3f O (%.3f %.3f %.3f | %.4f) rate (%.3f %.3f %.3f | %.4f): %zu tracked, %zu sleeping\n", k,
                S.pose.t[0], S.pose.t[1], S.pose.t[2], S.pose.rmin, S.pose.rmax, S.O[0], S.O[1], S.O[2], S.O[3], S.rate[0], S.rate[1], S.rate[2], S.rate[3],
                S.tracked.size(), S.sleep.n);
    auto line = [&](const char *tag, std::uint32_t pid, float left) {
        const Patch &pa = pw->patches[pid];
        const Cert c = Certify(S.pose, pa, S.rate);
        std::printf("    %s p%u mem %u edges %u mat %u%s n (%.3f %.3f %.3f) box ext (%.2f %.2f %.2f) cert m %.3f w (%.2f %.2f %.2f) score %.1f poses; left %.3f\n", tag, pid,
                    pa.memberCount, pa.edgeCount, pa.meta & 0xff, (pa.meta & tw::kMetaTrigger) ? " trig" : "", pa.n[0], pa.n[1], pa.n[2], pa.box.hi[0] - pa.box.lo[0],
                    pa.box.hi[1] - pa.box.lo[1], pa.box.hi[2] - pa.box.lo[2], c.m, c.w[0], c.w[1], c.w[2], c.score, left);
    };
    for (const Track &t : S.tracked) line("T", t.patch, 0.0f);
    for (std::size_t i = 0; i < S.sleep.n; ++i) {
        const float w[3] = {S.sleep.wx[i], S.sleep.wy[i], S.sleep.wz[i]};
        const float left = S.sleep.base[i] - Lhs(w, S.O);
        if (left < 0.5f) line("S", S.sleep.pid[i], left);
    }
}


void CarCache::MicroBench(double ghz) {
    (void)ghz;
#if CC_SPLIT || CC_MICROB
    if (lazySub_) {  // whole Step / probe on the same pose again (no motion: no expiry, no query)
        const tw::Substep sub = *lazySub_;
        CarCache tmp = *this;
        KSet K0[kShapes];
        tmp.Step(sub);
        tmp.KOf(K0);
        const int iters = std::getenv("CC_ITERS") ? std::atoi(std::getenv("CC_ITERS")) : 5000;
        std::uint64_t a = __rdtsc();
        for (int r = 0; r < iters; ++r) tmp.Step(sub);
        const double st1 = double(__rdtsc() - a) / iters;
        a = __rdtsc();
        for (int r = 0; r < 5000; ++r) tmp.ProbeDiffers(sub, K0, nullptr);
        const double pr = double(__rdtsc() - a) / 5000;
        std::size_t tr = 0;
        for (int k = 4; k < kSlots; ++k) tr += tmp.s[k].tracked.size();
        for (unsigned bit : {1u, 2u, 4u, 8u, 16u}) {
            CarCache t2 = tmp;
            t2.dbgSkip_ = bit;
            std::uint64_t a2 = __rdtsc();
            for (int r = 0; r < iters; ++r) t2.Step(sub);
            std::printf("  micro: Step skipping part %u: %.0f TSC\n", bit, double(__rdtsc() - a2) / iters);
        }
        std::printf("  micro state:");
        for (int k = 4; k < kSlots; ++k) std::printf(" slot %d: %zu sleepers budget %.3g, %zu tracked;", k, tmp.s[k].sleep.n, double(tmp.s[k].budget), tmp.s[k].tracked.size());
        std::printf(" car %zu budget %.3g\n", tmp.car_.n, double(tmp.carBudget_));
        std::printf("  micro: Step on a repeated pose %.0f TSC, probe %.0f TSC (%zu tracked, car level %zu)\n", st1, pr, tr, tmp.car_.n);
#if CC_FINE
        {
            for (auto &x : g_fine) x = 0;
            for (int r = 0; r < 5000; ++r) tmp.Step(sub);
            std::uint64_t e0 = __rdtsc(), e1 = e0;
            for (int r = 0; r < 5000; ++r) { const std::uint64_t q = __rdtsc(); e1 += __rdtsc() - q; }
            const double ovh = double(e1 - e0) / 5000;  // one rdtsc pair
            const char *names[10] = {"teleport check + rotation delta", "4 wheel poses + odometers", "proxy pose + odometers", "car level + horizon check",
                                     "query", "car-level expiry", "slot expiry checks", "wheel FACE kernel", "other tracked (proxy, full path)", "merge (8 shapes)"};
            std::printf("  fine (TSC per Step, rdtsc pair %.0f subtracted per segment):", ovh);
            double sum = 0;
            for (int i = 0; i < 10; ++i) { const double v = double(g_fine[i]) / 5000 - ovh; sum += v; std::printf(" %s %.0f;", names[i], v); }
            std::printf(" sum %.0f\n", sum);
            std::printf("\n");
        }
#endif
    }
    for (int k = 4; k < kSlots; ++k) {
        ShapeState &S = s[k];
        if (S.sleep.n == 0 && S.tracked.empty()) continue;
        const int reps = 20000;
        float acc = 0;
        std::uint64_t a = __rdtsc();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < S.sleep.n; ++i) acc += Certify(S.pose, pw->patches[S.sleep.pid[i]], S.rate).m;
        const double cert = double(__rdtsc() - a) / (double(reps) * std::max<std::size_t>(1, S.sleep.n));
        std::vector<Emitted> out;
        a = __rdtsc();
        for (int r = 0; r < reps; ++r)
            for (Track &t : S.tracked) {
                if (k == kProxy) continue;
                Cert c; std::int32_t h = t.member; out.clear(); EvalPatch(S.pose, t.patch, h, out, k, S.rate, &c);
            }
        const double ev = double(__rdtsc() - a) / (double(reps) * std::max<std::size_t>(1, S.tracked.size()));
        a = __rdtsc();
        const float budget = S.budget;
        for (int r = 0; r < reps; ++r) { Expire(k); }
        S.budget = budget;
        const double ex = double(__rdtsc() - a) / reps;
        if (k == 4) {
            CarCache tmp = *this;
            std::uint64_t q0 = __rdtsc();
            for (int r = 0; r < 2000; ++r) tmp.Query();
            const double qq = double(__rdtsc() - q0) / 2000;
            std::size_t cnt = 0;
            Box3 q = tw::EmptyBox();
            for (int kk = 4; kk < kSlots; ++kk) {
                Box3 b = s[kk].pose.B();
                for (int a = 0; a < 3; ++a) { b.lo[a] -= qL_[a]; b.hi[a] += qL_[a]; }
                q = tw::Union(q, b);
            }
            q0 = __rdtsc();
            for (int r = 0; r < 2000; ++r) pw->bvh.Query(q, [&](std::uint32_t) { ++cnt; });
            const double bq = double(__rdtsc() - q0) / 2000;
            q0 = __rdtsc();
            for (int r = 0; r < 2000; ++r) { Pose pp; for (int kk = 4; kk < kShapes; ++kk) SetupPose(*s[kk].pose.src, &pp); acc += pp.t[0]; }
            const double sp = double(__rdtsc() - q0) / 2000;
            std::printf("  micro: Query %.0f TSC (BVH part %.0f, %zu patches, qL %.2f %.2f %.2f); SetupPose x8 %.0f TSC\n", qq, bq, cnt / 2000, qL_[0], qL_[1], qL_[2], sp);
        }
        std::printf("  micro shape %d: %zu sleepers, %zu tracked: Certify %.0f TSC each, EvalPatch %.0f TSC each, Expire scan %.0f TSC (acc %g)\n", k, S.sleep.n,
                    S.tracked.size(), cert, ev, ex, double(acc));
    }
#endif
}

void CarCache::Distribute(std::uint32_t pid) {
    const Patch &pa = pw->patches[pid];
    CC_CNT(cur_->distributed);
    for (int k = 4; k < kSlots; ++k) {
        ShapeState &S = s[k];
        float w[3] = {0, 0, 0}, mg = 0.0f, bs = -1.0f;
        for (int a = 0; a < 3; ++a) {
            const float g = std::max(pa.box.lo[a] - S.pose.B().hi[a], S.pose.B().lo[a] - pa.box.hi[a]) - 1e-5f;
            const float rr = S.rate[a] + S.rate[3] + 1e-4f;
            if (g > 0.0f && g / rr > bs) { bs = g / rr; mg = g; w[0] = w[1] = w[2] = 0.0f; w[a] = 1.0f; }
        }
        S.sleep.Push(pid, w, Lhs(w, S.O) + mg - 3e-5f, pa.box);  // mg = 0: wakes at the slot's next scan
        S.budget = std::min(S.budget, mg * 0.999f - 1e-6f);
    }
}

void CarCache::Query() {
    busy_ = true; busyBits_ |= 1u;
    CC_CNT(cur_->queries);
    // per-axis inflation: Lmin plus Lposes poses of the recent motion along the axis (incl. rotation)
    float r[3] = {0, 0, 0};
    for (int k = 4; k < kSlots; ++k)
        for (int a = 0; a < 3; ++a) r[a] = std::max(r[a], std::min(1.5f, s[k].rate[a] + s[k].rate[3]));
    for (int a = 0; a < 3; ++a) qL_[a] = L + Lposes * r[a] + reserve;
    Box3 q = tw::EmptyBox();
    for (int k = 4; k < kSlots; ++k) {
        Box3 ib;
        const Box3 &b = s[k].pose.B();
        for (int a = 0; a < 3; ++a) { ib.lo[a] = b.lo[a] - qL_[a]; ib.hi[a] = b.hi[a] + qL_[a]; }
        q = tw::Union(q, ib);
    }
    qlist_.clear();
    // Only the part of q outside the previous query box can hold patches nobody holds (every patch overlapping
    // the previous box was found then, and sweeps drop only patches outside the box of their query): query q
    // minus the previous box as up to 6 slabs; the holder map removes duplicates.
    static const bool kNoDiff = std::getenv("CC_NODIFF") != nullptr;
    const std::uint64_t qa = CC_TSC();
    const bool overlapPrev = !kNoDiff && havePrevQ_ && tw::Overlap(q, prevQ_);
    if (!overlapPrev) {
        pw->bvh.Query(q, [&](std::uint32_t pid) { qlist_.push_back(pid); });
    } else {
        Box3 rest = q;
        for (int a = 0; a < 3; ++a) {
            if (rest.lo[a] < prevQ_.lo[a]) {
                Box3 sl = rest;
                sl.hi[a] = prevQ_.lo[a];
                pw->bvh.Query(sl, [&](std::uint32_t pid) { qlist_.push_back(pid); });
                rest.lo[a] = prevQ_.lo[a];
            }
            if (rest.hi[a] > prevQ_.hi[a]) {
                Box3 sl = rest;
                sl.lo[a] = prevQ_.hi[a];
                pw->bvh.Query(sl, [&](std::uint32_t pid) { qlist_.push_back(pid); });
                rest.hi[a] = prevQ_.hi[a];
            }
        }
    }
    prevQ_ = q;
    havePrevQ_ = true;
    const std::uint64_t qb = CC_TSC();
    g_split[4] += qb - qa;
    CC_ADD(cur_->queryPatches, qlist_.size());
    // Invariant: every patch held is either at the car level or in all 5 slots (sleeping or tracked), so a
    // queried patch is new exactly when the holder map does not know it. Entries outside the query box are
    // dropped lazily by a sweep (every 8th query or when the lists grow), which also rebuilds the map.
    std::size_t held = car_.n;
    for (int k = 4; k < kSlots; ++k) held += s[k].sleep.n;
    if (++queriesSinceSweep_ >= 8 || held > 2 * heldAtSweep_ + 64 || holders_.key.empty() ||
        float(holders_.used + qlist_.size()) > 0.5f * float(holders_.key.size())) {
        const std::uint64_t sa = CC_TSC();
        Sweep(q);
        g_split[5] += CC_TSC() - sa;
        ++g_split[7];
    }
    ++g_split[6];
    for (std::uint32_t pid : qlist_) {
        if (!holders_.Insert(pid)) continue;  // already held somewhere
        const Patch &pa = pw->patches[pid];
        CC_TOUCH(&pa, sizeof(Patch), kTouchQuery);
        float w[3] = {0, 0, 0}, mg = 0.0f, bs = -1.0f;
        for (int a = 0; a < 3; ++a) {
            const float g = std::max(pa.box.lo[a] - carBox_.hi[a], carBox_.lo[a] - pa.box.hi[a]) - 1e-5f;
            const float rr = carRate_[a] + 1e-4f;
            if (g > 0.0f && g / rr > bs) { bs = g / rr; mg = g; w[0] = w[1] = w[2] = 0.0f; w[a] = 1.0f; }
        }
        if (mg > 0.0f) {
            car_.Push(pid, w, Lhs(w, carO_) + mg - 3e-5f, pa.box);
            carBudget_ = std::min(carBudget_, (mg - 3e-5f) * 0.999f - 1e-6f);  // a new sleeper may wake first
        } else {
            Distribute(pid);  // already touches the car box
        }
    }
    for (int k = 4; k < kSlots; ++k)
        for (int a = 0; a < 4; ++a) s[k].Oq[a] = s[k].O[a];
}

// Drop entries outside the query box (from the car level, and from all 5 slots at once: the same box test
// on the same patch), rebuild the holder map, and re-base the odometers when they have grown past 16 m
// (float resolution of the certificate bases).
void CarCache::Sweep(const Box3 &q) {
    CC_CNT(cur_->sweeps);
    queriesSinceSweep_ = 0;
    {
        std::size_t total = car_.n + qlist_.size();
        for (int k = 4; k < kSlots; ++k) total += s[k].sleep.n + s[k].tracked.size();
        holders_.Reserve(total);
    }
    using V = m::vf<16>;
    auto sweepList = [&](SleepList &Lk) {
        removed_.clear();
        alignas(64) float keep[16];
        for (std::size_t b = 0; b < Lk.n; b += 16) {
            const auto out = m::cmp_gt(V::load(&Lk.lo[0][b]), V(q.hi[0])) | m::cmp_lt(V::load(&Lk.hi[0][b]), V(q.lo[0])) |
                             m::cmp_gt(V::load(&Lk.lo[1][b]), V(q.hi[1])) | m::cmp_lt(V::load(&Lk.hi[1][b]), V(q.lo[1])) |
                             m::cmp_gt(V::load(&Lk.lo[2][b]), V(q.hi[2])) | m::cmp_lt(V::load(&Lk.hi[2][b]), V(q.lo[2]));
            m::select(out, V(0.0f), V(1.0f)).store(keep);
            for (std::size_t i = b; i < std::min(Lk.n, b + 16); ++i)
                if (keep[i - b] == 0.0f) removed_.push_back(static_cast<std::uint32_t>(i));
        }
        for (std::size_t r2 = removed_.size(); r2-- > 0;) Lk.Remove(removed_[r2]);
        for (std::size_t i = 0; i < Lk.n; ++i) holders_.Insert(Lk.pid[i]);
    };
    auto rebase = [&](SleepList &Lk, float *O) {
        for (std::size_t i = 0; i < Lk.n; ++i) {
            const float w[3] = {Lk.wx[i], Lk.wy[i], Lk.wz[i]};
            Lk.base[i] -= Lhs(w, O) + 1e-6f;
        }
    };
    sweepList(car_);
    if (std::max(carO_[0], std::max(carO_[1], carO_[2])) > 16.0f) {
        rebase(car_, carO_);
        for (int a = 0; a < 4; ++a) carO_[a] = 0.0f;
    }
    for (int k = 4; k < kSlots; ++k) {
        ShapeState &S = s[k];
        sweepList(S.sleep);
        // tracked patches outside the box too (beyond every slot's horizon): a patch must leave all 5 slots at
        // once, or a slot that still holds it would hide it from the others at the next query
        for (std::size_t i = 0; i < S.tracked.size();) {
            if (!tw::Overlap(q, pw->patches[S.tracked[i].patch].box)) { TrackRemove(S, i); continue; }
            ++i;
        }
        for (const Track &t : S.tracked) holders_.Insert(t.patch);
        if (std::max(S.O[0], std::max(S.O[1], S.O[2])) + S.O[3] > 16.0f) {
            rebase(S.sleep, S.O);
            for (int a = 0; a < 4; ++a) { S.O[a] = 0.0f; S.Oq[a] = 0.0f; }
        }
        S.budget = 0.0f;  // rescan
    }
    carBudget_ = 0.0f;
    heldAtSweep_ = car_.n;
    for (int k = 4; k < kSlots; ++k) heldAtSweep_ += s[k].sleep.n;
}

void CarCache::ExpireCar() {
    busy_ = true; busyBits_ |= 2u;
    CC_CNT(cur_->carScans);
    SleepList &L = car_;
    using V = m::vf<16>;
    const V ox(carO_[0]), oy(carO_[1]), oz(carO_[2]);
    V mn(1e30f);
    removed_.clear();
    for (std::size_t b = 0; b < L.n; b += 16) {
        V base = V::load(&L.base[b]);
        const V lhs = (V::load(&L.wx[b]) * ox + V::load(&L.wy[b]) * oy) + V::load(&L.wz[b]) * oz;
        const auto exp = m::cmp_ge(lhs, base);
        if (m::any(exp)) {
            V bestS(-1.0f), bestM(0.0f), bwx(0.0f), bwy(0.0f), bwz(0.0f), bl(0.0f);
            for (int a = 0; a < 3; ++a) {
                const V g = m::max(V::load(&L.lo[a][b]) - V(carBox_.hi[a]), V(carBox_.lo[a]) - V::load(&L.hi[a][b])) - V(1e-5f);
                const V sc = g * V(1.0f / (carRate_[a] + 1e-4f));
                const auto better = m::cmp_gt(sc, bestS);
                bestS = m::select(better, sc, bestS);
                bestM = m::select(better, g, bestM);
                bwx = m::select(better, V(a == 0 ? 1.0f : 0.0f), bwx);
                bwy = m::select(better, V(a == 1 ? 1.0f : 0.0f), bwy);
                bwz = m::select(better, V(a == 2 ? 1.0f : 0.0f), bwz);
                bl = m::select(better, V(carO_[a]), bl);
            }
            const auto good = exp & m::cmp_ge(bestS, V(1.0f)) & m::cmp_gt(bestM, V(0.0f));
            base = m::select(good, bl + bestM - V(3e-5f), base);
            base.store(&L.base[b]);
            m::select(good, bwx, V::load(&L.wx[b])).store(&L.wx[b]);
            m::select(good, bwy, V::load(&L.wy[b])).store(&L.wy[b]);
            m::select(good, bwz, V::load(&L.wz[b])).store(&L.wz[b]);
            float rest[16];
            m::select(exp & ~good, V(1.0f), V(0.0f)).store(rest);
#if CC_STATS
            float gd[16];
            m::select(good, V(1.0f), V(0.0f)).store(gd);
            for (float x : gd) cur_->carRecerts += x != 0.0f;
#endif
            const std::size_t e = std::min(L.n, b + 16);
            for (std::size_t i = b; i < e; ++i) {
                if (rest[i - b] == 0.0f) continue;
                CC_CNT(cur_->carWakes);
                Distribute(L.pid[i]);
                removed_.push_back(static_cast<std::uint32_t>(i));
                L.base[i] = 1e30f;
            }
            const V lhs2 = (V::load(&L.wx[b]) * ox + V::load(&L.wy[b]) * oy) + V::load(&L.wz[b]) * oz;
            mn = m::min(mn, V::load(&L.base[b]) - lhs2);
        } else {
            mn = m::min(mn, base - lhs);
        }
    }
    std::vector<std::uint32_t> rm;
    rm.swap(removed_);
    for (std::size_t r = rm.size(); r-- > 0;) L.Remove(rm[r]);
    rm.swap(removed_);
    float tmp[16];
    mn.store(tmp);
    float b0 = 1e30f;
    for (float x : tmp) b0 = std::min(b0, x);
    carBudget_ = b0 * 0.999f - 1e-6f;
}

void CarCache::Expire(int k) {
    busy_ = true; busyBits_ |= 4u;
    ShapeState &S = s[k];
    SleepList &L = S.sleep;
    CC_CNT(cur_->expireScans);
    CC_TOUCH(L.base.data(), L.base.size() * 4, kTouchHot);
    CC_TOUCH(L.wx.data(), L.wx.size() * 4, kTouchHot);
    CC_TOUCH(L.wy.data(), L.wy.size() * 4, kTouchHot);
    CC_TOUCH(L.wz.data(), L.wz.size() * 4, kTouchHot);
    using V = m::vf<16>;
    const V ox(S.O[0]), oy(S.O[1]), oz(S.O[2]), om(S.O[3]);
    V mn(1e30f);
    removed_.clear();
    float lv[16];
    for (std::size_t b = 0; b < L.n; b += 16) {
        const V lhs = ((V::load(&L.wx[b]) * ox + V::load(&L.wy[b]) * oy) + V::load(&L.wz[b]) * oz) + om;
        const V base = V::load(&L.base[b]);
        const auto exp = m::cmp_ge(lhs, base);
        if (m::any(exp)) {
            // vector re-certification by the AABB gap along the axis predicted to last longest
            V bestS(-1.0f), bestM(0.0f), bwx(0.0f), bwy(0.0f), bwz(0.0f), bl(0.0f);
            for (int a = 0; a < 3; ++a) {
                const V g = m::max(V::load(&L.lo[a][b]) - V(S.pose.B().hi[a]), V(S.pose.B().lo[a]) - V::load(&L.hi[a][b])) - V(1e-5f);
                const V sc = g * V(1.0f / (S.rate[a] + S.rate[3] + 1e-4f));
                const auto better = m::cmp_gt(sc, bestS);
                bestS = m::select(better, sc, bestS);
                bestM = m::select(better, g, bestM);
                bwx = m::select(better, V(a == 0 ? 1.0f : 0.0f), bwx);
                bwy = m::select(better, V(a == 1 ? 1.0f : 0.0f), bwy);
                bwz = m::select(better, V(a == 2 ? 1.0f : 0.0f), bwz);
                bl = m::select(better, V(S.O[a]), bl);
            }
            const auto good = exp & m::cmp_ge(bestS, V(earlyOut_)) & m::cmp_gt(bestM, V(0.0f));
            m::select(good, (bl + om) + bestM - V(3e-5f), base).store(&L.base[b]);
            m::select(good, bwx, V::load(&L.wx[b])).store(&L.wx[b]);
            m::select(good, bwy, V::load(&L.wy[b])).store(&L.wy[b]);
            m::select(good, bwz, V::load(&L.wz[b])).store(&L.wz[b]);
#if CC_STATS
            {
                float gd[16];
                m::select(good, V(1.0f), V(0.0f)).store(gd);
                for (float x : gd) cur_->boxRecerts += x != 0.0f;
            }
#endif
            lhs.store(lv);
            float rest[16];
            m::select(exp & ~good, V(1.0f), V(0.0f)).store(rest);
            const std::size_t e = std::min(L.n, b + 16);
            for (std::size_t i = b; i < e; ++i) {
                if (rest[i - b] == 0.0f) continue;
                CC_CNT(cur_->marginEvals); CC_CNT(cur_->perShapeMargins[k]);
                const std::uint32_t pid = L.pid[i];
                CC_TOUCH(&L.pid[i], 4, kTouchHot);
                const Cert c = Certify(S.pose, pw->patches[pid], S.rate);
                // track: may touch, or the slab is the binding certificate and it would wake again soon
                if (c.m <= 0.0f || (c.kind == kCertPlane && c.score < promote)) {
                    CC_CNT(cur_->promotions);
                    TrackPush(S, pid);
                    removed_.push_back(static_cast<std::uint32_t>(i));
                    L.base[i] = 1e30f;  // out of the slack minimum below
                    continue;
                }
                L.base[i] = Lhs(c.w, S.O) + c.m - 3e-5f;
                L.wx[i] = c.w[0]; L.wy[i] = c.w[1]; L.wz[i] = c.w[2];
            }
            const V lhs2 = ((V::load(&L.wx[b]) * ox + V::load(&L.wy[b]) * oy) + V::load(&L.wz[b]) * oz) + om;
            mn = m::min(mn, V::load(&L.base[b]) - lhs2);
        } else {
            mn = m::min(mn, base - lhs);
        }
    }
    for (std::size_t r = removed_.size(); r-- > 0;) L.Remove(removed_[r]);  // descending: the swapped-in tail is never marked
    // smallest slack left: no sleeper can wake before |dO_xyz|_2 + dO_M grows by it (|w|_2 = 1)
    float tmp[16];
    mn.store(tmp);
    float b0 = 1e30f;
    for (float x : tmp) b0 = std::min(b0, x);
    S.budget = b0 * 0.999f - 1e-6f;
}

namespace {
// advance a slot's odometers from its old pose to np (support change of a box is <= sqrt(3) ||dM||_F)
inline void Advance(CarCache::ShapeState &S, const Pose &np, float mScale) {
    float f = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) { const float d = np.M[i][j] - S.pose.M[i][j]; f += d * d; }
    const float dM = m::sqrt(f) * (1.0001f * mScale) + 1e-7f;
    float d[3];
    for (int a = 0; a < 3; ++a) {
        d[a] = m::abs(np.t[a] - S.pose.t[a]) * 1.0001f + 1e-7f;
        S.O[a] += d[a];
        S.rate[a] = 0.5f * (S.rate[a] + d[a]);
    }
    S.O[3] += dM;
    S.rate[3] = 0.5f * (S.rate[3] + dM);
    S.budget -= m::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) * 1.0001f + dM;
}
}  // namespace

// ---- wheel FACE fast path -----------------------------------------------------------------------------
// One lane per (wheel, tracked patch) whose last evaluation located member T. FV's FACE on T's own plane is the
// patch's whole merged contribution when T is reached (0 < d < 0.99985), P lies inside T, and the contact disk
// of the ellipsoid with the slab around T's plane (half thickness tauP = 2 tau + alpha r_max) clears T's edges:
// every other member is interior-disjoint from T, so none can touch. Any lane that fails takes the full path.
// Same formulas as EvalPatch; one template for 4 lanes and for 1, so the bits do not depend on the width.
namespace {
using face::oOk; using face::oP; using face::oSep; using face::oK;
constexpr int kFaceMemb = face::kMemb, kFaceOut = face::kOut;
template <class F, bool kShared = false>
TW_INLINE void FaceKernel(const Pose *const *pz, const float (*mb)[4], float (*out)[4]) { face::Kernel<F, kShared>(pz, mb, out); }
inline void FaceMembLoad(float (*mb)[4], int lane, const Member &T, float tauP, float rmax) {
    face::LoadMember(mb, lane, T.v0, T.n, T.u, T.a, tauP, rmax);
}
inline void FaceEmit(const float (*out)[4], int l, const Member &T, Emitted *e) {
    e->c.kind = tw::kFace;
    for (int q = 0; q < 3; ++q) { e->c.P[q] = out[oP + q][l]; e->c.N[q] = T.n[q]; e->c.Sep[q] = out[oSep + q][l]; e->c.K[q] = out[oK + q][l]; }
    e->tri = T.tri;
    e->material = static_cast<std::uint8_t>(T.meta & tw::kMetaMaterialMask);
    e->primary = 1;
}
}  // namespace

int CarCache::FaceBatch(const FaceItem *items, int n, Emitted *res, bool *ok) {
    alignas(16) float mb[kFaceMemb][4], out[kFaceOut][4];
    for (int b = 0; b < n; b += 4) {
        const int w = std::min(4, n - b);
        const Pose *pz[4];
        for (int l = 0; l < 4; ++l) {
            const FaceItem &it = items[b + (l < w ? l : 0)];
            const Patch &pa = pw->patches[it.patch];
            const Member &T = pw->members[static_cast<std::size_t>(it.member)];
            pz[l] = it.pose;
            FaceMembLoad(mb, l, T, 2.0f * pa.tau + pa.alpha * it.pose->rmax, it.pose->rmax);
        }
        if (w == 1) FaceKernel<m::f32x1>(pz, mb, out); else FaceKernel<m::f32x4>(pz, mb, out);
        for (int l = 0; l < w; ++l) {
            ok[b + l] = out[oOk][l] != 0.0f;
            if (ok[b + l]) FaceEmit(out, l, pw->members[static_cast<std::size_t>(items[b + l].member)], &res[b + l]);
        }
    }
    return n;
}

void CarCache::TrackPush(ShapeState &S, std::uint32_t pid) {
    const std::size_t n = S.tracked.size();
    S.tracked.push_back(Track{pid, -1, 0u});
    if (n % 16 == 0)
        for (auto &v : S.sl) v.resize(n + 16, 0.0f);
    const Patch &pa = pw->patches[pid];
    const float rc = (std::fabs(pa.ref[0] - pa.cen[0]) + std::fabs(pa.ref[1] - pa.cen[1])) + std::fabs(pa.ref[2] - pa.cen[2]);
    const float v[9] = {pa.ref[0], pa.ref[1], pa.ref[2], pa.n[0], pa.n[1], pa.n[2], (pa.flags & kPatchExactOnly) ? 1e30f : pa.tau, pa.alpha,
                        (rc + pa.rad) * 1.0001f + 1e-4f};
    for (int f = 0; f < 9; ++f) S.sl[f][n] = v[f];
}

void CarCache::TrackRemove(ShapeState &S, std::size_t i) {
    const std::size_t last = S.tracked.size() - 1;
    S.tracked[i] = S.tracked[last];
    S.tracked.pop_back();
    for (auto &v : S.sl) v[i] = v[last];
    if (last % 16 == 0)
        for (auto &v : S.sl) v.resize(last);
}

void CarCache::SlabPass(ShapeState &S, const Pose &p, std::vector<std::uint8_t> &cert, std::vector<float> &mg, std::vector<float> &sc) {
    using V = m::vf<16>;
    const std::size_t n = S.tracked.size(), np = (n + 15) & ~std::size_t(15);
    if (cert.size() < np) { cert.resize(np); mg.resize(np); sc.resize(np); }
    const V tx(p.t[0]), ty(p.t[1]), tz(p.t[2]);
    const float ir[4] = {S.rate[0], S.rate[1], S.rate[2], S.rate[3] + 1e-4f};
    alignas(64) float cf[16];
    for (std::size_t b = 0; b < np; b += 16) {
        const V dx = tx - V::load(&S.sl[0][b]), dy = ty - V::load(&S.sl[1][b]), dz = tz - V::load(&S.sl[2][b]);
        const V nx = V::load(&S.sl[3][b]), ny = V::load(&S.sl[4][b]), nz = V::load(&S.sl[5][b]);
        const V tau = V::load(&S.sl[6][b]);
        const V sd = (nx * dx + ny * dy) + nz * dz;
        const V v0 = (V(p.M[0][0]) * nx + V(p.M[1][0]) * ny) + V(p.M[2][0]) * nz;
        const V v1 = (V(p.M[0][1]) * nx + V(p.M[1][1]) * ny) + V(p.M[2][1]) * nz;
        const V v2 = (V(p.M[0][2]) * nx + V(p.M[1][2]) * ny) + V(p.M[2][2]) * nz;
        const V dc = (m::abs(dx) + m::abs(dy)) + m::abs(dz);
        V behind = (V(0.0f) - sd) - (tau + V::load(&S.sl[7][b]) * (dc + V::load(&S.sl[8][b]))) - V(1e-5f);
        V far, farM;
        if (p.isBox) {
            const V c0 = (V(p.cM[0][0]) * nx + V(p.cM[1][0]) * ny) + V(p.cM[2][0]) * nz;
            const V c1 = (V(p.cM[0][1]) * nx + V(p.cM[1][1]) * ny) + V(p.cM[2][1]) * nz;
            const V c2 = (V(p.cM[0][2]) * nx + V(p.cM[1][2]) * ny) + V(p.cM[2][2]) * nz;
            behind = behind - ((m::abs(c0) + m::abs(c1)) + m::abs(c2));
            far = ((sd - tau) - ((m::abs(v0) + m::abs(v1)) + m::abs(v2))) - V(1e-5f);
            farM = far;
        } else {
            const V far0 = sd - tau - V(1e-5f), H2 = (v0 * v0 + v1 * v1) + v2 * v2;
            const auto ok = m::cmp_gt(far0, V(0.0f)) & m::cmp_gt(far0 * far0, H2 * V(1.00001f));
            farM = ((far0 * far0 - H2) / (far0 + m::sqrt(H2))) * V(0.9999f);  // = far0 - H without cancellation
            far = m::select(ok, farM, V(-1.0f));
        }
        const auto cb = m::cmp_gt(behind, V(0.0f)), cfm = m::cmp_gt(far, V(0.0f));
        const V margin = m::select(cb, behind * V(0.99f), farM);
        const V rate = ((m::abs(nx) * V(ir[0]) + m::abs(ny) * V(ir[1])) + m::abs(nz) * V(ir[2])) + V(ir[3]);
        m::select(cb | cfm, V(1.0f), V(0.0f)).store(cf);
        margin.store(&mg[b]);
        (margin / rate).store(&sc[b]);
        for (int l = 0; l < 16; ++l) cert[b + l] = cf[l] != 0.0f;
    }
}

// The 4 body ellipsoids against one patch slab (4 lanes): true when every body is behind every member plane
// or out of the slab's reach (the same test EvalPatch starts with), so the proxy needs no body evaluation.
bool CarCache::BodiesCertified(Pose *const *bodies, std::uint32_t pid) const {
    const Patch &pa = pw->patches[pid];
    if (pa.flags & kPatchExactOnly) return false;
    return proxy::BodiesSlab(bodies, pa.n, pa.ref, pa.tau, pa.alpha, pa.cen, pa.rad);
}

void CarCache::SetPrimary(int lane, std::uint32_t patch, std::int32_t member) {
    const Patch &pa = pw->patches[patch];
    const Member &T = pw->members[static_cast<std::size_t>(member)];
    const float rmax = s[4 + lane].pose.rmax;
    FaceMembLoad(primM_, lane, T, 2.0f * pa.tau + pa.alpha * rmax, rmax);
    primPatch_[lane] = patch;
    primMember_[lane] = member;
    primValid_[lane] = true;
}

// Steady state: one FACE kernel call for the 4 wheels' primary patches. Fills out[][] and returns the lane mask.
unsigned CarCache::PrimaryKernel(const Pose *const *poses, float (*out)[4]) {
    unsigned any = 0;
    for (int l = 0; l < 4; ++l) any |= primValid_[l] ? 1u << l : 0u;
    if (!any) return 0;
    if (sharedWheelM_ && poses[0] == &s[4].pose) FaceKernel<m::f32x4, true>(poses, primM_, out);
    else FaceKernel<m::f32x4>(poses, primM_, out);
    unsigned okm = 0;
    for (int l = 0; l < 4; ++l) if (((any >> l) & 1u) && out[oOk][l] != 0.0f) okm |= 1u << l;
    return okm;
}

void CarCache::WheelPose(const tw::Shape &sh, Pose *p) {
    for (int i = 0; i < 3; ++i) p->t[i] = sh.t[i];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) p->M[i][j] = sh.axis[j][i] * sh.r[j];
    p->boxValid = false;
    p->smValid = false;
    p->isBox = false;
    p->src = &sh;
}

void CarCache::Step(const tw::Substep &sub) {
    CC_FT0();
    busy_ = false;
    busyBits_ = 0;
    CC_CNT(cur_->poses);
    ++poseCount_;
    if (init_) {  // teleport (respawn, replay jump): start over
        const float *a = sub.carPos, *b = lastCarPos_;
        if (m::abs(a[0] - b[0]) + m::abs(a[1] - b[1]) + m::abs(a[2] - b[2]) > 8.0f) { CC_CNT(cur_->resets); Reset(); }
    }
    for (int i = 0; i < 3; ++i) lastCarPos_[i] = sub.carPos[i];
    bool need = !init_;
    const std::uint64_t t0 = CC_TSC();
    if (!init_) {  // body proxy box in the car frame (proxy::Build): the 4 body ellipsoids in the car frame
        const float *R = sub.carRot;
        float cen[4][3], ax[4][3][3], rr[4][3];
        for (int k = 0; k < 4; ++k) {
            const tw::Shape &sh = sub.shapes[k];
            for (int i = 0; i < 3; ++i) {  // car-frame axis i = R column i
                cen[k][i] = (R[3 * i] * (sh.t[0] - sub.carPos[0]) + R[3 * i + 1] * (sh.t[1] - sub.carPos[1])) + R[3 * i + 2] * (sh.t[2] - sub.carPos[2]);
                for (int j = 0; j < 3; ++j) ax[k][j][i] = (R[3 * i] * sh.axis[j][0] + R[3 * i + 1] * sh.axis[j][1]) + R[3 * i + 2] * sh.axis[j][2];
                rr[k][i] = sh.r[i];
            }
        }
        const proxy::BodyBox b = proxy::Build(cen, ax, rr);
        for (int i = 0; i < 3; ++i) { proxyC[i] = b.c[i]; proxyH[i] = b.h[i]; proxyHC[i] = b.hc[i]; }
        sharedWheelM_ = true;
        for (int k = 4; k < kShapes; ++k) {
            const float *r = sub.shapes[k].r;
            s[k].pose.rmax = std::max(r[0], std::max(r[1], r[2]));
            s[k].pose.rmin = std::min(r[0], std::min(r[1], r[2]));
            for (int j = 0; j < 3; ++j) {
                sharedWheelM_ = sharedWheelM_ && r[j] == sub.shapes[4].r[j];
                for (int i = 0; i < 3; ++i) sharedWheelM_ = sharedWheelM_ && sub.shapes[k].axis[j][i] == sub.carRot[3 * j + i];
            }
        }
    }
    lazySub_ = &sub;  // bodies: pose only, computed when the proxy expands (EvalProxyInto)
    bodiesValid_ = false;
    // car rotation change per column: the wheel and proxy axes are the car's columns scaled (rigid in the car
    // frame), so ||dM||_F = sqrt(sum_j r_j^2 |dR_j|^2) for every slot from one 3x3 difference
    float cj[3] = {0, 0, 0};
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) { const float d = sub.carRot[3 * j + i] - prevRot_[3 * j + i]; cj[j] += d * d; }
    }
    for (int q = 0; q < 9; ++q) prevRot_[q] = sub.carRot[q];
#if CC_STATS
    for (int k = 4; k < kShapes; ++k)  // sanity for the shared rotation: wheel axes are the car's columns
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                if (m::abs(sub.shapes[k].axis[j][i] - sub.carRot[3 * j + i]) > 1e-6f) CC_CNT(cur_->axisMismatch);
#endif
    CC_FT(0);
    float inc[3] = {0, 0, 0};
    if (!(dbgSkip_ & 1u)) {
        // The wheel axes are the car's columns (bit-exact, checked in stats builds) and the 4 wheels share radii,
        // so M and ||dM||_F are the same for all 4: computed once (per wheel otherwise).
        const tw::Shape &s4 = sub.shapes[4];
        float Mw[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) Mw[i][j] = sub.carRot[3 * j + i] * s4.r[j];
        const float dMw = m::sqrt((s4.r[0] * s4.r[0] * cj[0] + s4.r[1] * s4.r[1] * cj[1]) + s4.r[2] * s4.r[2] * cj[2]) * 1.0002f + 1e-7f;
        for (int k = 4; k < kShapes; ++k) {
            ShapeState &S = s[k];
            const tw::Shape &sh = sub.shapes[k];
            CC_TOUCH(&S, sizeof(Pose) + 64, kTouchHot);
            if (init_) {
                float d[3];
                for (int a = 0; a < 3; ++a) {
                    d[a] = m::abs(sh.t[a] - S.pose.t[a]) * 1.0001f + 1e-7f;
                    S.O[a] += d[a];
                    S.rate[a] = 0.5f * (S.rate[a] + d[a]);
                    inc[a] = std::max(inc[a], d[a] + dMw);
                }
                S.O[3] += dMw;
                S.rate[3] = 0.5f * (S.rate[3] + dMw);
                S.budget -= ((d[0] + d[1]) + d[2]) + dMw;  // |w.d| <= |d|_2 <= |d|_1: a looser budget, no square root
            }
            Pose &p = S.pose;
            for (int i = 0; i < 3; ++i) p.t[i] = sh.t[i];
            if (sharedWheelM_) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) p.M[i][j] = Mw[i][j];
                p.boxValid = false;
                p.smValid = false;
                p.isBox = false;
                p.src = &sh;
            } else {
                WheelPose(sh, &p);
            }
        }
    }
    CC_FT(1);
    if (!(dbgSkip_ & 2u)) {
        ShapeState &S = s[kProxy];
        const float *R = sub.carRot;
        if (init_) {
            const float dM = m::sqrt((proxyH[0] * proxyH[0] * cj[0] + proxyH[1] * proxyH[1] * cj[1]) + proxyH[2] * proxyH[2] * cj[2]) * (1.0002f * 1.7321f) + 1e-7f;
            float d[3];
            for (int a = 0; a < 3; ++a) {
                const float nt = sub.carPos[a] + ((R[a] * proxyC[0] + R[3 + a] * proxyC[1]) + R[6 + a] * proxyC[2]);
                d[a] = m::abs(nt - S.pose.t[a]) * 1.0001f + 1e-7f;
                S.O[a] += d[a];
                S.rate[a] = 0.5f * (S.rate[a] + d[a]);
                inc[a] = std::max(inc[a], d[a] + dM);
            }
            S.O[3] += dM;
            S.rate[3] = 0.5f * (S.rate[3] + dM);
            S.budget -= m::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) * 1.0001f + dM;
        }
        ProxyPose(sub, &S.pose);
#if CC_STATS
        for (int k = 0; k < 4; ++k) {  // sanity: each body ellipsoid's exact car-frame AABB lies inside the proxy box
            const tw::Shape &sh = sub.shapes[k];
            for (int i = 0; i < 3; ++i) {
                const float c = (R[3 * i] * (sh.t[0] - sub.carPos[0]) + R[3 * i + 1] * (sh.t[1] - sub.carPos[1])) + R[3 * i + 2] * (sh.t[2] - sub.carPos[2]);
                float h2 = 0;
                for (int j = 0; j < 3; ++j) {
                    const float a = ((R[3 * i] * sh.axis[j][0] + R[3 * i + 1] * sh.axis[j][1]) + R[3 * i + 2] * sh.axis[j][2]) * sh.r[j];
                    h2 += a * a;
                }
                const float h = m::sqrt(h2);
                if (c - h < proxyC[i] - proxyH[i] || c + h > proxyC[i] + proxyH[i]) CC_CNT(cur_->proxyViolations);
            }
        }
#endif
    }
    CC_FT(2);
    // car level: largest slot increment per axis (rotation folded in)
    if (init_) {
        float mx = 0;
        for (int a = 0; a < 3; ++a) {
            const float d = inc[a] * 1.0001f + 1e-7f;
            carO_[a] += d;
            carRate_[a] = 0.5f * (carRate_[a] + d);
            mx = std::max(mx, d);
        }
        carBudget_ -= mx;
    }
    for (int k = 4; k < kSlots; ++k)
        for (int a = 0; a < 3; ++a) need = need || (s[k].O[a] - s[k].Oq[a]) + (s[k].O[3] - s[k].Oq[3]) + reserve >= qL_[a];
    init_ = true;
    CC_FT(3);
    const std::uint64_t t1 = CC_TSC();
    if (need) { CarBox(); Query(); }
    CC_FT(4);
    const std::uint64_t t2 = CC_TSC();
    std::uint64_t tExp = 0;
    for (int k = 0; k < kShapes; ++k) s[k].raw.clear();
    if (car_.n && carBudget_ <= 0.0f) { CarBox(); ExpireCar(); }
    CC_ADD(cur_->carHeld, car_.n);
    CC_FT(5);
    for (int k = 4; k < kSlots; ++k) {
        ShapeState &S = s[k];
        if (S.sleep.n && S.budget <= 0.0f) {
            const std::uint64_t e0 = CC_TSC();
            Expire(k);
            tExp += CC_TSC() - e0;
        }
        CC_ADD(cur_->sleepersHeld, S.sleep.n);
        CC_ADD(cur_->trackedHeld, S.tracked.size());
        CC_ADD(cur_->perShapeTracked[k == kProxy ? 0 : k], S.tracked.size());
#if CC_STATS
        cur_->maxTracked = std::max<std::uint64_t>(cur_->maxTracked, S.tracked.size());
#endif
    }
    CC_FT(6);
    // tracked patches: the 4 wheels' primary patches go through one FACE kernel; everything else takes the full path
    if (!(dbgSkip_ & 8u)) {
        alignas(16) float out[kFaceOut][4];
        const Pose *wp[4] = {&s[4].pose, &s[5].pose, &s[6].pose, &s[7].pose};
        const unsigned okm = PrimaryKernel(wp, out);
        for (int l = 0; l < 4; ++l) {
            if (!primValid_[l]) continue;
            ShapeState &S = s[4 + l];
            std::size_t i = 0;
            while (i < S.tracked.size() && S.tracked[i].patch != primPatch_[l]) ++i;
            if (i == S.tracked.size()) { primValid_[l] = false; continue; }
            if (!((okm >> l) & 1u)) { primValid_[l] = false; continue; }  // re-chosen by the full path below
            CC_CNT(cur_->trackedEvals); CC_CNT(cur_->faceFast);
            S.raw.emplace_back();
            FaceEmit(out, l, pw->members[static_cast<std::size_t>(primMember_[l])], &S.raw.back());
            S.tracked[i].member = primMember_[l];
            S.tracked[i].stamp = poseCount_;
        }
    }
    CC_FT(7);
    Pose *bodies[4] = {&s[0].pose, &s[1].pose, &s[2].pose, &s[3].pose};
    for (int k = 4; k < kSlots && !(dbgSkip_ & 16u); ++k) {
        ShapeState &S = s[k];
        if (S.tracked.empty()) continue;
        std::size_t open = 0;
        for (const Track &t : S.tracked) open += t.stamp != poseCount_;
        if (open == 0) continue;
        if (open >= 3) {
            SlabPass(S, S.pose, slCert_, slM_, slS_);
        } else {  // a short list: the scalar evaluation starts with the same slab test
            if (slCert_.size() < S.tracked.size()) { slCert_.resize(S.tracked.size() + 16); slM_.resize(slCert_.size()); slS_.resize(slCert_.size()); }
            std::fill(slCert_.begin(), slCert_.begin() + static_cast<std::ptrdiff_t>(S.tracked.size()), std::uint8_t(0));
        }
        for (std::size_t i = 0; i < S.tracked.size();) {
            Track &tr = S.tracked[i];
            if (tr.stamp == poseCount_) { ++i; continue; }
            Cert c;
            Eval e;
            if (slCert_[i]) {  // the slab certifies the whole patch: no member can touch
                CC_CNT(cur_->trackedEvals); CC_CNT(cur_->slabCertified);
                e = kEmptyFar;
                const Patch &pa = pw->patches[tr.patch];
                c.w[0] = m::abs(pa.n[0]); c.w[1] = m::abs(pa.n[1]); c.w[2] = m::abs(pa.n[2]);
                c.m = slM_[i];
                c.score = slS_[i];
                c.kind = kCertPlane;
            } else if (k == kProxy) {
                std::vector<Emitted> *raws[4] = {&s[0].raw, &s[1].raw, &s[2].raw, &s[3].raw};
                e = EvalProxyInto(S.pose, tr.patch, bodies, raws, S.rate, &c, lazySub_, &bodiesValid_);
            } else {
                const std::size_t before = S.raw.size();
                e = EvalPatch(S.pose, tr.patch, tr.member, S.raw, k, S.rate, &c);
                // a FACE from a located member becomes the wheel's primary (validated by the kernel next pose)
                if (!primValid_[k - 4] && e == kContact && tr.member >= 0 && S.raw.size() == before + 1 && S.raw.back().primary)
                    SetPrimary(k - 4, tr.patch, tr.member);
            }
            if (e != kContact) {
                // demote when the best certificate (slab, outline, or AABB gap) is predicted to last long
                if (e != kEmptyOutside && ((poseCount_ + i) & 31u) == 0) {  // full certificate every 32nd pose only
                    const Cert g = Certify(S.pose, pw->patches[tr.patch], S.rate);
                    if (g.score > c.score) c = g;
                }
                // edge-bound (P outside the outline, separated): sleep at once; slab-bound: when it lasts
                if (c.m > 0.0f && (c.kind != kCertPlane || c.score > 2.0f * promote)) {
                    CC_CNT(cur_->demotions);
                    if (k != kProxy && primValid_[k - 4] && primPatch_[k - 4] == tr.patch) primValid_[k - 4] = false;
                    S.sleep.Push(tr.patch, c.w, Lhs(c.w, S.O) + c.m - 3e-5f, pw->patches[tr.patch].box);
                    S.budget = std::min(S.budget, c.m * 0.999f - 1e-5f);
                    // keep the slab-pass results aligned with the swap
                    const std::size_t last = S.tracked.size() - 1;
                    slCert_[i] = slCert_[last]; slM_[i] = slM_[last]; slS_[i] = slS_[last];
                    TrackRemove(S, i);
                    continue;
                }
            }
            ++i;
        }
    }
    CC_FT(8);
    for (int k = 0; k < kShapes && !(dbgSkip_ & 4u); ++k) {
        ShapeState &S = s[k];
        if (S.raw.size() <= 1) { S.merged.swap(S.raw); continue; }
        bool prim = false, feat = false;
        for (const Emitted &c : S.raw) { prim |= c.primary != 0; feat |= c.primary == 0; }
        if (prim && feat) { CC_CNT(cur_->merges); MergeShape(S.raw, &S.merged); }
        else S.merged.swap(S.raw);
    }
    CC_FT(9);
    if (!busy_) CC_CNT(cur_->steadyPoses);
#if CC_STATS
    for (int b = 0; b < 6; ++b) if ((busyBits_ >> b) & 1u) ++cur_->busyBy[b];
#endif
    const std::uint64_t t3 = CC_TSC();
    g_split[0] += t1 - t0; g_split[1] += t2 - t1; g_split[2] += tExp; g_split[3] += t3 - t2 - tExp;
}

void CarCache::CarBox() {
    carBox_ = tw::EmptyBox();
    for (int k = 4; k < kSlots; ++k) carBox_ = tw::Union(carBox_, s[k].pose.B());
}

bool CarCache::ProbeDiffers(const tw::Substep &pose, const KSet K0[kShapes], KSet *out) {
    cur_ = &pst;
    CC_CNT(cur_->probes);
    bool differs = false;
    Pose *qbp[4] = {&qb_[0], &qb_[1], &qb_[2], &qb_[3]};
    bool qbValid = false;
    for (int k = 0; k < 4; ++k) praw4_[k].clear();
    std::vector<Emitted> *raws[4] = {&praw4_[0], &praw4_[1], &praw4_[2], &praw4_[3]};
    // probe poses and odometers of every slot, as if the probe pose were the next pose on the chain
    float O[kSlots][4], iso[kSlots], inc[3] = {0, 0, 0};
    for (int k = 4; k < kSlots; ++k) {
        ShapeState &S = s[k];
        Pose &q = probe_[k];
        if (k == kProxy) ProxyPose(pose, &q); else SetupPose(pose.shapes[k], &q);
        const float mScale = k == kProxy ? 1.7321f : 1.0f;
        float f = 0;
        for (int i = 0; i < 3; ++i)
            for (int jj = 0; jj < 3; ++jj) { const float d = q.M[i][jj] - S.pose.M[i][jj]; f += d * d; }
        float d[3];
        for (int a = 0; a < 3; ++a) { d[a] = m::abs(q.t[a] - S.pose.t[a]) * 1.0001f + 1e-7f; O[k][a] = S.O[a] + d[a]; }
        const float dM = m::sqrt(f) * (1.0001f * mScale) + 1e-7f;
        O[k][3] = S.O[3] + dM;
        iso[k] = m::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) * 1.0001f + dM;
        for (int a = 0; a < 3; ++a) inc[a] = std::max(inc[a], d[a] + dM);
        extra_[k].clear();
    }
    // car level: patches that may wake at the probe pose and overlap a slot's probe box go to that slot
    const float incMax = std::max(inc[0], std::max(inc[1], inc[2])) * 1.0001f + 1e-7f;
    if (car_.n && carBudget_ <= incMax) {
        const float Oc[4] = {carO_[0] + inc[0] * 1.0001f + 1e-7f, carO_[1] + inc[1] * 1.0001f + 1e-7f, carO_[2] + inc[2] * 1.0001f + 1e-7f, 0.0f};
        using V = m::vf<16>;
        const V ox(Oc[0]), oy(Oc[1]), oz(Oc[2]);
        for (std::size_t b = 0; b < car_.n; b += 16) {
            const V lhs = (V::load(&car_.wx[b]) * ox + V::load(&car_.wy[b]) * oy) + V::load(&car_.wz[b]) * oz;
            if (!m::any(m::cmp_ge(lhs, V::load(&car_.base[b])))) continue;
            for (std::size_t i = b; i < std::min(car_.n, b + 16); ++i) {
                const float w[3] = {car_.wx[i], car_.wy[i], car_.wz[i]};
                if (Lhs(w, Oc) < car_.base[i]) continue;
                const Box3 &pb = pw->patches[car_.pid[i]].box;
                for (int k = 4; k < kSlots; ++k)
                    if (tw::Overlap(probe_[k].B(), pb)) extra_[k].push_back(car_.pid[i]);
            }
        }
    }
    // the wheels' primary patches: FACE kernel at the probe poses
    alignas(16) float pout[kFaceOut][4];
    const Pose *pp[4] = {&probe_[4], &probe_[5], &probe_[6], &probe_[7]};
    const unsigned pok = PrimaryKernel(pp, pout);
    // slots in order: wheels 4..7, then the proxy (bodies 0..3)
    for (int k = 4; k < kSlots && !(differs && !out); ++k) {
        ShapeState &S = s[k];
        Pose &q = probe_[k];
        praw_.clear();
        std::uint32_t skipPatch = 0xffffffffu;
        if (k != kProxy && ((pok >> (k - 4)) & 1u)) {
            skipPatch = primPatch_[k - 4];
            praw_.emplace_back();
            FaceEmit(pout, k - 4, pw->members[static_cast<std::size_t>(primMember_[k - 4])], &praw_.back());
            CC_CNT(cur_->faceFast);
        }
        Cert c;
        auto eval = [&](std::uint32_t pid, std::int32_t hint) {
            if (k == kProxy) EvalProxyInto(q, pid, qbp, raws, S.rate, &c, &pose, &qbValid);
            else { std::int32_t h = hint; EvalPatch(q, pid, h, praw_, k, S.rate, &c); }
        };
        const float *Oq = S.Oq;
        if ((O[k][0] - Oq[0]) + (O[k][3] - Oq[3]) >= qL_[0] || (O[k][1] - Oq[1]) + (O[k][3] - Oq[3]) >= qL_[1] ||
            (O[k][2] - Oq[2]) + (O[k][3] - Oq[3]) >= qL_[2]) {  // beyond the query horizon: local query
            CC_CNT(cur_->probeHorizon);
            pw->bvh.Query(q.B(), [&](std::uint32_t pid) { eval(pid, -1); });
        } else {
            if (S.tracked.size() >= 3) {
                SlabPass(S, q, slCert_, slM_, slS_);
                for (std::size_t i = 0; i < S.tracked.size(); ++i)
                    if (S.tracked[i].patch != skipPatch && !slCert_[i]) eval(S.tracked[i].patch, S.tracked[i].member);
            } else {
                for (std::size_t i = 0; i < S.tracked.size(); ++i)
                    if (S.tracked[i].patch != skipPatch) eval(S.tracked[i].patch, S.tracked[i].member);
            }
            for (std::uint32_t pid : extra_[k]) {
                CC_CNT(cur_->probeMargins);
                if (Certify(q, pw->patches[pid], S.rate).m > 0.0f) continue;
                eval(pid, -1);
            }
            const SleepList &L = S.sleep;
            if (S.budget <= iso[k]) {  // some sleeper may wake at the probe pose
                using V = m::vf<16>;
                const V ox(O[k][0]), oy(O[k][1]), oz(O[k][2]), om(O[k][3]);
                for (std::size_t b = 0; b < L.n; b += 16) {
                    const V lhs = ((V::load(&L.wx[b]) * ox + V::load(&L.wy[b]) * oy) + V::load(&L.wz[b]) * oz) + om;
                    if (!m::any(m::cmp_ge(lhs, V::load(&L.base[b])))) continue;
                    for (std::size_t i = b; i < std::min(L.n, b + 16); ++i) {
                        const float w[3] = {L.wx[i], L.wy[i], L.wz[i]};
                        if (Lhs(w, O[k]) < L.base[i]) continue;
                        if (!tw::Overlap(q.B(), pw->patches[L.pid[i]].box)) continue;  // AABB-disjoint at the probe pose
                        CC_CNT(cur_->probeMargins);
                        if (Certify(q, pw->patches[L.pid[i]], S.rate).m > 0.0f) continue;
                        eval(L.pid[i], -1);
                    }
                }
            }
        }
        auto kOf = [&](std::vector<Emitted> &raw) {
            bool prim = false, feat = false;
            for (const Emitted &e : raw) { prim |= e.primary != 0; feat |= e.primary == 0; }
            const std::vector<Emitted> *res = &raw;
            if (prim && feat) { MergeShape(raw, &pmerged_); res = &pmerged_; }
            KSet kq{};
            for (const Emitted &e : *res) KAdd(kq, e.material);
            return kq;
        };
        if (k == kProxy) {
            for (int b = 0; b < 4; ++b) {
                const KSet kq = kOf(praw4_[b]);
                if (out) out[b] = kq;
                if (kq != K0[b]) { differs = true; if (!out) break; }
            }
        } else {
            const KSet kq = kOf(praw_);
            if (out) out[k] = kq;
            if (kq != K0[k]) differs = true;
        }
    }
    cur_ = &st;
    return differs;
}

}  // namespace cc
