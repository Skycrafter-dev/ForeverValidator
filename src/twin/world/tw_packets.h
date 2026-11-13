// X1: board 20 §3.1 ellipsoid-vs-triangle test over lane types (tw::math,
// src/twin/math/tw_lanes.h). One (shape, triangle) pair per lane; the op
// sequence is ExactTest's (tw_exact.h) exactly, so every width gives the same
// bits as the scalar reference. Stage 1 = unit-space transform, normal and
// plane reach; stage 2 = first-violated-edge features and payload, run only if
// some lane survives stage 1.
#ifndef TW_PACKETS_H
#define TW_PACKETS_H

#include <cstdint>

#include "../math/tw_lanes.h"
#include "tw_exact.h"
#include "tw_world.h"

namespace tw {

template <class F>
struct PairIn {
    F A[3][3], t[3], M[3][3], N[3][3], v[3][3];  // shape unit map + inverse maps; triangle vertices (stored order)
};
template <class F>
struct PairOut {
    typename F::int_type kind;
    F P[3], Nrm[3], Sep[3], K[3];
};

template <class F>
struct Stage1State {
    F X[3][3];
    F n[3];
    F d;
    typename F::mask_type alive;
};

// Stage 1: unit-space vertices, normal, plane reach. False if no lane survives.
template <class F>
TW_INLINE bool Stage1(const F (&A)[3][3], const F (&t)[3], const F (&v)[3][3], Stage1State<F> &s) {
    using namespace tw::math;
    for (int k = 0; k < 3; ++k) {
        const F d0 = v[k][0] - t[0], d1 = v[k][1] - t[1], d2 = v[k][2] - t[2];
        for (int j = 0; j < 3; ++j) s.X[k][j] = (A[j][0] * d0 + A[j][1] * d1) + A[j][2] * d2;
    }
    const F e1[3] = {s.X[1][0] - s.X[0][0], s.X[1][1] - s.X[0][1], s.X[1][2] - s.X[0][2]};
    const F e2[3] = {s.X[2][0] - s.X[0][0], s.X[2][1] - s.X[0][1], s.X[2][2] - s.X[0][2]};
    s.n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    s.n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    s.n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    const F l2 = (s.n[1] * s.n[1] + s.n[0] * s.n[0]) + s.n[2] * s.n[2];
    s.alive = cmp_gt(l2, F(1e-10f));
    const F il = F(1.0f) / sqrt(select(s.alive, l2, F(1.0f)));
    s.n[0] = s.n[0] * il; s.n[1] = s.n[1] * il; s.n[2] = s.n[2] * il;
    s.d = -((s.X[0][0] * s.n[0] + s.X[0][1] * s.n[1]) + s.X[0][2] * s.n[2]);
    s.alive = s.alive & cmp_ge(s.d, F(0.0f)) & cmp_le(s.d, F(1.0f));
    return any(s.alive);
}

// Stage 2: first violated edge in stored order, feature acceptance, payload.
template <class F>
TW_INLINE void Stage2(const Stage1State<F> &s, const F (&t)[3], const F (&M)[3][3], const F (&N)[3][3], PairOut<F> &o) {
    using namespace tw::math;
    using Mk = typename F::mask_type;
    using I = typename F::int_type;
    const F *n = s.n;
    const F d = s.d;
    Mk alive = s.alive;
    const F h = sqrt(F(1.0f) - d * d);
    const F nd = -d;
    const F p[3] = {nd * n[0], nd * n[1], nd * n[2]};
    I kind(0);
    F f[3] = {F(0.0f), F(0.0f), F(0.0f)};
    Mk decided = ~alive;  // dead lanes count as decided (they never set a feature)
    for (int i = 0; i < 3; ++i) {
        const F *a = s.X[i], *b = s.X[(i + 1) % 3];
        F e[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const F el2 = (e[0] * e[0] + e[1] * e[1]) + e[2] * e[2];
        const Mk eok = cmp_gt(el2, F(1e-10f));
        const F ie = F(1.0f) / sqrt(select(eok, el2, F(1.0f)));
        e[0] = select(eok, e[0] * ie, e[0]); e[1] = select(eok, e[1] * ie, e[1]); e[2] = select(eok, e[2] * ie, e[2]);
        const F m[3] = {e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]};
        const F r[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
        const F sE = (r[0] * m[0] + r[1] * m[1]) + r[2] * m[2];
        const Mk over = cmp_gt(sE, h);
        const Mk out = ~decided & over;
        const Mk here = ~decided & ~over & cmp_gt(sE, F(0.0f));
        alive = alive & ~out;
        const F aa = (r[0] * e[0] + r[1] * e[1]) + r[2] * e[2];
        const F bb = ((p[0] - b[0]) * e[0] + (p[1] - b[1]) * e[1]) + (p[2] - b[2]) * e[2];
        const Mk kA = here & cmp_lt(aa, F(0.0f));
        const Mk bpos = cmp_gt(bb, F(0.0f));
        const Mk kB = here & ~kA & bpos;
        const Mk kE = here & ~kA & ~bpos;
        for (int c = 0; c < 3; ++c) f[c] = select(kA, a[c], select(kB, b[c], select(kE, p[c] - sE * m[c], f[c])));
        kind = select(kA, I(kVertexA), select(kB, I(kVertexB), select(kE, I(kEdge), kind)));
        decided = decided | out | here;
    }
    const Mk face = alive & ~decided & cmp_gt(d, F(0.0f));
    kind = select(face, I(kFace), kind);
    if (!any(alive & decided)) {  // no feature lane: FACE payload only (same bits for every FACE lane)
        const F dm1 = d - F(1.0f);
        F g[3];
        for (int i = 0; i < 3; ++i) {
            o.P[i] = ((M[i][0] * p[0] + M[i][1] * p[1]) + M[i][2] * p[2]) + t[i];
            const F Sp0 = n[0] * dm1, Sp1 = n[1] * dm1, Sp2 = n[2] * dm1;
            o.Sep[i] = (M[i][0] * Sp0 + M[i][1] * Sp1) + M[i][2] * Sp2;
            g[i] = (N[i][0] * n[0] + N[i][1] * n[1]) + N[i][2] * n[2];
            o.K[i] = n[i];
        }
        const F g2 = (g[0] * g[0] + g[1] * g[1]) + g[2] * g[2];
        const Mk gok = cmp_gt(g2, F(1e-10f));
        const F ig = select(gok, F(1.0f) / sqrt(select(gok, g2, F(1.0f))), F(1.0f));
        for (int i = 0; i < 3; ++i) o.Nrm[i] = g[i] * ig;
        o.kind = select(face, kind, I(0));
        return;
    }
    const F w[3] = {-f[0], -f[1], -f[2]};
    const F w2 = (w[0] * w[0] + w[1] * w[1]) + w[2] * w[2];
    const F wl = sqrt(w2);
    const Mk isA = cmp_eq(kind, I(kVertexA)), isE = cmp_eq(kind, I(kEdge)), isB = cmp_eq(kind, I(kVertexB));
    const F den = select(isB, sqrt(wl), wl);
    const Mk okA = isA & cmp_gt(w2, F(1e-10f)) & cmp_le(w2, F(1.0f));
    const Mk okE = isE & cmp_gt(w2, F(1e-5f));
    const Mk okB = isB & cmp_gt(wl, F(1e-10f)) & cmp_le(wl, F(1.0f));
    const Mk feat = alive & (okA | okE | okB);
    alive = face | feat;
    const F idn = F(1.0f) / select(feat, den, F(1.0f));
    const F sep = d * (den - F(1.0f)) * idn;
    const F dm1 = d - F(1.0f);
    F Np[3], Sp[3], Pp[3];
    for (int c = 0; c < 3; ++c) {
        Np[c] = select(face, n[c], w[c] * idn);
        Sp[c] = select(face, n[c] * dm1, n[c] * sep);
        Pp[c] = select(face, p[c], f[c]);
    }
    F g[3];
    for (int i = 0; i < 3; ++i) {
        o.P[i] = ((M[i][0] * Pp[0] + M[i][1] * Pp[1]) + M[i][2] * Pp[2]) + t[i];
        o.Sep[i] = (M[i][0] * Sp[0] + M[i][1] * Sp[1]) + M[i][2] * Sp[2];
        g[i] = (N[i][0] * Np[0] + N[i][1] * Np[1]) + N[i][2] * Np[2];
        o.K[i] = n[i];
    }
    const F g2 = (g[0] * g[0] + g[1] * g[1]) + g[2] * g[2];
    const Mk gok = cmp_gt(g2, F(1e-10f));
    const F ig = select(gok, F(1.0f) / sqrt(select(gok, g2, F(1.0f))), F(1.0f));
    for (int i = 0; i < 3; ++i) o.Nrm[i] = g[i] * ig;
    o.kind = select(alive, kind, I(0));
}

// Both stages (scalar reference and simple callers). False if stage 2 was skipped (kinds = 0).
template <class F>
TW_INLINE bool ExactLanes(const PairIn<F> &q, PairOut<F> &o) {
    Stage1State<F> s;
    if (!Stage1(q.A, q.t, q.v, s)) { o.kind = typename F::int_type(0); return false; }
    Stage2(s, q.t, q.M, q.N, o);
    return true;
}

}  // namespace tw

#endif
