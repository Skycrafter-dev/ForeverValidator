// Board 20 §3.1 ellipsoid-vs-triangle test in unit-sphere space (FV form),
// plus the conservative world-space reach margin used by the coherent cache.
#ifndef TW_EXACT_H
#define TW_EXACT_H

#include "tw_world.h"

namespace tw {

enum ContactKind : int { kNone = 0, kFace = 1, kVertexA = 2, kEdge = 3, kVertexB = 4 };

struct Contact {
    int kind;
    float P[3], N[3], Sep[3], K[3];
};

inline void UnitPoint(const ShapeMap &s, const float x[3], float u[3]) {
    const float d0 = x[0] - s.t[0], d1 = x[1] - s.t[1], d2 = x[2] - s.t[2];  // subtract first
    for (int j = 0; j < 3; ++j) u[j] = (s.A[j][0] * d0 + s.A[j][1] * d1) + s.A[j][2] * d2;
}

// Returns the contact kind (0 = none). Branch structure and thresholds follow
// board 20 §3.1 exactly: back-face cull, 0 <= d <= 1, first violated edge in
// stored vertex order decides, endpoint-B sqrt quirk, feature separation
// along the triangle normal.
inline int ExactTest(const ShapeMap &s, const Tri &t, Contact *out) {
    float v[3][3];
    for (int k = 0; k < 3; ++k) UnitPoint(s, t.v[k], v[k]);
    const float e1[3] = {v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]};
    const float e2[3] = {v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]};
    float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const float l2 = (n[1] * n[1] + n[0] * n[0]) + n[2] * n[2];
    if (!(l2 > 1e-10f)) return kNone;
    const float il = 1.0f / std::sqrt(l2);
    n[0] *= il; n[1] *= il; n[2] *= il;
    const float d = -((v[0][0] * n[0] + v[0][1] * n[1]) + v[0][2] * n[2]);
    if (!(d >= 0.0f) || d > 1.0f) return kNone;
    const float h = std::sqrt(1.0f - d * d);
    const float p[3] = {-d * n[0], -d * n[1], -d * n[2]};
    int kind = kNone;
    float f[3] = {0, 0, 0};
    bool decided = false;
    for (int i = 0; i < 3 && !decided; ++i) {
        const float *a = v[i], *b = v[(i + 1) % 3];
        float e[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float el2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        if (el2 > 1e-10f) { const float ie = 1.0f / std::sqrt(el2); e[0] *= ie; e[1] *= ie; e[2] *= ie; }
        const float m[3] = {e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]};
        const float r[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
        const float sE = r[0] * m[0] + r[1] * m[1] + r[2] * m[2];
        if (sE > h) return kNone;
        if (sE > 0.0f) {
            decided = true;
            const float aa = r[0] * e[0] + r[1] * e[1] + r[2] * e[2];
            if (aa < 0.0f) { kind = kVertexA; f[0] = a[0]; f[1] = a[1]; f[2] = a[2]; break; }
            const float bb = (p[0] - b[0]) * e[0] + (p[1] - b[1]) * e[1] + (p[2] - b[2]) * e[2];
            if (!(bb > 0.0f)) { kind = kEdge; f[0] = p[0] - sE * m[0]; f[1] = p[1] - sE * m[1]; f[2] = p[2] - sE * m[2]; }
            else { kind = kVertexB; f[0] = b[0]; f[1] = b[1]; f[2] = b[2]; }
        }
    }
    float Np[3], Sp[3], Pp[3];
    if (!decided) {
        if (!(d > 0.0f)) return kNone;
        kind = kFace;
        for (int k = 0; k < 3; ++k) { Np[k] = n[k]; Sp[k] = n[k] * (d - 1.0f); Pp[k] = p[k]; }
    } else {
        const float w[3] = {-f[0], -f[1], -f[2]};
        const float w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        const float wl = std::sqrt(w2);
        float den = wl;
        if (kind == kVertexA) { if (!(w2 > 1e-10f) || w2 > 1.0f) return kNone; }
        else if (kind == kEdge) { if (!(w2 > 1e-5f)) return kNone; }
        else { if (!(wl > 1e-10f) || wl > 1.0f) return kNone; den = std::sqrt(wl); }
        const float idn = 1.0f / den;
        const float sep = d * (den - 1.0f) * idn;
        for (int k = 0; k < 3; ++k) { Np[k] = w[k] * idn; Sp[k] = n[k] * sep; Pp[k] = f[k]; }
    }
    if (out) {
        out->kind = kind;
        float g[3];
        for (int i = 0; i < 3; ++i) {
            out->P[i] = ((s.M[i][0] * Pp[0] + s.M[i][1] * Pp[1]) + s.M[i][2] * Pp[2]) + s.t[i];
            out->Sep[i] = (s.M[i][0] * Sp[0] + s.M[i][1] * Sp[1]) + s.M[i][2] * Sp[2];
            g[i] = (s.N[i][0] * Np[0] + s.N[i][1] * Np[1]) + s.N[i][2] * Np[2];
            out->K[i] = n[i];
        }
        const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        const float ig = g2 > 1e-10f ? 1.0f / std::sqrt(g2) : 1.0f;
        for (int i = 0; i < 3; ++i) out->N[i] = g[i] * ig;
    }
    return kind;
}

// Support radius of the ellipsoid along a world direction u: h = |Mᵀ u|.
inline float Support(const ShapeMap &s, const float u[3]) {
    const float a = s.M[0][0] * u[0] + s.M[1][0] * u[1] + s.M[2][0] * u[2];
    const float b = s.M[0][1] * u[0] + s.M[1][1] * u[1] + s.M[2][1] * u[2];
    const float c = s.M[0][2] * u[0] + s.M[1][2] * u[1] + s.M[2][2] * u[2];
    return std::sqrt(a * a + b * b + c * c);
}

// Conservative world-space distance (m) the shape must still travel before the
// exact test could accept this triangle; <= 0 means "test now". Necessary
// conditions for any board 20 contact: the centre on the front side and the
// plane within reach (0 <= sd <= h(n)), and the ellipsoid reaching the inner
// side of every edge's side plane (se_i <= h(u_i)). All distances are computed
// subtract-first; eps absorbs float error. Degenerate triangles return 0.
inline float ReachMargin(const ShapeMap &s, const Tri &t, const TriExtra &x, float eps) {
    const float *n = x.n;
    if (!(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.5f)) return 0.0f;
    const float dc[3] = {s.t[0] - t.v[0][0], s.t[1] - t.v[0][1], s.t[2] - t.v[0][2]};
    const float sd = n[0] * dc[0] + n[1] * dc[1] + n[2] * dc[2];
    const float hn = Support(s, n);
    float m = -sd;                       // behind the plane
    if (sd - hn > m) m = sd - hn;        // plane out of reach
    for (int i = 0; i < 3; ++i) {
        const float *a = t.v[i], *b = t.v[(i + 1) % 3];
        const float e[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        float u[3] = {e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]};
        const float ul = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if (!(ul > 1e-6f)) continue;
        const float iu = 1.0f / ul;
        u[0] *= iu; u[1] *= iu; u[2] *= iu;
        const float se = u[0] * (s.t[0] - a[0]) + u[1] * (s.t[1] - a[1]) + u[2] * (s.t[2] - a[2]);
        const float me = se - Support(s, u);
        if (me > m) m = me;
    }
    return m - eps;
}

// Per-triangle world edge normals (outward, unit), precomputed once per map.
struct TriEdges { float u[3][3]; };
inline void MakeTriEdges(const Tri &t, const TriExtra &x, TriEdges *e) {
    for (int i = 0; i < 3; ++i) {
        const float *a = t.v[i], *b = t.v[(i + 1) % 3];
        const double d[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
        const double u[3] = {d[1] * x.n[2] - d[2] * x.n[1], d[2] * x.n[0] - d[0] * x.n[2], d[0] * x.n[1] - d[1] * x.n[0]};
        const double l = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        for (int k = 0; k < 3; ++k) e->u[i][k] = l > 1e-9 ? static_cast<float>(u[k] / l) : 0.0f;
    }
}
// ReachMargin with precomputed edge normals (same bound, fewer sqrt/div).
inline float ReachMarginPre(const ShapeMap &s, const Tri &t, const TriExtra &x, const TriEdges &e, float eps) {
    const float *n = x.n;
    if (!(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.5f)) return 0.0f;
    const float sd = n[0] * (s.t[0] - t.v[0][0]) + n[1] * (s.t[1] - t.v[0][1]) + n[2] * (s.t[2] - t.v[0][2]);
    float m = -sd;
    if (m > eps) return m - eps;  // behind the plane: no support needed
    const float hn = Support(s, n);
    if (sd - hn > m) m = sd - hn;
    for (int i = 0; i < 3; ++i) {
        const float *u = e.u[i], *a = t.v[i];
        if (u[0] == 0.0f && u[1] == 0.0f && u[2] == 0.0f) continue;
        const float se = u[0] * (s.t[0] - a[0]) + u[1] * (s.t[1] - a[1]) + u[2] * (s.t[2] - a[2]);
        if (se - (std::fabs(s.half[0] * u[0]) + std::fabs(s.half[1] * u[1]) + std::fabs(s.half[2] * u[2])) <= m) continue;  // cheap upper bound of h(u) cannot raise m
        const float me = se - Support(s, u);
        if (me > m) m = me;
    }
    return m - eps;
}

// Squared distance from point p to triangle t (double precision; gate only).
inline double PointTriDist2(const double p[3], const Tri &t) {
    double a[3], b[3], c[3];
    for (int k = 0; k < 3; ++k) { a[k] = t.v[0][k]; b[k] = t.v[1][k]; c[k] = t.v[2][k]; }
    auto sub = [](const double *x, const double *y, double *r) { for (int k = 0; k < 3; ++k) r[k] = x[k] - y[k]; };
    auto dot = [](const double *x, const double *y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    double ab[3], ac[3], ap[3], bp[3], cp[3];
    sub(b, a, ab); sub(c, a, ac); sub(p, a, ap);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    double q[3];
    auto setq = [&](double s, double u) { for (int k = 0; k < 3; ++k) q[k] = a[k] + s * ab[k] + u * ac[k]; };
    if (d1 <= 0 && d2 <= 0) setq(0, 0);
    else {
        sub(p, b, bp);
        const double d3 = dot(ab, bp), d4 = dot(ac, bp);
        sub(p, c, cp);
        const double d5 = dot(ab, cp), d6 = dot(ac, cp);
        const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
        if (d3 >= 0 && d4 <= d3) setq(1, 0);
        else if (vc <= 0 && d1 >= 0 && d3 <= 0) setq(d1 / (d1 - d3), 0);
        else if (d6 >= 0 && d5 <= d6) setq(0, 1);
        else if (vb <= 0 && d2 >= 0 && d6 <= 0) setq(0, d2 / (d2 - d6));
        else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
            const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            for (int k = 0; k < 3; ++k) q[k] = b[k] + w * (c[k] - b[k]);
        } else {
            const double den = 1.0 / (va + vb + vc);
            setq(vb * den, vc * den);
        }
    }
    double r[3]; sub(p, q, r);
    return dot(r, r);
}

}  // namespace tw

#endif
