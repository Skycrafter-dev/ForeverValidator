// Closed-form FACE contact of one car ellipsoid against one world triangle (architect D, gen2 S2).
// FV-free, header-only. Shared by every tier (E's T1 fast path, D's S2 fallback), so a FACE has the same
// bits whichever tier produces it (gen2-D-synthesis.md "FACE arithmetic"). Compile without FMA contraction.
//
// Board 20 §3.1 (SSphereMeshCollide::CollideTriangle) in unit-sphere space, mapped through M = R·diag(r):
//   unit plane distance  d = sd / σ,   sd = n·t − ρ (signed centre distance), σ = |Mᵀ n| (support along n)
//   FACE iff 0 < d ≤ 1 and no edge is violated, i.e. the foot point P lies inside the triangle (an affine
//   map with positive determinant keeps the side of a line, so "no violated edge in unit space" equals
//   "P inside the world triangle")
//   P   = t − (sd/σ²)·Q n          (= M p + t, p = −d n')          Q = M Mᵀ
//   N   = n                        (= normalize(R diag(1/r) n'))
//   Sep = Q n · (d − 1)/σ          (= M n' (d − 1))
//   K   = Mᵀ n / σ                 (= n', FV's merge key, in the ellipsoid's unit frame)
// Each line is an identity of FV's formula; only rounding differs (docs/twin/design-gen2-d.md §2.4.1).
//
// Precision: the board's triangle is FV's mesh-local triangle through the instance transform, in exact
// arithmetic (FV's ellipsoid path reads mesh-local vertices through meshToUnit, never the rounded world
// floats): build FaceTri from WorldVertD. The plane and edge offsets are doubles, and the two distances of
// the shape centre (to the plane, to each edge line) are taken in double from the world float centre, so a
// triangle's size or distance never costs precision. The rest is float on offsets of at most r_max, and P
// is the centre plus its offset, rounded once. The result is a pure function of (triangle, shape pose): the
// same bits for any caller.
//
// Near ties (E's margins, agreed 2026-11-08): the closed form does not decide d ≤ kFaceDistTie,
// d ≥ 1 − kFaceDistTie, or a foot point within kFaceEdgeTie of an edge line, nor a tiny triangle (FV's own
// degenerate test is shape dependent). There ClosedFace reports a negative tieMargin and the caller runs
// FV's own triangle test. Both choices are pure functions of the pose.
#ifndef TW_ATLAS_FACE_H
#define TW_ATLAS_FACE_H

#include <cmath>
#include <cstdint>

#include "response.h"   // model::RespContact (src/twin/model on the include path)

namespace tw::atlas {

constexpr float kFaceEdgeTie = 2e-4f;        // m: foot point this close to an edge line -> FV form
constexpr float kFaceDistTie = 1e-5f;        // unit-space d band at 0 and at 1 (FV's form decides inside)
constexpr double kFaceMinArea = 1e-5;        // m^2: FV skips a triangle whose unit-space cross product is <= 1e-5 in length
                                             // (world area <= ~7e-7 m^2 for a wheel); below 1e-5 m^2 FV decides

// One world triangle (built once per map, in double). 128 B. The plane distance uses the double normal: a
// float normal tilts the plane by ~6e-8 rad about the vertices, ~3e-5 m at 500 m on a map-sized triangle.
struct FaceTri {
    // line 0: the plane (the reach pre-test reads only this line), the contact normal, the instance origin
    double nd[3];         // own front unit normal (double), from the cross product (v1 - v0) x (v2 - v0)
    double rho;           // plane nd . x = rho (vertex mean)
    float n[3];           // nd rounded (the contact normal N and the float parts)
    std::uint32_t valid;  // 0: area below kFaceMinArea, or an instance rotation that is not an exact signed permutation:
                          // the caller runs FV's form (which skips or decides it)
    float tm[3];          // the instance's translation (FV localToWorld t): FV's c2w carries (R R^T - I)(t - tm), below
    std::uint32_t pad0;
    // line 1: the edge lines
    double c[3];          // edge i (v_i -> v_i+1): u_i . x - c_i > 0 outside (u_i the float normal below)
    float u[3][3];        // edge i outward unit normal in the plane: (v_i+1 - v_i) x n, normalised
    std::uint32_t pad1;
};
static_assert(sizeof(FaceTri) == 128, "FaceTri layout");

// The shape, once per (shape, substep).
struct ShapeLocal {
    double t[3];          // world centre, widened (ShapePose.world.t)
    float M[3][3];        // world M = axes · diag(r): M[i][j] = axis_j[i] · r_j (ShapePose.world.r column j, radii[j])
    float E[3][3];        // R R^T - I of the shape's float rotation (MakeShapeLocal)
    float Nw[3][3];       // R diag(1/r) (axes / r): FV's n2w rotation part, unit frame -> world normals
    float rmin;
};

struct FaceOut {
    float P[3];           // contact point (world): the centre plus its offset, rounded once (kFaceYes)
    float N[3], Sep[3], K[3];
    float d;              // unit-space plane distance (set once the plane is in reach)
    // set whenever the foot point was formed (kFaceYes, kFaceOutside, most kFaceTie): the fallback's prefilter
    float off[3];         // foot point minus centre (world)
    float m[3];           // M^T n
    float s2;             // |M^T n|^2 = sigma^2
    float se[3];          // signed in-plane distance of the foot point outside edge e (m)
};

// The board's world vertex of a mesh-local vertex: FV localToWorld (iso: rotation columns basisX, basisY,
// basisZ, then translation; WorldPackView Instance::iso), in double.
inline void WorldVertD(const float iso[12], const float vm[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = ((double(iso[i]) * vm[0] + double(iso[3 + i]) * vm[1]) + double(iso[6 + i]) * vm[2]) + double(iso[9 + i]);
}

// v: the board's world vertices in double (WorldVertD), FV stored order.
inline void MakeFaceTriD(const double v[3][3], FaceTri *f) {
    double e1[3], e2[3];
    for (int k = 0; k < 3; ++k) { e1[k] = v[1][k] - v[0][k]; e2[k] = v[2][k] - v[0][k]; }
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    f->valid = 0.5 * l >= kFaceMinArea ? 1u : 0u;
    f->pad0 = 0u;
    f->pad1 = 0u;
    f->tm[0] = f->tm[1] = f->tm[2] = 0.0f;
    if (!(l > 0.0)) { n[0] = 0; n[1] = 1; n[2] = 0; } else { for (double &x : n) x /= l; }
    for (int k = 0; k < 3; ++k) { f->n[k] = float(n[k]); f->nd[k] = n[k]; }
    double rho = 0.0;
    for (int j = 0; j < 3; ++j) rho += (n[0] * v[j][0] + n[1] * v[j][1] + n[2] * v[j][2]) / 3.0;
    f->rho = rho;
    for (int i = 0; i < 3; ++i) {
        const double *p = v[i], *q = v[(i + 1) % 3];
        const double e[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
        double m[3] = {e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]};
        const double ml = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
        if (ml > 0.0) for (double &x : m) x /= ml;
        for (int k = 0; k < 3; ++k) f->u[i][k] = float(m[k]);
        f->c[i] = double(f->u[i][0]) * p[0] + double(f->u[i][1]) * p[1] + double(f->u[i][2]) * p[2];
    }
}

// The instance a triangle belongs to: its translation, and whether its rotation is an exact signed permutation
// (WorldPackView Instance::rotation != kRotGeneral). For a general rotation, FV's transforms add terms in
// (R_m R_m^T - I) times the distance to the instance origin; those triangles go to FV's form.
inline void SetFaceInstance(const float iso[12], bool permutation, FaceTri *f) {
    for (int k = 0; k < 3; ++k) f->tm[k] = iso[9 + k];
    if (!permutation) f->valid = 0u;
}

// shapeR: ShapePose.world.r (rotation columns), shapeT: ShapePose.world.t, radii.
// FV's contact point carries a term of its transforms (GmCollision_Ellipsoid_Mesh, inverses by transposes):
// c2w(u) = R D u + t + (R R^T - I)(t - t_m), with R the shape's (not exactly orthonormal) float rotation and t_m
// the instance's origin; with |t - t_m| ~ 100 m that is ~1e-5-5e-5 m, and the board in double has it too.
// E = R R^T - I (in double from the float rotation, rounded) reproduces it.
inline void MakeShapeLocal(const float shapeR[9], const float shapeT[3], const float radii[3], ShapeLocal *s) {
    for (int k = 0; k < 3; ++k) s->t[k] = shapeT[k];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double rr = 0.0;
            for (int c = 0; c < 3; ++c) rr += double(shapeR[3 * c + i]) * double(shapeR[3 * c + j]);
            s->E[i][j] = float(rr - (i == j ? 1.0 : 0.0));
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            s->M[i][j] = shapeR[3 * j + i] * radii[j];
            s->Nw[i][j] = shapeR[3 * j + i] / radii[j];
        }
    const float m01 = radii[0] < radii[1] ? radii[0] : radii[1];
    s->rmin = m01 < radii[2] ? m01 : radii[2];
}

// The centre's signed distance to the triangle plane (double, rounded once): the cheap first test.
inline float PlaneDistance(const FaceTri &f, const ShapeLocal &s) {
    return float(((f.nd[0] * s.t[0] + f.nd[1] * s.t[1]) + f.nd[2] * s.t[2]) - f.rho);
}

// True iff FV's test yields a FACE for this triangle, decided outside the near-tie band; o filled.
// tieMargin: < 0 inside the band (the caller runs FV's form), else the distance to the band (diagnostic).
// why (optional): kFaceYes / kFaceTie / kFaceBehind / kFaceFar / kFaceOutside (outside an edge, in reach: a
// feature is possible) / kFaceInvalid (tiny triangle: FV's form decides, like a tie).
enum : int { kFaceYes = 0, kFaceTie = 1, kFaceBehind = 2, kFaceFar = 3, kFaceOutside = 4, kFaceInvalid = 5 };
inline bool ClosedFace(const FaceTri &f, const ShapeLocal &s, FaceOut *o, float *tieMargin, int *why = nullptr) {
    int dummy;
    int &w = why ? *why : dummy;
    w = kFaceInvalid;
    if (!f.valid) { *tieMargin = -1.0f; return false; }
    const float sd = PlaneDistance(f, s);
    // m = M^T n from the double normal, rounded once: the rounding varies with the pose, so no fixed per-triangle
    // error (float(n)) enters the contact (a fixed one leans in the bias test when many contacts share a triangle)
    float m[3];
    for (int j = 0; j < 3; ++j) m[j] = float((double(s.M[0][j]) * f.nd[0] + double(s.M[1][j]) * f.nd[1]) + double(s.M[2][j]) * f.nd[2]);
    const float s2 = (m[0] * m[0] + m[1] * m[1]) + m[2] * m[2];
    const float sg = std::sqrt(s2);
    const float d = sd / sg;
    if (d < -kFaceDistTie) { *tieMargin = -kFaceDistTie - d; w = kFaceBehind; return false; }      // FV rejects d < 0
    if (d > 1.0f + kFaceDistTie) { *tieMargin = d - 1.0f - kFaceDistTie; w = kFaceFar; return false; }
    // foot point offset from the centre and the edge distances
    float g[3];
    for (int i = 0; i < 3; ++i) g[i] = (s.M[i][0] * m[0] + s.M[i][1] * m[1]) + s.M[i][2] * m[2];   // Q n
    const float k1 = sd / s2;
    const float off[3] = {-k1 * g[0], -k1 * g[1], -k1 * g[2]};
    float smax = -1e30f;
    for (int e = 0; e < 3; ++e) {
        const float sc = float(((double(f.u[e][0]) * s.t[0] + double(f.u[e][1]) * s.t[1]) + double(f.u[e][2]) * s.t[2]) - f.c[e]);
        const float se = sc + ((f.u[e][0] * off[0] + f.u[e][1] * off[1]) + f.u[e][2] * off[2]);
        o->se[e] = se;
        smax = se > smax ? se : smax;
    }
    for (int i = 0; i < 3; ++i) { o->off[i] = off[i]; o->m[i] = m[i]; }
    o->s2 = s2;
    o->d = d;
    const bool dTie = d <= kFaceDistTie || d >= 1.0f - kFaceDistTie;
    if (smax > kFaceEdgeTie) {   // outside an edge: not a FACE (a feature if in reach; a near-reach d is a tie)
        w = dTie ? kFaceTie : kFaceOutside;
        *tieMargin = dTie ? -1.0f : smax - kFaceEdgeTie;
        return false;
    }
    if (smax > -kFaceEdgeTie || dTie) { *tieMargin = -1.0f; w = kFaceTie; return false; }   // near tie: FV decides
    w = kFaceYes;
    const float md = d - kFaceDistTie < 1.0f - kFaceDistTie - d ? d - kFaceDistTie : 1.0f - kFaceDistTie - d;
    const float me = -kFaceEdgeTie - smax;
    *tieMargin = md < me ? md : me;
    const float k2 = (d - 1.0f) / sg;
    double dt[3];
    for (int i = 0; i < 3; ++i) dt[i] = s.t[i] - double(f.tm[i]);
    // N: FV's route and its shape. FV normalises the unit-space triangle normal n' = D R^-1 n / |.|, maps it with
    // n2w = R D^-1 and normalises again (NormalizeForCollision). In reals that is n. Doing the same steps keeps
    // FV's rounding pattern: both normalisations act on vectors far from unit length (|D R^-1 n| ~ sigma, |R D^-1 n'|
    // ~ 1 / sigma), unlike a renormalised near-unit vector, whose roundings near 1 lean (the harness's search maps).
    // R^-1 = R^T (R R^T)^-1 ~ R^T (I - E).
    float kp[3];
    {
        double a[3];
        for (int i = 0; i < 3; ++i) a[i] = f.nd[i] - ((double(s.E[i][0]) * f.nd[0] + double(s.E[i][1]) * f.nd[1]) + double(s.E[i][2]) * f.nd[2]);
        float mp[3];
        for (int j = 0; j < 3; ++j) mp[j] = float((double(s.M[0][j]) * a[0] + double(s.M[1][j]) * a[1]) + double(s.M[2][j]) * a[2]);
        const float l2 = (mp[0] * mp[0] + mp[1] * mp[1]) + mp[2] * mp[2];
        const float il = 1.0f / std::sqrt(l2);
        for (int j = 0; j < 3; ++j) kp[j] = mp[j] * il;
    }
    float nv[3];
    for (int i = 0; i < 3; ++i) nv[i] = (s.Nw[i][0] * kp[0] + s.Nw[i][1] * kp[1]) + s.Nw[i][2] * kp[2];
    const float nl2 = (nv[0] * nv[0] + nv[1] * nv[1]) + nv[2] * nv[2];
    const float inl = 1.0f / std::sqrt(nl2);
    for (int i = 0; i < 3; ++i) {
        const double shift = (double(s.E[i][0]) * dt[0] + double(s.E[i][1]) * dt[1]) + double(s.E[i][2]) * dt[2];
        o->P[i] = float((s.t[i] + double(off[i])) + shift);
        o->N[i] = nv[i] * inl;
        o->Sep[i] = g[i] * k2;
        o->K[i] = m[i] / sg;
    }
    return true;
}

// The payload into a contact record (P, N, Sep); the caller fills the other fields.
inline void FaceToWorld(const FaceOut &o, model::RespContact *c) {
    for (int i = 0; i < 3; ++i) { c->P[i] = o.P[i]; c->N[i] = o.N[i]; c->Sep[i] = o.Sep[i]; }
}

}  // namespace tw::atlas

#endif
