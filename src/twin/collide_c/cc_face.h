// Wheel FACE closed form on one known member triangle, 1 or 4 lanes (architect C, docs/twin/collision-c.md §8).
//
// Use: a shape (ellipsoid x = M u + t, M = axes * diag(r)) and a triangle T of a coplanar patch. The kernel
// returns, per lane, "ok" and FV's FACE contact of T (board 20 §3.1):
//   P = t - (sd/H^2) M M^T n,  N = n_T,  K = M^T n_T / H,  Sep = ((sd - H)/H^2) M M^T n_T,
// with sd = n_T.(t - v0) and H = |M^T n_T|.
//
// When ok is set, this FACE is the patch's whole merged contribution:
//   - T is reached (1e-5 < sd, d^2 < 0.9997, so tangent contacts with a noisy d take the exact path);
//   - P lies strictly inside T (2e-5 m from its edges, in-plane outward edge normals u_e, edge origins a_e);
//   - the contact disk of the ellipsoid with the slab of half thickness tauP around T's plane clears every
//     edge of T. Its radius is at most r_max sqrt(1 - dlo^2), with dlo = (sd - tauP)/H, and its centre moves
//     by at most tauP r_max/H.
//   Every other member of the patch lies within tauP of T's plane and is interior-disjoint from T, so it cannot
//   touch the ellipsoid: no FV feature or second FACE comes from the patch.
// The caller supplies tauP = 2 tau + alpha r_max:
//   - tau: the patch's slab half thickness, covering every member's float vertices;
//   - alpha: the largest |n_member - n_patch|.
// When the lane is not ok, run the full test (FV's exact test on the patch's members, plus FV's merge).
//
// One template for 4 lanes and for 1. Every operation is a tw::math lane op, so the bits do not depend on the
// width or the ISA. PoseT is any type with float t[3] and float M[3][3]; with kShared, all lanes take M from
// lane 0, for example the 4 wheels, whose axes are the car's columns and whose radii are equal.
#ifndef CC_FACE_H
#define CC_FACE_H

#include "../math/tw_lanes.h"
#include "../math/tw_math.h"

namespace cc {
namespace face {

namespace m = tw::math;

// member data, SoA, 4 lanes: v0 xyz, n xyz, u[3] xyz, a[3] xyz, tauP, r_max
enum Memb : int { fV0 = 0, fN = 3, fU = 6, fA = 15, fTau = 24, fR = 25, kMemb = 26 };
// outputs, SoA, 4 lanes: ok (1/0), P xyz, Sep xyz, K xyz
enum Out : int { oOk = 0, oP = 1, oSep = 4, oK = 7, kOut = 10 };

// Pose lanes, built in registers: 4 scalar stores read back as one vector would stall store forwarding.
template <class F> struct Lanes {
    template <class P> static F T(const P *const *p, int q) {
        alignas(64) float v[F::width];
        for (int l = 0; l < F::width; ++l) v[l] = p[l]->t[q];
        return F::load(v);
    }
    template <class P> static F M(const P *const *p, int a, int b) {
        alignas(64) float v[F::width];
        for (int l = 0; l < F::width; ++l) v[l] = p[l]->M[a][b];
        return F::load(v);
    }
};
template <> struct Lanes<m::f32x1> {
    template <class P> static m::f32x1 T(const P *const *p, int q) { return m::f32x1(p[0]->t[q]); }
    template <class P> static m::f32x1 M(const P *const *p, int a, int b) { return m::f32x1(p[0]->M[a][b]); }
};
#if TW_HAVE_SSE2
template <> struct Lanes<m::f32x4> {
    template <class P> static m::f32x4 T(const P *const *p, int q) {
        return m::f32x4(_mm_setr_ps(p[0]->t[q], p[1]->t[q], p[2]->t[q], p[3]->t[q]));
    }
    template <class P> static m::f32x4 M(const P *const *p, int a, int b) {
        return m::f32x4(_mm_setr_ps(p[0]->M[a][b], p[1]->M[a][b], p[2]->M[a][b], p[3]->M[a][b]));
    }
};
#endif

template <class F, bool kShared = false, class P>
TW_INLINE void Kernel(const P *const *pz, const float (*mb)[4], float (*out)[4]) {
    auto L = [&](int f) { return F::load(&mb[f][0]); };
    using PL = Lanes<F>;
    const F tx = PL::T(pz, 0), ty = PL::T(pz, 1), tz = PL::T(pz, 2);
    F M[9];
    for (int q = 0; q < 9; ++q) M[q] = kShared ? F(pz[0]->M[q / 3][q % 3]) : PL::M(pz, q / 3, q % 3);
    const F nx = L(fN), ny = L(fN + 1), nz = L(fN + 2);
    const F sd = (nx * (tx - L(fV0)) + ny * (ty - L(fV0 + 1))) + nz * (tz - L(fV0 + 2));
    const F Mn0 = (M[0] * nx + M[3] * ny) + M[6] * nz, Mn1 = (M[1] * nx + M[4] * ny) + M[7] * nz, Mn2 = (M[2] * nx + M[5] * ny) + M[8] * nz;
    const F H2 = (Mn0 * Mn0 + Mn1 * Mn1) + Mn2 * Mn2;
    const auto reach = m::cmp_gt(sd, F(1e-5f)) & m::cmp_le(sd * sd, H2 * F(0.9997f));
    const F iH2 = F(1.0f) / H2, kk = sd * iH2;
    const F MM0 = (M[0] * Mn0 + M[1] * Mn1) + M[2] * Mn2, MM1 = (M[3] * Mn0 + M[4] * Mn1) + M[5] * Mn2, MM2 = (M[6] * Mn0 + M[7] * Mn1) + M[8] * Mn2;
    const F Px = tx - kk * MM0, Py = ty - kk * MM1, Pz = tz - kk * MM2;
    F smax(-1e30f);
    for (int e = 0; e < 3; ++e) {
        const F se = (L(fU + 3 * e) * (Px - L(fA + 3 * e)) + L(fU + 3 * e + 1) * (Py - L(fA + 3 * e + 1))) + L(fU + 3 * e + 2) * (Pz - L(fA + 3 * e + 2));
        smax = m::max(smax, se);
    }
    const auto inside = m::cmp_lt(smax, F(-2e-5f));
    const F H = m::sqrt(H2), ih = H * iH2, tau = L(fTau), rmax = L(fR);
    const F dlo = m::max(F(0.0f), (sd - tau) * ih);
    const F g = (F(0.0f) - smax) - (tau * rmax * ih + F(2e-4f));
    const auto disk = m::cmp_ge(g, F(0.0f)) & m::cmp_ge(g * g, ((rmax * rmax) * (F(1.0f) - dlo * dlo)) * F(1.0002f));
    m::select(reach & inside & disk, F(1.0f), F(0.0f)).store(&out[oOk][0]);
    Px.store(&out[oP][0]); Py.store(&out[oP + 1][0]); Pz.store(&out[oP + 2][0]);
    const F ks = (sd - H) * iH2;
    (ks * MM0).store(&out[oSep][0]); (ks * MM1).store(&out[oSep + 1][0]); (ks * MM2).store(&out[oSep + 2][0]);
    (Mn0 * ih).store(&out[oK][0]); (Mn1 * ih).store(&out[oK + 1][0]); (Mn2 * ih).store(&out[oK + 2][0]);
}

// Fill one lane of the member data. v0: T's vertex 0; n: T's unit normal (front side); u[e]: in-plane outward
// unit normal of edge e (v_e -> v_e+1, = normalize(e x n)); a[e] = v_e.
inline void LoadMember(float (*mb)[4], int lane, const float v0[3], const float n[3], const float u[3][3], const float a[3][3], float tauP,
                       float rmax) {
    for (int q = 0; q < 3; ++q) { mb[fV0 + q][lane] = v0[q]; mb[fN + q][lane] = n[q]; }
    for (int e = 0; e < 3; ++e)
        for (int q = 0; q < 3; ++q) { mb[fU + 3 * e + q][lane] = u[e][q]; mb[fA + 3 * e + q][lane] = a[e][q]; }
    mb[fTau][lane] = tauP;
    mb[fR][lane] = rmax;
}

}  // namespace face
}  // namespace cc

#endif
