// FV-form geometry over lanes (scalar float or 4 SSE lanes), operation for operation as
// tw_exact_fast.h (which follows FV's source), so a lane gives the scalar bits. Only IEEE-exact
// operations are used per lane (add, sub, mul, div, sqrt, compare, select); -ffp-contract=off
// keeps the compiler from fusing them. (Architect C, src/twin/collide_c/cc_stage.cpp; shared with
// the M3 stage.)
//   MakeFrameFv   exact::MakeFrameFv's meshToUnit, c2w, n2w (4 frames per call on F4)
//   UnitNormal    exact::UnitNormal
//   Decide4       exact::Decide on 4 lanes (masks; plus the feature-only "fvViol" lanes)
//   FacePayload   exact::EmitFace's payload
#ifndef TW_EXACT_LANES_H
#define TW_EXACT_LANES_H

#include <cmath>
#include <cstdint>

#include <xmmintrin.h>
#include <emmintrin.h>

namespace tw::exact::lanes {

#define TW_EXL_INLINE inline __attribute__((always_inline))
// The lane-generic templates and the scalar form are also device functions in a CUDA/HIP TU (the GPU engine
// runs them with F = float, one pair per thread); the SSE lanes stay host-only.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TW_EXL_GENERIC __host__ __device__ inline __attribute__((always_inline))
#else
#define TW_EXL_GENERIC TW_EXL_INLINE
#endif
// The SSE lanes exist in host code only (a GPU device pass sees the scalar form and the templates).
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define TW_EXL_SSE 0
#else
#define TW_EXL_SSE 1
#endif

#if TW_EXL_SSE
// ---- 4 SSE lanes ----------------------------------------------------------------------
struct F4 {
    __m128 v;
};
TW_EXL_INLINE F4 operator+(F4 a, F4 b) { return {_mm_add_ps(a.v, b.v)}; }
TW_EXL_INLINE F4 operator-(F4 a, F4 b) { return {_mm_sub_ps(a.v, b.v)}; }
TW_EXL_INLINE F4 operator*(F4 a, F4 b) { return {_mm_mul_ps(a.v, b.v)}; }
TW_EXL_INLINE F4 operator/(F4 a, F4 b) { return {_mm_div_ps(a.v, b.v)}; }
TW_EXL_INLINE F4 operator-(F4 a) { return {_mm_xor_ps(a.v, _mm_set1_ps(-0.0f))}; }
TW_EXL_INLINE F4 Sqrt(F4 a) { return {_mm_sqrt_ps(a.v)}; }
TW_EXL_INLINE F4 Abs(F4 a) { return {_mm_andnot_ps(_mm_set1_ps(-0.0f), a.v)}; }
TW_EXL_INLINE F4 K(float s, F4) { return {_mm_set1_ps(s)}; }
// masks (all-ones lanes)
struct B4 {
    __m128 v;
};
TW_EXL_INLINE B4 operator<(F4 a, F4 b) { return {_mm_cmplt_ps(a.v, b.v)}; }
TW_EXL_INLINE B4 operator>(F4 a, F4 b) { return {_mm_cmpgt_ps(a.v, b.v)}; }
TW_EXL_INLINE B4 operator<=(F4 a, F4 b) { return {_mm_cmple_ps(a.v, b.v)}; }
TW_EXL_INLINE B4 operator&(B4 a, B4 b) { return {_mm_and_ps(a.v, b.v)}; }
TW_EXL_INLINE B4 operator|(B4 a, B4 b) { return {_mm_or_ps(a.v, b.v)}; }
TW_EXL_INLINE B4 AndNot(B4 a, B4 b) { return {_mm_andnot_ps(b.v, a.v)}; }   // a & ~b
TW_EXL_INLINE F4 Select(B4 m, F4 a, F4 b) { return {_mm_or_ps(_mm_and_ps(m.v, a.v), _mm_andnot_ps(m.v, b.v))}; }   // m ? a : b
TW_EXL_INLINE int Bits(B4 m) { return _mm_movemask_ps(m.v); }

#endif

// ---- scalar (the reference form) ---------------------------------------------------------
TW_EXL_GENERIC float Sqrt(float a) { return std::sqrt(a); }
TW_EXL_GENERIC float Abs(float a) { return std::fabs(a); }
TW_EXL_GENERIC float K(float s, float) { return s; }
TW_EXL_GENERIC float Select(bool m, float a, float b) { return m ? a : b; }

template <class F> struct V3 { F x, y, z; };
template <class F> struct M3 { V3<F> c[3]; };   // GmMat3 columns
template <class F> struct I3 { M3<F> r; V3<F> t; };   // GmIso4

template <class F> TW_EXL_GENERIC F Dot(const V3<F> &a, const V3<F> &b) {
    const F xy = a.x * b.x + a.y * b.y;
    return xy + a.z * b.z;
}
template <class F> TW_EXL_GENERIC V3<F> Add(const V3<F> &a, const V3<F> &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
template <class F> TW_EXL_GENERIC V3<F> Sub(const V3<F> &a, const V3<F> &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
template <class F> TW_EXL_GENERIC V3<F> Scale(const V3<F> &a, F s) { return {a.x * s, a.y * s, a.z * s}; }
template <class F> TW_EXL_GENERIC V3<F> Neg(const V3<F> &a) { return {-a.x, -a.y, -a.z}; }
template <class F> TW_EXL_GENERIC V3<F> Cross(const V3<F> &a, const V3<F> &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <class F> TW_EXL_GENERIC V3<F> Row(const M3<F> &m, int a) {
    return a == 0 ? V3<F>{m.c[0].x, m.c[1].x, m.c[2].x} : a == 1 ? V3<F>{m.c[0].y, m.c[1].y, m.c[2].y} : V3<F>{m.c[0].z, m.c[1].z, m.c[2].z};
}
template <class F> TW_EXL_GENERIC V3<F> TDir(const M3<F> &m, const V3<F> &v) { return {Dot(Row(m, 0), v), Dot(Row(m, 1), v), Dot(Row(m, 2), v)}; }
template <class F> TW_EXL_GENERIC V3<F> TPoint(const I3<F> &t, const V3<F> &p) { return Add(TDir(t.r, p), t.t); }
template <class F> TW_EXL_GENERIC M3<F> Compose(const M3<F> &first, const M3<F> &second) {
    return M3<F>{{TDir(second, first.c[0]), TDir(second, first.c[1]), TDir(second, first.c[2])}};
}
template <class F> TW_EXL_GENERIC I3<F> SetMult(const I3<F> &lhs, const I3<F> &rhs) { return I3<F>{Compose(lhs.r, rhs.r), TPoint(rhs, lhs.t)}; }
template <class F> TW_EXL_GENERIC M3<F> Transpose(const M3<F> &m) { return M3<F>{{Row(m, 0), Row(m, 1), Row(m, 2)}}; }
template <class F> TW_EXL_GENERIC I3<F> Inverse(const I3<F> &rhs) {
    const M3<F> inv = Transpose(rhs.r);
    return I3<F>{inv, TDir(inv, Neg(rhs.t))};
}
template <class F> TW_EXL_GENERIC I3<F> MultInverse(const I3<F> &self, const I3<F> &rhs) { return SetMult(self, Inverse(rhs)); }
template <class F> TW_EXL_GENERIC I3<F> Diagonal(const V3<F> &s) {
    const F z = K(0.0f, s.x);
    return I3<F>{M3<F>{{V3<F>{s.x, z, z}, V3<F>{z, s.y, z}, V3<F>{z, z, s.z}}}, V3<F>{z, z, z}};
}
template <class F> TW_EXL_GENERIC void ScaleRows(I3<F> &t, const V3<F> &s) {
    t.r.c[0].x = s.x * t.r.c[0].x; t.r.c[1].x = t.r.c[1].x * s.x; t.r.c[2].x = s.x * t.r.c[2].x; t.t.x = t.t.x * s.x;
    t.r.c[0].y = s.y * t.r.c[0].y; t.r.c[1].y = t.r.c[1].y * s.y; t.r.c[2].y = s.y * t.r.c[2].y; t.t.y = t.t.y * s.y;
    t.r.c[0].z = s.z * t.r.c[0].z; t.r.c[1].z = t.r.c[1].z * s.z; t.r.c[2].z = s.z * t.r.c[2].z; t.t.z = t.t.z * s.z;
}

// exact::MakeFrameFv's FV-form part: meshToUnit, c2w, n2w.
template <class F> struct FrameFv { I3<F> meshToUnit, c2w, n2w; };
template <class F>
TW_EXL_GENERIC FrameFv<F> MakeFrameFv(const I3<F> &ell, const V3<F> &rad, const I3<F> &mesh) {
    const F one = K(1.0f, rad.x);
    const V3<F> invRadii = {one / rad.x, one / rad.y, one / rad.z};
    const I3<F> ellipsoidToMesh = MultInverse(ell, mesh);
    const I3<F> meshToEllipsoid = Inverse(ellipsoidToMesh);
    FrameFv<F> f;
    f.meshToUnit = meshToEllipsoid;
    ScaleRows(f.meshToUnit, invRadii);
    f.c2w = SetMult(MultInverse(Diagonal(rad), meshToEllipsoid), mesh);
    f.n2w = SetMult(MultInverse(Diagonal(invRadii), meshToEllipsoid), mesh);
    return f;
}

// exact::UnitNormal
template <class F> TW_EXL_GENERIC V3<F> UnitNormal(const V3<F> vu[3], F *lenSq) {
    const V3<F> e01 = Sub(vu[1], vu[0]), e02 = Sub(vu[2], vu[0]);
    const F nX = e02.z * e01.y - e02.y * e01.z;
    const F nY = e01.z * e02.x - e02.z * e01.x;
    const F nZ = e01.x * e02.y - e02.x * e01.y;
    *lenSq = (nY * nY + nX * nX) + nZ * nZ;
    return V3<F>{nX, nY, nZ};
}

template <class F> TW_EXL_GENERIC F Abs1(const V3<F> &a) { return (Abs(a.x) + Abs(a.y)) + Abs(a.z); }

#if TW_EXL_SSE
// exact::Decide over 4 lanes: the same values per lane, the scalar's early returns resolved by
// masks in the same order. Returns the lane masks of kReject and kFace (the rest: kFvForm).
// fvViol: the kFvForm lanes that left through a certainly violated edge within reach (all earlier
// edges certainly not violated): FV's edge loop then ends at that edge, so its result is a feature
// (non-primary) contact or none, never a FACE.
TW_EXL_INLINE void Decide4(const V3<F4> vu[3], const V3<F4> &nr, F4 lenSq, float kDirEpsSq, float kMarginEps, B4 *reject, B4 *face, B4 *fvViol) {
    const F4 zero = {_mm_setzero_ps()}, one = K(1.0f, zero), two = K(2.0f, zero), eps = K(kDirEpsSq, zero), km = K(kMarginEps, zero);
    B4 und = {_mm_castsi128_ps(_mm_set1_epi32(-1))};   // still undecided
    B4 rej = {_mm_setzero_ps()};
    // if (!(lenSq > kDirEpsSq)) return kReject
    const B4 degen = AndNot(und, lenSq > eps);
    rej = rej | degen;
    und = AndNot(und, degen);
    const F4 P0 = -Dot(vu[0], nr);
    const F4 d0 = km * Abs1(vu[0]);
    const F4 P02 = P0 * P0;
    const F4 lo2 = (d0 * d0) * lenSq;
    {   // if (P0 < 0 && P02 > lo2) return kReject
        const B4 c = und & (P0 < zero) & (P02 > lo2);
        rej = rej | c;
        und = AndNot(und, c);
    }
    const F4 up = one + d0;
    {   // if (P0 > 0 && P02 > (up * up) * lenSq) return kReject
        const B4 c = und & (P0 > zero) & (P02 > (up * up) * lenSq);
        rej = rej | c;
        und = AndNot(und, c);
    }
    F4 R2 = lenSq - P02;
    R2 = {_mm_max_ps(R2.v, zero.v)};   // R2 > 0 ? R2 : 0
    const F4 slack0 = two * d0 + d0 * d0;
    B4 fvb = {_mm_setzero_ps()};
    for (int e = 0; e < 3; ++e) {
        const V3<F4> &a = vu[e];
        const V3<F4> &b = vu[e == 2 ? 0 : e + 1];
        const V3<F4> E = Sub(b, a);
        const F4 el2 = Dot(E, E);
        const F4 L2 = Select(el2 > eps, el2, one);
        const V3<F4> m = Cross(E, nr);
        const F4 S = -Dot(a, m);
        const F4 de = km * (Abs1(a) + one);
        const F4 S2 = S * S;
        const F4 L2len = L2 * lenSq;
        // if (S2 <= (de * de) * L2len) return kFvForm
        const B4 unsure = und & (S2 <= (de * de) * L2len);
        und = AndNot(und, unsure);
        // if (S < 0) continue; else return beyond ? kReject : kFvForm
        const B4 viol = AndNot(und, S < zero);
        const B4 beyond = S2 > L2 * R2 + L2len * (slack0 + two * de * (one + d0) + de * de);
        rej = rej | (viol & beyond);
        fvb = fvb | AndNot(viol, beyond);
        und = AndNot(und, viol);
    }
    const F4 dn = one - d0;
    *face = und & (P0 > zero) & (P02 > lo2) & (dn > zero) & (P02 < (dn * dn) * lenSq);
    *reject = rej;
    *fvViol = fvb;
}
#endif

// exact::NormalizeForCollision
template <class F, class Bm> TW_EXL_GENERIC V3<F> NormalizeSel(const V3<F> &v, float epsSq) {
    const F l2 = Dot(v, v);
    const F s = K(1.0f, l2) / Sqrt(l2);
    const Bm m = K(epsSq, l2) < l2;
    return V3<F>{Select(m, v.x * s, v.x), Select(m, v.y * s, v.y), Select(m, v.z * s, v.z)};
}

// exact::EmitFace's payload: P, N, Sep (world) and K = n (unit space).
template <class F, class Bm>
TW_EXL_GENERIC void FacePayload(const FrameFv<F> &f, const V3<F> vu[3], const V3<F> &nr, F lenSq, float kDirEpsSq, V3<F> *P, V3<F> *N, V3<F> *Sep,
                       V3<F> *Kn) {
    const F len = Sqrt(lenSq);
    const F invLen = K(1.0f, len) / len;
    const V3<F> n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};
    const F z = K(0.0f, len);
    const V3<F> zero = {z, z, z};
    const F pd = Dot(Sub(zero, vu[0]), n);
    const V3<F> projected = Add(zero, Scale(n, -pd));
    const V3<F> sep = Scale(n, pd - K(1.0f, len));
    *P = TPoint(f.c2w, projected);
    *N = NormalizeSel<F, Bm>(TDir(f.n2w.r, n), kDirEpsSq);
    *Sep = TDir(f.c2w.r, sep);
    *Kn = n;
}

}  // namespace tw::exact::lanes

#endif
