// Architect C: lanes for the rival stage. The FV-form (exact) part is shared with the M3 stage in
// src/twin/world/tw_exact_lanes.h (tw::exact::lanes); this file adds C's noise-level narrow phase.
#ifndef CC_LANES_H
#define CC_LANES_H

#include "tw_exact_lanes.h"

#define CC_LI TW_EXL_INLINE

namespace tw::collide_c {
namespace lanes = tw::exact::lanes;
}

namespace tw::collide_c::noise {
using namespace tw::exact::lanes;

// ---- noise-level narrow phase (lead decision 2026-11-08: contacts within gate-1 floors, decisions
// equal to FV's except at near-ties) ---------------------------------------------------------

// lane-type helpers for UnitContact (scalar float/bool for the GPU engine, F4/B4 here; F16/B16 below on AVX-512)
TW_EXL_GENERIC float Zero(float) { return 0.0f; }
TW_EXL_GENERIC bool NoMask(bool) { return false; }
TW_EXL_GENERIC float Max(float a, float b) { return a > b ? a : b; }   // as _mm_max_ps
TW_EXL_GENERIC bool AndNot(bool a, bool b) { return a && !b; }
TW_EXL_GENERIC int Bits(bool m) { return m ? 1 : 0; }
#if TW_EXL_SSE
CC_LI F4 Zero(F4) { return F4{_mm_setzero_ps()}; }
CC_LI B4 NoMask(B4) { return B4{_mm_setzero_ps()}; }
CC_LI F4 Max(F4 a, F4 b) { return F4{_mm_max_ps(a.v, b.v)}; }   // a > b ? a : b
#endif

#if defined(__AVX512F__) && TW_EXL_SSE
// ---- 16 AVX-512 lanes ------------------------------------------------------------------
struct F16 { __m512 v; };
struct B16 { __mmask16 m; };
CC_LI F16 operator+(F16 a, F16 b) { return {_mm512_add_ps(a.v, b.v)}; }
CC_LI F16 operator-(F16 a, F16 b) { return {_mm512_sub_ps(a.v, b.v)}; }
CC_LI F16 operator*(F16 a, F16 b) { return {_mm512_mul_ps(a.v, b.v)}; }
CC_LI F16 operator/(F16 a, F16 b) { return {_mm512_div_ps(a.v, b.v)}; }
CC_LI F16 operator-(F16 a) { return {_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(static_cast<int>(0x80000000u))))}; }
CC_LI F16 Sqrt(F16 a) { return {_mm512_sqrt_ps(a.v)}; }
CC_LI F16 K(float s, F16) { return {_mm512_set1_ps(s)}; }
CC_LI F16 Zero(F16) { return {_mm512_setzero_ps()}; }
CC_LI F16 Max(F16 a, F16 b) { return {_mm512_max_ps(a.v, b.v)}; }
CC_LI B16 operator<(F16 a, F16 b) { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ)}; }
CC_LI B16 operator>(F16 a, F16 b) { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ)}; }
CC_LI B16 operator<=(F16 a, F16 b) { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ)}; }
CC_LI B16 operator&(B16 a, B16 b) { return {static_cast<__mmask16>(a.m & b.m)}; }
CC_LI B16 operator|(B16 a, B16 b) { return {static_cast<__mmask16>(a.m | b.m)}; }
CC_LI B16 AndNot(B16 a, B16 b) { return {static_cast<__mmask16>(a.m & ~b.m)}; }
CC_LI B16 NoMask(B16) { return {0}; }
CC_LI F16 Select(B16 m, F16 a, F16 b) { return {_mm512_mask_blend_ps(m.m, b.v, a.v)}; }   // m ? a : b
CC_LI int Bits(B16 m) { return m.m; }
// 16 rows of 16 floats -> 16 columns (r[i] lane l = row l's float i)
CC_LI void Transpose16(__m512 r[16]) {
    __m512 t[16];
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_shuffle_ps(t[i], t[i + 2], _MM_SHUFFLE(1, 0, 1, 0));
        r[i + 1] = _mm512_shuffle_ps(t[i], t[i + 2], _MM_SHUFFLE(3, 2, 3, 2));
        r[i + 2] = _mm512_shuffle_ps(t[i + 1], t[i + 3], _MM_SHUFFLE(1, 0, 1, 0));
        r[i + 3] = _mm512_shuffle_ps(t[i + 1], t[i + 3], _MM_SHUFFLE(3, 2, 3, 2));
    }
#pragma GCC unroll 16
    for (int k = 0; k < 4; ++k) {
        const __m512 a = r[k], b = r[4 + k], c = r[8 + k], d = r[12 + k];
        const __m512 ab0 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512 ab1 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(3, 1, 3, 1));
        const __m512 cd0 = _mm512_shuffle_f32x4(c, d, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512 cd1 = _mm512_shuffle_f32x4(c, d, _MM_SHUFFLE(3, 1, 3, 1));
        t[k] = _mm512_shuffle_f32x4(ab0, cd0, _MM_SHUFFLE(2, 0, 2, 0));
        t[4 + k] = _mm512_shuffle_f32x4(ab1, cd1, _MM_SHUFFLE(2, 0, 2, 0));
        t[8 + k] = _mm512_shuffle_f32x4(ab0, cd0, _MM_SHUFFLE(3, 1, 3, 1));
        t[12 + k] = _mm512_shuffle_f32x4(ab1, cd1, _MM_SHUFFLE(3, 1, 3, 1));
    }
#pragma GCC unroll 16
    for (int i = 0; i < 16; ++i) r[i] = t[i];
}
#endif

// exact::MakeFrameFv's meshToUnit alone (FV's operations; the maps back to world are not needed).
// invRadii = 1 / radii (the caller's, the same IEEE quotients).
template <class F>
TW_EXL_GENERIC I3<F> MakeMeshToUnitInv(const I3<F> &ell, const V3<F> &invRadii, const I3<F> &mesh, I3<F> *backOut = nullptr) {
    const I3<F> ellipsoidToMesh = MultInverse(ell, mesh);
    I3<F> m2u = Inverse(ellipsoidToMesh);
    if (backOut) *backOut = Inverse(m2u);
    ScaleRows(m2u, invRadii);
    return m2u;
}
template <class F>
TW_EXL_GENERIC I3<F> MakeMeshToUnit(const I3<F> &ell, const V3<F> &rad, const I3<F> &mesh, I3<F> *backOut = nullptr) {
    const F one = K(1.0f, rad.x);
    const V3<F> invRadii = {one / rad.x, one / rad.y, one / rad.z};
    const I3<F> ellipsoidToMesh = MultInverse(ell, mesh);
    I3<F> m2u = Inverse(ellipsoidToMesh);
    // the way back FV's c2w takes (its MultInverse(Diagonal(rad), meshToEllipsoid) factor, uncomposed): the
    // inverse of the rounded meshToEllipsoid, so a unit-space point returns through the same rounding
    if (backOut) *backOut = Inverse(m2u);
    ScaleRows(m2u, invRadii);
    return m2u;
}

// FV's SSphereMeshCollide::CollideTriangle on unit-space vertices, operation for operation
// (tw_exact_fast.h FvCollide / FvTriangle, lanes as masks): degenerate skip, plane reach
// 0 <= pd <= 1, the first violated edge in stored order (reject beyond reach; else a feature at the
// start vertex, the edge point or the end vertex, with FV's distance tests and its endpoint-B form:
// normal from sqrt(distance)), else the FACE. On FV's unit-space vertices the decisions and the
// unit-space payload are FV's bits. Masks: face, feat (a contact each); outputs P, N, Sep, K = n in
// unit space.
template <class F, class B>
TW_EXL_GENERIC void UnitContact(const V3<F> vu[3], float kDirEpsSq, float kCollisionDistance, B *face, B *feat, V3<F> *P, V3<F> *N, V3<F> *Sep, V3<F> *Kn) {
    const F zero = Zero(F{}), one = K(1.0f, zero), eps = K(kDirEpsSq, zero), cdist = K(kCollisionDistance, zero);
    const B none = NoMask(B{});
    F lenSq;
    const V3<F> nr = UnitNormal(vu, &lenSq);
    B und = eps < lenSq;
    const F invLen = one / Sqrt(lenSq);
    const V3<F> n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};
    const V3<F> z3 = {zero, zero, zero};
    const F pd = Dot(Sub(z3, vu[0]), n);
    und = AndNot(AndNot(und, one < pd), pd < zero);
    const F edgeReach = Sqrt(Max(one - pd * pd, zero));
    const V3<F> proj = Add(z3, Scale(n, -pd));
    B fe = none, cont = none, endB = none;
    V3<F> fpt = {zero, zero, zero};
    F minD = zero;
    for (int e = 0; e < 3; ++e) {
        const V3<F> &start = vu[e];
        const V3<F> &end = vu[e == 2 ? 0 : e + 1];
        const V3<F> E = Sub(end, start);
        const F el2 = Dot(E, E);
        const F s = one / Sqrt(el2);
        const B big = eps < el2;
        const V3<F> dir = {Select(big, E.x * s, E.x), Select(big, E.y * s, E.y), Select(big, E.z * s, E.z)};
        const V3<F> enrm = Cross(dir, n);
        const V3<F> rel = Sub(proj, start);
        const F ed = Dot(rel, enrm);
        und = AndNot(und, edgeReach < ed);   // beyond reach: no contact
        const B viol = und & (zero < ed);
        if (Bits(viol)) {
            const F fs = Dot(rel, dir);
            const F fend = Dot(Sub(proj, end), dir);
            const B cA = viol & (fs < zero);
            const B cB = AndNot(AndNot(viol, cA), zero < fend);
            const B cC = AndNot(AndNot(viol, cA), cB);
            const V3<F> pB = Sub(proj, Scale(enrm, ed));
            fpt = {Select(cA, start.x, Select(cB, pB.x, Select(cC, end.x, fpt.x))), Select(cA, start.y, Select(cB, pB.y, Select(cC, end.y, fpt.y))),
                   Select(cA, start.z, Select(cB, pB.z, Select(cC, end.z, fpt.z)))};
            minD = Select(cB, cdist, Select(viol, eps, minD));
            cont = cont | cA;
            endB = endB | cC;
            fe = fe | viol;
            und = AndNot(und, viol);
        }
    }
    *face = und & (zero < pd);
    // features: FV's EmitFeature (squared distance tests) or EmitEndpointB (distance tests, sqrt(d))
    const V3<F> toC = Sub(z3, fpt);
    const F dSq = Dot(toC, toC);
    const F d = Sqrt(dSq);
    const B okF = AndNot(AndNot(AndNot(fe, endB), cont & (one < dSq)), AndNot(fe, minD < dSq));
    const B okB = AndNot(AndNot(endB, one < d), AndNot(endB, minD < d));
    *feat = okF | okB;
    const F dd = Select(endB, Sqrt(d), d);
    const F inv = one / dd;
    const V3<F> normal = Scale(toC, inv);
    const V3<F> pen = Scale(toC, (dd - one) * inv);
    const F along = Dot(pen, n);
    const B f = *face;
    *P = {Select(f, proj.x, fpt.x), Select(f, proj.y, fpt.y), Select(f, proj.z, fpt.z)};
    *N = {Select(f, n.x, normal.x), Select(f, n.y, normal.y), Select(f, n.z, normal.z)};
    const F sepS = Select(f, pd - one, along);
    *Sep = Scale(n, sepS);
    *Kn = n;
}


#if TW_EXL_SSE
CC_LI void UnitContact4(const V3<F4> vu[3], float kDirEpsSq, float kCollisionDistance, B4 *face, B4 *feat, V3<F4> *P, V3<F4> *N, V3<F4> *Sep,
                        V3<F4> *Kn) {
    UnitContact<F4, B4>(vu, kDirEpsSq, kCollisionDistance, face, feat, P, N, Sep, Kn);
}
#endif

}  // namespace tw::collide_c::noise

#endif
