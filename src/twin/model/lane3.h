// Twin model: lane-generic vector helpers shared by the batch kernels
// (forces, response). F is a tw::math lane type (f32x1 at L = 1, f32x4,
// f32x8, f32x16; fpair types work through the generic paths). Every op
// mirrors vec3.h op for op, so a lane of an L-wide call gives the bits of
// the L = 1 call (docs/twin/m3-forces.md §7).
//
// State access works on both layouts with the same source:
//   TwinState (AoS, L = 1):  s.rpm is float,       s.vel is float[3]
//   TwinLanes<L> (SoA):      s.rpm is float[1][L], s.vel is float[3][L]
// Ld<F>(SF(s.rpm)), Ld<F>(s.absorb[w]), Ld3<F>(s.vel), Ld3At<F>(s.normalSum, 3 * w),
// and the matching St / St3 / St3At / masked StM, resolve by overload.
#ifndef TW_MODEL_LANE3_H
#define TW_MODEL_LANE3_H

#include <cstdint>
#include <cstring>

#include "tw_math.h"
#include "vec3.h"

namespace tw::model {

using tw::math::f32x1;
using tw::math::i32x1;
using tw::math::m32x1;

template <class F> using LaneI = typename F::int_type;
template <class F> using LaneM = typename F::mask_type;

// ---- masks and selects -------------------------------------------------------------
template <class F> TW_INLINE LaneM<F> MaskTrue() { return tw::math::cmp_eq(F(0.0f), F(0.0f)); }
template <class F> TW_INLINE LaneM<F> MaskFalse() { return tw::math::cmp_lt(F(0.0f), F(0.0f)); }
using tw::math::any;
using tw::math::all;
using tw::math::select;

// |x| (bitwise). f32x1 as one andps; tw::math::abs is bitwise for every width.
template <class F> TW_INLINE F FabsL(F x) { return tw::math::abs(x); }
TW_INLINE f32x1 FabsL(f32x1 x) { return f32x1(__builtin_fabsf(x.v)); }

// ---- int lanes -----------------------------------------------------------------------
template <class I> TW_INLINE auto IEq(I a, I b) { return tw::math::cmp_eq(a, b); }
template <class I> TW_INLINE auto INe(I a, I b) { return ~tw::math::cmp_eq(a, b); }
template <class I> TW_INLINE auto IGt(I a, I b) { return tw::math::cmp_gt(a, b); }
template <class I> TW_INLINE auto ILt(I a, I b) { return tw::math::cmp_gt(b, a); }
// unsigned 32-bit compares (tick times): flip the sign bit, compare signed
template <class I> TW_INLINE auto ULt(I a, I b) {
    const I s(static_cast<std::int32_t>(0x80000000u));
    return tw::math::cmp_gt(b ^ s, a ^ s);
}
template <class I> TW_INLINE auto ULe(I a, I b) { return ~ULt(b, a); }
TW_INLINE m32x1 ULt(i32x1 a, i32x1 b) { return m32x1(static_cast<std::uint32_t>(a.v) < static_cast<std::uint32_t>(b.v)); }
TW_INLINE m32x1 ULe(i32x1 a, i32x1 b) { return m32x1(static_cast<std::uint32_t>(a.v) <= static_cast<std::uint32_t>(b.v)); }
template <class F> TW_INLINE LaneI<F> IOf(LaneM<F> m) { return select(m, LaneI<F>(1), LaneI<F>(0)); }

// Lane bit set of a mask (bit i = lane i) and the mask of a bit set.
template <class F> TW_INLINE std::uint32_t MaskBits(LaneM<F> m) {
    alignas(64) std::int32_t t[F::width];
    select(m, LaneI<F>(1), LaneI<F>(0)).store(t);
    std::uint32_t b = 0u;
    for (int i = 0; i < F::width; ++i) b |= static_cast<std::uint32_t>(t[i] & 1) << i;
    return b;
}
template <> TW_INLINE std::uint32_t MaskBits<f32x1>(m32x1 m) { return m.v ? 1u : 0u; }
template <class F> TW_INLINE LaneM<F> MaskOf(std::uint32_t bits) {
    alignas(64) std::int32_t t[F::width];
    for (int i = 0; i < F::width; ++i) t[i] = static_cast<std::int32_t>((bits >> i) & 1u);
    return IGt(LaneI<F>::load(t), LaneI<F>(0));
}
template <> TW_INLINE m32x1 MaskOf<f32x1>(std::uint32_t bits) { return m32x1((bits & 1u) != 0u); }
#if TW_HAVE_AVX512F
template <> TW_INLINE std::uint32_t MaskBits<tw::math::f32x16>(tw::math::m32x16 m) { return static_cast<std::uint32_t>(m.v); }
template <> TW_INLINE tw::math::m32x16 MaskOf<tw::math::f32x16>(std::uint32_t bits) {
    return tw::math::m32x16(static_cast<__mmask16>(bits));
}
#endif

// ---- small-table lane lookups (exact loads: same value as table[idx]) --------------
// Deduced from the int lane type; the float result type is cvt_f32's.
template <class I> TW_INLINE auto LookupF(const float *table, I idx) -> decltype(tw::math::cvt_f32(idx)) {
    using F = decltype(tw::math::cvt_f32(idx));
    alignas(64) std::int32_t i[F::width];
    alignas(64) float v[F::width];
    idx.store(i);
    for (int l = 0; l < F::width; ++l) v[l] = table[i[l]];
    return F::load(v);
}
TW_INLINE f32x1 LookupF(const float *table, i32x1 idx) { return f32x1(table[idx.v]); }
// 16- and 32-entry tables (curve regions, material rows): register permutes where available.
// The index is wrapped to the table (as vpermps does), so an inactive lane holding stale data
// cannot read outside it; active lanes always hold an index inside the table.
template <class I> TW_INLINE auto Lookup16(const float *t16, I idx) -> decltype(tw::math::cvt_f32(idx)) { return LookupF(t16, idx & I(15)); }
template <class I> TW_INLINE auto Lookup32(const float *t32, I idx) -> decltype(tw::math::cvt_f32(idx)) { return LookupF(t32, idx & I(31)); }
TW_INLINE f32x1 Lookup16(const float *t, i32x1 idx) { return f32x1(t[idx.v]); }
TW_INLINE f32x1 Lookup32(const float *t, i32x1 idx) { return f32x1(t[idx.v]); }
#if TW_HAVE_AVX512F
TW_INLINE tw::math::f32x16 Lookup16(const float *t, tw::math::i32x16 idx) {
    return tw::math::f32x16(_mm512_permutexvar_ps(idx.v, _mm512_loadu_ps(t)));
}
TW_INLINE tw::math::f32x16 Lookup32(const float *t, tw::math::i32x16 idx) {
    return tw::math::f32x16(_mm512_permutex2var_ps(_mm512_loadu_ps(t), idx.v, _mm512_loadu_ps(t + 16)));
}
#endif
#if TW_HAVE_AVX2
TW_INLINE tw::math::f32x8 Lookup16(const float *t, tw::math::i32x8 idx) {
    const __m256 lo = _mm256_permutevar8x32_ps(_mm256_loadu_ps(t), idx.v);
    const __m256 hi = _mm256_permutevar8x32_ps(_mm256_loadu_ps(t + 8), idx.v);
    const __m256 sel = _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx.v, _mm256_set1_epi32(7)));
    return tw::math::f32x8(_mm256_blendv_ps(lo, hi, sel));
}
TW_INLINE tw::math::f32x8 Lookup32(const float *t, tw::math::i32x8 idx) {
    return tw::math::f32x8(_mm256_i32gather_ps(t, _mm256_and_si256(idx.v, _mm256_set1_epi32(31)), 4));
}
#endif

// per-lane variable left shift (count < 32)
template <class I> TW_INLINE I IShlVar(I a, I n) {
    alignas(64) std::int32_t x[sizeof(I) / 4], c[sizeof(I) / 4];
    a.store(x);
    n.store(c);
    for (std::size_t l = 0; l < sizeof(I) / 4; ++l) x[l] = static_cast<std::int32_t>(static_cast<std::uint32_t>(x[l]) << c[l]);
    return I::load(x);
}
TW_INLINE i32x1 IShlVar(i32x1 a, i32x1 n) {
    return i32x1(static_cast<std::int32_t>(static_cast<std::uint32_t>(a.v) << n.v));
}
#if TW_HAVE_AVX512F
TW_INLINE tw::math::i32x16 IShlVar(tw::math::i32x16 a, tw::math::i32x16 n) {
    return tw::math::i32x16(_mm512_sllv_epi32(a.v, n.v));
}
#endif
#if TW_HAVE_AVX2
TW_INLINE tw::math::i32x8 IShlVar(tw::math::i32x8 a, tw::math::i32x8 n) { return tw::math::i32x8(_mm256_sllv_epi32(a.v, n.v)); }
#endif

// per-lane product with a constant (cell indices)
template <class I> TW_INLINE I IMulC(I a, std::int32_t c) {
    alignas(64) std::int32_t x[sizeof(I) / 4];
    a.store(x);
    for (std::size_t l = 0; l < sizeof(I) / 4; ++l) x[l] = static_cast<std::int32_t>(static_cast<std::uint32_t>(x[l]) * static_cast<std::uint32_t>(c));
    return I::load(x);
}
TW_INLINE i32x1 IMulC(i32x1 a, std::int32_t c) {
    return i32x1(static_cast<std::int32_t>(static_cast<std::uint32_t>(a.v) * static_cast<std::uint32_t>(c)));
}
#if TW_HAVE_AVX512F
TW_INLINE tw::math::i32x16 IMulC(tw::math::i32x16 a, std::int32_t c) {
    return tw::math::i32x16(_mm512_mullo_epi32(a.v, _mm512_set1_epi32(c)));
}
#endif
#if TW_HAVE_AVX2
TW_INLINE tw::math::i32x8 IMulC(tw::math::i32x8 a, std::int32_t c) {
    return tw::math::i32x8(_mm256_mullo_epi32(a.v, _mm256_set1_epi32(c)));
}
#endif

// bytes base[idx] per lane (the 128 x 128 fake-contact texture); base must allow 3 bytes of overread
template <class F> TW_INLINE LaneI<F> LoadBytes(const std::uint8_t *base, LaneI<F> idx) {
    alignas(64) std::int32_t i[F::width];
    idx.store(i);
    for (int l = 0; l < F::width; ++l) i[l] = base[i[l]];
    return LaneI<F>::load(i);
}
template <> TW_INLINE i32x1 LoadBytes<f32x1>(const std::uint8_t *base, i32x1 idx) { return i32x1(base[idx.v]); }
#if TW_HAVE_AVX512F
template <> TW_INLINE tw::math::i32x16 LoadBytes<tw::math::f32x16>(const std::uint8_t *base, tw::math::i32x16 idx) {
    return tw::math::i32x16(_mm512_and_si512(_mm512_i32gather_epi32(idx.v, base, 1), _mm512_set1_epi32(0xff)));
}
#endif

// ---- 3-vectors and rotations ----------------------------------------------------------
template <class F> struct V3L {
    F x, y, z;
};
template <class F> TW_INLINE V3L<F> v3l(F x, F y, F z) { return V3L<F>{x, y, z}; }
template <class F> TW_INLINE V3L<F> v3l(const float *p) { return V3L<F>{F(p[0]), F(p[1]), F(p[2])}; }   // broadcast constant
template <class F> TW_INLINE V3L<F> v3l(V3 a) { return V3L<F>{F(a.x), F(a.y), F(a.z)}; }
template <class F> TW_INLINE V3L<F> operator+(V3L<F> a, V3L<F> b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
template <class F> TW_INLINE V3L<F> operator-(V3L<F> a, V3L<F> b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
template <class F> TW_INLINE V3L<F> operator-(V3L<F> a) { return {-a.x, -a.y, -a.z}; }
template <class F> TW_INLINE V3L<F> operator*(V3L<F> a, F s) { return {a.x * s, a.y * s, a.z * s}; }
template <class F> TW_INLINE V3L<F> &operator+=(V3L<F> &a, V3L<F> b) { a = a + b; return a; }
template <class F> TW_INLINE F dotl(V3L<F> a, V3L<F> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
template <class F> TW_INLINE F len2l(V3L<F> a) { return dotl(a, a); }
template <class F> TW_INLINE V3L<F> crossl(V3L<F> a, V3L<F> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <class F> TW_INLINE V3L<F> Sel3(LaneM<F> m, V3L<F> a, V3L<F> b) {
    return {select(m, a.x, b.x), select(m, a.y, b.y), select(m, a.z, b.z)};
}
// vec3.h normalize_or, lane-wise: a / |a| where |a|^2 > minLen2, else `fallback`
template <class F> TW_INLINE V3L<F> NormalizeOrL(V3L<F> a, V3L<F> fallback, float minLen2) {
    const F l2 = len2l(a);
    const LaneM<F> ok = tw::math::cmp_gt(l2, F(minLen2));
    if (!any(ok)) return fallback;
    return Sel3(ok, a * (F(1.0f) / tw::math::sqrt(l2)), fallback);
}

template <class F> struct M3L {
    V3L<F> c0, c1, c2;
};
template <class F> TW_INLINE V3L<F> mull(const M3L<F> &m, V3L<F> v) {
    return {m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z, m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z,
            m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z};
}
template <class F> TW_INLINE V3L<F> mul_tl(const M3L<F> &m, V3L<F> v) { return {dotl(m.c0, v), dotl(m.c1, v), dotl(m.c2, v)}; }
// vec3.h rot_from_quat (gm_matrix.cpp:55-82)
template <class F> TW_INLINE M3L<F> RotFromQuatL(F x, F y, F z, F w) {
    const F x2 = x * F(2.0f), y2 = y * F(2.0f), z2 = z * F(2.0f);
    const F xw = w * x2, yw = w * y2, zw = w * z2;
    const F xx = x * x2, xy = x * y2, xz = x * z2;
    const F yy = y * y2, yz = y * z2, zz = z * z2;
    const F one(1.0f);
    M3L<F> m;
    m.c0 = {(one - yy) - zz, xy + zw, xz - yw};
    m.c1 = {xy - zw, (one - xx) - zz, yz + xw};
    m.c2 = {xz + yw, yz - xw, (one - xx) - yy};
    return m;
}

// ---- state access (both layouts) ------------------------------------------------------
// scalar-field view: TwinState field as is, TwinLanes [1][L] field as its row
template <class T> TW_INLINE T &SF(T &x) { return x; }
template <class T, int L> TW_INLINE T (&SF(T (&x)[1][L]))[L] { return x[0]; }

// floats
template <class F> TW_INLINE F Ld(const float &x) { return F(x); }
template <class F, int L> TW_INLINE F Ld(const float (&x)[L]) { static_assert(L == F::width, "lane width"); return F::load(x); }
TW_INLINE void St(float &x, f32x1 v) { x = v.v; }
template <class F, int L> TW_INLINE void St(float (&x)[L], F v) { v.store(x); }
// ints: AoS u8 / i8 / u16 / u32 fields, SoA u32 / i32 lane words
template <class F> TW_INLINE LaneI<F> LdI(const std::uint8_t &x) { return LaneI<F>(static_cast<std::int32_t>(x)); }
template <class F> TW_INLINE LaneI<F> LdI(const std::int8_t &x) { return LaneI<F>(static_cast<std::int32_t>(x)); }
template <class F> TW_INLINE LaneI<F> LdI(const std::uint16_t &x) { return LaneI<F>(static_cast<std::int32_t>(x)); }
template <class F> TW_INLINE LaneI<F> LdI(const std::uint32_t &x) { return LaneI<F>(static_cast<std::int32_t>(x)); }
template <class F, int L> TW_INLINE LaneI<F> LdI(const std::uint32_t (&x)[L]) {
    return LaneI<F>::load(reinterpret_cast<const std::int32_t *>(x));
}
template <class F, int L> TW_INLINE LaneI<F> LdI(const std::int32_t (&x)[L]) { return LaneI<F>::load(x); }
TW_INLINE void StI(std::uint8_t &x, i32x1 v) { x = static_cast<std::uint8_t>(v.v); }
TW_INLINE void StI(std::int8_t &x, i32x1 v) { x = static_cast<std::int8_t>(v.v); }
TW_INLINE void StI(std::uint16_t &x, i32x1 v) { x = static_cast<std::uint16_t>(v.v); }
TW_INLINE void StI(std::uint32_t &x, i32x1 v) { x = static_cast<std::uint32_t>(v.v); }
template <class I, int L> TW_INLINE void StI(std::uint32_t (&x)[L], I v) { v.store(reinterpret_cast<std::int32_t *>(x)); }
template <class I, int L> TW_INLINE void StI(std::int32_t (&x)[L], I v) { v.store(x); }
// 3-vectors: AoS float[3] / SoA float[3][L]; at an offset of a [12] / [12][L] field
template <class F> TW_INLINE V3L<F> Ld3(const float (&x)[3]) { return {F(x[0]), F(x[1]), F(x[2])}; }
template <class F, int L> TW_INLINE V3L<F> Ld3(const float (&x)[3][L]) { return {F::load(x[0]), F::load(x[1]), F::load(x[2])}; }
TW_INLINE void St3(float (&x)[3], V3L<f32x1> v) { x[0] = v.x.v; x[1] = v.y.v; x[2] = v.z.v; }
template <class F, int L> TW_INLINE void St3(float (&x)[3][L], V3L<F> v) { v.x.store(x[0]); v.y.store(x[1]); v.z.store(x[2]); }
template <class F, int N> TW_INLINE V3L<F> Ld3At(const float (&x)[N], int o) { return {F(x[o]), F(x[o + 1]), F(x[o + 2])}; }
template <class F, int N, int L> TW_INLINE V3L<F> Ld3At(const float (&x)[N][L], int o) {
    return {F::load(x[o]), F::load(x[o + 1]), F::load(x[o + 2])};
}
template <int N> TW_INLINE void St3At(float (&x)[N], int o, V3L<f32x1> v) { x[o] = v.x.v; x[o + 1] = v.y.v; x[o + 2] = v.z.v; }
template <class F, int N, int L> TW_INLINE void St3At(float (&x)[N][L], int o, V3L<F> v) {
    v.x.store(x[o]); v.y.store(x[o + 1]); v.z.store(x[o + 2]);
}
// masked stores: lanes outside `m` keep their value
template <class F, class D> TW_INLINE void StM(D &dst, LaneM<F> m, F v) { St(dst, select(m, v, Ld<F>(dst))); }
template <class F, class D> TW_INLINE void StIM(D &dst, LaneM<F> m, LaneI<F> v) { StI(dst, select(m, v, LdI<F>(dst))); }
template <class F, class D> TW_INLINE void St3M(D &dst, LaneM<F> m, V3L<F> v) { St3(dst, Sel3(m, v, Ld3<F>(dst))); }
template <class F, class D> TW_INLINE void St3AtM(D &dst, int o, LaneM<F> m, V3L<F> v) { St3At(dst, o, Sel3(m, v, Ld3At<F>(dst, o))); }

}  // namespace tw::model

#endif
