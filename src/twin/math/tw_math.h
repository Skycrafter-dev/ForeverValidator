// Twin deterministic math: functions.
//
// Every function is a template over a lane type F (tw_lanes.h: f32x1,
// f32x4, f32x8, f32x16, fpair<F>, vf<W>) and runs one fixed sequence of
// correctly rounded IEEE binary32 operations (+ - * / sqrt), compares,
// selects, bit operations and exact conversions. No FMA, no libm, no
// rcp/rsqrt, no table lookups. Scalar `float` overloads call the f32x1 form,
// so scalar and every SIMD width return identical bits for every input
// (tested exhaustively, see docs/twin/math.md). The scalar compiler may turn
// a select into a branch; values are unaffected.
//
// Requirements on the caller: compile with -ffp-contract=off (GCC/Clang) or
// /fp:precise without /fp:contract (MSVC), no fast-math, and run under the
// canonical FP mode (tw_fpenv.h: round to nearest, IEEE denormals).
//
// Max error in ulp vs a double reference, every float of the domain
// (docs/twin/math.md has where each maximum occurs):
//   sin, cos, sincos   0.85 for |x| <= 12867 (= 2^13 pi/2); beyond that not
//                      accurate but finite, deterministic and in [-1, 1];
//                      NaN for inf/NaN
//   sincos_small       0.79 for |x| <= pi/4 (no reduction; equals sincos there)
//   asin 0.90, acos 0.91 on [-1, 1] (NaN outside); asin_small 0.67 on [-1/2, 1/2]
//   atan 0.85 (all inputs); atan2 1.48 (60M samples; IEEE zero/inf cases exact)
//   exp 0.99 (all inputs: +0 below -103.97, inf above 88.72, denormals rounded once)
//   rsqrt 1.49 (1/sqrt, two roundings); sqrt correctly rounded
//   floor, ceil, trunc, rint, fmod_pow2, conversions: exact
// NaN results are always the canonical quiet NaN 0x7fc00000 (so NaN bits do
// not depend on compiler operand order), except floor/ceil/trunc/rint, which
// return a NaN input unchanged, and sqrt/rsqrt/recip, which return the
// hardware NaN like any basic operation.
#ifndef TW_MATH_H
#define TW_MATH_H

#include <type_traits>

#include "tw_lanes.h"

namespace tw {
namespace math {

template <class F> using lane_int_t = typename F::int_type;
template <class F> using lane_mask_t = typename F::mask_type;
template <class F, class R = F>
using if_lane_t = typename std::enable_if<is_f32_lane<F>::value, R>::type;

namespace k {
// Bit-exact constants (hex float literals; values in docs/twin/math.md).
constexpr float kMagicRound = 0x1.8p23f;  // 1.5*2^23: (x + M) - M rounds to an integer, |x| < 2^22
constexpr float kTwoPow23 = 0x1p23f;
constexpr float kTwoOverPi = 0x1.45f306p-1f;
// pi/2 = P1 + P2 + P3 + P4 + P5 (+2.7e-24). P1 has 8 significant bits and
// P2..P4 have 11, so k*P1 .. k*P4 are exact for integer k <= 2^13
// (|x| <= 12867); P5 is the rounded remainder.
constexpr float kP1 = 0x1.92p+0f;
constexpr float kP2 = 0x1.fb4p-12f;
constexpr float kP3 = 0x1.444p-24f;
constexpr float kP4 = 0x1.68cp-39f;
constexpr float kP5 = 0x1.1a6264p-54f;
constexpr float kPio2Hi = 0x1.921fb6p+0f;   // float(pi/2)
constexpr float kPio2Lo = -0x1.777a5cp-25f; // pi/2 - kPio2Hi
constexpr float kPio4Hi = 0x1.921fb6p-1f;
constexpr float kPio4Lo = -0x1.777a5cp-26f;
constexpr float kPiHi = 0x1.921fb6p+1f;
constexpr float kPiLo = -0x1.777a5cp-24f;
// sin(r) = r + r^3 (S1 + S2 r^2 + S3 r^4), cos(r) = 1 - r^2/2 + r^4 (C1 + C2 r^2 + C3 r^4), |r| <= pi/4
constexpr float kS1 = -0x1.555546p-3f;
constexpr float kS2 = 0x1.11073cp-7f;
constexpr float kS3 = -0x1.9943f2p-13f;
constexpr float kC1 = 0x1.55554ap-5f;
constexpr float kC2 = -0x1.6c0c34p-10f;
constexpr float kC3 = 0x1.99eb9cp-16f;
// asin(s) = s + s z (A0 + A1 z + ... + A4 z^4), z = s^2 <= 1/4
constexpr float kA0 = 0x1.5555c8p-3f;
constexpr float kA1 = 0x1.3301ecp-4f;
constexpr float kA2 = 0x1.747d8ep-5f;
constexpr float kA3 = 0x1.8c2fc6p-6f;
constexpr float kA4 = 0x1.5966a4p-5f;
// atan(u) = u - u (z (T0 + w (T2 + w T4)) + w (T1 + w T3)), z = u^2, w = z^2,
// |u| <= 7/16 (fdlibm s_atanf.c coefficients)
constexpr float kT0 = 0x1.555552p-2f;
constexpr float kT1 = -0x1.99953p-3f;
constexpr float kT2 = 0x1.23ea1ap-3f;
constexpr float kT3 = -0x1.b4248ep-4f;
constexpr float kT4 = 0x1.f9584ap-5f;
constexpr float kAtanHalfHi = 0x1.dac67p-2f;        // atan(0.5) = hi + lo
constexpr float kAtanHalfLo = 0x1.586ed4p-28f;
constexpr float kAtanThreeHalfHi = 0x1.f730bep-1f;  // atan(1.5)
constexpr float kAtanThreeHalfLo = -0x1.afc12cp-26f;
// pi minus the offsets above (used for x < 0, so pi - atan(t) keeps both tails)
constexpr float kPiMinusAtanHalfHi = 0x1.56c6e8p+1f;
constexpr float kPiMinusAtanHalfLo = -0x1.8d014ap-24f;
constexpr float kThreePio4Hi = 0x1.2d97c8p+1f;
constexpr float kThreePio4Lo = -0x1.99bc5cp-28f;
constexpr float kPiMinusAtanThreeHalfHi = 0x1.145386p+1f;
constexpr float kPiMinusAtanThreeHalfLo = -0x1.71423ap-29f;
// exp(r) = 1 + r + r^2 (E0 + E1 r + ... + E5 r^5), |r| <= ln2/2
constexpr float kLog2e = 0x1.715476p+0f;
constexpr float kLn2Hi = 0x1.63p-1f;  // 9 significant bits: k*kLn2Hi exact for |k| <= 2^15
constexpr float kLn2Lo = -0x1.bd0106p-13f;
constexpr float kE0 = 0x1p-1f;
constexpr float kE1 = 0x1.555554p-3f;
constexpr float kE2 = 0x1.555382p-5f;
constexpr float kE3 = 0x1.11121p-7f;
constexpr float kE4 = 0x1.6e879cp-10f;
constexpr float kE5 = 0x1.a0d2cep-13f;
constexpr float kExpMin = -104.0f;  // exp(-104) < 2^-150: rounds to +0
constexpr float kExpMax = 89.0f;    // exp(89) > FLT_MAX: +inf
constexpr std::int32_t kSignBit = INT32_MIN;
constexpr std::int32_t kAbsMask = 0x7fffffff;
constexpr std::int32_t kQNaN = 0x7fc00000;
constexpr std::int32_t kInfBits = 0x7f800000;
}  // namespace k

// ============================================================ basic helpers

template <class F> TW_HD TW_INLINE if_lane_t<F> qnan() { return as_float(lane_int_t<F>(k::kQNaN)); }
template <class F> TW_HD TW_INLINE if_lane_t<F> infinity() { return as_float(lane_int_t<F>(k::kInfBits)); }

template <class F> TW_HD TW_INLINE if_lane_t<F> abs(F x) {
    return as_float(as_int(x) & lane_int_t<F>(k::kAbsMask));
}
// magnitude of `mag` with the sign bit of `sgn` (exact, also for NaN/zero)
template <class F> TW_HD TW_INLINE if_lane_t<F> copysign(F mag, F sgn) {
    using I = lane_int_t<F>;
    return as_float((as_int(mag) & I(k::kAbsMask)) | (as_int(sgn) & I(k::kSignBit)));
}
// x with its sign flipped where s is negative (sign bit set)
template <class F> TW_HD TW_INLINE if_lane_t<F> mulsign(F x, F s) {
    return as_float(as_int(x) ^ (as_int(s) & lane_int_t<F>(k::kSignBit)));
}
template <class F> TW_HD TW_INLINE if_lane_t<F, lane_mask_t<F>> signbit(F x) {
    using I = lane_int_t<F>;
    return cmp_gt(I(0), as_int(x));
}
template <class F> TW_HD TW_INLINE if_lane_t<F, lane_mask_t<F>> isnan(F x) { return cmp_unord(x, x); }
template <class F> TW_HD TW_INLINE if_lane_t<F, lane_mask_t<F>> isfinite(F x) {
    return cmp_le(abs(x), F(FLT_MAX));
}

// min/max with the SSE minps/maxps rule: the second operand is returned when
// either is NaN or both are zero. Never use std::min/std::max on floats in
// the twin (their operand order differs).
template <class F> TW_HD TW_INLINE if_lane_t<F> min(F a, F b) { return select(cmp_lt(a, b), a, b); }
template <class F> TW_HD TW_INLINE if_lane_t<F> max(F a, F b) { return select(cmp_gt(a, b), a, b); }
// clamp(x, lo, hi) = min(max(x, lo), hi); NaN -> lo
template <class F> TW_HD TW_INLINE if_lane_t<F> clamp(F x, F lo, F hi) { return min(max(x, lo), hi); }

template <class F> TW_HD TW_INLINE if_lane_t<F> recip(F x) { return F(1.0f) / x; }
// 1/sqrt(x) as two correctly rounded operations (<= 1 ulp + 0.5 ulp); never
// the approximate rsqrtps/vrsqrt14ps, which differ between CPU vendors.
template <class F> TW_HD TW_INLINE if_lane_t<F> rsqrt(F x) { return F(1.0f) / sqrt(x); }

// ============================================================ rounding (exact)

template <class F> TW_HD TW_INLINE if_lane_t<F> trunc(F x) {
    TW_NO_CONTRACT
    const lane_mask_t<F> small = cmp_lt(abs(x), F(k::kTwoPow23));  // false for NaN, inf, |x| >= 2^23
    const F t = cvt_f32(cvtt_i32(select(small, x, F(0.0f))));
    return select(small, copysign(t, x), x);
}
template <class F> TW_HD TW_INLINE if_lane_t<F> floor(F x) {
    TW_NO_CONTRACT
    const F t = trunc(x);
    return select(cmp_gt(t, x), t - F(1.0f), t);
}
template <class F> TW_HD TW_INLINE if_lane_t<F> ceil(F x) {
    TW_NO_CONTRACT
    const F t = trunc(x);
    return select(cmp_lt(t, x), t + F(1.0f), t);
}
// round to nearest integer, ties to even (C rint/nearbyint under RN)
template <class F> TW_HD TW_INLINE if_lane_t<F> rint(F x) {
    TW_NO_CONTRACT
    const F ax = abs(x);
    const F r = (ax + F(k::kTwoPow23)) - F(k::kTwoPow23);
    return select(cmp_lt(ax, F(k::kTwoPow23)), copysign(r, x), x);
}
// C fmod(x, m) for m a positive power of two (2^-126 <= m <= 2^127); exact.
// The result has the sign of x and |result| < m; NaN for NaN or infinite x.
template <class F> TW_HD TW_INLINE if_lane_t<F> fmod_pow2(F x, F m) {
    TW_NO_CONTRACT
    const F q = trunc(x / m);  // x/m exact (power-of-two scale) unless it underflows below 1
    const F r = copysign(x - q * m, x);
    return select(isnan(r), qnan<F>(), r);
}

// ============================================================ conversions

// float -> int32, truncating toward zero; NaN -> 0; saturates to
// [-2^31, 2^31 - 128] (the largest float below 2^31). Same result on every ISA.
template <class F> TW_HD TW_INLINE if_lane_t<F, lane_int_t<F>> to_i32_trunc_sat(F x) {
    const F c = clamp(select(isnan(x), F(0.0f), x), F(-0x1p31f), F(0x1.fffffep30f));
    return cvtt_i32(c);
}
// float -> uint32 with ForeverValidator's Binary32::TruncateToUint32Modulo
// semantics: trunc(x) modulo 2^32 (negative: 0 - (|trunc x| mod 2^32)),
// 0 for NaN, inf and |x| >= 2^64. Returned as the int32 bit pattern.
template <class F> TW_HD TW_INLINE if_lane_t<F, lane_int_t<F>> to_u32_trunc_wrap(F x) {
    TW_NO_CONTRACT
    using I = lane_int_t<F>;
    const F ax = abs(x);
    const lane_mask_t<F> ok = cmp_lt(ax, F(0x1p64f));
    const F axs = select(ok, ax, F(0.0f));
    const F hi = trunc(axs * F(0x1p-32f));       // exact
    const F r = axs - hi * F(0x1p32f);           // exact, 0 <= r < 2^32
    const lane_mask_t<F> top = cmp_ge(r, F(0x1p31f));
    const I u = cvtt_i32(select(top, r - F(0x1p31f), r)) + select(top, I(k::kSignBit), I(0));
    return select(signbit(x), I(0) - u, u);
}
// uint32 (passed as int32 bits) -> float, correctly rounded (round to
// nearest even), identical to static_cast<float>(uint32_t).
template <class I> TW_HD TW_INLINE auto from_u32(I u) -> decltype(cvt_f32(u)) {
    TW_NO_CONTRACT
    using F = decltype(cvt_f32(u));
    const F hi = cvt_f32(shr_logical<16>(u)) * F(65536.0f);  // exact
    const F lo = cvt_f32(u & I(0xffff));                     // exact
    return hi + lo;                                          // one rounding
}

// ============================================================ sin / cos

template <class F> struct sincos_result {
    F sin;
    F cos;
};

// sin and cos of x together; sin(x) and cos(x) below return exactly these.
//
// Reduction: k = nearest integer to |x|*2/pi; r = |x| - k*pi/2 as an
// unevaluated sum rh + rl. For k <= 2^13 every product k*Pi (i <= 4) is
// exact, |x| - k*P1 - k*P2 is exact, the P3 and P4 steps are two-sums and
// k*P5 goes into the tail, so rh + rl carries r to ~2^-65 absolute even next
// to a multiple of pi/2. Beyond that the products stop being exact and the
// result degrades gradually (measured in math.md); for huge x the quotient
// is clamped and the result clamped to [-1, 1] (deterministic, not
// meaningful).
template <class F> TW_HD TW_INLINE if_lane_t<F, sincos_result<F>> sincos(F x) {
    TW_NO_CONTRACT
    using I = lane_int_t<F>;
    const F ax = abs(x);
    F kf = (ax * F(k::kTwoOverPi) + F(k::kMagicRound)) - F(k::kMagicRound);
    kf = min(kf, F(0x1p30f));  // inf/NaN/huge -> 2^30: keeps cvtt_i32 defined
    const F d2 = (ax - kf * F(k::kP1)) - kf * F(k::kP2);  // exact
    const F t3 = kf * F(k::kP3);                           // exact
    const F t4 = kf * F(k::kP4);                           // exact
    const F d3 = d2 - t3;                                  // two-sum: d3 + e3 = d2 - t3
    const F b3 = d3 - d2;
    const F e3 = (d2 - (d3 - b3)) - (t3 + b3);
    const F rh = d3 - t4;                                  // two-sum: rh + e4 = d3 - t4
    const F b4 = rh - d3;
    const F e4 = (d3 - (rh - b4)) - (t4 + b4);
    const F rl = (e3 + e4) - kf * F(k::kP5);
    const F z = rh * rh;
    const F hz = z * F(0.5f);
    const F w = F(1.0f) - hz;
    const F sp = (F(k::kS3) * z + F(k::kS2)) * z + F(k::kS1);
    const F cp = (F(k::kC3) * z + F(k::kC2)) * z + F(k::kC1);
    // sin(rh + rl) ~ rh + (rl*cos(rh) + rh^3 S(z)),  cos(rh + rl) ~ cos(rh) - rl*rh
    F s = rh + (rl * w + (rh * z) * sp);
    F c = w + (((F(1.0f) - w) - hz) + ((z * z) * cp - rh * rl));
    s = clamp(s, F(-1.0f), F(1.0f));  // no-op on the accurate domain (tested)
    c = clamp(c, F(-1.0f), F(1.0f));
    const I q = cvtt_i32(kf);
    const lane_mask_t<F> swap = cmp_eq(q & I(1), I(1));
    const I sflip = shl<30>(q & I(2)) ^ (as_int(x) & I(k::kSignBit));  // quadrant sign, sin is odd
    const I cflip = shl<30>((q + I(1)) & I(2));
    const F sv = as_float(as_int(select(swap, c, s)) ^ sflip);
    const F cv = as_float(as_int(select(swap, s, c)) ^ cflip);
    const lane_mask_t<F> fin = cmp_le(ax, F(FLT_MAX));
    return {select(fin, sv, qnan<F>()), select(fin, cv, qnan<F>())};
}
template <class F> TW_HD TW_INLINE if_lane_t<F> sin(F x) { return sincos(x).sin; }
template <class F> TW_HD TW_INLINE if_lane_t<F> cos(F x) { return sincos(x).cos; }

// Reduction-free sincos for the latency-critical steering chain: accurate for
// |x| <= pi/4 (StadiumCar's visual steer angle stays within pi/6). Outside
// that it is deterministic but not accurate. Same polynomials as sincos, so
// on |x| <= pi/4 - 2^-12 it returns exactly sincos(x) (tested).
template <class F> TW_HD TW_INLINE if_lane_t<F, sincos_result<F>> sincos_small(F x) {
    TW_NO_CONTRACT
    const F a = abs(x);  // odd part on |x| then the sign: keeps sin(-0) = -0
    const F z = a * a;
    const F hz = z * F(0.5f);
    const F w = F(1.0f) - hz;
    const F sp = (F(k::kS3) * z + F(k::kS2)) * z + F(k::kS1);
    const F cp = (F(k::kC3) * z + F(k::kC2)) * z + F(k::kC1);
    const F s = mulsign(a + (a * z) * sp, x);
    const F c = w + (((F(1.0f) - w) - hz) + (z * z) * cp);
    const lane_mask_t<F> nan = isnan(x);
    return {select(nan, qnan<F>(), s), select(nan, qnan<F>(), c)};
}

// ============================================================ asin / acos

namespace detail {
template <class F> TW_HD TW_INLINE F asin_poly(F z) {
    TW_NO_CONTRACT
    return (((F(k::kA4) * z + F(k::kA3)) * z + F(k::kA2)) * z + F(k::kA1)) * z + F(k::kA0);
}
// sqrt(zb) = sh + c with sh = sqrt(zb) truncated to 12 bits (sh*sh exact)
template <class F> TW_HD TW_INLINE void split_sqrt(F zb, F &s, F &sh, F &c) {
    TW_NO_CONTRACT
    using I = lane_int_t<F>;
    s = sqrt(zb);
    sh = as_float(as_int(s) & I(-4096));  // 0xfffff000
    // + 2^-126 keeps 0/0 out at zb = 0 and changes nothing when s + sh >= 2^-13
    c = (zb - sh * sh) / ((s + sh) + F(0x1p-126f));
}
}  // namespace detail

// asin for |x| <= 1/2 only (no sqrt, no division: about a third of asin's
// latency). StadiumCar's steering asin argument 1/(|v.z| 0.85 + 2) is in
// (0, 1/2]. Deterministic but inaccurate outside; equals asin(x) bitwise on
// [-1/2, 1/2] (tested).
template <class F> TW_HD TW_INLINE if_lane_t<F> asin_small(F x) {
    TW_NO_CONTRACT
    const F z = x * x;
    const F r = x + x * (z * detail::asin_poly(z));
    return select(isnan(x), qnan<F>(), r);
}

// asin on [-1, 1]: |x| <= 1/2 uses x + x z P(z); |x| > 1/2 uses
// pi/2 - 2 asin(sqrt((1-|x|)/2)) with the sqrt split into sh + c (fdlibm).
template <class F> TW_HD TW_INLINE if_lane_t<F> asin(F x) {
    TW_NO_CONTRACT
    const F a = abs(x);
    const lane_mask_t<F> big = cmp_gt(a, F(0.5f));
    const F zb = max((F(1.0f) - a) * F(0.5f), F(0.0f));  // exact for 1/2 <= a <= 1
    const F z = select(big, zb, a * a);
    const F zP = z * detail::asin_poly(z);
    const F rs = a + a * zP;
    F s, sh, c;
    detail::split_sqrt(zb, s, sh, c);
    const F rb = F(k::kPio4Hi) -
                 (((F(2.0f) * s) * zP - (F(k::kPio2Lo) - F(2.0f) * c)) - (F(k::kPio4Hi) - F(2.0f) * sh));
    const F r = mulsign(select(big, rb, rs), x);
    return select(cmp_le(a, F(1.0f)), r, qnan<F>());
}

// acos on [-1, 1] (fdlibm structure, three branches evaluated as selects)
template <class F> TW_HD TW_INLINE if_lane_t<F> acos(F x) {
    TW_NO_CONTRACT
    const F a = abs(x);
    const lane_mask_t<F> big = cmp_gt(a, F(0.5f));
    const F zb = max((F(1.0f) - a) * F(0.5f), F(0.0f));
    const F z = select(big, zb, x * x);
    const F zP = z * detail::asin_poly(z);
    // |x| <= 1/2: pi/2 - (x + x zP)
    const F r_mid = F(k::kPio2Hi) - (x - (F(k::kPio2Lo) - x * zP));
    F s, sh, c;
    detail::split_sqrt(zb, s, sh, c);
    // x > 1/2: 2 asin(s);  x < -1/2: pi - 2 asin(s)
    const F r_pos = F(2.0f) * (sh + (s * zP + c));
    const F r_neg = F(k::kPiHi) - F(2.0f) * (s + (s * zP - F(k::kPio2Lo)));
    const F r = select(big, select(signbit(x), r_neg, r_pos), r_mid);
    return select(cmp_le(a, F(1.0f)), r, qnan<F>());
}

// ============================================================ atan / atan2

// atan2(y, x) in [-pi, pi], with the IEEE special cases for signed zeros
// and infinities. t = |y|/|x| is reduced as in fdlibm atanf, without forming
// t: [0, 7/16): t; [7/16, 11/16): atan(1/2) + atan((2t-1)/(2+t));
// [11/16, 19/16): pi/4 + atan((t-1)/(t+1)); [19/16, 39/16): atan(3/2) +
// atan((t-3/2)/(1+3t/2)); above: pi/2 + atan(-1/t). For x < 0 the offset
// is pi - y0 (hi + lo) and u is negated, so no rounded angle is reused.
// The division rounds once (|u| has 0.5 ulp error), which bounds atan2 at
// about 1.5 ulp; atan(x) has exact u and stays below 0.9 ulp. Inputs above 2^125 or
// below 2^-100 are first scaled by a power of two (exact), so the whole range
// keeps full accuracy.
template <class F> TW_HD TW_INLINE if_lane_t<F> atan2(F y, F x) {
    TW_NO_CONTRACT
    using M = lane_mask_t<F>;
    F ax = abs(x);
    F ay = abs(y);
    const F m = max(ax, ay);
    const F sc = select(cmp_gt(m, F(0x1p125f)), F(0x1p-8f), select(cmp_lt(m, F(0x1p-100f)), F(0x1p50f), F(1.0f)));
    ax = ax * sc;
    ay = ay * sc;
    const F inf = infinity<F>();
    const M both_inf = cmp_eq(ay, inf) & cmp_eq(ax, inf);
    const M i1 = cmp_gt(ay, ax * F(0.4375f));
    const M i2 = cmp_gt(ay, ax * F(0.6875f)) | both_inf;
    const M i3 = cmp_gt(ay, ax * F(1.1875f));
    const M i4 = cmp_gt(ay, ax * F(2.4375f));
    F num = select(i1, F(2.0f) * ay - ax, ay);
    F den = select(i1, F(2.0f) * ax + ay, ax);
    num = select(i2, ay - ax, num);
    den = select(i2, ay + ax, den);
    num = select(i3, ay - F(1.5f) * ax, num);
    den = select(i3, ax + F(1.5f) * ay, den);
    num = select(i4, -ax, num);
    den = select(i4, ay, den);
    num = select(both_inf, F(0.0f), num);
    den = select(both_inf | cmp_eq(ay, F(0.0f)), F(1.0f), den);
    // result = y0 + atan(u) for x >= 0 and (pi - y0) + atan(-u) for x < 0
    const M xneg = signbit(x);
    num = select(xneg, -num, num);
    F y0h = select(xneg, F(k::kPiHi), F(0.0f));
    F y0l = select(xneg, F(k::kPiLo), F(0.0f));
    y0h = select(i1, select(xneg, F(k::kPiMinusAtanHalfHi), F(k::kAtanHalfHi)), y0h);
    y0l = select(i1, select(xneg, F(k::kPiMinusAtanHalfLo), F(k::kAtanHalfLo)), y0l);
    y0h = select(i2, select(xneg, F(k::kThreePio4Hi), F(k::kPio4Hi)), y0h);
    y0l = select(i2, select(xneg, F(k::kThreePio4Lo), F(k::kPio4Lo)), y0l);
    y0h = select(i3, select(xneg, F(k::kPiMinusAtanThreeHalfHi), F(k::kAtanThreeHalfHi)), y0h);
    y0l = select(i3, select(xneg, F(k::kPiMinusAtanThreeHalfLo), F(k::kAtanThreeHalfLo)), y0l);
    y0h = select(i4, F(k::kPio2Hi), y0h);  // pi - pi/2 = pi/2
    y0l = select(i4, F(k::kPio2Lo), y0l);
    const F u = num / den;
    const F z = u * u;
    const F w = z * z;
    const F s1 = z * (F(k::kT0) + w * (F(k::kT2) + w * F(k::kT4)));
    const F s2 = w * (F(k::kT1) + w * F(k::kT3));
    F r = y0h - ((u * (s1 + s2) - y0l) - u);
    r = copysign(r, y);
    return select(cmp_unord(x, y), qnan<F>(), r);
}
template <class F> TW_HD TW_INLINE if_lane_t<F> atan(F x) { return atan2(x, F(1.0f)); }

// ============================================================ exp

template <class F> TW_HD TW_INLINE if_lane_t<F> exp(F x) {
    TW_NO_CONTRACT
    using I = lane_int_t<F>;
    F xc = select(cmp_lt(x, F(k::kExpMin)), F(k::kExpMin), x);
    xc = select(cmp_gt(xc, F(k::kExpMax)), F(k::kExpMax), xc);  // NaN passes through
    F kf = (xc * F(k::kLog2e) + F(k::kMagicRound)) - F(k::kMagicRound);  // nearest integer, -150..128
    kf = select(isnan(kf), F(0.0f), kf);
    const F r = (xc - kf * F(k::kLn2Hi)) - kf * F(k::kLn2Lo);
    const F z = r * r;
    const F p = ((((F(k::kE5) * r + F(k::kE4)) * r + F(k::kE3)) * r + F(k::kE2)) * r + F(k::kE1)) * r +
                F(k::kE0);
    const F e = (p * z + r) + F(1.0f);
    // 2^k in two normal factors so results down to the denormal range are
    // rounded once, from e: k1 = floor(k/2), k2 = k - k1, both in [-75, 64]
    const I ki = cvtt_i32(kf);
    const I k1 = shr_arith<1>(ki);
    const I k2 = ki - k1;
    const F s1 = as_float(shl<23>(k1 + I(127)));
    const F s2 = as_float(shl<23>(k2 + I(127)));
    const F res = (e * s1) * s2;
    return select(isnan(x), qnan<F>(), res);
}

// ============================================================ canary

// True when this translation unit evaluates a*b + c with two roundings (no
// contraction). Deliberately NOT protected by TW_NO_CONTRACT: it reports the
// includer's flags. (1+2^-12)^2 = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11
// (tie to even), so the unfused result is exactly 0 and the fused one 2^-24.
inline bool contraction_canary() {
    volatile float va = 1.0f + 0x1p-12f;
    volatile float vc = -(1.0f + 0x1p-11f);
    const float a = va;
    const float c = vc;
    const float raw = a * a + c;
    const f32x1 lane = f32x1(a) * f32x1(a) + f32x1(c);
    return raw == 0.0f && lane.v == 0.0f;
}

// ============================================================ scalar API

TW_HD TW_INLINE float abs(float x) { return abs(f32x1(x)).v; }
TW_HD TW_INLINE float copysign(float m, float s) { return copysign(f32x1(m), f32x1(s)).v; }
TW_HD TW_INLINE float min(float a, float b) { return min(f32x1(a), f32x1(b)).v; }
TW_HD TW_INLINE float max(float a, float b) { return max(f32x1(a), f32x1(b)).v; }
TW_HD TW_INLINE float clamp(float x, float lo, float hi) { return clamp(f32x1(x), f32x1(lo), f32x1(hi)).v; }
TW_HD TW_INLINE float sqrt(float x) { return sqrt(f32x1(x)).v; }
TW_HD TW_INLINE float rsqrt(float x) { return rsqrt(f32x1(x)).v; }
TW_HD TW_INLINE float recip(float x) { return recip(f32x1(x)).v; }
TW_HD TW_INLINE float trunc(float x) { return trunc(f32x1(x)).v; }
TW_HD TW_INLINE float floor(float x) { return floor(f32x1(x)).v; }
TW_HD TW_INLINE float ceil(float x) { return ceil(f32x1(x)).v; }
TW_HD TW_INLINE float rint(float x) { return rint(f32x1(x)).v; }
TW_HD TW_INLINE float fmod_pow2(float x, float m) { return fmod_pow2(f32x1(x), f32x1(m)).v; }
TW_HD TW_INLINE std::int32_t to_i32_trunc_sat(float x) { return to_i32_trunc_sat(f32x1(x)).v; }
TW_HD TW_INLINE std::uint32_t to_u32_trunc_wrap(float x) {
    return static_cast<std::uint32_t>(to_u32_trunc_wrap(f32x1(x)).v);
}
TW_HD TW_INLINE float from_u32(std::uint32_t u) { return from_u32(i32x1(wrap_i32(u))).v; }
TW_HD TW_INLINE float from_i32(std::int32_t i) { return cvt_f32(i32x1(i)).v; }
TW_HD TW_INLINE sincos_result<float> sincos(float x) {
    const sincos_result<f32x1> r = sincos(f32x1(x));
    return {r.sin.v, r.cos.v};
}
TW_HD TW_INLINE sincos_result<float> sincos_small(float x) {
    const sincos_result<f32x1> r = sincos_small(f32x1(x));
    return {r.sin.v, r.cos.v};
}
TW_HD TW_INLINE float asin_small(float x) { return asin_small(f32x1(x)).v; }
TW_HD TW_INLINE float sin(float x) { return sin(f32x1(x)).v; }
TW_HD TW_INLINE float cos(float x) { return cos(f32x1(x)).v; }
TW_HD TW_INLINE float asin(float x) { return asin(f32x1(x)).v; }
TW_HD TW_INLINE float acos(float x) { return acos(f32x1(x)).v; }
TW_HD TW_INLINE float atan(float x) { return atan(f32x1(x)).v; }
TW_HD TW_INLINE float atan2(float y, float x) { return atan2(f32x1(y), f32x1(x)).v; }
TW_HD TW_INLINE float exp(float x) { return exp(f32x1(x)).v; }

}  // namespace math
}  // namespace tw

#endif  // TW_MATH_H
