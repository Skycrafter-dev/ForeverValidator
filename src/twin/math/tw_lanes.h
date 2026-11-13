// Twin deterministic math: lane types.
//
// One interface, four native widths:
//   f32x1  scalar (plain C++; the reference and the future GPU thread form)
//   f32x4  SSE2   (SSE4.1 blendv when the TU has it; same bits)
//   f32x8  AVX2   (no FMA is ever emitted by the wrappers)
//   f32x16 AVX-512F (only F instructions: no DQ/VL needed)
// each with an int32 lane type (i32xN) and a mask type (m32xN), plus
// fpair<F> (two halves) and vf<W>, which picks a native or paired type so
// that every logical width W in {1, 4, 8, 16} exists on every ISA.
//
// Every operation is lane-pure: lane i of the result depends only on lane i
// of the inputs, and each wrapper is one IEEE-754 binary32 operation that is
// correctly rounded (+ - * / sqrt, int->float), exact (compare, select, bit
// ops, float->int truncation in range) or integer. So any width computes the
// same bits as the scalar form. There is deliberately no rcp/rsqrt, no FMA,
// no horizontal operation and no rounding-mode-dependent conversion except
// int->float (round to nearest under the canonical MXCSR, see tw_fpenv.h).
//
// Preconditions shared by all widths:
//   cvtt_i32(x): x must be in (-2^31, 2^31) and not NaN (x86 returns
//                0x80000000 out of range, ARM saturates, C++ calls it UB);
//                the math functions always clamp before converting.
//   shift counts are compile-time constants in [0, 31].
#ifndef TW_LANES_H
#define TW_LANES_H

#include "tw_math_config.h"

#if TW_HAVE_SSE2
#include <emmintrin.h>
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#endif
#if TW_HAVE_AVX2 || TW_HAVE_AVX512F
#include <immintrin.h>
#endif

namespace tw {
namespace math {

// ===================================================================== scalar

struct m32x1 {
    bool v;
    TW_HD TW_INLINE m32x1() : v(false) {}
    TW_HD TW_INLINE explicit m32x1(bool b) : v(b) {}
};

struct i32x1 {
    std::int32_t v;
    static constexpr int width = 1;
    TW_HD TW_INLINE i32x1() : v(0) {}
    TW_HD TW_INLINE i32x1(std::int32_t s) : v(s) {}
    TW_HD static TW_INLINE i32x1 load(const std::int32_t *p) { return i32x1(p[0]); }
    TW_HD TW_INLINE void store(std::int32_t *p) const { p[0] = v; }
};

struct f32x1 {
    float v;
    using int_type = i32x1;
    using mask_type = m32x1;
    static constexpr int width = 1;
    TW_HD TW_INLINE f32x1() : v(0.0f) {}
    TW_HD TW_INLINE f32x1(float s) : v(s) {}
    TW_HD static TW_INLINE f32x1 load(const float *p) { return f32x1(p[0]); }
    TW_HD TW_INLINE void store(float *p) const { p[0] = v; }
};

// NVIDIA device form of sqrt.rn.f32 with the same bits for every input (f32x1 sqrt on the device). The library's
// sqrtf range-checks its operand and calls a slow path out of [2^-101, max]: +-0 (common in the forces, whose arms run
// for every lane of a warp that has one lane in them) took the call, and the branch around it splits the code into
// blocks. SqrtAll has no branch: ptxas's own fast-path instructions for sqrt.rn (MUFU.RSQ, two FTZ products, two FFMA,
// as tw_gpu_collide.cuh SqrtF), exact on [2^-101, max]; an input in (0, 2^-101) (denormals too) is scaled by 2^64 first
// and its root by 2^-32 after (both exact, so the rounding is the scaled root's); +-0 and +inf are returned as they are;
// a negative input or a NaN gives the fast path's NaN. Checked against __fsqrt_rn under the twin's GPU flags on all
// 2^32 inputs (tools/twin/search/gpu-zeroguard-check.cu). (The same treatment of division's zero operands, DivZ in that
// tool: the guard's instructions on every division cost more than the calls it saved; gpu-forces-exp.)
#if defined(__CUDACC__) && !defined(__HIP__)
namespace dev {
__device__ __forceinline__ float SqrtAll(float x) {
    const unsigned b = __float_as_uint(x);
    const bool tiny = b - 1u < 0x0cffffffu;   // 0 < x < 2^-101
    const float xs = tiny ? x * 0x1p64f : x;
    float r, q, h;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(xs));
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(q) : "f"(xs), "f"(r));
    asm("mul.ftz.f32 %0, %1, 0f3F000000;" : "=f"(h) : "f"(r));
    const float e = __fmaf_rn(-q, q, xs);
    const float y = __fmaf_rn(e, h, q);
    const float ys = tiny ? y * 0x1p-32f : y;
    return ((b << 1) == 0u || b == 0x7f800000u) ? x : ys;
}
// 1 / sqrt(x) as its two correctly rounded operations (1.0f / sqrtf), no branch: SqrtAll, then ptxas's fast-path
// instructions for the reciprocal (MUFU.RCP, FFMA, an FTZ negation, FFMA: tw_gpu_collide.cuh RcpF), exact for |s| normal
// below 2^126, which holds every root but +-0 and +inf: 1 / +-0 = +-inf and 1 / +inf = +0 by select. Checked against
// 1.0f / __fsqrt_rn(x) on all 2^32 inputs (tools/twin/search/gpu-zeroguard-check.cu).
__device__ __forceinline__ float RsqrtAll(float x) {
    const float s = SqrtAll(x);
    const unsigned b = __float_as_uint(s);
    float r, t2;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
    const float t = __fmaf_rn(r, s, -1.0f);
    asm("add.ftz.f32 %0, %1, 0f80000000;" : "=f"(t2) : "f"(-t));
    const float y = __fmaf_rn(r, t2, r);
    return (b << 1) == 0u ? __uint_as_float((b & 0x80000000u) | 0x7f800000u) : (b == 0x7f800000u ? 0.0f : y);
}
}  // namespace dev
#endif

TW_HD TW_INLINE f32x1 operator+(f32x1 a, f32x1 b) { return f32x1(a.v + b.v); }
TW_HD TW_INLINE f32x1 operator-(f32x1 a, f32x1 b) { return f32x1(a.v - b.v); }
TW_HD TW_INLINE f32x1 operator*(f32x1 a, f32x1 b) { return f32x1(a.v * b.v); }
TW_HD TW_INLINE f32x1 operator/(f32x1 a, f32x1 b) { return f32x1(a.v / b.v); }
TW_HD TW_INLINE f32x1 operator-(f32x1 a) { return f32x1(-a.v); }  // sign flip (xorps)

TW_HD TW_INLINE f32x1 sqrt(f32x1 a) {
#if defined(__CUDA_ARCH__) && !defined(TW_LANES_LIBRARY_SQRT)
    return f32x1(dev::SqrtAll(a.v));  // __fsqrt_rn's bits without its slow-path call (above)
#elif defined(__CUDA_ARCH__)
    return f32x1(__fsqrt_rn(a.v));  // correctly rounded on NVIDIA (not on HIP: see math.md)
#elif TW_HAVE_SSE2
    return f32x1(_mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(a.v))));
#else
    return f32x1(__builtin_sqrtf(a.v));
#endif
}

TW_HD TW_INLINE m32x1 cmp_lt(f32x1 a, f32x1 b) { return m32x1(a.v < b.v); }
TW_HD TW_INLINE m32x1 cmp_le(f32x1 a, f32x1 b) { return m32x1(a.v <= b.v); }
TW_HD TW_INLINE m32x1 cmp_gt(f32x1 a, f32x1 b) { return m32x1(a.v > b.v); }
TW_HD TW_INLINE m32x1 cmp_ge(f32x1 a, f32x1 b) { return m32x1(a.v >= b.v); }
TW_HD TW_INLINE m32x1 cmp_eq(f32x1 a, f32x1 b) { return m32x1(a.v == b.v); }
TW_HD TW_INLINE m32x1 cmp_neq(f32x1 a, f32x1 b) { return m32x1(!(a.v == b.v)); }  // true if unordered
TW_HD TW_INLINE m32x1 cmp_unord(f32x1 a, f32x1 b) { return m32x1(a.v != a.v || b.v != b.v); }

TW_HD TW_INLINE m32x1 operator&(m32x1 a, m32x1 b) { return m32x1(a.v && b.v); }
TW_HD TW_INLINE m32x1 operator|(m32x1 a, m32x1 b) { return m32x1(a.v || b.v); }
TW_HD TW_INLINE m32x1 operator^(m32x1 a, m32x1 b) { return m32x1(a.v != b.v); }
TW_HD TW_INLINE m32x1 operator~(m32x1 a) { return m32x1(!a.v); }
TW_HD TW_INLINE bool any(m32x1 m) { return m.v; }
TW_HD TW_INLINE bool all(m32x1 m) { return m.v; }

TW_HD TW_INLINE f32x1 select(m32x1 m, f32x1 a, f32x1 b) { return m.v ? a : b; }
TW_HD TW_INLINE i32x1 select(m32x1 m, i32x1 a, i32x1 b) { return m.v ? a : b; }

TW_HD TW_INLINE i32x1 as_int(f32x1 a) { return i32x1(static_cast<std::int32_t>(bits_of(a.v))); }
TW_HD TW_INLINE f32x1 as_float(i32x1 a) { return f32x1(float_of(static_cast<std::uint32_t>(a.v))); }
TW_HD TW_INLINE i32x1 cvtt_i32(f32x1 a) { return i32x1(static_cast<std::int32_t>(a.v)); }
TW_HD TW_INLINE f32x1 cvt_f32(i32x1 a) { return f32x1(static_cast<float>(a.v)); }

TW_HD TW_INLINE std::int32_t wrap_i32(std::uint32_t u) {
    std::int32_t s;
    std::memcpy(&s, &u, sizeof s);
    return s;
}
TW_HD TW_INLINE i32x1 operator+(i32x1 a, i32x1 b) {
    return i32x1(wrap_i32(static_cast<std::uint32_t>(a.v) + static_cast<std::uint32_t>(b.v)));
}
TW_HD TW_INLINE i32x1 operator-(i32x1 a, i32x1 b) {
    return i32x1(wrap_i32(static_cast<std::uint32_t>(a.v) - static_cast<std::uint32_t>(b.v)));
}
TW_HD TW_INLINE i32x1 operator&(i32x1 a, i32x1 b) { return i32x1(a.v & b.v); }
TW_HD TW_INLINE i32x1 operator|(i32x1 a, i32x1 b) { return i32x1(a.v | b.v); }
TW_HD TW_INLINE i32x1 operator^(i32x1 a, i32x1 b) { return i32x1(a.v ^ b.v); }
template <int N> TW_HD TW_INLINE i32x1 shl(i32x1 a) {
    return i32x1(wrap_i32(static_cast<std::uint32_t>(a.v) << N));
}
template <int N> TW_HD TW_INLINE i32x1 shr_logical(i32x1 a) {
    return i32x1(wrap_i32(static_cast<std::uint32_t>(a.v) >> N));
}
template <int N> TW_HD TW_INLINE i32x1 shr_arith(i32x1 a) {
    // arithmetic shift written without implementation-defined behaviour
    const std::uint32_t u = static_cast<std::uint32_t>(a.v);
    const std::uint32_t sign = 0u - (u >> 31);
    return i32x1(wrap_i32((u >> N) | (N == 0 ? 0u : (sign << (31 - N) << 1))));
}
TW_HD TW_INLINE m32x1 cmp_eq(i32x1 a, i32x1 b) { return m32x1(a.v == b.v); }
TW_HD TW_INLINE m32x1 cmp_gt(i32x1 a, i32x1 b) { return m32x1(a.v > b.v); }

// ======================================================================= SSE2
#if TW_HAVE_SSE2

struct m32x4 {
    __m128 v;
    TW_INLINE m32x4() : v(_mm_setzero_ps()) {}
    TW_INLINE explicit m32x4(__m128 x) : v(x) {}
};

struct i32x4 {
    __m128i v;
    static constexpr int width = 4;
    TW_INLINE i32x4() : v(_mm_setzero_si128()) {}
    TW_INLINE i32x4(std::int32_t s) : v(_mm_set1_epi32(s)) {}
    TW_INLINE explicit i32x4(__m128i x) : v(x) {}
    static TW_INLINE i32x4 load(const std::int32_t *p) {
        return i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
    }
    TW_INLINE void store(std::int32_t *p) const { _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v); }
};

struct f32x4 {
    __m128 v;
    using int_type = i32x4;
    using mask_type = m32x4;
    static constexpr int width = 4;
    TW_INLINE f32x4() : v(_mm_setzero_ps()) {}
    TW_INLINE f32x4(float s) : v(_mm_set1_ps(s)) {}
    TW_INLINE explicit f32x4(__m128 x) : v(x) {}
    static TW_INLINE f32x4 load(const float *p) { return f32x4(_mm_loadu_ps(p)); }
    TW_INLINE void store(float *p) const { _mm_storeu_ps(p, v); }
};

TW_INLINE f32x4 operator+(f32x4 a, f32x4 b) { return f32x4(_mm_add_ps(a.v, b.v)); }
TW_INLINE f32x4 operator-(f32x4 a, f32x4 b) { return f32x4(_mm_sub_ps(a.v, b.v)); }
TW_INLINE f32x4 operator*(f32x4 a, f32x4 b) { return f32x4(_mm_mul_ps(a.v, b.v)); }
TW_INLINE f32x4 operator/(f32x4 a, f32x4 b) { return f32x4(_mm_div_ps(a.v, b.v)); }
TW_INLINE f32x4 operator-(f32x4 a) { return f32x4(_mm_xor_ps(a.v, _mm_set1_ps(-0.0f))); }
TW_INLINE f32x4 sqrt(f32x4 a) { return f32x4(_mm_sqrt_ps(a.v)); }

TW_INLINE m32x4 cmp_lt(f32x4 a, f32x4 b) { return m32x4(_mm_cmplt_ps(a.v, b.v)); }
TW_INLINE m32x4 cmp_le(f32x4 a, f32x4 b) { return m32x4(_mm_cmple_ps(a.v, b.v)); }
TW_INLINE m32x4 cmp_gt(f32x4 a, f32x4 b) { return m32x4(_mm_cmplt_ps(b.v, a.v)); }
TW_INLINE m32x4 cmp_ge(f32x4 a, f32x4 b) { return m32x4(_mm_cmple_ps(b.v, a.v)); }
TW_INLINE m32x4 cmp_eq(f32x4 a, f32x4 b) { return m32x4(_mm_cmpeq_ps(a.v, b.v)); }
TW_INLINE m32x4 cmp_neq(f32x4 a, f32x4 b) { return m32x4(_mm_cmpneq_ps(a.v, b.v)); }
TW_INLINE m32x4 cmp_unord(f32x4 a, f32x4 b) { return m32x4(_mm_cmpunord_ps(a.v, b.v)); }

TW_INLINE m32x4 operator&(m32x4 a, m32x4 b) { return m32x4(_mm_and_ps(a.v, b.v)); }
TW_INLINE m32x4 operator|(m32x4 a, m32x4 b) { return m32x4(_mm_or_ps(a.v, b.v)); }
TW_INLINE m32x4 operator^(m32x4 a, m32x4 b) { return m32x4(_mm_xor_ps(a.v, b.v)); }
TW_INLINE m32x4 operator~(m32x4 a) {
    return m32x4(_mm_xor_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(-1))));
}
TW_INLINE bool any(m32x4 m) { return _mm_movemask_ps(m.v) != 0; }
TW_INLINE bool all(m32x4 m) { return _mm_movemask_ps(m.v) == 0xf; }

TW_INLINE f32x4 select(m32x4 m, f32x4 a, f32x4 b) {
#if defined(__SSE4_1__)
    return f32x4(_mm_blendv_ps(b.v, a.v, m.v));
#else
    return f32x4(_mm_or_ps(_mm_and_ps(m.v, a.v), _mm_andnot_ps(m.v, b.v)));
#endif
}
TW_INLINE i32x4 select(m32x4 m, i32x4 a, i32x4 b) {
    const __m128i mi = _mm_castps_si128(m.v);
    return i32x4(_mm_or_si128(_mm_and_si128(mi, a.v), _mm_andnot_si128(mi, b.v)));
}

TW_INLINE i32x4 as_int(f32x4 a) { return i32x4(_mm_castps_si128(a.v)); }
TW_INLINE f32x4 as_float(i32x4 a) { return f32x4(_mm_castsi128_ps(a.v)); }
TW_INLINE i32x4 cvtt_i32(f32x4 a) { return i32x4(_mm_cvttps_epi32(a.v)); }
TW_INLINE f32x4 cvt_f32(i32x4 a) { return f32x4(_mm_cvtepi32_ps(a.v)); }

TW_INLINE i32x4 operator+(i32x4 a, i32x4 b) { return i32x4(_mm_add_epi32(a.v, b.v)); }
TW_INLINE i32x4 operator-(i32x4 a, i32x4 b) { return i32x4(_mm_sub_epi32(a.v, b.v)); }
TW_INLINE i32x4 operator&(i32x4 a, i32x4 b) { return i32x4(_mm_and_si128(a.v, b.v)); }
TW_INLINE i32x4 operator|(i32x4 a, i32x4 b) { return i32x4(_mm_or_si128(a.v, b.v)); }
TW_INLINE i32x4 operator^(i32x4 a, i32x4 b) { return i32x4(_mm_xor_si128(a.v, b.v)); }
template <int N> TW_INLINE i32x4 shl(i32x4 a) { return i32x4(_mm_slli_epi32(a.v, N)); }
template <int N> TW_INLINE i32x4 shr_logical(i32x4 a) { return i32x4(_mm_srli_epi32(a.v, N)); }
template <int N> TW_INLINE i32x4 shr_arith(i32x4 a) { return i32x4(_mm_srai_epi32(a.v, N)); }
TW_INLINE m32x4 cmp_eq(i32x4 a, i32x4 b) { return m32x4(_mm_castsi128_ps(_mm_cmpeq_epi32(a.v, b.v))); }
TW_INLINE m32x4 cmp_gt(i32x4 a, i32x4 b) { return m32x4(_mm_castsi128_ps(_mm_cmpgt_epi32(a.v, b.v))); }

#endif  // TW_HAVE_SSE2

// ======================================================================= AVX2
#if TW_HAVE_AVX2

struct m32x8 {
    __m256 v;
    TW_INLINE m32x8() : v(_mm256_setzero_ps()) {}
    TW_INLINE explicit m32x8(__m256 x) : v(x) {}
};

struct i32x8 {
    __m256i v;
    static constexpr int width = 8;
    TW_INLINE i32x8() : v(_mm256_setzero_si256()) {}
    TW_INLINE i32x8(std::int32_t s) : v(_mm256_set1_epi32(s)) {}
    TW_INLINE explicit i32x8(__m256i x) : v(x) {}
    static TW_INLINE i32x8 load(const std::int32_t *p) {
        return i32x8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
    }
    TW_INLINE void store(std::int32_t *p) const { _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v); }
};

struct f32x8 {
    __m256 v;
    using int_type = i32x8;
    using mask_type = m32x8;
    static constexpr int width = 8;
    TW_INLINE f32x8() : v(_mm256_setzero_ps()) {}
    TW_INLINE f32x8(float s) : v(_mm256_set1_ps(s)) {}
    TW_INLINE explicit f32x8(__m256 x) : v(x) {}
    static TW_INLINE f32x8 load(const float *p) { return f32x8(_mm256_loadu_ps(p)); }
    TW_INLINE void store(float *p) const { _mm256_storeu_ps(p, v); }
};

TW_INLINE f32x8 operator+(f32x8 a, f32x8 b) { return f32x8(_mm256_add_ps(a.v, b.v)); }
TW_INLINE f32x8 operator-(f32x8 a, f32x8 b) { return f32x8(_mm256_sub_ps(a.v, b.v)); }
TW_INLINE f32x8 operator*(f32x8 a, f32x8 b) { return f32x8(_mm256_mul_ps(a.v, b.v)); }
TW_INLINE f32x8 operator/(f32x8 a, f32x8 b) { return f32x8(_mm256_div_ps(a.v, b.v)); }
TW_INLINE f32x8 operator-(f32x8 a) { return f32x8(_mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f))); }
TW_INLINE f32x8 sqrt(f32x8 a) { return f32x8(_mm256_sqrt_ps(a.v)); }

TW_INLINE m32x8 cmp_lt(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)); }
TW_INLINE m32x8 cmp_le(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)); }
TW_INLINE m32x8 cmp_gt(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)); }
TW_INLINE m32x8 cmp_ge(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)); }
TW_INLINE m32x8 cmp_eq(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)); }
TW_INLINE m32x8 cmp_neq(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_UQ)); }
TW_INLINE m32x8 cmp_unord(f32x8 a, f32x8 b) { return m32x8(_mm256_cmp_ps(a.v, b.v, _CMP_UNORD_Q)); }

TW_INLINE m32x8 operator&(m32x8 a, m32x8 b) { return m32x8(_mm256_and_ps(a.v, b.v)); }
TW_INLINE m32x8 operator|(m32x8 a, m32x8 b) { return m32x8(_mm256_or_ps(a.v, b.v)); }
TW_INLINE m32x8 operator^(m32x8 a, m32x8 b) { return m32x8(_mm256_xor_ps(a.v, b.v)); }
TW_INLINE m32x8 operator~(m32x8 a) {
    return m32x8(_mm256_xor_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
}
TW_INLINE bool any(m32x8 m) { return _mm256_movemask_ps(m.v) != 0; }
TW_INLINE bool all(m32x8 m) { return _mm256_movemask_ps(m.v) == 0xff; }

TW_INLINE f32x8 select(m32x8 m, f32x8 a, f32x8 b) { return f32x8(_mm256_blendv_ps(b.v, a.v, m.v)); }
TW_INLINE i32x8 select(m32x8 m, i32x8 a, i32x8 b) {
    return i32x8(_mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v), m.v)));
}

TW_INLINE i32x8 as_int(f32x8 a) { return i32x8(_mm256_castps_si256(a.v)); }
TW_INLINE f32x8 as_float(i32x8 a) { return f32x8(_mm256_castsi256_ps(a.v)); }
TW_INLINE i32x8 cvtt_i32(f32x8 a) { return i32x8(_mm256_cvttps_epi32(a.v)); }
TW_INLINE f32x8 cvt_f32(i32x8 a) { return f32x8(_mm256_cvtepi32_ps(a.v)); }

TW_INLINE i32x8 operator+(i32x8 a, i32x8 b) { return i32x8(_mm256_add_epi32(a.v, b.v)); }
TW_INLINE i32x8 operator-(i32x8 a, i32x8 b) { return i32x8(_mm256_sub_epi32(a.v, b.v)); }
TW_INLINE i32x8 operator&(i32x8 a, i32x8 b) { return i32x8(_mm256_and_si256(a.v, b.v)); }
TW_INLINE i32x8 operator|(i32x8 a, i32x8 b) { return i32x8(_mm256_or_si256(a.v, b.v)); }
TW_INLINE i32x8 operator^(i32x8 a, i32x8 b) { return i32x8(_mm256_xor_si256(a.v, b.v)); }
template <int N> TW_INLINE i32x8 shl(i32x8 a) { return i32x8(_mm256_slli_epi32(a.v, N)); }
template <int N> TW_INLINE i32x8 shr_logical(i32x8 a) { return i32x8(_mm256_srli_epi32(a.v, N)); }
template <int N> TW_INLINE i32x8 shr_arith(i32x8 a) { return i32x8(_mm256_srai_epi32(a.v, N)); }
TW_INLINE m32x8 cmp_eq(i32x8 a, i32x8 b) { return m32x8(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a.v, b.v))); }
TW_INLINE m32x8 cmp_gt(i32x8 a, i32x8 b) { return m32x8(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a.v, b.v))); }

#endif  // TW_HAVE_AVX2

// ==================================================================== AVX-512F
#if TW_HAVE_AVX512F

struct m32x16 {
    __mmask16 v;
    TW_INLINE m32x16() : v(0) {}
    TW_INLINE explicit m32x16(__mmask16 x) : v(x) {}
};

struct i32x16 {
    __m512i v;
    static constexpr int width = 16;
    TW_INLINE i32x16() : v(_mm512_setzero_si512()) {}
    TW_INLINE i32x16(std::int32_t s) : v(_mm512_set1_epi32(s)) {}
    TW_INLINE explicit i32x16(__m512i x) : v(x) {}
    static TW_INLINE i32x16 load(const std::int32_t *p) { return i32x16(_mm512_loadu_si512(p)); }
    TW_INLINE void store(std::int32_t *p) const { _mm512_storeu_si512(p, v); }
};

struct f32x16 {
    __m512 v;
    using int_type = i32x16;
    using mask_type = m32x16;
    static constexpr int width = 16;
    TW_INLINE f32x16() : v(_mm512_setzero_ps()) {}
    TW_INLINE f32x16(float s) : v(_mm512_set1_ps(s)) {}
    TW_INLINE explicit f32x16(__m512 x) : v(x) {}
    static TW_INLINE f32x16 load(const float *p) { return f32x16(_mm512_loadu_ps(p)); }
    TW_INLINE void store(float *p) const { _mm512_storeu_ps(p, v); }
};

TW_INLINE f32x16 operator+(f32x16 a, f32x16 b) { return f32x16(_mm512_add_ps(a.v, b.v)); }
TW_INLINE f32x16 operator-(f32x16 a, f32x16 b) { return f32x16(_mm512_sub_ps(a.v, b.v)); }
TW_INLINE f32x16 operator*(f32x16 a, f32x16 b) { return f32x16(_mm512_mul_ps(a.v, b.v)); }
TW_INLINE f32x16 operator/(f32x16 a, f32x16 b) { return f32x16(_mm512_div_ps(a.v, b.v)); }
TW_INLINE f32x16 operator-(f32x16 a) {
    return f32x16(_mm512_castsi512_ps(
            _mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(INT32_MIN))));
}
TW_INLINE f32x16 sqrt(f32x16 a) { return f32x16(_mm512_sqrt_ps(a.v)); }

TW_INLINE m32x16 cmp_lt(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ)); }
TW_INLINE m32x16 cmp_le(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ)); }
TW_INLINE m32x16 cmp_gt(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ)); }
TW_INLINE m32x16 cmp_ge(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ)); }
TW_INLINE m32x16 cmp_eq(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ)); }
TW_INLINE m32x16 cmp_neq(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_UQ)); }
TW_INLINE m32x16 cmp_unord(f32x16 a, f32x16 b) { return m32x16(_mm512_cmp_ps_mask(a.v, b.v, _CMP_UNORD_Q)); }

TW_INLINE m32x16 operator&(m32x16 a, m32x16 b) { return m32x16(static_cast<__mmask16>(a.v & b.v)); }
TW_INLINE m32x16 operator|(m32x16 a, m32x16 b) { return m32x16(static_cast<__mmask16>(a.v | b.v)); }
TW_INLINE m32x16 operator^(m32x16 a, m32x16 b) { return m32x16(static_cast<__mmask16>(a.v ^ b.v)); }
TW_INLINE m32x16 operator~(m32x16 a) { return m32x16(static_cast<__mmask16>(~a.v)); }
TW_INLINE bool any(m32x16 m) { return m.v != 0; }
TW_INLINE bool all(m32x16 m) { return m.v == 0xffff; }

TW_INLINE f32x16 select(m32x16 m, f32x16 a, f32x16 b) { return f32x16(_mm512_mask_blend_ps(m.v, b.v, a.v)); }
TW_INLINE i32x16 select(m32x16 m, i32x16 a, i32x16 b) { return i32x16(_mm512_mask_blend_epi32(m.v, b.v, a.v)); }

TW_INLINE i32x16 as_int(f32x16 a) { return i32x16(_mm512_castps_si512(a.v)); }
TW_INLINE f32x16 as_float(i32x16 a) { return f32x16(_mm512_castsi512_ps(a.v)); }
TW_INLINE i32x16 cvtt_i32(f32x16 a) { return i32x16(_mm512_cvttps_epi32(a.v)); }
TW_INLINE f32x16 cvt_f32(i32x16 a) { return f32x16(_mm512_cvtepi32_ps(a.v)); }

TW_INLINE i32x16 operator+(i32x16 a, i32x16 b) { return i32x16(_mm512_add_epi32(a.v, b.v)); }
TW_INLINE i32x16 operator-(i32x16 a, i32x16 b) { return i32x16(_mm512_sub_epi32(a.v, b.v)); }
TW_INLINE i32x16 operator&(i32x16 a, i32x16 b) { return i32x16(_mm512_and_si512(a.v, b.v)); }
TW_INLINE i32x16 operator|(i32x16 a, i32x16 b) { return i32x16(_mm512_or_si512(a.v, b.v)); }
TW_INLINE i32x16 operator^(i32x16 a, i32x16 b) { return i32x16(_mm512_xor_si512(a.v, b.v)); }
template <int N> TW_INLINE i32x16 shl(i32x16 a) { return i32x16(_mm512_slli_epi32(a.v, N)); }
template <int N> TW_INLINE i32x16 shr_logical(i32x16 a) { return i32x16(_mm512_srli_epi32(a.v, N)); }
template <int N> TW_INLINE i32x16 shr_arith(i32x16 a) { return i32x16(_mm512_srai_epi32(a.v, N)); }
TW_INLINE m32x16 cmp_eq(i32x16 a, i32x16 b) { return m32x16(_mm512_cmpeq_epi32_mask(a.v, b.v)); }
TW_INLINE m32x16 cmp_gt(i32x16 a, i32x16 b) { return m32x16(_mm512_cmpgt_epi32_mask(a.v, b.v)); }

#endif  // TW_HAVE_AVX512F

// ================================================================ lane pairs
//
// fpair<F> is a lane type of twice F's width made of two F halves (each
// operation is applied to lo then hi). It gives every logical width on every
// ISA: 8 lanes on SSE2 = fpair<f32x4>, 16 lanes on AVX2 = fpair<f32x8>, 4
// lanes without SIMD = fpair<fpair<f32x1>>. Same bits as the native width.

template <class M> struct mpair {
    M lo, hi;
    TW_HD TW_INLINE mpair() {}
    TW_HD TW_INLINE mpair(M l, M h) : lo(l), hi(h) {}
};

template <class I> struct ipair {
    I lo, hi;
    static constexpr int width = 2 * I::width;
    TW_HD TW_INLINE ipair() {}
    TW_HD TW_INLINE ipair(std::int32_t s) : lo(s), hi(s) {}
    TW_HD TW_INLINE ipair(I l, I h) : lo(l), hi(h) {}
    TW_HD static TW_INLINE ipair load(const std::int32_t *p) { return ipair(I::load(p), I::load(p + I::width)); }
    TW_HD TW_INLINE void store(std::int32_t *p) const {
        lo.store(p);
        hi.store(p + I::width);
    }
};

template <class F> struct fpair {
    F lo, hi;
    using int_type = ipair<typename F::int_type>;
    using mask_type = mpair<typename F::mask_type>;
    static constexpr int width = 2 * F::width;
    TW_HD TW_INLINE fpair() {}
    TW_HD TW_INLINE fpair(float s) : lo(s), hi(s) {}
    TW_HD TW_INLINE fpair(F l, F h) : lo(l), hi(h) {}
    TW_HD static TW_INLINE fpair load(const float *p) { return fpair(F::load(p), F::load(p + F::width)); }
    TW_HD TW_INLINE void store(float *p) const {
        lo.store(p);
        hi.store(p + F::width);
    }
};

template <class F> TW_HD TW_INLINE fpair<F> operator+(fpair<F> a, fpair<F> b) { return {a.lo + b.lo, a.hi + b.hi}; }
template <class F> TW_HD TW_INLINE fpair<F> operator-(fpair<F> a, fpair<F> b) { return {a.lo - b.lo, a.hi - b.hi}; }
template <class F> TW_HD TW_INLINE fpair<F> operator*(fpair<F> a, fpair<F> b) { return {a.lo * b.lo, a.hi * b.hi}; }
template <class F> TW_HD TW_INLINE fpair<F> operator/(fpair<F> a, fpair<F> b) { return {a.lo / b.lo, a.hi / b.hi}; }
template <class F> TW_HD TW_INLINE fpair<F> operator-(fpair<F> a) { return {-a.lo, -a.hi}; }
template <class F> TW_HD TW_INLINE fpair<F> sqrt(fpair<F> a) { return {sqrt(a.lo), sqrt(a.hi)}; }

#define TW_PAIR_CMP(name)                                                                          \
    template <class F> TW_HD TW_INLINE typename fpair<F>::mask_type name(fpair<F> a, fpair<F> b) { \
        return {name(a.lo, b.lo), name(a.hi, b.hi)};                                               \
    }                                                                                              \
    template <class I> TW_HD TW_INLINE auto name(ipair<I> a, ipair<I> b)                           \
            ->mpair<decltype(name(a.lo, b.lo))> {                                                  \
        return {name(a.lo, b.lo), name(a.hi, b.hi)};                                               \
    }
TW_PAIR_CMP(cmp_lt)
TW_PAIR_CMP(cmp_le)
TW_PAIR_CMP(cmp_gt)
TW_PAIR_CMP(cmp_ge)
TW_PAIR_CMP(cmp_eq)
TW_PAIR_CMP(cmp_neq)
TW_PAIR_CMP(cmp_unord)
#undef TW_PAIR_CMP

template <class M> TW_HD TW_INLINE mpair<M> operator&(mpair<M> a, mpair<M> b) { return {a.lo & b.lo, a.hi & b.hi}; }
template <class M> TW_HD TW_INLINE mpair<M> operator|(mpair<M> a, mpair<M> b) { return {a.lo | b.lo, a.hi | b.hi}; }
template <class M> TW_HD TW_INLINE mpair<M> operator^(mpair<M> a, mpair<M> b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
template <class M> TW_HD TW_INLINE mpair<M> operator~(mpair<M> a) { return {~a.lo, ~a.hi}; }
template <class M> TW_HD TW_INLINE bool any(mpair<M> m) { return any(m.lo) || any(m.hi); }
template <class M> TW_HD TW_INLINE bool all(mpair<M> m) { return all(m.lo) && all(m.hi); }

template <class M, class F>
TW_HD TW_INLINE fpair<F> select(mpair<M> m, fpair<F> a, fpair<F> b) {
    return {select(m.lo, a.lo, b.lo), select(m.hi, a.hi, b.hi)};
}
template <class M, class I>
TW_HD TW_INLINE ipair<I> select(mpair<M> m, ipair<I> a, ipair<I> b) {
    return {select(m.lo, a.lo, b.lo), select(m.hi, a.hi, b.hi)};
}

template <class F> TW_HD TW_INLINE typename fpair<F>::int_type as_int(fpair<F> a) { return {as_int(a.lo), as_int(a.hi)}; }
template <class I> TW_HD TW_INLINE auto as_float(ipair<I> a) -> fpair<decltype(as_float(a.lo))> {
    return {as_float(a.lo), as_float(a.hi)};
}
template <class F> TW_HD TW_INLINE typename fpair<F>::int_type cvtt_i32(fpair<F> a) {
    return {cvtt_i32(a.lo), cvtt_i32(a.hi)};
}
template <class I> TW_HD TW_INLINE auto cvt_f32(ipair<I> a) -> fpair<decltype(cvt_f32(a.lo))> {
    return {cvt_f32(a.lo), cvt_f32(a.hi)};
}

template <class I> TW_HD TW_INLINE ipair<I> operator+(ipair<I> a, ipair<I> b) { return {a.lo + b.lo, a.hi + b.hi}; }
template <class I> TW_HD TW_INLINE ipair<I> operator-(ipair<I> a, ipair<I> b) { return {a.lo - b.lo, a.hi - b.hi}; }
template <class I> TW_HD TW_INLINE ipair<I> operator&(ipair<I> a, ipair<I> b) { return {a.lo & b.lo, a.hi & b.hi}; }
template <class I> TW_HD TW_INLINE ipair<I> operator|(ipair<I> a, ipair<I> b) { return {a.lo | b.lo, a.hi | b.hi}; }
template <class I> TW_HD TW_INLINE ipair<I> operator^(ipair<I> a, ipair<I> b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
template <int N, class I> TW_HD TW_INLINE ipair<I> shl(ipair<I> a) { return {shl<N>(a.lo), shl<N>(a.hi)}; }
template <int N, class I> TW_HD TW_INLINE ipair<I> shr_logical(ipair<I> a) {
    return {shr_logical<N>(a.lo), shr_logical<N>(a.hi)};
}
template <int N, class I> TW_HD TW_INLINE ipair<I> shr_arith(ipair<I> a) {
    return {shr_arith<N>(a.lo), shr_arith<N>(a.hi)};
}

// ================================================================ lane traits

template <class F> struct is_f32_lane { static constexpr bool value = false; };
template <class F> struct is_f32_lane<fpair<F>> { static constexpr bool value = is_f32_lane<F>::value; };
template <> struct is_f32_lane<f32x1> { static constexpr bool value = true; };
#if TW_HAVE_SSE2
template <> struct is_f32_lane<f32x4> { static constexpr bool value = true; };
#endif
#if TW_HAVE_AVX2
template <> struct is_f32_lane<f32x8> { static constexpr bool value = true; };
#endif
#if TW_HAVE_AVX512F
template <> struct is_f32_lane<f32x16> { static constexpr bool value = true; };
#endif

// vf<W>: the W-lane float type of this TU for W in {1, 4, 8, 16}: the native
// register type when the ISA has it, otherwise pairs of the next narrower
// width (vi<W>, vm<W>: its int and mask types).
template <int W> struct lane_of;
template <> struct lane_of<1> { using type = f32x1; };
#if TW_HAVE_SSE2
template <> struct lane_of<4> { using type = f32x4; };
#else
template <> struct lane_of<4> { using type = fpair<fpair<f32x1>>; };
#endif
#if TW_HAVE_AVX2
template <> struct lane_of<8> { using type = f32x8; };
#else
template <> struct lane_of<8> { using type = fpair<lane_of<4>::type>; };
#endif
#if TW_HAVE_AVX512F
template <> struct lane_of<16> { using type = f32x16; };
#else
template <> struct lane_of<16> { using type = fpair<lane_of<8>::type>; };
#endif
template <int W> using vf = typename lane_of<W>::type;
template <int W> using vi = typename vf<W>::int_type;
template <int W> using vm = typename vf<W>::mask_type;

}  // namespace math
}  // namespace tw

#endif  // TW_LANES_H
