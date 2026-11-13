// Twin deterministic math: configuration and compile-time guards.
//
// The twin must give the same bits on every supported CPU, ISA level and
// compiler. This header refuses builds whose flags break that (fast-math,
// x87 evaluation) and defines the small set of macros the math headers use.
// It has no dependency on ForeverValidator.
//
// Contraction (a*b+c fused into one FMA) cannot be detected by a macro on
// GCC. Every TU that uses this library must be compiled with
// -ffp-contract=off (GCC/Clang) or /fp:precise without /fp:contract (MSVC);
// tw::math::contraction_canary() (tw_math.h) checks it at run time, and the
// test suite runs it in every build.
#ifndef TW_MATH_CONFIG_H
#define TW_MATH_CONFIG_H

#include <cfloat>
#include <cstdint>
#include <cstring>

#if defined(__FAST_MATH__)
#error "tw::math: -ffast-math (or -Ofast) breaks determinism"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "tw::math: -ffinite-math-only breaks the NaN/inf rules"
#endif
#if defined(_M_FP_FAST)
#error "tw::math: /fp:fast breaks determinism"
#endif
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "tw::math: float expressions must be evaluated in float (FLT_EVAL_METHOD 0: SSE2 math, no x87)"
#endif
#if defined(__i386__) && !defined(__SSE2_MATH__) && !defined(__CUDACC__)
#error "tw::math: 32-bit x86 needs -msse2 -mfpmath=sse"
#endif

// Device annotation for a later CUDA/HIP port of the scalar (L = 1) path.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TW_HD __host__ __device__
#else
#define TW_HD
#endif

// In a CUDA/HIP TU every TW_INLINE function is also a device function: the GPU engine (src/twin/gpu)
// runs the scalar (L = 1) path, one candidate per thread.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TW_INLINE __host__ __device__ inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(__clang__)
#define TW_INLINE __forceinline
#else
#define TW_INLINE inline __attribute__((always_inline))
#endif

// Out of line, for the rare paths of large inlined kernels (keeps the hot code compact).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TW_NOINLINE __host__ __device__ __attribute__((noinline))
#define TW_NOINLINE_COLD __host__ __device__ __attribute__((noinline, cold))
#elif defined(_MSC_VER) && !defined(__clang__)
#define TW_NOINLINE __declspec(noinline)
#define TW_NOINLINE_COLD __declspec(noinline)
#else
#define TW_NOINLINE __attribute__((noinline))
#define TW_NOINLINE_COLD __attribute__((noinline, cold))
#endif

// Device compiles keep the loop after it rolled (GPU engine, docs/twin/gpu.md: a warp's instruction fetch is its
// limit, and a copy of a large body per iteration multiplies the code a substep runs). Nothing on the host.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define TW_DEVICE_ROLLED _Pragma("unroll 1")
#else
#define TW_DEVICE_ROLLED
#endif

// Defence in depth for Clang: forbid contraction inside every library
// function body even if the includer forgot -ffp-contract=off. GCC has no
// equivalent pragma (it ignores STDC FP_CONTRACT), so the flag stays
// mandatory there.
#if defined(__clang__)
#define TW_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define TW_NO_CONTRACT
#endif

// ISA levels available in this TU (the lane types exist only when the TU is
// compiled for them; the scalar lane always exists).
#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
#define TW_HAVE_SSE2 1
#else
#define TW_HAVE_SSE2 0
#endif
#if defined(__AVX2__)
#define TW_HAVE_AVX2 1
#else
#define TW_HAVE_AVX2 0
#endif
#if defined(__AVX512F__)
#define TW_HAVE_AVX512F 1
#else
#define TW_HAVE_AVX512F 0
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#undef TW_HAVE_SSE2
#undef TW_HAVE_AVX2
#undef TW_HAVE_AVX512F
#define TW_HAVE_SSE2 0
#define TW_HAVE_AVX2 0
#define TW_HAVE_AVX512F 0
#endif

namespace tw {
namespace math {

TW_HD TW_INLINE std::uint32_t bits_of(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

TW_HD TW_INLINE float float_of(std::uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

}  // namespace math
}  // namespace tw

#endif  // TW_MATH_CONFIG_H
