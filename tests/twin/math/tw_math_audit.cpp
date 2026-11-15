// Object-code audit TU: every tw::math function at every lane width this
// build supports, as out-of-line extern "C" symbols. audit_objects.sh checks
// the object for FMA / approximate reciprocal instructions and for any libm
// symbol. Nothing here calls the C or C++ math library.
#include "tw_fpenv.h"
#include "tw_math.h"

namespace twm = tw::math;

#if defined(__GNUC__)
#define TW_AUDIT extern "C" __attribute__((noinline, used))
#else
#define TW_AUDIT extern "C" __declspec(noinline)
#endif

#define TW_AUDIT_UNARY(sfx, F)                                                                     \
    TW_AUDIT void tw_audit_unary_##sfx(const float *in, float *out, int fn) {                      \
        const F x = F::load(in);                                                                   \
        F r;                                                                                       \
        switch (fn) {                                                                              \
        case 0: r = twm::sin(x); break;                                                            \
        case 1: r = twm::cos(x); break;                                                            \
        case 2: { const auto sc = twm::sincos(x); r = sc.sin + sc.cos; } break;                    \
        case 3: r = twm::asin(x); break;                                                           \
        case 4: r = twm::acos(x); break;                                                           \
        case 5: r = twm::atan(x); break;                                                           \
        case 6: r = twm::atan2(x, F::load(in + F::width)); break;                                  \
        case 7: r = twm::exp(x); break;                                                            \
        case 8: r = twm::sqrt(x); break;                                                           \
        case 9: r = twm::rsqrt(x); break;                                                          \
        case 10: r = twm::floor(x); break;                                                         \
        case 11: r = twm::ceil(x); break;                                                          \
        case 12: r = twm::trunc(x); break;                                                         \
        case 13: r = twm::rint(x); break;                                                          \
        case 14: r = twm::fmod_pow2(x, F(32.0f)); break;                                           \
        case 15: r = twm::as_float(twm::to_i32_trunc_sat(x)); break;                               \
        case 16: r = twm::as_float(twm::to_u32_trunc_wrap(x)); break;                              \
        case 17: r = twm::from_u32(twm::as_int(x)); break;                                         \
        case 18: { const auto sc = twm::sincos_small(x); r = sc.sin + sc.cos; } break;             \
        case 19: r = twm::asin_small(x); break;                                                    \
        default: r = twm::clamp(x, F(-1.0f), F(1.0f)); break;                                      \
        }                                                                                          \
        r.store(out);                                                                              \
    }

TW_AUDIT_UNARY(x1, twm::f32x1)
TW_AUDIT_UNARY(v8, twm::vf<8>)    // native or paired, whatever this ISA gives
TW_AUDIT_UNARY(v16, twm::vf<16>)
#if TW_HAVE_SSE2
TW_AUDIT_UNARY(x4, twm::f32x4)
#endif
#if TW_HAVE_AVX2
TW_AUDIT_UNARY(x8, twm::f32x8)
#endif
#if TW_HAVE_AVX512F
TW_AUDIT_UNARY(x16, twm::f32x16)
#endif

static_assert(twm::vf<1>::width == 1 && twm::vf<4>::width == 4 && twm::vf<8>::width == 8 &&
                      twm::vf<16>::width == 16,
              "vf<W> widths");

TW_AUDIT unsigned tw_audit_guard(void) {
    twm::FpGuard guard;
    return twm::fp_mode_get();
}
