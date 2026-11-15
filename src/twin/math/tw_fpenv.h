// Twin deterministic math: floating-point environment.
//
// Policy (docs/twin/math.md, "Float mode"): the twin runs with
//   round to nearest even, IEEE gradual underflow (FTZ = 0, DAZ = 0),
//   all exceptions masked
// i.e. MXCSR = 0x1F80 on x86, FPCR = 0 on AArch64. Host code (a game overlay,
// a driver, an audio library, a fast-math DLL) may have set FTZ/DAZ or a
// directed rounding mode, so every twin entry point (Step, RunBatch, ...)
// opens an FpGuard: it saves the mode, sets the canonical one and restores
// the caller's on exit. Cost: one stmxcsr + ldmxcsr pair each way (~20-40
// cycles), so put it at API entry, not around single math calls.
//
// Compilers do not model the FP environment as a dependency (no
// -frounding-math), so keep the guarded computation in a separate,
// non-inlined function from the one that opens the guard:
//     void Step(...) { tw::math::FpGuard guard; StepKernel(...); }  // StepKernel noinline
#ifndef TW_FPENV_H
#define TW_FPENV_H

#include "tw_math_config.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#define TW_FPENV_X86 1
#else
#define TW_FPENV_X86 0
#endif

#if !TW_FPENV_X86
#include <cfenv>
#endif

namespace tw {
namespace math {

#if TW_FPENV_X86
// RN (RC = 00), FTZ (bit 15) = 0, DAZ (bit 6) = 0, all six exception masks set.
constexpr unsigned kCanonicalFpMode = 0x1F80u;
// Bits that define behaviour (everything except the six sticky status flags).
constexpr unsigned kFpModeControlMask = 0xFFC0u;

inline unsigned fp_mode_get() { return _mm_getcsr(); }
inline void fp_mode_set(unsigned mode) { _mm_setcsr(mode); }
#elif defined(__aarch64__)
// AArch64 FPCR: RMode (23:22) = 00, FZ (24) = 0, DN (25) = 0, AH/FIZ = 0,
// trap enables = 0. Not yet tested on hardware (see math.md).
constexpr unsigned kCanonicalFpMode = 0u;
constexpr unsigned kFpModeControlMask = 0xFFFFFFFFu;
inline unsigned fp_mode_get() {
    unsigned long long v;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(v));
    return static_cast<unsigned>(v);
}
inline void fp_mode_set(unsigned mode) {
    unsigned long long v = mode;
    __asm__ __volatile__("msr fpcr, %0" : : "r"(v));
}
#else
// Generic fallback: only the rounding mode can be controlled portably.
constexpr unsigned kCanonicalFpMode = 0u;
constexpr unsigned kFpModeControlMask = 0u;
inline unsigned fp_mode_get() { return static_cast<unsigned>(std::fegetround()); }
inline void fp_mode_set(unsigned) { std::fesetround(FE_TONEAREST); }
#endif

// True when the current thread runs in the canonical mode.
inline bool fp_mode_is_canonical() {
    return (fp_mode_get() & kFpModeControlMask) == (kCanonicalFpMode & kFpModeControlMask);
}

class FpGuard {
public:
    FpGuard() : saved_(fp_mode_get()) { fp_mode_set(kCanonicalFpMode); }
    ~FpGuard() { fp_mode_set(saved_); }
    FpGuard(const FpGuard &) = delete;
    FpGuard &operator=(const FpGuard &) = delete;

private:
    unsigned saved_;
};

}  // namespace math
}  // namespace tw

#endif  // TW_FPENV_H
