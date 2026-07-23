#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"

#include "engine/core/binary32_math.h"
#include "simulation/runtime/replay_deterministic_execution.h"

#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>

#if (defined(__i386__) || defined(__x86_64__)) && \
        (defined(__GNUC__) || defined(__clang__))
#define FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH 1
#include <emmintrin.h>
#include <xmmintrin.h>
#else
#define FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH 0
#endif

namespace forevervalidator::simulation {
namespace {

[[maybe_unused]] constexpr bool HasSupportedBinaryFormats =
        std::numeric_limits<float>::is_iec559 &&
        std::numeric_limits<float>::radix == 2 &&
        std::numeric_limits<float>::digits == 24 &&
        std::numeric_limits<float>::max_exponent == 128 &&
        sizeof(float) == sizeof(std::uint32_t) &&
        std::numeric_limits<double>::is_iec559 &&
        std::numeric_limits<double>::radix == 2 &&
        std::numeric_limits<double>::digits == 53 &&
        std::numeric_limits<double>::max_exponent == 1024 &&
        sizeof(double) == sizeof(std::uint64_t);

template<typename To, typename From>
To BitCopy(const From &value) noexcept {
    static_assert(sizeof(To) == sizeof(From), "bit-copy size mismatch");
    To result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

#if FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH

#define FOREVERVALIDATOR_TARGET_SSE2 __attribute__((target("sse2")))
#define FOREVERVALIDATOR_NOINLINE __attribute__((noinline))

constexpr unsigned int MxcsrControlMask = 0xffc0u;
constexpr unsigned int DeterministicMxcsrControl = 0x1f80u;
constexpr std::uint64_t DoubleSignMask = 0x8000000000000000ull;
constexpr std::uint64_t MaximumFloatAsDoubleBits = 0x47efffffe0000000ull;
constexpr std::uint32_t PositiveInfinityBits = 0x7f800000u;
constexpr std::uint32_t NegativeZeroBits = 0x80000000u;

bool CpuHasSse2() noexcept {
#if defined(__x86_64__)
    return true;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2") != 0;
#endif
}

FOREVERVALIDATOR_TARGET_SSE2 unsigned int ReadMxcsrSse2() noexcept {
    return _mm_getcsr();
}

#endif

}  // namespace

#if FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH

FOREVERVALIDATOR_TARGET_SSE2 FOREVERVALIDATOR_NOINLINE
float OptimizedCpuBinary32FromDoubleX86Sse2(double value) noexcept {
    const std::uint64_t magnitudeBits =
            BitCopy<std::uint64_t>(value) & ~DoubleSignMask;
    if (magnitudeBits > MaximumFloatAsDoubleBits) {
        return Binary32::FromDouble(value);
    }

    const __m128 converted = _mm_cvtsd_ss(_mm_setzero_ps(), _mm_set_sd(value));
    return _mm_cvtss_f32(converted);
}

FOREVERVALIDATOR_TARGET_SSE2 FOREVERVALIDATOR_NOINLINE
float OptimizedCpuBinary32SqrtX86Sse2(float value) noexcept {
    const std::uint32_t bits = BitCopy<std::uint32_t>(value);
    if (bits > PositiveInfinityBits && bits != NegativeZeroBits) {
        return CIsqrt(value);
    }

    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(value)));
}

#undef FOREVERVALIDATOR_NOINLINE
#undef FOREVERVALIDATOR_TARGET_SSE2

#else

float OptimizedCpuBinary32FromDoubleX86Sse2(double value) noexcept {
    return Binary32::FromDouble(value);
}

float OptimizedCpuBinary32SqrtX86Sse2(float value) noexcept {
    return CIsqrt(value);
}

#endif

OptimizedCpuBinary32MathPath
SelectOptimizedCpuBinary32MathPathForActiveExecution() noexcept {
#if FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH
    if (!HasSupportedBinaryFormats ||
        !tmnf::simulation::DeterministicExecutionScope::IsActive() ||
        std::fegetround() != FE_TONEAREST || !CpuHasSse2()) {
        return OptimizedCpuBinary32MathPath::Reference;
    }
    if ((ReadMxcsrSse2() & MxcsrControlMask) !=
        DeterministicMxcsrControl) {
        return OptimizedCpuBinary32MathPath::Reference;
    }
    return OptimizedCpuBinary32MathPath::X86Sse2;
#else
    return OptimizedCpuBinary32MathPath::Reference;
#endif
}

float OptimizedCpuBinary32FromDouble(
        double value,
        OptimizedCpuBinary32MathPath path) noexcept {
#if FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH
    if (HasSupportedBinaryFormats &&
        path == OptimizedCpuBinary32MathPath::X86Sse2) {
        return OptimizedCpuBinary32FromDoubleX86Sse2(value);
    }
#else
    static_cast<void>(path);
#endif
    return Binary32::FromDouble(value);
}

float OptimizedCpuBinary32Sqrt(
        float value,
        OptimizedCpuBinary32MathPath path) noexcept {
#if FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH
    if (HasSupportedBinaryFormats &&
        path == OptimizedCpuBinary32MathPath::X86Sse2) {
        return OptimizedCpuBinary32SqrtX86Sse2(value);
    }
#else
    static_cast<void>(path);
#endif
    return CIsqrt(value);
}

}  // namespace forevervalidator::simulation

#undef FOREVERVALIDATOR_HAS_X86_SSE2_BINARY32_PATH
