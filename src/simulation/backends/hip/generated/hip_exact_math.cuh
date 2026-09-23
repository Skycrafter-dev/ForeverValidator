// Generated from src/simulation/backends/cuda/cuda_exact_math.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_EXACT_MATH_CUH
#define FOREVERVALIDATOR_HIP_EXACT_MATH_CUH

#include <hip/hip_runtime.h>

#include <cstdint>

namespace forevervalidator::simulation::hip::exact {
namespace detail {

constexpr double Pi = 3.14159265358979323846264338327950288;
constexpr double HalfPi = 1.57079632679489661923132169163975144;
constexpr double QuarterPi = 0.785398163397448309615660845819875721;
constexpr double TwoOverPi = 0.636619772367581343075535053490057448;
constexpr double HalfPiHigh = 1.570796326734125614166;
constexpr double HalfPiLow = 6.07710050650619224932e-11;
constexpr double Log2E = 1.44269504088896340735992468100189214;
constexpr double Ln2High = 0.693147180559945286227;
constexpr double Ln2Low = 2.31904681384629955842e-17;

__device__ inline float QuietNaN(bool negative = false) {
    return __uint_as_float((negative ? 0x80000000u : 0u) | 0x7fc00000u);
}

__device__ inline double PowerOfTwo(int exponent) {
    if (exponent < -1022) {
        return 0.0;
    }
    if (exponent > 1023) {
        return __longlong_as_double(0x7ff0000000000000ull);
    }
    return ldexp(1.0, exponent);
}

__device__ inline double AtanUnit(double value) {
    const bool aroundOne = value > 0.4142135623730950488;
    const double reduced =
            aroundOne ? (value - 1.0) / (value + 1.0) : value;
    const double square = reduced * reduced;
    double power = reduced;
    double result = reduced;
    for (unsigned termIndex = 1u; termIndex <= 24u; ++termIndex) {
        power *= -square;
        result += power / static_cast<double>(termIndex * 2u + 1u);
    }
    return aroundOne ? QuarterPi + result : result;
}

__device__ inline double Atan2(double y, double x) {
    const bool yNegative = signbit(y);
    const bool xNegative = signbit(x);
    const double absY = fabs(y);
    const double absX = fabs(x);
    if (absY == 0.0) {
        if (xNegative) {
            return yNegative ? -Pi : Pi;
        }
        return y;
    }
    if (absX == 0.0) {
        return yNegative ? -HalfPi : HalfPi;
    }
    double angle = absY <= absX
            ? AtanUnit(absY / absX)
            : HalfPi - AtanUnit(absX / absY);
    if (xNegative) {
        angle = Pi - angle;
    }
    return yNegative ? -angle : angle;
}

struct AtanApproximation {
    double value;
    double error;
};

__device__ inline AtanApproximation AtanUnitApproximation(
        double value) {
    const bool aroundOne = value > 0.4142135623730950488;
    const double reduced =
            aroundOne ? (value - 1.0) / (value + 1.0) : value;
    const double square = reduced * reduced;
    double power = reduced;
    double result = reduced;
    for (unsigned termIndex = 1u; termIndex <= 8u; ++termIndex) {
        power *= -square;
        result += power / static_cast<double>(termIndex * 2u + 1u);
    }
    const double firstOmitted = power * -square;
    const double tailBound =
            fabs(firstOmitted) /
            (19.0 * (1.0 - square));
    return {
            aroundOne ? QuarterPi + result : result,
            tailBound + 2.0e-14,
    };
}

__device__ inline AtanApproximation Atan2Approximation(
        double y, double x) {
    const bool yNegative = signbit(y);
    const bool xNegative = signbit(x);
    const double absY = fabs(y);
    const double absX = fabs(x);
    AtanApproximation result =
            absY <= absX
            ? AtanUnitApproximation(absY / absX)
            : AtanUnitApproximation(absX / absY);
    if (absY > absX) {
        result.value = HalfPi - result.value;
    }
    if (xNegative) {
        result.value = Pi - result.value;
    }
    if (yNegative) {
        result.value = -result.value;
    }
    return result;
}

__device__ inline bool CertifiedFloatRound(
        const AtanApproximation &approximation,
        float *result) {
    const float rounded = __double2float_rn(
            approximation.value);
    const float previous = nextafterf(
            rounded,
            -__uint_as_float(0x7f800000u));
    const float next = nextafterf(
            rounded,
            __uint_as_float(0x7f800000u));
    const double lower =
            (static_cast<double>(previous) +
             static_cast<double>(rounded)) *
            0.5;
    const double upper =
            (static_cast<double>(rounded) +
             static_cast<double>(next)) *
            0.5;
    if (approximation.value - approximation.error <= lower ||
        approximation.value + approximation.error >= upper) {
        return false;
    }
    *result = rounded;
    return true;
}

__device__ inline double SinPolynomial(double value) {
    const double square = value * value;
    double coefficient = -1.0 / 121645100408832000.0;
    coefficient = 1.0 / 355687428096000.0 + square * coefficient;
    coefficient = -1.0 / 1307674368000.0 + square * coefficient;
    coefficient = 1.0 / 6227020800.0 + square * coefficient;
    coefficient = -1.0 / 39916800.0 + square * coefficient;
    coefficient = 1.0 / 362880.0 + square * coefficient;
    coefficient = -1.0 / 5040.0 + square * coefficient;
    coefficient = 1.0 / 120.0 + square * coefficient;
    coefficient = -1.0 / 6.0 + square * coefficient;
    return value + value * square * coefficient;
}

__device__ inline double CosPolynomial(double value) {
    const double square = value * value;
    double coefficient = -1.0 / 6402373705728000.0;
    coefficient = 1.0 / 20922789888000.0 + square * coefficient;
    coefficient = -1.0 / 87178291200.0 + square * coefficient;
    coefficient = 1.0 / 479001600.0 + square * coefficient;
    coefficient = -1.0 / 3628800.0 + square * coefficient;
    coefficient = 1.0 / 40320.0 + square * coefficient;
    coefficient = -1.0 / 720.0 + square * coefficient;
    coefficient = 1.0 / 24.0 + square * coefficient;
    coefficient = -0.5 + square * coefficient;
    return 1.0 + square * coefficient;
}

struct ReducedAngle {
    double value;
    unsigned quadrant;
};

struct FixedProduct {
    std::uint64_t limbs[4];
};

__device__ inline FixedProduct MultiplyTwoOverPi(
        std::uint32_t significand) {
    constexpr std::uint64_t fixed[3] = {
            0xdb6295993c439041ull,
            0xfc2757d1f534ddc0ull,
            0xa2f9836e4e441529ull,
    };
    FixedProduct result{};
    std::uint64_t carry = 0u;
    for (unsigned index = 0u; index < 3u; ++index) {
        const std::uint64_t limb = fixed[index];
        const std::uint64_t low =
                static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(limb)) *
                        significand +
                carry;
        const std::uint64_t high =
                static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(limb >> 32u)) *
                        significand +
                (low >> 32u);
        result.limbs[index] =
                (low & 0xffffffffu) | (high << 32u);
        carry = high >> 32u;
    }
    result.limbs[3] = carry;
    return result;
}

__device__ inline bool FixedBit(const FixedProduct &value,
                                unsigned bitIndex) {
    return bitIndex < 256u &&
           ((value.limbs[bitIndex / 64u] >>
             (bitIndex % 64u)) &
            1u) != 0u;
}

__device__ inline bool AnyFixedBitsBelow(
        const FixedProduct &value,
        unsigned bitCount) {
    const unsigned fullLimbs = bitCount / 64u;
    for (unsigned index = 0u;
         index < fullLimbs && index < 4u;
         ++index) {
        if (value.limbs[index] != 0u) {
            return true;
        }
    }
    const unsigned remaining = bitCount % 64u;
    return fullLimbs < 4u && remaining != 0u &&
           (value.limbs[fullLimbs] &
            ((std::uint64_t{1} << remaining) - 1u)) != 0u;
}

__device__ inline ReducedAngle ReduceLargeAngle(float input) {
    int binaryExponent = 0;
    const float mantissa = frexpf(fabsf(input), &binaryExponent);
    const int exponent = binaryExponent - 1;
    const std::uint32_t significand =
            static_cast<std::uint32_t>(
                    ldexp(static_cast<double>(mantissa), 24));
    const FixedProduct product = MultiplyTwoOverPi(significand);
    const unsigned binaryPoint =
            static_cast<unsigned>(215 - exponent);
    unsigned nearest =
            (FixedBit(product, binaryPoint) ? 1u : 0u) |
            (FixedBit(product, binaryPoint + 1u) ? 2u : 0u);
    const bool halfway = FixedBit(product, binaryPoint - 1u);
    const bool below = AnyFixedBitsBelow(product, binaryPoint - 1u);
    const bool roundUp = halfway && (below || (nearest & 1u) != 0u);
    if (roundUp) {
        nearest = (nearest + 1u) & 3u;
    }
    double fraction = 0.0;
    double weight = 0.5;
    for (unsigned offset = 1u; offset <= 64u; ++offset) {
        if (FixedBit(product, binaryPoint - offset)) {
            fraction += weight;
        }
        weight *= 0.5;
    }
    if (roundUp) {
        fraction -= 1.0;
    }
    if (signbit(input)) {
        fraction = -fraction;
        nearest = (0u - nearest) & 3u;
    }
    return {fraction * HalfPi, nearest};
}

__device__ inline ReducedAngle ReduceAngle(float input) {
    if (fabsf(input) > 1000000.0f) {
        return ReduceLargeAngle(input);
    }
    const double value = static_cast<double>(input);
    const double scaled = value * TwoOverPi;
    const std::int64_t nearest = scaled >= 0.0
            ? static_cast<std::int64_t>(scaled + 0.5)
            : static_cast<std::int64_t>(scaled - 0.5);
    const double count = static_cast<double>(nearest);
    return {
            (value - count * HalfPiHigh) - count * HalfPiLow,
            static_cast<unsigned>(nearest) & 3u,
    };
}

}  // namespace detail

__device__ inline float FromDouble(double value) {
    if (isnan(value)) {
        return detail::QuietNaN(signbit(value));
    }
    return __double2float_rn(value);
}

__device__ inline float FromUnsignedInteger(std::uint32_t value) {
    return __uint2float_rn(value);
}

__device__ inline float Divide(float numerator, float denominator) {
    return numerator / denominator;
}

__device__ inline std::uint32_t TruncateToUint32Modulo(float value) {
    if (!isfinite(value) ||
        fabsf(value) >= 18446744073709551616.0) {
        return 0u;
    }
    const double truncated = trunc(static_cast<double>(value));
    const double wrapped = fmod(fabs(truncated), 4294967296.0);
    const std::uint32_t magnitude =
            static_cast<std::uint32_t>(wrapped);
    return signbit(value) ? 0u - magnitude : magnitude;
}

__device__ inline float Sqrt(float value) {
    if (value >= 0.0f) {
        return sqrtf(value);
    }
    return FromDouble(sqrt(static_cast<double>(value)));
}

__device__ inline float Atan2(float y, float x) {
    if (isnan(x) || isnan(y)) {
        return detail::QuietNaN();
    }
    if (y != 0.0f && x != 0.0f &&
        isfinite(y) && isfinite(x)) {
        const detail::AtanApproximation approximation =
                detail::Atan2Approximation(
                        static_cast<double>(y),
                        static_cast<double>(x));
        float result = 0.0f;
        if (detail::CertifiedFloatRound(
                    approximation, &result)) {
            return result;
        }
    }
    return FromDouble(detail::Atan2(
            static_cast<double>(y), static_cast<double>(x)));
}

__device__ inline float Acos(float value) {
    if (isnan(value) || value < -1.0f || value > 1.0f) {
        return detail::QuietNaN();
    }
    const float positive = 1.0f + value;
    const float negative = 1.0f - value;
    return Atan2(Sqrt(positive * negative), value);
}

__device__ inline float Asin(float value) {
    if (isnan(value) || value < -1.0f || value > 1.0f) {
        return detail::QuietNaN();
    }
    const float positive = 1.0f + value;
    const float negative = 1.0f - value;
    return Atan2(value, Sqrt(positive * negative));
}

__device__ inline float Sin(float value) {
    if (!isfinite(value)) {
        return detail::QuietNaN();
    }
    const detail::ReducedAngle reduced = detail::ReduceAngle(value);
    double result = 0.0;
    switch (reduced.quadrant) {
    case 0u: result = detail::SinPolynomial(reduced.value); break;
    case 1u: result = detail::CosPolynomial(reduced.value); break;
    case 2u: result = -detail::SinPolynomial(reduced.value); break;
    default: result = -detail::CosPolynomial(reduced.value); break;
    }
    return FromDouble(result);
}

__device__ inline float Cos(float value) {
    if (!isfinite(value)) {
        return detail::QuietNaN();
    }
    const detail::ReducedAngle reduced = detail::ReduceAngle(value);
    double result = 0.0;
    switch (reduced.quadrant) {
    case 0u: result = detail::CosPolynomial(reduced.value); break;
    case 1u: result = -detail::SinPolynomial(reduced.value); break;
    case 2u: result = -detail::CosPolynomial(reduced.value); break;
    default: result = detail::SinPolynomial(reduced.value); break;
    }
    return FromDouble(result);
}

struct SinCosResult {
    float sine;
    float cosine;
};

__device__ inline SinCosResult SinCos(float value) {
    if (!isfinite(value)) {
        const float nan = detail::QuietNaN();
        return {nan, nan};
    }
    const detail::ReducedAngle reduced = detail::ReduceAngle(value);
    double sine = 0.0;
    double cosine = 0.0;
    switch (reduced.quadrant) {
    case 0u:
        sine = detail::SinPolynomial(reduced.value);
        cosine = detail::CosPolynomial(reduced.value);
        break;
    case 1u:
        sine = detail::CosPolynomial(reduced.value);
        cosine = -detail::SinPolynomial(reduced.value);
        break;
    case 2u:
        sine = -detail::SinPolynomial(reduced.value);
        cosine = -detail::CosPolynomial(reduced.value);
        break;
    default:
        sine = -detail::CosPolynomial(reduced.value);
        cosine = detail::SinPolynomial(reduced.value);
        break;
    }
    return {FromDouble(sine), FromDouble(cosine)};
}

__device__ inline float Tan(float value) {
    if (!isfinite(value)) {
        return detail::QuietNaN();
    }
    const detail::ReducedAngle reduced = detail::ReduceAngle(value);
    const double sine = detail::SinPolynomial(reduced.value);
    const double cosine = detail::CosPolynomial(reduced.value);
    return FromDouble((reduced.quadrant & 1u) == 0u
            ? sine / cosine
            : -cosine / sine);
}

__device__ inline float Exp(float value) {
    if (isnan(value)) {
        return value;
    }
    if (value > 89.0f) {
        return __uint_as_float(0x7f800000u);
    }
    if (value < -110.0f) {
        return 0.0f;
    }
    const double input = static_cast<double>(value);
    const double scaled = input * detail::Log2E;
    const int exponent = scaled >= 0.0
            ? static_cast<int>(scaled + 0.5)
            : static_cast<int>(scaled - 0.5);
    const double reduced =
            (input - static_cast<double>(exponent) * detail::Ln2High) -
            static_cast<double>(exponent) * detail::Ln2Low;
    double tail = 1.0 / 355687428096000.0;
    tail = 1.0 / 20922789888000.0 + reduced * tail;
    tail = 1.0 / 1307674368000.0 + reduced * tail;
    tail = 1.0 / 87178291200.0 + reduced * tail;
    tail = 1.0 / 6227020800.0 + reduced * tail;
    tail = 1.0 / 479001600.0 + reduced * tail;
    tail = 1.0 / 39916800.0 + reduced * tail;
    tail = 1.0 / 3628800.0 + reduced * tail;
    tail = 1.0 / 362880.0 + reduced * tail;
    tail = 1.0 / 40320.0 + reduced * tail;
    tail = 1.0 / 5040.0 + reduced * tail;
    tail = 1.0 / 720.0 + reduced * tail;
    tail = 1.0 / 120.0 + reduced * tail;
    tail = 1.0 / 24.0 + reduced * tail;
    tail = 1.0 / 6.0 + reduced * tail;
    tail = 0.5 + reduced * tail;
    const double polynomial =
            1.0 + reduced * (1.0 + reduced * tail);
    return FromDouble(polynomial * detail::PowerOfTwo(exponent));
}

__device__ inline float Fmod(float value, float modulus) {
    return fmodf(value, modulus);
}

}  // namespace forevervalidator::simulation::hip::exact

#endif
