// Gate 1' yardstick: a binary32 number with stochastic rounding (SR).
//
// Every arithmetic result is computed in double (exact for + - * of two floats
// whose exponents differ by <= 29, otherwise within 2^-53, which only matters
// for the rare result that lies that close to a float), then rounded to one of
// the two floats that bracket it, up with probability (r - lo) / (hi - lo).
// Each op is therefore a faithful rounding (error < 1 ulp, expected error 0).
// A board evaluation with this type is one valid float implementation of the
// board's operation chain; many evaluations with different seeds sample the
// float rounding error that the chain accumulates within one tick
// (docs/twin/harness.md "Gate 1'"). Used by tools/twin/ref (the double and SR
// instantiations of the twin's scalar L = 1 path) and by the collision
// reference if it is instantiated on this type.
//
// Header-only, FV-free. The RNG is thread-local; Seed() before each sample.
#ifndef TW_REF_SR_H
#define TW_REF_SR_H

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace tw::sr {

inline std::uint64_t &RngState() {
    static thread_local std::uint64_t s = 0x9E3779B97F4A7C15ull;
    return s;
}
inline void Seed(std::uint64_t seed) {
    std::uint64_t z = seed + 0x9E3779B97F4A7C15ull;   // splitmix64 of the seed
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    RngState() = z != 0u ? z : 1u;
}
inline double Uniform() {   // xorshift64*, [0, 1)
    std::uint64_t &x = RngState();
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return static_cast<double>((x * 0x2545F4914F6CDD1Dull) >> 11) * 0x1p-53;
}

// Count of rounding decisions taken (all calls, for the report).
inline std::uint64_t &RoundCount() {
    static thread_local std::uint64_t n = 0u;
    return n;
}

inline float Round(double r) {
    const float f = static_cast<float>(r);   // nearest
    if (!std::isfinite(r) || static_cast<double>(f) == r) return f;
    ++RoundCount();
    float lo, hi;
    if (static_cast<double>(f) < r) {
        lo = f;
        hi = std::nextafter(f, INFINITY);
    } else {
        hi = f;
        lo = std::nextafter(f, -INFINITY);
    }
    if (!std::isfinite(lo)) return hi;
    if (!std::isfinite(hi)) return lo;
    const double p = (r - static_cast<double>(lo)) / (static_cast<double>(hi) - static_cast<double>(lo));
    return Uniform() < p ? hi : lo;
}

struct sr {
    float v;
    constexpr sr() : v(0.0f) {}
    constexpr sr(float x) : v(x) {}                       // exact
    sr(double x) : v(Round(x)) {}                         // one rounding
    template <class I, class = std::enable_if_t<std::is_integral<I>::value>>
    sr(I x) : v(Round(static_cast<double>(x))) {}         // FV int -> float: one rounding
    explicit operator double() const { return static_cast<double>(v); }

    sr &operator+=(sr b) { return *this = *this + b; }
    sr &operator-=(sr b) { return *this = *this - b; }
    sr &operator*=(sr b) { return *this = *this * b; }
    sr &operator/=(sr b) { return *this = *this / b; }

    friend sr operator+(sr a, sr b) { return sr(Round(static_cast<double>(a.v) + static_cast<double>(b.v))); }
    friend sr operator-(sr a, sr b) { return sr(Round(static_cast<double>(a.v) - static_cast<double>(b.v))); }
    friend sr operator*(sr a, sr b) { return sr(Round(static_cast<double>(a.v) * static_cast<double>(b.v))); }
    friend sr operator/(sr a, sr b) { return sr(Round(static_cast<double>(a.v) / static_cast<double>(b.v))); }
    friend constexpr sr operator-(sr a) { return sr(-a.v); }
    friend constexpr sr operator+(sr a) { return a; }
    friend constexpr bool operator<(sr a, sr b) { return a.v < b.v; }
    friend constexpr bool operator<=(sr a, sr b) { return a.v <= b.v; }
    friend constexpr bool operator>(sr a, sr b) { return a.v > b.v; }
    friend constexpr bool operator>=(sr a, sr b) { return a.v >= b.v; }
    friend constexpr bool operator==(sr a, sr b) { return a.v == b.v; }
    friend constexpr bool operator!=(sr a, sr b) { return a.v != b.v; }
};
static_assert(sizeof(sr) == 4 && std::is_trivially_copyable<sr>::value, "sr is a float");

// Functions: libm in double on the stored float, one rounding (a faithful float libm).
inline sr sqrt(sr x) { return sr(Round(std::sqrt(static_cast<double>(x.v)))); }
inline sr exp(sr x) { return sr(Round(std::exp(static_cast<double>(x.v)))); }
inline sr sin(sr x) { return sr(Round(std::sin(static_cast<double>(x.v)))); }
inline sr cos(sr x) { return sr(Round(std::cos(static_cast<double>(x.v)))); }
inline sr asin(sr x) { return sr(Round(std::asin(static_cast<double>(x.v)))); }
inline sr acos(sr x) { return sr(Round(std::acos(static_cast<double>(x.v)))); }
inline sr atan2(sr y, sr x) { return sr(Round(std::atan2(static_cast<double>(y.v), static_cast<double>(x.v)))); }
inline sr fabs(sr x) { return sr(std::fabs(x.v)); }

}  // namespace tw::sr

#endif
