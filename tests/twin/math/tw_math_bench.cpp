// Cycles per call of src/twin/math, scalar and per lane, with glibc/msvcrt
// float libm for context. Core cycles come from a calibration loop of
// dependent integer adds (1 cycle each), so the numbers do not depend on the
// TSC rate. Runs in ~10 s.
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tw_fpenv.h"
#include "tw_math.h"

namespace twm = tw::math;

namespace {

inline void clobber() {
#if defined(__GNUC__)
    __asm__ __volatile__("" ::: "memory");
#endif
}
// Hide a pointer's value from the optimiser so a repeated loop cannot be hoisted.
template <class T> inline T *launder_ptr(T *p) {
#if defined(__GNUC__)
    __asm__ __volatile__("" : "+r"(p) : : "memory");
#endif
    return p;
}

double now_ns() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
}

double core_ghz() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        std::uint64_t x = 0;
        const std::uint64_t iters = 20000000;
        const double t0 = now_ns();
        for (std::uint64_t i = 0; i < iters; ++i) {
            __asm__ __volatile__(
                    "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                    "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                    : "+r"(x));
        }
        const double t1 = now_ns();
        const double ghz = static_cast<double>(iters) * 10.0 / (t1 - t0);
        if (ghz > best) best = ghz;
    }
    return best;
#else
    return 0.0;
#endif
}

constexpr std::size_t kN = 4096;
constexpr int kReps = 2000;

template <class F, class Fn> double ns_per_elem(Fn fn, const std::vector<float> &in, std::vector<float> &out) {
    double best = 1e30;
    for (int trial = 0; trial < 3; ++trial) {
        const double t0 = now_ns();
        for (int r = 0; r < kReps; ++r) {
            const float *pi = launder_ptr(in.data());
            float *po = launder_ptr(out.data());
            for (std::size_t i = 0; i < kN; i += F::width) fn(F::load(pi + i)).store(po + i);
            clobber();
        }
        const double t1 = now_ns();
        const double ns = (t1 - t0) / (static_cast<double>(kReps) * kN);
        if (ns < best) best = ns;
    }
    return best;
}

template <class Fn> double ns_libm(Fn fn, const std::vector<float> &in, std::vector<float> &out) {
    double best = 1e30;
    for (int trial = 0; trial < 3; ++trial) {
        const double t0 = now_ns();
        for (int r = 0; r < kReps / 4; ++r) {
            const float *pi = launder_ptr(in.data());
            float *po = launder_ptr(out.data());
            for (std::size_t i = 0; i < kN; ++i) po[i] = fn(pi[i]);
            clobber();
        }
        const double t1 = now_ns();
        const double ns = (t1 - t0) / (static_cast<double>(kReps / 4) * kN);
        if (ns < best) best = ns;
    }
    return best;
}

template <class Step> double latency_ns(Step step, float x0) {
    double best = 1e30;
    for (int trial = 0; trial < 3; ++trial) {
        float x = x0;
        const int iters = 2000000;
        const double t0 = now_ns();
        for (int i = 0; i < iters; ++i) x = step(x);
        const double t1 = now_ns();
        volatile float sink = x;
        (void)sink;
        const double ns = (t1 - t0) / iters;
        if (ns < best) best = ns;
    }
    return best;
}

std::vector<float> inputs(double lo, double hi, unsigned seed) {
    std::vector<float> v(kN);
    std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + 7;
    for (std::size_t i = 0; i < kN; ++i) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        const double u = static_cast<double>((s * 0x2545F4914F6CDD1Dull) >> 11) / 9007199254740992.0;
        v[i] = static_cast<float>(lo + (hi - lo) * u);
    }
    return v;
}

struct Row {
    const char *name;
    double scalar, x4, x8, x16, libm, lat;
};

double ghz_g = 0.0;

void print_row(const Row &r) {
    auto cyc = [](double ns) { return ns * ghz_g; };
    std::printf("%-10s scalar %6.1f   x4 %5.1f/lane %6.1f/call   x8 %5.1f/lane %6.1f/call   x16 %5.1f/lane %6.1f/call"
                "   latency %5.1f   libm %6.1f\n",
                r.name, cyc(r.scalar), cyc(r.x4), cyc(r.x4) * 4, cyc(r.x8), cyc(r.x8) * 8, cyc(r.x16), cyc(r.x16) * 16,
                cyc(r.lat), cyc(r.libm));
}

template <class Op, class Lib, class Lat>
Row measure(const char *name, Op op, Lib lib, Lat lat, float x0, double lo, double hi) {
    const std::vector<float> in = inputs(lo, hi, 11);
    std::vector<float> out(kN);
    Row r{name, 0, 0, 0, 0, 0, 0};
    r.scalar = ns_per_elem<twm::f32x1>([&](twm::f32x1 x) { return op(x); }, in, out);
#if TW_HAVE_SSE2
    r.x4 = ns_per_elem<twm::f32x4>([&](twm::f32x4 x) { return op(x); }, in, out);
#endif
#if TW_HAVE_AVX2
    r.x8 = ns_per_elem<twm::f32x8>([&](twm::f32x8 x) { return op(x); }, in, out);
#endif
#if TW_HAVE_AVX512F
    r.x16 = ns_per_elem<twm::f32x16>([&](twm::f32x16 x) { return op(x); }, in, out);
#endif
    r.libm = ns_libm(lib, in, out);
    r.lat = latency_ns(lat, x0);
    return r;
}

}  // namespace

int main() {
    twm::FpGuard guard;
    ghz_g = core_ghz();
    std::printf("core clock %.2f GHz (dependent-add calibration); all numbers in core cycles per element\n", ghz_g);
    std::printf("(per lane = throughput per element; per call = one vector call; latency = scalar dependent chain\n"
                " incl. one extra mul/add to stay in domain; libm = std:: float function, scalar)\n");
#define OP(expr) [](auto x) { return expr; }
    print_row(measure("sin", OP(twm::sin(x)), [](float x) { return std::sin(x); },
                      [](float x) { return twm::sin(x) + 1.0f; }, 0.5f, -3.14159, 3.14159));
    print_row(measure("cos", OP(twm::cos(x)), [](float x) { return std::cos(x); },
                      [](float x) { return twm::cos(x) + 1.0f; }, 0.5f, -3.14159, 3.14159));
    print_row(measure("sincos", OP(twm::sincos(x).sin + twm::sincos(x).cos), [](float x) { return std::sin(x) + std::cos(x); },
                      [](float x) { auto r = twm::sincos(x); return r.sin + r.cos; }, 0.5f, -3.14159, 3.14159));
    print_row(measure("sincos_sm", OP(twm::sincos_small(x).sin + twm::sincos_small(x).cos),
                      [](float x) { return std::sin(x) + std::cos(x); },
                      [](float x) { auto r = twm::sincos_small(x); return (r.sin + r.cos) * 0.5f; }, 0.5f, -0.785, 0.785));
    print_row(measure("asin_sm", OP(twm::asin_small(x)), [](float x) { return std::asin(x); },
                      [](float x) { return twm::asin_small(x) * 0.9f; }, 0.4f, -0.5, 0.5));
    print_row(measure("asin", OP(twm::asin(x)), [](float x) { return std::asin(x); },
                      [](float x) { return twm::asin(x) * 0.6f; }, 0.5f, -1.0, 1.0));
    print_row(measure("acos", OP(twm::acos(x)), [](float x) { return std::acos(x); },
                      [](float x) { return twm::acos(x) * 0.6f - 0.9f; }, 0.5f, -1.0, 1.0));
    print_row(measure("atan", OP(twm::atan(x)), [](float x) { return std::atan(x); },
                      [](float x) { return twm::atan(x) * 3.0f; }, 0.5f, -10.0, 10.0));
    print_row(measure("atan2", OP(twm::atan2(x, x * 0.75f + 0.25f)),
                      [](float x) { return std::atan2(x, x * 0.75f + 0.25f); },
                      [](float x) { return twm::atan2(x, 1.5f) * 2.0f; }, 0.5f, -1.0, 1.0));
    print_row(measure("exp", OP(twm::exp(x)), [](float x) { return std::exp(x); },
                      [](float x) { return -twm::exp(x) * 0.5f; }, 0.5f, -10.0, 10.0));
    print_row(measure("sqrt", OP(twm::sqrt(x)), [](float x) { return std::sqrt(x); },
                      [](float x) { return twm::sqrt(x) + 1.0f; }, 2.0f, 0.0, 100.0));
    print_row(measure("rsqrt", OP(twm::rsqrt(x)), [](float x) { return 1.0f / std::sqrt(x); },
                      [](float x) { return twm::rsqrt(x) + 1.0f; }, 2.0f, 0.01, 100.0));
    print_row(measure("floor", OP(twm::floor(x)), [](float x) { return std::floor(x); },
                      [](float x) { return twm::floor(x * 1.5f) - 1.0f; }, 2.5f, -1000.0, 1000.0));
    print_row(measure("fmod32", OP(twm::fmod_pow2(x, decltype(x)(32.0f))), [](float x) { return std::fmod(x, 32.0f); },
                      [](float x) { return twm::fmod_pow2(x, 32.0f) + 33.0f; }, 2.5f, -2000.0, 2000.0));
#undef OP
    return 0;
}
