// Tests for src/twin/math (docs/twin/math.md has the commands and results).
//
//   tw_math_tests canary                 contraction canary (raw expression, every lane width)
//   tw_math_tests identity               fixed input set -> one hash per function; all lane widths
//                                        must equal scalar; FTZ/DAZ hostility under FpGuard
//   tw_math_tests exhaustive <fn>        all 2^32 inputs (unary): hash of the outputs, every lane
//                                        width compared to scalar, exact functions checked
//                                        against their reference
//   tw_math_tests accuracy <fn> <lo> <hi>  every float in [lo, hi]: max ulp error vs double libm
//   tw_math_tests atan2 <samples>        atan2 accuracy: random pairs, unit circle, axes
//   tw_math_tests equal <fnA> <fnB> <lo> <hi>  bitwise equality of two functions on [lo, hi]
//   tw_math_tests list                   function names
// Threads: TW_THREADS (default 6). Each invocation is sized to stay < 60 s.
#include <atomic>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tw_fpenv.h"
#include "tw_math.h"

namespace twm = tw::math;

namespace {

// ------------------------------------------------------------------ utilities

std::uint32_t fbits(float x) { return twm::bits_of(x); }
float bfloat(std::uint32_t u) { return twm::float_of(u); }

struct Rng {  // xorshift64*, fixed seeds: inputs are integer-generated (no float ops)
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    std::uint32_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return static_cast<std::uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
};

struct Hash {  // order-dependent 64-bit hash of 32-bit words
    std::uint64_t h = 0x84222325cbf29ce4ull;
    void add(std::uint32_t w) {
        h ^= w;
        h *= 0x100000001b3ull;
        h ^= h >> 29;
    }
};

unsigned thread_count() {
    const char *e = std::getenv("TW_THREADS");
    const int n = e ? std::atoi(e) : 6;
    return n > 0 ? static_cast<unsigned>(n) : 1u;
}

template <class Body> void parallel_chunks(std::size_t chunks, Body body) {
    std::atomic<std::size_t> next(0);
    std::vector<std::thread> pool;
    const unsigned n = thread_count();
    for (unsigned t = 0; t < n; ++t) {
        pool.emplace_back([&] {
            twm::FpGuard guard;
            for (;;) {
                const std::size_t c = next.fetch_add(1);
                if (c >= chunks) break;
                body(c);
            }
        });
    }
    for (auto &th : pool) th.join();
}

// ------------------------------------------------------------------ functions

// Each functor: eval<F>(F x) for every lane type; ref(double) reference for
// accuracy (transcendentals) or exact_ok(in, out) for exact functions.
struct FnSin {
    static constexpr const char *name = "sin";
    template <class F> static F eval(F x) { return twm::sin(x); }
    static double ref(double x) { return std::sin(x); }
    static long double refl(long double x) { return sinl(x); }
};
struct FnCos {
    static constexpr const char *name = "cos";
    template <class F> static F eval(F x) { return twm::cos(x); }
    static double ref(double x) { return std::cos(x); }
    static long double refl(long double x) { return cosl(x); }
};
struct FnSincosS {  // sin from sincos (must equal sin bitwise; separate hash)
    static constexpr const char *name = "sincos.sin";
    template <class F> static F eval(F x) { return twm::sincos(x).sin; }
    static double ref(double x) { return std::sin(x); }
    static long double refl(long double x) { return sinl(x); }
};
struct FnSincosC {
    static constexpr const char *name = "sincos.cos";
    template <class F> static F eval(F x) { return twm::sincos(x).cos; }
    static double ref(double x) { return std::cos(x); }
    static long double refl(long double x) { return cosl(x); }
};
struct FnSincosSmallS {
    static constexpr const char *name = "sincos_small.sin";
    template <class F> static F eval(F x) { return twm::sincos_small(x).sin; }
    static double ref(double x) { return std::sin(x); }
    static long double refl(long double x) { return sinl(x); }
};
struct FnSincosSmallC {
    static constexpr const char *name = "sincos_small.cos";
    template <class F> static F eval(F x) { return twm::sincos_small(x).cos; }
    static double ref(double x) { return std::cos(x); }
    static long double refl(long double x) { return cosl(x); }
};
struct FnAsinSmall {
    static constexpr const char *name = "asin_small";
    template <class F> static F eval(F x) { return twm::asin_small(x); }
    static double ref(double x) { return std::asin(x); }
    static long double refl(long double x) { return asinl(x); }
};
struct FnAsin {
    static constexpr const char *name = "asin";
    template <class F> static F eval(F x) { return twm::asin(x); }
    static double ref(double x) { return std::asin(x); }
    static long double refl(long double x) { return asinl(x); }
};
struct FnAcos {
    static constexpr const char *name = "acos";
    template <class F> static F eval(F x) { return twm::acos(x); }
    static double ref(double x) { return std::acos(x); }
    static long double refl(long double x) { return acosl(x); }
};
struct FnAtan {
    static constexpr const char *name = "atan";
    template <class F> static F eval(F x) { return twm::atan(x); }
    static double ref(double x) { return std::atan(x); }
    static long double refl(long double x) { return atanl(x); }
};
struct FnExp {
    static constexpr const char *name = "exp";
    template <class F> static F eval(F x) { return twm::exp(x); }
    static double ref(double x) { return std::exp(x); }
    static long double refl(long double x) { return expl(x); }
};
struct FnSqrt {
    static constexpr const char *name = "sqrt";
    template <class F> static F eval(F x) { return twm::sqrt(x); }
    static double ref(double x) { return std::sqrt(x); }
    static long double refl(long double x) { return sqrtl(x); }
    static bool exact_ok(float in, float out) {  // correctly rounded (double rounding is innocuous for sqrt)
        const float r = static_cast<float>(std::sqrt(static_cast<double>(in)));
        return (r != r) ? (out != out) : fbits(r) == fbits(out);
    }
};
struct FnRsqrt {
    static constexpr const char *name = "rsqrt";
    template <class F> static F eval(F x) { return twm::rsqrt(x); }
    static double ref(double x) { return 1.0 / std::sqrt(x); }
    static long double refl(long double x) { return 1.0L / sqrtl(x); }
};
// NaN inputs: trunc/floor/ceil/rint return the input unchanged
bool same_or_nan_passthrough(float in, float out, float ref) {
    if (in != in) return fbits(in) == fbits(out);
    return fbits(ref) == fbits(out);
}
struct FnTrunc {
    static constexpr const char *name = "trunc";
    template <class F> static F eval(F x) { return twm::trunc(x); }
    static bool exact_ok(float in, float out) { return same_or_nan_passthrough(in, out, std::trunc(in)); }
};
struct FnFloor {
    static constexpr const char *name = "floor";
    template <class F> static F eval(F x) { return twm::floor(x); }
    static bool exact_ok(float in, float out) { return same_or_nan_passthrough(in, out, std::floor(in)); }
};
struct FnCeil {
    static constexpr const char *name = "ceil";
    template <class F> static F eval(F x) { return twm::ceil(x); }
    static bool exact_ok(float in, float out) { return same_or_nan_passthrough(in, out, std::ceil(in)); }
};
struct FnRint {
    static constexpr const char *name = "rint";
    template <class F> static F eval(F x) { return twm::rint(x); }
    static bool exact_ok(float in, float out) { return same_or_nan_passthrough(in, out, std::nearbyint(in)); }
};
struct FnFmod32 {
    static constexpr const char *name = "fmod_pow2.32";
    template <class F> static F eval(F x) { return twm::fmod_pow2(x, F(32.0f)); }
    static bool exact_ok(float in, float out) {
        const float r = std::fmod(in, 32.0f);
        return (r != r) ? fbits(out) == 0x7fc00000u : fbits(r) == fbits(out);
    }
};
struct FnToI32 {
    static constexpr const char *name = "to_i32_trunc_sat";
    template <class F> static F eval(F x) { return twm::as_float(twm::to_i32_trunc_sat(x)); }
    static bool exact_ok(float in, float out) {
        std::int32_t r;
        if (in != in) r = 0;
        else if (in >= 2147483520.0f) r = 2147483520;
        else if (in <= -2147483648.0f) r = INT32_MIN;
        else r = static_cast<std::int32_t>(in);
        return static_cast<std::uint32_t>(r) == fbits(out);
    }
};
struct FnToU32Wrap {
    static constexpr const char *name = "to_u32_trunc_wrap";
    template <class F> static F eval(F x) { return twm::as_float(twm::to_u32_trunc_wrap(x)); }
    static bool exact_ok(float in, float out) {  // ForeverValidator Binary32::TruncateToUint32Modulo
        std::uint32_t r;
        if (!std::isfinite(in) || std::fabs(in) >= 18446744073709551616.0) {
            r = 0u;
        } else {
            const double t = std::trunc(static_cast<double>(in));
            const std::uint32_t m = static_cast<std::uint32_t>(std::fmod(std::fabs(t), 4294967296.0));
            r = std::signbit(in) ? 0u - m : m;
        }
        return r == fbits(out);
    }
};
struct FnFromU32 {
    static constexpr const char *name = "from_u32";
    template <class F> static F eval(F x) { return twm::from_u32(twm::as_int(x)); }
    static bool exact_ok(float in, float out) { return fbits(static_cast<float>(fbits(in))) == fbits(out); }
};
struct FnFromI32 {
    static constexpr const char *name = "from_i32";
    template <class F> static F eval(F x) { return twm::cvt_f32(twm::as_int(x)); }
    static bool exact_ok(float in, float out) {
        return fbits(static_cast<float>(static_cast<std::int32_t>(fbits(in)))) == fbits(out);
    }
};

// ------------------------------------------------------------------ lane runner

template <class Fn, class F> void run_lanes(const float *in, float *out, std::size_t n) {
    std::size_t i = 0;
    for (; i + F::width <= n; i += F::width) Fn::template eval<F>(F::load(in + i)).store(out + i);
    for (; i < n; ++i) out[i] = Fn::template eval<twm::f32x1>(twm::f32x1(in[i])).v;
}

using RunFn = void (*)(const float *, float *, std::size_t);
struct LaneSet {
    const char *names[8];
    RunFn fns[8];
    int count;
};

template <class Fn> LaneSet lanes_for() {
    LaneSet s{};
    s.names[s.count] = "x1";
    s.fns[s.count++] = &run_lanes<Fn, twm::f32x1>;
#if TW_HAVE_SSE2
    s.names[s.count] = "x4";
    s.fns[s.count++] = &run_lanes<Fn, twm::f32x4>;
#endif
#if TW_HAVE_AVX2
    s.names[s.count] = "x8";
    s.fns[s.count++] = &run_lanes<Fn, twm::f32x8>;
#endif
#if TW_HAVE_AVX512F
    s.names[s.count] = "x16";
    s.fns[s.count++] = &run_lanes<Fn, twm::f32x16>;
#endif
    // paired forms (what vf<W> uses when the ISA lacks the native width);
    // TW_NATIVE_LANES_ONLY=1 skips them (unoptimised -O0 builds are too slow)
    if (std::getenv("TW_NATIVE_LANES_ONLY")) return s;
    s.names[s.count] = "x4=pair(pair(x1))";
    s.fns[s.count++] = &run_lanes<Fn, twm::fpair<twm::fpair<twm::f32x1>>>;
#if TW_HAVE_SSE2
    s.names[s.count] = "x8=pair(x4)";
    s.fns[s.count++] = &run_lanes<Fn, twm::fpair<twm::f32x4>>;
#endif
#if TW_HAVE_AVX2
    s.names[s.count] = "x16=pair(x8)";
    s.fns[s.count++] = &run_lanes<Fn, twm::fpair<twm::f32x8>>;
#endif
    return s;
}

int native_widths() { return 1 + TW_HAVE_SSE2 + TW_HAVE_AVX2 + TW_HAVE_AVX512F; }

template <class T, class = void> struct has_exact : std::false_type {};
template <class T>
struct has_exact<T, decltype(void(T::exact_ok(0.0f, 0.0f)))> : std::true_type {};
template <class Fn> bool call_exact(float in, float out, std::true_type) { return Fn::exact_ok(in, out); }
template <class Fn> bool call_exact(float, float, std::false_type) { return true; }

struct Entry {
    const char *name;
    LaneSet lanes;
    bool exact;
    bool (*exact_ok)(float, float);
    double (*ref)(double);
    long double (*refl)(long double);
};

template <class T, class = void> struct has_ref : std::false_type {};
template <class T> struct has_ref<T, decltype(void(T::ref(0.0)))> : std::true_type {};
template <class Fn> double (*ref_ptr(std::true_type))(double) { return &Fn::ref; }
template <class Fn> double (*ref_ptr(std::false_type))(double) { return nullptr; }
template <class Fn> long double (*refl_ptr(std::true_type))(long double) { return &Fn::refl; }
template <class Fn> long double (*refl_ptr(std::false_type))(long double) { return nullptr; }

template <class Fn> Entry make_entry() {
    Entry e;
    e.name = Fn::name;
    e.lanes = lanes_for<Fn>();
    e.exact = has_exact<Fn>::value;
    e.exact_ok = [](float in, float out) { return call_exact<Fn>(in, out, has_exact<Fn>()); };
    e.ref = ref_ptr<Fn>(has_ref<Fn>());
    e.refl = refl_ptr<Fn>(has_ref<Fn>());
    return e;
}

const std::vector<Entry> &entries() {
    static const std::vector<Entry> v = {
            make_entry<FnSin>(),     make_entry<FnCos>(),       make_entry<FnSincosS>(),
            make_entry<FnSincosC>(), make_entry<FnSincosSmallS>(), make_entry<FnSincosSmallC>(),
            make_entry<FnAsin>(),    make_entry<FnAsinSmall>(), make_entry<FnAcos>(),
            make_entry<FnAtan>(),    make_entry<FnExp>(),       make_entry<FnSqrt>(),
            make_entry<FnRsqrt>(),   make_entry<FnTrunc>(),     make_entry<FnFloor>(),
            make_entry<FnCeil>(),    make_entry<FnRint>(),      make_entry<FnFmod32>(),
            make_entry<FnToI32>(),   make_entry<FnToU32Wrap>(), make_entry<FnFromU32>(),
            make_entry<FnFromI32>(),
    };
    return v;
}

const Entry *find_entry(const char *name) {
    for (const Entry &e : entries())
        if (std::strcmp(e.name, name) == 0) return &e;
    return nullptr;
}

// atan2 runs over pairs
template <class F> void run_atan2(const float *y, const float *x, float *out, std::size_t n) {
    std::size_t i = 0;
    for (; i + F::width <= n; i += F::width) twm::atan2(F::load(y + i), F::load(x + i)).store(out + i);
    for (; i < n; ++i) out[i] = twm::atan2(y[i], x[i]);
}
using RunFn2 = void (*)(const float *, const float *, float *, std::size_t);
struct LaneSet2 {
    const char *names[8];
    RunFn2 fns[8];
    int count;
};
LaneSet2 atan2_lanes() {
    LaneSet2 s{};
    s.names[s.count] = "x1";
    s.fns[s.count++] = &run_atan2<twm::f32x1>;
#if TW_HAVE_SSE2
    s.names[s.count] = "x4";
    s.fns[s.count++] = &run_atan2<twm::f32x4>;
#endif
#if TW_HAVE_AVX2
    s.names[s.count] = "x8";
    s.fns[s.count++] = &run_atan2<twm::f32x8>;
#endif
#if TW_HAVE_AVX512F
    s.names[s.count] = "x16";
    s.fns[s.count++] = &run_atan2<twm::f32x16>;
#endif
    if (std::getenv("TW_NATIVE_LANES_ONLY")) return s;
    s.names[s.count] = "x4=pair(pair(x1))";
    s.fns[s.count++] = &run_atan2<twm::fpair<twm::fpair<twm::f32x1>>>;
#if TW_HAVE_SSE2
    s.names[s.count] = "x8=pair(x4)";
    s.fns[s.count++] = &run_atan2<twm::fpair<twm::f32x4>>;
#endif
    return s;
}

// ------------------------------------------------------------------ ulp error

// error of `got` against the exact value `ref`, in ulps of the float format
// at ref (denormal ulp 2^-149 below 2^-126; overflow treated as 2^128).
double ulp_error(float got, double ref) {
    if (ref != ref) return (got != got) ? 0.0 : 1e30;
    if (got != got) return 1e30;
    double g = static_cast<double>(got);
    // |ref| >= FLT_MAX + half an ulp rounds to inf: inf is then the exact answer
    if (std::fabs(ref) >= 0x1.ffffffp127) return (std::isinf(g) && (g > 0) == (ref > 0)) ? 0.0 : 1e30;
    if (std::isinf(g)) g = std::copysign(std::ldexp(1.0, 128), g);
    int e = 0;
    std::frexp(ref, &e);  // |ref| = m 2^e, m in [0.5, 1)
    int fe = e - 1;
    if (fe < -126) fe = -126;
    if (fe > 127) fe = 127;
    return std::fabs(g - ref) / std::ldexp(1.0, fe - 23);
}

// ------------------------------------------------------------------ commands

int cmd_canary() {
    int bad = 0;
    const bool core = twm::contraction_canary();
    std::printf("canary raw+f32x1: %s\n", core ? "no contraction" : "CONTRACTION DETECTED");
    bad += !core;
    // lane forms
    volatile float va = 1.0f + 0x1p-12f, vc = -(1.0f + 0x1p-11f);
    float a[16], c[16], r[16];
    for (int i = 0; i < 16; ++i) {
        a[i] = va;
        c[i] = vc;
    }
    auto check = [&](const char *nm, auto tag) {
        using F = decltype(tag);
        for (int i = 0; i < 16; i += F::width) (F::load(a + i) * F::load(a + i) + F::load(c + i)).store(r + i);
        bool ok = true;
        for (int i = 0; i < 16; ++i) ok = ok && r[i] == 0.0f;
        std::printf("canary %s: %s\n", nm, ok ? "no contraction" : "CONTRACTION DETECTED");
        bad += !ok;
    };
#if TW_HAVE_SSE2
    check("f32x4", twm::f32x4());
#endif
#if TW_HAVE_AVX2
    check("f32x8", twm::f32x8());
#endif
#if TW_HAVE_AVX512F
    check("f32x16", twm::f32x16());
#endif
    // auto-vectorizable raw loop (what kernel code looks like)
    for (int i = 0; i < 16; ++i) r[i] = a[i] * a[i] + c[i];
    bool ok = true;
    for (int i = 0; i < 16; ++i) ok = ok && r[i] == 0.0f;
    std::printf("canary raw loop: %s\n", ok ? "no contraction" : "CONTRACTION DETECTED");
    bad += !ok;
    return bad ? 1 : 0;
}

// Fixed input set: specials + random bit patterns (whole range) + random
// patterns inside [0, hi] with random sign. Integer-generated only.
std::vector<float> identity_inputs(std::uint32_t hi_bits, std::uint64_t seed) {
    std::vector<float> v;
    const std::uint32_t specials[] = {
            0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u,
            0x80800000u, 0x3f800000u, 0xbf800000u, 0x3f000000u, 0xbf000000u, 0x7f7fffffu, 0xff7fffffu,
            0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7fa00001u, 0xff812345u, 0x40490fdbu,
            0xc0490fdbu, 0x3fc90fdbu, 0xbfc90fdbu, 0x3f490fdbu, 0x46000000u, 0x46000001u, 0x4b000000u,
            0x4b7fffffu, 0x4f000000u, 0xcf000000u, 0x4effffffu, 0x5f800000u, 0x42b17218u, 0x42b17217u,
            0xc2cff1b5u, 0xc2ce8ed0u, 0x3f3504f3u, 0x3effffffu, 0x3f000001u, 0x3f7fffffu, 0x3f800001u,
    };
    for (std::uint32_t s : specials) v.push_back(bfloat(s));
    Rng rng(seed);
    for (int i = 0; i < (1 << 20); ++i) v.push_back(bfloat(rng.next()));
    for (int i = 0; i < (1 << 20); ++i) {
        const std::uint32_t r = rng.next();
        const std::uint32_t mag = static_cast<std::uint32_t>((static_cast<std::uint64_t>(r) * (hi_bits + 1ull)) >> 32);
        v.push_back(bfloat(mag | (rng.next() & 0x80000000u)));
    }
    while (v.size() % 16) v.push_back(0.0f);
    return v;
}

std::uint32_t domain_hi_bits(const char *name) {
    const std::string n = name;
    if (n == "asin" || n == "acos") return 0x3f800000u;                       // 1
    if (n == "asin_small") return 0x3f000000u;                                // 1/2
    if (n.rfind("sincos_small", 0) == 0) return 0x3f490fdbu;                  // pi/4
    if (n == "exp") return 0x42d00000u;                                       // 104
    if (n.rfind("sin", 0) == 0 || n == "cos" || n.rfind("sincos", 0) == 0) return 0x40c90fdbu;  // 2 pi
    return 0x47800000u;  // 65536
}

struct IdentityResult {
    std::uint64_t hash;
    bool lanes_equal;
};

IdentityResult identity_one(const Entry &e, const std::vector<float> &in) {
    std::vector<float> ref(in.size()), out(in.size());
    e.lanes.fns[0](in.data(), ref.data(), in.size());
    bool eq = true;
    for (int l = 1; l < e.lanes.count; ++l) {
        e.lanes.fns[l](in.data(), out.data(), in.size());
        for (std::size_t i = 0; i < in.size(); ++i) {
            if (fbits(out[i]) != fbits(ref[i])) {
                if (eq)
                    std::printf("  MISMATCH %s lane %s input 0x%08x: 0x%08x vs scalar 0x%08x\n", e.name,
                                e.lanes.names[l], fbits(in[i]), fbits(out[i]), fbits(ref[i]));
                eq = false;
            }
        }
    }
    Hash h;
    for (float f : ref) h.add(fbits(f));
    return {h.h, eq};
}

std::uint64_t atan2_identity(bool &eq) {
    Rng rng(77);
    const std::size_t n = (1u << 20);
    std::vector<float> y(n), x(n), ref(n), out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int mode = static_cast<int>(i & 3u);
        if (mode == 0) {  // whole range
            y[i] = bfloat(rng.next());
            x[i] = bfloat(rng.next());
        } else {  // magnitudes in [2^-8, 2^8], random signs, plus zeros/infs sprinkled
            y[i] = bfloat((0x3b800000u + rng.next() % 0x08000000u) | (rng.next() & 0x80000000u));
            x[i] = bfloat((0x3b800000u + rng.next() % 0x08000000u) | (rng.next() & 0x80000000u));
            const std::uint32_t sp = rng.next() % 64u;
            const float specials[] = {0.0f, -0.0f, INFINITY, -INFINITY};
            if (sp < 4) y[i] = specials[sp];
            if (sp >= 4 && sp < 8) x[i] = specials[sp - 4];
        }
    }
    const LaneSet2 ls = atan2_lanes();
    ls.fns[0](y.data(), x.data(), ref.data(), n);
    eq = true;
    for (int l = 1; l < ls.count; ++l) {
        ls.fns[l](y.data(), x.data(), out.data(), n);
        for (std::size_t i = 0; i < n; ++i) eq = eq && fbits(out[i]) == fbits(ref[i]);
    }
    Hash h;
    for (float f : ref) h.add(fbits(f));
    return h.h;
}

std::uint64_t identity_all(bool verbose, bool &all_eq) {
    Hash total;
    all_eq = true;
    for (const Entry &e : entries()) {
        const std::vector<float> in = identity_inputs(domain_hi_bits(e.name), 1234);
        const IdentityResult r = identity_one(e, in);
        all_eq = all_eq && r.lanes_equal;
        if (verbose)
            std::printf("identity %-18s %016" PRIx64 "  lanes(%d) %s\n", e.name, r.hash, e.lanes.count,
                        r.lanes_equal ? "equal" : "DIFFER");
        total.add(static_cast<std::uint32_t>(r.hash));
        total.add(static_cast<std::uint32_t>(r.hash >> 32));
    }
    bool eq2 = true;
    const std::uint64_t h2 = atan2_identity(eq2);
    all_eq = all_eq && eq2;
    if (verbose) std::printf("identity %-18s %016" PRIx64 "  lanes %s\n", "atan2", h2, eq2 ? "equal" : "DIFFER");
    total.add(static_cast<std::uint32_t>(h2));
    total.add(static_cast<std::uint32_t>(h2 >> 32));
    return total.h;
}

// The hostile caller and the guarded kernel live in different functions.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
std::uint64_t guarded_identity(bool &eq) {
    twm::FpGuard guard;
    return identity_all(false, eq);
}

int cmd_identity() {
    int bad = 0;
    {
        twm::FpGuard guard;
        std::printf("fp mode canonical: %s\n", twm::fp_mode_is_canonical() ? "yes" : "NO");
    }
    bool eq = false;
    std::uint64_t base;
    {
        twm::FpGuard guard;
        base = identity_all(true, eq);
    }
    std::printf("identity TOTAL %016" PRIx64 "  all lane widths equal to scalar: %s\n", base, eq ? "yes" : "NO");
    bad += !eq;
#if TW_FPENV_X86
    // FTZ+DAZ+round-up preset by a hostile caller: the guard must hide it.
    const unsigned saved = twm::fp_mode_get();
    twm::fp_mode_set(0x9FC0u | 0x4000u);  // FTZ | DAZ | RC=up, masked
    bool eq2 = false;
    const std::uint64_t g = guarded_identity(eq2);
    twm::fp_mode_set(0x9FC0u);  // FTZ | DAZ, RN, no guard
    bool eq3 = false;
    const std::uint64_t u = identity_all(false, eq3);
    twm::fp_mode_set(saved);
    std::printf("hostile FTZ/DAZ/round-up + FpGuard: %016" PRIx64 " %s\n", g, g == base ? "same" : "DIFFERENT");
    std::printf("hostile FTZ/DAZ without guard:      %016" PRIx64 " %s (expected to differ)\n", u,
                u == base ? "same" : "different");
    std::printf("fp mode restored: %s\n", (twm::fp_mode_get() & 0xFFC0u) == (saved & 0xFFC0u) ? "yes" : "NO");
    bad += (g != base);
#endif
    return bad ? 1 : 0;
}

int cmd_exhaustive(const char *name) {
    const Entry *e = find_entry(name);
    if (!e) {
        std::fprintf(stderr, "unknown function %s\n", name);
        return 2;
    }
    const std::size_t chunks = 256, per = (std::size_t(1) << 32) / chunks;
    std::vector<std::uint64_t> hashes(chunks);
    std::vector<std::uint64_t> lane_bad(chunks), exact_bad(chunks);
    std::vector<std::uint32_t> first_bad(chunks, 0xffffffffu);
    parallel_chunks(chunks, [&](std::size_t c) {
        const std::size_t block = 4096;
        std::vector<float> in(block), ref(block), out(block);
        Hash h;
        std::uint64_t lb = 0, xb = 0;
        for (std::size_t b = 0; b < per; b += block) {
            const std::uint32_t base = static_cast<std::uint32_t>(c * per + b);
            for (std::size_t i = 0; i < block; ++i) in[i] = bfloat(base + static_cast<std::uint32_t>(i));
            e->lanes.fns[0](in.data(), ref.data(), block);
            for (int l = 1; l < e->lanes.count; ++l) {
                e->lanes.fns[l](in.data(), out.data(), block);
                for (std::size_t i = 0; i < block; ++i)
                    if (fbits(out[i]) != fbits(ref[i])) {
                        if (lb == 0) first_bad[c] = fbits(in[i]);
                        ++lb;
                    }
            }
            for (std::size_t i = 0; i < block; ++i) {
                h.add(fbits(ref[i]));
                if (e->exact && !e->exact_ok(in[i], ref[i])) {
                    if (xb == 0 && lb == 0) first_bad[c] = fbits(in[i]);
                    ++xb;
                }
            }
        }
        hashes[c] = h.h;
        lane_bad[c] = lb;
        exact_bad[c] = xb;
    });
    Hash total;
    std::uint64_t lb = 0, xb = 0;
    std::uint32_t fb = 0xffffffffu;
    for (std::size_t c = 0; c < chunks; ++c) {
        total.add(static_cast<std::uint32_t>(hashes[c]));
        total.add(static_cast<std::uint32_t>(hashes[c] >> 32));
        lb += lane_bad[c];
        xb += exact_bad[c];
        if (fb == 0xffffffffu) fb = first_bad[c];
    }
    std::printf("exhaustive %-18s %016" PRIx64 "  lanes(%d) mismatches %" PRIu64, e->name, total.h,
                e->lanes.count, lb);
    if (e->exact) std::printf("  reference mismatches %" PRIu64, xb);
    if (lb || xb) std::printf("  first at input 0x%08x", fb);
    std::printf("\n");
    return (lb || xb) ? 1 : 0;
}

struct AccChunk {
    double max_err = 0.0;
    std::uint32_t worst = 0;
    std::uint64_t n = 0, gt05 = 0, gt1 = 0, gt15 = 0, gt2 = 0, out_of_unit = 0;
    double sum = 0.0;
};

std::uint32_t order_key(float f) {  // monotone map float -> uint32 (for ranges incl. negatives)
    const std::uint32_t u = fbits(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
float from_order_key(std::uint32_t k) {
    return bfloat((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

int cmd_accuracy(const char *name, float lo, float hi) {
    const Entry *e = find_entry(name);
    if (!e || !e->ref) {
        std::fprintf(stderr, "no reference for %s\n", name);
        return 2;
    }
    const bool trig = std::strncmp(name, "sin", 3) == 0 || std::strcmp(name, "cos") == 0 ||
                      std::strncmp(name, "sincos", 6) == 0;
    const std::uint64_t k0 = order_key(lo), k1 = order_key(hi);
    const std::uint64_t total = k1 - k0 + 1;
    const std::size_t chunks = 512;
    std::vector<AccChunk> res(chunks);
    // widest native lane for speed; equality with scalar is the exhaustive command's job
    const RunFn fn = e->lanes.fns[native_widths() - 1];
    parallel_chunks(chunks, [&](std::size_t c) {
        const std::uint64_t b0 = k0 + total * c / chunks, b1 = k0 + total * (c + 1) / chunks;
        const std::size_t block = 4096;
        std::vector<float> in(block), out(block);
        AccChunk a;
        for (std::uint64_t b = b0; b < b1; b += block) {
            const std::size_t m = static_cast<std::size_t>(std::min<std::uint64_t>(block, b1 - b));
            for (std::size_t i = 0; i < m; ++i) in[i] = from_order_key(static_cast<std::uint32_t>(b + i));
            fn(in.data(), out.data(), m);
            for (std::size_t i = 0; i < m; ++i) {
                const double err = ulp_error(out[i], e->ref(static_cast<double>(in[i])));
                ++a.n;
                a.sum += err;
                if (err > 0.5) ++a.gt05;
                if (err > 1.0) ++a.gt1;
                if (err > 1.5) ++a.gt15;
                if (err > 2.0) ++a.gt2;
                if (trig && !(out[i] >= -1.0f && out[i] <= 1.0f)) ++a.out_of_unit;
                if (err > a.max_err) {
                    a.max_err = err;
                    a.worst = fbits(in[i]);
                }
            }
        }
        res[c] = a;
    });
    AccChunk t;
    for (const AccChunk &a : res) {
        t.n += a.n;
        t.sum += a.sum;
        t.gt05 += a.gt05;
        t.gt1 += a.gt1;
        t.gt15 += a.gt15;
        t.gt2 += a.gt2;
        t.out_of_unit += a.out_of_unit;
        if (a.max_err > t.max_err) {
            t.max_err = a.max_err;
            t.worst = a.worst;
        }
    }
    const float wx = bfloat(t.worst);
    float wout;
    fn(&wx, &wout, 1);
    const long double rl = e->refl(static_cast<long double>(wx));
    const double errl = ulp_error(wout, static_cast<double>(rl));
    std::printf("accuracy %-11s [%.9g, %.9g] n=%" PRIu64 " max %.4f ulp at x=%.9g (0x%08x) got %.9g ref %.12Lg"
                " (long double check %.4f)  mean %.4f  >0.5:%" PRIu64 " >1:%" PRIu64 " >1.5:%" PRIu64 " >2:%" PRIu64,
                e->name, lo, hi, t.n, t.max_err, wx, t.worst, wout, rl, errl, t.sum / static_cast<double>(t.n),
                t.gt05, t.gt1, t.gt15, t.gt2);
    if (trig) std::printf(" outside[-1,1]:%" PRIu64, t.out_of_unit);
    std::printf("\n");
    return 0;
}

// Bitwise equality of two functions on every float in [lo, hi].
int cmd_equal(const char *na, const char *nb, float lo, float hi) {
    const Entry *a = find_entry(na), *b = find_entry(nb);
    if (!a || !b) {
        std::fprintf(stderr, "unknown function\n");
        return 2;
    }
    const std::uint64_t k0 = order_key(lo), k1 = order_key(hi), total = k1 - k0 + 1;
    const std::size_t chunks = 256;
    std::vector<std::uint64_t> diff(chunks);
    std::vector<std::uint32_t> first(chunks, 0);
    parallel_chunks(chunks, [&](std::size_t c) {
        const std::uint64_t b0 = k0 + total * c / chunks, b1 = k0 + total * (c + 1) / chunks;
        const std::size_t block = 4096;
        std::vector<float> in(block), oa(block), ob(block);
        std::uint64_t d = 0;
        for (std::uint64_t q = b0; q < b1; q += block) {
            const std::size_t m = static_cast<std::size_t>(std::min<std::uint64_t>(block, b1 - q));
            for (std::size_t i = 0; i < m; ++i) in[i] = from_order_key(static_cast<std::uint32_t>(q + i));
            a->lanes.fns[0](in.data(), oa.data(), m);
            b->lanes.fns[0](in.data(), ob.data(), m);
            for (std::size_t i = 0; i < m; ++i)
                if (fbits(oa[i]) != fbits(ob[i])) {
                    if (d == 0) first[c] = fbits(in[i]);
                    ++d;
                }
        }
        diff[c] = d;
    });
    std::uint64_t d = 0;
    std::uint32_t f = 0;
    for (std::size_t c = 0; c < chunks; ++c) {
        if (d == 0 && diff[c]) f = first[c];
        d += diff[c];
    }
    std::printf("equal %s == %s on [%.9g, %.9g] (n=%" PRIu64 "): %s", na, nb, lo, hi, total,
                d ? "DIFFERENT" : "bitwise identical");
    if (d) std::printf(" (%" PRIu64 " inputs, first 0x%08x)", d, f);
    std::printf("\n");
    return d ? 1 : 0;
}

int cmd_atan2(std::uint64_t samples) {
    struct A2 {
        double max_err = 0;
        float wy = 0, wx = 0;
        std::uint64_t n = 0, gt1 = 0;
    };
    const std::size_t chunks = 96;
    std::vector<A2> res(chunks);
    parallel_chunks(chunks, [&](std::size_t c) {
        Rng rng(1000 + c);
        A2 a;
        const std::uint64_t per = samples / chunks;
        const std::size_t block = 4096;
        std::vector<float> y(block), x(block), out(block);
        const int mode = static_cast<int>(c % 3);
        for (std::uint64_t done = 0; done < per; done += block) {
            for (std::size_t i = 0; i < block; ++i) {
                if (mode == 0) {  // log-uniform magnitudes 2^-40..2^40, random signs
                    y[i] = bfloat((0x2b800000u + rng.next() % 0x28000000u) | (rng.next() & 0x80000000u));
                    x[i] = bfloat((0x2b800000u + rng.next() % 0x28000000u) | (rng.next() & 0x80000000u));
                } else if (mode == 1) {  // unit vectors (normalised physics directions)
                    const double th = (static_cast<double>(rng.next()) / 4294967296.0) * 6.283185307179586;
                    y[i] = static_cast<float>(std::sin(th));
                    x[i] = static_cast<float>(std::cos(th));
                } else {  // close ratios around the reduction boundaries
                    const float xs = bfloat(0x3f000000u + rng.next() % 0x01000000u);
                    const double kr[] = {0.41421356237309503, 2.414213562373095, 1.0};
                    const double ratio = kr[rng.next() % 3] * (1.0 + (static_cast<double>(rng.next()) / 4294967296.0 - 0.5) * 1e-3);
                    x[i] = (rng.next() & 1) ? -xs : xs;
                    y[i] = static_cast<float>(static_cast<double>(xs) * ratio) * ((rng.next() & 1) ? -1.0f : 1.0f);
                }
            }
            const LaneSet2 ls = atan2_lanes();
            ls.fns[native_widths() - 1](y.data(), x.data(), out.data(), block);
            for (std::size_t i = 0; i < block; ++i) {
                const double err = ulp_error(out[i], std::atan2(static_cast<double>(y[i]), static_cast<double>(x[i])));
                ++a.n;
                if (err > 1.0) ++a.gt1;
                if (err > a.max_err) {
                    a.max_err = err;
                    a.wy = y[i];
                    a.wx = x[i];
                }
            }
        }
        res[c] = a;
    });
    const char *names[] = {"log-uniform pairs", "unit circle", "near tan(pi/8), 1, tan(3pi/8)"};
    for (int m = 0; m < 3; ++m) {
        A2 t;
        for (std::size_t c = m; c < chunks; c += 3) {
            t.n += res[c].n;
            t.gt1 += res[c].gt1;
            if (res[c].max_err > t.max_err) {
                t.max_err = res[c].max_err;
                t.wy = res[c].wy;
                t.wx = res[c].wx;
            }
        }
        std::printf("accuracy atan2 %-30s n=%" PRIu64 " max %.4f ulp at (y=%.9g, x=%.9g)  >1:%" PRIu64 "\n", names[m],
                    t.n, t.max_err, t.wy, t.wx, t.gt1);
    }
    // special cases (IEEE 754 atan2 table)
    struct Sp {
        float y, x, want;
    } sp[] = {
            {0.0f, 1.0f, 0.0f},           {-0.0f, 1.0f, -0.0f},         {0.0f, -1.0f, 3.14159274f},
            {-0.0f, -1.0f, -3.14159274f}, {0.0f, 0.0f, 0.0f},           {-0.0f, 0.0f, -0.0f},
            {0.0f, -0.0f, 3.14159274f},   {-0.0f, -0.0f, -3.14159274f}, {1.0f, 0.0f, 1.57079637f},
            {1.0f, -0.0f, 1.57079637f},   {-1.0f, 0.0f, -1.57079637f},  {INFINITY, 1.0f, 1.57079637f},
            {1.0f, INFINITY, 0.0f},       {1.0f, -INFINITY, 3.14159274f}, {INFINITY, INFINITY, 0.785398185f},
            {INFINITY, -INFINITY, 2.35619449f}, {-INFINITY, -INFINITY, -2.35619449f},
    };
    int bad = 0;
    for (const Sp &s : sp) {
        const float r = twm::atan2(s.y, s.x);
        if (fbits(r) != fbits(s.want)) {
            std::printf("  atan2 special (%g, %g) = %.9g (0x%08x), want %.9g\n", s.y, s.x, r, fbits(r), s.want);
            ++bad;
        }
    }
    std::printf("atan2 IEEE special cases: %s\n", bad ? "FAIL" : "all match");
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s canary|identity|exhaustive <fn>|accuracy <fn> <lo> <hi>|atan2 <n>|equal <a> <b> <lo> <hi>|list\n",
                     argv[0]);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "list") {
        for (const Entry &e : entries()) std::printf("%s%s\n", e.name, e.exact ? " (exact)" : "");
        return 0;
    }
    if (cmd == "canary") return cmd_canary();
    if (cmd == "identity") return cmd_identity();
    if (cmd == "exhaustive" && argc >= 3) return cmd_exhaustive(argv[2]);
    if (cmd == "accuracy" && argc >= 5)
        return cmd_accuracy(argv[2], std::strtof(argv[3], nullptr), std::strtof(argv[4], nullptr));
    if (cmd == "atan2" && argc >= 3) return cmd_atan2(std::strtoull(argv[2], nullptr, 10));
    if (cmd == "equal" && argc >= 6)
        return cmd_equal(argv[2], argv[3], std::strtof(argv[4], nullptr), std::strtof(argv[5], nullptr));
    std::fprintf(stderr, "bad arguments\n");
    return 2;
}
