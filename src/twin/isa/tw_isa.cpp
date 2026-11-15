// Twin ISA dispatch: variant selection (tw_isa.h).
#include "tw_isa.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

extern "C" const tw::isa::TickTable *tw_isa_tick_table_baseline();
#if TW_ISA_HAVE_AVX2
extern "C" const tw::isa::TickTable *tw_isa_tick_table_avx2();
#endif
#if TW_ISA_HAVE_AVX512
extern "C" const tw::isa::TickTable *tw_isa_tick_table_avx512();
#endif

namespace tw {
namespace isa {

namespace {

std::atomic<int> g_override{-1};  // -1: none (environment / CPUID), else a Kind

int ParseName(const char *name) {
    if (name == nullptr || *name == '\0' || std::strcmp(name, "auto") == 0) return -1;
    if (std::strcmp(name, "baseline") == 0 || std::strcmp(name, "sse2") == 0) return 0;
    if (std::strcmp(name, "avx2") == 0) return 1;
    if (std::strcmp(name, "avx512") == 0) return 2;
    return -2;  // unknown
}

}  // namespace

bool Layout::operator==(const Layout &o) const {
    return abi == o.abi && std::memcmp(words, o.words, sizeof(words)) == 0;
}

const char *KindName(Kind kind) {
    switch (kind) {
    case Kind::Baseline: return "baseline";
    case Kind::Avx2: return "avx2";
    case Kind::Avx512: return "avx512";
    }
    return "unknown";
}

const TickTable *Table(Kind kind) {
    switch (kind) {
    case Kind::Baseline: return tw_isa_tick_table_baseline();
    case Kind::Avx2:
#if TW_ISA_HAVE_AVX2
        return tw_isa_tick_table_avx2();
#else
        return nullptr;
#endif
    case Kind::Avx512:
#if TW_ISA_HAVE_AVX512
        return tw_isa_tick_table_avx512();
#else
        return nullptr;
#endif
    }
    return nullptr;
}

// The ISA level names of __builtin_cpu_supports (GCC 12, Clang 18); older compilers check the features only.
#if (defined(__clang__) && __clang_major__ >= 18) || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 12)
#define TW_CPU_LEVEL(name) (__builtin_cpu_supports(name) != 0)
#else
#define TW_CPU_LEVEL(name) true
#endif

bool Supported(Kind kind) {
// (Windows: no dispatch, the baseline only; __builtin_cpu_supports would need compiler-rt there)
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && !defined(_WIN32)
    switch (kind) {
    // The variants are built for x86-64-v3 / -v4 (FOREVERVALIDATOR_TWIN_MARCH) or for AVX2 + FMA /
    // AVX-512F + VL + FMA; the levels cover both.
    case Kind::Baseline: return true;
    case Kind::Avx2:
        return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0 &&
               __builtin_cpu_supports("bmi") != 0 && __builtin_cpu_supports("bmi2") != 0 &&
               TW_CPU_LEVEL("x86-64-v3");
    case Kind::Avx512:
        return Supported(Kind::Avx2) && __builtin_cpu_supports("avx512f") != 0 &&
               __builtin_cpu_supports("avx512vl") != 0 && __builtin_cpu_supports("avx512bw") != 0 &&
               __builtin_cpu_supports("avx512dq") != 0 && __builtin_cpu_supports("avx512cd") != 0 &&
               TW_CPU_LEVEL("x86-64-v4");
    }
    return false;
#else
    return kind == Kind::Baseline;
#endif
}

void SetOverride(const char *name) {
    const int k = ParseName(name);
    g_override.store(k >= 0 ? k : -1);
}

Kind Select(const char **why) {
    auto usable = [](Kind k) {
        const TickTable *t = Table(k);
        return t != nullptr && Supported(k) && t->layout() == Table(Kind::Baseline)->layout();
    };
    int requested = g_override.load();
    if (requested < 0) {
        const int env = ParseName(std::getenv("FOREVERVALIDATOR_TWIN_ISA"));
        if (env == -2) {
            if (why) *why = "FOREVERVALIDATOR_TWIN_ISA is not baseline|sse2|avx2|avx512|auto; using CPUID";
        } else {
            requested = env;
        }
    }
    if (requested >= 0) {
        const Kind k = static_cast<Kind>(requested);
        if (usable(k)) {
            if (why) *why = "override";
            return k;
        }
        if (why) *why = "override not usable here (variant not built, CPU lacks it, or layout differs); using CPUID";
    }
    for (Kind k : {Kind::Avx512, Kind::Avx2}) {
        if (usable(k)) {
            if (why && requested < 0) *why = "cpuid";
            return k;
        }
    }
    if (why && requested < 0) *why = "cpuid (no wider variant usable)";
    return Kind::Baseline;
}

}  // namespace isa
}  // namespace tw
