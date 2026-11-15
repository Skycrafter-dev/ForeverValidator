// M1 curve check against FV (links forevervalidator_core): every StadiumCar curve,
// 10k+ inputs each, TwCurveEval vs FV CFuncKeysReal::GetValue as Model 6 calls it
// (key index 0, scene_vehicle_car_tuning.cpp:153-165), raw and km/h speed forms.
// Usage: tw_curve_fvcheck src/twin/model/stadiumcar-values.json
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "engine/core/func_keys_real.h"
#include "../tw_compile_car.h"

using namespace tw::model;

static std::uint32_t Bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
static std::int64_t Ulps(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return 0;
    std::int64_t ia = static_cast<std::int32_t>(Bits(a)), ib = static_cast<std::int32_t>(Bits(b));
    if (ia < 0) ia = static_cast<std::int64_t>(0x80000000LL) - ia;
    if (ib < 0) ib = static_cast<std::int64_t>(0x80000000LL) - ib;
    return ia > ib ? ia - ib : ib - ia;
}

static const char *kNames[] = {
#define TW_CURVE(name, json) json,
#include "../curve_ids.inc"
#undef TW_CURVE
};

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: tw_curve_fvcheck JSON\n"); return 2; }
    static TwCarPack pack;
    std::string err;
    if (!CompileCarFromJson(argv[1], &pack, &err)) { std::fprintf(stderr, "compile: %s\n", err.c_str()); return 1; }
    std::mt19937 rng(12345u);
    std::int64_t worstAll = 0;
    long total = 0, exact = 0, relFail = 0;
    for (std::uint32_t c = 0; c < kTwCurveCount; ++c) {
        const TwCurve &tc = pack.curve[c];
        CFuncKeysReal fv;
        for (std::uint32_t k = 0; k < tc.count; ++k) fv.AddKeyReal(tc.x[k], tc.v[k]);
        // stadiumcar-values.json: interpolation 0 = Constant, 1 = Linear (FV enum: Linear 0, Constant 1)
        fv.SetInterpolation(tc.linear ? CFuncKeysReal::Linear : CFuncKeysReal::Constant);
        const float lo = tc.count ? tc.x[0] : 0.0f, hi = tc.count ? tc.x[tc.count - 1] : 1.0f;
        const float range = (hi - lo) > 1.0f ? hi - lo : 1.0f;
        std::uniform_real_distribution<float> uni(lo - 0.25f * range - 1.0f, hi + 0.25f * range + 1.0f);
        std::vector<float> xs;
        for (int i = 0; i < 10000; ++i) xs.push_back(uni(rng));
        auto around = [&](float c0) {   // c0 and its 6 float neighbours on each side (sign-correct)
            float lo2 = c0, hi2 = c0;
            xs.push_back(c0);
            for (int d = 0; d < 6; ++d) {
                lo2 = std::nextafter(lo2, -INFINITY); hi2 = std::nextafter(hi2, INFINITY);
                xs.push_back(lo2); xs.push_back(hi2);
            }
        };
        for (std::uint32_t k = 0; k < tc.count; ++k) { around(tc.x[k]); around(tc.lower[k]); around(tc.upper[k]); }
        xs.push_back(NAN); xs.push_back(-NAN); xs.push_back(INFINITY); xs.push_back(-INFINITY); xs.push_back(0.0f); xs.push_back(-0.0f);
        std::int64_t worst = 0;
        long n = 0, ex = 0, rf = 0;
        for (float x : xs) {
            // raw form (EvaluateCurve)
            unsigned long key = 0ul;
            float fvRaw = 0.0f;
            fv.GetValue(x, fvRaw, key);
            const float twRaw = TwCurveEval(tc, x);
            // speed form (EvaluateSpeedCurve): x is a speed in m/s
            unsigned long key2 = 0ul;
            float fvSpd = 0.0f;
            fv.GetValue(static_cast<float>(static_cast<double>(x / 3.6f) * static_cast<double>(3.6f)), fvSpd, key2);
            const float twSpd = TwCurveEvalSpeed(tc, x / 3.6f);
            for (int form = 0; form < 2; ++form) {
                const float a = form ? twSpd : twRaw, b = form ? fvSpd : fvRaw;
                const std::int64_t u = Ulps(a, b);
                ++n;
                ex += u == 0;
                if (u > worst) worst = u;
                const float tol = 1e-6f * std::fabs(b);
                if (u > 2 && std::fabs(a - b) > tol) {
                    ++rf;
                    if (rf <= 4) std::printf("    %s x=%.9g (0x%08x) form=%s tw=%.9g fv=%.9g\n", kNames[c], x, Bits(x), form ? "speed" : "raw", a, b);
                }
            }
        }
        std::printf("%-40s keys %2u %s: %6ld evals, exact %6ld, worst %lld ulp, fails(>2 ulp and >1e-6 rel) %ld\n", kNames[c], tc.count,
                    tc.linear ? "lin" : "con", n, ex, static_cast<long long>(worst), rf);
        total += n; exact += ex; relFail += rf;
        if (worst > worstAll) worstAll = worst;
    }
    std::printf("ALL: %ld evals, %ld exact, worst %lld ulp, %ld failures -> %s\n", total, exact, static_cast<long long>(worstAll), relFail,
                relFail == 0 ? "PASS" : "FAIL");
    return relFail == 0 ? 0 : 1;
}
