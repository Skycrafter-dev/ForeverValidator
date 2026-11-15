// Compiled tuning curve (design.md §3.3, board 10 §2.20).
//
// Same results as FV's CFuncKeysReal::GetValue as Model 6 calls it, by
// construction rather than by approximation:
// - Every Model-6 call site starts the key search at index 0
//   (scene_vehicle_car_tuning.cpp:155, scene_vehicle_car_feedback.cpp:427), so
//   FV picks the FIRST segment whose tolerant bounds [x_i - 1e-5, x_{i+1} + 1e-5]
//   contain x. Here that index is the number of interior keys whose upper bound
//   is below x: a compare count, no state, identical at every lane width.
// - The tolerant bounds are precomputed with FV's rounding (double +- 1e-5f,
//   rounded once to float, func_keys_real.cpp:13-23).
// - The blend is FV's: b = |span| >= 1e-5 ? (x - x0) / span : 0, then
//   (1 - b) * v0 + b * v1, or v0 for Constant interpolation.
// - Speed curves take km/h: x = speed * 3.6f in float, which equals FV's
//   FromDouble(double(speed) * double(3.6f)) bit for bit because the exact
//   product of two binary32 values fits in a double.
#ifndef TW_CURVE_H
#define TW_CURVE_H

#include <cstdint>

namespace tw {
namespace model {

constexpr int kTwCurveMaxKeys = 16;

struct TwCurve {
    std::uint32_t count = 0u;           // 0 = absent curve: evaluates to 0 (FV: empty key list)
    std::uint32_t linear = 1u;          // 1 Linear, 0 Constant (authored interpolation)
    float x[kTwCurveMaxKeys] = {};      // key positions as authored (km/h for speed curves)
    float v[kTwCurveMaxKeys] = {};      // key values
    float lower[kTwCurveMaxKeys] = {};  // FV LowerKeyBound(x_i)
    float upper[kTwCurveMaxKeys] = {};  // FV UpperKeyBound(x_i)
    float span[kTwCurveMaxKeys] = {};   // x_{i+1} - x_i (float, as FV computes it)
};

// Builds the precomputed bounds and spans. Positions must be nondecreasing.
inline bool TwCurveCompile(TwCurve *c, std::uint32_t count, const float *positions, const float *values, bool linear) {
    if (count > static_cast<std::uint32_t>(kTwCurveMaxKeys)) return false;
    c->count = count;
    c->linear = linear ? 1u : 0u;
    for (std::uint32_t i = 0; i < count; ++i) {
        c->x[i] = positions[i];
        c->v[i] = values[i];
        c->lower[i] = static_cast<float>(static_cast<double>(positions[i]) - static_cast<double>(1.0e-5f));
        c->upper[i] = static_cast<float>(static_cast<double>(positions[i]) + static_cast<double>(1.0e-5f));
        if (i > 0 && positions[i] < positions[i - 1]) return false;
    }
    for (std::uint32_t i = 0; i + 1 < count; ++i) c->span[i] = c->x[i + 1] - c->x[i];
    return true;
}

// Raw input (FV EvaluateCurve). NaN input: FV's search wraps through all keys and
// stops on key 1, so a Constant curve returns v[1] and a Linear one NaN; mirrored.
inline float TwCurveEval(const TwCurve &c, float x) {
    const std::uint32_t n = c.count;
    if (n == 0u) return 0.0f;
    if (n == 1u || x < c.lower[0]) return c.v[0];
    if (c.linear == 0u && !(x == x)) return c.v[1];
    if (x > c.upper[n - 1u]) return c.v[n - 1u];
    std::uint32_t i = 0u;
    for (std::uint32_t k = 1u; k + 1u < n; ++k) i += (x > c.upper[k]) ? 1u : 0u;
    if (c.linear == 0u) return c.v[i];
    const float s = c.span[i];
    const float b = (s >= 1.0e-5f || s <= -1.0e-5f) ? (x - c.x[i]) / s : 0.0f;
    return (1.0f - b) * c.v[i] + b * c.v[i + 1u];
}

// Speed input in m/s (FV EvaluateSpeedCurve).
inline float TwCurveEvalSpeed(const TwCurve &c, float speed) { return TwCurveEval(c, speed * 3.6f); }

}  // namespace model
}  // namespace tw

#endif
