// Twin StadiumCar force kernel: folded per-car constants and compiled curves.
//
// Kernel-private view of the car (docs/twin/m3-forces.md §2). It is filled
// once per session by a compile step (src/twin/fv/compile_car.cpp from FV's
// CudaHostStaticConfiguration; later from M1's CarPack) and is read-only in
// the substep. Plain POD, no FV dependency, saved byte for byte by the
// recorder (tools/twin/twin_car_record.cpp) so FV-free tools can reload it.
#ifndef TW_MODEL_CAR_FORCE_CONSTS_H
#define TW_MODEL_CAR_FORCE_CONSTS_H

#include <cstdint>

#include "tw_math.h"   // TW_HAVE_* (off in GPU device code)

#if TW_HAVE_SSE2
#include <immintrin.h>
#endif

namespace tw::model {

// ---------------------------------------------------------------------------
// Compiled tuning curve (FV CFuncKeysReal semantics, board 10 §2.20).
//
// FV: x below key0 - 1e-5 -> v0; above key(n-1) + 1e-5 -> v(n-1); otherwise
// the FIRST segment s with x in [L(k_s), U(k_s+1)] (L/U = key -/+ 1e-5,
// rounded through double), value (1-t) v_s + t v_s+1, t = (x - k_s)/span
// (t = 0 when |span| < 1e-5), or v_s for Constant curves.
// Twin: region r = #{j : x > thr[j]} with thr = {prev(L(k0)), U(k1), ...,
// U(k(n-1))}; value = y0[r] + slope[r] * (x - x0[r]). The region choice is
// FV's exactly (same bounds, same first-segment rule, so Constant curves are
// bit-exact); linear values differ by float noise (<= 2 ulp of the value).
struct CompiledCurve {
    static constexpr int kMaxKeys = 15;
    std::uint32_t thresholds = 0u;     // n keys -> n thresholds, n + 1 regions
    std::uint32_t linear = 0u;
    alignas(64) float thr[16] = {};    // n used, the rest +inf (never passed, NaN passes none)
    float x0[kMaxKeys + 1] = {};
    float y0[kMaxKeys + 1] = {};
    float slope[kMaxKeys + 1] = {};
};

// Region = number of padded thresholds below x: an integer, so every ISA path
// gives the same region (AVX-512: one compare + popcount; SSE2: four compares,
// four movemasks and a 4-bit popcount table; else scalar).
TW_INLINE std::uint32_t CurveRegion(const CompiledCurve &c, float x) {
#if TW_HAVE_AVX512F
    const __mmask16 m = _mm512_cmp_ps_mask(_mm512_load_ps(c.thr), _mm512_set1_ps(x), _CMP_LT_OQ);
    return static_cast<std::uint32_t>(__builtin_popcount(static_cast<unsigned>(m)));
#elif TW_HAVE_SSE2
    static constexpr std::uint8_t kBits[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    const __m128 xv = _mm_set1_ps(x);
    return kBits[_mm_movemask_ps(_mm_cmplt_ps(_mm_load_ps(c.thr), xv))] +
           kBits[_mm_movemask_ps(_mm_cmplt_ps(_mm_load_ps(c.thr + 4), xv))] +
           kBits[_mm_movemask_ps(_mm_cmplt_ps(_mm_load_ps(c.thr + 8), xv))] +
           kBits[_mm_movemask_ps(_mm_cmplt_ps(_mm_load_ps(c.thr + 12), xv))];
#else
    std::uint32_t r = 0u;
    for (int j = 0; j < 16; ++j) r += x > c.thr[j] ? 1u : 0u;
    return r;
#endif
}

TW_INLINE float EvalCurve(const CompiledCurve &c, float x) {
    const std::uint32_t r = CurveRegion(c, x);
    return c.y0[r] + c.slope[r] * (x - c.x0[r]);
}

// Speed curves take km/h: FV rounds double(s) * double(3.6f) to float, which
// is the correctly rounded float product s * 3.6f.
TW_INLINE float EvalSpeedCurve(const CompiledCurve &c, float speed) {
    return EvalCurve(c, speed * 3.6f);
}

enum CurveId : std::uint32_t {
    kCurveMaxSideFriction,        // speed
    kCurveSteerDriveTorque,       // speed
    kCurveSteerSlowDown,          // speed (M5 getter: authored interpolation)
    kCurveDamperModulation,       // raw: (a - minAbs) / (maxAbs - minAbs)
    kCurveAirZScale,              // raw |omega.z|
    kCurveAccel,                  // speed (slipResponseAccelFromSpeed)
    kCurveSlippingAccel,          // speed (x slippingAccelScale folded in)
    kCurveRearGearAccel,          // speed
    kCurveRollLatRatio,           // speed-like (burnoutRolloverLateralFromSpeedRatio)
    kCurveBurnoutRadius,          // speed
    kCurveBurnoutLatSpeed,        // raw radius (result / 3.6 at use)
    kCurveDonutRollover,          // speed
    kCurveBurnoutRollover,        // speed
    kCurveSplashVertical,         // raw
    kCurveSplashHorizontal,       // raw
    kCurveWaterFriction,          // speed
    kCurveLatSlowDown,            // speed (M5 getter: authored interpolation)
    kCurveCount,
};

// ---------------------------------------------------------------------------
struct CarMaterial {
    float blend[4];               // x top speed, y accel, z brake cap, w side grip
    float fakePeriodX, fakePeriodZ, fakeSpeedScale, fakeDepthMax;
    std::uint32_t usesFakeTexture;
    float fakeInvPeriodX, fakeInvPeriodZ;   // 1 / period (exact: powers of two; derived)
};

struct WheelConsts {
    float rest[3];                // restSurfacePose.translation (car local)
    float forcePoint[3];          // forceApplicationPoint
    float forceLever[3];          // forcePoint - com (derived)
    float halfSpanSigned;         // (front ? span : -span) * 0.5
    std::uint32_t front;
    std::uint32_t killsLateral;
};

struct BurnoutConsts {
    float reverseForceThreshold, donutSpeedHigh, donutSpeedLow;
    float lateralCorrectionScale, angleTorqueScale, angularDampingLinear;
    float angleReturnQuadratic, tangentAngularDamping, radiusCorrectionScale;
    float radiusCorrectionSpeedScale, radiusMin, tangentSpeedMax, angleLimit;
    float angleLimitPositive, angleLimitNegative;
    std::uint32_t durationTicks, exitDurationTicks;
    float driveFadeScale, sideForceFadeScale, exitAccelFadeScale, exitBonusAccelScale;
    float durationF, duration2F, exitDurationF;   // FromUnsignedInteger of the above
};

struct EngineConsts {
    float rpmMax, burnoutHoldRise, airRise, airFall, groundBrake, groundRise;
    float transitionRise, groundFall, fwdHigh, fwdLow, revHigh, revLow;
    float ratio[6], upshift[6], downshift[6];
};

struct CarForceConsts {
    static constexpr std::uint32_t kMagic = 0x43465754u;  // "TWFC"
    static constexpr std::uint32_t kVersion = 5u;
    std::uint32_t magic = kMagic;
    std::uint32_t version = kVersion;
    std::uint32_t size = sizeof(CarForceConsts);

    // body
    float mass, invMass;
    float inertiaInvBody[3];      // diagonal of I^-1 (body frame)
    float com[3];
    float maxAngularSpeed;        // <= 0: no limit
    float gravity[3];             // sum of uniform force fields (world)
    std::uint32_t gravityYOnly;   // gravity = (0, gy, 0) (derived)
    float halfSpanFront, halfSpanRear;   // +span/2, -span/2 (derived from the wheels)
    float invAssistFullSpeed;            // 1 / assistFullSpeed (derived)
    std::uint32_t steerSmall;            // wheelSpeedBase >= 2, scale >= 0: asin arg <= 1/2, |delta| <= pi/6 (derived)
    float slippingAccelScale;            // slipResponse.slippingAccelScale (1 for StadiumCar)
    // fake-contact bump impulse in closed form (n = +y, p = rest point; derived):
    // J = (0, (1 - e) s / bumpDenom, 0), d omega_car = bumpW * J.y
    float bumpDenom[4];                  // 1/m + n . ((I^-1 (r x n)) x r) = 1/m + Dz r.x^2 + Dx r.z^2
    float bumpW[4][3];                   // D (r x (0, 1, 0)) * angularImpulseScale, y scaled by YScale
    float spare[13];                     // room for derived constants without a layout change
    // material rows by natural surface id (derived; lane lookups of the batch kernel)
    alignas(64) float byIdBlend[4][32];
    alignas(64) float byIdFakePeriodX[32], byIdFakePeriodZ[32], byIdFakeInvX[32], byIdFakeInvZ[32];
    alignas(64) float byIdFakeSpeed[32], byIdFakeDepth[32], byIdFakeUses[32];
    float zoneLinDamp, zoneAngDamp, angularDampingScale;
    float fsGrounded, fsAirborne, ldsAirborne;   // groundedSolidFeedback1, airborne1, airborne0
    float linearSpeedCap, reverseGearSpeedThreshold;
    float waterBoxCenter[3], waterBoxHalf[3];
    std::uint32_t frozenByWaterBox;  // waterBoxLocal.halfExtents.x < 0 -> C1 always

    WheelConsts wheel[4];
    std::int8_t wheelOfShape[8];          // collision tree shape -> board wheel (FV CudaVehicleCollisionShape::wheelIndex), -1 body
    alignas(16) float wLeverX[4];          // forceLever x / z per wheel, lane form (derived)
    alignas(16) float wLeverZ[4];
    alignas(16) std::uint32_t wFrontMask[4];   // ~0u for front wheels (derived)

    // suspension (wheel force mode 2)
    float restAbsorb, springK, damperC, followCoef, minAbs, maxAbs;
    float damperModRange;         // maxAbs - minAbs (0 -> modulation input 0)

    // contact response (fake contacts and, later, M5)
    float wheelImpulseMetal, wheelImpulseOther, bodyImpulseMetal, bodyImpulseOther;
    float tangentLimitMetal, tangentLimitOther;
    float angularImpulseScale, angularImpulseYScale, growthLimitSq;

    // steering, visual
    float slewRate, assistFullSpeed, slowDownScale, wheelSpeedBase, wheelSpeedScale;

    // body air response
    float slope1Min, slope1Max, slope2Min, slope2Max;
    float airTorqueLin, airTorqueQuad, airYSwitch;
    std::uint32_t airMemWindow;

    // coasting, lateral slowdown
    float lowSpeedFrictionMagnitude, lowSpeedLinearDamping;
    std::uint32_t lateralSlowDownWindow;

    // geared drive
    float engineSpeedNorm, reverseSpeedNorm;
    float accelBase, accelSpeedCoef, capSlipping, cap, capSlippingReverse, capReverse;
    float speedLimitForce, forceZScale, sideToDriveTorque, slippingSteerTorqueScale;
    float lateralForceScale, slippingSideFrictionScale, sideFrictionSlipBlend;
    float driveSideFrictionSlipBlend, slipRatioScale, currentForceTorqueMin;
    float currentTorqueXScale, currentTorqueZScale, perSlippingWheelAccelScale;
    float lowSpeedBSlippingGripScale;
    float dirtSideScale, dirtGateScale, dirtFwdScale, dirtFwdGateScale;
    float forceDivisor, invForceDivisor;
    BurnoutConsts burnout;
    EngineConsts engine;

    // turbo
    float turboScaleA, turboScaleB;
    std::uint32_t turboDurationA, turboDurationB;

    // water
    float buoyancy, splashHSpeed, splashTotalSpeed, waterAngLinDamp, waterAngSpeedDamp;

    // materials
    std::uint32_t materialCount;
    std::uint8_t remap[32];       // natural surface id -> material row
    CarMaterial material[32];
    std::uint8_t fakeTextureRed[128 * 128];   // row-major (z index * 128 + x index)

    CompiledCurve curve[kCurveCount];
};

// Derived fields and curve padding; idempotent (compile and loaders call it).
inline void FinalizeCarForceConsts(CarForceConsts *k) {
    for (int w = 0; w < 4; ++w)
        for (int i = 0; i < 3; ++i) k->wheel[w].forceLever[i] = k->wheel[w].forcePoint[i] - k->com[i];
    k->gravityYOnly = (k->gravity[0] == 0.0f && k->gravity[2] == 0.0f) ? 1u : 0u;
    for (int w = 0; w < 4; ++w) {
        k->wLeverX[w] = k->wheel[w].forceLever[0];
        k->wLeverZ[w] = k->wheel[w].forceLever[2];
        k->wFrontMask[w] = k->wheel[w].front ? ~0u : 0u;
    }
    k->invAssistFullSpeed = 1.0f / k->assistFullSpeed;
    k->steerSmall = (k->wheelSpeedBase >= 2.0f && k->wheelSpeedScale >= 0.0f) ? 1u : 0u;
    for (auto &m : k->material) {
        m.fakeInvPeriodX = m.fakePeriodX != 0.0f ? 1.0f / m.fakePeriodX : 0.0f;
        m.fakeInvPeriodZ = m.fakePeriodZ != 0.0f ? 1.0f / m.fakePeriodZ : 0.0f;
    }
    for (int id = 0; id < 32; ++id) {
        const CarMaterial &m = k->material[k->remap[id]];
        for (int i = 0; i < 4; ++i) k->byIdBlend[i][id] = m.blend[i];
        k->byIdFakePeriodX[id] = m.fakePeriodX;
        k->byIdFakePeriodZ[id] = m.fakePeriodZ;
        k->byIdFakeInvX[id] = m.fakeInvPeriodX;
        k->byIdFakeInvZ[id] = m.fakeInvPeriodZ;
        k->byIdFakeSpeed[id] = m.fakeSpeedScale;
        k->byIdFakeDepth[id] = m.fakeDepthMax;
        k->byIdFakeUses[id] = m.usesFakeTexture ? 1.0f : 0.0f;
    }
    for (int w = 0; w < 4; ++w) {
        const float rx = k->wheel[w].rest[0] - k->com[0], rz = k->wheel[w].rest[2] - k->com[2];
        k->bumpDenom[w] = k->invMass + (k->inertiaInvBody[2] * (rx * rx) + k->inertiaInvBody[0] * (rz * rz));
        k->bumpW[w][0] = k->inertiaInvBody[0] * -rz * k->angularImpulseScale;
        k->bumpW[w][1] = 0.0f;
        k->bumpW[w][2] = k->inertiaInvBody[2] * rx * k->angularImpulseScale;
    }
    for (int w = 3; w >= 0; --w) (k->wheel[w].front ? k->halfSpanFront : k->halfSpanRear) = k->wheel[w].halfSpanSigned;
    for (auto &c : k->curve)
        for (std::uint32_t j = c.thresholds; j < 16u; ++j) c.thr[j] = __builtin_inff();
}

// Map water zone (world data, board 20 owns it; FV CSceneVehicleWaterZone).
struct WaterZone {
    std::uint32_t enabled = 0u;
    float cellSize[2] = {1.0f, 1.0f};
    float origin[2] = {0.0f, 0.0f};
    std::uint32_t dims[2] = {0u, 0u};
    std::uint8_t outsideValue = 0u;
    float surfaceHeight = -1000000.0f;
    float secondaryCullHeight = 1000000.0f;
    const std::uint8_t *cells = nullptr;   // dims[0] * dims[1], row-major in y, plus >= 3 bytes of padding

    TW_INLINE std::uint8_t ValueAt(float sx, float sz, bool *inside) const {
        const std::uint32_t cx = tw::math::to_u32_trunc_wrap((sx - origin[0]) / cellSize[0]);
        const std::uint32_t cy = tw::math::to_u32_trunc_wrap((sz - origin[1]) / cellSize[1]);
        *inside = cx < dims[0] && cy < dims[1];
        return *inside ? cells[cx + dims[0] * cy] : outsideValue;
    }
    // CSceneVehicleWaterZone::AcceptsRegion (collision_zone.cpp:82-101); Water = 1.
    TW_INLINE bool AcceptsRegion(float sx, float sz, float lowerY, float upperY) const {
        if (!enabled) return false;
        bool inside;
        const std::uint8_t v = ValueAt(sx, sz, &inside);
        if (!inside && outsideValue == 1u && surfaceHeight > lowerY) return true;
        if (!(upperY > secondaryCullHeight) || !(surfaceHeight > lowerY)) return false;
        return v == 1u;
    }
};

}  // namespace tw::model

#endif
