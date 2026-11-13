// Twin FV adapter: CarForceConsts from CudaHostStaticConfiguration.
#include "compile_car.h"

#include <cmath>
#include <cstring>

#include "simulation/backends/cuda/cuda_static_configuration.h"

namespace tw::fv {

using forevervalidator::simulation::CudaHostStaticConfiguration;
using forevervalidator::simulation::CudaTuningCurveId;
using forevervalidator::simulation::CudaTransmissionArrayId;
using tw::model::CarForceConsts;
using tw::model::CompiledCurve;

namespace {

// FV func_keys_real.cpp:14-22: key -/+ 1e-5 evaluated in double, rounded to float.
float LowerBound(float key) { return static_cast<float>(static_cast<double>(key) - static_cast<double>(1.0e-5f)); }
float UpperBound(float key) { return static_cast<float>(static_cast<double>(key) + static_cast<double>(1.0e-5f)); }

bool CompileCurve(const CudaHostStaticConfiguration &cfg, CudaTuningCurveId id, float scale, CompiledCurve *out,
                  std::string *diag, const char *name) {
    *out = CompiledCurve{};
    const auto &d = cfg.tuning.curves[static_cast<std::size_t>(id)];
    const std::uint32_t n = d.keyCount;
    if (n == 0u) return true;                 // empty curve evaluates to 0
    if (n > static_cast<std::uint32_t>(CompiledCurve::kMaxKeys) || d.firstKey + n > cfg.curveKeys.size()) {
        *diag = std::string("curve ") + name + ": too many keys";
        return false;
    }
    const auto *keys = cfg.curveKeys.data() + d.firstKey;
    for (std::uint32_t i = 1; i < n; ++i) {
        if (!(keys[i - 1].position <= keys[i].position)) {
            *diag = std::string("curve ") + name + ": keys not sorted";
            return false;
        }
    }
    const bool linear = d.interpolation == 1u;   // ReplayTuningCurveInterpolation: 0 Constant, 1 Linear
    out->thresholds = n;
    out->linear = linear ? 1u : 0u;
    out->thr[0] = std::nextafter(LowerBound(keys[0].position), -INFINITY);   // x >= L(k0)  <=>  x > prev(L(k0))
    for (std::uint32_t j = 1; j < n; ++j) out->thr[j] = UpperBound(keys[j].position);
    out->x0[0] = keys[0].position;
    out->y0[0] = keys[0].value * scale;
    out->slope[0] = 0.0f;
    for (std::uint32_t r = 1; r < n; ++r) {
        const auto &a = keys[r - 1];
        const auto &b = keys[r];
        const float span = b.position - a.position;
        out->x0[r] = a.position;
        out->y0[r] = a.value * scale;
        out->slope[r] = (linear && std::fabs(span) >= 1.0e-5f) ? ((b.value - a.value) / span) * scale : 0.0f;
    }
    out->x0[n] = keys[n - 1].position;
    out->y0[n] = keys[n - 1].value * scale;
    out->slope[n] = 0.0f;
    return true;
}

bool IsPow2(float v) {
    if (!(v >= 1.0f)) return false;
    int e = 0;
    return std::frexp(v, &e) == 0.5f;
}

}  // namespace

bool CompileCarForceConsts(const CudaHostStaticConfiguration &cfg, CarForceConsts *out, std::string *diag) {
    std::string local;
    if (diag == nullptr) diag = &local;
    CarForceConsts k{};
    const auto &t = cfg.tuning;
    const auto &gd = t.gearedDrive;
    const auto &dp = cfg.dynaParameters;
    if (t.handlingModel != 5u) { *diag = "handling model is not GearedDrive (5)"; return false; }
    if (t.wheelForceMode != 2u) { *diag = "wheel force mode is not FollowAbsorbWithImpulse (2)"; return false; }

    k.mass = dp.mass;
    k.invMass = 1.0f / dp.mass;
    const auto &I = dp.inverseBodyInertia;
    if (I.basisX.y != 0.0f || I.basisX.z != 0.0f || I.basisY.x != 0.0f || I.basisY.z != 0.0f || I.basisZ.x != 0.0f ||
        I.basisZ.y != 0.0f) {
        *diag = "body inverse inertia is not diagonal";
        return false;
    }
    k.inertiaInvBody[0] = I.basisX.x;
    k.inertiaInvBody[1] = I.basisY.y;
    k.inertiaInvBody[2] = I.basisZ.z;
    k.com[0] = dp.localCenterOfMass.x;
    k.com[1] = dp.localCenterOfMass.y;
    k.com[2] = dp.localCenterOfMass.z;
    const float maxW = t.contactResponse.pointImpulseAngularSpeedMax;
    k.maxAngularSpeed = maxW > 1.0e-5f ? maxW : 0.0f;
    k.gravity[0] = k.gravity[1] = k.gravity[2] = 0.0f;
    for (const auto &f : cfg.forceFields) {
        if (f.type != forevervalidator::simulation::CudaForceFieldType::Uniform) {
            *diag = "Ball force field (position dependent) not modelled";
            return false;
        }
        k.gravity[0] += f.vector.x;
        k.gravity[1] += f.vector.y;
        k.gravity[2] += f.vector.z;
    }
    k.zoneLinDamp = cfg.zoneLinearDampingCoefficient;
    k.zoneAngDamp = cfg.zoneAngularDampingCoefficient;
    k.angularDampingScale = dp.angularDampingScale;
    const auto &air = t.bodyAirResponse;
    k.fsGrounded = air.groundedSolidFeedback1;
    k.fsAirborne = air.airborneSolidFeedback1;
    k.ldsAirborne = air.airborneSolidFeedback0;
    const auto &ip = cfg.initialParameters;
    k.linearSpeedCap = ip.linearSpeedCap;
    k.reverseGearSpeedThreshold = ip.reverseGearSpeedThreshold;
    k.waterBoxCenter[0] = ip.waterBoxLocal.center.x;
    k.waterBoxCenter[1] = ip.waterBoxLocal.center.y;
    k.waterBoxCenter[2] = ip.waterBoxLocal.center.z;
    k.waterBoxHalf[0] = ip.waterBoxLocal.halfExtents.x;
    k.waterBoxHalf[1] = ip.waterBoxLocal.halfExtents.y;
    k.waterBoxHalf[2] = ip.waterBoxLocal.halfExtents.z;
    k.frozenByWaterBox = ip.waterBoxLocal.halfExtents.x < 0.0f ? 1u : 0u;

    // wheels; span = 2 * halfExtent.z of the rest points (wheels.cpp:810-857)
    float zmin = 0.0f, zmax = 0.0f;
    for (int w = 0; w < 4; ++w) {
        const auto &d = cfg.wheels.wheels[static_cast<std::size_t>(w)];
        const auto &p = d.restSurfacePose.translation;
        k.wheel[w].rest[0] = p.x;
        k.wheel[w].rest[1] = p.y;
        k.wheel[w].rest[2] = p.z;
        k.wheel[w].forcePoint[0] = d.forceApplicationPoint.x;
        k.wheel[w].forcePoint[1] = d.forceApplicationPoint.y;
        k.wheel[w].forcePoint[2] = d.forceApplicationPoint.z;
        k.wheel[w].front = d.axle == VehicleWheelAxle::Front ? 1u : 0u;
        k.wheel[w].killsLateral = d.killsLateralSpeedOnContact ? 1u : 0u;
        if (w == 0) {
            zmin = zmax = p.z;
        } else {
            if (p.z < zmin) zmin = p.z;
            if (zmax < p.z) zmax = p.z;
        }
    }
    if (cfg.collisionShapes.size() != 8u) { *diag = "car collision model is not 8 shapes"; return false; }
    for (int i = 0; i < 8; ++i) k.wheelOfShape[i] = -2;
    for (std::size_t i = 0; i < 8u; ++i) {   // indexed by tree (traversal) order
        const auto &sh = cfg.collisionShapes[i];
        const std::uint32_t wi = sh.wheelIndex;
        if ((wi != UINT32_MAX && wi > 3u) || sh.traversalOrder >= 8u || k.wheelOfShape[sh.traversalOrder] != -2) {
            *diag = "collision shapes: bad wheel index or traversal order";
            return false;
        }
        k.wheelOfShape[sh.traversalOrder] = wi == UINT32_MAX ? std::int8_t{-1} : static_cast<std::int8_t>(wi);
    }
    const float halfZ = (zmax - zmin) * 0.5f;   // GmBoxAligned::SetMinMax half extent
    const float span = halfZ + halfZ;
    for (int w = 0; w < 4; ++w) k.wheel[w].halfSpanSigned = (k.wheel[w].front ? span : -span) * 0.5f;

    const auto &su = t.suspension;
    k.restAbsorb = su.wheelRestDamperAbsorb;
    k.springK = su.wheelSpringCoef;
    k.damperC = su.wheelDamperCoef;
    k.followCoef = su.wheelAbsorbFollowCoef;
    k.minAbs = su.damperModulationMinAbsorb;
    k.maxAbs = su.damperModulationMaxAbsorb;
    k.damperModRange = (su.damperModulationMinAbsorb == su.damperModulationMaxAbsorb)
                               ? 0.0f
                               : su.damperModulationMaxAbsorb - su.damperModulationMinAbsorb;

    const auto &cr = t.contactResponse;
    k.wheelImpulseMetal = cr.wheelContactImpulseMetal;
    k.wheelImpulseOther = cr.wheelContactImpulseOther;
    k.bodyImpulseMetal = cr.bodyContactImpulseMetal;
    k.bodyImpulseOther = cr.bodyContactImpulseOther;
    k.tangentLimitMetal = cr.bodyContactTangentLimitMetal;
    k.tangentLimitOther = cr.bodyContactTangentLimitOther;
    k.angularImpulseScale = cr.pointImpulseAngularScale;
    k.angularImpulseYScale = cr.pointImpulseAngularYScale;
    k.growthLimitSq = cr.pointImpulseLinearSpeedGrowthLimitSq;

    k.slewRate = t.steering.slewRate;
    k.assistFullSpeed = t.steering.assistFullSpeed;
    k.slowDownScale = t.steering.slowDownScale;
    k.wheelSpeedBase = t.visual.wheelSpeedBase;
    k.wheelSpeedScale = t.visual.wheelSpeedScale;

    k.slope1Min = air.slopeAdherence1Min;
    k.slope1Max = air.slopeAdherence1Max;
    k.slope2Min = air.slopeAdherence2Min;
    k.slope2Max = air.slopeAdherence2Max;
    k.airTorqueLin = air.airTorqueLinearCoef;
    k.airTorqueQuad = air.airTorqueQuadraticCoef;
    k.airYSwitch = air.airControlYSwitchThreshold;
    k.airMemWindow = air.airControlMemoryTickWindow;

    k.slippingAccelScale = t.slipResponse.slippingAccelScale;
    k.lowSpeedFrictionMagnitude = t.lowSpeedFrictionMagnitude;
    k.lowSpeedLinearDamping = t.lowSpeedLinearDamping;
    k.lateralSlowDownWindow = t.slipResponse.lateralSlowDownTickWindow;

    k.engineSpeedNorm = t.engineSpeedNorm;
    k.reverseSpeedNorm = gd.reverseSpeedNorm;
    k.accelBase = gd.forwardAccelBase;
    k.accelSpeedCoef = gd.forwardAccelSpeedCoef;
    k.capSlipping = gd.forwardAccelCapWhenSlipping;
    k.cap = gd.forwardAccelCap;
    k.capSlippingReverse = gd.forwardAccelCapWhenSlippingReverse;
    k.capReverse = gd.forwardAccelCapReverse;
    k.speedLimitForce = gd.speedLimitForce;
    k.forceZScale = gd.forceZScale;
    k.sideToDriveTorque = gd.sideForceToDriveTorqueScale;
    k.slippingSteerTorqueScale = gd.slippingSteerTorqueScale;
    k.lateralForceScale = gd.lateralForceScale;
    k.slippingSideFrictionScale = gd.slippingSideFrictionScale;
    k.sideFrictionSlipBlend = gd.sideFrictionSlipBlend;
    k.driveSideFrictionSlipBlend = gd.driveSideFrictionSlipBlend;
    k.slipRatioScale = gd.slipRatioScale;
    k.currentForceTorqueMin = gd.currentForceTorqueMin;
    k.currentTorqueXScale = gd.currentTorqueXScale;
    k.currentTorqueZScale = gd.currentTorqueZScale;
    k.perSlippingWheelAccelScale = gd.perSlippingWheelAccelScale;
    k.lowSpeedBSlippingGripScale = gd.lowSpeedBSlippingGripScale;
    k.dirtSideScale = gd.dirtSlideSideForceScale;
    k.dirtGateScale = gd.dirtSlideGateScale;
    k.dirtFwdScale = gd.dirtSlideForwardForceScale;
    k.dirtFwdGateScale = gd.dirtSlideForwardGateScale;
    k.forceDivisor = t.feedback.forceDivisor;
    k.invForceDivisor = 1.0f / t.feedback.forceDivisor;

    const auto &b = gd.burnout;
    auto &kb = k.burnout;
    kb.reverseForceThreshold = b.reverseForceThreshold;
    kb.donutSpeedHigh = b.donutSpeedHigh;
    kb.donutSpeedLow = b.donutSpeedLow;
    kb.lateralCorrectionScale = b.lateralCorrectionScale;
    kb.angleTorqueScale = b.angleTorqueScale;
    kb.angularDampingLinear = b.angularDampingLinear;
    kb.angleReturnQuadratic = b.angleReturnQuadratic;
    kb.tangentAngularDamping = b.tangentAngularDamping;
    kb.radiusCorrectionScale = b.radiusCorrectionScale;
    kb.radiusCorrectionSpeedScale = b.radiusCorrectionSpeedScale;
    kb.radiusMin = b.radiusMin;
    kb.tangentSpeedMax = b.tangentSpeedMax;
    kb.angleLimit = b.angleLimit;
    kb.angleLimitPositive = b.angleLimitPositive;
    kb.angleLimitNegative = b.angleLimitNegative;
    kb.durationTicks = b.durationTicks;
    kb.exitDurationTicks = b.exitDurationTicks;
    kb.driveFadeScale = b.driveFadeScale;
    kb.sideForceFadeScale = b.sideForceFadeScale;
    kb.exitAccelFadeScale = b.exitAccelFadeScale;
    kb.exitBonusAccelScale = b.exitBonusAccelScale;
    kb.durationF = static_cast<float>(b.durationTicks);
    kb.duration2F = static_cast<float>(b.durationTicks * 2u);
    kb.exitDurationF = static_cast<float>(b.exitDurationTicks);
    if (b.exitDurationTicks == 0u || b.durationTicks == 0u) { *diag = "zero burnout duration"; return false; }

    const auto &in = gd.input;
    auto &ke = k.engine;
    ke.rpmMax = in.engineInputMaximum;
    ke.burnoutHoldRise = in.burnoutHoldInputRise;
    ke.airRise = in.airborneInputRise;
    ke.airFall = in.airborneInputFall;
    ke.groundBrake = in.groundInputBrake;
    ke.groundRise = in.groundInputRise;
    ke.transitionRise = in.transitionInputRise;
    ke.groundFall = in.groundInputFall;
    ke.fwdHigh = in.forwardTransitionSpeedHigh;
    ke.fwdLow = in.forwardTransitionSpeedLow;
    ke.revHigh = in.reverseTransitionSpeedHigh;
    ke.revLow = in.reverseTransitionSpeedLow;
    auto array = [&](CudaTransmissionArrayId id, float *dst, const char *name) {
        const auto &d = cfg.transmissionArrays[static_cast<std::size_t>(id)];
        if (d.valueCount < 6u || d.firstValue + d.valueCount > cfg.transmissionValues.size()) {
            *diag = std::string("transmission array ") + name + " has fewer than 6 values";
            return false;
        }
        for (int i = 0; i < 6; ++i) dst[i] = cfg.transmissionValues[d.firstValue + static_cast<std::uint32_t>(i)];
        return true;
    };
    if (!array(CudaTransmissionArrayId::GearSpeedRatio, ke.ratio, "gearSpeedRatio") ||
        !array(CudaTransmissionArrayId::UpshiftThreshold, ke.upshift, "upshift") ||
        !array(CudaTransmissionArrayId::DownshiftThreshold, ke.downshift, "downshift"))
        return false;

    k.turboScaleA = t.turbo.impulseScaleA;
    k.turboScaleB = t.turbo.impulseScaleB;
    k.turboDurationA = t.turbo.durationA;
    k.turboDurationB = t.turbo.durationB;

    k.buoyancy = t.water.buoyancyForce;
    k.splashHSpeed = t.water.splashHorizontalSpeedThreshold;
    k.splashTotalSpeed = t.water.splashTotalSpeedThreshold;
    k.waterAngLinDamp = t.water.angularLinearDamping;
    k.waterAngSpeedDamp = t.water.angularSpeedDamping;

    // materials
    if (cfg.materials.empty() || cfg.materials.size() > 32u) { *diag = "material table size outside 1..32"; return false; }
    k.materialCount = static_cast<std::uint32_t>(cfg.materials.size());
    for (std::size_t i = 0; i < cfg.materials.size(); ++i) {
        const auto &m = cfg.materials[i];
        auto &o = k.material[i];
        o.blend[0] = m.blendableValues.x;
        o.blend[1] = m.blendableValues.y;
        o.blend[2] = m.blendableValues.z;
        o.blend[3] = m.blendableValues.w;
        o.fakePeriodX = m.fakeContactPeriodX;
        o.fakePeriodZ = m.fakeContactPeriodZ;
        o.fakeSpeedScale = m.fakeContactSpeedScale;
        o.fakeDepthMax = m.fakeContactDepthMax;
        o.usesFakeTexture = (m.usesFakeContactTexture && !cfg.fakeContactTextureRgb.empty()) ? 1u : 0u;
        if (o.usesFakeTexture && (!IsPow2(o.fakePeriodX) || !IsPow2(o.fakePeriodZ))) {
            *diag = "fake-contact period is not a power of two (fmod_pow2)";
            return false;
        }
    }
    for (int id = 0; id < 32; ++id) {
        std::uint32_t idx = 0u;
        if (static_cast<std::size_t>(id) < cfg.materialIndexByNaturalId.size()) idx = cfg.materialIndexByNaturalId[id];
        if (idx >= k.materialCount) {
            if (static_cast<std::size_t>(id) < cfg.materialIndexByNaturalId.size() && id < 31) {
                *diag = "material remap points outside the table";
                return false;
            }
            idx = 0u;
        }
        k.remap[id] = static_cast<std::uint8_t>(idx);
    }
    if (!cfg.fakeContactTextureRgb.empty()) {
        if (cfg.fakeContactTextureRgb.size() != 128u * 128u * 3u) { *diag = "fake-contact texture is not 128x128 RGB"; return false; }
        for (std::size_t p = 0; p < 128u * 128u; ++p) k.fakeTextureRed[p] = cfg.fakeContactTextureRgb[3u * p];
    }

    struct CurveMap {
        tw::model::CurveId twin;
        CudaTuningCurveId fv;
        float scale;
        const char *name;
    };
    const CurveMap curves[] = {
            {tw::model::kCurveMaxSideFriction, CudaTuningCurveId::MaxSideFrictionFromSpeed, 1.0f, "maxSideFriction"},
            {tw::model::kCurveSteerDriveTorque, CudaTuningCurveId::SteeringDriveTorqueFromSpeed, 1.0f, "steerDriveTorque"},
            {tw::model::kCurveSteerSlowDown, CudaTuningCurveId::SteerSlowDownFromSpeed, 1.0f, "steerSlowDown"},
            {tw::model::kCurveDamperModulation, CudaTuningCurveId::SuspensionDamperAbsorbModulation, 1.0f, "damperModulation"},
            {tw::model::kCurveAirZScale, CudaTuningCurveId::AirControlZScale, 1.0f, "airControlZScale"},
            {tw::model::kCurveAccel, CudaTuningCurveId::SlipResponseAccelFromSpeed, 1.0f, "accel"},
            {tw::model::kCurveSlippingAccel, CudaTuningCurveId::SlipResponseSlippingAccelFromSpeed, 1.0f, "slippingAccel"},
            {tw::model::kCurveRearGearAccel, CudaTuningCurveId::ReverseGearAccelFromSpeed, 1.0f, "rearGearAccel"},
            {tw::model::kCurveRollLatRatio, CudaTuningCurveId::BurnoutRolloverLateralFromSpeedRatio, 1.0f, "rolloverLateralRatio"},
            {tw::model::kCurveBurnoutRadius, CudaTuningCurveId::BurnoutRadiusFromSpeed, 1.0f, "burnoutRadius"},
            {tw::model::kCurveBurnoutLatSpeed, CudaTuningCurveId::BurnoutLateralSpeedFromRadius, 1.0f, "burnoutLateralSpeed"},
            {tw::model::kCurveDonutRollover, CudaTuningCurveId::DonutRolloverFromSpeed, 1.0f, "donutRollover"},
            {tw::model::kCurveBurnoutRollover, CudaTuningCurveId::BurnoutRolloverFromSpeed, 1.0f, "burnoutRollover"},
            {tw::model::kCurveSplashVertical, CudaTuningCurveId::SplashVerticalImpulse, 1.0f, "splashVertical"},
            {tw::model::kCurveSplashHorizontal, CudaTuningCurveId::SplashHorizontalImpulse, 1.0f, "splashHorizontal"},
            {tw::model::kCurveWaterFriction, CudaTuningCurveId::WaterFrictionFromSpeed, 1.0f, "waterFriction"},
            {tw::model::kCurveLatSlowDown, CudaTuningCurveId::LateralContactSlowDownFromSpeed, 1.0f, "lateralSlowDown"},
    };
    for (const auto &c : curves)
        if (!CompileCurve(cfg, c.fv, c.scale, &k.curve[c.twin], diag, c.name)) return false;

    tw::model::FinalizeCarForceConsts(&k);
    *out = k;
    return true;
}

bool CompileWaterZone(const CudaHostStaticConfiguration &cfg, tw::model::WaterZone *out, std::vector<std::uint8_t> *cells,
                      std::string *diag) {
    *out = tw::model::WaterZone{};
    cells->clear();
    if (!cfg.water.present) return true;      // no water: zone disabled
    const auto &g = cfg.water;
    const std::size_t count = static_cast<std::size_t>(g.dimensions.x) * g.dimensions.y;
    if (cfg.waterOccupancy.size() != count) {
        if (diag) *diag = "water occupancy size mismatch";
        return false;
    }
    // CSceneVehicleWaterZone::Configure (collision_zone.cpp:35-80)
    const bool declaresPlaneMap = !cfg.waterPlaneIndices.empty() || !cfg.waterPlanes.empty() || g.outsidePlaneIndex != 0u;
    bool hasPlaneMap = declaresPlaneMap && cfg.waterPlaneIndices.size() == count && g.outsidePlaneIndex <= cfg.waterPlanes.size();
    if (hasPlaneMap)
        for (std::uint8_t v : cfg.waterPlaneIndices) hasPlaneMap = hasPlaneMap && v <= cfg.waterPlanes.size();
    if (declaresPlaneMap && !hasPlaneMap) return true;   // FV disables the zone
    *cells = hasPlaneMap ? cfg.waterPlaneIndices : cfg.waterOccupancy;
    cells->resize(count + 4u, 0u);   // the batch kernel's byte gather reads up to 3 bytes past a cell
    out->enabled = 1u;
    out->cellSize[0] = g.cellSize.x;
    out->cellSize[1] = g.cellSize.y;
    out->origin[0] = g.origin.x;
    out->origin[1] = g.origin.y;
    out->dims[0] = g.dimensions.x;
    out->dims[1] = g.dimensions.y;
    out->outsideValue = static_cast<std::uint8_t>(hasPlaneMap ? g.outsidePlaneIndex : g.outsideOccupancy);
    out->surfaceHeight = g.surfaceHeight;
    out->secondaryCullHeight = g.secondaryCullHeight;
    out->cells = cells->data();
    return true;
}

}  // namespace tw::fv
