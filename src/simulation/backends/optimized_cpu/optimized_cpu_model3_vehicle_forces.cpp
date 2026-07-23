#include "simulation/backends/optimized_cpu/optimized_cpu_model3_vehicle_forces.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <typeinfo>

#include "engine/core/binary32_math.h"
#include "engine/core/func_keys_real.h"
#include "engine/core/mw_cmd_buffer_core.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_force_field.h"
#include "engine/physics/geometry/physics_tolerances.h"
#include "engine/physics/world/hms_zone.h"
#include "engine/scene/scene_vehicle_car_internal.h"

#if defined(__GNUC__) || defined(__clang__)
#define FV_E019_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define FV_E019_ALWAYS_INLINE inline
#endif

using forevervalidator::simulation::OptimizedCpuBinary32FromDoubleX86Sse2;
using forevervalidator::simulation::OptimizedCpuBinary32MathPath;
using forevervalidator::simulation::OptimizedCpuBinary32SqrtX86Sse2;
using namespace SceneVehicleCarDynamics;

namespace {

constexpr float Model3CurveKeyEpsilon = 1.0e-5f;

template<bool NativeBinary32>
FV_E019_ALWAYS_INLINE float Model3FromDouble(double value) noexcept {
    if constexpr (NativeBinary32) {
        return OptimizedCpuBinary32FromDoubleX86Sse2(value);
    }
    return Binary32::FromDouble(value);
}

template<bool NativeBinary32>
FV_E019_ALWAYS_INLINE float Model3Sqrt(float value) noexcept {
    if constexpr (NativeBinary32) {
        return OptimizedCpuBinary32SqrtX86Sse2(value);
    }
    return CIsqrt(value);
}

struct Model3ReducedAngle {
    double value;
    unsigned quadrant;
};

FV_E019_ALWAYS_INLINE Model3ReducedAngle Model3ReduceSmallAngle(
        float input) noexcept {
    constexpr double TwoOverPi =
            0.636619772367581343075535053490057448;
    constexpr double HalfPiHigh = 1.570796326734125614166;
    constexpr double HalfPiLow = 6.07710050650619224932e-11;
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

FV_E019_ALWAYS_INLINE double Model3SinPolynomial(double value) noexcept {
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

FV_E019_ALWAYS_INLINE double Model3CosPolynomial(double value) noexcept {
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

template<bool NativeBinary32>
FV_E019_ALWAYS_INLINE float Model3Sin(float input) noexcept {
    if constexpr (!NativeBinary32) {
        return CIsin(input);
    }
    if (!std::isfinite(input) || std::fabs(input) > 1000000.0f) {
        return CIsin(input);
    }
    const Model3ReducedAngle reduced = Model3ReduceSmallAngle(input);
    double result;
    switch (reduced.quadrant) {
    case 0u:
        result = Model3SinPolynomial(reduced.value);
        break;
    case 1u:
        result = Model3CosPolynomial(reduced.value);
        break;
    case 2u:
        result = -Model3SinPolynomial(reduced.value);
        break;
    default:
        result = -Model3CosPolynomial(reduced.value);
        break;
    }
    return Model3FromDouble<true>(result);
}

template<bool NativeBinary32>
FV_E019_ALWAYS_INLINE float Model3Cos(float input) noexcept {
    if constexpr (!NativeBinary32) {
        return CIcos(input);
    }
    if (!std::isfinite(input) || std::fabs(input) > 1000000.0f) {
        return CIcos(input);
    }
    const Model3ReducedAngle reduced = Model3ReduceSmallAngle(input);
    double result;
    switch (reduced.quadrant) {
    case 0u:
        result = Model3CosPolynomial(reduced.value);
        break;
    case 1u:
        result = -Model3SinPolynomial(reduced.value);
        break;
    case 2u:
        result = -Model3CosPolynomial(reduced.value);
        break;
    default:
        result = Model3SinPolynomial(reduced.value);
        break;
    }
    return Model3FromDouble<true>(result);
}

FV_E019_ALWAYS_INLINE double Model3AtanUnit(double value) noexcept {
    constexpr double QuarterPi =
            0.785398163397448309615660845819875721;
    const bool aroundOne = value > 0.4142135623730950488;
    const double reduced = aroundOne
            ? (value - 1.0) / (value + 1.0)
            : value;
    const double square = reduced * reduced;
    double power = reduced;
    double result = reduced;
    for (unsigned termIndex = 1u; termIndex <= 24u; ++termIndex) {
        power *= -square;
        result += power / static_cast<double>(termIndex * 2u + 1u);
    }
    return aroundOne ? QuarterPi + result : result;
}

FV_E019_ALWAYS_INLINE double Model3Atan2(double y, double x) noexcept {
    constexpr double Pi =
            3.14159265358979323846264338327950288;
    constexpr double HalfPi =
            1.57079632679489661923132169163975144;
    const bool yNegative = std::signbit(y);
    const bool xNegative = std::signbit(x);
    const double absY = std::fabs(y);
    const double absX = std::fabs(x);
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
            ? Model3AtanUnit(absY / absX)
            : HalfPi - Model3AtanUnit(absX / absY);
    if (xNegative) {
        angle = Pi - angle;
    }
    return yNegative ? -angle : angle;
}

template<bool NativeBinary32>
FV_E019_ALWAYS_INLINE float Model3Asin(float value) noexcept {
    if constexpr (!NativeBinary32) {
        return CIasin(value);
    }
    if (std::isnan(value) || value < -1.0f || value > 1.0f) {
        return CIasin(value);
    }
    const float positiveFactor = 1.0f + value;
    const float negativeFactor = 1.0f - value;
    const float radical = Model3Sqrt<true>(
            positiveFactor * negativeFactor);
    return Model3FromDouble<true>(
            Model3Atan2(
                    static_cast<double>(value),
                    static_cast<double>(radical)));
}

FV_E019_ALWAYS_INLINE float Model3LengthSquared(const GmVec3 &value) noexcept {
    const float xy = value.x * value.x + value.y * value.y;
    return xy + value.z * value.z;
}

template<bool NativeBinary32>
FV_E019_ALWAYS_INLINE GmVec3 Model3NormalizeOr(
        const GmVec3 &value,
        const GmVec3 &shortVectorResult,
        float minimumLengthSquared) noexcept {
    const float lengthSquared = Model3LengthSquared(value);
    if (!(lengthSquared > minimumLengthSquared)) {
        return shortVectorResult;
    }
    const float scale = 1.0f / Model3Sqrt<NativeBinary32>(lengthSquared);
    return {
            value.x * scale,
            value.y * scale,
            value.z * scale,
    };
}

}  // namespace

struct OptimizedCpuModel3VehicleForceAccess {
    static CSceneVehicleCarTuning *ActiveTuning(
            CSceneVehicleCar &car) noexcept {
        return car.ActiveTuningOrNull();
    }

    static bool HasRequiredModel3Configuration(
            const CSceneVehicleCarTuning &tuning) noexcept {
        return tuning.handlingModel == CSceneVehicleCarHandlingModel_Lateral &&
               tuning.maxSideFrictionFromSpeedCurve.IsBound() &&
               tuning.rolloverLateralFromSpeedCurve.IsBound() &&
               tuning.rolloverLateralCoefFromAngleCurve.IsBound() &&
               tuning.slipResponse.accelFromSpeedCurve.IsBound() &&
               tuning.steering.driveTorqueFromSpeedCurve.IsBound() &&
               tuning.steering.slowDownFromSpeedCurve.IsBound();
    }

    static bool HasStableEligibility(
            CSceneVehicleCar &car,
            CHmsItem *item,
            CSceneVehicleCarTuning *tuning,
            OptimizedCpuBinary32MathPath mathPath) noexcept {
        if (mathPath != OptimizedCpuBinary32MathPath::X86Sse2 || item == nullptr ||
            tuning == nullptr || typeid(car) != typeid(CSceneVehicleCar) ||
            typeid(*tuning) != typeid(CSceneVehicleCarTuning) ||
            car.HmsItem() != item) {
            return false;
        }
        return HasRequiredModel3Configuration(*tuning);
    }

    static FV_E019_ALWAYS_INLINE GmVec3 LocalDirectionToWorld(
            const CHmsDyna &dyna,
            const GmVec3 &local) noexcept {
        const GmMat3 &rotation = dyna.CurrentState().rotation;
        return {
                rotation.Element(GmAxis::X, GmAxis::Y) * local.y +
                        rotation.Element(GmAxis::X, GmAxis::X) * local.x +
                        rotation.Element(GmAxis::X, GmAxis::Z) * local.z,
                rotation.Element(GmAxis::Y, GmAxis::X) * local.x +
                        rotation.Element(GmAxis::Y, GmAxis::Y) * local.y +
                        rotation.Element(GmAxis::Y, GmAxis::Z) * local.z,
                rotation.Element(GmAxis::Z, GmAxis::X) * local.x +
                        rotation.Element(GmAxis::Z, GmAxis::Y) * local.y +
                        rotation.Element(GmAxis::Z, GmAxis::Z) * local.z,
        };
    }

    static FV_E019_ALWAYS_INLINE void AddVehicleCentralForce(
            CSceneVehicleCar &car,
            CHmsDyna &dyna,
            const GmVec3 &localForce) noexcept {
        const GmVec3 force = LocalDirectionToWorld(dyna, localForce);
        CHmsDyna::CHmsStateDyna &state = dyna.CurrentState();
        state.force.x = state.force.x + force.x;
        state.force.y = force.y + state.force.y;
        state.force.z = force.z + state.force.z;
        car.forceAccumulators.force.x =
                car.forceAccumulators.force.x + localForce.x;
        car.forceAccumulators.force.y =
                localForce.y + car.forceAccumulators.force.y;
        car.forceAccumulators.force.z =
                localForce.z + car.forceAccumulators.force.z;
    }

    static FV_E019_ALWAYS_INLINE void AddVehicleTorque(
            CHmsDyna &dyna,
            const GmVec3 &localTorque) noexcept {
        const GmVec3 torque = LocalDirectionToWorld(dyna, localTorque);
        CHmsDyna::CHmsStateDyna &state = dyna.CurrentState();
        state.torque.x = state.torque.x + torque.x;
        state.torque.y = torque.y + state.torque.y;
        state.torque.z = torque.z + state.torque.z;
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float LowerKeyBound(float key) noexcept {
        return Model3FromDouble<NativeBinary32>(
                static_cast<double>(key) -
                static_cast<double>(Model3CurveKeyEpsilon));
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float UpperKeyBound(float key) noexcept {
        return Model3FromDouble<NativeBinary32>(
                static_cast<double>(key) +
                static_cast<double>(Model3CurveKeyEpsilon));
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE bool IsWithinKeyBounds(
            float x,
            float lower,
            float upper) noexcept {
        const float lowerBound = LowerKeyBound<NativeBinary32>(lower);
        const float upperBound = UpperKeyBound<NativeBinary32>(upper);
        return !std::isnan(x) && !std::isnan(lowerBound) &&
               !std::isnan(upperBound) && x >= lowerBound && x <= upperBound;
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void GetBoundingIndices(
            const CFuncKeysReal &curve,
            float x,
            unsigned long &keyIndex,
            unsigned long &nextKeyIndex) noexcept {
        const unsigned long count =
                static_cast<unsigned long>(curve.keyPositions.size());
        if (count == 0u) {
            keyIndex = InvalidEngineIndex;
            nextKeyIndex = InvalidEngineIndex;
            return;
        }
        if (count == 1u ||
            x < LowerKeyBound<NativeBinary32>(curve.keyPositions.front())) {
            keyIndex = 0u;
            nextKeyIndex = 0u;
            return;
        }
        if (x > UpperKeyBound<NativeBinary32>(curve.keyPositions.back())) {
            keyIndex = count - 1u;
            nextKeyIndex = count - 1u;
            return;
        }

        unsigned long current = keyIndex < count ? keyIndex : 0ul;
        for (unsigned long scanned = 0ul; scanned <= count; ++scanned) {
            const unsigned long next =
                    current + 1ul < count ? current + 1ul : 0ul;
            if (IsWithinKeyBounds<NativeBinary32>(
                        x,
                        curve.keyPositions[current],
                        curve.keyPositions[next])) {
                keyIndex = current;
                nextKeyIndex = next;
                return;
            }
            current = next;
        }
        keyIndex = current;
        nextKeyIndex = current + 1u < count ? current + 1u : 0u;
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE bool ComputeBlendCoefficient(
            const CFuncKeysReal &curve,
            float x,
            unsigned long &keyIndex,
            unsigned long &nextKeyIndex,
            float &blendCoefficient) noexcept {
        GetBoundingIndices<NativeBinary32>(
                curve, x, keyIndex, nextKeyIndex);
        if (keyIndex == InvalidEngineIndex) {
            return false;
        }
        if (keyIndex == nextKeyIndex) {
            blendCoefficient = 0.0f;
            return true;
        }

        const float x0 = curve.keyPositions[keyIndex];
        const float span = curve.keyPositions[nextKeyIndex] - x0;
        blendCoefficient = std::fabs(span) >= Model3CurveKeyEpsilon
                ? (x - x0) / span
                : 0.0f;
        return true;
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float EvaluateCurve(
            const CFuncKeysReal &curve,
            float x) noexcept {
        unsigned long keyIndex = 0ul;
        unsigned long nextKeyIndex = keyIndex + 1ul;
        float blendCoefficient = 0.0f;
        if (!ComputeBlendCoefficient<NativeBinary32>(
                    curve,
                    x,
                    keyIndex,
                    nextKeyIndex,
                    blendCoefficient)) {
            return 0.0f;
        }

        if (curve.interpolationMode == CFuncKeysReal::Constant) {
            return curve.values[keyIndex];
        }
        const float value0 = curve.values[keyIndex];
        const float value1 = curve.values[nextKeyIndex];
        return (1.0f - blendCoefficient) * value0 +
               blendCoefficient * value1;
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float EvaluateSpeedCurve(
            const CSceneVehicleTuningCurve &curve,
            float speed) noexcept {
        const float kilometersPerHour = Model3FromDouble<NativeBinary32>(
                static_cast<double>(speed) * static_cast<double>(3.6f));
        return EvaluateCurve<NativeBinary32>(curve.Value(), kilometersPerHour);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float EvaluateLinearSpeedCurve(
            const CSceneVehicleTuningCurve &curve,
            float speed) noexcept {
        CFuncKeysReal &values = curve.Value();
        values.interpolationMode = CFuncKeysReal::Constant;
        return EvaluateSpeedCurve<NativeBinary32>(curve, speed);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void IntegrateWheelState(
            CSceneVehicleCar::SSimulationWheel::SRealTimeState &wheel,
            float dt) {
        if constexpr (!NativeBinary32) {
            wheel.Integrate(dt);
            return;
        }

        const float spinInput =
                wheel.wheelAngularSpeed * dt + wheel.wheelSpinAngle;
        wheel.wheelSpinAngle = GmFunc::Mod(
                spinInput,
                0.0f,
                SceneVehicleMath::WheelSpinAnglePeriod);

        const float normalLenSq =
                (wheel.accumulatedContactNormal.y *
                         wheel.accumulatedContactNormal.y +
                 wheel.accumulatedContactNormal.x *
                         wheel.accumulatedContactNormal.x) +
                wheel.accumulatedContactNormal.z *
                        wheel.accumulatedContactNormal.z;
        if (normalLenSq > VectorEpsilonSquared) {
            const float len = Model3Sqrt<true>(normalLenSq);
            const float invLen = 1.0f / len;
            wheel.accumulatedContactNormal.x =
                    wheel.accumulatedContactNormal.x * invLen;
            wheel.accumulatedContactNormal.y =
                    wheel.accumulatedContactNormal.y * invLen;
            wheel.accumulatedContactNormal.z =
                    invLen * wheel.accumulatedContactNormal.z;

            const GmVec3 directionOfView = {
                    0.0f,
                    -wheel.accumulatedContactNormal.z,
                    wheel.accumulatedContactNormal.y,
            };
            const GmVec3 sideSeed = GmMath::Cross(
                    wheel.accumulatedContactNormal,
                    directionOfView);
            const GmVec3 side = Model3NormalizeOr<true>(
                    sideSeed,
                    sideSeed,
                    PhysicsTolerance::SurfaceDirectionLengthSquared);
            const GmVec3 normalizedUp = Model3NormalizeOr<true>(
                    wheel.accumulatedContactNormal,
                    wheel.accumulatedContactNormal,
                    PhysicsTolerance::SurfaceDirectionLengthSquared);
            wheel.contactFrame.basisX = side;
            wheel.contactFrame.basisY = normalizedUp;
            wheel.contactFrame.basisZ = GmMath::Cross(side, normalizedUp);
        }

        const float current = wheel.currentVisualSteerAngle;
        const float target = wheel.targetVisualSteerAngle;
        if (target > current) {
            const float next = current + dt;
            wheel.currentVisualSteerAngle = next;
            if (next > target) {
                wheel.currentVisualSteerAngle = target;
            }
        } else {
            const float next = current - dt;
            wheel.currentVisualSteerAngle = next;
            if (target > next) {
                wheel.currentVisualSteerAngle = target;
            }
        }
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void UpdateWheelSpeed(
            CSceneVehicleCar &car,
            CSceneVehicleCar::SSimulationWheel &wheel,
            float vehicleForwardSpeed,
            float dt) {
        if constexpr (!NativeBinary32) {
            car.WheelUpdateSpeedFromVehicleSpeed(
                    wheel, vehicleForwardSpeed, dt);
            return;
        }

        if (wheel.realTimeState.contactPresent) {
            if (car.gearedDrive.wheelSpeedOverrideActive != 0 &&
                car.gearedDrive.wheelDriveSpeedInhibited == 0) {
                CSceneVehicleCarTuning *tuning =
                        car.Tunings()->ActiveTuning();
                wheel.realTimeState.wheelAngularSpeed =
                        tuning->gearedDrive.burnout.
                                wheelAngularSpeedOverride;
                return;
            }
            wheel.realTimeState.wheelAngularSpeed =
                    vehicleForwardSpeed / wheel.rollingRadius;
            return;
        }

        float targetAngularSpeed = 0.0f;
        float angularAcceleration = 0.0f;
        if (car.controls.lowSpeedGateB > ScalarEpsilon) {
            targetAngularSpeed = 1.0f - car.controls.lowSpeedGateB;
            if (targetAngularSpeed <= 0.0f) {
                targetAngularSpeed = 0.0f;
            } else if (targetAngularSpeed >= 1.0f) {
                targetAngularSpeed = 1.0f;
            }
            angularAcceleration = WheelAngularAccelNegative;
        } else if (car.controls.lowSpeedGateA > ScalarEpsilon &&
                   car.gearedDrive.wheelDriveSpeedInhibited == 0 &&
                   car.controls.forcedLowSpeedFriction == 0) {
            targetAngularSpeed = Model3FromDouble<true>(
                    static_cast<double>(car.controls.lowSpeedGateA) *
                    WheelLowSpeedGateASpeedScale);
            angularAcceleration = WheelAngularAccelPositive;
        } else {
            wheel.realTimeState.wheelAngularSpeed =
                    Model3FromDouble<true>(
                            static_cast<double>(
                                    wheel.realTimeState.wheelAngularSpeed) *
                            WheelNoContactDamping);
        }

        const float absAngularAcceleration =
                std::fabs(angularAcceleration);
        if (!(absAngularAcceleration < ScalarEpsilon)) {
            const float nextAngularSpeed =
                    angularAcceleration * dt +
                    wheel.realTimeState.wheelAngularSpeed;
            wheel.realTimeState.wheelAngularSpeed = nextAngularSpeed;
            if (angularAcceleration > 0.0f &&
                nextAngularSpeed > targetAngularSpeed) {
                wheel.realTimeState.wheelAngularSpeed = targetAngularSpeed;
            } else if (angularAcceleration < 0.0f &&
                       nextAngularSpeed < targetAngularSpeed) {
                wheel.realTimeState.wheelAngularSpeed = targetAngularSpeed;
            }
        }
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void UpdateWheelVisualState(
            CSceneVehicleCar &car,
            CSceneVehicleCar::SSimulationWheel &wheel,
            CSceneVehicleCarTuning *tuning,
            float vehicleForwardSpeed,
            float dt,
            float visualSpeedDenominator) {
        if constexpr (!NativeBinary32) {
            car.UpdateWheelVisualState(
                    wheel,
                    tuning,
                    vehicleForwardSpeed,
                    dt,
                    visualSpeedDenominator);
            return;
        }

        wheel.realTimeState.visualRotation.Set(
                wheel.surfaceHandler.RestRotation());
        float wheelVisualSteerAngle = 0.0f;
        if (IsFrontVehicleWheel(wheel.axle)) {
            float yaw = 0.0f;
            if (!(visualSpeedDenominator < 1.0e-5f)) {
                yaw = -car.controls.currentSteering /
                      visualSpeedDenominator;
            }
            if constexpr (NativeBinary32) {
                const float sine = Model3Sin<true>(yaw);
                const float cosine = Model3Cos<true>(yaw);
                const GmVec3 oldX =
                        wheel.realTimeState.visualRotation.Row(GmAxis::X);
                const GmVec3 oldZ =
                        wheel.realTimeState.visualRotation.Row(GmAxis::Z);
                wheel.realTimeState.visualRotation.SetRow(
                        GmAxis::X,
                        GmMath::Add(
                                GmMath::Scale(oldX, cosine),
                                GmMath::Scale(oldZ, sine)));
                wheel.realTimeState.visualRotation.SetRow(
                        GmAxis::Z,
                        GmMath::Add(
                                GmMath::Scale(oldX, -sine),
                                GmMath::Scale(oldZ, cosine)));
            } else {
                wheel.realTimeState.visualRotation.RotateY(yaw);
            }

            float maxSteerDegrees = WheelVisualDefaultMaxSteerDegrees;
            if (tuning->visual.wheelSteerAngleFromSpeedCurve.IsBound()) {
                const float curveInput =
                        std::fabs(vehicleForwardSpeed) *
                        SceneVehicleMath::SpeedKilometersPerHourScale;
                maxSteerDegrees = EvaluateCurve<true>(
                        tuning->visual.wheelSteerAngleFromSpeedCurve.Value(),
                        curveInput);
            }
            const float maxSteerRadians =
                    (maxSteerDegrees * SceneVehicleMath::Pi) /
                    DegreesToRadiansDivisor;
            wheelVisualSteerAngle =
                    -car.controls.currentSteering * maxSteerRadians;
        }

        wheel.realTimeState.targetVisualSteerAngle =
                wheelVisualSteerAngle;
        UpdateWheelSpeed<true>(car, wheel, vehicleForwardSpeed, dt);
        IntegrateWheelState<true>(wheel.realTimeState, dt);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void IntegrateVehicle(
            CSceneVehicleCar &car,
            float dt) {
        if constexpr (!NativeBinary32) {
            car.IntegrateVehicle(dt);
            return;
        }

        GmVec3 linearSpeed;
        car.HmsItem()->GetLinearSpeed(linearSpeed);
        const float vehicleForwardSpeed = linearSpeed.z;
        CSceneVehicleCarTuning *tuning =
                car.Tunings()->ActiveTuning();

        if (car.integration.updateWheelVisuals) {
            const float visualSpeedDenominator =
                    std::fabs(vehicleForwardSpeed) *
                            tuning->visual.wheelSpeedScale +
                    tuning->visual.wheelSpeedBase;
            const u32 wheelCount = car.WheelGetCount();
            for (u32 wheelIndex = 0; wheelIndex < wheelCount; ++wheelIndex) {
                CSceneVehicleCar::SSimulationWheel &wheel =
                        car.WheelAt(wheelIndex);
                UpdateWheelVisualState<true>(
                        car,
                        wheel,
                        tuning,
                        vehicleForwardSpeed,
                        dt,
                        visualSpeedDenominator);
            }
        }

        if (car.integration.integrateWheels) {
            const u32 wheelCount = car.WheelGetCount();
            for (u32 wheelIndex = 0; wheelIndex < wheelCount; ++wheelIndex) {
                car.WheelIntegrate(car.WheelAt(wheelIndex), dt);
            }
        }

        if (car.integration.integrateEngine) {
            if (car.controls.forcedLowSpeedFriction == 0) {
                const float input = !car.engine.useLowSpeedGateB
                        ? car.controls.lowSpeedGateA
                        : car.controls.lowSpeedGateB;
                car.EngineIntegrate(input, dt);
            } else {
                car.engine.engineInputMemory = 0.0f;
            }
        }

        car.UpdateCurrentSteering(tuning, dt);
        car.RefreshCollisionTree();
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void GetSlopeAdherence(
            CSceneVehicleCar &car,
            const GmVec3 &normal,
            float &outFirst,
            float &outSecond) {
        if constexpr (!NativeBinary32) {
            car.GetSlopeAdherence(normal, outFirst, outSecond);
            return;
        }

        const float lenSq =
                (normal.y * normal.y + normal.x * normal.x) +
                normal.z * normal.z;
        if (!(1.0e-10f < lenSq)) {
            return;
        }
        CSceneVehicleCarTuning *tuning =
                car.Tunings()->ActiveTuning();
        const float len = Model3Sqrt<true>(lenSq);
        float slope = Model3FromDouble<true>(
                static_cast<double>(normal.y) /
                static_cast<double>(len));
        slope = std::fabs(slope);

        outFirst = slope;
        const auto slopeAdherenceBlend = [](float value,
                                            float minimum,
                                            float maximum) {
            if (!(minimum <= value)) {
                return 0.0f;
            }
            if (!(maximum >= value)) {
                return 1.0f;
            }
            const float angle =
                    ((value - minimum) / (maximum - minimum)) *
                    static_cast<double>(SceneVehicleMath::Pi) * 0.5;
            return 1.0f - Model3Cos<true>(angle);
        };
        outFirst = slopeAdherenceBlend(
                slope,
                tuning->bodyAirResponse.slopeAdherence1Min,
                tuning->bodyAirResponse.slopeAdherence1Max);
        outSecond = slope;
        outSecond = slopeAdherenceBlend(
                slope,
                tuning->bodyAirResponse.slopeAdherence2Min,
                tuning->bodyAirResponse.slopeAdherence2Max);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float ComputeVisualSteerYaw(
            CSceneVehicleCar &car,
            CSceneVehicleCarTuning *tuning,
            const GmVec3 &linearSpeed) {
        if constexpr (!NativeBinary32) {
            return car.ComputeVisualSteerYaw(tuning, linearSpeed);
        }
        const float absSpeedZ = std::fabs(linearSpeed.z);
        const float denominator =
                absSpeedZ * tuning->visual.wheelSpeedScale +
                tuning->visual.wheelSpeedBase;
        float asinValue = 0.0f;
        if (!(denominator < ScalarEpsilon)) {
            constexpr float SafeTrigInteriorLimit = 1.0f - 1.0e-6f;
            const float asinInput = 1.0f / denominator;
            if (asinInput < -SafeTrigInteriorLimit ||
                SafeTrigInteriorLimit < asinInput) {
                asinValue = GmFunc::AsinSafe(asinInput);
            } else {
                asinValue = Model3Asin<true>(asinInput);
            }
        }
        return -car.controls.currentSteering * asinValue;
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void UpdateFeedbackTail(
            CSceneVehicleCar &car,
            CSceneVehicleCarTuning *tuning,
            float dt,
            const GmVec3 &linearSpeed,
            const GmVec3 &savedForce,
            const GmVec3 &savedImpulse,
            float surfaceFeedback) {
        if constexpr (!NativeBinary32) {
            car.UpdateFeedbackTail(
                    tuning,
                    dt,
                    linearSpeed,
                    savedForce,
                    savedImpulse,
                    surfaceFeedback);
            return;
        }

        car.HmsItem()->GetForce(car.gearedDrive.scaledCurrentForce);
        const float forceScale = 1.0f / tuning->feedback.forceDivisor;
        car.gearedDrive.scaledCurrentForce.x =
                forceScale * car.gearedDrive.scaledCurrentForce.x;
        car.gearedDrive.scaledCurrentForce.y =
                forceScale * car.gearedDrive.scaledCurrentForce.y;
        car.gearedDrive.scaledCurrentForce.z =
                forceScale * car.gearedDrive.scaledCurrentForce.z;

        const float surfaceCurveValue = EvaluateCurve<true>(
                tuning->feedback.surfaceCurve.Value(), surfaceFeedback);
        const float feedbackRate =
                surfaceCurveValue + tuning->feedback.surfaceBaseRate;
        const float unclampedSurfaceFeedback =
                feedbackRate * dt + car.feedback.surfaceAccumulator;
        car.feedback.surfaceAccumulator = unclampedSurfaceFeedback;
        car.feedback.surfaceAccumulator =
                ClampZeroOne(unclampedSurfaceFeedback);

        car.UpdateFeedbackSpringAxis(
                car.feedback.sideSpring,
                dt,
                savedForce.x,
                savedImpulse.x,
                0);
        car.UpdateFeedbackSpringAxis(
                car.feedback.forwardSpring,
                dt,
                savedForce.z,
                savedImpulse.z,
                1);

        const float contactRampDirection =
                car.IsAllWheelGroundContactId(FeedbackRampContactId) != 0
                ? 1.0f
                : -1.0f;
        const float curveInput = std::fabs(
                linearSpeed.z *
                SceneVehicleMath::SpeedKilometersPerHourScale);
        const float ramp0 = EvaluateCurve<true>(
                car.FeedbackRamp0Curve(), curveInput);
        car.feedback.ramp0 = ClampZeroOne(
                car.feedback.ramp0 +
                ramp0 * dt * contactRampDirection);
        const float ramp1 = EvaluateCurve<true>(
                car.FeedbackRamp1Curve(), curveInput);
        car.feedback.ramp1 = ClampZeroOne(
                car.feedback.ramp1 +
                ramp1 * dt * contactRampDirection);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float MaxSideFrictionFromSpeed(
            const CSceneVehicleCarTuning &tuning,
            float speed) noexcept {
        return EvaluateSpeedCurve<NativeBinary32>(
                tuning.maxSideFrictionFromSpeedCurve, speed);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float AccelFromSpeed(
            const CSceneVehicleCarTuning &tuning,
            float speed) noexcept {
        return EvaluateLinearSpeedCurve<NativeBinary32>(
                tuning.slipResponse.accelFromSpeedCurve, speed);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float RolloverLateralFromSpeed(
            const CSceneVehicleCarTuning &tuning,
            float speed) noexcept {
        return EvaluateSpeedCurve<NativeBinary32>(
                tuning.rolloverLateralFromSpeedCurve, speed);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float RolloverLateralCoefficientFromAngle(
            const CSceneVehicleCarTuning &tuning,
            float angle) noexcept {
        return EvaluateCurve<NativeBinary32>(
                tuning.rolloverLateralCoefFromAngleCurve.Value(), angle);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float SteerDriveTorqueFromSpeed(
            const CSceneVehicleCarTuning &tuning,
            float speed) noexcept {
        return EvaluateSpeedCurve<NativeBinary32>(
                tuning.steering.driveTorqueFromSpeedCurve, speed);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE float SteerSlowDownFromSpeed(
            const CSceneVehicleCarTuning &tuning,
            float speed) noexcept {
        return EvaluateLinearSpeedCurve<NativeBinary32>(
                tuning.steering.slowDownFromSpeedCurve, speed);
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void ApplyContactForces(
            CSceneVehicleCar &car,
            CHmsDyna &dyna,
            const CSceneVehicleCar::LegacyForceRequest &request,
            const CSceneVehicleCarTuning &tuning) {
        const u32 wheelCount = static_cast<u32>(car.wheels.size());
        for (u32 wheelIndex = 0; wheelIndex < wheelCount; wheelIndex++) {
            CSceneVehicleCar::SSimulationWheel *wheel =
                    &car.wheels[wheelIndex];
            car.WheelAddForceToVehicle(*wheel, request.currentForce);

            CSceneVehicleMaterial *material = car.GetWheelMaterial(*wheel);
            if (!wheel->realTimeState.contactPresent ||
                !(tuning.gearedDrive.lateralForceScale > 0.0f) ||
                tuning.handlingModel != CSceneVehicleCarHandlingModel_Lateral) {
                continue;
            }

            float slipGrip = wheel->realTimeState.slipping
                    ? tuning.gearedDrive.slippingSideFrictionScale
                    : 1.0f;
            float maxSide = material->blendableVals.w *
                            request.slopeAdherenceA *
                            MaxSideFrictionFromSpeed<NativeBinary32>(
                                    tuning, request.linearSpeed.z) *
                            slipGrip;
            GmVec3 lateral = {
                    wheel->realTimeState.accumulatedContactNormal.y,
                    -wheel->realTimeState.accumulatedContactNormal.x,
                    0.0f,
            };
            lateral = Model3NormalizeOr<NativeBinary32>(
                    lateral,
                    GmVec3{1.0f, 0.0f, 0.0f},
                    VectorEpsilonSquared);
            if (IsFrontVehicleWheel(wheel->axle)) {
                float visualSteerYawCos =
                        Model3Cos<NativeBinary32>(request.visualSteerYaw);
                float negVisualSteerYawSin =
                        -Model3Sin<NativeBinary32>(request.visualSteerYaw);
                lateral = GmVec3{
                        visualSteerYawCos * lateral.x,
                        visualSteerYawCos * lateral.y,
                        negVisualSteerYawSin +
                                visualSteerYawCos * lateral.z,
                };
            }

            float sideForce = -tuning.gearedDrive.lateralForceScale * 0.5f *
                              SceneVehicleMath::Dot(
                                      request.linearSpeed, lateral);
            float sideForceAbs = std::fabs(sideForce);
            if (!(maxSide < sideForceAbs)) {
                wheel->realTimeState.slipping = false;
            } else {
                float capped = SignNonNegative(sideForce) * maxSide;
                sideForce =
                        (1.0f -
                         tuning.gearedDrive.sideFrictionSlipBlend) *
                                capped +
                        tuning.gearedDrive.sideFrictionSlipBlend * sideForce;
                wheel->realTimeState.slipping = true;
            }
            if (wheel->realTimeState.slipping) {
                request.outSlipFlag = 1;
            }

            GmVec3 lateralForce =
                    SceneVehicleMath::Scale(lateral, sideForce);
            AddVehicleCentralForce(car, dyna, lateralForce);

            float rollover =
                    -RolloverLateralFromSpeed<NativeBinary32>(
                            tuning, request.linearSpeed.z) *
                    request.slopeAdherenceA *
                    RolloverLateralCoefficientFromAngle<NativeBinary32>(
                            tuning, std::fabs(lateral.y));
            GmVec3 rolloverTorque = {
                    lateralForce.z * rollover,
                    0.0f,
                    -rollover * lateralForce.x,
            };
            AddVehicleTorque(dyna, rolloverTorque);
        }
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void ApplySteeringTorques(
            CSceneVehicleCar &car,
            CHmsDyna &dyna,
            const CSceneVehicleCar::LegacyForceRequest &request,
            const CSceneVehicleCarTuning &tuning,
            float speedMagnitude) {
        const u32 wheelCount = static_cast<u32>(car.wheels.size());
        for (u32 wheelIndex = 0; wheelIndex < wheelCount; wheelIndex++) {
            CSceneVehicleCar::SSimulationWheel *wheel =
                    &car.wheels[wheelIndex];
            float halfTrack =
                    (IsFrontVehicleWheel(wheel->axle)
                                     ? car.gearedDrive.wheelLongitudinalSpan
                                     : -car.gearedDrive.wheelLongitudinalSpan) *
                    0.5f;
            float steerRamp = 1.0f;
            if (!(tuning.steering.assistFullSpeed < speedMagnitude)) {
                steerRamp = Model3Sin<NativeBinary32>(static_cast<float>(
                        (speedMagnitude /
                         tuning.steering.assistFullSpeed) *
                        SceneVehicleMath::HalfPi));
            }
            float maxSide =
                    MaxSideFrictionFromSpeed<NativeBinary32>(
                            tuning, request.linearSpeed.z) *
                    request.materialVals.w;
            float wheelSideSpeed = request.linearSpeed.x +
                                   request.angularSpeed.y * halfTrack;
            float sideForce = -tuning.gearedDrive.lateralForceScale * 0.5f *
                              wheelSideSpeed;
            if (maxSide < std::fabs(sideForce)) {
                float blended =
                        (1.0f -
                         tuning.gearedDrive.driveSideFrictionSlipBlend) *
                                maxSide +
                        tuning.gearedDrive.driveSideFrictionSlipBlend *
                                std::fabs(sideForce);
                sideForce = SignNonNegative(sideForce) * blended;
            }

            float sideTorque =
                    tuning.gearedDrive.sideForceToDriveTorqueScale * sideForce;
            if (IsFrontVehicleWheel(wheel->axle)) {
                float reverseSign =
                        !car.engine.useLowSpeedGateB ? 1.0f : -1.0f;
                float slipScale = wheel->realTimeState.slipping
                        ? tuning.gearedDrive.slippingSteerTorqueScale
                        : 1.0f;
                float steerTorque =
                        SteerDriveTorqueFromSpeed<NativeBinary32>(
                                tuning, request.linearSpeed.z);
                const float steerAssist =
                        reverseSign * steerRamp *
                        car.controls.currentSteering * steerTorque * slipScale;
                sideTorque = sideTorque - steerAssist;
            }
            AddVehicleTorque(
                    dyna, {0.0f, sideTorque * halfTrack, 0.0f});
        }
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void ApplyDriveForces(
            CSceneVehicleCar &car,
            CHmsDyna &dyna,
            const CSceneVehicleCar::LegacyForceRequest &request,
            const CSceneVehicleCarTuning &tuning) {
        float accelBase =
                AccelFromSpeed<NativeBinary32>(tuning, request.linearSpeed.z);
        float sideLimit =
                MaxSideFrictionFromSpeed<NativeBinary32>(
                        tuning, request.linearSpeed.z) *
                request.materialVals.w;
        float sideSlowdownInput = std::fabs(
                tuning.gearedDrive.lateralForceScale * 0.5f *
                request.linearSpeed.x);
        if (sideLimit < sideSlowdownInput) {
            sideSlowdownInput = sideLimit;
        }
        float driveScale =
                request.materialVals.y * car.controls.lowSpeedGateA +
                (car.engine.useLowSpeedGateB ? -1.0f : 0.0f) *
                        request.materialVals.y * car.controls.lowSpeedGateB +
                (car.turbo.type != CSceneVehicleCar::ETurboType_Inactive
                         ? car.turbo.impulseScale
                         : 0.0f);
        float driveForce =
                (accelBase -
                 tuning.steering.slowDownScale * sideSlowdownInput *
                         std::fabs(car.controls.currentSteering) *
                         SteerSlowDownFromSpeed<NativeBinary32>(
                                 tuning, request.linearSpeed.z)) *
                driveScale;
        if (car.controls.forcedLowSpeedFriction != 0) {
            driveForce =
                    car.turbo.type == CSceneVehicleCar::ETurboType_Inactive
                            ? 0.0f
                            : accelBase * car.turbo.impulseScale;
        }

        float opposingLongitudinalForce = 0.0f;
        if (request.linearSpeed.z > 0.0f) {
            opposingLongitudinalForce =
                    (tuning.gearedDrive.forwardAccelBase +
                     tuning.gearedDrive.forwardAccelSpeedCoef *
                             request.linearSpeed.z) *
                    car.controls.lowSpeedGateB;
            float cap =
                    request.materialVals.z *
                    (request.outSlipFlag == 0
                             ? tuning.gearedDrive.forwardAccelCap
                             : tuning.gearedDrive
                                       .forwardAccelCapWhenSlipping);
            if (cap < opposingLongitudinalForce) {
                opposingLongitudinalForce = cap;
                car.MarkAllWheelsSlipping();
            }
        }
        if (request.linearSpeed.z < 0.0f &&
            car.controls.forcedLowSpeedFriction != 0) {
            opposingLongitudinalForce =
                    (tuning.gearedDrive.forwardAccelBase -
                     tuning.gearedDrive.forwardAccelSpeedCoef *
                             request.linearSpeed.z) *
                    car.controls.lowSpeedGateA;
            float cap =
                    request.materialVals.z *
                    (request.outSlipFlag == 0
                             ? tuning.gearedDrive.forwardAccelCap
                             : tuning.gearedDrive
                                       .forwardAccelCapWhenSlipping);
            if (cap < opposingLongitudinalForce) {
                opposingLongitudinalForce = cap;
                car.MarkAllWheelsSlipping();
            }
            opposingLongitudinalForce = -opposingLongitudinalForce;
        }
        if (car.controls.forcedLowSpeedFriction != 0 &&
            std::fabs(request.linearSpeed.z) < 1.0f) {
            opposingLongitudinalForce *= std::fabs(request.linearSpeed.z);
        }

        request.outSurfaceFeedback = opposingLongitudinalForce;
        float netLongitudinal = driveForce - opposingLongitudinalForce;
        if (tuning.engineSpeedNorm * request.materialVals.x <
            request.linearSpeed.z) {
            netLongitudinal = -tuning.gearedDrive.speedLimitForce;
        }
        if (request.linearSpeed.z <
            -(tuning.gearedDrive.transmission.reverseSpeedNorm *
              request.materialVals.x)) {
            netLongitudinal = tuning.gearedDrive.speedLimitForce;
        }

        float longitudinalForceZ =
                netLongitudinal * request.slopeAdherenceB;
        AddVehicleCentralForce(
                car, dyna, {0.0f, 0.0f, longitudinalForceZ});
        AddVehicleTorque(dyna, {
                -longitudinalForceZ *
                        tuning.slipResponse.longitudinalTorqueScale,
                0.0f,
                0.0f,
        });
        AddVehicleCentralForce(car, dyna, {
                0.0f,
                0.0f,
                (-tuning.gearedDrive.forceZScale * request.currentForce.z) /
                        tuning.bodyAirResponse.groundedSolidFeedback1,
        });
    }

    template<bool NativeBinary32>
    static FV_E019_ALWAYS_INLINE void ComputeModel3(
            CSceneVehicleCar &car,
            CHmsDyna &dyna,
            float dt,
            const GmVec3 &currentForce,
            float slopeAdherenceA,
            float slopeAdherenceB,
            const GmVec3 &linearSpeed,
            const GmVec3 &angularSpeed,
            float visualSteerYaw,
            int hasGroundMaterial,
            CSceneVehicleMaterial::SBlendableVals &materialVals,
            int &outSlipFlag,
            float &outSurfaceFeedback,
            const CSceneVehicleCarTuning &tuning) {
        CSceneVehicleCar::LegacyForceRequest request{
                dt,
                currentForce,
                slopeAdherenceA,
                slopeAdherenceB,
                linearSpeed,
                angularSpeed,
                visualSteerYaw,
                hasGroundMaterial != 0,
                materialVals,
                outSlipFlag,
                outSurfaceFeedback,
        };
        ApplyContactForces<NativeBinary32>(car, dyna, request, tuning);
        if (!request.hasGroundMaterial ||
            tuning.handlingModel != CSceneVehicleCarHandlingModel_Lateral) {
            return;
        }

        float speedMagnitude =
                Model3Sqrt<NativeBinary32>(Model3LengthSquared(linearSpeed));
        if (car.controls.forcedLowSpeedFriction == 0) {
            if (!(speedMagnitude < car.reverseGearSpeedThreshold)) {
                if (car.controls.lowSpeedGateA > LowSpeedGateThreshold) {
                    car.engine.useLowSpeedGateB = false;
                }
            } else if (!(LowSpeedGateThreshold <
                         car.controls.lowSpeedGateB)) {
                car.engine.useLowSpeedGateB = false;
            } else {
                car.engine.useLowSpeedGateB = true;
            }
        }
        ApplySteeringTorques<NativeBinary32>(
                car, dyna, request, tuning, speedMagnitude);
        ApplyDriveForces<NativeBinary32>(car, dyna, request, tuning);
    }

    template<bool NativeBinary32>
    static void ComputeForces(
            CSceneVehicleCar &car,
            CHmsDyna &dyna,
            float dt) {
        GmVec3 savedForce;
        GmVec3 savedImpulse;
        car.SaveAndClearAccumulatedFeedback(savedForce, savedImpulse);

        if (car.integration.speedBlocked ||
            car.integration.speedBlockedSecondary ||
            car.WaterState().boxLocal.halfExtents.x < 0.0f) {
            car.SetZeroDynamics();
            return;
        }

        car.CreateFakeContacts();
        IntegrateVehicle<NativeBinary32>(car, dt);

        u32 tick = CMwCmdBufferCore::Current()->Timer().GetTickTime();
        int isGroundContact = car.IsGroundContact();
        CSceneVehicleCarTuning *tuning = car.Tunings()->ActiveTuning();
        car.UpdateDynaParamsForGroundContact(tuning, isGroundContact);

        if (!car.integration.integrateWheels) {
            return;
        }

        GmVec3 linearSpeed;
        GmVec3 angularSpeed;
        GmVec3 currentForce;
        CSceneVehicleMaterial::SBlendableVals materialVals = {
                1.0f, 1.0f, 1.0f, 1.0f};
        int hasGroundMaterial = 0;
        float slopeAdherenceA = 1.0f;
        float slopeAdherenceB = 1.0f;
        float visualSteerYaw = 0.0f;
        int modelSlipFlag = 0;
        int hasSideSpeedKillContact = 0;
        int hasAnyContact = 0;
        car.HmsItem()->GetLinearSpeed(linearSpeed);

        float surfaceFeedback = 0.0f;
        if (car.integration.zeroHorizontalSpeed) {
            linearSpeed.x = 0.0f;
            linearSpeed.z = 0.0f;
            car.HmsItem()->SetLinearSpeed(linearSpeed);
        } else {
            car.HmsItem()->GetAngularSpeed(angularSpeed);
            car.HmsItem()->GetForce(currentForce);

            car.engine.lowSpeedFeedbackForce = 0.0f;
            car.ApplyFrictionForces(linearSpeed);
            car.ClampLinearSpeed(linearSpeed);

            car.ComputeVehicleGroundMaterialVals(
                    materialVals, hasGroundMaterial);
            GetSlopeAdherence<NativeBinary32>(
                    car,
                    currentForce,
                    slopeAdherenceA,
                    slopeAdherenceB);

            visualSteerYaw = ComputeVisualSteerYaw<NativeBinary32>(
                    car, tuning, linearSpeed);
            car.gearedDrive.localSpeed = linearSpeed;

            ComputeModel3<NativeBinary32>(
                    car,
                    dyna,
                    dt,
                    currentForce,
                    slopeAdherenceA,
                    slopeAdherenceB,
                    linearSpeed,
                    angularSpeed,
                    visualSteerYaw,
                    hasGroundMaterial,
                    materialVals,
                    modelSlipFlag,
                    surfaceFeedback,
                    *tuning);

            hasAnyContact = car.ScanWheelSideSpeedKillContacts(
                    hasSideSpeedKillContact);
            car.UpdateLowSpeedFeedback(tuning, hasAnyContact);
            car.KillSideSpeedForTaggedContact(
                    tuning, hasSideSpeedKillContact, linearSpeed);

            car.ComputeAirControl(
                    angularSpeed,
                    tick,
                    isGroundContact,
                    hasSideSpeedKillContact);
            car.ApplySpecialContactResponse(
                    tuning, currentForce, tick, isGroundContact);
            car.UpdateImpactStates(tuning);
            car.lastComputeForcesTick = tick;
            car.ProcessTurboContacts(tuning, tick);
            car.UpdateTurbo(tick);

            car.OtherVehicleForces();
        }

        UpdateFeedbackTail<NativeBinary32>(
                car,
                tuning,
                dt,
                linearSpeed,
                savedForce,
                savedImpulse,
                surfaceFeedback);
        car.ClearWheelContactScratch();
        car.ResetPerTickContactFeedback();
    }
};

namespace forevervalidator::simulation {

void OptimizedCpuModel3VehicleForceContext::BeginTick(
        CSceneVehicleCar &car,
        OptimizedCpuBinary32MathPath mathPath,
        CHmsItem::CCallback *enabledComputeForcesCallback) noexcept {
    tickEligible_ = false;
    if (mathPath != OptimizedCpuBinary32MathPath::X86Sse2 ||
        enabledComputeForcesCallback == nullptr ||
        car.ArePhysicsUpdatesEnabled() == 0) {
        return;
    }

    CHmsItem *item = car.HmsItem();
    if (item == nullptr ||
        item->CallbackGet(CHmsItem::ECallback_ComputeForces) !=
                enabledComputeForcesCallback) {
        return;
    }

    CSceneVehicleCarTuning *tuning =
            OptimizedCpuModel3VehicleForceAccess::ActiveTuning(car);
    const bool identityChanged =
            car_ != &car || item_ != item || tuning_ != tuning;
    if (identityChanged) {
        car_ = &car;
        item_ = item;
        tuning_ = tuning;
        canonicalCallback_ = enabledComputeForcesCallback;
        stableEligible_ = false;
    } else if (canonicalCallback_ != enabledComputeForcesCallback) {
        return;
    }

    if (!stableEligible_) {
        stableEligible_ =
                OptimizedCpuModel3VehicleForceAccess::HasStableEligibility(
                        car, item, tuning, mathPath);
    }
    tickEligible_ =
            stableEligible_ && item->SceneMobilOwner() == &car &&
            OptimizedCpuModel3VehicleForceAccess::
                    HasRequiredModel3Configuration(*tuning);
}

void OptimizedCpuModel3VehicleForceContext::Reset(void) noexcept {
    car_ = nullptr;
    item_ = nullptr;
    tuning_ = nullptr;
    canonicalCallback_ = nullptr;
    stableEligible_ = false;
    tickEligible_ = false;
}

bool OptimizedCpuModel3VehicleForceContext::WouldUseSpecializationFor(
        const CHmsItem *item) const noexcept {
    return tickEligible_ && car_ != nullptr && item_ != nullptr &&
           tuning_ != nullptr && item == item_ && car_->HmsItem() == item_ &&
           item_->SceneMobilOwner() == car_ &&
           car_->ArePhysicsUpdatesEnabled() != 0 &&
           OptimizedCpuModel3VehicleForceAccess::ActiveTuning(*car_) ==
                   tuning_ &&
           tuning_->handlingModel == CSceneVehicleCarHandlingModel_Lateral &&
           item_->CallbackGet(CHmsItem::ECallback_ComputeForces) ==
                   canonicalCallback_;
}

bool OptimizedCpuModel3VehicleForceContext::TryComputeOwnerForces(
        CHmsCorpus *corpus,
        float dt) {
    if (corpus == nullptr ||
        !WouldUseSpecializationFor(corpus->Item()) ||
        item_->CorpusCount() != 1u ||
        item_->CorpusAt(0u) != corpus ||
        corpus->Dynamics() == nullptr) {
        return false;
    }
    OptimizedCpuModel3VehicleForceAccess::ComputeForces<true>(
            *car_, *corpus->Dynamics(), dt);
    return true;
}

float OptimizedCpuEvaluateModel3CurveForDifferential(
        CFuncKeysReal &curve,
        float input,
        bool convertSpeedToKmh,
        bool forceConstantInterpolation,
        OptimizedCpuBinary32MathPath mathPath) noexcept {
    const auto evaluate = [&curve, input](auto nativeTag) noexcept {
        constexpr bool NativeBinary32 = decltype(nativeTag)::value;
        return OptimizedCpuModel3VehicleForceAccess::
                EvaluateCurve<NativeBinary32>(curve, input);
    };
    const auto evaluateSpeed = [&curve, input](auto nativeTag) noexcept {
        constexpr bool NativeBinary32 = decltype(nativeTag)::value;
        const float converted = Model3FromDouble<NativeBinary32>(
                static_cast<double>(input) * static_cast<double>(3.6f));
        return OptimizedCpuModel3VehicleForceAccess::
                EvaluateCurve<NativeBinary32>(curve, converted);
    };

    if (forceConstantInterpolation) {
        curve.SetInterpolation(CFuncKeysReal::Constant);
    }
    if (mathPath == OptimizedCpuBinary32MathPath::X86Sse2) {
        return convertSpeedToKmh
                ? evaluateSpeed(std::true_type{})
                : evaluate(std::true_type{});
    }
    return convertSpeedToKmh
            ? evaluateSpeed(std::false_type{})
            : evaluate(std::false_type{});
}

}  // namespace forevervalidator::simulation

void CHmsZoneDynamic::ComputeCorpusForcesOptimizedCpuModel3(
        CHmsCorpus *corpus,
        float dt,
        forevervalidator::simulation::
                OptimizedCpuModel3VehicleForceContext &context) {
    CHmsDyna *dyna = corpus->Dynamics();
    if (dyna == 0) {
        return;
    }

    CHmsDynaParams *dynaParams = &dyna->Parameters();

    dyna->ValidateDynamicState();

    if (corpus->Item()->GetProperties().kinematicOnly) {
        GmVec3 zero = GmVec3::ZeroForComputeCorpusForces();
        dyna->SetForce(zero);
        dyna->SetTorque(zero);
        return;
    }

    GmVec3 accumulatedForce = GmVec3::ZeroForComputeCorpusForces();

    for (CHmsForceField *field : ForceFields()) {
        GmVec3 fieldValue;
        if (field->GetValue(dyna->CurrentState().position, fieldValue)) {
            accumulatedForce.AddScaledForComputeCorpusForces(
                    fieldValue,
                    dynaParams->forceScale * dynaParams->mass);
        }
    }

    GmVec3 linearSpeed;
    dyna->GetLinearSpeed(linearSpeed);
    accumulatedForce.AddScaledForComputeCorpusForces(
            linearSpeed,
            -linearDampingCoef_ * dynaParams->linearDampingScale);
    dyna->SetForce(accumulatedForce);

    if (dyna->DynamicType() ==
        CHmsDyna::EDynamicType_FullAngularDynamics) {
        GmVec3 angularSpeed;
        dyna->GetAngularSpeed(angularSpeed);
        GmVec3 dampingTorque = angularSpeed;
        dampingTorque.ScaleInPlaceForComputeCorpusForces(
                -angularDampingCoef_ * dynaParams->angularDampingScale);
        dyna->SetTorque(dampingTorque);
    }

    if (!context.TryComputeOwnerForces(corpus, dt)) {
        corpus->ComputeOwnerForces(dt);
    }
}

#undef FV_E019_ALWAYS_INLINE
