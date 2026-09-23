#ifndef FOREVERVALIDATOR_CUDA_VEHICLE_WHEELS_CUH
#define FOREVERVALIDATOR_CUDA_VEHICLE_WHEELS_CUH

#include "simulation/backends/cuda/cuda_dynamics.cuh"
#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_tuning.cuh"
#include "simulation/backends/cuda/cuda_vehicle_powertrain.cuh"

namespace forevervalidator::simulation::cuda::vehicle {
namespace wheel_detail {

constexpr float ScalarEpsilon = 1.0e-5f;
constexpr float VectorEpsilonSquared = 1.0e-10f;
constexpr float WheelSpinAnglePeriod = 512.0f * 3.1415927f;
constexpr float DefaultMaxSteerDegrees = 30.0f;
constexpr float Pi = 3.1415927f;
constexpr float DegreesDivisor = 180.0f;

__device__ inline bool UpdateWheelVisuals(
        const CudaVehicleState &vehicle) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_STEADY_INTEGRATION)
    return true;
#else
    return vehicle.integration.updateWheelVisuals;
#endif
}

__device__ inline bool IntegrateWheels(
        const CudaVehicleState &vehicle) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_STEADY_INTEGRATION)
    return true;
#else
    return vehicle.integration.integrateWheels;
#endif
}

__device__ inline bool IntegrateEngine(
        const CudaVehicleState &vehicle) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_STEADY_INTEGRATION)
    return true;
#else
    return vehicle.integration.integrateEngine;
#endif
}

__device__ inline bool SpeedBlocked(
        const CudaVehicleState &vehicle) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_STEADY_INTEGRATION)
    return false;
#else
    return vehicle.integration.speedBlocked ||
            vehicle.integration.speedBlockedSecondary;
#endif
}

__device__ inline float Mod(float value,
                            float minimum,
                            float maximum) {
    if (minimum < value && value < maximum) {
        return value;
    }
    const float period = maximum - minimum;
    float wrapped = exact::Fmod(value - minimum, period);
    if (wrapped < 0.0f) {
        wrapped += period;
    }
    return wrapped + minimum;
}

__device__ inline GmVec3 Cross(const GmVec3 &left,
                               const GmVec3 &right) {
    return {
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x,
    };
}

__device__ inline GmVec3 NormalizeOr(
        const GmVec3 &value,
        const GmVec3 &shortResult,
        float minimumLengthSquared) {
    const float xy = value.x * value.x + value.y * value.y;
    const float lengthSquared = xy + value.z * value.z;
    if (!(lengthSquared > minimumLengthSquared)) {
        return shortResult;
    }
    const float scale = exact::Divide(
            1.0f, exact::Sqrt(lengthSquared));
    return {
            value.x * scale,
            value.y * scale,
            value.z * scale,
    };
}

__device__ inline void IntegrateRealTime(
        CSceneVehicleCar::SSimulationWheel::SRealTimeState &wheel,
        float dt) {
    const float spinInput =
            wheel.wheelAngularSpeed * dt + wheel.wheelSpinAngle;
    wheel.wheelSpinAngle =
            Mod(spinInput, 0.0f, WheelSpinAnglePeriod);
    const float normalLengthSquared =
            (wheel.accumulatedContactNormal.y *
                     wheel.accumulatedContactNormal.y +
             wheel.accumulatedContactNormal.x *
                     wheel.accumulatedContactNormal.x) +
            wheel.accumulatedContactNormal.z *
                    wheel.accumulatedContactNormal.z;
    if (normalLengthSquared > VectorEpsilonSquared) {
        const float inverseLength = exact::Divide(
                1.0f, exact::Sqrt(normalLengthSquared));
        wheel.accumulatedContactNormal.x *= inverseLength;
        wheel.accumulatedContactNormal.y *= inverseLength;
        wheel.accumulatedContactNormal.z =
                inverseLength *
                wheel.accumulatedContactNormal.z;
        const GmVec3 directionOfView = {
                0.0f,
                -wheel.accumulatedContactNormal.z,
                wheel.accumulatedContactNormal.y,
        };
        const GmVec3 sideSeed = Cross(
                wheel.accumulatedContactNormal,
                directionOfView);
        const GmVec3 side = NormalizeOr(
                sideSeed, sideSeed, 1.0e-10f);
        const GmVec3 up = NormalizeOr(
                wheel.accumulatedContactNormal,
                wheel.accumulatedContactNormal, 1.0e-10f);
        wheel.contactFrame.basisX = side;
        wheel.contactFrame.basisY = up;
        wheel.contactFrame.basisZ = Cross(side, up);
    }
    const float current = wheel.currentVisualSteerAngle;
    const float target = wheel.targetVisualSteerAngle;
    if (target > current) {
        const float next = current + dt;
        wheel.currentVisualSteerAngle =
                next > target ? target : next;
    } else {
        const float next = current - dt;
        wheel.currentVisualSteerAngle =
                target > next ? target : next;
    }
}

__device__ inline void UpdateSpeed(
        CudaVehicleState &vehicle,
        CudaWheelState &wheel,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t wheelIndex,
        float vehicleForwardSpeed,
        float dt) {
    if (wheel.realTime.contactPresent) {
        if (vehicle.gearedDrive.wheelSpeedOverrideActive &&
            !vehicle.gearedDrive.wheelDriveSpeedInhibited) {
            wheel.realTime.wheelAngularSpeed =
                    configuration->tuning.gearedDrive.burnout.
                            wheelAngularSpeedOverride;
            return;
        }
        wheel.realTime.wheelAngularSpeed =
                exact::Divide(
                        vehicleForwardSpeed,
                        facts::WheelRollingRadius(
                                configuration, wheelIndex));
        return;
    }
    float targetAngularSpeed = 0.0f;
    float acceleration = 0.0f;
    if (vehicle.controls.lowSpeedGateB > ScalarEpsilon) {
        targetAngularSpeed =
                1.0f - vehicle.controls.lowSpeedGateB;
        if (targetAngularSpeed <= 0.0f) {
            targetAngularSpeed = 0.0f;
        } else if (targetAngularSpeed >= 1.0f) {
            targetAngularSpeed = 1.0f;
        }
        acceleration = -100.0f;
    } else if (
            vehicle.controls.lowSpeedGateA > ScalarEpsilon &&
            !vehicle.gearedDrive.wheelDriveSpeedInhibited &&
            !vehicle.controls.forcedLowSpeedFriction) {
        targetAngularSpeed =
                vehicle.controls.lowSpeedGateA * 200.0f;
        acceleration = 100.0f;
    } else {
        wheel.realTime.wheelAngularSpeed *= 0.995f;
    }
    if (!(fabsf(acceleration) < ScalarEpsilon)) {
        const float next =
                acceleration * dt +
                wheel.realTime.wheelAngularSpeed;
        wheel.realTime.wheelAngularSpeed = next;
        if (acceleration > 0.0f && next > targetAngularSpeed) {
            wheel.realTime.wheelAngularSpeed = targetAngularSpeed;
        } else if (acceleration < 0.0f &&
                   next < targetAngularSpeed) {
            wheel.realTime.wheelAngularSpeed = targetAngularSpeed;
        }
    }
}

__device__ inline void RotateVisualY(
        GmMat3 &rotation,
        const exact::SinCosResult &sinCos) {
    const float sine = sinCos.sine;
    const float cosine = sinCos.cosine;
    const GmVec3 oldX = dynamics::detail::Row(rotation, 0u);
    const GmVec3 oldZ = dynamics::detail::Row(rotation, 2u);
    const GmVec3 newX = {
            oldX.x * cosine + oldZ.x * sine,
            oldX.y * cosine + oldZ.y * sine,
            oldX.z * cosine + oldZ.z * sine,
    };
    const GmVec3 newZ = {
            oldX.x * -sine + oldZ.x * cosine,
            oldX.y * -sine + oldZ.y * cosine,
            oldX.z * -sine + oldZ.z * cosine,
    };
    rotation.basisX.x = newX.x;
    rotation.basisY.x = newX.y;
    rotation.basisZ.x = newX.z;
    rotation.basisX.z = newZ.x;
    rotation.basisY.z = newZ.y;
    rotation.basisZ.z = newZ.z;
}

__device__ inline void RotateVisualY(
        GmMat3 &rotation,
        float angle) {
    const exact::SinCosResult sinCos = exact::SinCos(angle);
    const float sine = sinCos.sine;
    const float cosine = sinCos.cosine;
    const GmVec3 oldX = dynamics::detail::Row(rotation, 0u);
    const GmVec3 oldZ = dynamics::detail::Row(rotation, 2u);
    const GmVec3 newX = {
            oldX.x * cosine + oldZ.x * sine,
            oldX.y * cosine + oldZ.y * sine,
            oldX.z * cosine + oldZ.z * sine,
    };
    const GmVec3 newZ = {
            oldX.x * -sine + oldZ.x * cosine,
            oldX.y * -sine + oldZ.y * cosine,
            oldX.z * -sine + oldZ.z * cosine,
    };
    rotation.basisX.x = newX.x;
    rotation.basisY.x = newX.y;
    rotation.basisZ.x = newX.z;
    rotation.basisX.z = newZ.x;
    rotation.basisY.z = newZ.y;
    rotation.basisZ.z = newZ.z;
}

}  // namespace wheel_detail

struct WheelVisualInvariants {
    exact::SinCosResult frontSteerSinCos{};
    float frontVisualSteerAngle = 0.0f;
};

__device__ inline WheelVisualInvariants
ComputeWheelVisualInvariants(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        float vehicleForwardSpeed,
        float visualSpeedDenominator) {
    float yaw = 0.0f;
    if (!(visualSpeedDenominator < 1.0e-5f)) {
        yaw = exact::Divide(
                -vehicle.controls.currentSteering,
                visualSpeedDenominator);
    }
    float maximumDegrees =
            wheel_detail::DefaultMaxSteerDegrees;
    const auto *curves =
            reinterpret_cast<const CudaTuningCurve *>(
                    &configuration->tuning.curves);
    const CudaTuningCurve &curve =
            curves[static_cast<std::uint32_t>(
                    CudaTuningCurveId::
                            WheelVisualSteerAngleFromSpeed)];
    if (curve.keyCount != 0u) {
        maximumDegrees = tuning::Evaluate(
                configuration,
                CudaTuningCurveId::
                        WheelVisualSteerAngleFromSpeed,
                fabsf(vehicleForwardSpeed) * 3.6f);
    }
    const float maximumRadians =
            exact::Divide(
                    maximumDegrees * wheel_detail::Pi,
                    wheel_detail::DegreesDivisor);
    return {
            exact::SinCos(yaw),
            -vehicle.controls.currentSteering *
                    maximumRadians,
    };
}

template <bool ReuseFrontInvariants = false>
__device__ inline void UpdateWheelVisual(
        CudaVehicleState &vehicle,
        CudaWheelState &wheel,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t wheelIndex,
        float vehicleForwardSpeed,
        float dt,
        float visualSpeedDenominator,
        const WheelVisualInvariants &invariants = {}) {
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_CANONICAL_WHEEL_FACTS)
    wheel.realTime.visualRotation = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
    };
#else
    const VehicleWheelDefinition &definition =
            facts::Wheel(configuration, wheelIndex);
    wheel.realTime.visualRotation =
            definition.restSurfacePose.rotation;
#endif
    float visualSteerAngle = 0.0f;
    if (facts::WheelAxle(
                configuration,
                wheelIndex) == VehicleWheelAxle::Front) {
        if constexpr (ReuseFrontInvariants) {
            wheel_detail::RotateVisualY(
                    wheel.realTime.visualRotation,
                    invariants.frontSteerSinCos);
            visualSteerAngle =
                    invariants.frontVisualSteerAngle;
        } else {
            float yaw = 0.0f;
            if (!(visualSpeedDenominator < 1.0e-5f)) {
                yaw = exact::Divide(
                        -vehicle.controls.currentSteering,
                        visualSpeedDenominator);
            }
            wheel_detail::RotateVisualY(
                    wheel.realTime.visualRotation, yaw);
            float maximumDegrees =
                    wheel_detail::DefaultMaxSteerDegrees;
            const auto *curves =
                    reinterpret_cast<const CudaTuningCurve *>(
                            &configuration->tuning.curves);
            const CudaTuningCurve &curve =
                    curves[static_cast<std::uint32_t>(
                            CudaTuningCurveId::
                                    WheelVisualSteerAngleFromSpeed)];
            if (curve.keyCount != 0u) {
                maximumDegrees = tuning::Evaluate(
                        configuration,
                        CudaTuningCurveId::
                                WheelVisualSteerAngleFromSpeed,
                        fabsf(vehicleForwardSpeed) * 3.6f);
            }
            const float maximumRadians =
                    exact::Divide(
                            maximumDegrees * wheel_detail::Pi,
                            wheel_detail::DegreesDivisor);
            visualSteerAngle =
                    -vehicle.controls.currentSteering *
                    maximumRadians;
        }
    }
    wheel.realTime.targetVisualSteerAngle = visualSteerAngle;
    wheel_detail::UpdateSpeed(
            vehicle, wheel, configuration,
            wheelIndex, vehicleForwardSpeed, dt);
    wheel_detail::IntegrateRealTime(wheel.realTime, dt);
}

__device__ inline void IntegrateWheelSuspension(
        CudaWheelState &wheel,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t wheelIndex,
        float dt) {
    const ReplayVehicleTuningSuspension &tuning =
            configuration->tuning.suspension;
    const GmIso4 &restPose =
            facts::Wheel(
                    configuration,
                    wheelIndex).restSurfacePose;
    switch (facts::WheelForceMode(configuration)) {
    case CSceneVehicleCarWheelForceMode_DirectSpring: {
        wheel.realTime.damperAbsorb =
                wheel.realTime.damperAbsorb -
                wheel.realTime.maxReplacementY;
        wheel.realTime.maxReplacementY = 0.0f;
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_CANONICAL_WHEEL_FACTS)
        wheel.currentPose = {
                {{1.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}},
                restPose.translation,
        };
#else
        wheel.currentPose = restPose;
#endif
        const float acceleration =
                (tuning.wheelRestDamperAbsorb -
                 wheel.realTime.damperAbsorb) *
                        tuning.wheelSpringCoef -
                tuning.wheelDamperCoef *
                        wheel.realTime.damperVelocity;
        wheel.realTime.damperVelocity =
                acceleration * dt +
                wheel.realTime.damperVelocity;
        wheel.realTime.damperAbsorb =
                dt * wheel.realTime.damperVelocity +
                wheel.realTime.damperAbsorb;
        wheel.currentPose.translation.y +=
                -wheel.realTime.damperAbsorb;
        break;
    }
    case CSceneVehicleCarWheelForceMode_FollowAbsorb:
    case CSceneVehicleCarWheelForceMode_FollowAbsorbWithImpulse: {
        const float baseAbsorb =
                wheel.realTime.damperAbsorb -
                wheel.realTime.maxReplacementY;
#if defined(FOREVERVALIDATOR_CUDA_RESEARCH_CANONICAL_WHEEL_FACTS)
        wheel.currentPose = {
                {{1.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}},
                restPose.translation,
        };
#else
        wheel.currentPose = restPose;
#endif
        const float displacement =
                tuning.wheelRestDamperAbsorb - baseAbsorb;
        const float target =
                displacement * dt *
                        tuning.wheelAbsorbFollowCoef +
                baseAbsorb;
        wheel.realTime.damperVelocity = exact::Divide(
                target - wheel.realTime.damperAbsorb, dt);
        wheel.realTime.damperAbsorb = target;
        wheel.realTime.maxReplacementY = 0.0f;
        wheel.currentPose.translation.y += -target;
        break;
    }
    default:
        break;
    }
    wheel.surfaceMovedByUpdate = true;
}

template <
        bool ReuseWheelPassInvariants = false,
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic>
__device__ inline void IntegrateVehiclePrefix(
    CudaCandidatePhysicsState &candidate,
    const CudaPackedStaticConfigurationHeader *configuration,
    float dt) {
    CudaVehicleState &vehicle = candidate.vehicle;
    const GmVec3 &worldSpeed = candidate.body.current.linearSpeed;
    const GmMat3 &rotation = candidate.body.current.rotation;
    const GmVec3 localSpeed = {
            dynamics::detail::Dot(rotation.basisX, worldSpeed),
            dynamics::detail::Dot(rotation.basisY, worldSpeed),
            dynamics::detail::Dot(rotation.basisZ, worldSpeed),
    };
    const float forwardSpeed = localSpeed.z;
    if (wheel_detail::UpdateWheelVisuals(vehicle)) {
        const float denominator =
                fabsf(forwardSpeed) *
                        configuration->tuning.visual.wheelSpeedScale +
                configuration->tuning.visual.wheelSpeedBase;
        WheelVisualInvariants invariants;
        if constexpr (ReuseWheelPassInvariants) {
            invariants = ComputeWheelVisualInvariants(
                    vehicle, configuration,
                    forwardSpeed, denominator);
        }
        for (std::uint32_t index = 0u;
             index < facts::WheelCount(vehicle); ++index) {
            UpdateWheelVisual<ReuseWheelPassInvariants>(
                    vehicle, vehicle.wheels.values[index],
                    configuration, index,
                    forwardSpeed, dt, denominator,
                    invariants);
        }
    }
    if (wheel_detail::IntegrateWheels(vehicle)) {
        for (std::uint32_t index = 0u;
             index < facts::WheelCount(vehicle); ++index) {
            IntegrateWheelSuspension(
                    vehicle.wheels.values[index],
                    configuration, index, dt);
        }
    }
    if (wheel_detail::IntegrateEngine(vehicle)) {
        if (!vehicle.controls.forcedLowSpeedFriction) {
            const float input =
                    !vehicle.engine.useLowSpeedGateB
                    ? vehicle.controls.lowSpeedGateA
                    : vehicle.controls.lowSpeedGateB;
            IntegrateEngine<Handling>(
                    vehicle, configuration, input, dt);
        } else {
            vehicle.engine.engineInputMemory = 0.0f;
        }
    }
    UpdateCurrentSteering(vehicle, configuration, dt);
}

}  // namespace forevervalidator::simulation::cuda::vehicle

#endif
