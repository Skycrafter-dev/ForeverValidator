// Generated from src/simulation/backends/cuda/cuda_vehicle_powertrain.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_VEHICLE_POWERTRAIN_CUH
#define FOREVERVALIDATOR_HIP_VEHICLE_POWERTRAIN_CUH

#include "simulation/backends/hip/generated/hip_state_layout.h"
#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_tuning.cuh"

namespace forevervalidator::simulation::hip::vehicle {
namespace detail {

constexpr float EngineInputActiveThreshold = 0.1f;
constexpr float GearShiftInputMemoryThreshold = 1000.0f;
constexpr float GearShiftCooldown = 0.025f;
constexpr float BurnoutReverseGearShiftCooldown = 0.002f;
constexpr float SlipRpmScaleMaximum = 1.15f;
constexpr float SlipRpmScaleApproachRate = 0.3f;
constexpr float LegacyEngineGearChangeCooldown = 0.04f;
constexpr float LegacyEngineWeightedSpeedXScale = 0.3f;
constexpr float LegacyEngineWeightedSpeedYScale = 0.1f;
constexpr float LegacyEngineSpeedNormScale = 0.2f;
constexpr float LegacyEngineCooldownDecayScale = 1.9f;
constexpr float LegacyEngineGroundInputRate = 12.0f;
constexpr float LegacyEngineAirborneInputRate = 3.5f;

__device__ inline float TransmissionValue(
        const HipPackedStaticConfigurationHeader *configuration,
        HipTransmissionArrayId array,
        int gear) {
    std::uint32_t count = 0u;
    const float *values = tuning::TransmissionValues(
            configuration, array, count);
    const std::uint32_t index =
            gear >= 0 ? static_cast<std::uint32_t>(gear) : 0u;
    return index < count ? values[index] : 0.0f;
}

__device__ inline bool AllWheelsAirborne(
        const HipVehicleState &vehicle) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        if (vehicle.wheels.values[index].realTime.contactPresent) {
            return false;
        }
    }
    return true;
}

__device__ inline void ClampEngineInput(
        HipVehicleState &vehicle) {
    const float memory = vehicle.engine.engineInputMemory;
    const float maximum = vehicle.engine.engineInputMax;
    float clamped = 0.0f;
    if (!(memory <= 0.0f)) {
        clamped = memory;
        if (!(maximum > memory)) {
            clamped = maximum;
        }
    }
    vehicle.engine.engineInputMemory = clamped;
}

__device__ inline void UpdateDirectionalTransitionInput(
        HipVehicleState &vehicle,
        const HipPackedStaticConfigurationHeader *configuration,
        float dt,
        bool inputActive,
        CSceneVehicleCarEngineControlState state) {
    const HipVehicleGearedDriveTuning &tuning =
            configuration->tuning.gearedDrive;
    const int gear =
            state ==
                    CSceneVehicleCarEngineControlState_ForwardTransition
            ? 1
            : 0;
    const float speed = vehicle.gearedDrive.localSpeed.z;
    bool outside;
    if (state ==
        CSceneVehicleCarEngineControlState_ForwardTransition) {
        outside =
                tuning.input.forwardTransitionSpeedHigh < speed ||
                speed < tuning.input.forwardTransitionSpeedLow;
    } else {
        outside =
                speed <
                        tuning.input.reverseTransitionSpeedLow ||
                tuning.input.reverseTransitionSpeedHigh < speed;
    }
    vehicle.gearedDrive.inputWindowExceeded = outside;
    vehicle.engine.slipRpmScale = 1.0f;
    const float absoluteSpeed = fabsf(speed);
    vehicle.engine.targetTransmissionInput =
            absoluteSpeed *
                    TransmissionValue(
                            configuration,
                            HipTransmissionArrayId::GearSpeedRatio,
                            gear) +
            TransmissionValue(
                    configuration,
                    HipTransmissionArrayId::TargetInputBias,
                    gear) *
                    0.0f;
    if (outside || !inputActive) {
        const float next =
                vehicle.engine.engineInputMemory -
                tuning.input.groundInputFall * dt;
        vehicle.engine.engineInputMemory = next;
        if (next <= vehicle.engine.targetTransmissionInput) {
            vehicle.engine.engineInputMemory =
                    vehicle.engine.targetTransmissionInput;
            vehicle.gearedDrive.engineState =
                    CSceneVehicleCarEngineControlState_Steady;
            vehicle.gearedDrive.inputWindowExceeded = false;
        }
    } else {
        vehicle.engine.engineInputMemory =
                tuning.input.transitionInputRise * dt +
                vehicle.engine.engineInputMemory;
    }
}

__device__ inline void UpdateTransmissionTransitions(
        HipVehicleState &vehicle,
        const HipPackedStaticConfigurationHeader *configuration,
        bool inputActive,
        bool burnoutState) {
    const HipVehicleGearedDriveTuning &tuning =
            configuration->tuning.gearedDrive;
    const float speed = vehicle.gearedDrive.localSpeed.z;
    if (tuning.input.forwardTransitionSpeedLow < speed &&
        speed < tuning.input.forwardTransitionSpeedHigh &&
        inputActive && !vehicle.engine.useLowSpeedGateB &&
        vehicle.gearedDrive.engineState ==
                CSceneVehicleCarEngineControlState_Steady) {
        vehicle.gearedDrive.engineState =
                CSceneVehicleCarEngineControlState_ForwardTransition;
        vehicle.gearedDrive.inputWindowExceeded = false;
        if (vehicle.engine.gearIndex == 0) {
            vehicle.engine.shiftCooldown = GearShiftCooldown;
            vehicle.engine.gearIndex = 1;
        }
    } else if (
            tuning.input.reverseTransitionSpeedLow < speed &&
            speed < tuning.input.reverseTransitionSpeedHigh &&
            inputActive && vehicle.engine.useLowSpeedGateB &&
            vehicle.gearedDrive.engineState ==
                    CSceneVehicleCarEngineControlState_Steady) {
        vehicle.gearedDrive.engineState =
                CSceneVehicleCarEngineControlState_ReverseTransition;
        vehicle.gearedDrive.inputWindowExceeded = false;
        if (vehicle.engine.gearIndex != 0) {
            vehicle.engine.gearIndex = 0;
            vehicle.engine.shiftCooldown = GearShiftCooldown;
        }
    }

    if (vehicle.gearedDrive.engineState !=
                CSceneVehicleCarEngineControlState_Steady &&
        vehicle.gearedDrive.engineState !=
                CSceneVehicleCarEngineControlState_GearShift) {
        return;
    }
    if (!vehicle.engine.useLowSpeedGateB) {
        if (vehicle.engine.gearIndex == 0) {
            vehicle.gearedDrive.engineState =
                    CSceneVehicleCarEngineControlState_GearShift;
            vehicle.gearedDrive.shiftDirection =
                    CSceneVehicleCarShiftDirection_Up;
            if (vehicle.engine.engineInputMemory <
                GearShiftInputMemoryThreshold) {
                vehicle.engine.shiftCooldown = GearShiftCooldown;
                vehicle.engine.gearIndex = 1;
            }
        }
        const int gear = vehicle.engine.gearIndex;
        if (gear > 0) {
            const float upshift =
                    TransmissionValue(
                            configuration,
                            HipTransmissionArrayId::UpshiftThreshold,
                            gear) *
                    vehicle.engine.engineInputMax;
            if (upshift <
                        vehicle.engine.targetTransmissionInput &&
                gear < 5) {
                vehicle.engine.shiftCooldown = GearShiftCooldown;
                vehicle.engine.gearIndex = gear + 1;
                vehicle.gearedDrive.engineState =
                        CSceneVehicleCarEngineControlState_GearShift;
                vehicle.gearedDrive.shiftDirection =
                        CSceneVehicleCarShiftDirection_Up;
                return;
            }
            const float downshift =
                    TransmissionValue(
                            configuration,
                            HipTransmissionArrayId::DownshiftThreshold,
                            gear) *
                    vehicle.engine.engineInputMax;
            if (vehicle.engine.targetTransmissionInput < downshift &&
                gear > 1) {
                vehicle.engine.shiftCooldown = GearShiftCooldown;
                vehicle.engine.gearIndex = gear - 1;
                vehicle.gearedDrive.engineState =
                        CSceneVehicleCarEngineControlState_GearShift;
                vehicle.gearedDrive.shiftDirection =
                        CSceneVehicleCarShiftDirection_Down;
            }
        }
    } else if (vehicle.engine.gearIndex != 0) {
        vehicle.gearedDrive.engineState =
                CSceneVehicleCarEngineControlState_GearShift;
        vehicle.gearedDrive.shiftDirection =
                CSceneVehicleCarShiftDirection_Down;
        if (vehicle.engine.engineInputMemory <
            GearShiftInputMemoryThreshold) {
            vehicle.engine.gearIndex = 0;
            vehicle.engine.shiftCooldown =
                    burnoutState
                    ? BurnoutReverseGearShiftCooldown
                    : GearShiftCooldown;
        }
    }
}

}  // namespace detail

__device__ inline void IntegrateLegacyEngine(
        HipVehicleState &vehicle,
        const HipPackedStaticConfigurationHeader *configuration,
        float input,
        float dt,
        bool blocked) {
    input = fabsf(input);
    const bool groundReady =
            !blocked &&
            !(0.0f < vehicle.engine.shiftCooldown);
    const int storedGear = vehicle.engine.gearIndex;
    const int ratioIndex =
            storedGear < 2 ? 0 : storedGear - 1;
    const GmVec3 &speed = vehicle.gearedDrive.localSpeed;
    const float weightedSpeed = exact::Sqrt(
            speed.z * speed.z +
            detail::LegacyEngineWeightedSpeedXScale *
                    speed.x * speed.x +
            detail::LegacyEngineWeightedSpeedYScale *
                    speed.y * speed.y);
    const float targetInput =
            exact::Divide(
                    weightedSpeed,
                    configuration->tuning.engineSpeedNorm *
                            detail::LegacyEngineSpeedNormScale) *
            detail::TransmissionValue(
                    configuration,
                    HipTransmissionArrayId::GearSpeedRatio,
                    ratioIndex);
    if (groundReady) {
        input = targetInput;
        if (!vehicle.engine.useLowSpeedGateB) {
            if (storedGear == 0) {
                vehicle.engine.gearIndex = 1;
                vehicle.engine.shiftCooldown =
                        detail::
                                LegacyEngineGearChangeCooldown;
            } else if (
                    detail::TransmissionValue(
                            configuration,
                            HipTransmissionArrayId::
                                    UpshiftThreshold,
                            ratioIndex) <
                            targetInput &&
                    storedGear < 5) {
                vehicle.engine.gearIndex = storedGear + 1;
                vehicle.engine.shiftCooldown =
                        detail::
                                LegacyEngineGearChangeCooldown;
            } else if (
                    targetInput <
                            detail::TransmissionValue(
                                    configuration,
                                    HipTransmissionArrayId::
                                            DownshiftThreshold,
                                    ratioIndex) &&
                    storedGear > 1) {
                vehicle.engine.gearIndex = storedGear - 1;
                vehicle.engine.shiftCooldown =
                        detail::
                                LegacyEngineGearChangeCooldown;
            }
        } else if (storedGear != 0) {
            vehicle.engine.gearIndex = 0;
            vehicle.engine.shiftCooldown =
                    detail::LegacyEngineGearChangeCooldown;
        }
    } else if (
            !(vehicle.engine.shiftCooldown < 0.0f) &&
            !(dt + dt < vehicle.engine.shiftCooldown)) {
        vehicle.engine.engineInputMemory -=
                vehicle.engine.engineInputMax * dt *
                detail::LegacyEngineCooldownDecayScale;
    }
    const float rate = groundReady
            ? detail::LegacyEngineGroundInputRate
            : detail::LegacyEngineAirborneInputRate;
    vehicle.engine.engineInputMemory +=
            (vehicle.engine.engineInputMax * input -
             vehicle.engine.engineInputMemory) *
            dt * rate;
}

__device__ inline void IntegrateGearedEngine(
        HipVehicleState &vehicle,
        const HipPackedStaticConfigurationHeader *configuration,
        float dt,
        bool inputActive,
        bool blocked) {
    const HipVehicleGearedDriveTuning &tuning =
            configuration->tuning.gearedDrive;
    if (blocked) {
        if (inputActive) {
            vehicle.engine.engineInputMemory =
                    tuning.input.airborneInputRise * dt +
                    vehicle.engine.engineInputMemory;
        } else {
            vehicle.engine.engineInputMemory =
                    vehicle.engine.engineInputMemory -
                    tuning.input.airborneInputFall * dt;
        }
        return;
    }

    const bool burnoutState =
            vehicle.gearedDrive.burnoutPhase ==
                    CSceneVehicleCarBurnoutPhase_TimedSpin ||
            vehicle.gearedDrive.burnoutPhase ==
                    CSceneVehicleCarBurnoutPhase_CircularDrift;
    if (burnoutState && vehicle.engine.gearIndex != 0) {
        vehicle.gearedDrive.engineState =
                CSceneVehicleCarEngineControlState_BurnoutHold;
    } else if (vehicle.gearedDrive.engineState ==
               CSceneVehicleCarEngineControlState_BurnoutHold) {
        vehicle.gearedDrive.engineState =
                CSceneVehicleCarEngineControlState_Steady;
    }

    switch (vehicle.gearedDrive.engineState) {
    case CSceneVehicleCarEngineControlState_ForwardTransition:
    case CSceneVehicleCarEngineControlState_ReverseTransition:
        detail::UpdateDirectionalTransitionInput(
                vehicle, configuration, dt, inputActive,
                vehicle.gearedDrive.engineState);
        break;
    case CSceneVehicleCarEngineControlState_BurnoutHold:
        vehicle.engine.targetTransmissionInput =
                vehicle.engine.engineInputMax;
        vehicle.engine.slipRpmScale =
                detail::SlipRpmScaleMaximum;
        if (vehicle.engine.engineInputMemory <
            vehicle.engine.engineInputMax) {
            vehicle.engine.engineInputMemory =
                    tuning.input.burnoutHoldInputRise * dt +
                    vehicle.engine.engineInputMemory;
        } else if (vehicle.engine.engineInputMax <
                   vehicle.engine.engineInputMemory) {
            vehicle.engine.engineInputMemory =
                    tuning.input.airborneInputFall * dt +
                    vehicle.engine.engineInputMemory;
        }
        break;
    default: {
        if (vehicle.slipMemory.active && inputActive) {
            if (vehicle.engine.slipRpmScale <
                detail::SlipRpmScaleMaximum) {
                vehicle.engine.slipRpmScale =
                        (detail::SlipRpmScaleMaximum -
                         vehicle.engine.slipRpmScale) *
                                detail::SlipRpmScaleApproachRate *
                                dt +
                        vehicle.engine.slipRpmScale;
            } else {
                vehicle.engine.slipRpmScale =
                        detail::SlipRpmScaleMaximum;
            }
        } else {
            vehicle.engine.slipRpmScale = 1.0f;
        }
        const int gear = vehicle.engine.gearIndex;
        const float slipSpeed =
                vehicle.gearedDrive.localSpeed.z *
                vehicle.engine.slipRpmScale;
        const float absoluteSlipSpeed = fabsf(slipSpeed);
        vehicle.engine.targetTransmissionInput =
                absoluteSlipSpeed *
                detail::TransmissionValue(
                        configuration,
                        HipTransmissionArrayId::GearSpeedRatio,
                        gear);
        if ((!vehicle.engine.useLowSpeedGateB && gear == 0) ||
            (vehicle.engine.useLowSpeedGateB && gear != 0)) {
            vehicle.engine.targetTransmissionInput = 0.0f;
        }
        if (!(vehicle.engine.engineInputMemory <
              vehicle.engine.targetTransmissionInput)) {
            float fall;
            if (inputActive) {
                fall = !vehicle.slipMemory.active || burnoutState
                        ? tuning.input.groundInputBrake
                        : tuning.input.airborneInputFall;
            } else {
                fall = tuning.input.groundInputFall;
            }
            const float next =
                    vehicle.engine.engineInputMemory - fall * dt;
            vehicle.engine.engineInputMemory = next;
            if (next <
                vehicle.engine.targetTransmissionInput) {
                vehicle.gearedDrive.engineState =
                        CSceneVehicleCarEngineControlState_Steady;
            }
        } else {
            const float next =
                    tuning.input.groundInputRise * dt +
                    vehicle.engine.engineInputMemory;
            vehicle.engine.engineInputMemory = next;
            if (vehicle.engine.targetTransmissionInput < next) {
                vehicle.gearedDrive.engineState =
                        CSceneVehicleCarEngineControlState_Steady;
            }
        }
        break;
    }
    }
    detail::UpdateTransmissionTransitions(
            vehicle, configuration, inputActive, burnoutState);
}

template <
        HipHandlingSpecialization Handling =
                HipHandlingSpecialization::Generic>
__device__ inline void IntegrateEngine(
        HipVehicleState &vehicle,
        const HipPackedStaticConfigurationHeader *configuration,
        float input,
        float dt) {
    const bool inputActive =
            input > detail::EngineInputActiveThreshold;
    if (0.0f < vehicle.engine.shiftCooldown) {
        vehicle.engine.shiftCooldown -= dt;
    }
    const bool blocked =
            detail::AllWheelsAirborne(vehicle) ||
            0.0f < vehicle.engine.shiftCooldown;
    if constexpr (Handling == HipHandlingSpecialization::Legacy) {
        IntegrateLegacyEngine(
                vehicle, configuration, input, dt, blocked);
    } else if constexpr (
            Handling ==
                    HipHandlingSpecialization::GearedDriveDry ||
            Handling ==
                    HipHandlingSpecialization::GearedDriveWater) {
        IntegrateGearedEngine(
                vehicle, configuration, dt,
                inputActive, blocked);
    } else if (
            configuration->tuning.handlingModel !=
            static_cast<std::uint32_t>(
                    CSceneVehicleCarHandlingModel_GearedDrive)) {
        IntegrateLegacyEngine(
                vehicle, configuration, input, dt, blocked);
    } else {
        IntegrateGearedEngine(
                vehicle, configuration, dt,
                inputActive, blocked);
    }
    detail::ClampEngineInput(vehicle);
}

__device__ inline void UpdateCurrentSteering(
        HipVehicleState &vehicle,
        const HipPackedStaticConfigurationHeader *configuration,
        float dt) {
    const float slewRate =
            configuration->tuning.steering.slewRate;
    if (slewRate <= 0.0f) {
        vehicle.controls.currentSteering =
                vehicle.controls.steeringControl;
        return;
    }
    float direction = -1.0f;
    if (vehicle.controls.currentSteering -
                vehicle.controls.steeringControl <
        0.0f) {
        direction = 1.0f;
    }
    const float candidate =
            slewRate * direction * dt +
            vehicle.controls.currentSteering;
    float next = candidate;
    if (vehicle.controls.steeringControl <=
        vehicle.controls.currentSteering) {
        if (vehicle.controls.steeringControl > candidate) {
            next = vehicle.controls.steeringControl;
        }
    } else if (vehicle.controls.steeringControl < candidate) {
        next = vehicle.controls.steeringControl;
    }
    vehicle.controls.currentSteering = next;
}

}  // namespace forevervalidator::simulation::hip::vehicle

#endif
