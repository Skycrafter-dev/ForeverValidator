#ifndef FOREVERVALIDATOR_CUDA_VEHICLE_TRANSITIONS_CUH
#define FOREVERVALIDATOR_CUDA_VEHICLE_TRANSITIONS_CUH

#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/backends/cuda/cuda_timeline_executor.h"
#include "simulation/backends/cuda/cuda_dynamics.cuh"

namespace forevervalidator::simulation::cuda::transition {
namespace detail {

__device__ inline GmMat3 IdentityMatrix() {
    return {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
    };
}

__device__ inline GmIso4 IdentityIso() {
    return {IdentityMatrix(), {}};
}

__device__ inline float MatrixElement(
        const GmMat3 &matrix,
        std::uint32_t row,
        std::uint32_t column) {
    const GmVec3 &basis = column == 0u
            ? matrix.basisX
            : (column == 1u
                       ? matrix.basisY
                       : matrix.basisZ);
    return row == 0u ? basis.x
                     : (row == 1u ? basis.y : basis.z);
}

__device__ inline void SetQuaternionComponent(
        GmQuat &value, std::uint32_t index, float component) {
    if (index == 0u) value.x = component;
    else if (index == 1u) value.y = component;
    else value.z = component;
}

__device__ inline GmQuat QuaternionFromMatrix(
        const GmMat3 &matrix) {
    GmQuat result{};
    const float trace =
            (MatrixElement(matrix, 0u, 0u) +
             MatrixElement(matrix, 1u, 1u)) +
            MatrixElement(matrix, 2u, 2u);
    if (trace > 0.0f) {
        const float root = cuda::exact::Sqrt(trace + 1.0f);
        const float scale = cuda::exact::Divide(0.5f, root);
        result.w = root * 0.5f;
        result.x =
                (MatrixElement(matrix, 2u, 1u) -
                 MatrixElement(matrix, 1u, 2u)) *
                scale;
        result.y =
                (MatrixElement(matrix, 0u, 2u) -
                 MatrixElement(matrix, 2u, 0u)) *
                scale;
        result.z =
                (MatrixElement(matrix, 1u, 0u) -
                 MatrixElement(matrix, 0u, 1u)) *
                scale;
        return result;
    }
    constexpr std::uint32_t nextAxis[3] = {1u, 2u, 0u};
    std::uint32_t dominant = 0u;
    if (MatrixElement(matrix, 0u, 0u) <
        MatrixElement(matrix, 1u, 1u)) {
        dominant = 1u;
    }
    if (MatrixElement(matrix, dominant, dominant) <
        MatrixElement(matrix, 2u, 2u)) {
        dominant = 2u;
    }
    const std::uint32_t next = nextAxis[dominant];
    const std::uint32_t final = nextAxis[next];
    const float root = cuda::exact::Sqrt(
            (MatrixElement(matrix, dominant, dominant) -
             (MatrixElement(matrix, final, final) +
              MatrixElement(matrix, next, next))) +
            1.0f);
    const float scale = cuda::exact::Divide(0.5f, root);
    SetQuaternionComponent(result, dominant, root * 0.5f);
    result.w =
            (MatrixElement(matrix, final, next) -
             MatrixElement(matrix, next, final)) *
            scale;
    SetQuaternionComponent(result, next,
            (MatrixElement(matrix, dominant, next) +
             MatrixElement(matrix, next, dominant)) *
            scale);
    SetQuaternionComponent(result, final,
            (MatrixElement(matrix, dominant, final) +
             MatrixElement(matrix, final, dominant)) *
            scale);
    return result;
}

__device__ inline void ResetDynaState(
        CHmsDyna::CHmsStateDyna &state) {
    state.linearSpeed = {};
    state.linearCorrectionSpeed = {};
    state.angularSpeed = {};
    state.force = {};
    state.torque = {};
    state.tweakedLinearSpeedValid = false;
    state.tweakedLinearSpeed = {};
}

__device__ inline void SetDynaLocation(
        CudaDynamicBodyState &body,
        const GmIso4 &location) {
    body.write.rotationQuat =
            QuaternionFromMatrix(location.rotation);
    body.write.rotation = location.rotation;
    body.write.position = location.translation;
    body.write.inverseInertiaWorld =
            cuda::dynamics::detail::Transpose(
                    body.write.rotation);
    body.write.inverseInertiaWorld =
            cuda::dynamics::detail::Compose(
                    body.write.inverseInertiaWorld,
                    body.parameters.bodyInertiaLike);
    body.write.inverseInertiaWorld =
            cuda::dynamics::detail::Compose(
                    body.write.inverseInertiaWorld,
                    body.write.rotation);
    body.current.rotationQuat = body.write.rotationQuat;
    body.current.rotation = body.write.rotation;
    body.current.position = body.write.position;
    body.current.inverseInertiaWorld =
            body.write.inverseInertiaWorld;
}

__device__ inline void ResetFrame(
        CudaVehicleCarFrameState &frame) {
    frame = {};
    frame.corpusIso = IdentityIso();
    frame.vehicleEvent0Value = 1u;
    frame.waterSplashEventCounter = 1u;
    frame.engineControlState =
            CSceneVehicleCarEngineControlState_Steady;
    frame.shiftDirection =
            CSceneVehicleCarShiftDirection_Up;
}

template<typename State>
__device__ inline void ResetWheelSnapshot(State &state) {
    const EPlugSurfaceMaterialId material =
            state.contactMaterial;
    state = {};
    state.contactMaterial = material;
    state.visualRotation = IdentityMatrix();
}

__device__ inline void ResetVehiclePassthrough(
        CudaCandidatePhysicsState &) {}

__device__ inline void ResetVehiclePassthrough(
        CudaCandidateState &candidate) {
    ResetFrame(candidate.vehiclePassthrough.asyncCurrent);
    ResetFrame(candidate.vehiclePassthrough.asyncPrevious);
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(candidate.vehicle); ++index) {
        ResetWheelSnapshot(
                candidate.vehiclePassthrough.
                        wheels[index].previousAsync);
        ResetWheelSnapshot(
                candidate.vehiclePassthrough.
                        wheels[index].currentAsync);
    }
}

__device__ inline void ResetWheel(
        CudaWheelState &wheel,
        const CudaPackedStaticConfigurationHeader *configuration) {
    wheel.realTime.maxReplacementY = 0.0f;
    wheel.realTime.damperVelocity = 0.0f;
    wheel.realTime.damperAbsorb =
            configuration->tuning.suspension.
                    wheelRestDamperAbsorb;
    wheel.currentPose = wheel.restPose;
    wheel.currentPose.translation.y -=
            wheel.realTime.damperAbsorb;
    wheel.surfaceMovedByUpdate = true;
    wheel.realTime.wheelAngularSpeed = 0.0f;
    wheel.realTime.wheelSpinAngle = 0.0f;
    wheel.realTime.contactPresent = false;
    wheel.realTime.contactFrame = IdentityMatrix();
    wheel.realTime.accumulatedContactNormal = {};
    wheel.realTime.currentVisualSteerAngle = 0.0f;
    wheel.realTime.targetVisualSteerAngle = 0.0f;
    wheel.realTime.slipping = false;
    wheel.realTime.contactNormalSampleCount = 0u;
    wheel.realTime.rejectedNormalContact = false;
    wheel.realTime.rejectedNormalContactPoint = {};
    ResetWheelSnapshot(wheel.previousPhysics);
    ResetWheelSnapshot(wheel.currentPhysics);
}

template<typename Candidate>
__device__ inline void VehicleReset(
        Candidate &candidate,
        const CudaPackedStaticConfigurationHeader *configuration) {
    CudaVehicleState &vehicle = candidate.vehicle;
    vehicle.controls.lowSpeedGateA = 0.0f;
    vehicle.controls.lowSpeedGateB = 0.0f;
    vehicle.controls.steeringControl = 0.0f;
    vehicle.controls.currentSteering = 0.0f;
    vehicle.controls.specialContactResponseGate = 0.0f;
    vehicle.contacts.specialContactImpulseCooldownUntil = 0u;
    vehicle.controls.forcedLowSpeedFriction = false;
    vehicle.water.splashPending = false;

    ResetVehiclePassthrough(candidate);
    ResetFrame(vehicle.frameHistory.physicsPrevious);
    ResetFrame(vehicle.frameHistory.physicsCurrent);
    vehicle.frameHistory.physicsCurrent.vehicleEvent0Value =
            vehicle.vehicleEvents[0].value;
    vehicle.frameHistory.physicsPrevious.vehicleEvent0Value =
            vehicle.vehicleEvents[0].value;
    vehicle.frameHistory.physicsCurrent.waterSplashEventCounter =
            vehicle.vehicleEvents[1].value;
    vehicle.frameHistory.physicsPrevious.waterSplashEventCounter =
            vehicle.vehicleEvents[1].value;

    vehicle.turbo.progressRatio = 0.0f;
    vehicle.turbo.type =
            CSceneVehicleCar::ETurboType_Inactive;
    vehicle.turbo.impulseScale = 0.0f;
    vehicle.airControl.refreshMemory = false;
    vehicle.airControl.memoryTick = 0u;
    vehicle.airControl.memoryAngular = {};
    vehicle.radiusSteering.steerAngle = 0.0f;
    vehicle.radiusSteering.phase =
            CSceneVehicleCarRadiusSteeringPhase_Idle;
    vehicle.radiusSteering.previousSteerSign = 0.0f;
    vehicle.slipMemory.steeringMemoryTick = UINT32_MAX;
    vehicle.slipMemory.active = false;
    vehicle.slipMemory.lastTick = UINT32_MAX;
    vehicle.slipMemory.startTick = UINT32_MAX;
    vehicle.gearedDrive.wheelSpeedOverrideActive = false;
    vehicle.gearedDrive.burnoutStartTick = UINT32_MAX;
    vehicle.gearedDrive.burnoutExitStartTick = UINT32_MAX;
    vehicle.gearedDrive.frameIso = IdentityIso();
    vehicle.gearedDrive.burnoutContactNormal = {};
    vehicle.gearedDrive.localSpeed = {};
    vehicle.gearedDrive.burnoutPhase =
            CSceneVehicleCarBurnoutPhase_Inactive;
    vehicle.gearedDrive.engineState =
            CSceneVehicleCarEngineControlState_Steady;
    vehicle.gearedDrive.inputWindowExceeded = false;
    vehicle.gearedDrive.wheelDriveSpeedInhibited = false;

    vehicle.contacts.bodyContactPresent = false;
    vehicle.contacts.lateralSlowDownContactActive = false;
    vehicle.contacts.lateralSlowDownLastTick = UINT32_MAX;
    vehicle.controls.noGroundFrictionGuard = false;
    vehicle.contacts.frontWheelImpactState =
            CSceneVehicleCarImpactState_None;
    vehicle.contacts.rearWheelImpactState =
            CSceneVehicleCarImpactState_None;
    vehicle.contacts.bodyImpactState =
            CSceneVehicleCarImpactState_None;
    vehicle.contacts.lastWheelContactMaterial =
            EPlugSurfaceMaterialId_Concrete;
    vehicle.contacts.lastBodyContactMaterial =
            EPlugSurfaceMaterialId_Concrete;
    vehicle.contacts.peakRearWheelImpactState =
            CSceneVehicleCarImpactState_None;
    vehicle.contacts.peakFrontWheelImpactState =
            CSceneVehicleCarImpactState_None;
    vehicle.contacts.peakBodyImpactState =
            CSceneVehicleCarImpactState_None;
    vehicle.contacts.frontWheelImpactBucket = 0.0f;
    vehicle.contacts.peakWheelImpactMaterial =
            EPlugSurfaceMaterialId_Concrete;
    vehicle.contacts.rearWheelImpactBucket = 0.0f;
    vehicle.contacts.peakBodyImpactMaterial =
            EPlugSurfaceMaterialId_Concrete;
    vehicle.contacts.bodyImpactBucket = 0.0f;
    vehicle.lastComputeForcesTick = 0u;
    vehicle.forceAccumulators = {};
    vehicle.contacts.bodyContactPointSum = {};
    vehicle.contacts.bodyContactNormalSum = {};
    vehicle.contacts.bodyContactCount = 0u;
    vehicle.contacts.wheelContactCount = 0u;
    vehicle.feedback.surfaceAccumulator = 0.0f;
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        ResetWheel(vehicle.wheels.values[index], configuration);
    }
    vehicle.engine.useLowSpeedGateB = false;
    vehicle.engine.engineInputMemory = 0.0f;
    vehicle.engine.gearIndex = 1;
    vehicle.engine.targetTransmissionInput = 0.0f;
    vehicle.engine.lowSpeedFeedbackForce = 0.0f;
    vehicle.engine.shiftCooldown = 0.0f;
    vehicle.engine.slipRpmScale = 1.0f;
    for (GmSpring<float> &spring : vehicle.dynaPartSprings) {
        spring.value = 0.0f;
        spring.target = 0.0f;
        spring.velocity = 0.0f;
    }
    candidate.body.physicalParameters.
            vehicleContactFeedbackScale =
            configuration->tuning.bodyAirResponse.
                    groundedSolidFeedback1;
    candidate.body.physicalParameters.linearFluidFriction =
            0.0f;
}

}  // namespace detail

__device__ inline void ApplyControls(
        CudaCandidatePhysicsState &candidate,
        const ReplayVehicleControlState &controls) {
    candidate.vehicle.controls.lowSpeedGateA =
            controls.lowSpeedGateA;
    candidate.vehicle.controls.lowSpeedGateB =
            controls.lowSpeedGateB;
    candidate.vehicle.controls.steeringControl =
            controls.steering;
    candidate.vehicle.frameHistory.physicsCurrent.lowSpeedGateA =
            controls.lowSpeedGateA;
    candidate.vehicle.frameHistory.physicsCurrent.lowSpeedGateB =
            controls.lowSpeedGateB;
    candidate.vehicle.frameHistory.physicsCurrent.steeringControl =
            controls.steering;
}

__device__ inline void RefreshPhysicalParameters(
        CudaCandidatePhysicsState &candidate) {
    candidate.body.physicalParameters.mass =
            candidate.body.parameters.mass;
    candidate.body.physicalParameters.impulseInertia =
            candidate.body.parameters.bodyInertiaLike;
    candidate.body.physicalParameters.linearFluidFriction =
            candidate.body.parameters.linearDampingScale;
    candidate.body.physicalParameters.physicalResponseCoefA =
            candidate.body.parameters.angularDampingScale;
    candidate.body.physicalParameters.physicalResponseCoefB =
            candidate.body.parameters.maxStepDistance;
    candidate.body.physicalParameters.
            vehicleContactFeedbackScale =
            candidate.body.parameters.forceScale;
    candidate.body.physicalParameters.localCenterOfMass =
            candidate.body.parameters.localCenterOfMass;
}

__device__ inline void PrepareSteadyStep(
        CudaCandidatePhysicsState &candidate,
        const CudaControlTick &tick) {
    ApplyControls(candidate, tick.controls);
    RefreshPhysicalParameters(candidate);
}

template<typename Candidate>
__device__ inline void PrepareStep(
        Candidate &candidate,
        const CudaControlTick &tick,
        const CudaPackedStaticConfigurationHeader *configuration) {
    ApplyControls(candidate, tick.controls);
    if ((tick.actionFlags &
         CudaControlActionEstablishRaceSpawn) != 0u) {
        const GmIso4 spawn = {
                candidate.body.current.rotation,
                candidate.body.current.position,
        };
        candidate.vehicle.gearedDrive.frameIso = spawn;
        candidate.vehicle.slipMemory.active = false;
        candidate.vehicle.slipMemory.lastTick = UINT32_MAX;
        candidate.vehicle.slipMemory.startTick = UINT32_MAX;
        candidate.race.player.previousSpawnLocation = spawn;
        candidate.race.player.currentSpawnLocation = spawn;
        candidate.race.playerSpawnLocation.present = true;
        candidate.race.playerSpawnLocation.value = spawn;
        candidate.race.currentSpawnLocationInitialized = true;
    }
    if ((tick.actionFlags &
         CudaControlActionEnableRaceSimulation) != 0u) {
        candidate.vehicle.integration.updateWheelVisuals = true;
        candidate.vehicle.integration.integrateWheels = true;
        candidate.vehicle.integration.integrateEngine = true;
        candidate.vehicle.integration.zeroHorizontalSpeed = false;
        candidate.vehicle.integration.speedBlocked = false;
        candidate.vehicle.integration.speedBlockedSecondary = false;
    }
    if ((tick.actionFlags &
         CudaControlActionResetAtRaceStart) != 0u) {
        candidate.vehicle.turbo.rouletteTickOrigin =
                tick.timeMs;
        candidate.vehicle.integration.speedBlocked = false;
        detail::VehicleReset(candidate, configuration);
    }
    RefreshPhysicalParameters(candidate);
}

template<typename Candidate>
__device__ inline bool Respawn(
        Candidate &candidate,
        const CudaPackedStaticConfigurationHeader *configuration) {
    if (!candidate.race.playerSpawnLocation.present) {
        return false;
    }
    const ReplayVehicleControlState controls = {
            candidate.vehicle.controls.lowSpeedGateA,
            candidate.vehicle.controls.lowSpeedGateB,
            candidate.vehicle.controls.steeringControl,
    };
    detail::VehicleReset(candidate, configuration);
    detail::ResetDynaState(candidate.body.current);
    detail::ResetDynaState(candidate.body.write);
    detail::ResetDynaState(candidate.body.temporary);
    candidate.body.collisionReplacements = {};
    candidate.vehicle.integration.speedBlockedSecondary = false;
    candidate.body.parameters.linearDampingScale = 0.0f;
    candidate.body.parameters.angularDampingScale = 0.0f;
    candidate.body.parameters.forceScale =
            configuration->tuning.bodyAirResponse.
                    groundedSolidFeedback1;
    detail::SetDynaLocation(
            candidate.body,
            candidate.race.player.currentSpawnLocation);
    ApplyControls(candidate, controls);
    return true;
}

}  // namespace forevervalidator::simulation::cuda::transition

#endif
