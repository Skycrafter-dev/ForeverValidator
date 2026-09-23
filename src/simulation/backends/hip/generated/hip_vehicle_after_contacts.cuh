// Generated from src/simulation/backends/cuda/cuda_vehicle_after_contacts.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_VEHICLE_AFTER_CONTACTS_CUH
#define FOREVERVALIDATOR_HIP_VEHICLE_AFTER_CONTACTS_CUH

#include "simulation/backends/hip/generated/hip_collision.cuh"

namespace forevervalidator::simulation::hip::vehicle {
namespace after_detail {

constexpr float VectorEpsilonSquared = 1.0e-10f;
constexpr float Pi = 3.1415927f;

__device__ inline GmVec3 Scale(
        const GmVec3 &value, float scale) {
    return {
            value.x * scale,
            value.y * scale,
            value.z * scale,
    };
}

__device__ inline float Smooth(
        float current, float target, float maximumStep) {
    if (target + maximumStep < current) {
        return current - maximumStep;
    }
    if (target - maximumStep > current) {
        return current + maximumStep;
    }
    return target;
}

__device__ inline const VehicleMaterialDefinition &Material(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t naturalId) {
    const std::uint32_t *remap =
            tuning::Section<std::uint32_t>(
                    configuration,
                    configuration->materialIndexByNaturalId);
    const VehicleMaterialDefinition *materials =
            tuning::Section<VehicleMaterialDefinition>(
                    configuration, configuration->materials);
    std::uint32_t index = 0u;
    if (naturalId <
        configuration->materialIndexByNaturalId.count) {
        index = remap[naturalId];
    }
    if (index >= configuration->materials.count) {
        index = 0u;
    }
    return materials[index];
}

__device__ inline void UpdateBodyContactSnapshot(
        HipCandidatePhysicsState &candidate) {
    auto &contacts = candidate.vehicle.contacts;
    auto &frame =
            candidate.vehicle.frameHistory.physicsCurrent;
    if (contacts.bodyContactCount == 0u) {
        frame.hasBodyContact = false;
        frame.bodyContactVerticalAngle = 0.0f;
        frame.bodyContactHorizontalAngle = 0.0f;
        return;
    }
    const float normalLength =
            (contacts.bodyContactNormalSum.x *
                     contacts.bodyContactNormalSum.x +
             contacts.bodyContactNormalSum.y *
                     contacts.bodyContactNormalSum.y) +
            contacts.bodyContactNormalSum.z *
                    contacts.bodyContactNormalSum.z;
    if (normalLength > VectorEpsilonSquared) {
        contacts.bodyContactNormalSum = Scale(
                contacts.bodyContactNormalSum,
                exact::Divide(1.0f, exact::Sqrt(normalLength)));
    }
    contacts.bodyContactNormalSum =
            Scale(contacts.bodyContactNormalSum, -1.0f);
    const float inverseCount = exact::Divide(
            1.0f,
            exact::FromUnsignedInteger(contacts.bodyContactCount));
    contacts.bodyContactPointSum =
            Scale(contacts.bodyContactPointSum, inverseCount);

    GmVec3 horizontal = {
            contacts.bodyContactNormalSum.x,
            contacts.bodyContactNormalSum.y,
            0.0f,
    };
    const float horizontalLength =
            horizontal.x * horizontal.x +
            horizontal.y * horizontal.y;
    if (horizontalLength > VectorEpsilonSquared) {
        horizontal = Scale(
                horizontal,
                exact::Divide(1.0f, exact::Sqrt(horizontalLength)));
    }
    frame.bodyContactHorizontalAngle =
            exact::Divide(
                    fabsf(exact::Atan2(
                            fabsf(horizontal.x), -horizontal.y)),
                    Pi);
    GmVec3 vertical = {
            0.0f,
            contacts.bodyContactNormalSum.y,
            contacts.bodyContactNormalSum.z,
    };
    const float verticalLength =
            vertical.y * vertical.y +
            vertical.z * vertical.z;
    if (verticalLength > VectorEpsilonSquared) {
        vertical = Scale(
                vertical,
                exact::Divide(1.0f, exact::Sqrt(verticalLength)));
    }
    frame.bodyContactVerticalAngle =
            exact::Divide(
                    fabsf(exact::Atan2(
                            fabsf(vertical.z), -vertical.y)),
                    Pi);
    frame.bodyContactZPositive = 0.0f < vertical.z;
    frame.hasBodyContact = true;
}

__device__ inline void UpdateWheelSnapshot(
        HipCandidatePhysicsState &candidate,
        HipWheelState &wheel) {
    wheel.previousPhysics = wheel.currentPhysics;
    auto &snapshot = wheel.currentPhysics;
    snapshot.damperAbsorb = wheel.realTime.damperAbsorb;
    snapshot.wheelSpinAngle = wheel.realTime.wheelSpinAngle;
    snapshot.currentVisualSteerAngle =
            wheel.realTime.currentVisualSteerAngle;
    snapshot.contactMaterial = wheel.realTime.contactMaterial;
    snapshot.contactPresent = wheel.realTime.contactPresent;
    snapshot.slipping = wheel.realTime.slipping;
    snapshot.localSurfacePoint =
            wheel.currentPose.translation;
    snapshot.localSurfacePoint.y -= wheel.rollingRadius;
    snapshot.worldSurfacePoint = collision::detail::TransformPoint(
            candidate.vehicle.frameHistory.physicsCurrent.corpusIso,
            snapshot.localSurfacePoint);
    snapshot.visualRotation = wheel.realTime.visualRotation;
    snapshot.rejectedNormalContact =
            wheel.realTime.rejectedNormalContact;
    snapshot.rejectedNormalContactPoint =
            wheel.realTime.rejectedNormalContactPoint;
}

__device__ inline void UpdateMaterialFeedback(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration) {
    auto &vehicle = candidate.vehicle;
    const VehicleMaterialDefinition &material =
            Material(configuration,
                     static_cast<std::uint32_t>(
                             vehicle.contacts.
                                     lastWheelContactMaterial));
    const float absoluteForward = fabsf(
            vehicle.frameHistory.physicsCurrent.forwardSpeed);
    float depth = 0.0f;
    if (vehicle.airControl.refreshMemory) {
        const float scaled =
                material.fakeContactSpeedScale *
                absoluteForward;
        depth = material.fakeContactDepthMax;
        if (!(depth < scaled)) depth = scaled;
    }
    const float targetIntensity =
            material.feedbackScale * depth;
    float cappedIntensity = 0.15f;
    if (targetIntensity < cappedIntensity) {
        cappedIntensity = targetIntensity;
    }
    const float speedTarget = exact::Divide(
            absoluteForward, material.feedbackSpeedDivisor);
    auto &frame = vehicle.frameHistory.physicsCurrent;
    const float feedbackSpeed = Smooth(
            frame.materialFeedbackSpeed,
            speedTarget, 0.5f);
    frame.materialFeedbackSpeed = feedbackSpeed;
    const float feedbackIntensity = Smooth(
            frame.materialFeedbackIntensity,
            cappedIntensity, 0.05f);
    frame.materialFeedbackIntensity = feedbackIntensity;
    if (vehicle.contacts.bodyImpactState >=
                CSceneVehicleCarImpactState_High ||
        (vehicle.contacts.frontWheelImpactState >=
                 CSceneVehicleCarImpactState_High &&
         vehicle.contacts.rearWheelImpactState !=
                 CSceneVehicleCarImpactState_None)) {
        frame.materialFeedbackSpeed = 2.0f;
        if (2.0f <= feedbackSpeed) {
            frame.materialFeedbackSpeed = feedbackSpeed;
        }
        frame.materialFeedbackIntensity = 1.0f;
        if (1.0f <= feedbackIntensity) {
            frame.materialFeedbackIntensity =
                    feedbackIntensity;
        }
        return;
    }
    if (vehicle.contacts.bodyImpactState ==
                CSceneVehicleCarImpactState_None &&
        (vehicle.contacts.frontWheelImpactState ==
                 CSceneVehicleCarImpactState_None ||
         vehicle.contacts.rearWheelImpactState ==
                 CSceneVehicleCarImpactState_None)) {
        return;
    }
    frame.materialFeedbackSpeed = 4.0f;
    if (4.0f <= feedbackSpeed) {
        frame.materialFeedbackSpeed = feedbackSpeed;
    }
    frame.materialFeedbackIntensity = 0.66f;
    if (0.66f <= feedbackIntensity) {
        frame.materialFeedbackIntensity = feedbackIntensity;
    }
}

}  // namespace after_detail

__device__ inline void AfterContacts(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration) {
    auto &vehicle = candidate.vehicle;
    vehicle.frameHistory.physicsPrevious =
            vehicle.frameHistory.physicsCurrent;
    for (std::uint32_t wheel = 0u;
         wheel < facts::WheelCount(vehicle); ++wheel) {
        vehicle.wheels.values[wheel].previousPhysics =
                vehicle.wheels.values[wheel].currentPhysics;
    }
    const GmVec3 worldLinearSpeed =
            candidate.body.current.linearSpeed;
    const GmVec3 linearSpeed = {
            collision::detail::Dot(
                    candidate.body.current.rotation.basisX,
                    worldLinearSpeed),
            collision::detail::Dot(
                    candidate.body.current.rotation.basisY,
                    worldLinearSpeed),
            collision::detail::Dot(
                    candidate.body.current.rotation.basisZ,
                    worldLinearSpeed),
    };
    auto &frame = vehicle.frameHistory.physicsCurrent;
    frame.forwardSpeed = linearSpeed.z;
    frame.sideSpeed = linearSpeed.x;
    frame.engineInputMemory =
            vehicle.engine.engineInputMemory;
    frame.turboActive =
            vehicle.turbo.type !=
            CSceneVehicleCar::ETurboType_Inactive;
    frame.turboProgressRatio = vehicle.turbo.progressRatio;
    frame.corpusIso = candidate.body.corpusLocalIso;
    frame.vehicleEvent0Value = vehicle.vehicleEvents[0].value;
    frame.waterSplashEventCounter =
            vehicle.vehicleEvents[1].value;
    frame.localLinearSpeed =
            collision::detail::TransformDirection(
                    frame.corpusIso.rotation, linearSpeed);
    frame.wheelSpeedOverrideActive =
            vehicle.gearedDrive.wheelSpeedOverrideActive;
    frame.airControlRefreshMemory =
            vehicle.airControl.refreshMemory;
    frame.surfaceFeedbackAccumulator =
            vehicle.feedback.surfaceAccumulator;
    frame.steeringControl = vehicle.controls.steeringControl;
    frame.lowSpeedGateB = vehicle.controls.lowSpeedGateB;
    frame.lowSpeedGateA = vehicle.controls.lowSpeedGateA;
    frame.engineControlState = vehicle.gearedDrive.engineState;
    frame.shiftDirection = vehicle.gearedDrive.shiftDirection;
    frame.feedbackSideSpringValue =
            vehicle.feedback.sideSpring.value;
    frame.feedbackForwardSpringValue =
            vehicle.feedback.forwardSpring.value;
    frame.feedbackRamp1 = vehicle.feedback.ramp1;
    frame.feedbackRamp0 = vehicle.feedback.ramp0;

    after_detail::UpdateBodyContactSnapshot(candidate);
    frame.hasWheelContact =
            vehicle.contacts.wheelContactCount != 0u;
    frame.noGroundFrictionGuard =
            vehicle.controls.noGroundFrictionGuard;
    vehicle.contacts.wheelContactCount = 0u;
    vehicle.contacts.bodyContactCount = 0u;
    vehicle.contacts.bodyContactPointSum = {};
    vehicle.contacts.bodyContactNormalSum = {};
    for (std::uint32_t wheel = 0u;
         wheel < facts::WheelCount(vehicle); ++wheel) {
        after_detail::UpdateWheelSnapshot(
                candidate, vehicle.wheels.values[wheel]);
    }
    after_detail::UpdateMaterialFeedback(
            candidate, configuration);
}

__device__ inline void AfterContactsWithoutSnapshots(
        HipCandidatePhysicsState &candidate) {
    auto &contacts = candidate.vehicle.contacts;
    contacts.wheelContactCount = 0u;
    contacts.bodyContactCount = 0u;
    contacts.bodyContactPointSum = {};
    contacts.bodyContactNormalSum = {};
}

}  // namespace forevervalidator::simulation::hip::vehicle

#endif
