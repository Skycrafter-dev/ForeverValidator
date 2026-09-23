// Generated from src/simulation/backends/cuda/cuda_collision_response.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_COLLISION_RESPONSE_CUH
#define FOREVERVALIDATOR_HIP_COLLISION_RESPONSE_CUH

#include "simulation/backends/hip/generated/hip_collision.cuh"
#include "simulation/backends/hip/generated/hip_dynamics.cuh"
#include "simulation/backends/hip/generated/hip_race.cuh"

namespace forevervalidator::simulation::hip::collision {

namespace response_detail {

constexpr float ScalarEpsilon = 1.0e-5f;

struct Contact {
    GmVec3 localNormal{};
    GmVec3 localPoint{};
    GmVec3 localSpeed{};
    GmVec3 replacement{};
    std::uint32_t peerMaterial = 0u;
    GmVec3 peerZAxis{};
    std::uint32_t peerCorpusId = 0u;
    std::uint32_t wheelIndex = UINT32_MAX;
    bool accepted = true;
};

__device__ inline GmVec3 TransposeDirection(
        const GmMat3 &rotation, const GmVec3 &value) {
    return {
            rotation.basisX.x * value.x +
                    rotation.basisX.y * value.y +
                    rotation.basisX.z * value.z,
            rotation.basisY.x * value.x +
                    rotation.basisY.y * value.y +
                    rotation.basisY.z * value.z,
            rotation.basisZ.x * value.x +
                    rotation.basisZ.y * value.y +
                    rotation.basisZ.z * value.z,
    };
}

__device__ inline GmVec3 LocalToWorldSideA(
        const GmMat3 &rotation, const GmVec3 &value) {
    return {
            (rotation.basisY.x * value.y +
             rotation.basisX.x * value.x) +
                    rotation.basisZ.x * value.z,
            (rotation.basisY.y * value.y +
             rotation.basisX.y * value.x) +
                    rotation.basisZ.y * value.z,
            (rotation.basisY.z * value.y +
             rotation.basisX.z * value.x) +
                    rotation.basisZ.z * value.z,
    };
}

__device__ inline GmVec3 WorldCenterOfMass(
        const HipCandidatePhysicsState &candidate) {
    return detail::TransformPoint(
            {candidate.body.current.rotation,
             candidate.body.current.position},
            candidate.body.parameters.localCenterOfMass);
}

__device__ inline GmVec3 SpeedAtPoint(
        const HipCandidatePhysicsState &candidate,
        const GmVec3 &point) {
    const auto &state = candidate.body.current;
    const GmVec3 center = WorldCenterOfMass(candidate);
    const float rx = point.x - center.x;
    const float ry = point.y - center.y;
    const float rz = point.z - center.z;
    return {
            state.linearSpeed.x +
                    (rz * state.angularSpeed.y -
                     state.angularSpeed.z * ry),
            state.linearSpeed.y +
                    (state.angularSpeed.z * rx -
                     rz * state.angularSpeed.x),
            state.linearSpeed.z +
                    (ry * state.angularSpeed.x -
                     rx * state.angularSpeed.y),
    };
}

__device__ inline std::uint32_t WheelIndexForShape(
        const HipVehicleCollisionShape &shape,
        const HipCandidatePhysicsState &candidate) {
    return shape.wheelIndex < facts::WheelCount(candidate.vehicle)
            ? shape.wheelIndex
            : UINT32_MAX;
}

template <typename Collision>
__device__ inline Contact MakeVehicleContact(
        const Collision &collision,
        const HipVehicleCollisionShape &shape,
        const HipSceneActor &actor,
        const HipPackedStaticConfigurationHeader *configuration,
        const HipCandidatePhysicsState &candidate) {
    const GmMat3 &rotation = candidate.body.current.rotation;
    const GmVec3 pointFromBody = {
            collision.contactPoint.x -
                    candidate.body.current.position.x,
            collision.contactPoint.y -
                    candidate.body.current.position.y,
            collision.contactPoint.z -
                    candidate.body.current.position.z,
    };
    Contact contact;
    contact.localNormal = TransposeDirection(
            rotation, collision.impulseNormal);
    contact.localPoint = TransposeDirection(
            rotation, pointFromBody);
    contact.localSpeed = TransposeDirection(
            rotation, SpeedAtPoint(candidate,
                                   collision.contactPoint));
    contact.replacement = TransposeDirection(
            rotation, detail::Negate(collision.separation));
    contact.peerMaterial = collision.materialB;
    contact.peerZAxis = TransposeDirection(
            candidate.body.write.rotation,
            actor.worldPose.rotation.basisZ);
    contact.peerCorpusId = actor.installationOrder + 1u;
    contact.wheelIndex = WheelIndexForShape(
            shape, candidate);
    contact.accepted = true;
    return contact;
}

template<bool CompactReplacements = false, typename Scratch>
__device__ inline void AddReplacement(
        HipCandidatePhysicsState &candidate,
        Scratch &scratch,
        const GmVec3 &replacement,
        bool &overflow) {
    if constexpr (CompactReplacements) {
        if (scratch.replacementOverflowCount >=
            HipCollisionReplacementCapacity) {
            overflow = true;
            return;
        }
        if (scratch.replacementOverflowCount++ == 0u) {
            scratch.replacementSumX = replacement.x;
            scratch.replacementSumY = replacement.y;
            scratch.replacementSumZ = replacement.z;
        } else {
            dynamics::AccumulateReplacement(
                    replacement,
                    scratch.replacementSumX,
                    scratch.replacementSumY,
                    scratch.replacementSumZ);
        }
        candidate.body.dynamicActive = true;
        return;
    }
    auto &items = candidate.body.collisionReplacements;
    if (items.count <
        sizeof(items.values) / sizeof(items.values[0])) {
        items.values[items.count++] = replacement;
        candidate.body.dynamicActive = true;
        return;
    }
    if (scratch.replacementOverflowCount >=
        HipCollisionReplacementOverflowCapacity) {
        overflow = true;
        return;
    }
    detail::ReplacementOverflowAt(
            scratch,
            scratch.replacementOverflowCount++) = replacement;
    candidate.body.dynamicActive = true;
}

__device__ inline void Accumulate(
        GmVec3 &target, const GmVec3 &value) {
    target.x = value.x + target.x;
    target.y = value.y + target.y;
    target.z = value.z + target.z;
}

__device__ inline void AddPointImpulse(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration,
        const GmVec3 &localImpulse,
        const GmVec3 &localPoint) {
    auto &state = candidate.body.current;
    const GmVec3 worldImpulse = detail::TransformDirection(
            state.rotation, localImpulse);
    const GmIso4 statePose = {
            state.rotation, state.position};
    const GmVec3 worldPoint =
            detail::TransformPoint(statePose, localPoint);
    const GmVec3 worldCom = detail::TransformPoint(
            statePose,
            candidate.body.parameters.localCenterOfMass);
    const float inverseMass = exact::Divide(
            1.0f, candidate.body.parameters.mass);
    GmVec3 linear = {
            state.linearSpeed.x +
                    worldImpulse.x * inverseMass,
            state.linearSpeed.y +
                    worldImpulse.y * inverseMass,
            state.linearSpeed.z +
                    worldImpulse.z * inverseMass,
    };
    const float oldLength =
            (state.linearSpeed.y * state.linearSpeed.y +
             state.linearSpeed.x * state.linearSpeed.x) +
            state.linearSpeed.z * state.linearSpeed.z;
    const float newLength =
            (linear.y * linear.y + linear.x * linear.x) +
            linear.z * linear.z;
    if (oldLength < newLength &&
        configuration->tuning.contactResponse.
                        pointImpulseLinearSpeedGrowthLimitSq <
                newLength - oldLength) {
        linear = {};
    }
    state.linearSpeed = linear;

    const GmVec3 lever = {
            worldPoint.x - worldCom.x,
            worldPoint.y - worldCom.y,
            worldPoint.z - worldCom.z,
    };
    GmVec3 angular = {
            lever.y * worldImpulse.z -
                    lever.z * worldImpulse.y,
            lever.z * worldImpulse.x -
                    worldImpulse.z * lever.x,
            worldImpulse.y * lever.x -
                    lever.y * worldImpulse.x,
    };
    angular = detail::TransformDirection(
            state.inverseInertiaWorld, angular);
    const float scale = configuration->tuning.contactResponse.
            pointImpulseAngularScale;
    angular.x = scale * angular.x;
    angular.y = scale * angular.y;
    angular.z = scale * angular.z;
    angular.y =
            configuration->tuning.contactResponse.
                    pointImpulseAngularYScale * angular.y;
    state.angularSpeed.x += angular.x;
    state.angularSpeed.y += angular.y;
    state.angularSpeed.z += angular.z;

    const float maximum = configuration->tuning.contactResponse.
            pointImpulseAngularSpeedMax;
    const float angularLength =
            (state.angularSpeed.y * state.angularSpeed.y +
             state.angularSpeed.x * state.angularSpeed.x) +
            state.angularSpeed.z * state.angularSpeed.z;
    if (angularLength > maximum * maximum) {
        const float length = exact::Sqrt(angularLength);
        const float clamp = exact::FromDouble(
                static_cast<double>(maximum) /
                static_cast<double>(length));
        state.angularSpeed.x *= clamp;
        state.angularSpeed.y *= clamp;
        state.angularSpeed.z *= clamp;
    }
    Accumulate(candidate.vehicle.forceAccumulators.impulse,
               localImpulse);
    candidate.body.dynamicActive = true;
}

__device__ inline void ComputeAndApplyContactImpulse(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration,
        float restitution,
        const GmVec3 &speed,
        const GmVec3 &normal,
        const GmVec3 &point) {
    const GmVec3 lever = {
            point.x -
                    candidate.body.parameters.localCenterOfMass.x,
            point.y -
                    candidate.body.parameters.localCenterOfMass.y,
            point.z -
                    candidate.body.parameters.localCenterOfMass.z,
    };
    GmVec3 angularUnit = {
            normal.z * lever.y - normal.y * lever.z,
            normal.x * lever.z - lever.x * normal.z,
            normal.y * lever.x - normal.x * lever.y,
    };
    angularUnit = detail::TransformDirection(
            candidate.body.parameters.bodyInertiaLike,
            angularUnit);
    const GmVec3 angularAtPoint = {
            lever.z * angularUnit.y -
                    angularUnit.z * lever.y,
            lever.x * angularUnit.z -
                    lever.z * angularUnit.x,
            angularUnit.x * lever.y -
                    lever.x * angularUnit.y,
    };
    const float speedAlongNormal =
            normal.y * speed.y + normal.x * speed.x +
            normal.z * speed.z;
    const float numerator =
            (restitution - 1.0f) * speedAlongNormal;
    const float angularMass =
            normal.x * angularAtPoint.x +
            normal.y * angularAtPoint.y +
            normal.z * angularAtPoint.z;
    const float denominator = exact::Divide(
            1.0f, candidate.body.parameters.mass) + angularMass;
    GmVec3 impulse{};
    if (!(fabsf(denominator) < ScalarEpsilon)) {
        const float impulseScale = exact::Divide(numerator, denominator);
        impulse = {
                normal.x * impulseScale,
                normal.y * impulseScale,
                impulseScale * normal.z,
        };
    }
    AddPointImpulse(candidate, configuration, impulse, point);
}

struct MaterialData {
    float friction = 1.0f;
    float restitution = 0.5f;
};

__device__ inline MaterialData Material(
        std::uint32_t material) {
    switch (material) {
    case EPlugSurfaceMaterialId_Ice:
    case EPlugSurfaceMaterialId_Rubber:
    case EPlugSurfaceMaterialId_Test:
        return {0.0f, 0.0f};
    case EPlugSurfaceMaterialId_SlidingRubber:
        return {0.0f, -0.5f};
    case EPlugSurfaceMaterialId_GolfBall:
        return {1.0f, 0.95f};
    case EPlugSurfaceMaterialId_GolfWall:
    case EPlugSurfaceMaterialId_GolfGround:
        return {1.0f, 0.8f};
    default:
        return {1.0f, 0.5f};
    }
}

__device__ inline float Restitution(
        const MaterialData &left,
        const MaterialData &right) {
    if (left.restitution > 0.0f) {
        if (right.restitution > 0.0f) {
            return right.restitution * left.restitution;
        }
        return right.restitution;
    }
    if (right.restitution > 0.0f) {
        return left.restitution;
    }
    return right.restitution + left.restitution;
}

__device__ inline GmVec3 ClampTangentSpeed(
        const GmVec3 &speed,
        const GmVec3 &normal,
        float friction) {
    const float normalSpeed =
            normal.z * speed.z + normal.x * speed.x +
            normal.y * speed.y;
    const GmVec3 normalComponent = {
            normal.x * normalSpeed,
            normal.y * normalSpeed,
            normalSpeed * normal.z,
    };
    GmVec3 tangent = {
            speed.x - normalComponent.x,
            speed.y - normalComponent.y,
            speed.z - normalComponent.z,
    };
    const float normalLength = exact::Sqrt(
            (normalComponent.y * normalComponent.y +
             normalComponent.x * normalComponent.x) +
            normalComponent.z * normalComponent.z);
    const float tangentLength = exact::Sqrt(
            (tangent.y * tangent.y + tangent.x * tangent.x) +
            tangent.z * tangent.z);
    const float limit = normalLength * friction;
    if (tangentLength > limit) {
        const float scale = exact::Divide(limit, tangentLength);
        tangent.x = scale * tangent.x;
        tangent.y *= scale;
        tangent.z = scale * tangent.z;
    }
    return {
            tangent.x + normalComponent.x,
            tangent.y + normalComponent.y,
            tangent.z + normalComponent.z,
    };
}

__device__ inline void AddWorldImpulseAtPoint(
        HipCandidatePhysicsState &candidate,
        const GmVec3 &impulse,
        const GmVec3 &point) {
    auto &state = candidate.body.current;
    const float inverseMass = exact::Divide(
            1.0f, candidate.body.parameters.mass);
    state.linearSpeed.x += impulse.x * inverseMass;
    state.linearSpeed.y += impulse.y * inverseMass;
    state.linearSpeed.z =
            impulse.z * inverseMass + state.linearSpeed.z;
    if (dynamics::IsFullAngular(candidate.body)) {
        const GmVec3 center = WorldCenterOfMass(candidate);
        const GmVec3 lever = {
                point.x - center.x,
                point.y - center.y,
                point.z - center.z,
        };
        GmVec3 angular = {
                lever.y * impulse.z -
                        lever.z * impulse.y,
                lever.z * impulse.x -
                        lever.x * impulse.z,
                lever.x * impulse.y -
                        lever.y * impulse.x,
        };
        angular = detail::TransformDirection(
                state.inverseInertiaWorld, angular);
        state.angularSpeed.x += angular.x;
        state.angularSpeed.y += angular.y;
        state.angularSpeed.z =
                angular.z + state.angularSpeed.z;
    }
    candidate.body.dynamicActive = true;
}

template <typename Collision>
__device__ inline void ApplyGenericImpulse(
        HipCandidatePhysicsState &candidate,
        const Collision &collision,
        const GmVec3 &originalSpeed) {
    const MaterialData materialA =
            Material(collision.materialA);
    const MaterialData materialB =
            Material(collision.materialB);
    const float restitution =
            Restitution(materialA, materialB);
    const GmVec3 adjusted = ClampTangentSpeed(
            originalSpeed, collision.impulseNormal,
            materialA.friction * materialB.friction);
    const GmVec3 negative = {
            -adjusted.x, -adjusted.y, -adjusted.z};
    const float speed = exact::Sqrt(
            (negative.y * negative.y +
             negative.x * negative.x) +
            negative.z * negative.z);
    if (!(speed > ScalarEpsilon)) return;
    const float inverseSpeed = exact::Divide(1.0f, speed);
    const GmVec3 direction = {
            inverseSpeed * negative.x,
            negative.y * inverseSpeed,
            inverseSpeed * negative.z,
    };
    float denominator = exact::Divide(
            1.0f, candidate.body.parameters.mass);
    if (dynamics::IsFullAngular(candidate.body)) {
        const GmVec3 center = WorldCenterOfMass(candidate);
        const GmVec3 lever = {
                collision.contactPoint.x - center.x,
                collision.contactPoint.y - center.y,
                collision.contactPoint.z - center.z,
        };
        GmVec3 angular = detail::Cross(lever, direction);
        angular = detail::TransformDirection(
                candidate.body.current.inverseInertiaWorld,
                angular);
        const GmVec3 atPoint =
                detail::Cross(angular, lever);
        denominator =
                (atPoint.y * direction.y +
                 atPoint.x * direction.x) +
                atPoint.z * direction.z +
                denominator;
    }
    const float magnitude =
            exact::Divide(
                    speed * (restitution + 1.0f), denominator);
    const GmVec3 impulse = {
            direction.x * magnitude,
            direction.y * magnitude,
            magnitude * direction.z,
    };
    AddWorldImpulseAtPoint(
            candidate, impulse, collision.contactPoint);
}

__device__ inline float WheelContactImpulse(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t material) {
    return material ==
                   static_cast<std::uint32_t>(
                           EPlugSurfaceMaterialId_Metal)
            ? configuration->tuning.contactResponse.
                      wheelContactImpulseMetal
            : configuration->tuning.contactResponse.
                      wheelContactImpulseOther;
}

__device__ inline float BodyContactImpulse(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t material) {
    return material ==
                   static_cast<std::uint32_t>(
                           EPlugSurfaceMaterialId_Metal)
            ? configuration->tuning.contactResponse.
                      bodyContactImpulseMetal
            : configuration->tuning.contactResponse.
                      bodyContactImpulseOther;
}

__device__ inline float BodyContactTangentLimit(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t material) {
    return material ==
                   static_cast<std::uint32_t>(
                           EPlugSurfaceMaterialId_Metal)
            ? configuration->tuning.contactResponse.
                      bodyContactTangentLimitMetal
            : configuration->tuning.contactResponse.
                      bodyContactTangentLimitOther;
}

__device__ inline void ApplyBodyContactImpulse(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration,
        Contact &contact) {
    const float speedAlongNormal =
            contact.localNormal.y * contact.localSpeed.y +
            contact.localNormal.x * contact.localSpeed.x +
            contact.localNormal.z * contact.localSpeed.z;
    if (speedAlongNormal < 0.0f) {
        const float restitution =
                -BodyContactImpulse(
                        configuration, contact.peerMaterial);
        const float tangentLimit =
                BodyContactTangentLimit(
                        configuration, contact.peerMaterial);
        const GmVec3 normalSpeed = {
                contact.localNormal.x * speedAlongNormal,
                contact.localNormal.y * speedAlongNormal,
                speedAlongNormal * contact.localNormal.z,
        };
        GmVec3 tangentSpeed = {
                contact.localSpeed.x - normalSpeed.x,
                contact.localSpeed.y - normalSpeed.y,
                contact.localSpeed.z - normalSpeed.z,
        };
        const float normalLength = exact::Sqrt(
                (normalSpeed.y * normalSpeed.y +
                 normalSpeed.x * normalSpeed.x) +
                normalSpeed.z * normalSpeed.z);
        const float tangentLength = exact::Sqrt(
                (tangentSpeed.y * tangentSpeed.y +
                 tangentSpeed.x * tangentSpeed.x) +
                tangentSpeed.z * tangentSpeed.z);
        const float tangentMaximum =
                normalLength * tangentLimit;
        if (tangentLength > tangentMaximum) {
            const float scale = exact::Divide(
                    tangentMaximum, tangentLength);
            tangentSpeed.x = scale * tangentSpeed.x;
            tangentSpeed.y *= scale;
            tangentSpeed.z = scale * tangentSpeed.z;
        }
        GmVec3 impulseNormal = {
                -(tangentSpeed.x + normalSpeed.x),
                -(tangentSpeed.y + normalSpeed.y),
                -(tangentSpeed.z + normalSpeed.z),
        };
        const float impulseLength = exact::Sqrt(
                (impulseNormal.y * impulseNormal.y +
                 impulseNormal.x * impulseNormal.x) +
                impulseNormal.z * impulseNormal.z);
        if (ScalarEpsilon < impulseLength) {
            const float inverse = exact::Divide(1.0f, impulseLength);
            impulseNormal.x = inverse * impulseNormal.x;
            impulseNormal.y *= inverse;
            impulseNormal.z = inverse * impulseNormal.z;
            ComputeAndApplyContactImpulse(
                    candidate, configuration, restitution,
                    contact.localSpeed, impulseNormal,
                    contact.localPoint);
        }
    }
    contact.accepted = false;
}

__device__ inline bool ApplyWheelReplacement(
        HipWheelState &wheel,
        Contact &contact,
        const HipPackedStaticConfigurationHeader *configuration) {
    const float replacementY = contact.replacement.y;
    bool blocked = false;
    if (replacementY > 0.0f) {
        float candidate = replacementY;
        const float maximum =
                configuration->tuning.suspension.
                        damperModulationMaxAbsorb;
        if (maximum >= -ScalarEpsilon) {
            const float clamped =
                    wheel.realTime.damperAbsorb - maximum;
            if (clamped <= replacementY) {
                candidate = clamped;
                blocked = true;
            }
        }
        wheel.realTime.maxReplacementY =
                candidate > wheel.realTime.maxReplacementY
                ? candidate
                : wheel.realTime.maxReplacementY;
        const float zero = 0.0f * candidate;
        contact.replacement.x -= zero;
        contact.replacement.y -= candidate;
        contact.replacement.z -= zero;
    } else {
        blocked = true;
    }
    return blocked;
}

__device__ inline void ApplyWheelImpulse(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration,
        HipWheelState &wheel,
        Contact &contact,
        bool replacementBlocked) {
    const float normalSpeed =
            contact.localNormal.y * contact.localSpeed.y +
            contact.localNormal.x * contact.localSpeed.x +
            contact.localNormal.z * contact.localSpeed.z;
    if (!(normalSpeed < 0.0f)) return;
    const GmVec3 component = {
            contact.localNormal.x * normalSpeed,
            contact.localNormal.y * normalSpeed,
            normalSpeed * contact.localNormal.z,
    };
    if (replacementBlocked || !(component.y < 0.0f)) {
        ComputeAndApplyContactImpulse(
                candidate, configuration,
                -WheelContactImpulse(
                        configuration, contact.peerMaterial),
                contact.localSpeed, contact.localNormal,
                contact.localPoint);
        return;
    }
    const float zero = 0.0f * component.y;
    const GmVec3 withoutVertical = {
            contact.localSpeed.x - zero,
            contact.localSpeed.y - component.y,
            contact.localSpeed.z - zero,
    };
    const float remainingNormal =
            contact.localNormal.y * withoutVertical.y +
            contact.localNormal.x * withoutVertical.x +
            contact.localNormal.z * withoutVertical.z;
    if (remainingNormal < 0.0f) {
        ComputeAndApplyContactImpulse(
                candidate, configuration,
                -WheelContactImpulse(
                        configuration, contact.peerMaterial),
                withoutVertical, contact.localNormal,
                wheel.currentPose.translation);
    }
}

__device__ inline void AbsorbWheel(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration,
        Contact &contact) {
    HipWheelState &wheel =
            candidate.vehicle.wheels.values[contact.wheelIndex];
    const float maximumNormalX =
            exact::Sin(3.1415927f * 0.25f);
    wheel.realTime.contactPresent =
            fabsf(contact.localNormal.x) < maximumNormalX;
    if (!wheel.realTime.contactPresent) {
        candidate.vehicle.contacts.
                lateralSlowDownContactActive = true;
        wheel.realTime.rejectedNormalContactPoint =
                contact.localPoint;
        wheel.realTime.rejectedNormalContact = true;
    } else {
        ++wheel.realTime.contactNormalSampleCount;
        Accumulate(wheel.realTime.accumulatedContactNormal,
                   contact.localNormal);
        wheel.realTime.contactMaterial =
                static_cast<EPlugSurfaceMaterialId>(
                        contact.peerMaterial);
    }
    contact.accepted = false;
    wheel.realTime.latestContactPoint = contact.localPoint;
    wheel.realTime.peerZAxisInCarLocal =
            contact.peerZAxis;
    wheel.realTime.peerCorpusId =
            CHmsCorpusId::FromValue(contact.peerCorpusId);
    if (::forevervalidator::simulation::hip::facts::
                WheelForceMode(configuration) !=
        static_cast<std::uint32_t>(
                CSceneVehicleCarWheelForceMode_FollowAbsorbWithImpulse)) {
        return;
    }
    const bool blocked = ApplyWheelReplacement(
            wheel, contact, configuration);
    if (wheel.realTime.contactPresent) {
        ApplyWheelImpulse(candidate, configuration, wheel,
                          contact, blocked);
        return;
    }
    const float speedAlongNormal =
            contact.localNormal.y * contact.localSpeed.y +
            contact.localNormal.x * contact.localSpeed.x +
            contact.localNormal.z * contact.localSpeed.z;
    if (speedAlongNormal < 0.0f) {
        const GmVec3 point = {
                contact.localPoint.x,
                candidate.body.parameters.localCenterOfMass.y,
                contact.localPoint.z,
        };
        ComputeAndApplyContactImpulse(
                candidate, configuration,
                -BodyContactImpulse(
                        configuration, contact.peerMaterial),
                contact.localSpeed, contact.localNormal, point);
    }
}

__device__ inline void AbsorbVehicle(
        HipCandidatePhysicsState &candidate,
        const HipPackedStaticConfigurationHeader *configuration,
        Contact &contact) {
    if (contact.peerMaterial ==
                static_cast<std::uint32_t>(
                        EPlugSurfaceMaterialId_Water) ||
        contact.peerMaterial ==
                static_cast<std::uint32_t>(
                        EPlugSurfaceMaterialId_GolfBall)) {
        contact.accepted = false;
        contact.replacement = {};
        return;
    }
    candidate.vehicle.airControl.refreshMemory = true;
    const float impact = fabsf(
            contact.localNormal.y * contact.localSpeed.y +
            contact.localNormal.x * contact.localSpeed.x +
            contact.localNormal.z * contact.localSpeed.z);
    if (contact.wheelIndex == UINT32_MAX ||
        !(contact.localNormal.y > 0.2f)) {
        candidate.vehicle.contacts.bodyImpactBucket += impact;
    } else {
        if (facts::WheelAxle(
                    configuration,
                    contact.wheelIndex) ==
            VehicleWheelAxle::Rear) {
            candidate.vehicle.contacts.rearWheelImpactBucket +=
                    impact;
        } else {
            candidate.vehicle.contacts.frontWheelImpactBucket =
                    impact +
                    candidate.vehicle.contacts.
                            frontWheelImpactBucket;
        }
    }
    if (contact.wheelIndex != UINT32_MAX) {
        candidate.vehicle.contacts.lastWheelContactMaterial =
                static_cast<EPlugSurfaceMaterialId>(
                        contact.peerMaterial);
        AbsorbWheel(candidate, configuration, contact);
        ++candidate.vehicle.contacts.wheelContactCount;
        return;
    }
    if (contact.localNormal.y < -0.75f &&
        configuration->tuning.handlingModel ==
                static_cast<std::uint32_t>(
                        CSceneVehicleCarHandlingModel_GearedDrive)) {
        const float along =
                contact.localNormal.y * contact.replacement.y +
                contact.localNormal.x * contact.replacement.x +
                contact.localNormal.z * contact.replacement.z;
        contact.replacement = {
                contact.localNormal.x * along,
                contact.localNormal.y * along,
                contact.localNormal.z * along,
        };
    }
    Accumulate(candidate.vehicle.contacts.bodyContactPointSum,
               contact.localPoint);
    Accumulate(candidate.vehicle.contacts.bodyContactNormalSum,
               contact.localNormal);
    ++candidate.vehicle.contacts.bodyContactCount;
    candidate.vehicle.contacts.bodyContactPresent = true;
    candidate.vehicle.contacts.lastBodyContactMaterial =
            static_cast<EPlugSurfaceMaterialId>(
                    contact.peerMaterial);
    if (::forevervalidator::simulation::hip::facts::
                WheelForceMode(configuration) ==
        static_cast<std::uint32_t>(
                CSceneVehicleCarWheelForceMode_FollowAbsorbWithImpulse)) {
        ApplyBodyContactImpulse(
                candidate, configuration, contact);
    }
}

}  // namespace response_detail

template <
        bool TrackDiagnostics = true,
        bool TrustedInputs = false,
        bool CompactReplacements = false,
        typename Scratch = HipCollisionScratch>
__device__ inline Status Respond(
        const HipPackedSceneHeader *scene,
        const HipPackedStaticConfigurationHeader *configuration,
        HipCandidatePhysicsState &candidate,
        Scratch &scratch) {
    if constexpr (!TrustedInputs) {
        if (scene == nullptr || configuration == nullptr) {
            return Status::InvalidScene;
        }
    }
    const HipSceneActor *actors =
            detail::SceneSection<HipSceneActor>(
                    scene, scene->actors);
    const HipVehicleCollisionShape *shapes =
            tuning::Section<HipVehicleCollisionShape>(
                    configuration,
                    configuration->collisionShapes);
    bool overflow = false;
    for (std::uint32_t index = 0u;
        index < scratch.collisionCount; ++index) {
        decltype(auto) collision =
                detail::OrderedCollisionAt(scratch, index);
        if (collision.staticActorIndex >= scene->actors.count ||
            collision.movingShapeIndex >=
                    configuration->collisionShapes.count) {
            return Status::InvalidScene;
        }
        const std::uint32_t targetGroup =
                static_cast<std::uint32_t>(
                        actors[collision.staticActorIndex].
                                itemProperties.collisionGroup);
        if (targetGroup == 1u) {
            race::OnTriggerContact(
                    candidate,
                    actors[collision.staticActorIndex]);
            continue;
        }
        if (targetGroup != 4u) {
            return Status::UnsupportedGeometry;
        }
        response_detail::Contact contact =
                response_detail::MakeVehicleContact(
                        collision,
                        shapes[collision.movingShapeIndex],
                        actors[collision.staticActorIndex],
                        configuration, candidate);
        const GmVec3 originalSpeed =
                response_detail::SpeedAtPoint(
                        candidate, collision.contactPoint);
        if constexpr (TrackDiagnostics) {
            if (index == 0u) {
                scratch.firstResponseWheelIndex =
                        contact.wheelIndex;
                scratch.firstResponseReplacementBefore =
                        contact.replacement;
            }
        }
        if (candidate.vehicle.mobil.absorbContactEnabled) {
            response_detail::AbsorbVehicle(
                    candidate, configuration, contact);
        }
        if constexpr (TrackDiagnostics) {
            if (index == 0u) {
                scratch.firstResponseReplacementAfter =
                        contact.replacement;
            }
        }
        const GmVec3 worldReplacement =
                candidate.vehicle.mobil.absorbContactEnabled
                ? response_detail::LocalToWorldSideA(
                          candidate.body.current.rotation,
                          contact.replacement)
                : detail::Negate(collision.separation);
        response_detail::AddReplacement<CompactReplacements>(
                candidate, scratch, worldReplacement, overflow);
        if (overflow) {
            scratch.overflowReason =
                    OverflowReason::CollisionReplacementCapacity;
            return Status::Overflow;
        }
        if (candidate.vehicle.mobil.absorbContactEnabled &&
            !contact.accepted) {
            continue;
        }
        response_detail::ApplyGenericImpulse(
                candidate, collision, originalSpeed);
    }
    return Status::Success;
}

}  // namespace forevervalidator::simulation::hip::collision

#endif
