// Generated from src/simulation/backends/cuda/cuda_dynamics.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_DYNAMICS_CUH
#define FOREVERVALIDATOR_HIP_DYNAMICS_CUH

#include "simulation/backends/hip/generated/hip_collision.cuh"
#include "simulation/backends/hip/generated/hip_exact_math.cuh"
#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation::hip::dynamics {

__device__ inline bool IsFrozen(
        const HipDynamicBodyState &body) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_FULL_ANGULAR_DYNAMICS)
    return false;
#else
    return body.dynamicType ==
            static_cast<std::uint32_t>(
                    CHmsDyna::EDynamicType_Frozen);
#endif
}

__device__ inline bool IsLinearOnly(
        const HipDynamicBodyState &body) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_FULL_ANGULAR_DYNAMICS)
    return false;
#else
    return body.dynamicType ==
            static_cast<std::uint32_t>(
                    CHmsDyna::EDynamicType_LinearOnly);
#endif
}

__device__ inline bool IsFullAngular(
        const HipDynamicBodyState &body) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_FULL_ANGULAR_DYNAMICS)
    return true;
#else
    return body.dynamicType ==
            static_cast<std::uint32_t>(
                    CHmsDyna::EDynamicType_FullAngularDynamics);
#endif
}

namespace detail {

__device__ inline float Dot(const GmVec3 &left,
                            const GmVec3 &right) {
    const float xy = left.x * right.x + left.y * right.y;
    return xy + left.z * right.z;
}

__device__ inline GmVec3 Row(const GmMat3 &matrix,
                             unsigned index) {
    if (index == 0u) {
        return {matrix.basisX.x, matrix.basisY.x,
                matrix.basisZ.x};
    }
    if (index == 1u) {
        return {matrix.basisX.y, matrix.basisY.y,
                matrix.basisZ.y};
    }
    return {matrix.basisX.z, matrix.basisY.z,
            matrix.basisZ.z};
}

__device__ inline GmVec3 Transform(const GmMat3 &matrix,
                                   const GmVec3 &vector) {
    return {
            Dot(Row(matrix, 0u), vector),
            Dot(Row(matrix, 1u), vector),
            Dot(Row(matrix, 2u), vector),
    };
}

__device__ inline GmMat3 Transpose(const GmMat3 &matrix) {
    return {Row(matrix, 0u), Row(matrix, 1u), Row(matrix, 2u)};
}

__device__ inline GmMat3 Compose(const GmMat3 &first,
                                 const GmMat3 &second) {
    return {
            Transform(second, first.basisX),
            Transform(second, first.basisY),
            Transform(second, first.basisZ),
    };
}

__device__ inline void Normalize(GmQuat &quaternion) {
    const float xy = quaternion.x * quaternion.x +
                     quaternion.w * quaternion.w;
    const float lengthSquared =
            (xy + quaternion.y * quaternion.y) +
            quaternion.z * quaternion.z;
    const float inverseLength = exact::Divide(
            1.0f, exact::Sqrt(lengthSquared));
    quaternion.w *= inverseLength;
    quaternion.x *= inverseLength;
    quaternion.y *= inverseLength;
    quaternion.z *= inverseLength;
}

__device__ inline GmMat3 MatrixFromQuaternion(
        GmQuat quaternion) {
    const float twoX = quaternion.x * 2.0f;
    const float twoY = quaternion.y * 2.0f;
    const float twoZ = quaternion.z * 2.0f;
    const float xw2 = quaternion.w * twoX;
    const float yw2 = quaternion.w * twoY;
    const float zw2 = quaternion.w * twoZ;
    const float xx2 = quaternion.x * twoX;
    const float xy2 = quaternion.x * twoY;
    const float xz2 = quaternion.x * twoZ;
    const float yy2 = quaternion.y * twoY;
    const float yz2 = quaternion.y * twoZ;
    const float zz2 = quaternion.z * twoZ;
    return {
            {(1.0f - yy2) - zz2, xy2 + zw2, xz2 - yw2},
            {xy2 - zw2, (1.0f - xx2) - zz2, yz2 + xw2},
            {xz2 + yw2, yz2 - xw2, (1.0f - xx2) - yy2},
    };
}

}  // namespace detail

__device__ inline void IntegrateStep(
        const HipDynamicBodyState &body,
        const CHmsDyna::CHmsStateDyna &source,
        CHmsDyna::CHmsStateDyna &destination,
        float dt) {
    if (IsFrozen(body)) {
        destination = source;
        return;
    }

    const float inverseMass = exact::Divide(
            1.0f, body.parameters.mass);
    const float forceMassX = source.force.x * inverseMass;
    const float forceMassY = source.force.y * inverseMass;
    const float forceMassZ = source.force.z * inverseMass;

    const float positionDeltaX = source.linearSpeed.x * dt;
    const float positionDeltaY = source.linearSpeed.y * dt;
    const float positionDeltaZ = source.linearSpeed.z * dt;
    destination.position.x = positionDeltaX + source.position.x;
    destination.position.y = source.position.y + positionDeltaY;
    destination.position.z = source.position.z + positionDeltaZ;

    const float correctionX =
            source.linearCorrectionSpeed.x * dt;
    const float correctionY =
            source.linearCorrectionSpeed.y * dt;
    const float correctionZ =
            source.linearCorrectionSpeed.z * dt;
    destination.position.x += correctionX;
    destination.position.y += correctionY;
    destination.position.z += correctionZ;
    destination.linearCorrectionSpeed = {};

    const float forceDeltaX = forceMassX * dt;
    const float forceDeltaY = forceMassY * dt;
    const float forceDeltaZ = dt * forceMassZ;
    destination.linearSpeed.x =
            source.linearSpeed.x + forceDeltaX;
    destination.linearSpeed.y =
            source.linearSpeed.y + forceDeltaY;
    destination.linearSpeed.z =
            source.linearSpeed.z + forceDeltaZ;

    if (IsLinearOnly(body)) {
        destination.rotation = source.rotation;
        return;
    }

    const GmVec3 angularAcceleration =
            detail::Transform(
                    source.inverseInertiaWorld, source.torque);
    const float angularLengthSquared =
            (source.angularSpeed.y * source.angularSpeed.y +
             source.angularSpeed.x * source.angularSpeed.x) +
            source.angularSpeed.z * source.angularSpeed.z;
    if (!(1.0e-10f < angularLengthSquared)) {
        destination.rotation = source.rotation;
        destination.rotationQuat = source.rotationQuat;
    } else {
        const GmQuat sourceQuaternion = source.rotationQuat;
        const GmVec3 angularSpeed = source.angularSpeed;
        constexpr float Half = 0.5f;
        GmQuat derivative;
        derivative.w =
                ((-angularSpeed.x * sourceQuaternion.x -
                  angularSpeed.y * sourceQuaternion.y) -
                 sourceQuaternion.z * angularSpeed.z) *
                Half;
        derivative.x =
                ((sourceQuaternion.w * angularSpeed.x +
                  angularSpeed.y * sourceQuaternion.z) -
                 sourceQuaternion.y * angularSpeed.z) *
                Half;
        derivative.y =
                ((angularSpeed.y * sourceQuaternion.w -
                  sourceQuaternion.z * angularSpeed.x) +
                 angularSpeed.z * sourceQuaternion.x) *
                Half;
        derivative.z =
                ((sourceQuaternion.y * angularSpeed.x -
                  angularSpeed.y * sourceQuaternion.x) +
                 sourceQuaternion.w * angularSpeed.z) *
                Half;

        destination.rotationQuat = source.rotationQuat;
        destination.rotationQuat.w =
                derivative.w * dt + destination.rotationQuat.w;
        destination.rotationQuat.x =
                derivative.x * dt + destination.rotationQuat.x;
        destination.rotationQuat.y =
                derivative.y * dt + destination.rotationQuat.y;
        destination.rotationQuat.z =
                dt * derivative.z + destination.rotationQuat.z;
        detail::Normalize(destination.rotationQuat);
        destination.rotation = detail::MatrixFromQuaternion(
                destination.rotationQuat);

        const GmVec3 oldCenter = detail::Transform(
                source.rotation, body.parameters.localCenterOfMass);
        const GmVec3 newCenter = detail::Transform(
                destination.rotation,
                body.parameters.localCenterOfMass);
        const float centerDeltaX = newCenter.x - oldCenter.x;
        const float centerDeltaY = newCenter.y - oldCenter.y;
        const float centerDeltaZ = newCenter.z - oldCenter.z;
        destination.position.x -= centerDeltaX;
        destination.position.y -= centerDeltaY;
        destination.position.z -= centerDeltaZ;
    }

    const float torqueDeltaX = angularAcceleration.x * dt;
    const float torqueDeltaY = angularAcceleration.y * dt;
    const float torqueDeltaZ = dt * angularAcceleration.z;
    destination.angularSpeed.x =
            source.angularSpeed.x + torqueDeltaX;
    destination.angularSpeed.y =
            source.angularSpeed.y + torqueDeltaY;
    destination.angularSpeed.z =
            source.angularSpeed.z + torqueDeltaZ;

    if (body.maxAngularSpeed.present) {
        const float angularLength =
                (destination.angularSpeed.y *
                         destination.angularSpeed.y +
                 destination.angularSpeed.x *
                         destination.angularSpeed.x) +
                destination.angularSpeed.z *
                        destination.angularSpeed.z;
        const float maximum = body.maxAngularSpeed.value;
        if (maximum * maximum < angularLength) {
            const float scale = exact::Divide(
                    maximum, exact::Sqrt(angularLength));
            destination.angularSpeed.x *= scale;
            destination.angularSpeed.y =
                    scale * destination.angularSpeed.y;
            destination.angularSpeed.z =
                    scale * destination.angularSpeed.z;
        }
    }

    destination.inverseInertiaWorld =
            detail::Transpose(destination.rotation);
    destination.inverseInertiaWorld = detail::Compose(
            destination.inverseInertiaWorld,
            body.parameters.bodyInertiaLike);
    destination.inverseInertiaWorld = detail::Compose(
            destination.inverseInertiaWorld,
            destination.rotation);
}

__device__ inline void PreCollision(
        HipDynamicBodyState &body,
        HipFixedArray<
                GmVec3,
                HipCollisionReplacementOverflowCapacity>
                &overflowReplacements,
        float dt) {
    const CHmsDyna::CHmsStateDyna source = body.current;
    IntegrateStep(body, source, body.current, dt);
    body.collisionReplacements = {};
    overflowReplacements.count = 0u;
}

template<bool CompactReplacements = false, typename Scratch>
__device__ inline void PreCollision(
        HipDynamicBodyState &body,
        Scratch &scratch,
        float dt) {
    const CHmsDyna::CHmsStateDyna source = body.current;
    IntegrateStep(body, source, body.current, dt);
    if constexpr (!CompactReplacements) {
        body.collisionReplacements = {};
    }
    scratch.replacementOverflowCount = 0u;
    if constexpr (CompactReplacements) {
        scratch.replacementSumX = 0.0f;
        scratch.replacementSumY = 0.0f;
        scratch.replacementSumZ = 0.0f;
    }
}

__device__ inline void AccumulateReplacement(
        const GmVec3 &next,
        float &sumX,
        float &sumY,
        float &sumZ) {
    float workX = sumX;
    float workY = sumY;
    float workZ = sumZ;
    const float projection =
            (next.x * sumX + sumY * next.y) +
            sumZ * next.z;
    if (projection > 0.0f) {
        const float sumLengthSquared =
                sumZ * sumZ +
                (sumY * sumY + workX * workX);
        if (1.0e-10f < sumLengthSquared) {
            float clampedProjection = projection;
            if (sumLengthSquared < clampedProjection) {
                clampedProjection = sumLengthSquared;
            }
            const float scale = exact::Divide(
                    clampedProjection, sumLengthSquared);
            const float projectedX = scale * workX;
            const float projectedY = scale * workY;
            const float projectedZ = scale * workZ;
            workX = workX - projectedX;
            workY = workY - projectedY;
            workZ = workZ - projectedZ;
        }
    }
    sumX = workX + next.x;
    sumY = workY + next.y;
    sumZ = workZ + next.z;
}

__device__ inline GmVec3 FinalizeReplacement(
        float sumX, float sumY, float sumZ) {
    const float lengthSquared =
            sumZ * sumZ + (sumY * sumY + sumX * sumX);
    constexpr float DirectionEpsilon = 0.01f;
    if (!(DirectionEpsilon * DirectionEpsilon < lengthSquared)) {
        return {};
    }
    const float inverseLength = exact::Divide(
            1.0f, exact::Sqrt(lengthSquared));
    const float unitX = inverseLength * sumX;
    const float unitY = inverseLength * sumY;
    const float unitZ = inverseLength * sumZ;
    return {
            sumX - DirectionEpsilon * unitX,
            sumY - DirectionEpsilon * unitY,
            sumZ - DirectionEpsilon * unitZ,
    };
}

__device__ inline GmVec3 SynthesizeReplacement(
        const HipFixedArray<
                GmVec3,
                HipCollisionReplacementInlineCapacity> &replacements,
        const HipFixedArray<
                GmVec3,
                HipCollisionReplacementOverflowCapacity>
                &overflowReplacements) {
    if (replacements.count == 0u &&
        overflowReplacements.count == 0u) {
        return {};
    }
    const bool hasInline = replacements.count != 0u;
    const GmVec3 &first = hasInline
            ? replacements.values[0]
            : overflowReplacements.values[0];
    float sumX = first.x;
    float sumY = first.y;
    float sumZ = first.z;
    for (std::uint32_t index = hasInline ? 1u : 0u;
         index < replacements.count; ++index) {
        AccumulateReplacement(
                replacements.values[index], sumX, sumY, sumZ);
    }
    for (std::uint32_t index = hasInline ? 0u : 1u;
         index < overflowReplacements.count; ++index) {
        AccumulateReplacement(
                overflowReplacements.values[index],
                sumX, sumY, sumZ);
    }
    return FinalizeReplacement(sumX, sumY, sumZ);
}

template<typename Scratch>
__device__ inline GmVec3 SynthesizeReplacement(
        const HipFixedArray<
                GmVec3,
                HipCollisionReplacementInlineCapacity> &replacements,
        const Scratch &scratch) {
    if (replacements.count == 0u &&
        scratch.replacementOverflowCount == 0u) {
        return {};
    }
    const bool hasInline = replacements.count != 0u;
    const GmVec3 &first = hasInline
            ? replacements.values[0]
            : collision::detail::ReplacementOverflowAt(scratch, 0u);
    float sumX = first.x;
    float sumY = first.y;
    float sumZ = first.z;
    for (std::uint32_t index = hasInline ? 1u : 0u;
         index < replacements.count; ++index) {
        AccumulateReplacement(
                replacements.values[index], sumX, sumY, sumZ);
    }
    for (std::uint32_t index = hasInline ? 0u : 1u;
         index < scratch.replacementOverflowCount; ++index) {
        AccumulateReplacement(
                collision::detail::ReplacementOverflowAt(
                        scratch, index),
                sumX, sumY, sumZ);
    }
    return FinalizeReplacement(sumX, sumY, sumZ);
}

__device__ inline void PostCollision(
        HipDynamicBodyState &body,
        const HipFixedArray<
                GmVec3,
                HipCollisionReplacementOverflowCapacity>
                &overflowReplacements) {
    const GmVec3 replacement =
            SynthesizeReplacement(
                    body.collisionReplacements,
                    overflowReplacements);
    body.current.position.x += replacement.x;
    body.current.position.y =
            replacement.y + body.current.position.y;
    body.current.position.z += replacement.z;
}

template<bool CompactReplacements = false, typename Scratch>
__device__ inline void PostCollision(
        HipDynamicBodyState &body,
        const Scratch &scratch) {
    const GmVec3 replacement = [&]() {
        if constexpr (CompactReplacements) {
            return scratch.replacementOverflowCount == 0u
                    ? GmVec3{}
                    : FinalizeReplacement(
                              scratch.replacementSumX,
                              scratch.replacementSumY,
                              scratch.replacementSumZ);
        } else {
            return SynthesizeReplacement(
                    body.collisionReplacements, scratch);
        }
    }();
    body.current.position.x += replacement.x;
    body.current.position.y =
            replacement.y + body.current.position.y;
    body.current.position.z += replacement.z;
}

}  // namespace forevervalidator::simulation::hip::dynamics

#endif
