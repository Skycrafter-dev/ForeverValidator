// Generated from src/simulation/backends/cuda/cuda_environment.cuh by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_ENVIRONMENT_CUH
#define FOREVERVALIDATOR_HIP_ENVIRONMENT_CUH

#include "simulation/backends/hip/generated/hip_state_layout.h"
#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_tuning.cuh"

namespace forevervalidator::simulation::hip::environment {

__device__ inline void AddScaled(
        GmVec3 &accumulator,
        GmVec3 value,
        float scale) {
    value.x = value.x * scale;
    value.y = value.y * scale;
    value.z = scale * value.z;
    accumulator.x = value.x + accumulator.x;
    accumulator.y = value.y + accumulator.y;
    accumulator.z = value.z + accumulator.z;
}

__device__ inline bool ForceFieldValue(
        const HipForceField &field,
        const GmVec3 &position,
        GmVec3 &value) {
    if (field.type == HipForceFieldType::Uniform) {
        value = field.vector;
        return true;
    }
    const GmVec3 delta = {
            field.vector.x - position.x,
            field.vector.y - position.y,
            field.vector.z - position.z,
    };
    const float distanceSquared =
            (delta.z * delta.z + delta.x * delta.x) +
            delta.y * delta.y;
    const float radiusSquared = field.radius * field.radius;
    if (!(radiusSquared > distanceSquared) ||
        !(distanceSquared > 1.0e-10f)) {
        return false;
    }
    const float scale = exact::Divide(-field.strength, distanceSquared);
    value.x = delta.x * scale;
    value.y = delta.y * scale;
    value.z = scale * delta.z;
    return true;
}

__device__ inline void BeginForcePass(
        HipDynamicBodyState &body,
        const HipPackedStaticConfigurationHeader *configuration) {
    body.write = body.current;
    GmVec3 accumulatedForce{};
    const float fieldScale =
            body.parameters.forceScale * body.parameters.mass;
#if !defined(FOREVERVALIDATOR_HIP_RESEARCH_NO_FORCE_FIELDS)
    const HipForceField *fields =
            tuning::Section<HipForceField>(
                    configuration, configuration->forceFields);
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_ONE_UNIFORM_FORCE_FIELD)
    AddScaled(accumulatedForce, fields[0].vector, fieldScale);
#else
    for (std::uint32_t index = 0u;
         index < configuration->forceFields.count; ++index) {
        GmVec3 value;
        if (ForceFieldValue(
                    fields[index], body.current.position, value)) {
            AddScaled(accumulatedForce, value, fieldScale);
        }
    }
#endif
#else
    (void)fieldScale;
#endif
    AddScaled(
            accumulatedForce, body.current.linearSpeed,
            -configuration->zoneLinearDampingCoefficient *
                    body.parameters.linearDampingScale);
    body.current.force = accumulatedForce;

#if defined(FOREVERVALIDATOR_HIP_RESEARCH_FULL_ANGULAR_DYNAMICS)
    {
#else
    if (body.dynamicType ==
        static_cast<std::uint32_t>(
                CHmsDyna::EDynamicType_FullAngularDynamics)) {
#endif
        GmVec3 dampingTorque = body.current.angularSpeed;
        const float scale =
                -configuration->zoneAngularDampingCoefficient *
                body.parameters.angularDampingScale;
        dampingTorque.x = dampingTorque.x * scale;
        dampingTorque.y = dampingTorque.y * scale;
        dampingTorque.z = scale * dampingTorque.z;
        body.current.torque = dampingTorque;
    }
}

}  // namespace forevervalidator::simulation::hip::environment

#endif
