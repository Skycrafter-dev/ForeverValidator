// Generated from src/simulation/backends/cuda/cuda_physics_step_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_PHYSICS_STEP_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_PHYSICS_STEP_CERTIFICATION_H

#include <string>

#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

struct HipPhysicsStepExecution {
    bool success = false;
    HipCandidateState finalState{};
    std::string diagnostic;
};

HipPhysicsStepExecution ExecuteHipPhysicsStepForCertification(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const HipCandidateState &state) noexcept;

HipPhysicsStepExecution ExecuteHipPreCollisionForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &state,
        float dt) noexcept;

HipPhysicsStepExecution ExecuteHipCollisionSubstepForCertification(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const HipCandidateState &state,
        float dt) noexcept;

}  // namespace forevervalidator::simulation

#endif
