// Generated from src/simulation/backends/cuda/cuda_dynamics_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_DYNAMICS_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_DYNAMICS_CERTIFICATION_H

#include <cstdint>
#include <string>

namespace forevervalidator::simulation {

struct HipDynamicsCertificationResult {
    bool success = false;
    std::uint64_t checkedStates = 0u;
    std::uint64_t checkedFields = 0u;
    std::uint64_t firstMismatchState = UINT64_MAX;
    std::string diagnostic;
};

HipDynamicsCertificationResult
CertifyHipPreCollisionDynamics() noexcept;

}  // namespace forevervalidator::simulation

#endif
