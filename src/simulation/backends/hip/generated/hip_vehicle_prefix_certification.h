// Generated from src/simulation/backends/cuda/cuda_vehicle_prefix_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_VEHICLE_PREFIX_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_VEHICLE_PREFIX_CERTIFICATION_H

#include <string>

#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

struct HipVehiclePrefixExecution {
    bool success = false;
    HipCandidateState finalState{};
    std::string diagnostic;
};

HipVehiclePrefixExecution ExecuteHipVehiclePrefixForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt) noexcept;

}  // namespace forevervalidator::simulation

#endif
