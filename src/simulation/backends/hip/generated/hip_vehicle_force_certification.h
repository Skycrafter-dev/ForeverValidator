// Generated from src/simulation/backends/cuda/cuda_vehicle_force_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_VEHICLE_FORCE_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_VEHICLE_FORCE_CERTIFICATION_H

#include <string>

#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

struct HipVehicleForceExecution {
    bool success = false;
    bool supported = false;
    HipCandidateState finalState{};
    std::string diagnostic;
};

HipVehicleForceExecution ExecuteHipVehicleForceForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt) noexcept;

HipVehicleForceExecution ExecuteHipVehicleForcePassForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt) noexcept;

}  // namespace forevervalidator::simulation

#endif
