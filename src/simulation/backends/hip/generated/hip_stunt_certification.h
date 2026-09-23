// Generated from src/simulation/backends/cuda/cuda_stunt_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_STUNT_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_STUNT_CERTIFICATION_H

#include <cstdint>
#include <string>
#include <vector>

#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

enum class HipStuntCommandKind : std::uint32_t {
    Update,
    RespawnPenalty,
    TimePenalty,
};

struct HipStuntCommand {
    HipStuntCommandKind kind = HipStuntCommandKind::Update;
    std::uint32_t overtimeMs = 0u;
    ReplayStuntSimulationState state{};
};

struct HipStuntExecution {
    bool success = false;
    std::uint32_t failureCommand = UINT32_MAX;
    std::uint32_t failureDetail = 0u;
    HipRaceState finalState{};
    std::string diagnostic;
};

HipStuntExecution ExecuteHipStuntCommandsForCertification(
        const HipRaceState &initialState,
        const std::vector<HipStuntCommand> &commands) noexcept;

}  // namespace forevervalidator::simulation

#endif
