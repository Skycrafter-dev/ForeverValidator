// Generated from src/simulation/backends/cuda/cuda_race_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_RACE_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_RACE_CERTIFICATION_H

#include <string>

#include "simulation/backends/hip/generated/hip_scene_layout.h"
#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

struct HipRaceContactExecution {
    bool success = false;
    HipCandidateState finalState{};
    std::string diagnostic;
};

HipRaceContactExecution ExecuteHipRaceContactForCertification(
        const HipCandidateState &initialState,
        const HipSceneActor &actor) noexcept;

}  // namespace forevervalidator::simulation

#endif
