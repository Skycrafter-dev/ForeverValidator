// Generated from src/simulation/backends/cuda/cuda_collision_certification.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_COLLISION_CERTIFICATION_H
#define FOREVERVALIDATOR_HIP_COLLISION_CERTIFICATION_H

#include <string>
#include <vector>

#include "simulation/backends/hip/generated/hip_collision_layout.h"
#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

struct HipCollisionExecution {
    bool success = false;
    hip::collision::Status status =
            hip::collision::Status::InvalidScene;
    std::vector<hip::collision::HipCollision> collisions;
    HipCandidateState finalState{};
    std::uint32_t accelerationCellVisits = 0u;
    std::uint32_t accelerationSurfaceVisits = 0u;
    std::uint32_t meshCellVisits = 0u;
    std::uint32_t meshCellIntersections = 0u;
    std::uint32_t meshTriangleCells = 0u;
    std::uint32_t triangleTests = 0u;
    std::uint32_t triangleHits = 0u;
    std::uint32_t firstVisitedShape = UINT32_MAX;
    std::uint32_t firstVisitedSurface = UINT32_MAX;
    GmIso4 firstShapeWorld{};
    GmBoxAligned firstMovingBounds{};
    GmBoxAligned firstEllipsoidBox{};
    GmBoxAligned firstSurfaceWorldBounds{};
    GmBoxAligned firstMeshRootBounds{};
    std::uint32_t firstResponseWheelIndex = UINT32_MAX;
    GmVec3 firstResponseReplacementBefore{};
    GmVec3 firstResponseReplacementAfter{};
    std::string diagnostic;
};

HipCollisionExecution ExecuteHipCollisionForCertification(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const HipCandidateState &state) noexcept;

struct HipCollisionOrderingExecution {
    bool success = false;
    std::vector<hip::collision::HipCollision> collisions;
    std::string diagnostic;
};

HipCollisionOrderingExecution
ExecuteHipCollisionOrderingForCertification(
        const std::vector<hip::collision::HipCollision> &collisions)
        noexcept;

struct HipShapeWorldPoseExecution {
    bool success = false;
    GmIso4 worldPose{};
    std::string diagnostic;
};

HipShapeWorldPoseExecution ExecuteHipShapeWorldPoseForCertification(
        const std::vector<HipVehicleCollisionShape> &shapes,
        const HipCandidateState &state,
        std::uint32_t shapeIndex) noexcept;

}  // namespace forevervalidator::simulation

#endif
