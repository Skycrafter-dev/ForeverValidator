#ifndef FOREVERVALIDATOR_CUDA_PHYSICS_STEP_CUH
#define FOREVERVALIDATOR_CUDA_PHYSICS_STEP_CUH

#include "simulation/backends/cuda/cuda_collision_response.cuh"
#include "simulation/backends/cuda/cuda_dynamics.cuh"
#include "simulation/backends/cuda/cuda_environment.cuh"
#include "simulation/backends/cuda/cuda_vehicle_after_contacts.cuh"
#include "simulation/backends/cuda/cuda_vehicle_forces.cuh"

namespace forevervalidator::simulation::cuda::physics {

enum class Status : std::uint32_t {
    Success,
    UnsupportedVehicleForce,
    UnsupportedForceBase = 100u,
    CollisionFailureBase = 200u,
};

template <
        bool ReuseWheelPassInvariants = false,
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic,
        bool CollectHotPathMetrics = false>
__device__ inline vehicle::ForceStatus ForcePass(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        float dt,
        collision::CudaHotPathCounters *hotPathCounters = nullptr) {
    environment::BeginForcePass(candidate.body, configuration);
    if (!candidate.vehicle.mobil.physicsUpdatesEnabled) {
        if constexpr (CollectHotPathMetrics) {
            ++hotPathCounters->physicsCallbackDisabledForcePassCount;
        }
        return vehicle::ForceStatus::Success;
    }
    return vehicle::ComputeForcesModel6<
            ReuseWheelPassInvariants,
            Handling,
            CollectHotPathMetrics>(
            candidate, configuration, dt, hotPathCounters);
}

template <
        bool TrackCollisionDiagnostics = true,
        bool ReuseWheelPassInvariants = false,
        bool TrustedInputs = false,
        bool CompactReplacements = false,
        bool EightOrderedEllipsoids = false,
        bool WarpCoherentAcceleration = false,
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic,
        bool CollectHotPathMetrics = false,
        typename Scratch = collision::CudaCollisionScratch>
__device__ inline Status CollisionSubstep(
        const CudaPackedSceneHeader *scene,
        const CudaPackedStaticConfigurationHeader *configuration,
        CudaCandidatePhysicsState &candidate,
        float dt,
        Scratch &scratch,
        collision::CudaHotPathCounters *hotPathCounters = nullptr) {
    if constexpr (CollectHotPathMetrics) {
        ++hotPathCounters->physicsSubstepCount;
    }
    const vehicle::ForceStatus forceStatus =
            ForcePass<
                    ReuseWheelPassInvariants,
                    Handling,
                    CollectHotPathMetrics>(
                    candidate, configuration, dt, hotPathCounters);
    if (forceStatus != vehicle::ForceStatus::Success) {
        return static_cast<Status>(
                static_cast<std::uint32_t>(
                        Status::UnsupportedForceBase) +
                static_cast<std::uint32_t>(forceStatus));
    }
    dynamics::PreCollision<CompactReplacements>(
            candidate.body, scratch, dt);
    collision::Status collisionStatus =
            collision::Detect<
                    TrackCollisionDiagnostics,
                    TrustedInputs,
                    EightOrderedEllipsoids,
                    WarpCoherentAcceleration,
                    false,
                    CollectHotPathMetrics>(
                    scene, configuration, candidate, scratch,
                    hotPathCounters);
    if (collisionStatus == collision::Status::Success) {
        collisionStatus =
                collision::Respond<
                        TrackCollisionDiagnostics,
                        TrustedInputs,
                        CompactReplacements>(
                        scene, configuration, candidate, scratch);
    }
    if (collisionStatus != collision::Status::Success) {
        return static_cast<Status>(
                static_cast<std::uint32_t>(
                        Status::CollisionFailureBase) +
                static_cast<std::uint32_t>(collisionStatus));
    }
    dynamics::PostCollision<CompactReplacements>(
            candidate.body, scratch);
    return Status::Success;
}

template <
        bool TrackCollisionDiagnostics = true,
        bool ReuseWheelPassInvariants = false,
        bool TrustedInputs = false,
        bool CompactReplacements = false,
        bool EightOrderedEllipsoids = false,
        bool WarpCoherentAcceleration = false,
        bool WriteOutputSnapshots = true,
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic,
        bool CollectHotPathMetrics = false,
        typename Scratch = collision::CudaCollisionScratch>
__device__ inline Status Step(
        const CudaPackedSceneHeader *scene,
        const CudaPackedStaticConfigurationHeader *configuration,
        CudaCandidatePhysicsState &candidate,
        Scratch &scratch,
        collision::CudaHotPathCounters *hotPathCounters = nullptr) {
    const float dt =
            __int2float_rn(static_cast<std::int32_t>(
                    candidate.world.schemePeriodMs)) *
            0.001f;
    // StadiumCar is registered in the dynamic collision group, so the
    // authoritative zone excludes it from the initial ungrouped force pass.
    if (candidate.body.dynamicActive) {
        candidate.body.temporary = candidate.body.current;
        const GmVec3 &linear =
                candidate.body.current.linearSpeed;
        const GmVec3 &angular =
                candidate.body.current.angularSpeed;
        const float linearLength = exact::Sqrt(
                (linear.y * linear.y + linear.x * linear.x) +
                linear.z * linear.z);
        const float angularLength = exact::Sqrt(
                (angular.x * angular.x +
                 angular.y * angular.y) +
                angular.z * angular.z);
        const float scaled =
                ((linearLength + angularLength) * dt) /
                candidate.body.parameters.maxStepDistance;
        std::uint32_t substeps =
                exact::TruncateToUint32Modulo(scaled) + 1u;
        if (substeps > 1000u) substeps = 1000u;
        if constexpr (CollectHotPathMetrics) {
            if (substeps > hotPathCounters->maximumSubstepsPerTick) {
                hotPathCounters->maximumSubstepsPerTick = substeps;
            }
        }
        float remaining = dt;
        if (substeps > 1u) {
            const float split =
                    dt / exact::FromUnsignedInteger(substeps);
            for (std::uint32_t count = substeps - 1u;
                 count != 0u; --count) {
                const Status status =
                        CollisionSubstep<
                                TrackCollisionDiagnostics,
                                ReuseWheelPassInvariants,
                                TrustedInputs,
                                CompactReplacements,
                                EightOrderedEllipsoids,
                                WarpCoherentAcceleration,
                                Handling,
                                CollectHotPathMetrics>(
                                scene, configuration, candidate,
                                split, scratch, hotPathCounters);
                if (status != Status::Success) return status;
                remaining -= split;
            }
        }
        const Status finalStatus =
                CollisionSubstep<
                        TrackCollisionDiagnostics,
                        ReuseWheelPassInvariants,
                        TrustedInputs,
                        CompactReplacements,
                        EightOrderedEllipsoids,
                        WarpCoherentAcceleration,
                        Handling,
                        CollectHotPathMetrics>(
                        scene, configuration, candidate,
                        remaining, scratch, hotPathCounters);
        if (finalStatus != Status::Success) {
            return finalStatus;
        }
        candidate.body.write = candidate.body.temporary;
    }
    if (candidate.vehicle.mobil.physicsUpdatesEnabled) {
        if constexpr (WriteOutputSnapshots) {
            vehicle::AfterContacts(candidate, configuration);
        } else {
            vehicle::AfterContactsWithoutSnapshots(candidate);
        }
    }
    return Status::Success;
}

}  // namespace forevervalidator::simulation::cuda::physics

#endif
