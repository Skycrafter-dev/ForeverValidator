// Generated from src/simulation/backends/cuda/cuda_finish_time_refinement.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_FINISH_TIME_REFINEMENT_CUH
#define FOREVERVALIDATOR_HIP_FINISH_TIME_REFINEMENT_CUH

#include <cstdint>

#include <forevervalidator/finish_time.h>

#include "simulation/backends/hip/generated/hip_exact_math.cuh"
#include "simulation/backends/hip/generated/hip_finish_time_origin.cuh"
#include "simulation/backends/hip/generated/hip_physics_step.cuh"

namespace forevervalidator::simulation::hip::finish {

struct Refinement {
    bool present = false;
    bool failed = false;
    bool rejected = false;
    forevervalidator::FinishTimeEstimate estimate{};
};

template <
        bool TrackCollisionDiagnostics,
        bool ReuseWheelPassInvariants,
        bool TrustedInputs,
        bool CompactReplacements,
        bool EightOrderedEllipsoids,
        bool WarpCoherentAcceleration,
        typename Scratch>
__device__ inline physics::Status ProbeFinishSubstep(
        const HipPackedSceneHeader *scene,
        const HipPackedStaticConfigurationHeader *configuration,
        HipCandidatePhysicsState &candidate,
        float dt,
        Scratch &scratch) {
    const vehicle::ForceStatus forceStatus =
            physics::ForcePass<
                    ReuseWheelPassInvariants,
                    HipHandlingSpecialization::Generic>(
                    candidate, configuration, dt);
    if (forceStatus != vehicle::ForceStatus::Success) {
        return static_cast<physics::Status>(
                static_cast<std::uint32_t>(
                        physics::Status::UnsupportedForceBase) +
                static_cast<std::uint32_t>(forceStatus));
    }
    dynamics::PreCollision<CompactReplacements>(
            candidate.body, scratch, dt);
    const collision::Status collisionStatus =
            collision::Detect<
                    TrackCollisionDiagnostics,
                    TrustedInputs,
                    EightOrderedEllipsoids,
                    WarpCoherentAcceleration,
                    true>(
                    scene, configuration, candidate, scratch);
    if (collisionStatus != collision::Status::Success) {
        return static_cast<physics::Status>(
                static_cast<std::uint32_t>(
                        physics::Status::CollisionFailureBase) +
                static_cast<std::uint32_t>(collisionStatus));
    }
    const HipSceneActor *actors =
            collision::detail::SceneSection<HipSceneActor>(
                    scene, scene->actors);
    for (std::uint32_t index = 0u;
         index < scratch.collisionCount; ++index) {
        decltype(auto) contact =
                collision::detail::OrderedCollisionAt(scratch, index);
        if (contact.staticActorIndex >= scene->actors.count) {
            return static_cast<physics::Status>(
                    static_cast<std::uint32_t>(
                            physics::Status::CollisionFailureBase) +
                    static_cast<std::uint32_t>(
                            collision::Status::InvalidScene));
        }
        const HipSceneActor &actor =
                actors[contact.staticActorIndex];
        if (static_cast<std::uint32_t>(
                    actor.itemProperties.collisionGroup) == 1u) {
            race::OnTriggerContact(candidate, actor);
        }
    }
    return physics::Status::Success;
}

template <
        bool TrackCollisionDiagnostics = true,
        bool ReuseWheelPassInvariants = false,
        bool TrustedInputs = false,
        bool CompactReplacements = false,
        bool EightOrderedEllipsoids = false,
        bool WarpCoherentAcceleration = false,
        typename Scratch>
__device__ inline bool RefineTransition(
        const HipPackedSceneHeader *scene,
        const HipPackedStaticConfigurationHeader *configuration,
        const HipCandidatePhysicsState &preSubstep,
        float fullDt,
        double substepStartNs,
        Scratch &scratch,
        forevervalidator::FinishTimeEstimate *estimate,
        std::uint64_t targetResolutionNs = 1u,
        std::uint64_t incumbentUpperNs =
                ~std::uint64_t{0},
        bool *rejected = nullptr) {
    double lower = substepStartNs;
    double upper =
            lower + static_cast<double>(fullDt) * 1000000000.0;
    const auto reject = [&]() {
        if (rejected != nullptr) {
            *rejected = true;
        }
        return true;
    };
    if (lower >= static_cast<double>(incumbentUpperNs)) {
        return reject();
    }
    for (;;) {
        const std::uint64_t intervalLower =
                static_cast<std::uint64_t>(floor(lower));
        const std::uint64_t intervalUpper =
                static_cast<std::uint64_t>(ceil(upper));
        if (intervalUpper >= intervalLower &&
            intervalUpper - intervalLower <= targetResolutionNs) {
            break;
        }
        const std::uint64_t firstInterior =
                static_cast<std::uint64_t>(floor(lower)) + 1u;
        const std::uint64_t upperCeiling =
                static_cast<std::uint64_t>(ceil(upper));
        if (upperCeiling == 0u || firstInterior >= upperCeiling) {
            break;
        }
        const std::uint64_t lastInterior = upperCeiling - 1u;
        const std::uint64_t candidateNs =
                firstInterior +
                (lastInterior - firstInterior) / 2u;
        const float partialDt = static_cast<float>(
                (static_cast<double>(candidateNs) -
                 substepStartNs) /
                1000000000.0);
        if (!(partialDt > 0.0f)) {
            lower = static_cast<double>(candidateNs);
            if (lower >= static_cast<double>(incumbentUpperNs)) {
                return reject();
            }
            continue;
        }
        HipCandidatePhysicsState probe = preSubstep;
        const physics::Status status =
                ProbeFinishSubstep<
                        TrackCollisionDiagnostics,
                        ReuseWheelPassInvariants,
                        TrustedInputs,
                        CompactReplacements,
                        EightOrderedEllipsoids,
                        WarpCoherentAcceleration>(
                        scene, configuration, probe,
                        partialDt, scratch);
        if (status != physics::Status::Success) {
            return false;
        }
        if (probe.race.progress.raceCompleted) {
            upper = static_cast<double>(candidateNs);
        } else {
            lower = static_cast<double>(candidateNs);
            if (lower >= static_cast<double>(incumbentUpperNs)) {
                return reject();
            }
        }
    }
    if (upper >= static_cast<double>(incumbentUpperNs)) {
        return reject();
    }
    estimate->lowerBoundNs =
            static_cast<std::uint64_t>(floor(lower));
    estimate->upperBoundNs =
            static_cast<std::uint64_t>(ceil(upper));
    estimate->estimatedNs = estimate->upperBoundNs;
    return estimate->lowerBoundNs < estimate->upperBoundNs &&
           estimate->upperBoundNs - estimate->lowerBoundNs <=
                   targetResolutionNs;
}

template <
        bool TrackCollisionDiagnostics = true,
        bool ReuseWheelPassInvariants = false,
        bool TrustedInputs = false,
        bool CompactReplacements = false,
        bool EightOrderedEllipsoids = false,
        bool WarpCoherentAcceleration = false,
        typename Scratch>
__device__ inline physics::Status StepAndRefine(
        const HipPackedSceneHeader *scene,
        const HipPackedStaticConfigurationHeader *configuration,
        HipCandidatePhysicsState &candidate,
        const HipControlTick &tick,
        Scratch &scratch,
        Refinement &output,
        std::uint64_t targetResolutionNs = 1u,
        std::uint64_t incumbentUpperNs =
                ~std::uint64_t{0}) {
    const float dt =
            __int2float_rn(static_cast<std::int32_t>(
                    candidate.world.schemePeriodMs)) *
            0.001f;
    if (candidate.body.dynamicActive) {
        candidate.body.temporary = candidate.body.current;
        const GmVec3 &linear = candidate.body.current.linearSpeed;
        const GmVec3 &angular = candidate.body.current.angularSpeed;
        const float linearLength = exact::Sqrt(
                (linear.y * linear.y + linear.x * linear.x) +
                linear.z * linear.z);
        const float angularLength = exact::Sqrt(
                (angular.x * angular.x + angular.y * angular.y) +
                angular.z * angular.z);
        const float scaled =
                exact::Divide(
                        (linearLength + angularLength) * dt,
                        candidate.body.parameters.maxStepDistance);
        std::uint32_t substeps =
                exact::TruncateToUint32Modulo(scaled) + 1u;
        if (substeps > 1000u) substeps = 1000u;

        float remaining = dt;
        double elapsed = 0.0;
        const std::uint64_t tickStartNs =
                TickStartNanoseconds(tick.timeMs);
        for (std::uint32_t index = 0u; index < substeps; ++index) {
            const float substepDt = index + 1u < substeps
                    ? exact::Divide(
                              dt, exact::FromUnsignedInteger(substeps))
                    : remaining;
            const HipCandidatePhysicsState preSubstep = candidate;
            const bool wasFinished =
                    preSubstep.race.progress.raceCompleted;
            const physics::Status status =
                    physics::CollisionSubstep<
                            TrackCollisionDiagnostics,
                            ReuseWheelPassInvariants,
                            TrustedInputs,
                            CompactReplacements,
                            EightOrderedEllipsoids,
                            WarpCoherentAcceleration,
                            HipHandlingSpecialization::Generic>(
                            scene, configuration, candidate,
                            substepDt, scratch);
            if (status != physics::Status::Success) {
                return status;
            }
            if (!wasFinished &&
                candidate.race.progress.raceCompleted) {
                const double substepStartNs =
                        static_cast<double>(tickStartNs) +
                        elapsed * 1000000000.0;
                bool rejected = false;
                output.present = RefineTransition<
                        TrackCollisionDiagnostics,
                        ReuseWheelPassInvariants,
                        TrustedInputs,
                        CompactReplacements,
                        EightOrderedEllipsoids,
                        WarpCoherentAcceleration>(
                        scene, configuration, preSubstep,
                        substepDt, substepStartNs, scratch,
                        &output.estimate, targetResolutionNs,
                        incumbentUpperNs,
                        &rejected);
                output.rejected = rejected;
                output.failed = !output.present && !output.rejected;
                return status;
            }
            elapsed += static_cast<double>(substepDt);
            remaining -= substepDt;
        }
        candidate.body.write = candidate.body.temporary;
    }
    if (candidate.vehicle.mobil.physicsUpdatesEnabled) {
        vehicle::AfterContacts(candidate, configuration);
    }
    return physics::Status::Success;
}

}  // namespace forevervalidator::simulation::hip::finish

#endif
