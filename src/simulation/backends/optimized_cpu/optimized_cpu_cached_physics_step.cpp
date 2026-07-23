// OptimizedCpu physics step using immutable static-transform sidecars.

#include "engine/core/binary32_math.h"
#include "engine/core/mw_cmd_buffer_core.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/physics/world/hms_zone.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"

namespace {

constexpr u32 CachedPhysicsStepMaxSubsteps = 1000u;
constexpr float CachedPhysicsStepMwTimeToSeconds = 0.001f;

float CachedPhysicsStepSecondsFromMwTime(u32 schemePeriodMwTime) {
    return static_cast<float>(static_cast<int32_t>(schemePeriodMwTime)) *
           CachedPhysicsStepMwTimeToSeconds;
}

u32 CachedPhysicsStepSubstepCount(
        float linearSpeedLength,
        float angularSpeedLength,
        float dt,
        float maxStepDistance) {
    const float speedSum = linearSpeedLength + angularSpeedLength;
    const float scaled = (speedSum * dt) / maxStepDistance;
    return Binary32::TruncateToUint32Modulo(scaled) + 1u;
}

}  // namespace

void CHmsZoneDynamic::PhysicsStep2OptimizedCpuCachedImpl(
        const OptimizedCpuStaticSurfaceTransformCache &transforms,
        bool nativeBinary32,
        forevervalidator::simulation::
                OptimizedCpuModel3VehicleForceContext *model3Context) {
    const auto detectCollisions =
            [&transforms, nativeBinary32](
                    CHmsCollisionManagerSZone &collisionManagerZone,
                    CHmsCollisionBuffer &collisionBuffer,
                    CHmsCorpus *corpus) {
        if (nativeBinary32) {
            collisionManagerZone.
                    DetectCollisionsCorpusOptimizedCpuNativeBinary32Cached(
                            collisionBuffer, corpus, transforms);
        } else {
            collisionManagerZone.DetectCollisionsCorpusOptimizedCpuCached(
                    collisionBuffer, corpus, transforms);
        }
    };
    const auto computeForces =
            [this, model3Context](CHmsCorpus *corpus, float dt) {
        if (model3Context != nullptr) {
            ComputeCorpusForcesOptimizedCpuModel3(
                    corpus, dt, *model3Context);
        } else {
            ComputeCorpusForces(corpus, dt);
        }
    };
    float dt = CachedPhysicsStepSecondsFromMwTime(
            CMwCmdBufferCore::Current()->Timer().GetSchemePeriod());
    u32 corpusCount = static_cast<u32>(dynamicCorpuses_.size());
    for (u32 corpusIndex = 0; corpusIndex < corpusCount; corpusIndex++) {
        CHmsCorpus *corpus = dynamicCorpuses_[corpusIndex];
        if (!corpus->IsExcludedFromInitialForcePass()) {
            computeForces(corpus, dt);
            corpus->Dynamics()->DoPreCollisionDynamic(dt);
        }
    }
    collisionManagerZone_->PrepareCollisions();
    CHmsCollisionManagerSZone *collisionManagerZone = collisionManagerZone_;
    for (u32 groupIndex = 0;
         groupIndex < CHmsCollisionManager_GroupCount;
         groupIndex++) {
        CHmsCollisionManagerSGroup *group =
                collisionManagerZone->GroupAtOrNull(groupIndex);
        if (group->IsDisabled()) {
            continue;
        }

        u32 groupCorpusCount = group->NonStaticCorpusCount();
        for (u32 groupCorpusIndex = 0;
             groupCorpusIndex < groupCorpusCount;
             groupCorpusIndex++) {
            CHmsCorpus *corpus = group->NonStaticCorpusAt(groupCorpusIndex);
            CHmsDyna *dyna = corpus->Dynamics();

            if (dyna == 0) {
                collisionBuffer_.Clear();
                detectCollisions(
                        *collisionManagerZone, collisionBuffer_, corpus);
                ComputeCollisionResponse();
                continue;
            }
            if (!dyna->IsDynamicActive()) {
                continue;
            }

            dyna->CopyStateToTemp();
            float maxStepDistance = dyna->Parameters().maxStepDistance;
            GmVec3 linearSpeed;
            GmVec3 angularSpeed;
            dyna->GetLinearSpeed(linearSpeed);
            dyna->GetAngularSpeed(angularSpeed);

            float linearSpeedLength =
                    linearSpeed.PhysicsStep2LinearSpeedLength();
            float angularSpeedLength =
                    angularSpeed.PhysicsStep2AngularSpeedLength();
            u32 substeps = CachedPhysicsStepSubstepCount(
                    linearSpeedLength,
                    angularSpeedLength,
                    dt,
                    maxStepDistance);
            if (substeps > CachedPhysicsStepMaxSubsteps) {
                substeps = CachedPhysicsStepMaxSubsteps;
            }

            float remainingDt = dt;
            if (substeps > 1) {
                float splitDt =
                        ((dt) / Binary32::FromUnsignedInteger((substeps)));
                for (u32 remainingSplitCount = substeps - 1;
                     remainingSplitCount != 0;
                     remainingSplitCount--) {
                    computeForces(corpus, splitDt);
                    dyna->DoPreCollisionDynamic(splitDt);
                    collisionBuffer_.Clear();
                    detectCollisions(
                            *collisionManagerZone, collisionBuffer_, corpus);
                    ComputeCollisionResponse();
                    dyna->DoPostCollisionDynamic();
                    remainingDt = ((remainingDt) - (splitDt));
                }
            }

            computeForces(corpus, remainingDt);
            dyna->DoPreCollisionDynamic(remainingDt);
            collisionBuffer_.Clear();
            detectCollisions(
                    *collisionManagerZone, collisionBuffer_, corpus);
            ComputeCollisionResponse();
            dyna->DoPostCollisionDynamic();
            dyna->CopyTempToState();
        }
    }

    for (u32 corpusIndex = 0; corpusIndex < corpusCount; corpusIndex++) {
        CHmsCorpus *corpus = dynamicCorpuses_[corpusIndex];
        corpus->NotifyOwnerAfterContacts();
    }
}

void CHmsZoneDynamic::PhysicsStep2OptimizedCpuCached(
        const OptimizedCpuStaticSurfaceTransformCache &transforms) {
    PhysicsStep2OptimizedCpuCachedImpl(transforms, false, nullptr);
}

void CHmsZoneDynamic::PhysicsStep2OptimizedCpuNativeBinary32Cached(
        const OptimizedCpuStaticSurfaceTransformCache &transforms,
        forevervalidator::simulation::
                OptimizedCpuModel3VehicleForceContext &model3Context) {
    PhysicsStep2OptimizedCpuCachedImpl(
            transforms, true, &model3Context);
}
