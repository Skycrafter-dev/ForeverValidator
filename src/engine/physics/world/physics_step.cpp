// CHmsZoneDynamic integration, substep scheduling, and contact processing.

#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/physics/world/hms_zone.h"
#include "engine/core/binary32_math.h"
#include "engine/core/mw_cmd_buffer_core.h"

constexpr u32 CHmsZoneDynamic_PhysicsStep2MaxSubsteps = 1000u;

static constexpr float CHmsZoneDynamic_MwTimeToSeconds = 0.001f;

static float secondsFromMwTimeForPhysicsStep2(u32 schemePeriodMwTime) {
    return static_cast<float>(static_cast<int32_t>(schemePeriodMwTime)) *
           CHmsZoneDynamic_MwTimeToSeconds;
}

static u32 ComputeSubstepCount(
        float linearSpeedLength,
        float angularSpeedLength,
        float dt,
        float maxStepDistance) {
    const float speedSum = linearSpeedLength + angularSpeedLength;
    const float scaled = (speedSum * dt) / maxStepDistance;
    return Binary32::TruncateToUint32Modulo(scaled) + 1u;
}

void CHmsZoneDynamic::PhysicsStep2() {
    // The command core owns the simulation clock used by every physics consumer.
    float dt = secondsFromMwTimeForPhysicsStep2(
            CMwCmdBufferCore::Current()->Timer().GetSchemePeriod());
    u32 corpusCount = static_cast<u32>(dynamicCorpuses_.size());
    for (u32 corpusIndex = 0; corpusIndex < corpusCount; corpusIndex++) {
        CHmsCorpus *corpus = dynamicCorpuses_[corpusIndex];
        if (!corpus->IsExcludedFromInitialForcePass()) {
            ComputeCorpusForces(corpus, dt);
            corpus->Dynamics()->DoPreCollisionDynamic(dt);
        }
    }
    collisionManagerZone_->PrepareCollisions();
    CHmsCollisionManagerSZone *collisionManagerZone = collisionManagerZone_;
    for (u32 groupIndex = 0; groupIndex < CHmsCollisionManager_GroupCount; groupIndex++) {
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
                collisionManagerZone->DetectCollisionsCorpus(collisionBuffer_, corpus);
                ComputeCollisionResponse();
                continue;
            }
            if (!dyna->IsDynamicActive()) {
                continue;
            }

            dyna->CopyStateToTemp();
            // trunc(dt * (|linearSpeed| + |angularSpeed|) / maxStepDistance) + 1.
            float maxStepDistance = dyna->Parameters().maxStepDistance;
            GmVec3 linearSpeed;
            GmVec3 angularSpeed;
            dyna->GetLinearSpeed(linearSpeed);
            dyna->GetAngularSpeed(angularSpeed);

            float linearSpeedLength =
                    linearSpeed.PhysicsStep2LinearSpeedLength();
            float angularSpeedLength =
                    angularSpeed.PhysicsStep2AngularSpeedLength();
            u32 substeps = ComputeSubstepCount(
                    linearSpeedLength,
                    angularSpeedLength,
                    dt,
                    maxStepDistance);
            if (substeps > CHmsZoneDynamic_PhysicsStep2MaxSubsteps) {
                substeps = CHmsZoneDynamic_PhysicsStep2MaxSubsteps;
            }

            float remainingDt = dt;
            if (substeps > 1) {
                float splitDt = ((dt) / Binary32::FromUnsignedInteger((substeps)));
                for (u32 remainingSplitCount = substeps - 1;
                 remainingSplitCount != 0;
                 remainingSplitCount--) {
                    ComputeCorpusForces(corpus, splitDt);
                    dyna->DoPreCollisionDynamic(splitDt);
                    collisionBuffer_.Clear();
                    collisionManagerZone->DetectCollisionsCorpus(collisionBuffer_, corpus);
                    ComputeCollisionResponse();
                    dyna->DoPostCollisionDynamic();
                    remainingDt = ((remainingDt) - (splitDt));
                }
            }

            ComputeCorpusForces(corpus, remainingDt);
            dyna->DoPreCollisionDynamic(remainingDt);
            collisionBuffer_.Clear();
            collisionManagerZone->DetectCollisionsCorpus(collisionBuffer_, corpus);
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

void CHmsZoneDynamic::PhysicsStep2OptimizedCpu() {
    float dt = secondsFromMwTimeForPhysicsStep2(
            CMwCmdBufferCore::Current()->Timer().GetSchemePeriod());
    u32 corpusCount = static_cast<u32>(dynamicCorpuses_.size());
    for (u32 corpusIndex = 0; corpusIndex < corpusCount; corpusIndex++) {
        CHmsCorpus *corpus = dynamicCorpuses_[corpusIndex];
        if (!corpus->IsExcludedFromInitialForcePass()) {
            ComputeCorpusForces(corpus, dt);
            corpus->Dynamics()->DoPreCollisionDynamic(dt);
        }
    }
    collisionManagerZone_->PrepareCollisions();
    CHmsCollisionManagerSZone *collisionManagerZone = collisionManagerZone_;
    for (u32 groupIndex = 0; groupIndex < CHmsCollisionManager_GroupCount; groupIndex++) {
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
                collisionManagerZone->DetectCollisionsCorpusOptimizedCpu(
                        collisionBuffer_, corpus);
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
            u32 substeps = ComputeSubstepCount(
                    linearSpeedLength,
                    angularSpeedLength,
                    dt,
                    maxStepDistance);
            if (substeps > CHmsZoneDynamic_PhysicsStep2MaxSubsteps) {
                substeps = CHmsZoneDynamic_PhysicsStep2MaxSubsteps;
            }

            float remainingDt = dt;
            if (substeps > 1) {
                float splitDt = ((dt) / Binary32::FromUnsignedInteger((substeps)));
                for (u32 remainingSplitCount = substeps - 1;
                 remainingSplitCount != 0;
                 remainingSplitCount--) {
                    ComputeCorpusForces(corpus, splitDt);
                    dyna->DoPreCollisionDynamic(splitDt);
                    collisionBuffer_.Clear();
                    collisionManagerZone->DetectCollisionsCorpusOptimizedCpu(
                            collisionBuffer_, corpus);
                    ComputeCollisionResponse();
                    dyna->DoPostCollisionDynamic();
                    remainingDt = ((remainingDt) - (splitDt));
                }
            }

            ComputeCorpusForces(corpus, remainingDt);
            dyna->DoPreCollisionDynamic(remainingDt);
            collisionBuffer_.Clear();
            collisionManagerZone->DetectCollisionsCorpusOptimizedCpu(
                    collisionBuffer_, corpus);
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
