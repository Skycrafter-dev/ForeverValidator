#pragma once

#include "engine/core/engine_types.h"
#include <optional>
#include <vector>

#include "engine/core/gm_types.h"
struct SHmsPhysicalCollision;

struct CHmsDynaParams {
    float mass; // inverse used by IntegrateStep/AddImpulse.
    GmMat3 bodyInertiaLike;
    float linearDampingScale; // multiplies zone linear damping in ComputeCorpusForces.
    float angularDampingScale; // multiplies zone angular damping in ComputeCorpusForces.
    float maxStepDistance; // PhysicsStep2 substep divisor.
    float forceScale; // ComputeCorpusForces force-field scale.
    GmVec3 localCenterOfMass; // transformed to world for angular velocity/impulses.
};

struct CHmsDyna {
    enum EDynamicType : u32 {
        EDynamicType_LinearOnly = 0u,
        EDynamicType_FullAngularDynamics = 1u,
        EDynamicType_Frozen = 2u,
    };

    struct CHmsStateDyna {
        GmQuat rotationQuat{};
        GmMat3 rotation{};
        GmVec3 position{};
        GmVec3 linearSpeed{};
        GmVec3 linearCorrectionSpeed{};
        GmVec3 angularSpeed{};
        GmVec3 force{};
        GmVec3 torque{};
        GmMat3 inverseInertiaWorld{};
        bool tweakedLinearSpeedValid = false;
        GmVec3 tweakedLinearSpeed{};

        void Reset(void);
    };

    struct RuntimeClone {
        std::optional<float> maxAngularSpeed;
        CHmsDynaParams dynaParams{};
        CHmsStateDyna tempState{};
        CHmsStateDyna writeState{};
        CHmsStateDyna currentState{};
        std::vector<GmVec3> pendingCollisionReplacements;
        bool isDynamicActive = false;
        EDynamicType dynamicType = EDynamicType_LinearOnly;
    };

    CHmsDyna(void);
    ~CHmsDyna(void);
    CHmsDynaParams &Parameters(void) { return dynaParams; }
    const CHmsDynaParams &Parameters(void) const { return dynaParams; }
    CHmsStateDyna &CurrentState(void) { return currentState; }
    const CHmsStateDyna &CurrentState(void) const { return currentState; }
    CHmsStateDyna &WriteState(void) { return writeState; }
    const CHmsStateDyna &WriteState(void) const { return writeState; }
    CHmsStateDyna &TemporaryState(void) { return tempState; }
    const CHmsStateDyna &TemporaryState(void) const { return tempState; }
    void SetAngularSpeedLimit(std::optional<float> limit) {
        maxAngularSpeed = limit;
    }
    const std::optional<float> &AngularSpeedLimit(void) const {
        return maxAngularSpeed;
    }
    void SetDynamicActive(bool active) { isDynamicActive = active; }
    bool IsDynamicActive(void) const { return isDynamicActive; }
    EDynamicType DynamicType(void) const { return dynamicType; }
    void Reset(void);
    void ClearCollisionReplacements(void);
    void ValidateDynamicState(
            void);
    void CopyStateToTemp(
            void);
    void CopyTempToState(
            void);
    void SetDynamicType(
            EDynamicType dynamicType);
    float InverseMass(void) const;
    int UsesFullAngularDynamics(void) const;
    int UsesAngularIntegration(void) const;
    int IsKinematicOrFrozen(void) const;
    void Activate(void);
    GmVec3 WorldCenterOfMass(const CHmsStateDyna &state) const;
    float AngularEffectiveMassTermForSolveImpulse(
            const SHmsPhysicalCollision *collision,
            GmVec3 unitImpulse,
            int sideB);
    GmVec3 WorldCenterOfMassForAddForceAtPoint(const CHmsStateDyna &state) const;
    GmVec3 LocalDirectionToWorld(const GmVec3 &local) const;
    GmVec3 LocalPointToWorld(const GmVec3 &local) const;
    void WorldDirectionToLocal(GmVec3 &world) const;
    void ApplyReplacement(
            const GmVec3 &replacement);
    void SetForce(
            const GmVec3 &force);
    void SetTorque(
            const GmVec3 &torque);
    void AddForce(
            const GmVec3 &force,
            const GmVec3 &point);
    void AddForce(
            const GmVec3 &force);
    void GetForce(
            GmVec3 &out) const;
    void AddTorque(
            const GmVec3 &torque);
    void IntegrateStep(
            const CHmsStateDyna &src,
            CHmsStateDyna &dst,
            float dt) const;
    void DoPreCollisionDynamic(
            float dt);
    void AddImpulse(
            const GmVec3 &impulse,
            const GmVec3 &point);
    void AddImpulse(
            const GmVec3 &impulse);
    void GetSpeed(
            const GmVec3 &point,
            GmVec3 &out) const;
    void SetLinearSpeed(
            const GmVec3 &speed);
    void GetLinearSpeed(
            GmVec3 &out) const;
    void SetAngularSpeed(
            const GmVec3 &speed);
    void GetAngularSpeed(
            GmVec3 &out) const;
    void DoPostCollisionDynamic(
            void);
    void SetLocation(
            const GmIso4 &location);
    void SetTranslation(
            const GmVec3 &translation);
    void RotateOf(
            const GmMat3 &rotation);
    void AddLocalForce(
            const GmVec3 &force,
            const GmVec3 &point);
    void SetLocalForce(
            const GmVec3 &force);
    void AddLocalForce(
            const GmVec3 &force);
    void GetLocalForce(
            GmVec3 &out) const;
    void SetLocalTorque(
            const GmVec3 &torque);
    void AddLocalTorque(
            const GmVec3 &torque);
    void AddLocalImpulse(
            const GmVec3 &impulse,
            const GmVec3 &point);
    void AddLocalImpulse(
            const GmVec3 &impulse);
    void SetLocalLinearSpeed(
            const GmVec3 &speed);
    void GetLocalLinearSpeed(
            GmVec3 &out) const;
    void SetLocalAngularSpeed(
            const GmVec3 &speed);
    void GetLocalAngularSpeed(
            GmVec3 &out) const;
    void AddReplacement(
            const GmVec3 &replacement);
    void ComputeSynthetizedReplacement(
            GmVec3 &out);
    RuntimeClone CaptureRuntimeClone(void) const;
    bool PrepareRuntimeCloneRestore(const RuntimeClone &clone);
    void RestoreRuntimeClone(RuntimeClone clone) noexcept;

private:
    std::optional<float> maxAngularSpeed;
    CHmsDynaParams dynaParams{};
    CHmsStateDyna tempState{};
    CHmsStateDyna writeState{};
    CHmsStateDyna currentState{};
    std::vector<GmVec3> pendingCollisionReplacements_;
    bool isDynamicActive = false;
    EDynamicType dynamicType = EDynamicType_LinearOnly;
};
