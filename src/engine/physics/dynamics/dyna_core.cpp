// CHmsDyna state and force helpers used by PhysicsStep2.

#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/physics/geometry/geometry_helpers.h"
#include "engine/core/binary32_math.h"
namespace {

constexpr float kVectorLengthEpsilonSquared = 1.0e-10f;
constexpr float kReplacementDirectionEpsilon = 0.01f;

}

void CHmsDyna::CHmsStateDyna::Reset(void) {
    const GmVec3 zero = GmVec3::Zero();
    linearSpeed = zero;
    linearCorrectionSpeed = zero;
    angularSpeed = zero;
    force = zero;
    torque = zero;
    tweakedLinearSpeedValid = false;
    tweakedLinearSpeed = zero;
}

void CHmsDyna::Reset(void) {
    tempState.Reset();
    writeState.Reset();
    currentState.Reset();
    ClearCollisionReplacements();
    maxAngularSpeed.reset();
    isDynamicActive = false;
    dynamicType = EDynamicType_LinearOnly;
}

static float SubtractProductsRounded(float a, float b, float c, float d) {
    return (a * b - c * d);
}

static void CopyDynaState(CHmsDyna::CHmsStateDyna &destination,
                          const CHmsDyna::CHmsStateDyna &source) {
    destination = source;
}

static void CopyIso4PoseToDynaState(CHmsDyna::CHmsStateDyna &state,
                                    const GmIso4 &location) {
    state.rotation = location.rotation;
    state.position = location.translation;
}

float CHmsDyna::InverseMass(void) const {
    return 1.0f / dynaParams.mass;
}

int CHmsDyna::UsesFullAngularDynamics(void) const {
    return dynamicType == EDynamicType_FullAngularDynamics;
}

int CHmsDyna::UsesAngularIntegration(void) const {
    return dynamicType != EDynamicType_LinearOnly;
}

int CHmsDyna::IsKinematicOrFrozen(void) const {
    return dynamicType == EDynamicType_Frozen;
}

void CHmsDyna::Activate(void) {
    if (!isDynamicActive) {
        isDynamicActive = true;
    }
}

GmVec3 CHmsDyna::WorldCenterOfMass(const CHmsDyna::CHmsStateDyna &state) const {
    GmVec3 out;
    GmIso4 stateLocation;
    stateLocation.Set(state.rotation, state.position);
    out.SetMult(dynaParams.localCenterOfMass, stateLocation);
    return out;
}

GmVec3 CHmsDyna::WorldCenterOfMassForAddForceAtPoint(const CHmsDyna::CHmsStateDyna &state) const {
    const GmVec3 *com = &dynaParams.localCenterOfMass;
    GmVec3 out = {
        (
                state.rotation.Element(GmAxis::X, GmAxis::Y) * com->y +
                state.rotation.Element(GmAxis::X, GmAxis::X) * com->x +
                state.rotation.Element(GmAxis::X, GmAxis::Z) * com->z +
                state.position.x),
        (
                state.rotation.Element(GmAxis::Y, GmAxis::Y) * com->y +
                state.rotation.Element(GmAxis::Y, GmAxis::X) * com->x +
                state.rotation.Element(GmAxis::Y, GmAxis::Z) * com->z +
                state.position.y),
        (
                state.rotation.Element(GmAxis::Z, GmAxis::Y) * com->y +
                state.rotation.Element(GmAxis::Z, GmAxis::X) * com->x +
                state.rotation.Element(GmAxis::Z, GmAxis::Z) * com->z +
                state.position.z),
    };
    return out;
}

void CHmsDyna::ValidateDynamicState(void) {
    CHmsDyna *dyna = this;
    CopyDynaState(dyna->writeState, dyna->currentState);
}

void CHmsDyna::CopyStateToTemp(void) {
    CHmsDyna *dyna = this;
    CopyDynaState(dyna->tempState, dyna->currentState);
}

void CHmsDyna::CopyTempToState(void) {
    CHmsDyna *dyna = this;
    CopyDynaState(dyna->writeState, dyna->tempState);
}

void CHmsDyna::SetDynamicType(EDynamicType newDynamicType) {
    dynamicType = newDynamicType;

    const GmVec3 zero = GmVec3::Zero();
    writeState.angularSpeed = zero;
    currentState.angularSpeed = zero;
    writeState.torque = zero;
    currentState.torque = zero;
}

void CHmsDyna::SetLocation(const GmIso4 &location) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *writeState = &dyna->writeState;
    CHmsDyna::CHmsStateDyna *currentState = &dyna->currentState;

    writeState->rotationQuat.Set(location.RotationMatrix());
    CopyIso4PoseToDynaState(*writeState, location);
    writeState->inverseInertiaWorld.SetMult(writeState->rotation,
                                            dyna->dynaParams.bodyInertiaLike);
    writeState->inverseInertiaWorld.MultTranspose(writeState->rotation);

    currentState->rotationQuat.Set(writeState->rotationQuat);
    CopyIso4PoseToDynaState(*currentState, location);
    currentState->inverseInertiaWorld.Set(writeState->inverseInertiaWorld);
}

void CHmsDyna::SetTranslation(const GmVec3 &translation) {
    GmIso4 location;
    location.Set(currentState.rotation, translation);
    SetLocation(location);
}

void CHmsDyna::RotateOf(const GmMat3 &rotation) {
    GmMat3 rotated;
    rotated.SetMult(rotation, currentState.rotation);
    rotated.OrthoNormalize();

    GmIso4 location;
    location.Set(rotated, currentState.position);
    SetLocation(location);
}

void CHmsDyna::ApplyReplacement(
        const GmVec3 &replacement) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->position.x = (state->position.x + replacement.x);
    state->position.y = (replacement.y + state->position.y);
    state->position.z = (replacement.z + state->position.z);
}

void CHmsDyna::SetForce(const GmVec3 &force) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->force.x = (force.x);
    state->force.y = (force.y);
    state->force.z = (force.z);
}

void CHmsDyna::SetTorque(const GmVec3 &torque) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->torque.x = (torque.x);
    state->torque.y = (torque.y);
    state->torque.z = (torque.z);
}

void CHmsDyna::AddForce(
        const GmVec3 &force,
        const GmVec3 &point) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    // Accumulate force, then add the lever-arm torque at the application point.
    state->force.x = (force.x + state->force.x);
    state->force.y = (state->force.y + force.y);
    state->force.z = (state->force.z + force.z);

    GmVec3 center = dyna->WorldCenterOfMassForAddForceAtPoint(*state);
    const float rx = (point.x - center.x);
    const float ry = (point.y - center.y);
    const float rz = (point.z - center.z);
    const float torqueX = SubtractProductsRounded(ry, force.z, rz, force.y);
    const float torqueY = SubtractProductsRounded(rz, force.x, rx, force.z);
    const float torqueZ = SubtractProductsRounded(rx, force.y, ry, force.x);
    state->torque.x = (state->torque.x + torqueX);
    state->torque.y = (state->torque.y + torqueY);
    state->torque.z = (state->torque.z + torqueZ);
}

void CHmsDyna::AddForce(const GmVec3 &force) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->force.x = (state->force.x + force.x);
    state->force.y = (force.y + state->force.y);
    state->force.z = (force.z + state->force.z);
}

void CHmsDyna::GetForce(GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    const CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    out.x = (state->force.x);
    out.y = (state->force.y);
    out.z = (state->force.z);
}

void CHmsDyna::AddTorque(const GmVec3 &torque) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->torque.x = (state->torque.x + torque.x);
    state->torque.y = (torque.y + state->torque.y);
    state->torque.z = (torque.z + state->torque.z);
}

void CHmsDyna::IntegrateStep(
        const CHmsDyna::CHmsStateDyna &source,
        CHmsDyna::CHmsStateDyna &destination,
        float dt) const {
    const CHmsDyna *dyna = this;
    const CHmsDyna::CHmsStateDyna *src = &source;
    CHmsDyna::CHmsStateDyna *dst = &destination;
    if (dyna->IsKinematicOrFrozen()) {
        CopyDynaState(*dst, *src);
        return;
    }

    const float invMass = (dyna->InverseMass());
    const float forceMassX = (src->force.x * invMass);
    const float forceMassY = (src->force.y * invMass);
    const float forceMassZ = (src->force.z * invMass);

    float linearPositionDeltaX = (src->linearSpeed.x * dt);
    float linearPositionDeltaY = (src->linearSpeed.y * dt);
    float linearPositionDeltaZ = (src->linearSpeed.z * dt);
    dst->position.x = (linearPositionDeltaX + src->position.x);
    dst->position.y = (src->position.y + linearPositionDeltaY);
    dst->position.z = (src->position.z + linearPositionDeltaZ);

    float correctionPositionDeltaX =
            (src->linearCorrectionSpeed.x * dt);
    float correctionPositionDeltaY =
            (src->linearCorrectionSpeed.y * dt);
    float correctionPositionDeltaZ =
            (src->linearCorrectionSpeed.z * dt);
    dst->position.x = (dst->position.x + correctionPositionDeltaX);
    dst->position.y = (dst->position.y + correctionPositionDeltaY);
    dst->position.z = (dst->position.z + correctionPositionDeltaZ);
    dst->linearCorrectionSpeed = GmVec3::Zero();

    float forceSpeedDeltaX = (forceMassX * dt);
    float forceSpeedDeltaY = (forceMassY * dt);
    float forceSpeedDeltaZ = (dt * forceMassZ);
    dst->linearSpeed.x = (src->linearSpeed.x + forceSpeedDeltaX);
    dst->linearSpeed.y = (src->linearSpeed.y + forceSpeedDeltaY);
    dst->linearSpeed.z = (src->linearSpeed.z + forceSpeedDeltaZ);

    if (!dyna->UsesAngularIntegration()) {
        dst->rotation.Set(src->rotation);
        return;
    }

    GmVec3 angularAccel;
    angularAccel.SetMult(src->torque, src->inverseInertiaWorld);

    const float angularSpeedLen2 = src->angularSpeed.LengthSquaredYXZ();
    if (!(kVectorLengthEpsilonSquared < angularSpeedLen2)) {
        dst->rotation.Set(src->rotation);
        dst->rotationQuat.Set(src->rotationQuat);
    } else {
        const GmQuat sourceQuat = src->rotationQuat;
        const GmVec3 angularSpeed = src->angularSpeed;
        const float halfScale = 0.5f;
        GmQuat quatDerivative;

        quatDerivative.w = (
                (((-angularSpeed.x * sourceQuat.x -
                   angularSpeed.y * sourceQuat.y) -
                  sourceQuat.z * angularSpeed.z) *
                 halfScale));
        quatDerivative.x = (
                (((sourceQuat.w * angularSpeed.x +
                   angularSpeed.y * sourceQuat.z) -
                  sourceQuat.y * angularSpeed.z) *
                 halfScale));
        quatDerivative.y = (
                (((angularSpeed.y * sourceQuat.w -
                   sourceQuat.z * angularSpeed.x) +
                  angularSpeed.z * sourceQuat.x) *
                 halfScale));
        quatDerivative.z = (
                (((sourceQuat.y * angularSpeed.x -
                   angularSpeed.y * sourceQuat.x) +
                  sourceQuat.w * angularSpeed.z) *
                 halfScale));

        dst->rotationQuat.Set(src->rotationQuat);
        dst->rotationQuat.w = (
                quatDerivative.w * dt + dst->rotationQuat.w);
        dst->rotationQuat.x = (
                quatDerivative.x * dt + dst->rotationQuat.x);
        dst->rotationQuat.y = (
                quatDerivative.y * dt + dst->rotationQuat.y);
        dst->rotationQuat.z = (
                dt * quatDerivative.z + dst->rotationQuat.z);
        dst->rotationQuat.Normalize();
        dst->rotation.Set(dst->rotationQuat);

        GmVec3 oldCom;
        GmVec3 newCom;
        oldCom.SetMult(dyna->dynaParams.localCenterOfMass, src->rotation);
        newCom.SetMult(dyna->dynaParams.localCenterOfMass, dst->rotation);
        const float deltaComX = (newCom.x - oldCom.x);
        const float deltaComY = (newCom.y - oldCom.y);
        const float deltaComZ = (newCom.z - oldCom.z);
        dst->position.x = (dst->position.x - deltaComX);
        dst->position.y = (dst->position.y - deltaComY);
        dst->position.z = (dst->position.z - deltaComZ);
    }

    float torqueSpeedDeltaX = (angularAccel.x * dt);
    float torqueSpeedDeltaY = (angularAccel.y * dt);
    float torqueSpeedDeltaZ = (dt * angularAccel.z);
    dst->angularSpeed.x = (src->angularSpeed.x + torqueSpeedDeltaX);
    dst->angularSpeed.y = (src->angularSpeed.y + torqueSpeedDeltaY);
    dst->angularSpeed.z = (src->angularSpeed.z + torqueSpeedDeltaZ);

    if (dyna->maxAngularSpeed.has_value()) {
        float angularLen2 = (
            (dst->angularSpeed.y * dst->angularSpeed.y +
                             dst->angularSpeed.x * dst->angularSpeed.x) +
            dst->angularSpeed.z * dst->angularSpeed.z);
        const float configuredMax = *dyna->maxAngularSpeed;
        if (configuredMax * configuredMax < angularLen2) {
            float maxAngularSpeed = (configuredMax);
            float angularSpeedClampScale =
                    (maxAngularSpeed / CIsqrt(angularLen2));
            dst->angularSpeed.x =
                    (dst->angularSpeed.x * angularSpeedClampScale);
            dst->angularSpeed.y =
                    (angularSpeedClampScale * dst->angularSpeed.y);
            dst->angularSpeed.z =
                    (angularSpeedClampScale * dst->angularSpeed.z);
        }
    }

    dst->inverseInertiaWorld.SetTranspose(dst->rotation);
    dst->inverseInertiaWorld.Mult(dyna->dynaParams.bodyInertiaLike);
    dst->inverseInertiaWorld.Mult(dst->rotation);
}

void CHmsDyna::DoPreCollisionDynamic(float dt) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna srcCopy;
    CopyDynaState(srcCopy, dyna->currentState);
    dyna->IntegrateStep(srcCopy, dyna->currentState, dt);
    dyna->ClearCollisionReplacements();
}

void CHmsDyna::AddImpulse(
        const GmVec3 &impulse,
        const GmVec3 &point) {
    CHmsDyna *dyna = this;
    if (dyna->IsKinematicOrFrozen()) {
        return;
    }

    dyna->Activate();
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    const float invMass = (dyna->InverseMass());
    const float impulseMassX = (impulse.x * invMass);
    const float impulseMassY = (impulse.y * invMass);
    const float impulseMassZ = (impulse.z * invMass);
    state->linearSpeed.x = (state->linearSpeed.x + impulseMassX);
    state->linearSpeed.y = (state->linearSpeed.y + impulseMassY);
    state->linearSpeed.z = (impulseMassZ + state->linearSpeed.z);

    if (dyna->UsesFullAngularDynamics()) {
        GmVec3 center = dyna->WorldCenterOfMass(*state);
        const float rx = (point.x - center.x);
        const float ry = (point.y - center.y);
        const float rz = (point.z - center.z);
        GmVec3 angularImpulse = {
            (ry * impulse.z - rz * impulse.y),
            (rz * impulse.x - rx * impulse.z),
            (rx * impulse.y - ry * impulse.x),
        };
        angularImpulse.Mult(state->inverseInertiaWorld);
        state->angularSpeed.x = (state->angularSpeed.x + angularImpulse.x);
        state->angularSpeed.y = (state->angularSpeed.y + angularImpulse.y);
        state->angularSpeed.z = (angularImpulse.z + state->angularSpeed.z);
    }
}

void CHmsDyna::AddImpulse(const GmVec3 &impulse) {
    CHmsDyna *dyna = this;
    if (dyna->IsKinematicOrFrozen()) {
        return;
    }

    dyna->Activate();
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    const float invMass = (dyna->InverseMass());
    const float impulseMassX = (impulse.x * invMass);
    const float impulseMassY = (impulse.y * invMass);
    const float impulseMassZ = (impulse.z * invMass);
    state->linearSpeed.x = (impulseMassX + state->linearSpeed.x);
    state->linearSpeed.y = (state->linearSpeed.y + impulseMassY);
    state->linearSpeed.z = (impulseMassZ + state->linearSpeed.z);
}

void CHmsDyna::GetSpeed(
        const GmVec3 &point,
        GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    const CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    if (dyna->IsKinematicOrFrozen()) {
        out = GmVec3::Zero();
        return;
    }

    out.x = (state->linearSpeed.x);
    out.y = (state->linearSpeed.y);
    out.z = (state->linearSpeed.z);
    if (dyna->UsesFullAngularDynamics()) {
        GmVec3 center;
        GmIso4 stateLocation;
        stateLocation.Set(
                state->rotation,
                state->position);
        center.SetMult(dyna->dynaParams.localCenterOfMass, stateLocation);
        const float rx = (point.x - center.x);
        const float ry = (point.y - center.y);
        const float rz = (point.z - center.z);
        const float speedX = (rz * state->angularSpeed.y -
                                              state->angularSpeed.z * ry);
        const float speedY = (state->angularSpeed.z * rx -
                                              rz * state->angularSpeed.x);
        const float speedZ = (ry * state->angularSpeed.x -
                                              rx * state->angularSpeed.y);
        out.x = (out.x + speedX);
        out.y = (out.y + speedY);
        out.z = (out.z + speedZ);
    }
}

void CHmsDyna::GetLinearSpeed(GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    const CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    out.x = (state->linearSpeed.x);
    out.y = (state->linearSpeed.y);
    out.z = (state->linearSpeed.z);
}

void CHmsDyna::SetLinearSpeed(const GmVec3 &speed) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->linearSpeed.x = (speed.x);
    state->linearSpeed.y = (speed.y);
    state->linearSpeed.z = (speed.z);
}

void CHmsDyna::GetAngularSpeed(GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    const CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    out.x = (state->angularSpeed.x);
    out.y = (state->angularSpeed.y);
    out.z = (state->angularSpeed.z);
}

void CHmsDyna::SetAngularSpeed(const GmVec3 &speed) {
    CHmsDyna *dyna = this;
    CHmsDyna::CHmsStateDyna *state = &dyna->currentState;
    state->angularSpeed.x = (speed.x);
    state->angularSpeed.y = (speed.y);
    state->angularSpeed.z = (speed.z);
}

GmVec3 CHmsDyna::LocalDirectionToWorld(const GmVec3 &local) const {
    const CHmsDyna::CHmsStateDyna *state = &currentState;
    const GmMat3 *rot = &state->rotation;
    // dot product and rounds only the final component before AddForce/AddTorque.
    GmVec3 world = {
        (
                rot->Element(GmAxis::X, GmAxis::Y) * local.y +
                rot->Element(GmAxis::X, GmAxis::X) * local.x +
                rot->Element(GmAxis::X, GmAxis::Z) * local.z),
        (
                rot->Element(GmAxis::Y, GmAxis::X) * local.x +
                rot->Element(GmAxis::Y, GmAxis::Y) * local.y +
                rot->Element(GmAxis::Y, GmAxis::Z) * local.z),
        (
                rot->Element(GmAxis::Z, GmAxis::X) * local.x +
                rot->Element(GmAxis::Z, GmAxis::Y) * local.y +
                rot->Element(GmAxis::Z, GmAxis::Z) * local.z),
    };
    return world;
}

GmVec3 CHmsDyna::LocalPointToWorld(const GmVec3 &local) const {
    const CHmsDyna::CHmsStateDyna *state = &currentState;
    const GmMat3 *rot = &state->rotation;
    // position term before the single component rounds.
    GmVec3 world = {
        (
                rot->Element(GmAxis::X, GmAxis::Y) * local.y +
                rot->Element(GmAxis::X, GmAxis::X) * local.x +
                rot->Element(GmAxis::X, GmAxis::Z) * local.z +
                state->position.x),
        (
                rot->Element(GmAxis::Y, GmAxis::Y) * local.y +
                rot->Element(GmAxis::Y, GmAxis::X) * local.x +
                rot->Element(GmAxis::Y, GmAxis::Z) * local.z +
                state->position.y),
        (
                rot->Element(GmAxis::Z, GmAxis::Y) * local.y +
                rot->Element(GmAxis::Z, GmAxis::X) * local.x +
                rot->Element(GmAxis::Z, GmAxis::Z) * local.z +
                state->position.z),
    };
    return world;
}

void CHmsDyna::WorldDirectionToLocal(GmVec3 &world) const {
    world.MultTranspose(currentState.rotation);
}

void CHmsDyna::AddLocalForce(
        const GmVec3 &localForce,
        const GmVec3 &localPoint) {
    CHmsDyna *dyna = this;
    GmVec3 force = dyna->LocalDirectionToWorld(localForce);
    GmVec3 point = dyna->LocalPointToWorld(localPoint);
    dyna->AddForce(force, point);
}

void CHmsDyna::SetLocalForce(const GmVec3 &localForce) {
    CHmsDyna *dyna = this;
    GmVec3 force = dyna->LocalDirectionToWorld(localForce);
    dyna->SetForce(force);
}

void CHmsDyna::AddLocalForce(const GmVec3 &localForce) {
    CHmsDyna *dyna = this;
    GmVec3 force = dyna->LocalDirectionToWorld(localForce);
    dyna->AddForce(force);
}

void CHmsDyna::GetLocalForce(GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    dyna->GetForce(out);
    dyna->WorldDirectionToLocal(out);
}

void CHmsDyna::SetLocalTorque(const GmVec3 &localTorque) {
    CHmsDyna *dyna = this;
    GmVec3 torque = dyna->LocalDirectionToWorld(localTorque);
    dyna->SetTorque(torque);
}

void CHmsDyna::AddLocalTorque(const GmVec3 &localTorque) {
    CHmsDyna *dyna = this;
    GmVec3 torque = dyna->LocalDirectionToWorld(localTorque);
    dyna->AddTorque(torque);
}

void CHmsDyna::AddLocalImpulse(
        const GmVec3 &localImpulse,
        const GmVec3 &localPoint) {
    CHmsDyna *dyna = this;
    GmVec3 impulse = dyna->LocalDirectionToWorld(localImpulse);
    GmVec3 point = dyna->LocalPointToWorld(localPoint);
    dyna->AddImpulse(impulse, point);
}

void CHmsDyna::AddLocalImpulse(const GmVec3 &localImpulse) {
    CHmsDyna *dyna = this;
    GmVec3 impulse = dyna->LocalDirectionToWorld(localImpulse);
    dyna->AddImpulse(impulse);
}

void CHmsDyna::SetLocalLinearSpeed(
        const GmVec3 &localSpeed) {
    CHmsDyna *dyna = this;
    GmVec3 speed = dyna->LocalDirectionToWorld(localSpeed);
    dyna->SetLinearSpeed(speed);
}

void CHmsDyna::GetLocalLinearSpeed(GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    dyna->GetLinearSpeed(out);
    dyna->WorldDirectionToLocal(out);
}

void CHmsDyna::SetLocalAngularSpeed(
        const GmVec3 &localSpeed) {
    CHmsDyna *dyna = this;
    GmVec3 speed = dyna->LocalDirectionToWorld(localSpeed);
    dyna->SetAngularSpeed(speed);
}

void CHmsDyna::GetLocalAngularSpeed(GmVec3 &out) const {
    const CHmsDyna *dyna = this;
    dyna->GetAngularSpeed(out);
    dyna->WorldDirectionToLocal(out);
}

void CHmsDyna::AddReplacement(const GmVec3 &replacement) {
    CHmsDyna *dyna = this;
    dyna->Activate();
    dyna->pendingCollisionReplacements_.push_back(replacement);
}

void CHmsDyna::ClearCollisionReplacements(void) {
    pendingCollisionReplacements_.clear();
}

void CHmsDyna::ComputeSynthetizedReplacement(GmVec3 &out) {
    CHmsDyna *dyna = this;
    u32 count = static_cast<u32>(dyna->pendingCollisionReplacements_.size());
    if (count == 0) {
        out = GmVec3::Zero();
        return;
    }

    // Remove forward projections so replacements do not reinforce each other.
    const GmVec3 *items = dyna->pendingCollisionReplacements_.data();
    float sumX = (items[0].x);
    float sumY = (items[0].y);
    float sumZ = (items[0].z);
    for (u32 replacementIndex = 1; replacementIndex < count; replacementIndex++) {
        const GmVec3 *next = &items[replacementIndex];
        float workX = sumX;
        float workY = sumY;
        float workZ = sumZ;
        float projection = (
                (next->x * sumX +
                 sumY * next->y) +
                sumZ * next->z);
        if (projection > 0.0f) {
            float sumLen2 = (
                    sumZ * sumZ +
                    (sumY * sumY + workX * workX));
            if (kVectorLengthEpsilonSquared < sumLen2) {
                float projectionClamped = projection;
                if (sumLen2 < projectionClamped) {
                    projectionClamped = sumLen2;
                }
                float projectionScale = (
                        projectionClamped / sumLen2);
                float projectedX = (
                        projectionScale * workX);
                float projectedY = (
                        projectionScale * workY);
                float projectedZ = (
                        projectionScale * workZ);
                float adjustedX = (workX - projectedX);
                float adjustedY = (workY - projectedY);
                float adjustedZ = (workZ - projectedZ);
                workX = adjustedX;
                workY = adjustedY;
                workZ = adjustedZ;
            }
        }
        sumX = (workX + next->x);
        sumY = (workY + next->y);
        sumZ = (workZ + next->z);
    }

    float len2 = (
            sumZ * sumZ +
            (sumY * sumY +
             sumX * sumX));
    const float epsilonSq = (
            kReplacementDirectionEpsilon *
            kReplacementDirectionEpsilon);
    if (!(epsilonSq < len2)) {
        out = GmVec3::Zero();
        return;
    }

    const float root = (CIsqrt(len2));
    const float invRoot = (1.0f / root);
    const float unitX = (invRoot * sumX);
    const float unitY = (invRoot * sumY);
    const float unitZ = (invRoot * sumZ);
    const float epsilonX = (
            kReplacementDirectionEpsilon * unitX);
    const float epsilonY = (
            kReplacementDirectionEpsilon * unitY);
    const float epsilonZ = (
            kReplacementDirectionEpsilon * unitZ);
    out.x = (sumX - epsilonX);
    out.y = (sumY - epsilonY);
    out.z = (sumZ - epsilonZ);
}

void CHmsDyna::DoPostCollisionDynamic(void) {
    CHmsDyna *dyna = this;
    GmVec3 replacement;
    dyna->ComputeSynthetizedReplacement(replacement);
    dyna->ApplyReplacement(replacement);
}
CHmsDyna::RuntimeClone CHmsDyna::CaptureRuntimeClone(void) const {
    RuntimeClone clone;
    clone.maxAngularSpeed = maxAngularSpeed;
    clone.dynaParams = dynaParams;
    clone.tempState = tempState;
    clone.writeState = writeState;
    clone.currentState = currentState;
    clone.pendingCollisionReplacements = pendingCollisionReplacements_;
    clone.isDynamicActive = isDynamicActive;
    clone.dynamicType = dynamicType;
    return clone;
}

bool CHmsDyna::PrepareRuntimeCloneRestore(const RuntimeClone &clone) {
    try {
        pendingCollisionReplacements_.reserve(
                clone.pendingCollisionReplacements.size());
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

void CHmsDyna::RestoreRuntimeClone(RuntimeClone clone) noexcept {
    maxAngularSpeed = clone.maxAngularSpeed;
    dynaParams = clone.dynaParams;
    tempState = clone.tempState;
    writeState = clone.writeState;
    currentState = clone.currentState;
    pendingCollisionReplacements_.swap(clone.pendingCollisionReplacements);
    isDynamicActive = clone.isDynamicActive;
    dynamicType = clone.dynamicType;
}
