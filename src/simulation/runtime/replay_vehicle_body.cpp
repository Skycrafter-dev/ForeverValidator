#include "simulation/runtime/replay_vehicle_body.h"
#include <memory>

#include "engine/physics/world/hms_zone.h"
namespace {

ReplayDynaFrameState CaptureFrame(const CHmsDyna::CHmsStateDyna &source) {
    ReplayDynaFrameState frame;
    frame.rotationQuaternion = source.rotationQuat;
    frame.rotation = source.rotation;
    frame.position = source.position;
    frame.linearSpeed = source.linearSpeed;
    frame.linearCorrectionSpeed = source.linearCorrectionSpeed;
    frame.angularSpeed = source.angularSpeed;
    frame.force = source.force;
    frame.torque = source.torque;
    frame.inverseInertiaWorld = source.inverseInertiaWorld;
    return frame;
}

} // namespace

CHmsDyna::CHmsDyna(void) = default;

CHmsDyna::~CHmsDyna(void) = default;

CHmsCorpus::CHmsCorpus(void) = default;

CHmsCorpus::~CHmsCorpus(void) {
    DetachFromWorld();
}

ReplayVehicleBody::ReplayVehicleBody() {
    CMwNodRef<CPlugSolid> solid = MakeMwNod<CPlugSolid>();
    item.SetSolid(solid.Get());
}

void CHmsCorpus::SetItem(CHmsItem *newItem) {
    if (item == newItem) {
        return;
    }
    if (item != nullptr) {
        item->RemoveCorpus(this);
    }
    item = newItem;
    if (item != nullptr) {
        item->AddCorpus(this);
    }
}

void CHmsCorpus::DetachFromWorld(void) {
    if (zone_ != nullptr) {
        zone_->RemoveCorpus(this);
    }
    if (collisionManagerZone != nullptr) {
        collisionManagerZone->RemoveCorpus(this);
    }
    SetItem(nullptr);
}

CHmsZone *CHmsCorpus::OwningZone(void) const {
    return zone_;
}

void CHmsCorpus::BindCollisionManagerRegistration(
        CHmsCollisionManagerSZone &managerZone,
        CHmsCollisionManagerSGroup *managerGroup,
        CHmsCorpusId id) {
    collisionManagerZone = &managerZone;
    collisionManagerGroup = managerGroup;
    collisionManagerId = id;
}

void CHmsCorpus::ClearCollisionManagerRegistration(
        const CHmsCollisionManagerSZone &managerZone) {
    if (collisionManagerZone != &managerZone) {
        return;
    }
    collisionManagerZone = nullptr;
    collisionManagerGroup = nullptr;
    collisionManagerId = {};
}

CHmsCollisionManagerSZone *
CHmsCorpus::RegisteredCollisionManagerZone(void) const {
    return collisionManagerZone;
}

CHmsCollisionManagerSGroup *
CHmsCorpus::RegisteredCollisionManagerGroup(void) const {
    return collisionManagerGroup;
}

void CHmsCorpus::Reset(void) {
    if (dyna != nullptr) {
        dyna->Reset();
    }
}

void ReplayVehicleBody::InitializeAtSpawn(
        const ReplayDynaParameters &parameters,
        const GmIso4 &spawnLocation) {
    dyna.SetDynamicType(CHmsDyna::EDynamicType_FullAngularDynamics);
    InstallDynaParameters(parameters);
    dyna.SetLocation(spawnLocation);
    dyna.SetAngularSpeedLimit(100.0f);
    dyna.SetDynamicActive(true);
}

void ReplayVehicleBody::SetSpawnLocation(
        const std::optional<GmIso4> &location) {
    if (!location.has_value()) {
        return;
    }
    dyna.SetLocation(*location);
}

void ReplayVehicleBody::ConstructItem(
        const CHmsItem::Properties &properties) {
    item.SetProperties(properties);
}

void ReplayVehicleBody::BuildCorpus() {
    corpus.SetDynamics(&dyna);
    corpus.SetItem(&item);
}

ReplayDynaParameters
ReplayVehicleBody::CaptureDynaParameters() const {
    const CHmsDynaParams &source = dyna.Parameters();
    ReplayDynaParameters parameters;
    parameters.mass = source.mass;
    parameters.inverseBodyInertia = source.bodyInertiaLike;
    parameters.linearDampingScale = source.linearDampingScale;
    parameters.angularDampingScale = source.angularDampingScale;
    parameters.maxStepDistance = source.maxStepDistance;
    parameters.forceScale = source.forceScale;
    parameters.localCenterOfMass = source.localCenterOfMass;
    return parameters;
}

ReplayDynaFrameState ReplayVehicleBody::CaptureCurrentFrame() const {
    return CaptureFrame(dyna.CurrentState());
}

ReplayDynaFrameState
ReplayVehicleBody::CaptureWriteState() const {
    return CaptureFrame(dyna.WriteState());
}

void ReplayVehicleBody::InstallDynaParameters(
        const ReplayDynaParameters &parameters) {
    CHmsDynaParams &target = dyna.Parameters();
    target.mass = parameters.mass;
    target.bodyInertiaLike = parameters.inverseBodyInertia;
    target.linearDampingScale = parameters.linearDampingScale;
    target.angularDampingScale = parameters.angularDampingScale;
    target.maxStepDistance = parameters.maxStepDistance;
    target.forceScale = parameters.forceScale;
    target.localCenterOfMass = parameters.localCenterOfMass;
}

void ReplayVehicleBody::InstallPhysicalParameters(
        const ReplayDynaParameters &parameters) {
    CPlugPhysicalObject &target = Solid().Physical();
    target.Parameters().mass = parameters.mass;
    target.Parameters().impulseInertia = parameters.inverseBodyInertia;
    target.Parameters().linearFluidFriction = parameters.linearDampingScale;
    target.Parameters().physicalResponseCoefA = parameters.angularDampingScale;
    target.Parameters().physicalResponseCoefB = parameters.maxStepDistance;
    target.Parameters().vehicleContactFeedbackScale = parameters.forceScale;
    target.Parameters().localCenterOfMass = parameters.localCenterOfMass;
}

void ReplayVehicleBody::InstallEmptyCollisionTree() {
    auto root = std::make_unique<CPlugTree>();
    CPlugTree::SFlags state;
    state.visible = false;
    state.collisionEnabled = true;
    state.pickableVisual = false;
    state.castsShadows = false;
    state.rooted = false;
    state.locationDirty = false;
    root->ApplyLoadedState(state);
    Solid().SetTree(root.release(), 0);
}

CHmsDyna &ReplayVehicleBody::Dyna() {
    return dyna;
}

CHmsItem &ReplayVehicleBody::Item() {
    return item;
}

CPlugSolid &ReplayVehicleBody::Solid() {
    return *item.Solid();
}

CHmsCorpus &ReplayVehicleBody::Corpus() {
    return corpus;
}

ReplayVehicleBody::RuntimeClone
ReplayVehicleBody::CaptureRuntimeClone() const {
    return dyna.CaptureRuntimeClone();
}

bool ReplayVehicleBody::PrepareRuntimeCloneRestore(
        const RuntimeClone &clone) {
    return dyna.PrepareRuntimeCloneRestore(clone);
}

void ReplayVehicleBody::RestoreRuntimeClone(RuntimeClone clone) noexcept {
    dyna.RestoreRuntimeClone(std::move(clone));
}
