#include "engine/scene/scene_mobil.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "engine/physics/dynamics/hms_item.h"
#include "engine/rendering/plug_material.h"
#include "engine/rendering/plug_shader.h"
#include "engine/scene/plug_solid.h"
#include "engine/rendering/plug_tree.h"
#include "engine/rendering/plug_visual.h"
void CSceneMobil::SetOwningBlock(CGameCtnBlock *block) {
    owningBlock = block;
}

CGameCtnBlock *CSceneMobil::OwningBlock(void) const {
    return owningBlock;
}

int CSceneMobil::IsZombie(void) const {
    const CHmsItem *item = HmsItem();
    return item != nullptr && item->GetProperties().zombie;
}

CPlugSolid *CSceneMobil::ExistingSolid(void) const {
    CHmsItem *item = HmsItem();
    return item != nullptr ? item->Solid() : nullptr;
}

int CSceneMobil::DoesMotionChangeLocations(void) {
    return replayMotionChangesLocations ? 1 : 0;
}

void CSceneMobil::SetReplayMotionChangesLocations(int changesLocations) {
    replayMotionChangesLocations = changesLocations != 0;
}

void CSceneMobil::SetReplayCollisionUsesInitialTransform(
        int usesInitialTransform) {
    replayCollisionUsesInitialTransform = usesInitialTransform != 0;
}

int CSceneMobil::ReplayCollisionUsesInitialTransform(void) const {
    return replayCollisionUsesInitialTransform ? 1 : 0;
}

void CSceneMobil::AddTree(CPlugTree *tree) {
    if (tree == nullptr) {
        return;
    }
    CMwNodRef<CPlugSolid> createdSolid;
    CPlugSolid *solid = ExistingSolid();
    if (solid == nullptr) {
        createdSolid = MakeMwNod<CPlugSolid>();
        solid = createdSolid.Get();
        HmsItem()->SetSolid(solid);
    }
    solid->AddTree(tree);
}


CPlugTree *CSceneMobil::FindOrAddNewTreeWithId(
        const CMwId &id) {
    CPlugSolid *solid = ExistingSolid();
    if (solid != nullptr && solid->CollisionTree() != nullptr) {
        CPlugTree *found = solid->CollisionTree()->GetPlugFromId(id);
        if (found != nullptr) {
            return found;
        }
    }

    auto newTree = std::make_unique<CPlugTree>();
    newTree->SetPlugId(id);
    CPlugTree *result = newTree.get();
    AddTree(newTree.release());
    return result;
}


CPlugTree *CSceneMobil::SetVisual(
        CPlugVisual *visual,
        CPlugShader *shader,
        CPlugMaterial *material) {
    CMwNodRef<CPlugSolid> createdSolid;
    CPlugSolid *solid = ExistingSolid();
    if (solid == nullptr) {
        createdSolid = MakeMwNod<CPlugSolid>();
        solid = createdSolid.Get();
        HmsItem()->SetSolid(solid);
    }
    auto tree = std::make_unique<CPlugTree>();
    tree->SetVisual(visual, shader, material, 1);
    tree->UpdateBoundingBox(1u);
    solid->SetOwnedTree(std::move(tree), 1u);
    return solid->CollisionTree();
}


CPlugTree *CSceneMobil::AddVisual(
        CPlugVisual *visual,
        CPlugShader *shader,
        CPlugMaterial *material) {
    auto tree = std::make_unique<CPlugTree>();
    tree->SetVisual(visual, shader, material, 1);
    CPlugTree *result = tree.get();
    AddTree(tree.release());
    return result;
}


class CSceneMobilCallbackBinding {
public:
    explicit CSceneMobilCallbackBinding(CSceneMobil &owner)
        : owner_(&owner),
          callbacks_(*this) {}

    void Install(CHmsItem &item,
                 bool physicsUpdatesEnabled,
                 bool absorbContactEnabled) {
        item.CallbackSet(
                CHmsItem::ECallback_ComputeForces,
                physicsUpdatesEnabled
                        ? static_cast<CHmsItem::CCallbackComputeForces *>(
                                  &callbacks_)
                        : nullptr);
        item.CallbackSet(
                CHmsItem::ECallback_AfterContacts,
                physicsUpdatesEnabled
                        ? static_cast<CHmsItem::CCallbackAfterContacts *>(
                                  &callbacks_)
                        : nullptr);
        item.CallbackSet(
                CHmsItem::ECallback_AbsorbContact,
                absorbContactEnabled
                        ? static_cast<CHmsItem::CCallbackAbsorbContact *>(
                                  &callbacks_)
                        : nullptr);
    }

    void Uninstall(CHmsItem &item) {
        auto *computeForces =
                static_cast<CHmsItem::CCallbackComputeForces *>(&callbacks_);
        auto *afterContacts =
                static_cast<CHmsItem::CCallbackAfterContacts *>(&callbacks_);
        auto *absorbContact =
                static_cast<CHmsItem::CCallbackAbsorbContact *>(&callbacks_);
        if (item.CallbackGet(CHmsItem::ECallback_ComputeForces) ==
            computeForces) {
            item.CallbackSet(CHmsItem::ECallback_ComputeForces, nullptr);
        }
        if (item.CallbackGet(CHmsItem::ECallback_AfterContacts) ==
            afterContacts) {
            item.CallbackSet(CHmsItem::ECallback_AfterContacts, nullptr);
        }
        if (item.CallbackGet(CHmsItem::ECallback_AbsorbContact) ==
            absorbContact) {
            item.CallbackSet(CHmsItem::ECallback_AbsorbContact, nullptr);
        }
    }

    void InvalidateOwner() noexcept { owner_ = nullptr; }

private:
    class Callbacks final
            : public CHmsItem::CCallbackComputeForces,
              public CHmsItem::CCallbackAfterContacts,
              public CHmsItem::CCallbackAbsorbContact {
    public:
        explicit Callbacks(CSceneMobilCallbackBinding &binding)
            : binding_(binding) {}

        void ComputeForces(CHmsItem *, float dt) override {
            if (binding_.owner_ != nullptr) {
                binding_.owner_->HmsComputeForces(dt);
            }
        }

        void AfterContacts(CHmsItem *) override {
            if (binding_.owner_ != nullptr) {
                binding_.owner_->HmsAfterContacts();
            }
        }

        void AbsorbContact(
                CHmsItem *, CHmsPhysicalContact &contact) override {
            if (binding_.owner_ != nullptr) {
                binding_.owner_->AbsorbContact(contact);
            }
        }

    private:
        CSceneMobilCallbackBinding &binding_;
    };

    CSceneMobil *owner_;
    Callbacks callbacks_;
};

CSceneMobil::CSceneMobil(void)
        : callbackBinding(
                  std::make_unique<CSceneMobilCallbackBinding>(*this)) {
    hmsItem = std::make_unique<CHmsItem>();
    hmsItem->SetSceneMobilOwner(this);
    RefreshPhysicsCallbacks(*hmsItem);
}

CSceneMobil::~CSceneMobil(void) {
    DetachPhysicsItem();
    for (CHmsItem *item : triggerPhysicsItems) {
        if (item != nullptr) {
            ClearPhysicsCallbacks(*item);
            if (item->SceneMobilOwner() == this) {
                item->SetSceneMobilOwner(nullptr);
            }
        }
    }
    triggerPhysicsItems.clear();
    if (hmsItem != nullptr) {
        ClearPhysicsCallbacks(*hmsItem);
        hmsItem->SetSceneMobilOwner(nullptr);
    }
    callbackBinding->InvalidateOwner();
    if (visibilityMotion.Get() != nullptr) {
        visibilityMotion->Stop();
    }
    for (CMwNodRef<CSceneObjectLink> &link : links) {
        if (link.Get() != nullptr) {
            link->SetMobil(nullptr);
        }
    }
}

CHmsItem *CSceneMobil::HmsItem(void) const {
    return attachedPhysicsItem != nullptr
            ? attachedPhysicsItem
            : hmsItem.get();
}

void CSceneMobil::AttachPhysicsItem(CHmsItem &item) {
    if (attachedPhysicsItem == &item) {
        return;
    }
    CSceneMobil *itemOwner = item.SceneMobilOwner();
    if (itemOwner != nullptr && itemOwner != this) {
        throw std::logic_error("physics item already belongs to another mobil");
    }

    DetachPhysicsItem();
    if (hmsItem != nullptr) {
        hmsItem->SetSceneMobilOwner(nullptr);
    }
    attachedPhysicsItem = &item;
    attachedPhysicsItem->SetSceneMobilOwner(this);
    RefreshPhysicsCallbacks(*attachedPhysicsItem);
}

void CSceneMobil::DetachPhysicsItem(void) {
    if (attachedPhysicsItem == nullptr) {
        return;
    }
    ClearPhysicsCallbacks(*attachedPhysicsItem);
    if (attachedPhysicsItem->SceneMobilOwner() == this) {
        attachedPhysicsItem->SetSceneMobilOwner(nullptr);
    }
    attachedPhysicsItem = nullptr;
    if (hmsItem != nullptr) {
        hmsItem->SetSceneMobilOwner(this);
        RefreshPhysicsCallbacks(*hmsItem);
    }
}

void CSceneMobil::AttachTriggerPhysicsItem(CHmsItem &item) {
    if (std::find(triggerPhysicsItems.begin(), triggerPhysicsItems.end(),
                  &item) != triggerPhysicsItems.end()) {
        return;
    }
    CSceneMobil *itemOwner = item.SceneMobilOwner();
    if (itemOwner != nullptr && itemOwner != this) {
        throw std::logic_error("trigger physics item already has another owner");
    }
    triggerPhysicsItems.push_back(&item);
    item.SetSceneMobilOwner(this);
    RefreshPhysicsCallbacks(item);
}

void CSceneMobil::DetachTriggerPhysicsItem(CHmsItem &item) {
    auto found = std::find(triggerPhysicsItems.begin(),
                           triggerPhysicsItems.end(), &item);
    if (found == triggerPhysicsItems.end()) {
        return;
    }
    ClearPhysicsCallbacks(item);
    triggerPhysicsItems.erase(found);
    if (item.SceneMobilOwner() == this) {
        item.SetSceneMobilOwner(nullptr);
    }
}

void CSceneMobil::OnPhysicsItemDestroyed(CHmsItem &item) {
    if (attachedPhysicsItem == &item) {
        attachedPhysicsItem = nullptr;
        if (hmsItem != nullptr) {
            hmsItem->SetSceneMobilOwner(this);
            RefreshPhysicsCallbacks(*hmsItem);
        }
    }
    auto trigger = std::find(triggerPhysicsItems.begin(),
                             triggerPhysicsItems.end(), &item);
    if (trigger != triggerPhysicsItems.end()) {
        triggerPhysicsItems.erase(trigger);
    }
}

void CSceneMobil::RefreshPhysicsCallbacks(CHmsItem &item) {
    callbackBinding->Install(
            item, physicsUpdatesEnabled, absorbContactEnabled);
}

void CSceneMobil::ClearPhysicsCallbacks(CHmsItem &item) {
    callbackBinding->Uninstall(item);
}

CSceneMobil *CSceneMobil::CreateModelInstance(void) {
    CSceneMobil *clone = new CSceneMobil();
    clone->SetModel(this);
    clone->id = id;
    clone->staticSolidAsset = staticSolidAsset;
    clone->initialItemProperties = initialItemProperties;
    clone->replayMotionChangesLocations = replayMotionChangesLocations;
    clone->replayCollisionUsesInitialTransform =
            replayCollisionUsesInitialTransform;
    if (hmsItem != nullptr) {
        clone->HmsItem()->SetProperties(hmsItem->GetProperties());
    }
    CPlugSolid *solid = ExistingSolid();
    if (solid != nullptr) {
        clone->SetSolid(solid->CreateModelInstance());
    }
    clone->CopyLinksFromMobil(this);
    return clone;
}

void CSceneMobil::SetModel(CSceneMobil *newModel) {
    model.MwSetNod(newModel);
}

CSceneMobil *CSceneMobil::GetModel(void) {
    return model.Get();
}

void CSceneMobil::SetMessageHandler(CSceneMessageHandler *handler) {
    messageHandler.MwSetNod(handler);
}

void CSceneMobil::DisconnectFromModel(void) {
    SetModel(nullptr);
}

void CSceneMobil::Show(void) {
    CPlugTree *tree = GetTree();
    if (tree != nullptr) {
        tree->SetIsVisible(1);
    }
}

void CSceneMobil::Hide(void) {
    CPlugTree *tree = GetTree();
    if (tree != nullptr) {
        tree->SetIsVisible(0);
    }
}

void CSceneMobil::SetIsVisible(int isVisible) {
    if (isVisible != 0) {
        Show();
    } else {
        Hide();
    }
}

int CSceneMobil::GetIsVisible(void) const {
    const CPlugTree *tree = GetTree();
    return tree != nullptr ? tree->IsVisible() : 0;
}

void CSceneMobil::EnableAbsorbContactCallback(int enabled) {
    absorbContactEnabled = enabled != 0;
    CHmsItem *item = HmsItem();
    if (item != nullptr) {
        RefreshPhysicsCallbacks(*item);
    }
    for (CHmsItem *triggerItem : triggerPhysicsItems) {
        if (triggerItem != nullptr) {
            RefreshPhysicsCallbacks(*triggerItem);
        }
    }
}

void CSceneMobil::EnablePhysicsUpdates(int enabled) {
    physicsUpdatesEnabled = enabled != 0;
    CHmsItem *item = HmsItem();
    if (item != nullptr) {
        RefreshPhysicsCallbacks(*item);
    }
    for (CHmsItem *triggerItem : triggerPhysicsItems) {
        if (triggerItem != nullptr) {
            RefreshPhysicsCallbacks(*triggerItem);
        }
    }
}

int CSceneMobil::IsAbsorbContactEnabled(void) const {
    return absorbContactEnabled ? 1 : 0;
}

int CSceneMobil::ArePhysicsUpdatesEnabled(void) const {
    return physicsUpdatesEnabled ? 1 : 0;
}

void CSceneMobil::AbsorbContact(CHmsPhysicalContact &contact) {
    (void)contact;
}

void CSceneMobil::HmsComputeForces(float dt) {
    (void)dt;
}

void CSceneMobil::HmsAfterContacts(void) {
}

void CSceneMobil::VehicleReset(void) {
}

void CSceneMobil::VehicleBlockSpeedSet(int isBlocked) {
    (void)isBlocked;
}

void CSceneMobil::VehicleBlockSpeed2Set(int isBlocked) {
    (void)isBlocked;
}

void CSceneMobil::SetSolid(CPlugSolid *solid) {
    CMwNodRef<CPlugSolid> defaultSolid;
    CPlugSolid *resolvedSolid = solid;
    if (resolvedSolid == nullptr) {
        defaultSolid = MakeMwNod<CPlugSolid>();
        auto tree = std::make_unique<CPlugTree>();
        tree->SetIsVisible(0);
        defaultSolid->SetOwnedTree(std::move(tree), 1u);
        resolvedSolid = defaultSolid.Get();
    }
    CHmsItem *item = HmsItem();
    if (item != nullptr) {
        item->SetSolid(resolvedSolid);
    }
}

CPlugTree *CSceneMobil::GetTree(void) const {
    CHmsItem *item = HmsItem();
    return item != nullptr && item->Solid() != nullptr
            ? item->Solid()->CollisionTree()
            : nullptr;
}

void CSceneMobil::SetTree(CPlugTree *tree) {
    CMwNodRef<CPlugSolid> createdSolid;
    CPlugSolid *solid = ExistingSolid();
    if (solid == nullptr) {
        createdSolid = MakeMwNod<CPlugSolid>();
        solid = createdSolid.Get();
        HmsItem()->SetSolid(solid);
    }
    solid->SetTree(tree, 1u);
}

CSceneObjectLink::CSceneObjectLink(void) {
    relativeLocation.SetIdentity();
}

CSceneObjectLink::~CSceneObjectLink(void) {
    OnLeaveScene();
    SetMobil(nullptr);
    SetObject(nullptr);
}

void CSceneObjectLink::SetObject(CSceneObject *newObject) {
    object.MwSetNod(newObject);
}

void CSceneObjectLink::SetMobil(CSceneMobil *newMobil) {
    mobilTree = nullptr;
    mobil = newMobil;
}

void CSceneObjectLink::SetMobilTree(CPlugTree *tree) {
    mobilTree = tree;
}

void CSceneObjectLink::SetMobilTreeId(CMwId treeId) {
    mobilTreeId = treeId;
}

void CSceneObjectLink::SetIsActive(int isActive) {
    active = isActive != 0;
}

void CSceneObjectLink::OnEnterScene(CScene *newScene) {
    scene = newScene;
}

void CSceneObjectLink::OnLeaveScene(void) {
    scene = nullptr;
}

void CSceneObjectLink::SetRelativeLocation(const GmIso4 &iso4) {
    relativeLocation = iso4;
}

CSceneObject *CSceneObjectLink::Object(void) const {
    return object.Get();
}

CSceneMobil *CSceneObjectLink::Mobil(void) const {
    return mobil;
}

CPlugTree *CSceneObjectLink::MobilTree(void) const {
    return mobilTree;
}

const CMwId &CSceneObjectLink::TreeId(void) const {
    return mobilTreeId;
}

const GmIso4 &CSceneObjectLink::RelativeLocation(void) const {
    return relativeLocation;
}

bool CSceneObjectLink::IsActive(void) const {
    return active;
}

unsigned long CSceneMobil::LinkFind(CSceneObjectLink *link) const {
    for (std::size_t index = 0u; index < links.size(); ++index) {
        if (links[index].Get() == link) {
            return static_cast<unsigned long>(index);
        }
    }
    return InvalidEngineIndex;
}

unsigned long CSceneMobil::LinkFindFromObjectId(const CMwId &objectId) const {
    for (std::size_t index = 0u; index < links.size(); ++index) {
        CSceneObjectLink *link = links[index].Get();
        if (link != nullptr && link->object.Get() != nullptr &&
            link->object->id == objectId) {
            return static_cast<unsigned long>(index);
        }
    }
    return InvalidEngineIndex;
}

unsigned long CSceneMobil::LinkAdd(CSceneObjectLink *link) {
    links.emplace_back(link);
    return static_cast<unsigned long>(links.size() - 1u);
}

void CSceneMobil::LinkRemove(unsigned long index) {
    if (index >= links.size()) {
        return;
    }
    CSceneObjectLink *link = links[index].Get();
    if (link != nullptr) {
        link->OnLeaveScene();
        CSceneObject *target = link->object.Get();
        if (target != nullptr && target->scene != nullptr) {
            CMwNodRef<CSceneObject> targetOwner(target);
            link->SetObject(nullptr);
            target->RemoveFromScene();
        }
    }
    CMwNodRef<CSceneObjectLink> removed(std::move(links[index]));
    links.erase(links.begin() + index);
    removed.MwSetNod(nullptr);
}

void CSceneMobil::AddObject(CSceneObject *newObject, CSceneObjectLink *&link) {
    if (newObject == nullptr || newObject == this) {
        return;
    }
    if (link == nullptr) {
        link = new CSceneObjectLink();
        link->SetIsActive(1);
    }
    link->SetObject(newObject);
    link->SetMobil(this);
    (void)LinkAdd(link);
    newObject->ownedBySceneMobilLink = true;

    if (scene != nullptr) {
        if (newObject->scene != nullptr && newObject->scene != scene) {
            newObject->RemoveFromScene();
        }
        if (newObject->scene == nullptr) {
            newObject->scene = scene;
            newObject->OnEnterScene();
        }
        link->OnEnterScene(scene);
    }
}

void CSceneMobil::CopyLinksFromMobil(CSceneMobil *sourceMobil) {
    if (sourceMobil == nullptr) {
        return;
    }
    for (const CMwNodRef<CSceneObjectLink> &sourceLinkRef :
         sourceMobil->links) {
        CSceneObjectLink *sourceLink = sourceLinkRef.Get();
        CSceneMobil *sourceChild = sourceLink != nullptr
                ? dynamic_cast<CSceneMobil *>(sourceLink->object.Get())
                : nullptr;
        if (sourceChild == nullptr) {
            continue;
        }
        CSceneMobil *child = sourceChild->CreateModelInstance();
        CSceneObjectLink *link = nullptr;
        AddObject(child, link);
        if (link != nullptr) {
            link->SetRelativeLocation(sourceLink->RelativeLocation());
            link->SetMobilTreeId(sourceLink->TreeId());
            link->SetIsActive(sourceLink->IsActive());
        }
    }
}

void CSceneMobil::OnEnterScene(void) {
    CSceneObject::OnEnterScene();
    for (CMwNodRef<CSceneObjectLink> &link : links) {
        if (link.Get() != nullptr) {
            link->OnEnterScene(scene);
        }
    }
}

void CSceneMobil::OnLeaveScene(void) {
    for (CMwNodRef<CSceneObjectLink> &link : links) {
        if (link.Get() != nullptr) {
            link->OnLeaveScene();
        }
    }
    CSceneObject::OnLeaveScene();
}

StaticSolidAssetReference &CSceneMobil::StaticSolidAsset(void) {
    return staticSolidAsset;
}

const StaticSolidAssetReference &CSceneMobil::StaticSolidAsset(void) const {
    return staticSolidAsset;
}

void CSceneMobil::SetStaticSolidAsset(StaticSolidAssetReference reference) {
    staticSolidAsset = std::move(reference);
}

void CSceneMobil::SetInitialItemProperties(
        const CHmsItem::Properties &properties) {
    initialItemProperties = properties;
    if (HmsItem() != nullptr) {
        HmsItem()->SetProperties(properties);
    }
}

const std::optional<CHmsItem::Properties> &
CSceneMobil::InitialItemProperties(void) const {
    return initialItemProperties;
}

const std::vector<CMwNodRef<CSceneObjectLink>> &CSceneMobil::Links(void) const {
    return links;
}
CSceneMobil::RuntimeClone CSceneMobil::CaptureRuntimeClone(void) const noexcept {
    return {absorbContactEnabled, physicsUpdatesEnabled};
}

void CSceneMobil::RestoreRuntimeClone(
        const RuntimeClone &clone) noexcept {
    EnableAbsorbContactCallback(clone.absorbContactEnabled ? 1 : 0);
    EnablePhysicsUpdates(clone.physicsUpdatesEnabled ? 1 : 0);
}
