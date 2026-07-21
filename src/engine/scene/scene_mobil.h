#pragma once

#include <optional>
#include <memory>
#include <vector>

#include "engine/core/gm_types.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/scene/scene_object.h"
#include "engine/resources/static_solid_asset.h"
struct CGameCtnBlock;
struct CHmsPhysicalContact;
struct CPlugMaterial;
struct CPlugShader;
struct CPlugSolid;
struct CPlugTree;
struct CPlugVisual;

class CSceneMobil;
class CSceneMobilCallbackBinding;

class CSceneObjectLink : public CMwNod {
public:
    CSceneObjectLink(void);
    ~CSceneObjectLink(void) override;

    void SetObject(CSceneObject *object);
    void SetMobil(CSceneMobil *mobil);
    void SetMobilTree(CPlugTree *tree);
    void SetMobilTreeId(CMwId treeId);
    void SetIsActive(int isActive);
    void OnEnterScene(CScene *scene);
    void OnLeaveScene(void);
    void SetRelativeLocation(const GmIso4 &iso4);
    CSceneObject *Object(void) const;
    CSceneMobil *Mobil(void) const;
    CPlugTree *MobilTree(void) const;
    const CMwId &TreeId(void) const;
    const GmIso4 &RelativeLocation(void) const;
    bool IsActive(void) const;

private:
    friend class CSceneMobil;

    CMwNodRef<CSceneObject> object;
    CSceneMobil *mobil = nullptr;
    CPlugTree *mobilTree = nullptr;
    CMwId mobilTreeId;
    GmIso4 relativeLocation{};
    CScene *scene = nullptr;
    bool active = false;
};

class CSceneMobil : public CSceneObject {
public:
    struct RuntimeClone {
        bool absorbContactEnabled = false;
        bool physicsUpdatesEnabled = false;
    };
    CSceneMobil(void);
    ~CSceneMobil(void) override;

    CHmsItem *HmsItem(void) const;
    void AttachPhysicsItem(CHmsItem &item);
    void DetachPhysicsItem(void);
    void AttachTriggerPhysicsItem(CHmsItem &item);
    void DetachTriggerPhysicsItem(CHmsItem &item);
    void OnPhysicsItemDestroyed(CHmsItem &item);
    void SetOwningBlock(CGameCtnBlock *block);
    CGameCtnBlock *OwningBlock(void) const;
    CPlugSolid *ExistingSolid(void) const;
    CPlugTree *GetTree(void) const;
    void AddTree(CPlugTree *tree);
    CPlugTree *FindOrAddNewTreeWithId(const CMwId &id);
    CPlugTree *SetVisual(CPlugVisual *visual,
                        CPlugShader *shader,
                        CPlugMaterial *material);
    CPlugTree *AddVisual(CPlugVisual *visual,
                        CPlugShader *shader,
                        CPlugMaterial *material);
    void SetTree(CPlugTree *tree);
    virtual CSceneMobil *CreateModelInstance(void);
    virtual void DisconnectFromModel(void);
    virtual void SetSolid(CPlugSolid *solid);
    virtual int DoesMotionChangeLocations(void);
    CSceneMobil *GetModel(void);
    void SetMessageHandler(CSceneMessageHandler *handler);
    unsigned long LinkFind(CSceneObjectLink *link) const;
    unsigned long LinkFindFromObjectId(const CMwId &objectId) const;
    unsigned long LinkAdd(CSceneObjectLink *link);
    void LinkRemove(unsigned long index);
    void AddObject(CSceneObject *object, CSceneObjectLink *&link);
    void CopyLinksFromMobil(CSceneMobil *sourceMobil);
    void OnEnterScene(void) override;
    void OnLeaveScene(void) override;
    int IsZombie(void) const;
    virtual void Show(void);
    virtual void Hide(void);
    void SetIsVisible(int isVisible);
    virtual int GetIsVisible(void) const;
    void EnableAbsorbContactCallback(int enabled);
    void EnablePhysicsUpdates(int enabled);
    int IsAbsorbContactEnabled(void) const;
    int ArePhysicsUpdatesEnabled(void) const;
    virtual void AbsorbContact(CHmsPhysicalContact &contact);
    virtual void VehicleBlockSpeedSet(int isBlocked);
    virtual void VehicleBlockSpeed2Set(int isBlocked);
    virtual void VehicleReset(void);
    StaticSolidAssetReference &StaticSolidAsset(void);
    const StaticSolidAssetReference &StaticSolidAsset(void) const;
    void SetStaticSolidAsset(StaticSolidAssetReference reference);
    void SetInitialItemProperties(const CHmsItem::Properties &properties);
    const std::optional<CHmsItem::Properties> &InitialItemProperties(void) const;
    const std::vector<CMwNodRef<CSceneObjectLink>> &Links(void) const;
    void SetReplayMotionChangesLocations(int changesLocations);
    void SetReplayCollisionUsesInitialTransform(int usesInitialTransform);
    int ReplayCollisionUsesInitialTransform(void) const;
    RuntimeClone CaptureRuntimeClone(void) const noexcept;
    void RestoreRuntimeClone(const RuntimeClone &clone) noexcept;

protected:
    virtual void HmsComputeForces(float dt);
    virtual void HmsAfterContacts(void);

private:
    friend class CSceneMobilCallbackBinding;

    void RefreshPhysicsCallbacks(CHmsItem &item);
    void ClearPhysicsCallbacks(CHmsItem &item);

    std::unique_ptr<CHmsItem> hmsItem;
    CHmsItem *attachedPhysicsItem = nullptr;
    std::vector<CHmsItem *> triggerPhysicsItems;
    CGameCtnBlock *owningBlock = nullptr;
    CMwNodRef<CSceneMobil> model;
    CMwNodRef<CMotion> solidMotion;
    CMwNodRef<CMotion> visibilityMotion;
    std::vector<CMwNodRef<CSceneObjectLink>> links;
    CMwNodRef<CSceneMessageHandler> messageHandler;
    StaticSolidAssetReference staticSolidAsset;
    std::optional<CHmsItem::Properties> initialItemProperties;
    bool replayMotionChangesLocations = false;
    bool replayCollisionUsesInitialTransform = false;

    bool absorbContactEnabled = false;
    bool physicsUpdatesEnabled = false;
    std::unique_ptr<CSceneMobilCallbackBinding> callbackBinding;

    void SetModel(CSceneMobil *model);
};
