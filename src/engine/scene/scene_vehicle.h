#pragma once

#include "engine/core/engine_types.h"
#include <array>
#include <optional>
#include <vector>

#include "engine/core/gm_types.h"
#include "engine/core/mw_id.h"
#include "engine/scene/scene_mobil.h"
#include "engine/game/vehicle_wheel_surface_id.h"
struct CPlugSolid;
struct CPlugTree;
struct CSceneVehicleMaterialContainer;
class CSceneVehicleTuning;
struct CSceneVehicleTunings;

using CSceneVehicleMaterialRemapIndex = u32;

class SSurfaceId {
  public:
    SSurfaceId(void);

    SSurfaceId &operator=(VehicleWheelSurfaceId surfaceId);
    void Reset(void);
    const CMwId &Id(void) const;

  private:
    CMwId id_;
};

class CSceneVehicle : public CSceneMobil {
  public:
    struct RuntimeClone;
    enum EVehicleEvent {
        EVehicleEvent_None = 0,
        EVehicleEvent_WaterSplash = 1,
    };

    struct SVehicleState {
        float forwardSpeed = 0.0f;
        float sideSpeed = 0.0f;
        float steeringControl = 0.0f;
        float lowSpeedGateA = 0.0f;
        float lowSpeedGateB = 0.0f;
        u32 turboActive = 0u;
        float turboProgressRatio = 0.0f;
        u32 wheelSpeedOverrideActive = 0u;
        float surfaceFeedbackAccumulator = 0.0f;
        float feedbackSideSpringValue = 0.0f;
        float feedbackForwardSpringValue = 0.0f;
        float feedbackRamp1 = 0.0f;
        float feedbackRamp0 = 0.0f;
        GmIso4 corpusIso{};
        u32 vehicleEvent0Value = 1u;
        u32 waterSplashEventCounter = 1u;
        GmVec3 localLinearSpeed{};
        float materialFeedbackSpeed = 0.0f;
        float materialFeedbackIntensity = 0.0f;

        void VehicleStateReset(void);
        void VehicleStateSet(const SVehicleState &rhs);
        void VehicleStateSetBlend(const SVehicleState &from, const SVehicleState &to, float blend);
    };

    struct SStateSampleWindow {
        u32 previousTick = 0u;
        u32 currentTick = 0u;
    };

    class SSurfaceHandler {
      public:
        SSurfaceHandler(void);

        void Init(const CPlugSolid &solid, const SSurfaceId &surfaceId);
        void Reset(void);
        void UpdateSurface(void);
        GmMat3 RestRotation(void) const { return restIso.RotationMatrix(); }
        GmVec3 RestPoint(void) const { return restIso.Translation(); }
        GmMat3 CurrentRotation(void) const { return currentIso.RotationMatrix(); }
        GmVec3 CurrentPoint(void) const { return currentIso.Translation(); }
        CPlugTree *Tree(void) const { return tree; }
        void BindTree(CPlugTree *newTree) { tree = newTree; }
        const GmIso4 &RestPose(void) const { return restIso; }
        const GmIso4 &CurrentPose(void) const { return currentIso; }
        void SetRestPose(const GmMat3 &rotation, const GmVec3 &point) {
            restIso.Set(rotation, point);
        }
        void SetCurrentPose(const GmMat3 &rotation, const GmVec3 &point) {
            currentIso.Set(rotation, point);
        }
        void SetRestPose(const GmIso4 &pose) { restIso = pose; }
        void SetCurrentPose(const GmIso4 &pose) { currentIso = pose; }
        void ResetCurrentPose(void) { currentIso = restIso; }
        void OffsetCurrentY(float offset) {
            currentIso.translation.y += offset;
        }

      private:
        CPlugTree *tree = nullptr;
        GmIso4 restIso{};
        GmIso4 currentIso{};
    };

    struct SEventSlot {
        bool initialized = false;
        u32 value = 1u;
    };

    struct SWaterState {
        GmBoxAligned boxLocal{{0.0f, 0.0f, 0.0f}, {-1.0f, -1.0f, -1.0f}};
        bool splashPending = false;
        GmVec3 splashLocalSpeed{};
    };

    CSceneVehicle(void);
    ~CSceneVehicle(void) override;
    void TuningsSet(CSceneVehicleTunings *tunings);
    CSceneVehicleTunings *Tunings(void) const { return tuningContainer; }
    void BindMaterialContainer(CSceneVehicleMaterialContainer &container);
    CSceneVehicleMaterialContainer *MaterialContainer(void) const {
        return materialContainer;
    }
    void ClearMaterialRemap(void);
    void SetMaterialRemap(
            const std::vector<CSceneVehicleMaterialRemapIndex> &remap) {
        materialRemap = remap;
    }
    CSceneVehicleMaterialRemapIndex &MaterialRemapAt(u32 materialId);
    const CSceneVehicleMaterialRemapIndex &MaterialRemapAt(u32 materialId) const;
    void SetWaterBoxLocal(const GmBoxAligned &box);
    const GmBoxAligned &WaterBoxLocal(void) const;
    SWaterState &WaterState(void) { return water; }
    const SWaterState &WaterState(void) const { return water; }
    u32 VehicleEventValue(EVehicleEvent event) const;
    bool WaterSplashPending(void) const;
    const GmVec3 &WaterSplashLocalSpeed(void) const;
    int UpdateEvent(EVehicleEvent event, unsigned long value);
    void WaterSplash(const GmVec3 &localSpeed);
    void BuildVehicleMaterialsRemap(void);
    void SetIsUpdateAsync(int isUpdateAsync);
    bool IsUpdateAsync(void) const { return updateAsync; }
    bool IsNetworked(void) const { return networked; }
    float AsyncPeriodSeconds(void) const { return asyncPeriodSeconds; }
    virtual void VehicleIsNetworkedSet(int isNetworked);
    float VehicleStateComputeBlendVal(void);
    void SetSolid(CPlugSolid *solid) override;
    void OnEnterScene(void) override;
    void VehicleReset(void) override;
    virtual void VehicleInitFromSolid(void);
    virtual void VehicleUpdateAsync(void);
    virtual const SVehicleState &VehicleStateAsyncGet(void) const = 0;
    virtual const SVehicleState &VehicleStatePrevAsyncGet(void) const = 0;
    virtual void UpdateParamsFromTuning(void);
    virtual const CSceneVehicleTuning &GetVehicleTuning(void);
    virtual unsigned long WheelGetCount(void) const;
    virtual int WheelIsSliding(unsigned long wheelIndex) const;
    virtual unsigned short WheelGetContactMaterial(unsigned long wheelIndex) const;
    virtual const GmVec3 &WheelGetAsyncGroundContactPos(unsigned long wheelIndex) const;

    struct RuntimeClone {
        CSceneMobil::RuntimeClone mobil{};
        std::array<SEventSlot, 2> vehicleEvents{};
        SWaterState water{};
        bool updateAsync = true;
        bool networked = false;
        u32 predictionDelayTicks = 0u;
        std::optional<SStateSampleWindow> stateSampleWindow;
        float asyncPeriodSeconds = 0.0f;
    };
    RuntimeClone CaptureRuntimeClone(void) const noexcept;
    void RestoreRuntimeClone(const RuntimeClone &clone) noexcept;

private:
    CSceneVehicleTunings *tuningContainer = nullptr;
    CSceneVehicleMaterialContainer *materialContainer = nullptr;
    std::vector<CSceneVehicleMaterialRemapIndex> materialRemap;
    std::array<SEventSlot, 2> vehicleEvents{};
    SWaterState water;
    bool updateAsync = true;
    bool networked = false;
    u32 predictionDelayTicks = 0u;
    std::optional<SStateSampleWindow> stateSampleWindow;
    float asyncPeriodSeconds = 0.0f;
};
