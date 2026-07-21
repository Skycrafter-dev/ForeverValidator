// Shared scene-vehicle state and behavior.

#include "engine/scene/scene_vehicle.h"
#include "engine/core/binary32_math.h"
#include "engine/core/mw_cmd_buffer_core.h"
#include "engine/scene/plug_solid.h"
#include "engine/scene/scene_vehicle_car_tuning.h"
#include "engine/scene/scene_vehicle_material.h"
namespace {

float BlendVehicleStateValue(float from, float to, float blend) {
    return from * (1.0f - blend) + to * blend;
}

} // namespace

SSurfaceId::SSurfaceId(void) { Reset(); }

SSurfaceId &SSurfaceId::operator=(VehicleWheelSurfaceId surfaceId) {
    if (surfaceId == VehicleWheelSurfaceId::Invalid) {
        id_.SetInvalid();
    } else {
        id_.SetNumber(static_cast<u32>(surfaceId));
    }
    return *this;
}

void SSurfaceId::Reset(void) { id_.SetInvalid(); }

const CMwId &SSurfaceId::Id(void) const { return id_; }

void CSceneVehicle::SVehicleState::VehicleStateReset(void) {
    forwardSpeed = 0.0f;
    sideSpeed = 0.0f;
    steeringControl = 0.0f;
    lowSpeedGateA = 0.0f;
    lowSpeedGateB = 0.0f;
    turboActive = 0u;
    turboProgressRatio = 0.0f;
    wheelSpeedOverrideActive = 0u;
    surfaceFeedbackAccumulator = 0.0f;
    feedbackSideSpringValue = 0.0f;
    feedbackForwardSpringValue = 0.0f;
    feedbackRamp1 = 0.0f;
    feedbackRamp0 = 0.0f;
    corpusIso.SetIdentity();
    vehicleEvent0Value = 1u;
    waterSplashEventCounter = 1u;
    localLinearSpeed = {};
    materialFeedbackSpeed = 0.0f;
    materialFeedbackIntensity = 0.0f;
}

void CSceneVehicle::SVehicleState::VehicleStateSet(const SVehicleState &rhs) { *this = rhs; }

void CSceneVehicle::SVehicleState::VehicleStateSetBlend(const SVehicleState &from,
                                                        const SVehicleState &to, float blend) {
    forwardSpeed = BlendVehicleStateValue(from.forwardSpeed, to.forwardSpeed, blend);
    sideSpeed = BlendVehicleStateValue(from.sideSpeed, to.sideSpeed, blend);
    steeringControl = BlendVehicleStateValue(from.steeringControl, to.steeringControl, blend);
    lowSpeedGateA = BlendVehicleStateValue(from.lowSpeedGateA, to.lowSpeedGateA, blend);
    lowSpeedGateB = BlendVehicleStateValue(from.lowSpeedGateB, to.lowSpeedGateB, blend);
    turboActive = to.turboActive;
    turboProgressRatio =
            BlendVehicleStateValue(from.turboProgressRatio, to.turboProgressRatio, blend);
    wheelSpeedOverrideActive = to.wheelSpeedOverrideActive;
    surfaceFeedbackAccumulator = BlendVehicleStateValue(from.surfaceFeedbackAccumulator,
                                                        to.surfaceFeedbackAccumulator, blend);
    feedbackSideSpringValue =
            BlendVehicleStateValue(from.feedbackSideSpringValue, to.feedbackSideSpringValue, blend);
    feedbackForwardSpringValue = BlendVehicleStateValue(from.feedbackForwardSpringValue,
                                                        to.feedbackForwardSpringValue, blend);
    feedbackRamp1 = BlendVehicleStateValue(from.feedbackRamp1, to.feedbackRamp1, blend);
    feedbackRamp0 = BlendVehicleStateValue(from.feedbackRamp0, to.feedbackRamp0, blend);
    corpusIso.SetBlend(from.corpusIso, to.corpusIso, blend);
    vehicleEvent0Value = to.vehicleEvent0Value;
    waterSplashEventCounter = to.waterSplashEventCounter;
    localLinearSpeed.x =
            BlendVehicleStateValue(from.localLinearSpeed.x, to.localLinearSpeed.x, blend);
    localLinearSpeed.y =
            BlendVehicleStateValue(from.localLinearSpeed.y, to.localLinearSpeed.y, blend);
    localLinearSpeed.z =
            BlendVehicleStateValue(from.localLinearSpeed.z, to.localLinearSpeed.z, blend);
    materialFeedbackSpeed =
            BlendVehicleStateValue(from.materialFeedbackSpeed, to.materialFeedbackSpeed, blend);
    materialFeedbackIntensity = BlendVehicleStateValue(from.materialFeedbackIntensity,
                                                       to.materialFeedbackIntensity, blend);
}

CSceneVehicle::CSceneVehicle(void) {
    vehicleEvents[EVehicleEvent_None] = {false, 1u};
    vehicleEvents[EVehicleEvent_WaterSplash] = {false, 1u};
}

CSceneVehicle::~CSceneVehicle(void) = default;

CSceneVehicle::RuntimeClone
CSceneVehicle::CaptureRuntimeClone(void) const noexcept {
    RuntimeClone clone;
    clone.mobil = CSceneMobil::CaptureRuntimeClone();
    clone.vehicleEvents = vehicleEvents;
    clone.water = water;
    clone.updateAsync = updateAsync;
    clone.networked = networked;
    clone.predictionDelayTicks = predictionDelayTicks;
    clone.stateSampleWindow = stateSampleWindow;
    clone.asyncPeriodSeconds = asyncPeriodSeconds;
    return clone;
}

void CSceneVehicle::RestoreRuntimeClone(
        const RuntimeClone &clone) noexcept {
    CSceneMobil::RestoreRuntimeClone(clone.mobil);
    vehicleEvents = clone.vehicleEvents;
    water = clone.water;
    updateAsync = clone.updateAsync;
    networked = clone.networked;
    predictionDelayTicks = clone.predictionDelayTicks;
    stateSampleWindow = clone.stateSampleWindow;
    asyncPeriodSeconds = clone.asyncPeriodSeconds;
}

void CSceneVehicle::TuningsSet(CSceneVehicleTunings *tunings) { tuningContainer = tunings; }

void CSceneVehicle::BindMaterialContainer(CSceneVehicleMaterialContainer &container) {
    materialContainer = &container;
}

void CSceneVehicle::ClearMaterialRemap(void) { materialRemap.clear(); }

CSceneVehicleMaterialRemapIndex &CSceneVehicle::MaterialRemapAt(u32 materialId) {
    return materialRemap[materialId];
}

const CSceneVehicleMaterialRemapIndex &CSceneVehicle::MaterialRemapAt(u32 materialId) const {
    return materialRemap[materialId];
}

void CSceneVehicle::SetWaterBoxLocal(const GmBoxAligned &box) { water.boxLocal = box; }

const GmBoxAligned &CSceneVehicle::WaterBoxLocal(void) const { return water.boxLocal; }

u32 CSceneVehicle::VehicleEventValue(EVehicleEvent event) const {
    return vehicleEvents[static_cast<std::size_t>(event)].value;
}

bool CSceneVehicle::WaterSplashPending(void) const { return water.splashPending; }

const GmVec3 &CSceneVehicle::WaterSplashLocalSpeed(void) const { return water.splashLocalSpeed; }

int CSceneVehicle::UpdateEvent(EVehicleEvent event, unsigned long value) {
    SEventSlot &slot = vehicleEvents[static_cast<std::size_t>(event)];
    if (!slot.initialized) {
        slot.initialized = true;
        slot.value = static_cast<u32>(value);
        return 0;
    }

    if (slot.value == value) {
        return 0;
    }

    slot.value = static_cast<u32>(value);
    return 1;
}

void CSceneVehicle::WaterSplash(const GmVec3 &localSpeed) {
    if (!IsZombie()) {
        ++vehicleEvents[EVehicleEvent_WaterSplash].value;
    } else if (!GetIsVisible()) {
        return;
    }

    water.splashLocalSpeed = localSpeed;
    water.splashPending = true;
}

void CSceneVehicle::BuildVehicleMaterialsRemap(void) {
    materialRemap.assign(EPlugSurfaceMaterialId_Count, 0u);
    if (materialContainer == nullptr) {
        return;
    }

    const u32 materialCount = materialContainer->MaterialCount();
    for (u32 materialIndex = 0; materialIndex < materialCount; ++materialIndex) {
        CSceneVehicleMaterial *material = materialContainer->MaterialAt(materialIndex);
        if (material != nullptr && material->materialNaturalId < materialRemap.size()) {
            MaterialRemapAt(material->materialNaturalId) = materialIndex;
        }
    }
}

void CSceneVehicle::SetIsUpdateAsync(int isUpdateAsync) { updateAsync = isUpdateAsync != 0; }

void CSceneVehicle::VehicleIsNetworkedSet(int isNetworked) { networked = isNetworked != 0; }

float CSceneVehicle::VehicleStateComputeBlendVal(void) {
    const CMwCmdBufferCore *commandBuffer = CMwCmdBufferCore::Current();
    if (commandBuffer == nullptr) {
        return 0.0f;
    }

    u32 tick = static_cast<u32>(commandBuffer->Timer().GetTickTime());
    if (networked) {
        tick = tick > predictionDelayTicks ? tick - predictionDelayTicks : 0ul;
    }

    float blend = 0.0f;
    if (IsZombie()) {
        if (!stateSampleWindow.has_value() ||
            stateSampleWindow->previousTick == stateSampleWindow->currentTick) {
            return 0.0f;
        }
        if (tick <= stateSampleWindow->previousTick) {
            return 0.0f;
        }
        if (tick >= stateSampleWindow->currentTick) {
            return 1.0f;
        }
        blend = ((Binary32::FromUnsignedInteger(
                         static_cast<u32>(tick - stateSampleWindow->previousTick))) /
                 (Binary32::FromUnsignedInteger(static_cast<u32>(
                         stateSampleWindow->currentTick - stateSampleWindow->previousTick))));
    } else {
        const u32 period =
                static_cast<u32>(commandBuffer->Timer().GetSchemePeriod());
        if (period == 0u || tick % period == 0u) {
            blend = 1.0f;
        } else {
            blend = ((Binary32::FromUnsignedInteger(static_cast<u32>(tick % period))) /
                     (Binary32::FromUnsignedInteger(static_cast<u32>(period))));
        }
    }

    if (blend < 0.0f) {
        return 0.0f;
    }
    if (blend > 1.0f) {
        return 1.0f;
    }
    return blend;
}

void CSceneVehicle::SetSolid(CPlugSolid *solid) {
    CSceneMobil::SetSolid(solid);
    VehicleInitFromSolid();
}

void CSceneVehicle::OnEnterScene(void) {
    CSceneMobil::OnEnterScene();
}

void CSceneVehicle::VehicleReset(void) {
    CSceneMobil::VehicleReset();
    water.splashPending = false;
}

void CSceneVehicle::VehicleInitFromSolid(void) {}

void CSceneVehicle::VehicleUpdateAsync(void) {}

void CSceneVehicle::UpdateParamsFromTuning(void) {}

const CSceneVehicleTuning &CSceneVehicle::GetVehicleTuning(void) {
    return *tuningContainer->ActiveTuning();
}

unsigned long CSceneVehicle::WheelGetCount(void) const { return 0u; }

int CSceneVehicle::WheelIsSliding(unsigned long wheelIndex) const {
    (void)wheelIndex;
    return 0;
}

unsigned short CSceneVehicle::WheelGetContactMaterial(unsigned long wheelIndex) const {
    (void)wheelIndex;
    return 0u;
}

const GmVec3 &CSceneVehicle::WheelGetAsyncGroundContactPos(unsigned long wheelIndex) const {
    (void)wheelIndex;
    static const GmVec3 noContactPosition{};
    return noContactPosition;
}
