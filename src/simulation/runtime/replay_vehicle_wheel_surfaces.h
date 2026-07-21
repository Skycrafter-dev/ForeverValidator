#ifndef TMNF_REPLAY_VEHICLE_WHEEL_SURFACES_H
#define TMNF_REPLAY_VEHICLE_WHEEL_SURFACES_H

#include "engine/core/engine_types.h"
#include <memory>
#include <vector>

#include "engine/game/replay_vehicle_wheel_definition.h"
#include "engine/scene/scene_vehicle_car.h"
class ReplayVehicleCollisionModel;

class ReplayVehicleWheelSurfaces final
        : public CSceneVehicleCarWheelSurfaceObserver {
public:
    struct RuntimeClone {
        std::vector<bool> movedByUpdateSurface;
    };
    void Reset(CSceneVehicleCar *car = nullptr);
    void MarkWheelSurfaceUpdated(CSceneVehicleCar *car,
                                 CSceneVehicleCar::SSimulationWheel *wheel);
    void BeginWheelFromReplayState(u32 wheelIndex,
                                   CSceneVehicleCar::SSimulationWheel *wheel);
    void UpdateLocalIsoFromWheel(u32 wheelIndex,
                                 const CSceneVehicleCar::SSimulationWheel *wheel);
    void BindCollisionShapes(
            const ReplayVehicleCollisionModel &collisionModel,
            const VehicleWheelSetDefinition &wheelDefinitions,
            CSceneVehicleCar &car);
    void OnWheelSurfaceUpdated(
            CSceneVehicleCar &car,
            CSceneVehicleCar::SSimulationWheel &wheel) override;
    RuntimeClone CaptureRuntimeClone() const;
    bool CanRestoreRuntimeClone(const RuntimeClone &clone,
                                const CSceneVehicleCar &car) const noexcept;
    void RestoreRuntimeClone(const RuntimeClone &clone,
                             CSceneVehicleCar &car) noexcept;

private:
    struct WheelSurfaceBinding {
        std::unique_ptr<CPlugTree> ownedLocalTree;
        bool movedByUpdateSurface = false;
    };

    WheelSurfaceBinding *EnsureBinding(u32 wheelIndex);
    WheelSurfaceBinding *BindingAt(u32 wheelIndex);
    const WheelSurfaceBinding *BindingAt(u32 wheelIndex) const;
    CPlugTree *LocalTree(u32 wheelIndex);
    void UseLocalHandlerTree(u32 wheelIndex,
                             CSceneVehicleCar::SSimulationWheel *wheel);
    void BindWheelToSourceTree(u32 wheelIndex,
                               CPlugTree *tree,
                               CSceneVehicleCar::SSimulationWheel *wheel);
    bool ShouldPreserveBoundWheelRestBox(u32 wheelIndex) const;

    static void SetBoundSourceTreeLocation(CPlugTree *tree,
                                           const GmIso4 *iso);
    static void PreserveBoundWheelSurfaceRestBox(
            CPlugTree *tree,
            const CSceneVehicleCar::SSimulationWheel *wheel);
    static void SanitizeWheelBorrowedPointers(CSceneVehicleCar *car);

    std::vector<WheelSurfaceBinding> wheelBindings;
};

#endif // TMNF_REPLAY_VEHICLE_WHEEL_SURFACES_H
