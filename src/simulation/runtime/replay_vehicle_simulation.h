#ifndef TMNF_REPLAY_VEHICLE_SIMULATION_H
#define TMNF_REPLAY_VEHICLE_SIMULATION_H

#include <cstdint>
#include <optional>

#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_simulation_definition.h"
#include "simulation/runtime/replay_vehicle_body.h"
#include "simulation/runtime/replay_vehicle_collision_model.h"
#include "simulation/runtime/replay_vehicle_materials.h"
#include "simulation/runtime/replay_vehicle_tuning_assembly.h"
#include "simulation/runtime/replay_vehicle_turbo_sound.h"
#include "simulation/runtime/replay_vehicle_wheel_surfaces.h"
class CTrackManiaRace;

enum class ReplayVehiclePreparationResult {
    Ready,
    CollisionModelConstructionFailed,
    CollisionTreeUnavailable,
};

class ReplayVehicleSimulation {
public:
    struct RuntimeClone {
        CSceneVehicleCar::RuntimeClone car;
        ReplayVehicleWheelSurfaces::RuntimeClone wheelSurfaces;
    };
    explicit ReplayVehicleSimulation(CTrackManiaRace &race);
    ~ReplayVehicleSimulation();

    ReplayVehicleSimulation(const ReplayVehicleSimulation &) = delete;
    ReplayVehicleSimulation &operator=(const ReplayVehicleSimulation &) = delete;

    ReplayVehiclePreparationResult Start(
            const ReplaySimulationDefinition &definition,
            const ReplayControlTick &tick,
            ReplayVehicleBody &body,
            bool staticSceneReady);
    void PrepareStep(const ReplayControlTick &tick,
                     ReplayVehicleBody &body);
    bool Respawn(ReplayVehicleBody &body);

    CSceneVehicleCar &Car() { return car_; }
    std::optional<ReplayDynaParameters> BuildDynaParameters() const;
    std::optional<u32> FinishTimeMs() const;
    RuntimeClone CaptureRuntimeClone() const;
    bool CanRestoreRuntimeClone(const RuntimeClone &clone) const noexcept;
    void RestoreRuntimeClone(const RuntimeClone &clone) noexcept;

private:
    void InstallActiveTuning();
    void ApplyControls(const ReplayVehicleControlState &controls);
    void ApplyTransitionActions(const ReplayControlTick &tick,
                                ReplayVehicleBody &body);
    void ApplyWheelMaterialFromTuning();

    CTrackManiaRace &race_;
    CSceneVehicleCar car_;
    CSceneVehicleCarTuning tuning_;
    CSceneVehicleTunings tuningContainer_;
    CSceneVehicleStruct vehicleStruct_;
    ReplayVehicleTuningInstallation tuningInstallation_;
    ReplayVehicleWheelSurfaces wheelSurfaces_;
    ReplayVehicleMaterials materials_;
    ReplayVehicleTurboSound turboSound_;
    ReplayVehicleCollisionModel collisionModel_;
};

#endif
