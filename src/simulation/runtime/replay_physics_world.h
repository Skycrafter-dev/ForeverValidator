#ifndef TMNF_REPLAY_PHYSICS_WORLD_H
#define TMNF_REPLAY_PHYSICS_WORLD_H

#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_environment_definition.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/world/hms_zone.h"
#include "engine/core/mw_cmd_buffer_core.h"
#include "simulation/runtime/replay_environment.h"
#include "simulation/replay/replay_map_scene.h"
class CHmsCorpus;
class CSceneVehicleCar;
class CTrackManiaRace;

class ReplayPhysicsWorld {
public:
    struct RuntimeClone {
        std::uint32_t schemePeriodMs = 0u;
        std::uint32_t tickTimeMs = 0u;
    };
    ReplayPhysicsWorld();

    ReplayMapSceneResult ConnectMapScene(
            ReplayMapScene &mapScene,
            CSceneVehicleCar *vehicle,
            CTrackManiaRace &race);
    void InstallEnvironment(
            ReplayEnvironment &environment,
            const ReplayEnvironmentDefinition &definition,
            bool vehicleForceCallbacksEnabled);
    void AddVehicleBody(CHmsCorpus &corpus);
    void SetSimulationTime(const ReplayControlTick &tick);
    void Step();

    CHmsZoneDynamic &Zone() { return zone_; }
    CHmsCollisionManagerSZone &CollisionZone() { return *collisionZone_; }
    RuntimeClone CaptureRuntimeClone() const noexcept;
    void RestoreRuntimeClone(const RuntimeClone &clone) noexcept;

private:
    CHmsCollisionManagerSZone localCollisionZone_{0u, nullptr};
    CHmsZoneDynamic zone_;
    CHmsCollisionManagerSZone *collisionZone_ = &localCollisionZone_;
    CMwCmdBufferCore commandBuffer_;
};

#endif
