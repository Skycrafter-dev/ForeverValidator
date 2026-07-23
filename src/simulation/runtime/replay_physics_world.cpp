#include "simulation/runtime/replay_physics_world.h"
#include "engine/physics/dynamics/hms_corpus.h"
ReplayPhysicsWorld::ReplayPhysicsWorld() {
    zone_.ResetSimulationState();
    localCollisionZone_.Reset();
    commandBuffer_.Reset();
}

ReplayMapSceneResult ReplayPhysicsWorld::ConnectMapScene(
        ReplayMapScene &mapScene,
        CSceneVehicleCar *vehicle,
        CTrackManiaRace &race) {
    const ReplayMapSceneResult result = mapScene.SelectCollisionZone(
            localCollisionZone_, vehicle, race, collisionZone_);
    if (result == ReplayMapSceneResult::Ready &&
        collisionZone_ == &localCollisionZone_) {
        localCollisionZone_.Reset();
    }
    return result;
}

void ReplayPhysicsWorld::InstallEnvironment(
        ReplayEnvironment &environment,
        const ReplayEnvironmentDefinition &definition,
        bool vehicleForceCallbacksEnabled) {
    environment.InstallOnZone(
            zone_, definition, vehicleForceCallbacksEnabled);
}

void ReplayPhysicsWorld::AddVehicleBody(CHmsCorpus &corpus) {
    zone_.SetCollisionManagerZone(*collisionZone_);
    zone_.AddCorpus(&corpus);
}

void ReplayPhysicsWorld::SetSimulationTime(const ReplayControlTick &tick) {
    commandBuffer_.SetSimulationTime(tick.periodMs, tick.timeMs);
    commandBuffer_.InstallAsCurrent();
}

void ReplayPhysicsWorld::Step() {
    zone_.PhysicsStep2();
}

void ReplayPhysicsWorld::StepOptimizedCpu() {
    zone_.PhysicsStep2OptimizedCpu();
}

ReplayPhysicsWorld::RuntimeClone
ReplayPhysicsWorld::CaptureRuntimeClone() const noexcept {
    return {
            static_cast<std::uint32_t>(
                    commandBuffer_.Timer().GetSchemePeriod()),
            static_cast<std::uint32_t>(
                    commandBuffer_.Timer().GetTickTime()),
    };
}

void ReplayPhysicsWorld::RestoreRuntimeClone(
        const RuntimeClone &clone) noexcept {
    commandBuffer_.SetSimulationTime(
            clone.schemePeriodMs, clone.tickTimeMs);
}
