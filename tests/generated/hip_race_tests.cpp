// Generated from tests/cuda_race_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "engine/game/game_ctn_block.h"
#include "engine/game/game_ctn_block_info.h"
#include "engine/game/trackmania_race.h"
#include "engine/scene/scene_vehicle_car.h"
#include "simulation/backends/hip/generated/hip_race_certification.h"

#include <cstdint>
#include <iostream>

namespace {

struct ContactOutcome {
    bool forcedLowSpeedFriction = false;
    std::uint32_t freewheelClearCount = 0u;
};

ContactOutcome CpuContact(bool currentTransform) {
    CMwNodRef<CGameCtnBlockInfo> info = MakeMwNod<CGameCtnBlockInfo>();
    info->SetRaceRole(BlockRaceRole::Checkpoint);
    info->SetRespawnUsesCurrentTransform(currentTransform);
    CGameCtnBlock block;
    block.SetBlockInfo(info.Get());

    CTrackManiaRace race;
    CSceneVehicleCar vehicle;
    race.BindVehicle(&vehicle);
    race.SetCurrentTransformCheckpointFreewheelClearEnabled(true);
    vehicle.VehicleFreeWheelingSet(1);
    race.OnCheckpoint(nullptr, &block);
    return {
        vehicle.CaptureRuntimeClone().controls.forcedLowSpeedFriction,
        race.Progress().freewheelClearCount,
    };
}

ContactOutcome HipContact(bool currentTransform) {
    using namespace forevervalidator::simulation;
    HipCandidateState initial;
    initial.vehicle.controls.forcedLowSpeedFriction = true;
    HipSceneActor actor;
    actor.hasCheckpoint = true;
    actor.checkpointRole =
            static_cast<std::uint32_t>(BlockRaceRole::Checkpoint);
    actor.checkpointSlot = UINT32_MAX;
    actor.respawnUsesCurrentTransform = currentTransform;

    const HipRaceContactExecution executed =
            ExecuteHipRaceContactForCertification(initial, actor);
    if (!executed.success) {
        std::cerr << executed.diagnostic << '\n';
        return {true, UINT32_MAX};
    }
    return {
        executed.finalState.vehicle.controls.forcedLowSpeedFriction,
        executed.finalState.race.progress.freewheelClearCount,
    };
}

bool CheckEqual(
        const ContactOutcome &cpu,
        const ContactOutcome &hip,
        const char *name) {
    if (cpu.forcedLowSpeedFriction ==
                hip.forcedLowSpeedFriction &&
        cpu.freewheelClearCount == hip.freewheelClearCount) {
        return true;
    }
    std::cerr << name << " CPU/HIP mismatch: friction="
              << cpu.forcedLowSpeedFriction << "/"
              << hip.forcedLowSpeedFriction << " clears="
              << cpu.freewheelClearCount << "/"
              << hip.freewheelClearCount << '\n';
    return false;
}

}  // namespace

int main() {
    const ContactOutcome currentCpu = CpuContact(true);
    const ContactOutcome currentHip = HipContact(true);
    if (!CheckEqual(
                currentCpu, currentHip,
                "current-transform rejected checkpoint") ||
        currentCpu.forcedLowSpeedFriction ||
        currentCpu.freewheelClearCount != 1u) {
        std::cerr << "current-transform checkpoint did not clear freewheel\n";
        return 1;
    }

    const ContactOutcome ordinaryCpu = CpuContact(false);
    const ContactOutcome ordinaryHip = HipContact(false);
    if (!CheckEqual(
                ordinaryCpu, ordinaryHip,
                "ordinary rejected checkpoint") ||
        !ordinaryCpu.forcedLowSpeedFriction ||
        ordinaryCpu.freewheelClearCount != 0u) {
        std::cerr << "ordinary checkpoint changed freewheel\n";
        return 1;
    }
    return 0;
}
