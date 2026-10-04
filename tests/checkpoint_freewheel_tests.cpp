#include "engine/game/game_ctn_block.h"
#include "engine/game/game_ctn_block_info.h"
#include "engine/game/trackmania_race.h"
#include "engine/scene/scene_vehicle_car.h"
#include "simulation/runtime/replay_simulation_runtime.h"

#include <iostream>

namespace {

bool Check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

bool TestCurrentTransformCheckpointContactClearsFreewheel() {
    CTrackManiaRace race;
    CSceneVehicleCar vehicle;
    CMwNodRef<CGameCtnBlockInfo> currentTransformInfo =
            MakeMwNod<CGameCtnBlockInfo>();
    currentTransformInfo->SetRespawnUsesCurrentTransform(true);
    CGameCtnBlock currentTransformBlock;
    currentTransformBlock.SetBlockInfo(currentTransformInfo.Get());
    race.BindVehicle(&vehicle);
    race.SetCurrentTransformCheckpointFreewheelClearEnabled(true);

    vehicle.VehicleFreeWheelingSet(1);
    race.OnCheckpoint(nullptr, &currentTransformBlock);
    bool okay = Check(
            !vehicle.CaptureRuntimeClone()
                     .controls.forcedLowSpeedFriction,
            "current-transform checkpoint contact retained freewheel");
    okay &= Check(
            race.Progress().freewheelClearCount == 1u,
            "current-transform checkpoint contact was not counted");

    race.SetCurrentTransformCheckpointFreewheelClearEnabled(false);
    vehicle.VehicleFreeWheelingSet(1);
    race.OnCheckpoint(nullptr, &currentTransformBlock);
    okay &= Check(
            vehicle.CaptureRuntimeClone()
                    .controls.forcedLowSpeedFriction,
            "disabled Reference rule changed another backend");
    okay &= Check(
            race.Progress().freewheelClearCount == 1u,
            "disabled Reference rule changed the clear count");

    race.SetCurrentTransformCheckpointFreewheelClearEnabled(true);
    CMwNodRef<CGameCtnBlockInfo> ordinaryInfo =
            MakeMwNod<CGameCtnBlockInfo>();
    ordinaryInfo->SetRespawnUsesCurrentTransform(false);
    CGameCtnBlock ordinaryBlock;
    ordinaryBlock.SetBlockInfo(ordinaryInfo.Get());
    vehicle.VehicleFreeWheelingSet(1);
    race.OnCheckpoint(nullptr, &ordinaryBlock);
    okay &= Check(
            vehicle.CaptureRuntimeClone()
                    .controls.forcedLowSpeedFriction,
            "ordinary checkpoint contact cleared freewheel");
    okay &= Check(
            race.Progress().freewheelClearCount == 1u,
            "ordinary checkpoint contact changed the clear count");
    return okay;
}

bool BackendClearsCurrentTransformCheckpointFreewheel(
        forevervalidator::SimulationBackend backend) {
    CMwNodRef<CGameCtnBlockInfo> info = MakeMwNod<CGameCtnBlockInfo>();
    info->SetRespawnUsesCurrentTransform(true);
    CGameCtnBlock block;
    block.SetBlockInfo(info.Get());

    CTrackManiaRace race;
    ReplaySimulationRuntime runtime(race, backend);
    CSceneVehicleCar vehicle;
    race.BindVehicle(&vehicle);
    vehicle.VehicleFreeWheelingSet(1);
    race.OnCheckpoint(nullptr, &block);
    return !vehicle.CaptureRuntimeClone()
                     .controls.forcedLowSpeedFriction;
}

bool TestParityBackendRouting() {
    bool okay = Check(
            BackendClearsCurrentTransformCheckpointFreewheel(
                    forevervalidator::SimulationBackend::Reference),
            "Reference backend did not enable the checkpoint rule");
    okay &= Check(
            BackendClearsCurrentTransformCheckpointFreewheel(
                    forevervalidator::SimulationBackend::OptimizedCpu),
            "Optimized CPU backend did not enable the checkpoint rule");
    okay &= Check(
            BackendClearsCurrentTransformCheckpointFreewheel(
                    forevervalidator::SimulationBackend::Cuda),
            "CUDA staging runtime did not enable the checkpoint rule");
    okay &= Check(
            !BackendClearsCurrentTransformCheckpointFreewheel(
                    forevervalidator::SimulationBackend::SpeculativeTicking),
            "unrelated speculative backend enabled the checkpoint rule");
    return okay;
}

bool TestLapCountersRemainSeparate() {
    bool okay = true;
    for (const unsigned checkpoints : {0u, 2u}) {
        CTrackManiaRace race;
        auto setup = race.CaptureRuntimeClone();
        setup.progress.requiredCheckpointCount = checkpoints;
        setup.progress.requiredLapCount = 2u;
        setup.checkpointSlotsPassed.assign(checkpoints + 1u, 0u);
        race.RestoreRuntimeClone(setup);
        for (unsigned lap = 0; lap < 2u; ++lap) {
            for (unsigned checkpoint = 0; checkpoint < checkpoints; ++checkpoint) {
                okay &= Check(race.InternalOnCheckpoint(10, 0, checkpoint, checkpoint,
                        &race.GetPlayingPlayer()->Info(), nullptr, nullptr, 0) == 1,
                        "ordinary checkpoint was not accepted");
                okay &= Check(race.InternalOnCheckpoint(10, 0, checkpoint, checkpoint,
                        &race.GetPlayingPlayer()->Info(), nullptr, nullptr, 0) == 0,
                        "duplicate ordinary checkpoint was accepted");
            }
            race.OnFinishLine(nullptr, nullptr);
            const auto saved = race.CaptureRuntimeClone();
            race.RestoreRuntimeClone(saved);
            okay &= Check(race.Progress().completedLapCount == lap + 1u &&
                          race.Progress().checkpointCount == checkpoints * (lap + 1u) &&
                          race.Progress().raceCompleted == (lap == 1u),
                          "finish passage changed ordinary counts or snapshot lost laps");
        }
        race.OnFinishLine(nullptr, nullptr);
        okay &= Check(race.Progress().completedLapCount == 2u,
                      "completed race counted another finish passage");
    }
    return okay;
}

bool TestAcceptedCheckpointJournal() {
    CTrackManiaRace race;
    auto setup = race.CaptureRuntimeClone();
    setup.progress.requiredCheckpointCount = 2;
    setup.progress.requiredLapCount = 2;
    setup.checkpointSlotsPassed.assign(3, 0);
    race.RestoreRuntimeClone(setup);
    bool okay = true;
    for (unsigned lap = 1; lap <= 2; ++lap) {
        race.ClearAcceptedCheckpointEvents();
        race.OnFinishLine(nullptr, nullptr);
        okay &= Check(race.AcceptedCheckpointEvents().empty(), "premature finish was journaled");
        for (unsigned index = 0; index < 2; ++index) {
            const auto slot = 1u - index;
            const auto accept = [&]() { return race.InternalOnCheckpoint(10, 0, index, slot,
                    &race.GetPlayingPlayer()->Info(), nullptr, nullptr, 0); };
            okay &= Check(accept() == 1 && accept() == 0, "journal changed acceptance rules");
            const auto &event = race.AcceptedCheckpointEvents().back();
            okay &= Check(event.checkpointSlot == slot && event.checkpointIndex == index &&
                          event.lap == lap && event.eventIndex == (lap - 1u) * 3u + index + 1u && !event.finish,
                          "journal lost identity, lap, ordinal or global sequence");
        }
        race.OnFinishLine(nullptr, nullptr);
        const auto &events = race.AcceptedCheckpointEvents();
        okay &= Check(events.size() == 3 && events.back().finish && events.back().lap == lap &&
                      events.back().checkpointSlot == 2 && events.back().eventIndex == lap * 3u,
                      "simultaneous checkpoint/finish events were collapsed");
        const auto saved = race.CaptureRuntimeClone();
        race.ClearAcceptedCheckpointEvents();
        race.RestoreRuntimeClone(saved);
        okay &= Check(race.AcceptedCheckpointEvents().size() == 3 &&
                      race.AcceptedCheckpointEvents().back().eventIndex == lap * 3u,
                      "snapshot lost accepted events or their sequence");
    }
    race.ClearAcceptedCheckpointEvents();
    race.OnFinishLine(nullptr, nullptr);
    okay &= Check(race.AcceptedCheckpointEvents().empty(), "completed race produced another finish event");
    race.ResetValidationSession();
    okay &= Check(race.AcceptedCheckpointEvents().empty() &&
                  race.CaptureRuntimeClone().acceptedCheckpointEventCount == 0,
                  "new validation session retained accepted history");

    // An accepted unusual ordinal is not a checkpoint-count delta or a finish.
    race.RestoreRuntimeClone(setup);
    okay &= Check(race.InternalOnCheckpoint(10, 0, 2, 0, &race.GetPlayingPlayer()->Info(),
                    nullptr, nullptr, 0) == 1 && race.Progress().checkpointCount == 0 &&
                  race.AcceptedCheckpointEvents().size() == 1 && !race.AcceptedCheckpointEvents()[0].finish,
                  "accepted bug-checkpoint event was inferred from counters instead of acceptance");
    const auto beforeRejected = race.AcceptedCheckpointEvents().size();
    race.InternalOnCheckpoint(10, 0, 0, 99, &race.GetPlayingPlayer()->Info(), nullptr, nullptr, 0);
    race.InternalOnCheckpoint(10, 0, 0, 1, nullptr, nullptr, nullptr, 0);
    okay &= Check(race.AcceptedCheckpointEvents().size() == beforeRejected,
                  "invalid checkpoint request appeared in the journal");

    race.ResetValidationSession();
    auto many = race.CaptureRuntimeClone();
    many.progress.requiredCheckpointCount = 1024;
    many.checkpointSlotsPassed.assign(1025, 0);
    race.RestoreRuntimeClone(many);
    for (unsigned index = 0; index < 1024; ++index)
        race.InternalOnCheckpoint(10, 0, index, index, &race.GetPlayingPlayer()->Info(), nullptr, nullptr, 0);
    race.OnFinishLine(nullptr, nullptr);
    okay &= Check(race.AcceptedCheckpointEvents().size() == 1025,
                  "same-tick journal silently truncated accepted transitions");

    race.ResetValidationSession();
    race.SetReplayChallengePlayMode(EChallengePlayMode::Shortcut);
    race.OnFinishLine(nullptr, nullptr);
    okay &= Check(race.AcceptedCheckpointEvents().size() == 1 &&
                  race.AcceptedCheckpointEvents()[0].finish &&
                  race.AcceptedCheckpointEvents()[0].eventIndex == 1,
                  "shortcut finish bypassed the accepted-event journal");
    return okay;
}

}  // namespace

int main() {
    const bool contact = TestCurrentTransformCheckpointContactClearsFreewheel();
    const bool routing = TestParityBackendRouting();
    return contact && routing && TestLapCountersRemainSeparate() && TestAcceptedCheckpointJournal() ? 0 : 1;
}
