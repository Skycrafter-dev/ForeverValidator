// Generated from tests/cuda_stunt_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_state_layout.h"
#include "simulation/backends/hip/generated/hip_stunt_certification.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

GmIso4 Location(float rotationX,
                float rotationY,
                float rotationZ,
                float x,
                float y,
                float z) {
    GmMat3 rotation;
    rotation.SetIdentity();
    rotation.RotateX(rotationX);
    rotation.RotateY(rotationY);
    rotation.RotateZ(rotationZ);
    return {rotation, {x, y, z}};
}

ReplayStuntSimulationState State(
        std::uint32_t tick,
        const GmIso4 &location,
        bool wheelContact,
        bool bodyContact = false) {
    ReplayStuntSimulationState result;
    result.tickTimeMs = tick;
    result.inputQueryTimeOffsetMs = 100u;
    result.vehicleLocation = location;
    result.forwardSpeed = 20.0f;
    result.sideSpeed = 0.0f;
    result.hasWheelContact = wheelContact;
    result.hasBodyContact = bodyContact;
    return result;
}

void AddUpdate(
        CTrackManiaRace &cpu,
        std::vector<forevervalidator::simulation::HipStuntCommand> &commands,
        ReplayStuntSimulationState state) {
    cpu.SetReplayStuntSimulationState(state);
    cpu.UpdateStunts();
    forevervalidator::simulation::HipStuntCommand command;
    command.state = state;
    commands.push_back(command);
}

}  // namespace

int main() {
    using namespace forevervalidator::simulation;

    CTrackManiaRace cpu;
    cpu.ConfigureReplayStuntsSimulation(true, 100000u);
    HipRaceState initial;
    if (EncodeHipRaceState(cpu.CaptureRuntimeClone(), &initial) !=
        HipStateConversionResult::Success) {
        std::cerr << "could not encode initial stunt state\n";
        return 1;
    }

    std::vector<HipStuntCommand> commands;
    ReplayStuntSimulationState start =
            State(0u, Location(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f), true);
    start.raceStart = true;
    AddUpdate(cpu, commands, start);

    AddUpdate(
            cpu, commands,
            State(100u, Location(0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f),
                  false));
    for (std::uint32_t tick = 200u; tick <= 600u; tick += 100u) {
        const float angle =
                static_cast<float>(tick / 100u - 1u) * 0.7f;
        AddUpdate(
                cpu, commands,
                State(tick,
                      Location(angle, 0.0f, 0.0f,
                               0.1f * static_cast<float>(tick / 100u),
                               1.0f, 0.0f),
                      false));
    }
    AddUpdate(
            cpu, commands,
            State(700u, Location(3.5f, 0.0f, 0.0f, 0.7f, 0.0f, 0.0f),
                  true));

    AddUpdate(
            cpu, commands,
            State(800u, Location(3.5f, 0.0f, 0.0f, 0.8f, 0.0f, 0.0f),
                  true));
    AddUpdate(
            cpu, commands,
            State(900u, Location(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f),
                  false));
    for (std::uint32_t tick = 1000u; tick <= 1400u; tick += 100u) {
        ReplayStuntSimulationState airborne =
                State(tick,
                      Location(0.0f,
                               0.65f *
                                       static_cast<float>(
                                               tick / 100u - 9u),
                               0.35f *
                                       static_cast<float>(
                                               tick / 100u - 9u),
                               1.0f, 1.0f, 0.2f),
                      false);
        airborne.inputLastChangeTimeMs[0] = 1300u;
        AddUpdate(cpu, commands, airborne);
    }
    ReplayStuntSimulationState badLanding =
            State(1500u,
                  Location(0.0f, 3.25f, 1.75f, 1.2f, 0.0f, 0.2f),
                  false, true);
    badLanding.bodyContactHorizontalAngle = 0.75f;
    badLanding.bodyContactVerticalAngle = 0.6f;
    AddUpdate(cpu, commands, badLanding);

    for (std::uint32_t tick = 1600u; tick < 5200u; tick += 100u) {
        AddUpdate(
                cpu, commands,
                State(tick,
                      Location(0.0f, 0.0f, 0.0f,
                               static_cast<float>(tick) * 0.001f,
                               0.0f, 0.0f),
                      true));
    }

    cpu.ApplyReplayStuntRespawnPenalty(5200u);
    HipStuntCommand respawn;
    respawn.kind = HipStuntCommandKind::RespawnPenalty;
    commands.push_back(respawn);

    cpu.ApplyReplayStuntTimePenalty(1200u);
    HipStuntCommand timePenalty;
    timePenalty.kind = HipStuntCommandKind::TimePenalty;
    timePenalty.overtimeMs = 1200u;
    commands.push_back(timePenalty);

    ReplayStuntSimulationState finish =
            State(100200u,
                  Location(0.0f, 0.0f, 0.0f, 5.2f, 0.0f, 0.0f),
                  true);
    finish.finishRace = true;
    AddUpdate(cpu, commands, finish);

    const HipStuntExecution gpu =
            ExecuteHipStuntCommandsForCertification(initial, commands);
    if (!gpu.success) {
        std::cerr << gpu.diagnostic
                  << " command=" << gpu.failureCommand
                  << " detail=" << gpu.failureDetail << '\n';
        return 1;
    }

    HipRaceState expected;
    if (EncodeHipRaceState(cpu.CaptureRuntimeClone(), &expected) !=
        HipStateConversionResult::Success) {
        std::cerr << "could not encode expected stunt state\n";
        return 1;
    }
    const auto *cpuBytes =
            reinterpret_cast<const std::uint8_t *>(&expected);
    const auto *gpuBytes =
            reinterpret_cast<const std::uint8_t *>(&gpu.finalState);
    std::size_t mismatch = 0u;
    while (mismatch < sizeof(HipRaceState) &&
           cpuBytes[mismatch] == gpuBytes[mismatch]) {
        ++mismatch;
    }
    if (mismatch != sizeof(HipRaceState)) {
        std::cerr << "stunt state mismatch at byte " << mismatch
                  << " expected=" << static_cast<unsigned>(cpuBytes[mismatch])
                  << " actual=" << static_cast<unsigned>(gpuBytes[mismatch])
                  << " scores=" << expected.stunts.stuntsScore << "/"
                  << gpu.finalState.stunts.stuntsScore
                  << " events=" << expected.stuntEvents.count << "/"
                  << gpu.finalState.stuntEvents.count << '\n';
        return 1;
    }
    if (expected.stuntEvents.count < 2u ||
        !expected.stunts.stuntScoreAtTimeLimit.present ||
        expected.stunts.stuntInputHistorySize != 32u ||
        expected.stunts.stuntLocationHistorySize != 20u) {
        std::cerr << "synthetic stunt coverage did not exercise "
                     "events, latching, and bounded histories\n";
        return 1;
    }
    HipRaceState overflowInitial = initial;
    overflowInitial.stunts.replayStuntsEnabled = true;
    overflowInitial.stuntEvents.count = 2048u;
    HipStuntCommand overflowCommand;
    overflowCommand.kind = HipStuntCommandKind::TimePenalty;
    overflowCommand.overtimeMs = 100u;
    const HipStuntExecution overflow =
            ExecuteHipStuntCommandsForCertification(
                    overflowInitial, {overflowCommand});
    if (overflow.success || overflow.failureCommand != 0u ||
        overflow.failureDetail == 0u) {
        std::cerr << "stunt event overflow was not reported explicitly\n";
        return 1;
    }
    return 0;
}
