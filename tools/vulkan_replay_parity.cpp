#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/input_state.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "validation/api/physics_sandbox_cuda_test_access.h"
#include "simulation/runtime/replay_simulation_session.h"

namespace {

int Fail(const std::string &message) {
    std::cerr << "vulkan_replay_parity: " << message << '\n';
    return 1;
}

std::size_t FirstDifferentByte(
        const forevervalidator::simulation::CudaCandidateState &left,
        const forevervalidator::simulation::CudaCandidateState &right) {
    const auto *leftBytes =
            reinterpret_cast<const std::uint8_t *>(&left);
    const auto *rightBytes =
            reinterpret_cast<const std::uint8_t *>(&right);
    for (std::size_t index = 0u; index < sizeof(left); ++index) {
        if (leftBytes[index] != rightBytes[index]) return index;
    }
    return sizeof(left);
}

bool SameCandidateState(
        const forevervalidator::simulation::CudaCandidateState &left,
        const forevervalidator::simulation::CudaCandidateState &right) {
    using namespace forevervalidator::simulation;
    if (left.schemaVersion != right.schemaVersion ||
        left.candidateId != right.candidateId ||
        left.validationSeed != right.validationSeed ||
        left.randomState != right.randomState ||
        left.controlCursor != right.controlCursor) {
        return false;
    }
    ReplaySimulationInstanceClone leftClone;
    ReplaySimulationInstanceClone rightClone;
    return DecodeCudaCandidateState(left, &leftClone) ==
                   CudaStateConversionResult::Success &&
           DecodeCudaCandidateState(right, &rightClone) ==
                   CudaStateConversionResult::Success &&
           ReplaySimulationInstanceSemanticHash(leftClone) ==
                   ReplaySimulationInstanceSemanticHash(rightClone);
}

void AddSteeringMutation(
        std::vector<forevervalidator::experimental::
                            PhysicsSandboxInputEvent> &events,
        std::int32_t timeMs,
        forevervalidator::AnalogInputState value) {
    using namespace forevervalidator::experimental;
    PhysicsSandboxInputEvent event;
    event.timeMs = timeMs;
    event.action = PhysicsSandboxInputAction::Steer;
    event.value.kind = PhysicsSandboxInputValueKind::Analog;
    event.value.analog = value;
    events.push_back(event);
}

bool Run(
        const char *packs,
        const char *replayPath,
        const forevervalidator::AssetBytes &replay,
        bool mutate) {
    using namespace forevervalidator;
    using namespace forevervalidator::experimental;
    using Access = cuda_test::PhysicsSandboxCudaTestAccess;

    auto source = OpenInstalledPackDirectory(packs);
    if (!source) return false;
    PhysicsSandboxOptions options;
    options.backend = SimulationBackend::Vulkan;
    auto created = CreatePhysicsSandbox(
            std::move(source).Value(), options);
    if (!created) return false;
    PhysicsSandbox sandbox = std::move(created).Value();
    auto loaded = sandbox.LoadReplay(
            {replay.data(), replay.size()},
            ReplayIdentity{replayPath});
    if (!loaded) {
        std::cerr << loaded.Error().diagnostic << '\n';
        return false;
    }
    if (mutate) {
        auto inputs = sandbox.ReadInputs();
        if (!inputs) return false;
        std::vector<PhysicsSandboxInputEvent> events =
                std::move(inputs).Value();
        AddSteeringMutation(events, 100, 16384);
        AddSteeringMutation(events, 850, -24576);
        AddSteeringMutation(events, 1600, 0);
        std::stable_sort(
                events.begin(), events.end(),
                [](const auto &left, const auto &right) {
                    return left.timeMs < right.timeMs;
                });
        if (!sandbox.ReplaceInputs(std::move(events))) return false;
    }
    const std::size_t end = Access::TimelineSize(sandbox);
    const std::size_t begin = Access::Cursor(sandbox);
    std::uint64_t checkedBytes = 0u;
    for (std::size_t cursor = begin; cursor < end; ++cursor) {
        auto snapshot = sandbox.CaptureState();
        const auto before = Access::CaptureCandidateState(sandbox);
        if (!snapshot || !before.has_value()) return false;
        const ReplayCudaVehiclePrefixDifferential differential =
                Access::RunNextTimelineTick(sandbox);
        if (!differential.success) {
            std::cerr << differential.diagnostic
                      << " cursor=" << cursor << '\n';
            return false;
        }
        checkedBytes += differential.checkedBytes;
        const auto after = Access::CaptureCandidateState(sandbox);
        if (!after.has_value() ||
            !SameCandidateState(*before, *after)) {
            std::cerr << "differential restoration changed state"
                      << " cursor=" << cursor;
            if (after.has_value()) {
                std::cerr << " byte="
                          << FirstDifferentByte(*before, *after);
            }
            std::cerr << '\n';
            return false;
        }
        if (!sandbox.RestoreState(snapshot.Value())) {
            std::cerr << "snapshot restoration failed"
                      << " cursor=" << cursor << '\n';
            return false;
        }
        const auto restored = Access::CaptureCandidateState(sandbox);
        if (!restored.has_value() ||
            !SameCandidateState(*before, *restored)) {
            std::cerr << "snapshot restoration changed state"
                      << " cursor=" << cursor;
            if (restored.has_value()) {
                std::cerr << " byte="
                          << FirstDifferentByte(*before, *restored);
            }
            std::cerr << '\n';
            return false;
        }
        if (!sandbox.AdvanceTicks(1u)) {
            std::cerr << "Vulkan timeline advance failed"
                      << " cursor=" << cursor
                      << " diagnostic="
                      << Access::Diagnostic(sandbox) << '\n';
            return false;
        }
    }
    const auto finalState = sandbox.ReadState();
    if (!finalState) {
        std::cerr << "could not read final sandbox state\n";
        return false;
    }
    std::cout << "{\"mode\":\""
              << (mutate ? "mutated" : "recorded")
              << "\",\"checked_ticks\":" << (end - begin)
              << ",\"checked_state_bytes\":" << checkedBytes
              << ",\"restorations\":" << (end - begin)
              << ",\"race_completed\":"
              << (finalState.Value().raceCompleted ? "true" : "false")
              << ",\"finish_time_ms\":";
    if (finalState.Value().finishTimeMs.has_value()) {
        std::cout << *finalState.Value().finishTimeMs;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"finish_time_ns\":";
    if (finalState.Value().finishTime.has_value()) {
        std::cout << finalState.Value().finishTime->estimatedNs;
    } else {
        std::cout << "null";
    }
    std::cout
              << "}\n";
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        return Fail("usage: PACKS REPLAY [recorded|mutated|both]");
    }
    const std::string mode = argc == 4 ? argv[3] : "both";
    if (mode != "recorded" && mode != "mutated" && mode != "both") {
        return Fail("mode must be recorded, mutated, or both");
    }
    auto replay = forevervalidator::ReadNativeReplayFile(
            argv[2], forevervalidator::ReplayIdentity{argv[2]});
    if (!replay) return Fail("could not read replay");
    if ((mode == "recorded" || mode == "both") &&
        !Run(argv[1], argv[2], replay.Value(), false)) {
        return Fail("recorded replay parity failed");
    }
    if ((mode == "mutated" || mode == "both") &&
        !Run(argv[1], argv[2], replay.Value(), true)) {
        return Fail("mutated replay parity failed");
    }
    return 0;
}
