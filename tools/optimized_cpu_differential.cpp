#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using forevervalidator::AssetBytes;
using forevervalidator::ReplayIdentity;
using forevervalidator::SimulationBackend;
using forevervalidator::Vector3;
using forevervalidator::experimental::CreatePhysicsSandbox;
using forevervalidator::experimental::PhysicsSandbox;
using forevervalidator::experimental::PhysicsSandboxOptions;
using forevervalidator::experimental::PhysicsSandboxState;
using forevervalidator::experimental::PhysicsSandboxStateView;

std::uint32_t FloatBits(float value) {
    std::uint32_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value), "binary32 size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool CompareFloat(const char *name,
                  float reference,
                  float optimized,
                  std::string *difference) {
    const std::uint32_t referenceBits = FloatBits(reference);
    const std::uint32_t optimizedBits = FloatBits(optimized);
    if (referenceBits == optimizedBits) {
        return true;
    }
    *difference = std::string(name) + " differs: reference_bits=" +
            std::to_string(referenceBits) + " optimized_bits=" +
            std::to_string(optimizedBits);
    return false;
}

bool CompareVector(const char *name,
                   const Vector3 &reference,
                   const Vector3 &optimized,
                   std::string *difference) {
    const std::string xName = std::string(name) + ".x";
    const std::string yName = std::string(name) + ".y";
    const std::string zName = std::string(name) + ".z";
    return CompareFloat(
                   xName.c_str(), reference.x, optimized.x, difference) &&
           CompareFloat(
                   yName.c_str(), reference.y, optimized.y, difference) &&
           CompareFloat(
                   zName.c_str(), reference.z, optimized.z, difference);
}

template<typename T>
bool CompareValue(const char *name,
                  const T &reference,
                  const T &optimized,
                  std::string *difference) {
    if (reference == optimized) {
        return true;
    }
    *difference = std::string(name) + " differs";
    return false;
}

bool CompareState(const PhysicsSandboxStateView &reference,
                  const PhysicsSandboxStateView &optimized,
                  std::string *difference) {
    return CompareValue(
                   "tick", reference.tick, optimized.tick, difference) &&
           CompareValue("timeMs",
                        reference.timeMs,
                        optimized.timeMs,
                        difference) &&
           CompareValue("mapEnvironment",
                        reference.mapEnvironment,
                        optimized.mapEnvironment,
                        difference) &&
           CompareValue("vehicleModel",
                        reference.vehicleModel,
                        optimized.vehicleModel,
                        difference) &&
           CompareValue("playMode",
                        reference.playMode,
                        optimized.playMode,
                        difference) &&
           CompareFloat("car.rotationX",
                        reference.car.rotationX,
                        optimized.car.rotationX,
                        difference) &&
           CompareFloat("car.rotationY",
                        reference.car.rotationY,
                        optimized.car.rotationY,
                        difference) &&
           CompareFloat("car.rotationZ",
                        reference.car.rotationZ,
                        optimized.car.rotationZ,
                        difference) &&
           CompareFloat("car.rotationW",
                        reference.car.rotationW,
                        optimized.car.rotationW,
                        difference) &&
           CompareVector("car.position",
                         reference.car.position,
                         optimized.car.position,
                         difference) &&
           CompareVector("car.linearSpeed",
                         reference.car.linearSpeed,
                         optimized.car.linearSpeed,
                         difference) &&
           CompareVector("car.angularSpeed",
                         reference.car.angularSpeed,
                         optimized.car.angularSpeed,
                         difference) &&
           CompareVector("car.force",
                         reference.car.force,
                         optimized.car.force,
                         difference) &&
           CompareVector("car.torque",
                         reference.car.torque,
                         optimized.car.torque,
                         difference) &&
           CompareFloat("accelerate",
                        reference.accelerate,
                        optimized.accelerate,
                        difference) &&
           CompareFloat(
                   "brake", reference.brake, optimized.brake, difference) &&
           CompareFloat("steering",
                        reference.steering,
                        optimized.steering,
                        difference) &&
           CompareValue("checkpointsCollected",
                        reference.checkpointsCollected,
                        optimized.checkpointsCollected,
                        difference) &&
           CompareValue("checkpointsTotal",
                        reference.checkpointsTotal,
                        optimized.checkpointsTotal,
                        difference) &&
           CompareValue("completedLaps",
                        reference.completedLaps,
                        optimized.completedLaps,
                        difference) &&
           CompareValue("totalLaps",
                        reference.totalLaps,
                        optimized.totalLaps,
                        difference) &&
           CompareValue("raceCompleted",
                        reference.raceCompleted,
                        optimized.raceCompleted,
                        difference) &&
           CompareValue("finishTimeMs",
                        reference.finishTimeMs,
                        optimized.finishTimeMs,
                        difference) &&
           CompareValue("respawnCount",
                        reference.respawnCount,
                        optimized.respawnCount,
                        difference) &&
           CompareValue("stuntsScore",
                        reference.stuntsScore,
                        optimized.stuntsScore,
                        difference);
}

bool ParseTickCount(const char *text, std::uint32_t *value) {
    try {
        std::size_t consumed = 0u;
        const unsigned long parsed = std::stoul(text, &consumed);
        if (text[consumed] != '\0' || parsed == 0u ||
            parsed > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        *value = static_cast<std::uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

int Fail(const std::string &message) {
    std::cerr << "optimized_cpu_differential: " << message << '\n';
    return 1;
}

forevervalidator::experimental::PhysicsSandboxResult<PhysicsSandbox>
CreateLoadedSandbox(const std::string &packs,
                    const std::string &replayPath,
                    const AssetBytes &replay,
                    SimulationBackend backend) {
    auto source = forevervalidator::OpenInstalledPackDirectory(packs);
    if (!source) {
        return forevervalidator::experimental::PhysicsSandboxResult<
                PhysicsSandbox>::Failure({
                forevervalidator::experimental::PhysicsSandboxErrorCode::InvalidRequest,
                {},
                "could not open the installed pack directory"});
    }
    PhysicsSandboxOptions options;
    options.backend = backend;
    auto sandbox = CreatePhysicsSandbox(std::move(source).Value(), options);
    if (!sandbox) {
        return sandbox;
    }
    auto loaded = sandbox.Value().LoadReplay(
            {replay.data(), replay.size()}, ReplayIdentity{replayPath});
    if (!loaded) {
        return forevervalidator::experimental::PhysicsSandboxResult<
                PhysicsSandbox>::Failure(std::move(loaded).Error());
    }
    return sandbox;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        return Fail("usage: PACKS REPLAY TICKS");
    }
    std::uint32_t tickCount = 0u;
    if (!ParseTickCount(argv[3], &tickCount)) {
        return Fail("ticks must be a positive uint32");
    }

    auto replayResult = forevervalidator::ReadNativeReplayFile(
            argv[2], ReplayIdentity{argv[2]});
    if (!replayResult) {
        return Fail("could not read the replay");
    }
    AssetBytes replay = std::move(replayResult).Value();
    auto referenceResult = CreateLoadedSandbox(
            argv[1], argv[2], replay, SimulationBackend::Reference);
    if (!referenceResult) {
        return Fail("could not load Reference: " +
                    referenceResult.Error().diagnostic);
    }
    auto optimizedResult = CreateLoadedSandbox(
            argv[1], argv[2], replay, SimulationBackend::OptimizedCpu);
    if (!optimizedResult) {
        return Fail("could not load OptimizedCpu: " +
                    optimizedResult.Error().diagnostic);
    }
    PhysicsSandbox reference = std::move(referenceResult).Value();
    PhysicsSandbox optimized = std::move(optimizedResult).Value();

    auto referenceView = reference.ReadState();
    auto optimizedView = optimized.ReadState();
    if (!referenceView || !optimizedView) {
        return Fail("could not read tick-zero state");
    }
    std::string difference;
    if (!CompareState(referenceView.Value(), optimizedView.Value(), &difference)) {
        return Fail("tick 0: " + difference);
    }

    const std::uint32_t checkpointTick =
            tickCount == 1u ? 1u : tickCount / 2u;
    std::optional<PhysicsSandboxState> referenceCheckpoint;
    std::optional<PhysicsSandboxState> optimizedCheckpoint;
    std::vector<PhysicsSandboxStateView> referenceTail;
    std::vector<PhysicsSandboxStateView> optimizedTail;
    referenceTail.reserve(tickCount - checkpointTick);
    optimizedTail.reserve(tickCount - checkpointTick);

    for (std::uint32_t tick = 1u; tick <= tickCount; ++tick) {
        referenceView = reference.AdvanceTicks(1u);
        optimizedView = optimized.AdvanceTicks(1u);
        if (!referenceView || !optimizedView) {
            return Fail("advance failed at tick " + std::to_string(tick));
        }
        if (!CompareState(
                    referenceView.Value(), optimizedView.Value(), &difference)) {
            return Fail("tick " + std::to_string(tick) + ": " + difference);
        }
        if (tick == checkpointTick) {
            auto capturedReference = reference.CaptureState();
            auto capturedOptimized = optimized.CaptureState();
            if (!capturedReference || !capturedOptimized) {
                return Fail("could not capture midpoint state");
            }
            referenceCheckpoint.emplace(
                    std::move(capturedReference).Value());
            optimizedCheckpoint.emplace(
                    std::move(capturedOptimized).Value());
        } else if (tick > checkpointTick) {
            referenceTail.push_back(referenceView.Value());
            optimizedTail.push_back(optimizedView.Value());
        }
    }

    if (!referenceCheckpoint || !optimizedCheckpoint) {
        return Fail("midpoint state was not captured");
    }
    auto restoredReference = reference.RestoreState(*referenceCheckpoint);
    auto restoredOptimized = optimized.RestoreState(*optimizedCheckpoint);
    if (!restoredReference || !restoredOptimized) {
        return Fail("could not restore midpoint state");
    }
    if (!CompareState(restoredReference.Value(),
                      referenceCheckpoint->View(),
                      &difference)) {
        return Fail("Reference restore: " + difference);
    }
    if (!CompareState(restoredOptimized.Value(),
                      optimizedCheckpoint->View(),
                      &difference)) {
        return Fail("OptimizedCpu restore: " + difference);
    }
    if (!CompareState(restoredReference.Value(),
                      restoredOptimized.Value(),
                      &difference)) {
        return Fail("restored midpoint: " + difference);
    }

    for (std::uint32_t tailIndex = 0u;
         tailIndex < referenceTail.size();
         ++tailIndex) {
        referenceView = reference.AdvanceTicks(1u);
        optimizedView = optimized.AdvanceTicks(1u);
        const std::uint32_t tick = checkpointTick + tailIndex + 1u;
        if (!referenceView || !optimizedView) {
            return Fail("repeat advance failed at tick " +
                        std::to_string(tick));
        }
        if (!CompareState(referenceView.Value(),
                          optimizedView.Value(),
                          &difference)) {
            return Fail("repeat tick " + std::to_string(tick) + ": " +
                        difference);
        }
        if (!CompareState(referenceTail[tailIndex],
                          referenceView.Value(),
                          &difference)) {
            return Fail("Reference repeat tick " + std::to_string(tick) +
                        ": " + difference);
        }
        if (!CompareState(optimizedTail[tailIndex],
                          optimizedView.Value(),
                          &difference)) {
            return Fail("OptimizedCpu repeat tick " +
                        std::to_string(tick) + ": " + difference);
        }
    }

    std::cout << "ticks_compared=" << tickCount
              << " repeated_tail_ticks=" << referenceTail.size()
              << " result=identical\n";
    return 0;
}
