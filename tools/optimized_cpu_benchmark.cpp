#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using forevervalidator::AssetBytes;
using forevervalidator::ReplayIdentity;
using forevervalidator::SimulationBackend;
using forevervalidator::experimental::CreatePhysicsSandbox;
using forevervalidator::experimental::PhysicsSandbox;
using forevervalidator::experimental::PhysicsSandboxOptions;
using forevervalidator::experimental::PhysicsSandboxState;
using forevervalidator::experimental::PhysicsSandboxStateView;

constexpr std::uint64_t FnvOffset = 1469598103934665603ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;

template<typename T>
void HashValue(std::uint64_t &hash, const T &value) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        hash ^= bytes[index];
        hash *= FnvPrime;
    }
}

void HashOptional(std::uint64_t &hash,
                  const std::optional<std::uint32_t> &value) {
    const bool present = value.has_value();
    HashValue(hash, present);
    if (present) {
        HashValue(hash, *value);
    }
}

std::uint64_t Fingerprint(const PhysicsSandboxStateView &view) {
    std::uint64_t hash = FnvOffset;
    HashValue(hash, view.tick);
    HashValue(hash, view.timeMs);
    HashValue(hash, view.mapEnvironment);
    HashValue(hash, view.vehicleModel);
    const bool playModePresent = view.playMode.has_value();
    HashValue(hash, playModePresent);
    if (playModePresent) {
        HashValue(hash, *view.playMode);
    }
    HashValue(hash, view.car.rotationX);
    HashValue(hash, view.car.rotationY);
    HashValue(hash, view.car.rotationZ);
    HashValue(hash, view.car.rotationW);
    HashValue(hash, view.car.position.x);
    HashValue(hash, view.car.position.y);
    HashValue(hash, view.car.position.z);
    HashValue(hash, view.car.linearSpeed.x);
    HashValue(hash, view.car.linearSpeed.y);
    HashValue(hash, view.car.linearSpeed.z);
    HashValue(hash, view.car.angularSpeed.x);
    HashValue(hash, view.car.angularSpeed.y);
    HashValue(hash, view.car.angularSpeed.z);
    HashValue(hash, view.car.force.x);
    HashValue(hash, view.car.force.y);
    HashValue(hash, view.car.force.z);
    HashValue(hash, view.car.torque.x);
    HashValue(hash, view.car.torque.y);
    HashValue(hash, view.car.torque.z);
    HashValue(hash, view.accelerate);
    HashValue(hash, view.brake);
    HashValue(hash, view.steering);
    HashValue(hash, view.checkpointsCollected);
    HashValue(hash, view.checkpointsTotal);
    HashValue(hash, view.completedLaps);
    HashValue(hash, view.totalLaps);
    HashValue(hash, view.raceCompleted);
    HashOptional(hash, view.finishTimeMs);
    HashValue(hash, view.respawnCount);
    HashOptional(hash, view.stuntsScore);
    return hash;
}

bool ParsePositive(const char *text, std::uint32_t *value) {
    try {
        const unsigned long long parsed = std::stoull(text);
        if (parsed == 0u ||
            parsed > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        *value = static_cast<std::uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

const char *BackendName(SimulationBackend backend) {
    return backend == SimulationBackend::Reference
                   ? "reference"
                   : "optimized-cpu";
}

int Fail(const std::string &message) {
    std::cerr << "optimized_cpu_benchmark: " << message << '\n';
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        return Fail("usage: PACKS REPLAY reference|optimized-cpu "
                    "TICKS WARMUPS REPETITIONS");
    }

    SimulationBackend backend = SimulationBackend::Reference;
    const std::string backendArgument = argv[3];
    if (backendArgument == "optimized-cpu") {
        backend = SimulationBackend::OptimizedCpu;
    } else if (backendArgument != "reference") {
        return Fail("backend must be reference or optimized-cpu");
    }

    std::uint32_t tickCount = 0u;
    std::uint32_t warmupCount = 0u;
    std::uint32_t repetitionCount = 0u;
    if (!ParsePositive(argv[4], &tickCount) ||
        !ParsePositive(argv[5], &warmupCount) ||
        !ParsePositive(argv[6], &repetitionCount) ||
        (repetitionCount % 2u) == 0u) {
        return Fail("ticks and warmups must be positive; repetitions must be "
                    "positive and odd");
    }

    auto sourceResult = forevervalidator::OpenInstalledPackDirectory(argv[1]);
    if (!sourceResult) {
        return Fail("could not open the installed pack directory");
    }
    auto replayResult = forevervalidator::ReadNativeReplayFile(
            argv[2], ReplayIdentity{argv[2]});
    if (!replayResult) {
        return Fail("could not read the replay");
    }
    AssetBytes replay = std::move(replayResult).Value();

    PhysicsSandboxOptions options;
    options.backend = backend;
    options.tickDurationMs = 10u;
    options.prestartDurationMs = 2600u;
    auto sandboxResult = CreatePhysicsSandbox(
            std::move(sourceResult).Value(), options);
    if (!sandboxResult) {
        return Fail("could not create the physics sandbox");
    }
    PhysicsSandbox sandbox = std::move(sandboxResult).Value();
    auto loaded = sandbox.LoadReplay(
            {replay.data(), replay.size()}, ReplayIdentity{argv[2]});
    if (!loaded) {
        return Fail("could not load the replay: " + loaded.Error().diagnostic);
    }
    auto initialResult = sandbox.CaptureState();
    if (!initialResult) {
        return Fail("could not capture the initial state");
    }
    PhysicsSandboxState initial = std::move(initialResult).Value();

    std::vector<std::uint64_t> samples;
    samples.reserve(repetitionCount);
    std::uint64_t resultChecksum = FnvOffset;
    std::uint64_t finalFingerprint = 0u;
    const std::uint32_t totalIterations = warmupCount + repetitionCount;
    for (std::uint32_t iteration = 0u;
         iteration < totalIterations;
         ++iteration) {
        auto restored = sandbox.RestoreState(initial);
        if (!restored) {
            return Fail("could not restore the initial state");
        }

        const auto start = std::chrono::steady_clock::now();
        auto advanced = sandbox.AdvanceTicks(tickCount);
        const auto stop = std::chrono::steady_clock::now();
        if (!advanced) {
            return Fail("could not advance the sandbox: " +
                        advanced.Error().diagnostic);
        }
        if (advanced.Value().tick != tickCount) {
            return Fail("advance ended on an unexpected tick");
        }

        const std::uint64_t fingerprint = Fingerprint(advanced.Value());
        if (iteration == 0u) {
            finalFingerprint = fingerprint;
        } else if (fingerprint != finalFingerprint) {
            return Fail("repeated advance produced a different final state");
        }
        HashValue(resultChecksum, fingerprint);

        if (iteration >= warmupCount) {
            samples.push_back(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            stop - start).count()));
        }
    }

    std::sort(samples.begin(), samples.end());
    const std::uint64_t median = samples[samples.size() / 2u];
    std::cout << "backend=" << BackendName(backend)
              << " ticks=" << tickCount
              << " warmups=" << warmupCount
              << " repetitions=" << repetitionCount
              << " median_ns=" << median
              << " min_ns=" << samples.front()
              << " max_ns=" << samples.back()
              << " final_fingerprint=" << finalFingerprint
              << " result_checksum=" << resultChecksum
              << " samples_ns=";
    for (std::size_t index = 0u; index < samples.size(); ++index) {
        if (index != 0u) {
            std::cout << ',';
        }
        std::cout << samples[index];
    }
    std::cout << '\n';
    return 0;
}
