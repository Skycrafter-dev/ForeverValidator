#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "simulation/backends/optimized_cpu/optimized_cpu_static_scene_fingerprint.h"
#include "validation/api/physics_sandbox_static_scene_test_access.h"

namespace {

using forevervalidator::AssetBytes;
using forevervalidator::ReplayIdentity;
using forevervalidator::SimulationBackend;
using forevervalidator::experimental::CreatePhysicsSandbox;
using forevervalidator::experimental::PhysicsSandbox;
using forevervalidator::experimental::PhysicsSandboxInputAction;
using forevervalidator::experimental::PhysicsSandboxInputEvent;
using forevervalidator::experimental::PhysicsSandboxInputValueKind;
using forevervalidator::experimental::PhysicsSandboxOptions;
using forevervalidator::experimental::PhysicsSandboxState;
using forevervalidator::experimental::PhysicsSandboxSwitchState;
using StaticSceneTestAccess = forevervalidator::experimental::static_scene_test::
        PhysicsSandboxStaticSceneTestAccess;

constexpr std::uint32_t TickDurationMs = 10u;

int Fail(const std::string &message) {
    std::cerr << "optimized_cpu_static_scene_lifecycle_differential: "
              << message << '\n';
    return 1;
}

bool CompareField(const char *stage,
                  const char *field,
                  std::uint64_t expected,
                  std::uint64_t actual) {
    if (expected == actual) {
        return true;
    }
    std::cerr << "optimized_cpu_static_scene_lifecycle_differential: "
              << stage << " changed " << field << '\n';
    return false;
}

bool CompareFingerprint(
        const char *stage,
        const OptimizedCpuStaticSceneFingerprint &expected,
        const OptimizedCpuStaticSceneFingerprint &actual) {
    return CompareField(stage, "identityHash",
                        expected.identityHash, actual.identityHash) &&
           CompareField(stage, "staticTreeRecordHash",
                        expected.staticTreeRecordHash,
                        actual.staticTreeRecordHash) &&
           CompareField(stage, "staticTransformHash",
                        expected.staticTransformHash,
                        actual.staticTransformHash) &&
           CompareField(stage, "meshTriangleHash",
                        expected.meshTriangleHash,
                        actual.meshTriangleHash) &&
           CompareField(stage, "groupCount",
                        expected.groupCount, actual.groupCount) &&
           CompareField(stage, "staticTreeRecordCount",
                        expected.staticTreeRecordCount,
                        actual.staticTreeRecordCount) &&
           CompareField(stage, "staticSurfaceCount",
                        expected.staticSurfaceCount,
                        actual.staticSurfaceCount) &&
           CompareField(stage, "meshSurfaceCount",
                        expected.meshSurfaceCount,
                        actual.meshSurfaceCount) &&
           CompareField(stage, "meshVertexCount",
                        expected.meshVertexCount,
                        actual.meshVertexCount) &&
           CompareField(stage, "meshTriangleCount",
                        expected.meshTriangleCount,
                        actual.meshTriangleCount) &&
           CompareField(stage, "meshOctreeCellCount",
                        expected.meshOctreeCellCount,
                        actual.meshOctreeCellCount);
}

std::optional<OptimizedCpuStaticSceneFingerprint> Capture(
        const PhysicsSandbox &sandbox,
        const char *stage) {
    std::optional<OptimizedCpuStaticSceneFingerprint> fingerprint =
            StaticSceneTestAccess::CaptureStaticSceneFingerprint(sandbox);
    if (!fingerprint.has_value()) {
        std::cerr << "optimized_cpu_static_scene_lifecycle_differential: "
                  << stage << " has no OptimizedCpu static-scene cache\n";
    }
    return fingerprint;
}

PhysicsSandboxInputEvent RespawnEvent(std::int32_t timeMs) {
    PhysicsSandboxInputEvent event;
    event.timeMs = timeMs;
    event.action = PhysicsSandboxInputAction::Respawn;
    event.value.kind = PhysicsSandboxInputValueKind::Switch;
    event.value.switchState = PhysicsSandboxSwitchState::Pressed;
    return event;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        return Fail("usage: PACKS REPLAY");
    }

    auto replayResult = forevervalidator::ReadNativeReplayFile(
            argv[2], ReplayIdentity{argv[2]});
    if (!replayResult) {
        return Fail("could not read the replay");
    }
    AssetBytes replay = std::move(replayResult).Value();

    auto source = forevervalidator::OpenInstalledPackDirectory(argv[1]);
    if (!source) {
        return Fail("could not open the installed pack directory");
    }
    PhysicsSandboxOptions options;
    options.backend = SimulationBackend::OptimizedCpu;
    options.tickDurationMs = TickDurationMs;
    auto sandboxResult = CreatePhysicsSandbox(
            std::move(source).Value(), options);
    if (!sandboxResult) {
        return Fail("could not create OptimizedCpu sandbox: " +
                    sandboxResult.Error().diagnostic);
    }
    PhysicsSandbox sandbox = std::move(sandboxResult).Value();
    auto loaded = sandbox.LoadReplay(
            {replay.data(), replay.size()}, ReplayIdentity{argv[2]});
    if (!loaded) {
        return Fail("could not load OptimizedCpu replay: " +
                    loaded.Error().diagnostic);
    }

    std::optional<OptimizedCpuStaticSceneFingerprint> baseline =
            Capture(sandbox, "initial start");
    if (!baseline.has_value()) {
        return 1;
    }
    if (baseline->groupCount == 0u ||
        baseline->staticTreeRecordCount == 0u ||
        baseline->staticSurfaceCount == 0u ||
        baseline->meshSurfaceCount == 0u ||
        baseline->meshVertexCount == 0u ||
        baseline->meshTriangleCount == 0u) {
        return Fail("fixture does not exercise a populated static mesh scene");
    }

    auto advanced = sandbox.AdvanceTicks(1u);
    if (!advanced) {
        return Fail("ordinary advance failed: " + advanced.Error().diagnostic);
    }
    std::optional<OptimizedCpuStaticSceneFingerprint> afterAdvance =
            Capture(sandbox, "ordinary advance");
    if (!afterAdvance.has_value() ||
        !CompareFingerprint("ordinary advance", *baseline, *afterAdvance)) {
        return 1;
    }

    auto inputsResult = sandbox.ReadInputs();
    if (!inputsResult) {
        return Fail("could not read replay inputs: " +
                    inputsResult.Error().diagnostic);
    }
    if (advanced.Value().timeMs >
        static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max() - TickDurationMs)) {
        return Fail("fixture time is too large for a future respawn event");
    }
    std::vector<PhysicsSandboxInputEvent> inputs =
            std::move(inputsResult).Value();
    inputs.push_back(RespawnEvent(static_cast<std::int32_t>(
            advanced.Value().timeMs + TickDurationMs)));
    std::stable_sort(
            inputs.begin(), inputs.end(),
            [](const PhysicsSandboxInputEvent &left,
               const PhysicsSandboxInputEvent &right) {
                return left.timeMs < right.timeMs;
            });
    auto replaced = sandbox.ReplaceInputs(std::move(inputs));
    if (!replaced) {
        return Fail("could not install future respawn: " +
                    replaced.Error().diagnostic);
    }

    const std::uint32_t respawnCountBefore = advanced.Value().respawnCount;
    auto respawned = sandbox.AdvanceTicks(1u);
    if (!respawned) {
        return Fail("respawn advance failed: " +
                    respawned.Error().diagnostic);
    }
    if (respawned.Value().respawnCount <= respawnCountBefore) {
        return Fail("injected future respawn was not executed");
    }
    std::optional<OptimizedCpuStaticSceneFingerprint> afterRespawn =
            Capture(sandbox, "respawn");
    if (!afterRespawn.has_value() ||
        !CompareFingerprint("respawn", *baseline, *afterRespawn)) {
        return 1;
    }

    auto captured = sandbox.CaptureState();
    if (!captured) {
        return Fail("state capture failed: " + captured.Error().diagnostic);
    }
    PhysicsSandboxState checkpoint = std::move(captured).Value();
    std::optional<OptimizedCpuStaticSceneFingerprint> afterCapture =
            Capture(sandbox, "state capture");
    if (!afterCapture.has_value() ||
        !CompareFingerprint("state capture", *baseline, *afterCapture)) {
        return 1;
    }

    auto advancedPastCapture = sandbox.AdvanceTicks(2u);
    if (!advancedPastCapture) {
        return Fail("post-capture advance failed: " +
                    advancedPastCapture.Error().diagnostic);
    }
    std::optional<OptimizedCpuStaticSceneFingerprint> afterCapturedAdvance =
            Capture(sandbox, "post-capture advance");
    if (!afterCapturedAdvance.has_value() ||
        !CompareFingerprint(
                "post-capture advance", *baseline, *afterCapturedAdvance)) {
        return 1;
    }

    auto restored = sandbox.RestoreState(checkpoint);
    if (!restored) {
        return Fail("state restore failed: " + restored.Error().diagnostic);
    }
    std::optional<OptimizedCpuStaticSceneFingerprint> afterRestore =
            Capture(sandbox, "state restore");
    if (!afterRestore.has_value() ||
        !CompareFingerprint("state restore", *baseline, *afterRestore)) {
        return 1;
    }

    auto restarted = StaticSceneTestAccess::RestartAtRaceTick(sandbox, 0u);
    if (!restarted) {
        return Fail("restart failed: " + restarted.Error().diagnostic);
    }
    if (restarted.Value().tick != 0u) {
        return Fail("restart did not return to race tick zero");
    }
    std::optional<OptimizedCpuStaticSceneFingerprint> afterRestart =
            Capture(sandbox, "restart");
    if (!afterRestart.has_value() ||
        !CompareFingerprint("restart", *baseline, *afterRestart)) {
        return 1;
    }

    std::cout << "static_groups=" << baseline->groupCount
              << " static_tree_records="
              << baseline->staticTreeRecordCount
              << " static_surfaces=" << baseline->staticSurfaceCount
              << " mesh_surfaces=" << baseline->meshSurfaceCount
              << " mesh_vertices=" << baseline->meshVertexCount
              << " mesh_triangles=" << baseline->meshTriangleCount
              << " mesh_octree_cells=" << baseline->meshOctreeCellCount
              << " lifecycle_stages=7 respawns_executed="
              << (respawned.Value().respawnCount - respawnCountBefore)
              << " result=identical\n";
    return 0;
}
