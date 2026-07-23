#pragma once

#include <cstdint>
#include <optional>

#include <forevervalidator/experimental/physics_sandbox.h>

#include "simulation/backends/optimized_cpu/optimized_cpu_static_scene_fingerprint.h"

namespace forevervalidator::experimental::static_scene_test {

// Deliberately internal: lifecycle differentials may inspect the static scene
// and invoke the same restart routine used by LoadReplay, but cannot mutate it.
struct PhysicsSandboxStaticSceneTestAccess {
    static std::optional<OptimizedCpuStaticSceneFingerprint>
            CaptureStaticSceneFingerprint(
                    const PhysicsSandbox &sandbox) noexcept;
    static PhysicsSandboxResult<PhysicsSandboxStateView> RestartAtRaceTick(
            PhysicsSandbox &sandbox,
            std::uint64_t raceTick) noexcept;
};

}  // namespace forevervalidator::experimental::static_scene_test
