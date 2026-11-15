#ifndef FOREVERVALIDATOR_PHYSICS_SANDBOX_TWIN_TEST_ACCESS_H
#define FOREVERVALIDATOR_PHYSICS_SANDBOX_TWIN_TEST_ACCESS_H

#include <memory>

#include <forevervalidator/experimental/physics_sandbox.h>

#include "simulation/runtime/replay_simulation_session.h"

namespace forevervalidator::experimental::twin_test {

// Internal hook for the twin acceptance harness (tools/twin) and the twin
// API (src/twin/api/): reads the runtime clone inside a captured sandbox
// state, builds a state whose clone is replaced (same cursor, inputs and
// compatibility keys), and reaches a loaded sandbox's session. Not part of
// the supported public sandbox API.
struct PhysicsSandboxTwinTestAccess {
    static std::shared_ptr<const ReplaySimulationInstanceClone> RuntimeClone(
            const PhysicsSandboxState &state) noexcept;
    static bool ReplaceRuntimeClone(
            PhysicsSandboxState &state,
            const ReplaySimulationInstanceClone &clone) noexcept;
    // The loaded sandbox's session and simulation definition (twin world
    // compile, src/twin/api/twin_api.cpp). Null when nothing is loaded.
    static const ReplaySimulationSession *Session(
            const PhysicsSandbox &sandbox) noexcept;
    static const ReplaySimulationDefinition *Definition(
            const PhysicsSandbox &sandbox) noexcept;
    // The bridge (tools/twin/twin_bridge.cpp): the vehicle controls of the
    // tick AdvanceTicks(1) would run next, and that tick run with other
    // controls (the control plan's tick otherwise: its time, race actions);
    // the cursor advances as with AdvanceTicks(1).
    struct TickControls {
        float gateA = 0.0f;     // accelerate (ReplayVehicleControlState::lowSpeedGateA)
        float gateB = 0.0f;     // brake (lowSpeedGateB)
        float steering = 0.0f;  // analog / 65536, or -1 / 1 (digital)
    };
    static bool NextTickControls(const PhysicsSandbox &sandbox,
                                 TickControls *controls) noexcept;
    static PhysicsSandboxResult<PhysicsSandboxStateView> AdvanceTickWithControls(
            PhysicsSandbox &sandbox, const TickControls &controls) noexcept;
};

}  // namespace forevervalidator::experimental::twin_test

#endif
