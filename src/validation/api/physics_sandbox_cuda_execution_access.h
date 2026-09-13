#ifndef FOREVERVALIDATOR_PHYSICS_SANDBOX_CUDA_EXECUTION_ACCESS_H
#define FOREVERVALIDATOR_PHYSICS_SANDBOX_CUDA_EXECUTION_ACCESS_H

#include <forevervalidator/experimental/physics_sandbox.h>
#include "simulation/backends/cuda/cuda_execution_context.h"
#include "simulation/backends/cuda/cuda_timeline_executor.h"

namespace forevervalidator::experimental {

struct PhysicsSandboxCudaExecutionContext {
    simulation::CudaExecutionContext physics;
    PhysicsSandboxStateView state;
    std::vector<simulation::CudaControlTick> ticks;
    std::uint32_t prestartDurationMs = 0;
    std::uint64_t firstCursor = 0;
};

// Internal, build-specific CUDA ABI. Consumers compile against the same
// pinned engine sources. No native mutation, evaluator or search policy is
// selected here; programs own every input edit, tick and result decision.
struct PhysicsSandboxCudaExecutionAccess {
    static PhysicsSandboxResult<PhysicsSandboxCudaExecutionContext> Capture(
        const PhysicsSandbox &sandbox) noexcept;
    static PhysicsSandboxResult<PhysicsSandboxState> ImportState(
        const PhysicsSandboxState &origin,
        const simulation::CudaCandidateState &state,
        const PhysicsSandboxStateView &view,
        std::vector<PhysicsSandboxInputEvent> inputs) noexcept;
};

} // namespace forevervalidator::experimental
#endif
