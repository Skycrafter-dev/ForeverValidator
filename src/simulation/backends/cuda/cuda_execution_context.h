#ifndef FOREVERVALIDATOR_CUDA_EXECUTION_CONTEXT_H
#define FOREVERVALIDATOR_CUDA_EXECUTION_CONTEXT_H

#include "simulation/backends/cuda/cuda_state_layout.h"

namespace forevervalidator::simulation {

// Borrowed immutable GPU resources plus an owned branch state. This is the
// physics boundary for programmable executors, not a search configuration.
// Device pointers remain valid while the originating session is loaded.
struct CudaExecutionContext {
    const void *deviceScene = nullptr;
    const void *deviceStaticConfiguration = nullptr;
    CudaCandidateState initialState{};
};

} // namespace forevervalidator::simulation
#endif
