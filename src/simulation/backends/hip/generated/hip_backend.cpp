// Generated from src/simulation/backends/cuda/cuda_backend.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_backend.h"

#include "simulation/backends/simulation_backend.h"

namespace forevervalidator::simulation {

#if FOREVERVALIDATOR_HAS_HIP
HipBackendDiagnostics QueryCompiledHipRuntimeDiagnostics() noexcept;
#endif

HipBackendDiagnostics QueryHipRuntimeDiagnostics() noexcept {
#if FOREVERVALIDATOR_HAS_HIP
    return QueryCompiledHipRuntimeDiagnostics();
#else
    HipBackendDiagnostics result;
    result.status = HipBackendStatus::NotCompiled;
    result.diagnostic =
            "HIP backend support was not compiled into this build";
    return result;
#endif
}

bool IsHipBackendReady() noexcept {
    return QueryHipRuntimeDiagnostics().IsReady();
}

}  // namespace forevervalidator::simulation

namespace forevervalidator {

HipBackendDiagnostics QueryHipBackendDiagnostics() noexcept {
    return simulation::QueryHipRuntimeDiagnostics();
}

}  // namespace forevervalidator
